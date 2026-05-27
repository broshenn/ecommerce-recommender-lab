from __future__ import annotations

import json
import re
from typing import Any

from app.models import AgentResult, Product, RecommendRequest
from app.personalization import score_product
from app.services import llm_client
from app.services.vector_store import VectorRecallUnavailable, get_product_vector_store

from app.agents.base_agent import BaseAgent


RERANK_PROMPT = """你是电商推荐排序专家。根据用户偏好和商品信息，对候选商品按相关性从高到低重新排序。

排序原则:
1. 优先匹配用户的真实购物意图（参考推荐提示）
2. 类目/品牌偏好匹配的商品排在前面
3. 价格在用户预算范围内优先
4. 适当保持类目多样性，避免全部推同一类目

请输出按推荐优先级排序的商品ID列表（JSON数组）:
["商品ID1", "商品ID2", ...]

只输出JSON数组，不要其他内容。"""


class ProductRecAgent(BaseAgent):
    """商品推荐 Agent：先向量召回，再用 LLM 或规则重排。"""

    def __init__(self):
        super().__init__(name="product_rec", timeout=18.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        request: RecommendRequest = kwargs["request"]
        products: list[Product] = kwargs.get("products", [])
        limit: int = kwargs.get("limit", request.num_items)
        mode: str = kwargs.get("mode", "rerank")

        if mode == "recall":
            return self._recall(request, products, limit)

        return self._rerank(request, products, limit, backend="rule_rerank")

    def _recall(
        self,
        request: RecommendRequest,
        products: list[Product],
        limit: int,
    ) -> AgentResult:
        try:
            vector_store = get_product_vector_store()
            product_ids = vector_store.recall(request, products, limit)
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "product_ids": product_ids,
                    "scores": {},
                    "candidate_count": len(products),
                    "returned_count": len(product_ids),
                    "mode": "recall",
                    "backend": vector_store.backend_name,
                    "query": vector_store.status(),
                },
                confidence=0.8,
            )
        except VectorRecallUnavailable as exc:
            fallback = self._rerank(
                request,
                products,
                limit,
                backend="rule_fallback_after_vector_unavailable",
            )
            fallback.data["fallback_reason"] = str(exc)
            return fallback

    def _rerank(
        self,
        request: RecommendRequest,
        products: list[Product],
        limit: int,
        backend: str,
    ) -> AgentResult:
        if request.context.get("llm_hint"):
            llm_result = self._llm_rerank(request, products, limit, backend)
            if llm_result is not None:
                return llm_result

        scored = [(score_product(product, request), product) for product in products]
        scored.sort(
            key=lambda item: (
                item[0].value,
                item[1].rating or 0,
                item[1].rating_count or 0,
            ),
            reverse=True,
        )

        selected = scored[:limit]
        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "product_ids": [product.product_id for _, product in selected],
                "scores": {
                    product.product_id: {
                        "score": score.value,
                        "reason": score.reason,
                    }
                    for score, product in selected
                },
                "candidate_count": len(products),
                "returned_count": len(selected),
                "mode": "rerank",
                "backend": backend,
            },
            confidence=0.85,
        )

    def _llm_rerank(
        self,
        request: RecommendRequest,
        products: list[Product],
        limit: int,
        backend: str,
    ) -> AgentResult | None:
        llm_candidates = self._llm_candidate_products(products, limit)
        text = llm_client.chat(
            system_prompt=RERANK_PROMPT,
            user_message=self._build_llm_rerank_message(request, llm_candidates, limit),
            max_tokens=512,
        )
        if not text:
            return None

        selected_ids = self._normalize_llm_product_ids(text, products, limit)
        if not selected_ids:
            return None

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "product_ids": selected_ids,
                "scores": {
                    product_id: {
                        "score": round(1.0 - index / max(len(selected_ids), 1), 2),
                        "reason": f"LLM 重排序第{index + 1}位",
                    }
                    for index, product_id in enumerate(selected_ids)
                },
                "candidate_count": len(products),
                "llm_candidate_count": len(llm_candidates),
                "returned_count": len(selected_ids),
                "mode": "llm_rerank",
                "backend": f"llm+{backend}",
                "llm_client": llm_client.status(),
            },
            confidence=0.85,
        )

    def _build_llm_rerank_message(
        self,
        request: RecommendRequest,
        products: list[Product],
        limit: int,
    ) -> str:
        budget_range = "无"
        if request.budget_min is not None or request.budget_max is not None:
            budget_range = f"¥{request.budget_min or 0} - ¥{request.budget_max or '不限'}"

        product_lines = [
            (
                f"{index + 1}. ID:{product.product_id} {product.name} "
                f"类目:{product.category} 价格:¥{product.price} "
                f"品牌:{product.brand} 标签:{','.join(product.tags)}"
            )
            for index, product in enumerate(products)
        ]

        return "\n".join(
            [
                "## 用户偏好",
                f"- 偏好类目: {', '.join(request.preferred_categories) or '无'}",
                f"- 偏好品牌: {', '.join(request.liked_brands) or '无'}",
                f"- 偏好标签: {', '.join(request.preferred_tags) or '无'}",
                f"- 预算范围: {budget_range}",
                f"- 推荐提示: {request.context.get('llm_hint', '')}",
                "",
                f"## 候选商品（{len(products)}件）",
                *product_lines,
                "",
                f"请从中选出最合适的{limit}件，按优先级排序。",
            ]
        )

    def _llm_candidate_products(self, products: list[Product], limit: int) -> list[Product]:
        return products[: min(len(products), max(limit * 2, 10))]

    def _normalize_llm_product_ids(
        self,
        raw_output: str,
        products: list[Product],
        limit: int,
    ) -> list[str]:
        product_ids = {product.product_id for product in products}
        raw_ids = self._extract_product_ids(raw_output, product_ids)
        selected: list[str] = []
        seen: set[str] = set()

        for raw_id in raw_ids:
            product_id = str(raw_id).strip()
            if product_id in product_ids and product_id not in seen:
                selected.append(product_id)
                seen.add(product_id)
                if len(selected) >= limit:
                    return selected

        # LLM 可能漏掉部分商品；按原候选顺序补齐，保证下游数量稳定。
        for product in products:
            if product.product_id not in seen:
                selected.append(product.product_id)
                seen.add(product.product_id)
                if len(selected) >= limit:
                    break

        return selected

    def _extract_product_ids(self, raw_output: str, valid_ids: set[str]) -> list[str]:
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]

        ordered_ids: list[str] = []
        for product_id in re.findall(r"[A-Z0-9]{10}", raw_output):
            if product_id in valid_ids and product_id not in ordered_ids:
                ordered_ids.append(product_id)
        return ordered_ids
