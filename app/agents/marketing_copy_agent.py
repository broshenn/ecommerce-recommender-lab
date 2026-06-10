from __future__ import annotations

import re
from typing import Any

import os

from app.models import AgentResult, MarketingCopy, Product, UserProfile
from app.services import llm_client
from app.services.local_hf_copy_client import local_hf_copy_client

from app.agents.base_agent import BaseAgent

# Ollama LoRA 文案模型配置
_COPY_BACKEND = os.getenv("COPY_LLM_BACKEND", "").strip().lower()
_OLLAMA_COPY_MODEL = os.getenv("OLLAMA_COPY_MODEL", "ecom-copy-lora:qwen25-3b-gguf")
_OLLAMA_COPY_BASE_URL = os.getenv("OLLAMA_COPY_BASE_URL", "http://127.0.0.1:11434/v1")


SEGMENT_TEMPLATES = {
    "new_user": """你是电商营销文案专家。为新用户撰写欢迎+推荐文案。
风格要求：热情友好、突出新人专属优惠感、降低决策门槛。
每个商品生成一条文案（25-40字）。""",
    "high_value": """你是电商营销文案专家。为高价值VIP用户撰写推荐文案。
风格要求：品质感、尊享感、突出商品高端属性和品牌价值。
每个商品生成一条文案（25-40字）。""",
    "price_sensitive": """你是电商营销文案专家。为价格敏感用户撰写推荐文案。
风格要求：突出性价比、价格亲民、实用耐用、高评分口碑。
每个商品生成一条文案（25-40字）。

严格规则：
- 只能使用商品名称、真实价格、品牌、标签、评分、库存等真实信息
- 禁止编造具体的折扣金额、满减活动、限时秒杀、买一送一
- 禁止出现"促销价""近期活动""限时优惠""直降""打折""原价"等词汇
- 可写"性价比""价格实惠""预算友好""物有所值""高评分""口碑好"
- 如果商品价格低于同品类平均，可写"亲民价"；如果商品评分高，可写"高好评率""用户推荐"
- 实在没有卖点就描述商品本身功能和使用场景""",
    "churn_risk": """你是电商营销文案专家。为即将流失的用户撰写召回文案。
风格要求：情感唤回、品质保障、服务承诺、温和提醒。
每个商品生成一条文案（25-40字）。

严格规则：
- 只能使用商品名称、品牌、评分、库存、标签等真实信息
- 禁止编造具体折扣金额、专属优惠券、限时活动、最后机会
- 禁止出现"专属折扣""限时优惠""错过不再""最后机会"等制造假紧迫感的话术
- 可写"为你保留""品质之选""售后无忧""值得拥有""欢迎回来""我们一直在"
- 强调商品品质和用户口碑，用情感连接替代虚假促销""",
    "active": """你是电商营销文案专家。为活跃用户撰写推荐文案。
风格要求：突出商品亮点和使用场景，引发共鸣。
每个商品生成一条文案（25-40字）。""",
    "category_explorer": """你是电商营销文案专家。为探索型用户撰写推荐文案。
风格要求：突出商品新奇特点，激发探索欲和好奇心。
每个商品生成一条文案（25-40字）。""",
    "brand_loyal": """你是电商营销文案专家。为品牌忠实用户撰写推荐文案。
风格要求：强化品牌认同感，突出品牌生态和兼容优势。
每个商品生成一条文案（25-40字）。""",
    "_default": """你是电商营销文案专家。为用户撰写个性化推荐文案。
风格要求：结合商品特点，简洁有力，突出实用价值。
每个商品生成一条文案（25-40字）。""",
}

FORBIDDEN_WORDS = [
    "最好",
    "第一",
    "国家级",
    "全球首",
    "绝对",
    "100%",
    "永久",
    "万能",
    "祖传",
    "纯天然",
    "最便宜",
    "全网最低",
]

OUTPUT_INSTRUCTION = """
请以JSON数组格式输出,每个元素格式:
[{"product_id": "xxx", "copy": "文案内容（25-40字）"}]
只输出JSON,不要其他内容。"""

OUTPUT_SAFETY_RULES = """
通用安全规则（所有分群都必须遵守）:
- 只能使用商品信息中真实存在的字段作为卖点
- 禁止编造任何具体数字（折扣金额、降价幅度、优惠券面额等）
- 禁止编造活动类型（秒杀、满减、买赠、限时等）
- 如果实在没有卖点，直接描述商品功能和使用场景，不要硬凑
- 不要在每个商品的文案中都重复相同的句式"""


def _chat_ollama_copy(system_prompt: str, user_message: str) -> list[dict] | None:
    """调 Ollama 本地 LoRA 模型生成文案。不可用时返回 None。"""
    if _COPY_BACKEND not in {"ollama", "local_ollama"}:
        return None
    try:
        import json as _json
        from openai import OpenAI as _OpenAI

        client = _OpenAI(api_key="ollama", base_url=_OLLAMA_COPY_BASE_URL)
        response = client.chat.completions.create(
            model=_OLLAMA_COPY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=768,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) >= 3 else content.strip("`")
        return _json.loads(content)
    except Exception as exc:
        import sys as _sys
        print(f"[_chat_ollama_copy] ERROR: {exc}", file=_sys.stderr)
        return None


class MarketingCopyAgent(BaseAgent):
    """营销文案 Agent：优先 LLM 个性化生成，失败时使用规则文案兜底。"""

    def __init__(self):
        super().__init__(name="marketing_copy", timeout=10.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        products: list[Product] = kwargs.get("products", [])
        profile: UserProfile | None = kwargs.get("profile")
        llm_profile: dict[str, Any] = kwargs.get("llm_profile", {})
        experiment_group: str = kwargs.get("experiment_group", "")

        if not products:
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={"copies": [], "template": "empty", "copy_count": 0, "mode": "empty"},
                confidence=1.0,
            )

        if experiment_group == "control":
            copies = [
                MarketingCopy(
                    product_id=product.product_id,
                    text=self._copy_for_product(product, profile),
                )
                for product in products
            ]
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "copies": [copy.model_dump(mode="json") for copy in copies],
                    "template": "control_rule",
                    "copy_count": len(copies),
                    "mode": "control_rule",
                    "llm_client": llm_client.status(),
                    "local_hf_copy_client": local_hf_copy_client.status(),
                },
                confidence=0.85,
            )

        llm_copies = self._generate_via_llm(products, profile, llm_profile)
        if llm_copies is not None:
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "copies": llm_copies,
                    "template": f"llm_{self._pick_segment(llm_profile)}",
                    "copy_count": len(llm_copies),
                    "mode": "llm",
                    "llm_client": llm_client.status(),
                    "local_hf_copy_client": local_hf_copy_client.status(),
                    "ollama_copy_model": _OLLAMA_COPY_MODEL,
                },
                confidence=0.9,
            )

        copies = [
            MarketingCopy(
                product_id=product.product_id,
                text=self._copy_for_product(product, profile),
            )
            for product in products
        ]
        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "copies": [copy.model_dump(mode="json") for copy in copies],
                "template": "rule_based_fallback",
                "copy_count": len(copies),
                "mode": "rule_fallback",
                "llm_client": llm_client.status(),
                "local_hf_copy_client": local_hf_copy_client.status(),
            },
            confidence=0.6,
        )

    def _generate_via_llm(
        self,
        products: list[Product],
        profile: UserProfile | None,
        llm_profile: dict[str, Any],
    ) -> list[dict[str, str]] | None:
        segment = self._pick_segment(llm_profile)
        system_prompt = SEGMENT_TEMPLATES.get(segment, SEGMENT_TEMPLATES["_default"])
        prompt = f"{system_prompt}\n\n{OUTPUT_SAFETY_RULES}\n\n{OUTPUT_INSTRUCTION}"
        message = self._build_llm_message(products, profile, llm_profile)
        # ① Ollama LoRA GGUF 模型（本地最快）
        result = _chat_ollama_copy(prompt, message)
        if result is None:
            # ② DeepSeek API 兜底
            result = llm_client.chat_json(
                system_prompt=prompt,
                user_message=message,
                default=None,
            )
        if not isinstance(result, list):
            return None

        normalized = self._normalize_llm_copies(result, products)
        return normalized or None

    def _build_llm_message(
        self,
        products: list[Product],
        profile: UserProfile | None,
        llm_profile: dict[str, Any],
    ) -> str:
        product_lines = [
            (
                f"- ID:{product.product_id} 名称:{product.name} 类目:{product.category} "
                f"价格:¥{product.price} 品牌:{product.brand} 标签:{','.join(product.tags)}"
            )
            for product in products
        ]
        return "\n".join(
            [
                f"用户分群: {', '.join(llm_profile.get('segments', [])) or '无'}",
                f"用户意图: {llm_profile.get('intent_summary', '无')}",
                f"推荐提示: {llm_profile.get('recommendation_hint', '无')}",
                f"价格敏感度: {llm_profile.get('price_sensitivity', 'medium')}",
                f"长期偏好类目: {', '.join(profile.preferred_categories) if profile else '无'}",
                "",
                "商品列表:",
                *product_lines,
            ]
        )

    def _normalize_llm_copies(
        self,
        raw_items: list[Any],
        products: list[Product],
    ) -> list[dict[str, str]]:
        valid_product_ids = {product.product_id for product in products}
        copies_by_id: dict[str, str] = {}

        for item in self._iter_copy_items(raw_items):
            if not isinstance(item, dict):
                continue
            product_id = str(item.get("product_id", "")).strip()
            text = str(item.get("copy") or item.get("text") or "").strip()
            if product_id in valid_product_ids and text:
                copies_by_id[product_id] = self._compliance_check(text)

        return [
            MarketingCopy(
                product_id=product.product_id,
                text=copies_by_id[product.product_id],
            ).model_dump(mode="json")
            for product in products
            if product.product_id in copies_by_id
        ]

    def _iter_copy_items(self, raw_items: list[Any]) -> list[Any]:
        flattened: list[Any] = []
        for item in raw_items:
            if isinstance(item, list):
                flattened.extend(self._iter_copy_items(item))
            else:
                flattened.append(item)
        return flattened

    def _pick_segment(self, llm_profile: dict[str, Any]) -> str:
        segments = llm_profile.get("segments", [])
        if not isinstance(segments, list):
            return "active"
        for segment in segments:
            if segment in SEGMENT_TEMPLATES:
                return segment
        return "active"

    def _copy_for_product(self, product: Product, profile: UserProfile | None) -> str:
        if profile and product.category in profile.preferred_categories:
            return f"根据你的浏览偏好，为你优先推荐 {product.name}，库存和价格都已为你校验。"
        if product.stock <= 100:
            return f"{product.name} 当前库存紧张，适合尽快决策。"
        return f"{product.name} 为你精选，兼顾品质、价格与实用性。"

    def _compliance_check(self, text: str) -> str:
        checked = text
        for word in FORBIDDEN_WORDS:
            checked = re.sub(re.escape(word), "***", checked)
        return checked

    def _fallback(self, latency_ms: float, exc: Exception, **kwargs: Any) -> AgentResult:
        products: list[Product] = kwargs.get("products", [])
        copies = [
            MarketingCopy(
                product_id=product.product_id,
                text=f"{product.name} 为你精选，兼顾品质与实用，适合当前浏览偏好。",
            ).model_dump(mode="json")
            for product in products
        ]
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            data={
                "copies": copies,
                "template": "fallback",
                "copy_count": len(copies),
                "mode": "agent_fallback",
            },
            confidence=0.4,
        )
