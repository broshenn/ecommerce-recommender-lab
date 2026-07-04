from __future__ import annotations

import re
import os
from typing import Any

from app.agents.base_agent import BaseAgent
from app.catalog import list_products
from app.models import AgentResult, ChatMessage, ConversationState, IntentResult
from app.services import llm_client

INTENT_PROMPT = """You are an intent classifier for a conversational commerce agent.
Return JSON with these fields:
- intent: one of recommend_products, refine_preferences, compare_products, explain_recommendation, record_feedback, ask_product, smalltalk
- slots: object with optional shopping_goal, budget_min, budget_max, preferred_categories, liked_brands, preferred_tags, rejected_reasons, event_type
- product_refs: product references mentioned by the user, such as "first", "second", "第一个", "第二个"
- needs_recommendation: boolean
- confidence: number from 0 to 1

Do not invent discounts, inventory, or facts. Extract only what the user said."""


class IntentAgent(BaseAgent):
    """Classifies chat intent and extracts slots, with rule fallback."""

    def __init__(self):
        super().__init__(name="intent", timeout=8.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        message: str = kwargs["message"]
        state: ConversationState = kwargs["state"]
        recent_messages: list[ChatMessage] = kwargs.get("recent_messages", [])

        result = self._llm_intent(message, state, recent_messages)
        if result is None:
            result = self._rule_intent(message, state)

        return AgentResult(
            agent_name=self.name,
            success=True,
            data=result.model_dump(mode="json"),
            confidence=result.confidence,
        )

    def _llm_intent(
        self,
        message: str,
        state: ConversationState,
        recent_messages: list[ChatMessage],
    ) -> IntentResult | None:
        if os.getenv("CHAT_LLM_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            return None
        history = "\n".join(
            f"{item.role}: {item.content}"
            for item in recent_messages[-8:]
        )
        raw = llm_client.chat_json(
            system_prompt=INTENT_PROMPT,
            user_message="\n".join(
                [
                    f"Current state: {state.model_dump(mode='json')}",
                    f"Recent messages:\n{history or '(none)'}",
                    f"User message: {message}",
                ]
            ),
            default=None,
        )
        if not isinstance(raw, dict):
            return None
        try:
            result = IntentResult.model_validate(raw)
        except ValueError:
            return None
        result.source = "llm"
        return result

    def _rule_intent(self, message: str, state: ConversationState) -> IntentResult:
        text = message.strip()
        lowered = text.lower()
        if self._is_meta_smalltalk(text):
            return IntentResult(
                intent="smalltalk",
                slots={},
                product_refs=[],
                needs_recommendation=False,
                confidence=0.92,
                source="rule",
            )

        slots = self._extract_slots(text)
        product_refs = self._extract_product_refs(text, state)
        confidence = 0.72
        needs_recommendation = False

        if self._has_any(
            text,
            ["比较", "对比", "哪个好", "哪款好", "哪个更", "哪款更", "区别"],
        ) or "compare" in lowered:
            intent = "compare_products"
        elif self._has_any(text, ["为什么", "原因", "解释", "为啥", "推荐理由"]) or "why" in lowered:
            intent = "explain_recommendation"
        elif self._has_any(
            text,
            ["库存", "有货", "详情", "参数", "评分", "评价", "价格多少", "多少钱", "价格"],
        ):
            intent = "ask_product"
        elif self._looks_like_recommendation(text):
            intent = "recommend_products"
            needs_recommendation = True
            confidence = 0.82
        elif self._has_any(
            text,
            ["喜欢", "不喜欢", "不要", "太贵", "便宜", "换", "购买", "买了", "加入购物车", "下单"],
        ):
            intent = "record_feedback"
            needs_recommendation = self._has_any(text, ["换", "便宜", "重新", "再来", "替换"])
            slots.update(self._feedback_slots(text))
        elif slots:
            intent = "refine_preferences"
            needs_recommendation = True
        else:
            intent = "smalltalk"
            confidence = 0.65

        if intent == "record_feedback" and product_refs:
            slots["product_refs"] = product_refs

        return IntentResult(
            intent=intent,
            slots=slots,
            product_refs=product_refs,
            needs_recommendation=needs_recommendation,
            confidence=confidence,
            source="rule",
        )

    def _extract_slots(self, text: str) -> dict[str, Any]:
        products = list_products()
        categories = self._unique(product.category for product in products)
        brands = self._unique(product.brand for product in products)
        tags = self._unique(tag for product in products for tag in product.tags)
        slots: dict[str, Any] = {}

        budget = self._extract_budget(text)
        if budget:
            slots.update(budget)

        matched_categories = [category for category in categories if category and category in text]
        matched_brands = [brand for brand in brands if brand and brand.lower() in text.lower()]
        matched_tags = [tag for tag in tags if tag and tag in text]
        product_synonyms = {
            "电脑": {"categories": ["电子数码"], "tags": ["电脑配件"]},
            "笔记本": {"categories": ["电子数码"], "tags": ["电脑配件"]},
            "键盘": {"categories": ["电子数码"], "tags": ["电脑配件", "键盘"]},
            "摄像头": {"categories": ["电子数码"], "tags": ["摄像头"]},
            "相机": {"categories": ["电子数码"], "tags": ["摄像头"]},
            "耳机": {"categories": ["电子数码"], "tags": ["耳机"]},
            "耳麦": {"categories": ["电子数码"], "tags": ["耳机"]},
            "手机": {"categories": ["手机"], "tags": ["手机配件"]},
            "保护壳": {"categories": ["手机"], "tags": ["手机配件", "保护壳"]},
            "保护膜": {"categories": ["手机"], "tags": ["手机配件", "保护膜"]},
            "数据线": {"categories": ["电子数码"], "tags": ["数据线"]},
        }
        for keyword, mapped in product_synonyms.items():
            if keyword in text:
                matched_categories.extend(mapped["categories"])
                matched_tags.extend(mapped["tags"])

        generic_terms = [
            "耳机",
            "通勤",
            "防水",
            "轻便",
            "办公",
            "游戏",
            "保护壳",
            "保护膜",
            "数据线",
            "电脑配件",
            "键盘",
            "摄像头",
            "电子产品",
            "PlayStation",
            "任天堂",
        ]
        for term in generic_terms:
            if term in text and term not in matched_tags and term not in matched_categories:
                matched_tags.append(term)

        matched_categories = self._unique(matched_categories)
        matched_tags = self._unique(
            tag for tag in matched_tags
            if tag not in matched_categories
        )

        if matched_categories:
            slots["preferred_categories"] = matched_categories[:5]
        if matched_brands:
            slots["liked_brands"] = matched_brands[:5]
        if matched_tags:
            slots["preferred_tags"] = matched_tags[:8]

        goal = self._extract_goal(text, matched_categories, matched_tags)
        if goal:
            slots["shopping_goal"] = goal

        return slots

    def _extract_budget(self, text: str) -> dict[str, float] | None:
        range_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:元|块|块钱|人民币|rmb|RMB)?\s*(?:-|~|到|至|—)\s*"
            r"(\d+(?:\.\d+)?)\s*(?:元|块|块钱|人民币|rmb|RMB)?",
            text,
        )
        if range_match:
            left = float(range_match.group(1))
            right = float(range_match.group(2))
            return {
                "budget_min": min(left, right),
                "budget_max": max(left, right),
            }

        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:元|块|块钱|人民币|rmb|RMB)?", text)
        if not match:
            return None
        value = float(match.group(1))
        if any(marker in text for marker in ["以上", "起", "至少", "不低于", "不少于"]):
            return {"budget_min": value}
        if any(marker in text for marker in ["以内", "以下", "不超过", "别超过", "低于", "最多"]):
            return {"budget_max": value}
        return {"budget_max": value}

    def _extract_goal(
        self,
        text: str,
        categories: list[str],
        tags: list[str],
    ) -> str:
        if categories or tags:
            return " ".join(self._unique([*categories, *tags])[:4])
        match = re.search(r"(?:想买|想要|帮我找|推荐)(.+)", text)
        if match:
            return match.group(1).strip(" ，。,.")
        return ""

    def _extract_product_refs(self, text: str, state: ConversationState) -> list[str]:
        refs = []
        aliases = {
            "第一个": ["第一个", "第一款", "1号", "一号", "first"],
            "第二个": ["第二个", "第二款", "2号", "二号", "second"],
            "第三个": ["第三个", "第三款", "3号", "三号", "third"],
            "这款": ["这款", "这个", "刚才那个"],
        }
        for canonical, values in aliases.items():
            if any(value in text.lower() for value in values):
                product_id = state.active_product_refs.get(canonical)
                refs.append(product_id or canonical)
        return self._unique(refs)

    def _feedback_slots(self, text: str) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        reasons = []
        if "太贵" in text or "便宜" in text:
            reasons.append("too_expensive")
        if "不喜欢" in text or "不要" in text:
            slots["event_type"] = "dislike"
            reasons.append("disliked")
        elif "喜欢" in text:
            slots["event_type"] = "like"
        elif "购买" in text or "买了" in text or "加入购物车" in text or "下单" in text:
            slots["event_type"] = "purchase"
        if reasons:
            slots["rejected_reasons"] = reasons
        return slots

    def _looks_like_recommendation(self, text: str) -> bool:
        return self._has_any(
            text,
            [
                "推荐",
                "找",
                "想买",
                "想要",
                "想看",
                "有没有",
                "来个",
                "买个",
                "帮我挑",
                "适合",
                "筛",
                "看看",
            ],
        )

    def _is_meta_smalltalk(self, text: str) -> bool:
        lowered = text.lower()
        meta_markers = [
            "你好",
            "你是谁",
            "你是什么",
            "什么agent",
            "什么 agent",
            "介绍一下",
            "你能做什么",
            "能说画面",
            "看画面",
            "看图片",
            "看截图",
            "说画面",
        ]
        return any(marker in lowered for marker in meta_markers)

    def _has_any(self, text: str, markers: list[str]) -> bool:
        lowered = text.lower()
        return any(marker.lower() in lowered for marker in markers)

    def _unique(self, values) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value and value not in seen:
                result.append(value)
                seen.add(value)
        return result
