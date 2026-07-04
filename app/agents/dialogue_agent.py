from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from app.agents.base_agent import BaseAgent
from app.models import AgentResult, ConversationState, MarketingCopy, RecommendedProduct
from app.services import llm_client

DIALOGUE_PROMPT = """You are a professional, concise e-commerce shopping assistant.
Speak in Chinese. Be friendly but not pushy. Do not claim discounts, lowest price,
medical effects, guarantees, or inventory facts that are not present in the data.
Keep the reply under 80 Chinese characters unless comparing products."""


class DialogueAgent(BaseAgent):
    """Generates user-facing chat replies with safe rule fallback."""

    def __init__(self):
        super().__init__(name="dialogue", timeout=8.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        intent: str = kwargs["intent"]
        state: ConversationState = kwargs["state"]
        products: list[RecommendedProduct] = kwargs.get("products", [])
        marketing_copies: list[MarketingCopy] = kwargs.get("marketing_copies", [])
        extra: dict[str, Any] = kwargs.get("extra", {})
        message: str = kwargs.get("message", "")

        reply = self._llm_reply(intent, state, products, marketing_copies, extra)
        mode = "llm"
        if not reply:
            reply = self._rule_reply(intent, state, products, marketing_copies, extra, message)
            mode = "rule"

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={"reply": reply, "mode": mode},
            confidence=0.8,
        )

    def _llm_reply(
        self,
        intent: str,
        state: ConversationState,
        products: list[RecommendedProduct],
        marketing_copies: list[MarketingCopy],
        extra: dict[str, Any],
    ) -> str | None:
        if os.getenv("CHAT_LLM_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            return None
        product_lines = [
            f"{index + 1}. {product.name} 品牌:{product.brand} 价格:{product.price} 库存:{product.stock}"
            for index, product in enumerate(products[:3])
        ]
        text = llm_client.chat(
            system_prompt=DIALOGUE_PROMPT,
            user_message="\n".join(
                [
                    f"intent: {intent}",
                    f"state: {state.model_dump(mode='json')}",
                    f"products: {product_lines or '(none)'}",
                    f"marketing copies: {[copy.model_dump(mode='json') for copy in marketing_copies[:3]]}",
                    f"extra: {extra}",
                ]
            ),
            max_tokens=256,
        )
        if not text:
            return None
        cleaned = text.strip()
        return cleaned if self._is_safe_reply(cleaned) else None

    def _is_safe_reply(self, text: str) -> bool:
        if len(text) > 180:
            return False
        leaked_markers = [
            "intent:",
            "state:",
            "products:",
            "marketing copies:",
            "system_prompt",
            "用户意图是",
            "我们被要求",
            "要求：",
            "我应该",
        ]
        lowered = text.lower()
        return not any(marker.lower() in lowered for marker in leaked_markers)

    def _rule_reply(
        self,
        intent: str,
        state: ConversationState,
        products: list[RecommendedProduct],
        marketing_copies: list[MarketingCopy],
        extra: dict[str, Any],
        message: str = "",
    ) -> str:
        if intent in {"recommend_products", "refine_preferences", "record_feedback"} and products:
            goal = state.shopping_goal or "你的需求"
            budget = f"，预算不超过 {state.budget_max:g} 元" if state.budget_max else ""
            return f"我按{goal}{budget}筛了这几款，优先考虑相关度、库存和价格。"
        if intent == "compare_products":
            return extra.get("comparison") or "我可以基于价格、评分、库存和匹配度帮你比较这几款。"
        if intent == "explain_recommendation":
            return extra.get("explanation") or "这款主要因为匹配你的偏好、预算和当前可售状态。"
        if intent == "ask_product":
            return extra.get("answer") or "我先根据当前商品数据回答，复杂参数可以继续追问。"
        if intent == "record_feedback":
            return "收到，我会记住这个反馈，并调整后续推荐。"
        return self._smalltalk_reply(message)

    def _smalltalk_reply(self, message: str) -> str:
        text = message.lower()
        if any(marker in text for marker in ["你是谁", "你是什么", "什么agent", "什么 agent", "你能做什么", "介绍一下"]):
            if "模型" in text:
                mode = "已启用" if os.getenv("CHAT_LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"} else "未启用"
                return f"当前导购对话默认用规则意图识别，CHAT_LLM_ENABLED={mode}；项目配置的大模型是 {llm_client.model}。"
            return (
                "我是对话式电商导购 Agent。"
                "我可以理解预算、品类、品牌和反馈，记住本轮偏好，并调用推荐链路给出商品卡片。"
            )
        if any(marker in text for marker in ["星期几", "周几", "几号", "日期", "今天"]):
            now = datetime.now()
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            return f"按服务端当前时间，今天是 {now:%Y-%m-%d}，{weekdays[now.weekday()]}。"
        if any(marker in text for marker in ["画面", "图片", "截图", "看图"]):
            return (
                "在这个网页里我不能直接感知屏幕画面，"
                "但可以根据你发来的文字、商品卡片和截图描述帮你分析。"
            )
        if any(marker in text for marker in ["你好", "hello", "hi"]):
            return "你好，我是你的电商导购。你可以告诉我想买什么、预算多少、用途是什么。"
        return "我主要负责电商导购和推荐。你可以直接说需求，也可以问我为什么推荐某款商品。"
