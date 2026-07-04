from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.behavior import record_event
from app.catalog import list_products
from app.models import (
    AgentResult,
    ConversationState,
    IntentResult,
    MarketingCopy,
    Product,
    RecommendRequest,
    RecommendedProduct,
    UserEventCreate,
)
from app.orchestrator.graph import recommend_with_graph


@dataclass
class ToolObservation:
    tool_name: str
    success: bool
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None

    def to_trace(self) -> dict[str, Any]:
        return {
            "step": "tool",
            "tool_name": self.tool_name,
            "success": self.success,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


@dataclass
class BusinessToolResult:
    observation: ToolObservation
    products: list[RecommendedProduct] = field(default_factory=list)
    marketing_copies: list[MarketingCopy] = field(default_factory=list)
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BusinessToolContext:
    user_id: str
    state: ConversationState
    intent_result: IntentResult
    resolved_product_ids: list[str]
    recommend_request: RecommendRequest


class BusinessTool:
    tool_name = "BusinessTool"

    def run(self, context: BusinessToolContext) -> BusinessToolResult:
        started = time.perf_counter()
        try:
            result = self._run(context)
            result.observation.latency_ms = (time.perf_counter() - started) * 1000
            return result
        except Exception as exc:
            return BusinessToolResult(
                observation=ToolObservation(
                    tool_name=self.tool_name,
                    success=False,
                    input_summary=self._input_summary(context),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=str(exc),
                )
            )

    def _run(self, context: BusinessToolContext) -> BusinessToolResult:
        raise NotImplementedError

    def _input_summary(self, context: BusinessToolContext) -> dict[str, Any]:
        return {
            "intent": context.intent_result.intent,
            "slots": context.intent_result.slots,
            "product_ids": context.resolved_product_ids,
        }


class PreferenceUpdateTool(BusinessTool):
    tool_name = "PreferenceUpdateTool"

    def _run(self, context: BusinessToolContext) -> BusinessToolResult:
        state = context.state
        return BusinessToolResult(
            observation=ToolObservation(
                tool_name=self.tool_name,
                success=True,
                input_summary={
                    "intent": context.intent_result.intent,
                    "slots": context.intent_result.slots,
                },
                output_summary={
                    "shopping_goal": state.shopping_goal,
                    "budget_min": state.budget_min,
                    "budget_max": state.budget_max,
                    "preferred_categories": state.preferred_categories,
                    "liked_brands": state.liked_brands,
                    "preferred_tags": state.preferred_tags,
                    "rejected_reasons": state.rejected_reasons,
                },
            )
        )


class RecommendGraphTool(BusinessTool):
    tool_name = "RecommendGraphTool"

    def _run(self, context: BusinessToolContext) -> BusinessToolResult:
        response = recommend_with_graph(context.recommend_request)
        return BusinessToolResult(
            observation=ToolObservation(
                tool_name=self.tool_name,
                success=True,
                input_summary={
                    "user_id": context.recommend_request.user_id,
                    "scene": context.recommend_request.scene,
                    "categories": context.recommend_request.preferred_categories,
                    "brands": context.recommend_request.liked_brands,
                    "tags": context.recommend_request.preferred_tags,
                    "budget_min": context.recommend_request.budget_min,
                    "budget_max": context.recommend_request.budget_max,
                },
                output_summary={
                    "strategy": response.strategy,
                    "product_count": len(response.products),
                    "experiment_group": response.experiment_group,
                },
            ),
            products=response.products,
            marketing_copies=response.marketing_copies,
            agent_results=response.agent_results,
            extra={
                "recommend_strategy": response.strategy,
                "experiment_group": response.experiment_group,
            },
        )


class FeedbackTool(BusinessTool):
    tool_name = "FeedbackTool"

    def _run(self, context: BusinessToolContext) -> BusinessToolResult:
        event_type = context.intent_result.slots.get("event_type")
        if event_type not in {"like", "dislike", "purchase", "view"}:
            event_type = "dislike" if context.resolved_product_ids else None
        recorded = []
        products_by_id = {product.product_id: product for product in list_products()}
        if event_type:
            for product_id in context.resolved_product_ids[:3]:
                record_event(
                    UserEventCreate(
                        user_id=context.user_id,
                        product_id=product_id,
                        event_type=event_type,
                    )
                )
                recorded.append({"product_id": product_id, "event_type": event_type})
                self._update_state_from_feedback(
                    context.state,
                    products_by_id.get(product_id),
                    product_id,
                    event_type,
                )
        return BusinessToolResult(
            observation=ToolObservation(
                tool_name=self.tool_name,
                success=True,
                input_summary={
                    "product_ids": context.resolved_product_ids,
                    "event_type": context.intent_result.slots.get("event_type"),
                },
                output_summary={"recorded_count": len(recorded)},
            ),
            extra={"feedback": {"recorded": recorded}},
        )

    def _update_state_from_feedback(
        self,
        state: ConversationState,
        product: Product | None,
        product_id: str,
        event_type: str,
    ) -> None:
        if event_type == "dislike":
            state.disliked_products = _unique([*state.disliked_products, product_id])
            return
        if event_type not in {"like", "purchase"} or not product:
            return
        state.preferred_categories = _unique([*state.preferred_categories, product.category])
        state.liked_brands = _unique([*state.liked_brands, product.brand])
        state.preferred_tags = _unique([*state.preferred_tags, *product.tags])


class CompareProductTool(BusinessTool):
    tool_name = "CompareProductTool"

    def _run(self, context: BusinessToolContext) -> BusinessToolResult:
        products = _products_by_refs(context.state, context.resolved_product_ids, fallback_count=2)
        if len(products) < 2:
            comparison = "我还没有足够的商品可比较，可以先让我推荐几款。"
        else:
            lines = []
            for product in products[:3]:
                lines.append(
                    f"{product.name}：价格 {product.price:g} 元，评分 {product.rating or '-'}，库存 {product.stock}。"
                )
            comparison = " ".join(lines)
        return BusinessToolResult(
            observation=ToolObservation(
                tool_name=self.tool_name,
                success=True,
                input_summary={"product_ids": context.resolved_product_ids},
                output_summary={"compared_count": len(products)},
            ),
            extra={"comparison": comparison},
        )


class ExplainRecommendationTool(BusinessTool):
    tool_name = "ExplainRecommendationTool"

    def _run(self, context: BusinessToolContext) -> BusinessToolResult:
        product = next(
            iter(_products_by_refs(context.state, context.resolved_product_ids, fallback_count=1)),
            None,
        )
        if not product:
            explanation = "我还没有可解释的商品，可以先发起一次推荐。"
        else:
            matched = []
            if product.category in context.state.preferred_categories:
                matched.append("品类匹配")
            if product.brand in context.state.liked_brands:
                matched.append("品牌匹配")
            if set(product.tags) & set(context.state.preferred_tags):
                matched.append("标签匹配")
            if context.state.budget_max is None or product.price <= context.state.budget_max:
                matched.append("预算友好")
            explanation = f"{product.name} 的推荐依据是：{', '.join(matched) or '综合评分较好'}。"
        return BusinessToolResult(
            observation=ToolObservation(
                tool_name=self.tool_name,
                success=True,
                input_summary={"product_ids": context.resolved_product_ids},
                output_summary={"has_product": product is not None},
            ),
            extra={"explanation": explanation},
        )


class ProductInfoTool(BusinessTool):
    tool_name = "ProductInfoTool"

    def _run(self, context: BusinessToolContext) -> BusinessToolResult:
        product = next(
            iter(_products_by_refs(context.state, context.resolved_product_ids, fallback_count=1)),
            None,
        )
        if not product:
            answer = "我还没有定位到具体商品，可以先说“第一款”或让我推荐几款。"
        else:
            answer = (
                f"{product.name}，品牌 {product.brand}，价格 {product.price:g} 元，"
                f"库存 {product.stock}，评分 {product.rating or '-'}。"
            )
        return BusinessToolResult(
            observation=ToolObservation(
                tool_name=self.tool_name,
                success=True,
                input_summary={"product_ids": context.resolved_product_ids},
                output_summary={"has_product": product is not None},
            ),
            extra={"answer": answer},
        )


class SmalltalkTool(BusinessTool):
    tool_name = "SmalltalkTool"

    def _run(self, context: BusinessToolContext) -> BusinessToolResult:
        return BusinessToolResult(
            observation=ToolObservation(
                tool_name=self.tool_name,
                success=True,
                input_summary={"intent": context.intent_result.intent},
                output_summary={"handled_by": "DialogueAgent"},
            )
        )


class ToolRouter:
    def __init__(self):
        self.preference_tool = PreferenceUpdateTool()
        self.recommend_tool = RecommendGraphTool()
        self.feedback_tool = FeedbackTool()
        self.compare_tool = CompareProductTool()
        self.explain_tool = ExplainRecommendationTool()
        self.product_info_tool = ProductInfoTool()
        self.smalltalk_tool = SmalltalkTool()

    def route(self, intent_result: IntentResult, *, should_recommend: bool) -> list[BusinessTool]:
        tools: list[BusinessTool] = []
        if intent_result.intent in {
            "recommend_products",
            "refine_preferences",
            "record_feedback",
        }:
            tools.append(self.preference_tool)
        if intent_result.intent == "record_feedback":
            tools.append(self.feedback_tool)
        if should_recommend:
            tools.append(self.recommend_tool)
            return tools
        if intent_result.intent == "compare_products":
            tools.append(self.compare_tool)
        elif intent_result.intent == "explain_recommendation":
            tools.append(self.explain_tool)
        elif intent_result.intent == "ask_product":
            tools.append(self.product_info_tool)
        elif intent_result.intent == "smalltalk":
            tools.append(self.smalltalk_tool)
        return tools


def _products_by_refs(
    state: ConversationState,
    product_ids: list[str],
    *,
    fallback_count: int,
) -> list[Product]:
    ids = product_ids or state.last_recommended_product_ids[:fallback_count]
    products_by_id = {product.product_id: product for product in list_products()}
    return [products_by_id[product_id] for product_id in ids if product_id in products_by_id]


def _unique(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(str(value))
            seen.add(str(value))
    return result
