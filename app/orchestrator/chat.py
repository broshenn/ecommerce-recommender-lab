from __future__ import annotations

import json
import time
from typing import Any, Iterable

from app.agents import DialogueAgent, IntentAgent
from app.behavior import record_event
from app.catalog import list_products
from app.models import (
    AgentResult,
    ChatRequest,
    ChatResponse,
    ConversationState,
    IntentResult,
    Product,
    RecommendRequest,
    RecommendedProduct,
    UserEventCreate,
)
from app.orchestrator.graph import recommend_with_graph
from app.services.memory import MemoryService


class ChatOrchestrator:
    """Outer conversational shell around the existing recommendation graph."""

    def __init__(self):
        self.memory = MemoryService()
        self.intent_agent = IntentAgent()
        self.dialogue_agent = DialogueAgent()

    def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        state = self.memory.get_or_create_state(
            user_id=request.user_id,
            session_id=request.session_id,
        )
        recent_messages = self.memory.recent_messages(state.session_id)
        self.memory.append_message(state.session_id, "user", request.message)

        trace: list[dict[str, Any]] = []
        agent_results: dict[str, AgentResult] = {}
        products: list[RecommendedProduct] = []
        marketing_copies = []
        extra: dict[str, Any] = {}

        intent_agent_result = self.intent_agent.run(
            message=request.message,
            state=state,
            recent_messages=recent_messages,
        )
        agent_results["intent"] = intent_agent_result
        intent_result = IntentResult.model_validate(intent_agent_result.data)
        trace.append(
            {
                "step": "intent",
                "intent": intent_result.intent,
                "source": intent_result.source,
                "latency_ms": round(intent_agent_result.latency_ms, 2),
            }
        )

        self._apply_intent_to_state(state, intent_result)
        resolved_product_ids = self._resolve_product_refs(state, intent_result.product_refs)

        if intent_result.intent == "record_feedback":
            extra["feedback"] = self._record_feedback(
                user_id=request.user_id,
                product_ids=resolved_product_ids,
                event_type=intent_result.slots.get("event_type"),
            )

        if self._should_recommend(intent_result):
            recommend_response = recommend_with_graph(self._recommend_request_from_state(state))
            products = recommend_response.products
            marketing_copies = recommend_response.marketing_copies
            agent_results.update(recommend_response.agent_results)
            state.last_recommended_product_ids = [product.product_id for product in products]
            state.active_product_refs = self._build_product_refs(products)
            trace.append(
                {
                    "step": "recommend_graph",
                    "strategy": recommend_response.strategy,
                    "product_count": len(products),
                    "experiment_group": recommend_response.experiment_group,
                }
            )
        elif intent_result.intent == "compare_products":
            extra["comparison"] = self._compare_products(state, resolved_product_ids)
        elif intent_result.intent == "explain_recommendation":
            extra["explanation"] = self._explain_product(state, resolved_product_ids)
        elif intent_result.intent == "ask_product":
            extra["answer"] = self._answer_product_question(state, resolved_product_ids)

        dialogue_result = self.dialogue_agent.run(
            intent=intent_result.intent,
            state=state,
            products=products,
            marketing_copies=marketing_copies,
            extra=extra,
            message=request.message,
        )
        agent_results["dialogue"] = dialogue_result
        reply = dialogue_result.data.get("reply", "")
        trace.append(
            {
                "step": "dialogue",
                "mode": dialogue_result.data.get("mode"),
                "latency_ms": round(dialogue_result.latency_ms, 2),
            }
        )

        self.memory.save_state(state)
        self.memory.append_message(state.session_id, "assistant", reply)
        self.memory.record_memory_facts(
            user_id=request.user_id,
            facts=self._facts_from_state(state),
            source="chat",
        )
        trace.append(
            {
                "step": "done",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )

        return ChatResponse(
            session_id=state.session_id,
            intent=intent_result.intent,
            reply=reply,
            state=state,
            products=products,
            marketing_copies=marketing_copies,
            agent_results=agent_results,
            trace=trace,
        )

    def stream_chat(self, request: ChatRequest) -> Iterable[str]:
        try:
            response = self.chat(request)
            yield self._sse("state", response.state.model_dump(mode="json"))
            for char in response.reply:
                yield self._sse("token", {"text": char})
            yield self._sse(
                "products",
                {
                    "products": [product.model_dump(mode="json") for product in response.products],
                    "marketing_copies": [
                        copy.model_dump(mode="json")
                        for copy in response.marketing_copies
                    ],
                },
            )
            for item in response.trace:
                yield self._sse("trace", item)
            yield self._sse("done", response.model_dump(mode="json"))
        except Exception as exc:
            yield self._sse("error", {"error": str(exc)})

    def _apply_intent_to_state(
        self,
        state: ConversationState,
        intent_result: IntentResult,
    ) -> None:
        state.recent_intents = self._merge(
            [*state.recent_intents, intent_result.intent],
            [],
        )[-8:]
        if intent_result.intent in {
            "smalltalk",
            "compare_products",
            "explain_recommendation",
            "ask_product",
        }:
            return

        slots = intent_result.slots
        state.shopping_goal = slots.get("shopping_goal") or state.shopping_goal
        state.budget_min = self._coalesce_number(slots.get("budget_min"), state.budget_min)
        state.budget_max = self._coalesce_number(slots.get("budget_max"), state.budget_max)
        explicit_categories = slots.get("preferred_categories", [])
        explicit_tags = slots.get("preferred_tags", [])
        explicit_brands = slots.get("liked_brands", [])
        should_replace_goal = (
            intent_result.intent == "recommend_products"
            and bool(explicit_categories or explicit_tags)
        )
        if should_replace_goal:
            state.preferred_categories = self._unique_strings(explicit_categories)
            state.preferred_tags = self._unique_strings(explicit_tags)
            state.liked_brands = self._unique_strings(explicit_brands)
        else:
            state.preferred_categories = self._merge(
                state.preferred_categories,
                explicit_categories,
            )
            state.liked_brands = self._merge(state.liked_brands, explicit_brands)
            state.preferred_tags = self._merge(state.preferred_tags, explicit_tags)
        state.disliked_products = self._merge(
            state.disliked_products,
            slots.get("disliked_products", []),
        )
        state.rejected_reasons = self._merge(
            state.rejected_reasons,
            slots.get("rejected_reasons", []),
        )

    def _recommend_request_from_state(self, state: ConversationState) -> RecommendRequest:
        return RecommendRequest(
            user_id=state.user_id,
            scene="chat",
            num_items=3,
            preferred_categories=state.preferred_categories,
            liked_brands=state.liked_brands,
            preferred_tags=state.preferred_tags,
            budget_min=state.budget_min,
            budget_max=state.budget_max,
            disliked_products=state.disliked_products,
            context={
                "shopping_goal": state.shopping_goal,
                "conversation_session_id": state.session_id,
                "force_experiment_group": "control",
            },
        )

    def _should_recommend(self, intent_result: IntentResult) -> bool:
        return intent_result.needs_recommendation or intent_result.intent in {
            "recommend_products",
            "refine_preferences",
        }

    def _record_feedback(
        self,
        *,
        user_id: str,
        product_ids: list[str],
        event_type: str | None,
    ) -> dict[str, Any]:
        if event_type not in {"like", "dislike", "purchase", "view"}:
            event_type = "dislike" if product_ids else None
        recorded = []
        if event_type:
            for product_id in product_ids[:3]:
                record_event(
                    UserEventCreate(
                        user_id=user_id,
                        product_id=product_id,
                        event_type=event_type,
                    )
                )
                recorded.append({"product_id": product_id, "event_type": event_type})
        return {"recorded": recorded}

    def _compare_products(
        self,
        state: ConversationState,
        product_ids: list[str],
    ) -> str:
        products = self._products_by_refs(state, product_ids, fallback_count=2)
        if len(products) < 2:
            return "我还没有足够的商品可比较，可以先让我推荐几款。"
        lines = []
        for product in products[:3]:
            lines.append(
                f"{product.name}：价格 {product.price:g} 元，评分 {product.rating or '-'}，库存 {product.stock}。"
            )
        return " ".join(lines)

    def _explain_product(
        self,
        state: ConversationState,
        product_ids: list[str],
    ) -> str:
        product = next(iter(self._products_by_refs(state, product_ids, fallback_count=1)), None)
        if not product:
            return "我还没有可解释的商品，可以先发起一次推荐。"
        matched = []
        if product.category in state.preferred_categories:
            matched.append("品类匹配")
        if product.brand in state.liked_brands:
            matched.append("品牌匹配")
        if set(product.tags) & set(state.preferred_tags):
            matched.append("标签匹配")
        if state.budget_max is None or product.price <= state.budget_max:
            matched.append("预算友好")
        return f"{product.name} 的推荐依据是：{', '.join(matched) or '综合评分较好'}。"

    def _answer_product_question(
        self,
        state: ConversationState,
        product_ids: list[str],
    ) -> str:
        product = next(iter(self._products_by_refs(state, product_ids, fallback_count=1)), None)
        if not product:
            return "我还没有定位到具体商品，可以先说“第一款”或让我推荐几款。"
        return (
            f"{product.name}，品牌 {product.brand}，价格 {product.price:g} 元，"
            f"库存 {product.stock}，评分 {product.rating or '-'}。"
        )

    def _products_by_refs(
        self,
        state: ConversationState,
        product_ids: list[str],
        *,
        fallback_count: int,
    ) -> list[Product]:
        ids = product_ids or state.last_recommended_product_ids[:fallback_count]
        products_by_id = {product.product_id: product for product in list_products()}
        return [products_by_id[product_id] for product_id in ids if product_id in products_by_id]

    def _resolve_product_refs(self, state: ConversationState, refs: list[str]) -> list[str]:
        resolved = []
        for ref in refs:
            product_id = state.active_product_refs.get(ref, ref)
            if product_id in state.last_recommended_product_ids and product_id not in resolved:
                resolved.append(product_id)
        return resolved

    def _build_product_refs(self, products: list[RecommendedProduct]) -> dict[str, str]:
        labels = [
            ("第一个", "第一款", "1号", "first"),
            ("第二个", "第二款", "2号", "second"),
            ("第三个", "第三款", "3号", "third"),
        ]
        refs: dict[str, str] = {}
        for index, product in enumerate(products[:3]):
            for label in labels[index]:
                refs[label] = product.product_id
        if products:
            refs["这款"] = products[0].product_id
            refs["这个"] = products[0].product_id
            refs["刚才那个"] = products[0].product_id
        return refs

    def _facts_from_state(self, state: ConversationState) -> dict[str, Any]:
        return {
            "shopping_goal": state.shopping_goal,
            "budget_min": state.budget_min,
            "budget_max": state.budget_max,
            "preferred_category": state.preferred_categories,
            "liked_brand": state.liked_brands,
            "preferred_tag": state.preferred_tags,
            "rejected_reason": state.rejected_reasons,
        }

    def _coalesce_number(self, value: Any, fallback: float | None) -> float | None:
        if value in (None, ""):
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _merge(self, first: list[str], second: list[str]) -> list[str]:
        return self._unique_strings([*first, *second])

    def _unique_strings(self, values) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value and value not in seen:
                result.append(str(value))
                seen.add(str(value))
        return result

    def _sse(self, event: str, payload: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


chat_orchestrator = ChatOrchestrator()
