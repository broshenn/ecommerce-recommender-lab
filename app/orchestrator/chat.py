from __future__ import annotations

import json
import time
from typing import Any, Iterable

from app.behavior import build_user_profile
from app.agents import DialogueAgent, IntentAgent
from app.models import (
    AgentResult,
    ChatRequest,
    ChatResponse,
    ConversationState,
    IntentResult,
    RecommendRequest,
    RecommendedProduct,
)
from app.services.memory import MemoryService
from app.tools import BusinessToolContext, ToolRouter


class ChatOrchestrator:
    """Outer conversational shell around the existing recommendation graph."""

    def __init__(self):
        self.memory = MemoryService()
        self.intent_agent = IntentAgent()
        self.dialogue_agent = DialogueAgent()
        self.tool_router = ToolRouter()

    def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        state = self.memory.get_or_create_state(
            user_id=request.user_id,
            session_id=request.session_id,
        )
        memory_summary = self.memory.user_memory_summary(request.user_id)
        behavior_profile = build_user_profile(request.user_id)
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
            intent_mode=request.intent_mode,
        )
        intent_result = IntentResult.model_validate(intent_agent_result.data)
        intent_result, intent_agent_result = self._enrich_intent_with_memory(
            intent_result=intent_result,
            intent_agent_result=intent_agent_result,
            message=request.message,
            state=state,
            memory_summary=memory_summary,
            behavior_profile=behavior_profile,
        )
        agent_results["intent"] = intent_agent_result
        trace.append(
            {
                "step": "intent",
                "intent": intent_result.intent,
                "source": intent_result.source,
                "intent_mode": request.intent_mode,
                "latency_ms": round(intent_agent_result.latency_ms, 2),
            }
        )
        if self._has_memory_summary(memory_summary):
            trace.append(
                {
                    "step": "memory",
                    "source": "sqlite_user_memory_facts",
                    "summary": self._compact_memory_summary(memory_summary),
                }
            )
        intent_memory = intent_agent_result.data.get("rule_debug", {}).get(
            "memory_enrichment",
            {},
        )
        if intent_memory:
            trace.append(
                {
                    "step": "query_memory",
                    "source": "conversation+long_term+behavior_profile",
                    "summary": intent_memory,
                }
            )

        self._apply_intent_to_state(state, intent_result)
        resolved_product_ids = self._resolve_product_refs(state, intent_result.product_refs)
        should_recommend = self._should_recommend(intent_result)
        tool_context = BusinessToolContext(
            user_id=request.user_id,
            state=state,
            intent_result=intent_result,
            resolved_product_ids=resolved_product_ids,
            recommend_request=self._recommend_request_from_state(
                state,
                memory_summary,
                force_experiment_group=request.force_experiment_group,
            ),
        )
        for tool in self.tool_router.route(intent_result, should_recommend=should_recommend):
            tool_result = tool.run(tool_context)
            trace.append(tool_result.observation.to_trace())
            extra.update(tool_result.extra)
            agent_results.update(tool_result.agent_results)
            if tool_result.products:
                products = tool_result.products
                marketing_copies = tool_result.marketing_copies
                state.last_recommended_product_ids = [
                    product.product_id for product in products
                ]
                state.active_product_refs = self._build_product_refs(products)

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

    def _enrich_intent_with_memory(
        self,
        *,
        intent_result: IntentResult,
        intent_agent_result: AgentResult,
        message: str,
        state: ConversationState,
        memory_summary: dict[str, Any],
        behavior_profile,
    ) -> tuple[IntentResult, AgentResult]:
        is_continuation = self._is_continuation_message(message)
        if intent_result.intent == "smalltalk" and is_continuation:
            intent_result = intent_result.model_copy(
                update={
                    "intent": "recommend_products",
                    "needs_recommendation": True,
                    "confidence": max(intent_result.confidence, 0.76),
                    "source": f"{intent_result.source}+continuation",
                }
            )
        if intent_result.intent not in {"recommend_products", "refine_preferences"}:
            return intent_result, intent_agent_result
        explicit_slots = self._business_slot_keys(intent_result.slots)
        if explicit_slots and not is_continuation:
            return intent_result, intent_agent_result

        merged_slots: dict[str, Any] = {}
        sources: list[dict[str, Any]] = []
        self._merge_memory_source(
            merged_slots,
            sources,
            "short_term_session",
            {
                "shopping_goal": state.shopping_goal,
                "budget_min": state.budget_min,
                "budget_max": state.budget_max,
                "preferred_categories": state.preferred_categories,
                "liked_brands": state.liked_brands,
                "preferred_tags": state.preferred_tags,
            },
        )
        self._merge_memory_source(
            merged_slots,
            sources,
            "long_term_memory",
            memory_summary,
        )
        self._merge_memory_source(
            merged_slots,
            sources,
            "behavior_profile",
            {
                "preferred_categories": behavior_profile.preferred_categories,
                "liked_brands": behavior_profile.liked_brands,
                "preferred_tags": behavior_profile.preferred_tags,
                "recent_views": behavior_profile.recent_views,
            },
        )
        if not merged_slots:
            return intent_result, intent_agent_result

        enriched = intent_result.model_copy(
            update={
                "slots": {**intent_result.slots, **merged_slots},
                "needs_recommendation": True,
                "source": f"{intent_result.source}+memory",
                "confidence": max(intent_result.confidence, 0.78),
            }
        )
        data = dict(intent_agent_result.data)
        rule_debug = dict(data.get("rule_debug", {}))
        rule_debug["memory_enrichment"] = {
            "applied": True,
            "reason": "empty_business_slots_continuation",
            "sources": sources,
            "slot_keys": sorted(merged_slots.keys()),
        }
        data.update(enriched.model_dump(mode="json"))
        data["rule_debug"] = rule_debug
        return enriched, intent_agent_result.model_copy(
            update={"data": data, "confidence": enriched.confidence}
        )

    def _is_continuation_message(self, message: str) -> bool:
        text = message.strip().lower()
        markers = [
            "再推荐",
            "再来",
            "再给",
            "继续",
            "还有",
            "换几个",
            "换几款",
            "推荐几个",
            "推荐几款",
            "more",
            "again",
        ]
        return any(marker in text for marker in markers)

    def _business_slot_keys(self, slots: dict[str, Any]) -> set[str]:
        business_keys = {
            "shopping_goal",
            "budget_min",
            "budget_max",
            "preferred_categories",
            "liked_brands",
            "preferred_tags",
        }
        return {
            key
            for key, value in slots.items()
            if key in business_keys and value not in (None, "", [])
        }

    def _merge_memory_source(
        self,
        merged_slots: dict[str, Any],
        sources: list[dict[str, Any]],
        source: str,
        data: dict[str, Any],
    ) -> None:
        applied: dict[str, Any] = {}
        for key in (
            "shopping_goal",
            "budget_min",
            "budget_max",
            "preferred_categories",
            "liked_brands",
            "preferred_tags",
            "recent_views",
        ):
            value = data.get(key)
            if value in (None, "", []):
                continue
            if key == "recent_views":
                applied[key] = value[:3] if isinstance(value, list) else value
                continue
            if isinstance(value, list):
                existing = merged_slots.get(key, [])
                merged_slots[key] = self._merge(existing, value)
                applied[key] = value[:5]
            elif key not in merged_slots or merged_slots.get(key) in (None, ""):
                merged_slots[key] = value
                applied[key] = value
        if applied:
            sources.append({"source": source, "slots": applied})

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

    def _recommend_request_from_state(
        self,
        state: ConversationState,
        memory_summary: dict[str, Any] | None = None,
        force_experiment_group: str | None = None,
    ) -> RecommendRequest:
        memory_summary = memory_summary or {}
        shopping_goal = state.shopping_goal or memory_summary.get("shopping_goal", "")
        preferred_categories = state.preferred_categories or memory_summary.get(
            "preferred_categories",
            [],
        )
        liked_brands = state.liked_brands or memory_summary.get("liked_brands", [])
        preferred_tags = state.preferred_tags or memory_summary.get("preferred_tags", [])
        budget_min = state.budget_min
        if budget_min is None:
            budget_min = memory_summary.get("budget_min")
        budget_max = state.budget_max
        if budget_max is None:
            budget_max = memory_summary.get("budget_max")
        context = {
            "shopping_goal": shopping_goal,
            "conversation_session_id": state.session_id,
            "long_term_memory": self._compact_memory_summary(memory_summary),
        }
        if force_experiment_group in {"control", "treatment"}:
            context["force_experiment_group"] = force_experiment_group
        return RecommendRequest(
            user_id=state.user_id,
            scene="chat",
            num_items=3,
            preferred_categories=preferred_categories,
            liked_brands=liked_brands,
            preferred_tags=preferred_tags,
            budget_min=budget_min,
            budget_max=budget_max,
            disliked_products=state.disliked_products,
            context=context,
        )

    def _should_recommend(self, intent_result: IntentResult) -> bool:
        return intent_result.needs_recommendation or intent_result.intent in {
            "recommend_products",
            "refine_preferences",
        }

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

    def _has_memory_summary(self, memory_summary: dict[str, Any]) -> bool:
        return any(
            value not in (None, "", [])
            for value in memory_summary.values()
        )

    def _compact_memory_summary(self, memory_summary: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value[:5] if isinstance(value, list) else value
            for key, value in memory_summary.items()
            if value not in (None, "", [])
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
