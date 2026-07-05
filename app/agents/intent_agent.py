from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Any

from app.agents.base_agent import BaseAgent
from app.catalog import list_products
from app.models import AgentResult, ChatMessage, ConversationState, IntentResult
from app.services import llm_client
from app.services.intent_classifier import intent_model_classifier

INTENT_PROMPT = """You are an intent classifier for a conversational commerce agent.
Return JSON with these fields:
- intent: one of recommend_products, refine_preferences, compare_products, explain_recommendation, record_feedback, ask_product, smalltalk
- slots: object with optional shopping_goal, budget_min, budget_max, preferred_categories, liked_brands, preferred_tags, rejected_reasons, event_type
- product_refs: product references mentioned by the user, such as "first", "second", "第一个", "第二个"
- needs_recommendation: boolean
- confidence: number from 0 to 1

Do not invent discounts, inventory, or facts. Extract only what the user said."""
DEFAULT_RULE_PATH = Path(__file__).with_name("intent_rules.json")
VALID_INTENT_MODES = {"rule", "bert", "llm"}


class IntentAgent(BaseAgent):
    """Classifies chat intent and extracts slots, with rule fallback."""

    def __init__(self):
        super().__init__(name="intent", timeout=8.0)
        self.rules = self._load_rules(DEFAULT_RULE_PATH)

    def _execute(self, **kwargs: Any) -> AgentResult:
        message: str = kwargs["message"]
        state: ConversationState = kwargs["state"]
        recent_messages: list[ChatMessage] = kwargs.get("recent_messages", [])
        intent_mode = kwargs.get("intent_mode", "rule")
        if intent_mode not in VALID_INTENT_MODES:
            intent_mode = "rule"

        self._last_intent_mode = intent_mode
        rule_result = self._rule_intent(message, state)
        rule_debug = getattr(self, "_last_rule_debug", {})
        result = rule_result

        if intent_mode == "bert":
            result = self._apply_model_intent(message, rule_result, force=True)
            result = self._apply_business_guard(result, rule_result, rule_debug)
        elif intent_mode == "llm":
            llm_result = self._llm_intent(message, state, recent_messages, force=True)
            if llm_result is None:
                self._last_rule_debug = {
                    **rule_debug,
                    "llm_fallback": True,
                    "business_guard": {
                        "applied": False,
                        "reason": "llm_unavailable_rule_fallback",
                        "final_intent": rule_result.intent,
                    },
                }
                result = rule_result
            else:
                result = self._merge_rule_slots(llm_result, rule_result)
                result = self._apply_business_guard(result, rule_result, rule_debug)

        return AgentResult(
            agent_name=self.name,
            success=True,
            data=self._result_payload(result),
            confidence=result.confidence,
        )

    def _load_rules(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _result_payload(self, result: IntentResult) -> dict[str, Any]:
        payload = result.model_dump(mode="json")
        payload["rule_debug"] = getattr(self, "_last_rule_debug", {})
        payload["rule_config"] = str(DEFAULT_RULE_PATH.name)
        payload["intent_mode"] = getattr(self, "_last_intent_mode", "rule")
        payload["intent_model_status"] = intent_model_classifier.status()
        return payload

    def _llm_intent(
        self,
        message: str,
        state: ConversationState,
        recent_messages: list[ChatMessage],
        force: bool = False,
    ) -> IntentResult | None:
        if (
            not force
            and os.getenv("CHAT_LLM_ENABLED", "false").strip().lower()
            not in {"1", "true", "yes", "on"}
        ):
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
        self._last_rule_debug = {
            "mode": "llm",
            "matched_rules": [],
            "fallback": False,
        }
        return result

    def _apply_model_intent(
        self,
        message: str,
        rule_result: IntentResult,
        force: bool = False,
    ) -> IntentResult:
        if rule_result.intent == "smalltalk" and rule_result.confidence >= 0.9:
            self._last_rule_debug = {
                **getattr(self, "_last_rule_debug", {}),
                "model_intent": {
                    "skipped": True,
                    "reason": "high_confidence_smalltalk_guard",
                },
            }
            return rule_result
        try:
            model_result = intent_model_classifier.classify(message, force=force)
        except TypeError:
            model_result = intent_model_classifier.classify(message)
        if not model_result:
            return rule_result
        merged = rule_result.model_copy(
            update={
                "intent": model_result["intent"],
                "confidence": model_result["confidence"],
                "source": "bert+rule_slots",
                "needs_recommendation": self._intent_needs_recommendation(
                    model_result["intent"],
                    rule_result.needs_recommendation,
                ),
            }
        )
        self._last_rule_debug = {
            **getattr(self, "_last_rule_debug", {}),
            "model_intent": {
                "intent": model_result["intent"],
                "confidence": round(model_result["confidence"], 4),
                "model_dir": model_result["model_dir"],
            },
        }
        return merged

    def _apply_business_guard(
        self,
        candidate: IntentResult,
        rule_result: IntentResult,
        rule_debug: dict[str, Any],
    ) -> IntentResult:
        guard = {
            "applied": False,
            "reason": "model_accepted",
            "rule_intent": rule_result.intent,
            "candidate_intent": candidate.intent,
            "final_intent": candidate.intent,
        }
        candidate_debug = getattr(self, "_last_rule_debug", {})
        debug_payload = {**rule_debug}
        if "model_intent" in candidate_debug:
            debug_payload["model_intent"] = candidate_debug["model_intent"]
        if candidate_debug.get("mode") == "llm":
            debug_payload["llm_intent"] = {
                "fallback": candidate_debug.get("fallback", False),
                "matched_rules": candidate_debug.get("matched_rules", []),
            }
        final = candidate
        guarded_source = f"rule_guarded_{candidate.source}"

        if rule_result.intent == "smalltalk" and rule_result.confidence >= 0.9:
            final = rule_result.model_copy(
                update={
                    "source": guarded_source,
                    "confidence": max(rule_result.confidence, candidate.confidence),
                }
            )
            guard.update(
                {
                    "applied": True,
                    "reason": "smalltalk_guard",
                    "final_intent": final.intent,
                }
            )
        elif rule_result.intent == "record_feedback" and self._has_feedback_evidence(rule_result):
            final = rule_result.model_copy(
                update={
                    "source": guarded_source,
                    "confidence": max(rule_result.confidence, candidate.confidence),
                }
            )
            guard.update(
                {
                    "applied": True,
                    "reason": "feedback_guard",
                    "final_intent": final.intent,
                }
            )
        elif (
            rule_result.needs_recommendation
            and self._has_business_slots(rule_result)
            and candidate.intent
            not in {"recommend_products", "refine_preferences", "record_feedback"}
        ):
            final = rule_result.model_copy(
                update={
                    "source": guarded_source,
                    "confidence": max(rule_result.confidence, candidate.confidence),
                }
            )
            guard.update(
                {
                    "applied": True,
                    "reason": "business_slot_recommendation_guard",
                    "final_intent": final.intent,
                }
            )

        self._last_rule_debug = {**debug_payload, "business_guard": guard}
        return final

    def _merge_rule_slots(
        self,
        candidate: IntentResult,
        rule_result: IntentResult,
    ) -> IntentResult:
        merged_slots = self._clean_slots(candidate.slots)
        for key, value in self._clean_slots(rule_result.slots).items():
            if isinstance(value, list):
                merged_slots[key] = self._unique(
                    [*merged_slots.get(key, []), *value]
                    if isinstance(merged_slots.get(key), list)
                    else value
                )
            else:
                merged_slots[key] = value
        product_refs = self._unique([*candidate.product_refs, *rule_result.product_refs])
        return candidate.model_copy(
            update={
                "slots": merged_slots,
                "product_refs": product_refs,
                "needs_recommendation": (
                    candidate.needs_recommendation or rule_result.needs_recommendation
                ),
            }
        )

    def _clean_slots(self, slots: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in slots.items():
            if value in (None, "", []):
                continue
            cleaned[key] = value
        return cleaned

    def _has_business_slots(self, result: IntentResult) -> bool:
        slots = self._clean_slots(result.slots)
        business_keys = {
            "shopping_goal",
            "budget_min",
            "budget_max",
            "preferred_categories",
            "liked_brands",
            "preferred_tags",
        }
        return any(key in slots for key in business_keys)

    def _has_feedback_evidence(self, result: IntentResult) -> bool:
        slots = self._clean_slots(result.slots)
        return bool(
            result.product_refs
            or slots.get("event_type")
            or slots.get("rejected_reasons")
            or result.needs_recommendation
        )

    def _intent_needs_recommendation(self, intent: str, fallback: bool) -> bool:
        if intent in {"recommend_products", "refine_preferences"}:
            return True
        if intent == "record_feedback":
            return fallback
        return False

    def _rule_intent(self, message: str, state: ConversationState) -> IntentResult:
        text = message.strip()
        if self._is_meta_smalltalk(text):
            self._last_rule_debug = {
                "mode": "rule",
                "matched_rules": ["smalltalk"],
                "matched_keywords": {"smalltalk": self._matched_markers(text, "smalltalk")},
                "slot_sources": {},
            }
            return IntentResult(
                intent="smalltalk",
                slots={},
                product_refs=[],
                needs_recommendation=False,
                confidence=0.92,
                source="rule",
            )

        slots, slot_debug = self._extract_slots(text)
        product_refs = self._extract_product_refs(text, state)
        confidence = 0.72
        needs_recommendation = False
        matched_keywords: dict[str, list[str]] = {}

        if self._has_intent(text, "compare_products", matched_keywords):
            intent = "compare_products"
        elif self._has_intent(text, "explain_recommendation", matched_keywords):
            intent = "explain_recommendation"
        elif self._has_intent(text, "ask_product", matched_keywords):
            intent = "ask_product"
        elif self._looks_like_recommendation(text):
            intent = "recommend_products"
            matched_keywords["recommend_products"] = self._matched_markers(text, "recommend_products")
            needs_recommendation = True
            confidence = 0.82
        elif self._has_intent(text, "record_feedback", matched_keywords):
            intent = "record_feedback"
            needs_recommendation = self._has_any(
                text,
                self.rules["intent_markers"]["feedback_recommend"],
            )
            slots.update(self._feedback_slots(text))
        elif slots:
            intent = "refine_preferences"
            needs_recommendation = True
        else:
            intent = "smalltalk"
            confidence = 0.65

        if intent == "record_feedback" and product_refs:
            slots["product_refs"] = product_refs

        self._last_rule_debug = {
            "mode": "rule",
            "matched_rules": [intent],
            "matched_keywords": matched_keywords,
            "slot_sources": slot_debug,
            "product_refs": product_refs,
        }

        return IntentResult(
            intent=intent,
            slots=slots,
            product_refs=product_refs,
            needs_recommendation=needs_recommendation,
            confidence=confidence,
            source="rule",
        )

    def _extract_slots(self, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
        products = list_products()
        categories = self._unique(product.category for product in products)
        brands = self._unique(product.brand for product in products)
        tags = self._unique(tag for product in products for tag in product.tags)
        slots: dict[str, Any] = {}
        slot_debug: dict[str, Any] = {}

        budget = self._extract_budget(text)
        if budget:
            slots.update(budget)
            slot_debug["budget"] = budget

        matched_categories = [category for category in categories if category and category in text]
        matched_brands = [brand for brand in brands if brand and brand.lower() in text.lower()]
        matched_tags = [tag for tag in tags if tag and tag in text]
        if matched_categories:
            slot_debug["catalog_categories"] = matched_categories
        if matched_brands:
            slot_debug["catalog_brands"] = matched_brands
        if matched_tags:
            slot_debug["catalog_tags"] = matched_tags
        synonym_hits = []
        for keyword, mapped in self.rules["product_synonyms"].items():
            if keyword in text:
                matched_categories.extend(mapped["categories"])
                matched_tags.extend(mapped["tags"])
                synonym_hits.append(keyword)
        if synonym_hits:
            slot_debug["synonyms"] = synonym_hits

        generic_hits = []
        for term in self.rules["generic_tags"]:
            if term in text and term not in matched_tags and term not in matched_categories:
                matched_tags.append(term)
                generic_hits.append(term)
        if generic_hits:
            slot_debug["generic_tags"] = generic_hits

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

        return slots, slot_debug

    def _extract_budget(self, text: str) -> dict[str, float] | None:
        joiners = "|".join(re.escape(item) for item in self.rules["budget"]["range_joiners"])
        range_match = re.search(
            rf"(\d+(?:\.\d+)?)\s*(?:元|块|块钱|人民币|rmb|RMB)?\s*(?:{joiners})\s*"
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
        if any(marker in text for marker in self.rules["budget"]["min_markers"]):
            return {"budget_min": value}
        if any(marker in text for marker in self.rules["budget"]["max_markers"]):
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
        for canonical, values in self.rules["product_refs"].items():
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
        return self._has_any(text, self.rules["intent_markers"]["recommend_products"])

    def _is_meta_smalltalk(self, text: str) -> bool:
        return self._has_any(text, self.rules["intent_markers"]["smalltalk"])

    def _has_intent(
        self,
        text: str,
        intent: str,
        matched_keywords: dict[str, list[str]],
    ) -> bool:
        markers = self._matched_markers(text, intent)
        if not markers:
            return False
        matched_keywords[intent] = markers
        return True

    def _matched_markers(self, text: str, intent: str) -> list[str]:
        return [
            marker
            for marker in self.rules["intent_markers"].get(intent, [])
            if marker.lower() in text.lower()
        ]

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
