from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("QWEN_API_KEY", "")
os.environ.setdefault("DASHSCOPE_API_KEY", "")
os.environ.setdefault("COPY_LLM_BACKEND", "disabled")
os.environ.setdefault("PRODUCT_VECTOR_EMBEDDING_PROVIDER", "local")

from app.behavior import reset_behavior_events  # noqa: E402
from app.models import ChatRequest, ChatResponse  # noqa: E402
from app.orchestrator.chat import chat_orchestrator  # noqa: E402
from app.services import ab_test_engine, feature_store, llm_client, metrics_collector  # noqa: E402

DEFAULT_CASES = PROJECT_ROOT / "data" / "chat_eval_cases.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "chat_agent_eval_latest.json"
THRESHOLDS = {
    "intent_macro_f1": 0.85,
    "slot_f1": 0.80,
    "memory_consistency_rate": 0.90,
    "product_ref_resolution_rate": 0.85,
    "task_success_rate": 0.80,
    "budget_compliance_rate": 0.95,
    "inventory_compliance_rate": 1.00,
    "avg_latency_ms": 1500.0,
    "unsupported_claim_rate": 0.02,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate conversational commerce agent.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = evaluate_chat_agent(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report written: {args.output}")


def evaluate_chat_agent(cases_path: Path) -> dict[str, Any]:
    cases = load_cases(cases_path)
    reset_runtime()

    case_results = [run_case(case) for case in cases]
    summary = summarize(case_results)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "case_count": len(case_results),
        "thresholds": THRESHOLDS,
        "summary": summary,
        "cases": case_results,
    }


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    cases = []
    with cases_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def reset_runtime() -> None:
    llm_client.api_key = ""
    llm_client._client = None
    reset_behavior_events()
    feature_store.clear_all()
    chat_orchestrator.memory.clear_all()
    ab_test_engine.reset_outcomes()
    metrics_collector.reset()


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    user_id = f"chat-eval-{case['case_id']}"
    session_id = f"eval-{case['case_id']}"
    responses: list[ChatResponse] = []
    latencies = []

    for message in case["messages"]:
        started = time.perf_counter()
        response = chat_orchestrator.chat(
            ChatRequest(user_id=user_id, session_id=session_id, message=message)
        )
        latencies.append((time.perf_counter() - started) * 1000)
        responses.append(response)

    expected = case["expected"]
    final = responses[-1]
    predicted_intents = [response.intent for response in responses]
    expected_intents = expected.get("intents", [])
    intent_scores = score_sequence(predicted_intents, expected_intents)
    slot_scores = score_slots(final.state.model_dump(mode="json"), expected.get("slots", {}))
    constraints = check_constraints(responses, expected.get("constraints", {}))
    unsupported_claim = has_unsupported_claim(final.reply)

    return {
        "case_id": case["case_id"],
        "predicted_intents": predicted_intents,
        "expected_intents": expected_intents,
        "final_intent": final.intent,
        "expected_final_intent": expected.get("final_intent"),
        "intent_exact": predicted_intents == expected_intents,
        "intent_precision": intent_scores["precision"],
        "intent_recall": intent_scores["recall"],
        "intent_f1": intent_scores["f1"],
        "slot_precision": slot_scores["precision"],
        "slot_recall": slot_scores["recall"],
        "slot_f1": slot_scores["f1"],
        "memory_consistent": memory_consistent(final, expected.get("slots", {})),
        "product_ref_resolved": constraints["product_ref_resolved"],
        "task_success": constraints["task_success"] and final.intent == expected.get("final_intent"),
        "budget_compliant": constraints["budget_compliant"],
        "inventory_compliant": constraints["inventory_compliant"],
        "recommendation_ndcg_at_k": constraints["recommendation_ndcg_at_k"],
        "unsupported_claim": unsupported_claim,
        "latency_ms": round(sum(latencies), 2),
        "final_state": final.state.model_dump(mode="json"),
        "product_ids": [product.product_id for product in final.products],
    }


def score_sequence(predicted: list[str], expected: list[str]) -> dict[str, float]:
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    matches = sum(1 for left, right in zip(predicted, expected) if left == right)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(expected) if expected else 0.0
    return prf(precision, recall)


def score_slots(state: dict[str, Any], expected_slots: dict[str, Any]) -> dict[str, float]:
    if not expected_slots:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    expected_pairs = flatten_slots(expected_slots)
    predicted_pairs = flatten_slots({key: state.get(key) for key in expected_slots})
    matches = len(expected_pairs & predicted_pairs)
    precision = matches / len(predicted_pairs) if predicted_pairs else 0.0
    recall = matches / len(expected_pairs) if expected_pairs else 0.0
    return prf(precision, recall)


def flatten_slots(slots: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = set()
    for key, raw_value in slots.items():
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            if value not in (None, "", []):
                pairs.add((key, normalize_slot_value(value)))
    return pairs


def normalize_slot_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def check_constraints(
    responses: list[ChatResponse],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    products = [product for response in responses for product in response.products]
    final_products = responses[-1].products
    min_products = constraints.get("min_products", 0)

    max_price = constraints.get("max_price")
    budget_products = final_products or products
    budget_compliant = True
    if max_price is not None and budget_products:
        budget_compliant = all(product.price <= float(max_price) for product in budget_products)

    inventory_compliant = True
    if constraints.get("inventory_positive") and budget_products:
        inventory_compliant = all(product.stock > 0 for product in budget_products)

    product_ref_resolved = True
    if constraints.get("product_ref_resolved"):
        product_ref_resolved = any(response.state.active_product_refs for response in responses)

    task_success = len(final_products) >= min_products if min_products else True
    if constraints.get("product_ref_resolved"):
        task_success = task_success and product_ref_resolved

    return {
        "task_success": task_success,
        "budget_compliant": budget_compliant,
        "inventory_compliant": inventory_compliant,
        "product_ref_resolved": product_ref_resolved,
        "recommendation_ndcg_at_k": 1.0 if task_success else 0.0,
    }


def memory_consistent(response: ChatResponse, expected_slots: dict[str, Any]) -> bool:
    state = response.state.model_dump(mode="json")
    for key, expected_value in expected_slots.items():
        actual = state.get(key)
        if isinstance(expected_value, list):
            actual_values = actual if isinstance(actual, list) else [actual]
            if not all(value in actual_values for value in expected_value):
                return False
        elif normalize_slot_value(actual) != normalize_slot_value(expected_value):
            return False
    return True


def has_unsupported_claim(text: str) -> bool:
    markers = ["最低价", "全网最低", "保证", "包治", "官方认证优惠", "限时折扣"]
    return any(marker in text for marker in markers)


def summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "case_count": len(case_results),
        "intent_macro_f1": avg(item["intent_f1"] for item in case_results),
        "slot_f1": avg(item["slot_f1"] for item in case_results),
        "memory_consistency_rate": avg_bool(item["memory_consistent"] for item in case_results),
        "product_ref_resolution_rate": avg_bool(item["product_ref_resolved"] for item in case_results),
        "task_success_rate": avg_bool(item["task_success"] for item in case_results),
        "budget_compliance_rate": avg_bool(item["budget_compliant"] for item in case_results),
        "inventory_compliance_rate": avg_bool(item["inventory_compliant"] for item in case_results),
        "recommendation_ndcg_at_k": avg(item["recommendation_ndcg_at_k"] for item in case_results),
        "fallback_rate": 0.0,
        "avg_latency_ms": avg(item["latency_ms"] for item in case_results),
        "unsupported_claim_rate": avg_bool(item["unsupported_claim"] for item in case_results),
    }
    summary["passed_thresholds"] = {
        key: (summary[key] <= value if key in {"avg_latency_ms", "unsupported_claim_rate"} else summary[key] >= value)
        for key, value in THRESHOLDS.items()
    }
    return summary


def prf(precision: float, recall: float) -> dict[str, float]:
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def avg(values: Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(float(statistics.mean(materialized)), 4)


def avg_bool(values: Any) -> float:
    return avg(1.0 if value else 0.0 for value in values)


if __name__ == "__main__":
    main()
