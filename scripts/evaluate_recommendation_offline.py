from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
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

from app.catalog import list_products  # noqa: E402
from app.models import Product, RecommendRequest  # noqa: E402
from app.orchestrator.graph import recommend_with_graph  # noqa: E402
from app.recommender import recommend_products  # noqa: E402
from app.services import ab_test_engine, feature_store, metrics_collector  # noqa: E402


DEFAULT_EVENTS = PROJECT_ROOT / "data" / "amazon_user_events_sample.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "recommendation_offline_eval_latest.json"
POSITIVE_EVENT_TYPES = {"like", "purchase"}
RELEVANCE = {"purchase": 3.0, "like": 2.0, "view": 0.5, "dislike": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate end-to-end recommendation quality without online traffic."
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--orchestrator", choices=["graph", "supervisor"], default="graph")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-events-per-user", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=50)
    args = parser.parse_args()

    report = evaluate_offline(
        events_path=args.events,
        orchestrator=args.orchestrator,
        k=args.k,
        min_events_per_user=args.min_events_per_user,
        max_users=args.max_users,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report written: {args.output}")


def evaluate_offline(
    *,
    events_path: Path,
    orchestrator: str = "graph",
    k: int = 5,
    min_events_per_user: int = 5,
    max_users: int = 50,
) -> dict[str, Any]:
    products = list_products()
    products_by_id = {product.product_id: product for product in products}
    user_events = load_events(events_path, products_by_id)
    cases = build_cases(
        user_events,
        products_by_id=products_by_id,
        k=k,
        min_events_per_user=min_events_per_user,
        max_users=max_users,
    )

    ab_test_engine.reset_outcomes()
    metrics_collector.reset()
    feature_store.clear_all()

    case_results = []
    started = time.perf_counter()
    for case in cases:
        case_results.append(run_case(case, orchestrator=orchestrator, k=k))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events_path": str(events_path),
        "orchestrator": orchestrator,
        "k": k,
        "case_count": len(case_results),
        "summary": summarize(case_results, elapsed_seconds=time.perf_counter() - started),
        "cases": case_results,
    }


def load_events(
    events_path: Path,
    products_by_id: dict[str, Product],
) -> dict[str, list[dict[str, Any]]]:
    user_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with events_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            product_id = row["product_id"]
            if product_id not in products_by_id:
                continue
            event_type = row["event_type"]
            if event_type not in {"view", "like", "dislike", "purchase"}:
                continue
            user_events[row["user_id"]].append(
                {
                    "user_id": row["user_id"],
                    "product_id": product_id,
                    "event_type": event_type,
                    "rating": float(row.get("rating") or 0),
                    "timestamp": int(float(row.get("timestamp") or 0)),
                }
            )

    for events in user_events.values():
        events.sort(key=lambda event: event["timestamp"])
    return dict(user_events)


def build_cases(
    user_events: dict[str, list[dict[str, Any]]],
    *,
    products_by_id: dict[str, Product],
    k: int,
    min_events_per_user: int,
    max_users: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for user_id, events in sorted(user_events.items()):
        if len(events) < min_events_per_user:
            continue
        split_index = max(1, int(len(events) * 0.7))
        history = events[:split_index]
        future = events[split_index:]
        targets = [
            event
            for event in future
            if event["event_type"] in POSITIVE_EVENT_TYPES
        ]
        if not targets:
            continue

        request = request_from_history(
            user_id=user_id,
            history=history,
            products_by_id=products_by_id,
            k=k,
        )
        cases.append(
            {
                "user_id": user_id,
                "request": request.model_dump(mode="json"),
                "history_event_count": len(history),
                "target_events": targets,
            }
        )
        if len(cases) >= max_users:
            break
    return cases


def request_from_history(
    *,
    user_id: str,
    history: list[dict[str, Any]],
    products_by_id: dict[str, Product],
    k: int,
) -> RecommendRequest:
    liked_products = [
        products_by_id[event["product_id"]]
        for event in history
        if event["event_type"] in POSITIVE_EVENT_TYPES and event["product_id"] in products_by_id
    ]
    viewed_products = [
        products_by_id[event["product_id"]]
        for event in history
        if event["event_type"] == "view" and event["product_id"] in products_by_id
    ]
    disliked_products = [
        event["product_id"]
        for event in history
        if event["event_type"] == "dislike"
    ]
    price_values = [product.price for product in liked_products if product.price > 0]
    budget_max = max(price_values) * 1.25 if price_values else None

    return RecommendRequest(
        user_id=f"offline-{user_id}",
        num_items=k,
        preferred_categories=top_values(product.category for product in liked_products),
        liked_brands=top_values(product.brand for product in liked_products),
        preferred_tags=top_values(tag for product in liked_products for tag in product.tags),
        budget_min=0 if price_values else None,
        budget_max=round(budget_max, 2) if budget_max else None,
        recent_views=[product.product_id for product in viewed_products[-10:]],
        disliked_products=disliked_products[-10:],
    )


def run_case(case: dict[str, Any], *, orchestrator: str, k: int) -> dict[str, Any]:
    request = RecommendRequest.model_validate(case["request"])
    started = time.perf_counter()
    response = (
        recommend_with_graph(request)
        if orchestrator == "graph"
        else recommend_products(request)
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    recommended_ids = [product.product_id for product in response.products]
    target_relevance = target_relevance_map(case["target_events"])
    target_ids = set(target_relevance)
    intent_relevance = intent_relevance_map(case["target_events"])

    exact_hit_count = sum(1 for product_id in recommended_ids if product_id in target_ids)
    intent_hit_count = sum(
        1
        for product in response.products
        if intent_score(product, intent_relevance) > 0
    )
    return {
        "user_id": case["user_id"],
        "history_event_count": case["history_event_count"],
        "target_product_ids": list(target_ids),
        "recommended_product_ids": recommended_ids,
        "exact_hit": exact_hit_count > 0,
        "exact_hit_count": exact_hit_count,
        "exact_recall_at_k": round(exact_hit_count / len(target_ids), 4) if target_ids else 0.0,
        "exact_ndcg_at_k": round(ndcg_at_k(recommended_ids, target_relevance, k), 4),
        "intent_hit": intent_hit_count > 0,
        "intent_hit_count": intent_hit_count,
        "intent_recall_at_k": round(intent_hit_count / min(k, len(response.products)), 4)
        if response.products
        else 0.0,
        "intent_ndcg_at_k": round(intent_ndcg_at_k(response.products, intent_relevance, k), 4),
        "budget_compliant": budget_compliant(request, response.products),
        "inventory_compliant": all(product.stock > 0 for product in response.products),
        "diversity_at_k": round(diversity_at_k(response.products), 4),
        "latency_ms": latency_ms,
        "experiment_group": response.experiment_group,
        "marketing_mode": response.agent_results.get("marketing_copy").data.get("mode")
        if response.agent_results.get("marketing_copy")
        else "unknown",
        "rerank_mode": response.agent_results.get("product_rerank").data.get("mode")
        if response.agent_results.get("product_rerank")
        else "unknown",
    }


def summarize(case_results: list[dict[str, Any]], *, elapsed_seconds: float) -> dict[str, Any]:
    if not case_results:
        return {
            "case_count": 0,
            "exact_hit_rate_at_k": 0.0,
            "exact_recall_at_k": 0.0,
            "exact_ndcg_at_k": 0.0,
            "intent_hit_rate_at_k": 0.0,
            "intent_recall_at_k": 0.0,
            "intent_ndcg_at_k": 0.0,
            "budget_compliance_rate": 0.0,
            "inventory_compliance_rate": 0.0,
            "avg_diversity_at_k": 0.0,
            "avg_latency_ms": 0.0,
            "fallback_rate": 0.0,
            "elapsed_seconds": round(elapsed_seconds, 2),
        }

    return {
        "case_count": len(case_results),
        "exact_hit_rate_at_k": avg(1.0 if item["exact_hit"] else 0.0 for item in case_results),
        "exact_recall_at_k": avg(item["exact_recall_at_k"] for item in case_results),
        "exact_ndcg_at_k": avg(item["exact_ndcg_at_k"] for item in case_results),
        "intent_hit_rate_at_k": avg(1.0 if item["intent_hit"] else 0.0 for item in case_results),
        "intent_recall_at_k": avg(item["intent_recall_at_k"] for item in case_results),
        "intent_ndcg_at_k": avg(item["intent_ndcg_at_k"] for item in case_results),
        "budget_compliance_rate": avg(1.0 if item["budget_compliant"] else 0.0 for item in case_results),
        "inventory_compliance_rate": avg(1.0 if item["inventory_compliant"] else 0.0 for item in case_results),
        "avg_diversity_at_k": avg(item["diversity_at_k"] for item in case_results),
        "avg_latency_ms": avg(item["latency_ms"] for item in case_results),
        "fallback_rate": avg(
            1.0 if item["marketing_mode"] in {"rule_fallback", "agent_fallback"} else 0.0
            for item in case_results
        ),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }


def target_relevance_map(events: list[dict[str, Any]]) -> dict[str, float]:
    relevance: dict[str, float] = {}
    for event in events:
        product_id = event["product_id"]
        relevance[product_id] = max(relevance.get(product_id, 0.0), RELEVANCE[event["event_type"]])
    return relevance


def intent_relevance_map(events: list[dict[str, Any]]) -> dict[str, Any]:
    products_by_id = {product.product_id: product for product in list_products()}
    categories: dict[str, float] = defaultdict(float)
    brands: dict[str, float] = defaultdict(float)
    tags: dict[str, float] = defaultdict(float)
    for event in events:
        product = products_by_id.get(event["product_id"])
        if not product:
            continue
        relevance = RELEVANCE[event["event_type"]]
        categories[product.category] = max(categories[product.category], relevance)
        brands[product.brand] = max(brands[product.brand], relevance)
        for tag in product.tags:
            tags[tag] = max(tags[tag], relevance)
    return {
        "categories": dict(categories),
        "brands": dict(brands),
        "tags": dict(tags),
    }


def intent_score(product: Product, intent_relevance: dict[str, Any]) -> float:
    score = 0.0
    score += intent_relevance["categories"].get(product.category, 0.0)
    score += intent_relevance["brands"].get(product.brand, 0.0) * 0.8
    if product.tags:
        tag_score = max(
            (intent_relevance["tags"].get(tag, 0.0) for tag in product.tags),
            default=0.0,
        )
        score += tag_score * 0.5
    return score


def intent_ndcg_at_k(
    products: list[Product],
    intent_relevance: dict[str, Any],
    k: int,
) -> float:
    gains = [intent_score(product, intent_relevance) for product in products[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    candidate_gains = sorted(gains, reverse=True)
    if not candidate_gains:
        return 0.0
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(candidate_gains[:k]))
    return dcg / idcg if idcg else 0.0


def ndcg_at_k(recommended_ids: list[str], target_relevance: dict[str, float], k: int) -> float:
    gains = [target_relevance.get(product_id, 0.0) for product_id in recommended_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = sorted(target_relevance.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg else 0.0


def budget_compliant(request: RecommendRequest, products: list[Product]) -> bool:
    for product in products:
        if request.budget_min is not None and product.price < request.budget_min:
            return False
        if request.budget_max is not None and product.price > request.budget_max:
            return False
    return True


def diversity_at_k(products: list[Product]) -> float:
    if not products:
        return 0.0
    categories = {product.category for product in products}
    brands = {product.brand for product in products}
    return ((len(categories) / len(products)) + (len(brands) / len(products))) / 2


def top_values(values: Any, *, limit: int = 5) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        if value:
            counts[value] += 1
    return [
        value
        for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def avg(values: Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(float(statistics.mean(materialized)), 4)


if __name__ == "__main__":
    main()
