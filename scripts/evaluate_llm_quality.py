from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LLM_API_KEY", "ollama")
os.environ.setdefault("LLM_API_BASE", "http://127.0.0.1:11434/v1")
os.environ.setdefault("LLM_MODEL", "qwen2.5:3b")
os.environ.setdefault("LLM_MAX_TOKENS", "1024")

from app.agents.marketing_copy_agent import MarketingCopyAgent  # noqa: E402
from app.agents.user_profile_agent import UserProfileAgent  # noqa: E402
from app.catalog import list_products  # noqa: E402
from app.models import Product, UserProfile  # noqa: E402
from app.services import llm_client  # noqa: E402


SEGMENTS = [
    "new_user",
    "active",
    "high_value",
    "price_sensitive",
    "churn_risk",
    "category_explorer",
    "brand_loyal",
]

STYLE_MARKERS = {
    "new_user": ["新人", "新用户", "首次", "入门", "欢迎", "首单"],
    "active": ["浏览", "常看", "偏好", "精选", "适合", "日常"],
    "high_value": ["品质", "高端", "尊享", "旗舰", "质感", "品牌"],
    "price_sensitive": ["性价比", "预算", "实惠", "价格", "划算", "省心"],
    "churn_risk": ["回归", "回来", "错过", "专属", "重逢", "唤回", "想念"],
    "category_explorer": ["探索", "新奇", "尝鲜", "发现", "灵感", "多样"],
    "brand_loyal": ["品牌", "生态", "官方", "忠实", "同款", "系列"],
}

UNSUPPORTED_PROMO_PATTERNS = [
    r"买一送一",
    r"\d+折",
    r"折扣",
    r"促销",
    r"立减",
    r"直降",
    r"省下?\s*\d+",
    r"优惠券?",
    r"限时",
    r"历史低价",
    r"全网最低",
    r"秒杀",
    r"赠送?",
    r"免邮",
    r"包邮",
]

MONEY_PATTERN = re.compile(
    r"(?:[¥￥]\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*元)"
)


def main() -> None:
    started = time.perf_counter()
    products = _sample_products()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_client": llm_client.status(),
        "environment": {
            "LLM_API_BASE": os.getenv("LLM_API_BASE"),
            "LLM_MODEL": os.getenv("LLM_MODEL"),
            "LLM_TEMPERATURE": os.getenv("LLM_TEMPERATURE", "default"),
        },
        "sample_products": [product.model_dump(mode="json") for product in products],
        "marketing_copy": evaluate_marketing_copy(products),
        "user_profile": evaluate_user_profile(),
    }
    report["consistency"] = evaluate_consistency(products[0])
    report["summary"] = summarize(report)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    latest_path = reports_dir / "llm_quality_eval_latest.json"
    latest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report written: {latest_path}")


def evaluate_marketing_copy(products: list[Product]) -> dict[str, Any]:
    agent = MarketingCopyAgent()
    segment_results = []

    for segment in SEGMENTS:
        profile = UserProfile(
            user_id=f"eval-{segment}",
            preferred_categories=_unique([product.category for product in products]),
            liked_brands=_unique([product.brand for product in products[:2]]),
            preferred_tags=_unique(tag for product in products for tag in product.tags[:2]),
            event_count=20,
        )
        if segment == "new_user":
            profile.event_count = 2
        elif segment == "churn_risk":
            profile.event_count = 80

        result = agent.run(
            products=products,
            profile=profile,
            llm_profile={
                "segments": [segment],
                "intent_summary": f"用户属于 {segment} 分群，需要匹配该分群的推荐语气。",
                "recommendation_hint": "根据商品真实字段生成推荐文案，不要编造未提供的优惠或折扣。",
                "price_sensitivity": "high" if segment == "price_sensitive" else "medium",
                "rfm_interpretation": "评测样例",
            },
            experiment_group="treatment",
        )

        copies = result.data.get("copies", [])
        copy_checks = [
            check_copy(copy, products_by_id={product.product_id: product for product in products}, segment=segment)
            for copy in copies
        ]
        segment_results.append(
            {
                "segment": segment,
                "success": result.success,
                "mode": result.data.get("mode"),
                "latency_ms": round(result.latency_ms, 2),
                "copy_count": len(copies),
                "expected_count": len(products),
                "copies": copy_checks,
                "error": result.error,
            }
        )

    return {"segments": segment_results}


def evaluate_user_profile() -> dict[str, Any]:
    agent = UserProfileAgent()
    cases = [
        {
            "name": "new_user_low_history",
            "expected": "new_user",
            "profile": UserProfile(user_id="eval-new", event_count=2),
            "features": _features(recency=0.8, frequency=0.05, monetary=0.0),
        },
        {
            "name": "churn_risk_inactive",
            "expected": "churn_risk",
            "profile": UserProfile(
                user_id="eval-churn",
                preferred_categories=["Electronics"],
                liked_brands=["Sony"],
                event_count=80,
            ),
            "features": _features(recency=0.02, frequency=0.55, monetary=0.35),
        },
        {
            "name": "high_value_active",
            "expected": "high_value",
            "profile": UserProfile(
                user_id="eval-vip",
                preferred_categories=["Electronics"],
                liked_brands=["Apple"],
                cart_items=["P1", "P2", "P3"],
                event_count=120,
            ),
            "features": _features(
                recency=0.95,
                frequency=0.9,
                monetary=0.92,
                view_count_24h=12,
                like_count_24h=5,
                purchase_count_7d=8,
                recent_brands=["Apple"],
            ),
        },
        {
            "name": "category_explorer_multi_category",
            "expected": "category_explorer",
            "profile": UserProfile(
                user_id="eval-explorer",
                preferred_categories=["Books", "Beauty", "Electronics", "Sports"],
                event_count=35,
            ),
            "features": _features(
                recency=0.7,
                frequency=0.5,
                monetary=0.3,
                view_count_24h=10,
                recent_categories=["Books", "Beauty", "Electronics", "Sports"],
            ),
        },
        {
            "name": "brand_loyal_repeat_brand",
            "expected": "brand_loyal",
            "profile": UserProfile(
                user_id="eval-brand",
                preferred_categories=["Electronics"],
                liked_brands=["Apple"],
                event_count=50,
            ),
            "features": _features(
                recency=0.8,
                frequency=0.7,
                monetary=0.65,
                view_count_24h=8,
                like_count_24h=4,
                recent_brands=["Apple"],
            ),
        },
    ]

    results = []
    for case in cases:
        started = time.perf_counter()
        output = agent._build_llm_profile(case["profile"], case["features"])
        latency_ms = (time.perf_counter() - started) * 1000
        segments = output.get("segments", []) if isinstance(output, dict) else []
        results.append(
            {
                "case": case["name"],
                "expected": case["expected"],
                "segments": segments,
                "hit": case["expected"] in segments,
                "schema_ok": _profile_schema_ok(output),
                "latency_ms": round(latency_ms, 2),
                "output": output,
            }
        )

    return {"cases": results}


def evaluate_consistency(product: Product) -> dict[str, Any]:
    copy_agent = MarketingCopyAgent()
    profile_agent = UserProfileAgent()
    copy_outputs = []
    profile_outputs = []

    profile = UserProfile(
        user_id="eval-consistency-copy",
        preferred_categories=[product.category],
        liked_brands=[product.brand],
        preferred_tags=product.tags,
        event_count=20,
    )
    llm_profile = {
        "segments": ["price_sensitive"],
        "intent_summary": "用户预算敏感，关注价格透明和性价比。",
        "recommendation_hint": "只基于真实商品价格，不要编造折扣。",
        "price_sensitivity": "high",
        "rfm_interpretation": "评测样例",
    }

    for _ in range(3):
        result = copy_agent.run(
            products=[product],
            profile=profile,
            llm_profile=llm_profile,
            experiment_group="treatment",
        )
        copies = result.data.get("copies", [])
        copy_outputs.append(copies[0].get("text", "") if copies else "")

    churn_profile = UserProfile(
        user_id="eval-consistency-profile",
        preferred_categories=["Electronics"],
        liked_brands=["Sony"],
        event_count=80,
    )
    churn_features = _features(recency=0.02, frequency=0.55, monetary=0.35)
    for _ in range(3):
        output = profile_agent._build_llm_profile(churn_profile, churn_features)
        profile_outputs.append(output.get("segments", []) if isinstance(output, dict) else [])

    return {
        "copy": {
            "runs": copy_outputs,
            "unique_outputs": len(set(copy_outputs)),
            "stable": len(set(copy_outputs)) == 1,
        },
        "profile": {
            "runs": profile_outputs,
            "unique_outputs": len({tuple(item) for item in profile_outputs}),
            "stable": len({tuple(item) for item in profile_outputs}) == 1,
        },
    }


def check_copy(copy: dict[str, Any], products_by_id: dict[str, Product], segment: str) -> dict[str, Any]:
    product_id = str(copy.get("product_id", ""))
    text = str(copy.get("text", ""))
    product = products_by_id.get(product_id)
    char_count = len(re.sub(r"\s+", "", text))
    promo_hits = _pattern_hits(text, UNSUPPORTED_PROMO_PATTERNS)
    money_values = _money_values(text)
    price_mismatches = []
    if product is not None:
        expected_price = round(float(product.price), 2)
        for value in money_values:
            if abs(value - expected_price) > 0.01:
                price_mismatches.append(value)

    return {
        "product_id": product_id,
        "text": text,
        "char_count": char_count,
        "length_ok": 25 <= char_count <= 40,
        "style_marker_hit": any(marker in text for marker in STYLE_MARKERS[segment]),
        "unsupported_promo_hits": promo_hits,
        "money_values": money_values,
        "price_mismatches": price_mismatches,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    copy_checks = [
        copy
        for segment in report["marketing_copy"]["segments"]
        for copy in segment["copies"]
    ]
    segment_results = report["marketing_copy"]["segments"]
    profile_cases = report["user_profile"]["cases"]
    latencies = [segment["latency_ms"] for segment in segment_results]

    expected_copies = sum(segment["expected_count"] for segment in segment_results)
    actual_copies = sum(segment["copy_count"] for segment in segment_results)
    length_passes = sum(1 for copy in copy_checks if copy["length_ok"])
    style_passes = sum(1 for copy in copy_checks if copy["style_marker_hit"])
    promo_failures = sum(1 for copy in copy_checks if copy["unsupported_promo_hits"])
    price_mismatches = sum(1 for copy in copy_checks if copy["price_mismatches"])
    schema_passes = sum(1 for case in profile_cases if case["schema_ok"])
    label_hits = sum(1 for case in profile_cases if case["hit"])

    return {
        "copy_json_success_rate": _round_rate(
            sum(1 for segment in segment_results if segment["success"] and segment["mode"] == "llm"),
            len(segment_results),
        ),
        "copy_completion_rate": _round_rate(actual_copies, expected_copies),
        "copy_length_pass_rate": _round_rate(length_passes, len(copy_checks)),
        "copy_style_marker_pass_rate": _round_rate(style_passes, len(copy_checks)),
        "copy_unsupported_promo_rate": _round_rate(promo_failures, len(copy_checks)),
        "copy_price_mismatch_rate": _round_rate(price_mismatches, len(copy_checks)),
        "copy_avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "profile_schema_pass_rate": _round_rate(schema_passes, len(profile_cases)),
        "profile_expected_label_accuracy": _round_rate(label_hits, len(profile_cases)),
        "copy_consistency_unique_outputs": report["consistency"]["copy"]["unique_outputs"],
        "profile_consistency_unique_outputs": report["consistency"]["profile"]["unique_outputs"],
        "overall_verdict": _verdict(
            length_passes=length_passes,
            copy_count=len(copy_checks),
            promo_failures=promo_failures,
            label_hits=label_hits,
            profile_count=len(profile_cases),
        ),
    }


def _verdict(
    *,
    length_passes: int,
    copy_count: int,
    promo_failures: int,
    label_hits: int,
    profile_count: int,
) -> str:
    if copy_count == 0 or profile_count == 0:
        return "blocked"
    if promo_failures > 0:
        return "not_ready_price_hallucination"
    if length_passes / copy_count < 0.8:
        return "not_ready_copy_length"
    if label_hits / profile_count < 0.8:
        return "not_ready_profile_labels"
    return "usable_with_monitoring"


def _sample_products() -> list[Product]:
    selected: list[Product] = []
    seen_categories: set[str] = set()
    for product in list_products():
        if product.category in seen_categories:
            continue
        selected.append(product)
        seen_categories.add(product.category)
        if len(selected) >= 3:
            break
    if len(selected) < 3:
        selected = list_products()[:3]
    return selected


def _features(
    *,
    recency: float,
    frequency: float,
    monetary: float,
    view_count_1h: int = 0,
    view_count_24h: int = 0,
    like_count_24h: int = 0,
    dislike_count_24h: int = 0,
    purchase_count_7d: int = 0,
    recent_categories: list[str] | None = None,
    recent_brands: list[str] | None = None,
    recent_tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "view_count_1h": view_count_1h,
        "view_count_24h": view_count_24h,
        "like_count_24h": like_count_24h,
        "dislike_count_24h": dislike_count_24h,
        "purchase_count_7d": purchase_count_7d,
        "recent_categories": recent_categories or [],
        "recent_brands": recent_brands or [],
        "recent_tags": recent_tags or [],
        "rfm": {
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
        },
    }


def _profile_schema_ok(output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    required = {
        "segments": list,
        "intent_summary": str,
        "recommendation_hint": str,
        "price_sensitivity": str,
        "rfm_interpretation": str,
    }
    return all(isinstance(output.get(key), expected_type) for key, expected_type in required.items())


def _money_values(text: str) -> list[float]:
    values = []
    for match in MONEY_PATTERN.finditer(text):
        raw_value = match.group(1) or match.group(2)
        if raw_value:
            values.append(round(float(raw_value), 2))
    return values


def _pattern_hits(text: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]


def _round_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _unique(values) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


if __name__ == "__main__":
    main()
