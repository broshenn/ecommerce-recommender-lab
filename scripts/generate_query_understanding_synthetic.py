from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog import list_products  # noqa: E402
from app.services.llm_client import LLMClient  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "query_understanding_synthetic_deepseek.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "query_understanding_train.jsonl"
DEFAULT_EVAL = PROJECT_ROOT / "data" / "query_understanding_eval.jsonl"

INTENTS = {
    "recommend_products",
    "refine_preferences",
    "compare_products",
    "explain_recommendation",
    "record_feedback",
    "ask_product",
    "smalltalk",
}
INTENT_MIX = {
    "recommend_products": 0.38,
    "refine_preferences": 0.16,
    "compare_products": 0.10,
    "explain_recommendation": 0.10,
    "record_feedback": 0.12,
    "ask_product": 0.08,
    "smalltalk": 0.06,
}
SYSTEM_PROMPT = """你是电商 Query Understanding 数据合成器。
只输出 JSON 数组，不要 markdown，不要解释。
每条样本用于训练意图/类目分类和结构化 query 解析。
要求：
1. 文本必须自然、口语化、中文为主，可以少量中英混合品牌词。
2. 类目只能来自给定 category 列表；非购物/反馈/解释/比较/问商品时如果类目不明确，category 用空字符串。
3. 不要编造商品库外的大类，比如鞋服、美妆、食品、家电。
4. slots 只放用户明确表达或强语义隐含的字段。
5. rewrite_queries 只为购物/补充偏好生成，其他意图用空数组。
6. need_clarify 用于需求太宽泛、缺少关键条件时。
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek synthetic Query Understanding data."
    )
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    products = list_products()
    categories = sorted({product.category for product in products})
    tags = sorted({tag for product in products for tag in product.tags})
    brands = sorted({product.brand for product in products})[:120]
    client = LLMClient()
    status = client.status()
    if not status["available"]:
        raise RuntimeError(f"LLM is not available: {status['last_error']}")

    examples = load_existing(args.output)
    seen_texts = {item["text"] for item in examples}
    print(f"Loaded existing examples: {len(examples)}")
    while len(examples) < args.target:
        remaining = args.target - len(examples)
        batch_size = min(args.batch_size, remaining)
        batch_intents = sample_intents(batch_size)
        prompt = build_prompt(
            batch_size=batch_size,
            batch_intents=batch_intents,
            categories=categories,
            tags=tags,
            brands=brands,
            existing_count=len(examples),
        )
        batch = request_batch(
            client=client,
            prompt=prompt,
            categories=categories,
            max_retries=args.max_retries,
        )
        accepted = []
        for item in batch:
            normalized = normalize_item(item, categories, client.model)
            if not normalized:
                continue
            if normalized["text"] in seen_texts:
                continue
            examples.append(normalized)
            seen_texts.add(normalized["text"])
            accepted.append(normalized)
            if len(examples) >= args.target:
                break

        save_jsonl(args.output, examples)
        print(
            f"generated={len(examples)}/{args.target}, "
            f"accepted_batch={len(accepted)}, requested_batch={batch_size}"
        )
        if not accepted:
            time.sleep(1.5)

    examples = examples[: args.target]
    save_jsonl(args.output, examples)
    split_train_eval(
        examples=examples,
        train_path=args.train_output,
        eval_path=args.eval_output,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print(json.dumps(summarize(examples), ensure_ascii=False, indent=2))
    print(f"Output: {args.output}")
    print(f"Train:  {args.train_output}")
    print(f"Eval:   {args.eval_output}")


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    examples = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = normalize_item(
                item,
                sorted({product.category for product in list_products()}),
                item.get("teacher_model", ""),
            )
            if normalized:
                examples.append(normalized)
    return examples


def sample_intents(batch_size: int) -> list[str]:
    intents = list(INTENT_MIX)
    weights = [INTENT_MIX[intent] for intent in intents]
    return random.choices(intents, weights=weights, k=batch_size)


def build_prompt(
    *,
    batch_size: int,
    batch_intents: list[str],
    categories: list[str],
    tags: list[str],
    brands: list[str],
    existing_count: int,
) -> str:
    sampled_tags = random.sample(tags, min(16, len(tags)))
    sampled_brands = random.sample(brands, min(18, len(brands))) if brands else []
    return json.dumps(
        {
            "task": "generate_query_understanding_examples",
            "batch_size": batch_size,
            "existing_count": existing_count,
            "intent_labels": sorted(INTENTS),
            "required_intents_in_order": batch_intents,
            "categories": categories,
            "available_tags": sampled_tags,
            "available_brands": sampled_brands,
            "schema": {
                "text": "用户原话",
                "intent": "one intent label",
                "category": "one of categories or empty string",
                "slots": {
                    "budget_min": "number or null",
                    "budget_max": "number or null",
                    "preferred_categories": "list[str]",
                    "liked_brands": "list[str]",
                    "preferred_tags": "list[str]",
                    "negative_tags": "list[str]",
                    "event_type": "view/like/dislike/purchase or empty string",
                    "product_refs": "list[str], e.g. 第一个/第二个/这款",
                },
                "rewrite_queries": "list[str]",
                "need_recommendation": "bool",
                "need_clarify": "bool",
                "clarify_question": "string",
            },
            "style_requirements": [
                "覆盖简单高频 query 和长尾口语表达",
                "包含预算、品牌、品类、用途、负反馈、比较、解释、库存问答、闲聊",
                "不要重复生成同一句",
                "smalltalk 的 category 必须为空字符串",
                "record_feedback 可包含 event_type 和 product_refs",
            ],
        },
        ensure_ascii=False,
    )


def request_batch(
    *,
    client: LLMClient,
    prompt: str,
    categories: list[str],
    max_retries: int,
) -> list[dict[str, Any]]:
    for attempt in range(max_retries):
        text = client.chat(
            system_prompt=SYSTEM_PROMPT,
            user_message=prompt,
            temperature=0.8,
            max_tokens=2048,
        )
        if not text:
            continue
        parsed = parse_json_array(text)
        if parsed:
            return [
                item
                for item in parsed
                if isinstance(item, dict)
                and item.get("category", "") in {"", *categories}
            ]
        time.sleep(1 + attempt)
    return []


def parse_json_array(text: str) -> list[Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
    candidates = [cleaned]
    match = re.search(r"\[.*\]", cleaned, re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return []


def normalize_item(
    item: dict[str, Any],
    categories: list[str],
    teacher_model: str,
) -> dict[str, Any] | None:
    text = str(item.get("text", "")).strip()
    intent = str(item.get("intent", "")).strip()
    category = str(item.get("category", "") or "").strip()
    if len(text) < 2 or intent not in INTENTS or category not in {"", *categories}:
        return None

    slots = item.get("slots") if isinstance(item.get("slots"), dict) else {}
    rewrite_queries = item.get("rewrite_queries")
    if not isinstance(rewrite_queries, list):
        rewrite_queries = []

    return {
        "text": text,
        "intent": intent,
        "category": category,
        "slots": normalize_slots(slots, categories),
        "rewrite_queries": unique_strings(rewrite_queries)[:6],
        "need_recommendation": bool(item.get("need_recommendation", intent in {"recommend_products", "refine_preferences"})),
        "need_clarify": bool(item.get("need_clarify", False)),
        "clarify_question": str(item.get("clarify_question", "") or "").strip(),
        "source": "deepseek_synthetic",
        "teacher_model": teacher_model,
    }


def normalize_slots(slots: dict[str, Any], categories: list[str]) -> dict[str, Any]:
    return {
        "budget_min": number_or_none(slots.get("budget_min")),
        "budget_max": number_or_none(slots.get("budget_max")),
        "preferred_categories": [
            value
            for value in unique_strings(slots.get("preferred_categories", []))
            if value in categories
        ],
        "liked_brands": unique_strings(slots.get("liked_brands", []))[:5],
        "preferred_tags": unique_strings(slots.get("preferred_tags", []))[:8],
        "negative_tags": unique_strings(slots.get("negative_tags", []))[:6],
        "event_type": str(slots.get("event_type", "") or "")
        if str(slots.get("event_type", "") or "") in {"", "view", "like", "dislike", "purchase"}
        else "",
        "product_refs": unique_strings(slots.get("product_refs", []))[:4],
    }


def unique_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def number_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def save_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in examples:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_train_eval(
    *,
    examples: list[dict[str, Any]],
    train_path: Path,
    eval_path: Path,
    eval_ratio: float,
    seed: int,
) -> None:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_items = shuffled[:eval_size]
    train_items = shuffled[eval_size:]
    save_jsonl(train_path, train_items)
    save_jsonl(eval_path, eval_items)


def summarize(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_intent: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for item in examples:
        by_intent[item["intent"]] = by_intent.get(item["intent"], 0) + 1
        by_category[item["category"] or "(none)"] = by_category.get(item["category"] or "(none)", 0) + 1
    return {
        "count": len(examples),
        "by_intent": dict(sorted(by_intent.items())),
        "by_category": dict(sorted(by_category.items())),
    }


if __name__ == "__main__":
    main()
