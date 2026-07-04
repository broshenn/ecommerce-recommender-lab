from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog import list_products  # noqa: E402

DEFAULT_SEEDS = [
    PROJECT_ROOT / "data" / "query_understanding_synthetic_deepseek_smoke.jsonl",
    PROJECT_ROOT / "data" / "query_understanding_synthetic_deepseek_smoke2.jsonl",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "query_understanding_synthetic_deepseek.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "query_understanding_train.jsonl"
DEFAULT_EVAL = PROJECT_ROOT / "data" / "query_understanding_eval.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment DeepSeek seed Query Understanding data to a larger JSONL dataset."
    )
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--seed-files", nargs="*", type=Path, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    random.seed(args.seed)
    seed_examples = load_seed_examples(args.seed_files)
    examples = dedupe(seed_examples)
    seen = {item["text"] for item in examples}
    support_limit = int(args.target * 0.55)
    for item in generate_intent_support(args.target):
        if item["text"] in seen or len(examples) >= support_limit:
            continue
        examples.append(item)
        seen.add(item["text"])
    attempts = 0
    while len(examples) < args.target and attempts < 30:
        attempts += 1
        generated = generate_from_catalog(args.target * 2, attempt=attempts)
        for item in generated:
            if len(examples) >= args.target:
                break
            if item["text"] in seen:
                continue
            examples.append(item)
            seen.add(item["text"])

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


def load_seed_examples(paths: list[Path]) -> list[dict[str, Any]]:
    examples = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item["source"] = item.get("source") or "deepseek_synthetic"
                examples.append(item)
    return examples


def generate_from_catalog(limit: int, *, attempt: int = 0) -> list[dict[str, Any]]:
    products = list_products()
    products_by_category: dict[str, list[Any]] = {}
    for product in products:
        products_by_category.setdefault(product.category, []).append(product)

    examples = []
    for category, category_products in products_by_category.items():
        tags = sorted({tag for product in category_products for tag in product.tags})
        brands = sorted({product.brand for product in category_products})
        for _ in range(max(80, limit // max(len(products_by_category), 1))):
            tag = random.choice(tags) if tags else category
            brand = random.choice(brands) if brands else ""
            budget = random.choice([50, 80, 100, 150, 200, 300, 500])
            examples.extend(
                [
                    recommend_example(f"推荐几款{category}{tag}", category, [tag]),
                    recommend_example(f"我想买{budget}元以内的{tag}", category, [tag], budget_max=budget),
                    recommend_example(f"有没有{brand}的{tag}", category, [tag], brands=[brand] if brand else []),
                    recommend_example(f"帮我找个适合日常用的{tag}，别太贵", category, [tag]),
                    recommend_example(f"想看看{category}里评分高一点的{tag}", category, [tag]),
                    recommend_example(f"第{attempt}轮，给我来点{category}相关的{tag}", category, [tag]),
                    recommend_example(f"{brand}{tag}有合适的吗，预算大概{budget}" if brand else f"{tag}有合适的吗，预算大概{budget}", category, [tag], brands=[brand] if brand else [], budget_max=budget),
                    recommend_example(f"想要一个不踩雷的{tag}，偏{category}类", category, [tag]),
                    recommend_example(f"平时用的{tag}推荐一下，价格别超过{budget}", category, [tag], budget_max=budget),
                    recommend_example(f"有没有评价好一点的{category}{tag}", category, [tag]),
                    refine_example(f"预算控制在{budget}以内", budget_max=budget),
                    refine_example(f"最好是{tag}，不要太复杂", tags=[tag], negative_tags=["复杂"]),
                    refine_example(f"换成{tag}方向的，预算还是{budget}", tags=[tag], budget_max=budget),
                ]
            )

    examples.extend(meta_examples())
    random.shuffle(examples)
    return examples[:limit]


def generate_intent_support(limit: int) -> list[dict[str, Any]]:
    products = list_products()
    categories = sorted({product.category for product in products})
    tags = sorted({tag for product in products for tag in product.tags})
    refs = ["第一个", "第二个", "第三个", "这款", "刚才那个"]
    support: list[dict[str, Any]] = []

    for index in range(max(80, limit // 8)):
        tag = random.choice(tags)
        category = random.choice(categories)
        ref = random.choice(refs)
        budget = random.choice([50, 80, 100, 150, 200, 300, 500])
        support.extend(
            [
                refine_example(f"第{index}个补充条件：预算别超过{budget}", budget_max=budget),
                refine_example(f"还是想要{tag}，但别太复杂", tags=[tag], negative_tags=["复杂"]),
                refine_example(f"换成更适合日常用的{category}{tag}", tags=[tag]),
                base_item(text=f"{ref}和另一个比哪个更划算 {index}", intent="compare_products", category=""),
                base_item(text=f"帮我比较一下{ref}和第二款的区别 {index}", intent="compare_products", category=""),
                base_item(text=f"{ref}为什么推荐给我 {index}", intent="explain_recommendation", category=""),
                base_item(text=f"这款的推荐理由是什么 {index}", intent="explain_recommendation", category=""),
                base_item(
                    text=f"{ref}太贵了，换个便宜点的 {index}",
                    intent="record_feedback",
                    category="",
                    slots={
                        "budget_min": None,
                        "budget_max": None,
                        "preferred_categories": [],
                        "liked_brands": [],
                        "preferred_tags": [],
                        "negative_tags": ["太贵"],
                        "event_type": "dislike",
                        "product_refs": [ref],
                    },
                    need_recommendation=True,
                ),
                base_item(
                    text=f"我喜欢{ref}，类似的再来几个 {index}",
                    intent="record_feedback",
                    category="",
                    slots={
                        "budget_min": None,
                        "budget_max": None,
                        "preferred_categories": [],
                        "liked_brands": [],
                        "preferred_tags": [],
                        "negative_tags": [],
                        "event_type": "like",
                        "product_refs": [ref],
                    },
                    need_recommendation=True,
                ),
                base_item(text=f"{ref}库存还有多少 {index}", intent="ask_product", category=""),
                base_item(text=f"{ref}评分怎么样 {index}", intent="ask_product", category=""),
                base_item(text=f"这个商品是什么品牌 {index}", intent="ask_product", category=""),
                base_item(text=f"你好，我先随便看看 {index}", intent="smalltalk", category=""),
                base_item(text=f"你是什么模型呀 {index}", intent="smalltalk", category=""),
                base_item(text=f"今天星期几 {index}", intent="smalltalk", category=""),
            ]
        )

    random.shuffle(support)
    return support


def recommend_example(
    text: str,
    category: str,
    tags: list[str],
    *,
    brands: list[str] | None = None,
    budget_max: float | None = None,
) -> dict[str, Any]:
    return base_item(
        text=text,
        intent="recommend_products",
        category=category,
        slots={
            "budget_min": None,
            "budget_max": budget_max,
            "preferred_categories": [category],
            "liked_brands": brands or [],
            "preferred_tags": tags,
            "negative_tags": [],
            "event_type": "",
            "product_refs": [],
        },
        rewrite_queries=[f"{category} {tag}" for tag in tags],
        need_recommendation=True,
    )


def refine_example(
    text: str,
    *,
    tags: list[str] | None = None,
    negative_tags: list[str] | None = None,
    budget_max: float | None = None,
) -> dict[str, Any]:
    return base_item(
        text=text,
        intent="refine_preferences",
        category="",
        slots={
            "budget_min": None,
            "budget_max": budget_max,
            "preferred_categories": [],
            "liked_brands": [],
            "preferred_tags": tags or [],
            "negative_tags": negative_tags or [],
            "event_type": "",
            "product_refs": [],
        },
        rewrite_queries=tags or [],
        need_recommendation=True,
    )


def meta_examples() -> list[dict[str, Any]]:
    records = []
    for text in [
        "你好",
        "你是谁",
        "你是什么模型",
        "今天星期几",
        "你能看图片吗",
        "你能做什么",
        "谢谢",
        "先不用推荐",
    ]:
        records.append(base_item(text=text, intent="smalltalk", category=""))
    for text in ["第一款和第二款哪个好", "帮我比较一下前两个", "这两个哪个更划算"]:
        records.append(base_item(text=text, intent="compare_products", category=""))
    for text in ["为什么推荐第一个", "第二款为什么适合我", "推荐理由是什么"]:
        records.append(base_item(text=text, intent="explain_recommendation", category=""))
    for text in ["第二个太贵了", "我不喜欢第一款", "这款我买了", "换个便宜点的"]:
        records.append(
            base_item(
                text=text,
                intent="record_feedback",
                category="",
                slots={
                    "budget_min": None,
                    "budget_max": None,
                    "preferred_categories": [],
                    "liked_brands": [],
                    "preferred_tags": [],
                    "negative_tags": ["太贵"] if "贵" in text else [],
                    "event_type": "purchase" if "买了" in text else "dislike" if "不喜欢" in text else "",
                    "product_refs": ["第二个"] if "第二" in text else ["第一款"] if "第一" in text else ["这款"] if "这款" in text else [],
                },
            )
        )
    for text in ["这款库存怎么样", "第一款评分多少", "这个是什么品牌", "价格是多少"]:
        records.append(base_item(text=text, intent="ask_product", category=""))
    return records


def base_item(
    *,
    text: str,
    intent: str,
    category: str,
    slots: dict[str, Any] | None = None,
    rewrite_queries: list[str] | None = None,
    need_recommendation: bool = False,
) -> dict[str, Any]:
    return {
        "text": text,
        "intent": intent,
        "category": category,
        "slots": slots
        or {
            "budget_min": None,
            "budget_max": None,
            "preferred_categories": [],
            "liked_brands": [],
            "preferred_tags": [],
            "negative_tags": [],
            "event_type": "",
            "product_refs": [],
        },
        "rewrite_queries": rewrite_queries or [],
        "need_recommendation": need_recommendation,
        "need_clarify": False,
        "clarify_question": "",
        "source": "deepseek_seed_augmented",
        "teacher_model": "deepseek-v4-pro",
    }


def dedupe(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for item in examples:
        text = item.get("text")
        if text and text not in seen:
            result.append(item)
            seen.add(text)
    return result


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
    save_jsonl(eval_path, shuffled[:eval_size])
    save_jsonl(train_path, shuffled[eval_size:])


def summarize(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_intent: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for item in examples:
        by_intent[item["intent"]] = by_intent.get(item["intent"], 0) + 1
        by_category[item["category"] or "(none)"] = by_category.get(item["category"] or "(none)", 0) + 1
        by_source[item["source"]] = by_source.get(item["source"], 0) + 1
    return {
        "count": len(examples),
        "by_intent": dict(sorted(by_intent.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_source": dict(sorted(by_source.items())),
    }


if __name__ == "__main__":
    main()
