from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.marketing_copy_agent import SEGMENT_TEMPLATES  # noqa: E402
from app.catalog import list_products  # noqa: E402
from scripts.generate_copy_sft_data import OUTPUT_RULES, build_user_message  # noqa: E402


PROMO_PATTERN = re.compile(
    r"(原价|折扣|优惠券|优惠码|限时|买一送一|满减|赠品|促销|见面礼|专享|特惠|优惠|低至|元/件|单价|首单)"
)
PRICE_PATTERN = re.compile(r"(?:¥\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*元)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training/data/copy_sft_raw.jsonl")
    parser.add_argument("--output", default="training/data/copy_sft_prepared.jsonl")
    parser.add_argument("--llamafactory-output", default="")
    parser.add_argument("--augment-price-products", type=int, default=0)
    parser.add_argument("--augment-new-user-products", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    products = list_products()
    product_by_id = {product.product_id: product for product in products}
    rows, rejected = clean_rows(PROJECT_ROOT / args.input, product_by_id)
    rows.extend(build_augmented_rows(products[: args.augment_price_products], "price_sensitive", args.batch_size))
    rows.extend(build_augmented_rows(products[: args.augment_new_user_products], "new_user", args.batch_size))

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    if args.llamafactory_output:
        llama_path = Path(args.llamafactory_output)
        llama_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(llama_path, rows)

    print(
        json.dumps(
            {
                "prepared_records": len(rows),
                "segments": Counter(row["segment"] for row in rows),
                "rejected_records": len(rejected),
                "rejected_preview": rejected[:10],
                "output": str(output_path),
                "llamafactory_output": args.llamafactory_output,
            },
            ensure_ascii=False,
            default=dict,
            indent=2,
        )
    )


def clean_rows(input_path: Path, product_by_id: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    rejected = []
    for line_no, line in enumerate(input_path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        reasons = validate_record(record, product_by_id)
        if reasons:
            rejected.append({"line": line_no, "segment": record.get("segment"), "reasons": reasons[:3]})
            continue
        rows.append(record)
    return rows, rejected


def validate_record(record: dict[str, Any], product_by_id: dict[str, Any]) -> list[str]:
    reasons = []
    try:
        output = json.loads(record["output"])
    except (KeyError, json.JSONDecodeError):
        return ["invalid_output_json"]

    product_ids = record.get("product_ids", [])
    if len(output) != len(product_ids):
        reasons.append("length_mismatch")
    if any("信息清楚" in item.get("copy", "") for item in output):
        reasons.append("fallback_copy")
    if any(PROMO_PATTERN.search(item.get("copy", "")) for item in output):
        reasons.append("promo_word")

    for item in output:
        product_id = item.get("product_id")
        product = product_by_id.get(product_id)
        if not product:
            reasons.append(f"unknown_product:{product_id}")
            continue
        for match in PRICE_PATTERN.finditer(item.get("copy", "")):
            number = match.group(1) or match.group(2)
            if int(round(float(number))) != int(round(float(product.price))):
                reasons.append(f"price_mismatch:{product_id}:{number}!={product.price}")
                break
    return reasons


def build_augmented_rows(products: list[Any], segment: str, batch_size: int) -> list[dict[str, Any]]:
    rows = []
    for batch in chunks(products, batch_size):
        rows.append(build_rule_record(segment, batch))
    return rows


def build_rule_record(segment: str, products: list[Any]) -> dict[str, Any]:
    system_prompt = f"{SEGMENT_TEMPLATES[segment]}\n\n{OUTPUT_RULES}"
    user_message = build_user_message(segment, products)
    answer = [{"product_id": product.product_id, "copy": rule_copy(segment, product)} for product in products]
    return {
        "task": "marketing_copy",
        "segment": segment,
        "product_ids": [product.product_id for product in products],
        "instruction": "请根据用户分群和商品真实字段，为每个商品生成一条电商推荐文案。",
        "input": user_message,
        "output": json.dumps(answer, ensure_ascii=False),
        "system": system_prompt,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ],
    }


def rule_copy(segment: str, product: Any) -> str:
    name = "".join(str(product.name).split())[:16]
    price = int(round(float(product.price)))
    rating = product.rating or "暂无"
    if segment == "price_sensitive":
        return f"{name}¥{price}，价格透明预算友好，评分{rating}，日常实用不虚夸。"
    if segment == "new_user":
        return f"新手可看{name}¥{price}，信息清楚易判断，先从实用配件开始。"
    return f"{name}¥{price}，字段真实清楚，结合当前偏好稳妥选择。"


def chunks(items: list[Any], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
