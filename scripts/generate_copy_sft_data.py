from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.marketing_copy_agent import SEGMENT_TEMPLATES  # noqa: E402
from app.catalog import list_products  # noqa: E402


SEGMENTS = [
    "new_user",
    "active",
    "high_value",
    "price_sensitive",
    "churn_risk",
    "category_explorer",
    "brand_loyal",
]

OUTPUT_RULES = """输出要求:
1. 只输出 JSON 数组, 格式为 [{"product_id": "...", "copy": "..."}]。
2. 每条 copy 去掉空格后必须 25-40 个中文字符左右。
3. 只能使用商品列表提供的真实字段。
4. 没有 original_price、discount、coupon、promotion 字段时, 禁止写原价、现价、折扣、优惠、限时、省钱金额、买一送一。
5. 可以表达价格透明、预算友好、性价比, 但不能编造促销事实。
6. 文案必须有当前分群辨识度, 但不要夸大功效。
7. 禁止写低至、起、元/件、单价、首单、专享、特惠、优惠、送礼等未提供价格或活动口径。
8. 若表达价格敏感, 只能写商品字段中的 price, 并用价格透明、预算友好、实用等表述。"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training/data/copy_sft.jsonl")
    parser.add_argument("--limit-products", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY"),
        base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        timeout=60,
        max_retries=2,
    )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = _load_done_keys(output_path)

    products = list_products()
    if args.limit_products > 0:
        products = products[: args.limit_products]

    written = 0
    with output_path.open("a", encoding="utf-8") as file:
        for segment in SEGMENTS:
            for batch in _chunks(products, args.batch_size):
                key = _record_key(segment, batch)
                if key in done_keys:
                    continue

                record = build_training_record(
                    client=client,
                    model=normalize_model(args.model),
                    segment=segment,
                    products=batch,
                    max_tokens=args.max_tokens,
                )
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                written += 1
                print(f"written={written} segment={segment} products={len(batch)}")
                time.sleep(args.sleep)

    print(f"done output={output_path} new_records={written}")


def build_training_record(
    *,
    client: OpenAI,
    model: str,
    segment: str,
    products: list[Any],
    max_tokens: int,
) -> dict[str, Any]:
    system_prompt = f"{SEGMENT_TEMPLATES[segment]}\n\n{OUTPUT_RULES}"
    user_message = build_user_message(segment, products)
    answer = call_teacher(client, model, system_prompt, user_message, products, max_tokens)
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


def call_teacher(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_message: str,
    products: list[Any],
    max_tokens: int,
) -> list[dict[str, str]]:
    valid_ids = {product.product_id for product in products}
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            parsed = _parse_json_array(content)
            normalized = _normalize_answer(parsed, valid_ids)
            if len(normalized) == len(products):
                return normalized
            print(f"retry teacher output attempt={attempt + 1} raw={content[:200]}")
        except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
            wait_seconds = min(2**attempt, 20)
            print(f"retry teacher request attempt={attempt + 1} error={exc!r} wait={wait_seconds}s")
            time.sleep(wait_seconds)
    return [{"product_id": product.product_id, "copy": fallback_copy(product)} for product in products]


def build_user_message(segment: str, products: list[Any]) -> str:
    lines = [
        f"用户分群: {segment}",
        "可用字段: product_id、名称、类目、价格、品牌、标签、评分、库存。",
        "不可用字段: 原价、折扣、优惠券、满减、限时活动、赠品、邮费。",
        "商品列表:",
    ]
    for product in products:
        lines.append(
            (
                f"- ID:{product.product_id} 名称:{product.name} 类目:{product.category} "
                f"价格:¥{product.price} 品牌:{product.brand} 标签:{','.join(product.tags)} "
                f"评分:{product.rating or '无'} 库存:{product.stock}"
            )
        )
    return "\n".join(lines)


def fallback_copy(product: Any) -> str:
    name = "".join(str(product.name).split())[:18]
    return f"{name}信息清楚，价格透明，适合结合当前偏好稳妥选择。"


def normalize_model(model: str) -> str:
    model = model.strip()
    if model.endswith("]") and "[" in model:
        return model.rsplit("[", 1)[0]
    return model


def _parse_json_array(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else text.strip("`")
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _normalize_answer(parsed: Any, valid_ids: set[str]) -> list[dict[str, str]]:
    if not isinstance(parsed, list):
        return []
    result = []
    seen = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        product_id = str(item.get("product_id", "")).strip()
        copy = str(item.get("copy") or item.get("text") or "").strip()
        if product_id in valid_ids and copy and product_id not in seen:
            result.append({"product_id": product_id, "copy": copy})
            seen.add(product_id)
    return result


def _chunks(items: list[Any], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _record_key(segment: str, products: list[Any]) -> str:
    return f"{segment}:{','.join(product.product_id for product in products)}"


def _load_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            product_ids = record.get("product_ids", [])
            keys.add(f"{record.get('segment')}:{','.join(product_ids)}")
    return keys


if __name__ == "__main__":
    main()
