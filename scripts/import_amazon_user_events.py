from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog import list_products  # noqa: E402
from app.models import Product  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "amazon_user_events_sample.csv"
POSITIVE_EVENTS = {"like", "purchase"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Amazon Reviews style user-item-rating rows into project user "
            "events, or generate a small weak-label sample from the current product catalog."
        )
    )
    parser.add_argument("--input", type=Path, help="Raw Amazon reviews jsonl/jsonl.gz/csv file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-events", type=int, default=500)
    parser.add_argument("--min-rating-positive", type=float, default=4.0)
    parser.add_argument("--generate-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    products = list_products()
    product_ids = {product.product_id for product in products}

    if args.generate_sample or args.input is None:
        events = generate_sample_events(products, max_events=args.max_events, seed=args.seed)
    else:
        events = convert_raw_events(
            args.input,
            product_ids=product_ids,
            max_events=args.max_events,
            min_rating_positive=args.min_rating_positive,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_events(args.output, events)
    print(f"Wrote {len(events)} events to {args.output}")


def convert_raw_events(
    input_path: Path,
    *,
    product_ids: set[str],
    max_events: int,
    min_rating_positive: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in read_rows(input_path):
        product_id = first_value(row, "parent_asin", "asin", "product_id", "item_id")
        if not product_id or product_id not in product_ids:
            continue

        rating = parse_float(first_value(row, "rating", "overall", "score"))
        if rating is None:
            continue

        user_id = first_value(row, "user_id", "reviewerID", "reviewer_id")
        if not user_id:
            continue

        timestamp = normalize_timestamp(first_value(row, "timestamp", "unixReviewTime", "time"))
        event_type = rating_to_event_type(rating, min_rating_positive=min_rating_positive)
        events.append(
            {
                "user_id": str(user_id),
                "product_id": str(product_id),
                "event_type": event_type,
                "rating": rating,
                "timestamp": timestamp,
                "source": "amazon_reviews_2023",
            }
        )
        if len(events) >= max_events:
            break
    return events


def generate_sample_events(
    products: list[Product],
    *,
    max_events: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_category: dict[str, list[Product]] = {}
    by_brand: dict[str, list[Product]] = {}
    for product in products:
        by_category.setdefault(product.category, []).append(product)
        by_brand.setdefault(product.brand, []).append(product)

    categories = [category for category, items in by_category.items() if len(items) >= 3]
    brands = [brand for brand, items in by_brand.items() if len(items) >= 2]
    start_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    events: list[dict[str, Any]] = []

    user_count = max(12, min(80, max_events // 8))
    for user_index in range(user_count):
        category = rng.choice(categories)
        brand = rng.choice(brands)
        preferred_pool = unique_products(
            [*by_category.get(category, []), *by_brand.get(brand, [])]
        )
        if len(preferred_pool) < 4:
            preferred_pool = by_category[category]

        noise_pool = [
            product
            for product in products
            if product.category != category and product.brand != brand
        ]
        history_size = rng.randint(5, 9)
        sampled = rng.sample(preferred_pool, min(history_size, len(preferred_pool)))
        if noise_pool:
            sampled.extend(rng.sample(noise_pool, min(2, len(noise_pool))))

        for event_index, product in enumerate(sampled):
            rating = weak_rating(product, preferred_category=category, preferred_brand=brand, rng=rng)
            event_time = start_time + timedelta(days=user_index, minutes=event_index * 17)
            events.append(
                {
                    "user_id": f"amazon-u{user_index + 1:03d}",
                    "product_id": product.product_id,
                    "event_type": rating_to_event_type(rating),
                    "rating": rating,
                    "timestamp": int(event_time.timestamp()),
                    "source": "amazon_metadata_weak_label",
                }
            )
            if len(events) >= max_events:
                return events
    return events


def weak_rating(
    product: Product,
    *,
    preferred_category: str,
    preferred_brand: str,
    rng: random.Random,
) -> float:
    score = 3.0
    if product.category == preferred_category:
        score += 0.8
    if product.brand == preferred_brand:
        score += 0.7
    if product.rating:
        score += (product.rating - 3.0) * 0.35
    score += rng.uniform(-0.6, 0.6)
    return round(max(1.0, min(5.0, score)), 1)


def read_rows(input_path: Path) -> Iterable[dict[str, Any]]:
    suffixes = "".join(input_path.suffixes).lower()
    if suffixes.endswith(".jsonl.gz"):
        with gzip.open(input_path, "rt", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)
        return

    if suffixes.endswith(".jsonl"):
        with input_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)
        return

    if input_path.suffix.lower() == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as file:
            yield from csv.DictReader(file)
        return

    raise ValueError(f"Unsupported input file: {input_path}")


def write_events(output_path: Path, events: list[dict[str, Any]]) -> None:
    fieldnames = ["user_id", "product_id", "event_type", "rating", "timestamp", "source"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


def rating_to_event_type(rating: float, *, min_rating_positive: float = 4.0) -> str:
    if rating >= 4.8:
        return "purchase"
    if rating >= min_rating_positive:
        return "like"
    if rating <= 2.0:
        return "dislike"
    return "view"


def normalize_timestamp(value: Any) -> int:
    if value is None or value == "":
        return int(datetime.now(timezone.utc).timestamp())
    if isinstance(value, (int, float)):
        timestamp = float(value)
    else:
        text = str(value).strip()
        if text.isdigit():
            timestamp = float(text)
        else:
            return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    if timestamp > 10_000_000_000:
        timestamp = timestamp / 1000
    return int(timestamp)


def first_value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return value
    return None


def parse_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def unique_products(products: Iterable[Product]) -> list[Product]:
    result: list[Product] = []
    seen: set[str] = set()
    for product in products:
        if product.product_id in seen:
            continue
        seen.add(product.product_id)
        result.append(product)
    return result


if __name__ == "__main__":
    main()
