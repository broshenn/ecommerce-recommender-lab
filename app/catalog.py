from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from app.models import Product

BASE_DIR = Path(__file__).resolve().parent.parent
PRODUCTS_FILE = BASE_DIR / "data" / "products_amazon_sample.csv"


@lru_cache(maxsize=1)
def _load_products() -> tuple[Product, ...]:
    with PRODUCTS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        products = [_row_to_product(row) for row in rows]
    return tuple(products)


def list_products() -> list[Product]:
    return [product.model_copy(deep=True) for product in _load_products()]


def _row_to_product(row: dict[str, str]) -> Product:
    return Product(
        product_id=row["product_id"],
        name=row["name"],
        category=row["category"],
        price=float(row["price"]),
        brand=row["brand"],
        stock=int(row["stock"]),
        tags=_split_tags(row.get("tags", "")),
        source_name=_empty_to_none(row.get("source_name")),
        source_category=_empty_to_none(row.get("source_category")),
        source_dataset=row.get("source_dataset") or "Amazon Reviews 2023",
        image_url=_empty_to_none(row.get("image_url")),
        rating=_float_or_none(row.get("rating")),
        rating_count=_int_or_none(row.get("rating_count")),
    )


def _split_tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.split("|") if tag.strip()]


def _empty_to_none(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    return value or None


def _float_or_none(value: str | None) -> float | None:
    if not value:
        return None
    return float(value)


def _int_or_none(value: str | None) -> int | None:
    if not value:
        return None
    return int(value)
