from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

USD_TO_CNY = 7.2
DATASET_NAME = "Amazon Reviews 2023"
BASE_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw/meta_categories/{file_name}"
)


@dataclass(frozen=True)
class SourceFile:
    file_name: str
    category: str


SOURCE_FILES = [
    SourceFile("meta_Cell_Phones_and_Accessories.jsonl", "手机"),
    SourceFile("meta_Electronics.jsonl", "电子数码"),
    SourceFile("meta_Video_Games.jsonl", "游戏"),
    SourceFile("meta_Office_Products.jsonl", "办公"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/products_amazon_sample.csv"),
    )
    args = parser.parse_args()

    rows = list(iter_product_rows(args.limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"wrote {len(rows)} products -> {args.output}")


def iter_product_rows(limit: int) -> Iterable[dict[str, object]]:
    seen: set[str] = set()
    index = 1
    base_quota = limit // len(SOURCE_FILES)
    remainder = limit % len(SOURCE_FILES)

    for source_index, source in enumerate(SOURCE_FILES):
        quota = base_quota + (1 if source_index < remainder else 0)
        collected = 0
        url = BASE_URL.format(file_name=source.file_name)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            for raw_line in response:
                if index > limit:
                    return
                if collected >= quota:
                    break
                try:
                    item = json.loads(raw_line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                row = build_row(item, source.category, index)
                if row is None:
                    continue
                if row["product_id"] in seen:
                    continue

                seen.add(str(row["product_id"]))
                yield row
                collected += 1
                index += 1


def build_row(
    item: dict[str, object],
    category: str,
    index: int,
) -> dict[str, object] | None:
    title = clean_text(item.get("title"))
    brand = clean_text(item.get("store"))
    price = parse_price(item.get("price"))
    asin = clean_text(item.get("parent_asin"))

    if not title or not brand or price is None or not asin:
        return None

    image_url = first_image_url(item.get("images"))
    source_category = clean_text(item.get("main_category")) or category
    rating = parse_float(item.get("average_rating"))
    rating_count = parse_int(item.get("rating_number"))
    cny_price = round(price * USD_TO_CNY)

    return {
        "product_id": asin,
        "name": build_display_name(title, brand, category),
        "category": category,
        "brand": brand,
        "price": cny_price,
        "stock": deterministic_stock(asin, index),
        "tags": "|".join(build_tags(item, title, category)),
        "source_name": title,
        "source_category": source_category,
        "source_dataset": DATASET_NAME,
        "image_url": image_url,
        "rating": rating if rating is not None else "",
        "rating_count": rating_count if rating_count is not None else "",
    }


def build_tags(item: dict[str, object], title: str, category: str) -> list[str]:
    tags = [category]
    lower_title = title.lower()
    categories = item.get("categories")

    if isinstance(categories, list):
        for value in flatten(categories):
            mapped = map_tag(str(value))
            if mapped:
                tags.append(mapped)

    keyword_map = {
        "case": "保护壳",
        "protector": "保护膜",
        "charger": "充电器",
        "cable": "数据线",
        "keyboard": "键盘",
        "mouse": "鼠标",
        "headset": "耳机",
        "playstation": "PlayStation",
        "nintendo": "任天堂",
        "office": "办公",
        "camera": "摄像头",
        "drive": "存储",
    }
    for keyword, tag in keyword_map.items():
        if keyword in lower_title:
            tags.append(tag)

    return unique(tags)[:5]


def build_display_name(title: str, brand: str, category: str) -> str:
    lower_title = title.lower()
    keyword_names = [
        ("screen protector", "屏幕保护膜"),
        ("lens protector", "镜头保护膜"),
        ("case", "保护壳"),
        ("cover", "保护套"),
        ("charger", "充电器"),
        ("cable", "数据线"),
        ("power bank", "移动电源"),
        ("headset", "耳机"),
        ("headphone", "耳机"),
        ("hard drive", "移动硬盘"),
        ("flash drive", "U 盘"),
        ("camera", "摄像头"),
        ("dvr", "安防录像机"),
        ("keyboard", "键盘"),
        ("mouse pad", "鼠标垫"),
        ("mouse", "鼠标"),
        ("calculator", "计算器"),
        ("stapler", "订书机"),
        ("playstation", "PlayStation 游戏"),
        ("nintendo", "任天堂游戏"),
        ("xbox", "Xbox 游戏"),
        ("game", "游戏商品"),
    ]
    for keyword, chinese_name in keyword_names:
        if keyword in lower_title:
            return f"{brand} {chinese_name}"
    return f"{brand} {category}商品"


def map_tag(value: str) -> str | None:
    normalized = value.lower()
    mapping = {
        "cell phones & accessories": "手机配件",
        "cases, holsters & sleeves": "保护壳",
        "screen protectors": "保护膜",
        "electronics": "电子产品",
        "computers & accessories": "电脑配件",
        "video games": "游戏",
        "playstation 4": "PlayStation",
        "office products": "办公",
        "calculators": "计算器",
    }
    return mapping.get(normalized)


def deterministic_stock(asin: str, index: int) -> int:
    digest = hashlib.sha256(asin.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 900
    if index % 97 == 0:
        return 0
    if index % 31 == 0:
        return 50 + value % 50
    return 100 + value


def first_image_url(images: object) -> str:
    if not isinstance(images, list) or not images:
        return ""
    first = images[0]
    if not isinstance(first, dict):
        return ""
    for key in ("large", "hi_res", "thumb"):
        value = clean_text(first.get(key))
        if value:
            return value
    return ""


def parse_price(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None
    return float(match.group())


def parse_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def flatten(values: list[object]) -> Iterable[object]:
    for value in values:
        if isinstance(value, list):
            yield from flatten(value)
        else:
            yield value


def unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "product_id",
        "name",
        "category",
        "brand",
        "price",
        "stock",
        "tags",
        "source_name",
        "source_category",
        "source_dataset",
        "image_url",
        "rating",
        "rating_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
