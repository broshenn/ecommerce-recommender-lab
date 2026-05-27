from __future__ import annotations

from app.models import Product, RecommendedProduct

LOW_STOCK_THRESHOLD = 100
HOT_ITEM_LIMIT_THRESHOLD = 300


def is_available(product: Product) -> bool:
    """判断商品是否可推荐。"""
    return product.stock > 0


def enrich_inventory(
    product: Product,
    *,
    recommendation_score: float = 0,
    recommendation_reason: str = "基础推荐",
) -> RecommendedProduct:
    """为商品补充库存状态和限购信息。"""
    stock_status = "normal"
    stock_message = "库存充足"
    purchase_limit: int | None = None

    if product.stock <= LOW_STOCK_THRESHOLD:
        stock_status = "low"
        stock_message = "库存紧张"
        purchase_limit = 1
    elif _is_hot_product(product) and product.stock <= HOT_ITEM_LIMIT_THRESHOLD:
        stock_status = "limited"
        stock_message = "热销限购"
        purchase_limit = 2

    return RecommendedProduct(
        **product.model_dump(),
        stock_status=stock_status,
        stock_message=stock_message,
        purchase_limit=purchase_limit,
        recommendation_score=recommendation_score,
        recommendation_reason=recommendation_reason,
    )


def _is_hot_product(product: Product) -> bool:
    hot_tags = {"旗舰", "新品", "热销"}
    return bool(hot_tags.intersection(product.tags))
