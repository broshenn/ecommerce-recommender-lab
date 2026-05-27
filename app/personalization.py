from __future__ import annotations

from dataclasses import dataclass

from app.models import Product, RecommendRequest


@dataclass(frozen=True)
class ProductScore:
    value: float
    reason: str


def score_product(product: Product, request: RecommendRequest) -> ProductScore:
    """根据当前用户的轻量画像给商品打规则分。"""
    score = 0.0
    reasons: list[str] = []

    if _matches(product.category, request.preferred_categories):
        score += 40
        reasons.append("类目匹配")

    if _matches(product.brand, request.liked_brands):
        score += 25
        reasons.append("品牌匹配")

    matched_tags = [
        tag for tag in product.tags
        if _matches(tag, request.preferred_tags)
    ]
    if matched_tags:
        score += len(matched_tags) * 10
        reasons.append("标签匹配：" + "、".join(matched_tags))

    if _is_in_budget(product, request):
        score += 20
        reasons.append("价格符合预算")
    elif request.budget_min is not None or request.budget_max is not None:
        score -= 20
        reasons.append("价格超出预算")

    if product.rating is not None:
        rating_score = product.rating * 4
        score += rating_score
        reasons.append(f"评分 {product.rating:.1f}")

    if product.product_id in set(request.disliked_products):
        score -= 100
        reasons.append("用户不感兴趣，强降权")

    if product.product_id in set(request.recent_views):
        score -= 30
        reasons.append("最近浏览过，降低重复推荐")

    if not reasons:
        reasons.append("默认排序")

    return ProductScore(
        value=round(score, 2),
        reason="；".join(reasons),
    )


def _matches(value: str, candidates: list[str]) -> bool:
    return value.lower() in {candidate.lower() for candidate in candidates}


def _is_in_budget(product: Product, request: RecommendRequest) -> bool:
    if request.budget_min is not None and product.price < request.budget_min:
        return False
    if request.budget_max is not None and product.price > request.budget_max:
        return False
    return request.budget_min is not None or request.budget_max is not None
