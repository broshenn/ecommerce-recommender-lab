from __future__ import annotations

from app.behavior import merge_behavior_profile
from app.catalog import list_products
from app.inventory import enrich_inventory, is_available
from app.models import RecommendRequest, RecommendResponse
from app.personalization import score_product


def recommend_products(request: RecommendRequest) -> RecommendResponse:
    """Return products using profile-based scoring.

    Step 4 scores every available product against the current user's profile,
    then returns the highest scoring products with inventory status attached.
    """
    effective_request = merge_behavior_profile(request)
    all_products = list_products()
    available_products = [product for product in all_products if is_available(product)]

    scored_products = [
        (score_product(product, effective_request), product)
        for product in available_products
    ]
    scored_products.sort(
        key=lambda item: (
            item[0].value,
            item[1].rating or 0,
            item[1].rating_count or 0,
        ),
        reverse=True,
    )

    products = [
        enrich_inventory(
            product,
            recommendation_score=score.value,
            recommendation_reason=score.reason,
        )
        for score, product in scored_products[: request.num_items]
    ]
    strategy = "behavior_profile_scoring+inventory_filter"
    reason = "合并手动画像与当前用户行为画像后，对商品打分排序，并过滤缺货商品。"

    return RecommendResponse(
        user_id=request.user_id,
        scene=request.scene,
        products=products,
        strategy=strategy,
        reason=reason,
    )
