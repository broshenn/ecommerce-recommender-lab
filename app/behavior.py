from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from app.catalog import list_products
from app.models import RecommendRequest, UserEvent, UserEventCreate, UserProfile

_EVENTS_BY_USER: dict[str, list[UserEvent]] = defaultdict(list)
_NEXT_EVENT_ID = 1


def record_event(event: UserEventCreate) -> UserEvent:
    global _NEXT_EVENT_ID

    stored_event = UserEvent(
        **event.model_dump(),
        event_id=_NEXT_EVENT_ID,
        created_at=datetime.utcnow(),
    )
    _NEXT_EVENT_ID += 1
    _EVENTS_BY_USER[event.user_id].append(stored_event)
    return stored_event


def list_user_events(user_id: str) -> list[UserEvent]:
    return list(_EVENTS_BY_USER.get(user_id, []))


def build_user_profile(user_id: str) -> UserProfile:
    events = list_user_events(user_id)
    products_by_id = {product.product_id: product for product in list_products()}

    preferred_categories: list[str] = []
    liked_brands: list[str] = []
    preferred_tags: list[str] = []
    recent_views: list[str] = []
    disliked_products: list[str] = []
    cart_items: list[str] = []

    for event in events:
        product = products_by_id.get(event.product_id)

        if event.event_type == "view":
            recent_views.append(event.product_id)

        if event.event_type == "dislike":
            disliked_products.append(event.product_id)
            continue

        if event.event_type == "add_to_cart":
            cart_items.append(event.product_id)

        if event.event_type in {"like", "add_to_cart"} and product:
            preferred_categories.append(product.category)
            liked_brands.append(product.brand)
            preferred_tags.extend(product.tags)

    return UserProfile(
        user_id=user_id,
        preferred_categories=_unique(preferred_categories),
        liked_brands=_unique(liked_brands),
        preferred_tags=_unique(preferred_tags),
        recent_views=_unique(reversed(recent_views))[:20],
        disliked_products=_unique(disliked_products),
        cart_items=_unique(cart_items),
        event_count=len(events),
    )


def merge_behavior_profile(request: RecommendRequest) -> RecommendRequest:
    profile = build_user_profile(request.user_id)
    return request.model_copy(
        update={
            "preferred_categories": _unique(
                [*request.preferred_categories, *profile.preferred_categories]
            ),
            "liked_brands": _unique([*request.liked_brands, *profile.liked_brands]),
            "preferred_tags": _unique([*request.preferred_tags, *profile.preferred_tags]),
            "recent_views": _unique([*request.recent_views, *profile.recent_views]),
            "disliked_products": _unique(
                [*request.disliked_products, *profile.disliked_products]
            ),
        }
    )


def reset_behavior_events() -> None:
    global _NEXT_EVENT_ID
    _EVENTS_BY_USER.clear()
    _NEXT_EVENT_ID = 1


def _unique(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
