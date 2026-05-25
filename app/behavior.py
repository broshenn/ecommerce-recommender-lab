from __future__ import annotations

from datetime import datetime, timezone

from app.catalog import list_products
from app.database import get_connection, init_db
from app.models import RecommendRequest, UserEvent, UserEventCreate, UserProfile
from app.services import feature_store


def record_event(event: UserEventCreate) -> UserEvent:
    init_db()
    created_at = datetime.now(timezone.utc)

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO user_events (user_id, product_id, event_type, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                event.user_id,
                event.product_id,
                event.event_type,
                created_at.isoformat(),
            ),
        )
        event_id = int(cursor.lastrowid)

    stored_event = UserEvent(
        **event.model_dump(),
        event_id=event_id,
        created_at=created_at,
    )
    products_by_id = {product.product_id: product for product in list_products()}
    feature_store.invalidate_profile(stored_event.user_id)
    feature_store.record_behavior(
        stored_event,
        products_by_id.get(stored_event.product_id),
    )
    return stored_event


def list_user_events(user_id: str) -> list[UserEvent]:
    init_db()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT event_id, user_id, product_id, event_type, created_at
            FROM user_events
            WHERE user_id = ?
            ORDER BY event_id ASC
            """,
            (user_id,),
        ).fetchall()

    return [_row_to_event(row) for row in rows]


def build_user_profile(user_id: str) -> UserProfile:
    cached_profile = feature_store.get_cached_profile(user_id)
    if cached_profile:
        return cached_profile

    profile = _build_user_profile_from_sqlite(user_id)
    feature_store.set_cached_profile(profile)
    return profile


def _build_user_profile_from_sqlite(user_id: str) -> UserProfile:
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
    init_db()
    with get_connection() as connection:
        connection.execute("DELETE FROM user_events")
        connection.execute(
            "DELETE FROM sqlite_sequence WHERE name = ?",
            ("user_events",),
        )
    feature_store.clear_all()


def _row_to_event(row) -> UserEvent:
    return UserEvent(
        event_id=row["event_id"],
        user_id=row["user_id"],
        product_id=row["product_id"],
        event_type=row["event_type"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _unique(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
