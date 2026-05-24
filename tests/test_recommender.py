import pytest
from fastapi.testclient import TestClient

from app.behavior import record_event, reset_behavior_events
from app.catalog import list_products
from app.main import app
from app.models import RecommendRequest, UserEventCreate
from app.personalization import score_product
from app.recommender import recommend_products


@pytest.fixture(autouse=True)
def clear_behavior_events():
    reset_behavior_events()
    yield
    reset_behavior_events()


def test_recommend_prefers_selected_categories():
    response = recommend_products(
        RecommendRequest(
            user_id="u001",
            num_items=2,
            preferred_categories=["电子数码"],
        )
    )

    assert len(response.products) == 2
    assert all(product.category == "电子数码" for product in response.products)
    assert response.strategy == "behavior_profile_scoring+inventory_filter"


def test_products_endpoint_returns_catalog():
    client = TestClient(app)
    response = client.get("/api/v1/products")

    assert response.status_code == 200
    products = response.json()
    assert len(products) >= 10
    assert products[0]["source_dataset"] == "Amazon Reviews 2023"
    assert products[0]["source_name"]


def test_recommend_endpoint_returns_products():
    client = TestClient(app)
    response = client.post(
        "/api/v1/recommend",
        json={
            "user_id": "u001",
            "num_items": 3,
            "preferred_categories": ["手机"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "u001"
    assert len(data["products"]) == 3


def test_out_of_stock_products_are_filtered():
    response = recommend_products(
        RecommendRequest(
            user_id="u001",
            num_items=3,
            preferred_categories=["游戏"],
        )
    )

    assert all(product.stock > 0 for product in response.products)
    assert all(product.product_id != "AMZ009" for product in response.products)


def test_low_stock_product_gets_inventory_message():
    low_stock_product = next(
        product for product in list_products()
        if 0 < product.stock <= 100
    )
    response = recommend_products(
        RecommendRequest(
            user_id="u001",
            num_items=1,
            preferred_categories=[low_stock_product.category],
            liked_brands=[low_stock_product.brand],
            preferred_tags=low_stock_product.tags,
            budget_min=0,
            budget_max=low_stock_product.price + 1,
        )
    )

    product = response.products[0]
    assert product.product_id == low_stock_product.product_id
    assert product.stock_status == "low"
    assert product.stock_message == "库存紧张"
    assert product.purchase_limit == 1


def test_amazon_rating_and_image_fields_are_preserved():
    response = recommend_products(
        RecommendRequest(
            user_id="u001",
            num_items=1,
            preferred_categories=["手机"],
        )
    )

    product = response.products[0]
    assert product.source_name
    assert product.image_url
    assert product.rating is not None
    assert product.rating_count is not None


def test_profile_score_uses_brand_tags_budget_and_recent_views():
    products = list_products()
    recently_viewed = products[0]
    base_request = RecommendRequest(
        user_id="u001",
        preferred_categories=[recently_viewed.category],
        liked_brands=[recently_viewed.brand],
        preferred_tags=recently_viewed.tags,
        budget_min=0,
        budget_max=recently_viewed.price + 1,
    )
    normal_score = score_product(recently_viewed, base_request)
    viewed_score = score_product(
        recently_viewed,
        base_request.model_copy(
            update={"recent_views": [recently_viewed.product_id]}
        ),
    )

    assert normal_score.value > viewed_score.value
    assert "价格符合预算" in normal_score.reason
    assert "最近浏览过" in viewed_score.reason


def test_event_endpoint_updates_user_profile():
    client = TestClient(app)
    product = list_products()[0]

    response = client.post(
        "/api/v1/events",
        json={
            "user_id": "behavior-user",
            "product_id": product.product_id,
            "event_type": "like",
        },
    )

    assert response.status_code == 200
    assert response.json()["event_id"] == 1

    profile_response = client.get("/api/v1/users/behavior-user/profile")
    assert profile_response.status_code == 200
    profile = profile_response.json()
    assert profile["event_count"] == 1
    assert product.category in profile["preferred_categories"]
    assert product.brand in profile["liked_brands"]


def test_events_are_loaded_from_sqlite_storage():
    product = list_products()[0]
    stored_event = record_event(
        UserEventCreate(
            user_id="sqlite-user",
            product_id=product.product_id,
            event_type="view",
        )
    )

    client = TestClient(app)
    events_response = client.get("/api/v1/users/sqlite-user/events")

    assert events_response.status_code == 200
    events = events_response.json()
    assert events[0]["event_id"] == stored_event.event_id
    assert events[0]["product_id"] == product.product_id
    assert events[0]["event_type"] == "view"


def test_recorded_behavior_affects_recommendation_profile():
    product = list_products()[0]
    record_event(
        UserEventCreate(
            user_id="auto-profile-user",
            product_id=product.product_id,
            event_type="add_to_cart",
        )
    )

    response = recommend_products(
        RecommendRequest(
            user_id="auto-profile-user",
            num_items=3,
        )
    )

    assert response.strategy == "behavior_profile_scoring+inventory_filter"
    assert response.products[0].category == product.category
    assert "类目匹配" in response.products[0].recommendation_reason


def test_dislike_event_strongly_demotes_product():
    product = list_products()[0]
    record_event(
        UserEventCreate(
            user_id="dislike-user",
            product_id=product.product_id,
            event_type="dislike",
        )
    )

    response = recommend_products(
        RecommendRequest(
            user_id="dislike-user",
            num_items=20,
            preferred_categories=[product.category],
            liked_brands=[product.brand],
            preferred_tags=product.tags,
        )
    )

    disliked = [
        item for item in response.products
        if item.product_id == product.product_id
    ]
    assert not disliked or "强降权" in disliked[0].recommendation_reason
