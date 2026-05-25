import pytest
from fastapi.testclient import TestClient

from app.behavior import record_event, reset_behavior_events
from app.catalog import list_products
from app.main import app
from app.models import RecommendRequest, UserEventCreate
from app.personalization import score_product
from app.recommender import recommend_products
from app.services import ab_test_engine, feature_store, metrics_collector
from app.services.vector_store import get_product_vector_store


@pytest.fixture(autouse=True)
def clear_runtime_state(monkeypatch):
    monkeypatch.setenv("PRODUCT_VECTOR_EMBEDDING_PROVIDER", "local")
    get_product_vector_store.cache_clear()
    reset_behavior_events()
    feature_store.clear_all()
    metrics_collector.reset()
    yield
    metrics_collector.reset()
    reset_behavior_events()
    feature_store.clear_all()
    get_product_vector_store.cache_clear()


def test_recommend_prefers_selected_categories():
    category = list_products()[0].category
    response = recommend_products(
        RecommendRequest(
            user_id="u001",
            num_items=2,
            preferred_categories=[category],
        )
    )

    assert len(response.products) == 2
    assert all(product.category == category for product in response.products)
    assert response.strategy.startswith("supervisor_agents")
    assert "vector_recall" in response.strategy
    assert response.experiment_group in {"control", "treatment"}
    assert "user_profile" in response.agent_results
    assert response.agent_results["product_recall"].data["mode"] == "recall"
    assert response.agent_results["product_recall"].data["backend"].startswith("chroma:")
    assert "product_rerank" in response.agent_results


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
    category = list_products()[0].category
    response = client.post(
        "/api/v1/recommend",
        json={
            "user_id": "u001",
            "num_items": 3,
            "preferred_categories": [category],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "u001"
    assert data["experiment_group"] in {"control", "treatment"}
    assert len(data["products"]) == 3


def test_out_of_stock_products_are_filtered():
    out_of_stock = next(product for product in list_products() if product.stock == 0)
    response = recommend_products(
        RecommendRequest(
            user_id="u001",
            num_items=10,
            preferred_categories=[out_of_stock.category],
            liked_brands=[out_of_stock.brand],
            preferred_tags=out_of_stock.tags,
        )
    )

    assert all(product.stock > 0 for product in response.products)
    assert all(product.product_id != out_of_stock.product_id for product in response.products)


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
    assert product.purchase_limit == 1


def test_amazon_rating_and_image_fields_are_preserved():
    category = list_products()[0].category
    response = recommend_products(
        RecommendRequest(
            user_id="u001",
            num_items=1,
            preferred_categories=[category],
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

    feature_response = client.get("/api/v1/feature-store/behavior-user")
    assert feature_response.status_code == 200
    feature_data = feature_response.json()
    if feature_data["status"]["available"]:
        assert feature_data["features"]["like_count_24h"] == 1
        assert product.brand in feature_data["features"]["recent_brands"]


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

    assert response.strategy.startswith("supervisor_agents")
    effective_request = response.agent_results["user_profile"].data["effective_request"]
    assert product.category in effective_request["preferred_categories"]
    assert response.agent_results["product_rerank"].data["mode"] == "rerank"
    feature_status = response.agent_results["user_profile"].data["feature_store"]["status"]
    assert feature_status["backend"] == "redis"


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

    assert all(item.product_id != product.product_id for item in response.products[:5])


def test_ab_assignment_is_stable_for_same_user():
    first = ab_test_engine.assign("u001")
    second = ab_test_engine.assign("u001")

    assert first.group == second.group
    assert first.experiment_id == "recommendation_strategy_v1"
    assert first.group in {"control", "treatment"}


def test_experiments_endpoint_returns_assignment():
    client = TestClient(app)
    response = client.get("/api/v1/experiments", params={"user_id": "u001"})

    assert response.status_code == 200
    data = response.json()
    assert data["default_experiment_id"] == "recommendation_strategy_v1"
    assert data["assignment"]["group"] in {"control", "treatment"}


def test_metrics_endpoint_records_agent_calls():
    client = TestClient(app)
    category = list_products()[0].category
    response = client.post(
        "/api/v1/recommend",
        json={
            "user_id": "metrics-user",
            "num_items": 2,
            "preferred_categories": [category],
        },
    )
    assert response.status_code == 200

    metrics_response = client.get("/api/v1/metrics")

    assert metrics_response.status_code == 200
    data = metrics_response.json()
    agents = {metric["agent"]: metric for metric in data["agent_metrics"]}
    assert agents["user_profile"]["call_count"] == 1
    assert agents["product_recall"]["call_count"] == 1
    assert agents["product_rerank"]["call_count"] == 1
    assert agents["inventory"]["call_count"] == 1
    assert agents["marketing_copy"]["call_count"] == 1
    assert data["business_events"]["recommend_success"] == 1


def test_vector_store_endpoint_returns_chroma_status():
    client = TestClient(app)
    response = client.get("/api/v1/vector-store")

    assert response.status_code == 200
    data = response.json()
    assert data["backend"].startswith("chroma:")
    assert data["collection"].startswith("products_")


def test_feature_store_profile_cache_is_rebuilt_from_sqlite():
    product = list_products()[0]
    record_event(
        UserEventCreate(
            user_id="cache-user",
            product_id=product.product_id,
            event_type="add_to_cart",
        )
    )

    profile = feature_store.get_cached_profile("cache-user")
    assert profile is None

    client = TestClient(app)
    profile_response = client.get("/api/v1/users/cache-user/profile")
    assert profile_response.status_code == 200

    feature_response = client.get("/api/v1/feature-store/cache-user")
    assert feature_response.status_code == 200
    data = feature_response.json()
    if data["status"]["available"]:
        assert data["cached_profile"]["user_id"] == "cache-user"
        assert data["features"]["add_to_cart_count_7d"] == 1
