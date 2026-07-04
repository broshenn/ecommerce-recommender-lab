import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.behavior import record_event, reset_behavior_events
from app.catalog import list_products
from app.agents.marketing_copy_agent import MarketingCopyAgent
from app.agents.product_rec_agent import ProductRecAgent
from app.main import app
from app.models import RecommendRequest, UserEventCreate
from app.orchestrator.chat import chat_orchestrator
from app.personalization import score_product
from app.recommender import recommend_products
from app.services import ab_test_engine, feature_store, llm_client, metrics_collector
from app.services.llm_client import LLMClient
from app.services.vector_store import get_product_vector_store
from scripts.evaluate_chat_agent import evaluate_chat_agent
from scripts.evaluate_recommendation_offline import ndcg_at_k
from scripts.import_amazon_user_events import normalize_timestamp, rating_to_event_type


@pytest.fixture(autouse=True)
def clear_runtime_state(monkeypatch):
    monkeypatch.setenv("PRODUCT_VECTOR_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LLM_API_KEY", "")
    llm_client.api_key = ""
    llm_client._client = None
    get_product_vector_store.cache_clear()
    reset_behavior_events()
    feature_store.clear_all()
    chat_orchestrator.memory.clear_all()
    ab_test_engine.reset_outcomes()
    metrics_collector.reset()
    yield
    metrics_collector.reset()
    ab_test_engine.reset_outcomes()
    reset_behavior_events()
    feature_store.clear_all()
    chat_orchestrator.memory.clear_all()
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
    llm_profile = response.agent_results["user_profile"].data["llm_profile"]
    assert llm_profile["intent_summary"] == "LLM不可用，默认画像"
    recall_data = response.agent_results["product_recall"].data
    assert recall_data["mode"] in {"recall", "rerank"}
    if recall_data["mode"] == "recall":
        assert recall_data["backend"].startswith("chroma:")
    else:
        assert recall_data["backend"] == "rule_fallback_after_vector_unavailable"
        assert recall_data["fallback_reason"]
    assert "product_rerank" in response.agent_results
    assert response.agent_results["marketing_copy"].data["mode"] == "control_rule"


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
    assert len(data["marketing_copies"]) == 3


def test_graph_recommend_endpoint_returns_products():
    client = TestClient(app)
    response = client.post(
        "/api/v1/recommend/graph",
        json={
            "user_id": "graph-user",
            "num_items": 3,
            "preferred_categories": [list_products()[0].category],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "graph-user"
    assert data["strategy"].startswith("langgraph_agents")
    assert data["experiment_group"] in {"control", "treatment"}
    assert len(data["products"]) == 3
    assert len(data["marketing_copies"]) == 3
    assert "product_rerank" in data["agent_results"]


def test_graph_recommendation_records_ab_exposure():
    client = TestClient(app)
    response = client.post(
        "/api/v1/recommend/graph",
        json={"user_id": "graph-exposure-user", "num_items": 2},
    )

    assert response.status_code == 200
    data = response.json()
    stats = ab_test_engine.get_stats(data["experiment"]["experiment_id"])
    assert stats[data["experiment_group"]]["exposures"] == 1


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


def test_product_rerank_can_use_llm_hint(monkeypatch):
    products = list_products()[:4]
    expected_order = [products[2].product_id, products[0].product_id]

    def fake_chat(**kwargs):
        return f'["{products[2].product_id}", "missing-product", "{products[0].product_id}"]'

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    monkeypatch.setattr(
        llm_client,
        "status",
        lambda: {
            "available": True,
            "base_url": "mock://llm",
            "model": "mock-model",
            "last_error": None,
        },
    )

    result = ProductRecAgent().run(
        request=RecommendRequest(
            user_id="llm-rerank-user",
            num_items=2,
            context={"llm_hint": "优先推荐手机配件和办公商品"},
        ),
        products=products,
        limit=2,
        mode="rerank",
    )

    assert result.data["mode"] == "llm_rerank"
    assert result.data["backend"] == "llm+rule_rerank"
    assert result.data["product_ids"] == expected_order
    assert result.data["scores"][expected_order[0]]["reason"] == "LLM 重排序第1位"


def test_qwen3_models_disable_thinking_by_default(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("LLM_API_BASE", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3.5:4b")
    monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)

    client = LLMClient()

    assert client.enable_thinking is False
    assert client._extra_body() == {"enable_thinking": False, "think": False}


def test_llm_enable_thinking_env_can_override_qwen3(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("LLM_API_BASE", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("LLM_ENABLE_THINKING", "true")

    client = LLMClient()

    assert client.enable_thinking is True
    assert client._extra_body() == {"enable_thinking": True, "think": True}


def test_llm_json_parser_repairs_missing_closing_bracket(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("LLM_API_BASE", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:3b")

    client = LLMClient()
    result = client._parse_json_text('[[{"product_id": "p1", "copy": "ok"}]')

    assert result == [[{"product_id": "p1", "copy": "ok"}]]


def test_marketing_copy_normalizes_nested_llm_items():
    products = list_products()[:2]
    agent = MarketingCopyAgent()

    copies = agent._normalize_llm_copies(
        [
            [
                {"product_id": products[0].product_id, "copy": "copy one"},
                {"product_id": products[1].product_id, "copy": "copy two"},
            ]
        ],
        products,
    )

    assert [copy["product_id"] for copy in copies] == [
        products[0].product_id,
        products[1].product_id,
    ]
    assert [copy["text"] for copy in copies] == ["copy one", "copy two"]


def test_amazon_rating_to_user_event_mapping():
    assert rating_to_event_type(5.0) == "purchase"
    assert rating_to_event_type(4.0) == "like"
    assert rating_to_event_type(3.0) == "view"
    assert rating_to_event_type(1.0) == "dislike"
    assert normalize_timestamp(1_700_000_000_000) == 1_700_000_000


def test_offline_ndcg_scores_exact_target_order():
    relevance = {"p1": 3.0, "p2": 2.0}

    assert ndcg_at_k(["p1", "p2"], relevance, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["p3", "p2"], relevance, 2) < 1.0


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
            event_type="purchase",
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
    assert "llm_profile" in response.agent_results["user_profile"].data
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
    assert first.config["strategy"] in {"rule", "llm"}


def test_experiments_endpoint_returns_assignment():
    client = TestClient(app)
    response = client.get("/api/v1/experiments", params={"user_id": "u001"})

    assert response.status_code == 200
    data = response.json()
    assert data["default_experiment_id"] == "recommendation_strategy_v1"
    assert data["assignment"]["group"] in {"control", "treatment"}
    stats = data["experiments"]["recommendation_strategy_v1"]["stats"]
    assert set(stats) == {"control", "treatment"}


def test_recommendation_records_ab_exposure():
    response = recommend_products(
        RecommendRequest(
            user_id="ab-user-1",
            num_items=2,
        )
    )

    stats = ab_test_engine.get_stats(response.experiment.experiment_id)
    assert stats[response.experiment_group]["exposures"] == 1
    assert stats[response.experiment_group]["clicks"] == 0
    assert stats[response.experiment_group]["ctr"] == 0.0


def test_experiment_outcome_endpoint_updates_ctr_and_beta():
    client = TestClient(app)

    recommend_response = client.post(
        "/api/v1/recommend",
        json={"user_id": "ab-user-1", "num_items": 2},
    )
    assert recommend_response.status_code == 200
    recommend_data = recommend_response.json()
    experiment_id = recommend_data["experiment"]["experiment_id"]
    group = recommend_data["experiment_group"]
    product_id = recommend_data["products"][0]["product_id"]

    outcome_response = client.post(
        f"/api/v1/experiments/{experiment_id}/outcome",
        json={
            "experiment_id": experiment_id,
            "group": group,
            "user_id": "ab-user-1",
            "success": True,
            "product_id": product_id,
        },
    )

    assert outcome_response.status_code == 200
    data = outcome_response.json()
    group_stats = data["stats"][group]
    assert group_stats["exposures"] == 1
    assert group_stats["clicks"] == 1
    assert group_stats["ctr"] == 1.0
    assert group_stats["alpha"] == 2
    assert group_stats["beta"] == 1


def test_thompson_assignment_uses_outcome_counters():
    for _ in range(10):
        ab_test_engine.record_outcome(
            "recommendation_strategy_v1",
            "treatment",
            "winner",
            True,
        )
    for _ in range(3):
        ab_test_engine.record_outcome(
            "recommendation_strategy_v1",
            "control",
            "baseline",
            False,
        )

    seen = {
        ab_test_engine.assign_thompson(f"sample-user-{index}").group
        for index in range(30)
    }
    stats = ab_test_engine.get_stats("recommendation_strategy_v1")

    assert stats["treatment"]["alpha"] == 11
    assert stats["control"]["beta"] == 4
    assert seen <= {"control", "treatment"}


def test_control_group_uses_rule_pipeline(monkeypatch):
    def fail_if_llm_called(**kwargs):
        raise AssertionError("control group must not call LLM")

    monkeypatch.setattr(llm_client, "chat", fail_if_llm_called)
    monkeypatch.setattr(llm_client, "chat_json", fail_if_llm_called)

    response = recommend_products(
        RecommendRequest(
            user_id="ab-user-1",
            num_items=2,
            preferred_categories=[list_products()[0].category],
        )
    )

    assert response.experiment_group == "control"
    assert response.experiment.config["strategy"] == "rule"
    assert response.agent_results["user_profile"].data["llm_profile"]["recommendation_hint"] == ""
    assert response.agent_results["product_rerank"].data["mode"] == "rerank"
    assert response.agent_results["marketing_copy"].data["mode"] == "control_rule"


def test_treatment_group_can_use_llm_pipeline(monkeypatch):
    products = list_products()
    first_id = products[1].product_id
    second_id = products[0].product_id

    def fake_chat_json(**kwargs):
        user_message = kwargs.get("user_message", "")
        product_ids = re.findall(r"[A-Z0-9]{10}", user_message)
        if product_ids:
            return [
                {"product_id": product_id, "copy": f"LLM copy {index + 1}"}
                for index, product_id in enumerate(product_ids[:2])
            ]
        return {
            "segments": ["active"],
            "intent_summary": "Looking for practical accessories",
            "recommendation_hint": "Prefer practical accessories with good ratings.",
            "price_sensitivity": "medium",
            "rfm_interpretation": "Active user",
        }

    def fake_chat(**kwargs):
        return f'["{first_id}", "{second_id}"]'

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)
    monkeypatch.setattr(
        llm_client,
        "status",
        lambda: {
            "available": True,
            "base_url": "mock://llm",
            "model": "mock-model",
            "last_error": None,
        },
    )

    response = recommend_products(
        RecommendRequest(
            user_id="ab-user-2",
            num_items=2,
        )
    )

    assert response.experiment_group == "treatment"
    assert response.experiment.config["strategy"] == "llm"
    assert response.agent_results["user_profile"].data["effective_request"]["context"]["llm_hint"]
    assert response.agent_results["product_rerank"].data["mode"] == "llm_rerank"
    assert response.agent_results["marketing_copy"].data["mode"] == "llm"


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
            event_type="purchase",
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
        assert data["features"]["purchase_count_7d"] == 1


def test_chat_endpoint_returns_conversational_recommendation():
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-user",
            "message": "我想买个200元以内的手机保护壳",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"]
    assert data["intent"] == "recommend_products"
    assert data["state"]["budget_max"] == 200
    assert "手机" in data["state"]["preferred_categories"]
    assert data["products"]
    assert all(product["stock"] > 0 for product in data["products"])
    assert "intent" in data["agent_results"]


def test_chat_stream_endpoint_emits_sse_events():
    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "user_id": "chat-stream-user",
            "message": "推荐几款手机保护壳",
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: state" in body
    assert "event: token" in body
    assert "event: products" in body
    assert "event: done" in body


def test_chat_goal_switching_replaces_previous_preferences():
    client = TestClient(app)
    session_id = "chat-goal-switch-session"

    phone = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-goal-switch-user",
            "session_id": session_id,
            "message": "我想要手机",
        },
    )
    assert phone.status_code == 200
    phone_data = phone.json()
    assert phone_data["state"]["preferred_categories"] == ["手机"]
    assert "手机配件" in phone_data["state"]["preferred_tags"]
    assert all(product["category"] == "手机" for product in phone_data["products"])

    computer = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-goal-switch-user",
            "session_id": session_id,
            "message": "想要电脑",
        },
    )
    assert computer.status_code == 200
    computer_data = computer.json()
    assert computer_data["state"]["preferred_categories"] == ["电子数码"]
    assert computer_data["state"]["preferred_tags"] == ["电脑配件"]
    assert computer_data["state"]["liked_brands"] == []
    assert all(product["category"] == "电子数码" for product in computer_data["products"])
    assert any("电脑配件" in product["tags"] for product in computer_data["products"])

    earphones = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-goal-switch-user",
            "session_id": session_id,
            "message": "我想买个200元以内的通勤耳机",
        },
    )
    assert earphones.status_code == 200
    earphone_data = earphones.json()
    assert earphone_data["state"]["preferred_categories"] == ["电子数码"]
    assert earphone_data["state"]["preferred_tags"] == ["耳机", "通勤"]
    assert earphone_data["state"]["budget_max"] == 200
    assert all(product["category"] == "电子数码" for product in earphone_data["products"])
    assert any("耳机" in product["tags"] for product in earphone_data["products"])


def test_chat_memory_resolves_product_reference_and_feedback():
    client = TestClient(app)
    first = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-feedback-user",
            "session_id": "chat-feedback-session",
            "message": "推荐几款手机保护壳",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-feedback-user",
            "session_id": "chat-feedback-session",
            "message": "第二个太贵了，换便宜点",
        },
    )

    assert second.status_code == 200
    data = second.json()
    assert data["intent"] == "record_feedback"
    assert "too_expensive" in data["state"]["rejected_reasons"]
    assert data["state"]["active_product_refs"]


def test_chat_meta_questions_do_not_trigger_recommendations():
    client = TestClient(app)
    for message in ["你好", "你能说画面", "你是什么agent", "你是什么模型", "今天星期几"]:
        response = client.post(
            "/api/v1/chat",
            json={
                "user_id": "chat-meta-user",
                "session_id": "chat-meta-session",
                "message": message,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "smalltalk"
        assert data["products"] == []

    assert "星期" in data["reply"]


def test_chat_eval_script_reports_core_metrics():
    report = evaluate_chat_agent(Path("data/chat_eval_cases.jsonl"))

    summary = report["summary"]
    assert summary["case_count"] >= 8
    assert "intent_macro_f1" in summary
    assert "slot_f1" in summary
    assert "task_success_rate" in summary
