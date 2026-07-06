import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.behavior import list_user_events, record_event, reset_behavior_events
from app.catalog import list_products
from app.agents.dialogue_agent import DialogueAgent
from app.agents.intent_agent import IntentAgent
from app.agents.marketing_copy_agent import MarketingCopyAgent
from app.agents.product_rec_agent import ProductRecAgent
from app.main import app
from app.models import ConversationState, IntentResult, RecommendRequest, UserEventCreate
from app.orchestrator.chat import chat_orchestrator
from app.personalization import score_product
from app.recommender import recommend_products
from app.services import ab_test_engine, feature_store, llm_client, metrics_collector
from app.services.intent_classifier import intent_model_classifier
from app.services.llm_client import LLMClient
from app.services.vector_store import get_product_vector_store
from scripts.evaluate_chat_agent import evaluate_chat_agent
from scripts.evaluate_query_understanding_hard import evaluate_hard_set
from scripts.evaluate_query_understanding_models import evaluate_models
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
    intent_data = data["agent_results"]["intent"]["data"]
    assert intent_data["source"] == "rule"
    assert "rule_debug" in intent_data
    assert intent_data["rule_debug"]["matched_rules"] == ["recommend_products"]
    tool_names = [item.get("tool_name") for item in data["trace"] if item.get("step") == "tool"]
    assert "PreferenceUpdateTool" in tool_names
    assert "RecommendGraphTool" in tool_names


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
    assert "event: trace" in body
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


def test_intent_agent_extracts_business_slots_and_budget_ranges():
    state = ConversationState(session_id="intent-slots", user_id="intent-user")
    agent = IntentAgent()
    result = agent._rule_intent("想要100到300元的防水耳机，最好是Samsung", state)

    assert result.intent == "recommend_products"
    assert result.needs_recommendation is True
    assert result.slots["budget_min"] == 100
    assert result.slots["budget_max"] == 300
    assert result.slots["preferred_categories"] == ["电子数码"]
    assert "耳机" in result.slots["preferred_tags"]
    assert "防水" in result.slots["preferred_tags"]
    assert result.slots["liked_brands"] == ["SAMSUNG"]
    assert agent.rules["product_synonyms"]["耳机"]["tags"] == ["耳机"]
    assert agent._last_rule_debug["matched_keywords"]["recommend_products"] == ["想要"]
    assert "耳机" in agent._last_rule_debug["slot_sources"]["synonyms"]


def test_optional_bert_intent_classifier_keeps_rule_slots(monkeypatch):
    def fake_classify(text):
        return {
            "intent": "recommend_products",
            "confidence": 0.93,
            "model_dir": "mock://intent-bert",
        }

    monkeypatch.setattr(intent_model_classifier, "classify", fake_classify)
    agent = IntentAgent()
    state = ConversationState(session_id="bert-intent", user_id="intent-user")

    result = agent._apply_model_intent(
        "预算200以内",
        IntentResult(
            intent="refine_preferences",
            slots={"budget_max": 200},
            needs_recommendation=True,
            confidence=0.72,
            source="rule",
        ),
    )

    assert result.intent == "recommend_products"
    assert result.source == "bert+rule_slots"
    assert result.slots["budget_max"] == 200
    assert agent._last_rule_debug["model_intent"]["model_dir"] == "mock://intent-bert"


def test_intent_agent_supports_explicit_rule_and_bert_modes(monkeypatch):
    calls = []

    def fake_classify(text, force=False):
        calls.append(force)
        return {
            "intent": "compare_products",
            "confidence": 0.91,
            "model_dir": "mock://intent-bert",
        }

    monkeypatch.setattr(intent_model_classifier, "classify", fake_classify)
    agent = IntentAgent()
    state = ConversationState(session_id="intent-mode-session", user_id="intent-user")

    message = "鑷冲皯500鍏冪殑鐢佃剳閰嶄欢"
    rule = agent.run(
        message=message,
        state=state,
        recent_messages=[],
        intent_mode="rule",
    )
    assert rule.data["source"] == "rule"
    assert rule.data["intent_mode"] == "rule"
    assert calls == []

    bert = agent.run(
        message=message,
        state=state,
        recent_messages=[],
        intent_mode="bert",
    )
    assert bert.data["intent"] == "refine_preferences"
    assert bert.data["source"] == "rule_guarded_bert+rule_slots"
    assert bert.data["intent_mode"] == "bert"
    assert bert.data["slots"]["budget_max"] == 500
    assert bert.data["rule_debug"]["business_guard"]["applied"] is True
    assert (
        bert.data["rule_debug"]["business_guard"]["reason"]
        == "business_slot_recommendation_guard"
    )
    assert calls == [True]


def test_intent_agent_handles_min_budget_product_info_and_compare():
    agent = IntentAgent()
    state = ConversationState(session_id="intent-branches", user_id="intent-user")

    min_budget = agent._rule_intent("至少500元的电脑配件", state)
    assert min_budget.intent == "refine_preferences"
    assert min_budget.slots["budget_min"] == 500
    assert min_budget.slots["preferred_categories"] == ["电子数码"]
    assert "电脑配件" in min_budget.slots["preferred_tags"]

    ask_product = agent._rule_intent("第一款库存和价格多少", state)
    assert ask_product.intent == "ask_product"

    compare = agent._rule_intent("第一个和第二个有什么区别，哪个更好", state)
    assert compare.intent == "compare_products"


def test_business_guard_prevents_llm_from_overriding_budget_need(monkeypatch):
    monkeypatch.setattr(
        llm_client,
        "chat_json",
        lambda **kwargs: {
            "intent": "ask_product",
            "slots": {},
            "product_refs": [],
            "needs_recommendation": False,
            "confidence": 0.88,
        },
    )
    agent = IntentAgent()
    state = ConversationState(session_id="llm-guard-session", user_id="intent-user")

    result = agent.run(
        message="鑷冲皯500鍏冪殑鐢佃剳閰嶄欢",
        state=state,
        recent_messages=[],
        intent_mode="llm",
    )

    assert result.data["intent"] == "refine_preferences"
    assert result.data["source"] == "rule_guarded_llm"
    assert result.data["slots"]["budget_max"] == 500
    assert result.data["rule_debug"]["business_guard"]["applied"] is True
    assert (
        result.data["rule_debug"]["business_guard"]["reason"]
        == "business_slot_recommendation_guard"
    )


def test_intent_agent_handles_feedback_events():
    agent = IntentAgent()
    state = ConversationState(session_id="intent-feedback", user_id="intent-user")

    dislike = agent._rule_intent("第二个太贵了，不喜欢，换便宜点", state)
    assert dislike.intent == "record_feedback"
    assert dislike.needs_recommendation is True
    assert dislike.slots["event_type"] == "dislike"
    assert "too_expensive" in dislike.slots["rejected_reasons"]
    assert "disliked" in dislike.slots["rejected_reasons"]

    purchase = agent._rule_intent("我要购买第一个", state)
    assert purchase.intent == "record_feedback"
    assert purchase.slots["event_type"] == "purchase"


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
    first_data = first.json()
    disliked_product_id = first_data["products"][1]["product_id"]

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
    assert disliked_product_id in data["state"]["disliked_products"]
    events = list_user_events("chat-feedback-user")
    assert any(
        event.product_id == disliked_product_id and event.event_type == "dislike"
        for event in events
    )
    tool_names = [item.get("tool_name") for item in data["trace"] if item.get("step") == "tool"]
    assert "FeedbackTool" in tool_names
    assert "RecommendGraphTool" in tool_names


def test_chat_long_term_memory_seeds_new_session_without_overriding_explicit_goal():
    client = TestClient(app)
    first = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-memory-user",
            "session_id": "chat-memory-first-session",
            "message": "我喜欢Bastmei，想看手机保护壳",
        },
    )
    assert first.status_code == 200
    assert first.json()["state"]["liked_brands"] == ["Bastmei"]

    recalled = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-memory-user",
            "session_id": "chat-memory-new-session",
            "message": "再推荐几款",
        },
    )
    assert recalled.status_code == 200
    recalled_data = recalled.json()
    memory_trace = [item for item in recalled_data["trace"] if item.get("step") == "memory"]
    assert memory_trace
    recommend_tool = next(
        item for item in recalled_data["trace"]
        if item.get("tool_name") == "RecommendGraphTool"
    )
    assert recommend_tool["input_summary"]["categories"] == ["手机"]
    assert recommend_tool["input_summary"]["brands"] == ["Bastmei"]

    switched = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-memory-user",
            "session_id": "chat-memory-switch-session",
            "message": "想要电脑",
        },
    )
    assert switched.status_code == 200
    switched_tool = next(
        item for item in switched.json()["trace"]
        if item.get("tool_name") == "RecommendGraphTool"
    )
    assert switched_tool["input_summary"]["categories"] == ["电子数码"]
    assert switched_tool["input_summary"]["tags"] == ["电脑配件"]


def test_chat_continuation_uses_short_term_query_memory():
    client = TestClient(app)
    session_id = "chat-short-memory-session"

    first = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-short-memory-user",
            "session_id": session_id,
            "message": "我想买个200元以内的通勤耳机",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-short-memory-user",
            "session_id": session_id,
            "message": "再推荐几个",
        },
    )

    assert second.status_code == 200
    data = second.json()
    assert data["intent"] == "recommend_products"
    assert data["state"]["budget_max"] == 200
    assert "耳机" in data["state"]["preferred_tags"]
    intent_debug = data["agent_results"]["intent"]["data"]["rule_debug"]
    memory = intent_debug["memory_enrichment"]
    assert memory["applied"] is True
    assert memory["sources"][0]["source"] == "short_term_session"
    recommend_tool = next(
        item for item in data["trace"]
        if item.get("tool_name") == "RecommendGraphTool"
    )
    assert recommend_tool["input_summary"]["tags"] == ["耳机", "通勤"]


def test_chat_query_understanding_uses_behavior_profile_for_empty_need():
    product = next(item for item in list_products() if item.stock > 0)
    record_event(
        UserEventCreate(
            user_id="chat-behavior-memory-user",
            product_id=product.product_id,
            event_type="like",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-behavior-memory-user",
            "session_id": "chat-behavior-memory-session",
            "message": "给我推荐几个",
        },
    )

    assert response.status_code == 200
    data = response.json()
    intent_debug = data["agent_results"]["intent"]["data"]["rule_debug"]
    memory = intent_debug["memory_enrichment"]
    sources = {item["source"]: item["slots"] for item in memory["sources"]}
    assert memory["applied"] is True
    assert "behavior_profile" in sources
    assert product.category in data["state"]["preferred_categories"]
    assert product.brand in data["state"]["liked_brands"]
    recommend_tool = next(
        item for item in data["trace"]
        if item.get("tool_name") == "RecommendGraphTool"
    )
    assert product.category in recommend_tool["input_summary"]["categories"]
    assert product.brand in recommend_tool["input_summary"]["brands"]


def test_chat_business_tools_handle_compare_explain_and_product_info():
    client = TestClient(app)
    session_id = "chat-business-tools-session"
    first = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-business-tools-user",
            "session_id": session_id,
            "message": "推荐几款手机保护壳",
        },
    )
    assert first.status_code == 200

    compare = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-business-tools-user",
            "session_id": session_id,
            "message": "第一个和第二个有什么区别",
        },
    )
    compare_data = compare.json()
    compare_tools = [
        item.get("tool_name") for item in compare_data["trace"] if item.get("step") == "tool"
    ]
    assert compare_data["intent"] == "compare_products"
    assert "CompareProductTool" in compare_tools
    assert compare_data["products"] == []

    explain = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-business-tools-user",
            "session_id": session_id,
            "message": "为什么推荐第一款",
        },
    )
    explain_data = explain.json()
    explain_tools = [
        item.get("tool_name") for item in explain_data["trace"] if item.get("step") == "tool"
    ]
    assert explain_data["intent"] == "explain_recommendation"
    assert "ExplainRecommendationTool" in explain_tools

    ask = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-business-tools-user",
            "session_id": session_id,
            "message": "第一款价格和库存多少",
        },
    )
    ask_data = ask.json()
    ask_tools = [
        item.get("tool_name") for item in ask_data["trace"] if item.get("step") == "tool"
    ]
    assert ask_data["intent"] == "ask_product"
    assert "ProductInfoTool" in ask_tools


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
        tool_names = [
            item.get("tool_name") for item in data["trace"] if item.get("step") == "tool"
        ]
        assert tool_names == ["SmalltalkTool"]

    assert "星期" in data["reply"]


def test_smalltalk_open_chat_can_use_llm_fallback(monkeypatch):
    monkeypatch.setenv("CHAT_LLM_ENABLED", "true")
    monkeypatch.setattr(
        llm_client,
        "chat",
        lambda **kwargs: "可以，轻松聊两句也没问题。你想买东西时，直接告诉我预算和用途就行。",
    )
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-smalltalk-llm-user",
            "session_id": "chat-smalltalk-llm-session",
            "message": "不是问商品，随便聊聊",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "smalltalk"
    assert data["products"] == []
    assert data["agent_results"]["dialogue"]["data"]["mode"] == "llm_smalltalk"
    policy = data["agent_results"]["dialogue"]["data"]["smalltalk_policy"]
    assert policy["category"] == "open_smalltalk"
    assert policy["allow_llm"] is True


def test_smalltalk_identity_uses_rule_even_when_llm_enabled(monkeypatch):
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return "I should not be used here"

    monkeypatch.setenv("CHAT_LLM_ENABLED", "true")
    monkeypatch.setattr(llm_client, "chat", fake_chat)
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={
            "user_id": "chat-smalltalk-rule-user",
            "session_id": "chat-smalltalk-rule-session",
            "message": "你是什么agent",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "smalltalk"
    assert data["products"] == []
    assert data["agent_results"]["dialogue"]["data"]["mode"] == "rule"
    policy = data["agent_results"]["dialogue"]["data"]["smalltalk_policy"]
    assert policy["allow_llm"] is False
    assert calls == []


def test_chat_eval_script_reports_core_metrics():
    report = evaluate_chat_agent(Path("data/chat_eval_cases.jsonl"))

    summary = report["summary"]
    assert summary["case_count"] >= 12
    assert "intent_macro_f1" in summary
    assert "slot_f1" in summary
    assert "task_success_rate" in summary
    assert "tool_success_rate" in summary
    assert "no_recommendation_guard_rate" in summary
    assert summary["tool_success_rate"] >= 0.9
    assert report["scenario_summary"]
    assert "smalltalk_fallback" in report["scenario_summary"]
    assert "long_term_memory" in report["scenario_summary"]
    assert report["failures"] == []


def test_query_understanding_model_compare_reports_tradeoffs():
    report = evaluate_models(
        train_path=Path("data/query_understanding_train.jsonl"),
        eval_path=Path("data/query_understanding_eval.jsonl"),
    )

    summary = report["summary"]
    assert report["train_count"] >= 500
    assert report["eval_count"] >= 100
    assert "rule_baseline" in summary["completed_models"]
    assert "char_ngram_nb_classifier" in summary["completed_models"]
    assert "llm_classifier" in summary["skipped_models"]
    assert "distilbert_classifier" in summary["skipped_models"]

    models = {model["name"]: model for model in report["models"]}
    assert models["char_ngram_nb_classifier"]["intent_macro_f1"] >= 0.9
    assert models["rule_baseline"]["slot_f1"] > 0.5
    assert report["recommendation"]["winner"]


def test_query_understanding_hard_eval_reports_generalization_gaps():
    report = evaluate_hard_set(
        train_path=Path("data/query_understanding_train.jsonl"),
        eval_path=Path("data/query_understanding_hard_eval.jsonl"),
    )

    assert report["case_count"] >= 30
    assert "smalltalk" in report["scenario_distribution"]
    assert "negative_feedback" in report["scenario_distribution"]
    assert "unsupported_catalog" in report["scenario_distribution"]

    models = {model["name"]: model for model in report["models"]}
    assert models["rule_baseline"]["hard_slot_f1"] > 0.5
    assert models["bert_rule_slots"]["hard_intent_macro_f1"] >= 0.85
    assert models["bert_rule_slots"]["smalltalk_guard_rate"] >= 0.9
    assert report["summary"]["best_by_hard_intent_macro_f1"] == "bert_rule_slots"


def test_query_understanding_eval_summary_endpoint_returns_model_metrics():
    client = TestClient(app)
    response = client.get("/api/v1/query-understanding/eval-summary")

    assert response.status_code == 200
    data = response.json()
    assert data["hard_eval"]["status"] == "available"
    models = {model["name"]: model for model in data["hard_eval"]["models"]}
    assert "rule_baseline" in models
    assert "bert_rule_slots" in models
    assert models["bert_rule_slots"]["hard_intent_macro_f1"] >= 0.85


def test_query_understanding_compare_endpoint_runs_all_modes(monkeypatch):
    def fake_classify(text, force=False):
        return {
            "intent": "recommend_products",
            "confidence": 0.94,
            "model_dir": "mock://intent-bert",
        }

    monkeypatch.setattr(intent_model_classifier, "classify", fake_classify)
    monkeypatch.setattr(llm_client, "chat_json", lambda **kwargs: None)

    client = TestClient(app)
    response = client.post(
        "/api/v1/query-understanding/compare",
        json={
            "user_id": "compare-user",
            "message": "鑷冲皯500鍏冪殑鐢佃剳閰嶄欢",
        },
    )

    assert response.status_code == 200
    data = response.json()
    rows = {row["mode"]: row for row in data["modes"]}
    assert set(rows) == {"rule", "bert", "llm"}
    assert rows["bert"]["source"] == "bert+rule_slots"
    assert rows["bert"]["slots"]["budget_max"] == 500
    assert data["summary"]["intents"]["bert"] == "recommend_products"
    assert "bert" in data["summary"]["recommend_modes"]


def test_chat_recommend_request_does_not_force_control_group():
    state = ConversationState(
        session_id="chat-ab-session",
        user_id="chat-ab-user",
        shopping_goal="电子数码 电脑配件",
        preferred_categories=["电子数码"],
        preferred_tags=["电脑配件"],
        budget_max=200,
    )

    request = chat_orchestrator._recommend_request_from_state(state, {})

    assert request.context["conversation_session_id"] == "chat-ab-session"
    assert "force_experiment_group" not in request.context

    forced_request = chat_orchestrator._recommend_request_from_state(
        state,
        {},
        force_experiment_group="control",
    )
    assert forced_request.context["force_experiment_group"] == "control"


def test_dialogue_explains_computer_catalog_maps_to_accessories():
    product = next(
        item for item in list_products()
        if item.category == "电子数码" and "电脑配件" in item.tags
    )
    state = ConversationState(
        session_id="dialogue-computer-session",
        user_id="dialogue-computer-user",
        shopping_goal="电子数码 电脑配件",
        preferred_categories=["电子数码"],
        preferred_tags=["电脑配件"],
        budget_max=200,
    )

    result = DialogueAgent().run(
        intent="recommend_products",
        state=state,
        products=[product],
        marketing_copies=[],
        extra={},
        message="想要个200块以内的电脑",
    )

    assert "商品库主要是电脑配件" in result.data["reply"]
