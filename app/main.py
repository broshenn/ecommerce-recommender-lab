from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.behavior import build_user_profile, list_user_events, record_event
from app.catalog import list_products
from app.database import init_db
from app.models import (
    ExperimentOutcome,
    ChatRequest,
    ChatResponse,
    Product,
    RecommendRequest,
    RecommendResponse,
    UserEvent,
    UserEventCreate,
    UserProfile,
)
from app.recommender import recommend_products
from app.orchestrator.chat import chat_orchestrator
from app.services import ab_test_engine, feature_store, metrics_collector
from app.services.vector_store import get_product_vector_store

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="E-Commerce Recommendation Rebuild",
    version="0.14.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(
        STATIC_DIR / "index.html",
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
def health():
    init_db()
    return {
        "status": "healthy",
        "step": "17",
        "storage": "sqlite",
        "orchestrator": "supervisor",
        "experiments": "ab_test",
        "metrics": "in_memory",
        "vector_recall": "chroma",
        "feature_store": "redis",
        "llm_profile": "openai_compatible",
        "llm_marketing_copy": "openai_compatible",
        "llm_rerank": "openai_compatible",
        "ab_experiment_gating": "control_rule_vs_treatment_llm",
        "ab_outcome_stats": "exposure_click_ctr_thompson",
        "langgraph_orchestration": "available",
        "chat_agent": "conversational_commerce",
        "chat_stream": "sse",
    }


@app.get("/api/v1/products", response_model=list[Product])
def products():
    return list_products()


@app.post("/api/v1/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    return recommend_products(request)


@app.post("/api/v1/recommend/graph", response_model=RecommendResponse)
def recommend_via_graph(request: RecommendRequest):
    from app.orchestrator.graph import recommend_with_graph

    return recommend_with_graph(request)


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return chat_orchestrator.chat(request)


@app.post("/api/v1/chat/stream")
def chat_stream(request: ChatRequest):
    return StreamingResponse(
        chat_orchestrator.stream_chat(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/query-understanding/eval-summary")
def query_understanding_eval_summary():
    hard_report = _load_json_report(
        PROJECT_ROOT / "reports" / "query_understanding_hard_eval_latest.json"
    )
    model_report = _load_json_report(
        PROJECT_ROOT / "reports" / "query_understanding_model_compare_latest.json"
    )
    return {
        "hard_eval": _compact_query_eval_report(hard_report),
        "synthetic_eval": _compact_query_eval_report(model_report),
    }


def _load_json_report(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    data["status"] = "available"
    return data


def _compact_query_eval_report(report: dict) -> dict:
    if report.get("status") == "missing":
        return report
    models = []
    for item in report.get("models", []):
        models.append(
            {
                "name": item.get("name"),
                "intent_macro_f1": item.get("intent_macro_f1"),
                "slot_f1": item.get("slot_f1"),
                "hard_intent_macro_f1": item.get("hard_intent_macro_f1"),
                "hard_slot_f1": item.get("hard_slot_f1"),
                "smalltalk_guard_rate": item.get("smalltalk_guard_rate"),
                "avg_latency_ms": item.get("avg_latency_ms"),
            }
        )
    return {
        "status": "available",
        "case_count": report.get("case_count") or report.get("eval_count"),
        "summary": report.get("summary", {}),
        "models": models,
    }


@app.get("/api/v1/experiments")
def experiments(user_id: str | None = None):
    payload = ab_test_engine.list_experiments()
    if user_id:
        payload["assignment"] = ab_test_engine.assign(user_id).model_dump()
    return payload


@app.post("/api/v1/experiments/{experiment_id}/outcome")
def record_experiment_outcome(experiment_id: str, outcome: ExperimentOutcome):
    ab_test_engine.record_outcome(
        experiment_id=experiment_id,
        group=outcome.group,
        user_id=outcome.user_id,
        success=outcome.success,
        product_id=outcome.product_id,
    )
    return {
        "status": "recorded",
        "experiment_id": experiment_id,
        "group": outcome.group,
        "stats": ab_test_engine.get_stats(experiment_id),
    }


@app.get("/api/v1/metrics")
def metrics():
    return metrics_collector.snapshot()


@app.get("/api/v1/vector-store")
def vector_store():
    return get_product_vector_store().status()


@app.get("/api/v1/feature-store/{user_id}")
def feature_store_status(user_id: str):
    cached_profile = feature_store.get_cached_profile(user_id)
    return {
        "status": feature_store.status(),
        "features": feature_store.get_user_features(user_id),
        "cached_profile": cached_profile.model_dump(mode="json") if cached_profile else None,
    }


@app.post("/api/v1/events", response_model=UserEvent)
def create_event(event: UserEventCreate):
    return record_event(event)


@app.get("/api/v1/users/{user_id}/events", response_model=list[UserEvent])
def user_events(user_id: str):
    return list_user_events(user_id)


@app.get("/api/v1/users/{user_id}/profile", response_model=UserProfile)
def user_profile(user_id: str):
    return build_user_profile(user_id)
