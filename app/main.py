from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.behavior import build_user_profile, list_user_events, record_event
from app.catalog import list_products
from app.models import (
    Product,
    RecommendRequest,
    RecommendResponse,
    UserEvent,
    UserEventCreate,
    UserProfile,
)
from app.recommender import recommend_products

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="E-Commerce Recommendation Rebuild",
    version="0.5.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
def health():
    return {"status": "healthy", "step": 5}


@app.get("/api/v1/products", response_model=list[Product])
def products():
    return list_products()


@app.post("/api/v1/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    return recommend_products(request)


@app.post("/api/v1/events", response_model=UserEvent)
def create_event(event: UserEventCreate):
    return record_event(event)


@app.get("/api/v1/users/{user_id}/events", response_model=list[UserEvent])
def user_events(user_id: str):
    return list_user_events(user_id)


@app.get("/api/v1/users/{user_id}/profile", response_model=UserProfile)
def user_profile(user_id: str):
    return build_user_profile(user_id)
