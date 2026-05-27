from __future__ import annotations

from app.models import RecommendRequest, RecommendResponse
from app.orchestrator import SupervisorOrchestrator

supervisor = SupervisorOrchestrator()


def recommend_products(request: RecommendRequest) -> RecommendResponse:
    """运行传统 Supervisor + Agent 推荐链路。"""
    return supervisor.recommend(request)
