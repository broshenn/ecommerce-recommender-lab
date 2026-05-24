from __future__ import annotations

from app.models import RecommendRequest, RecommendResponse
from app.orchestrator import SupervisorOrchestrator

supervisor = SupervisorOrchestrator()


def recommend_products(request: RecommendRequest) -> RecommendResponse:
    """Run the Supervisor + Agent recommendation pipeline."""
    return supervisor.recommend(request)
