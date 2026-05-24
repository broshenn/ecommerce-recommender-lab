from __future__ import annotations

from typing import Any

from app.behavior import build_user_profile, merge_behavior_profile
from app.models import AgentResult, RecommendRequest

from app.agents.base_agent import BaseAgent


class UserProfileAgent(BaseAgent):
    """Build the current user's profile from SQLite behavior plus request hints."""

    def __init__(self):
        super().__init__(name="user_profile", timeout=3.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        request: RecommendRequest = kwargs["request"]
        behavior_profile = build_user_profile(request.user_id)
        effective_request = merge_behavior_profile(request)

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "profile": behavior_profile.model_dump(mode="json"),
                "effective_request": effective_request.model_dump(mode="json"),
            },
            confidence=0.9,
        )

    def _fallback(self, latency_ms: float, exc: Exception, **kwargs: Any) -> AgentResult:
        request: RecommendRequest = kwargs.get("request", RecommendRequest(user_id="unknown"))
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            data={
                "profile": {"user_id": request.user_id, "event_count": 0},
                "effective_request": request.model_dump(mode="json"),
            },
            confidence=0.3,
        )
