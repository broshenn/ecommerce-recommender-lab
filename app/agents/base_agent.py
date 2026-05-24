from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from app.models import AgentResult


class BaseAgent(ABC):
    """Common agent wrapper with timing, error capture, and fallback."""

    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
        self.call_count = 0
        self.error_count = 0

    def run(self, **kwargs: Any) -> AgentResult:
        start = time.perf_counter()
        self.call_count += 1

        try:
            result = self._execute(**kwargs)
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as exc:
            self.error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            return self._fallback(latency_ms, exc, **kwargs)

    @abstractmethod
    def _execute(self, **kwargs: Any) -> AgentResult:
        """Run the agent's core work."""

    def _fallback(self, latency_ms: float, exc: Exception, **kwargs: Any) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            confidence=0.0,
        )

    @property
    def error_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.error_count / self.call_count
