from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any

from app.models import AgentResult


class MetricsCollector:
    """学习版内存指标收集器。"""

    def __init__(self):
        self._lock = Lock()
        self._agent_stats: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "call_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_latency_ms": 0.0,
                "last_latency_ms": 0.0,
            }
        )
        self._business_events: dict[str, int] = defaultdict(int)

    def record_agent_result(self, metric_key: str, result: AgentResult) -> None:
        with self._lock:
            stats = self._agent_stats[metric_key]
            stats["call_count"] += 1
            stats["success_count"] += 1 if result.success else 0
            stats["error_count"] += 0 if result.success else 1
            stats["total_latency_ms"] += result.latency_ms
            stats["last_latency_ms"] = result.latency_ms

    def record_business_event(self, event_name: str) -> None:
        with self._lock:
            self._business_events[event_name] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            agent_metrics = []
            for metric_key, stats in sorted(self._agent_stats.items()):
                call_count = int(stats["call_count"])
                success_count = int(stats["success_count"])
                total_latency = float(stats["total_latency_ms"])
                agent_metrics.append(
                    {
                        "agent": metric_key,
                        "call_count": call_count,
                        "success_count": success_count,
                        "error_count": int(stats["error_count"]),
                        "success_rate": round(success_count / call_count, 4) if call_count else 0.0,
                        "avg_latency_ms": round(total_latency / call_count, 2) if call_count else 0.0,
                        "last_latency_ms": round(float(stats["last_latency_ms"]), 2),
                    }
                )

            return {
                "agent_metrics": agent_metrics,
                "business_events": dict(sorted(self._business_events.items())),
                "total_agent_calls": sum(metric["call_count"] for metric in agent_metrics),
            }

    def reset(self) -> None:
        with self._lock:
            self._agent_stats.clear()
            self._business_events.clear()
