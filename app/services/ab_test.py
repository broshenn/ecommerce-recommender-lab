from __future__ import annotations

import hashlib
from typing import Any

from app.models import ExperimentAssignment


class ABTestEngine:
    """Stable user bucketing for recommendation experiments."""

    DEFAULT_EXPERIMENT_ID = "recommendation_strategy_v1"

    def __init__(self):
        self.experiments: dict[str, dict[str, Any]] = {
            self.DEFAULT_EXPERIMENT_ID: {
                "name": "Recommendation strategy experiment v1",
                "description": "Compare a pure rule pipeline with an LLM-enhanced pipeline.",
                "variants": [
                    {
                        "group": "control",
                        "traffic_percent": 50,
                        "description": "Rule pipeline: rule profile, rule rerank, rule copy",
                        "config": {
                            "strategy": "rule",
                            "profile": "rule",
                            "rerank": "rule",
                            "copy": "rule",
                        },
                    },
                    {
                        "group": "treatment",
                        "traffic_percent": 50,
                        "description": "LLM pipeline: LLM profile, LLM rerank, LLM copy",
                        "config": {
                            "strategy": "llm",
                            "profile": "llm",
                            "rerank": "llm",
                            "copy": "llm",
                        },
                    },
                ],
            }
        }

    def assign(
        self,
        user_id: str,
        experiment_id: str | None = None,
    ) -> ExperimentAssignment:
        experiment_id = experiment_id or self.DEFAULT_EXPERIMENT_ID
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        bucket = self._bucket(user_id, experiment_id)
        cumulative = 0
        selected = experiment["variants"][-1]

        for variant in experiment["variants"]:
            cumulative += int(variant["traffic_percent"] * 100)
            if bucket < cumulative:
                selected = variant
                break

        return ExperimentAssignment(
            experiment_id=experiment_id,
            group=selected["group"],
            reason=f"stable user_id hash bucket={bucket / 100:.2f}",
            config=selected.get("config", {}),
        )

    def list_experiments(self) -> dict[str, Any]:
        return {
            "default_experiment_id": self.DEFAULT_EXPERIMENT_ID,
            "experiments": self.experiments,
        }

    def _bucket(self, user_id: str, experiment_id: str) -> int:
        key = f"{experiment_id}:{user_id}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return int(digest[:8], 16) % 10000
