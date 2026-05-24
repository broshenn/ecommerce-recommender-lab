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
                "name": "推荐策略实验 v1",
                "description": "先完成稳定分桶，后续接入向量召回、LLM 重排等策略对比。",
                "variants": [
                    {
                        "group": "control",
                        "traffic_percent": 50,
                        "description": "当前规则排序策略",
                        "config": {"strategy": "rule_ranking"},
                    },
                    {
                        "group": "treatment",
                        "traffic_percent": 50,
                        "description": "预留增强推荐策略入口",
                        "config": {"strategy": "rule_ranking_plus"},
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
            reason=f"user_id 稳定哈希分桶，bucket={bucket / 100:.2f}",
            config=selected.get("config", {}),
        )

    def list_experiments(self) -> dict[str, Any]:
        return {"default_experiment_id": self.DEFAULT_EXPERIMENT_ID, "experiments": self.experiments}

    def _bucket(self, user_id: str, experiment_id: str) -> int:
        key = f"{experiment_id}:{user_id}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return int(digest[:8], 16) % 10000
