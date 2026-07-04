from __future__ import annotations

import hashlib
import random
import threading
import time
from typing import Any

from app.models import ExperimentAssignment


class ABTestEngine:
    """A/B 实验引擎：稳定分桶 + 内存实验统计。"""

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
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._init_beta_priors()

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

    def assign_thompson(
        self,
        user_id: str,
        experiment_id: str | None = None,
    ) -> ExperimentAssignment:
        """使用 Thompson Sampling 动态选择实验组。"""
        experiment_id = experiment_id or self.DEFAULT_EXPERIMENT_ID
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return self.assign(user_id, experiment_id)

        best_variant: dict[str, Any] | None = None
        best_sample = -1.0
        with self._lock:
            variants = list(experiment["variants"])

        for variant in variants:
            sample = random.betavariate(
                variant.get("alpha", 1),
                variant.get("beta", 1),
            )
            if sample > best_sample:
                best_sample = sample
                best_variant = variant

        if best_variant is None:
            return self.assign(user_id, experiment_id)

        return ExperimentAssignment(
            experiment_id=experiment_id,
            group=best_variant["group"],
            reason=(
                "Thompson Sampling: "
                f"alpha={best_variant.get('alpha', 1)}, "
                f"beta={best_variant.get('beta', 1)}, "
                f"sample={best_sample:.4f}"
            ),
            config=best_variant.get("config", {}),
        )

    def record_exposure(
        self,
        experiment_id: str,
        group: str,
        user_id: str,
    ) -> None:
        """记录一次推荐结果曝光。"""
        self._append_event(
            experiment_id=experiment_id,
            group=group,
            user_id=user_id,
            event_type="exposure",
            product_id=None,
        )

    def record_outcome(
        self,
        experiment_id: str,
        group: str,
        user_id: str,
        success: bool,
        product_id: str | None = None,
    ) -> None:
        """记录点击/购买等正反馈，或跳过/点踩等负反馈。"""
        self._append_event(
            experiment_id=experiment_id,
            group=group,
            user_id=user_id,
            event_type="click" if success else "skip",
            product_id=product_id,
        )

        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            return

        with self._lock:
            for variant in experiment["variants"]:
                if variant["group"] != group:
                    continue
                if success:
                    variant["alpha"] = variant.get("alpha", 1) + 1
                else:
                    variant["beta"] = variant.get("beta", 1) + 1
                break

    def get_stats(self, experiment_id: str | None = None) -> dict[str, Any]:
        """返回每个实验组的曝光、点击、CTR 和 Thompson 计数。"""
        experiment_id = experiment_id or self.DEFAULT_EXPERIMENT_ID
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            return {}

        with self._lock:
            events = list(self._events)
            variants = list(experiment["variants"])

        stats: dict[str, dict[str, Any]] = {}
        for variant in variants:
            group = variant["group"]
            group_events = [
                event
                for event in events
                if event["experiment_id"] == experiment_id and event["group"] == group
            ]
            exposures = sum(1 for event in group_events if event["type"] == "exposure")
            clicks = sum(1 for event in group_events if event["type"] == "click")
            skips = sum(1 for event in group_events if event["type"] == "skip")
            alpha = variant.get("alpha", 1)
            beta = variant.get("beta", 1)
            stats[group] = {
                "exposures": exposures,
                "clicks": clicks,
                "skips": skips,
                "ctr": round(clicks / exposures, 4) if exposures else 0.0,
                "alpha": alpha,
                "beta": beta,
                "expected_ctr": round(alpha / (alpha + beta), 4),
            }
        return stats

    def list_experiments(self) -> dict[str, Any]:
        experiments_info: dict[str, Any] = {}
        for experiment_id, experiment in self.experiments.items():
            experiments_info[experiment_id] = {
                "name": experiment["name"],
                "description": experiment.get("description", ""),
                "variants": [
                    {
                        "group": variant["group"],
                        "traffic_percent": variant.get("traffic_percent", 0),
                        "description": variant.get("description", ""),
                        "config": variant.get("config", {}),
                    }
                    for variant in experiment["variants"]
                ],
                "stats": self.get_stats(experiment_id),
            }
        return {
            "default_experiment_id": self.DEFAULT_EXPERIMENT_ID,
            "experiments": experiments_info,
        }

    def reset_outcomes(self) -> None:
        """清空内存事件并重置 Beta 先验，主要用于测试。"""
        with self._lock:
            self._events.clear()
            self._init_beta_priors()

    def _append_event(
        self,
        experiment_id: str,
        group: str,
        user_id: str,
        event_type: str,
        product_id: str | None,
    ) -> None:
        with self._lock:
            self._events.append(
                {
                    "experiment_id": experiment_id,
                    "group": group,
                    "user_id": user_id,
                    "type": event_type,
                    "product_id": product_id,
                    "timestamp": time.time(),
                }
            )

    def _init_beta_priors(self) -> None:
        for experiment in self.experiments.values():
            for variant in experiment["variants"]:
                variant["alpha"] = 1
                variant["beta"] = 1

    def _bucket(self, user_id: str, experiment_id: str) -> int:
        key = f"{experiment_id}:{user_id}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return int(digest[:8], 16) % 10000
