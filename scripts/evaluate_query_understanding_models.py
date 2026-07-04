from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("QWEN_API_KEY", "")
os.environ.setdefault("DASHSCOPE_API_KEY", "")
os.environ.setdefault("PRODUCT_VECTOR_EMBEDDING_PROVIDER", "local")

from app.agents.intent_agent import IntentAgent  # noqa: E402
from app.models import ConversationState  # noqa: E402
from app.services import llm_client  # noqa: E402

DEFAULT_TRAIN = PROJECT_ROOT / "data" / "query_understanding_train.jsonl"
DEFAULT_EVAL = PROJECT_ROOT / "data" / "query_understanding_eval.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "query_understanding_model_compare_latest.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "query_understanding_model_compare_latest.md"

SLOT_KEYS = [
    "budget_min",
    "budget_max",
    "preferred_categories",
    "liked_brands",
    "preferred_tags",
    "event_type",
    "product_refs",
]
INTENTS = [
    "recommend_products",
    "refine_preferences",
    "compare_products",
    "explain_recommendation",
    "record_feedback",
    "ask_product",
    "smalltalk",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Query Understanding models for conversational commerce."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--llm-limit", type=int, default=30)
    parser.add_argument("--bert-predictions", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate_models(
        train_path=args.train,
        eval_path=args.eval,
        include_llm=args.include_llm,
        llm_limit=args.llm_limit,
        bert_predictions=args.bert_predictions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report written: {args.output}")
    print(f"Markdown report written: {args.markdown_output}")


def evaluate_models(
    *,
    train_path: Path,
    eval_path: Path,
    include_llm: bool = False,
    llm_limit: int = 30,
    bert_predictions: Path | None = None,
) -> dict[str, Any]:
    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    models = [
        evaluate_rule_baseline(eval_rows),
        evaluate_char_ngram_nb(train_rows, eval_rows),
    ]
    models.append(evaluate_llm_classifier(eval_rows[:llm_limit], include_llm))
    models.append(evaluate_distilbert_predictions(eval_rows, bert_predictions))
    completed = [model for model in models if model["status"] == "completed"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "dataset_summary": {
            "train": summarize_dataset(train_rows),
            "eval": summarize_dataset(eval_rows),
        },
        "summary": summarize_models(models),
        "models": models,
        "recommendation": recommend_model(completed),
        "notes": [
            "rule_baseline is the production fallback used by IntentAgent today.",
            "char_ngram_nb is a small trainable baseline for intent only; it does not extract slots.",
            "llm_classifier and distilbert_classifier are optional to avoid making the main demo depend on external APIs or GPU training.",
        ],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_rule_baseline(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent = IntentAgent()
    predictions = []
    latencies = []
    for index, row in enumerate(eval_rows):
        state = ConversationState(session_id=f"qu-rule-{index}", user_id="query-eval")
        started = time.perf_counter()
        result = agent._rule_intent(row["text"], state)
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append(
            {
                "intent": result.intent,
                "slots": normalize_predicted_slots(result.slots, result.product_refs),
            }
        )
    return score_predictions(
        name="rule_baseline",
        status="completed",
        eval_rows=eval_rows,
        predictions=predictions,
        latencies=latencies,
        description="Current production fallback: deterministic rules in IntentAgent.",
    )


def evaluate_char_ngram_nb(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    classifier = CharNgramNaiveBayes()
    classifier.fit([(row["text"], row["intent"]) for row in train_rows])
    predictions = []
    latencies = []
    for row in eval_rows:
        started = time.perf_counter()
        intent = classifier.predict(row["text"])
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append({"intent": intent, "slots": empty_slots()})
    return score_predictions(
        name="char_ngram_nb_classifier",
        status="completed",
        eval_rows=eval_rows,
        predictions=predictions,
        latencies=latencies,
        description="Trainable intent-only baseline using character n-gram Naive Bayes.",
    )


def evaluate_llm_classifier(
    eval_rows: list[dict[str, Any]],
    include_llm: bool,
) -> dict[str, Any]:
    if not include_llm:
        return skipped_model(
            "llm_classifier",
            "Pass --include-llm to call the configured OpenAI-compatible LLM.",
            "Optional classifier for complex natural language; not required for local demo.",
        )
    status = llm_client.status()
    if not status.get("available"):
        return skipped_model(
            "llm_classifier",
            f"LLM is not available: {status.get('last_error')}",
            "Optional classifier for complex natural language; not required for local demo.",
        )

    predictions = []
    latencies = []
    for row in eval_rows:
        started = time.perf_counter()
        raw = llm_client.chat_json(
            system_prompt=(
                "你是电商 Query Understanding 分类器。只输出 JSON："
                "{intent, slots}。intent 必须是固定枚举之一。"
            ),
            user_message=json.dumps(
                {
                    "text": row["text"],
                    "intents": INTENTS,
                    "slot_schema": SLOT_KEYS,
                },
                ensure_ascii=False,
            ),
            default={},
        )
        latencies.append((time.perf_counter() - started) * 1000)
        intent = raw.get("intent") if isinstance(raw, dict) else ""
        slots = raw.get("slots") if isinstance(raw, dict) and isinstance(raw.get("slots"), dict) else {}
        predictions.append(
            {
                "intent": intent if intent in INTENTS else "smalltalk",
                "slots": normalize_predicted_slots(slots, slots.get("product_refs", [])),
            }
        )
    return score_predictions(
        name="llm_classifier",
        status="completed",
        eval_rows=eval_rows,
        predictions=predictions,
        latencies=latencies,
        description="OpenAI-compatible LLM classifier, evaluated only when explicitly enabled.",
    )


def evaluate_distilbert_predictions(
    eval_rows: list[dict[str, Any]],
    predictions_path: Path | None,
) -> dict[str, Any]:
    if not predictions_path:
        return skipped_model(
            "distilbert_classifier",
            "No --bert-predictions file provided.",
            "Use an offline DistilBERT/BERT trainer to write JSONL predictions with text and intent.",
        )
    if not predictions_path.exists():
        return skipped_model(
            "distilbert_classifier",
            f"Prediction file does not exist: {predictions_path}",
            "Use an offline DistilBERT/BERT trainer to write JSONL predictions with text and intent.",
        )

    prediction_rows = load_jsonl(predictions_path)
    by_text = {row.get("text", ""): row for row in prediction_rows}
    predictions = []
    for row in eval_rows:
        item = by_text.get(row["text"], {})
        predictions.append(
            {
                "intent": item.get("intent", "smalltalk"),
                "slots": normalize_predicted_slots(item.get("slots", {}), []),
            }
        )
    return score_predictions(
        name="distilbert_classifier",
        status="completed",
        eval_rows=eval_rows,
        predictions=predictions,
        latencies=[0.0 for _ in eval_rows],
        description="External DistilBERT/BERT intent classifier evaluated from saved predictions.",
    )


def score_predictions(
    *,
    name: str,
    status: str,
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    latencies: list[float],
    description: str,
) -> dict[str, Any]:
    intent_metrics = score_intents(
        [row["intent"] for row in eval_rows],
        [prediction["intent"] for prediction in predictions],
    )
    slot_metrics = score_slots(
        [normalize_expected_slots(row.get("slots", {})) for row in eval_rows],
        [prediction["slots"] for prediction in predictions],
    )
    need_recommendation_accuracy = accuracy(
        [bool(row.get("need_recommendation")) for row in eval_rows],
        [prediction["intent"] in {"recommend_products", "refine_preferences", "record_feedback"} for prediction in predictions],
    )
    return {
        "name": name,
        "status": status,
        "description": description,
        "case_count": len(eval_rows),
        "intent_accuracy": intent_metrics["accuracy"],
        "intent_macro_f1": intent_metrics["macro_f1"],
        "intent_per_label": intent_metrics["per_label"],
        "slot_f1": slot_metrics["f1"],
        "slot_precision": slot_metrics["precision"],
        "slot_recall": slot_metrics["recall"],
        "need_recommendation_accuracy": need_recommendation_accuracy,
        "avg_latency_ms": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "cost_level": cost_level(name),
        "online_suitability": online_suitability(name),
        "sample_errors": sample_errors(eval_rows, predictions),
    }


def skipped_model(name: str, reason: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "skipped",
        "description": description,
        "skip_reason": reason,
    }


class CharNgramNaiveBayes:
    def __init__(self, ngram_range: tuple[int, int] = (1, 3), alpha: float = 1.0):
        self.ngram_range = ngram_range
        self.alpha = alpha
        self.class_counts: Counter[str] = Counter()
        self.feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.feature_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, rows: list[tuple[str, str]]) -> None:
        for text, label in rows:
            self.class_counts[label] += 1
            for feature in self.features(text):
                self.feature_counts[label][feature] += 1
                self.feature_totals[label] += 1
                self.vocabulary.add(feature)

    def predict(self, text: str) -> str:
        features = self.features(text)
        total_docs = sum(self.class_counts.values())
        best_label = ""
        best_score = -math.inf
        vocab_size = max(1, len(self.vocabulary))
        for label in self.class_counts:
            score = math.log(self.class_counts[label] / total_docs)
            denom = self.feature_totals[label] + self.alpha * vocab_size
            counts = self.feature_counts[label]
            for feature in features:
                score += math.log((counts[feature] + self.alpha) / denom)
            if score > best_score:
                best_label = label
                best_score = score
        return best_label or "smalltalk"

    def features(self, text: str) -> list[str]:
        cleaned = "".join(ch.lower() for ch in text if not ch.isspace())
        result = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            result.extend(cleaned[index:index + n] for index in range(max(0, len(cleaned) - n + 1)))
        return result or [cleaned]


def score_intents(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = sorted(set(expected) | set(predicted) | set(INTENTS))
    per_label = {}
    f1_values = []
    correct = 0
    for label in labels:
        tp = sum(1 for exp, pred in zip(expected, predicted) if exp == label and pred == label)
        fp = sum(1 for exp, pred in zip(expected, predicted) if exp != label and pred == label)
        fn = sum(1 for exp, pred in zip(expected, predicted) if exp == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for exp in expected if exp == label),
        }
        if per_label[label]["support"]:
            f1_values.append(f1)
    correct = sum(1 for exp, pred in zip(expected, predicted) if exp == pred)
    return {
        "accuracy": round(correct / len(expected), 4) if expected else 0.0,
        "macro_f1": round(statistics.mean(f1_values), 4) if f1_values else 0.0,
        "per_label": per_label,
    }


def score_slots(expected: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, float]:
    expected_pairs = []
    predicted_pairs = []
    for index, item in enumerate(expected):
        expected_pairs.extend((index, *pair) for pair in flatten_slots(item))
    for index, item in enumerate(predicted):
        predicted_pairs.extend((index, *pair) for pair in flatten_slots(item))
    expected_set = set(expected_pairs)
    predicted_set = set(predicted_pairs)
    matches = len(expected_set & predicted_set)
    precision = matches / len(predicted_set) if predicted_set else (1.0 if not expected_set else 0.0)
    recall = matches / len(expected_set) if expected_set else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def normalize_expected_slots(slots: dict[str, Any]) -> dict[str, Any]:
    return {
        key: slots.get(key)
        for key in SLOT_KEYS
        if slots.get(key) not in (None, "", [])
    }


def normalize_predicted_slots(slots: dict[str, Any], product_refs: list[str]) -> dict[str, Any]:
    normalized = {
        "budget_min": slots.get("budget_min"),
        "budget_max": slots.get("budget_max"),
        "preferred_categories": slots.get("preferred_categories", []),
        "liked_brands": slots.get("liked_brands", []),
        "preferred_tags": slots.get("preferred_tags", []),
        "event_type": slots.get("event_type", ""),
        "product_refs": slots.get("product_refs") or product_refs or [],
    }
    return {
        key: value
        for key, value in normalized.items()
        if value not in (None, "", [])
    }


def empty_slots() -> dict[str, Any]:
    return {}


def flatten_slots(slots: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = set()
    for key, raw_value in slots.items():
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            if value not in (None, "", []):
                pairs.add((key, normalize_value(value)))
    return pairs


def normalize_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def accuracy(expected: list[bool], predicted: list[bool]) -> float:
    if not expected:
        return 0.0
    return round(sum(1 for exp, pred in zip(expected, predicted) if exp == pred) / len(expected), 4)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return round(ordered[index], 4)


def sample_errors(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    errors = []
    for row, prediction in zip(eval_rows, predictions):
        if row["intent"] != prediction["intent"]:
            errors.append(
                {
                    "text": row["text"],
                    "expected_intent": row["intent"],
                    "predicted_intent": prediction["intent"],
                }
            )
        if len(errors) >= limit:
            break
    return errors


def summarize_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "intent_distribution": dict(Counter(row["intent"] for row in rows)),
        "source_distribution": dict(Counter(row.get("source", "") for row in rows)),
    }


def summarize_models(models: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [model for model in models if model["status"] == "completed"]
    return {
        "completed_models": [model["name"] for model in completed],
        "skipped_models": {
            model["name"]: model.get("skip_reason", "")
            for model in models
            if model["status"] == "skipped"
        },
        "best_by_intent_macro_f1": max(
            completed,
            key=lambda item: item["intent_macro_f1"],
        )["name"] if completed else "",
        "best_by_latency": min(
            completed,
            key=lambda item: item["avg_latency_ms"],
        )["name"] if completed else "",
    }


def recommend_model(completed: list[dict[str, Any]]) -> dict[str, str]:
    if not completed:
        return {"winner": "", "reason": "No completed model evaluation."}
    best_intent = max(completed, key=lambda item: item["intent_macro_f1"])
    best_slots = max(completed, key=lambda item: item["slot_f1"])
    if best_intent["name"] != best_slots["name"] and best_slots["slot_f1"] > 0:
        return {
            "winner": f"{best_intent['name']} + {best_slots['name']}",
            "reason": (
                "Use the trainable classifier for high-frequency intent classification, "
                "and keep rule extraction for slots until a sequence-labeling model is trained."
            ),
        }
    return {
        "winner": best_intent["name"],
        "reason": "Highest intent macro F1 among completed models.",
    }


def cost_level(name: str) -> str:
    if name == "llm_classifier":
        return "high"
    if name == "distilbert_classifier":
        return "medium"
    return "low"


def online_suitability(name: str) -> str:
    if name == "rule_baseline":
        return "excellent fallback and demo baseline"
    if name == "char_ngram_nb_classifier":
        return "good for high-frequency intent classification, weak for slots"
    if name == "distilbert_classifier":
        return "good after offline training, still needs slot extractor"
    if name == "llm_classifier":
        return "good for complex language, less ideal for every request due to cost and latency"
    return ""


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Query Understanding Model Compare Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Train set: `{report['train_count']}` rows",
        f"- Eval set: `{report['eval_count']}` rows",
        f"- Recommendation: `{report['recommendation']['winner']}` - {report['recommendation']['reason']}",
        "",
        "## Model Summary",
        "",
        "| Model | Status | Intent Acc | Intent Macro F1 | Slot F1 | Need Rec Acc | Avg Latency ms | Cost |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model in report["models"]:
        if model["status"] == "skipped":
            lines.append(
                f"| `{model['name']}` | skipped: {model['skip_reason']} | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            "| "
            f"`{model['name']}` | completed | {model['intent_accuracy']} | "
            f"{model['intent_macro_f1']} | {model['slot_f1']} | "
            f"{model['need_recommendation_accuracy']} | {model['avg_latency_ms']} | "
            f"{model['cost_level']} |"
        )
    lines.extend(["", "## Dataset", ""])
    for split in ["train", "eval"]:
        summary = report["dataset_summary"][split]
        lines.append(f"### {split}")
        lines.append("")
        lines.append(f"- Count: `{summary['count']}`")
        lines.append(f"- Intent distribution: `{summary['intent_distribution']}`")
        lines.append(f"- Source distribution: `{summary['source_distribution']}`")
        lines.append("")
    lines.extend(["## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
