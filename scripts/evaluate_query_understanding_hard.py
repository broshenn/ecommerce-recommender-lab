from __future__ import annotations

import argparse
import json
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
from app.services.intent_classifier import intent_model_classifier  # noqa: E402
from scripts.evaluate_query_understanding_models import (  # noqa: E402
    CharNgramNaiveBayes,
    INTENTS,
    load_jsonl,
    normalize_expected_slots,
    normalize_predicted_slots,
    score_intents,
    score_slots,
)

DEFAULT_TRAIN = PROJECT_ROOT / "data" / "query_understanding_train.jsonl"
DEFAULT_HARD_EVAL = PROJECT_ROOT / "data" / "query_understanding_hard_eval.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "query_understanding_hard_eval_latest.json"
DEFAULT_MARKDOWN_OUTPUT = PROJECT_ROOT / "reports" / "query_understanding_hard_eval_latest.md"
DEFAULT_BERT_MODEL = PROJECT_ROOT / "training" / "query_intent_bert"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate query understanding on a hand-written hard set.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_HARD_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--bert-model", type=Path, default=DEFAULT_BERT_MODEL)
    args = parser.parse_args()

    report = evaluate_hard_set(
        train_path=args.train,
        eval_path=args.eval,
        bert_model_path=args.bert_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Hard eval report written: {args.output}")
    print(f"Hard eval markdown written: {args.markdown_output}")


def evaluate_hard_set(
    *,
    train_path: Path = DEFAULT_TRAIN,
    eval_path: Path = DEFAULT_HARD_EVAL,
    bert_model_path: Path = DEFAULT_BERT_MODEL,
) -> dict[str, Any]:
    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    models = [
        evaluate_rule(eval_rows),
        evaluate_char_ngram(train_rows, eval_rows),
        evaluate_bert_rule_slots(eval_rows, bert_model_path),
    ]
    completed = [model for model in models if model["status"] == "completed"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "case_count": len(eval_rows),
        "scenario_distribution": dict(Counter(row.get("scenario", "") for row in eval_rows)),
        "summary": {
            "completed_models": [model["name"] for model in completed],
            "best_by_hard_intent_macro_f1": max(
                completed,
                key=lambda model: model["hard_intent_macro_f1"],
            )["name"] if completed else "",
            "best_by_hard_slot_f1": max(
                completed,
                key=lambda model: model["hard_slot_f1"],
            )["name"] if completed else "",
        },
        "models": models,
        "notes": [
            "This hard set is manually simulated to stress realistic phrasing, target switching, references, feedback, and smalltalk guards.",
            "BERT is evaluated as intent-only; slots still come from the rule extractor.",
            "Scores are more useful for relative comparison than as production guarantees.",
        ],
    }


def evaluate_rule(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent = IntentAgent()
    predictions = []
    latencies = []
    for index, row in enumerate(eval_rows):
        state = hard_state(index)
        started = time.perf_counter()
        result = agent._rule_intent(row["text"], state)
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append(
            {
                "intent": result.intent,
                "slots": normalize_predicted_slots(result.slots, result.product_refs),
            }
        )
    return score_model(
        name="rule_baseline",
        description="Production rule baseline with configured keywords and slot extraction.",
        eval_rows=eval_rows,
        predictions=predictions,
        latencies=latencies,
    )


def evaluate_char_ngram(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    classifier = CharNgramNaiveBayes()
    classifier.fit([(row["text"], row["intent"]) for row in train_rows])
    predictions = []
    latencies = []
    for row in eval_rows:
        started = time.perf_counter()
        intent = classifier.predict(row["text"])
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append({"intent": intent, "slots": {}})
    return score_model(
        name="char_ngram_nb_classifier",
        description="Trainable intent-only character n-gram baseline.",
        eval_rows=eval_rows,
        predictions=predictions,
        latencies=latencies,
    )


def evaluate_bert_rule_slots(eval_rows: list[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        return {
            "name": "bert_rule_slots",
            "status": "skipped",
            "skip_reason": f"model path does not exist: {model_path}",
        }
    intent_model_classifier.enabled = True
    intent_model_classifier.model_dir = model_path
    agent = IntentAgent()
    predictions = []
    latencies = []
    for index, row in enumerate(eval_rows):
        state = hard_state(index)
        rule_result = agent._rule_intent(row["text"], state)
        started = time.perf_counter()
        merged_result = agent._apply_model_intent(row["text"], rule_result)
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append(
            {
                "intent": merged_result.intent,
                "slots": normalize_predicted_slots(merged_result.slots, merged_result.product_refs),
            }
        )
    return score_model(
        name="bert_rule_slots",
        description="Fine-tuned BERT for intent classification plus rule-based slots.",
        eval_rows=eval_rows,
        predictions=predictions,
        latencies=latencies,
    )


def score_model(
    *,
    name: str,
    description: str,
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    latencies: list[float],
) -> dict[str, Any]:
    expected_intents = [row["intent"] for row in eval_rows]
    predicted_intents = [prediction["intent"] for prediction in predictions]
    intent_metrics = score_intents(expected_intents, predicted_intents)
    slot_metrics = score_slots(
        [normalize_expected_slots(row.get("slots", {})) for row in eval_rows],
        [prediction["slots"] for prediction in predictions],
    )
    scenario_summary = summarize_by_scenario(eval_rows, predictions)
    return {
        "name": name,
        "status": "completed",
        "description": description,
        "case_count": len(eval_rows),
        "hard_intent_accuracy": intent_metrics["accuracy"],
        "hard_intent_macro_f1": intent_metrics["macro_f1"],
        "hard_slot_f1": slot_metrics["f1"],
        "hard_slot_precision": slot_metrics["precision"],
        "hard_slot_recall": slot_metrics["recall"],
        "need_recommendation_accuracy": score_need_recommendation(eval_rows, predictions),
        "smalltalk_guard_rate": scenario_rate(eval_rows, predictions, "smalltalk", "smalltalk"),
        "negative_feedback_accuracy": scenario_rate(eval_rows, predictions, "negative_feedback", "record_feedback"),
        "goal_switch_accuracy": scenario_intent_accuracy(eval_rows, predictions, "goal_switch"),
        "product_ref_resolution_rate": product_ref_resolution_rate(eval_rows, predictions),
        "unsupported_catalog_guard_rate": unsupported_catalog_guard_rate(eval_rows, predictions),
        "avg_latency_ms": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "scenario_summary": scenario_summary,
        "sample_errors": sample_errors(eval_rows, predictions),
    }


def hard_state(index: int) -> ConversationState:
    return ConversationState(
        session_id=f"hard-eval-{index}",
        user_id="hard-eval-user",
        last_recommended_product_ids=["p-first", "p-second", "p-third"],
        active_product_refs={
            "第一个": "p-first",
            "第一款": "p-first",
            "1号": "p-first",
            "这款": "p-first",
            "这个": "p-first",
            "刚才那个": "p-first",
            "第二个": "p-second",
            "第二款": "p-second",
            "2号": "p-second",
            "第三个": "p-third",
            "第三款": "p-third",
            "3号": "p-third",
        },
    )


def summarize_by_scenario(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row, prediction in zip(eval_rows, predictions):
        buckets[row.get("scenario", "")].append((row, prediction))
    summary = {}
    for scenario, pairs in buckets.items():
        total = len(pairs)
        correct = sum(1 for row, pred in pairs if row["intent"] == pred["intent"])
        summary[scenario] = {
            "count": total,
            "intent_accuracy": round(correct / total, 4) if total else 0.0,
        }
    return summary


def score_need_recommendation(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> float:
    expected = [bool(row.get("need_recommendation")) for row in eval_rows]
    predicted = [
        prediction["intent"] in {"recommend_products", "refine_preferences"}
        or (
            prediction["intent"] == "record_feedback"
            and bool(row.get("need_recommendation"))
        )
        for row, prediction in zip(eval_rows, predictions)
    ]
    return round(sum(1 for exp, pred in zip(expected, predicted) if exp == pred) / len(expected), 4)


def scenario_rate(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scenario: str,
    expected_intent: str,
) -> float:
    pairs = [
        (row, prediction)
        for row, prediction in zip(eval_rows, predictions)
        if row.get("scenario") == scenario
    ]
    if not pairs:
        return 0.0
    return round(sum(1 for _, pred in pairs if pred["intent"] == expected_intent) / len(pairs), 4)


def scenario_intent_accuracy(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scenario: str,
) -> float:
    pairs = [
        (row, prediction)
        for row, prediction in zip(eval_rows, predictions)
        if row.get("scenario") == scenario
    ]
    if not pairs:
        return 0.0
    return round(sum(1 for row, pred in pairs if row["intent"] == pred["intent"]) / len(pairs), 4)


def product_ref_resolution_rate(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> float:
    rows_with_refs = []
    for row, prediction in zip(eval_rows, predictions):
        expected_refs = row.get("slots", {}).get("product_refs", [])
        if expected_refs:
            rows_with_refs.append((set(expected_refs), set(prediction["slots"].get("product_refs", []))))
    if not rows_with_refs:
        return 0.0
    return round(sum(1 for expected, predicted in rows_with_refs if expected <= predicted) / len(rows_with_refs), 4)


def unsupported_catalog_guard_rate(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> float:
    pairs = [
        (row, prediction)
        for row, prediction in zip(eval_rows, predictions)
        if row.get("constraints", {}).get("unsupported_catalog_awareness")
    ]
    if not pairs:
        return 0.0
    correct = 0
    for _, prediction in pairs:
        slots = prediction["slots"]
        if slots.get("preferred_categories") == ["电子数码"] and "电脑配件" in slots.get("preferred_tags", []):
            correct += 1
    return round(correct / len(pairs), 4)


def sample_errors(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    errors = []
    for row, prediction in zip(eval_rows, predictions):
        if row["intent"] != prediction["intent"]:
            errors.append(
                {
                    "case_id": row.get("case_id"),
                    "scenario": row.get("scenario"),
                    "text": row["text"],
                    "expected_intent": row["intent"],
                    "predicted_intent": prediction["intent"],
                }
            )
        if len(errors) >= limit:
            break
    return errors


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Query Understanding Hard Eval Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Case count: `{report['case_count']}`",
        f"- Best by hard intent macro F1: `{report['summary']['best_by_hard_intent_macro_f1']}`",
        f"- Best by hard slot F1: `{report['summary']['best_by_hard_slot_f1']}`",
        "",
        "## Model Summary",
        "",
        "| Model | Intent Acc | Intent Macro F1 | Slot F1 | Smalltalk Guard | Feedback Acc | Ref Rate | Unsupported Guard | Avg Lat ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in report["models"]:
        if model["status"] != "completed":
            lines.append(f"| `{model['name']}` | skipped: {model['skip_reason']} | - | - | - | - | - | - | - |")
            continue
        lines.append(
            "| "
            f"`{model['name']}` | {model['hard_intent_accuracy']} | {model['hard_intent_macro_f1']} | "
            f"{model['hard_slot_f1']} | {model['smalltalk_guard_rate']} | "
            f"{model['negative_feedback_accuracy']} | {model['product_ref_resolution_rate']} | "
            f"{model['unsupported_catalog_guard_rate']} | {model['avg_latency_ms']} |"
        )
    lines.extend(["", "## Scenario Distribution", ""])
    for scenario, count in report["scenario_distribution"].items():
        lines.append(f"- `{scenario}`: {count}")
    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
