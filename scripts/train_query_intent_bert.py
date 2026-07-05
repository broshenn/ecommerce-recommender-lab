from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INTENTS = [
    "recommend_products",
    "refine_preferences",
    "compare_products",
    "explain_recommendation",
    "record_feedback",
    "ask_product",
    "smalltalk",
]
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "query_understanding_train.jsonl"
DEFAULT_EVAL = PROJECT_ROOT / "data" / "query_understanding_eval.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training" / "query_intent_bert"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "reports" / "query_understanding_bert_predictions_latest.jsonl"
DEFAULT_METRICS = PROJECT_ROOT / "reports" / "query_understanding_bert_metrics_latest.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small BERT-style classifier for query intent classification."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--model-name", default="uer/chinese_roberta_L-2_H-128")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    report = train_and_evaluate(
        train_path=args.train,
        eval_path=args.eval,
        model_name=args.model_name,
        output_dir=args.output_dir,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Model saved: {args.output_dir}")
    print(f"Predictions written: {args.predictions}")
    print(f"Metrics written: {args.metrics}")


def train_and_evaluate(
    *,
    train_path: Path,
    eval_path: Path,
    model_name: str,
    output_dir: Path,
    predictions_path: Path,
    metrics_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    torch, transformers = import_ml_dependencies()
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    set_seed(seed, torch)
    device = resolve_device(device_name, torch)
    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    label_to_id = {label: index for index, label in enumerate(INTENTS)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(INTENTS),
        id2label={str(key): value for key, value in id_to_label.items()},
        label2id=label_to_id,
    )
    model.to(device)

    class QueryIntentDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.rows[index]
            encoded = tokenizer(
                row["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(label_to_id[row["intent"]], dtype=torch.long),
            }

    train_loader = DataLoader(
        QueryIntentDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        QueryIntentDataset(eval_rows),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        epoch_metrics, _ = evaluate_model(
            model=model,
            rows=eval_rows,
            loader=eval_loader,
            id_to_label=id_to_label,
            torch=torch,
            device=device,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(sum(losses) / len(losses), 6) if losses else 0.0,
                **epoch_metrics,
            }
        )

    final_metrics, prediction_rows = evaluate_model(
        model=model,
        rows=eval_rows,
        loader=eval_loader,
        id_to_label=id_to_label,
        torch=torch,
        device=device,
    )
    expected = [row["intent"] for row in eval_rows]
    predicted = [row["intent"] for row in prediction_rows]
    final_metrics["classification_report"] = classification_report(
        expected,
        predicted,
        labels=INTENTS,
        zero_division=0,
        output_dict=True,
    )
    final_metrics["accuracy"] = round(accuracy_score(expected, predicted), 4)
    final_metrics["macro_f1"] = round(f1_score(expected, predicted, labels=INTENTS, average="macro"), 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "intent_labels.json").write_text(
        json.dumps(
            {
                "intents": INTENTS,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "model_name": model_name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions_path, prediction_rows)
    report = {
        "summary": {
            "model_name": model_name,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "epochs": epochs,
            "batch_size": batch_size,
            "device": str(device),
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        },
        "history": history,
        "metrics": final_metrics,
        "output_dir": str(output_dir),
        "predictions_path": str(predictions_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def evaluate_model(*, model, rows, loader, id_to_label, torch, device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    prediction_rows = []
    expected = []
    predicted = []
    latencies = []
    with torch.no_grad():
        row_offset = 0
        for batch in loader:
            labels = batch["labels"]
            features = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            started = time.perf_counter()
            outputs = model(**features)
            elapsed_ms = (time.perf_counter() - started) * 1000
            probabilities = torch.softmax(outputs.logits, dim=-1).cpu()
            pred_ids = probabilities.argmax(dim=-1).tolist()
            for local_index, pred_id in enumerate(pred_ids):
                row = rows[row_offset + local_index]
                confidence = float(probabilities[local_index][pred_id])
                predicted_intent = id_to_label[int(pred_id)]
                expected.append(row["intent"])
                predicted.append(predicted_intent)
                latencies.append(elapsed_ms / max(1, len(pred_ids)))
                prediction_rows.append(
                    {
                        "text": row["text"],
                        "intent": predicted_intent,
                        "confidence": round(confidence, 6),
                        "expected_intent": row["intent"],
                        "latency_ms": round(elapsed_ms / max(1, len(pred_ids)), 4),
                        "slots": {},
                    }
                )
            row_offset += len(pred_ids)
    return (
        {
            "accuracy": round(accuracy_score(expected, predicted), 4),
            "macro_f1": round(f1_score(expected, predicted, labels=INTENTS, average="macro"), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        },
        prediction_rows,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def import_ml_dependencies():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependency. Install with: pip install -r requirements-ml.txt"
        ) from exc
    return torch, transformers


def resolve_device(device_name: str, torch):
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
