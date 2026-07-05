from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "training" / "query_intent_bert"


class IntentModelClassifier:
    """Optional local BERT-style intent classifier.

    The application keeps rule-based slot extraction as the stable baseline.
    This classifier is loaded only when CHAT_INTENT_MODEL_ENABLED=true.
    """

    def __init__(self):
        self.enabled = os.getenv("CHAT_INTENT_MODEL_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.model_dir = Path(os.getenv("CHAT_INTENT_MODEL_PATH", str(DEFAULT_MODEL_DIR)))
        self.confidence_threshold = float(os.getenv("CHAT_INTENT_MODEL_MIN_CONFIDENCE", "0.65"))
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._id_to_label: dict[int, str] = {}
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_dir": str(self.model_dir),
            "loaded": self._model is not None,
            "confidence_threshold": self.confidence_threshold,
            "last_error": self._last_error,
        }

    def classify(self, text: str, force: bool = False) -> dict[str, Any] | None:
        if not self.enabled and not force:
            return None
        if not self._ensure_loaded():
            return None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=64,
            padding="max_length",
            return_tensors="pt",
        )
        with self._torch.no_grad():
            outputs = self._model(**encoded)
            probabilities = self._torch.softmax(outputs.logits, dim=-1)[0]
            confidence, label_id = self._torch.max(probabilities, dim=-1)
        intent = self._id_to_label.get(int(label_id), "")
        score = float(confidence)
        if not intent or score < self.confidence_threshold:
            return None
        return {
            "intent": intent,
            "confidence": score,
            "model_dir": str(self.model_dir),
        }

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.model_dir.exists():
            self._last_error = f"model dir does not exist: {self.model_dir}"
            return False
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            labels_path = self.model_dir / "intent_labels.json"
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            self._id_to_label = {
                int(key): value
                for key, value in labels["id_to_label"].items()
            }
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self._model.eval()
            self._torch = torch
            self._last_error = None
            return True
        except Exception as exc:
            self._last_error = str(exc)
            return False


intent_model_classifier = IntentModelClassifier()
