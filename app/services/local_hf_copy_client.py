from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = r"D:\models\ecom-copy-lora-merged"


class LocalHFCopyClient:
    """Lazy local HuggingFace runner for the fine-tuned marketing copy model."""

    def __init__(self) -> None:
        load_dotenv(BASE_DIR / ".env")
        self.enabled = os.getenv("COPY_LLM_BACKEND", "").strip().lower() in {"hf", "local_hf"}
        self.model_path = os.getenv("COPY_LLM_HF_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.device = os.getenv("COPY_LLM_HF_DEVICE", "cuda").strip().lower()
        self.max_new_tokens = int(os.getenv("COPY_LLM_HF_MAX_NEW_TOKENS", "768"))
        self._tokenizer = None
        self._model = None
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "model_path": self.model_path,
            "device": self.device,
            "loaded": self._model is not None,
            "last_error": self._last_error,
        }

    @property
    def available(self) -> bool:
        return self.enabled and Path(self.model_path).exists()

    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        default: Any = None,
    ) -> Any:
        text = self.chat(system_prompt, user_message)
        if text is None:
            return default
        for candidate in self._json_candidates(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return default

    def chat(self, system_prompt: str, user_message: str) -> str | None:
        if not self.available:
            self._last_error = "local HF copy model is not enabled or path does not exist"
            return None
        if not self._ensure_loaded():
            return None

        try:
            import torch

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer([prompt], return_tensors="pt")
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = output[0][inputs["input_ids"].shape[-1] :]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            self._last_error = None
            return text.strip()
        except Exception as exc:
            self._last_error = str(exc)
            return None

    def _ensure_loaded(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            if self.device == "cuda" and not torch.cuda.is_available():
                self._last_error = "COPY_LLM_HF_DEVICE=cuda but CUDA is not available"
                return False

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
            self._last_error = None
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._model = None
            self._tokenizer = None
            return False

    def _json_candidates(self, text: str) -> list[str]:
        cleaned = self._strip_json_fence(text.strip())
        candidates = [cleaned]
        extracted = self._extract_json_candidate(cleaned)
        if extracted and extracted not in candidates:
            candidates.append(extracted)
        return candidates

    def _strip_json_fence(self, text: str) -> str:
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return text.strip("`").strip()

    def _extract_json_candidate(self, text: str) -> str | None:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end >= start:
            return text[start : end + 1].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            return text[start : end + 1].strip()
        return None


local_hf_copy_client = LocalHFCopyClient()
