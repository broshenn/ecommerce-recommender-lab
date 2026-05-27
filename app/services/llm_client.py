from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
JSON_OUTPUT_INSTRUCTION = "Return valid JSON only. Do not include markdown, comments, or extra text."


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat APIs."""

    def __init__(self):
        load_dotenv(BASE_DIR / ".env")
        self.api_key, self.base_url, self.model = self._load_config()
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))
        self.enable_thinking = self._load_enable_thinking()
        self._client = None
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "available": self._openai is not None,
            "base_url": self._masked_base_url(),
            "model": self.model,
            "enable_thinking": self.enable_thinking,
            "last_error": self._last_error,
        }

    @property
    def _openai(self):
        if self._is_placeholder_key(self.api_key):
            self._last_error = "LLM API key is not configured"
            return None
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
            self._last_error = None
            return self._client
        except Exception as exc:
            self._last_error = str(exc)
            return None

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        client = self._openai
        if client is None:
            return None

        try:
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            extra_body = self._extra_body()
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**request_kwargs)
            message = response.choices[0].message
            content = (
                message.content
                or getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
            )
            self._last_error = None
            return content.strip() if content else None
        except Exception as exc:
            self._last_error = str(exc)
            return None

    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        default: Any = None,
    ) -> Any:
        text = self.chat(
            system_prompt=system_prompt,
            user_message=f"{user_message}\n\n{JSON_OUTPUT_INSTRUCTION}",
        )
        if text is None:
            return default

        return self._parse_json_text(text, default=default)

    def _parse_json_text(self, text: str, *, default: Any = None) -> Any:
        for candidate in self._json_candidates(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return default

    def _strip_json_fence(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return cleaned.strip("`").strip()

    def _json_candidates(self, text: str) -> list[str]:
        cleaned = self._strip_json_fence(text)
        candidates = [cleaned]
        extracted = self._extract_json_candidate(cleaned)
        if extracted and extracted not in candidates:
            candidates.append(extracted)
        balanced_candidates: list[str] = []
        for candidate in candidates:
            balanced = self._balance_json_brackets(candidate)
            if balanced != candidate:
                balanced_candidates.append(balanced)
        return candidates + balanced_candidates

    def _extract_json_candidate(self, text: str) -> str | None:
        starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
        if not starts:
            return None
        start = min(starts)
        ends = [index for index in (text.rfind("]"), text.rfind("}")) if index >= start]
        end = max(ends) if ends else len(text) - 1
        return text[start : end + 1].strip()

    def _balance_json_brackets(self, text: str) -> str:
        stack: list[str] = []
        in_string = False
        escaped = False
        for char in text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                stack.append("]")
            elif char == "{":
                stack.append("}")
            elif char in "]}":
                if not stack or stack[-1] != char:
                    return text
                stack.pop()
        if in_string:
            return text
        return text + "".join(reversed(stack))

    def _extra_body(self) -> dict[str, Any]:
        if self.enable_thinking is None:
            return {}
        return {
            "enable_thinking": self.enable_thinking,
            "think": self.enable_thinking,
        }

    def _load_config(self) -> tuple[str, str, str]:
        llm_api_key = self._env_api_key("LLM_API_KEY")
        if llm_api_key:
            return (
                llm_api_key,
                os.getenv("LLM_API_BASE", "https://api.deepseek.com"),
                self._normalize_model(os.getenv("LLM_MODEL", "deepseek-chat")),
            )
        deepseek_api_key = self._env_api_key("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            return (
                deepseek_api_key,
                os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
                self._normalize_model(os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            )
        qwen_api_key = self._env_api_key("QWEN_API_KEY")
        if qwen_api_key:
            return (
                qwen_api_key,
                os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                self._normalize_model(os.getenv("QWEN_MODEL", "qwen-plus")),
            )
        dashscope_api_key = self._env_api_key("DASHSCOPE_API_KEY")
        if dashscope_api_key:
            return (
                dashscope_api_key,
                os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                self._normalize_model(os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")),
            )
        return "", "https://api.deepseek.com", "deepseek-chat"

    def _env_api_key(self, name: str) -> str:
        api_key = os.getenv(name, "")
        return "" if self._is_placeholder_key(api_key) else api_key

    def _normalize_model(self, model: str) -> str:
        model = model.strip()
        if model.endswith("]") and "[" in model:
            return model.rsplit("[", 1)[0]
        return model

    def _load_enable_thinking(self) -> bool | None:
        raw = os.getenv("LLM_ENABLE_THINKING", "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        if self._is_qwen_thinking_model(self.model):
            return False
        return None

    def _is_qwen_thinking_model(self, model: str) -> bool:
        normalized = model.strip().lower()
        return (
            normalized.startswith("qwen3")
            or "qwen3." in normalized
            or "qwen3-" in normalized
            or "qwen3:" in normalized
        )

    def _is_placeholder_key(self, api_key: str) -> bool:
        lowered = api_key.strip().lower()
        return (
            not lowered
            or "your-" in lowered
            or lowered
            in {
                "sk-xxx",
                "sk-placeholder",
                "sk-your-deepseek-api-key",
                "sk-your-qwen-api-key",
                "sk-your-dashscope-api-key",
            }
        )

    def _masked_base_url(self) -> str:
        return self.base_url.replace(self.api_key, "***") if self.api_key else self.base_url


llm_client = LLMClient()
