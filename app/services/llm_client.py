from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat APIs."""

    def __init__(self):
        load_dotenv(BASE_DIR / ".env")
        self.api_key, self.base_url, self.model = self._load_config()
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))
        self._client = None
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "available": self._openai is not None,
            "base_url": self._masked_base_url(),
            "model": self.model,
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
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            message = response.choices[0].message
            content = message.content or getattr(message, "reasoning_content", None)
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
            user_message=f"{user_message}\n\n只输出JSON，不要其他内容。",
        )
        if text is None:
            return default

        try:
            return json.loads(self._strip_json_fence(text))
        except json.JSONDecodeError:
            return default

    def _strip_json_fence(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return cleaned.strip("`").strip()

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
