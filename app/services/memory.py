from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from app.database import get_connection, init_db
from app.models import ChatMessage, ConversationState
from app.services import feature_store

RECENT_MESSAGE_LIMIT = 8


class MemoryService:
    """SQLite-backed conversation memory with optional Redis state cache."""

    def get_or_create_state(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> ConversationState:
        init_db()
        resolved_session_id = session_id or f"s-{uuid.uuid4().hex[:12]}"
        cached = self._get_cached_state(resolved_session_id)
        if cached and cached.user_id == user_id:
            return cached

        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT state_json
                FROM conversation_states
                WHERE session_id = ? AND user_id = ?
                """,
                (resolved_session_id, user_id),
            ).fetchone()

        if row:
            state = ConversationState.model_validate(json.loads(row["state_json"]))
            self._set_cached_state(state)
            return state

        now = self._now()
        state = ConversationState(session_id=resolved_session_id, user_id=user_id)
        with get_connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversation_sessions
                    (session_id, user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (resolved_session_id, user_id, now, now),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO conversation_states
                    (session_id, user_id, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    resolved_session_id,
                    user_id,
                    json.dumps(state.model_dump(mode="json"), ensure_ascii=False),
                    now,
                ),
            )
        self._set_cached_state(state)
        return state

    def save_state(self, state: ConversationState) -> None:
        init_db()
        now = self._now()
        with get_connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversation_sessions
                    (session_id, user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (state.session_id, state.user_id, now, now),
            )
            connection.execute(
                """
                UPDATE conversation_sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (now, state.session_id),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO conversation_states
                    (session_id, user_id, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    state.session_id,
                    state.user_id,
                    json.dumps(state.model_dump(mode="json"), ensure_ascii=False),
                    now,
                ),
            )
        self._set_cached_state(state)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        init_db()
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO conversation_messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, self._now()),
            )

    def recent_messages(self, session_id: str, limit: int = RECENT_MESSAGE_LIMIT) -> list[ChatMessage]:
        init_db()
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY message_id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        messages = [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in reversed(rows)
        ]
        return messages

    def record_memory_facts(
        self,
        *,
        user_id: str,
        facts: dict[str, Any],
        source: str,
    ) -> None:
        existing = self._existing_fact_keys(user_id)
        rows: list[tuple[str, str, str, str, str]] = []
        now = self._now()
        for fact_type, raw_value in facts.items():
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                if value in (None, "", []):
                    continue
                fact_value = str(value)
                fact_key = (fact_type, fact_value, source)
                if fact_key in existing:
                    continue
                existing.add(fact_key)
                rows.append((user_id, fact_type, fact_value, source, now))
        if not rows:
            return

        init_db()
        with get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO user_memory_facts
                    (user_id, fact_type, fact_value, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def user_memory_summary(self, user_id: str, limit: int = 200) -> dict[str, Any]:
        init_db()
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT fact_type, fact_value
                FROM user_memory_facts
                WHERE user_id = ?
                ORDER BY fact_id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        summary = {
            "shopping_goal": "",
            "budget_min": None,
            "budget_max": None,
            "preferred_categories": [],
            "liked_brands": [],
            "preferred_tags": [],
            "rejected_reasons": [],
        }
        list_fields = {
            "preferred_category": "preferred_categories",
            "liked_brand": "liked_brands",
            "preferred_tag": "preferred_tags",
            "rejected_reason": "rejected_reasons",
        }
        for row in rows:
            fact_type = row["fact_type"]
            fact_value = row["fact_value"]
            if fact_type == "shopping_goal" and not summary["shopping_goal"]:
                summary["shopping_goal"] = fact_value
            elif fact_type in {"budget_min", "budget_max"} and summary[fact_type] is None:
                summary[fact_type] = self._to_float(fact_value)
            elif fact_type in list_fields:
                target = list_fields[fact_type]
                if fact_value not in summary[target]:
                    summary[target].append(fact_value)
        return summary

    def clear_all(self) -> None:
        init_db()
        with get_connection() as connection:
            connection.execute("DELETE FROM conversation_messages")
            connection.execute("DELETE FROM conversation_states")
            connection.execute("DELETE FROM conversation_sessions")
            connection.execute("DELETE FROM user_memory_facts")
            connection.execute(
                "DELETE FROM sqlite_sequence WHERE name IN (?, ?)",
                ("conversation_messages", "user_memory_facts"),
            )
        redis_client = self._redis()
        if redis_client:
            for pattern in ("conversation_state:*",):
                keys = list(redis_client.scan_iter(match=pattern))
                if keys:
                    redis_client.delete(*keys)

    def _get_cached_state(self, session_id: str) -> ConversationState | None:
        redis_client = self._redis()
        if not redis_client:
            return None
        try:
            raw = redis_client.get(self._state_key(session_id))
        except Exception:
            return None
        if not raw:
            return None
        try:
            return ConversationState.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            return None

    def _set_cached_state(self, state: ConversationState) -> None:
        redis_client = self._redis()
        if not redis_client:
            return
        try:
            redis_client.set(
                self._state_key(state.session_id),
                json.dumps(state.model_dump(mode="json"), ensure_ascii=False),
                ex=3600,
            )
        except Exception:
            return

    def _redis(self):
        if not feature_store.is_available():
            return None
        return feature_store._redis()

    def _state_key(self, session_id: str) -> str:
        return f"conversation_state:{session_id}"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _existing_fact_keys(self, user_id: str) -> set[tuple[str, str, str]]:
        init_db()
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT fact_type, fact_value, source
                FROM user_memory_facts
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchall()
        return {
            (row["fact_type"], row["fact_value"], row["source"])
            for row in rows
        }

    def _to_float(self, value: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
