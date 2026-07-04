from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.models import Product, UserEvent, UserProfile

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ONE_HOUR_SECONDS = 3600
ONE_DAY_SECONDS = 86400
SEVEN_DAYS_SECONDS = 604800


class RedisFeatureStore:
    """Redis 特征服务：在线画像缓存 + 实时行为时间窗。"""

    def __init__(self):
        load_dotenv(BASE_DIR / ".env")
        self.redis_url = os.getenv("REDIS_URL") or os.getenv("ECOM_REDIS_URL") or "redis://localhost:6379/0"
        self.behavior_ttl_seconds = int(os.getenv("FEATURE_STORE_BEHAVIOR_TTL_SECONDS", "604800"))
        self.profile_ttl_seconds = int(os.getenv("FEATURE_STORE_PROFILE_TTL_SECONDS", "600"))
        self._client = None
        self._available: bool | None = None
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        available = self.is_available()
        return {
            "backend": "redis",
            "available": available,
            "redis_url": self._masked_url(),
            "behavior_ttl_seconds": self.behavior_ttl_seconds,
            "profile_ttl_seconds": self.profile_ttl_seconds,
            "last_error": self._last_error,
        }

    def is_available(self) -> bool:
        return self._redis() is not None

    def record_behavior(
        self,
        event: UserEvent,
        product: Product | None,
    ) -> bool:
        redis_client = self._redis()
        if not redis_client:
            return False

        key = self._behavior_key(event.user_id, event.event_type)
        timestamp = event.created_at.timestamp()
        payload = {
            "event_id": event.event_id,
            "user_id": event.user_id,
            "product_id": event.product_id,
            "event_type": event.event_type,
            "ts": timestamp,
        }
        if product:
            payload.update(
                {
                    "category": product.category,
                    "brand": product.brand,
                    "tags": product.tags,
                    "price": product.price,
                }
            )

        try:
            redis_client.zadd(
                key,
                {json.dumps(payload, ensure_ascii=False): timestamp},
            )
            redis_client.expire(key, self.behavior_ttl_seconds)
            return True
        except Exception as exc:
            self._mark_error(exc)
            return False

    def get_recent_behaviors(
        self,
        user_id: str,
        event_type: str,
        window_seconds: int,
    ) -> list[dict[str, Any]]:
        redis_client = self._redis()
        if not redis_client:
            return []

        key = self._behavior_key(user_id, event_type)
        cutoff = time.time() - window_seconds
        try:
            raw_items = redis_client.zrangebyscore(key, cutoff, "+inf")
        except Exception as exc:
            self._mark_error(exc)
            return []

        behaviors = []
        for raw in raw_items:
            try:
                behaviors.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return behaviors

    def get_user_features(self, user_id: str) -> dict[str, Any]:
        views_1h = self.get_recent_behaviors(user_id, "view", ONE_HOUR_SECONDS)
        views_24h = self.get_recent_behaviors(user_id, "view", ONE_DAY_SECONDS)
        likes_24h = self.get_recent_behaviors(user_id, "like", ONE_DAY_SECONDS)
        dislikes_24h = self.get_recent_behaviors(user_id, "dislike", ONE_DAY_SECONDS)
        carts_7d = self.get_recent_behaviors(user_id, "purchase", SEVEN_DAYS_SECONDS)

        return {
            "user_id": user_id,
            "view_count_1h": len(views_1h),
            "view_count_24h": len(views_24h),
            "like_count_24h": len(likes_24h),
            "dislike_count_24h": len(dislikes_24h),
            "purchase_count_7d": len(carts_7d),
            "recent_views": self._recent_product_ids(views_24h, 20),
            "recent_likes": self._recent_product_ids(likes_24h, 20),
            "recent_dislikes": self._recent_product_ids(dislikes_24h, 20),
            "recent_cart_items": self._recent_product_ids(carts_7d, 20),
            "recent_categories": self._unique(
                [item.get("category", "") for item in [*views_24h, *likes_24h, *carts_7d]]
            ),
            "recent_brands": self._unique(
                [item.get("brand", "") for item in [*likes_24h, *carts_7d]]
            ),
            "recent_tags": self._unique(
                tag
                for item in [*likes_24h, *carts_7d]
                for tag in item.get("tags", [])
            ),
            "rfm": self._compute_rfm(carts_7d),
        }

    def get_cached_profile(self, user_id: str) -> UserProfile | None:
        redis_client = self._redis()
        if not redis_client:
            return None

        try:
            raw = redis_client.get(self._profile_key(user_id))
        except Exception as exc:
            self._mark_error(exc)
            return None
        if not raw:
            return None
        try:
            return UserProfile.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            self.invalidate_profile(user_id)
            return None

    def set_cached_profile(self, profile: UserProfile) -> bool:
        redis_client = self._redis()
        if not redis_client:
            return False

        try:
            redis_client.set(
                self._profile_key(profile.user_id),
                json.dumps(profile.model_dump(mode="json"), ensure_ascii=False),
                ex=self.profile_ttl_seconds,
            )
            return True
        except Exception as exc:
            self._mark_error(exc)
            return False

    def invalidate_profile(self, user_id: str) -> bool:
        redis_client = self._redis()
        if not redis_client:
            return False
        try:
            redis_client.delete(self._profile_key(user_id))
            return True
        except Exception as exc:
            self._mark_error(exc)
            return False

    def clear_all(self) -> None:
        redis_client = self._redis()
        if not redis_client:
            return
        try:
            for pattern in ("behavior:*", "profile:*"):
                keys = list(redis_client.scan_iter(match=pattern))
                if keys:
                    redis_client.delete(*keys)
        except Exception as exc:
            self._mark_error(exc)

    def _redis(self):
        if self._available is False:
            return None
        if self._client is not None:
            return self._client

        try:
            import redis

            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self._client.ping()
            self._available = True
            self._last_error = None
            return self._client
        except Exception as exc:
            self._mark_error(exc)
            return None

    def _mark_error(self, exc: Exception) -> None:
        self._available = False
        self._client = None
        self._last_error = str(exc)

    def _masked_url(self) -> str:
        if "@" not in self.redis_url:
            return self.redis_url
        scheme, rest = self.redis_url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"

    def _behavior_key(self, user_id: str, event_type: str) -> str:
        return f"behavior:{user_id}:{event_type}"

    def _profile_key(self, user_id: str) -> str:
        return f"profile:{user_id}"

    def _recent_product_ids(self, behaviors: list[dict[str, Any]], limit: int) -> list[str]:
        return self._unique(item.get("product_id", "") for item in reversed(behaviors))[:limit]

    def _compute_rfm(self, cart_behaviors: list[dict[str, Any]]) -> dict[str, float]:
        """学习版用购买行为近似计算 RFM。"""
        if not cart_behaviors:
            return {"recency": 0.0, "frequency": 0.0, "monetary": 0.0}

        now = time.time()
        latest_ts = max(float(item.get("ts", 0)) for item in cart_behaviors)
        days_since = (now - latest_ts) / 86400
        avg_amount = sum(float(item.get("price", 100)) for item in cart_behaviors) / len(cart_behaviors)

        return {
            "recency": round(max(0.0, 1.0 - days_since / 30), 3),
            "frequency": round(min(1.0, len(cart_behaviors) / 10), 3),
            "monetary": round(min(1.0, avg_amount / 1000), 3),
        }

    def _unique(self, values) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value and value not in seen:
                result.append(value)
                seen.add(value)
        return result
