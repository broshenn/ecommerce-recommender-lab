from __future__ import annotations

from typing import Any

from app.behavior import build_user_profile, merge_behavior_profile
from app.models import AgentResult, RecommendRequest, UserProfile
from app.services import feature_store, llm_client

from app.agents.base_agent import BaseAgent


SYSTEM_PROMPT = """你是一个电商用户画像分析专家。根据用户的长期行为数据和实时行为特征，分析用户购物意图和偏好。

你需要输出以下JSON格式:
{
  "segments": ["用户分群标签"],
  "intent_summary": "一句话描述用户当前购物意图",
  "recommendation_hint": "给推荐系统的策略建议（一句话）",
  "price_sensitivity": "high/medium/low",
  "rfm_interpretation": "对RFM指标的解读（一句话）"
}

用户分群可选值: new_user, active, high_value, price_sensitive, churn_risk, category_explorer, brand_loyal
只输出JSON，不要其他内容。"""


class UserProfileAgent(BaseAgent):
    """Build the current user's profile from behavior data and LLM analysis."""

    def __init__(self):
        super().__init__(name="user_profile", timeout=10.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        request: RecommendRequest = kwargs["request"]
        experiment_group: str = kwargs.get("experiment_group", "")
        behavior_profile = build_user_profile(request.user_id)
        effective_request = merge_behavior_profile(request)
        online_features = feature_store.get_user_features(request.user_id)
        if experiment_group == "control":
            llm_profile: dict[str, Any] = {
                "segments": ["active"],
                "intent_summary": "LLM不可用，默认画像",
                "recommendation_hint": "",
                "price_sensitivity": "medium",
                "rfm_interpretation": "",
            }
        else:
            llm_profile = self._build_llm_profile(behavior_profile, online_features)

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "profile": behavior_profile.model_dump(mode="json"),
                "effective_request": effective_request.model_dump(mode="json"),
                "feature_store": {
                    "status": feature_store.status(),
                    "online_features": online_features,
                },
                "llm_profile": llm_profile,
                "llm_client": llm_client.status(),
            },
            confidence=0.9 if llm_profile.get("recommendation_hint") else 0.7,
        )

    def _build_llm_profile(
        self,
        behavior_profile: UserProfile,
        online_features: dict[str, Any],
    ) -> dict[str, Any]:
        llm_profile = llm_client.chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_message=self._build_llm_message(behavior_profile, online_features),
            default={
                "segments": [],
                "intent_summary": "LLM不可用，默认画像",
                "recommendation_hint": "",
                "price_sensitivity": "medium",
                "rfm_interpretation": "",
            },
        )
        if isinstance(llm_profile, dict):
            return llm_profile
        return {
            "segments": [],
            "intent_summary": "LLM unavailable, fallback profile",
            "recommendation_hint": "",
            "price_sensitivity": "medium",
            "rfm_interpretation": "",
        }

    def _build_llm_message(
        self,
        profile: UserProfile,
        online_features: dict[str, Any],
    ) -> str:
        parts = [
            "## 用户长期偏好（SQLite累计统计）",
            f"- 偏好类目：{', '.join(profile.preferred_categories) or '无'}",
            f"- 偏好品牌：{', '.join(profile.liked_brands) or '无'}",
            f"- 偏好标签：{', '.join(profile.preferred_tags) or '无'}",
            f"- 累计行为次数：{profile.event_count}",
            f"- 加购商品数：{len(profile.cart_items)}",
            f"- 点踩商品数：{len(profile.disliked_products)}",
            "",
            "## 用户实时行为（Redis时间窗）",
            f"- 最近1小时浏览：{online_features.get('view_count_1h', 0)}次",
            f"- 最近24小时浏览：{online_features.get('view_count_24h', 0)}次",
            f"- 最近24小时点赞：{online_features.get('like_count_24h', 0)}次",
            f"- 最近24小时点踩：{online_features.get('dislike_count_24h', 0)}次",
            f"- 最近7天加购：{online_features.get('add_to_cart_count_7d', 0)}次",
            f"- 最近浏览类目：{', '.join(online_features.get('recent_categories', [])) or '无'}",
            f"- 最近兴趣品牌：{', '.join(online_features.get('recent_brands', [])) or '无'}",
            f"- 最近兴趣标签：{', '.join(online_features.get('recent_tags', [])) or '无'}",
            "",
            "## RFM得分",
            f"- Recency（最近活跃度）: {online_features.get('rfm', {}).get('recency', 0)}",
            f"- Frequency（行为频次）: {online_features.get('rfm', {}).get('frequency', 0)}",
            f"- Monetary（消费能力）: {online_features.get('rfm', {}).get('monetary', 0)}",
        ]
        return "\n".join(parts)

    def _fallback(self, latency_ms: float, exc: Exception, **kwargs: Any) -> AgentResult:
        request: RecommendRequest = kwargs.get("request", RecommendRequest(user_id="unknown"))
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            data={
                "profile": {"user_id": request.user_id, "event_count": 0},
                "effective_request": request.model_dump(mode="json"),
                "feature_store": {
                    "status": feature_store.status(),
                    "online_features": {},
                },
                "llm_profile": {"segments": [], "intent_summary": "Agent执行失败"},
            },
            confidence=0.3,
        )
