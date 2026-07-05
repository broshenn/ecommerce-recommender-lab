from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal["view", "like", "dislike", "purchase"]
ChatIntent = Literal[
    "recommend_products",
    "refine_preferences",
    "compare_products",
    "explain_recommendation",
    "record_feedback",
    "ask_product",
    "smalltalk",
]
ChatIntentMode = Literal["rule", "bert", "llm"]


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    brand: str
    stock: int
    tags: list[str] = Field(default_factory=list)
    source_name: str | None = None
    source_category: str | None = None
    source_dataset: str = "local"
    image_url: str | None = None
    rating: float | None = None
    rating_count: int | None = None


class RecommendedProduct(Product):
    stock_status: str = "normal"
    stock_message: str = "库存充足"
    purchase_limit: int | None = None
    recommendation_score: float = 0
    recommendation_reason: str = "基础推荐"


class RecommendRequest(BaseModel):
    user_id: str
    scene: str = "homepage"
    num_items: int = 3
    preferred_categories: list[str] = Field(default_factory=list)
    liked_brands: list[str] = Field(default_factory=list)
    preferred_tags: list[str] = Field(default_factory=list)
    budget_min: float | None = None
    budget_max: float | None = None
    recent_views: list[str] = Field(default_factory=list)
    disliked_products: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class ExperimentAssignment(BaseModel):
    experiment_id: str
    group: str
    reason: str
    config: dict[str, Any] = Field(default_factory=dict)


class ExperimentOutcome(BaseModel):
    experiment_id: str = "recommendation_strategy_v1"
    group: str
    user_id: str
    success: bool
    product_id: str | None = None


class RecommendResponse(BaseModel):
    user_id: str
    scene: str
    products: list[RecommendedProduct]
    strategy: str
    reason: str
    experiment_group: str | None = None
    experiment: ExperimentAssignment | None = None
    marketing_copies: list[MarketingCopy] = Field(default_factory=list)
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)


class UserEventCreate(BaseModel):
    user_id: str
    product_id: str
    event_type: EventType


class UserEvent(UserEventCreate):
    event_id: int
    created_at: datetime


class UserProfile(BaseModel):
    user_id: str
    preferred_categories: list[str] = Field(default_factory=list)
    liked_brands: list[str] = Field(default_factory=list)
    preferred_tags: list[str] = Field(default_factory=list)
    recent_views: list[str] = Field(default_factory=list)
    disliked_products: list[str] = Field(default_factory=list)
    cart_items: list[str] = Field(default_factory=list)
    event_count: int = 0


class MarketingCopy(BaseModel):
    product_id: str
    text: str


class AgentResult(BaseModel):
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class ConversationState(BaseModel):
    session_id: str
    user_id: str
    shopping_goal: str = ""
    budget_min: float | None = None
    budget_max: float | None = None
    preferred_categories: list[str] = Field(default_factory=list)
    liked_brands: list[str] = Field(default_factory=list)
    preferred_tags: list[str] = Field(default_factory=list)
    disliked_products: list[str] = Field(default_factory=list)
    rejected_reasons: list[str] = Field(default_factory=list)
    last_recommended_product_ids: list[str] = Field(default_factory=list)
    active_product_refs: dict[str, str] = Field(default_factory=dict)
    recent_intents: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime | None = None


class IntentResult(BaseModel):
    intent: ChatIntent
    slots: dict[str, Any] = Field(default_factory=dict)
    product_refs: list[str] = Field(default_factory=list)
    needs_recommendation: bool = False
    confidence: float = 0.0
    source: str = "rule"


class ChatRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    message: str
    stream: bool = False
    force_experiment_group: str | None = None
    intent_mode: ChatIntentMode = "rule"


class ChatResponse(BaseModel):
    session_id: str
    intent: ChatIntent
    reply: str
    state: ConversationState
    products: list[RecommendedProduct] = Field(default_factory=list)
    marketing_copies: list[MarketingCopy] = Field(default_factory=list)
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
