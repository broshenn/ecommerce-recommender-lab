from __future__ import annotations

from typing import Any

from app.models import AgentResult, MarketingCopy, Product, UserProfile

from app.agents.base_agent import BaseAgent


class MarketingCopyAgent(BaseAgent):
    """Generate deterministic template copy before the LLM step is introduced."""

    def __init__(self):
        super().__init__(name="marketing_copy", timeout=3.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        products: list[Product] = kwargs.get("products", [])
        profile: UserProfile | None = kwargs.get("profile")

        copies = [
            MarketingCopy(
                product_id=product.product_id,
                text=self._copy_for_product(product, profile),
            )
            for product in products
        ]

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "copies": [copy.model_dump(mode="json") for copy in copies],
                "template": "rule_based",
                "copy_count": len(copies),
            },
            confidence=0.75,
        )

    def _copy_for_product(self, product: Product, profile: UserProfile | None) -> str:
        if profile and product.category in profile.preferred_categories:
            return f"根据你的浏览偏好，为你优先推荐 {product.name}，库存和价格都已为你校验。"
        if product.stock <= 100:
            return f"{product.name} 当前库存紧张，适合尽快决策。"
        return f"{product.name} 为你精选，兼顾品质、价格与实用性。"
