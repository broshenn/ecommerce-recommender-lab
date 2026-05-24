from __future__ import annotations

from typing import Any

from app.models import AgentResult, Product, RecommendRequest
from app.personalization import score_product

from app.agents.base_agent import BaseAgent


class ProductRecAgent(BaseAgent):
    """Recall and rerank products with the current rule-based scorer."""

    def __init__(self):
        super().__init__(name="product_rec", timeout=5.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        request: RecommendRequest = kwargs["request"]
        products: list[Product] = kwargs.get("products", [])
        limit: int = kwargs.get("limit", request.num_items)

        scored = [(score_product(product, request), product) for product in products]
        scored.sort(
            key=lambda item: (
                item[0].value,
                item[1].rating or 0,
                item[1].rating_count or 0,
            ),
            reverse=True,
        )

        selected = scored[:limit]
        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "product_ids": [product.product_id for _, product in selected],
                "scores": {
                    product.product_id: {
                        "score": score.value,
                        "reason": score.reason,
                    }
                    for score, product in selected
                },
                "candidate_count": len(products),
                "returned_count": len(selected),
            },
            confidence=0.85,
        )
