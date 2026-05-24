from __future__ import annotations

from typing import Any

from app.inventory import is_available
from app.models import AgentResult, Product

from app.agents.base_agent import BaseAgent


class InventoryAgent(BaseAgent):
    """Check availability and surface inventory alerts."""

    def __init__(self):
        super().__init__(name="inventory", timeout=3.0)

    def _execute(self, **kwargs: Any) -> AgentResult:
        products: list[Product] = kwargs.get("products", [])
        available_ids: list[str] = []
        low_stock_alerts: list[dict[str, Any]] = []

        for product in products:
            if not is_available(product):
                continue
            available_ids.append(product.product_id)
            if product.stock <= 100:
                low_stock_alerts.append(
                    {
                        "product_id": product.product_id,
                        "name": product.name,
                        "stock": product.stock,
                    }
                )

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "available_ids": available_ids,
                "low_stock_alerts": low_stock_alerts,
                "checked_count": len(products),
                "available_count": len(available_ids),
            },
            confidence=0.95,
        )

    def _fallback(self, latency_ms: float, exc: Exception, **kwargs: Any) -> AgentResult:
        products: list[Product] = kwargs.get("products", [])
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            data={
                "available_ids": [product.product_id for product in products],
                "low_stock_alerts": [],
                "fallback": "assume_available",
            },
            confidence=0.4,
        )
