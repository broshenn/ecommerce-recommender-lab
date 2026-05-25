from __future__ import annotations

import concurrent.futures
from typing import Any

from app.agents import (
    InventoryAgent,
    MarketingCopyAgent,
    ProductRecAgent,
    UserProfileAgent,
)
from app.catalog import list_products
from app.inventory import enrich_inventory
from app.models import (
    AgentResult,
    ExperimentAssignment,
    MarketingCopy,
    Product,
    RecommendRequest,
    RecommendResponse,
    UserProfile,
)
from app.services import ab_test_engine, metrics_collector
from app.services.ab_test import ABTestEngine
from app.services.metrics import MetricsCollector


class SupervisorOrchestrator:
    """Coordinate the four-agent recommendation skeleton."""

    def __init__(
        self,
        ab_engine: ABTestEngine | None = None,
        metrics: MetricsCollector | None = None,
    ):
        self.user_profile_agent = UserProfileAgent()
        self.product_rec_agent = ProductRecAgent()
        self.inventory_agent = InventoryAgent()
        self.marketing_copy_agent = MarketingCopyAgent()
        self.ab_engine = ab_engine or ab_test_engine
        self.metrics = metrics or metrics_collector
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        experiment = self.ab_engine.assign(request.user_id)
        self.metrics.record_business_event("recommend_request")

        all_products = list_products()
        products_by_id = {product.product_id: product for product in all_products}

        # Phase 1: profile and first-pass recall can run independently.
        profile_future = self.executor.submit(
            self.user_profile_agent.run,
            request=request,
        )
        recall_future = self.executor.submit(
            self.product_rec_agent.run,
            request=request,
            products=all_products,
            limit=max(request.num_items * 20, 50),
            mode="recall",
        )

        profile_result = self._result_or_fallback(profile_future, self.user_profile_agent)
        recall_result = self._result_or_fallback(recall_future, self.product_rec_agent)

        effective_request = self._effective_request(profile_result, request)
        profile = self._profile(profile_result, request.user_id)
        recalled_products = self._products_from_ids(
            recall_result.data.get("product_ids", []),
            products_by_id,
        )
        if not recalled_products:
            recalled_products = all_products
        recalled_products = self._expand_candidates_with_profile(
            recalled_products,
            all_products,
            effective_request,
        )

        # Phase 2: rerank candidates while checking inventory on the same pool.
        rerank_future = self.executor.submit(
            self.product_rec_agent.run,
            request=effective_request,
            products=recalled_products,
            limit=request.num_items * 2,
            mode="rerank",
        )
        inventory_future = self.executor.submit(
            self.inventory_agent.run,
            products=recalled_products,
        )

        rerank_result = self._result_or_fallback(rerank_future, self.product_rec_agent)
        inventory_result = self._result_or_fallback(inventory_future, self.inventory_agent)

        ranked_products = self._products_from_ids(
            rerank_result.data.get("product_ids", []),
            products_by_id,
        )
        available_ids = set(inventory_result.data.get("available_ids", []))
        scores = rerank_result.data.get("scores", {})

        final_products = [
            product for product in ranked_products
            if product.product_id in available_ids
        ]
        if len(final_products) < request.num_items:
            existing = {product.product_id for product in final_products}
            final_products.extend(
                product for product in ranked_products
                if product.product_id not in existing
            )
        final_products = final_products[: request.num_items]

        recommended = [
            enrich_inventory(
                product,
                recommendation_score=scores.get(product.product_id, {}).get("score", 0),
                recommendation_reason=scores.get(product.product_id, {}).get(
                    "reason",
                    "Supervisor 推荐",
                ),
            )
            for product in final_products
        ]

        # Phase 3: copy generation depends on the final product list.
        copy_result = self.marketing_copy_agent.run(
            products=final_products,
            profile=profile,
        )
        marketing_copies = [
            MarketingCopy.model_validate(copy)
            for copy in copy_result.data.get("copies", [])
        ]

        agent_results = {
            "user_profile": profile_result,
            "product_recall": recall_result,
            "product_rerank": rerank_result,
            "inventory": inventory_result,
            "marketing_copy": copy_result,
        }
        self._record_agent_metrics(agent_results)
        self.metrics.record_business_event("recommend_success")

        return RecommendResponse(
            user_id=request.user_id,
            scene=request.scene,
            products=recommended,
            strategy="supervisor_agents+vector_recall+inventory_filter+ab_test",
            reason="Supervisor 编排用户画像、Chroma 向量召回、规则重排、库存决策、营销文案和 A/B 分桶后生成推荐。",
            experiment_group=experiment.group,
            experiment=experiment,
            marketing_copies=marketing_copies,
            agent_results=agent_results,
        )

    def assign_experiment(self, user_id: str) -> ExperimentAssignment:
        return self.ab_engine.assign(user_id)

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.snapshot()

    def _result_or_fallback(
        self,
        future: concurrent.futures.Future,
        agent,
    ) -> AgentResult:
        try:
            return future.result(timeout=agent.timeout + 0.5)
        except Exception as exc:
            return agent._fallback(0.0, exc)

    def _effective_request(
        self,
        result: AgentResult,
        fallback_request: RecommendRequest,
    ) -> RecommendRequest:
        raw = result.data.get("effective_request")
        if not raw:
            return fallback_request
        return RecommendRequest.model_validate(raw)

    def _profile(self, result: AgentResult, user_id: str) -> UserProfile:
        raw = result.data.get("profile") or {"user_id": user_id}
        return UserProfile.model_validate(raw)

    def _products_from_ids(
        self,
        product_ids: list[str],
        products_by_id: dict[str, Product],
    ) -> list[Product]:
        return [
            products_by_id[product_id]
            for product_id in product_ids
            if product_id in products_by_id
        ]

    def _expand_candidates_with_profile(
        self,
        recalled_products: list[Product],
        all_products: list[Product],
        request: RecommendRequest,
    ) -> list[Product]:
        existing_ids = {product.product_id for product in recalled_products}
        expanded = list(recalled_products)
        for product in all_products:
            if product.product_id in existing_ids:
                continue
            if self._matches_profile(product, request):
                expanded.append(product)
                existing_ids.add(product.product_id)
        return expanded

    def _matches_profile(self, product: Product, request: RecommendRequest) -> bool:
        if request.preferred_categories and product.category in request.preferred_categories:
            return True
        if request.liked_brands and product.brand in request.liked_brands:
            return True
        if request.preferred_tags and set(product.tags) & set(request.preferred_tags):
            return True
        return False

    def _record_agent_metrics(self, agent_results: dict[str, AgentResult]) -> None:
        for metric_key, result in agent_results.items():
            self.metrics.record_agent_result(metric_key, result)
