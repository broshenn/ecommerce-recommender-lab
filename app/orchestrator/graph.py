from __future__ import annotations

"""LangGraph 版本推荐编排。

整体仍是画像、召回/重排、库存、文案三阶段；相比普通 Supervisor，
这里额外在库存过滤后商品不足时走 expand 分支扩大召回。
"""

import concurrent.futures
import time
import uuid
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

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
    RecommendedProduct,
    UserProfile,
)
from app.services import ab_test_engine, metrics_collector

GRAPH_WORKERS = 4
RECALL_MULTIPLIER = 50
RECALL_FLOOR = 200
EXPANDED_RECALL_MULTIPLIER = 100
EXPANDED_RECALL_FLOOR = 400


class PipelineState(TypedDict, total=False):
    user_id: str
    scene: str
    num_items: int
    preferred_categories: list[str]
    liked_brands: list[str]
    preferred_tags: list[str]
    budget_min: float | None
    budget_max: float | None
    recent_views: list[str]
    disliked_products: list[str]
    context: dict[str, Any]

    request_id: str
    experiment: dict[str, Any]
    experiment_group: str
    total_latency_ms: float
    _start_time: float
    _expanded: bool

    all_products: list[dict[str, Any]]
    all_products_by_id: dict[str, dict[str, Any]]
    profile: dict[str, Any]
    effective_request: dict[str, Any]
    llm_profile: dict[str, Any]
    recalled_product_ids: list[str]
    reranked_product_ids: list[str]
    rerank_scores: dict[str, Any]
    available_ids: list[str]
    final_products: list[dict[str, Any]]
    marketing_copies: list[dict[str, Any]]
    agent_results: dict[str, dict[str, Any]]
    route_trace: list[str]


_user_profile_agent = UserProfileAgent()
_product_rec_agent = ProductRecAgent()
_inventory_agent = InventoryAgent()
_marketing_copy_agent = MarketingCopyAgent()
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=GRAPH_WORKERS)


def recommend_with_graph(request: RecommendRequest) -> RecommendResponse:
    """运行状态图，并把图里的 dict 状态转换回 API 响应模型。"""
    state = _state_from_request(request)
    result = rec_graph.invoke(state)

    experiment = ExperimentAssignment.model_validate(result["experiment"])
    final_products = [
        RecommendedProduct.model_validate(product)
        for product in result.get("final_products", [])
    ]
    marketing_copies = [
        MarketingCopy.model_validate(copy)
        for copy in result.get("marketing_copies", [])
    ]
    agent_results = {
        key: AgentResult.model_validate(value)
        for key, value in result.get("agent_results", {}).items()
    }

    return RecommendResponse(
        user_id=result["user_id"],
        scene=result.get("scene", "homepage"),
        products=final_products,
        strategy="langgraph_agents+vector_recall+inventory_filter+ab_test",
        reason=(
            "LangGraph 状态图编排用户画像、召回、重排、库存、文案、A/B 曝光，"
            "并在商品不足时自动扩大召回。"
        ),
        experiment_group=experiment.group,
        experiment=experiment,
        marketing_copies=marketing_copies,
        agent_results=agent_results,
    )


def _state_from_request(request: RecommendRequest) -> PipelineState:
    return {
        "user_id": request.user_id,
        "scene": request.scene,
        "num_items": request.num_items,
        "preferred_categories": request.preferred_categories,
        "liked_brands": request.liked_brands,
        "preferred_tags": request.preferred_tags,
        "budget_min": request.budget_min,
        "budget_max": request.budget_max,
        "recent_views": request.recent_views,
        "disliked_products": request.disliked_products,
        "context": request.context,
    }


def _init_node(state: PipelineState) -> PipelineState:
    state["request_id"] = str(uuid.uuid4())
    state["_start_time"] = time.perf_counter()
    state["_expanded"] = False
    state["agent_results"] = {}
    state["route_trace"] = ["init"]

    experiment = _assign_experiment(state)
    state["experiment"] = experiment.model_dump(mode="json")
    state["experiment_group"] = experiment.group

    all_products = list_products()
    serialized_products = [product.model_dump(mode="json") for product in all_products]
    state["all_products"] = serialized_products
    state["all_products_by_id"] = {
        product["product_id"]: product
        for product in serialized_products
    }

    metrics_collector.record_business_event("recommend_request")
    return state


def _assign_experiment(state: PipelineState) -> ExperimentAssignment:
    force_group = str(state.get("context", {}).get("force_experiment_group", "")).strip()
    if force_group:
        experiment = ab_test_engine.experiments.get(ab_test_engine.DEFAULT_EXPERIMENT_ID, {})
        for variant in experiment.get("variants", []):
            if variant.get("group") == force_group:
                return ExperimentAssignment(
                    experiment_id=ab_test_engine.DEFAULT_EXPERIMENT_ID,
                    group=force_group,
                    reason="forced by request context",
                    config=variant.get("config", {}),
                )
    return ab_test_engine.assign(state["user_id"])


def _phase1_node(state: PipelineState) -> PipelineState:
    state["route_trace"].append("phase1")
    products = _products_from_state(state)
    request = _request_from_state(state)

    # 用户画像和向量召回互不依赖，可以并行执行。
    profile_future = _executor.submit(
        _user_profile_agent.run,
        request=request,
        experiment_group=state["experiment_group"],
    )
    recall_future = _executor.submit(
        _product_rec_agent.run,
        request=request,
        products=products,
        limit=_recall_limit(state),
        mode="recall",
    )

    profile_result = _result_or_fallback(profile_future, _user_profile_agent)
    recall_result = _result_or_fallback(recall_future, _product_rec_agent)

    state["agent_results"]["user_profile"] = profile_result.model_dump(mode="json")
    state["agent_results"]["product_recall"] = recall_result.model_dump(mode="json")
    state["profile"] = profile_result.data.get("profile") or {"user_id": state["user_id"]}
    state["effective_request"] = (
        profile_result.data.get("effective_request")
        or request.model_dump(mode="json")
    )
    state["llm_profile"] = profile_result.data.get("llm_profile", {})

    recall_ids = recall_result.data.get("product_ids", [])
    state["recalled_product_ids"] = recall_ids or list(state["all_products_by_id"].keys())
    return state


def _merge1_node(state: PipelineState) -> PipelineState:
    state["route_trace"].append("merge1")
    effective_request = state.get("effective_request", {})
    llm_profile = state.get("llm_profile", {})
    context = dict(effective_request.get("context", {}))

    if state["experiment_group"] != "control" and llm_profile.get("recommendation_hint"):
        context["llm_hint"] = llm_profile["recommendation_hint"]
        effective_request["context"] = context

    state["effective_request"] = effective_request
    if "user_profile" in state["agent_results"]:
        state["agent_results"]["user_profile"]["data"]["effective_request"] = effective_request
    return state


def _phase2_node(state: PipelineState) -> PipelineState:
    state["route_trace"].append("phase2")
    recalled_products = _products_from_ids(state, state.get("recalled_product_ids", []))
    request = RecommendRequest.model_validate(state["effective_request"])

    # 重排和库存检查都只依赖候选集，可以并行执行。
    rerank_future = _executor.submit(
        _product_rec_agent.run,
        request=request,
        products=recalled_products,
        limit=state["num_items"] * 2,
        mode="rerank",
    )
    inventory_future = _executor.submit(
        _inventory_agent.run,
        products=recalled_products,
    )

    rerank_result = _result_or_fallback(rerank_future, _product_rec_agent)
    inventory_result = _result_or_fallback(inventory_future, _inventory_agent)

    state["agent_results"]["product_rerank"] = rerank_result.model_dump(mode="json")
    state["agent_results"]["inventory"] = inventory_result.model_dump(mode="json")
    state["reranked_product_ids"] = rerank_result.data.get("product_ids", [])
    state["rerank_scores"] = rerank_result.data.get("scores", {})
    state["available_ids"] = inventory_result.data.get("available_ids", [])
    return state


def _merge2_node(state: PipelineState) -> PipelineState:
    state["route_trace"].append("merge2")
    available_ids = set(state.get("available_ids", []))
    ranked_ids = state.get("reranked_product_ids", [])
    selected_ids = [
        product_id
        for product_id in ranked_ids
        if product_id in available_ids
    ]

    # 先触发 expand 扩召回，仍不足时再退回到原排序里的不可用商品兜底。
    if state.get("_expanded") and len(selected_ids) < state["num_items"]:
        for product_id in ranked_ids:
            if product_id not in selected_ids:
                selected_ids.append(product_id)
            if len(selected_ids) >= state["num_items"]:
                break

    state["final_products"] = [
        _enriched_product_dump(state, product_id)
        for product_id in selected_ids[: state["num_items"]]
    ]
    return state


def _expand_node(state: PipelineState) -> PipelineState:
    state["route_trace"].append("expand")
    state["_expanded"] = True
    products = _products_from_state(state)
    request = RecommendRequest.model_validate(state["effective_request"])

    recall_result = _product_rec_agent.run(
        request=request,
        products=products,
        limit=_recall_limit(state),
        mode="recall",
    )
    state["agent_results"]["product_recall_expanded"] = recall_result.model_dump(mode="json")
    state["recalled_product_ids"] = (
        recall_result.data.get("product_ids", [])
        or list(state["all_products_by_id"].keys())
    )
    return _phase2_node(state)


def _phase3_node(state: PipelineState) -> PipelineState:
    state["route_trace"].append("phase3")
    products = [
        Product.model_validate(product)
        for product in state.get("final_products", [])
    ]
    profile = UserProfile.model_validate(state.get("profile") or {"user_id": state["user_id"]})

    copy_result = _marketing_copy_agent.run(
        products=products,
        profile=profile,
        llm_profile=state.get("llm_profile", {}),
        experiment_group=state["experiment_group"],
    )
    state["agent_results"]["marketing_copy"] = copy_result.model_dump(mode="json")
    state["marketing_copies"] = [
        MarketingCopy.model_validate(copy).model_dump(mode="json")
        for copy in copy_result.data.get("copies", [])
    ]
    return state


def _aggregate_node(state: PipelineState) -> PipelineState:
    state["route_trace"].append("aggregate")
    state["total_latency_ms"] = (
        time.perf_counter() - state.get("_start_time", time.perf_counter())
    ) * 1000

    for metric_key, result_json in state["agent_results"].items():
        metrics_collector.record_agent_result(
            metric_key,
            AgentResult.model_validate(result_json),
        )
    metrics_collector.record_business_event("recommend_success")

    experiment = ExperimentAssignment.model_validate(state["experiment"])
    ab_test_engine.record_exposure(
        experiment_id=experiment.experiment_id,
        group=experiment.group,
        user_id=state["user_id"],
    )
    return state


def _should_expand(state: PipelineState) -> Literal["expand", "phase3"]:
    if len(state.get("final_products", [])) < state["num_items"] and not state.get("_expanded"):
        return "expand"
    return "phase3"


def _request_from_state(state: PipelineState) -> RecommendRequest:
    return RecommendRequest(
        user_id=state["user_id"],
        scene=state.get("scene", "homepage"),
        num_items=state.get("num_items", 3),
        preferred_categories=state.get("preferred_categories", []),
        liked_brands=state.get("liked_brands", []),
        preferred_tags=state.get("preferred_tags", []),
        budget_min=state.get("budget_min"),
        budget_max=state.get("budget_max"),
        recent_views=state.get("recent_views", []),
        disliked_products=state.get("disliked_products", []),
        context=state.get("context", {}),
    )


def _products_from_state(state: PipelineState) -> list[Product]:
    return [Product.model_validate(product) for product in state["all_products"]]


def _products_from_ids(state: PipelineState, product_ids: list[str]) -> list[Product]:
    products_by_id = state["all_products_by_id"]
    return [
        Product.model_validate(products_by_id[product_id])
        for product_id in product_ids
        if product_id in products_by_id
    ]


def _enriched_product_dump(state: PipelineState, product_id: str) -> dict[str, Any]:
    product = Product.model_validate(state["all_products_by_id"][product_id])
    score_info = state.get("rerank_scores", {}).get(product_id, {})
    return enrich_inventory(
        product,
        recommendation_score=score_info.get("score", 0),
        recommendation_reason=score_info.get("reason", "LangGraph recommendation"),
    ).model_dump(mode="json")


def _recall_limit(state: PipelineState) -> int:
    product_count = len(state.get("all_products", []))
    multiplier = EXPANDED_RECALL_MULTIPLIER if state.get("_expanded") else RECALL_MULTIPLIER
    floor = EXPANDED_RECALL_FLOOR if state.get("_expanded") else RECALL_FLOOR
    return min(product_count, max(state.get("num_items", 3) * multiplier, floor))


def _result_or_fallback(
    future: concurrent.futures.Future,
    agent: Any,
) -> AgentResult:
    try:
        return future.result(timeout=agent.timeout + 0.5)
    except Exception as exc:
        return agent._fallback(0.0, exc)


def build_graph():
    """启动时编译一次推荐状态图。"""
    graph = StateGraph(PipelineState)
    graph.add_node("init", _init_node)
    graph.add_node("phase1", _phase1_node)
    graph.add_node("merge1", _merge1_node)
    graph.add_node("phase2", _phase2_node)
    graph.add_node("merge2", _merge2_node)
    graph.add_node("expand", _expand_node)
    graph.add_node("phase3", _phase3_node)
    graph.add_node("aggregate", _aggregate_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "phase1")
    graph.add_edge("phase1", "merge1")
    graph.add_edge("merge1", "phase2")
    graph.add_edge("phase2", "merge2")
    graph.add_conditional_edges(
        "merge2",
        _should_expand,
        {
            "expand": "expand",
            "phase3": "phase3",
        },
    )
    graph.add_edge("expand", "merge2")
    graph.add_edge("phase3", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


rec_graph = build_graph()
