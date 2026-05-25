# Step 12b: ProductRecAgent LLM 重排 — 执行手册

## 背景

当前 Step 12a 已完成：`UserProfileAgent` 和 `MarketingCopyAgent` 都接入了 LLM。`supervisor.py` 已经在 `effective_request.context["llm_hint"]` 中传入了 LLM 生成的推荐策略提示。但 `ProductRecAgent._rerank()` 仍然是纯规则打分：

```python
# 现在的 _rerank() (app/agents/product_rec_agent.py:62-98)
scored = [(score_product(product, request), product) for product in products]
# score_product: 类目+40, 品牌+25, 标签+10, 预算+20, 评分×4, dislike-100, recent_views-30
scored.sort(...)
# 取出 top N → 返回
```

**本步目标：`_rerank()` 先尝试 LLM 排序，LLM 不可用时降级走规则排序。**

---

## 1. 先读这些文件

| 顺序 | 文件 | 看什么 |
|------|------|------|
| 1 | `app/agents/product_rec_agent.py` | 当前 `_rerank()` 实现（99 行，等下要改） |
| 2 | `app/personalization.py` | `score_product()` 打分规则（改后作为 fallback 保留） |
| 3 | `app/agents/user_profile_agent.py:8-22` | SYSTEM_PROMPT 风格参考 |
| 4 | `app/agents/marketing_copy_agent.py:98-130` | `_generate_via_llm()` 的 LLM 调用 + fallback 模式（照抄这个思路） |
| 5 | `app/orchestrator/supervisor.py:77-82` | rerank 调用的入参（`request=effective_request`, `mode="rerank"`) |

---

## 2. 改造文件：`app/agents/product_rec_agent.py`

### 设计要点

- `_rerank()` 新增 LLM 路径，**不删原有规则逻辑**
- LLM prompt 传：`llm_hint` + 用户偏好类目/品牌/预算 + 候选商品列表
- LLM 输出格式：`["P001", "P003", "P002", ...]` — 按推荐优先级排序的商品 ID 数组
- LLM 不可用 → 直接走 `score_product()` 规则排序
- 数据格式不变：AgentResult.data 必须继续包含 `product_ids`, `scores`, `candidate_count`, `returned_count`, `mode`, `backend`

### 完整改动

#### 2.1 文件顶部加 import 和 RERANK_PROMPT

```python
from __future__ import annotations

import json
from typing import Any

from app.models import AgentResult, Product, RecommendRequest
from app.personalization import score_product
from app.services.vector_store import VectorRecallUnavailable, get_product_vector_store
from app.services import llm_client

from app.agents.base_agent import BaseAgent


RERANK_PROMPT = """你是电商推荐排序专家。根据用户偏好和商品信息，对候选商品按相关性从高到低重新排序。

排序原则:
1. 优先匹配用户的真实购物意图（参考推荐提示）
2. 类目/品牌偏好匹配的商品排在前面
3. 价格在用户预算范围内优先
4. 适当保持类目多样性，避免全部推同一类目

请输出按推荐优先级排序的商品ID列表（JSON数组）:
["商品ID1", "商品ID2", ...]

只输出JSON数组，不要其他内容。"""
```

#### 2.2 改 `_rerank()`

```python
def _rerank(
    self,
    request: RecommendRequest,
    products: list[Product],
    limit: int,
    backend: str,
) -> AgentResult:
    # === LLM 路径 ===
    if request.context.get("llm_hint"):
        llm_result = self._llm_rerank(request, products, limit, backend)
        if llm_result is not None:
            return llm_result

    # === 规则 fallback（原逻辑） ===
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
            "mode": "rerank",
            "backend": backend,
        },
        confidence=0.85,
    )
```

#### 2.3 新增 `_llm_rerank()`

```python
def _llm_rerank(
    self,
    request: RecommendRequest,
    products: list[Product],
    limit: int,
    backend: str,
) -> AgentResult | None:
    """Try LLM-based reranking. Returns None if unavailable."""
    # 构建商品摘要
    product_lines = []
    for i, product in enumerate(products):
        product_lines.append(
            f"{i+1}. ID:{product.product_id} {product.name} "
            f"类目:{product.category} 价格:¥{product.price} "
            f"品牌:{product.brand} 标签:{','.join(product.tags)}"
        )

    # 构建用户画像摘要
    pref_categories = request.preferred_categories or []
    liked_brands = request.liked_brands or []
    preferred_tags = request.preferred_tags or []
    budget_range = ""
    if request.budget_min is not None or request.budget_max is not None:
        budget_range = f"预算范围: ¥{request.budget_min or 0} - ¥{request.budget_max or '不限'}"
    llm_hint = request.context.get("llm_hint", "")

    user_message = (
        f"## 用户偏好\n"
        f"- 偏好类目: {', '.join(pref_categories) or '无'}\n"
        f"- 偏好品牌: {', '.join(liked_brands) or '无'}\n"
        f"- 偏好标签: {', '.join(preferred_tags) or '无'}\n"
        f"- {budget_range}\n"
        f"- 推荐提示: {llm_hint}\n"
        f"\n"
        f"## 候选商品（{len(products)}件）\n"
        + "\n".join(product_lines)
        + f"\n\n请从中选出最合适的{limit}件，按优先级排序。"
    )

    result = llm_client.chat_json(
        system_prompt=RERANK_PROMPT,
        user_message=user_message,
        default=None,
    )
    if not isinstance(result, list) or len(result) == 0:
        return None

    # 用 LLM 返回的排序结果，匹配商品
    id_to_product = {product.product_id: product for product in products}
    reranked_ids = []
    seen: set[str] = set()
    for raw_id in result:
        pid = str(raw_id).strip()
        if pid in id_to_product and pid not in seen:
            reranked_ids.append(pid)
            seen.add(pid)

    if len(reranked_ids) == 0:
        return None

    # 不足 limit 的用剩余商品补齐
    if len(reranked_ids) < limit:
        for product in products:
            if product.product_id not in seen:
                reranked_ids.append(product.product_id)
                seen.add(product.product_id)
                if len(reranked_ids) >= limit:
                    break

    selected_ids = reranked_ids[:limit]
    return AgentResult(
        agent_name=self.name,
        success=True,
        data={
            "product_ids": selected_ids,
            "scores": {
                pid: {
                    "score": round(1.0 - idx / max(len(selected_ids), 1), 2),
                    "reason": f"LLM 重排序第{idx + 1}位",
                }
                for idx, pid in enumerate(selected_ids)
            },
            "candidate_count": len(products),
            "returned_count": len(selected_ids),
            "mode": "llm_rerank",
            "backend": f"llm+{backend}",
            "llm_client": llm_client.status(),
        },
        confidence=0.85,
    )
```

### 关键逻辑说明

- **触发条件**：LLM 路径只在 `request.context.llm_hint` 存在时才尝试（即 `UserProfileAgent` LLM 给出了推荐策略提示）
- **降级链**：LLM 不可用 → 返回 None → 自动走 `score_product()` 规则
- **LLM 返回脏数据**：`_llm_rerank` 用 `id_to_product` 校验每个返回的 product_id 是否真实存在，去重，不足补全
- **scores 字段**：LLM 重排的 score 按位次递减（第1名=1.0，第N名=1/N），reason 标注"LLM 重排第X位"

---

## 3. 其他文件改动

### 3.1 不改 `personalization.py`

`score_product()` 全部保留，作为 LLM 的 fallback。两个路径并存。

### 3.2 不改 `supervisor.py`

rerank 的调用方式是 `product_rec_agent.run(mode="rerank", request=effective_request, ...)`，`effective_request` 已经有 `context.llm_hint`。`_execute()` 拿到 request 后透传给 `_rerank()`，不需要改 supervisor。

### 3.3 不改 `models.py`

`RecommendRequest.context` 是 `dict[str, Any]`，`llm_hint` 已经可以放在里面。

---

## 4. 验证

### 4.1 不加 API key 跑测试（LLM 降级路径）

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest tests/test_recommender.py -q
```

当前测试 mock 了 LLM（`LLM_API_KEY=""`），所以 `_llm_rerank()` 会返回 None，**必须走 `score_product()` 规则路径**。所有现有测试应该通过。

### 4.2 跑通后加 API key 测试 LLM 路径

确保 `.env` 有真实 `DEEPSEEK_API_KEY`，启动服务：

```powershell
D:\anaconda\envs\py3.10\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

先创建用户画像（让 LLM 生成 `llm_hint`）：

```powershell
curl -X POST http://127.0.0.1:8010/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test01","product_id":"P001","event_type":"like"}'

curl -X POST http://127.0.0.1:8010/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test01","product_id":"P003","event_type":"add_to_cart"}'
```

然后调推荐接口：

```powershell
curl -X POST http://127.0.0.1:8010/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test01","num_items":3,"preferred_categories":["耳机"]}'
```

检查：
- `agent_results.product_rerank.data.mode` 应该为 `"llm_rerank"`
- `agent_results.product_rerank.data.backend` 包含 `"llm+"` 前缀
- `agent_results.product_rerank.data.scores` 的 reason 是 "LLM 重排序第X位"

---

## 5. 改动清单

| 动作 | 文件 | 说明 |
|------|------|------|
| 改造 | `app/agents/product_rec_agent.py` | 加 RERANK_PROMPT + `_rerank()` 尝试 LLM + 新增 `_llm_rerank()` ~70 行 |
| 不改 | `app/personalization.py` | 规则 fallback 完整保留 |
| 不改 | `app/orchestrator/supervisor.py` | 调用方式不变 |
| 不改 | 其他所有文件 | — |

---

## 6. LLM 三 Agent 完成后的完整数据流

```
用户请求 POST /api/v1/recommend
  │
  ├── Phase 1
  │     ├── UserProfileAgent ── LLM → llm_profile (segments/intent/hint/sensitivity/rfm)
  │     └── ProductRecAgent (recall) ── Chroma 向量召回
  │
  ├── Phase 2
  │     ├── ProductRecAgent (rerank) ── LLM 读取 llm_hint + 候选 → 排序     ← 本 Step
  │     └── InventoryAgent ── 库存检查
  │
  └── Phase 3
        └── MarketingCopyAgent ── LLM 读取 segments + intent → 3套个性化文案

所有 LLM 路径失败 → 自动降级走规则逻辑
```
