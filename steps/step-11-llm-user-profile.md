# Step 11: LLM Client + UserProfileAgent 改造 — 执行手册

## 背景

这是 `ecommerce-rebuild-step-by-step` 项目，从零逐步重建一个多 Agent 电商推荐系统。当前到 Step 10，已经有：

- 4 个 Agent（用户画像/商品推荐/库存/文案）+ Supervisor 编排 + ThreadPoolExecutor 并行
- ChromaDB 向量召回、Redis 特征存储（实时行为窗口+RFM）、A/B 分桶、Metrics 采集
- 所有 Agent 都是规则驱动（if-else），没有 LLM 调用
- 项目根目录：`D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step`
- 环境变量已在 `.env` 配置好 DeepSeek

### 本步目标

1. 创建 `app/services/llm_client.py`：统一 LLM 调用模块
2. 改造 `app/agents/user_profile_agent.py`：用 LLM 分析用户数据，输出结构化画像
3. 微调 `app/orchestrator/supervisor.py`：消费 LLM 生成的画像摘要
4. 现有所有测试保持通过

### 关键约束

- **全程同步代码**，不用 async/await —— 项目目前全是同步的
- LLM SDK 用 `openai` 包（OpenAI 兼容接口），不用 LangChain
- `.env` 中 DeepSeek 配置已就绪：`DEEPSEEK_API_KEY`、`DEEPSEEK_API_BASE=https://api.deepseek.com`、`DEEPSEEK_MODEL=deepseek-v4-pro[1m]`
- **LLM 不可用时必须降级**走原规则逻辑，不能崩
- `UserProfileAgent` 的输出接口不能变——`AgentResult.data` 里必须继续包含 `profile`、`effective_request`、`feature_store` 三个 key，否则 supervisor 会挂

---

## 1. 先读这些文件（了解上下文）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 1 | `app/services/vector_store.py` 最后 30 行 | 参考已有的 `get_product_vector_store` 单例模式 + env 读取方式 |
| 2 | `app/services/feature_store.py:107-136` | `get_user_features()` 返回的数据结构（行为计数 + 类目/品牌 + RFM） |
| 3 | `app/behavior.py:60-67` | `build_user_profile()` 返回 `UserProfile` 对象 |
| 4 | `app/behavior.py:93-107` | `merge_behavior_profile()` 的逻辑 |
| 5 | `app/models.py:80-88` | `UserProfile` 字段定义 |
| 6 | `app/agents/user_profile_agent.py` | 当前实现（等下要改） |
| 7 | `app/orchestrator/supervisor.py:64-68` | `_effective_request` 和 `_profile` 怎么从 AgentResult 取值 |
| 8 | `tests/test_recommender.py` | 已有测试，改造后不能挂 |

---

## 2. 新增文件：`app/services/llm_client.py`

### 设计要求

- 读 `.env` 中的 `DEEPSEEK_API_KEY`、`DEEPSEEK_API_BASE`、`DEEPSEEK_MODEL`
- 用 `openai.OpenAI` 构造客户端（DeepSeek 兼容 OpenAI API）
- 暴露两个方法：`chat()` 和 `chat_json()`
- 模块级单例 `llm_client = LLMClient()`

### 完整代码

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat API. Supports DeepSeek."""

    def __init__(self):
        load_dotenv(BASE_DIR / ".env")
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self._client: OpenAI | None = None

    @property
    def _openai(self) -> OpenAI | None:
        if not self.api_key or "your-" in self.api_key:
            return None
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Send a chat request. Returns response text or None on failure."""
        client = self._openai
        if client is None:
            return None

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception:
            return None

    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a chat request and parse JSON response. Returns default on failure."""
        text = self.chat(system_prompt, user_message + "\n只输出JSON，不要其他内容。")
        if text is None:
            return default or {}

        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return default or {}


llm_client = LLMClient()
```

### 要点

- `_openai` property 做了 API key 占位符检查（`"your-" in api_key`），没填 key 时返回 None，所有调用降级
- `chat()` 失败全部 catch Exception 返回 None，不抛异常
- `chat_json()` 在 user_message 后自动追加 "只输出JSON" 指令，并处理 ```json ... ``` 包裹

---

## 3. 改造文件：`app/agents/user_profile_agent.py`

### 改造目标

`_execute()` 在采集完数据后，调 LLM 生成一份 "用户画像分析"，附在 `data["llm_profile"]` 里。原有的 `profile`、`effective_request`、`feature_store` 三个 key 保持不变。

### SYSTEM_PROMPT —— 定义 LLM 角色和输出格式

写一个模块级常量（参考原项目 `user_profile_agent.py` 的 SYSTEM_PROMPT 风格）：

```python
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
```

### 改造后的 `_execute()`

```python
def _execute(self, **kwargs: Any) -> AgentResult:
    request: RecommendRequest = kwargs["request"]
    behavior_profile = build_user_profile(request.user_id)
    effective_request = merge_behavior_profile(request)
    online_features = feature_store.get_user_features(request.user_id)

    # === 新增：LLM 画像分析 ===
    llm_profile = {}
    from app.services.llm_client import llm_client

    user_message = self._build_llm_message(behavior_profile, online_features)
    llm_profile = llm_client.chat_json(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        default={"segments": [], "intent_summary": "LLM不可用，默认画像"},
    )

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
            "llm_profile": llm_profile,          # ← 新增
        },
        confidence=0.9 if llm_profile.get("intent_summary") else 0.7,
    )
```

### `_build_llm_message()` —— 把数据拼成 LLM 能理解的 prompt

```python
def _build_llm_message(self, profile: UserProfile, online_features: dict) -> str:
    parts = [
        f"## 用户长期偏好（SQLite累计统计）",
        f"- 偏好类目：{', '.join(profile.preferred_categories) or '无'}",
        f"- 偏好品牌：{', '.join(profile.liked_brands) or '无'}",
        f"- 偏好标签：{', '.join(profile.preferred_tags) or '无'}",
        f"- 累计行为次数：{profile.event_count}",
        f"- 加购商品数：{len(profile.cart_items)}",
        f"- 点踩商品数：{len(profile.disliked_products)}",
        "",
        f"## 用户实时行为（Redis时间窗）",
        f"- 最近1小时浏览：{online_features.get('view_count_1h', 0)}次",
        f"- 最近24小时浏览：{online_features.get('view_count_24h', 0)}次",
        f"- 最近24小时点赞：{online_features.get('like_count_24h', 0)}次",
        f"- 最近24小时点踩：{online_features.get('dislike_count_24h', 0)}次",
        f"- 最近7天加购：{online_features.get('add_to_cart_count_7d', 0)}次",
        f"- 最近浏览类目：{', '.join(online_features.get('recent_categories', [])) or '无'}",
        f"- 最近兴趣品牌：{', '.join(online_features.get('recent_brands', [])) or '无'}",
        f"- 最近兴趣标签：{', '.join(online_features.get('recent_tags', [])) or '无'}",
        "",
        f"## RFM得分",
        f"- Recency（最近活跃度）: {online_features.get('rfm', {}).get('recency', 0)}",
        f"- Frequency（行为频次）: {online_features.get('rfm', {}).get('frequency', 0)}",
        f"- Monetary（消费能力）: {online_features.get('rfm', {}).get('monetary', 0)}",
    ]
    return "\n".join(parts)
```

### `_fallback()` —— 增加 `llm_profile` 空字典

```python
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
            "llm_profile": {"segments": [], "intent_summary": "Agent执行失败"},
        },
        confidence=0.3,
    )
```

### 需要新增的 import

```python
from app.models import AgentResult, RecommendRequest, UserProfile
from app.behavior import build_user_profile, merge_behavior_profile
from app.services import feature_store
```

注意：`UserProfile` 之前没导入（以前只存 dict），现在 `_build_llm_message` 需要类型注解。agent 文件顶部加 `from app.models import UserProfile`，`_execute` 里 `build_user_profile` 返回值本来就是 `UserProfile`，不需要改。

---

## 4. 微调文件：`app/orchestrator/supervisor.py`

### 改造内容

原 `_effective_request()` 和 `_profile()` 只读 `data["effective_request"]` 和 `data["profile"]`，**不需要改**。

但在 `recommend()` 方法里，可以把 `llm_profile` 中的 `recommendation_hint` 传给下游使用。最简单的做法——在 Phase 2 的 rerank 调用里，把 hint 塞进 context：

```python
# supervisor.py 行 67 后增加：
profile_result = self._result_or_fallback(profile_future, self.user_profile_agent)
recall_result = self._result_or_fallback(recall_future, self.product_rec_agent)

effective_request = self._effective_request(profile_result, request)
profile = self._profile(profile_result, request.user_id)

# === 新增：提取 LLM 画像摘要，传给下游 ===
llm_profile = profile_result.data.get("llm_profile", {})
if llm_profile.get("recommendation_hint"):
    effective_request.context["llm_hint"] = llm_profile["recommendation_hint"]
```

这样 `effective_request.context.llm_hint` 就能被 `ProductRecAgent` 和 `MarketingCopyAgent` 在后续 step 中消费。

**这一处改动不破坏任何现有逻辑**——`context` 本来就是个自由 dict，下游没读它就只是透传。

---

## 5. 修改 `requirements.txt`

增加一行：
```
openai>=1.0.0
```

---

## 6. 修改 `app/services/__init__.py`

增加 `LLMClient` 和 `llm_client`：
```python
from app.services.llm_client import LLMClient, llm_client

__all__ = [
    "ABTestEngine",
    "LLMClient",
    "RedisFeatureStore",
    "MetricsCollector",
    "ab_test_engine",
    "feature_store",
    "llm_client",
    "metrics_collector",
]
```

---

## 7. 验证方式

### 7.1 启动服务

```powershell
D:\anaconda\envs\py3.10\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

### 7.2 测试推荐接口

```powershell
curl -X POST http://127.0.0.1:8010/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test001","num_items":3,"preferred_categories":["耳机"]}'
```

检查返回的 `agent_results.user_profile.data.llm_profile` 里是否包含 `intent_summary`、`segments`、`recommendation_hint` 等字段。

### 7.3 测试 LLM 降级

注释掉 `.env` 中的 `DEEPSEEK_API_KEY`，重启服务，再次调推荐接口。确认：
- 不报错
- `llm_profile.intent_summary` 为 `"LLM不可用，默认画像"`
- 推荐结果仍然正常返回（走规则 fallback）

### 7.4 跑测试

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest tests/test_recommender.py -q
```

所有已有测试必须通过。如果现有测试因为新增 `llm_profile` 字段而导致断言失败，修正对应的断言（比如检查 `llm_profile` 为 dict 而不是检查具体值）。

---

## 8. 做完之后的 git diff 清单

```
requirements.txt                    +1 行
.env.example                       无需改（key 已在 Step 11 加好）
app/services/llm_client.py         新增文件 (~80 行)
app/services/__init__.py            +3 行
app/agents/user_profile_agent.py    ~60 行改造（加 SYSTEM_PROMPT + _build_llm_message + 改 _execute + 改 _fallback）
app/orchestrator/supervisor.py      +3 行（提取 llm_profile hint）
tests/test_recommender.py           可能微调（如果断言过不了）
```
