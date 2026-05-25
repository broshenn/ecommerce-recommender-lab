# Step 11: LLM 用户画像 Agent

## 这一阶段解决什么问题

Step 10 之前，`UserProfileAgent` 只会把 SQLite 历史行为和 Redis 实时窗口整理成结构化字段，例如偏好类目、品牌、标签、最近浏览、点踩商品。

Step 11 开始接入 LLM，但不直接让 LLM 推荐商品，而是先让它做一件更稳定的事：

```text
SQLite 长期画像 + Redis 实时行为窗口
  -> UserProfileAgent
  -> LLM 分析
  -> llm_profile
```

这样后续的商品重排、营销文案、推荐解释都可以消费同一份 LLM 画像摘要。

## 新增能力

```text
app/services/llm_client.py
```

统一封装 OpenAI-compatible Chat API，支持这些环境变量：

```env
LLM_API_KEY=
LLM_API_BASE=
LLM_MODEL=
LLM_TIMEOUT_SECONDS=15

DEEPSEEK_API_KEY=
DEEPSEEK_API_BASE=
DEEPSEEK_MODEL=

QWEN_API_KEY=
QWEN_API_BASE=
QWEN_MODEL=
```

优先级是：

```text
LLM_* > DEEPSEEK_* > QWEN/DASHSCOPE_*
```

如果没有配置 key，或者 `openai` 包不可用，LLM 调用会返回默认画像，不影响推荐流程。

## UserProfileAgent 输出变化

原有字段保持不变：

```json
{
  "profile": {},
  "effective_request": {},
  "feature_store": {}
}
```

新增：

```json
{
  "llm_profile": {
    "segments": [],
    "intent_summary": "LLM不可用，默认画像",
    "recommendation_hint": "",
    "price_sensitivity": "medium",
    "rfm_interpretation": ""
  },
  "llm_client": {
    "available": false,
    "base_url": "...",
    "model": "...",
    "last_error": "..."
  }
}
```

如果 LLM 可用，`llm_profile` 会变成类似：

```json
{
  "segments": ["active", "price_sensitive"],
  "intent_summary": "近期频繁浏览手机配件，对中低价位保护壳和办公配件兴趣较高",
  "recommendation_hint": "优先推荐手机配件、办公用品，价格控制在中低区间",
  "price_sensitivity": "medium",
  "rfm_interpretation": "近期行为较活跃，但加购金额中等"
}
```

## Supervisor 怎么使用

`SupervisorOrchestrator` 会读取：

```text
agent_results.user_profile.data.llm_profile.recommendation_hint
```

然后写入：

```text
effective_request.context["llm_hint"]
```

这一步只是把 LLM 画像摘要传给下游，当前 `ProductRecAgent` 和 `MarketingCopyAgent` 还没有正式消费它。这个入口是为 Step 12 做准备。

## 验证方式

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
D:\anaconda\envs\py3.10\python.exe -m compileall app tests
```

已验证：

```text
16 passed
compileall 通过
```

## 推荐阅读顺序

```text
app/services/llm_client.py
app/agents/user_profile_agent.py
app/orchestrator/supervisor.py
tests/test_recommender.py
```
