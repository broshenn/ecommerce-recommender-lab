# Step 18：稳定混合前端和对话入口

## 目标

把项目从“纯结构化推荐页面”升级为稳定的混合交互页面：

- 保留原来的结构化推荐能力。
- 在推荐页面上方加入自然语言对话框。
- 对话输入经过 IntentAgent 识别后，回填结构化字段并刷新下方商品列表。
- 核心推荐链路仍复用原 LangGraph，不重写推荐 pipeline。

## 已完成能力

### 1. 保留结构化推荐页面

左侧继续保留原项目字段：

- 用户 ID
- 推荐数量
- 偏好类目
- 偏好品牌
- 偏好标签
- 预算范围

右侧继续保留：

- LLM 用户画像
- 推荐策略摘要
- 商品卡片
- 行为按钮

行为按钮继续写入 `/api/v1/events`，当前统一使用 `view / like / dislike / purchase`。

### 2. 新增自然语言对话框

页面上方新增对话框。用户可以输入：

- `我想要手机`
- `想要电脑`
- `我想买个200元以内的通勤耳机`
- `第二个太贵了，换便宜点`

前端调用 `/api/v1/chat/stream`，通过 SSE 接收：

- state
- token
- products
- trace
- done
- error

### 3. 对话结果回填结构化字段

IntentAgent 输出的 slots 会同步回填左侧结构化字段：

- `preferred_categories`
- `liked_brands`
- `preferred_tags`
- `budget_min`
- `budget_max`

这样面试演示时可以清楚看到：自然语言输入不是直接“瞎生成结果”，而是被解析为业务字段，再驱动推荐链路。

### 4. 修复目标切换

同一个 session 内，明确的新商品目标会覆盖旧目标：

```text
手机 -> 手机商品
电脑 -> 电子数码 / 电脑配件
耳机 -> 电子数码 / 耳机
```

不会再出现“用户说电脑和耳机，下面仍然推荐手机壳”的问题。

补充预算、用途、负反馈时仍然继承当前上下文。

### 5. 保留闲聊兜底

以下问题不会误触推荐：

- `你好`
- `你是什么 agent`
- `你是什么模型`
- `今天星期几`
- `你能说画面吗`

## 涉及模块

- 前端：`app/static/index.html`
- 对话入口：`app/main.py`
- 意图识别：`app/agents/intent_agent.py`
- 对话编排：`app/orchestrator/chat.py`
- 记忆服务：`app/services/memory.py`
- 数据结构：`app/models.py`

## 验收标准

- 打开首页后，结构化推荐页面仍可用。
- 点击“生成推荐”仍走原 LangGraph 推荐链路。
- 输入“想要电脑”后，左侧字段切到 `电子数码 / 电脑配件`，下方商品变为电子数码商品。
- 输入“200元以内通勤耳机”后，预算变为 `200`，标签包含 `耳机 / 通勤`，下方商品变为耳机相关商品。
- 点击喜欢、不喜欢、购买按钮不报错，并能继续写入用户行为。
- 闲聊不返回商品列表。

## 测试

重点回归测试：

```bash
pytest tests/test_recommender.py::test_chat_endpoint_returns_conversational_recommendation
pytest tests/test_recommender.py::test_chat_stream_endpoint_emits_sse_events
pytest tests/test_recommender.py::test_chat_goal_switching_replaces_previous_preferences
pytest tests/test_recommender.py::test_chat_memory_resolves_product_reference_and_feedback
pytest tests/test_recommender.py::test_chat_meta_questions_do_not_trigger_recommendations
```

完整测试：

```bash
pytest
```

## 下一步

进入 Step 19：业务意图识别。

重点继续增强：

- 预算表达：`100-300`、`至少500`。
- 更多品类和用途映射。
- 品牌识别。
- 闲聊和业务意图边界。
- 规则 fallback 和 LLM intent classifier 的一致性。
