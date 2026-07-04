# Step 19：业务意图识别

## 目标

把对话入口前面的 IntentAgent 做稳：第一版不训练 BERT，不依赖 LLM，先用规则覆盖电商导购高频表达，并保留可选 LLM 路径。

核心目标：

- 能把自然语言解析为业务字段。
- 能区分推荐、补偏好、比较、解释、商品问答、反馈和闲聊。
- 规则 fallback 在无 LLM 环境下也能完成 demo。

## 已完成能力

### 1. 固定意图集合

当前支持：

- `recommend_products`
- `refine_preferences`
- `compare_products`
- `explain_recommendation`
- `record_feedback`
- `ask_product`
- `smalltalk`

### 2. 业务 slots 抽取

支持字段：

- `budget_min`
- `budget_max`
- `preferred_categories`
- `liked_brands`
- `preferred_tags`
- `rejected_reasons`
- `event_type`

预算表达：

- `200以内`
- `不超过300`
- `至少500`
- `100到300`
- `100-300`
- `100~300`

品类和用途：

- 手机 -> `手机 / 手机配件`
- 电脑、笔记本 -> `电子数码 / 电脑配件`
- 键盘 -> `电子数码 / 电脑配件 / 键盘`
- 耳机、耳麦 -> `电子数码 / 耳机`
- 摄像头、相机 -> `电子数码 / 摄像头`
- 保护壳、保护膜、数据线、防水、通勤、办公、游戏等标签

品牌：

- 从 catalog 中匹配真实品牌，例如 `SAMSUNG`、`Microsoft`、`Sharp`、`Bastmei`。

### 3. 业务意图区分

规则优先级：

1. 元问题和闲聊先兜底，例如“你好”“你是什么模型”“今天星期几”。
2. 比较：`比较 / 对比 / 哪个更好 / 区别`。
3. 解释：`为什么 / 原因 / 推荐理由`。
4. 商品问答：`库存 / 有货 / 参数 / 评分 / 价格 / 多少钱`。
5. 推荐：`推荐 / 找 / 想买 / 想要 / 买个 / 适合`。
6. 反馈：`太贵 / 不喜欢 / 换便宜点 / 购买 / 下单`。
7. 只有 slots 时作为 `refine_preferences`。

### 4. 反馈和指代

反馈：

- 太贵、便宜 -> `rejected_reasons=["too_expensive"]`
- 不喜欢、不要 -> `event_type="dislike"`
- 喜欢 -> `event_type="like"`
- 购买、买了、下单、加入购物车 -> `event_type="purchase"`

指代：

- 第一个、第一款、1号
- 第二个、第二款、2号
- 第三个、第三款、3号
- 这款、这个、刚才那个

## 测试

新增单测覆盖：

```bash
pytest tests/test_recommender.py::test_intent_agent_extracts_business_slots_and_budget_ranges
pytest tests/test_recommender.py::test_intent_agent_handles_min_budget_product_info_and_compare
pytest tests/test_recommender.py::test_intent_agent_handles_feedback_events
```

扩展 chat eval：

- `budget_range_brand_earphones`
- `min_budget_computer_accessory`

完整验证：

```bash
pytest
python scripts/evaluate_chat_agent.py
```

## 验收标准

- `pytest` 全部通过。
- chat eval 所有阈值通过。
- `100到300元的防水Samsung耳机` 能抽出：
  - `budget_min=100`
  - `budget_max=300`
  - `liked_brands=["SAMSUNG"]`
  - `preferred_categories=["电子数码"]`
  - `preferred_tags` 包含 `耳机`、`防水`
- `至少500元的电脑配件` 能抽出：
  - `budget_min=500`
  - `preferred_categories=["电子数码"]`
  - `preferred_tags=["电脑配件"]`
- 闲聊仍不触发推荐。

## 下一步

进入 Step 20：业务 Tool Layer。

下一步重点不是继续堆规则，而是把当前的业务能力封装成轻量工具：

- `RecommendGraphTool`
- `PreferenceUpdateTool`
- `FeedbackTool`
- `CompareProductTool`
- `ExplainRecommendationTool`
- `ProductInfoTool`
- `SmalltalkTool`

并让前端 trace 从节点级别升级为业务工具调用级别。
