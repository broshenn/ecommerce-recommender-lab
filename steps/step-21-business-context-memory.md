# Step 21：业务上下文和记忆

## 目标

本步骤把对话 Agent 的记忆从“会话状态可保存”推进到“短期上下文可执行、长期偏好可冷启动”。核心原则：

- 当前会话显式目标优先。
- 长期记忆只在当前槽位为空时补位。
- 闲聊不污染购物状态。
- 明确反馈既更新短期状态，也写入长期行为数据。

## 短期状态

`ConversationState` 继续维护当前会话里的业务上下文：

- `shopping_goal`
- `budget_min / budget_max`
- `preferred_categories`
- `liked_brands`
- `preferred_tags`
- `disliked_products`
- `rejected_reasons`
- `last_recommended_product_ids`
- `active_product_refs`
- `recent_intents`

状态规则：

- 新商品目标覆盖旧目标，例如手机切到电脑时不继承手机类目。
- 补充预算、品牌、用途时继承当前目标。
- 负反馈继承当前推荐列表，通过 `active_product_refs` 解析“第一个/第二个/这款”。
- 闲聊、元问题、时间问题不写入购物偏好。

## 长期记忆

`MemoryService` 新增：

- `user_memory_summary(user_id)`：从 `user_memory_facts` 汇总长期事实。
- `record_memory_facts(...)` 去重写入，避免每轮重复插入同一偏好。

推荐请求构造时会读取长期记忆：

```text
当前 state 有明确类目/品牌/标签/预算 -> 使用当前 state
当前 state 对应字段为空 -> 使用 user_memory_facts 里的长期偏好补位
```

这样新会话可以根据历史偏好冷启动，但用户明确说“想要电脑”时，不会被历史“手机保护壳”偏好带偏。

## 反馈记忆

`FeedbackTool` 增强：

- `dislike`：写 `/api/v1/events` 同款行为事件，并把商品加入当前会话 `disliked_products`。
- `like / purchase`：写行为事件，并把商品类目、品牌、标签合并进当前会话偏好。
- 后续推荐请求会带上 `disliked_products`，原推荐链路继续负责降权和过滤。

## 可观测性

当长期记忆存在时，聊天 trace 会增加：

```json
{
  "step": "memory",
  "source": "sqlite_user_memory_facts",
  "summary": {
    "preferred_categories": ["手机"],
    "liked_brands": ["Bastmei"],
    "preferred_tags": ["保护壳", "手机配件"]
  }
}
```

业务工具 trace 仍然展示具体工具调用，例如：

```text
intent:recommend_products -> memory -> PreferenceUpdateTool -> RecommendGraphTool -> dialogue -> done
```

## 验证

- `python -m pytest tests/test_recommender.py -q`
- `python scripts/evaluate_chat_agent.py`

测试覆盖：

- 负反馈解析商品指代，并写入 `disliked_products` 与行为事件。
- 长期记忆可冷启动新会话推荐。
- 明确目标切换不会被长期记忆覆盖。
- 闲聊仍不触发推荐、不污染购物状态。
