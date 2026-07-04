# Step 20：业务 Tool Layer

## 目标

本步骤把对话入口升级为轻量业务工具调用链路。项目仍然不是通用 coding agent，不做文件系统、代码编辑或复杂 planner；这里只借鉴 coding agent 的 router、tool calling、observation、trace 思路，让电商导购过程更可解释。

## 新增结构

新增 `app/tools/business_tools.py`：

- `ToolRouter`：根据业务意图选择工具。
- `ToolObservation`：统一记录工具名、输入摘要、输出摘要、耗时、错误。
- `RecommendGraphTool`：继续调用原有 `recommend_with_graph`，不重写推荐链路。
- `PreferenceUpdateTool`：把已合并到会话状态的偏好以 observation 形式输出。
- `FeedbackTool`：处理喜欢、不喜欢、购买等反馈，并复用现有行为事件写入。
- `CompareProductTool`：基于当前会话推荐商品做轻量比较。
- `ExplainRecommendationTool`：解释当前商品为什么被推荐。
- `ProductInfoTool`：回答价格、库存、评分等商品事实。
- `SmalltalkTool`：闲聊和元问题交给 `DialogueAgent` 兜底回复。

## 对话执行链路

```text
用户输入
-> IntentAgent
-> MemoryService 读取/更新状态
-> ToolRouter 选择业务工具
-> BusinessTool 执行并返回 observation
-> DialogueAgent 汇总回复
-> MemoryService 写回状态和消息
-> 前端 trace 展示工具调用过程
```

## Observation 格式

```json
{
  "step": "tool",
  "tool_name": "RecommendGraphTool",
  "success": true,
  "input_summary": {},
  "output_summary": {},
  "latency_ms": 12.3,
  "error": null
}
```

## 验证

- `pytest`
- `python scripts/evaluate_chat_agent.py`

测试覆盖：

- 推荐请求会调用 `PreferenceUpdateTool` 和 `RecommendGraphTool`
- 负反馈会调用 `FeedbackTool` 并继续触发推荐
- 比较、解释、商品信息分别调用对应业务工具
- 闲聊只走 `SmalltalkTool`，不触发推荐
