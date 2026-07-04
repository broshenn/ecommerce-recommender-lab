---
name: 业务型 Conversational Commerce Agent 路线图
description: 保留原电商推荐系统和 LangGraph 推荐链路，在前面增加对话入口、意图识别、上下文记忆、业务工具调用和闲聊兜底。
updated: 2026-07-04
---

# 业务型 Conversational Commerce Agent 路线图

## Summary

项目定位统一为 **业务型 Conversational Commerce Agent**：保留原电商推荐系统和 LangGraph 推荐链路，在前面增加对话入口、意图识别、上下文记忆、业务工具调用和闲聊兜底。

这里的 “coding agent 风格” 只借鉴 **tool calling、router、trace、observation、step-by-step 执行**，不做通用代码编辑、不做文件系统操作、不做完整 Claude Code 复刻。

核心目标：

- 用户可以用结构化字段推荐，也可以用自然语言推荐。
- 自然语言会被解析成业务字段：品类、预算、品牌、用途、反馈、指代。
- Agent 根据业务意图调用对应工具。
- 核心推荐仍走原 LangGraph。
- 前端能展示业务工具调用轨迹，便于面试说明。

## Roadmap

### Step 18：稳定混合前端

保留原项目结构化推荐页面：

- 用户 ID
- 推荐数量
- 类目
- 品牌
- 标签
- 预算
- 商品卡片
- 行为按钮

新增对话框：

- 对话输入识别出的 slots 回填结构化字段。
- 下方商品列表用对话返回结果刷新。
- 目标切换要正确：
  - 手机 -> 手机商品
  - 电脑 -> 电子数码/电脑配件
  - 耳机 -> 电子数码/耳机

这一阶段目标是 demo 稳定，不扩展复杂 RAG。

### Step 19：业务意图识别

固定业务意图集合：

- `recommend_products`：推荐商品
- `refine_preferences`：补充预算、品牌、用途
- `compare_products`：比较商品
- `explain_recommendation`：解释推荐原因
- `record_feedback`：喜欢、不喜欢、购买、太贵
- `ask_product`：问商品价格、库存、评分
- `smalltalk`：闲聊、元问题

第一版以规则为主，可选 LLM。

重点覆盖：

- 品类：手机、电脑、耳机、办公、游戏、保护壳、键盘、摄像头
- 预算：200 以内、100-300、至少 500
- 品牌：Sony、Sharp、Microsoft、Samsung 等
- 用途：通勤、办公、游戏、防水、轻便
- 反馈：太贵、不喜欢、换便宜点、喜欢、购买
- 指代：第一个、第二个、这款、刚才那个

输出结构：

- `intent`
- `slots`
- `product_refs`
- `needs_recommendation`
- `confidence`
- `source`

### Step 20：业务 Tool Layer

做轻量业务工具层，不做通用 coding agent。

工具只围绕电商业务：

```text
RecommendGraphTool
PreferenceUpdateTool
FeedbackTool
CompareProductTool
ExplainRecommendationTool
ProductInfoTool
SmalltalkTool
```

`ToolRouter` 负责根据 intent 选择工具。

每个工具返回统一 observation：

- `tool_name`
- `success`
- `input_summary`
- `output_summary`
- `latency_ms`
- `error`

对话流程：

```text
用户输入
-> IntentAgent
-> MemoryService 读取状态
-> ToolRouter 选择业务工具
-> 执行业务工具
-> DialogueAgent 汇总回复
-> MemoryService 写回状态
-> 前端展示结果和 trace
```

推荐相关工具必须继续调用原 LangGraph，不重写推荐链路。

### Step 21：业务上下文和记忆

短期会话状态：

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

- 新商品目标覆盖旧目标。
- 补充预算、品牌、用途继承当前目标。
- 负反馈继承当前推荐列表。
- 闲聊不污染购物状态。

长期行为：

- 喜欢、不喜欢、购买继续写 `/api/v1/events`。
- SQLite 构建长期用户画像。
- Redis 做实时特征缓存。

目标：多轮对话像真实导购，不像一次性搜索框。

### Step 22：业务测评体系

继续维护 chat eval。

场景覆盖：

- 单轮推荐
- 多轮补槽
- 目标切换
- 指代理解
- 负反馈
- 比较商品
- 解释推荐
- 商品信息问答
- 闲聊兜底
- LLM 不可用 fallback

指标：

- `intent_macro_f1`
- `slot_f1`
- `memory_consistency_rate`
- `product_ref_resolution_rate`
- `task_success_rate`
- `budget_compliance_rate`
- `inventory_compliance_rate`
- `avg_latency_ms`
- `fallback_rate`

初始目标：

- `intent_macro_f1 >= 0.85`
- `slot_f1 >= 0.80`
- `memory_consistency_rate >= 0.90`
- `budget_compliance_rate >= 0.95`
- `inventory_compliance_rate = 1.00`

### Step 23：Query Understanding 模型对比

这一步再做模型实验，不影响主功能。

数据来源：

- DeepSeek 合成业务对话样本
- 模板增强
- 人工校验 eval set

对比：

- 规则 baseline
- LLM classifier
- BERT/DistilBERT classifier

目标不是为了炫模型，而是回答业务问题：

- 成本
- 延迟
- 准确率
- 泛化能力
- 是否适合线上高频入口

结论预期：

- 规则适合兜底和 demo。
- LLM 适合理解复杂自然语言。
- BERT 适合高频意图分类，但 slot 抽取仍需要规则或额外模型。

## Deferred

- Product RAG 暂时不做主线。
- 不做通用 coding agent。
- 不做文件系统工具。
- 不做复杂 planner、多 Agent 自主循环。
- 不做自动购买或真实支付动作。
- 后续如果要增强商品事实问答，再单独做 Product RAG。

## Test Plan

后端：

- chat API
- chat stream
- 意图识别
- 目标切换
- 指代理解
- 负反馈
- 闲聊不触发推荐

前端：

- 结构化字段推荐可用
- 对话输入回填字段
- 商品列表刷新
- 行为按钮继续可用
- trace 显示业务工具调用

评测：

- `scripts/evaluate_chat_agent.py`
- 后续新增 query understanding 对比报告

## GitHub Workflow

每完成一个 Step，都必须同步写入 GitHub，保证代码、文档和路线图一致。

推荐流程：

1. 完成当前 Step 的代码、文档和测试。
2. 运行对应测试或评测脚本。
3. 更新相关 step 文档、`AGENTS.md` 或 README。
4. 检查 `git status`，确认只包含当前 Step 相关改动。
5. 提交 commit，commit message 使用清晰的 Step 编号，例如：
   - `Step 18: stabilize hybrid commerce UI`
   - `Step 19: add business intent routing`
   - `Step 20: add business tool layer`
6. 推送到 GitHub：
   - `git push`

约定：

- 每个 Step 至少一个 commit。
- 一个 Step 不混入无关重构。
- 如果 Step 较大，可以拆成多个 commit，但 commit message 必须能看出属于哪个 Step。
- 推送前必须说明测试结果；如果测试未跑或失败，必须在提交说明或交付说明里写清楚。

## Assumptions

- 项目目标是业务型电商 Agent，而不是通用 coding agent。
- “coding agent 思路”只借鉴工具调用和可观测执行过程。
- 原推荐链路继续是核心能力。
- 第一优先级是业务闭环和面试可讲性。
- LLM 可选，规则 fallback 必须稳定。
