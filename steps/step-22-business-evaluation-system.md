# Step 22：业务测评体系

## 目标

本步骤把对话 Agent 的评测从“能跑几个 case”升级为更完整的业务测评体系。评测不只看有没有返回商品，还要看：

- 意图是否识别正确
- 槽位是否抽取完整
- 多轮状态是否一致
- 第一个/第二个/这款是否解析成功
- 业务工具是否按预期调用
- 闲聊是否误触推荐链路
- 推荐商品是否满足预算和库存约束
- 回复是否包含无依据承诺

## 改动范围

### 1. 扩展评测数据

`data/chat_eval_cases.jsonl` 新增：

- `scenario`：场景标签，用于分场景汇总。
- `expected.tools`：期望出现的业务工具。
- 新增长期记忆和闲聊状态保护 case。

当前覆盖 13 个场景：

- 单轮推荐
- 多轮补槽
- 目标切换
- 预算范围和品牌
- 最低预算
- 指代反馈
- 比较商品
- 解释推荐
- 商品信息问答
- 闲聊兜底
- 偏好记忆
- 长期记忆冷启动
- 推荐后闲聊不刷新商品

### 2. 扩展评测脚本

`scripts/evaluate_chat_agent.py` 增强：

- 输出 JSON 报告：`reports/chat_agent_eval_latest.json`
- 输出 Markdown 报告：`reports/chat_agent_eval_latest.md`
- 新增 `tool_success_rate`
- 新增 `no_recommendation_guard_rate`
- 新增 `scenario_summary`
- 新增 `failures`
- 记录每个 case 的 `trace_tools`、`missing_tools`、`tool_errors`

### 3. 更新测试

`tests/test_recommender.py` 增加断言：

- case 数量至少 12
- `tool_success_rate` 存在并达到 0.9
- `no_recommendation_guard_rate` 存在
- `scenario_summary` 包含 `smalltalk_fallback` 和 `long_term_memory`
- 当前评测无失败 case

## 指标说明

| 指标 | 含义 | 合格线 |
|------|------|--------|
| `intent_macro_f1` | 多轮意图序列识别 F1 | `>= 0.85` |
| `slot_f1` | 期望槽位抽取 F1 | `>= 0.80` |
| `memory_consistency_rate` | 最终状态是否保留期望偏好 | `>= 0.90` |
| `product_ref_resolution_rate` | 指代是否能解析到商品 | `>= 0.85` |
| `task_success_rate` | 任务约束是否整体满足 | `>= 0.80` |
| `tool_success_rate` | 期望业务工具是否被调用且无错误 | `>= 0.90` |
| `budget_compliance_rate` | 推荐商品是否符合预算 | `>= 0.95` |
| `inventory_compliance_rate` | 推荐商品是否有库存 | `= 1.00` |
| `avg_latency_ms` | 平均端到端耗时 | `<= 1500` |
| `unsupported_claim_rate` | 是否出现无依据承诺 | `<= 0.02` |

## 运行方式

```powershell
python scripts\evaluate_chat_agent.py
```

输出：

```text
reports/chat_agent_eval_latest.json
reports/chat_agent_eval_latest.md
```

注意：不要和 `pytest` 并行运行 chat eval。测试 fixture 会清空 SQLite/Redis 运行状态，并行执行会干扰多轮评测。

## 当前结果

当前 13 个 case 全部通过：

```text
intent_macro_f1 = 1.0
slot_f1 = 1.0
memory_consistency_rate = 1.0
product_ref_resolution_rate = 1.0
task_success_rate = 1.0
tool_success_rate = 1.0
budget_compliance_rate = 1.0
inventory_compliance_rate = 1.0
unsupported_claim_rate = 0.0
```

## 面试表达

可以这样讲：

> 我没有只用“返回了商品”来判断对话 Agent 是否可用，而是做了端到端 eval。每条样本是一段多轮对话，评测会同时检查意图序列、槽位状态、业务工具调用、商品指代、预算和库存约束，以及是否出现无依据承诺。报告同时输出 JSON 和 Markdown，JSON 方便自动化检查，Markdown 方便面试或项目复盘展示。

## 下一步

Step 23 再做 Query Understanding 模型对比：

- 规则 baseline
- LLM classifier
- BERT/DistilBERT classifier

这个对比实验不影响主功能，只用于回答成本、延迟、准确率和线上入口选型问题。
