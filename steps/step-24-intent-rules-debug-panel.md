# Step 24：IntentAgent 规则配置化与 Debug 面板

## 目标

本步骤把原来写死在 `IntentAgent` 里的意图关键词、商品同义词、预算规则和商品指代规则拆到配置文件中，并在前端展示 Query Understanding 的中间结果。

核心目的不是换模型，而是让当前规则 baseline 更像一个可维护的业务系统：

```text
用户输入
-> IntentAgent 读取 intent_rules.json
-> 输出 intent / slots / confidence / rule_debug
-> ToolRouter 调业务工具
-> 前端展示 Query Understanding 过程
```

## 新增文件

```text
app/agents/intent_rules.json
steps/step-24-intent-rules-debug-panel.md
```

## 后端变化

`app/agents/intent_agent.py` 现在会在初始化时读取：

```text
app/agents/intent_rules.json
```

配置内容包括：

```text
intent_markers        意图关键词
product_synonyms      商品词到 category/tag 的映射
generic_tags          通用标签词
budget                预算范围、上限、下限 marker
product_refs          第一个/第二个/这款 等指代词
```

例如：

```json
{
  "电脑": {
    "categories": ["电子数码"],
    "tags": ["电脑配件"]
  },
  "耳机": {
    "categories": ["电子数码"],
    "tags": ["耳机"]
  }
}
```

## Debug 输出

`IntentAgent` 的 `agent_results.intent.data` 中新增：

```json
{
  "rule_debug": {
    "mode": "rule",
    "matched_rules": ["recommend_products"],
    "matched_keywords": {
      "recommend_products": ["想要"]
    },
    "slot_sources": {
      "budget": {
        "budget_max": 200
      },
      "synonyms": ["电脑"]
    },
    "product_refs": []
  },
  "rule_config": "intent_rules.json"
}
```

这样面试时可以直接解释：

```text
为什么识别成 recommend_products？
因为命中了 “想要”。

为什么 category 是 电子数码？
因为 “电脑” 在配置中映射到了 电子数码 + 电脑配件。

为什么预算是 200？
因为命中了 “200块以内” 的 budget_max 规则。
```

## 前端变化

`app/static/index.html` 左侧新增 Query Understanding 面板，展示：

```text
来源
置信度
是否触发推荐
slots
命中规则
```

这让页面不只是返回商品，也能展示 Agent 的理解过程，方便说明：

```text
自然语言 -> 结构化字段 -> 工具调用 -> LangGraph 推荐
```

## 当前策略

默认仍然是规则 baseline：

```text
CHAT_LLM_ENABLED 未开启 -> rule intent
CHAT_LLM_ENABLED=true -> 先 LLM JSON 分类，失败后 fallback 到 rule
```

规则 baseline 的价值：

```text
低延迟
低成本
可解释
可兜底
适合 demo 和高频入口
```

后续如果接 BERT/DistilBERT，建议仍然保留这套规则配置：

```text
BERT 负责 intent 分类
规则继续负责预算、品类、品牌、标签、指代等 slots
LLM 负责低置信度和复杂表达兜底
```

## 验证

```powershell
python -m pytest tests/test_recommender.py -q
```

通过结果：

```text
44 passed
```
