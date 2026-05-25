# Step 12a: LLM 营销文案 Agent

## 这一阶段解决什么问题

Step 11 里，`UserProfileAgent` 已经能产出 LLM 用户画像：

```text
segments
intent_summary
recommendation_hint
price_sensitivity
rfm_interpretation
```

Step 12a 把这份画像交给 `MarketingCopyAgent`，让它不再只输出固定模板，而是根据用户分群和商品信息生成个性化营销文案。

## 推荐流程变化

```text
最终推荐商品
  -> MarketingCopyAgent
  -> 读取 llm_profile
  -> 按用户分群选择 Prompt 模板
  -> 调 LLM 生成商品文案
  -> 广告法敏感词过滤
  -> 返回 marketing_copies
```

LLM 不可用时：

```text
MarketingCopyAgent
  -> rule_fallback
  -> 使用原来的固定模板文案
```

## 用户分群模板

当前支持：

```text
new_user
high_value
price_sensitive
churn_risk
active
category_explorer
brand_loyal
```

每个分群会使用不同的文案风格。例如：

```text
new_user       -> 欢迎、新人优惠感、降低决策门槛
high_value     -> 品质感、尊享感、品牌价值
price_sensitive -> 性价比、促销、省钱
brand_loyal    -> 品牌认同、生态兼容
```

## 合规过滤

LLM 生成文案后，会过滤这些敏感词：

```text
最好
第一
国家级
全球首
绝对
100%
永久
万能
祖传
纯天然
最便宜
全网最低
```

命中后会替换为：

```text
***
```

## 接口返回里怎么看

推荐接口：

```text
POST /api/v1/recommend
```

重点看：

```text
marketing_copies
agent_results.marketing_copy.data.mode
agent_results.marketing_copy.data.template
agent_results.marketing_copy.data.llm_client
```

如果 LLM 成功：

```json
{
  "mode": "llm",
  "template": "llm_new_user"
}
```

如果 LLM 不可用：

```json
{
  "mode": "rule_fallback",
  "template": "rule_based_fallback"
}
```

## 前端变化

商品卡片里新增文案展示区，会显示 `marketing_copies` 对应商品的文案。

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
app/agents/marketing_copy_agent.py
app/orchestrator/supervisor.py
app/services/llm_client.py
app/static/index.html
```
