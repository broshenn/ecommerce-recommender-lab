# Step 7：Supervisor + 4 Agent 骨架

## 这个 Step 做了什么

把原来的直接推荐函数升级成更接近原始项目的多 Agent 编排结构。

之前：

```text
/api/v1/recommend -> recommend_products() -> 直接过滤、打分、排序
```

现在：

```text
/api/v1/recommend
  -> SupervisorOrchestrator
  -> UserProfileAgent
  -> ProductRecAgent
  -> InventoryAgent
  -> MarketingCopyAgent
  -> RecommendResponse
```

## 新增模块

```text
app/agents/base_agent.py
app/agents/user_profile_agent.py
app/agents/product_rec_agent.py
app/agents/inventory_agent.py
app/agents/marketing_copy_agent.py
app/orchestrator/supervisor.py
```

## 四个 Agent 职责

```text
UserProfileAgent
  从 SQLite 行为和请求参数生成有效画像。

ProductRecAgent
  先使用当前规则分数进行候选排序，后续可替换成 Chroma 召回。

InventoryAgent
  检查商品是否可推荐，返回可用商品和低库存预警。

MarketingCopyAgent
  先用模板文案生成推荐理由，后续再接 LLM。
```

## 为什么先不接 LLM

这一阶段只搭架构骨架，避免同时引入 LLM、Chroma、Redis 后看不清主线。

后续应该按这个顺序继续：

```text
Step 8：A/B 测试和 Metrics
Step 9：Chroma 商品向量召回
Step 10：Redis 实时特征窗口
Step 11：LLM 营销文案
```

## 重点阅读文件

```text
app/models.py
app/agents/base_agent.py
app/agents/user_profile_agent.py
app/agents/product_rec_agent.py
app/agents/inventory_agent.py
app/agents/marketing_copy_agent.py
app/orchestrator/supervisor.py
app/recommender.py
tests/test_recommender.py
```
