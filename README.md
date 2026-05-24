# 电商推荐系统重构版

这是一个从 0 开始、按步骤重构的电商推荐学习项目。目标不是一次性做成大而全系统，而是每次只加一个核心能力，让你能看懂推荐系统如何从“规则推荐”逐步长成“多 Agent 推荐系统”。

## 当前阶段：Step 8 A/B 测试 + Metrics

当前已经实现：

- Amazon Reviews 2023 商品元数据样本，当前保留 1000 条商品
- 商品目录从 `data/products_amazon_sample.csv` 读取
- 当前用户画像：类目、品牌、标签、预算、最近浏览、不喜欢商品、加购商品
- 当前用户行为事件：`view`、`like`、`dislike`、`add_to_cart`
- SQLite 持久化用户行为事件
- Supervisor 编排推荐流程
- 4 个非 LLM Agent：用户画像、商品推荐、库存决策、营销文案
- 库存过滤、低库存提示、限购提示
- Vue 3 前端页面
- A/B 测试稳定分桶
- Metrics 指标统计：Agent 调用次数、成功率、平均耗时、业务事件次数
- 基础单元测试

暂时还没有实现：

- Chroma 向量召回
- Redis 实时特征窗口
- LLM 营销文案生成
- RAG 商品问答/推荐解释
- 训练出来的排序模型

## 推荐流程

```text
POST /api/v1/recommend
  -> SupervisorOrchestrator
  -> Phase 1: UserProfileAgent + ProductRecAgent recall
  -> Phase 2: ProductRecAgent rerank + InventoryAgent
  -> Phase 3: MarketingCopyAgent + ABTestEngine
  -> MetricsCollector 记录 Agent 指标
  -> 返回商品列表、实验组、营销文案、Agent 结果
```

## 重点文件

```text
app/main.py                       FastAPI 路由
app/models.py                     请求、响应、商品、画像、实验模型
app/catalog.py                    商品 CSV 读取
app/database.py                   SQLite 初始化
app/behavior.py                   当前用户行为记录和画像聚合
app/personalization.py            当前规则打分
app/inventory.py                  库存状态、低库存、限购规则
app/agents/base_agent.py          Agent 统一计时、错误捕获、降级
app/agents/user_profile_agent.py  用户画像 Agent
app/agents/product_rec_agent.py   商品召回/重排 Agent
app/agents/inventory_agent.py     库存决策 Agent
app/agents/marketing_copy_agent.py 营销文案 Agent
app/orchestrator/supervisor.py    Supervisor 编排器
app/services/ab_test.py           A/B 测试稳定分桶
app/services/metrics.py           内存指标统计
app/static/index.html             Vue 3 前端页面
tests/test_recommender.py         回归测试
```

## 运行

```powershell
cd D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step
D:\anaconda\envs\py3.10\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

打开：

```text
http://127.0.0.1:8010/
```

## 测试

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
```

## 常用接口

健康检查：

```http
GET /health
```

商品列表：

```http
GET /api/v1/products
```

生成推荐：

```http
POST /api/v1/recommend
```

示例请求：

```json
{
  "user_id": "u001",
  "num_items": 3,
  "preferred_categories": ["手机"],
  "liked_brands": ["Sharp"],
  "preferred_tags": ["手机配件"],
  "budget_min": 50,
  "budget_max": 500
}
```

查看实验配置和用户分桶：

```http
GET /api/v1/experiments?user_id=u001
```

查看指标：

```http
GET /api/v1/metrics
```

记录用户行为：

```http
POST /api/v1/events
```

```json
{
  "user_id": "u001",
  "product_id": "B07ZPSG8P5",
  "event_type": "like"
}
```

查看用户行为：

```http
GET /api/v1/users/u001/events
```

查看用户画像：

```http
GET /api/v1/users/u001/profile
```

## 下一步

下一步建议做 Step 9：把 Chroma 接进 `ProductRecAgent`，让商品召回从“全量规则排序”升级为“向量召回 + 规则重排”。
