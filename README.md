# 电商推荐系统重构版

这是一个从 0 开始、按步骤重构的电商推荐学习项目。目标不是一次性做完所有能力，而是每次只加一个核心功能，让你能看懂推荐系统如何从“规则推荐”逐步长成“多 Agent 推荐系统”。

## 当前阶段：Step 9 Chroma 向量召回

当前已经实现：

- Amazon Reviews 2023 商品元数据样本，当前保留 1000 条商品
- 商品目录从 `data/products_amazon_sample.csv` 读取
- 当前用户画像：类目、品牌、标签、预算、最近浏览、不喜欢商品、加购商品
- 当前用户行为事件：`view`、`like`、`dislike`、`add_to_cart`
- SQLite 持久化用户行为事件
- Supervisor 编排推荐流程
- 4 个非 LLM Agent：用户画像、商品推荐、库存决策、营销文案
- Chroma 商品向量召回
- 千问 / DashScope `text-embedding-v4` 可选向量化
- 本地 hash embedding 兜底，没配 key 也能跑通
- 规则重排、库存过滤、低库存提示、限购提示
- A/B 测试稳定分桶
- Metrics 指标统计
- Vue 3 前端页面
- 基础单元测试

暂时还没有实现：

- Redis 实时特征窗口
- LLM 营销文案生成
- RAG 商品问答/推荐解释
- 训练出来的排序模型

## 推荐流程

```text
POST /api/v1/recommend
  -> SupervisorOrchestrator
  -> Phase 1: UserProfileAgent + ProductRecAgent vector recall
  -> Phase 2: ProductRecAgent rule rerank + InventoryAgent
  -> Phase 3: MarketingCopyAgent + ABTestEngine
  -> MetricsCollector 记录 Agent 指标
  -> 返回商品列表、实验组、营销文案、Agent 结果
```

## 重点文件

```text
app/main.py                        FastAPI 路由
app/models.py                      请求、响应、商品、画像、实验模型
app/catalog.py                     商品 CSV 读取
app/database.py                    SQLite 初始化
app/behavior.py                    当前用户行为记录和画像聚合
app/personalization.py             当前规则打分
app/inventory.py                   库存状态、低库存、限购规则
app/agents/base_agent.py           Agent 统一计时、错误捕获、降级
app/agents/user_profile_agent.py   用户画像 Agent
app/agents/product_rec_agent.py    商品向量召回 / 规则重排 Agent
app/agents/inventory_agent.py      库存决策 Agent
app/agents/marketing_copy_agent.py 营销文案 Agent
app/orchestrator/supervisor.py     Supervisor 编排器
app/services/vector_store.py       Chroma + 千问 embedding / 本地 embedding
app/services/ab_test.py            A/B 测试稳定分桶
app/services/metrics.py            内存指标统计
app/static/index.html              Vue 3 前端页面
tests/test_recommender.py          回归测试
CODE_UPDATES.md                    每个阶段的代码改动记录
```

## 千问 API 配置

当前项目根目录已经有 `.env` 模板：

```text
D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step\.env
```

把里面的占位符替换成你的 DashScope Key：

```env
PRODUCT_VECTOR_EMBEDDING_PROVIDER=local
DASHSCOPE_API_KEY=sk-your-dashscope-api-key
```

要真正调用千问，把它改成：

```env
PRODUCT_VECTOR_EMBEDDING_PROVIDER=qwen
DASHSCOPE_API_KEY=你的真实DashScopeKey
```

如果你想先不调用外部 API，改成：

```env
PRODUCT_VECTOR_EMBEDDING_PROVIDER=local
```

`.env` 不会提交到 GitHub，提交用的是 `.env.example`。

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

生成推荐：

```http
POST /api/v1/recommend
```

查看向量库状态：

```http
GET /api/v1/vector-store
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

## 下一步

下一步建议做 Step 10：Redis 实时特征窗口。SQLite 继续保存长期行为，Redis 用来保存最近 1 小时、24 小时、7 天的实时行为特征。
