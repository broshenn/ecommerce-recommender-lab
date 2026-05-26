# Step 14: LangGraph 状态图编排

这一阶段新增了一套 LangGraph 编排器，但保留原来的 `SupervisorOrchestrator` 不动。

## 这一步为什么要做

之前的推荐流程写在 `app/orchestrator/supervisor.py` 里：

```text
Phase 1: UserProfileAgent + ProductRecAgent recall
Phase 2: ProductRecAgent rerank + InventoryAgent
Phase 3: MarketingCopyAgent
```

这种写法直观，但流程结构藏在普通 Python 代码里。后面如果要加条件分支，比如“库存过滤后商品不够就扩大召回”，会让 `recommend()` 越来越长。

LangGraph 的作用是把推荐流程显式建成一张状态图：

```text
init -> phase1 -> merge1 -> phase2 -> merge2 -> phase3 -> aggregate
                                      |
                                      +-- expand -> merge2
```

## 新增文件

```text
app/orchestrator/graph.py
```

里面包含：

```text
PipelineState        LangGraph 节点之间传递的状态
build_graph()        构建状态图
rec_graph            编译后的图实例
recommend_with_graph 对外封装，返回 RecommendResponse
```

## 图节点说明

| 节点 | 作用 |
|---|---|
| `init` | 生成 request_id、A/B 分桶、加载商品目录、记录请求指标 |
| `phase1` | 并行执行用户画像 Agent 和商品召回 Agent |
| `merge1` | 解析画像结果，treatment 组写入 `llm_hint` |
| `phase2` | 并行执行商品重排 Agent 和库存 Agent |
| `merge2` | 库存过滤、整理最终商品 |
| `expand` | 商品不足时扩大召回范围并重新重排 |
| `phase3` | 生成营销文案 |
| `aggregate` | 记录 Agent 指标、业务成功指标和 A/B 曝光 |

## 条件边

核心条件边在 `merge2` 后：

```text
final_products 数量 < request.num_items 且还没扩召回
  -> expand

否则
  -> phase3
```

这就是本阶段比原始线性 graph 更进一步的地方：流程不再只能一条线走到底，而是可以根据中间结果动态改道。

## 新增接口

```text
POST /api/v1/recommend/graph
```

请求体和原来的推荐接口一致：

```json
{
  "user_id": "u001",
  "num_items": 3,
  "preferred_categories": ["手机"]
}
```

返回结构也使用 `RecommendResponse`，所以前端或测试可以用同一种方式读取：

```text
products
experiment_group
experiment
marketing_copies
agent_results
```

## 和原 Supervisor 的关系

现在两套编排并存：

```text
POST /api/v1/recommend        -> 原 Supervisor
POST /api/v1/recommend/graph  -> LangGraph
```

这样做的好处是：

```text
1. 原来的稳定链路不被破坏。
2. LangGraph 可以作为实验编排器逐步增强。
3. 后面要做更多条件分支、人工审核节点、RAG 节点时，可以先放到 graph 里。
```

## 验证

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
D:\anaconda\envs\py3.10\python.exe -m compileall app tests
```

当前结果：

```text
24 passed
compileall 通过
```

手动请求：

```powershell
curl -X POST http://127.0.0.1:8010/api/v1/recommend/graph ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"u001\",\"num_items\":3}"
```

## 阅读重点

先看：

```text
app/orchestrator/graph.py
```

重点抓住这条主线：

```text
PipelineState 是“流程上下文”
每个 node 读取 state、写回 state
conditional_edges 根据 state 决定下一步走哪里
recommend_with_graph 把图结果转换成 RecommendResponse
```
