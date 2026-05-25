# Code Updates

这个文件记录每个阶段实际改动了哪些代码，方便你读项目时先看“变更说明”，再去看源码。

## 记录格式

每完成一个 Step，都按下面格式追加：

```text
## Step X：功能名

### 新增文件

### 修改文件

### 核心变化

### 运行和验证

### 你应该重点阅读
```

## Step 9：Chroma 向量召回

### 新增文件

```text
.env.example
app/services/vector_store.py
steps/step-09-chroma-vector-recall/README.md
```

### 修改文件

```text
app/agents/product_rec_agent.py
app/orchestrator/supervisor.py
app/main.py
requirements.txt
README.md
steps/README.md
tests/test_recommender.py
.codex/skills/ecommerce-rebuild-project/SKILL.md
.codex/skills/ecommerce-rebuild-project/references/architecture-alignment.md
.codex/skills/ecommerce-rebuild-project/references/project-state.md
.codex/skills/ecommerce-rebuild-project/references/roadmap.md
```

### 核心变化

```text
1. ProductRecAgent 拆成 recall 和 rerank 两种模式。
2. recall 阶段优先使用 Chroma 商品向量召回。
3. rerank 阶段继续复用原来的规则打分。
4. 新增千问 / DashScope text-embedding-v4 可选 embedding。
5. 默认使用 local hash embedding，保证没有 API Key 也能跑通。
6. Supervisor 流程变成：画像 -> 向量召回 -> 规则重排 -> 库存过滤 -> 文案 -> A/B -> Metrics。
7. 新增 /api/v1/vector-store 查看向量库状态。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest -q
15 passed

GET /health
step = 9

GET /api/v1/vector-store
backend = chroma:local_hash:hashing-64

POST /api/v1/recommend
strategy = supervisor_agents+vector_recall+inventory_filter+ab_test
```

### 你应该重点阅读

```text
app/services/vector_store.py
app/agents/product_rec_agent.py
app/orchestrator/supervisor.py
steps/step-09-chroma-vector-recall/README.md
```

## Step 9.1：对齐原项目粗召回与精排边界

### 新增文件

```text
无
```

### 修改文件

```text
app/orchestrator/supervisor.py
tests/test_recommender.py
CODE_UPDATES.md
```

### 核心变化

```text
1. 删除 Supervisor 里的 _expand_candidates_with_profile。
2. 删除 _matches_profile 辅助方法。
3. Phase 1 的 Chroma 只负责粗召回，不再在召回后补充画像匹配商品。
4. Phase 2 的 rerank 使用 UserProfileAgent 产出的 effective_request 做精排。
5. 粗召回数量从 max(num_items * 20, 50) 调整为 min(total, max(num_items * 50, 200))，避免候选过窄。
6. 测试改为验证历史行为进入 effective_request，而不是要求历史商品一定出现在粗召回结果顶部。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest -q
15 passed

D:\anaconda\envs\py3.10\python.exe -m compileall app tests
通过
```

### 你应该重点阅读

```text
app/orchestrator/supervisor.py
tests/test_recommender.py
```
