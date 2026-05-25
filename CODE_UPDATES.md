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

## Step 10：Redis 在线画像缓存 + 实时行为窗口

### 新增文件

```text
app/services/feature_store.py
steps/step-10-redis-feature-store/README.md
```

### 修改文件

```text
app/behavior.py
app/agents/user_profile_agent.py
app/main.py
app/services/__init__.py
requirements.txt
.env.example
README.md
steps/README.md
tests/test_recommender.py
CODE_UPDATES.md
```

### 核心变化

```text
1. 新增 RedisFeatureStore，对齐原项目 Feature Store 思路。
2. Redis Sorted Set 保存 behavior:{user_id}:{event_type}，score 为事件时间戳。
3. Redis profile:{user_id} 缓存 SQLite 聚合后的用户画像。
4. record_event 写入顺序变成：SQLite 成功 -> 删除 profile cache -> 写 Redis 实时行为窗口。
5. build_user_profile 读取顺序变成：Redis profile 命中直接返回，miss 后从 SQLite 重建并写回 Redis。
6. UserProfileAgent 的 AgentResult 增加 feature_store 字段，能看到 Redis 状态和实时窗口特征。
7. 新增 /api/v1/feature-store/{user_id} 查看在线特征。
8. Redis 不可用时自动降级，SQLite 和推荐流程仍然可用。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest -q
16 passed

D:\anaconda\envs\py3.10\python.exe -m compileall app tests
通过

本机 Redis 启动脚本：
D:\redis\Redis-8.4.0-Windows-x64-msys2-with-Service\start.bat
```

### 你应该重点阅读

```text
app/services/feature_store.py
app/behavior.py
app/agents/user_profile_agent.py
steps/step-10-redis-feature-store/README.md
```
