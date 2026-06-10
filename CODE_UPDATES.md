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

## Step 11: LLM 用户画像 Agent

### 新增文件

```text
app/services/llm_client.py
steps/step-11-llm-user-profile/README.md
```

### 修改文件

```text
.env.example
README.md
app/agents/user_profile_agent.py
app/main.py
app/orchestrator/supervisor.py
app/services/__init__.py
requirements.txt
steps/README.md
tests/test_recommender.py
.codex/skills/ecommerce-rebuild-project/SKILL.md
.codex/skills/ecommerce-rebuild-project/references/project-state.md
.codex/skills/ecommerce-rebuild-project/references/roadmap.md
CODE_UPDATES.md
```

### 核心变化

```text
1. 新增 LLMClient，统一封装 OpenAI-compatible Chat API。
2. LLM 配置支持 LLM_*、DEEPSEEK_*、QWEN/DASHSCOPE_* 三类变量，并会跳过占位符 key，避免 DeepSeek 占位符挡住千问配置。
3. UserProfileAgent 在原有 profile/effective_request/feature_store 基础上新增 llm_profile。
4. llm_profile 根据 SQLite 长期画像 + Redis 实时行为窗口生成用户分群、意图摘要、推荐提示和价格敏感度。
5. LLM 不可用时返回默认画像，推荐流程继续走规则 fallback。
6. Supervisor 会把 llm_profile.recommendation_hint 写入 effective_request.context["llm_hint"]，给后续重排和文案 Agent 使用。
7. /health 升级到 step=11，版本升级到 0.11.0。
8. LLM Client 默认超时为 8 秒，并关闭 SDK 自动重试；UserProfileAgent 超时调整为 10 秒，避免 LLM 正常响应被 Supervisor 提前 fallback。
9. Supervisor 会把带有 llm_hint 的 effective_request 同步回 agent_results，便于前端和调试接口直接观察。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest -q
16 passed

D:\anaconda\envs\py3.10\python.exe -m compileall app tests
通过
```

### 你应该重点阅读

```text
app/services/llm_client.py
app/agents/user_profile_agent.py
app/orchestrator/supervisor.py
steps/step-11-llm-user-profile/README.md
```

## Step 12a: LLM 营销文案 Agent

### 新增文件

```text
steps/step-12a-llm-marketing-copy/README.md
```

### 修改文件

```text
app/agents/marketing_copy_agent.py
app/orchestrator/supervisor.py
app/services/llm_client.py
app/main.py
app/static/index.html
tests/test_recommender.py
README.md
steps/README.md
CODE_UPDATES.md
```

### 核心变化

```text
1. MarketingCopyAgent 从固定模板改为 LLM 优先、规则 fallback。
2. 文案 Prompt 根据 llm_profile.segments 选择不同分群模板。
3. LLM 输入包含用户分群、意图摘要、recommendation_hint、价格敏感度和商品属性。
4. LLM 输出统一转换为 MarketingCopy 模型需要的 product_id + text。
5. 增加广告法敏感词过滤，命中后替换为 ***。
6. Supervisor 调用 MarketingCopyAgent 时传入 UserProfileAgent 产出的 llm_profile。
7. 前端商品卡片展示 marketing_copies 文案。
8. /health 升级为 step=12a，版本升级到 0.12.0。
9. llm_client.chat_json 支持 JSON 数组返回，方便文案批量生成。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest -q
16 passed

D:\anaconda\envs\py3.10\python.exe -m compileall app tests
通过
```

### 你应该重点阅读

```text
app/agents/marketing_copy_agent.py
app/orchestrator/supervisor.py
app/static/index.html
steps/step-12a-llm-marketing-copy/README.md
```

## Step 12b: LLM 商品重排 Agent

### 新增文件

```text
steps/step-12b-llm-rerank/README.md
```

### 修改文件

```text
app/agents/product_rec_agent.py
app/main.py
app/static/index.html
tests/test_recommender.py
README.md
steps/README.md
CODE_UPDATES.md
```

### 核心变化

```text
1. ProductRecAgent._rerank() 在 request.context["llm_hint"] 存在时优先尝试 LLM 重排。
2. 新增 RERANK_PROMPT，要求 LLM 返回商品 ID JSON 数组。
3. 新增 _llm_rerank()，把用户偏好、预算、llm_hint 和候选商品列表交给 LLM。
4. 新增 _normalize_llm_product_ids()，校验 LLM 返回的 product_id、去重、不足补齐。
5. LLM 重排成功时返回 mode=llm_rerank，backend=llm+rule_rerank。
6. LLM 不可用、返回格式异常或 ID 无效时自动回退原 score_product 规则排序。
7. 前端 LLM 面板新增“重排模式”和“文案模式”。
8. /health 升级为 step=12b，版本升级到 0.12.1。
9. LLM 重排只把前 10-12 个候选传给模型，避免 200 个候选导致请求超时；成功调用后会清空 llm_client.last_error。
10. llm_client.chat() 在 DeepSeek v4 的 content 为空时，会读取 reasoning_content；ProductRecAgent 可从非 JSON 文本里抽取有效商品 ID。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest -q
17 passed

D:\anaconda\envs\py3.10\python.exe -m compileall app tests
通过
```

### 你应该重点阅读

```text
app/agents/product_rec_agent.py
app/personalization.py
tests/test_recommender.py
steps/step-12b-llm-rerank/README.md
```

## Step 13a: A/B 实验驱动策略开关

### 新增文件

```text
steps/step-13a-ab-experiment-gating/README.md
```

### 修改文件

```text
app/services/ab_test.py
app/orchestrator/supervisor.py
app/agents/user_profile_agent.py
app/agents/marketing_copy_agent.py
app/main.py
tests/test_recommender.py
steps/README.md
CODE_UPDATES.md
```

### 核心变化

```text
1. A/B 实验配置从占位 strategy 改成真实策略：control=rule，treatment=llm。
2. Supervisor 会把 experiment.group 传给 UserProfileAgent 和 MarketingCopyAgent。
3. control 组 UserProfileAgent 不调用 LLM，直接返回规则画像占位结果。
4. control 组不会写入 effective_request.context.llm_hint，因此 ProductRecAgent 自动走规则重排。
5. treatment 组保留 LLM 画像 -> llm_hint -> LLM 重排 -> LLM 文案链路。
6. control 组 MarketingCopyAgent 直接走规则模板文案，返回 mode=control_rule。
7. /health 升级为 step=13a，版本升级到 0.13.0。
8. 新增 control/treatment 回归测试，确保 control 不会偷偷调用 LLM。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest -q
19 passed
```

### 你应该重点阅读

```text
app/services/ab_test.py
app/orchestrator/supervisor.py
app/agents/user_profile_agent.py
app/agents/marketing_copy_agent.py
steps/step-13a-ab-experiment-gating/README.md
```

## Step 13b: A/B 实验数据闭环

### 新增文件

```text
steps/step-13b-ab-outcome-stats/README.md
```

### 修改文件

```text
app/models.py
app/services/ab_test.py
app/orchestrator/supervisor.py
app/main.py
app/static/index.html
tests/test_recommender.py
steps/README.md
CODE_UPDATES.md
README.md
```

### 核心变化

```text
1. 新增 ExperimentOutcome 请求模型，用于实验点击/负反馈回传。
2. ABTestEngine 新增 record_exposure()，推荐成功后记录曝光。
3. ABTestEngine 新增 record_outcome()，记录 click/skip 并更新 alpha/beta。
4. ABTestEngine 新增 get_stats()，按 control/treatment 聚合 exposures、clicks、skips、ctr、expected_ctr。
5. ABTestEngine 新增 assign_thompson()，为后续动态流量分配准备 Thompson Sampling 入口。
6. Supervisor 在推荐成功后自动记录实验曝光。
7. FastAPI 新增 POST /api/v1/experiments/{experiment_id}/outcome。
8. 前端行为按钮会在记录用户行为后回传实验 outcome：查看/喜欢/加购为 success=true，不喜欢为 success=false。
9. /health 升级为 step=13b，版本升级到 0.13.1。
10. 测试新增曝光、outcome 统计、CTR、Beta 参数和 Thompson Sampling 入口验证。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest -q
22 passed

D:\anaconda\envs\py3.10\python.exe -m compileall app tests
通过
```

### 你应该重点阅读

```text
app/services/ab_test.py
app/orchestrator/supervisor.py
app/main.py
tests/test_recommender.py
steps/step-13b-ab-outcome-stats/README.md
```

## Step 14: LangGraph 状态图编排

### 新增文件

```text
app/orchestrator/graph.py
steps/step-14-langgraph-orchestration/README.md
```

### 修改文件

```text
app/main.py
requirements.txt
tests/test_recommender.py
steps/README.md
CODE_UPDATES.md
README.md
.codex/skills/ecommerce-rebuild-project/SKILL.md
.codex/skills/ecommerce-rebuild-project/references/project-state.md
```

### 核心变化

```text
1. 新增 LangGraph 编排器，保留原 Supervisor 不动。
2. 新增 PipelineState，用 TypedDict 表示节点之间传递的推荐流程状态。
3. 新增 init、phase1、merge1、phase2、merge2、expand、phase3、aggregate 节点。
4. phase1 继续并行执行 UserProfileAgent 和 ProductRecAgent recall。
5. phase2 继续并行执行 ProductRecAgent rerank 和 InventoryAgent。
6. merge2 后新增条件边：最终商品不足且未扩召回时，进入 expand。
7. expand 节点会扩大召回范围并重新执行 phase2，再回到 merge2。
8. 新增 POST /api/v1/recommend/graph，返回 RecommendResponse。
9. /health 升级为 step=14，版本升级到 0.14.0。
10. requirements.txt 新增 langgraph>=0.2.0。
11. 前端默认推荐请求从 /api/v1/recommend 切换到 /api/v1/recommend/graph。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest -q
24 passed

D:\anaconda\envs\py3.10\python.exe -m compileall app tests
通过
```

### 你应该重点阅读

```text
app/orchestrator/graph.py
app/main.py
tests/test_recommender.py
steps/step-14-langgraph-orchestration/README.md
```

## Step 14.1: Qwen3.5 LLM 适配

### 新增文件

```text
无
```

### 修改文件

```text
app/services/llm_client.py
.env.example
tests/test_recommender.py
CODE_UPDATES.md
```

### 核心变化

```text
1. 撤回 qwen2.5:3b 的本地小模型专项兼容改动，不再保留文案嵌套数组容错和额外营销禁词补丁。
2. LLMClient 新增 LLM_ENABLE_THINKING 配置。
3. 模型名匹配 qwen3/qwen3.5 时，默认关闭 thinking，避免 JSON Agent 拿到空 content 或 reasoning 内容。
4. 请求 OpenAI-compatible Chat API 时，会通过 extra_body 传入 enable_thinking 和 think 两个字段，兼容 DashScope 与 Ollama 风格接口。
5. LLMClient.status() 增加 enable_thinking，前端/调试接口可以看到当前 thinking 开关。
6. chat_json 的附加指令改成 ASCII 的 JSON-only 约束，避免中文编码显示异常影响本地模型。
7. 新增测试覆盖 qwen3.5 默认关闭 thinking，以及环境变量显式开启 thinking 的情况。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest tests\test_recommender.py -q
26 passed

D:\anaconda\envs\py3.10\python.exe -m compileall app tests
通过
```

### 你应该重点阅读

```text
app/services/llm_client.py
.env.example
tests/test_recommender.py
```

## Step 14.2: Qwen2.5 Local LLM Compatibility

### 新增文件

```text
无
```

### 修改文件

```text
app/services/llm_client.py
app/agents/marketing_copy_agent.py
tests/test_recommender.py
CODE_UPDATES.md
```

### 核心变化

```text
1. LLMClient.chat_json 增加 JSON 候选解析逻辑。
2. 当本地小模型少输出一个右括号时，尝试补齐 JSON 的 [] / {} 结构。
3. MarketingCopyAgent 支持把 LLM 返回的嵌套数组压平，兼容 qwen2.5:3b 偶发输出 [[{...}]] 的情况。
4. 新增测试覆盖 qwen2.5 风格的缺失右括号 JSON，以及嵌套文案数组归一化。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe -m pytest tests\test_recommender.py -q
28 passed

本地 Ollama qwen2.5:3b 临时验证：
UserProfileAgent latency_sec: 4.04，成功生成 segments / recommendation_hint
MarketingCopyAgent latency_sec: 0.96，mode=llm，成功生成 LLM 文案
```

### 你应该重点阅读

```text
app/services/llm_client.py
app/agents/marketing_copy_agent.py
tests/test_recommender.py
```

## Step 15: Offline Recommendation Evaluation

### 新增文件

```text
scripts/import_amazon_user_events.py
scripts/evaluate_recommendation_offline.py
data/amazon_user_events_sample.csv
reports/recommendation_offline_eval_latest.json
```

### 修改文件

```text
app/agents/marketing_copy_agent.py
tests/test_recommender.py
CODE_UPDATES.md
```

### 核心变化

```text
1. 新增 Amazon 用户行为导入脚本，支持 jsonl/jsonl.gz/csv 原始 review 数据转成项目事件格式。
2. 当前没有下载完整 Amazon Reviews 2023 原始行为时，脚本可基于现有商品元数据生成 weak-label 行为样本。
3. 生成 data/amazon_user_events_sample.csv，字段为 user_id/product_id/event_type/rating/timestamp/source。
4. 新增离线推荐评测脚本，按用户时间序列切分 history/target，再调用当前推荐链路计算指标。
5. 指标分为 exact 与 intent 两类：
   - exact: 是否命中未来同一个商品 ASIN。
   - intent: 是否命中未来正反馈商品的类目/品牌/标签意图。
6. 离线评测默认关闭外部 LLM 和本地 Ollama 文案模型，避免模型服务状态影响推荐链路评估。
7. MarketingCopyAgent 的本地 Ollama LoRA 文案模型改成显式开关：COPY_LLM_BACKEND=ollama 才启用。
8. 新增测试覆盖 rating -> event_type 映射、timestamp 归一化和 NDCG 计算。
```

### 运行和验证

```text
D:\anaconda\envs\py3.10\python.exe scripts\import_amazon_user_events.py --generate-sample --max-events 500 --output data\amazon_user_events_sample.csv
Wrote 500 events to data\amazon_user_events_sample.csv

D:\anaconda\envs\py3.10\python.exe scripts\evaluate_recommendation_offline.py --events data\amazon_user_events_sample.csv --orchestrator graph --k 5 --max-users 50
case_count: 44
exact_hit_rate_at_k: 0.0
intent_hit_rate_at_k: 0.9773
intent_recall_at_k: 0.9682
intent_ndcg_at_k: 0.9761
budget_compliance_rate: 1.0
inventory_compliance_rate: 1.0
avg_latency_ms: 137.2568
fallback_rate: 0.5455
```

### 你应该重点阅读

```text
scripts/import_amazon_user_events.py
scripts/evaluate_recommendation_offline.py
data/amazon_user_events_sample.csv
reports/recommendation_offline_eval_latest.json
```
