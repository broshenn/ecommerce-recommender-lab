# Project State

Project root:

```text
D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step
```

Run:

```powershell
D:\anaconda\envs\py3.10\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Test:

```powershell
D:\anaconda\envs\py3.10\python.exe -m pytest -q
```

Current step:

```text
Step 11: LLM user profile agent
```

Architecture alignment:

```text
Current project has Supervisor + AgentResult + 4 Agents + Chroma vector recall + Redis Feature Store + LLM user-profile analysis + ABTestEngine + MetricsCollector.
Next step should add LLM rerank and LLM marketing copy generation with rule fallback.
```

Important files:

```text
app/models.py                    request/response/product/event/profile/experiment schemas
app/catalog.py                   CSV product loader
app/database.py                  SQLite connection and table initialization
app/behavior.py                  current-user event store and profile builder
app/agents/                      BaseAgent and four concrete agents
app/orchestrator/supervisor.py   Supervisor orchestration
app/services/vector_store.py     Chroma vector recall and Qwen/local embedding
app/services/feature_store.py    Redis online profile cache and real-time behavior windows
app/services/llm_client.py       OpenAI-compatible chat client for LLM profile analysis
app/services/ab_test.py          stable user_id experiment bucketing
app/services/metrics.py          in-memory agent and business metrics
app/inventory.py                 stock status and purchase limit rules
app/personalization.py           user profile scoring rules
app/recommender.py               recommendation entrypoint
app/main.py                      FastAPI routes
app/static/index.html            Vue 3 frontend
data/products_amazon_sample.csv  1000 product rows
scripts/import_amazon_products.py Amazon metadata importer
tests/test_recommender.py        regression tests
steps/                           step notes only, no code snapshots
```

Current recommendation response includes:

```text
products
strategy
reason
experiment_group
experiment
marketing_copies
agent_results
```

Current experiment behavior:

```text
Step 14 is the latest root version.
control   -> rule profile + rule rerank + rule copy
treatment -> LLM profile + LLM rerank + LLM copy, with rule fallback
ab-user-1 -> stable control sample user
ab-user-2 -> stable treatment sample user
recommend success records exposure
POST /api/v1/experiments/{experiment_id}/outcome records click/skip outcome
GET /api/v1/experiments returns stats: exposures, clicks, skips, ctr, alpha, beta, expected_ctr
POST /api/v1/recommend keeps the original Supervisor orchestration
POST /api/v1/recommend/graph uses LangGraph orchestration with merge2 -> expand conditional routing
```

Behavior APIs:

```text
POST /api/v1/events
GET /api/v1/users/{user_id}/events
GET /api/v1/users/{user_id}/profile
```

Experiment and metrics APIs:

```text
GET /api/v1/experiments
GET /api/v1/experiments?user_id=u001
POST /api/v1/recommend/graph
GET /api/v1/metrics
GET /api/v1/vector-store
GET /api/v1/feature-store/{user_id}
```

LLM profile output:

```text
agent_results.user_profile.data.llm_profile
agent_results.user_profile.data.llm_client
effective_request.context.llm_hint
```

Redis keys:

```text
profile:{user_id}
behavior:{user_id}:view
behavior:{user_id}:like
behavior:{user_id}:dislike
behavior:{user_id}:add_to_cart
```

Runtime database:

```text
data/app.sqlite3
```

Persisted table:

```text
user_events(event_id, user_id, product_id, event_type, created_at)
```

Scoring rules:

```text
category match +40
brand match +25
each matching tag +10
within budget +20
outside budget -20
rating adds rating * 4
recently viewed -30
disliked product -100
out-of-stock products are filtered before scoring
```
