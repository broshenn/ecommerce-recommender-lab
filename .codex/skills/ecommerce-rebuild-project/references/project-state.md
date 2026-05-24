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
Step 8: A/B testing and metrics
```

Architecture alignment:

```text
Current project has Supervisor + AgentResult + 4 non-LLM Agents + ABTestEngine + MetricsCollector.
Next step should add Chroma vector recall inside ProductRecAgent.
```

Important files:

```text
app/models.py                    request/response/product/event/profile/experiment schemas
app/catalog.py                   CSV product loader
app/database.py                  SQLite connection and table initialization
app/behavior.py                  current-user event store and profile builder
app/agents/                      BaseAgent and four concrete agents
app/orchestrator/supervisor.py   Supervisor orchestration
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
GET /api/v1/metrics
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
