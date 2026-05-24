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
Step 6: SQLite persistence for current user behavior
```

Architecture alignment:

```text
Current project is still a direct recommender flow.
Next step should introduce Supervisor + AgentResult + 4 Agent skeleton.
```

Important files:

```text
app/models.py                  request/response/product/event/profile schemas
app/catalog.py                 CSV product loader
app/database.py                SQLite connection and table initialization
app/behavior.py                current-user event store and profile builder
app/inventory.py               stock status and purchase limit rules
app/personalization.py         user profile scoring rules
app/recommender.py             recommendation orchestration
app/main.py                    FastAPI routes
app/static/index.html          single-page frontend
data/products_amazon_sample.csv 1000 product rows
scripts/import_amazon_products.py Amazon metadata importer
tests/test_recommender.py      regression tests
steps/                         step notes
```

Current recommendation request fields:

```json
{
  "user_id": "u001",
  "scene": "homepage",
  "num_items": 3,
  "preferred_categories": ["手机", "电子数码"],
  "liked_brands": ["Bastmei", "Sharp"],
  "preferred_tags": ["手机配件", "办公"],
  "budget_min": 50,
  "budget_max": 500,
  "recent_views": ["B07ZPSG8P5"],
  "disliked_products": []
}
```

Behavior APIs:

```text
POST /api/v1/events
GET /api/v1/users/{user_id}/events
GET /api/v1/users/{user_id}/profile
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
