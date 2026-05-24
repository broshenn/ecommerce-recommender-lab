# Roadmap

## Completed

```text
Step 1: minimal catalog + rule recommendation + frontend
Step 2: inventory filtering and stock messages
Step 3: Amazon Reviews 2023 product metadata fields
Step 4: current user profile scoring and personalized ranking
Step 5: current user behavior collection and profile aggregation
Step 6: SQLite persistence for current user behavior
```

Step notes:

```text
steps/step-04-user-profile-ranking/README.md
steps/step-05-current-user-behavior/README.md
steps/step-06-sqlite-persistence/README.md
```

## Architecture Rule

Stay aligned with the original project:

```text
Supervisor + 4 Agent + Feature Store + Vector Recall + Inventory + Copy + A/B + Metrics
```

Do not jump directly to ML ranking training or RAG before the multi-agent spine exists.

## Immediate Next Step

Step 7 should introduce the multi-agent skeleton from the original project, without adding LLM calls yet.

Recommended scope:

```text
1. Add app/agents/base_agent.py with AgentResult, timeout, fallback, latency.
2. Split current direct logic into simple non-LLM agents:
   - UserProfileAgent: build profile from SQLite behavior + request.
   - ProductRecAgent: rule recall/rerank using current personalization score.
   - InventoryAgent: reuse current inventory filtering/status rules.
   - MarketingCopyAgent: template copy only, no LLM yet.
3. Add app/orchestrator/supervisor.py.
4. Keep /api/v1/recommend response backward-compatible for the frontend.
5. Add optional agent_results/debug field only if it does not break the UI.
6. Write step note to steps/step-07-supervisor-agent-skeleton/README.md.
```

## After Step 7

```text
Step 8: add A/B testing and metrics endpoints
Step 9: add Chroma product vector recall inside ProductRecAgent
Step 10: add Redis feature store for real-time sliding-window profile features
Step 11: add MarketingCopyAgent LLM generation and compliance fallback
Step 12: add RAG product Q&A or product explanation, after Chroma exists
Step 13: add offline evaluation and ML ranking training
```

## Why Redis Is Not Next

Redis in the original project is a real-time feature store:

```text
behavior:{user_id}:{behavior_type} sorted sets
sliding windows: 1h / 24h / 7d
profile:{user_id} offline tag cache
RFM and real-time feature aggregation
```

Our project already has durable behavior storage in SQLite. Redis should be added later as an online feature/cache layer, not as a replacement for SQLite.

## Deferred ML Ranking Training

Save this for after the architecture spine exists and after reviewing the original project again:

```text
D:\pycode\agent\cluade\multi-agent-ecommerce-system
```

Future training scope:

```text
1. Pull Amazon Reviews 2023 user review rows for selected categories.
2. Map rating >= 4 to positive preference.
3. Map rating <= 2 to negative preference.
4. Use verified_purchase as a stronger purchase-like signal.
5. Generate training samples:
   user/profile features + product features -> liked/disliked label.
6. Train a lightweight model first, such as LogisticRegression or LightGBM.
7. Replace hard-coded ranking weights with model probabilities.
8. Keep rule scoring as fallback when the model is unavailable.
```

## Data Expansion Notes

Use:

```powershell
D:\anaconda\envs\py3.10\python.exe scripts\import_amazon_products.py --limit 1000 --output data\products_amazon_sample.csv
```

The importer streams Hugging Face JSONL files and writes only adapted rows. It should not download whole multi-GB raw files.

