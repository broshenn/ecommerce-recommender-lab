# Architecture Alignment With Original Project

Original reference project:

```text
D:\pycode\agent\cluade\multi-agent-ecommerce-system
```

## Original Architecture Spine

The original project is not a plain recommender. Its spine is:

```text
FastAPI
  -> SupervisorOrchestrator
  -> Phase 1 parallel:
       UserProfileAgent
       ProductRecAgent recall
  -> Phase 2 parallel:
       ProductRecAgent rerank
       InventoryAgent
  -> Phase 3:
       MarketingCopyAgent
       ABTestEngine
  -> RecommendationResponse with agent_results and metrics
```

Original storage roles:

```text
SQLite/MySQL  -> business data and durable records
Redis         -> real-time feature store, sliding-window behavior features, offline tag cache
Milvus/Chroma -> product vector retrieval
```

Original Agent roles:

```text
UserProfileAgent  -> real-time features + RFM + segments
ProductRecAgent   -> multi-route recall + rerank
InventoryAgent    -> stock filtering + alerts + purchase limits
MarketingCopyAgent-> personalized copy + compliance filter
```

Original reliability ideas:

```text
BaseAgent
AgentResult
timeout
retry
fallback
latency_ms
success/error/confidence
MetricsCollector
A/B bucket assignment
```

## Current Project State

Current project has:

```text
FastAPI
Vue 3 frontend
Amazon Reviews 2023 product metadata sample
SQLite persisted current-user behavior events
behavior profile aggregation
rule-based personalization scoring
inventory filtering
```

Current project does not yet have:

```text
BaseAgent
AgentResult
SupervisorOrchestrator
MarketingCopyAgent
ABTestEngine
MetricsCollector
Redis feature store
Chroma vector recall
LLM fallback wrapper
```

## Do Not Drift

Do not jump straight to ML training, RAG, or a large frontend redesign before the architecture spine is aligned.

The next priority is to reshape the current direct recommender into the original project's modular multi-agent architecture while keeping every step runnable and easy to read.

