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
SQLite/MySQL   -> business data and durable records
Redis          -> real-time feature store, sliding-window behavior features, offline tag cache
Milvus/Chroma  -> product vector retrieval
```

Original Agent roles:

```text
UserProfileAgent   -> real-time features + RFM + segments
ProductRecAgent    -> multi-route recall + rerank
InventoryAgent     -> stock filtering + alerts + purchase limits
MarketingCopyAgent -> personalized copy + compliance filter
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

Current project now has:

```text
FastAPI
Vue 3 frontend
Amazon Reviews 2023 product metadata sample
SQLite persisted current-user behavior events
behavior profile aggregation
rule-based personalization scoring
inventory filtering
BaseAgent
AgentResult
SupervisorOrchestrator
UserProfileAgent
ProductRecAgent
InventoryAgent
MarketingCopyAgent
ABTestEngine
MetricsCollector
```

Current project does not yet have:

```text
Redis feature store
Chroma vector recall
LLM marketing copy generation
RAG product Q&A / recommendation explanation
trained ranking model
```

## Do Not Drift

Do not jump straight to ML training or RAG before vector recall exists.

The next priority is:

```text
Step 9: Chroma product vector recall inside ProductRecAgent
```

Redis should come after Chroma in this learning path, because SQLite already covers durable behavior storage and Redis should be added as the online feature/cache layer.
