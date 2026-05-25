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
Chroma vector recall
Redis feature store
InventoryAgent
MarketingCopyAgent
ABTestEngine
MetricsCollector
```

Current project does not yet have:

```text
LLM marketing copy generation
RAG product Q&A / recommendation explanation
trained ranking model
```

## Do Not Drift

Do not jump straight to ML training or RAG before the vector recall path is stable.

The next priority is:

```text
Step 11: LLM marketing copy generation and compliance fallback
```

Redis now serves as online profile cache and real-time behavior windows. SQLite remains the durable source of truth.
