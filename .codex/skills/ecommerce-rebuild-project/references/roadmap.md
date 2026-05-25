# Roadmap

## Completed

```text
Step 1: minimal catalog + rule recommendation + frontend
Step 2: inventory filtering and stock messages
Step 3: Amazon Reviews 2023 product metadata fields
Step 4: current user profile scoring and personalized ranking
Step 5: current user behavior collection and profile aggregation
Step 6: SQLite persistence for current user behavior
Step 7: Supervisor + 4 Agent skeleton
Step 8: A/B testing and metrics endpoints
Step 9: Chroma vector recall with optional Qwen embeddings
Step 10: Redis online profile cache and real-time behavior windows
Step 11: LLM Client and UserProfileAgent profile analysis
```

Step notes:

```text
steps/step-04-user-profile-ranking/README.md
steps/step-05-current-user-behavior/README.md
steps/step-06-sqlite-persistence/README.md
steps/step-07-supervisor-agent-skeleton/README.md
steps/step-08-ab-test-metrics/README.md
steps/step-09-chroma-vector-recall/README.md
steps/step-10-redis-feature-store/README.md
steps/step-11-llm-user-profile/README.md
```

## Architecture Rule

Stay aligned with the original project:

```text
Supervisor + 4 Agent + Feature Store + Vector Recall + Inventory + Copy + A/B + Metrics
```

## Immediate Next Step

Step 12 should add LLM rerank and LLM marketing copy generation.

Recommended scope:

```text
1. Let ProductRecAgent rerank consume effective_request.context.llm_hint.
2. Keep rule scoring as fallback when LLM is unavailable.
3. Add LLM generation inside MarketingCopyAgent.
4. Add simple compliance checks for exaggerated or unsafe claims.
5. Record LLM copy latency and failure metrics.
6. Write step note to steps/step-12-llm-rerank-copy/README.md.
```

## After Step 11

```text
Step 13: add RAG product Q&A or recommendation explanation
Step 14: add offline evaluation and ML ranking training
```

## Deferred ML Ranking Training

Save this for after the architecture spine, Chroma recall, Redis feature windows, and event tracking exist:

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
