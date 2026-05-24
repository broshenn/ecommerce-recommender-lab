# Roadmap

## Completed

```text
Step 1: minimal catalog + rule recommendation + frontend
Step 2: inventory filtering and stock messages
Step 3: Amazon Reviews 2023 product metadata fields
Step 4: current user profile scoring and personalized ranking
Step 5: current user behavior collection and profile aggregation
```

Step notes:

```text
steps/step-04-user-profile-ranking/README.md
steps/step-05-current-user-behavior/README.md
```

## Immediate Next Step

Step 6 should persist products and behavior with SQLite.

Recommended scope:

```text
1. Add SQLite connection and table creation.
2. Persist behavior events instead of using memory only.
3. Keep /api/v1/events and profile APIs unchanged.
4. Preserve the existing CSV importer as a product seed source.
5. Write step note to steps/step-06-sqlite-persistence/README.md.
```

Later steps:

```text
Step 7: add Chroma semantic retrieval for product title/features
Step 8: add RAG product Q&A based on product metadata
Step 9: add LLM-generated recommendation explanations
Step 10: add offline evaluation metrics such as CTR proxy, recall, diversity
```

## Deferred ML Ranking Training

Save this for after the full basic framework is in place and after reviewing the original project at:

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
7. Replace hard-coded Step 4 weights with model probabilities.
8. Keep rule scoring as fallback when the model is unavailable.
```

## Data Expansion Notes

Use:

```powershell
D:\anaconda\envs\py3.10\python.exe scripts\import_amazon_products.py --limit 1000 --output data\products_amazon_sample.csv
```

The importer streams Hugging Face JSONL files and writes only adapted rows. It should not download whole multi-GB raw files.

Keep this project focused on product metadata plus current-user behavior. Avoid introducing large historical multi-user behavior datasets unless the user changes the product direction.
