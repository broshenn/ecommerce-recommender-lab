# Query Understanding Model Compare Report

- Generated at: `2026-07-04T16:53:30.633341+00:00`
- Train set: `800` rows
- Eval set: `200` rows
- Recommendation: `char_ngram_nb_classifier + rule_baseline` - Use the trainable classifier for high-frequency intent classification, and keep rule extraction for slots until a sequence-labeling model is trained.

## Model Summary

| Model | Status | Intent Acc | Intent Macro F1 | Slot F1 | Need Rec Acc | Avg Latency ms | Cost |
|---|---|---:|---:|---:|---:|---:|---|
| `rule_baseline` | completed | 0.645 | 0.7147 | 0.5963 | 0.81 | 10.7716 | low |
| `char_ngram_nb_classifier` | completed | 0.985 | 0.9861 | 0.0 | 1.0 | 0.0701 | low |
| `llm_classifier` | skipped: Pass --include-llm to call the configured OpenAI-compatible LLM. | - | - | - | - | - | - |
| `distilbert_classifier` | skipped: No --bert-predictions file provided. | - | - | - | - | - | - |

## Dataset

### train

- Count: `800`
- Intent distribution: `{'smalltalk': 84, 'refine_preferences': 128, 'ask_product': 99, 'record_feedback': 56, 'recommend_products': 306, 'compare_products': 58, 'explain_recommendation': 69}`
- Source distribution: `{'deepseek_seed_augmented': 767, 'deepseek_synthetic': 33}`

### eval

- Count: `200`
- Intent distribution: `{'smalltalk': 24, 'explain_recommendation': 22, 'recommend_products': 87, 'refine_preferences': 29, 'compare_products': 6, 'record_feedback': 14, 'ask_product': 18}`
- Source distribution: `{'deepseek_seed_augmented': 193, 'deepseek_synthetic': 7}`

## Notes

- rule_baseline is the production fallback used by IntentAgent today.
- char_ngram_nb is a small trainable baseline for intent only; it does not extract slots.
- llm_classifier and distilbert_classifier are optional to avoid making the main demo depend on external APIs or GPU training.
