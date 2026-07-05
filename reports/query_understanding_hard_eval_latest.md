# Query Understanding Hard Eval Report

- Generated at: `2026-07-05T15:39:39.905935+00:00`
- Case count: `36`
- Best by hard intent macro F1: `bert_rule_slots`
- Best by hard slot F1: `rule_baseline`

## Model Summary

| Model | Intent Acc | Intent Macro F1 | Slot F1 | Smalltalk Guard | Feedback Acc | Ref Rate | Unsupported Guard | Avg Lat ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule_baseline` | 0.6944 | 0.692 | 0.7634 | 1.0 | 0.5 | 1.0 | 0.5 | 10.2579 |
| `char_ngram_nb_classifier` | 0.7222 | 0.7315 | 0.0 | 0.75 | 1.0 | 0.0 | 0.0 | 0.0689 |
| `bert_rule_slots` | 0.8889 | 0.9026 | 0.7634 | 1.0 | 1.0 | 1.0 | 0.5 | 427.392 |

## Scenario Distribution

- `single_recommend`: 4
- `multi_slot`: 4
- `goal_switch`: 3
- `negative_feedback`: 4
- `compare`: 3
- `explain`: 3
- `ask_product`: 3
- `smalltalk`: 4
- `unsupported_catalog`: 2
- `edge_mixed`: 6

## Notes

- This hard set is manually simulated to stress realistic phrasing, target switching, references, feedback, and smalltalk guards.
- BERT is evaluated as intent-only; slots still come from the rule extractor.
- Scores are more useful for relative comparison than as production guarantees.
