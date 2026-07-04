# Chat Agent Eval Report

- Generated at: `2026-07-04T16:37:41.311749+00:00`
- Cases: `13`
- Cases file: `D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step\data\chat_eval_cases.jsonl`

## Summary

| Metric | Value | Threshold | Passed |
|---|---:|---:|:---:|
| `intent_macro_f1` | 1.0 | 0.85 | yes |
| `slot_f1` | 1.0 | 0.8 | yes |
| `memory_consistency_rate` | 1.0 | 0.9 | yes |
| `product_ref_resolution_rate` | 1.0 | 0.85 | yes |
| `task_success_rate` | 1.0 | 0.8 | yes |
| `tool_success_rate` | 1.0 | 0.9 | yes |
| `budget_compliance_rate` | 1.0 | 0.95 | yes |
| `inventory_compliance_rate` | 1.0 | 1.0 | yes |
| `avg_latency_ms` | 235.6669 | 1500.0 | yes |
| `unsupported_claim_rate` | 0.0 | 0.02 | yes |
| `no_recommendation_guard_rate` | 1.0 | - | - |
| `recommendation_ndcg_at_k` | 1.0 | - | - |

## Scenario Summary

| Scenario | Cases | Intent F1 | Slot F1 | Task Success | Tool Success | Avg Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| `budget_brand_slots` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 138.21 |
| `budget_min_slots` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 128.66 |
| `compare_products` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 203.49 |
| `explain_recommendation` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 221.58 |
| `goal_switching` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 426.3 |
| `long_term_memory` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 312.63 |
| `multi_turn_slot_filling` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 385.03 |
| `preference_memory` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 252.5 |
| `product_info_question` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 243.56 |
| `product_reference_feedback` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 360.79 |
| `single_turn_recommend` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 161.06 |
| `smalltalk_fallback` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 61.43 |
| `smalltalk_state_guard` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 168.43 |

## Case Details

- `single_recommend_phone_case` (single_turn_recommend): intent=recommend_products, tools=PreferenceUpdateTool,RecommendGraphTool, task_success=True, latency_ms=161.06
- `multi_turn_budget_refine` (multi_turn_slot_filling): intent=refine_preferences, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool, task_success=True, latency_ms=385.03
- `goal_switch_phone_to_computer_to_earphones` (goal_switching): intent=recommend_products, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool, task_success=True, latency_ms=426.3
- `budget_range_brand_earphones` (budget_brand_slots): intent=recommend_products, tools=PreferenceUpdateTool,RecommendGraphTool, task_success=True, latency_ms=138.21
- `min_budget_computer_accessory` (budget_min_slots): intent=refine_preferences, tools=PreferenceUpdateTool,RecommendGraphTool, task_success=True, latency_ms=128.66
- `product_ref_feedback` (product_reference_feedback): intent=record_feedback, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,FeedbackTool,RecommendGraphTool, task_success=True, latency_ms=360.79
- `compare_products` (compare_products): intent=compare_products, tools=PreferenceUpdateTool,RecommendGraphTool,CompareProductTool, task_success=True, latency_ms=203.49
- `explain_recommendation` (explain_recommendation): intent=explain_recommendation, tools=PreferenceUpdateTool,RecommendGraphTool,ExplainRecommendationTool, task_success=True, latency_ms=221.58
- `ask_product_detail` (product_info_question): intent=ask_product, tools=PreferenceUpdateTool,RecommendGraphTool,ProductInfoTool, task_success=True, latency_ms=243.56
- `smalltalk_fallback` (smalltalk_fallback): intent=smalltalk, tools=SmalltalkTool, task_success=True, latency_ms=61.43
- `brand_memory` (preference_memory): intent=record_feedback, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,FeedbackTool,RecommendGraphTool, task_success=True, latency_ms=252.5
- `long_term_memory_cold_start` (long_term_memory): intent=recommend_products, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool, task_success=True, latency_ms=312.63
- `smalltalk_after_recommendation` (smalltalk_state_guard): intent=smalltalk, tools=PreferenceUpdateTool,RecommendGraphTool,SmalltalkTool, task_success=True, latency_ms=168.43

## Failures

No failed cases.
