# Chat Agent Eval Report

- Generated at: `2026-07-06T06:15:33.067033+00:00`
- Cases: `15`
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
| `avg_latency_ms` | 215.4513 | 1500.0 | yes |
| `unsupported_claim_rate` | 0.0 | 0.02 | yes |
| `memory_enrichment_success_rate` | 1.0 | 0.8 | yes |
| `smalltalk_policy_rate` | 1.0 | 0.9 | yes |
| `no_recommendation_guard_rate` | 1.0 | - | - |
| `recommendation_ndcg_at_k` | 1.0 | - | - |

## Scenario Summary

| Scenario | Cases | Intent F1 | Slot F1 | Task Success | Tool Success | Memory | Guard | Smalltalk Policy | Avg Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `budget_brand_slots` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 103.76 |
| `budget_min_slots` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 136.43 |
| `compare_products` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 215.41 |
| `explain_recommendation` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 265.7 |
| `goal_switching` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 351.73 |
| `long_term_memory` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 306.05 |
| `multi_turn_slot_filling` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 359.03 |
| `preference_memory` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 256.73 |
| `product_info_question` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 206.88 |
| `product_reference_feedback` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 241.71 |
| `short_term_memory` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 298.97 |
| `single_turn_recommend` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 125.45 |
| `smalltalk_fallback` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 96.88 |
| `smalltalk_policy` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 56.49 |
| `smalltalk_state_guard` | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 210.55 |

## Case Details

- `single_recommend_phone_case` (single_turn_recommend): intent=recommend_products, tools=PreferenceUpdateTool,RecommendGraphTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=125.45
- `multi_turn_budget_refine` (multi_turn_slot_filling): intent=refine_preferences, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=359.03
- `goal_switch_phone_to_computer_to_earphones` (goal_switching): intent=recommend_products, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=351.73
- `budget_range_brand_earphones` (budget_brand_slots): intent=recommend_products, tools=PreferenceUpdateTool,RecommendGraphTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=103.76
- `min_budget_computer_accessory` (budget_min_slots): intent=refine_preferences, tools=PreferenceUpdateTool,RecommendGraphTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=136.43
- `product_ref_feedback` (product_reference_feedback): intent=record_feedback, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,FeedbackTool,RecommendGraphTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=241.71
- `compare_products` (compare_products): intent=compare_products, tools=PreferenceUpdateTool,RecommendGraphTool,CompareProductTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=215.41
- `explain_recommendation` (explain_recommendation): intent=explain_recommendation, tools=PreferenceUpdateTool,RecommendGraphTool,ExplainRecommendationTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=265.7
- `ask_product_detail` (product_info_question): intent=ask_product, tools=PreferenceUpdateTool,RecommendGraphTool,ProductInfoTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=206.88
- `smalltalk_fallback` (smalltalk_fallback): intent=smalltalk, tools=SmalltalkTool, task_success=True, memory=False, guard=False, smalltalk_policy=True, latency_ms=96.88
- `brand_memory` (preference_memory): intent=record_feedback, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,FeedbackTool,RecommendGraphTool, task_success=True, memory=False, guard=False, smalltalk_policy=False, latency_ms=256.73
- `long_term_memory_cold_start` (long_term_memory): intent=recommend_products, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool, task_success=True, memory=True, guard=False, smalltalk_policy=False, latency_ms=306.05
- `smalltalk_after_recommendation` (smalltalk_state_guard): intent=smalltalk, tools=PreferenceUpdateTool,RecommendGraphTool,SmalltalkTool, task_success=True, memory=False, guard=False, smalltalk_policy=True, latency_ms=210.55
- `short_term_memory_continuation` (short_term_memory): intent=recommend_products, tools=PreferenceUpdateTool,RecommendGraphTool,PreferenceUpdateTool,RecommendGraphTool, task_success=True, memory=True, guard=False, smalltalk_policy=False, latency_ms=298.97
- `open_smalltalk_policy` (smalltalk_policy): intent=smalltalk, tools=SmalltalkTool, task_success=True, memory=False, guard=False, smalltalk_policy=True, latency_ms=56.49

## Failures

No failed cases.
