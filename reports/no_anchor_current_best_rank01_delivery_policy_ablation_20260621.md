# Current-Best Delivery Policy Ablation

- Source zip: old opponent-scheduler rank01 raw submission.
- Selection rule: policy order is predeclared; no GT metric is used to choose the submitted policy.
- Decision: keep `density_simple`; no delivery-only improvement found.

| policy | IDF1 | HOTA | AssA | DetRe | dropped rows | decision |
|---|---:|---:|---:|---:|---:|---|
| `density_simple` | 0.655817 | 0.519228 | 0.534791 | 0.574613 | 33685 | keep as standing best |
| `density_oracle_lite` | 0.655810 | 0.519220 | 0.534789 | 0.574541 | 34434 | reject; below density_simple |
| `confidence_tail` | 0.655800 | 0.519203 | 0.534773 | 0.574486 | 34783 | reject; below density_simple |

## Interpretation

- `density_oracle_lite` drops 749 additional rows and loses `0.000007` IDF1 versus `density_simple`.
- `confidence_tail` drops 1098 additional rows and loses `0.000017` IDF1 versus `density_simple`.
- The delivery filter has reached a local plateau for this source zip; further gain likely requires changing the assignment/component structure, not only filtering detections.

## Artifacts

- `local_runs/no_anchor_current_best_rank01_delivery_policy_ablation_20260621.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_delivery_policy_ablation_20260621/policies.json`
