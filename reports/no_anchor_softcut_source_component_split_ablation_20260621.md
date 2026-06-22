# No-Anchor Source-Component Softcut Split Ablation

Date: 2026-06-21

## Protocol

- No anchors were used.
- GT is used only by the evaluator for pair/full metrics, not for selecting split candidates or density policies.
- Standing best is the predeclared primary policy `density_oracle_lite`; `density_simple` is reported as an observed variant only.

## Result

- Previous standing best: IDF1 `0.655240` / HOTA `0.518652` / AssA `0.534359`.
- New protocol standing best: IDF1 `0.655378` / HOTA `0.518798` / AssA `0.534546`.
- Delta: `+0.000138` IDF1. This is a real but tiny win, still far below the e2e target `0.700000`.
- Observed non-primary variant: `density_simple` reached IDF1 `0.655385`.

## Ablation Table

| case | split shape | pair F1 / P / R | raw IDF1 | primary density IDF1 | HOTA | AssA | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `base_raw` | base, no new split | 0.769661 / 0.815773 / 0.728482 | 0.653071 |  | 0.516882 | 0.532486 | baseline |
| `k2_single` | 1 comp / k=2 / 172 tracklets / 2 parts | 0.769582 / 0.816131 / 0.728057 | 0.653150 | 0.655323 | 0.518736 | 0.534464 | keep as weaker positive |
| `k3_single` | 1 comp / k=3 / 172 tracklets / 3 parts | 0.769367 / 0.816518 / 0.727364 | 0.653210 | 0.655378 | 0.518798 | 0.534546 | promote as protocol best |
| `k2_two` | 2 comp / k=2 / 330 tracklets / 4 parts | 0.768757 / 0.816177 / 0.726544 | 0.652854 | 0.655021 | 0.518425 | 0.534245 | reject; side effects dominate |

## Density Policy Check

| case | density_oracle_lite | density_simple | confidence_tail |
| --- | ---: | ---: | ---: |
| `k2_single` | 0.655323 | 0.655330 | 0.655313 |
| `k3_single` | 0.655378 | 0.655385 | 0.655368 |
| `k2_two` | 0.655021 | 0.655031 | 0.655014 |

## Error Context

The base diagnosis shows the current output is dominated by both large false merges and high false splits. The largest false-merge component has 262 tracklets, 6 GT parts, dominant GT fraction 0.602, and false-merge mass 790,173,339. The largest false-split GT is split across 26 predicted components with false-split mass 1,046,536,325. Softcut source-component splitting helps one false-merge component, but it does not solve the larger false-split budget.

## Interpretation

- The useful move is not more aggressive splitting. The two-component split loses raw IDF1 and remains below the single-component split after density filtering.
- The best local move is the 3-way split of one conflicted 172-tracklet component: it raises precision enough to survive density filtering, but recall loss keeps the gain tiny.
- Next direction should be a new evidence source or admission rule: e.g. detector-quality quarantine tied to identity-state commit/defer, or a source-component split followed by constrained false-split bridge repair using a new verifier.

## Artifacts

- `local_runs/no_anchor_softcut_source_component_split_summary_20260621.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_base_error_diagnosis/base_error_diagnosis.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_best_nonnoop_fullscore/best_nonnoop_softcut.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_best_nonnoop_fullscore/best_nonnoop_full_export.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_best_nonnoop_fullscore/best_nonnoop_density_filter.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_k3_red010_fullscore/softcut.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_k3_red010_fullscore/full.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_k3_red010_fullscore/density_filter.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_two_components_fullscore/softcut.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_two_components_fullscore/full.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_two_components_fullscore/density_filter.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/full.zip`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/density_primary.zip`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_two_components_fullscore_20260621/full.zip`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_two_components_fullscore_20260621/density_primary.zip`
