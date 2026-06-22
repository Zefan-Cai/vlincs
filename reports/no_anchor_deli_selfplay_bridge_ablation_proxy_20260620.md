# No-Anchor Assignment Summary Proxy

This is a no-GT scheduler/referee aid. It is not completion evidence.

- training rows: `10`
- raw label observations: `12`
- candidate assignments: `141`
- skipped candidate assignments: `3`
- min candidate tracklets/videos: `7000.0` / `10.0`
- LOOCV alpha: `100.0`
- LOOCV MAE: `0.00023707598772491068`
- LOOCV RMSE: `0.0003025984210775069`
- LOOCV corr: `0.5536579878793966`

## Top Candidates

| rank | predicted full IDF1 | known full IDF1 | labeled | assignment |
| ---: | ---: | ---: | --- | --- |
| 1 | `0.664464` | `None` | `false` | `local_runs/no_anchor_local_pervideo_source_selector_balanced_20260620_assignments.csv` |
| 2 | `0.663957` | `None` | `false` | `local_runs/no_anchor_deli_selfplay_bridge_ablation_20260620/rank01_balanced_plus_provisional_44_to_8_assignments.csv` |
| 3 | `0.663641` | `None` | `false` | `local_runs/no_anchor_deli_selfplay_bridge_ablation_20260620/rank03_balanced_plus_quarantine_17_to_47_assignments.csv` |
| 4 | `0.663442` | `None` | `false` | `local_runs/no_anchor_local_pervideo_source_selector_conservative_20260620_assignments.csv` |
| 5 | `0.661941` | `None` | `false` | `local_runs/no_anchor_deli_selfplay_bridge_ablation_20260620/rank02_balanced_plus_quarantine_13_to_2_assignments.csv` |
| 6 | `0.660916` | `None` | `false` | `local_runs/no_anchor_deli_selfplay_bridge_ablation_20260620/rank99_balanced_plus_all_noncommitted_bridge_assignments.csv` |
| 7 | `0.655650` | `None` | `false` | `local_runs/no_anchor_edge_table_focused_recovered_base_local_export_20260620/assignments/rank01_edge_table_island_merge_no_anchor_edge_table_island_merge_focused_pair_20260620_assignments.csv` |
| 8 | `0.654289` | `None` | `false` | `local_runs/no_anchor_local_pervideo_consensus_selector_20260620_assignments.csv` |
| 9 | `0.654020` | `None` | `false` | `local_runs/no_anchor_crossqueue_singleedge68_localized_island_local_export_20260620/assignments/rank06_hub_bridge_portfolio_no_anchor_crossqueue_singleedge68_localized_island_portfolio_candidates_20260620_assignments.csv` |
| 10 | `0.654020` | `None` | `false` | `local_runs/no_anchor_crossqueue_singleedge68_localized_island_sample_export_20260620/assignments/rank06_hub_bridge_portfolio_no_anchor_crossqueue_singleedge68_localized_island_portfolio_candidates_20260620_assignments.csv` |

## Training Labels

- `0.653823` predicted `0.653548` from `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_top_assignments_20260620.csv` (conflict_subcluster_reassign_candidate_search)
- `0.653823` predicted `0.653548` from `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_unique_top_assignments_20260620.csv` (conflict_subcluster_reassign_candidate_search)
- `0.653823` predicted `0.653548` from `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_top1_assignments_20260620.csv` (conflict_subcluster_reassign)
- `0.653724` predicted `0.653566` from `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_narrow_top1_assignments_20260620.csv` (conflict_subcluster_reassign)
- `0.653686` predicted `0.653563` from `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_top1_assignments_20260620.csv` (None)
- `0.653541` predicted `0.653544` from `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_top_assignments_20260620.csv` (conflict_subcluster_reassign_candidate_search)
- `0.653097` predicted `0.653148` from `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_top1_assignments_20260620.csv` (edge_table_target_repair)
- `0.653084` predicted `0.653275` from `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_top1_assignments_20260620.csv` (edge_table_target_repair)
- `0.652957` predicted `0.653548` from `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_margin003_top1_assignments_20260620.csv` (conflict_subcluster_reassign)
- `0.652948` predicted `0.653216` from `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_target_repair_fallback_single_assignments_20260620.csv` (edge_table_target_repair)
