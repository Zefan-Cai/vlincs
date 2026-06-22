# Learned-Proxy Pair Candidate Ranking

- candidates: `326`
- all candidates before full-score filter: `326`
- min pair F1: `0.74`
- include full-scored: `True`
- delivery filter: `{'enabled': False, 'min_output_tracklets': 7000.0, 'min_eval_tracklets': 7000.0, 'min_coverage_ratio': 0.7, 'only_applies_to_present_fields': True}`
- dropped by delivery filter: `0`

| rank | learned full IDF1 | pair F1 | known full | output | eval | coverage | mode | artifact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `1` | `0.656057` | `0.767329` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| `2` | `0.655826` | `0.764808` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `3` | `0.655559` | `0.758282` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `4` | `0.655334` | `0.764840` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `5` | `0.655193` | `0.764257` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `6` | `0.654632` | `0.770090` | `` | `` | `7289` | `` | `conflict_subcluster_reassign` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_pair_20260620.json` |
| `7` | `0.654624` | `0.756637` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `8` | `0.654526` | `0.771653` | `` | `` | `7289` | `` | `conflict_subcluster_reassign` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_pair_20260620.json` |
| `9` | `0.654468` | `0.771045` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_stricttarget_pair_20260620.json` |
| `10` | `0.654418` | `0.767215` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| `11` | `0.654299` | `0.764062` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `12` | `0.654078` | `0.758632` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `13` | `0.654011` | `0.773540` | `0.654009` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| `14` | `0.653854` | `0.772654` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |
| `15` | `0.653837` | `0.775064` | `0.653541` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| `16` | `0.653703` | `0.774082` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |
| `17` | `0.653655` | `0.772399` | `` | `` | `7289` | `` | `conflict_subcluster_reassign` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_narrow_pair_20260620.json` |
| `18` | `0.653480` | `0.769774` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_tiny_pair_20260620.json` |
| `19` | `0.653478` | `0.772127` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_stricttarget_pair_20260620.json` |
| `20` | `0.653462` | `0.768930` | `` | `7487` | `7289` | `` | `cannotlink_nms_singleton` | `local_runs/remote_h100_test_3_20260619/no_anchor_cannotlink_nms_singleton_relaxed_pair_20260619.json` |
