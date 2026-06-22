# Learned-Proxy Pair Candidate Ranking

- candidates: `300`
- all candidates before full-score filter: `318`
- min pair F1: `0.74`
- include full-scored: `False`
- delivery filter: `{'enabled': True, 'min_output_tracklets': 7000.0, 'min_eval_tracklets': 7000.0, 'min_coverage_ratio': 0.7, 'only_applies_to_present_fields': True}`
- dropped by delivery filter: `68`

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
| `13` | `0.653854` | `0.772654` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |
| `14` | `0.653703` | `0.774082` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |
| `15` | `0.653655` | `0.772399` | `` | `` | `7289` | `` | `conflict_subcluster_reassign` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_narrow_pair_20260620.json` |
| `16` | `0.653480` | `0.769774` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_tiny_pair_20260620.json` |
| `17` | `0.653478` | `0.772127` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_stricttarget_pair_20260620.json` |
| `18` | `0.653462` | `0.768930` | `` | `7487` | `7289` | `` | `cannotlink_nms_singleton` | `local_runs/remote_h100_test_3_20260619/no_anchor_cannotlink_nms_singleton_relaxed_pair_20260619.json` |
| `19` | `0.653296` | `0.758097` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `20` | `0.653267` | `0.768743` | `` | `7487` | `7289` | `` | `cannotlink_nms_singleton` | `local_runs/remote_h100_test_3_20260619/no_anchor_cannotlink_nms_singleton_relaxed_pair_20260619.json` |
| `21` | `0.653130` | `0.757892` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `22` | `0.653091` | `0.758006` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `23` | `0.653029` | `0.770667` | `` | `` | `7289` | `` | `assignment_multiview_merge` | `local_runs/remote_h100_test_3_20260620/no_anchor_state_color_forced_multiview_merge_pair_20260620.json` |
| `24` | `0.652479` | `0.757008` | `` | `7538` | `7338` | `` | `louvain` | `local_runs/remote_h100_test_3_20260619/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |
| `25` | `0.652457` | `0.766254` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| `26` | `0.652359` | `0.766205` | `` | `` | `7289` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_tiny_pair_20260620.json` |
| `27` | `0.652038` | `0.769668` | `` | `` | `7289` | `` | `assignment_softcut_split` | `local_runs/remote_h100_test_3_20260619/no_anchor_softcut_split_relaxed_pair_20260619.json` |
| `28` | `0.652026` | `0.769760` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `29` | `0.652025` | `0.769755` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_paironly_20260620.json` |
| `30` | `0.652025` | `0.769756` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `31` | `0.652021` | `0.769751` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `32` | `0.652021` | `0.769753` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_target_repair_cached_pair_20260620.json` |
| `33` | `0.652020` | `0.769745` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_paironly_20260620.json` |
| `34` | `0.652019` | `0.769742` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `35` | `0.652019` | `0.769750` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_paironly_20260620.json` |
| `36` | `0.652018` | `0.769741` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_paironly_20260620.json` |
| `37` | `0.652018` | `0.769739` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `38` | `0.652018` | `0.769743` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `39` | `0.652017` | `0.769738` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `40` | `0.652017` | `0.769735` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_paironly_20260620.json` |
| `41` | `0.652017` | `0.769755` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_target_repair_cached_pair_20260620.json` |
| `42` | `0.652017` | `0.769739` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `43` | `0.652016` | `0.769736` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `44` | `0.652016` | `0.769754` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_target_repair_cached_pair_20260620.json` |
| `45` | `0.652016` | `0.769747` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `46` | `0.652016` | `0.769752` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_target_repair_cached_pair_20260620.json` |
| `47` | `0.652016` | `0.769754` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_target_repair_cached_pair_20260620.json` |
| `48` | `0.652015` | `0.769740` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json` |
| `49` | `0.652015` | `0.769753` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_target_repair_cached_pair_20260620.json` |
| `50` | `0.652014` | `0.769753` | `` | `` | `7289` | `` | `edge_table_target_repair` | `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_target_repair_cached_pair_20260620.json` |
