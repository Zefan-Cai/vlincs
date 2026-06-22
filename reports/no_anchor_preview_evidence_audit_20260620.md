# No-Anchor Accepted-Preview Evidence Audit

Current best full IDF1: `0.655240`

Preview rows: `5051`
Full-labelled preview rows: `38`
Labelled rows above current best: `0`

## By Mode

| mode | preview rows | labelled | max full | above best | max pair-mass proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| assignment_softcut_split | 183 | 0 |  | 0 |  |
| conflict_subcluster_reassign | 1410 | 5 | 0.653823 | 0 | 2016 |
| conflict_subcluster_reassign_candidate_search | 658 | 29 | 0.654009 | 0 | 2568 |
| edge_table_island_merge | 2364 | 0 |  | 0 | 1104 |
| provisional_subcluster_relink | 336 | 4 | 0.652265 | 0 |  |
| video_temporal_relink | 100 | 0 |  | 0 |  |

## Top Full-Labelled Preview Rows

| full | pair F1 | mode | source -> target | source size | target size | artifact |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 0.654009 | 0.77354 | conflict_subcluster_reassign_candidate_search | `21 -> 0` | 8 | 199 | `no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| 0.654009 | 0.77354 | conflict_subcluster_reassign_candidate_search | `40 -> 21` | 8 | 227 | `no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| 0.654009 | 0.77354 | conflict_subcluster_reassign_candidate_search | `32 -> 15` | 8 | 158 | `no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| 0.654009 | 0.77354 | conflict_subcluster_reassign_candidate_search | `4 -> 6` | 8 | 172 | `no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| 0.653823 | 0.772654 | conflict_subcluster_reassign | `21 -> 0` | 8 | 199 | `no_anchor_conflict_reassign_strict_top1_full_20260620.json` |
| 0.653724 | 0.772399 | conflict_subcluster_reassign | `26 -> 21` | 8 | 227 | `no_anchor_conflict_reassign_narrow_top1_full_20260620.json` |
| 0.653541 | 0.775234 | conflict_subcluster_reassign_candidate_search | `48 -> 9` | 8 | 77 | `no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |
| 0.652957 | 0.770554 | conflict_subcluster_reassign | `36 -> 5` | 8 | 237 | `no_anchor_conflict_reassign_margin003_top1_full_20260620.json` |
| 0.652265 | 0.767775 | provisional_subcluster_relink | `22 -> ` | 8 |  | `no_anchor_provisional_relink_narrow_20260619.json` |
| 0.652265 | 0.767775 | provisional_subcluster_relink | `27 -> ` | 8 |  | `no_anchor_provisional_relink_narrow_20260619.json` |
| 0.652265 | 0.767775 | provisional_subcluster_relink | `2 -> ` | 8 |  | `no_anchor_provisional_relink_narrow_20260619.json` |
| 0.652265 | 0.767775 | provisional_subcluster_relink | `3 -> ` | 8 |  | `no_anchor_provisional_relink_narrow_20260619.json` |

## Highest Unlabelled Pair-Mass Preview Rows

| pair-mass proxy | pair F1 | mode | source -> target | source size | target size | artifact |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 2568 | 0.763836 | conflict_subcluster_reassign_candidate_search | `15 -> 14` | 12 | 214 | `no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| 2064 | 0.766254 | conflict_subcluster_reassign_candidate_search | `4 -> 6` | 12 | 172 | `no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| 2064 | 0.763836 | conflict_subcluster_reassign_candidate_search | `19 -> 6` | 12 | 172 | `no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| 2016 | 0.76028 | conflict_subcluster_reassign | `10 -> 26` | 8 | 252 | `no_anchor_conflict_reassign_margin003_top1_full_20260620.json` |
| 1984 | 0.765617 | conflict_subcluster_reassign | `38 -> 3` | 8 | 248 | `no_anchor_conflict_reassign_margin003_top1_full_20260620.json` |
| 1896 | 0.766254 | conflict_subcluster_reassign_candidate_search | `32 -> 15` | 12 | 158 | `no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| 1896 | 0.770554 | conflict_subcluster_reassign | `36 -> 5` | 8 | 237 | `no_anchor_conflict_reassign_margin003_top1_full_20260620.json` |
| 1816 | 0.775234 | conflict_subcluster_reassign_candidate_search | `40 -> 21` | 8 | 227 | `no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |
| 1816 | 0.772399 | conflict_subcluster_reassign | `26 -> 21` | 8 | 227 | `no_anchor_conflict_reassign_narrow_pair_20260620.json` |
| 1784 | 0.766205 | conflict_subcluster_reassign_candidate_search | `2 -> 13` | 8 | 223 | `no_anchor_conflict_reassign_candidate_search_tiny_pair_20260620.json` |
| 1784 | 0.7656 | conflict_subcluster_reassign | `2 -> 13` | 8 | 223 | `no_anchor_conflict_reassign_margin003_top1_full_20260620.json` |
| 1776 | 0.757677 | conflict_subcluster_reassign | `7 -> 37` | 8 | 222 | `no_anchor_conflict_reassign_margin003_top1_full_20260620.json` |
| 1728 | 0.763836 | conflict_subcluster_reassign_candidate_search | `9 -> 7` | 8 | 216 | `no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| 1728 | 0.765617 | conflict_subcluster_reassign | `37 -> 7` | 8 | 216 | `no_anchor_conflict_reassign_margin003_top1_full_20260620.json` |
| 1712 | 0.772359 | conflict_subcluster_reassign_candidate_search | `15 -> 14` | 8 | 214 | `no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |
| 1648 | 0.757677 | conflict_subcluster_reassign | `3 -> 38` | 8 | 206 | `no_anchor_conflict_reassign_margin003_top1_full_20260620.json` |
| 1592 | 0.775234 | conflict_subcluster_reassign_candidate_search | `21 -> 0` | 8 | 199 | `no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |
| 1592 | 0.770554 | conflict_subcluster_reassign | `21 -> 0` | 8 | 199 | `no_anchor_conflict_reassign_margin003_top1_full_20260620.json` |
| 1392 | 0.76228 | conflict_subcluster_reassign_candidate_search | `20 -> 50` | 12 | 116 | `no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| 1376 | 0.772359 | conflict_subcluster_reassign_candidate_search | `4 -> 6` | 8 | 172 | `no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |

## Interpretation

- Existing full-labelled preview edits do not prove a route beyond the current best if `above best` is zero.
- High pair-mass unlabelled rows are candidates for future full-score slots only if their source artifacts are no-anchor and their accepted previews pass cannot-link/provenance checks.
- This audit is an ablation/routing tool; it does not train or select with GT.
