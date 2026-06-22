# No-Anchor Referee-Pruned Crossqueue Full-Score Refutation

Date: 2026-06-21

## Why This Branch Was Tested

After the MCAM04/06 local-track and temporal-continuity branch failed to improve full-score, I executed the next Deli-style ready candidate queue instead of sweeping another local threshold family. The queue was a `referee_pruned_crossqueue_singleedge68_localized_island` portfolio: it proposed component moves that looked safe under the no-GT proxy and model-side pair metrics.

The hypothesis was:

- if source-local/temporal repairs are too small, then high-mass component moves selected by a referee-pruned proxy may recover false splits;
- if the proxy is calibrated, selected rows with predicted full IDF1 around `0.662` to `0.667` should beat the standing verified IDF1 `0.655240`.

## Execution Notes

Remote scoring was restored on `h100-test-3`. `scp` was unstable and repeatedly closed at `0%`, so I used a chunked base64 uploader over normal SSH commands to place:

- `kit/no_anchor_fullscore_scheduler.py`
- `local_runs/no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json`

Then I materialized and full-scored all four selected ranks with:

- base assignment: `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`
- rank1 run: `/mnt/localssd/vlincs_reid_runs/no_anchor_referee_pruned_crossqueue_rank1_fullscore_20260621`
- rank2-4 run: `/mnt/localssd/vlincs_reid_runs/no_anchor_referee_pruned_crossqueue_ranks2_4_fullscore_20260621`

All rows remain no-anchor. GT is used only by the canonical evaluator.

## Full-Score Results

Standing best before this branch:

| metric | value |
| --- | ---: |
| IDF1 | 0.655240 |
| HOTA | 0.518652 |
| AssA | 0.534359 |

Referee-pruned crossqueue results:

| selection rank | moved tracklets | target components | pair F1 | proxy predicted IDF1 | full IDF1 | HOTA | AssA |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 28 | `6,7,15` | 0.768959 | 0.665788 | 0.650714 | 0.514125 | 0.529500 |
| 2 | 241 | `7,68` | 0.769301 | 0.664389 | 0.626798 | 0.494799 | 0.518994 |
| 3 | 261 | `6,7,15,68` | 0.769205 | 0.666790 | 0.624978 | 0.492621 | 0.516447 |
| 4 | 253 | `6,15,68` | 0.769356 | 0.662361 | 0.625017 | 0.492936 | 0.517068 |

Best row in this branch:

- rank1 IDF1 `0.650714`
- delta versus standing best: `-0.004526`

## Interpretation

This is a strong negative result. The proxy ranked all four rows as plausible gains, but canonical full-score rejected every one. The failure is especially clear for ranks 2-4: model-side pair F1 stays around `0.769`, yet full IDF1 collapses to about `0.625`.

That means pair-level quality is not enough for this family. Large component moves introduce end-to-end side effects that the current proxy does not price:

- false merges across a high-mass component,
- detector/tracklet coverage shifts that alter DetRe and HOTA,
- namespace side effects from moving hundreds of tracklets at once,
- component-local gains that damage global identity consistency.

This also explains why continuing to add crossqueue merge candidates is unlikely to reach e2e `0.70`.

## Deli-Style Verdict

Negative result accepted as evidence:

- proposer: referee-pruned crossqueue component repair;
- referee/proxy: predicted full-score gain;
- evaluator: canonical DS1 full-score;
- opponent verdict: proxy overestimates large moves, branch rejected.

## Next Direction

The next structural branch should shift from "find more merges" to "prevent harmful commitments":

- no-GT admission/quarantine before a component move is allowed;
- commit/defer/force delivery split instead of forced global ID for every tracklet;
- false-merge guard using cannot-link, same-frame overlap, and target impurity;
- side-effect proxy trained to penalize moved-tracklet mass and namespace churn;
- per-video detector-quality admission for MCAM04/06 rather than temporal stitching.

No goal completion is claimed. The global-id pair model remains above the `0.70` target, but verified no-anchor end-to-end IDF1 remains `0.655240`.
