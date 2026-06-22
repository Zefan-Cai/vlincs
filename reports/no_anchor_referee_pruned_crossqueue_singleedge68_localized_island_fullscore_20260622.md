# No-Anchor Referee-Pruned Crossqueue Full-Score Refutation

Date: 2026-06-22

## Why This Was Run

The Deli AutoResearch/self-play distillation says "ready means execute":
when a no-anchor proxy scheduler has a concrete candidate, run the canonical
DS1 scorer instead of stopping at a promising queue.  This experiment full-
scored the referee-pruned crossqueue/single-edge localized-island candidates
that had predicted full IDF1 around `0.662361-0.666790`.

Current standing no-anchor best:

- global-id pair model: F1/P/R = `0.775234 / 0.820504 / 0.734698`;
- end-to-end DS1: IDF1/HOTA/AssA = `0.655911 / 0.519311 / 0.534922`.

No anchors were used.  GT is used only by the canonical evaluator.

## Run

- remote: h100-test-3
- remote run dir:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_referee_pruned_crossqueue_singleedge68_localized_island_fullscore_20260622`
- local artifact mirror:
  `local_runs/remote_h100_test_3_20260622/no_anchor_referee_pruned_crossqueue_singleedge68_localized_island_fullscore_20260622`
- scheduler:
  `local_runs/no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json`
- base assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`
- ranks: `1,2,3,4`

## Full-Score Results

| rank | predicted full | pair F1 | moved tracklets | target components | IDF1 | HOTA | AssA | DetRe | DetPr | IDF1 - predicted | IDF1 - current best |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.665788 | 0.768959 | 28 | 6+7+15 | 0.650714 | 0.514125 | 0.529500 | 0.572723 | 0.753295 | -0.015074 | -0.005197 |
| 2 | 0.664389 | 0.769301 | 241 | 7+68 | 0.626798 | 0.494799 | 0.518994 | 0.551622 | 0.725698 | -0.037591 | -0.029113 |
| 3 | 0.666790 | 0.769205 | 261 | 6+7+15+68 | 0.624978 | 0.492621 | 0.516447 | 0.549856 | 0.723873 | -0.041812 | -0.030933 |
| 4 | 0.662361 | 0.769356 | 253 | 6+15+68 | 0.625017 | 0.492936 | 0.517068 | 0.549894 | 0.723914 | -0.037344 | -0.030894 |

Best rank is rank 1, but it is still below the current best by `0.005197`
IDF1.  The larger component `3 -> 68` move is a strong negative: whenever it is
present, IDF1 falls by about `0.029-0.031` against the current best.

## Best Rank Per-Video Metrics

| video | IDF1 | HOTA | AssA |
| --- | ---: | ---: | ---: |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.845793 | 0.764757 | 0.787740 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.808178 | 0.732419 | 0.782300 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.688387 | 0.580446 | 0.623006 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.626660 | 0.508952 | 0.549346 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.558658 | 0.445383 | 0.490265 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.710965 | 0.601047 | 0.639585 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.791599 | 0.697946 | 0.727424 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.606895 | 0.514826 | 0.596222 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.704148 | 0.590733 | 0.621206 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.763250 | 0.655372 | 0.674327 |

Against the current p0.5 weak-video oracle reference, even the best rank loses
on all three weak videos:

| video | current best IDF1 | rank 1 IDF1 | delta |
| --- | ---: | ---: | ---: |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.632479 | 0.626660 | -0.005819 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.564234 | 0.558658 | -0.005576 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.609699 | 0.606895 | -0.002804 |

## Verdict

This is a negative result.  The proxy/pair score is over-optimistic for this
old-base crossqueue component-reassignment family.  High pair F1 around
`0.769` did not transfer to DS1 IDF1; the canonical scorer exposed false-merge
or delivery side effects, especially for the large `3 -> 68` move.

Close this candidate family for now.  Per the AutoResearch stall rule, do not
continue with threshold tweaks inside this frame.  The next no-anchor branch
should change the evidence source or weak-label curriculum, not add more
crossqueue component rewrites on the same proxy.

## Next Direction

Recommended next branch:

1. keep the current DS1 best frozen at `0.655911`;
2. stop using pair-F1-only schedulers as promotion criteria;
3. build a stricter no-anchor weak-label curriculum for the sample/global-id
   model, with hard negatives from the failed `3 -> 68` and `32/4/9 -> 15/6/7`
   moves;
4. only re-enter DS1 full-score once sample identity evidence or candidate
   side-effect prediction improves materially.
