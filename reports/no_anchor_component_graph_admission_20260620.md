# No-Anchor Component-Graph Admission

- source: `local_runs/no_anchor_component_graph_low_vote_rescue_broad_20260620.json`
- raw rows: `8`
- unordered pair groups: `4`
- committed probes: `1`
- provisional probes: `1`
- quarantined groups: `2`
- rejected direction duplicates: `4`

| status | input | source | target | moved | best | mean | min-view | vote | overlap | score | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `provisional_probe` | `6` | `44` | `8` | `130` | `0.765438` | `0.439675` | `0.701892` | `0.300` | `0.015240` | `0.612289` | `medium_top_visual_match;multi_view_support;low_same_video_overlap;larger_or_equal_target;spiky_low_vote_evidence` |
| `committed_probe` | `2` | `31` | `24` | `44` | `0.841304` | `0.490128` | `0.721598` | `0.300` | `0.016901` | `0.609516` | `strong_top_visual_match;multi_view_support;low_same_video_overlap;larger_or_equal_target;spiky_low_vote_evidence` |
| `quarantine` | `5` | `13` | `2` | `223` | `0.719889` | `0.407463` | `0.697498` | `0.300` | `0.011163` | `0.592957` | `borderline_multi_view_support;low_same_video_overlap;larger_or_equal_target;weak_top_visual_match;spiky_low_vote_evidence` |
| `quarantine` | `8` | `17` | `47` | `140` | `0.724181` | `0.444104` | `0.691919` | `0.300` | `0.014423` | `0.589080` | `borderline_multi_view_support;low_same_video_overlap;larger_or_equal_target;weak_top_visual_match;spiky_low_vote_evidence` |

## Interpretation

- `committed_probe` means safe to spend canonical full-score budget when the scorer is reachable.
- `provisional_probe` keeps plausible but under-verified edges visible without forcing them into the next submission.
- `quarantine` rows carry explicit no-GT counter-evidence and should not be scheduled until a new verifier addresses that risk.
