# No-Anchor Component-Rescue Old-Base R1-3 Refutation

Date: 2026-06-21

## Why This Was Run

Following the Deli AutoResearch distillation, I tested a structurally different
candidate family instead of another delivery-threshold sweep.  The probe uses
the component-graph rescue queue from 2026-06-20: multi-edge identity edits
that move hundreds of tracklets into visually plausible target components.

This is no-anchor.  GT is used only by the canonical DS1 evaluator after the
submission is produced.

Important caveat: this probe uses the old 2026-06-20
softcut/softoverlap base namespace:

`/mnt/localssd/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260620/no_anchor_recovered_softcut_then_softoverlap_base_assignments_20260620.csv`

It is therefore a structural probe / hard negative source, not a current-k3
production candidate.

## Result

Current promoted best:

`IDF1/HOTA/AssA = 0.655911 / 0.519311 / 0.534922`

The component-rescue probe does not beat it.

| rank | source rank | moved tracklets | accepted edges | raw IDF1 | raw HOTA | raw AssA | p0.5 IDF1 | p0.5 HOTA | p0.5 AssA | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `32` | `320` | `5` | `0.653177` | `0.517094` | `0.532626` | `0.655290` | `0.519022` | `0.534605` | reject |
| 2 | `31` | `326` | `5` | `0.653177` | `0.517094` | `0.532626` | `0.655290` | `0.519022` | `0.534605` | reject |
| 3 | `30` | `406` | `5` | `0.639705` | `0.509136` | `0.532805` | `0.642169` | `0.511309` | `0.534868` | reject |

Remote run directory:

`/mnt/localssd/vlincs_reid_runs/no_anchor_component_rescue_dedup_probe_r1_3_fullscore_20260621`

Local mirrors:

- `local_runs/remote_h100_test_3_20260621/no_anchor_component_rescue_dedup_probe_r1_3_fullscore_20260621/summary.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_component_rescue_dedup_probe_r1_3_fullscore_20260621/manifest_assignments.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_component_rescue_dedup_probe_r1_3_fullscore_20260621/rank*_*.json`

## Interpretation

The proxy predicted these rows around `0.6683`, but the canonical full scorer
lands at `0.6532` raw and `0.6553` after the fixed p0.5 delivery filter.
Rank 3 is much worse.  This is a useful negative result: broad old-base
multi-edge rescue has high side-effect risk, and the proxy/reviewer was too
optimistic for this family.

The p0.5 delivery filter remains useful but cannot rescue bad identity edits.
The best p0.5 result here is still `0.000621` below the current best.

## Next Direction

Use this as a hard negative for the AutoResearch opponent/side-effect critic.
The next structural pivot should generate candidates in the current-k3
namespace or regenerate detector/tracklet evidence directly, especially for:

- MCAM04 Tc6
- MCAM06 Tc6
- MCAM03 Tc8

Do not schedule more old-base component-rescue ranks unless they are first
translated into the current-k3 namespace and pass a side-effect opponent that
penalizes broad multi-edge relabeling.
