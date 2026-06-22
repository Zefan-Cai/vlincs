# No-Anchor Current-k3 Small-Fragment Combo Refutation

Date: 2026-06-21

## Why This Was Run

The previous current-k3 small-fragment probe found a near miss: the best
single 4-tracklet attachment reached p0.5 IDF1 `0.655486`, only `0.000425`
below the current best `0.655911`.  This run tested whether independent
near-miss fragments can stack into a positive end-to-end gain.

The candidate construction is no-anchor.  It composes already selected
small-fragment rows by no-GT metadata only and rejects combinations that reuse
the same source seqs.

## Candidate Construction

Input:

`local_runs/no_anchor_current_k3_component_graph_small_fragment_20260621/small_fragment.json`

Composer:

`kit/compose_no_anchor_small_fragment_combos.py`

Output:

`local_runs/no_anchor_current_k3_small_fragment_combo_20260621/combo.json`

The composer used the top 6 small-fragment rows, generated `17` non-conflicting
2-edit and 3-edit combinations, then selected top 10 by no-GT combo score.  I
full-scored ranks `1-5`.

| rank | source ranks | source comps | target comps | moved | combo score |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `[21, 27]` | `[61, 2328]` | `[50, 20]` | `8` | `0.703408` |
| 2 | `[21, 27, 22]` | `[61, 2328, 60]` | `[50, 20, 14]` | `12` | `0.691177` |
| 3 | `[21, 38]` | `[61, 2328]` | `[50, 32]` | `8` | `0.691039` |
| 4 | `[21, 22]` | `[61, 60]` | `[50, 14]` | `8` | `0.687007` |
| 5 | `[21, 38, 22]` | `[61, 2328, 60]` | `[50, 32, 14]` | `12` | `0.681618` |

## Full-Score Result

Current promoted best:

`IDF1/HOTA/AssA = 0.655911 / 0.519311 / 0.534922`

The combinations do not improve over the best single small-fragment near miss.

| rank | edits | moved | raw IDF1 | raw HOTA | raw AssA | p0.5 IDF1 | p0.5 HOTA | p0.5 AssA | verdict |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `61->50 + 2328->20` | `8` | `0.653387` | `0.517323` | `0.533003` | `0.655486` | `0.519236` | `0.534963` | reject |
| 2 | `61->50 + 2328->20 + 60->14` | `12` | `0.653381` | `0.517315` | `0.532994` | `0.655481` | `0.519228` | `0.534954` | reject |
| 3 | `61->50 + 2328->32` | `8` | `0.653387` | `0.517323` | `0.533003` | `0.655486` | `0.519236` | `0.534963` | reject |
| 4 | `61->50 + 60->14` | `8` | `0.653381` | `0.517315` | `0.532994` | `0.655481` | `0.519228` | `0.534954` | reject |
| 5 | `61->50 + 2328->32 + 60->14` | `12` | `0.653381` | `0.517315` | `0.532994` | `0.655481` | `0.519228` | `0.534954` | reject |

Remote run:

`/mnt/localssd/vlincs_reid_runs/no_anchor_current_k3_small_fragment_combo_r1_5_fullscore_20260621`

Local mirror:

`local_runs/remote_h100_test_3_20260621/no_anchor_current_k3_small_fragment_combo_r1_5_fullscore_20260621/`

## Interpretation

This closes the simple-combination variant.  The extra small-fragment edits are
nearly neutral relative to the top edit, but they do not create additive
identity gain.  The effective score ceiling for this small-fragment candidate
family remains p0.5 IDF1 `0.655486`, below the current best.

The useful signal is diagnostic: `61->50` is the dominant near-miss edit, while
`2328->20/32` and `60->14` do not materially change DS1 identity score when
added.  A side-effect critic should learn this as "safe but low-impact", not as
a production improvement.

## Next Direction

Stop spending full-score budget on simple current-k3 component-graph
combinations.  The next productive direction is to change the evidence source:

1. detector/tracklet regeneration or admission repair for weak videos MCAM04
   Tc6, MCAM06 Tc6, and MCAM03 Tc8; or
2. train a no-anchor impact/side-effect critic using broad-negative,
   small-nearmiss, and combo-neutral examples before scheduling more identity
   edits.
