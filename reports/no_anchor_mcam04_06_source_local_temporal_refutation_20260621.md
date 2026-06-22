# No-Anchor MCAM04/06 Source-Local + Temporal Refutation

Date: 2026-06-21

## Deli AutoResearch Distillation Applied

I treated the Deli AutoResearch/self-play thread as an operating protocol, not as a model recipe:

- Keep state file-backed and resumable.
- Let negative results count as real evidence.
- Separate proposer, referee/opponent, and evaluator.
- After stale iterations, pivot the structural hypothesis instead of sweeping the same threshold family harder.
- Execute prepared candidates on canonical full-score when remote scoring is available.

For this VLINCS no-anchor loop, that means the current assignment/source-selector family is considered saturated below the standing verified e2e result, so this turn tested the next structural claim: MCAM04/06 failures may be local-track or temporal-continuity failures.

## Standing Baseline

- Verified standing best e2e: IDF1 `0.655240`, HOTA `0.518652`, AssA `0.534359`
- Current model-side pair quality remains above target:
  - pair F1 `0.775234`
  - pair precision `0.820504`
  - pair recall `0.734698`

The edge-table focused base used in this turn has pair F1 `0.772686`.

## Experiment 1: Local-Track Dominant Relink

Artifact: `local_runs/remote_mcam04_06_temporal_20260621/localtrack_group_diagnostic.json`

Question: do MCAM04/06 contain `(video, local_track_id)` groups whose tracklets were split across multiple predicted global IDs?

Result:

| source assignment | local-track groups | multi-pred groups | multi-pred tracklets |
| --- | ---: | ---: | ---: |
| edge-table focused | 3246 | 0 | 0 |
| balanced selector | 3246 | 0 | 0 |

Conclusion: this is a clean no-op. The current tracklet source format has one predicted ID per local-track group in MCAM04/06, so dominant local-track relink has no editable object. This refutes the simple "same tracker local id should be fused" hypothesis.

## Experiment 2: Targeted Temporal Relink

Code change: added `--only-videos` to `kit/no_anchor_assignment_video_temporal_relink_sweep.py`, so temporal edges can be restricted to MCAM04/06.

Artifacts:

- `local_runs/remote_mcam04_06_temporal_20260621/edge_temporal_target_mcam04_06_wide_proxy_summary.json`
- `local_runs/remote_mcam04_06_temporal_20260621/edge_temporal_target_mcam04_06_ultrawide_proxy_summary.json`
- `local_runs/remote_mcam04_06_temporal_20260621/edge_temporal_target_mcam04_06_ultrawide_top_full_summary.json`

The normal guarded temporal grid found almost no usable edges:

- top guarded row: `candidate_edges=1`, `accepted_edges=0`

The ultrawide diagnostic found a tiny repair:

- candidate edges: `97`
- accepted edges: `3`
- accepted by video: MCAM04 `2`, MCAM06 `1`
- mean accepted app sim: `0.578402`
- pair F1: `0.772685`, essentially unchanged from `0.772686`

Canonical full-score for the top ultrawide temporal rule:

| metric | value |
| --- | ---: |
| IDF1 | 0.653827 |
| HOTA | 0.517796 |
| AssA | 0.533344 |
| MCAM04 Tc6 IDF1 | 0.560501 |
| MCAM06 Tc6 IDF1 | 0.606895 |

This does not beat the standing best IDF1 `0.655240`, nor the previous edge-table density result `0.654041`.

Gate artifact: `local_runs/no_anchor_result_gate_after_mcam04_06_temporal_20260621.json`

- `pass_joint=false`
- best e2e among this batch: IDF1 `0.654041`
- best joint row with model-side pair metrics: temporal ultrawide, pair F1 `0.772685`, full IDF1 `0.653827`

## Refutation

The MCAM04/06 failure is not explained by:

1. local-track ID fragmentation inside assignment rows, because there are no multi-pred local-track groups;
2. ordinary temporal-adjacent appearance stitching, because even ultrawide temporal edges only produce three edits and no full-score gain.

## Next Direction

The next productive branch should stop treating MCAM04/06 as short-term temporal continuity. The evidence points back to higher-mass identity-component structure:

- component/global false-split repair with side-effect modeling,
- detector quality/admission on MCAM04/06,
- component-level candidate retrieval that is not constrained to temporal adjacency,
- case visualization for the few accepted temporal edges to understand why their visual scores look plausible but do not move IDF1.

No goal completion is claimed: global-id model metrics are above 0.70, but verified e2e remains below 0.70.
