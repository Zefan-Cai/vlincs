# No-Anchor Counter-Target Verifier Refutation

Date: 2026-06-21

## Verdict

Rejected as a production admission gate, kept as a useful opponent/referee.

The visual counter-target verifier is a good reducer: it rejected the prior k3
mass-bridge failure and reduced 18 compact candidate edges to 1 accepted edge.
But the one accepted edge still failed canonical DS1 full-score:

| Run | IDF1 | HOTA | AssA | Decision |
|---|---:|---:|---:|---|
| Standing best, k3 softcut + density_oracle_lite | 0.655378 | 0.518798 | 0.534546 | keep |
| Verified edge raw full-score | 0.652833 | 0.516616 | 0.532353 | reject |
| Verified edge + density_oracle_lite | 0.654999 | 0.518378 | 0.534212 | reject |
| Verified edge + density_simple | 0.655006 | 0.518387 | 0.534214 | observed, still below standing |
| Verified edge + confidence_tail | 0.654989 | 0.518361 | 0.534197 | reject |

This is a Deli-style honest downward score: the agent found a plausible
candidate, built a stronger referee, and then let the full evaluator mark the
result down.

## What Changed

Added `kit/audit_no_anchor_countertarget_verifier.py`.

The script is no-GT/no-anchor. It reads:

- current assignment CSV,
- candidate rows with `accepted_preview`,
- multiple feature NPZs,
- DB tracklet metadata only for indexing.

For every proposed source island -> target component edge, it asks:

1. Is the target component ranked first among all current assignment components?
2. Is the target margin over the best alternative large enough?
3. Do enough feature views agree?
4. Is the weighted target-vs-countertarget margin positive enough?

Current gate:

| Gate | Threshold |
|---|---:|
| max accepted target rank | 1 |
| min per-view margin | 0.03 |
| min weighted margin | 0.03 |
| min rank vote | 0.75 |
| min margin vote | 0.50 |

## Audit Results

### Prior k3 mass-bridge edge

The previous mass-bridge edge was source component `21` -> target component
`0`, moving seqs `2232, 2270, 2308, 2374, 2415, 2452, 2488, 2553`.

| View | target rank | target score | best alt component | best alt score | margin |
|---|---:|---:|---:|---:|---:|
| primary | 1 | 0.873492 | 40 | 0.703424 | 0.170068 |
| colorhist | 1 | 0.995547 | 55 | 0.989553 | 0.005994 |
| dino | 1 | 0.633814 | 26 | 0.626543 | 0.007271 |
| posecolor | 5 | 0.973117 | 26 | 0.980858 | -0.007741 |

Verifier output: `reject_countertarget`.

Reason: primary likes the target, but non-primary views have tiny or negative
margins. Rank vote was `0.75`, margin vote only `0.25`.

### Compact historical candidate pool

I compacted earlier scheduler rows into a 5.8KB no-GT verifier input containing
10 rows / 18 unique candidate edges. The verifier accepted only 1 edge.

| source component | target component | weighted margin | rank vote | margin vote | verdict |
|---:|---:|---:|---:|---:|---|
| 13 | 49 | 0.079118 | 0.75 | 0.50 | accept |
| 21 | 0 | 0.074511 | 0.75 | 0.25 | reject |
| 33 | 55 | 0.054091 | 0.50 | 0.50 | reject |
| 40 | 21 | 0.026933 | 0.50 | 0.25 | reject |
| 32 | 15 | 0.010745 | 0.25 | 0.25 | reject |

## Accepted Edge Case

Chosen edge:

- source component: `13`
- target component: `49`
- moved tracklets: `8`
- source seqs: `3408, 3518, 4311, 4889, 4938, 5026, 5062, 5158`
- target gid after replay: `96000049`

Verifier evidence:

| View | target rank | target score | best alt component | best alt score | margin |
|---|---:|---:|---:|---:|---:|
| primary | 1 | 0.734447 | 24 | 0.572960 | 0.161487 |
| dino | 1 | 0.617574 | 55 | 0.550716 | 0.066857 |
| posecolor | 1 | 0.961354 | 3 | 0.957977 | 0.003378 |
| colorhist | 2 | 0.978781 | 51 | 0.981305 | -0.002524 |

This passed the current visual gate because primary and DINO supplied real
counter-target margin, while posecolor/colorhist were weak but not catastrophic.

## Why It Still Failed

Full score localized the damage to `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6`:

| Video | raw base IDF1 | verified-edge raw IDF1 | delta |
|---|---:|---:|---:|
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.559050 | 0.558128 | -0.000922 |

No-GT spatiotemporal diagnostic shows the source and most competitors live in
the same camera/video and overlapping local time neighborhood:

| group | video | seq count | frame span |
|---|---|---:|---|
| source | MCAM04 Tc6 | 8 | 35232-46637 |
| target_colorhist | MCAM04 Tc6 | 3 | 36212-45808 |
| target_dino | MCAM04 Tc6 | 3 | 17758-45795 |
| target_posecolor | MCAM04 Tc6 | 2 | 45019-45795 |
| alt_posecolor | MCAM04 Tc6 | 3 | 41448-43127 |
| alt_colorhist | MCAM04 Tc6 | 3 | 32282-35281 |

Interpretation: this is exactly the crowded same-camera cell where visual
nearest-neighbor evidence is likely to over-merge. A production verifier needs a
temporal/local-track opponent, not just more visual threshold tuning.

## Next Structural Pivot

Use the accepted-but-bad edge as a hard negative for a self-play opponent.
I implemented the first version of that opponent in the same verifier as an
optional temporal cannot-link gate:

```bash
--max-same-video-overlap-frames 0
--temporal-local-window-frames 1500
```

With this gate enabled:

| Audit input | candidate edges | accepted before temporal | accepted after temporal |
|---|---:|---:|---:|
| k3 mass-bridge search | 6 | 0 | 0 |
| compact historical pool | 18 | 1 | 0 |

The previously accepted bad edge `13 -> 49` is now rejected as
`reject_temporal_overlap`. The largest same-camera source/target overlap is
`247` frames. One concrete overlap example:

| source seq | source span | target seq | target span | video | overlap |
|---:|---|---:|---|---|---:|
| 3408 | 35232-35531 | 3372 | 35062-35361 | MCAM04 Tc6 | 130 |

This converts the full-score failure into a reusable no-GT rule.

Next candidate-generation plan:

1. Keep visual counter-target verifier as stage 1 reducer.
2. Add same-camera temporal/local-track opponent:
   - penalize source islands whose target support overlaps the same local time
     band but lacks sequential continuity;
   - require target support to explain the source's preceding/following local
     tracklets better than competing components;
   - add cannot-link/near-coexistence checks inside MCAM04 Tc6.
3. Add detector-quality admission:
   - do not merge if the source island lives in a high-density low-confidence
     cell unless temporal continuity also supports the merge.
4. Only materialize edges that pass both visual counter-target and temporal
   opponent.

## Artifacts

- `kit/audit_no_anchor_countertarget_verifier.py`
- `local_runs/no_anchor_countertarget_inputs_20260621/no_anchor_countertarget_compact_edges_20260621.json`
- `local_runs/no_anchor_countertarget_audit_k3_mass_20260621/audit.json`
- `local_runs/no_anchor_countertarget_audit_k3_mass_20260621/audit.csv`
- `local_runs/no_anchor_countertarget_audit_compact_edges_20260621/audit.json`
- `local_runs/no_anchor_countertarget_audit_compact_edges_20260621/audit.csv`
- `local_runs/no_anchor_countertarget_audit_k3_mass_temporal_20260621/audit.json`
- `local_runs/no_anchor_countertarget_audit_compact_edges_temporal_20260621/audit.json`
- `local_runs/no_anchor_countertarget_verified_edge_fullscore_20260621/verified_edge_full.json`
- `local_runs/no_anchor_countertarget_verified_edge_fullscore_20260621/verified_edge_density_filter.json`
- `local_runs/no_anchor_countertarget_verified_edge_fullscore_20260621/verified_edge_no_gt_spacetime_diagnostic.json`
- `local_runs/no_anchor_countertarget_verified_edge_fullscore_20260621_summary.json`
- remote zip: `/mnt/localssd/vlincs_reid_runs/no_anchor_countertarget_verified_edge_fullscore_20260621/verified_edge_submission.zip`
- remote density zip: `/mnt/localssd/vlincs_reid_runs/no_anchor_countertarget_verified_edge_fullscore_20260621/verified_edge_density_primary.zip`
