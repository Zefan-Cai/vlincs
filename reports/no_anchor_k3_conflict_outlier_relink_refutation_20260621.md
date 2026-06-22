# No-Anchor K3 Conflict-Outlier Relink Refutation

Date: 2026-06-21

## Question

After the current k3/density error decomposition, the next hypothesis was that
large mixed components contain a few bad conflict/outlier tracklets that should
be relinked into a better existing component rather than split into a new ID.

This run tests that hypothesis without anchors:

- selection evidence: current assignment, same-stream temporal cannot-link, and
  fused tracklet appearance centroids;
- no GT or anchors used for candidate generation;
- GT used only after materialization for pair/full diagnostics.

## New Proposer

Added:

- `kit/no_anchor_conflict_outlier_relink.py`

The proposer searches only tracklets already involved in an internal
same-stream conflict.  A candidate must have weak similarity to its current
component centroid, stronger similarity to a physically compatible target
component, and a positive target-source margin.

Remote run:

- candidates:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_conflict_outlier_relink_candidates_20260621/result.json`
- full-score:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_conflict_outlier_relink_fullscore_20260621/result.json`
- density:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_conflict_outlier_relink_density_20260621/k1_density_filter.json`

Local mirrors:

- `local_runs/no_anchor_k3_conflict_outlier_relink_candidates_20260621/result.json`
- `local_runs/no_anchor_k3_conflict_outlier_relink_fullscore_20260621/result.json`
- `local_runs/no_anchor_k3_conflict_outlier_relink_density_20260621/k1_density_filter.json`

## Candidate Summary

The broad no-score candidate run produced 324 config rows.  The strongest
no-GT config was:

- `min_source_size=32`
- `max_source_sim=0.76`
- `min_target_sim=0.72`
- `min_margin=0.02`
- `candidate_count=44`

Top accepted candidates for `top_k=8`:

| rank | seq | move | camera | margin | note |
|---:|---:|---|---|---:|---|
| 1 | `5828` | `10 -> 22` | MCAM04 | `0.849035` | strongest one-tracklet relink |
| 2 | `7232` | `11 -> 21` | MCAM06 | `0.630224` | high margin but lower delivery value |
| 3 | `6241` | `32 -> 20` | MCAM05 | `0.516315` | small-video edit |
| 4 | `2206` | `11 -> 21` | MCAM03 | `0.512972` | same target as rank 2 |
| 5 | `7305` | `10 -> 21` | MCAM06 | `0.433928` | starts to dilute signal |
| 8 | `3776` | `35 -> 61` | MCAM04 | `0.237595` | overlaps prior clothing-referee clue |

## Raw Full-Score

| top-k relinks | pair F1 | pair precision | pair recall | full IDF1 | HOTA | AssA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.769463` | `0.816638` | `0.727441` | `0.653258` | `0.517085` | `0.532738` |
| 2 | `0.769157` | `0.816260` | `0.727193` | `0.653113` | `0.516930` | `0.532600` |
| 4 | `0.769130` | `0.816212` | `0.727184` | `0.653093` | `0.516914` | `0.532592` |
| 8 | `0.768856` | `0.816195` | `0.726708` | `0.653141` | `0.516917` | `0.532553` |

Best raw row is `top_k=1`, moving only seq `5828` from component `10` to
component `22`.  It is a tiny positive over raw k3 (`0.653210 -> 0.653258`) but
well below the standing k3+density best `0.655378`.

## Density Recheck

For the best raw row, applying the same no-GT per-video filter policies gives:

| policy | IDF1 | HOTA | AssA | dropped rows |
|---|---:|---:|---:|---:|
| `density_oracle_lite` | `0.653468` | `0.517078` | `0.532839` | `34434` |
| `density_simple` | `0.653473` | `0.517084` | `0.532838` | `33685` |
| `confidence_tail` | `0.653369` | `0.516938` | `0.532694` | `36385` |

This remains below standing best:

| standing run | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| k3 + density primary | `0.655378` | `0.518798` | `0.534546` |

## Interpretation

The action primitive is valid but too weak:

- single-tracklet relink can be directionally positive;
- blindly increasing `top_k` dilutes the signal and lowers pair/full metrics;
- fused-centroid margin finds plausible outliers, but it does not prioritize
  high-delivery MCAM04/MCAM08 false-merge/false-split mass well enough.

Decision: keep `no_anchor_conflict_outlier_relink.py` as a fast proposer, but do
not promote this candidate family as a production improvement.

## Eval-Only Candidate Audit

I also ran a post-hoc GT audit of the top candidates.  This is diagnostic only:
it is not used for no-anchor candidate generation or production selection.

Artifact:

- `local_runs/no_anchor_k3_conflict_outlier_relink_fullscore_20260621/candidate_gt_audit.json`

The audit explains the top-k behavior:

| rank | seq | move | seq GT | source top GT | target top GT | verdict |
|---:|---:|---|---:|---:|---:|---|
| 1 | `5828` | `10 -> 22` | `48` | `7` | `48` | true positive relink |
| 2 | `7232` | `11 -> 21` | `12` | `12` | `31` | false relink |
| 3 | `6241` | `32 -> 20` | `43` | `43` | `37` | false relink |

So the best row improves because the first candidate is a real component
outlier, but the next candidates are visually seductive false positives: the
tracklet already matches the source component's dominant identity, even though
its target centroid similarity is high.

This gives a sharper next critic target: do not rank by target margin alone.
The critic needs an additional source-retention score, such as local temporal
continuity inside the source component, source-neighbor agreement, and target
component purity/safety proxies.

## Next

Use the relink proposer as a candidate generator only.  The next self-play
branch should train or hand-code a no-GT critic that gives higher weight to:

- current k3 false-merge risk proxies: large component, high internal conflict,
  low source centroid support, and MCAM04/MCAM08 concentration;
- target-side safety: target component purity proxies, temporal isolation, and
  no competing high-sim target;
- delivery impact: whether the edit touches the known hard videos and changes
  enough detection mass to move IDF1.
