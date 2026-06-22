# No-Anchor K3 Grouped Relink Refutation

Date: 2026-06-21

## Purpose

This is the structural pivot after the source-retention single-tracklet relink
critic.  The critic can find eval-only true relinks, but strict single-node
admission only produced two candidates.  This run asks whether relaxing the
gate and grouping conflict outliers by `source_label -> target_label` can create
enough mass to move end-to-end IDF1.

No anchors are used.  GT is used only after candidate materialization for
diagnosis and canonical full-score.

## Code

Added:

- `kit/no_anchor_conflict_outlier_group_relink.py`

The script:

1. builds single-tracklet conflict-outlier evidence with the same
   source-retention features;
2. groups rows by source/target key;
3. ranks groups by mean score, neighbor margin, detection mass, and source
   retention;
4. materializes top group edits for canonical full-score.

## Candidate Results

Strict grouped run:

| run | candidate count | group count | interpretation |
|---|---:|---:|---|
| source/target/camera or source/target/video | `21` | `0` | no natural grouped evidence under strict admission |

Relaxed recall probe:

| config | candidates | groups | top-k groups | relinks |
|---|---:|---:|---:|---:|
| `source_target`, relaxed margins | `59` | `3` | `1 / 2 / 4` | `2 / 4 / 6` |

Top groups from the relaxed probe:

| rank | source -> target | seqs | dets | mean margin | mean neighbor margin | audit |
|---:|---|---|---:|---:|---:|---|
| 1 | `29 -> 11` | `3727, 7845` | `562` | `0.385709` | `0.168546` | both neither source-top nor target-top GT |
| 2 | `26 -> 20` | `5824, 2041` | `360` | `0.156858` | `0.099867` | one target-top true, one neither |
| 3 | `20 -> 12` | `4328, 5382` | `143` | `0.013792` | `0.050789` | one target-top true, one neither |

Artifacts:

- `local_runs/no_anchor_k3_source_retention_group_relink_candidates_20260621/result.json`
- `local_runs/no_anchor_k3_source_retention_group_relink_recall_probe_20260621/result.json`
- `local_runs/no_anchor_k3_source_retention_group_relink_recall_probe_20260621/group_gt_audit.json`

## Full-Score

Only the top group was full-scored because it already contains known eval-only
false evidence.

| run | moved relinks | IDF1 | HOTA | AssA |
|---|---:|---:|---:|---:|
| relaxed group top-1 | `2` | `0.653293` | `0.517126` | `0.532779` |
| standing k3 + density primary | - | `0.655378` | `0.518798` | `0.534546` |

The top grouped edit is below both the standing best and the stricter
source-retention single-tracklet top-2+density result (`IDF1 0.653600`).

## Interpretation

Grouping did create more mass only after relaxing the gate, but the extra mass
comes from minority/noise fragments:

- top group `29 -> 11` has strong no-GT margins, yet both moved tracklets are
  neither the source dominant GT (`3`) nor the target dominant GT (`12`);
- groups 2 and 3 are mixed, with only one member matching target dominant GT;
- grouping by camera/video under stricter gates produced no groups at all.

Decision: reject grouped conflict-outlier relink as the next production
direction.  It is useful as a negative result: source-retention is a good
single-tracklet critic, but relaxed grouping admits minority fragments before it
finds enough true mass.

## Next

The next structural pivot should stop using conflict nodes as the primary
recall source.  Better candidates:

1. target-fragment mining: start from impure target components and ask which
   minority fragments have consistent visual/temporal support;
2. local-track continuity: use same-camera temporal chains to create positives
   before cross-component relink;
3. detector-density admission: score whether a relink improves committed output
   density before it is full-scored.

