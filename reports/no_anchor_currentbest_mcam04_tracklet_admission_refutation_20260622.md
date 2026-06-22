# No-Anchor VLINCS: MCAM04 Tracklet Admission Refutation

Date: 2026-06-22

## Why this run

After local attach side-effect gates saturated, I tested a different no-anchor
hypothesis: maybe MCAM04 is weak because low-area tracklets add delivery false
positives.  This run keeps every predicted global ID fixed and only filters
tracklets by MCAM04 tracklet geometry.

No anchors were used.  Policy generation does not load GT.  GT appears only in
the canonical DS1 scorer.

## New utility

- `kit/export_no_anchor_assignment_admission_policies.py`

This materializes explicit no-GT admission policies as assignment CSVs:

`current assignment -> tracklet metadata filter -> filtered assignment CSV`

It does not score, train, or inspect GT.

## Policies generated

Remote run:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_currentbest_weakvideo_admission_20260622`

Manifest:

- `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_weakvideo_admission_20260622/admission_policy_manifest.json`

| Policy | Assignment rows | Dropped MCAM04 tracklets |
|---|---:|---:|
| `mcam04_area2600` | 7433 | 54 |
| `mcam04_area3000` | 7396 | 91 |
| `mcam04_area3200` | 7364 | 123 |
| `mcam04_area3600` | 7308 | 179 |
| `mcam04_area4000` | 7242 | 245 |
| `mcam04_area4800` | 7139 | 348 |

Only `mcam04_area2600` and `mcam04_area3000` were full-scored.  After both
were strongly negative, I stopped the more aggressive 3200-4800 evaluations.

## Canonical results

Baseline before admission:

| Policy | IDF1 | HOTA | AssA | DetPr | DetRe | p005 dropped rows |
|---|---:|---:|---:|---:|---:|---:|
| current best | 0.656225 | 0.519723 | 0.535329 | 0.764545 | 0.574789 | 7603 |

Admission policies:

| Policy | Dropped MCAM04 tracklets | Density IDF1 | p005 IDF1 | HOTA | AssA | DetPr | DetRe | Delta IDF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcam04_area2600` | 54 | 0.655490 | 0.655613 | 0.519060 | 0.534957 | 0.765802 | 0.573145 | -0.000612 |
| `mcam04_area3000` | 91 | 0.655281 | 0.655421 | 0.518834 | 0.534811 | 0.766496 | 0.572464 | -0.000804 |

MCAM04 slice itself:

| Policy | MCAM04 IDF1 | MCAM04 HOTA | MCAM04 AssA | MCAM04 DetPr | MCAM04 DetRe |
|---|---:|---:|---:|---:|---:|
| current best | 0.562439 | 0.448616 | 0.493734 | 0.685737 | 0.476722 |
| `mcam04_area2600` | 0.560655 | 0.447493 | 0.493984 | 0.687866 | 0.473152 |
| `mcam04_area3000` | 0.560031 | 0.447161 | 0.494190 | 0.689121 | 0.471674 |

## Interpretation

Tracklet-level low-area admission is refuted for MCAM04.

The policies do exactly what they should geometrically:

- DetPr rises.
- MCAM04 AssA rises slightly.

But the cost is too high:

- DetRe drops.
- MCAM04 IDF1 drops.
- Global IDF1 drops by 0.000612 to 0.000804 even for the two mildest policies.

This means MCAM04's weak score is not mainly caused by low-area tracklet false
positives.  It is more likely dominated by identity structure and missing/fragile
recall, so filtering tracklets is the wrong lever.

## Decision

Stop MCAM04 area-admission sweeps on the current best.  Keep the exporter for
future explicit admission policies, but pivot away from MCAM04 tracklet pruning.

Next direction:

1. High-mass component rollback/split must be subpart-aware, not whole-component.
2. Any MCAM04 repair should preserve recall first; admission-only policies are
   too lossy.
3. Candidate generation should use structural conflict evidence plus a recall
   guard, not only geometry quality.
