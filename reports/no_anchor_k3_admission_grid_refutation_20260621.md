# No-Anchor K3 Admission Grid Refutation

Date: 2026-06-21

## Verdict

Rejected as a direct route above the standing best.

This was the detector/admission structural pivot after the temporal-clean bridge
queue found zero admitted edges. It keeps the k3 standing-best global IDs fixed
and only changes which tracklets are delivered.

## Grid

Run on h100-test-3:

- assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`
- min confidence grid: `0.0, 0.08, 0.12, 0.16`
- min quality grid: `-inf, 0.50, 0.55, 0.60`
- ranking key: `coverage_sqrt_pair_score`
- full-scored top rows: `4`

## Results

| Row | min conf | min quality | output filtered tracklets | assigned after filter | pair F1 | full IDF1 | HOTA | AssA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | -inf | 0 | 7487 | 0.769367 | 0.653210 | 0.517030 | 0.532678 |
| 2 | 0.08 | -inf | 0 | 7487 | 0.769367 | 0.653210 | 0.517030 | 0.532678 |
| 3 | 0.12 | -inf | 8 | 7487 | 0.769367 | 0.653210 | 0.517030 | 0.532678 |
| 4 | 0.00 | 0.50 | 576 | 7487 | 0.769367 | 0.653210 | 0.517030 | 0.532678 |

Standing best remains:

| Run | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| k3 softcut + density_oracle_lite | 0.655378 | 0.518798 | 0.534546 |

## Interpretation

Global min-confidence/min-quality tracklet admission is not the current delivery
bottleneck at this interface.

The filters can remove output tracklets, but the assignment side remains fixed:
`assigned_after_filter = 7487` and pair F1 remains `0.769367` for all top rows.
The full-score rows are therefore identical, all below the standing best.

This does not refute density filtering in general. The earlier no-GT
`density_oracle_lite` selector is still the standing promoted policy. It refutes
only this global tracklet-level min-conf/min-quality gate.

## Next Structural Pivot

Do not run a larger global min-quality grid. The useful next branches are:

1. per-video density/admission policies targeting MCAM04 Tc6 and MCAM06 Tc6;
2. local-track continuity scoring before merge proposal;
3. learned counter-target verifier using temporal false positives as hard
   negatives.

## Artifacts

- `local_runs/no_anchor_k3_admission_grid_20260621_summary.json`
- `local_runs/no_anchor_k3_admission_grid_20260621/admission_grid.json`
- `local_runs/no_anchor_k3_admission_grid_20260621/top_assignment.csv`
- remote run dir:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_admission_grid_20260621`
