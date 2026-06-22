# No-Anchor Current-Best Detection Filter Sweep

Date: 2026-06-22

## Standing

- Setting: no-anchor global-id research; GT is used only for canonical evaluation.
- Current best before this sweep: IDF1 / HOTA / AssA = `0.657653 / 0.520723 / 0.535819`.
- New best found: no. The standing best remains `0.657653 / 0.520723 / 0.535819`.
- E2E target remains open: `0.657653 < 0.70`.

## Method

This sweep tested whether the canonical `p005_area` detection/admission filter is a local bottleneck after the latest promoted assignment, `weakmetric 10 -> 22`. The assignment was fixed and only the density/area post-filter changed. Six variants were full-scored on h100-test-3 using DS1 GT only for final evaluation.

A first wrapper attempt failed because `DATA_ROOT` was not exported in the remote shell. That failure produced empty cost matrices and is treated as a runner bug, not experiment evidence. The metrics below are from the corrected run with `DATA_ROOT=/mnt/localssd/vlincs_reid_data`.

## Results

| Config | Dropped rows | IDF1 | HOTA | AssA | DetPr | DetRe | MCAM04 IDF1 | MCAM06 Tc6 IDF1 | MCAM03 Tc8 IDF1 | Delta vs p005 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `area_scale0p75` | 3008 | 0.657608 | 0.520696 | 0.535762 | 0.764910 | 0.576708 | 0.565587 | 0.609917 | 0.628750 | -0.000045 |
| `p005_area` | 7603 | 0.657653 | 0.520723 | 0.535819 | 0.765620 | 0.576374 | 0.565566 | 0.610296 | 0.628528 | +0.000000 |
| `area_scale1p25` | 13667 | 0.657576 | 0.520577 | 0.535700 | 0.766470 | 0.575774 | 0.565479 | 0.610206 | 0.628097 | -0.000077 |
| `q01_area` | 13830 | 0.657577 | 0.520578 | 0.535700 | 0.766467 | 0.575778 | 0.565474 | 0.610202 | 0.628089 | -0.000076 |
| `lowvideo_area_half` | 3850 | 0.657625 | 0.520698 | 0.535749 | 0.764981 | 0.576694 | 0.565629 | 0.609665 | 0.628648 | -0.000028 |
| `mcam04_q05_area` | 34865 | 0.656550 | 0.519248 | 0.534720 | 0.769731 | 0.572387 | 0.561978 | 0.610296 | 0.628528 | -0.001103 |

## Interpretation

- The existing `p005_area` filter is still the best global setting among this local sweep: IDF1/HOTA/AssA `0.657653 / 0.520723 / 0.535819`.
- Relaxing area thresholds recovers a little recall but loses enough precision to lower IDF1. `area_scale0p75` drops only 3008 rows and has DetRe `0.576708`, but IDF1 falls to `0.657608`.
- Tightening area thresholds improves DetPr but loses recall and also lowers IDF1. `area_scale1p25` and `q01_area` both land at about `0.657576`.
- A strict MCAM04-only `q05_area` policy is clearly harmful: it drops 34865 rows and falls to IDF1 `0.656550`, with MCAM04 IDF1 `0.561978`.
- Conclusion: detection/admission filters are a local optimum in this neighborhood. The next useful direction should go back to no-anchor identity evidence: blacklisted side-effect-aware subpart candidate search, not broader area filtering.

## Artifacts

- Summary JSON: `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/filter_sweep_summary.json`
- Config manifest: `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/configs.json`
- `area_scale0p75` metrics/log: `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/01_area_scale0p75.json`, `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/01_area_scale0p75.log`
- `p005_area` metrics/log: `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/02_p005_area.json`, `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/02_p005_area.log`
- `area_scale1p25` metrics/log: `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/03_area_scale1p25.json`, `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/03_area_scale1p25.log`
- `q01_area` metrics/log: `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/04_q01_area.json`, `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/04_q01_area.log`
- `lowvideo_area_half` metrics/log: `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/05_lowvideo_area_half.json`, `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/05_lowvideo_area_half.log`
- `mcam04_q05_area` metrics/log: `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/06_mcam04_q05_area.json`, `local_runs/remote_h100_test_3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/06_mcam04_q05_area.log`
- Remote run: `/mnt/localssd/vlincs_reid_runs/no_anchor_filter_sweep_currentbest_20260622_env`
