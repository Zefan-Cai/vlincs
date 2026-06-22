# Source-Island Acceptor Audit

- rows: `61`
- positives: `12`
- features: `27`

## Average Precision

| ranker | AP |
| --- | ---: |
| `source_rank_score` | `0.302423` |
| `source_score` | `0.300976` |
| `source_quality` | `0.155425` |
| `ridge_logit_loocv` | `0.944124` |

## Top-k Positives

| ranker | k | positives | positive frac | sum positive delta |
| --- | ---: | ---: | ---: | ---: |
| `source_rank_score` | `5` | `3` | `0.600000` | `0.003605` |
| `source_rank_score` | `10` | `3` | `0.300000` | `0.003605` |
| `source_rank_score` | `20` | `5` | `0.250000` | `0.006006` |
| `source_rank_score` | `50` | `9` | `0.180000` | `0.011390` |
| `source_score` | `5` | `3` | `0.600000` | `0.003605` |
| `source_score` | `10` | `3` | `0.300000` | `0.003605` |
| `source_score` | `20` | `5` | `0.250000` | `0.006006` |
| `source_score` | `50` | `9` | `0.180000` | `0.011390` |
| `source_quality` | `5` | `1` | `0.200000` | `0.000202` |
| `source_quality` | `10` | `2` | `0.200000` | `0.002611` |
| `source_quality` | `20` | `2` | `0.100000` | `0.002611` |
| `source_quality` | `50` | `5` | `0.100000` | `0.008005` |
| `ridge_logit_loocv` | `5` | `5` | `1.000000` | `0.002834` |
| `ridge_logit_loocv` | `10` | `9` | `0.900000` | `0.009426` |
| `ridge_logit_loocv` | `20` | `12` | `0.600000` | `0.013374` |
| `ridge_logit_loocv` | `50` | `12` | `0.240000` | `0.013374` |

## Top Feature Correlations

| feature | corr |
| --- | ---: |
| `source_cross_mean_sim` | `-0.624109` |
| `source_cross_max_sim` | `-0.491114` |
| `source_quality` | `-0.387037` |
| `source_internal_sim` | `-0.338238` |
| `source_margin_mean` | `0.293822` |
| `source_size` | `-0.206881` |
| `source_rank_score` | `0.108548` |
| `source_score` | `0.107364` |
| `source_expand_sim` | `-0.045464` |
| `source_conflicts_to_rest` | `0.041690` |
| `source_rejected_overlap` | `0.031703` |
| `source_seed_sim` | `-0.031665` |

## Top Oracle Delta Rows

| delta pair F1 | source label | source rank | artifact |
| ---: | ---: | ---: | --- |
| `0.002993` | `21` | `1` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_tiny_20260620.json` |
| `0.002409` | `9` | `34` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_loose1_20260620.json` |
| `0.001547` | `40` | `15` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_loose1_20260620.json` |
| `0.001389` | `31` | `43` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_loose1_20260620.json` |
| `0.001384` | `32` | `12` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_loose1_20260620.json` |
| `0.001084` | `19` | `26` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_loose1_20260620.json` |
| `0.001017` | `26` | `5` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_tiny_20260620.json` |
| `0.000489` | `24` | `44` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_loose1_20260620.json` |
