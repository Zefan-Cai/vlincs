# No-Anchor BoTSORT Time-Agglom Top-k15 Sample Promotion

Date: 2026-06-22

## Question

After the Deli AutoResearch/self-play distillation and the crossqueue
full-score refutation, the loop required a structural pivot.  The next
question was whether the sample global-ID plateau came from weak visual
evidence or from the weak-label curriculum/search space.

This round re-used the current best sample evidence file:

`local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/features_botsort_osnet_color_s5_std_cpu_20260622.npz`

No anchors were used.  GT identities in the sample parquet are evaluation-only.

## Main Finding

The missing axis was not another pair model or prototype feature.  It was the
baseline `time_agglom` search space: `top_k=15, theta=0.035` is much better on
the sample than the previously emphasized pair-model settings.

Best sample setting:

| mode | top_k | theta | identity F1 | pair F1 | pair P/R | components | purity mean/p10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| time_agglom | 15 | 0.035 | 0.702092 | 0.639093 | 0.607764 / 0.673829 | 161 | 0.889784 / 0.632869 |

Previous reported best S5 mean/std pair-model setting:

| mode | solver | threshold | identity F1 | pair F1 | pair P/R | components |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pair_model | consensus_guard | 0.65 | 0.456966 | 0.360210 | 0.435531 / 0.307100 | 220 |

So the corrected sample result is:

- identity F1: `0.461553 -> 0.702092`
- pair F1: `0.360210 -> 0.639093`
- pair precision: `0.435531 -> 0.607764`
- pair recall: `0.307100 -> 0.673829`

This is a positive sample-level promotion.  It does not yet prove DS1
end-to-end improvement.

## Ablations

Three stricter curriculum variants were run:

| variant | pseudo positives | positive sources | best identity F1 | best pair F1 | best mode |
| --- | ---: | --- | ---: | ---: | --- |
| strict_v4 | 25,021 | consensus 25,014; strong visual 7 | 0.702092 | 0.639093 | time_agglom top_k15 theta0.035 |
| no_strong | 29,861 | consensus 29,861 | 0.702092 | 0.639093 | time_agglom top_k15 theta0.035 |
| precision_guard | 19,332 | consensus 18,844; strong visual 488 | 0.702092 | 0.639093 | time_agglom top_k15 theta0.035 |

The variants all converge to the same best row because the promoted row is the
baseline no-training resolver, not the learned pair-model row.  This means the
improvement is not from stricter pseudo-label training.  It is a resolver
search-space discovery.

## Top-k/Theta Shape

For `top_k=15`, increasing theta trades recall for precision.  The sample best
is the high-precision middle point:

| top_k | theta | identity F1 | pair F1 | pair P/R | components |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 0.035 | 0.702092 | 0.639093 | 0.607764 / 0.673829 | 161 |
| 15 | 0.030 | 0.688047 | 0.627000 | 0.571395 / 0.694594 | 155 |
| 15 | 0.022 | 0.659243 | 0.604047 | 0.523580 / 0.713738 | 146 |
| 15 | 0.020 | 0.632563 | 0.563970 | 0.464669 / 0.717249 | 144 |
| 15 | 0.018 | 0.607219 | 0.552115 | 0.445566 / 0.725640 | 141 |

`top_k=30` collapses back toward the old plateau, so the key is the smaller
neighbor set rather than only the threshold.

## DS1 Migration Probe

A DS1 no-anchor probe was launched on h100-test-3 with the analogous settings:

```bash
kit/no_anchor_resolve_sweep.py \
  --modes time_agglom \
  --thetas 0.025,0.030,0.035,0.040,0.045 \
  --top-ks 15 \
  --min-dets 10 \
  --exclude-same camera \
  --temporal-bonus 0.005 \
  --time-windows-ms 1000 \
  --sort-key tracklet_pair_f1 \
  --full-top-n 5
```

Remote run dir:

`/mnt/localssd/vlincs_reid_runs/no_anchor_time_agglom_topk15_theta_probe_20260622`

Local mirror:

`local_runs/remote_h100_test_3_20260622/no_anchor_time_agglom_topk15_theta_probe_20260622`

DS1 fast-pair rows showed high precision but low recall:

| theta | pair F1 | pair P/R | DS1 IDF1 | HOTA | AssA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.025 | 0.612019 | 0.782720 / 0.502443 | 0.587838 | 0.444591 | 0.463460 |
| 0.030 | 0.604274 | 0.799179 / 0.485797 | 0.586953 | 0.443659 | 0.462832 |
| 0.035 | 0.593153 | 0.814651 / 0.466354 | 0.584903 | 0.441464 | 0.461082 |
| 0.040 | 0.570349 | 0.841322 / 0.431403 | 0.580838 | 0.433922 | 0.450593 |
| 0.045 | 0.546508 | 0.847681 / 0.403240 | 0.570792 | 0.424738 | 0.443153 |

Best DS1 row per-video metrics:

| video | IDF1 | HOTA | AssA |
| --- | ---: | ---: | ---: |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.805727 | 0.724542 | 0.769106 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.727664 | 0.640596 | 0.711277 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.649887 | 0.529446 | 0.566829 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.596797 | 0.491143 | 0.551833 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.503407 | 0.386727 | 0.432641 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.236698 | 0.217928 | 0.352004 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.765602 | 0.671532 | 0.713334 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.611663 | 0.505218 | 0.566235 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.659631 | 0.559325 | 0.618793 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.693055 | 0.574302 | 0.608822 |

## Verdict

Promote the sample insight, but refute direct DS1 production transfer.
Sample `top_k=15/theta=0.035` proves that stricter small-neighborhood
agglomeration can cross 0.70 identity F1 on the BoTSORT sample.  On DS1, the
analogous full-resolver probe falls to best IDF1 `0.587838`, far below the
current no-anchor production best `0.655911`.

Next work should use this as a precision-biased candidate generator, not as the
whole production resolver: attach only small, high-margin components to the
current DS1 best, and require a side-effect critic before full-score.

## Artifacts

- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_curriculum_strict_v4_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_curriculum_no_strong_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_curriculum_precision_guard_20260622.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_time_agglom_topk15_theta_probe_20260622`
- `local_runs/remote_h100_test_3_20260622/no_anchor_time_agglom_topk15_theta_probe_20260622/time_agglom_topk15_theta_probe.json`
