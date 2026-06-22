# No-Anchor BoTSORT S5 Mean/Std Feature Ablation

Date: 2026-06-22

## Question

The previous no-anchor sample branch showed that stronger crop/person evidence
matters:

| Evidence | Identity F1 | Pair F1 | Pair P/R |
|---|---:|---:|---:|
| BoTSORT weak crop+bbox | `0.191162` | `0.032079` | `0.034875 / 0.029697` |
| BoTSORT OSNet+color, 3 crops mean | `0.380024` | `0.253572` | `0.315609 / 0.211916` |
| GT-box OSNet+color, 3 crops mean | `0.272212` | `0.095926` | `0.132012 / 0.075333` |

This round asks whether taking more crops per tracklet and exposing dispersion
features can raise the no-anchor sample global-ID model enough to justify DS1
pipeline insertion.

GT identity labels are eval-only.  They are not used as anchors, training
labels, assignment evidence, or candidate selection signals.

## Feature Extraction

Remote extraction ran on `h100-test-3` CPU because the H100 GPUs were occupied.
The extractor reads real BoTSORT sample tracklets and raw frames:

```bash
/mnt/localssd/vlincs_reid_venv/bin/python kit/extract_sample_tracklet_osnet_features.py \
  --tracklet-parquet /mnt/localssd/vlincs_reid_runs/botsort_no_anchor_sample_20260621/botsort_eval.parquet \
  --video-root /mnt/localssd/vlincs/VLINCS_Performer/sample/videos \
  --out /mnt/localssd/vlincs_reid_runs/botsort_no_anchor_sample_20260621/features_botsort_osnet_color_s5_std_cpu_20260622.npz \
  --samples 5 \
  --batch-size 64 \
  --device cpu
```

Extraction summary:

| Field | Value |
|---|---:|
| tracklets | `2,406` |
| seen crops | `11,971` |
| valid OSNet / color | `2,406 / 2,406` |
| missing video / crop | `0 / 0` |
| mean feature blocks | `features_osnet (512)`, `features_color (82)` |
| dispersion blocks | `features_osnet_std (512)`, `features_color_std (82)` |

Local artifact:

- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/features_botsort_osnet_color_s5_std_cpu_20260622.npz`

## Ablation Results

All runs use:

- sample parquet:
  `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/botsort_eval.parquet`
- eval minimum GT fraction: `0.50`
- baseline top-k: `45`
- pseudo-pair model: no-anchor weak positives / hard negatives only
- GT: evaluation only

| Setting | Best identity F1 | Best pair F1 | Pair P/R | Purity mean/p10 | Components | Best pair setting |
|---|---:|---:|---:|---:|---:|---|
| s5_mean_only | `0.448159` | `0.341138` | `0.401001 / 0.296827` | `0.841244 / 0.453805` | `221` | `consensus_attach @ 0.65` |
| s5_mean_std_small | `0.451135` | `0.337979` | `0.371661 / 0.309894` | `0.852176 / 0.445948` | `234` | `consensus_guard @ 0.65` |
| s5_mean_std_strong | `0.453773` | `0.349976` | `0.405765 / 0.307674` | `0.868229 / 0.475090` | `233` | `consensus_guard @ 0.65` |
| s5_mean_std_w045 | `0.461553` | `0.360210` | `0.435531 / 0.307100` | `0.851065 / 0.502871` | `220` | `consensus_guard @ 0.65` |
| s5_mean_std_w060 | `0.460016` | `0.352409` | `0.430832 / 0.298139` | `0.864744 / 0.523832` | `223` | `consensus_guard @ 0.65` |
| s5_mean_std_osonly030 | `0.450146` | `0.347279` | `0.416507 / 0.297784` | `0.854439 / 0.489589` | `223` | `consensus_guard @ 0.65` |

Best observed sample result:

- identity F1: `0.461553`
- pair F1/P/R: `0.360210 / 0.435531 / 0.307100`
- setting: `s5_mean_std_w045`, `consensus_guard @ 0.65`

Compared with the prior BoTSORT OSNet+color 3-crop mean result:

- identity F1 improved `0.380024 -> 0.461553`
- pair F1 improved `0.253572 -> 0.360210`
- pair precision improved `0.315609 -> 0.435531`
- pair recall improved `0.211916 -> 0.307100`

## Interpretation

The useful gain mostly comes from five-crop extraction plus better temporal
coverage, not from simply increasing dispersion weight.  Mean-only S5 already
raises pair F1 to `0.341138`; adding moderate OSNet/color dispersion raises the
best pair F1 to `0.360210`.  Pushing dispersion harder (`w060`) improves some
purity numbers but lowers pair F1, so std weight is not a monotonic knob.

This is a positive evidence promotion, but it is not close to the requested
`0.70` global-ID/sample target.  The current DS1 end-to-end best is still:

- IDF1/HOTA/AssA = `0.655911 / 0.519311 / 0.534922`

No new DS1 full-score is claimed from this sample branch.

## Prototype Pair-Feature Follow-Up

I then tested the natural next hypothesis: keep the same S5 mean/std embedding,
but expose crop-level prototypes to the pair model.

Added code paths:

- `kit/extract_sample_tracklet_osnet_features.py --save-prototypes`
- `kit/make_sample_prototype_pair_feature_views.py`
- 3D prototype support in `kit/no_anchor_global_id_model.py`
- `--pair-feature-npz` support in `kit/no_anchor_sample_parquet_sweep.py`

Prototype extraction produced:

| Field | Value |
|---|---:|
| feature file | `features_botsort_osnet_color_s5_proto_cpu_20260622.npz` |
| OSNet prototypes | `[2406, 5, 512]` |
| color prototypes | `[2406, 5, 82]` |
| valid OSNet prototypes | `11,971` |
| valid color prototypes | `11,971` |

Two prototype forms were tested:

| Setting | Feature form | Best identity F1 | Best pair F1 | Pair P/R |
|---|---|---:|---:|---:|
| slot views | five separate same-slot cosine views | `0.442073` | `0.350859` | `0.421102 / 0.300699` |
| all-cross view | 25 crop-to-crop max/top-k/mean/min/std stats | `0.443991` | `0.352414` | `0.428609 / 0.299221` |
| best S5 mean/std baseline | no prototype pair view | `0.461553` | `0.360210` | `0.435531 / 0.307100` |

Eval-only retrieval audit confirms the prototype branch is not merely a graph
threshold issue:

| Retrieval evidence | AP | AUC | top-45 hit | cross-video top-45 hit | cross-video top-positive > top-negative |
|---|---:|---:|---:|---:|---:|
| S5 mean OSNet | `0.430425` | `0.884725` | `0.994189` | `0.951400` | `0.788167` |
| all-cross prototype max | `0.372521` | `0.881577` | `0.989435` | `0.928118` | `0.685156` |

So this particular prototype form is a negative result.  It adds evidence but
adds more noise than margin.  Do not spend more sweeps on slot/all-cross
prototype pair features without changing the evidence source or pseudo-label
construction.

## Deli AutoResearch Decision

The Deli AutoResearch protocol says to keep durable state, separate proposer
from judge, treat score drops honestly, and pivot structurally after repeated
small gains.  Applied here:

- keep `features_botsort_osnet_color_s5_std_cpu_20260622.npz` as the current
  best sample evidence source;
- reject further scalar std-weight sweeps as the main path;
- reject slot/all-cross prototype pair-feature sweeps as the main path;
- do not insert this into DS1 until sample identity evidence moves far beyond
  the current `0.46 / 0.36` plateau;
- pivot to a stronger evidence source or a different pseudo-label curriculum,
  not just more aggregation of the same OSNet crops.

Next experiment should answer a sharper question:

> Can a stronger identity evidence source or a stricter weak-label curriculum
> separate true cross-video positives from hard negatives beyond the S5 mean
> OSNet retrieval margin?

## Artifacts

- `kit/extract_sample_tracklet_osnet_features.py`
- `kit/make_sample_prototype_pair_feature_views.py`
- `kit/no_anchor_global_id_model.py`
- `kit/no_anchor_sample_parquet_sweep.py`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/features_botsort_osnet_color_s5_std_cpu_20260622.npz`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/features_botsort_osnet_color_s5_proto_cpu_20260622.npz`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s5_mean_only_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s5_mean_std_small_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s5_mean_std_strong_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s5_mean_std_w045_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s5_mean_std_w060_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s5_mean_std_osonly030_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s5_proto_pair_w045_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s5_proto_allcross_w045_20260622.json`
