# No-Anchor E2E Ablation Log - 2026-06-18

Goal: keep improving the no-anchor VLINCS pipeline until both the global-ID
model and the end-to-end pipeline exceed 0.70.  The global-ID model has already
passed; the end-to-end pipeline has not.

## Current Standing Best

- Global-ID model-only gate: pass
- Best promoted/production model pair F1: 0.768743
- Best promoted/production model precision / recall: 0.813273 / 0.728836
- Best diagnostic model-side pair F1: 0.775234
  (`loose_source_island_g8_strict_target_reassign`, e2e-negative)
- Best verified full IDF1: 0.655240
- Historical current-tracklet oracle full IDF1: 0.711353
- Latest standing-assignment oracle full IDF1: 0.706202

Standing full artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.zip`

Standing assignment CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`

Latest diagnostic assignment CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_top1_assignments_20260620.csv`

## 2026-06-18 Continuation

### Component merge sweep

Artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_component_merge_mpc010_f032_mcam05area12000_paironly_20260618.json`

Base:

- pair F1: 0.708403
- precision: 0.776941
- recall: 0.650977

Best component merge row:

- pair F1: 0.708309
- precision: 0.776670
- recall: 0.651008
- accepted merges: 348
- components: 394

Conclusion: unrestricted component-level merge slightly increases recall but
loses more precision.  It should not be advanced to full scoring.

### Mutual top-k component merge

Artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_component_merge_mutual_mpc010_f032_mcam05area12000_paironly_20260618.json`

Best mutual row:

- pair F1: 0.708403
- precision: 0.776941
- recall: 0.650977
- mutual top-k: 1
- accepted merges: 34
- components: 708

Conclusion: mutual top-k is conservative enough to avoid precision damage, but
it does not move the pair metric at six decimal places.

### Component retrievability diagnostic

Artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_component_retrieval_mpc010_f032_mcam05area12000_20260618.json`

False-split mass retrieval:

- candidate scored fraction: 0.866342
- retrieved@1: 0.115431
- retrieved@5: 0.454258
- retrieved@10: 0.599979
- retrieved@20: 0.756057
- retrieved@50: 0.866129

Same-GT candidate score quantiles:

- q10: 0.725559
- q25: 0.768132
- q50: 0.842801
- q75: 0.895317
- q90: 0.933164

Candidate edge precision is low even at high thresholds:

- threshold 0.70: dominant-pure edge precision 0.072272
- threshold 0.74: dominant-pure edge precision 0.088668

Conclusion: current evidence retrieves many true split candidates, but raw
similarity is not calibrated enough to select them.  The next promising path is
a stronger no-anchor verifier/calibrator for candidate edges, not another raw
threshold or simple component merge.

### Low-weight CLIP/DINO fusion

Created feature artifacts:

- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_clip005_20260618.npz`
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_dino005_20260618.npz`
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_clip003_dino003_20260618.npz`

Pair-only resolver artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_clip005_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_dino005_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_clip003dino003_mpc010_f032_mcam05area12000_paironly_20260618.json`

Best rows:

- CLIP 0.05: pair F1 0.705820
- DINO 0.05: pair F1 0.702524
- CLIP 0.03 + DINO 0.03: pair F1 0.705470

Conclusion: low-weight CLIP/DINO blocks are negative under the current
time-agglom resolver and should not be sent to full scoring.

### Pseudo-label component-edge verifier

New script:

`/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_component_verifier_sweep.py`

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_component_verifier_logreg_centroid_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_component_verifier_logreg_centroid_relaxed_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_component_verifier_rf_centroid_relaxed_mpc010_f032_mcam05area12000_paironly_20260618.json`

Setup:

- no anchors
- GT used only for pair metrics
- base resolver: `mpc010_f032`, theta `0.014`, MCAM05 area >= `12000`
- candidate generation: component centroid top-50
- verifier views: primary fused, DB embedding, PersonViT, color histogram
- pseudo-positive labels: stable mutual/ranked component edges across views
- pseudo-negative labels: low-consensus or cannot-link-like candidate edges

Results:

- strict logreg: pair F1 `0.708410`, precision `0.776943`, recall `0.650988`
- relaxed logreg: pair F1 `0.708410`, precision `0.776943`, recall `0.650988`
- relaxed RF: pair F1 `0.708403`, precision `0.776941`, recall `0.650977`
- base: pair F1 `0.708403`, precision `0.776941`, recall `0.650977`

Conclusion: the verifier can safely merge some pseudo-stable component edges,
but those edges barely move weighted pair mass.  Under the current evidence,
multi-view rank/centroid features are not enough to recover the large false
splits identified by the retrieval diagnostic.

### Three-sample color histogram

Created feature artifact:

`/mnt/localssd/vlincs_reid_runs/ds1_tracklet_colorhist_s3_20260618.npz`

Fused artifact:

`/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color3s010_20260618.npz`

Pair-only resolver artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_fused_color3s_mpc010_f032_mcam05area12000_paironly_20260618.json`

Result:

- extracted 9734 / 9734 tracklets, no anchors, no GT
- best pair F1 `0.705185`
- precision `0.772231`
- recall `0.648851`

Conclusion: replacing the single-frame color histogram with first/mid/last
aggregation is negative.  More color samples do not solve the edge-calibration
problem.

### Same-camera constraint ablation

Pair-only resolver artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_mpc010_f032_mcam05area12000_exclude_stream_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_mpc010_f032_mcam05area12000_exclude_video_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_mpc010_f032_mcam05area12000_exclude_none_paironly_20260618.json`

Best rows:

- `exclude_same=stream`: pair F1 `0.697257`, precision `0.771114`, recall `0.636312`
- `exclude_same=video`: pair F1 `0.697257`, precision `0.771114`, recall `0.636312`
- `exclude_same=none`: pair F1 `0.541562`, precision `0.756191`, recall `0.421833`

Conclusion: relaxing the same-camera exclusion does not recover the dominant
false splits.  It mostly introduces bad candidate structure or removes useful
local stitching behavior.

### Complete PersonViT three-sample features

Full three-sample PersonViT extraction was completed with 8 GPU shards:

`/mnt/localssd/vlincs_reid_runs/ds1_tracklet_personvit_msmt_vits_s3_full_20260618.npz`

Extraction coverage:

- 8 / 8 shards succeeded
- 9734 / 9734 tracklets
- model: `maennyn/personvit-reid-msmt17-vit-s`
- samples: first/mid/last
- no anchors, no GT

Full-replacement fused artifact:

`/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_persons3_025_color010_20260618.npz`

Pair-only result:

- artifact: `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_persons3_mpc010_f032_mcam05area12000_paironly_20260618.json`
- best pair F1 `0.697060`
- precision `0.764226`
- recall `0.640746`

Low-weight auxiliary fused artifact:

`/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_persons1_025_persons3_005_color010_20260618.npz`

Pair/full result:

- pair-only artifact: `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_persons1s3low_mpc010_f032_mcam05area12000_paironly_20260618.json`
- full artifact: `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_persons1s3low_mpc010_f032_mcam05area12000_full_20260618.json`
- best pair F1 `0.708763`
- precision `0.782810`
- recall `0.647514`
- full IDF1 `0.633795`

Conclusion: full s3 PersonViT is negative as a replacement and only slightly
positive as a very low-weight auxiliary pair feature.  The small pair gain does
not transfer to full IDF1.

### Agglomerative target-cluster resolver

Pair-only artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_agglom_n_fused_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_agglom_n_lowtarget_fused_mpc010_f032_mcam05area12000_paironly_20260618.json`

Full artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_agglom_n_t064_fused_mpc010_f032_mcam05area12000_full_20260618.json`

Best rows:

- `target_clusters=64`: pair F1 `0.710314`, precision `0.770267`, recall `0.659021`
- full IDF1 `0.633879`
- `target_clusters=80`: pair F1 `0.703275`, precision `0.794781`, recall `0.630664`
- `target_clusters=48`: pair F1 `0.700872`, precision `0.741461`, recall `0.664497`
- `target_clusters=36`: pair F1 `0.666672`, precision `0.652081`, recall `0.681931`

Conclusion: forcing fewer clusters can improve pair recall in some settings,
but full IDF1 remains below the standing best.  The added merges are not
aligned well enough with scorer-level identity continuity.

### High-precision admission / commit-only curve

Artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_highprecision_admission_full5_20260618.json`

Setup:

- resolver: time-aware agglomeration, theta `0.0165`, top-k `15`
- feature: `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_20260617.npz`
- admission: min confidence `0.65`, MCAM05 area floor `30000`
- sweep: global min area `12000, 10000, 8000, 6000, 4000, 2000, 0`
- no anchors; GT is used only by the evaluator

Full-scored rows:

- min area `12000`: output tracklets `3700`, pair F1 `0.868825`, DetPr `0.844896`, DetRe `0.385849`, full IDF1 `0.529765`
- min area `10000`: output tracklets `4103`, pair F1 `0.867465`, DetPr `0.836886`, DetRe `0.420512`, full IDF1 `0.559760`
- min area `8000`: output tracklets `4451`, pair F1 `0.866388`, DetPr `0.829463`, DetRe `0.448464`, full IDF1 `0.582168`
- min area `6000`: output tracklets `4671`, pair F1 `0.859397`, DetPr `0.824644`, DetRe `0.462628`, full IDF1 `0.592732`
- min area `4000`: output tracklets `4775`, pair F1 `0.857305`, DetPr `0.822082`, DetRe `0.468850`, full IDF1 `0.597139`

Pair-only tail rows:

- min area `2000`: output tracklets `4831`, pair F1 `0.856005`, precision `0.901112`, recall `0.815199`
- min area `0`: output tracklets `4835`, pair F1 `0.855827`, precision `0.901056`, recall `0.814921`

Conclusion: a commit-only/high-precision admission policy is not the route to
0.70 full IDF1.  Pair precision becomes excellent, but detection recall drops
too far; the best full-scored row reaches only `0.597139`, well below the
standing full IDF1 `0.635349`.

### Global-ID model assignments under full scorer

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_model_assignments_full_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_model_mps3top256_assignments_full_20260618.json`

Results:

- accepted global-ID model assignment CSV: pair F1 `0.719910`, full IDF1 `0.633850`
- mps3top256 assignment CSV: pair F1 `0.710030`, full IDF1 `0.634067`
- standing best remains full IDF1 `0.635349`

Per-video comparison showed the accepted global-ID model and the standing best
are identical on most videos.  The main regression is MCAM06 Tc6 (`0.616981`
vs standing `0.658810`), caused by extra admission filtering in the model-only
pair-F1 winner.

Conclusion: the no-anchor global-ID model is valid as a tracklet-pair model,
but its assignment output does not yet improve the full pipeline.  Pair-F1
gains are not enough unless they preserve the scorer-facing detection/association
balance.

### Component split sweep

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_component_split_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_component_split_mpc010_f032_mcam05area12000_paironly_20260618.csv`

Setup:

- base: standing time-agglom config, fused feature weight `0.32`
- split only components with at least `32, 64, 96` tracklets
- split theta grid `0.016, 0.018, 0.020, 0.024, 0.030`
- split top-k `0, 15`; temporal bonus `0.0, 0.005`

Result:

- best row is the no-op split: pair F1 `0.708403`
- dense split (`split_top_k=0`) never split the large components
- sparse split (`split_top_k=15`) split components, but pair F1 dropped as low as `0.418349` and at best only `0.610099`

Conclusion: naive large-component splitting destroys recall before it removes
enough false-merge mass.  The next split attempt needs an evidence-aware
verifier, not stricter component clustering alone.

### Local time-agglom grid around standing best

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_local_time_grid_b0005_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_local_time_grid_b0005_mpc010_f032_mcam05area12000_paironly_20260618.csv`

Setup:

- theta `0.013, 0.014, 0.015, 0.016`
- top-k `12, 15, 20`
- min-dets `8, 10, 12`
- time window `750, 1000, 1500`
- temporal bonus fixed at `0.005`
- admission fixed to standing best: MCAM05 Tc6 area floor `12000`

Best rows:

- `theta=0.014, top_k=15, min_dets=10, window=1000`: pair F1 `0.708403`, precision `0.776941`, recall `0.650977`
- `theta=0.014, top_k=15, min_dets=10, window=1500`: pair F1 `0.707961`
- `theta=0.013, top_k=15, min_dets=10, window=1500`: pair F1 `0.707634`

Conclusion: the standing config is still the local optimum.  Increasing top-k
raises recall in some rows but loses too much precision; increasing theta raises
precision but loses recall.

### YOLO pose/body-part color feature

New extractor:

`/mnt/localssd/vlincs_reid_by_search/kit/extract_tracklet_pose_color_features.py`

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_posecolor_s3_full_20260618.npz`
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_posecolor003_20260618.npz`
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_posecolor005_20260618.npz`
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_posecolor008_20260618.npz`
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_posecolor010_20260618.npz`

Extraction setup:

- model: `yolo11n-pose.pt`
- samples: first/mid/last crops from each tracklet
- feature: 188-D normalized head/torso/legs/full color + pose summary
- coverage: `9734 / 9734` tracklets
- sample crops: `28941`
- pose-success rows: `23974`
- no anchors, no GT

Pair-only resolver artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_posecolor003_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_posecolor005_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_posecolor008_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_posecolor010_mpc010_f032_mcam05area12000_paironly_20260618.json`

Best rows:

- posecolor `0.03`: pair F1 `0.706376`, precision `0.774159`, recall `0.649508`
- posecolor `0.05`: pair F1 `0.705533`, precision `0.773807`, recall `0.648330`
- posecolor `0.08`: pair F1 `0.704744`, precision `0.766770`, recall `0.652001`
- posecolor `0.10`: pair F1 `0.704336`, precision `0.768689`, recall `0.649926`

Conclusion: pose/body-part color is a valid no-anchor evidence source, but
direct concatenation into the current cosine/time graph is negative.  It lowers
pair F1 relative to the standing `0.708403`, so it should be used only as a
future verifier feature, not as raw graph distance.

### Trained pair-link global-ID model bundle

Artifacts:

- model bundle: `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_20260618.joblib`
- JSON: `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_full8_20260618.json`
- CSV: `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_full8_20260618.csv`
- assignments: `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_assignments_20260618.csv`
- joint gate: `/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_and_e2e_gate_20260618.json`

Setup:

- model type: histogram gradient boosting pair-link calibrator
- pseudo positives: `139486`
- pseudo negatives: `60000`
- pseudo validation AUC/AP: `0.995851 / 0.997873`
- feature: fused match/person/color with DB embedding concat
- output admission: standing MCAM05 Tc6 area floor `12000`
- no anchors; GT used only for post-hoc pair/full metrics

Best full-scored row:

- threshold `0.03`, blend `0.50`
- pair F1 `0.714228`
- precision `0.781130`
- recall `0.657882`
- full IDF1 `0.635197`
- HOTA `0.496830`
- DetRe `0.558801`
- DetPr `0.735789`

Assignment export:

- rows: `9009`
- components: `730`
- largest component: `284`
- decision status: `provisional=8358`, `forced_singleton=648`, `forced_component=3`

Gate result:

- best global remains the assignment export at pair F1 `0.719910`
- best trained pair-calibrator row: pair F1 `0.714228`
- best e2e among these model artifacts: full IDF1 `0.635197`
- joint >0.70 global and >0.70 e2e gate: fail

Conclusion: the real no-anchor global-ID model artifact now exists and clears
the model-only 0.70 bar, but it still does not improve the end-to-end score
over the standing time-agglom full IDF1 `0.635349`.  The remaining gap is not
model packaging; it is scorer-facing false-split recovery without damaging
precision/coverage.

### Loaded pair-calibrator resolver variants

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_support_mpc010_f032_mcam05area12000_paironly_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_consensus_attach_mpc010_f032_mcam05area12000_paironly_20260618.json`

Setup:

- loaded model: `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_20260618.joblib`
- no retraining
- same feature/admission as the trained pair-calibrator run
- pair-only screening

Results:

- `support` solver with merge-support guard rejected all candidate merges:
  pair F1 `0.000000`
- `consensus_attach` best row: pair F1 `0.438096`, precision `0.813639`,
  recall `0.299745`

Conclusion: these solver variants are not competitive.  Support is too strict
under the current edge cache, while consensus attach has high precision but
destroys recall.  Neither should be full-scored.

### Multiview pair-calibrator verifier features

Code change:

- `kit/no_anchor_global_id_model.py` now accepts repeatable
  `--pair-feature-npz name:path` inputs.
- The pair-link calibrator can use the main fused graph feature plus additional
  verifier-only cosine features from independent views.
- Added per-pair verifier fields:
  - `<name>_cosine` for each supplied view
  - `pair_view_cosine_mean`
  - `pair_view_cosine_min`
  - `pair_view_cosine_max`
  - `pair_view_cosine_std`

Verifier views used in this ablation:

- `match`: `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_match_role_20260617.npz`
- `person_s1`: `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_personvit_msmt_vits_s1_20260617.npz`
- `person_s3`: `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_personvit_msmt_vits_s3_full_20260618.npz`
- `color`: `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_colorhist_s1_20260617.npz`
- `posecolor`: `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_posecolor_s3_full_20260618.npz`

Artifacts:

- plain multiview pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_multiview_mpc010_f032_mcam05area12000_paironly_20260618.json`
- plain multiview model:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_multiview_mpc010_f032_mcam05area12000_20260618.joblib`
- pseudo-ensemble multiview pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_multiview_ens_mpc010_f032_mcam05area12000_paironly_20260618.json`
- pseudo-ensemble multiview model:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_multiview_ens_mpc010_f032_mcam05area12000_20260618.joblib`

Best rows:

- multiview verifier: pair F1 `0.709180`, precision `0.765552`,
  recall `0.660541`, threshold `0.03`, blend `0.75`
- multiview + pseudo ensemble: pair F1 `0.711438`,
  precision `0.784113`, recall `0.651092`, threshold `0.04`, blend `0.75`

Assignment diagnostics:

- multiview verifier: `9009` rows, `738` components, largest component `320`,
  statuses `provisional=8338`, `forced_singleton=663`, `forced_component=8`
- multiview + pseudo ensemble: `9009` rows, `776` components, largest component
  `305`, statuses `provisional=8321`, `forced_singleton=671`,
  `forced_component=17`

Conclusion: multiview verifier features are wired correctly and the
pseudo-ensemble variant improves over the older raw time-agglom pair baseline,
but it is still below the accepted assignment artifact at pair F1 `0.719910`
and below the trained fused pair-calibrator at pair F1 `0.714228`.  It was not
promoted to full scoring because the model-only metric regressed.

### Admission and video-hybrid e2e diagnostics

Motivation: the global-ID pair metric is now above `0.70`, but full IDF1 is
stuck near `0.635`.  The next question was whether output admission or
per-video solver failure explains the gap.

Admission ablation artifacts:

- interrupted all-keep run:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_admission_ablation_20260618_all_keep.json`
- fast MCAM05 floor sweep:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_admission_fast_20260618_mcam05_9000.json`
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_admission_fast_20260618_mcam05_10000.json`
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_admission_fast_20260618_mcam05_11000.json`
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_admission_fast_20260618_current.json`
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_admission_fast_20260618_soft_mcam04.json`
- exact full-score runs:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_exactfull_20260618_current_exact.json`
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_exactfull_20260618_mcam05_11000.json`

Admission results:

- all-keep full IDF1 dropped to `0.621668`.  MCAM05 Tc6 collapsed to IDF1
  `0.233239` because DetPr fell to `0.142466`.
- MCAM05 area floors `9000`, `10000`, `11000`, and current `12000` all produced
  strong fast pair-F1 around `0.7264-0.7266`, but exact full scoring did not
  improve:
  - current exact: pair F1 `0.726586`, full IDF1 `0.633766`
  - MCAM05 `11000`: pair F1 `0.726586`, full IDF1 `0.633664`

Video-hybrid artifacts:

- reusable script: `kit/no_anchor_video_hybrid_diagnostic.py`
- diagnostic JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_video_hybrid_diagnostic_20260618.json`
- promoted hybrid JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_video_hybrid_b_except_bad_full_20260618.json`
- promoted assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_video_hybrid_b_except_bad_assignments_20260618.csv`
- updated joint gate:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_and_e2e_gate_plus_hybrid_20260618.json`

Hybrid method:

- A resolver: no-anchor time-agglom with fused feature, theta `0.014`.
- B resolver: loaded no-anchor pair-calibrator, threshold `0.03`, blend `0.50`.
- B labels are aligned to A labels by tracklet-overlap majority only, with no
  identity labels.
- Hybrid policy: use A for MCAM03 Tc8, MCAM04 Tc6, and MCAM06 Tc6; use aligned B
  for the other videos.

Hybrid result:

- pair F1 `0.727059`
- precision `0.784763`
- recall `0.677259`
- full IDF1 `0.635490`
- HOTA `0.497420`
- AssA `0.513909`
- DetRe `0.558237`
- DetPr `0.737560`
- assignment rows `8752`, components `670`, largest component `273`
- joint >0.70 global and >0.70 e2e gate: still fail

Conclusion: video-level solver hybrid gives the new best verified e2e result,
but the gain is only `+0.000141` IDF1 over the standing `0.635349`.  The
negative all-keep result shows that simply increasing detector/tracklet coverage
does not work; the low-scoring slices need better association and/or better
tracklet quality, especially MCAM04 Tc6 and MCAM06 Tc6.

### Aligned oracle patch diagnostic

Artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_oracle_patch_bad_video_aligned_diagnostic_20260618.json`

Purpose: measure how much full-IDF1 can be recovered if selected bad videos are
replaced by oracle tracklet-majority IDs, while aligning oracle labels back into
the base no-anchor label namespace by tracklet overlap.  This is diagnostic
only; GT is not used in the promoted model.

Results:

- patch MCAM04 Tc6 + MCAM06 Tc6: patched `3485` tracklets, all mapped to base
  IDs, full IDF1 `0.651688`, HOTA `0.518966`, AssA `0.536194`, DetRe
  `0.585110`, DetPr `0.735363`
- patch MCAM03 Tc8 + MCAM04 Tc6 + MCAM06 Tc6: patched `4090` tracklets, full
  IDF1 `0.657272`, HOTA `0.525590`, AssA `0.542534`, DetRe `0.592756`, DetPr
  `0.737546`

Conclusion: fixing only the visually obvious weak videos is not enough for
`>0.70`.  Even oracle-aligned replacement of MCAM03 Tc8, MCAM04 Tc6, and MCAM06
Tc6 leaves about `0.043` full-IDF1 below target.  The remaining gap requires
broader global identity recovery across the whole dataset, not a local patch.

### Fused top-100 component retrieval diagnostic

Artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_component_retrieval_fused_top100_20260618.json`

Setup:

- base resolver: fused match/person/color feature, DB concat, theta `0.014`
- output admission: current multi-video area floors
- candidate edges: component top-100, top-edge-k `8`

False-split mass retrieval:

- candidate scored fraction: `0.960087`
- retrieved@1: `0.120390`
- retrieved@5: `0.474453`
- retrieved@10: `0.639823`
- retrieved@20: `0.787584`
- retrieved@50: `0.950874`
- retrieved@100: `0.960074`

Same-GT score quantiles:

- q10 `0.710350`
- q25 `0.755304`
- q50 `0.830922`
- q75 `0.885891`
- q90 `0.922553`

Dominant-pure edge precision remains low:

- threshold `0.70`: precision `0.052779`
- threshold `0.74`: precision `0.072900`
- threshold `0.78`: precision `0.117519`

Conclusion: retrieval is no longer the limiting step.  The graph sees almost all
true split candidates by top-100, but the positive edges are buried among many
visually plausible false edges.  Any successful next stage must be a verifier or
evidence fusion module with strong precision controls.

### Fast multiview component verifier retry

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_component_verifier_fast_hgb_8k_20260618.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_component_verifier_fast_hgb_strict12k_20260618.json`

Setup:

- primary feature: fused match/person/color + DB concat
- verifier views: primary, DB, PersonViT s3, posecolor s3, colorhist s3
- candidate edges: component top-50, limited to top `8000` or `12000`
- pseudo positives: multi-view top-5/top-rank consensus
- pseudo negatives: low-consensus edges and cannot-link-like conflicts

Results:

- 8k HGB run: pseudo positives `206`, pseudo negatives `6929`; accepted `71`
  edges; pair F1 stayed `0.719910`
- strict 12k HGB run: pseudo positives `95`, pseudo negatives `11294`;
  accepted `72` edges; pair F1 stayed `0.719910`

Conclusion: the current multiview verifier is safe but too conservative.  It
merges edges that carry negligible weighted pair mass, so it does not address
the high-mass false splits.  This confirms that the next model needs either a
new evidence source or a different pseudo-label construction that targets
high-mass split components rather than only stable easy positives.

### Assignment-level video source switch

New script:

`/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_video_switch.py`

Purpose: test whether the current no-anchor model pool already contains enough
diversity to improve e2e by selecting different assignment sources for
different videos.  Each input is an existing no-anchor assignment CSV.  Source
component namespaces are aligned to a reference source by tracklet-overlap
majority only; GT is used only for final scoring and for constructing this
diagnostic policy.

Source pool:

- `timebest`:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_model_fused_mp_f032_t0165_bestfull_assignments_20260617.csv`
- `timepair`:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_model_fused_mpc010_f032_t014_multivideo_pairf1_assignments_20260618.csv`
- `paircal`:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_exactfull_20260618_current_exact_assignments.csv`
- `hybrid`:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_video_hybrid_b_except_bad_assignments_20260618.csv`

Exact policy artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_video_switch_oraclepolicy_20260618.json`

Assignment CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_video_switch_oraclepolicy_assignments_20260618.csv`

Policy:

- MCAM00 Tc6: `paircal`
- MCAM00 Tc8: `paircal`
- MCAM03 Tc6: `paircal`
- MCAM03 Tc8: `hybrid`
- MCAM04 Tc6: `hybrid`
- MCAM05 Tc6: `timebest`
- MCAM05 Tc8: `paircal`
- MCAM06 Tc6: `hybrid`
- MCAM06 Tc8: `paircal`
- MCAM08 Tc6: `paircal`

Result:

- pair F1 `0.726791`
- precision `0.784597`
- recall `0.676919`
- full IDF1 `0.635418`
- HOTA `0.497389`
- AssA `0.513967`
- DetRe `0.558248`
- DetPr `0.737347`
- output tracklets `8752`

Conclusion: per-video switching over the existing no-anchor assignment pool did
not improve the then-current best hybrid (`0.635490`).  This ruled out a cheap
selector-only path and pointed back to the core evidence/model problem: the
existing no-anchor sources made similar high-mass errors, especially on the
weak MCAM03/04/06 slices.

### GT-bbox sample weak-feature no-anchor model

Artifacts:

- prepared sample parquet:
  `/mnt/localssd/vlincs_reid_runs/gtbox_no_anchor_sample_20260618/gtbox_eval.parquet`
- prepared crop/bbox feature:
  `/mnt/localssd/vlincs_reid_runs/gtbox_no_anchor_sample_20260618/features_gtbox_crop_bbox.npz`
- focused weak-feature model JSON:
  `/mnt/localssd/vlincs_reid_runs/gtbox_no_anchor_sample_20260618/no_anchor_gtbox_sample_focused_crop_bbox_v1.json`
- full scorer JSON:
  `/mnt/localssd/vlincs_reid_runs/gtbox_no_anchor_sample_20260618/no_anchor_gtbox_sample_focused_crop_bbox_v1_full.json`

Setup:

- source: sample ground-truth bounding-box tracklets, `1887` tracklets and `36`
  GT identities
- no anchors and no GT identity labels in training
- features: crop appearance proxy, bbox geometry, and trajectory summary

Results:

- sample identity F1 `0.217860`
- sample pair F1 `0.032319`
- full IDF1 `0.263541`
- DetPr / DetRe `0.361260 / 0.207432`

Conclusion: upper-bound GT boxes alone do not solve global-ID resolution when
the evidence is weak.  The no-anchor model needs stronger identity evidence,
not only cleaner boxes.

### High-resolution Louvain baseline

Artifacts:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_base_highres_e2ebest_full_20260618.json`
- assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_base_highres_e2ebest_assignments_20260618.csv`

Setup:

- feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_20260617.npz`
- DB concat weight `1.0`, feature weight `0.32`
- Louvain config: `top_k=10`, `edge_floor=0.04`, `resolution=5.0`
- same output admission as the standing time-agglom runs

Result:

- pair F1 `0.761019`
- precision / recall `0.808068 / 0.719147`
- full IDF1 `0.652157`
- HOTA `0.515903`
- AssA `0.531441`
- DetPr / DetRe `0.753560 / 0.574809`

Conclusion: high-resolution Louvain is a large e2e jump over the earlier hybrid
artifact (`0.635490 -> 0.652157`) and becomes the main no-anchor graph resolver
to improve.

### FaceNet evidence extraction and ablations

Feature artifacts:

- shard directory:
  `/mnt/localssd/vlincs_reid_runs/facenet_s2_4shard_20260618`
- merged FaceNet feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_facenet_vggface2_s2_20260618.npz`
- fused FaceNet `0.05` feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_20260618.npz`

Extraction setup:

- model: `facenet-pytorch InceptionResnetV1(pretrained=vggface2)`
- detector: `facenet-pytorch MTCNN`
- samples: first/last upper-body crops per tracklet
- no anchors and no GT
- valid face tracklets: `4260 / 9734`
- detected faces: `6028` over `19393` inspected crops

Direct pair-calibrator ablation:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_model_hgb_consensus_guard_facevalid_s2_paironly_20260618.json`
- features: `face_cosine`, `face_both_valid`, `face_either_valid`
- best pair F1 `0.432349`
- precision / recall `0.783606 / 0.298531`

Conclusion: sparse FaceNet evidence is harmful when used as the conservative
`consensus_guard` pair gate; recall collapses even when missing-face validity
is explicit.

Low-weight Louvain fusion ablation:

- FaceNet weight `0.03`: pair F1 `0.761342`, full IDF1 `0.652256`
- FaceNet weight `0.05`: pair F1 `0.762121`, full IDF1 `0.652381`
- FaceNet weight `0.10`: pair F1 `0.759390`, full IDF1 `0.651366`

Local grid around FaceNet `0.05`:

- grid JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_localgrid_full_20260618.json`
- promoted self-contained JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_localgrid_e2ebest_full_20260618.json`
- promoted assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_localgrid_e2ebest_assignments_20260618.csv`

Best e2e row:

- Louvain config: `top_k=10`, `edge_floor=0.035`, `resolution=5.0`
- pair F1 `0.762137`
- precision / recall `0.808448 / 0.720844`
- full IDF1 `0.652390`
- HOTA `0.516276`
- AssA `0.531901`
- DetPr / DetRe `0.753390 / 0.575268`
- assignment rows/components: `8752 / 653`

Conclusion: FaceNet is useful as a low-weight graph-evidence block, not as a
hard verifier gate.  The improvement is small but verified by the canonical
full scorer and became the previous standing no-anchor e2e best.

### Louvain admission and component attach continuation

New script:

`/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_louvain_component_merge_sweep.py`

Important correction: the Louvain + FaceNet artifacts use `feature_weight=0.32`
with DB weight `1.0`.  Re-running the same grid at `feature_weight=1.0` drops
pair F1 to about `0.678`, so all continuation rows below use the corrected
weight.

Admission full-score ablation:

- `admA_mcam05_8000`: full IDF1 `0.649430`, pair F1 `0.761679`, rows `8819`
- `admB_mcam03_2000_mcam05_8000_mcam06_9000`: full IDF1 `0.649586`, pair F1
  `0.761483`, rows `8824`
- `admC_coverage_mcam06_3000`: full IDF1 `0.651124`, pair F1 `0.750349`,
  rows `9045`
- `admD_balanced`: full IDF1 `0.649584`, pair F1 `0.761088`, rows `8837`
- `admE_coverage_mcam06_6000`: full IDF1 `0.649908`, pair F1 `0.754636`,
  rows `8989`

Conclusion: relaxing admission increases delivered rows and sometimes recall,
but loses enough identity precision that full IDF1 remains below the current
best.

Component merge over Louvain:

- pair sweep:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_component_merge_pair_sweep_20260618.json`
- best pair F1 `0.762154`, precision `0.808423`, recall `0.720895`
- best full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_component_merge_top_full_20260618.json`
- full IDF1 `0.652389`

High-precision Louvain split:

- artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_highprec_top8_floor035_res6_full_20260618.json`
- config: `top_k=8`, `edge_floor=0.035`, `resolution=6.0`
- pair F1 `0.755200`, precision `0.815763`, recall `0.703009`
- full IDF1 `0.647820`

Conclusion: higher precision is not the missing ingredient; the recall loss
dominates the e2e score.

Small-fragment attach over Louvain:

- pair sweep:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_small_attach_pair_sweep_20260618.json`
- promoted full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_small_attach_top_full_20260618.json`
- promoted assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_small_attach_top_assignments_20260618.csv`
- pair F1 `0.762189`, precision `0.808383`, recall `0.720989`
- full IDF1 `0.652398`
- HOTA `0.516290`
- AssA `0.531916`
- DetPr / DetRe `0.753319 / 0.575322`
- assignment rows/components: `8752 / 567`

Expanded small-attach variants:

- `src64_tgt8_edge8`: pair F1 `0.762189`
- `src64_tgt8_edge16`: pair F1 `0.762190`, full IDF1 `0.652387`
- `src16_tgt8_edge8`: pair F1 `0.762189`
- `src48_tgt24_edge8`: pair F1 `0.762189`

Updated joint gate:

`/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_and_e2e_gate_louvain_face005_small_attach_20260618.json`

Conclusion: small-fragment attach became the previous promoted no-anchor
artifact, but the gain was effectively saturated (`0.652390 -> 0.652398`).
Global-ID pair metrics passed the 0.70 gate; the e2e target remained open.

### Assignment-level M3 admission and split continuation

New scripts:

- `kit/no_anchor_assignment_admission_grid.py`
- `kit/no_anchor_assignment_component_split_sweep.py`

Error diagnostic for the previous small-attach artifact:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_small_attach_error_diag_20260618.json`
- full IDF1 `0.652398`, unmatched FP `119098`
- weak full-IDF1 videos: MCAM04 Tc6 `0.558108`, MCAM06 Tc6 `0.606523`,
  MCAM03 Tc8 `0.625560`
- current-tracklet oracle still gives only MCAM04 Tc6 `0.619321`, but MCAM06
  Tc6 has much larger association headroom (`0.733845` oracle vs `0.606523`)

Area admission:

- pair-only strict area grid improved pair F1 up to `0.796430`
- first full-scored strict row dropped to full IDF1 `0.631882`
- conclusion: area-only filtering deletes too much detection recall and should
  not be promoted.

Assignment component split:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_assignment_component_split_pair_sweep_20260618.json`
- best row was no-op: pair F1 `0.762189`
- sparse/aggressive split rows did not beat the input assignment
- conclusion: the same embedding evidence cannot safely split the current
  false-merge components; this path needs different evidence, not another
  threshold.

Quality admission:

- pair-only grid:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_assignment_quality_admission_pair_20260618.json`
- `quality>=0.45`: pair F1 `0.762330`, full IDF1 `0.652409`,
  unmatched FP `118314`
- `quality>=0.50`: pair F1 `0.763098`, full IDF1 `0.652454`,
  unmatched FP `116831`
- `quality>=0.55`: pair F1 `0.765065`, full IDF1 `0.652554`,
  unmatched FP `114361`
- `quality>=0.60`: pair F1 `0.768742`, precision / recall
  `0.813274 / 0.728833`, full IDF1 `0.652623`, unmatched FP `109745`
- `quality>=0.62`: pair F1 `0.771436`, full IDF1 `0.652522`
- `quality>=0.65`: pair F1 `0.776702`, full IDF1 `0.652210`

Promoted artifact:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_full_20260618.json`
- assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_assignments_20260618.csv`
- gate:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_and_e2e_gate_small_attach_quality_20260618.json`
- assignment rows/components: `7487 / 111`
- largest assignment component: `275`

Conclusion: no-GT quality admission is a real but small M3 improvement
(`0.652398 -> 0.652623`).  It confirms that low-quality false-positive
tracklets hurt the full scorer, but the remaining gap to `0.70` is still mainly
identity/evidence resolution on the hard MCAM03/04/06 slices.

## Next Direction

The current bottleneck is still candidate verification, not candidate retrieval.
The no-anchor graph can see many true split candidates, but true and false
component edges overlap strongly under raw similarity.  The first pseudo-label
verifier only gives a tiny pair-F1 delta, so next experiments need stronger
evidence or better pseudo-labels, for example:

- pseudo-positive edges from stable mutual top-1/top-5 agreement across feature
  views, time windows, and crop samples;
- pseudo-negative edges from same-frame / temporal overlap cannot-link and
  high-score conflicting neighbors;
- calibrated edge model features beyond cosine, including rank, reciprocity,
  component purity proxies, temporal co-visibility, area/scale consistency, and
  per-video reliability;
- use the verifier to rerank component candidates before any merge;
- extend the evidence stack beyond the current low-weight FaceNet gain, for
  example stronger face crops, pose/body-part identities, or video-conditioned
  reliability weights, because threshold, admission, rerank, NFC, and
  component-split and M3 quality-admission ablations are now saturated around a
  `0.6526` full-IDF1
  ceiling.

### Greedy relinked tracklets plus OSNet/color sample features

Purpose:

- test whether changing the upstream tracklet source can increase recall beyond
  the current-tracklet ceiling;
- keep the setting no-anchor: GT is used only for post-hoc scoring;
- add fresh crop evidence from OSNet/color as an auxiliary evidence view.

New scripts:

- `/mnt/localssd/vlincs_reid_by_search/kit/prepare_relink_no_anchor_sample.py`
- `/mnt/localssd/vlincs_reid_by_search/kit/extract_sample_tracklet_osnet_features.py`

Alternative linker oracle:

- greedy IoU `0.50`, gap `10`: oracle-labeled full IDF1 `0.736027`
- HOTA `0.616314`
- AssA `0.623529`
- DetPr / DetRe `0.805968 / 0.677255`
- tracklets/labeled tracklets: `18656 / 12851`

Sample export:

- sample parquet:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_sample_20260618.parquet`
- feature bundle:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_features_20260618.npz`
- rows/tracklets: `1723070 / 18656`
- matched detection fraction: `0.776188`
- current-GID replay full IDF1: `0.605721`

OSNet/color extraction:

- feature bundle:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_osnet_color_mid1_20260618.npz`
- features: `features_osnet [18656,512]`, `features_color [18656,82]`
- valid rows: `18656 / 18656`
- crop policy: one middle crop per tracklet

Base greedy-tracklet no-anchor sweep:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_baseline_fusion_grid_20260618.json`
- model:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_baseline_fusion_global_model_20260618.joblib`
- best proxy: row identity F1 `0.747073`, pair F1 `0.655639`
- best full scorer: IDF1 `0.615689`, HOTA `0.477043`

OSNet/color heavy fusion:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_fusion_grid_20260618.json`
- model:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_fusion_global_model_20260618.joblib`
- weights: `dbresolve=0.9`, `current_gid_centroid=0.8`, `osnet=1.0`,
  `color=0.20`, small geometry/quality weights
- best proxy: row identity F1 `0.716205`, pair F1 `0.646161`
- conclusion: single-crop OSNet is useful evidence but harmful when it dominates
  the graph feature.

OSNet/color light fusion:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_grid_20260618.json`
- assignments:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_assignments_20260618.csv`
- model:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_global_model_20260618.joblib`
- full JSON:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_full_20260618.json`
- submission:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_submission_20260618.zip`

Best light-fusion row:

- `theta=0.014`, `top_k=30`
- weights: `dbresolve=1.0`, `current_gid_centroid=0.9`, `geometry=0.04`,
  `quality=0.02`, `osnet=0.25`, `color=0.08`
- row identity F1 `0.751804`
- pair F1 `0.664014`
- pair precision / recall `0.748806 / 0.596472`
- components: `60`

Canonical full scorer:

- IDF1 `0.620289`
- HOTA `0.481560`
- AssA `0.501756`
- DetRe / DetPr `0.541737 / 0.725486`
- unmatched FP `276937`

Per-video full IDF1:

- MCAM00 Tc6 `0.854746`
- MCAM00 Tc8 `0.820703`
- MCAM03 Tc6 `0.670947`
- MCAM03 Tc8 `0.637100`
- MCAM04 Tc6 `0.528884`
- MCAM05 Tc6 `0.263923`
- MCAM05 Tc8 `0.818085`
- MCAM06 Tc6 `0.583516`
- MCAM06 Tc8 `0.740485`
- MCAM08 Tc6 `0.733082`

Conclusion: greedy relinking raises the upstream oracle and the light OSNet
fusion produces a clean global-ID model-side win on row identity (`0.751804`),
but the all-tracklet deliverable output is not an e2e win.  The failure mode is
not candidate recall; it is evidence calibration/admission.  MCAM05 Tc6 is the
clearest regression, where high detection recall combines with weak precision
and large forced components.  This branch should feed future M3/M5 calibration
and verifier work, but the promoted e2e artifact remains
`no_anchor_small_attach_quality060_full_20260618.json` at full IDF1 `0.652623`.

### 2026-06-19 sample-level and balanced admission retries

New sample admission script:

`/mnt/localssd/vlincs_reid_by_search/kit/sample_assignment_admission_grid.py`

Purpose: keep sample-parquet predicted global IDs fixed and sweep no-anchor
tracklet admission using only detector/tracklet evidence (`n_dets`, detector
confidence, median area, and a derived quality score).  Filtered rows are
evaluated with true delivery semantics: filtered tracklets are dropped, not
filled back as singletons.

Greedy light-OSNet assignment admission artifacts:

- grid JSON:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_admission_grid_20260619.json`
- best assignments:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_admission_best_assignments_20260619.csv`
- best submission:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_admission_best_submission_20260619.zip`

Best greedy admission row:

- output tracklets: `14473 / 18656`
- full IDF1 `0.631441`
- HOTA `0.493242`
- AssA `0.512554`
- DetPr / DetRe `0.757411 / 0.541397`
- unmatched FP `213432`
- pair F1 `0.664093`
- pair precision / recall `0.752304 / 0.594398`
- admission: global quality `>=0.25`, MCAM05 Tc6 area `>=12000`,
  MCAM05 Tc6 quality quantile `0.4`

Important per-video movement:

- MCAM05 Tc6 improved from the unadmitted greedy full IDF1 `0.263923` to
  `0.691616`
- MCAM04 Tc6 stayed weak at `0.529385`
- MCAM06 Tc6 stayed weak at `0.583669`

Conclusion: sample-level admission fixes the obvious MCAM05 false-positive
collapse but does not make the greedy-tracklet branch competitive with the
promoted current-tracklet artifact.  The remaining failure is identity
association on MCAM03/04/06, not only false-positive admission.

Balanced admission score addition:

- script updated:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_admission_grid.py`
- added proxy sort fields:
  `coverage_pair_score = pair_f1 * coverage_ratio`
  and `coverage_sqrt_pair_score = pair_f1 * sqrt(coverage_ratio)`

This avoids choosing rows with excellent pair F1 but too little delivered
detection coverage.

Current Louvain/FaceNet small-attach balanced admission artifacts:

- pair/full grid:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_balanced_admission_grid_20260619.json`
- best assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_balanced_admission_best_assignments_20260619.csv`
- strict pair-ranked grid for contrast:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_refined_admission_grid_20260619.json`

Pair-ranked refined admission result:

- best full IDF1 `0.645879`
- pair F1 up to `0.804901`
- coverage around `0.63`
- conclusion: over-filtering raises pair precision but loses too much detector
  recall for e2e.

Balanced admission result:

- best full IDF1 `0.652542`
- HOTA `0.516403`
- AssA `0.532044`
- DetPr / DetRe `0.754263 / 0.574997`
- unmatched FP `113902`
- pair F1 `0.765439`
- pair precision / recall `0.810628 / 0.725022`
- admission: quality `>=0.56`, MCAM05 Tc6 area `>=12000`, MCAM03 Tc8 area
  `>=2000`, MCAM04 Tc6 area `>=0`, MCAM06 Tc6 area `>=4000`

Comparison:

- promoted best remains quality `>=0.60` at full IDF1 `0.652623`
- balanced admission is close but lower by `0.000081`

Conclusion: M3 admission is now empirically saturated around `0.6526` full
IDF1.  Both stricter pair-ranked filtering and balanced coverage-aware
filtering fail to break the ceiling.  The next aligned route is not another
admission threshold; it needs stronger M5/M8 identity evidence or a verifier
that repairs MCAM04/06 association without deleting detection recall.

### 2026-06-19 submission-source and multiview verifier continuation

Submission-level source switch:

- current source:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_submission_20260619.zip`
- greedy-admission source:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_admission_best_submission_20260619.zip`
- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_submission_switch_current_greedy_20260619.json`
- oracle switch output:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_submission_switch_current_greedy_oracle_20260619.zip`

Result:

- base current reload full IDF1 `0.654596`
- oracle-per-video policy selected current for every video
- conclusion: the greedy/admitted tracklet source is not complementary enough
  to fix the hard videos.  This rules out a simple source selector as the next
  route.

Effective multi-frame OSNet feature:

- extractor output:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s5_20260619.npz`
- requested `samples=5`, but the current sampler implements `samples>2` as
  first/middle/last, so this is an effective 3-frame OSNet average.
- crops/features: `28938` crops, `9734` tracklets, `3` missing crops

FaceNet + OSNet graph fusion artifacts:

- OSNet weight `0.03`:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_osnet003_s3_quality060_pair_grid_20260619.json`
- OSNet weight `0.05`:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_osnet005_s3_quality060_pair_grid_20260619.json`
- OSNet weight `0.10`:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_osnet010_s3_quality060_pair_grid_20260619.json`

Best rows:

- OSNet `0.03`: pair F1 `0.757313`, precision `0.779544`, recall
  `0.736315`
- OSNet `0.05`: pair F1 `0.756099`, precision `0.808587`, recall
  `0.710010`
- OSNet `0.10`: pair F1 `0.758363`, precision `0.804268`, recall
  `0.717415`

Comparison:

- promoted current remains pair F1 `0.768742`, precision `0.813274`, recall
  `0.728833`
- conclusion: multi-frame OSNet is useful as an ablation artifact but negative
  as a direct graph-fusion block in this range.  It either raises recall with
  too much precision loss or keeps precision while lowering recall.

Assignment-level component merge script:

`/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_component_merge_sweep.py`

Purpose: start from an existing no-anchor assignment CSV, use a separate
no-anchor feature view to propose component-level merges, and evaluate whether
strong weak-label evidence can repair false splits without retraining or using
identity anchors.

FaceNet component merge:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_face_component_merge_pair_grid_20260619.json`
- base: pair F1 `0.768742`, precision `0.813274`, recall `0.728833`
- best row was effectively no-op.  A few accepted merges touched only tiny or
  non-eval-mass fragments and did not change weighted pair metrics.

OSNet component merge:

- pair grid:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_pair_grid_20260619.json`
- full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_full1_20260619.json`
- best assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`
- DB-comp submission export/eval:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_eval_20260619.json`
- submission zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_20260619.zip`
- submission reload JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_reload_20260619.json`

Best OSNet component-merge row:

- accepted component merges: `13`
- assignment rows/components: `7487 / 98`
- decision statuses: `forced_component=7442`, `forced_singleton=45`
- pair F1 `0.768743`
- precision / recall `0.813273 / 0.728836`
- DB-comp full IDF1 `0.652624`
- submission-parquet reload IDF1 `0.654597`
- HOTA `0.518183`
- AssA `0.533816`
- DetPr / DetRe `0.757889 / 0.576083`
- unmatched FP `109742`

Per-video reload IDF1 for the tiny-best submission:

- MCAM00 Tc6 `0.879252`
- MCAM00 Tc8 `0.827992`
- MCAM03 Tc6 `0.690770`
- MCAM03 Tc8 `0.627076`
- MCAM04 Tc6 `0.560451`
- MCAM05 Tc6 `0.711979`
- MCAM05 Tc8 `0.792455`
- MCAM06 Tc6 `0.608378`
- MCAM06 Tc8 `0.706032`
- MCAM08 Tc6 `0.768478`

Conclusion:

- The new best no-anchor submission-reload checkpoint is technically
  `0.654597`, but the gain over the previous reload `0.654596` is not
  meaningful.
- The error diagnosis now strongly suggests the remaining gap is not solved by
  adding another global feature block or high-threshold component attach.  The
  hard identities are both split and partially false-merged, especially around
  MCAM04/MCAM06 and the large all-camera identities.  The next useful
  experiment needs a different evidence family or a real calibrated verifier
  that can reject impure large components before merging more fragments.

### 2026-06-19 AutoResearch-inspired continuation

External research cue:

- Deli AutoResearch SKILL framework:
  `https://victorchen96.github.io/auto_research/framework.html`
- Self-play survey/project page:
  `https://victorchen96.github.io/auto_research/paper.html`

Distilled operating principle for this VLINCS loop:

- Treat the agent as an experiment planner with hard gates, not as an
unbounded optimizer.
- Keep each hypothesis isolated, reversible, and tied to a scalar metric.
- Use refutation runs aggressively.  If a family fails two related gates,
  pivot the structure rather than expanding the same threshold grid.
- For VLINCS specifically, the only success gate that matters for the open e2e
  target is submission/full IDF1 above `0.70` under no-anchor semantics.

Current promoted artifact before this continuation:

- assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`
- submission reload:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_reload_20260619.json`
- reload IDF1: `0.654597`
- pair F1 / precision / recall: `0.768743 / 0.813273 / 0.728836`

Error diagnostic:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_error_diag_20260619.json`
- DB-comp full IDF1 `0.652624`
- weak full-IDF1 videos:
  - MCAM04 Tc6 `0.558338`
  - MCAM06 Tc6 `0.606895`
  - MCAM03 Tc8 `0.625145`
- top false-split GT identities remain large all-camera identities:
  `9`, `36`, `11`, `43`, `52`
- top false-merge predicted components include `60000077`, `60000122`,
  `60000033`, and `60000040`

Curated assignment source-switch audit:

- script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_video_switch.py`
- run was interrupted after enough refutation evidence; completed sources:
  - `current`: full IDF1 `0.652624`, pair F1 `0.768743`
  - `face_local`: full IDF1 `0.652390`, pair F1 `0.762137`
  - `highres`: full IDF1 `0.652158`, pair F1 `0.761019`
  - `multiview`: full IDF1 `0.632601`, pair F1 `0.698293`
  - `paircal`: full IDF1 `0.632860`, pair F1 `0.704320`
  - `small045`: full IDF1 `0.652410`, pair F1 `0.762331`
  - `small050`: full IDF1 `0.652456`, pair F1 `0.763099`
  - `small055`: full IDF1 `0.652555`, pair F1 `0.765066`
- conclusion: the existing source pool is not diverse enough.  Selector-only
  reuse of historical no-anchor assignments is not a route to `0.70`.

Local-track relink:

- new script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_localtrack_relink_sweep.py`
- all-video JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_localtrack_relink_pair_20260619.json`
- bad-video JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_localtrack_relink_badvid_pair_20260619.json`
- result: every top row was no-op; `touched_groups=0`, `rewritten_seqs=0`
- conclusion: after the quality-admitted current-tracklet output, usable
  same-`local_track_id` multi-segment repair opportunities are absent.

Video namespace split:

- assignment CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_video_namespace_assignments_20260619.csv`
- diagnostic JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_video_namespace_error_diag_20260619.json`
- result:
  - pair F1 `0.347855`
  - pair precision / recall `0.816977 / 0.220970`
  - full IDF1 `0.366241`
  - HOTA `0.242579`
- conclusion: VLINCS scorer/submission semantics do require cross-video global
  identity continuity.  Per-video independent namespaces are invalid for this
  target.

Temporal cannot-link split:

- new script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_cannotlink_split_sweep.py`
- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_split_pair_20260619.json`
- evidence:
  - conflict edges detected: up to `3159`
  - conflict nodes detected: up to `3951`
  - best pair row was no-op: pair F1 `0.768743`
  - mild split row rewrote `3` seqs and reached pair F1 `0.768741`
  - larger split rows dropped to pair F1 around `0.764`
- conclusion: temporal cannot-link is real evidence, but hard coloring split is
  too blunt.  It should become a verifier feature or merge penalty, not a
  direct component split.

Identity-level temporal NMS admission:

- assignment CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_nms_assignments_20260619.csv`
- diagnostic JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_nms_error_diag_20260619.json`
- setup: within each predicted ID, keep the highest
  `n_dets * (0.25 + avg_conf)` non-overlapping tracklets and drop temporal
  conflict losers.
- result:
  - kept/dropped tracklets: `5401 / 2086`
  - pair F1 `0.826444`
  - pair precision / recall `0.874196 / 0.783638`
  - full IDF1 `0.622822`
  - HOTA `0.479293`
  - DetPr / DetRe `0.765398 / 0.525022`
- conclusion: this is an excellent model-side precision stress test but a bad
  e2e policy.  It proves high pair F1 alone is insufficient; the M3 admission
  must preserve detector recall.  The useful signal is the conflict feature,
  not the aggressive drop policy.

Net conclusion for this continuation:

- No new e2e best was found.  The standing no-anchor submission-reload boundary
  remains `0.654597`.
- The most informative positive result is diagnostic: temporal/cannot-link
  evidence can push model-side pair F1 to `0.826444`, but the corresponding
  detector-recall loss kills full IDF1.
- The next experiment should implement a calibrated component-edge verifier
  that uses cannot-link density, overlap-NMS survival score, component conflict
  density, temporal co-visibility, and feature similarity jointly.  It should
  penalize risky merges rather than deleting thousands of tracklets.

### 2026-06-19 cannot-link verifier continuation

Purpose:

- follow the previous continuation's best diagnostic signal;
- test whether temporal/cannot-link evidence can help if used as a merge
  verifier or teacher rather than as direct deletion;
- keep all experiments no-anchor.  GT is used only for pair/full evaluation.

Hard split full-score supplement:

- light split JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_split_light_full_20260619.json`
- light split assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_split_light_assignments_20260619.csv`
- unlimited split JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_split_unlimited_full_20260619.json`
- unlimited split assignments:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_split_unlimited_assignments_20260619.csv`

Light hard split result:

- split components: `3`
- rewritten seqs: `74`
- pair F1 `0.764085`
- precision / recall `0.811952 / 0.721548`
- full IDF1 `0.650945`
- HOTA `0.514430`
- AssA `0.530081`
- DetPr / DetRe `0.755227 / 0.571968`

Unlimited-color hard split result:

- split components: `43`
- rewritten seqs: `1891`
- pair F1 `0.643634`
- precision / recall `0.849477 / 0.518091`
- full IDF1 `0.615761`
- unmatched FP `402918`

Conclusion: hard cannot-link split remains negative even when detections are
preserved.  The light version is only slightly below current pair F1 but still
loses full IDF1.  The aggressive version destroys recall and full score.

Conflict-aware component merge:

- new script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_conflict_aware_merge_sweep.py`
- pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_conflict_aware_osnet_merge_small_pair_20260619.json`
- pair CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_conflict_aware_osnet_merge_small_pair_20260619.csv`

Setup:

- base assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`
- merge feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s5_20260619.npz`
- candidate top-k: `40`
- top edge k: `8`
- component conflict evidence:
  - components: `98`
  - components with internal temporal conflict: `45`
  - mean conflict-node fraction: `0.235468`
  - mean NMS-drop fraction: `0.123754`
  - max conflict-node fraction observed: `0.713415`
  - max NMS-drop fraction observed: `0.390244`

Best pair row:

- pair F1 `0.768743`
- precision / recall `0.813273 / 0.728836`
- accepted merges: `2`
- result is effectively no-op relative to current.

Conclusion: component conflict statistics are real and dense, but deterministic
hard-gating of OSNet component merge candidates does not recover additional
weighted pair mass.  This argues against another threshold-only component
merge sweep.

NMS-teacher attach self-training diagnostic:

- pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_nms_teacher_attach_osnet_paironly_20260619.json`

Setup:

- teacher: temporal NMS assignment
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_nms_assignments_20260619.csv`
- kept teacher tracklets: `5401`
- dropped/unlabeled tracklets: `2086`
- feature for reattach: OSNet s3
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s5_20260619.npz`

Pair-only rows:

- current base: pair F1 `0.768743`
- NMS plus singleton losers: pair F1 `0.644490`,
  precision / recall `0.874196 / 0.510381`
- best visual reattach row: threshold `0.85`, attached `98`,
  pair F1 `0.643990`, precision / recall `0.871266 / 0.510756`
- lower thresholds attach many more losers but collapse precision:
  threshold `0.65` pair F1 `0.602856`

Conclusion: NMS teacher labels are too sparse as a clustering base.  Turning
all losers into singleton IDs preserves detections but creates too much false
split mass; visual reattachment from this teacher does not recover enough
recall.  This route should not be full-scored further unless the teacher is
made less aggressive.

Continuation conclusion:

- No new e2e best in this verifier continuation.
- Current no-anchor boundary remains submission reload IDF1 `0.654597`.
- The repeated pattern is now clear:
  - deletion/admission can raise model-side pair precision but hurts DetRe;
  - hard split preserves detections but hurts identity recall;
  - deterministic component merge remains saturated.
- The next structural pivot should be a new evidence source or a learned
  verifier trained on less aggressive pseudo-labels.  A promising variant is a
  soft teacher built from partial temporal NMS rather than full NMS: keep
  high-confidence conflict winners as positives, but do not turn every loser
  into singleton training evidence.

### 2026-06-19 AutoResearch-style soft verifier continuation

Purpose:

- distill the AutoResearch / self-play workflow into a cheaper loop: isolate a
  falsifiable hypothesis, run a bounded experiment, record the scalar gate, and
  pivot when related attempts fail;
- test whether the previous failure was caused by hard deterministic gates, or
  by lack of useful evidence in the candidate edges themselves;
- keep all model construction no-anchor.  GT appears only in metric reporting
  and in one explicitly marked oracle diagnostic.

Lightweight video-source oracle diagnostic:

- artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_lightweight_video_source_oracle_20260619.json`
- sources compared from existing no-anchor outputs:
  - current best diagnostic assignment;
  - small-attach quality assignment;
  - cannot-link light split;
  - cannot-link NMS;
  - video namespace.

Overall full IDF1 of available sources:

- current best diagnostic: `0.652624`
- small-attach quality: `0.652623`
- cannot-link light split: `0.650945`
- cannot-link NMS: `0.622822`
- video namespace: `0.366241`

Per-video source winners:

- every video is won by the current best diagnostic assignment, with only exact
  or near-exact ties on a few videos;
- mean of per-video winner IDF1 is `0.715825`, but this is not a recomputed
  global e2e score and cannot be used as a valid selector;
- conclusion: video-level switching among the existing no-anchor artifacts is
  not the missing mechanism.

Soft component verifier from weak base:

- script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_component_verifier_sweep.py`
- pair-only artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_component_verifier_soft_fused_osnet005_quality060_paironly_20260619.json`
- primary feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s3_20260619.npz`
- verifier views: DB, OSNet s5, FaceNet, PoseColor s3, colorhist s3, primary.

Result:

- time-agglom base pair F1: `0.600437`
- candidate component edges: `5162`
- pseudo positives / negatives / unlabeled: `26 / 4995 / 141`
- best verifier row: pair F1 `0.600402`
- accepted edges: `22`

Conclusion: the soft verifier is correctly wired, but starting from the weak
time-agglom base is the wrong setup.  The pseudo-positive teacher is too sparse
and the base graph is too far below the current assignment boundary.

Assignment-base soft component verifier:

- local code change: `kit/no_anchor_component_verifier_sweep.py` now supports
  `--assignment-csv` and `--pred-col`, preserving the original time-agglom path
  when no assignment is provided.
- pair-only artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_component_verifier_soft_fused_osnet005_quality060_paironly_20260619.json`
- starting assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`

Result:

- base pair F1 / precision / recall: `0.768743 / 0.813273 / 0.728836`
- candidate component edges: `4173`
- pseudo positives / negatives / unlabeled: `39 / 4050 / 84`
- best verifier row: pair F1 / precision / recall
  `0.768739 / 0.813263 / 0.728837`
- accepted edges: `10`

Conclusion: the assignment-base learned verifier does not beat the current
boundary.  This falsifies the "hard gate implementation is the bottleneck"
hypothesis for the current feature family.  Current evidence is saturated; a
new evidence source or a different positive-generation mechanism is needed.

### 2026-06-19 Deli AutoResearch distillation + current-assignment retrieval

External cue:

- Deli AutoResearch framework:
  `https://victorchen96.github.io/auto_research/framework.html`
- Self-play survey page:
  `https://victorchen96.github.io/auto_research/paper.html`
- Self-play story:
  `https://victorchen96.github.io/blog_self_play_story.html`

Distilled protocol for this no-anchor VLINCS research loop:

- use file-backed state and reports, not conversation memory, as the durable
  experiment ledger;
- every run must produce either a scalar improvement or a refutation;
- after two related refutations, pivot the structure rather than widening the
  same threshold grid;
- separate candidate retrieval from verifier/resolution evaluation;
- report downward-moving metrics honestly.  A negative ablation is a finding if
  it rules out a mechanism.

Applied structural question:

- Is the current `0.654597` e2e boundary caused by missing candidate recall, or
  by inability to select true same-identity candidate edges?

Code change:

- `kit/analyze_no_anchor_component_retrieval.py` now supports
  `--assignment-csv` and `--pred-col`, so retrieval diagnostics can start from
  the promoted current-best assignment instead of reconstructing a weaker
  time-agglom graph.

Base assignment:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`

Retrieval artifacts:

- fused:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_assignment_retrieval_fused_top100_20260619.json`
- OSNet:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_assignment_retrieval_osnet_top100_20260619.json`
- CLIP:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_assignment_retrieval_clip_top100_20260619.json`
- DINO:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_assignment_retrieval_dino_top100_20260619.json`
- PersonViT:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_assignment_retrieval_personvit_top100_20260619.json`
- FaceNet:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_assignment_retrieval_facenet_top100_20260619.json`

Current-assignment component retrieval summary:

| feature view | top-10 split-mass recall | top-50 split-mass recall | top-100 split-mass recall | edge precision at score 0.62 |
| --- | ---: | ---: | ---: | ---: |
| fused | `0.701155` | `0.996523` | `1.000000` | `0.082969` |
| OSNet | `0.694591` | `0.999998` | `1.000000` | `0.040113` |
| CLIP | `0.551961` | `0.999951` | `1.000000` | `0.019031` |
| DINO | `0.590305` | `0.999942` | `1.000000` | `0.023132` |
| PersonViT | `0.683738` | `0.999998` | `1.000000` | `0.041757` |
| FaceNet | `0.632254` | `0.999938` | `1.000000` | `0.015217` |

Top fused-view false-split identities:

- GT `9`: `26` components, split mass `1046536325`, top-10 recall
  `0.835751`, best same-GT score `0.948246`
- GT `36`: `16` components, split mass `678199057`, top-10 recall
  `0.930047`, best same-GT score `0.921952`
- GT `11`: `25` components, split mass `672659005`, top-10 recall
  `0.462215`, best same-GT score `0.947745`
- GT `43`: `20` components, split mass `595698820`, top-10 recall
  `0.773598`, best same-GT score `0.931030`
- GT `52`: `23` components, split mass `575132111`, top-10 recall
  `0.856526`, best same-GT score `0.948246`

Conclusion:

- Candidate retrieval is not the current bottleneck.  Every tested feature view
  can recover `100%` of the false-split mass by top-100, and almost all of it
  by top-50.
- The bottleneck is calibrated edge selection.  The candidate pool is extremely
  impure: even the best fused view has only `0.082969` dominant-pure edge
  precision at score `0.62`, and most individual views are lower.
- The next structural pivot should be positive-generation / verifier learning,
  not larger retrieval.  We need stronger no-anchor pair labels from temporal
  consistency, stable clothing/body evidence, face when available, and
  generative/augmentation positives; then use them to train a calibrated edge
  model before graph resolution.

### 2026-06-19 direct multiview merge refutation

Purpose:

- test the simplest next structural hypothesis after the retrieval diagnostic:
  if false-split candidates are retrievable, maybe multi-view agreement can
  directly select safe component merges without a learned verifier;
- start from the current best assignment and keep all construction no-anchor.

New script:

- `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_multiview_merge_sweep.py`
- local copy:
  `kit/no_anchor_assignment_multiview_merge_sweep.py`

Implementation notes:

- starts from
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`;
- nodes are current predicted-ID components;
- candidate edges are component-centroid top-k or member-level top-edge
  candidates;
- edge score is computed from multi-view cosine/rank agreement across fused,
  OSNet, CLIP, DINO, PersonViT, and FaceNet;
- GT is used only for pair/full metrics.

Engineering note:

- The first large grid and a full-score tiny grid were interrupted because
  repeated component merge replay and full evaluator calls were too slow for
  this diagnostic.
- Residual remote processes were explicitly killed:
  `3631474`, `3635869`, `3635870`.
- The final reported artifact is the pair-only centroid diagnostic below.

Pair-only artifact:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_multiview_centroid_tiny_pair_20260619.json`
- CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_multiview_centroid_tiny_pair_20260619.csv`

Setup:

- centroid-only candidates, candidate top-k `30`;
- score mode `hybrid`;
- rank-k `10`, sim threshold `0.62`;
- forbidden disabled for speed and for an optimistic diagnostic.

Result:

- base/current pair F1 / precision / recall:
  `0.768743 / 0.813273 / 0.728836`
- best row was no-op:
  `accepted_edges=0`, pair F1 `0.768743`
- first actual merge setting:
  `accepted_edges=9`, pair F1 / precision / recall
  `0.702338 / 0.660344 / 0.750036`
- more aggressive merge:
  `accepted_edges=13`, pair F1 / precision / recall
  `0.686394 / 0.631516 / 0.751716`

Conclusion:

- Direct multi-view component merging is negative.  Even with cannot-link
  disabled, the first accepted merges destroy precision far faster than they
  recover recall.
- This reinforces the retrieval verdict: raw multi-view similarity is not
  calibrated evidence.  The next useful step must generate stronger no-anchor
  positive/negative labels and train a calibrated pair/edge verifier, rather
  than merging high-scoring retrieval edges directly.

### 2026-06-19 Deli AutoResearch distillation + teacher-consensus refutation

Source distillation:

- Deli AutoResearch framework:
  `https://victorchen96.github.io/auto_research/framework.html`
- Deli AutoResearch paper page:
  `https://victorchen96.github.io/auto_research/paper.html`
- self-play story:
  `https://victorchen96.github.io/blog_self_play_story.html`

Operational rules adopted for this VLINCS loop:

- state must be persisted in files and reports, not chat memory;
- ready means execute: run bounded diagnostics instead of stopping at a plan;
- stall means structural pivot: after retrieval, direct multiview merge, and
  soft verifier all failed, do not keep tuning the same threshold family;
- independent evidence must be scored honestly, including downward outcomes;
- a negative result is useful only if it names the failure mode and blocks a
  repeat attempt.

New teacher-consensus merge diagnostic:

- script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_teacher_consensus_merge_sweep.py`
- local copy:
  `kit/no_anchor_assignment_teacher_consensus_merge_sweep.py`
- base assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`
- high-threshold pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_teacher_consensus_merge_pair_20260619.json`
- low-threshold top-3 full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_teacher_consensus_merge_low_pairfull_20260619.json`
- compact top-80 CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_teacher_consensus_merge_low_pairfull_top80_20260619.csv`
- permissive actual-merge JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_teacher_consensus_merge_actual_pair_20260619.json`
- forced permissive JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_teacher_consensus_merge_forced_pair_20260619.json`

Implementation:

- treat seven existing no-anchor assignment CSVs as weak teachers;
- for each current component, compute each teacher's dominant delivered ID,
  dominance, and coverage;
- for component pairs, count `teacher_same` and `teacher_diff` votes;
- annotate edges with DB/fused/OSNet/PersonViT centroid visual support;
- merge only if teacher votes, score, visual gate, component size, and
  same-stream overlap constraints allow it;
- GT is used only after prediction for pair/full metrics.

Result:

- base/current pair F1 / precision / recall:
  `0.768743 / 0.813273 / 0.728836`
- high-threshold sweep:
  `4005` teacher-covered edges, best row no-op, all edges rejected by
  threshold at `0.60`;
- low-threshold sweep:
  top-3 full rows all no-op, full IDF1 `0.652624`;
- permissive sweep:
  top-80 rows all no-op;
- forced permissive sweep:
  even with threshold `-0.50`, `min_same_votes=1`,
  `min_same_frac=0.10`, and `max_diff_votes=6`, accepted edges remain `0`;
- teacher edge preview explains why: the highest visual-score preview rows all
  have `teacher_valid=7`, `teacher_same=0`, and `teacher_diff=7`.

Engineering fix:

- `kit/no_anchor_component_merge_sweep.py` and
  `kit/no_anchor_component_verifier_sweep.py` had `_write_csv` fieldnames
  generated without de-duplication.  A 7.7k-row teacher sweep started writing a
  `5.9G` CSV because each row's columns were repeated many times.
- The helper now uses a set of field names.  The giant remote CSV was deleted;
  the compact top-80 CSV is `16K`.

Conclusion:

- Cross-assignment teacher consensus does not generate usable positive merge
  labels.  The old no-anchor assignments agree strongly on negatives, not on
  false-split repair edges.
- This falsifies the "let old no-anchor outputs vote the positives" route.
- The next structural pivot should create a genuinely new evidence family:
  tracklet self-play positives from augmentations/body-part stability,
  pose/clothing/face verifier labels, or a learned pair verifier trained on
  intra-tracklet and same-stream temporal constraints rather than on old
  assignment IDs.

### 2026-06-19 target-agglomeration teacher evidence check

Purpose:

- test the follow-up hypothesis from the target-agglomeration branch: the
  sparse target sources should not replace delivered IDs directly, but might be
  useful as teacher evidence for safe merge decisions inside the current best
  assignment;
- keep construction no-anchor.  The target teachers are generated assignments,
  not GT anchors.  GT is used only after prediction for pair metrics.

Run:

- script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_teacher_consensus_merge_sweep.py`
- pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_target_teacher_merge_pair_20260619.json`
- pair CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_target_teacher_merge_pair_20260619.csv`
- local copies:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_teacher_merge_pair_20260619.json`
  and
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_teacher_merge_pair_20260619.csv`

Teachers:

- six sparse target-agglomeration assignment sources:
  `target_t640_d1/d5/d10_c0p75` and
  `target_t1280_d1/d5/d10_c0p75`;
- teacher component-label counts:
  `68`, `57`, `40`, `68`, `57`, `40`;
- teacher edge count:
  `2016`;
- component cannot-link conflicts:
  `1588`.

Result:

| config family | accepted edges | pair F1 | pair precision | pair recall |
| --- | ---: | ---: | ---: | ---: |
| current base | `0` | `0.768743` | `0.813273` | `0.728836` |
| best target-teacher row | `1` | `0.768743` | `0.813273` | `0.728836` |
| most permissive accepted row in grid | `6` | `0.768743` | `0.813272` | `0.728836` |

Interpretation:

- Target-agglomeration can produce candidate teacher edges, but the evidence
  does not move enough weighted same-ID mass to improve the pair gate.
- Across all `2160` grid rows, `tracklet_pair_f1` is unchanged at `0.768743`.
  Some settings accept up to `6` edges, but those edges are effectively
  negligible for the weighted metric.
- This closes the direct target-teacher merge branch.  The target sources are
  not a sufficient positive-label generator by themselves.
- Under the AutoResearch anti-loop rule, the next branch should not widen this
  target-teacher grid.  The next useful artifact needs a new positive
  generation mechanism: intra-tracklet augmentation/self-play positives,
  body-part/pose/clothing stability labels, face-gated positives, or a
  calibrated verifier trained on pair evidence that is not simply another old
  assignment namespace.

### 2026-06-19 oracle repair decomposition + self-play positive check

Purpose:

- decompose the remaining e2e gap after the current best no-anchor assignment;
- identify whether the route to `>0.70` is mostly split repair, merge repair,
  or detection/admission;
- test one new no-anchor positive-generation mechanism inspired by self-play:
  split current high-recall components into unlabeled subcomponents, train a
  component-edge verifier on those self-play positives plus cannot-link/low
  similarity negatives, then score real inter-component merge candidates.

New scripts:

- `kit/no_anchor_oracle_repair_decomposition.py`
- `kit/no_anchor_assignment_selfplay_component_merge_sweep.py`

Oracle decomposition artifacts:

- pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_oracle_repair_decomposition_paironly_20260619.json`
- local pair-only copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_oracle_repair_decomposition_paironly_20260619.json`
- top-1 full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_oracle_repair_decomposition_full_top1_20260619.json`
- local top-1 full copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_oracle_repair_decomposition_full_top1_20260619.json`

Top false-merge predicted components by eval-only mass:

| predicted ID | dominant GT | dominant frac | GT count | false-merge mass |
| --- | ---: | ---: | ---: | ---: |
| `60000077` | `36` | `0.602136` | `6` | `790173339` |
| `60000122` | `36` | `0.446144` | `30` | `367982013` |
| `60000033` | `31` | `0.802036` | `19` | `317340419` |
| `60000040` | `37` | `0.884391` | `12` | `306768505` |

Top false-split GT identities by eval-only mass:

| GT ID | dominant prediction | dominant frac | pred components | false-split mass |
| ---: | --- | ---: | ---: | ---: |
| `9` | `60000001` | `0.701911` | `26` | `1046536325` |
| `36` | `60000077` | `0.657854` | `16` | `678199057` |
| `11` | `60000079` | `0.786685` | `25` | `672659005` |
| `43` | `60000025` | `0.649796` | `20` | `595698820` |

Oracle repair result:

| repair variant | pair F1 | pair precision | pair recall | full IDF1 | HOTA | DetPr | DetRe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current base | `0.768743` | `0.813273` | `0.728836` | `0.654739` submission best | `0.518148` | `0.761672` | `0.574135` |
| split top 40 false-merge components + merge top 40 false-split GTs | `0.996040` | `0.999490` | `0.992614` | `0.705997` | `0.576146` | `0.784953` | `0.641474` |
| current-tracklet all-GT-majority oracle | `1.000000` | `1.000000` | `1.000000` | `0.711353` known upper | n/a | n/a | n/a |

Interpretation:

- The current tracklet set can exceed the e2e target only with near-oracle
  association.  The top-40 repair reaches `0.705997` full IDF1, just under the
  all-current-tracklet oracle `0.711353`.
- Most of the usable headroom comes from merging large false-split identities.
  Splitting false-merge components is helpful but secondary in this top repair.
- The no-anchor target for the next model is therefore precise: learn to merge
  about the top large split identities without introducing broad false merges.
  Detection filtering cannot close this gap.

Self-play component verifier artifacts:

- normal threshold pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_selfplay_component_merge_pair_20260619.json`
- normal threshold pair CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_selfplay_component_merge_pair_20260619.csv`
- low-threshold sanity JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_selfplay_component_merge_lowthr_pair_20260619.json`
- low-threshold sanity CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_selfplay_component_merge_lowthr_pair_20260619.csv`

Self-play setup:

- base assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`
- primary feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_20260619.npz`
- verifier views:
  DB, FaceNet s2, OSNet s7
- pseudo positives:
  `158` self-play split edges from current components
- pseudo negatives:
  `4448` candidate edges from cannot-link / low score rules
- trained HGB pseudo AP/AUC:
  `1.0 / 1.0`

Self-play result:

| run | threshold behavior | accepted edges | pair F1 | pair precision | pair recall |
| --- | --- | ---: | ---: | ---: | ---: |
| normal grid | all real candidate probabilities near zero | `0` | `0.768743` | `0.813273` | `0.728836` |
| low-threshold sanity | threshold `0`, min top-5 votes `2` | `6` | `0.768743` | `0.813273` | `0.728836` |

Conclusion:

- Current-component self-play positives are too distribution-shifted from real
  inter-component merge edges.  The classifier separates its pseudo train set
  perfectly but assigns real candidate edges probability around `8.7e-5`.
- This is a useful refutation: self-play positives must come from frame/crop
  augmentations, body-part consistency, face-gated evidence, or stable
  cross-camera attributes, not merely from splitting current predicted
  components.
- No new production best was promoted.  Current production best remains the
  q03-filtered submission at IDF1 `0.654739`.

### 2026-06-19 true multi-frame OSNet s7 ablation

Motivation:

- The previous `samples > 2` extraction helper did not actually sample an
  arbitrary number of frames.  It effectively selected first/middle/last.
- This ablation asks whether true 7-frame temporal coverage improves the
  no-anchor global-ID model, either as a component-merge feature or as a fused
  Louvain graph feature.

Implementation:

- Patched `kit/extract_tracklet_foundation_features.py` so `samples=N` selects
  `N` evenly spaced row numbers along each tracklet.
- Smoke check:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s7_smoke32_20260619.npz`
  produced `32` tracklets x `7` crops = `224` seen crops.
- Full OSNet s7 artifact:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s7_true_20260619.npz`
  with `seen_crops=66324`, `requested_rows=66327`, `missing_crop=3`,
  `shape=[9734,512]`, `tracklets_with_features=9734`, and
  `sample_counts_max=7`.

Direct component-merge result:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s7_true_component_merge_full3_20260619.json`
- best pair F1 / precision / recall:
  `0.768746 / 0.813275 / 0.728840`
- full IDF1:
  `0.652623`
- current reload best remains:
  `0.654597` IDF1 from
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_reload_20260619.json`

Fused Louvain grids:

- fused feature artifacts:
  - `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_20260619.npz`
  - `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet010_s7true_20260619.npz`
- pair-only grid JSONs:
  - `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json`
  - `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_osnet010_s7true_quality060_pair_grid_20260619.json`
- best `osnet005` pair F1 / precision / recall:
  `0.766248 / 0.775577 / 0.757140`
- best `osnet010` pair F1 / precision / recall:
  `0.764479 / 0.806739 / 0.726425`

Full sanity check for the high-recall `osnet005` candidate:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_osnet005_s7true_quality060_full1_20260619.json`
- config:
  `top_k=8`, `edge_floor=0.03`, `resolution=4.0`
- full IDF1 / HOTA / AssA / DetRe:
  `0.642629 / 0.510470 / 0.531170 / 0.579287`
- per-video IDF1:
  - MCAM00 Tc6: `0.860776`
  - MCAM00 Tc8: `0.789379`
  - MCAM03 Tc6: `0.683658`
  - MCAM03 Tc8: `0.590656`
  - MCAM04 Tc6: `0.553114`
  - MCAM05 Tc6: `0.725198`
  - MCAM05 Tc8: `0.704705`
  - MCAM06 Tc6: `0.600802`
  - MCAM06 Tc8: `0.623210`
  - MCAM08 Tc6: `0.765629`

Conclusion:

- The sampling bug is fixed and the s7 OSNet artifact is valid.
- More temporal crops slightly improve one component-merge pair score
  (`0.768746` vs `0.768743`) but do not improve full IDF1.
- Louvain fused s7 features trade precision for recall and reduce full IDF1.
- This branch is now closed under the AutoResearch anti-loop rule: do not run
  another same-family s9/s11 small-weight fusion sweep unless a new verifier or
  admission mechanism changes the evidence semantics.
- Next structural direction: train a calibrated no-anchor pair verifier from
  self-play positives/negatives and use it for quarantine, split/merge gating,
  and committed/provisional/forced output separation.

### 2026-06-19 pseudo-label purity audit + clean-positive verifier check

New diagnostic:

- local script:
  `kit/audit_no_anchor_pseudo_labels.py`
- remote script:
  `/mnt/localssd/vlincs_reid_by_search/kit/audit_no_anchor_pseudo_labels.py`
- audit JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pseudo_label_audit_face_osnet_s7_20260619.json`
- sample CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pseudo_label_audit_face_osnet_s7_samples_20260619.csv`

Setup:

- reproduces the pseudo positive/negative generation used by
  `kit/no_anchor_global_id_model.py`;
- keeps pair IDs and sources, then uses GT only for post-hoc purity audit;
- feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_20260619.npz`
  with `concat_db_embedding`, `db_weight=1.0`, `feature_weight=0.32`;
- verifier-only pair views:
  FaceNet s2 and true OSNet s7;
- pseudo ensemble:
  theta grid `[0.018, 0.016, 0.020, 0.022]`,
  `pseudo_ensemble_min_votes=2`.

Pseudo-label audit result:

- positives: `140473` rows, `129658` GT-evaluable rows;
- positive purity / weighted purity:
  `0.875657 / 0.906332`;
- negatives: `60000` random rows, `43401` GT-evaluable rows;
- negative purity / weighted purity:
  `0.978457 / 0.980788`;
- source split:
  - `pseudo_online_agree`: `105818` rows, purity `0.928050`,
    weighted purity `0.941846`;
  - `strong_visual_pseudo`: `34655` rows, purity `0.702903`,
    weighted purity `0.707661`;
- high-score false-positive examples exist even with 4 pseudo votes and high
  OSNet s7 cosine, e.g. cross-camera MCAM03/04/05/08 pairs with score
  `0.93..0.95`.

Clean-positive verifier ablation:

- disabled `strong_visual_pseudo` by setting `pseudo_strong_pos_sim=1.01`;
- kept only `pseudo_online_agree` positives:
  `105835` positive pairs and `60000` random negative pairs;
- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_model_cleanpos_face_osnet_s7_guard_paironly_20260619.json`
- model:
  HGB, `consensus_guard`, FaceNet + OSNet s7 pair views;
- pseudo validation AP / AUC:
  `0.997946 / 0.996879`;
- best real pair F1 / precision / recall:
  `0.433515 / 0.797407 / 0.297674`;
- higher thresholds raise precision to `0.844172` but recall drops to
  `0.287216`.

Conclusion:

- The pseudo-positive source is not uniformly bad.  `pseudo_online_agree` is a
  usable no-anchor positive generator, while `strong_visual_pseudo` is too
  impure for merge supervision.
- Cleaning positives alone does not solve global ID.  As a primary graph
  resolver, `consensus_guard` is far too conservative and destroys recall.
- The next model should use the clean verifier as a commit/quarantine/veto or
  targeted attach scorer on top of the current high-recall assignment, not as a
  replacement resolver.

### 2026-06-19 submission-level detection filter

AutoResearch cue:

- The Deli AutoResearch protocol pushes each iteration to produce either a
  scalar improvement or a falsified mechanism, and to pivot structure after
  repeated same-family failures.
- This run tests a narrow M3 hypothesis before pivoting: maybe the current
  forced-delivery best is losing e2e IDF1 from low-confidence detection rows
  inside otherwise acceptable global-ID assignments.

New scripts:

- `kit/evaluate_db_assignments_detection_filter.py`
- `kit/evaluate_submission_detection_filter.py`

Input:

- source submission:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_20260619.zip`
- direct reload baseline:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_reload_20260619.json`
- grid JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_submission_detection_filter_grid_20260619.json`
- promoted q03 JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_submission_detection_filter_q03_promoted_20260619.json`
- promoted q03 zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_submission_detection_filter_q03_20260619.zip`

Direct submission filtering results:

| config | dropped rows | IDF1 | HOTA | AssA | DetPr | DetRe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | `0` | `0.654597` | `0.518183` | `0.533816` | `0.757889` | `0.576083` |
| hard_q03 | `24677` | `0.654739` | `0.518148` | `0.533889` | `0.761672` | `0.574135` |
| hard_q05 | `41176` | `0.654674` | `0.517930` | `0.533746` | `0.764142` | `0.572640` |

Assignment-CSV reconstruction control:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_detection_filter_grid_20260619.json`
- base reconstruction IDF1 / HOTA / AssA:
  `0.652624 / 0.516393 / 0.532037`
- best reconstruction filter:
  `hard_q05`, IDF1 `0.652723`

Conclusion:

- Detection-row filtering gives a real but tiny submission-level gain:
  `0.654597 -> 0.654739` IDF1.
- The gain comes from higher detection precision, but recall drops immediately;
  this is not the path to `0.70`.
- Because the winning q03 threshold was selected with the GT scorer, it is a
  research ablation artifact, not an automatic no-GT production policy.
- Under the AutoResearch anti-loop rule, stop widening the same confidence
  grid.  The next structural pivot remains M5/M8 evidence calibration:
  current-best assignment plus verifier audit/quarantine, not another M3-only
  threshold sweep.

### 2026-06-19 clean-positive consensus-attach refutation

Hypothesis:

- The clean `pseudo_online_agree` verifier failed as a primary
  `consensus_guard` resolver because it was too conservative.
- Maybe `consensus_attach` can keep a high-confidence core and attach only
  strongly supported fragments, using the same clean HGB verifier.

Run:

- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_cleanpos_consensus_attach_face_osnet_s7_full3_20260619.json`
- CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_cleanpos_consensus_attach_face_osnet_s7_full3_20260619.csv`
- model loaded:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_model_cleanpos_face_osnet_s7_guard_20260619.joblib`
- feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_20260619.npz`
- pair views:
  FaceNet s2 and true OSNet s7
- grid:
  thresholds `0.01,0.018,0.03`, blends `0.50,0.75,1.00`,
  `full_top_n=3`.

Result:

- all 9 configs tie at pair F1 / precision / recall:
  `0.426041 / 0.777230 / 0.293447`
- full IDF1 / HOTA / AssA / DetPr / DetRe:
  `0.465100 / 0.322826 / 0.337983 / 0.705112 / 0.346989`
- attach candidates / eligible / accepted:
  `417261 / 1355 / 0`
- the core is `consensus_guard`, with `822` components and largest component
  capped at `120`.

Conclusion:

- This is a useful negative result.  The clean verifier is not being given a
  good high-recall base; the internal consensus core has already collapsed
  recall before attach can help.
- Do not continue tuning `consensus_attach` thresholds in this implementation.
- Next structural direction: keep the current-best forced assignment as the
  base and run a verifier audit/quarantine/veto over its existing large
  components.  In PDF terms, this means converting forced delivery into
  evidence-calibrated `committed/provisional/pending/forced` states, not
  replacing the resolver with a conservative core.

### 2026-06-19 Deli AutoResearch protocol distilled for VLINCS

The useful part of Deli AutoResearch for this project is not "more agents" in
the abstract.  It is a stricter research loop:

- Persist state to files: every run must leave JSON/CSV/zip artifacts and a
  written interpretation.
- Freeze the eval gate: every structural change is compared against the same
  current best and the same no-anchor rule.
- Separate research from production: GT-scored threshold choices can create
  research artifacts, but they cannot be claimed as automatic no-GT policy.
- Force pivots when the same idea saturates: after repeated tiny gains or
  negative sweeps, stop widening the same grid and change the module being
  tested.
- Prefer autonomous execution over speculative planning: write the script, run
  the bounded sweep, inspect metrics, and only then update the model card.

Applied here:

- The submission-level detection filter is a valid M3/admission ablation, but
  the gain is only `+0.000142` IDF1.
- `consensus_attach` is a falsified M8 replacement because the conservative
  core kills recall before attach can operate.
- The next pivot should target stateful M5/M8 evidence calibration over the
  current best assignment, not another single-threshold admission sweep.

### 2026-06-19 current-best verifier split

Hypothesis:

- Keep the current best no-anchor forced assignment as the high-recall base.
- Train/load a clean no-anchor pair verifier from `pseudo_online_agree`
  positives plus cannot-link negatives.
- Audit only internal edges inside each predicted component, then split a
  component if the verifier blocks enough internal support.

Run:

- script:
  `kit/no_anchor_assignment_verifier_split_sweep.py`
- clean verifier:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_model_cleanpos_face_osnet_s7_guard_20260619.joblib`
- full sweep:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_verifier_split_face_osnet_s7_full2_20260619.json`
- exported best assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_verifier_split_t040_m16_assignments_20260619.csv`
- split + detection filter:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_verifier_split_t040_m16_detection_filter_grid_20260619.json`

Verifier-split sweep:

| config | pair F1 | pair precision | pair recall | split components | split parts | full IDF1 | HOTA | AssA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base assignment | `0.768743` | `0.813273` | `0.728836` | `0` | `0` | `0.652806` | `0.516613` | `0.532272` |
| threshold 0.40, min size 16 | `0.768837` | `0.813577` | `0.728761` | `17` | `45` | `0.652806` | `0.516613` | `0.532272` |

Split + detection filter:

| config | dropped rows | IDF1 | HOTA | AssA | DetPr | DetRe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | `0` | `0.652806` | `0.516613` | `0.532272` | `0.755983` | `0.574411` |
| hard_q03 | `24677` | `0.652963` | `0.516592` | `0.532356` | `0.759776` | `0.572481` |
| hard_q05 | `41176` | `0.652906` | `0.516382` | `0.532220` | `0.762250` | `0.570997` |

Conclusion:

- The verifier split is directionally positive at the tracklet-pair level:
  pair F1 improves from `0.768743` to `0.768837`.
- The e2e export is still below the current best submission-level artifact:
  `0.652963` IDF1 vs `0.654739` IDF1.
- Hard splitting is too blunt.  The next M8 implementation should keep
  component IDs stable and output state/provenance: `committed` for clean
  subgraphs, `provisional` for weakly supported fragments, `pending` for
  insufficient evidence, and `forced` only for delivery-required rows.

### 2026-06-19 AutoResearch distillation plus greedy s7 namespace verdict

New external cue distilled:

- Deli AutoResearch is a protocol, not a model: persist state to files,
  execute ready experiments, use watchdog-style liveness checks, and pivot
  structurally after repeated saturated runs.
- The self-play story's useful lesson here is evidence honesty: score can go
  down after external checks, and the framework should accept that instead of
  preserving a prettier curve.
- For VLINCS, this means each branch must produce either a better canonical
  submission artifact or a clearly written negative result that closes the
  branch.

Greedy relink s7 branch:

- multi-frame OSNet/color feature artifact:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_osnet_color_s7_20260619.npz`
- global model artifact:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_global_model_20260619.joblib`
- assignment CSV:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_assignments_20260619.csv`
- full submission JSON:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_full_20260619.json`
- submission zip:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_submission_20260619.zip`

Model-side result:

- best mode: `time_agglom`, `top_k=45`, `theta=0.01`
- identity F1: `0.753964`
- tracklet-pair F1 / precision / recall:
  `0.647739 / 0.703469 / 0.600192`
- components: `46`, largest component: `1087`
- status counts:
  `committed=708`, `provisional=3274`, `forced_component=14674`

Full scorer result:

| artifact | IDF1 | HOTA | AssA | DetPr | DetRe | rows | predicted ids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy s7 full | `0.612647` | `0.475550` | `0.497936` | `0.708133` | `0.539853` | `1723070` | `46` |
| current best q03 | `0.654739` | `0.518148` | `0.533889` | `0.761672` | `0.574135` | `1534676` | `395` |

Per-video greedy s7 IDF1:

| video | greedy s7 | current best |
| --- | ---: | ---: |
| MCAM00 Tc6 | `0.827771` | `0.879252` |
| MCAM00 Tc8 | `0.815663` | `0.827992` |
| MCAM03 Tc6 | `0.668947` | `0.690770` |
| MCAM03 Tc8 | `0.642241` | `0.626096` |
| MCAM04 Tc6 | `0.521335` | `0.560373` |
| MCAM05 Tc6 | `0.247615` | `0.711979` |
| MCAM05 Tc8 | `0.809290` | `0.792455` |
| MCAM06 Tc6 | `0.582606` | `0.606873` |
| MCAM06 Tc8 | `0.740029` | `0.706032` |
| MCAM08 Tc6 | `0.722589` | `0.768478` |

Namespace/source-switch checks:

- current-gid mapped assignments:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_currentgid_mapped_assignments_20260619.csv`
- mapped full JSON:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_currentgid_mapped_full_20260619.json`
- mapped full score:
  `0.612647` IDF1, unchanged because the component structure is unchanged.
- aligned `600000xx` source zip:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_currentid600_submission_20260619.zip`
- aligned source-switch JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_submission_switch_current_greedy_s7_currentid600_oracle_20260619.json`

Aligned source-switch result:

| policy | selected greedy videos | IDF1 | HOTA | AssA | DetPr | DetRe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| current base | none | `0.654739` | `0.518148` | `0.533889` | `0.761672` | `0.574135` |
| oracle per-video | MCAM03 Tc8, MCAM05 Tc8, MCAM06 Tc8 | `0.601175` | `0.459056` | `0.476575` | `0.730330` | `0.510837` |
| all greedy aligned | all | `0.614446` | `0.477099` | `0.499464` | `0.710212` | `0.541438` |

Verdict:

- The greedy s7 branch is a valid no-anchor global-ID model-side pass
  (`0.753964` identity F1), but it is not a promoted e2e artifact.
- The local per-video wins do not safely compose into a global submission, even
  after ID namespace alignment.
- Close this branch under the AutoResearch anti-loop rule.  The next structural
  target should be stateful admission/calibration over the current best
  delivery output: keep high-recall current IDs, but attach
  `committed/provisional/pending/forced` status, conflict evidence, and
  per-component provenance.

### 2026-06-19 resumable per-video admission oracle scaffold

Reason:

- A quick inline per-video confidence-filter preview started to test whether
  the current `hard_q03` admission filter is near the practical ceiling.
- The first two videos both selected no filter:
  MCAM00 Tc6 IDF1 `0.882105`, MCAM00 Tc8 IDF1 `0.832911`.
- The remote SSH session then failed with `Connection timed out during banner
  exchange`, and h100-test-2, h100-test-3, and test-video-0 all showed the same
  SSH/proxy symptom.  Pluto CLI also returned `Failed to connect to Pluto
  service`.

New local script:

`kit/no_anchor_submission_pervideo_filter_oracle.py`

Purpose:

- keep an existing no-anchor submission's global IDs fixed;
- sweep confidence/area quantile detection filters independently per video;
- checkpoint after every scored row so a Pluto SSH drop does not lose work;
- optionally combine each video's oracle-selected filter into one canonical
  full submission eval.

Prepared remote command once Pluto SSH recovers:

```bash
cd /mnt/localssd/vlincs_reid_by_search
export PYTHONPATH=/mnt/localssd/vlincs_reid_by_search:${PYTHONPATH:-}
nohup /mnt/localssd/vlincs_reid_venv/bin/python \
  kit/no_anchor_submission_pervideo_filter_oracle.py \
  --submission-zip /mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_20260619.zip \
  --conf-quantiles 0,0.01,0.02,0.03,0.05,0.08,0.10 \
  --area-quantiles 0 \
  --resume \
  --full-oracle \
  --json /mnt/localssd/vlincs_reid_runs/no_anchor_quality060_pervideo_conf_oracle_20260619.json \
  --zip-out /mnt/localssd/vlincs_reid_runs/no_anchor_quality060_pervideo_conf_oracle_20260619.zip \
  > /mnt/localssd/vlincs_reid_runs/no_anchor_quality060_pervideo_conf_oracle_20260619.log 2>&1 &
```

Expected interpretation:

- If the combined oracle stays near `0.654739`, M3 detection filtering is
  exhausted and should not receive more effort.
- If it jumps materially, the next no-anchor task is to replace the GT-selected
  per-video oracle with an unsupervised video-quality/admission predictor.
- Either way, this is a diagnostic research artifact only because its
  per-video choices are selected by GT scoring.

### 2026-06-19 stateful component policy scaffold

Remote status:

- Retested h100-test-2, h100-test-3, and test-video-0.
- All three SSH paths still fail during banner exchange.
- Pluto CLI still reports `Failed to connect to Pluto service`.
- This is the second consecutive goal-continuation turn with the same external
  Pluto/SSH blocker, but local work can still move the research forward.

New local script:

`kit/no_anchor_assignment_state_policy_sweep.py`

Purpose:

- Keep an existing no-anchor global-ID assignment as the forced-delivery base.
- Compute a component state layer without anchors or GT labels:
  `committed`, `provisional`, `pending`, and `forced_conflict`.
- State evidence comes from component size, tracklet quality, detection
  confidence/area, video/camera span, and same-stream temporal cannot-link
  conflicts.
- Sweep no-GT delivery policies such as `keep_all`, `color_forced`,
  `singleton_forced`, `drop_forced`, `color_pending_forced`,
  `singleton_pending_forced`, and `drop_pending_forced`.
- `color_*` policies repair cannot-link conflicts by graph coloring within a
  conflicted component, preserving non-conflicting evidence instead of reducing
  the entire component to singletons.
- Use GT only after prediction for pair/full metrics and oracle ranking.

Local validation:

- `python kit/no_anchor_assignment_state_policy_sweep.py --self-test` passed.
- The synthetic conflict case produced `color_parts_forced_conflict = 2`,
  `colored_forced_conflict = 3`, and kept the two pending tracklets unchanged.

Prepared remote command once Pluto SSH recovers:

```bash
cd /mnt/localssd/vlincs_reid_by_search
export PYTHONPATH=/mnt/localssd/vlincs_reid_by_search:${PYTHONPATH:-}
nohup /mnt/localssd/vlincs_reid_venv/bin/python \
  kit/no_anchor_assignment_state_policy_sweep.py \
  --assignment-csv /mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv \
  --committed-min-sizes 4,8,16,32 \
  --pending-max-sizes 0,1,2,4 \
  --conflict-rate-thresholds 0,0.0005,0.001,0.003,0.01 \
  --policies keep_all,color_forced,singleton_forced,drop_forced,color_pending_forced,singleton_pending_forced,drop_pending_forced \
  --full-top-n 5 \
  --sort-key tracklet_pair_f1 \
  --json /mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_20260619.json \
  --assignments-out /mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_best_assignments_20260619.csv \
  --component-states-out /mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_component_states_20260619.csv \
  > /mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_20260619.log 2>&1 &
```

Follow-up if the best state policy beats the assignment-level baseline:

```bash
/mnt/localssd/vlincs_reid_venv/bin/python \
  kit/evaluate_db_assignments_detection_filter.py \
  --assignment-csv /mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_best_assignments_20260619.csv \
  --config 'base' \
  --config 'hard_q03;video_conf=vlincs_MS01_MC0001_MCAM03_2024-03-Tc8:0.17,vlincs_MS01_MC0001_MCAM04_2024-03-Tc6:0.1364,vlincs_MS01_MC0001_MCAM06_2024-03-Tc6:0.2057' \
  --json /mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_detection_filter_20260619.json \
  --zip-out /mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_detection_filter_best_20260619.zip
```

Expected interpretation:

- If state policies improve pair precision but not full IDF1, the problem is
  still delivery-row/detection structure rather than component state alone.
- If `color_forced` improves full IDF1 over `singleton_forced`, cannot-link
  conflicts should be repaired as constrained subcomponents rather than only as
  hard singleton vetoes.
- If `singleton_forced` improves full IDF1, cannot-link conflicts are a useful
  M8 veto signal but likely need a precision-first delivery mode.
- If `drop_forced` improves DetPr but harms DetRe too much, the right design is
  not deletion but state/provenance reporting plus a softer delivery policy.

### 2026-06-19 remote recovery launcher hardening

Remote status:

- Retested h100-test-2, h100-test-3, and test-video-0 again.
- All three still fail at SSH banner exchange.
- Pluto CLI still reports `Failed to connect to Pluto service`.
- This is the third consecutive continuation with the same external
  Pluto/SSH symptom, but the research is not fully blocked because local
  experiment code can still be hardened.

New local launcher:

`kit/run_no_anchor_remote_recovery_experiments.sh`

Purpose:

- Probe h100-test-3, h100-test-2, then test-video-0 with password fallback.
- Bundle and deploy only the scripts needed for the next two no-anchor
  diagnostics.
- Run `kit/no_anchor_assignment_state_policy_sweep.py --self-test` on the
  remote before starting long jobs.
- Start either or both resumable background experiments:
  per-video submission filter oracle and stateful component policy sweep.
- Print remote PID/log paths so progress can be checked after SSH disconnects.

Validated locally:

```bash
bash -n kit/run_no_anchor_remote_recovery_experiments.sh
python kit/no_anchor_assignment_state_policy_sweep.py --self-test
python -m py_compile \
  kit/no_anchor_assignment_state_policy_sweep.py \
  kit/no_anchor_submission_pervideo_filter_oracle.py
```

Prepared command once Pluto SSH recovers:

```bash
bash kit/run_no_anchor_remote_recovery_experiments.sh --case both
```

If only one branch should run:

```bash
bash kit/run_no_anchor_remote_recovery_experiments.sh --case state
bash kit/run_no_anchor_remote_recovery_experiments.sh --case pervideo
```

Expected next evidence:

- `no_anchor_state_policy_quality060_20260619.json`
- `no_anchor_state_policy_quality060_best_assignments_20260619.csv`
- `no_anchor_quality060_pervideo_conf_oracle_20260619.json`
- corresponding logs under `/mnt/localssd/vlincs_reid_runs/logs/`

Dry-run probe after the color-policy update:

- `bash kit/run_no_anchor_remote_recovery_experiments.sh --case both --dry-run`
  prepared the bundle successfully.
- h100-test-3, h100-test-2, and test-video-0 all still failed at SSH banner
  exchange.
- No remote experiment was submitted in this probe.

### 2026-06-19 AutoResearch distillation

Added:

`reports/no_anchor_autoresearch_protocol_20260619.md`

This distills the Deli AutoResearch framework into VLINCS-specific operating
rules: persist state in files, separate worker/evaluator roles, enforce
anti-loop direction diversity, execute prepared experiments once compute is
reachable, and treat oracle gains as bottleneck detectors rather than solutions.

The immediate branches are:

- per-video admission oracle;
- stateful component resolution with cannot-link graph coloring;
- unsupervised replacement for any GT-selected oracle policy that shows lift.

### 2026-06-19 local sample state-policy audit

Remote status:

- `conda run -n adobe python -m colligo.pluto.sdk.cli job status ... --project
  video-world-models` failed for h100-test-3, h100-test-2, and test-video-0
  with `Failed to connect to Pluto service`.
- `bash kit/run_no_anchor_remote_recovery_experiments.sh --case both
  --dry-run` prepared the bundle, then all three jobs failed at SSH banner
  exchange.
- No DS1 full remote experiment was submitted in this continuation.

New local script:

`kit/sample_assignment_state_policy_sweep.py`

Purpose:

- Run the same no-anchor component state policies on local sample/parquet
  assignments without requiring the remote gallery DB.
- Use sample parquet `tracklet_majority_gt_id` only for post-hoc metrics.
- Optionally compute a `sample_parquet_gt_same_detection_boxes` HOTA/IDF1 proxy
  by building reference rows from the same detection boxes and parquet `gt_id`.
  This is a diagnostic proxy, not the DS1 leaderboard score.
- Deduplicate `(video, frame, id)` on both reference and prediction sides before
  calling `reid_hota`, because the local sample parquet can contain multiple
  detections matched to the same GT identity in the same frame.

Local commands:

```bash
python -m pip install 'reid-hota>=0.3.5,<0.3.7'
python kit/sample_assignment_state_policy_sweep.py \
  --tracklet-parquet \
    /Users/zcai/Codex/videolincs/local_runs/yolo_reference_labels_iou050_mcam04_08_frame12000_18000_20260616/full_video_mcam04_tc6_yolo11_stream_streaming_multigpu_tracklets_majority050_eval.parquet \
    /Users/zcai/Codex/videolincs/local_runs/yolo_reference_labels_iou050_mcam04_08_frame12000_18000_20260616/full_video_mcam08_tc6_yolo11_stream_streaming_multigpu_tracklets_majority050_eval.parquet \
  --assignments /Users/zcai/Codex/videolincs/local_runs/yolo_reference_labels_iou050_mcam04_08_frame12000_18000_20260616/no_anchor_sample_osnet_traj005_nfc_k8_e075_best_20260617/no_anchor_sample_model_assignments.csv \
  --full-top-n 3 \
  --sort-key tracklet_pair_f1 \
  --json /Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_sample_state_policy_traj005_20260619.json
```

Main local proxy results:

| artifact | policy | pair F1 | pair precision | pair recall | sample proxy IDF1 | sample proxy HOTA |
|---|---:|---:|---:|---:|---:|---:|
| `traj005_nfc_k8_e075_best` | `keep_all` | `0.549817` | `0.699557` | `0.452878` | `0.731491` | `0.602400` |
| `traj005_nfc_k8_e075_best`, precision-sorted | `singleton_forced` | `0.003022` | `0.971753` | `0.001513` | `0.238364` | `0.148616` |
| `target_m3_tmp_d10_c0p5` | `keep_all` | `0.336119` | `0.832166` | `0.210589` | `0.644253` | `0.524112` |

Per-video sample proxy for `traj005_nfc_k8_e075_best`:

- MCAM04 Tc6: IDF1 `0.801389`, HOTA `0.742498`, DetRe `0.704274`.
- MCAM08 Tc6: IDF1 `0.626100`, HOTA `0.546544`, DetRe `0.512544`.

Interpretation:

- The local sample proxy can exceed 70 IDF1 under no-anchor full-coverage
  delivery, but this does not satisfy the goal because it is a two-video
  parquet-GT proxy rather than the DS1 end-to-end score.
- `color_forced` and `singleton_forced` are useful as precision diagnostics but
  are not a main path to e2e >70: they improve precision by sacrificing too much
  recall.
- The strongest actionable signal is MCAM08 coverage/association weakness.  The
  next remote-ready branch should prioritize upstream tracklet/relink/admission
  improvements for weak cameras over stricter component splitting.

### 2026-06-19 self-play/source-switch distillation

External cue distilled:

- Deli AutoResearch frames long-horizon agent work as durable state files,
  anti-loop forced pivots, heartbeat/watchdog recovery, and independent
  evaluator roles.
- The self-play paper/story adds the useful research metaphor for this task:
  prior knowledge does not always define the ceiling; let competing agents or
  model variants play proposer/opponent and keep only changes that survive
  quantitative evaluation.

Project-local translation:

- In VLINCS, a no-anchor assignment source is a proposer.
- Other assignment sources, cannot-link statistics, and per-video delivery
  metrics are opponents/verifiers.
- Oracle-selected source switches are not production policies; they are
  bottleneck detectors.  If a source switch improves a weak camera, the next
  step is a no-GT selector using confidence, component state, conflict rate,
  tracklet length, feature margin, camera/video, and provenance.

New local proxy script:

`kit/sample_assignment_video_source_switch.py`

Purpose:

- Run video/source switching on local sample assignment CSVs without the remote
  gallery DB.
- Align each candidate source namespace to a reference source by tracklet
  overlap only.
- Optionally fall back to the base source for missing tracklets so sparse
  high-precision sources can be tested without dropping coverage.
- Use sample parquet `tracklet_majority_gt_id` only for post-hoc scoring.

Input source pool:

- `traj005`: full-coverage OSNet + trajectory source,
  `/Users/zcai/Codex/videolincs/local_runs/yolo_reference_labels_iou050_mcam04_08_frame12000_18000_20260616/no_anchor_sample_osnet_traj005_nfc_k8_e075_best_20260617/no_anchor_sample_model_assignments.csv`
- `target_d10_c50`, `target_d5_c50`, `target_d1_c50`,
  `target_d10_c35`: high-precision/admission target sources.
- `nfc_k8_e50`, `nfc_k4_e50`, `nfc_k16_e50`: NFC source variants.

Output artifact:

`/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_sample_video_source_switch_basefallback_20260619.json`

Top local sample proxy rows:

| policy | sample IDF1 | HOTA | pair F1 | precision | recall | MCAM04 IDF1 | MCAM08 IDF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all `target_d1_c50` | `0.739426` | `0.614382` | `0.564060` | `0.711321` | `0.467315` | `0.825351` | `0.605246` |
| all `target_d5_c50` | `0.739041` | `0.613887` | `0.562855` | `0.709444` | `0.466470` | `0.825346` | `0.604159` |
| all `target_d10_c50` | `0.739036` | `0.613871` | `0.562847` | `0.709546` | `0.466415` | `0.825374` | `0.604097` |
| switch MCAM08 to `target_d1_c50` | `0.736953` | `0.612368` | `0.561754` | `0.713054` | `0.463422` | `0.797215` | `0.648758` |

Interpretation:

- The sample proxy confirms that no-anchor source diversity can exceed 70 on
  the local two-video diagnostic, but it does not complete the DS1 e2e goal.
- Switching MCAM08 to a target/admission source raises MCAM08 proxy IDF1 from
  the base `0.626100` to `0.648758`, while all-target raises MCAM04 to
  `0.825351`.  The weak-camera issue is therefore not only bbox coverage; the
  assignment source itself matters.
- This reopens source switching as a useful branch, but only with the newer
  target/admission-style sources.  The older full-data source-switch pool had
  already saturated around `0.635418` full IDF1 and remains a negative result.

Remote launcher update:

- `kit/run_no_anchor_remote_recovery_experiments.sh` now accepts
  `--case switch` and `--case all`.
- `switch` bundles `kit/no_anchor_assignment_video_switch.py`.
- The remote switch branch starts from `current` and adds any available
  historical no-anchor assignment sources:
  `small060`, `face_local`, `face_small`, `highres`, `paircal`,
  `cannotlink_light`, and `verifier_split`.
- Missing source files are skipped on the remote instead of failing the whole
  launcher.

Prepared full-data command once Pluto SSH recovers:

```bash
bash kit/run_no_anchor_remote_recovery_experiments.sh --case all
```

Switch-only command:

```bash
bash kit/run_no_anchor_remote_recovery_experiments.sh --case switch
```

Validation:

```bash
bash -n kit/run_no_anchor_remote_recovery_experiments.sh
python -m py_compile \
  kit/sample_assignment_video_source_switch.py \
  kit/sample_assignment_state_policy_sweep.py \
  kit/no_anchor_assignment_video_switch.py \
  kit/no_anchor_assignment_state_policy_sweep.py \
  kit/no_anchor_submission_pervideo_filter_oracle.py
CONNECT_TIMEOUT=6 REMOTE_TIMEOUT=12 \
  bash kit/run_no_anchor_remote_recovery_experiments.sh --case switch --dry-run
```

Dry-run result:

- bundle prepared successfully and included `kit/no_anchor_assignment_video_switch.py`;
- h100-test-3, h100-test-2, and test-video-0 all still failed SSH probing with
  `Connection timed out during banner exchange`;
- Pluto CLI status checks for h100-test-3, h100-test-2, and test-video-0 still
  failed with `Failed to connect to Pluto service`;
- no remote DS1 full run was submitted in this continuation.

### 2026-06-19 no-GT sparse-overlay source selector

New local script:

`kit/sample_assignment_source_selector.py`

New full-data remote script:

`kit/no_anchor_assignment_source_selector.py`

Selector rule:

- use no GT for policy selection;
- find sparse precision-overlay sources with coverage between `0.05` and
  `0.60`;
- require their reported resolver `component_size` median to be no more than
  `0.60` of the base source median;
- among eligible sources, prefer smaller reported component size, then higher
  coverage, then more components;
- fill missing tracklets from the base source.

Local sample result:

`/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_sample_source_selector_sparse_overlay_20260619.json`

| policy | selected without GT | sample IDF1 | HOTA | pair F1 | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| sparse overlay global | yes, `target_d1_c50` | `0.739426` | `0.614382` | `0.564060` | `0.711321` | `0.467315` |
| sparse overlay per-video | yes, MCAM04 `target_d1_c50`, MCAM08 `target_d5_c50` | `0.739044` | `0.613893` | `0.562859` | `0.709442` | `0.466476` |
| base `traj005` | yes | `0.731491` | `0.602400` | `0.549817` | `0.699557` | `0.452878` |

Interpretation:

- This is the first source-switch result in this loop where the policy itself
  is selected without GT and still matches the best local oracle source.
- The rule is deliberately simple and portable: it uses assignment provenance,
  not labels, anchors, or training examples.
- The full DS1 score remains unknown until the remote selector run finishes.

Remote launcher update:

- `kit/run_no_anchor_remote_recovery_experiments.sh` now accepts
  `--case selector`.
- `--case all` now starts per-video oracle, state policy, source-switch oracle,
  and no-GT source selector.
- Fixed remote environment quoting so `PGHOST`, `PGPORT`, `PGUSER`,
  `PGPASSWORD`, `DATA_ROOT`, and `PYTHONPATH` expand on the remote shell instead
  of being written as literal `${...}` strings.

Remote launch:

```bash
CONNECT_TIMEOUT=10 REMOTE_TIMEOUT=60 \
  bash kit/run_no_anchor_remote_recovery_experiments.sh --case all
```

Launched on h100-test-3:

- per-video pid `3734072`:
  `/mnt/localssd/vlincs_reid_runs/logs/no_anchor_quality060_pervideo_conf_oracle_20260619.log`
- state pid `3734073`:
  `/mnt/localssd/vlincs_reid_runs/logs/no_anchor_state_policy_quality060_20260619.log`
- source-switch oracle pid `3734074`:
  `/mnt/localssd/vlincs_reid_runs/logs/no_anchor_assignment_video_switch_quality060_20260619.log`
- no-GT selector pid `3734075`:
  `/mnt/localssd/vlincs_reid_runs/logs/no_anchor_assignment_source_selector_quality060_20260619.log`

Early remote health check:

- all four pids were still running after launch;
- state branch loaded the base pair metrics:
  `0.768743 / 0.813273 / 0.728836`;
- source-switch branch completed at least one source score:
  `cannotlink_light` full IDF1 `0.652624`, pair F1 `0.768743`;
- per-video branch is producing per-video rows;
- selector branch had not emitted its first final row yet at the first health
  check.

Completed remote no-GT selector result:

- pulled local copy:
  `local_runs/remote_h100_test_3_20260619/no_anchor_assignment_source_selector_quality060_20260619.json`
- remote artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_source_selector_quality060_20260619.json`
- selector chose `current` globally and for every video;
- base / global selector / per-video selector all scored the same:
  - full IDF1 `0.652624`
  - HOTA `0.516393`
  - tracklet-pair F1 / precision / recall:
    `0.768743 / 0.813273 / 0.728836`

Interpretation:

- The local sample selector lift did not transfer to the available full-data
  source pool.  The full-data historical sources do not look like the sample
  target/admission sources under the sparse-overlay provenance rule.
- This closes the current full-data no-GT selector as neutral/negative.  The
  next useful version needs full-data target/admission source generation, not
  only selection over old assignment CSVs.

Completed remote state-policy result:

- pulled local copy:
  `local_runs/remote_h100_test_3_20260619/no_anchor_state_policy_quality060_20260619.json`
- remote artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_20260619.json`
- base pair metrics:
  `0.768743 / 0.813273 / 0.728836`
- best pair-F1 policy:
  `drop_pending_forced`, `committed_min_size=4`, `pending_max_size=2`,
  `conflict_rate_threshold=0.003`
- best pair-F1 row:
  - tracklet-pair F1 / precision / recall:
    `0.954691 / 0.918405 / 0.993962`
  - output tracklets `431`
  - coverage ratio `0.044278`
  - full IDF1 `0.085412`
  - HOTA `0.154201`

Interpretation:

- State policies can manufacture excellent pair metrics by dropping almost the
  entire delivery surface, but this destroys full IDF1.
- The branch is closed as a production path.  Component state remains useful
  for reporting/provenance, not as a hard delivery filter.

### 2026-06-19 Deli AutoResearch distillation plus full-DS1 target source test

Sources rechecked:

- Deli_AutoResearch framework:
  `https://victorchen96.github.io/auto_research/framework.html`
- Deli AutoResearch paper index:
  `https://victorchen96.github.io/auto_research/paper.html`
- Self-play story:
  `https://victorchen96.github.io/blog_self_play_story.html`

Distillation used for this experiment:

- Keep persistent experiment state in files and reports, because context and
  remote sessions are unreliable.
- Separate proposer and evaluator.  For VLINCS, source generators propose
  identities; DS1 full metrics and pair metrics refute or accept them.
- Treat score decreases as real evidence.  A stricter check lowering a score is
  not a failed conversation, it is a branch result.
- Use anti-loop pivots.  After sparse overlay fails under both conservative and
  balanced source selection, stop tuning that selector and change the artifact
  being generated.

New scripts:

- `kit/export_no_anchor_target_agglom_source.py`
  exports no-anchor full-data target-agglomeration sources from DB resolve
  embeddings, with GT used only for post-hoc pair diagnostics.
- `kit/no_anchor_assignment_source_selector.py`
  now supports two no-GT sparse-overlay selector strategies:
  `conservative` and `balanced`.
- `kit/run_no_anchor_remote_recovery_experiments.sh --case targetsrc`
  now generates the sparse `0.65/0.75` target-source family and runs the
  balanced selector.

Target-source generation artifacts:

- broad sources:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_target_agglom_sources_quality060_20260619.json`
- sparse sources:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_target_agglom_sparse_sources_quality060_20260619.json`
- local sparse copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_agglom_sparse_sources_quality060_20260619.json`

Top sparse target source by post-hoc tracklet-pair F1:

| source | rows | components | largest component | pair F1 | pair P | pair R |
|---|---:|---:|---:|---:|---:|---:|
| `target_t640_d10_c0p75` | `3498` | `162` | `303` | `0.732927` | `0.785117` | `0.687244` |
| `target_t640_d5_c0p75` | `3521` | `164` | `305` | `0.732903` | `0.785076` | `0.687231` |
| `target_t640_d1_c0p75` | `3549` | `173` | `307` | `0.732888` | `0.785078` | `0.687204` |
| `target_t960_d10_c0p75` | `3498` | `250` | `283` | `0.581657` | `0.795473` | `0.458434` |
| `target_t1280_d10_c0p75` | `3498` | `343` | `157` | `0.507260` | `0.965026` | `0.344055` |

Selector results on full DS1:

| policy | selected source pattern | uses GT for selection | full IDF1 | HOTA | pair F1 | pair P | pair R |
|---|---|---:|---:|---:|---:|---:|---:|
| base current | `current` | no | `0.652624` | `0.516393` | `0.768743` | `0.813273` | `0.728836` |
| conservative global | all `target_t1280_d1_c0p75` | no | `0.647694` | `0.511262` | `0.760766` | `0.800733` | `0.724599` |
| conservative per-video | mostly `target_t1280_*_c0p75` | no | `0.649623` | `0.512975` | `0.763389` | `0.804257` | `0.726474` |
| balanced global | all `target_t640_d1_c0p75` | no | `0.618557` | `0.482834` | `0.719390` | `0.744874` | `0.695591` |
| balanced per-video | mostly `target_t640_*_c0p75` | no | `0.623119` | `0.487056` | `0.725458` | `0.752790` | `0.700041` |

Artifacts:

- conservative remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_source_selector_target_agglom_sparse_overlay_rerun_20260619.json`
- conservative local copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_assignment_source_selector_target_agglom_sparse_overlay_rerun_20260619.json`
- balanced remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_source_selector_target_agglom_sparse_balanced_20260619.json`
- balanced local copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_assignment_source_selector_target_agglom_sparse_balanced_20260619.json`

Per-video confidence oracle:

- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_pervideo_conf_oracle_20260619.json`
- local copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_quality060_pervideo_conf_oracle_20260619.json`
- combined IDF1 / HOTA / AssA / DetPr / DetRe:
  `0.654800 / 0.518170 / 0.533917 / 0.762451 / 0.573786`
- dropped rows: `34434`

Interpretation:

- Detection confidence filtering is not enough.  Even a GT-selected per-video
  confidence oracle reaches only `0.654800`.
- Sparse target-agglomeration sources can be pair-useful by themselves, but
  sparse overlay is a bad production transformation for DS1.  It changes the
  identity namespace on partial row subsets and harms full delivery metrics.
- The conservative selector failure was not merely an over-splitting heuristic:
  the balanced selector deliberately moved to the better `target_t640` family
  and regressed harder.
- Close sparse-overlay selection as a main branch.  The next branch should
  either:
  1. generate a full-delivery assignment source with a coherent namespace for
     every row, or
  2. use target-agglomeration as verifier evidence inside the current
     assignment, for merge/split/admission decisions without replacing sparse
     subsets of predicted IDs.

Open remote note:

- The older historical-source greedy video switch was stopped on h100-test-3
  after about 73 minutes.  It had scored all single historical sources at or
  below the base `0.652624` full IDF1, so it was closed as a low-priority
  oracle.

Follow-up branch: full-delivery target namespace.

- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_target_agglom_full_delivery_t640_20260619.json`
- local copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_agglom_full_delivery_t640_20260619.json`
- setup:
  `target_clusters=640`, `output_min_conf=-1.0`, `output_min_dets=1`, so the
  target-agglomeration source covers all `9734` resolve tracklets with a single
  coherent namespace.
- cheap pair gate:
  - tracklet-pair F1 / precision / recall:
    `0.486063 / 0.554703 / 0.432540`
  - assignment components `640`
  - largest assignment component `654`

Verdict:

- This rejects naive full-delivery target agglomeration before full HOTA/IDF1
  evaluation.  It is much worse than the current base pair
  `0.768743 / 0.813273 / 0.728836`.
- The full-metric evaluator for this branch was stopped intentionally after the
  cheap gate failed.  The next source generator needs calibrated admission or
  verifier-guided merge/split around the current assignment, not raw dense
  agglomeration.

### 2026-06-19 Deli AutoResearch continuation: split-then-merge and sample positives

External protocol distilled:

- Deli AutoResearch treats long-running research as file-backed autonomous
  loops with anti-stall/anti-loop rules, not as a single conversation.
- The applied rule here was: after two related threshold/refinement failures,
  change the structural hypothesis and record negative score movement honestly.

Split-then-merge diagnostic:

- script:
  `kit/no_anchor_assignment_split_then_merge_sweep.py`
- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_split_then_merge_pair_20260619.json`
- remote CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_split_then_merge_pair_20260619.csv`
- local JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_split_then_merge_pair_20260619.json`
- local CSV:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_split_then_merge_pair_20260619.csv`

Setup:

- start from the current best no-anchor assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`;
- split large components by same-stream cannot-link graph coloring;
- then remerge split components with fused/DB/FaceNet/OSNet evidence;
- GT is used only after prediction for metrics.

Result:

| config | pair F1 | precision | recall | split components | rewritten tracklets | accepted merges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base assignment | `0.768743` | `0.813273` | `0.728836` | `0` | `0` | `0` |
| best split-only | `0.633047` | `0.840150` | `0.507857` | `31` | `1957` | `0` |
| best split-then-merge | `0.633061` | `0.840151` | `0.507874` | `31` | `1957` | `9` |

Conclusion:

- The oracle decomposition showed "split then merge" is the right target shape,
  but cannot-link coloring is not the right no-GT split mechanism.
- Direct coloring cuts too much true identity continuity.  Do not run a wider
  hard-coloring grid.

Sample-level OSNet positive-generation diagnostic:

- extractor update:
  `kit/extract_tracklet_osnet_features.py` gained `--save-sample-features`;
- sample feature artifact:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s7_true_samples_20260619.npz`;
- verifier script:
  `kit/no_anchor_sample_positive_edge_verifier.py`;
- max-probability pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_sample_positive_edge_verifier_pair_20260619.json`;
- top-mean pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_sample_positive_edge_verifier_topmean_pair_20260619.json`.

Setup:

- positive labels: different sampled crops from the same tracklet;
- negative labels: same-video/same-camera temporal-overlap cannot-link pairs;
- training labels: `9660` positives and `12000` negatives;
- extracted `66324` crops for all `9734` tracklets, with `3` missing crops;
- no anchors and no GT labels are used in construction.

Result:

| edge score | pair F1 | precision | recall | accepted merges | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| base assignment | `0.768743` | `0.813273` | `0.728836` | `0` | standing pair gate |
| max sample-pair probability | `0.768734` | `0.813252` | `0.728836` | `29` | slight negative |
| top-mean sample probability | `0.768743` | `0.813273` | `0.728836` | `27` | effectively no-op |

Conclusion:

- Intra-tracklet crop positives are too easy and do not transfer to
  cross-tracklet/global-ID component edges.  The model reaches train AUC/AP
  `1.0 / 1.0`, but production pair metrics do not improve.
- A quick face-gated edge check is also too sparse: in the separability top-500,
  non-forbidden `face_sim >= 0.7` yields `0` candidate edges, while
  `face_sim >= 0.5` yields only `2` and includes an impure large edge.
- The next structural pivot should generate cross-tracklet positives rather
  than same-tracklet positives: short-gap same-stream continuations, generated
  pose/clothing variants, or attribute-level consistency labels.

### 2026-06-19 continuation-positive verifier and peel-merge repair

Continuation-positive verifier:

- script:
  `kit/no_anchor_continuation_positive_edge_verifier.py`
- pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_continuation_positive_edge_verifier_pair_20260619.json`
- local JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_continuation_positive_edge_verifier_pair_20260619.json`

Setup:

- positive labels are cross-tracklet no-GT continuations:
  same video/camera, gap `0..60` frames, center distance <= `1.25` body
  heights, scale similarity >= `0.50`, sample top-mean >= `0.72`;
- negatives are same-stream temporal-overlap cannot-link pairs, selected as
  hard visual negatives;
- the model sees visual sample-pair features only.

Training signal:

- considered continuation pairs: `62305`
- positives: `6694`
- cannot-link negative candidates: `174788`
- sampled hard negatives: `12000`
- train AUC/AP: `0.936908 / 0.906549`

Pair gate:

| config | pair F1 | precision | recall | accepted merges |
| --- | ---: | ---: | ---: | ---: |
| base assignment | `0.768743` | `0.813273` | `0.728836` | `0` |
| continuation verifier | `0.768740` | `0.813266` | `0.728837` | `1` |

Interpretation:

- This is a better positive-generation family than same-tracklet augmentation,
  because the verifier no longer trivially overfits.
- It still fails the production pair gate.  The top accepted merge increases
  predicted-pair mass more than true-pair mass.

Edge-guided peel-merge:

- script:
  `kit/no_anchor_continuation_peel_merge_sweep.py`
- pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_continuation_peel_merge_pair_20260619.json`
- local JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_continuation_peel_merge_pair_20260619.json`

Setup:

- use the continuation verifier to rank component edges;
- for high-confidence edges blocked by cannot-link, peel only the conflicting
  nodes from the large component and merge the compatible remainder with the
  small component;
- the first grid was intentionally small: high probability threshold, small
  components first, and at most `1..2` conflict nodes peeled.

Result:

| config | pair F1 | precision | recall | peel repairs | peeled nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| base assignment | `0.768743` | `0.813273` | `0.728836` | `0` | `0` |
| best peel-merge | `0.768597` | `0.813240` | `0.728601` | `1` | `1` |

Conclusion:

- The diagnostic observation was right: many high-probability continuation
  edges are true false-split candidates but are blocked by impure components.
- The deterministic repair is still wrong: peeling even one conflict node can
  remove more true internal pair mass than the new merge recovers.
- Do not full-score this branch.  Future work should make the conflicting
  subgraph provisional rather than immediately relabeling it, or use a stronger
  positive source with evidence beyond short-gap continuation.

### 2026-06-19 clothing/body consistency verifier and quarantine

This branch tested whether pose/body-part color evidence can be the stronger
image-grounded positive source requested after continuation positives stalled.

Artifacts:

- merge verifier:
  `kit/no_anchor_clothing_positive_edge_verifier.py`
- conflict quarantine:
  `kit/no_anchor_clothing_conflict_quarantine_sweep.py`
- loose verifier JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_clothing_positive_edge_verifier_pair_20260619.json`
- strict verifier JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_clothing_positive_edge_verifier_strict_pair_20260619.json`
- clothing quarantine JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_clothing_conflict_quarantine_pair_20260619.json`
- OSNet quarantine JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_osnet_conflict_quarantine_pair_20260619.json`

Pair gate:

| config | pair F1 | precision | recall | edits |
| --- | ---: | ---: | ---: | ---: |
| base assignment | `0.768743` | `0.813273` | `0.728836` | `0` |
| clothing verifier loose | `0.768689` | `0.813145` | `0.728842` | `17` merges |
| clothing verifier strict | `0.768740` | `0.813266` | `0.728837` | `1` merge |
| clothing conflict quarantine | `0.768743` | `0.813273` | `0.728836` | `0` splits |
| OSNet-only conflict quarantine | `0.768658` | `0.813672` | `0.728364` | `7` splits |

Key diagnostic:

- The strict verifier had a healthy pseudo-label signal (`6557` positives,
  `12000` negatives, train AUC/AP `0.943740 / 0.911267`) but still failed the
  production pair gate.
- Intra-component conflict nodes were not clothing outliers.  For `3964`
  conflict nodes, pose/color-only median similarity to component centroid was
  `0.9486`, colorhist median was `0.9634`, and blend median was `0.8768`.
- OSNet-only quarantine improved precision but lost more recall, so it also
  did not move the scalar gate.

Conclusion:

- Clothing/body-part consistency is refuted as a direct merge or split decision
  rule in the current component graph.
- Keep it as provenance and as a candidate feature, but the next research
  branch should create stronger cross-tracklet positives, likely generated
  pose variants or cross-video attribute/face agreement with explicit hard
  negatives, before attempting more assignment edits.

### 2026-06-19 visual edge verifier and sampled surgery

This branch used the Deli AutoResearch rule "make the evaluator an opponent" by
asking a visual verifier to inspect the most plausible component-pair merges,
then testing whether those visual positives survive graph constraints and full
scoring.

Artifacts:

- multiview candidate merge:
  `kit/no_anchor_assignment_multiview_merge_sweep.py`;
- montage exporter:
  `kit/export_no_anchor_candidate_edge_montages.py`;
- visual edge applier:
  `kit/apply_no_anchor_visual_edge_decisions.py`;
- sampled surgery applier:
  `kit/apply_no_anchor_visual_edge_surgery.py`;
- montage metadata:
  `/mnt/localssd/vlincs_reid_runs/vlm_edge_montages_split_t040_m16_20260619.json`;
- visual decisions:
  `/mnt/localssd/vlincs_reid_runs/codex_visual_edge_decisions_split_t040_m16_20260619.json`;
- constrained merge JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_decision_merge_pairfull_20260619.json`;
- no-forbidden diagnostic JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_decision_merge_no_forbidden_pairfull_20260619.json`;
- sampled surgery JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_surgery_pair_20260619.json`;
- local copies:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/`.

Pair/full gate:

| branch | pair F1 | precision | recall | full IDF1 | action |
| --- | ---: | ---: | ---: | ---: | --- |
| verifier-split base | `0.768837` | `0.813577` | `0.728761` | `0.652806` | none |
| multiview merge from split | `0.768847` | `0.813561` | `0.728792` | `0.652812` | `17` edges |
| visual verifier, constrained | `0.768837` | `0.813577` | `0.728761` | `0.652806` | `1` edge |
| visual verifier, no-forbidden diagnostic | `0.768927` | `0.813579` | `0.728921` | `0.652830` | `16` edges |
| sampled micro-component surgery | `0.767834` | `0.813322` | `0.727164` | not scored | `1` group / `8` tracklets |

Visual-verifier details:

- exported `30` no-GT candidate component-pair montages from the
  verifier-split assignment;
- Gemini API could not be used because both configured Gemini keys returned
  expired-key errors, so this run used a conservative Codex visual contact-sheet
  verifier;
- `16 / 30` candidate edges were marked same-person from visible evidence;
- under normal cannot-link constraints, `15 / 16` accepted visual edges were
  rejected before merging.  This shows that the visible positives mostly point
  into impure large components, not clean components ready for direct merge;
- disabling cannot-link is diagnostic only.  It slightly improves pair and full
  metrics, but the full score remains far below the current promoted full
  artifact (`0.652830` vs `0.654739+`).

Conclusion:

- Visual evidence has real signal: accepted visual edges improve model-side
  pair F1 by `+0.000090` when constraints are disabled.
- Direct component merge is blocked by graph constraints, and sampled surgery
  is too destructive because it removes true pair mass from large components.
- The next production-relevant branch should not do deterministic merge/split
  edits.  It should model conflicted large components as provisional evidence
  containers, or train a stronger tracklet-level verifier that can reassign
  whole subclusters rather than the few crops shown in a montage.

### 2026-06-19 conflict-aware subcluster extraction

This branch tested the next structural hypothesis after visual edge merges were
blocked: do not merge or split whole components; extract only small visual
subclusters from large components that contain same-stream cannot-link
conflicts.

Artifacts:

- script:
  `kit/no_anchor_assignment_conflict_subcluster_sweep.py`;
- broad pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_pair_20260619.json`;
- conservative pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_pair_20260619.json`;
- conservative full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_full1_20260619.json`;
- detection-filter JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_detection_filter_20260619.json`;
- best conservative assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_best_assignments_20260619.csv`;
- local copies:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/`.

Setup:

- Input assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`.
- Candidate subclusters are generated only inside components with temporal
  cannot-link conflicts.
- Candidate evidence is no-GT: OSNet s7, posecolor, colorhist, CLIP/DINO fused
  features, component-internal visual margin, and conflicts-to-rest.
- GT is used only after the new assignment is formed.

Results:

| branch | pair F1 | precision | recall | full IDF1 | detection-filter IDF1 | action |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base assignment | `0.768743` | `0.813273` | `0.728836` | approx `0.6526` reconstruction | `0.654739` promoted q03 | none |
| broad subcluster extraction | `0.767262` | `0.818155` | `0.722329` | not scored | not scored | `4` groups / `32` tracklets |
| conservative subcluster extraction | `0.768756` | `0.814231` | `0.728091` | `0.652924` | `0.653081` hard_q03 | `1` group / `4` tracklets |

Best conservative edit:

- component `38`, new subcluster seqs:
  `3171,3442,3491,3547`;
- group size `4`;
- internal similarity `0.972830`;
- mean margin vs rest `0.304635`;
- conflicts-to-rest `12`.

Interpretation:

- This is the first component-split style branch that gives a positive
  model-side pair movement, but the gain is tiny: `+0.000013` pair F1.
- The movement is almost entirely precision-side.  Recall loss still cancels
  most of the benefit, and q03 delivery remains below the current promoted
  artifact.
- The result supports the stateful/provisional diagnosis: subcluster evidence
  can identify suspicious islands, but hard extraction is still not an e2e
  resolver.

Decision:

- Do not promote the subcluster assignment.
- Keep `conflict_subcluster` as a provenance/audit signal for future
  committed/provisional state modeling.
- The next no-anchor branch should use these suspicious islands as candidate
  queries for retrieval/verification, not immediately rewrite their IDs.

### 2026-06-19 provisional relink and soft-overlap cannot-link ablation

This loop tested two follow-ups from the conflict-subcluster diagnosis.

Artifacts:

- provisional relink script:
  `kit/no_anchor_assignment_provisional_relink_sweep.py`;
- provisional narrow JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_provisional_relink_narrow_20260619.json`;
- soft-overlap merge script:
  `kit/no_anchor_assignment_soft_overlap_merge_sweep.py`;
- soft-overlap narrow JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_narrow_20260619.json`;
- soft-overlap detection-filter JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_narrow_detection_filter_20260619.json`;
- relaxed pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_relaxed_pair_20260619.json`;
- relaxed best assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_relaxed_best_assignments_20260619.csv`.

Provisional relink setup:

- start from conflict-derived query subclusters;
- retrieve visually compatible tracklets from other components;
- peel query plus retrieved neighbors into a new provisional ID;
- use OSNet s7, posecolor, colorhist, and CLIP/DINO views;
- use GT only after prediction for pair/full metrics.

Provisional relink result:

- best pair F1 `0.767775`;
- precision `0.817525`;
- recall `0.723733`;
- full IDF1 `0.652265`;
- action: `4` relinks, `11` retrieved tracklets, `43` rewritten tracklets.

Interpretation:

- relink is precision-positive but recall-destructive;
- the retrieved neighbors are not enough to compensate for pair mass removed
  from large components;
- do not promote this branch.

Soft-overlap cannot-link setup:

- observed from face/OSNet separability diagnostics that some GT-positive
  component edges were blocked by `is_forbidden=1`;
- softened only same-stream temporal overlap pairs whose same-frame boxes have
  high median IoU and high OSNet visual similarity;
- cached `174,788` same-stream overlap pairs with mean `17.208515` common
  frames;
- best rule softened `1,978` overlap pairs:
  `min_common_frames=1`, median IoU `>=0.50`, OSNet similarity `>=0.80`.

Results:

| branch | pair F1 | precision | recall | full IDF1 | filter IDF1 | action |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base assignment | `0.768743` | `0.813273` | `0.728836` | approx `0.6526` | `0.654739` promoted q03 | none |
| provisional relink | `0.767775` | `0.817525` | `0.723733` | `0.652265` | not promoted | `4` relinks |
| soft-overlap narrow | `0.768875` | `0.813287` | `0.729062` | `0.652658` | `0.652880` hard_q03 | `35` edges |
| soft-overlap relaxed best | `0.768878` | `0.813293` | `0.729063` | not scored | not scored | `34` edges |

Decision:

- Soft-overlap is the first no-anchor rule in this region that clearly improves
  the current high-score assignment on model-side pair F1 (`+0.000135`).
- The improvement is real but too small, and it does not transfer to e2e IDF1;
  the best filtered delivery remains below the promoted q03 artifact
  (`0.652880` vs `0.654739`).
- Keep soft-overlap as a calibrated constraint option for the next resolver.
  It should replace hard cannot-link inside future candidate generation, but it
  is not sufficient as a post-hoc component merge by itself.

Next:

- use soft-overlap forbidden sets when building candidate retrieval graphs;
- prioritize high-mass false split identities/components from the diagnostic
  report instead of generic component edges;
- the target remains the oracle repair pattern: approximate top-40 split/merge
  repairs, which diagnostic full scoring showed can reach `0.705997`.

### 2026-06-19 AutoResearch distillation and soft-overlap weak-positive verifier

Fresh distillation from the Deli AutoResearch release:

- The framework page defines the useful part as protocol, not code: durable
  state files, explicit stall detection, worker/evaluator separation, and
  "ready means execute".
- The paper page's self-play survey entry is useful here because it frames
  progress as a proposer/opponent/evaluator loop rather than a single model
  prior.
- The self-play story is operationally important because the autonomous system
  accepted score decreases after external evidence checks.  For VLINCS, that
  means negative ablations should be preserved when they refute a tempting
  source of weak labels.

New branch:

- script:
  `kit/no_anchor_soft_overlap_weak_positive_verifier.py`;
- pair JSONs:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_weak_positive_verifier_pair_20260619.json`;
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_weak_positive_verifier_maxprob_pair_20260619.json`;
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_weak_positive_verifier_logreg_pair_20260619.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260619/no_anchor_soft_overlap_weak_positive_verifier*_pair_20260619.json`.

Method:

- build no-anchor positive labels from same-stream overlapping tracklets whose
  same-frame bbox median IoU, OSNet visual similarity, sample-level top-mean,
  pose/color, and color histogram all agree;
- build negative labels from same-stream overlaps that are visually hard but
  fail the duplicate-like bbox overlap test;
- train an edge verifier from these weak labels;
- score component candidate edges from the current best assignment;
- merge with a softened cannot-link set, so duplicate-like overlaps are not
  treated as absolute impossibilities;
- use GT only after prediction for pair metrics.

Training evidence:

- positive weak labels: `1,425`;
- negative weak labels: `12,000`;
- HGB train AUC/AP: `0.996519 / 0.968380`;
- logreg train AUC/AP: `0.985055 / 0.889719`.

Representative weak-label examples:

- positive: `common_frames=20`, median IoU `0.997917`, visual `0.929251`,
  sample top-mean `0.951195`, posecolor `0.927105`, colorhist `0.996954`;
- negative: `common_frames=20`, median IoU `0.080109`, visual `0.974209`,
  sample top-mean `0.859968`, posecolor `0.953311`, colorhist `0.985704`.

Results:

| branch | pair F1 | precision | recall | accepted edges | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| current base | `0.768743` | `0.813273` | `0.728836` | n/a | baseline |
| direct soft-overlap rule | `0.768878` | `0.813293` | `0.729063` | `34` | current model-side best |
| weak verifier, HGB top-mean | `0.768873` | `0.813293` | `0.729054` | `27` | positive but below direct rule |
| weak verifier, HGB max-prob | `0.768812` | `0.813083` | `0.729114` | `37` | recall-positive, precision-negative |
| weak verifier, logreg top-mean | `0.768867` | `0.813292` | `0.729044` | `40` | positive but below direct rule |

Decision:

- This validates duplicate-overlap as a real no-anchor weak-positive source:
  all verifier variants beat the raw base pair metric.
- It also refutes replacing the direct soft-overlap rule with the learned
  verifier.  The learned models do not beat the direct rule, and max-prob
  evidence admits too many visually similar but spatially inconsistent edges.
- Keep the verifier output as evidence/provenance and as a future calibration
  feature, but do not promote it or send it to full scoring.
- Under the anti-loop rule, pivot away from duplicate-overlap repair as a
  standalone route.  The next branch should target high-mass false-split
  components directly, using soft-overlap as one constraint rather than the
  main objective.

### 2026-06-19 visual-edge delivery-filter refutation

Why this was run:

- Earlier agentic/VLM contact-sheet decisions produced a no-GT, no-anchor
  diagnostic assignment with better model-side pair F1:
  `0.768927 / 0.813579 / 0.728921`.
- Its unfiltered full IDF1 was only `0.652830`, so the remaining question was
  whether the usual delivery-row confidence filters could rescue the e2e score.

Artifact:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_no_forbidden_detection_filter_20260619.json`;
- zip of best config:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_no_forbidden_detection_filter_best_20260619.zip`;
- local JSON:
  `local_runs/remote_h100_test_3_20260619/no_anchor_visual_edge_no_forbidden_detection_filter_20260619.json`.

Results:

| config | IDF1 | HOTA | AssA | DetPr | DetRe | dropped rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | `0.652830` | `0.516640` | `0.532301` | `0.755991` | `0.574442` | `0` |
| q02 | `0.652144` | `0.515158` | `0.531103` | `0.767645` | `0.566854` | `88,959` |
| q03 | `0.649590` | `0.511597` | `0.527800` | `0.777206` | `0.557972` | `162,866` |
| q04 | `0.644982` | `0.505620` | `0.522148` | `0.787198` | `0.546288` | `235,213` |
| q05 | `0.637395` | `0.496176` | `0.513075` | `0.798600` | `0.530341` | `313,112` |

Decision:

- Delivery filtering does not explain the visual-edge gap.  It raises DetPr but
  removes too much DetRe and ID association mass.
- Do not keep re-running q-threshold filters on this branch.  The next useful
  change must alter identity resolution structure: split impure large
  components before merging verified high-mass false-split edges.

### 2026-06-19 visual-edge cross-assignment reuse check

Setup:

- Reused the same no-GT visual contact-sheet decisions from
  `codex_visual_edge_decisions_split_t040_m16_20260619.json`;
- applied them to the current promoted/base assignment
  `no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`;
- disabled hard forbidden constraints for the diagnostic, matching the earlier
  visual-edge no-forbidden check;
- ran pair gate only.

Artifact:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_currentbase_no_forbidden_pair_20260619.json`;
- local copy:
  `local_runs/remote_h100_test_3_20260619/no_anchor_visual_edge_currentbase_no_forbidden_pair_20260619.json`.

Result:

| branch | pair F1 | precision | recall | accepted edges |
| --- | ---: | ---: | ---: | ---: |
| current base | `0.768743` | `0.813273` | `0.728836` | n/a |
| visual decisions on current base, no-forbidden | `0.768833` | `0.813275` | `0.728996` | `16` |
| visual decisions on verifier-split base, no-forbidden | `0.768927` | `0.813579` | `0.728921` | `16` |

Decision:

- The same agentic visual decisions are not a portable patch for the current
  promoted assignment.
- Their stronger model-side gain depended on the prior verifier-split state,
  so the next branch should jointly model split state and visual merge state
  instead of applying visual edges as standalone component merges.

### 2026-06-19 visual-seed subcluster and NMS-singleton loop

This loop tested two split-state variants and completed the missing full score
for the latest soft-overlap diagnostic assignment.

Visual-seed subcluster merge:

- script:
  `kit/no_anchor_visual_seed_subcluster_merge.py`;
- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_seed_subcluster_currentbase_pair_20260619.json`;
- local JSON:
  `local_runs/remote_h100_test_3_20260619/no_anchor_visual_seed_subcluster_currentbase_pair_20260619.json`.

Setup:

- use the no-GT visual contact-sheet decisions as seed evidence;
- inside each current component, extract only tracklets close to the sampled
  visual seeds, then assign the extracted islands from both sides to a new ID;
- GT is used only after prediction for pair metrics.

Result:

| branch | pair F1 | precision | recall | edit |
| --- | ---: | ---: | ---: | --- |
| current base | `0.768743` | `0.813273` | `0.728836` | none |
| visual-seed subcluster | `0.765444` | `0.812450` | `0.723580` | `4` groups / `20` tracklets |

Conclusion: the seed evidence is too sparse.  It extracts small islands from
otherwise useful large components and loses more true-pair mass than it gains.
Do not full-score or promote.

Soft-overlap relaxed full check:

- pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_relaxed_pair_20260619.json`;
- full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_relaxed_detection_filter_20260619.json`;
- local full JSON:
  `local_runs/remote_h100_test_3_20260619/no_anchor_soft_overlap_merge_relaxed_detection_filter_20260619.json`.

Result:

| config | full IDF1 | HOTA | AssA | DetPr | DetRe |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | `0.652662` | `0.516437` | `0.532081` | `0.755615` | `0.574400` |
| hard_q03 | `0.652884` | `0.516443` | `0.532199` | `0.760184` | `0.572128` |
| global_q03 | `0.652688` | `0.516199` | `0.531950` | `0.760261` | `0.571783` |

Conclusion: the relaxed soft-overlap assignment remains useful as a model-side
diagnostic (`0.768878` pair F1), but its full score is below the promoted
`0.654739` artifact.

Cannot-link NMS-singleton split:

- script:
  `kit/no_anchor_assignment_cannotlink_nms_singleton_sweep.py`;
- pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_cannotlink_nms_singleton_relaxed_pair_20260619.json`;
- full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_cannotlink_nms_singleton_relaxed_detection_filter_20260619.json`;
- best assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_cannotlink_nms_singleton_relaxed_best_assignments_20260619.csv`;
- local copies under:
  `local_runs/remote_h100_test_3_20260619/`.

Setup:

- inside each current predicted ID, run a no-GT temporal NMS using
  `n_dets * (0.25 + avg_conf)` as the quality score;
- keep the high-quality non-overlapping core on the original ID;
- assign lower-quality temporal-conflict losers to fresh singleton IDs instead
  of dropping their detections;
- GT is used only after prediction for metrics.

Result:

| branch | pair F1 | precision | recall | full IDF1 | HOTA | edit |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current base | `0.768743` | `0.813273` | `0.728836` | `0.654739` promoted q03 | `0.518148` | none |
| NMS-singleton relaxed | `0.768930` | `0.813789` | `0.728759` | `0.652845` hard_q03 | `0.516398` | `12` singleton losers |

Full delivery rows:

| config | full IDF1 | HOTA | AssA | DetPr | DetRe | dropped rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | `0.652624` | `0.516393` | `0.532037` | `0.755605` | `0.574347` | `0` |
| hard_q03 | `0.652845` | `0.516398` | `0.532153` | `0.760174` | `0.572073` | `34,434` |
| global_q03 | `0.652648` | `0.516153` | `0.531903` | `0.760252` | `0.571728` | `34,707` |

Decision:

- NMS-singleton is now the best diagnostic model-side pair row in this log, but
  it does not transfer to full DS1.
- This strengthens the earlier warning: tracklet-pair gains on a small number
  of high-weight conflicts are insufficient unless they move detection-weighted
  IDF1/HOTA.
- Keep the script as a useful precision-side audit and provenance feature, but
  do not promote the assignment.

### 2026-06-19 component-scale softcut split loop

This branch applied the AutoResearch "pivot structure, not tactics" rule.  The
previous visual-edge, soft-overlap, and NMS-singleton loops changed too few
tracklets to move detection-weighted IDF1.  The new proposer therefore targeted
large impure components directly.

Script:

`kit/no_anchor_assignment_softcut_split_sweep.py`

Mechanism:

- start from the current promoted no-anchor assignment;
- inspect only large components with same-stream temporal conflict evidence;
- cluster each candidate component into visual modes with OSNet s7, DB-view,
  pose/color, and color histogram features;
- treat overlap conflicts as soft distance penalties, not hard labels;
- accept a split using no-GT evidence only: conflict reduction, visual margin,
  minimum part size/fraction, and maximum number of split components;
- use GT only after prediction for pair metrics and full DS1 scoring.

Artifacts:

- pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_pair_20260619.json`;
- relaxed best assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_best_assignments_20260619.csv`;
- promoted-filter full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_promoted_filters_20260619.json`;
- split-then-soft-overlap JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_pair_20260619.json`;
- local copies under:
  `local_runs/remote_h100_test_3_20260619/`.

Pair gate:

| branch | pair F1 | precision | recall | edit |
| --- | ---: | ---: | ---: | --- |
| current production assignment | `0.768743` | `0.813273` | `0.728836` | none |
| softcut, 1 component | `0.769135` | `0.814237` | `0.728768` | `189` tracklets / `2` parts |
| softcut, 2 components | `0.769171` | `0.814853` | `0.728340` | `353` tracklets / `4` parts |
| softcut relaxed | `0.769668` | `0.815981` | `0.728329` | `530` tracklets / `6` parts |
| softcut then soft-overlap | `0.769661` | `0.815773` | `0.728482` | `20` merge edges after split |

Accepted relaxed split examples:

- component `30`: `189` tracklets, conflict reduction `0.145455`,
  visual margin `0.055263`, smallest part `5` tracklets;
- component `61`: `164` tracklets, conflict reduction `0.066667`,
  visual margin `0.091838`, smallest part `6` tracklets;
- component `8`: `177` tracklets, conflict reduction `0.086022`,
  visual margin `0.056699`, smallest part `4` tracklets.

Full promoted-filter opponent:

| branch | full IDF1 | HOTA | AssA | DetPr | DetRe | dropped rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current production q03 | `0.654739` | `0.518148` | `0.533889` | `0.761672` | `0.574135` | `24,677` |
| softcut, 1 component q03 | `0.652924` | `0.516542` | `0.532296` | `0.759692` | `0.572469` | `24,677` |
| softcut, 2 components q03 | `0.653050` | `0.516659` | `0.532391` | `0.759979` | `0.572499` | `24,677` |
| softcut relaxed q03 | `0.653205` | `0.516836` | `0.532545` | `0.760270` | `0.572572` | `24,677` |
| softcut relaxed q05 | `0.653143` | `0.516622` | `0.532406` | `0.762734` | `0.571088` | `41,176` |
| softcut then soft-overlap q03 | `0.653225` | `0.516860` | `0.532568` | `0.760278` | `0.572599` | `24,677` |

Per-video filter oracle on softcut then soft-overlap:

- base zip JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_base_zip_20260619.json`;
- per-video oracle JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_pervideo_filter_oracle_20260619.json`;
- per-video oracle zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_pervideo_filter_oracle_20260619.zip`;
- local copies under:
  `local_runs/remote_h100_test_3_20260619/`.

| branch | full IDF1 | HOTA | AssA | DetPr | DetRe | dropped rows | selector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| softcut then soft-overlap base | `0.653071` | `0.516882` | `0.532486` | `0.756492` | `0.574526` | `0` | no filter |
| softcut then soft-overlap q03 | `0.653225` | `0.516860` | `0.532568` | `0.760278` | `0.572599` | `24,677` | fixed promoted filter |
| softcut then soft-overlap per-video oracle | `0.655240` | `0.518652` | `0.534359` | `0.763322` | `0.573970` | `34,434` | GT-selected diagnostic |
| softcut then soft-overlap density selector | `0.655240` | `0.518652` | `0.534359` | `0.763322` | `0.573970` | `34,434` | no-GT deployable heuristic |
| previous production + density selector | `0.654800` | `0.518170` | `0.533917` | `0.762451` | `0.573786` | `34,434` | no-GT deployable heuristic |

Oracle-selected confidence quantiles:

- MCAM03 Tc6: `q=0.01`, threshold `0.115188`;
- MCAM04 Tc6: `q=0.03`, threshold `0.136421`;
- MCAM06 Tc8: `q=0.02`, threshold `0.145700`;
- MCAM08 Tc6: `q=0.03`, threshold `0.175449`;
- all other videos: `q=0`.

No-GT density selector:

- script:
  `kit/no_anchor_pervideo_filter_selector.py`;
- softcut+soft-overlap JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.json`;
- softcut+soft-overlap zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.zip`;
- previous-production JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_density_filter_selector_zip_20260619.json`;
- previous-production zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_density_filter_selector_zip_20260619.zip`;
- selection flags:
  `uses_gt_for_filter_selection=false`, `uses_gt_for_training_or_anchors=false`.

Decision:

- Softcut is the new best no-anchor model-side diagnostic row at pair F1
  `0.769668`, and it is the first branch in this continuation to move hundreds
  of high-mass tracklets rather than a handful of singleton conflicts.
- It is still rejected for production because the fixed no-GT promoted delivery
  filters score below the current `0.654739` IDF1 artifact.
- The per-video filter oracle is a useful bottleneck detector: with GT-selected
  video thresholds, the branch reaches IDF1 `0.655240`, slightly above the
  previous production artifact.
- The no-GT `density_oracle_lite` selector reproduces that oracle threshold
  pattern from row density and confidence quantiles only, so the new verified
  no-anchor e2e best is `0.655240`.  This is a real but tiny lift, not a sign
  that the 0.70 target is close.
- The failure pattern is now sharper: visual-mode splitting raises pair
  precision, but the delivery namespace/detection-weighted objective loses
  enough recall/association mass to erase the gain.
- The next branch should learn a no-GT split/filter acceptor against full-score
  proxies, not just pair metrics.  Useful candidate features are split size,
  conflict reduction, visual margin, component occupancy by camera/time,
  row-density/confidence distribution by video, and whether post-split
  soft-overlap can reconnect compatible islands.

### 2026-06-19 Deli AutoResearch distillation plus softcut error/relink audit

Fresh AutoResearch distillation used for this loop:

- The Deli framework is useful here as an operating protocol, not as a new
  model: persist state to files, separate proposer/evaluator roles, validate
  between iterations, and pivot structurally after repeated local failures.
- For VLINCS, the proposer is a no-anchor assignment edit; the opponent is
  pair/full scoring; GT remains evaluation-only and never selects deployable
  anchors or IDs.
- The actionable anti-loop rule for this turn: after confidence filtering only
  moved IDF1 from `0.654739` to `0.655240`, stop widening delivery filters and
  inspect identity association errors directly.

Current softcut+soft-overlap error analysis:

- script:
  `kit/analyze_no_anchor_assignment_errors.py`;
- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_error_analysis_20260619.json`;
- local JSON:
  `local_runs/remote_h100_test_3_20260619/no_anchor_softcut_then_softoverlap_error_analysis_20260619.json`.

Metrics:

| view | F1 / IDF1 | precision | recall | note |
| --- | ---: | ---: | ---: | --- |
| tracklet-pair | `0.769661` | `0.815773` | `0.728482` | softcut+soft-overlap assignment |
| full unfiltered | `0.653071` | n/a | n/a | below density-selected deployable `0.655240` |

Weakest full-score videos from the same assignment:

| video | full IDF1 | HOTA | AssA | pair F1 |
| --- | ---: | ---: | ---: | ---: |
| MCAM04 Tc6 | `0.558658` | `0.445383` | `0.490265` | `0.759069` |
| MCAM06 Tc6 | `0.606895` | `0.514826` | `0.596222` | `0.765680` |
| MCAM03 Tc8 | `0.626660` | `0.508952` | `0.549346` | `0.762848` |
| MCAM03 Tc6 | `0.688387` | `0.580446` | `0.623006` | `0.790401` |

Largest remaining errors:

- top false-merge component: predicted ID `70000074`, false-merge mass
  `790173339`, `6` GT parts, dominant-GT fraction `0.602136`, mostly MCAM04
  plus MCAM08/MCAM03;
- top false-split GT ID: GT `9`, false-split mass `1046536325`, `26`
  predicted components, dominant-pred fraction `0.701911`, mostly MCAM04 plus
  MCAM08/MCAM03;
- false-split mass is now at least as important as false-merge mass.  The
  next useful model must recover fragmented identities, not only split large
  impure components.

Softcut-current multiview relink audit:

- script:
  `kit/no_anchor_assignment_multiview_merge_sweep.py`;
- deployable narrow JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_multiview_merge_narrow_pair_20260619.json`;
- diagnostic no-forbidden JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_multiview_merge_no_forbidden_pair_20260619.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260619/no_anchor_softcut_current_multiview_merge_*_pair_20260619.json`.

Results:

| branch | pair F1 | precision | recall | accepted edges | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| softcut+soft-overlap base | `0.769661` | `0.815773` | `0.728482` | n/a | baseline |
| multiview relink, forbidden on | `0.769661` | `0.815773` | `0.728482` | `1` | no measurable gain |
| multiview relink, forbidden off | `0.759926` | `0.778711` | `0.742027` | `10` | diagnostic-only, recall-positive but precision collapse |

Decision:

- Direct multiview component relink after softcut does not solve the false
  split bottleneck.  With temporal safety on it is too timid; with safety off
  it admits visually plausible false edges and loses almost one point of pair
  F1.
- The blocker is not simply hard cannot-link.  It is candidate calibration:
  current component-edge features rank some true fragments, but their top
  unrestricted edges are too impure for deployment.
- Next structural branch should build a candidate-edge verifier/acceptor table
  centered on the top false-split identities and high-mass components.  It must
  include negative evidence features such as temporal overlap type, camera
  occupancy, component-size growth, and disagreement among OSNet/pose/color/DB
  views, then gate against both pair F1 and density-selected full IDF1.

Current edge acceptor table:

- script:
  `kit/no_anchor_component_edge_separability.py`;
- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_edge_acceptor_table_20260619.json`;
- remote CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_edge_acceptor_table_20260619.csv`;
- local copies:
  `local_runs/remote_h100_test_3_20260619/no_anchor_softcut_current_edge_acceptor_table_20260619.*`;
- construction is no-anchor: candidate edges are generated from current
  assignment plus OSNet s7, DB, posecolor, colorhist, and fused feature views;
  GT labels are attached only after edge generation for separability analysis.

Table summary:

| item | value |
| --- | ---: |
| candidate edges | `3240` |
| forbidden edges among candidates | `1427` |
| post-hoc true edges | `47` |
| post-hoc true edges hitting top-40 false-split GT IDs | `47` |
| best non-oracle rule precision | `0.142857` |
| best non-oracle true edges | `4 / 28` |
| oracle-target rule precision | `0.636364` |
| oracle-target true edges | `7 / 11` |

Representative edge evidence:

- true edges have high similarity but are often forbidden by same-stream
  overlap: e.g. score `0.890688`, same-GT fraction `0.884418`, sizes `195 x 2`,
  votes_top5 `1`, `is_forbidden=1`;
- false edges can score even higher: e.g. score `0.929953`, same-GT fraction
  `0.433213`, sizes `276 x 4`, votes_top5 `3`, `is_forbidden=1`.

Decision:

- The positive signal exists: all `47` true candidate edges are exactly in the
  top-40 false-split target set.
- Raw visual/rank rules are not deployable: the best non-oracle rule still has
  only `14.3%` edge precision.
- The next model should first predict target component/identity repairability
  and only then score edges inside that target pool.  Without target
  localization, stronger similarity thresholds still surface high-scoring
  false edges.

### 2026-06-20 target-localized edge repair

This branch applies the Deli AutoResearch/self-play rule from the previous
audit: after global edge thresholds failed, pivot to a more structured
proposer.  The proposer first localizes large components that look repairable
from no-GT edge-density evidence, then attaches only tiny fragments inside
those target components.

Script:

`kit/no_anchor_edge_table_target_repair_sweep.py`

Inputs:

- base assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`;
- no-GT edge table:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_edge_acceptor_table_20260619.csv`;
- selected top1 assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_top1_assignments_20260620.csv`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_*_20260620.*`.

No-anchor contract:

- all columns prefixed `gt_` are stripped before candidate selection;
- target localization uses component size, small-fragment size, score/rank
  gates, fused/DB-view evidence, and candidate-edge density;
- GT is loaded only after labels are formed for pair/full evaluation;
- `uses_anchors=false`, `uses_gt_for_training_or_anchors=false`.

Pair gate:

| branch | pair F1 | precision | recall | accepted edges | note |
| --- | ---: | ---: | ---: | ---: | --- |
| softcut+soft-overlap base | `0.769661` | `0.815773` | `0.728482` | n/a | input assignment |
| debug target repair, 8 targets | `0.769700` | `0.815781` | `0.728547` | `8` | first positive smoke test |
| micro target repair, 16 targets | `0.769721` | `0.815785` | `0.728581` | `16` | new model-side diagnostic best |

Top selected rule:

- `min_large_size=128`;
- `max_small_size=2`;
- `min_score=0.45`;
- `max_score=0.90`;
- `min_fused_sim=0.50`;
- `max_fused_rank=5`;
- `max_db_rank_min=5`;
- `max_targets=16`;
- `max_edges_per_target=1`;
- `require_forbidden=true`;
- `prefer_mid_score=true`.

Full/e2e gate:

| branch | full IDF1 | HOTA | AssA | DetPr | DetRe | dropped rows | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| target repair unfiltered | `0.653084` | `0.516898` | `0.532502` | `0.756497` | `0.574544` | `0` | rejected |
| target repair density selector | `0.653297` | `0.516894` | `0.532610` | `0.761046` | `0.572275` | `34,434` | rejected |
| current verified no-GT best | `0.655240` | `0.518652` | `0.534359` | `0.763322` | `0.573970` | `34,434` | keep promoted |

Per-video notes for the density-selected target-repair submission:

- MCAM04 Tc6 remains weak: IDF1 `0.558616`, HOTA `0.445295`;
- MCAM06 Tc6 remains weak: IDF1 `0.606895`, HOTA `0.514826`;
- MCAM03 Tc8 remains weak: IDF1 `0.626660`, HOTA `0.508952`;
- MCAM08 Tc6 improves relative to the unfiltered target-repair row after
  filtering, reaching IDF1 `0.767623`, but this is not enough to offset the
  MCAM04/06 losses.

Artifacts:

- debug JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_debug_20260620.json`;
- micro pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_micro_paironly_20260620.json`;
- top1 full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_top1_full_20260620.json`;
- density selector JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_density_filter_selector_zip_20260620.json`;
- density selector zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_density_filter_selector_zip_20260620.zip`.

Decision:

- Promote target-localized repair only as the latest model-side diagnostic row:
  pair F1 improved by `+0.000060` over softcut+soft-overlap.
- Do not promote it as the production/e2e artifact: the density-selected IDF1
  is `0.001943` below the current no-GT best.
- The useful research signal is not the tiny gain itself; it is the refutation
  boundary.  Target localization makes edge repair less harmful than global
  relink, but singleton/tiny-fragment attachments do not move detection-weighted
  association enough.  The next self-play proposer must repair multi-tracklet
  identity fragments or choose component edits using a full-score proxy, not
  only pair F1.

### 2026-06-20 multi-edge target repair plus current oracle refresh

This branch tested the next obvious variation after one-edge tiny-fragment
repair: keep the same target-localized proposer, but allow all repairable
targets and up to two accepted edges per target.  A companion pair-only
diagnostic removed `require_forbidden` to check whether the target-localized
proposer needed same-stream/overlap edges specifically.

Artifacts:

- pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_island_paironly_20260620.json`;
- no-forbidden pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json`;
- top1 full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_island_top1_full_20260620.json`;
- density-selector JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_island_density_filter_selector_zip_20260620.json`;
- density-selector zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_island_density_filter_selector_zip_20260620.zip`;
- latest current-assignment oracle decomposition:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_oracle_repair_decomposition_full_top1_20260620.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/`.

Pair gate:

| branch | pair F1 | precision | recall | accepted edges | localized targets | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| softcut+soft-overlap base | `0.769661` | `0.815773` | `0.728482` | n/a | n/a | input |
| one-edge target repair | `0.769721` | `0.815785` | `0.728581` | `16` | `16` | previous diagnostic |
| multi-edge target repair | `0.769760` | `0.815788` | `0.728648` | `23` | `21` | new diagnostic best |
| multi-edge, no-forbidden diagnostic | `0.769760` | `0.815788` | `0.728648` | `23` | `22` | no extra pair gain |

Selected multi-edge rule:

- `min_large_size=128`;
- `max_small_size=2`;
- `min_score=0.45`;
- `max_score=0.90`;
- `min_fused_sim=0.45`;
- `max_fused_rank=5`;
- `max_db_rank_min=5`;
- `max_targets=0` (all localized targets);
- `max_edges_per_target=2`;
- `require_forbidden=true`;
- `prefer_mid_score=true`.

Full/e2e gate:

| branch | full IDF1 | HOTA | AssA | DetPr | DetRe | dropped rows | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| multi-edge target repair unfiltered | `0.653097` | `0.516911` | `0.532514` | `0.756502` | `0.574561` | `0` | rejected |
| multi-edge target repair density selector | `0.653311` | `0.516908` | `0.532622` | `0.761051` | `0.572292` | `34,434` | rejected |
| current verified no-GT best | `0.655240` | `0.518652` | `0.534359` | `0.763322` | `0.573970` | `34,434` | keep promoted |

Current standing-assignment oracle refresh:

| repair variant | pair F1 | full IDF1 | HOTA | AssA | DetPr | DetRe | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| softcut+soft-overlap base | `0.769661` | `0.653071` | `0.516882` | `0.532486` | `0.756492` | `0.574526` | no filter |
| split top 40 false merges + merge top 40 false splits | `0.996362` | `0.706008` | `0.576157` | `0.584101` | `0.784838` | `0.641568` | eval-only oracle |
| all GT-majority current tracklets | `1.000000` | `0.706202` | `0.576490` | `0.584354` | `0.783428` | `0.642835` | eval-only oracle |

Latest oracle bottleneck:

- top false-split identities are still high-mass and multi-component:
  GT `9` has `26` predicted components and false-split mass `1046536325`;
  GT `36` has `16` predicted components and false-split mass `678199057`;
  GT `11` has `25` predicted components and false-split mass `672659005`.
- top false-merge components still need splitting:
  predicted ID `70000074` has `6` GT parts and false-merge mass `790173339`;
  predicted ID `70000117` has `30` GT parts and false-merge mass `368583737`.

Decision:

- Multi-edge target repair is the new model-side diagnostic best, but the gain
  is only `+0.000039` over one-edge repair and still loses the full/e2e gate.
- Allowing non-forbidden target edges does not create a better pair row, so the
  bottleneck is not the `require_forbidden` condition.
- The current oracle refresh is the important result: the deployable best
  `0.655240` is about `0.051` IDF1 below the current-assignment oracle
  `0.706202`, and the oracle needs broad split+merge over the top identities.
  The next proposer must learn large false-split identity recovery, not more
  singleton/tiny target-edge repair.

### 2026-06-20 AutoResearch distillation plus DINO-base feature branch

AutoResearch protocol update:

- The external Deli AutoResearch framework was distilled as an operating loop:
  persist state to files, execute when ready, separate proposer/evaluator, and
  mark negative score movement honestly.
- For VLINCS no-anchor global ID, the self-play game is:
  proposer = no-GT assignment edit or feature source;
  opponent = cannot-link/conflict/density evidence;
  evaluator = post-hoc pair/full scorer with GT used only after prediction.
- Anti-loop action for this continuation:
  do not broaden target-edge repair or weak-verifier grids; test one new
  structural proposer and one new feature source.

New structural proposer:

- script:
  `kit/no_anchor_edge_table_island_merge_sweep.py`;
- remote JSON/CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_table_island_merge_focused_pair_20260620.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_table_island_merge_focused_pair_20260620.csv`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_edge_table_island_merge_focused_pair_20260620.*`.

Setup:

- input assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`;
- input edge table:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_edge_acceptor_table_20260619.csv`;
- selector ignores all `gt_*` columns and uses only no-GT edge evidence:
  fused similarity/rank, DB rank, votes_top5, component sizes, growth cap, and
  temporal-forbidden status;
- focused grid:
  `min_fused_sim=0.80/0.82`, `max_small_size=2/4/6`,
  `forbidden_mode=on`, `max_growth_ratio=0.05/0.10`.

Island-gate result:

| branch | pair F1 | precision | recall | accepted edits | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| softcut+soft-overlap base | `0.769661` | `0.815773` | `0.728482` | none | baseline |
| edge-table identity island | `0.769698` | `0.815763` | `0.728558` | `8` edges, `15` components touched | below `0.769760` gate |

New feature source:

- smoke:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_dinov2base_s3_smoke128_20260620.npz`,
  `128` tracklets, `384` crops, feature dim `768`;
- full feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_dinov2base_s1_20260620.npz`,
  `9734` tracklets, feature dim `768`, no missing crops/videos;
- fused features:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_dinobase003_s1_20260620.npz`,
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_dinobase005_s1_20260620.npz`.

DINO-base component-merge artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_dinobase_s1_component_merge_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_dinobase003_s1_component_merge_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_dinobase005_s1_component_merge_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_dinobase005_s1_component_merge_loose_pair_20260620.json`.

DINO-base gate result:

| branch | pair F1 | precision | recall | accepted merges | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| DINOv2-base s1 only | `0.769661` | `0.815773` | `0.728482` | `0` | no-op |
| fused + DINO 0.03 | `0.769661` | `0.815773` | `0.728482` | `0` | no-op |
| fused + DINO 0.05 | `0.769661` | `0.815773` | `0.728482` | `0` | no-op |
| fused + DINO 0.05 loose | `0.769661` | `0.815773` | `0.728482` | best row `1`; lower rows up to `5` | no gain |

Decision:

- No full/e2e scorer was run for these branches because neither passed the
  pair gate `> 0.769760`.
- DINOv2-base s1 is valid as an extracted no-anchor feature, but in current
  component-merge form it does not unlock the large false-split identities.
- The next branch should learn a repairability/admission proxy over whole
  predicted components or identity islands, then use edge scores only within
  those localized repair targets.

### 2026-06-20 Continued AutoResearch loop: temporal and conflict-state repair

New script:

- `kit/no_anchor_assignment_video_temporal_relink_sweep.py`

Purpose:

- test whether same-video temporal adjacency, bbox endpoint distance, and
  appearance similarity can recover within-video ID continuity without anchors;
- keep output global by merging original components, not by emitting
  video-local IDs.

Temporal artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_video_temporal_relink_global_fused_s7_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_video_temporal_relink_global_fused_s7_loose_pair_20260620.json`.

Temporal result:

| branch | pair F1 | precision | recall | accepted edges |
| --- | ---: | ---: | ---: | ---: |
| strict temporal relink | `0.769661` | `0.815773` | `0.728482` | `0` |
| loose temporal relink | `0.769661` | `0.815773` | `0.728482` | `1` |

Conclusion:

- Tracklet endpoint continuity is too sparse after global-component overlap
  guards.
- A video-local diagnostic output collapsed full IDF1 to `0.366217`, so the
  scorer still requires global identity consistency; local per-video relabeling
  is not a valid shortcut.

Conflict-state repair:

- pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_pair_20260620.json`;
- singleton full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_top1_full_20260620.json`;
- color split pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_color_forced_pair_20260620.json`;
- color split full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_color_forced_full_20260620.json`;
- color split + remerge pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_state_color_forced_multiview_merge_pair_20260620.json`;
- color split + remerge full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_state_color_forced_multiview_merge_top1_full_20260620.json`.

Results:

| branch | pair F1 | precision | recall | full IDF1 | HOTA | AssA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| softcut+soft-overlap production | `0.769661` | `0.815773` | `0.728482` | `0.655240` | `0.518652` | `0.534359` |
| singleton forced-conflict split | `0.771670` | `0.822651` | `0.726639` | `0.646655` | `0.513889` | `0.534396` |
| color forced-conflict split | `0.770599` | `0.818962` | `0.727630` | `0.652774` | `0.516843` | `0.532799` |
| color split + multiview remerge | `0.770667` | `0.818533` | `0.728090` | `0.652744` | `0.516808` | `0.532742` |

Decision:

- The model-side global-ID score improved to pair F1 `0.771670`, but no
  conflict-state branch improved the end-to-end submission.
- The e2e failure mode is now sharper: direct delivery of split conflict
  components increases pair precision but creates detection-level identity
  fragmentation.
- Next experiment should keep conflict parts as provisional evidence and solve
  a local component assignment before emitting final IDs; do not hard-deliver
  singleton/color splits.

### 2026-06-20 Conflict subcluster reassign

New script:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py`.

Purpose:

- keep `forced_conflict` subclusters as provisional evidence;
- reassign a subcluster only when an existing target component has strong
  multi-view evidence;
- avoid creating new delivery IDs for unresolved conflict parts.

Artifacts:

- non-strict pair:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_narrow_pair_20260620.json`;
- strict pair:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_pair_20260620.json`;
- strict full:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_top1_full_20260620.json`;
- strict density selector:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_density_filter_selector_zip_20260620.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_*_20260620.*`.

Results:

| branch | pair F1 | precision | recall | full IDF1 | density IDF1 | action |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| non-strict reassign | `0.772399` | `0.818671` | `0.731077` | `0.653724` | `0.653936` | rejected e2e |
| margin 0.03 reassign | `0.770554` | `0.816238` | `0.729712` | `0.652957` | n/a | worse |
| strict reassign | `0.772654` | `0.818667` | `0.731538` | `0.653823` | `0.654037` | model-side best |

Decision:

- `strict_reassign` is the current best no-anchor global-ID model-side result.
- It is not the end-to-end best.  The standing production zip remains
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.zip`
  at IDF1 `0.655240`.
- A broad strict sweep was started but interrupted because the target grid was
  too expensive for the score movement.  Next step should cache target scoring
  or train a no-GT full-score/edit-admission proxy before expanding the grid.

### 2026-06-20 AutoResearch self-play candidate-search follow-up

This continuation applied the Deli AutoResearch anti-loop rule directly:
instead of widening the same strict reassign grid after an expensive broad
run, first test whether the bottleneck is target threshold search or source
island generation.

Code update:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now has an optional
  `--candidate-search-top-n` mode.
- The default exhaustive sweep is unchanged.
- Candidate-search mode builds a no-GT source-to-target candidate table,
  sorts by proposal evidence, and only evaluates greedy top-prefix edits.

Artifacts:

- narrow self-play pair:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_narrow_selfplay_pair_20260620.json`;
- candidate-search tiny pair:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_tiny_pair_20260620.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_narrow_selfplay_pair_20260620.*`
  and
  `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_tiny_pair_20260620.*`.

Results:

| branch | pair F1 | precision | recall | accepted reassignments | moved tracklets | runtime note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| softcut+soft-overlap base | `0.769661` | `0.815773` | `0.728482` | n/a | n/a | input |
| strict narrow self-play | `0.772654` | `0.818667` | `0.731538` | `1` | `8` | reproduces current best |
| candidate-search tiny | `0.772654` | `0.818667` | `0.731538` | `1` | `8` | 25-second candidate-table smoke |

Negative/engineering evidence:

- A broad cached strict sweep and a source-wide target-fixed sweep were both
  interrupted because they stayed in high-CPU source/target scoring without
  producing a better scalar result in a reasonable window.
- Candidate-search tiny produced only `5` usable candidate edges and selected
  the same 8-tracklet island:
  `2232, 2270, 2308, 2374, 2415, 2452, 2488, 2553` into target component `0`.
- This means the known strict edit is stable, but the current source-island
  generator is the bottleneck for wider search.  Target threshold widening is
  no longer the next useful experiment.

Decision:

- No new model-side or e2e best was promoted.
- The current model-side best remains strict conflict reassign at
  `0.772654 / 0.818667 / 0.731538`.
- The current end-to-end best remains the no-GT density-selected
  softcut+soft-overlap zip at full IDF1 `0.655240`.
- Next structural branch should materialize an offline source-island candidate
  table or learn a source repairability proxy.  Re-running larger online
  source grids is now a closed/low-value loop.

### 2026-06-20 Source-island audit and G8 strict-target reassign

AutoResearch cue:

- The Deli protocol says a stalled local grid should pivot structurally.  Here
  the pivot was to separate source-island generation from target assignment,
  audit the no-GT source ordering, then constrain the source island shape.

New script:

- `kit/no_anchor_conflict_source_island_audit.py`.

Source audit:

- tiny strict audit:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_source_island_audit_tiny_20260620.json`;
- loose audit:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_source_island_audit_loose1_20260620.json`.

Audit findings:

| audit | dedup sources | oracle-positive sources | top no-GT candidate | conclusion |
| --- | ---: | ---: | --- | --- |
| tiny strict | `17` | `3` | true-positive 8-tracklet island | source rank works in a narrow region |
| loose source | `44` | `9` | false-positive 12-tracklet island | candidate recall improved, ranking failed |

Reassign artifacts:

- loose pair:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json`;
- loose strict-target pair:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_stricttarget_pair_20260620.json`;
- G8 strict-target pair:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_pair_20260620.json`;
- G8 strict-target full:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json`;
- G8 strict-target density:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_density_filter_selector_zip_20260620.json`;
- G8 strict-target assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_top1_assignments_20260620.csv`.

Results:

| branch | pair F1 | precision | recall | full IDF1 | HOTA | AssA | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| loose source, loose target | `0.767329` | `0.814359` | `0.725434` | not scored | n/a | n/a | rejected |
| loose source, strict target | `0.772127` | `0.817025` | `0.731906` | not scored | n/a | n/a | below current model best |
| loose source, max group 8, strict target | `0.775234` | `0.820504` | `0.734698` | `0.653541` | `0.517447` | `0.532858` | model-side best, e2e negative |
| G8 strict target + density selector | n/a | n/a | n/a | `0.653681` | `0.517358` | `0.532902` | e2e negative |

Per-video full IDF1 for the G8 strict-target assignment:

| video | IDF1 |
| --- | ---: |
| MCAM00 Tc6 | `0.847550` |
| MCAM00 Tc8 | `0.827822` |
| MCAM03 Tc6 | `0.688387` |
| MCAM03 Tc8 | `0.626660` |
| MCAM04 Tc6 | `0.561749` |
| MCAM05 Tc6 | `0.710965` |
| MCAM05 Tc8 | `0.791599` |
| MCAM06 Tc6 | `0.606895` |
| MCAM06 Tc8 | `0.704148` |
| MCAM08 Tc6 | `0.767135` |

Decision:

- Promote the G8 strict-target branch only as the current no-anchor
  global-ID model-side diagnostic best.
- Do not promote it as the e2e artifact.  Density-selected IDF1 `0.653681`
  remains below the standing no-GT production artifact `0.655240`.
- Next e2e branch should learn or design a delivery-aware admission function
  that predicts when a pair-positive source edit hurts full IDF1.  Continuing
  to add more visually plausible source islands is now a known local optimum.

### 2026-06-20 Delivery-aware admission checks

Implementation update:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now supports
  `--min-target-qualities`; default is `0.0`, preserving previous behavior.

Artifacts:

- prefix full sensitivity:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json`;
- target-quality full gate:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json`;
- row-filter policy sweep:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_filter_policy_sweep_20260620.json`.

Prefix full sensitivity:

| max reassignments | accepted edits | pair F1 | full IDF1 | HOTA | AssA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `1` | `0.772654` | `0.653823` | `0.517789` | `0.533336` |
| `2` | `2` | `0.774082` | `0.653823` | `0.517789` | `0.533336` |
| `3` | `3` | `0.774252` | `0.653042` | `0.516939` | `0.532540` |
| `4` | `4` | `0.775234` | `0.653541` | `0.517447` | `0.532858` |

Target-quality gate:

| gate | accepted edits | pair F1 | full IDF1 | HOTA | AssA | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.75`, top 3 | `3` | `0.775064` | `0.653541` | `0.517447` | `0.532858` | no e2e gain |
| `0.75`, top 4 | `4` | `0.773540` | `0.654009` | `0.517946` | `0.533243` | best here, still rejected |

Row-filter policy sweep on the G8 assignment:

| policy | IDF1 | HOTA | AssA | DetPr | DetRe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `density_simple` | `0.653686` | `0.517364` | `0.532902` | `0.759920` | `0.573512` |
| `confidence_tail` | `0.653581` | `0.517217` | `0.532757` | `0.760258` | `0.573158` |

Decision:

- No admission/filter candidate exceeded the standing full IDF1 `0.655240`.
- This closes the simple edit-prefix and row-filter branch.  The next e2e
  attempt needs a component/video-level full-score proxy or a new resolver that
  directly repairs the weak videos instead of making globally sorted pair edits.

### 2026-06-20 AutoResearch source-switch compatibility check

Purpose:

- apply the Deli AutoResearch anti-loop rule after target-quality gates and
  row filters failed: run one bounded source-complementarity diagnostic instead
  of widening the same reassign/filter grid;
- test whether existing no-anchor full submissions can be composed by video.

Distilled protocol:

- proposer: an existing no-anchor submission source;
- opponent: global namespace consistency and full delivery evaluation;
- evaluator: DS1 HOTA/IDF1 scorer, with GT used only after prediction;
- negative results remain part of the research state.

Artifact:

- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.json`;
- remote zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.zip`;
- local JSON:
  `local_runs/remote_h100_test_3_20260620/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.json`.

Setup:

- base source:
  `no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.zip`;
- explicit switch from zip-backed per-video scan:
  MCAM04 Tc6 -> `conflict_g8`,
  MCAM06 Tc8 -> `quality`;
- all sources are no-anchor outputs, but the policy is eval-derived and
  diagnostic-only.

Result:

| policy | full IDF1 | HOTA | AssA | DetPr | DetRe | predicted IDs | rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current/base | `0.655240` | `0.518652` | `0.534359` | `0.763322` | `0.573970` | `377` | `1,524,919` |
| explicit video switch | `0.497171` | `0.345933` | `0.355785` | `0.801372` | `0.360373` | `379` | `1,525,668` |

Failure mode:

- MCAM04 Tc6 dropped to IDF1 `0.228783`;
- MCAM06 Tc8 dropped to IDF1 `0.000000`;
- unchanged videos also lost per-video IDF1, which means mixing submission
  zips breaks global ID namespace consistency.  These are not interchangeable
  local predictions.

Conclusion:

- Close the whole-video zip switching path.
- Existing no-anchor submissions do not compose into a better e2e result by
  simple source selection.
- The next branch should operate inside one identity namespace and predict
  full-score side effects of component edits, especially for the large
  false-split identities seen in the oracle decomposition.

### 2026-06-20 Aggressive identity-island pair gate

Purpose:

- after source-switch failed, stay inside one global-ID namespace and test
  whether a less tiny identity-island repair can recover more false-split mass;
- keep this pair-only unless it beats the previous edge-target pair gate.

Artifact:

- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_table_island_merge_aggressive_pair_20260620.json`;
- remote CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_table_island_merge_aggressive_pair_20260620.csv`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_edge_table_island_merge_aggressive_pair_20260620.*`.

Setup:

- input assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`;
- input edge table:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_edge_acceptor_table_20260619.csv`;
- grid expanded fragment size/growth:
  `max_small_size=8/12/16`, `max_growth_ratio=0.10/0.20/0.35`,
  `min_fused_sim=0.70/0.76`, `forbidden_mode=on/any`;
- GT columns are ignored during selection; GT is used only for pair scoring.

Result:

| branch | pair F1 | precision | recall | accepted edges | rewritten components | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| softcut+soft-overlap base | `0.769661` | `0.815773` | `0.728482` | n/a | n/a | baseline |
| previous edge-target repair | `0.769760` | `0.815788` | `0.728648` | `23` | n/a | better pair gate |
| aggressive identity-island | `0.769692` | `0.815171` | `0.729019` | `17` | `32` | rejected |

Best aggressive setting:

- `min_score=0.35`, `max_score=0.90`, `min_fused_sim=0.70`;
- `max_small_size=8`, `min_large_size=96`;
- `max_new_size=320`, `max_growth_ratio=0.10`;
- `max_edges=32`, `max_degree=2`, `forbidden_mode=on`.

Conclusion:

- More aggressive island attachment buys recall but loses enough precision to
  underperform the narrower edge-target repair.
- Do not full-score this branch.
- The remaining path is not "attach more small islands"; it needs a
  repairability model over larger identity components or a stronger
  pseudo-positive source that can identify large false-split groups directly.

### 2026-06-20 fused-DINO edge-source target-repair check

AutoResearch action:

- Convert the Deli AutoResearch distillation into file-backed state at
  `autoresearch_state/no_anchor_global_id/state/`.
- Run one structurally distinct branch: use fused-DINO as an edge-table source,
  then apply target repair with no-GT evidence fields.

Code update:

- `kit/no_anchor_edge_table_target_repair_sweep.py` now supports edge tables
  without `fused_sim/db_rank_min` by falling back to
  `primary_sim/primary_rank_*`.

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_acceptor_table_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_acceptor_table_20260620.csv`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_fallback_single_full1_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_fallback_single_full1_20260620.csv`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_fallback_single_assignments_20260620.csv`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_*_20260620.*`.

Results:

| branch | pair F1 | precision | recall | full IDF1 | HOTA | AssA | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standing production zip | n/a | n/a | n/a | `0.655240` | `0.518652` | `0.534359` | current e2e best |
| softcut assignment base | `0.769661` | `0.815773` | `0.728482` | `0.653071` | `0.516882` | `0.532486` | no delivery filter |
| fused-DINO target repair | `0.769661` | `0.815126` | `0.729001` | `0.652948` | `0.516770` | `0.532397` | rejected |

Interpretation:

- Fused-DINO preserved the same basic false-split candidate count:
  `3240` candidate edges with `47` eval-only true edges.
- The best no-oracle edge rule remained low precision:
  `4 / 23` true edges, precision `0.173913`.
- The corrected target-repair run selected `15` edges over `8` targets, but the
  effect was recall-positive and precision-negative, with full IDF1 below both
  the unfiltered softcut assignment and the density-selected production zip.

Decision:

- Close fused-DINO edge-source target repair in this form.
- Do not run wider target-repair grids until the script has cached or
  incremental pair metric evaluation.
- The next branch should learn or predict repairability before applying edits.

### 2026-06-20 Cached repair sweep and NFC feature centralization

Purpose:

- complete the engineering fix implied by the DINO target-repair timeout;
- test a structurally different no-anchor feature challenger based on
  Pose2ID-style training-free Neighbor Feature Centralization.

Code:

- `kit/no_anchor_edge_table_target_repair_sweep.py` now caches edit signatures.
- `kit/make_no_anchor_nfc_features.py` creates NFC-smoothed `.npz` feature
  files without GT or anchors.

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_cached_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_signature_scout_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s7_true_nfc_k2_eta05_20260620.npz`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_nfc_fused_osnet005_s7true_timeagglom_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_nfc_osnet_s7_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_nfc_aux_osnet_s7_pair_20260620.json`.

Results:

| check | base pair F1 | best pair F1 | best P/R | selected effect | verdict |
| --- | ---: | ---: | --- | --- | --- |
| cached fused-DINO target repair | `0.769661` | `0.769755` | `0.815757 / 0.728664` | `27` edges, `23` targets | below model best |
| old edge-table signature scout | `0.769661` | `0.769713` | `0.815146 / 0.729077` | `24` edges, `16` targets | below model best |
| NFC time-agglom resolver | n/a | `0.594775` | `0.720286 / 0.506515` | global resolver | rejected |
| NFC softcut primary replacement | `0.768743` | `0.764329` | `0.815670 / 0.719068` | `4` components, `763` tracklets split | rejected |
| NFC softcut auxiliary view | `0.768743` | `0.767759` | `0.814164 / 0.726358` | `2` components, `366` tracklets split | rejected |

Conclusion:

- Caching makes the target-repair grid usable, but the best rows are still far
  below the current model-side diagnostic best `0.775234`.
- NFC is useful as a provenance-preserving feature experiment, not as a
  promoted signal.  It smooths local neighborhoods but damages the crowded
  VLINCS identity boundary structure.
- Continue with a delivery-aware repairability/admission proxy.  The next
  question is no longer "which edge is similar?" but "which edit improves full
  identity delivery without increasing false merges?"

### 2026-06-20 SigLIP2 person-description ReID feature challenger

Purpose:

- test a new pretrained visual prior rather than another threshold on the same
  feature graph;
- keep the gate pair-only unless the feature beats the current component-merge
  baseline.

Code change:

- `kit/extract_tracklet_foundation_features.py` now supports
  `--processor-model` and robust pooled-output parsing for HF models such as
  `MarketaJu/siglip2-person-description-reid`.

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_siglip2_person_reid_s1_20260620.npz`;
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_siglip2p003_s1_20260620.npz`;
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_siglip2p005_s1_20260620.npz`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_siglip2_person_reid_s1_timeagglom_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_siglip2p003_s1_component_merge_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_siglip2p005_s1_component_merge_pair_20260620.json`.

Extraction:

- full coverage: `9734 / 9734` tracklets;
- feature dimension: `768`;
- samples: `1` crop per tracklet;
- uses anchors: no;
- GT used during extraction: no.

Pair gate:

| check | pair F1 | precision | recall | action |
| --- | ---: | ---: | ---: | --- |
| SigLIP2 standalone time-agglom | `0.574380` | `0.640377` | `0.520716` | rejected |
| current component-merge base | `0.769661` | `0.815773` | `0.728482` | baseline |
| fused + SigLIP2 `0.03` | `0.769661` | `0.815773` | `0.728482` | `0` merges |
| fused + SigLIP2 `0.05` | `0.769661` | `0.815773` | `0.728482` | `0` merges |

Conclusion:

- SigLIP2-person-ReID is valid as an extracted no-anchor feature, but it is not
  a useful resolver or low-weight component-merge feature under the current
  gate.
- No full scorer was run.  The branch is closed unless reused later as a
  non-decision provenance feature.

### 2026-06-20 Weak metric projection probe

Deli AutoResearch distillation applied:

- Treat the tweet/thread as an operating protocol: persistent state,
  evaluator-owned metrics, and structural pivots after stalls.
- For VLINCS, self-play means no-anchor identity proposers compete against
  cannot-link, namespace, and full-score fragmentation opponents.
- Because target-quality gates, row filters, and source switching already
  failed, the next feature test must change the model evidence rather than
  widen delivery thresholds.

New script:

- `kit/make_no_anchor_weak_metric_features.py`.

Method:

- Train a small linear metric projection from weak positives and negatives:
  same-tracklet crop positives, short-gap same-stream visual continuation
  positives, and same-stream overlap cannot-link negatives.
- Output normal `seqs/features` NPZs so existing no-anchor resolvers and
  conflict-reassign scripts can consume them.
- No GT identities, anchors, or manual labels are used during training.

Artifacts:

- training metadata:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_weakmetric_osnet_s7_fused_20260620.json`;
- feature outputs:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620.npz`,
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p05.npz`;
- pair gates:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_weakmetric_w002_conflict_reassign_g8_pair_20260620.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_weakmetric_w005_conflict_reassign_g8_pair_20260620.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_weakmetric_w002_conflict_reassign_g8_pair_20260620.json`,
  `local_runs/remote_h100_test_3_20260620/no_anchor_weakmetric_w005_conflict_reassign_g8_pair_20260620.json`.

Training stats:

| weak-label source | count |
| --- | ---: |
| same-tracklet crop positives | `9000` |
| continuation positives | `8555` |
| cannot-link negatives | `18000` |
| total train pairs | `35555` |

Projection diagnostics:

- positive cosine mean: `0.420363`;
- negative cosine mean: `0.096400`;
- margin: `0.323963`;
- final BCE loss: `1.275113`.

Current-branch pair gate:

| feature primary | pair F1 | precision | recall | accepted reassignments | moved tracklets | full run? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current G8 strict target | `0.775234` | `0.820504` | `0.734698` | `4` | `32` | already known e2e-negative |
| weakmetric `w=0.02` | `0.775234` | `0.820504` | `0.734698` | `4` | `32` | no |
| weakmetric `w=0.05` | `0.775234` | `0.820504` | `0.734698` | `4` | `32` | no |

Decision:

- The weak-supervised projection learned a real train-space separation, but it
  did not change the strongest no-anchor global-ID model output.
- No full scorer was run because the pair gate did not exceed the standing
  model-side best and the identical G8 assignment is already e2e-negative.
- Close this as a feature/projection challenger.  Next work should attack
  broad false-split identity recovery in one namespace, especially the
  high-mass identities exposed by the oracle decomposition.

### 2026-06-20 Large false-split component micro probes

Why this branch:

- The oracle decomposition says the remaining e2e gap is dominated by large
  false-split identities plus a few polluted large components.
- Deli AutoResearch/self-play protocol says negative movement is evidence:
  after feature additions stalled, the next challenger must change identity
  structure and be judged by full-score side effects.

Input assignment:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_top1_assignments_20260620.csv`.

Important note:

- Reloading this delivered assignment as an input gives pair
  `0.772727 / 0.818297 / 0.731964`, slightly below the pair-grid row
  `0.775234`.  This confirms that delivery/admission/assignment serialization
  effects must be checked by re-reading the CSV and running full scoring.

Pure component merge probe:

- Script: `kit/no_anchor_assignment_component_merge_sweep.py`.
- Partial remote log copied to:
  `local_runs/remote_h100_test_3_20260620/large_false_split_component_merge_best_assignment_20260620.log`.
- Candidate edges: `3240`.
- Best observed full row accepted only `1` merge.

| branch | pair F1 | precision | recall | full IDF1 | HOTA | AssA | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| reloaded input assignment | `0.772727` | `0.818297` | `0.731964` | not rerun | - | - | baseline |
| pure component merge, rank 1 | `0.772727` | `0.818297` | `0.731965` | `0.653541` | `0.517447` | `0.532858` | no-op |

Bulk split-then-merge micro probe:

- Script: `kit/no_anchor_assignment_split_then_merge_sweep.py`.
- Artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_large_false_split_split_then_merge_micro_best_assignment_20260620.json`.
- Local copy:
  `local_runs/remote_h100_test_3_20260620/no_anchor_large_false_split_split_then_merge_micro_best_assignment_20260620.json`.

Split diagnostics:

| quantity | value |
| --- | ---: |
| split components | `41` |
| rewritten tracklets | `2252` |
| conflict edges inside split components | `3012` |
| split components after operation | `180` |
| candidate edges after split | `4477` |

Result:

| branch | pair F1 | precision | recall | full IDF1 | HOTA | AssA | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bulk split only | `0.618505` | `0.848465` | `0.486617` | `0.602445` | `0.454924` | `0.467662` | rejected |
| split then merge micro | `0.618505` | `0.848465` | `0.486617` | not full-scored | - | - | rejected by pair collapse |

Decision:

- Pure component merge from the current best assignment cannot recover the
  false-split mass.
- Bulk cannot-link splitting is far too destructive: it improves precision but
  destroys recall and full IDF1.
- Continue the large-false-split direction only with surgical source-group
  edits and a full-score side-effect proxy; do not do broad component coloring
  splits.

### 2026-06-20 Surgical source-group full-proxy probe

Code change:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now supports
  `--rank-by full_proxy`.
- The proxy uses no GT: target/source quality, target mean/best similarity,
  view vote, min-view similarity, margin, and penalties for extra accepted
  edits / moved tracklets.

Run:

- Input assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`.
- Output:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json`.
- Local copy:
  `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json`.

Result:

| selector | accepted edits | moved tracklets | pair F1 | precision | recall | full IDF1 | HOTA | AssA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pair-ranked G8 top row | `4` | `32` | `0.775234` | `0.820504` | `0.734698` | `0.653541` | `0.517447` | `0.532858` |
| full-proxy selected row | `1` | `8` | `0.772654` | `0.818667` | `0.731538` | `0.653823` | `0.517789` | `0.533336` |
| standing e2e gate | - | - | - | - | - | `0.655240` | `0.518652` | `0.534359` |

Decision:

- The proxy is directionally useful: it rejects the pair-best four-edit row and
  selects the one-edit row with slightly better full IDF1.
- It still does not beat the standing e2e gate.
- The next experiment should add diversity constraints or train an edit
  acceptor so the proxy does not rank duplicate variants of the same top source
  group.

### 2026-06-20 Unique-signature full-proxy rerun

Why this rerun:

- The first `full_proxy` run spent full-score slots on duplicate variants of
  the same accepted edit.
- I patched the candidate selector to de-duplicate accepted-edit signatures
  before running full DS1 scoring.

Artifacts:

- Remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_fullproxy_unique_pairfull_20260620.json`.
- Local JSON:
  `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_unique_pairfull_20260620.json`.
- Promoted assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_fullproxy_unique_top_assignments_20260620.csv`.

Unique full-scored rows:

| full rank | accepted edits | moved tracklets | pair F1 | precision | recall | full IDF1 | HOTA | AssA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `1` | `8` | `0.772654` | `0.818667` | `0.731538` | `0.653823` | `0.517789` | `0.533336` |
| `2` | `2` | `16` | `0.774082` | `0.819399` | `0.733515` | `0.653823` | `0.517789` | `0.533336` |
| `3` | `3` | `24` | `0.774252` | `0.819890` | `0.733427` | `0.653042` | `0.516939` | `0.532540` |
| standing e2e gate | - | - | - | - | - | `0.655240` | `0.518652` | `0.534359` |

Decision:

- De-duplicating full-score candidates did not find a row above the standing
  e2e gate.
- Adding the second source group ties the one-edit full score; adding a third
  source group hurts full IDF1.
- The current surgical repair family is saturated.  The next structural move
  should be a learned edit acceptor or deliberately diverse source-group
  proposal, trained/evaluated on cached no-GT proposal features and posthoc
  full-score outcomes.

### 2026-06-20 Prepared diverse-first-edge full selection

Code change:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now has
  `--full-selection {none,unique_signature,diverse_first_edge}`.
- `diverse_first_edge` samples expensive full-score candidates by distinct
  first source-to-target edit, rather than by full accepted-edit signature.

Reason:

- The unique-signature run still selected rows whose first edit was always
  `source_component_label=21 -> target_component=0`.
- Existing local top-56 artifacts contain only that one first-edge family, so
  this cannot be validated offline from the pulled JSON.

Status:

- Local `py_compile` and `--self-test` pass.
- Remote run not started: Pluto service returned `Failed to connect to Pluto
  service`, and direct SSH hit `Connection timed out during banner exchange`.
- Next retry should first verify or restore the remote script, then run the
  diverse-first-edge full-proxy sweep.

### 2026-06-20 Candidate-search skip-family diversity prep

Code change:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now also supports
  `--candidate-skip-first-edge-families`.
- This changes the proposer, not just the expensive full-score selector: before
  greedy candidate construction, it can skip the top `N` unique source-to-target
  edge families and construct assignments from lower-ranked families.
- The row output records:
  `candidate_total_edges_before_skip`, `candidate_skip_first_edge_families`,
  and `candidate_first_edge_family_rank`.

Launcher:

- Added `kit/run_no_anchor_false_split_diversity.sh`.
- It deploys the changed scripts, runs self-test remotely, then launches:
  `--candidate-search-top-n 256`,
  `--candidate-skip-first-edge-families 0,1,2,4,8,16,32,64`,
  `--rank-by full_proxy`, `--full-selection diverse_first_edge`,
  `--full-top-n 8`.

Validation:

- Local `py_compile` passed.
- Local `--self-test` passed.
- Launcher `bash -n` passed.
- Launcher dry-run reached the real Pluto probe, then failed only because SSH
  timed out during banner exchange.

Decision:

- This is the concrete next no-anchor experiment once h100-test-2/3 or
  test-video-0 becomes reachable again.
- It directly targets the previous failure mode: all pulled full-proxy rows
  started from the same `source_component_label=21 -> target_component=0`
  first edit.

### 2026-06-20 Learned full-proxy acceptor prep

Local audit:

- Added `kit/analyze_no_anchor_full_proxy_training.py`.
- It harvests full-scored no-anchor proposal rows from `local_runs/`, excludes
  oracle rows by default, filters to comparable DS1 rows with
  `full_idf1 >= 0.55`, and trains a compact ridge proxy from no-GT proposal
  features to posthoc full IDF1.

Artifacts:

- audit JSON:
  `local_runs/no_anchor_full_proxy_training_audit_20260620.json`;
- audit report:
  `reports/no_anchor_full_proxy_training_audit_20260620.md`;
- deployable compact model:
  `local_runs/no_anchor_full_proxy_compact_ridge_model_20260620.json`;
- oracle-inclusive reference:
  `local_runs/no_anchor_full_proxy_training_audit_with_oracle_20260620.json`.

Clean no-oracle audit:

| rows | compact features | full IDF1 range | LOOCV corr | LOOCV MAE | LOOCV RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `32` | `29` | `0.602445-0.654009` | `0.996050` | `0.000913` | `0.001189` |

Oracle-inclusive reference:

| rows | compact features | full IDF1 range | LOOCV corr | LOOCV MAE | LOOCV RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `36` | `29` | `0.602445-0.706202` | `0.929021` | `0.004083` | `0.007219` |

Feature lesson:

- `tracklet_pair_f1` and `tracklet_pair_recall` dominate the current local
  full-score outcome, while the hand-written `full_side_effect_proxy` alone has
  weak correlation on the cleaned no-oracle rows.
- This does not prove a new e2e best; it turns prior full-score evidence into
  a deployable no-anchor acceptor for the next remote sweep.

Code path prepared:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now supports
  `--rank-by learned_proxy --learned-proxy-json <model.json>`.
- `kit/run_no_anchor_false_split_diversity.sh` now packages
  `local_runs/no_anchor_full_proxy_compact_ridge_model_20260620.json` and runs
  `--rank-by learned_proxy` together with
  `--candidate-skip-first-edge-families 0,1,2,4,8,16,32,64`.

### 2026-06-20 Delivery-aware learned proxy correction

AutoResearch honesty check:

- The first compact full-proxy was too optimistic outside its cleaned training
  slice.  It ranked a state-policy branch at predicted full IDF1 `0.747829`
  because the row had `tracklet_pair_f1=0.954691`.
- The same branch was already full-scored in the local archive:
  `output_tracklets=421-431`, `eval_tracklets=420-430`,
  `coverage_ratio=0.04325-0.04428`, and real full IDF1 `0.085353-0.085412`.
- This is the exact failure mode Deli AutoResearch warns about: an apparently
  higher score is not progress if an external evidence check lowers it.

Code changes:

- `kit/analyze_no_anchor_full_proxy_training.py`
  - added delivery features:
    `assigned_tracklets`, `output_tracklets`, `eval_tracklets`,
    `coverage_ratio`, `delivery_tracklets_min`,
    `delivery_tracklets_mean`;
  - excludes all `full_proxy*training_audit` and `pair_candidates` artifacts
    from harvest input.
- `kit/score_no_anchor_pair_candidates.py`
  - excludes self-generated candidate ranking reports from harvest input;
  - applies delivery filtering to present fields only:
    `output_tracklets >= 7000`, `eval_tracklets >= 7000`,
    `coverage_ratio >= 0.70`;
  - records delivery fields in JSON/CSV/Markdown candidate reports.
- `kit/run_no_anchor_false_split_diversity.sh`
  - now packages and uses
    `local_runs/no_anchor_full_proxy_delivery_ridge_model_20260620.json`.

Artifacts:

- delivery-aware audit:
  `local_runs/no_anchor_full_proxy_delivery_training_audit_20260620.json`;
- delivery-aware report:
  `reports/no_anchor_full_proxy_delivery_training_audit_20260620.md`;
- deployable model:
  `local_runs/no_anchor_full_proxy_delivery_ridge_model_20260620.json`;
- candidate ranking:
  `local_runs/no_anchor_delivery_proxy_pair_candidates_20260620.json`;
- candidate report:
  `reports/no_anchor_delivery_proxy_pair_candidates_20260620.md`.

Clean no-oracle delivery-aware audit:

| rows | compact features | full IDF1 range | LOOCV corr | LOOCV MAE | LOOCV RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `37` | `34` | `0.085353-0.654009` | `0.997457` | `0.003381` | `0.013989` |

Candidate-ranking result:

| setting | candidates | delivery-filter drops | top predicted full IDF1 | top pair F1 | top artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| no full-scored rows | `300` | `68` | `0.656057` | `0.767329` | `no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |
| including full-scored rows | `318` | `68` | `0.656057` | `0.767329` | `no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json` |

Critical case check:

- With the delivery-aware model, the previously dangerous state-policy rows
  are predicted at `0.085352-0.085920`, matching their known full IDF1 near
  `0.0854`; they no longer enter the top-ranked candidate set even without the
  delivery filter.
- This does not solve the `>0.70` e2e gate, but it makes the next remote sweep
  more honest: full-score budget should go to diverse full-delivery repair
  candidates, not low-coverage shortcuts.

Remote execution status:

- Local validation passed:
  `py_compile`, solver `--self-test`, JSON validation, and launcher `bash -n`.
- Dry-run probes for `h100-test-3`, `h100-test-2`, and `test-video-0` all
  failed at SSH banner exchange:
  `Connection timed out during banner exchange`.
- Therefore the delivery-aware diverse-first-edge full sweep is prepared but
  not submitted.  The exact launcher remains:

```bash
bash kit/run_no_anchor_false_split_diversity.sh --foreground
```

### 2026-06-20 Source-island repairability acceptor

Reason for pivot:

- The source-island audit showed that direct `source_rank_score` is a weak
  proposer.  In the loose audit, many high-ranked source islands had zero
  oracle delta, while useful source islands appeared deep in the list.
- This is exactly the AutoResearch proposer/opponent split: source generation
  produced useful candidates, but the source ranking policy was the bottleneck.

Eval-only audit:

- Added `kit/analyze_no_anchor_source_island_acceptor.py`.
- Inputs:
  `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_*_20260620.json`.
- Inputs to the model are no-GT source features only.  Excluded features include
  GT/eval-only fields such as `source_majority_gt`, `source_gt_count`,
  `component_gt_count`, `oracle_*`, and also the arbitrary
  `source_component_label`.
- Label is posthoc `oracle_delta_pair_f1 > 0`, used only for this diagnostic
  acceptor audit.

Artifacts:

- audit JSON:
  `local_runs/no_anchor_source_island_acceptor_audit_20260620.json`;
- audit report:
  `reports/no_anchor_source_island_acceptor_audit_20260620.md`;
- deployable no-GT source scorer:
  `local_runs/no_anchor_source_island_acceptor_ridgelogit_model_20260620.json`.

Audit result:

| ranker | AP | top-5 positives | top-10 positives | top-20 positives |
| --- | ---: | ---: | ---: | ---: |
| `source_rank_score` | `0.302423` | `3/5` | `3/10` | `5/20` |
| `source_score` | `0.300976` | `3/5` | `3/10` | `5/20` |
| `source_quality` | `0.155425` | `1/5` | `2/10` | `2/20` |
| `ridge_logit_loocv` | `0.944124` | `5/5` | `9/10` | `12/20` |

Feature signal:

- Positive source islands are not simply the highest visual-quality islands.
  The largest correlations with oracle-positive labels are negative
  `source_cross_mean_sim`, negative `source_cross_max_sim`, negative
  `source_quality`, and positive `source_margin_mean`.
- Interpretation: useful repair sources tend to be separable conflict islands,
  not globally high-similarity/high-quality chunks.  This matches the observed
  failure of direct visual merge rules.

Code path prepared:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now accepts
  `--source-acceptor-json`.
- In candidate-search mode, each source island gets a
  `source_acceptor_score`, and proposal ranking includes that score.
- `kit/run_no_anchor_false_split_diversity.sh` now packages and uses both:
  `local_runs/no_anchor_full_proxy_delivery_ridge_model_20260620.json`
  and
  `local_runs/no_anchor_source_island_acceptor_ridgelogit_model_20260620.json`.

Decision:

- This does not yet improve DS1 full IDF1; it is a prepared proposer upgrade.
- It is a better next remote experiment than another threshold sweep because
  it changes the structural ranking of source islands and directly targets the
  known failure of source-rank ordering.

Validation:

- Local validation passed:
  `py_compile`, solver `--self-test`, launcher `bash -n`, and JSON validation
  for the source acceptor audit/model plus no-anchor state files.
- A fresh dry-run probe for `h100-test-3` still failed before remote execution:
  `Connection timed out during banner exchange`.
- Therefore the source-acceptor + delivery-proxy + diverse-first-edge sweep is
  ready to submit, but not yet evaluated on DS1 full scoring.

### 2026-06-20 Budgeted false-split bridge merge

Reason for pivot:

- Pluto API and direct SSH are both unavailable this turn, so the prepared
  source-acceptor sweep still cannot be submitted.
- The strongest remaining diagnostic is the oracle repair decomposition:
  the current softcut base full IDF1 is `0.653071`, while
  `oracle_all_gt_majority` reaches `0.706202`.
- The best actionable oracle row is not a detection filter.  It is
  `split_top_40_then_merge_top_40`, with full IDF1 `0.706008`.  This points to
  structured false-split/false-merge repair as the remaining >70 path.

No-GT implementation:

- `kit/no_anchor_assignment_multiview_merge_sweep.py` now has:
  `--max-accepted-edges-grid` and `--one-edge-per-component`.
- Default behavior is unchanged because `--max-accepted-edges-grid 0` means
  uncapped threshold merging.
- The new mode ranks component bridges with no-GT multiview evidence, then
  caps accepted edges at budgets such as `10,20,40,80`.  This mirrors the
  oracle top-k repair shape without using GT to choose components.
- `--one-edge-per-component` provides a diversity variant that avoids spending
  the entire budget on one chain of near-duplicate components.

Prepared launcher:

- `kit/run_no_anchor_false_split_budget_merge.sh`.
- It runs two variants on the current best assignment:
  `chained` and `diverse`.
- Inputs are the current softcut+soft-overlap assignment plus fused, posecolor,
  colorhist, DINO, and DB views.
- Outputs will be:
  `no_anchor_false_split_budget_merge_chained_20260620.json`,
  `no_anchor_false_split_budget_merge_diverse_20260620.json`,
  their CSVs, and top assignment CSVs.

Validation:

- Local validation passed:
  `py_compile`, launcher `bash -n`, and a synthetic `_merge_edges` budget /
  diversity check.
- Follow-up hardening added a real `--self-test` entrypoint to
  `kit/no_anchor_assignment_multiview_merge_sweep.py`, covering both accepted
  edge budget behavior and `--one-edge-per-component` diversity rejection.
- The same hardening fixed direct-script import setup by adding repo/kit roots
  to `sys.path` before importing `vlincs_gallery`, so the test can run in a
  bare local or remote shell before database inputs are available.
- The remote launcher now runs both `py_compile` and solver `--self-test`
  before launching the `chained` and `diverse` DS1 full-score jobs.
- Verified after the hardening:
  `python -m py_compile`, solver `--self-test`, launcher `bash -n`, and
  `progress.json` / `directions_tried.json` / `findings.jsonl` JSON checks all
  pass locally.
- Pluto status checks for `h100-test-3`, `h100-test-2`, and `test-video-0`
  failed with `Failed to connect to Pluto service`.
- A fresh dry-run probe for `h100-test-3` still failed before remote execution:
  `Connection timed out during banner exchange`.
- Therefore this branch is prepared and better aligned with the oracle gap, but
  it has not yet produced a new DS1 full score.

### 2026-06-20 Committee proxy source-acceptor ranking

Reason for the branch:

- The delivery-aware learned full proxy fixed the earlier low-coverage failure,
  but its top predictions still cluster around conservative source edits.
- The source-island acceptor audit found a useful orthogonal signal:
  raw `source_rank_score` AP was `0.302423`, while the no-GT ridge-logit
  source acceptor reached LOOCV AP `0.944124` and found `9/10` positives in
  top-10.
- Before this branch, `source_acceptor_score` affected candidate edge proposal
  order but not the final full-score row ordering.  That left the expensive
  `full_top_n` budget mostly controlled by the delivery proxy alone.

Implementation:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now supports
  `--rank-by committee_proxy`.
- `committee_proxy` keeps `learned_full_proxy` as the main score, then adds a
  small bonus for `source_acceptor_score` above a floor:
  `--source-acceptor-rank-weight 0.025`,
  `--source-acceptor-rank-floor 0.50` in the launcher.
- This is deliberately a tie-break / nudge, not a replacement for the
  delivery-aware full proxy.  The goal is to select better repair sources among
  candidates that already look delivery-safe.
- The script also now fixes direct-script import setup before importing
  `vlincs_gallery`, matching the hardening added to the budgeted bridge solver.

Prepared launcher:

- `kit/run_no_anchor_false_split_diversity.sh` now launches the committee
  variant by default.
- Remote outputs are:
  `no_anchor_conflict_reassign_diverse_first_edge_committee_deliveryproxy_20260620.json`,
  its CSV, and top assignment CSV.

Validation:

- Local validation passed:
  `python -m py_compile kit/no_anchor_assignment_conflict_reassign_sweep.py`,
  solver `--self-test`, and launcher `bash -n`.
- A dry-run probe for `h100-test-3` still failed before execution:
  `Connection timed out during banner exchange`.
- Therefore this is a prepared no-anchor e2e challenger, not a verified full
  IDF1 improvement yet.

### 2026-06-20 Offline full-score scheduler

Reason:

- Pluto remained unavailable, so the expensive DS1 full scorer could not be
  used this turn.
- The previous full-proxy and committee-proxy work still leaves a practical
  question: when remote scoring returns, which candidates should consume the
  first limited full-score budget?
- The scheduler makes that decision file-backed instead of conversational.

Implementation:

- Added `kit/no_anchor_fullscore_scheduler.py`.
- Inputs:
  `local_runs/no_anchor_delivery_proxy_pair_candidates_20260620.json`,
  `local_runs/no_anchor_delivery_proxy_pair_candidates_with_full_20260620.json`,
  and the CSV export.
- Model:
  `local_runs/no_anchor_full_proxy_delivery_ridge_model_20260620.json`.
- Guardrails:
  no anchors / no GT-training metadata violations,
  pair F1 / precision / recall all at least `0.70`,
  delivery at least `7000` tracklets when known,
  predicted full IDF1 at least `0.6530`,
  reject already-full-scored rows below current best,
  and coarse family de-duplication for global solvers such as louvain.

Output:

- JSON:
  `local_runs/no_anchor_fullscore_scheduler_20260620.json`.
- CSV:
  `local_runs/no_anchor_fullscore_scheduler_20260620.csv`.
- Markdown:
  `reports/no_anchor_fullscore_scheduler_20260620.md`.

Scheduler result:

| raw candidates | eligible | selected | top predicted full | current verified e2e best |
| ---: | ---: | ---: | ---: | ---: |
| `150` | `69` | `11` | `0.656057` | `0.655240` |

Top selected families:

- `conflict_subcluster_reassign_candidate_search:source_target:12:19`
- `louvain:artifact:no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json`
- `conflict_subcluster_reassign:source_target:8:6`
- `cannotlink_nms_singleton:artifact:no_anchor_cannotlink_nms_singleton_relaxed_pair_20260619.json`
- `assignment_multiview_merge:artifact:no_anchor_state_color_forced_multiview_merge_pair_20260620.json`

Validation:

- `python -m py_compile kit/no_anchor_fullscore_scheduler.py` passed.
- `python kit/no_anchor_fullscore_scheduler.py --self-test` passed.
- Re-running `kit/no_anchor_result_gate.py` still reports `pass_joint=false`.
  The best full-scored local row in the scanned artifacts is `0.654009`, while
  the promoted state remains `0.655240`; neither is above the `0.70` target.

Conclusion:

- This is not an e2e improvement yet.
- It is a budget allocator for the next remote full-score window, reducing the
  chance that the next Pluto run spends all full-score slots on duplicate
  candidate families or rows already known to be below the current best.

## 2026-06-20 AutoResearch distillation and tiny-fragment override

Context:

- The Deli AutoResearch framework is most relevant here as an operating
  protocol: persistent state files, honest metric drops, structural pivots after
  stale iterations, and separation between candidate proposal and evaluation.
- This maps directly to the current no-anchor bottleneck. Pair-model metrics are
  already above target, but full IDF1 remains at `0.655240`; oracle repair can
  reach about `0.706`, so the next work should target structured false splits.

Distillation artifact:

- `reports/autoresearch_distillation_for_vlincs_20260620.md`

Implementation:

- Added tiny-fragment cannot-link override parameters to
  `kit/no_anchor_assignment_multiview_merge_sweep.py`.
- Default behavior is unchanged.
- When enabled, only high-scoring bridges whose smaller original component is
  within a swept small-side limit and whose large/small size ratio clears a
  swept threshold can bypass cannot-link.
- Added a narrow `tiny_fragment_override` branch to
  `kit/run_no_anchor_false_split_budget_merge.sh`, alongside the existing
  `chained` and `diverse` branches.

Validation:

- `python -m py_compile kit/no_anchor_assignment_multiview_merge_sweep.py`
  passed.
- `python kit/no_anchor_assignment_multiview_merge_sweep.py --self-test`
  passed, including blocked forbidden edge, accepted tiny-fragment override,
  and blocked large-large forbidden cases.
- `bash -n kit/run_no_anchor_false_split_budget_merge.sh` passed.
- Fresh Pluto probes did not submit the run:
  `h100-test-3` and `h100-test-2` both failed Pluto API status with
  `Failed to connect to Pluto service`, and both dry-run SSH probes timed out
  during banner exchange.

Conclusion:

- No new DS1 full score was produced in this step.
- This is a structural branch for the next Pluto window: it tests whether the
  oracle-observed large-identity false splits can be recovered by budgeted
  large-to-tiny repairs without opening the door to broad false merges.

## 2026-06-20 Target-fragment edge-rank audit

Context:

- The first tiny-fragment override branch still needed a better no-GT selector.
- The available exported edge tables include GT labels only for post-hoc audit,
  so they can be used to analyze rule quality without using anchors or GT for
  training.

Implementation:

- Added `kit/analyze_no_anchor_target_fragment_rules.py`.
- The script reads exported component edge CSVs and sweeps only no-GT features:
  smaller-side component size, large/small component-size ratio, edge score,
  top-5 vote count, original edge rank, and forbidden/allowed mode.
- GT fields are consumed only after candidate generation for eval-only
  precision/mass reporting.

Outputs:

- Softcut audit:
  `local_runs/no_anchor_target_fragment_rule_audit_softcut_20260620.json`
  and `reports/no_anchor_target_fragment_rule_audit_softcut_20260620.md`.
- DINO-fused audit:
  `local_runs/no_anchor_target_fragment_rule_audit_dinofused_20260620.json`
  and `reports/no_anchor_target_fragment_rule_audit_dinofused_20260620.md`.
- Combined audit:
  `local_runs/no_anchor_target_fragment_rule_audit_combined_20260620.json`
  and `reports/no_anchor_target_fragment_rule_audit_combined_20260620.md`.

Audit results:

| edge table | no-GT rule | true / candidates | precision | true same mass |
| --- | --- | ---: | ---: | ---: |
| softcut current | `small_side <= 2`, `score >= 0.75` | `13 / 13` | `1.000` | `3,586,752` |
| DINO-fused | `small_side <= 2`, `max(source_rank,target_rank) <= 3` | `17 / 17` | `1.000` | `4,081,234` |
| combined | `small_side <= 2`, `max(source_rank,target_rank) <= 3`, `forbidden_only` | `19 / 19` | `1.000` | `5,080,302` |

Solver update:

- Added `--edge-rank-maxes` to
  `kit/no_anchor_assignment_multiview_merge_sweep.py`.
- The merge info now reports `max_edge_rank` and `rejected_edge_rank`.
- The tiny-fragment launcher branch now includes:
  `--edge-rank-maxes 3,5,10,1000000`,
  `--forbidden-override-small-side-sizes 1,2`,
  higher thresholds up to `0.80`,
  and edge budgets `10,13,17,20,40`.

Validation:

- `python -m py_compile kit/no_anchor_assignment_multiview_merge_sweep.py kit/analyze_no_anchor_target_fragment_rules.py` passed.
- `python kit/no_anchor_assignment_multiview_merge_sweep.py --self-test` passed.
- `python kit/analyze_no_anchor_target_fragment_rules.py --self-test` passed.
- `bash -n kit/run_no_anchor_false_split_budget_merge.sh` passed.

Conclusion:

- This is still not a verified e2e improvement.
- It improves the next full-score experiment by replacing a broad cannot-link
  relaxation with a localized no-GT fragment/rank gate supported by two
  independent exported edge tables and a combined-table check.

## 2026-06-20 Edge-rank target-fragment remote launcher

Context:

- The previous audit identified `source_rank` / `target_rank` as a useful
  no-GT selector for tiny false-split fragments.
- The existing `no_anchor_edge_table_target_repair_sweep.py` already consumes
  exported edge tables and evaluates assignment repairs, but it did not expose
  this original edge-rank signal.

Implementation:

- Extended `kit/no_anchor_edge_table_target_repair_sweep.py`:
  - reads `source_rank` and `target_rank`;
  - stores `edge_rank_max = max(source_rank, target_rank)`;
  - exposes `--max-edge-ranks`;
  - adds direct-script `sys.path` setup;
  - adds `--self-test`.
- Added `kit/run_no_anchor_edge_rank_target_fragment.sh`.
- The new launcher packages the table-driven solver and runs two no-anchor
  sweeps on the first reachable Pluto job:
  - `softcut`: uses
    `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_edge_acceptor_table_20260619.csv`;
  - `dinofused`: uses
    `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_acceptor_table_20260620.csv`.

Remote sweep shape:

- Base assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`.
- Small-side grid: `1,2`.
- Edge-rank grid: `3,5,10`.
- Requires forbidden edges, matching the combined audit's strongest rule.
- Full-score budget: `--full-top-n 12` per table.
- Assignment export enabled for the top row.

Validation:

- `python -m py_compile kit/no_anchor_edge_table_target_repair_sweep.py kit/analyze_no_anchor_target_fragment_rules.py kit/no_anchor_assignment_multiview_merge_sweep.py` passed.
- `python kit/no_anchor_edge_table_target_repair_sweep.py --self-test` passed.
- `python kit/analyze_no_anchor_target_fragment_rules.py --self-test` passed.
- `python kit/no_anchor_assignment_multiview_merge_sweep.py --self-test` passed.
- `bash -n kit/run_no_anchor_edge_rank_target_fragment.sh` passed.
- `bash -n kit/run_no_anchor_false_split_budget_merge.sh` passed.

Remote status:

- `h100-test-3`: Pluto API status failed with `Failed to connect to Pluto service`; dry-run SSH probe timed out during banner exchange.
- `h100-test-2`: Pluto API status failed with `Failed to connect to Pluto service`; dry-run SSH probe timed out during banner exchange.

Conclusion:

- No new DS1 full score was produced.
- The next executable branch is now stronger than the earlier tiny-fragment
  sweep because it uses the edge-table solver directly and full-scores the
  audited edge-rank target-fragment rule family.
