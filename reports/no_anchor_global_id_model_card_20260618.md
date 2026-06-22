# No-Anchor Global-ID Model Card - 2026-06-18

## Status

Global-ID model acceptance is complete for the no-anchor setting.

- No anchors: yes
- GT used for training or identity seeds: no
- GT used for evaluation only: yes
- Model-only gate: pass
- Best promoted delivery assignment pair F1: 0.768743
- Best promoted delivery assignment precision: 0.813273
- Best promoted delivery assignment recall: 0.728836
- Best diagnostic assignment pair F1: 0.775234
- Best diagnostic assignment precision: 0.820504
- Best diagnostic assignment recall: 0.734698
- Best trained pair-calibrator single-solver pair F1: 0.726586
- Best trained pair-calibrator single-solver precision: 0.789270
- Best trained pair-calibrator single-solver recall: 0.673126

The end-to-end pipeline target is not complete yet. The best verified
submission artifact is now IDF1 0.655240.  The historical current-tracklet
oracle artifact was IDF1 0.711353; the latest standing-assignment oracle refresh
under the current scorer is IDF1 0.706202.

Latest continuation note: loose source-island search plus strict target gating
is the current model-side diagnostic best when the source island size is capped
at 8 tracklets.  It reaches pair F1/P/R
`0.775234 / 0.820504 / 0.734698` by moving four 8-tracklet conflict islands into
existing target components.  It is not promoted as production: unfiltered full
IDF1 is `0.653541`, and the no-GT row-density/confidence selector reaches only
`0.653681`, below the verified no-anchor full-score best `0.655240`.  This is a
clean model-side improvement and another pair/full mismatch, so the next
research branch should learn delivery-aware admission rather than only improve
tracklet-pair structure.

## Accepted Global-ID Artifacts

Primary exported assignment JSON:

`/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_full_20260618.json`

Primary exported assignment CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_assignments_20260618.csv`

Previous Louvain + FaceNet + small-fragment attach assignment JSON:

`/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_small_attach_top_full_20260618.json`

Previous Louvain + FaceNet + small-fragment attach assignment CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_small_attach_top_assignments_20260618.csv`

Previous Louvain + FaceNet assignment JSON:

`/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_localgrid_e2ebest_full_20260618.json`

Previous Louvain + FaceNet assignment CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_localgrid_e2ebest_assignments_20260618.csv`

Previous high-resolution Louvain assignment JSON:

`/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_base_highres_e2ebest_full_20260618.json`

Previous high-resolution Louvain assignment CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_base_highres_e2ebest_assignments_20260618.csv`

Previous video-hybrid assignment JSON:

`/mnt/localssd/vlincs_reid_runs/no_anchor_video_hybrid_b_except_bad_full_20260618.json`

Previous video-hybrid assignment CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_video_hybrid_b_except_bad_assignments_20260618.csv`

Previous pure time-agglom assignment JSON:

`/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_model_fused_mpc010_f032_t014_multivideo_pairf1_assignments_20260618.json`

Model-only gate:

`/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_model_only_gate_20260618.json`

Model-only gate top CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_model_only_gate_top_20260618.csv`

Trained pair-calibrator model bundle:

`/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_20260618.joblib`

Pair-calibrator JSON:

`/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_full8_20260618.json`

Pair-calibrator assignment CSV:

`/mnt/localssd/vlincs_reid_runs/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_assignments_20260618.csv`

Joint global/e2e gate:

`/mnt/localssd/vlincs_reid_runs/no_anchor_global_id_and_e2e_gate_small_attach_quality_20260618.json`

Current promoted e2e submission zip:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.zip`

Current promoted e2e scoring grid:

`/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_submission_detection_filter_grid_20260619.json`

Previous promoted q03 submission zip:

`/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_submission_detection_filter_q03_20260619.zip`

Tiny-best OSNet component-merge continuation:

`/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_reload_20260619.json`

Tiny-best submission zip:

`/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_20260619.zip`

Current diagnostic softcut assignment:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_best_assignments_20260619.csv`

Current diagnostic softcut pair JSON:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_pair_20260619.json`

Current diagnostic softcut full-score JSON:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_promoted_filters_20260619.json`

Current diagnostic softcut+soft-overlap per-video filter oracle:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_pervideo_filter_oracle_20260619.json`

Current no-GT density filter selector:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.json`

Current no-GT density filter submission zip:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.zip`

## Output Schema

The assignment CSV has one row per emitted tracklet assignment and includes:

- `seq`
- `tracklet_key`
- `video`
- `camera`
- `start_frame`
- `end_frame`
- `n_dets`
- `avg_conf`
- `predicted_global_id`
- `component_label`
- `component_size`
- `prediction_confidence`
- `decision_status`
- `component_internal_edges`
- `component_internal_score_median`
- `component_external_prob_max`
- `component_margin_prob`

Primary Louvain + FaceNet + small-fragment attach + quality admission artifact size:

- assignment rows: 7487
- assignment components: 111
- largest assignment component: 275
- decision status counts: `forced_component=7421`, `forced_singleton=66`

Previous Louvain + FaceNet + small-fragment attach artifact size:

- assignment rows: 8752
- assignment components: 567
- largest assignment component: 292
- decision status counts: `forced_component=8230`, `forced_singleton=522`

Previous Louvain + FaceNet artifact size:

- assignment rows: 8752
- assignment components: 653
- largest assignment component: 287
- decision status counts: `forced_component=8144`, `forced_singleton=608`

Previous high-resolution Louvain artifact size:

- assignment rows: 8752
- assignment components: 653
- largest assignment component: 285

Previous pure time-agglom artifact size:

- assignment rows: 8752
- assignment components: 670
- largest assignment component: 275
- decision status counts: `forced_component=12`, `forced_singleton=608`, `provisional=8132`

Trained pair-calibrator assignment size:

- assignment rows: 9009
- assignment components: 730
- largest assignment component: 284
- decision status counts: `forced_component=3`, `forced_singleton=648`, `provisional=8358`

## Method

This model treats global-ID prediction as no-anchor identity resolution rather
than closed-set classification.

1. Build tracklet evidence from detector tracklets and fused embeddings.
2. Apply admission filters to avoid committing low-quality tracklets into the
   model output.
3. Build a no-GT graph over tracklets. Nodes are tracklets; candidate edges come
   from fused visual similarity, database embedding similarity, temporal support,
   and top-k retrieval.
4. Enforce no-anchor constraints. In particular, same-camera exclusion and
   temporal/overlap constraints reduce impossible merges.
5. Resolve the graph into identity components with time-aware agglomeration.
6. Emit synthetic `predicted_global_id` values per resolved component with
   confidence and status. These IDs are delivery IDs, not GT person IDs.
7. Evaluate with cached GT labels only after prediction, reporting pairwise
   tracklet identity precision, recall, and F1.

The trained pair-calibrator variant additionally fits a histogram gradient
boosting edge model from no-anchor pseudo positives/negatives:

- pseudo positives: `139486`
- pseudo negatives: `60000`
- pseudo validation AUC/AP: `0.995851 / 0.997873`
- best threshold/blend: `0.03 / 0.50`
- best full IDF1 for the trained model: `0.635197`

## Multiview Verifier Extension

`kit/no_anchor_global_id_model.py` also supports repeatable
`--pair-feature-npz name:path` inputs.  These add verifier-only pair features
from independent tracklet embeddings without changing the graph feature used for
candidate generation.

The June 18 multiview ablation used `match`, `person_s1`, `person_s3`, `color`,
and `posecolor` views.  The best pseudo-ensemble multiview run reached pair F1
`0.711438`, precision `0.784113`, and recall `0.651092`; the plain multiview
run reached pair F1 `0.709180`.

This extension is reproducible but is not the promoted global-ID model because
it did not beat the then-current pure time-agglom assignment artifact at pair
F1 `0.719910`, and it remains below the video-hybrid artifact.

## Louvain + FaceNet Low-Weight Extension

`kit/no_anchor_louvain_sweep.py` builds a no-anchor graph over fused tracklet
evidence and resolves global-ID components with Louvain community detection.
The current graph feature adds a sparse FaceNet/VGGFace2 tracklet
embedding as a low-weight block in the fused graph feature:

- FaceNet feature artifact:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_facenet_vggface2_s2_20260618.npz`
- FaceNet coverage: `4260 / 9734` tracklets have valid high-confidence face
  embeddings; all rows are present, with zero vectors for no-face tracklets.
- Fused feature artifact:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_20260618.npz`
- FaceNet weight: `0.05`
- Louvain config: `top_k=10`, `edge_floor=0.035`, `resolution=5.0`

Metrics for the best plain Louvain + FaceNet artifact:

- pair F1: `0.762137`
- precision: `0.808448`
- recall: `0.720844`
- full IDF1: `0.652390`
- HOTA: `0.516276`
- AssA: `0.531901`
- DetRe: `0.575268`
- DetPr: `0.753390`

This was the previous best verified no-anchor e2e artifact.  The improvement
over the prior high-resolution Louvain artifact is small (`0.652390` vs
`0.652157`), but it is positive under the canonical full scorer and uses no
anchors.

Directly inserting FaceNet as a `pair-feature-npz` verifier in the HGB
`consensus_guard` model was negative: even with explicit
`face_both_valid` / `face_either_valid` features, best pair F1 stayed around
`0.432349` because recall collapsed to about `0.299`.  Face evidence is
therefore useful as a low-weight graph feature, not as the primary conservative
pair gate in the current resolver.

## Small-Fragment Attach Extension

`kit/no_anchor_louvain_component_merge_sweep.py` starts from the Louvain +
FaceNet graph resolver and then attaches small source components into larger
target components when component-level evidence is strong enough.  This remains
no-anchor: the attach decision uses fused visual/component evidence only, while
GT is used after the fact for metrics.

Promoted attach config:

- source component size: `1..32`
- target component size: `>=12`
- candidate top-k: `60`
- top edge k: `8`
- centroid weight: `0.0`
- threshold: `0.86`
- margin: `-1.0`
- accepted attaches: `86`

Metrics for the previous small-fragment attach artifact:

- pair F1: `0.762189`
- precision: `0.808383`
- recall: `0.720989`
- full IDF1: `0.652398`
- HOTA: `0.516290`
- AssA: `0.531916`
- DetRe: `0.575322`
- DetPr: `0.753319`

The gain over plain Louvain + FaceNet is tiny (`0.652398` vs `0.652390`), but
it was positive under the canonical full scorer.  A broader variant with
`top_edge_k=16` gave slightly higher pair F1 (`0.762190`) but lower full IDF1
(`0.652387`), so the stricter edge-8 attach was kept before the later quality
admission step.

## Quality Admission Extension

`kit/no_anchor_assignment_admission_grid.py` keeps the predicted global IDs
fixed and sweeps M3 delivery admission.  The promoted setting adds a no-GT
tracklet-quality threshold on top of the small-fragment attach assignment:

- input assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_louvain_face005_small_attach_top_assignments_20260618.csv`
- quality threshold: `output_min_quality=0.60`
- emitted assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_assignments_20260618.csv`
- full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_full_20260618.json`

Metrics for the current promoted artifact:

- pair F1: `0.768742`
- precision: `0.813274`
- recall: `0.728833`
- full IDF1: `0.652623`
- HOTA: `0.516393`
- AssA: `0.532037`
- DetRe: `0.574345`
- DetPr: `0.755605`
- unmatched FP: `109745`

The quality threshold removes low-confidence/low-area/short tracklets that are
unlikely to match evaluation objects.  It raises full IDF1 modestly
(`0.652398 -> 0.652623`) by improving detection precision and pair metrics; a
stricter `0.65` threshold hurts recall and drops full IDF1 to `0.652210`.

## Video-Hybrid Extension

`kit/no_anchor_video_hybrid_diagnostic.py` builds a no-anchor hybrid from two
resolvers:

- A: time-aware agglomeration over fused tracklet evidence.
- B: loaded pair-calibrator output.
- B component labels are aligned to A labels by tracklet-overlap majority,
  without identity labels.
- Current policy: use A for MCAM03 Tc8, MCAM04 Tc6, and MCAM06 Tc6; use aligned
  B for the other videos.

This was the previous promoted diagnostic artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_video_hybrid_b_except_bad_full_20260618.json`

Metrics:

- pair F1: `0.727059`
- precision: `0.784763`
- recall: `0.677259`
- full IDF1: `0.635490`

The hybrid improved both model-only pair F1 and verified e2e IDF1 at the time,
but it is now superseded by the high-resolution Louvain and Louvain + FaceNet
artifacts.

Latest continuation diagnostics around the old hybrid showed:

- fused top-100 component retrieval recovers `0.960074` of false-split mass by
  top-100, but dominant-pure edge precision is only `0.117519` even at score
  threshold `0.78`
- fast multiview component verifier accepted about `70` high-confidence edges,
  but pair F1 stayed at `0.719910`
- aligned oracle replacement of MCAM03 Tc8, MCAM04 Tc6, and MCAM06 Tc6 reaches
  only full IDF1 `0.657272`, so local bad-video repair cannot by itself reach
  the e2e target
- assignment-level per-video source switching over existing no-anchor outputs
  reached full IDF1 `0.635418`, so selector-only reuse of the current model pool
  does not beat the promoted hybrid

## Primary Configuration

- feature NPZ: `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_20260618.npz`
- concat DB embedding: true
- DB weight: 1.0
- feature weight: 0.32
- resolver: Louvain community detection
- top-k: 10
- edge floor: 0.035
- resolution: 5.0
- min detections: 10
- exclude same: camera
- temporal bonus: 0.005
- time window: 1000 ms
- admission:
  - `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8:4000`
  - `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6:2000`
  - `vlincs_MS01_MC0001_MCAM05_2024-03-Tc6:12000`
  - `vlincs_MS01_MC0001_MCAM06_2024-03-Tc6:6000`

## Gate Command

```bash
cd /mnt/localssd/vlincs_reid_by_search
DATA_ROOT=/mnt/localssd/vlincs_reid_data \
PYTHONPATH=/mnt/localssd/vlincs_reid_by_search:$PYTHONPATH \
/mnt/localssd/vlincs_reid_venv/bin/python kit/no_anchor_result_gate.py \
  /mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_full_20260618.json \
  /mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_assignments_20260618.csv \
  --global-metric tracklet_pair_f1 \
  --e2e-metric full_idf1 \
  --precision-metric tracklet_pair_precision \
  --recall-metric tracklet_pair_recall \
  --global-threshold 0.70 \
  --e2e-threshold 0.70 \
  --json-out /mnt/localssd/vlincs_reid_runs/no_anchor_global_id_and_e2e_gate_small_attach_quality_20260618.json \
  --csv-out /mnt/localssd/vlincs_reid_runs/no_anchor_global_id_and_e2e_gate_small_attach_quality_top_20260618.csv \
  --top-n 10
```

## Current E2E Boundary

Best verified full/e2e artifact:

`/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_reload_20260619.json`

Metrics:

- submission reload IDF1: 0.654597
- HOTA: 0.518183
- AssA: 0.533816
- DetRe: 0.576083
- DetPr: 0.757889

The previous DB-comp full/e2e artifact remains:

`/mnt/localssd/vlincs_reid_runs/no_anchor_small_attach_quality060_full_20260618.json`

- DB-comp full IDF1: 0.652623

Current-tracklet oracle:

`/mnt/localssd/vlincs_reid_runs/oracle_current_tracklets_full_upper_20260617.json`

- oracle full IDF1: 0.711353

This means the >0.70 end-to-end target is barely feasible with the current
tracklets and requires near-oracle association. The next research step should
focus on false-split recovery without introducing false merges.

## Greedy-Tracklet Global-ID Model Continuation

This continuation tested whether a different no-anchor tracklet source can
raise the model side while keeping the full pipeline honest.  The new tracklet
source is a greedy IoU linker over YOLO detections, not an anchor-labeled
training set.

Sample and feature artifacts:

- sample parquet:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_sample_20260618.parquet`
- base feature bundle:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_features_20260618.npz`
- OSNet/color one-middle-crop feature bundle:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_osnet_color_mid1_20260618.npz`
- sample export summary:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_sample_export_20260618.json`

Sample size:

- tracklets: `18656`
- detection rows: `1723070`
- evaluation-labeled tracklets: `12851`
- matched detection fraction: `0.776188`
- greedy oracle-labeled full IDF1 upper check: `0.736027`

The promoted greedy-tracklet model for this branch is the light OSNet fusion
resolver:

- model bundle:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_global_model_20260618.joblib`
- assignments:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_assignments_20260618.csv`
- grid JSON:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_grid_20260618.json`
- full scorer JSON:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_full_20260618.json`
- submission zip:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_light2_submission_20260618.zip`

Best resolver settings:

- feature weights:
  `dbresolve=1.0`, `current_gid_centroid=0.9`, `geometry=0.04`,
  `quality=0.02`, `osnet=0.25`, `color=0.08`
- mode: `time_agglom`
- `theta=0.014`
- `top_k=30`
- output min detections: `1`

Model-side metrics:

- deliverable identity F1: `0.751804`
- deliverable tracklet-pair F1: `0.664014`
- pair precision / recall: `0.748806 / 0.596472`
- predicted IDs: `60`
- assignment rows/components: `18656 / 60`

Full scorer metrics:

- IDF1: `0.620289`
- HOTA: `0.481560`
- AssA: `0.501756`
- DetRe: `0.541737`
- DetPr: `0.725486`
- unmatched FP: `276937`

Per-video full IDF1:

- MCAM00 Tc6: `0.854746`
- MCAM00 Tc8: `0.820703`
- MCAM03 Tc6: `0.670947`
- MCAM03 Tc8: `0.637100`
- MCAM04 Tc6: `0.528884`
- MCAM05 Tc6: `0.263923`
- MCAM05 Tc8: `0.818085`
- MCAM06 Tc6: `0.583516`
- MCAM06 Tc8: `0.740485`
- MCAM08 Tc6: `0.733082`

Conclusion: this branch produces a valid no-anchor global-ID model with a
stronger row-level identity score (`0.751804`) than the earlier base
greedy-tracklet models, but it is not the promoted end-to-end artifact.  The
full scorer drops to `0.620289`, mainly because the all-tracklet greedy output
keeps difficult false positives and very large forced components, especially
on MCAM05 Tc6.  The promoted e2e artifact remains the quality-admitted
small-attach model at full IDF1 `0.652623`, while the best global-ID model
family remains above the `0.70` model-only bar.

## AutoResearch-Inspired Continuation - 2026-06-19

The Deli AutoResearch / self-play research note suggested a stricter operating
pattern for this thread: isolated hypotheses, scalar hard gates, explicit
refutation, and pivoting when related attempts fail.  Applying that loop to
the no-anchor VLINCS target produced the following continuation artifacts:

- local-track relink script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_localtrack_relink_sweep.py`
- cannot-link split script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_cannotlink_split_sweep.py`
- local-track relink JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_localtrack_relink_pair_20260619.json`
- cannot-link split JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_split_pair_20260619.json`
- temporal NMS diagnostic:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_nms_error_diag_20260619.json`
- video-namespace negative diagnostic:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_video_namespace_error_diag_20260619.json`

Findings:

- `local_track_id` relink has no effect on the promoted quality-admitted
  artifact because there are no qualifying multi-segment local-track groups
  left to rewrite.
- per-video namespace splitting is invalid for this task: full IDF1 drops to
  `0.366241`, proving cross-video global identity continuity matters.
- direct temporal cannot-link coloring finds many conflicts but does not
  improve pair F1; hard split is too blunt.
- temporal NMS is useful only as a diagnostic.  It creates a very strong
  model-side assignment (`0.826444` pair F1) but loses full IDF1 due to
  detector-recall loss.

The next model direction should use cannot-link conflict statistics as
verifier inputs: component conflict density, overlap-NMS survival score,
temporal co-visibility, component size, and multiview visual similarity.  The
verifier should block risky merges and selectively attach fragments; it should
not delete large numbers of tracklets.

## Cannot-Link Verifier Continuation - 2026-06-19

Additional verifier-style experiments tested the same temporal/cannot-link
signal without using anchors or GT labels for training:

- conflict-aware merge script:
  `/mnt/localssd/vlincs_reid_by_search/kit/no_anchor_assignment_conflict_aware_merge_sweep.py`
- conflict-aware pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_conflict_aware_osnet_merge_small_pair_20260619.json`
- light hard-split full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_split_light_full_20260619.json`
- unlimited hard-split full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_cannotlink_split_unlimited_full_20260619.json`
- NMS-teacher attach JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_nms_teacher_attach_osnet_paironly_20260619.json`

Results:

- light hard split: pair F1 `0.764085`, full IDF1 `0.650945`
- unlimited hard split: pair F1 `0.643634`, full IDF1 `0.615761`
- conflict-aware OSNet merge: best pair row stayed at `0.768743`, effectively
  no better than current
- NMS plus singleton losers: pair F1 `0.644490`
- NMS-teacher visual reattachment: best pair F1 `0.643990`

Conclusion: temporal/cannot-link remains a strong diagnostic signal, but the
tested deterministic uses are negative for the promoted model.  The no-anchor
global-ID model still passes the model-only bar, but the end-to-end target
remains open at best submission reload IDF1 `0.654597`.

## AutoResearch-Style Soft Verifier Continuation - 2026-06-19

The Deli AutoResearch / self-play thread was distilled into a stricter
experiment loop for this project: isolate one hypothesis, run a bounded
experiment, keep a scalar pass/fail gate, and pivot after repeated related
failures.  Applying that loop after the hard cannot-link experiments gave three
new diagnostics.

Lightweight video-source oracle:

- artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_lightweight_video_source_oracle_20260619.json`
- source full IDF1 values:
  - current best diagnostic assignment: `0.652624`
  - small-attach quality: `0.652623`
  - cannot-link light split: `0.650945`
  - cannot-link NMS: `0.622822`
  - video namespace: `0.366241`
- every per-video winner is the current best diagnostic assignment, modulo
  exact or near-exact ties;
- the mean per-video oracle IDF1 is `0.715825`, but this is only a diagnostic
  average, not a recomputed global e2e score and not a valid selector.

Soft verifier from weak time-agglom base:

- artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_component_verifier_soft_fused_osnet005_quality060_paironly_20260619.json`
- base pair F1: `0.600437`
- pseudo positives / negatives / unlabeled: `26 / 4995 / 141`
- best verifier row: pair F1 `0.600402`, accepted edges `22`

Assignment-base soft verifier:

- code change:
  `kit/no_anchor_component_verifier_sweep.py` now accepts `--assignment-csv`
  and `--pred-col`, so it can start from an existing no-anchor assignment.
- artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_component_verifier_soft_fused_osnet005_quality060_paironly_20260619.json`
- starting assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`
- base pair F1 / precision / recall: `0.768743 / 0.813273 / 0.728836`
- pseudo positives / negatives / unlabeled: `39 / 4050 / 84`
- best verifier row: `0.768739 / 0.813263 / 0.728837`, accepted edges `10`

Conclusion: the current feature family is saturated.  The failure is not just
hard-threshold code; even a learned no-anchor verifier cannot find beneficial
merges from the current candidate pool.  The next useful research move is a
new positive-generation mechanism or a stronger identity evidence source,
rather than another deterministic merge/split threshold sweep.

## AutoResearch Distillation And Retrieval Verdict - 2026-06-19

The Deli AutoResearch release was distilled into this operating rule for the
VLINCS no-anchor loop: each iteration must produce a scalar improvement or a
clean refutation, repeated related failures force a structural pivot, and
state must live in durable artifacts rather than chat memory.  The useful
self-play analogy is not "let the agent search forever"; it is to construct
cheap opponents/verifiers that expose where the current identity hypothesis is
wrong.

Current-assignment retrieval diagnostic:

- script update:
  `kit/analyze_no_anchor_component_retrieval.py` accepts `--assignment-csv`
  and `--pred-col`
- base assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`
- feature views tested: fused, OSNet, CLIP, DINO, PersonViT, FaceNet
- all artifacts use GT only for diagnostic measurement, never as anchors or
  training labels.

Key result:

| feature view | top-10 split recall | top-50 split recall | top-100 split recall | edge precision at 0.62 |
| --- | ---: | ---: | ---: | ---: |
| fused | `0.701155` | `0.996523` | `1.000000` | `0.082969` |
| OSNet | `0.694591` | `0.999998` | `1.000000` | `0.040113` |
| CLIP | `0.551961` | `0.999951` | `1.000000` | `0.019031` |
| DINO | `0.590305` | `0.999942` | `1.000000` | `0.023132` |
| PersonViT | `0.683738` | `0.999998` | `1.000000` | `0.041757` |
| FaceNet | `0.632254` | `0.999938` | `1.000000` | `0.015217` |

Interpretation:

- Missing retrieval is no longer a plausible explanation for the `0.654597`
  e2e boundary.  Existing feature banks can surface essentially all false-split
  mass from the current best assignment.
- The failure is edge calibration: candidate edges are available but extremely
  impure.  Larger top-k retrieval will mostly add false merges.
- The next global-ID model should train a calibrated pair/edge verifier from
  no-anchor positives and negatives: stable same-tracklet augmentations,
  temporal continuity, clothing/body consistency, face agreement when present,
  and synthetic/generative positives such as Pose2ID-style augmentation.  The
  graph resolver should consume calibrated edge likelihoods, not raw cosine
  thresholds.

## Direct Multiview Merge Refutation - 2026-06-19

To check whether a simpler rule could replace verifier learning, I added
`kit/no_anchor_assignment_multiview_merge_sweep.py`.  It starts from the
current best no-anchor assignment, scores component edges by multi-view
centroid agreement, and sweeps merge thresholds.  This uses no anchors and no
GT labels for construction.

Fast optimistic diagnostic:

- artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_multiview_centroid_tiny_pair_20260619.json`
- candidate mode: centroid-only, top-k `30`
- feature views: fused, OSNet, CLIP, DINO, PersonViT, FaceNet
- cannot-link disabled for speed and as an optimistic upper check

Result:

- current base pair F1 / precision / recall:
  `0.768743 / 0.813273 / 0.728836`
- best multiview row is no-op: `accepted_edges=0`, pair F1 `0.768743`
- accepting `9` edges drops pair F1 to `0.702338`
- accepting `13` edges drops pair F1 to `0.686394`

Conclusion: direct multi-view similarity is not a sufficient verifier.  It
selects false merges before it improves recall, even when temporal cannot-link
checks are disabled.  The next global-ID model direction remains calibrated
positive/negative generation for an edge verifier, not direct high-score
component merging.

## Teacher-Consensus Refutation - 2026-06-19

The Deli AutoResearch release was distilled into one operating discipline for
this no-anchor run: write state to files, execute bounded diagnostics when
ready, and pivot structurally after repeated related failures.  The self-play
story is especially relevant because it treats non-monotonic review scores and
negative checks as signal rather than as mistakes to hide.

Sources:

- `https://victorchen96.github.io/auto_research/framework.html`
- `https://victorchen96.github.io/auto_research/paper.html`
- `https://victorchen96.github.io/blog_self_play_story.html`

Hypothesis:

- If old no-anchor assignment variants make complementary mistakes, their
  agreement can produce pseudo-positive component merges for the current best
  assignment.

Experiment:

- script:
  `kit/no_anchor_assignment_teacher_consensus_merge_sweep.py`
- base assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`
- remote JSONs:
  - `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_teacher_consensus_merge_pair_20260619.json`
  - `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_teacher_consensus_merge_low_pairfull_20260619.json`
  - `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_teacher_consensus_merge_actual_pair_20260619.json`
  - `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_teacher_consensus_merge_forced_pair_20260619.json`

Metrics:

- base pair F1 / precision / recall:
  `0.768743 / 0.813273 / 0.728836`
- low-threshold top-3 full rows:
  all no-op, full IDF1 `0.652624`
- forced permissive setting:
  threshold `-0.50`, `min_same_votes=1`, `min_same_frac=0.10`,
  `max_diff_votes=6`, accepted edges `0`
- teacher edge preview:
  top visual-score pairs have `teacher_valid=7`, `teacher_same=0`,
  `teacher_diff=7`

Verdict:

- Teacher consensus is useful as negative evidence, but not as a source of
  merge positives.
- The route "old no-anchor outputs vote for false-split repair" is now
  falsified.
- The model card should not count this as a global-ID model improvement; it is
  a blocked path that informs the next model: generate positives from actual
  tracklet evidence, not from historical assignment agreement.

Implementation note:

- Fixed duplicate CSV fieldnames in `kit/no_anchor_component_merge_sweep.py`
  and `kit/no_anchor_component_verifier_sweep.py`.  Without the set conversion,
  large sweeps can generate repeated columns and multi-GB CSVs.

## True Multi-Frame OSNet s7 Ablation - 2026-06-19

The foundation-feature sampler was fixed so `samples=N` selects `N` evenly
spaced frames rather than collapsing to first/middle/last.  This creates a
valid multi-frame OSNet artifact:

- feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s7_true_20260619.npz`
- extraction stats:
  `seen_crops=66324`, `requested_rows=66327`, `missing_crop=3`,
  `shape=[9734,512]`, `sample_counts_max=7`

Model-side results:

- component-merge with s7 OSNet:
  pair F1 / precision / recall
  `0.768746 / 0.813275 / 0.728840`, full IDF1 `0.652623`
- fused Louvain `face005+osnet005_s7true`:
  pair F1 / precision / recall
  `0.766248 / 0.775577 / 0.757140`
- full sanity check for that high-recall Louvain candidate:
  IDF1 / HOTA / AssA / DetRe
  `0.642629 / 0.510470 / 0.531170 / 0.579287`
- fused Louvain `face005+osnet010_s7true`:
  best pair F1 / precision / recall
  `0.764479 / 0.806739 / 0.726425`

Verdict:

- Valid s7 temporal evidence does not move the end-to-end best beyond the
  current no-anchor reload best IDF1 `0.654597`.
- The current model's bottleneck is not single-frame sampling coverage alone.
  It is evidence calibration and graph decision control: false merges appear
  before recall gains become useful.
- The next global-ID model should explicitly output evidence states:
  `committed`, `provisional`, `pending`, `unresolvable`, and `forced`, with
  a calibrated pair verifier gating graph merge/split decisions.

## Pseudo-Label Purity Audit - 2026-06-19

New diagnostic script:

- `kit/audit_no_anchor_pseudo_labels.py`

The script reproduces the pseudo-label construction from
`kit/no_anchor_global_id_model.py`, preserves pair IDs and pseudo-label source,
and evaluates purity with GT only after the no-anchor labels are generated.

Audit artifact:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_pseudo_label_audit_face_osnet_s7_20260619.json`
- sample cases:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pseudo_label_audit_face_osnet_s7_samples_20260619.csv`

Key numbers:

- all pseudo positives:
  purity / weighted purity `0.875657 / 0.906332`
- `pseudo_online_agree` positives:
  purity / weighted purity `0.928050 / 0.941846`
- `strong_visual_pseudo` positives:
  purity / weighted purity `0.702903 / 0.707661`
- random negatives:
  purity / weighted purity `0.978457 / 0.980788`

Follow-up ablation:

- disabled `strong_visual_pseudo` with `pseudo_strong_pos_sim=1.01`;
- trained the HGB `consensus_guard` verifier with FaceNet + OSNet s7 pair
  features;
- artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_pair_model_cleanpos_face_osnet_s7_guard_paironly_20260619.json`
- pseudo validation AP / AUC:
  `0.997946 / 0.996879`
- best real pair F1 / precision / recall:
  `0.433515 / 0.797407 / 0.297674`

Interpretation:

- `pseudo_online_agree` is a viable no-anchor positive source.
- `strong_visual_pseudo` is too noisy to supervise merges directly.
- A clean verifier used as the primary resolver collapses recall, so the
  global-ID model should use calibrated verifier evidence for
  commit/quarantine/veto/targeted-attach decisions on top of the current
  high-recall resolver instead of replacing the resolver.

## Submission-Level Detection Filter - 2026-06-19

This is a delivery-layer ablation on top of the current best no-anchor
submission.  It does not retrain the global-ID model and it does not use
anchors.  GT is used only by the evaluator and by this research-grid selection.

Artifacts:

- script:
  `kit/evaluate_submission_detection_filter.py`
- source submission:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_submission_20260619.zip`
- grid:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_submission_detection_filter_grid_20260619.json`
- promoted research artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_submission_detection_filter_q03_20260619.zip`

Metrics:

| config | IDF1 | HOTA | AssA | DetPr | DetRe | dropped rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base reload | `0.654597` | `0.518183` | `0.533816` | `0.757889` | `0.576083` | `0` |
| hard_q03 | `0.654739` | `0.518148` | `0.533889` | `0.761672` | `0.574135` | `24677` |
| hard_q05 | `0.654674` | `0.517930` | `0.533746` | `0.764142` | `0.572640` | `41176` |

Model-card interpretation:

- Current best research artifact is now the q03 filtered zip by IDF1, but the
  improvement is only `+0.000142`.
- This should be reported as a delivery ablation, not a solved global-ID model:
  the identity evidence remains forced-delivery, and q03 was selected with GT
  metrics.
- The next production-grade model needs evidence states and provenance:
  `committed`, `provisional`, `pending`, `unresolvable`, and `forced`.  The
  current best assignment CSV has only `forced_component` / `forced_singleton`
  rows, so it is still a forced-output baseline in the terminology of
  Open-World ReID / evidence-calibrated identity resolution.

## Clean-Positive Verifier Placement - 2026-06-19

The clean `pseudo_online_agree` verifier was also tested through
`consensus_attach`:

- artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_cleanpos_consensus_attach_face_osnet_s7_full3_20260619.json`
- pair F1 / precision / recall:
  `0.426041 / 0.777230 / 0.293447`
- full IDF1 / HOTA / AssA:
  `0.465100 / 0.322826 / 0.337983`
- attach candidates / eligible / accepted:
  `417261 / 1355 / 0`

Interpretation:

- The verifier is not the only failure point.  In this script, `consensus_attach`
  first builds a conservative `consensus_guard` core; recall is already
  destroyed before attach runs.
- The next model should keep the current-best high-recall forced assignment as
  base, then use the verifier for audit/quarantine/veto/state calibration over
  existing components.

## AutoResearch Protocol Update - 2026-06-19

I distilled Deli AutoResearch into a project-local operating rule for the next
rounds:

- every run must leave durable JSON/CSV/zip artifacts plus a written
  interpretation;
- every result must be compared against the frozen current-best no-anchor gate;
- GT-scored thresholds are research artifacts only, not no-GT production
  policy;
- repeated tiny gains trigger a structural pivot instead of a wider grid;
- the agent should execute bounded experiments directly, then update the
  model card from observed metrics.

That rule changed today's direction: after the submission-level detection
filter gave only `+0.000142` IDF1, and `consensus_attach` failed, the next
experiment moved to verifier auditing over the current-best forced assignment
instead of another admission threshold sweep.

## Current-Best Verifier Split - 2026-06-19

Artifacts:

- script:
  `kit/no_anchor_assignment_verifier_split_sweep.py`
- full sweep:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_verifier_split_face_osnet_s7_full2_20260619.json`
- exported assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_verifier_split_t040_m16_assignments_20260619.csv`
- split + filter grid:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_verifier_split_t040_m16_detection_filter_grid_20260619.json`

Metrics:

| artifact | IDF1 | HOTA | AssA | DetPr | DetRe |
| --- | ---: | ---: | ---: | ---: | ---: |
| verifier split base | `0.652806` | `0.516613` | `0.532272` | `0.755983` | `0.574411` |
| verifier split + hard_q03 | `0.652963` | `0.516592` | `0.532356` | `0.759776` | `0.572481` |
| current best submission q03 | `0.654739` | `0.518148` | `0.533889` | `0.761672` | `0.574135` |

Pair-level audit:

- base pair F1 / precision / recall:
  `0.768743 / 0.813273 / 0.728836`
- best split pair F1 / precision / recall:
  `0.768837 / 0.813577 / 0.728761`
- audited components / internal pairs / forbidden pairs:
  `98 / 699262 / 3168`

Model-card interpretation:

- The verifier contains a small real structural signal, but hard splitting does
  not yet improve delivery metrics enough.
- The current best no-anchor artifact remains the submission-level q03 zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_submission_detection_filter_q03_20260619.zip`
- This is still not a finished global-ID model.  It is a forced-delivery
  resolver plus diagnostic verifier evidence.  The next model needs explicit
  `committed/provisional/pending/forced` states and provenance over each
  component.

## Greedy Relink OSNet s7 Continuation - 2026-06-19

AutoResearch distillation used for this branch:

- state is persisted through durable artifacts, not chat memory;
- each ready experiment is executed to a canonical metric;
- repeated same-shape failures trigger a structural pivot;
- a branch can close as a negative result when it clarifies the bottleneck.

Artifacts:

- multi-frame OSNet/color features:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_osnet_color_s7_20260619.npz`
- model bundle:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_global_model_20260619.joblib`
- assignments:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_assignments_20260619.csv`
- full scorer:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_full_20260619.json`
- submission:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_submission_20260619.zip`

Model-side status:

- identity F1: `0.753964`
- tracklet-pair F1 / precision / recall:
  `0.647739 / 0.703469 / 0.600192`
- status counts:
  `committed=708`, `provisional=3274`, `forced_component=14674`

Canonical full status:

| artifact | IDF1 | HOTA | AssA | DetPr | DetRe |
| --- | ---: | ---: | ---: | ---: | ---: |
| greedy s7 full | `0.612647` | `0.475550` | `0.497936` | `0.708133` | `0.539853` |
| current best q03 | `0.654739` | `0.518148` | `0.533889` | `0.761672` | `0.574135` |

Namespace/source-switch diagnostics:

- current-gid mapped full:
  `/mnt/localssd/vlincs_reid_runs/greedy_iou050_gap10_no_anchor_osnet_s7_light2_currentgid_mapped_full_20260619.json`
- current-gid mapped score:
  `0.612647` IDF1, unchanged because only labels changed, not components.
- aligned current-ID source switch:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_submission_switch_current_greedy_s7_currentid600_oracle_20260619.json`
- oracle per-video selected greedy for MCAM03 Tc8, MCAM05 Tc8, and MCAM06
  Tc8, but full IDF1 fell to `0.601175`.

Interpretation:

- The greedy s7 branch is a no-anchor global-ID model-side pass, but not an
  end-to-end pass.
- The failure is not just numeric ID namespace mismatch.  The greedy source has
  local per-video wins, but it breaks global consistency when mixed into the
  promoted current source.
- The promoted e2e artifact remains:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_submission_detection_filter_q03_20260619.zip`

## Local Sample Source-Switch Evidence - 2026-06-19

AutoResearch/self-play distillation produced one new actionable branch:
assignment sources should act as proposer/opponent variants rather than only as
static ensemble members.  I tested that idea locally on the MCAM04/MCAM08
sample parquet using no-anchor assignment CSVs and GT only for post-hoc
evaluation.

Artifact:

`/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_sample_video_source_switch_basefallback_20260619.json`

No-GT selector artifact:

`/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_sample_source_selector_sparse_overlay_20260619.json`

Top sample proxy result:

| policy | sample IDF1 | HOTA | pair F1 | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-GT sparse-overlay selector, all `target_d1_c50` | `0.739426` | `0.614382` | `0.564060` | `0.711321` | `0.467315` |
| all `target_d1_c50` | `0.739426` | `0.614382` | `0.564060` | `0.711321` | `0.467315` |
| no-GT sparse-overlay selector, per-video | `0.739044` | `0.613893` | `0.562859` | `0.709442` | `0.466476` |
| switch MCAM08 to `target_d1_c50` | `0.736953` | `0.612368` | `0.561754` | `0.713054` | `0.463422` |
| base `traj005` | `0.731491` | `0.602400` | `0.549817` | `0.699557` | `0.452878` |

Per-video proxy:

- all `target_d1_c50`: MCAM04 IDF1 `0.825351`, MCAM08 IDF1 `0.605246`.
- switch only MCAM08 to `target_d1_c50`: MCAM04 IDF1 `0.797215`, MCAM08 IDF1
  `0.648758`.
- base `traj005`: MCAM04 IDF1 `0.801389`, MCAM08 IDF1 `0.626100`.

Model-card interpretation:

- This does not complete the project goal.  It is a local two-video
  `sample_parquet_gt_same_detection_boxes` proxy, not the DS1 e2e score.
- It does show that the newer target/admission-style assignment sources create
  real no-anchor diversity that the earlier full-data source-switch pool lacked.
- The no-GT sparse-overlay selector is the first production-shaped version of
  this idea: it selected `target_d1_c50` from assignment provenance alone and
  matched the best local oracle source-switch score.
- The next DS1 experiment should rerun assignment-level source switching with
  the current best source plus any available high-precision/verifier/cannot-link
  sources, and evaluate the no-GT selector separately from the GT diagnostic
  oracle.

Full-data selector follow-up:

- remote artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_source_selector_quality060_20260619.json`
- local copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_assignment_source_selector_quality060_20260619.json`
- no-GT sparse-overlay selector chose `current` for all videos;
- full IDF1 / HOTA:
  `0.652624 / 0.516393`
- tracklet-pair F1 / precision / recall:
  `0.768743 / 0.813273 / 0.728836`

Interpretation:

- The sample positive result did not transfer to the available historical
  full-data assignment pool.  Selection over old sources is not enough.
- The next model-side generator should create full-data target/admission-style
  sources analogous to the sample `target_d1_c50` family, then rerun the
  no-GT selector.

## Stateful Calibration Scaffold - 2026-06-19

The next no-anchor model iteration is now scaffolded as a component-state
policy sweep rather than another embedding or clustering replacement.

Script:

`kit/no_anchor_assignment_state_policy_sweep.py`

It keeps the current forced-delivery assignment as the base and computes
component states without anchors or GT identity labels:

- `committed`: sufficiently large, clean, non-conflicted components;
- `provisional`: usable components with weaker support;
- `pending`: very small components with insufficient evidence;
- `forced_conflict`: components containing same-stream temporal cannot-link
  conflicts.

The first planned sweep tests whether state-aware policies such as
`color_forced`, `singleton_forced`, `drop_forced`, `color_pending_forced`, or
`singleton_pending_forced` can improve the current assignment before applying the
existing q03 submission-level detection filter.  GT remains evaluation-only:
state assignment uses component metadata, tracklet quality, detection
confidence/area, video/camera span, and temporal cannot-link evidence.

The important new repair is cannot-link graph coloring.  When a component has
same-stream temporal conflicts, the `color_*` policies split only the conflicting
substructure into compatible color groups.  This keeps more usable evidence than
the earlier `singleton_*` policies, which turn every conflicted tracklet into a
one-tracklet identity.

Validated locally:

- `python kit/no_anchor_assignment_state_policy_sweep.py --self-test`
- `python -m py_compile kit/no_anchor_assignment_state_policy_sweep.py kit/no_anchor_submission_pervideo_filter_oracle.py`
- `bash -n kit/run_no_anchor_remote_recovery_experiments.sh`

## Local Sample State-Policy Audit - 2026-06-19

Added a local/offline auditor:

`kit/sample_assignment_state_policy_sweep.py`

This script mirrors the no-anchor state policy sweep on sample parquet
assignments when the remote gallery DB is unavailable.  It uses local sample
GT columns only for evaluation and labels the optional HOTA/IDF1 score as a
`sample_parquet_gt_same_detection_boxes` proxy.

Key results on the MCAM04/MCAM08 Tc6 local sample:

- Full-coverage no-anchor artifact
  `no_anchor_sample_osnet_traj005_nfc_k8_e075_best_20260617` reached sample
  proxy IDF1 `0.731491`, but MCAM08 remained weak at IDF1 `0.626100`.
- Precision-sorted `singleton_forced` reached pair precision `0.971753`, but
  recall collapsed to `0.001513` and sample proxy IDF1 fell to `0.238364`.
- High-precision admitted artifact `no_anchor_sample_osnet_target_m3_tmp_d10_c0p5`
  had pair precision `0.832166`, but sample proxy IDF1 was only `0.644253`.

Conclusion:

- The local sample proves that the no-anchor model can exceed 70 on a
  two-camera proxy, but this is not the DS1 end-to-end gate.
- State/color/singleton repair is a precision-control tool, not the main path to
  end-to-end >70.  The next full-data attempt should focus on weak-camera
  coverage and upstream relinking/admission, especially MCAM08-like failure
  slices.

This implements the Open-World ReID/Persistent Entity Resolution framing more
faithfully than a single forced global-ID column: every output identity can
carry status and provenance, and delivery policies can be evaluated separately
from evidence construction.

## Full-DS1 Target-Source Verdict - 2026-06-19

The Deli AutoResearch/self-play material was distilled into an execution rule:
every new branch must generate a concrete artifact, evaluate honestly, and stop
when a repeated mechanism fails.  The useful pieces for this no-anchor global-ID
model are:

- persist state to files, not chat memory;
- separate proposer from evaluator;
- accept score decreases as evidence;
- after a repeated failure, change the generated artifact rather than retuning
  the same selector.

Sources:

- `https://victorchen96.github.io/auto_research/framework.html`
- `https://victorchen96.github.io/auto_research/paper.html`
- `https://victorchen96.github.io/blog_self_play_story.html`

New full-data target source generator:

- script:
  `/Users/zcai/Codex/vlincs_reid_by_search/kit/export_no_anchor_target_agglom_source.py`
- sparse source summary:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_agglom_sparse_sources_quality060_20260619.json`

Best sparse target source by post-hoc pair evidence:

| source | rows | components | pair F1 | pair P | pair R |
|---|---:|---:|---:|---:|---:|
| `target_t640_d10_c0p75` | `3498` | `162` | `0.732927` | `0.785117` | `0.687244` |
| `target_t640_d5_c0p75` | `3521` | `164` | `0.732903` | `0.785076` | `0.687231` |
| `target_t640_d1_c0p75` | `3549` | `173` | `0.732888` | `0.785078` | `0.687204` |

No-GT selector follow-up:

- script:
  `/Users/zcai/Codex/vlincs_reid_by_search/kit/no_anchor_assignment_source_selector.py`
- conservative selector local copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_assignment_source_selector_target_agglom_sparse_overlay_rerun_20260619.json`
- balanced selector local copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_assignment_source_selector_target_agglom_sparse_balanced_20260619.json`

| model/policy | full IDF1 | HOTA | pair F1 | pair P | pair R | production verdict |
|---|---:|---:|---:|---:|---:|---|
| base current | `0.652624` | `0.516393` | `0.768743` | `0.813273` | `0.728836` | keep |
| conservative global sparse overlay | `0.647694` | `0.511262` | `0.760766` | `0.800733` | `0.724599` | reject |
| conservative per-video sparse overlay | `0.649623` | `0.512975` | `0.763389` | `0.804257` | `0.726474` | reject |
| balanced global sparse overlay | `0.618557` | `0.482834` | `0.719390` | `0.744874` | `0.695591` | reject |
| balanced per-video sparse overlay | `0.623119` | `0.487056` | `0.725458` | `0.752790` | `0.700041` | reject |

Per-video confidence oracle:

- local copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_quality060_pervideo_conf_oracle_20260619.json`
- combined IDF1 / HOTA / AssA / DetPr / DetRe:
  `0.654800 / 0.518170 / 0.533917 / 0.762451 / 0.573786`

Current model-card status:

- Model-side pair metrics remain above the 70 target with the current no-anchor
  assignment: pair `F1/P/R = 0.768743 / 0.813273 / 0.728836`.
- End-to-end DS1 remains below the 70 target.  The best verified score in this
  continuation is the GT-selected per-video confidence oracle at IDF1
  `0.654800`; the production no-GT base remains `0.652624`.
- Sparse target-agglomeration is not a production model by itself.  It is useful
  evidence for verifier/merge/split decisions, but replacing sparse subsets of
  predicted IDs breaks full delivery.

Naive full-delivery target-agglomeration check:

- local copy:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_agglom_full_delivery_t640_20260619.json`
- configuration:
  `target_clusters=640`, `output_min_conf=-1.0`, `output_min_dets=1`
- coverage:
  all `9734` resolve tracklets
- tracklet-pair F1 / precision / recall:
  `0.486063 / 0.554703 / 0.432540`

This branch was stopped at the cheap pair gate because it is far worse than the
current base pair `0.768743 / 0.813273 / 0.728836`.

Next research branch:

- Attach target-agglomeration evidence as a verifier inside the current
  assignment, using it for merge/split/admission decisions rather than raw
  namespace replacement.
- Stop tuning sparse-overlay selector heuristics unless the source itself
  changes from sparse replacement to full-delivery or verifier-only evidence.
- Do not use raw dense target agglomeration as a standalone full-delivery
  global-ID model without calibration.

## Target-Agglomeration Teacher Evidence - 2026-06-19

This follow-up tested the verifier-only version of the target branch: keep the
current best forced assignment fixed, then ask whether sparse target sources can
vote for additional component merges.

Artifacts:

- script:
  `kit/no_anchor_assignment_teacher_consensus_merge_sweep.py`
- pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_target_teacher_merge_pair_20260619.json`
- pair CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_target_teacher_merge_pair_20260619.csv`
- local copies:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_teacher_merge_pair_20260619.json`
  and
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_teacher_merge_pair_20260619.csv`

Setup:

- base assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv`
- teachers:
  six sparse target-agglomeration outputs,
  `target_t640_d1/d5/d10_c0p75` and
  `target_t1280_d1/d5/d10_c0p75`
- teacher edge count:
  `2016`
- component cannot-link conflicts:
  `1588`
- no anchors and no GT training signal.

Metrics:

| model/policy | pair F1 | pair precision | pair recall | accepted edges |
| --- | ---: | ---: | ---: | ---: |
| current base | `0.768743` | `0.813273` | `0.728836` | `0` |
| best target-teacher merge | `0.768743` | `0.813273` | `0.728836` | `1` |
| largest accepted target-teacher merge | `0.768743` | `0.813272` | `0.728836` | `6` |

Model-card interpretation:

- The target teachers are diagnostic evidence, not a production improvement.
- They produce candidate edges, but after teacher-vote, visual, conflict, and
  component-size gates, accepted edges are too few or too low-mass to move the
  tracklet-pair metric.
- This closes both direct target-overlay and target-teacher-merge as current
  production paths.
- The model-side no-anchor pair score remains above the 70 target, but the
  end-to-end DS1 target remains unsolved: current best submission-level IDF1 is
  still about `0.654739`, not `0.70+`.
- Next structural direction: generate stronger no-anchor positives for a
  calibrated edge verifier, especially from intra-tracklet self-play
  augmentations, stable clothing/body-part evidence, face-gated positives, and
  hard negatives from cannot-link/co-occurrence.

## Oracle Repair Decomposition And Self-Play Refutation - 2026-06-19

New eval-only decomposition:

- script:
  `kit/no_anchor_oracle_repair_decomposition.py`
- pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_oracle_repair_decomposition_paironly_20260619.json`
- top-1 full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_oracle_repair_decomposition_full_top1_20260619.json`

Key diagnostic:

| artifact | pair F1 | full IDF1 | interpretation |
| --- | ---: | ---: | --- |
| current no-anchor production best | `0.768743` | `0.654739` | still best production artifact |
| eval-only split top 40 + merge top 40 repair | `0.996040` | `0.705997` | near-oracle target shape, not production |
| all-current-tracklet GT-majority oracle | `1.000000` | `0.711353` | current-tracklet upper bound |

This shows that `>0.70` is still theoretically feasible with the current
tracklet source, but only with near-oracle global-ID association.  The repair
that crosses `0.70` is dominated by false-split recovery: large identities such
as GT `9`, `36`, `11`, and `43` are spread across many predicted components.

New no-anchor self-play verifier:

- script:
  `kit/no_anchor_assignment_selfplay_component_merge_sweep.py`
- normal pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_selfplay_component_merge_pair_20260619.json`
- low-threshold sanity JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_selfplay_component_merge_lowthr_pair_20260619.json`

Setup:

- no anchors and no GT training signal;
- pseudo positives are generated by splitting current components into
  camera/video/temporal/even-odd subcomponents;
- pseudo negatives come from cannot-link and low-score candidate edges;
- primary feature is fused FaceNet/color/OSNet-s7, with DB, FaceNet, and OSNet
  verifier views.

Result:

- pseudo positives / negatives:
  `158 / 4448`;
- pseudo train AP/AUC:
  `1.0 / 1.0`;
- normal verifier accepted edges:
  `0`;
- low-threshold sanity accepted edges:
  `6`;
- best pair remains unchanged:
  `0.768743 / 0.813273 / 0.728836`.

Model-card interpretation:

- Splitting existing predicted components is not a sufficient self-play
  positive generator.  It creates positives that the classifier can memorize,
  but the learned distribution does not transfer to real inter-component merge
  edges.
- The next no-anchor positive source must come from actual image evidence:
  intra-tracklet crop augmentations, body-part/clothing stability, face-gated
  positives, or multimodal attributes.  Assignment-namespace reuse and
  current-component splits are now both refuted as sufficient positive sources.

## Split And Image-Positive Continuation - 2026-06-19

This continuation applied the Deli AutoResearch rule that a branch must either
move a scalar gate or create a reusable refutation.

Split-then-merge experiment:

- script:
  `kit/no_anchor_assignment_split_then_merge_sweep.py`
- JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_split_then_merge_pair_20260619.json`

Metrics:

| artifact | pair F1 | precision | recall |
| --- | ---: | ---: | ---: |
| current base | `0.768743` | `0.813273` | `0.728836` |
| best split-then-merge | `0.633061` | `0.840151` | `0.507874` |

Interpretation:

- Whole-component cannot-link coloring is not a production approximation of the
  oracle split.  It preserves/raises precision but destroys identity recall.
- Future split behavior should be expressed as states and provenance
  (`committed`, `provisional`, `pending`, `forced`), not as hard relabeling of
  thousands of tracklets.

Sample-level positive verifier:

- extractor artifact:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s7_true_samples_20260619.npz`;
- verifier script:
  `kit/no_anchor_sample_positive_edge_verifier.py`;
- max-probability JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_sample_positive_edge_verifier_pair_20260619.json`;
- top-mean JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_sample_positive_edge_verifier_topmean_pair_20260619.json`.

Metrics:

| artifact | pair F1 | precision | recall | accepted merges |
| --- | ---: | ---: | ---: | ---: |
| current base | `0.768743` | `0.813273` | `0.728836` | `0` |
| max sample probability | `0.768734` | `0.813252` | `0.728836` | `29` |
| top-mean sample probability | `0.768743` | `0.813273` | `0.728836` | `27` |

Interpretation:

- The sample-positive verifier is a useful asset and reusable code path, but it
  does not improve the global-ID model yet.
- Same-tracklet crop positives are not enough; the next model must generate
  cross-tracklet positives from no-GT evidence such as short-gap continuations,
  body-part/clothing consistency, or generated pose variants.
- Current model-only gate remains passed, but the end-to-end `>0.70` target is
  still open.

## Cross-Tracklet Positive Continuation - 2026-06-19

The next no-anchor positive family used short-gap same-stream continuation
pairs rather than same-tracklet crop splits.

Artifacts:

- verifier:
  `kit/no_anchor_continuation_positive_edge_verifier.py`;
- verifier pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_continuation_positive_edge_verifier_pair_20260619.json`;
- peel-merge repair:
  `kit/no_anchor_continuation_peel_merge_sweep.py`;
- peel-merge pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_continuation_peel_merge_pair_20260619.json`.

Metrics:

| artifact | pair F1 | precision | recall |
| --- | ---: | ---: | ---: |
| current base | `0.768743` | `0.813273` | `0.728836` |
| continuation-positive verifier | `0.768740` | `0.813266` | `0.728837` |
| continuation peel-merge | `0.768597` | `0.813240` | `0.728601` |

Interpretation:

- Short-gap continuation positives are a better training signal than
  intra-tracklet positives: the train AUC/AP is `0.936908 / 0.906549`, not a
  trivial `1.0 / 1.0`.
- The signal is not yet production useful.  Direct merges are slightly
  negative, and local peel-merge repairs are more negative.
- The model-only no-anchor gate remains passed; the end-to-end `>0.70` gate
  remains unsatisfied.

## Clothing/Body Consistency Continuation - 2026-06-19

This continuation distilled the Deli AutoResearch rule "pivot structure, not
tactics" into a new evidence family: body-part/color consistency.  The branch
tested both merge-side use and split-side quarantine use.

Artifacts:

- clothing verifier:
  `kit/no_anchor_clothing_positive_edge_verifier.py`;
- conflict quarantine:
  `kit/no_anchor_clothing_conflict_quarantine_sweep.py`;
- pair JSONs:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_clothing_positive_edge_verifier_pair_20260619.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_clothing_positive_edge_verifier_strict_pair_20260619.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_clothing_conflict_quarantine_pair_20260619.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_osnet_conflict_quarantine_pair_20260619.json`.

Metrics:

| artifact | pair F1 | precision | recall | action |
| --- | ---: | ---: | ---: | --- |
| current base | `0.768743` | `0.813273` | `0.728836` | none |
| loose clothing verifier | `0.768689` | `0.813145` | `0.728842` | `17` merges |
| strict clothing verifier | `0.768740` | `0.813266` | `0.728837` | `1` merge |
| clothing quarantine | `0.768743` | `0.813273` | `0.728836` | `0` splits |
| OSNet-only quarantine | `0.768658` | `0.813672` | `0.728364` | `7` splits |

Interpretation:

- Clothing/body-part features produce plausible no-GT weak labels, but direct
  merges still add predicted-pair mass faster than true-pair mass.
- Conflict quarantine reveals why: many same-stream conflict nodes remain very
  close to their component centroids under pose/color features, so visual
  outlier peeling is not enough.
- This evidence family should remain in the model as provenance/calibration,
  not as a standalone global-ID resolver.

Next model direction:

- stop editing the current components with deterministic merge/split rules;
- build stronger no-GT positives from generated pose/body variants or
  cross-video attribute agreement with hard negatives;
- expose stateful outputs (`committed`, `provisional`, `forced`) before
  forced-delivery scoring so high-risk components are not silently treated as
  equally reliable IDs.

## Visual Edge Verifier Update - 2026-06-19

Latest tested branch:

- multiview component edge scoring plus exported raw-frame montages;
- conservative visual verifier decisions over `30` candidate component pairs;
- constrained component merge, no-forbidden diagnostic merge, and sampled
  micro-component surgery.

Artifacts:

- `kit/export_no_anchor_candidate_edge_montages.py`;
- `kit/apply_no_anchor_visual_edge_decisions.py`;
- `kit/apply_no_anchor_visual_edge_surgery.py`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_decision_merge_pairfull_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_decision_merge_no_forbidden_pairfull_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_surgery_pair_20260619.json`.

Metrics:

| branch | pair F1 | precision | recall | full IDF1 | promoted |
| --- | ---: | ---: | ---: | ---: | --- |
| verifier-split base | `0.768837` | `0.813577` | `0.728761` | `0.652806` | no |
| multiview merge | `0.768847` | `0.813561` | `0.728792` | `0.652812` | no |
| visual verifier constrained | `0.768837` | `0.813577` | `0.728761` | `0.652806` | no |
| visual verifier no-forbidden diagnostic | `0.768927` | `0.813579` | `0.728921` | `0.652830` | no |
| sampled surgery | `0.767834` | `0.813322` | `0.727164` | not scored | no |

Decision:

- Do not promote the visual verifier branch.  It shows a small positive signal
  only when cannot-link constraints are disabled, and it remains below the
  current promoted e2e artifact.
- The active production artifact remains the q03 submission/detection filter.
- The model card status remains: model-only global-ID gate passed; end-to-end
  pipeline gate not passed.

Next viable model step:

- Build a stateful identity resolver around `committed`, `provisional`,
  `pending`, and `forced` components.
- Use visual evidence as provenance for candidate subclusters, not as direct
  full-component merge permission.

## Conflict Subcluster Update - 2026-06-19

Latest tested branch:

- only inspect large components with same-stream cannot-link conflicts;
- extract small visual subclusters using OSNet s7, posecolor, colorhist, and
  CLIP/DINO feature agreement;
- evaluate the resulting forced assignment and the standard q03-style detection
  filter.

Artifacts:

- `kit/no_anchor_assignment_conflict_subcluster_sweep.py`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_full1_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_detection_filter_20260619.json`.

Metrics:

| branch | pair F1 | precision | recall | full IDF1 | q03/filter IDF1 | promoted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base assignment | `0.768743` | `0.813273` | `0.728836` | approx `0.6526` | `0.654739` | yes, previous q03 |
| broad conflict subcluster | `0.767262` | `0.818155` | `0.722329` | not scored | not scored | no |
| conservative conflict subcluster | `0.768756` | `0.814231` | `0.728091` | `0.652924` | `0.653081` | no |

Decision:

- Do not promote this branch.  It gives a tiny model-side pair-F1 lift and a
  useful precision diagnostic, but it still underperforms the current promoted
  end-to-end artifact.
- Keep conflict subclusters as provenance for future candidate retrieval and
  state calibration.

## Soft Cannot-Link Update - 2026-06-19

Latest tested branch:

- soften same-stream temporal cannot-link only when overlapping tracklets look
  like duplicate tracks of the same person;
- duplicate-track evidence is no-GT: same-frame bbox median IoU and OSNet
  visual similarity;
- merge current assignment components with multiview evidence after replacing
  hard cannot-link with the softened forbidden set.

Artifacts:

- `kit/no_anchor_assignment_provisional_relink_sweep.py`;
- `kit/no_anchor_assignment_soft_overlap_merge_sweep.py`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_provisional_relink_narrow_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_narrow_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_relaxed_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_relaxed_best_assignments_20260619.csv`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_narrow_detection_filter_20260619.json`.

Metrics:

| branch | pair F1 | precision | recall | full/filter IDF1 | promoted |
| --- | ---: | ---: | ---: | ---: | --- |
| base assignment | `0.768743` | `0.813273` | `0.728836` | `0.654739` current q03 | yes, previous |
| provisional relink | `0.767775` | `0.817525` | `0.723733` | `0.652265` full | no |
| soft-overlap narrow | `0.768875` | `0.813287` | `0.729062` | `0.652880` hard_q03 | no |
| soft-overlap relaxed best | `0.768878` | `0.813293` | `0.729063` | not full-scored | model-side candidate only |

Soft-overlap details:

- cached `174,788` same-stream overlap pairs;
- mean common frames per overlap pair: `17.208515`;
- best rule softened `1,978` pairs with median IoU `>=0.50` and OSNet
  similarity `>=0.80`;
- accepted `34` to `35` component edges depending on the merge scorer.

Interpretation:

- This validates that hard temporal cannot-link was over-constraining duplicate
  overlapping tracklets.
- The signal is model-positive (`+0.000135` pair F1) and should be retained in
  candidate generation.
- As a post-hoc component merge, it is still far from the e2e target because
  the remaining large errors require high-mass split/merge repair, not only
  duplicate-overlap repair.

Current status:

- model-side global-ID gate remains passed, with the best no-anchor pair F1 now
  `0.768878`;
- end-to-end gate remains not passed; current promoted full IDF1 is still
  `0.654739`.

## Soft-Overlap Weak-Positive Verifier - 2026-06-19

AutoResearch distillation used for this continuation:

- use file-backed state and reports as the durable research memory;
- treat every new idea as proposer/opponent/evaluator: propose a weak-label
  source, let metrics and counterexamples refute it, and keep negative results;
- pivot structurally after a repeated local failure instead of widening the same
  threshold grid.

New model branch:

- `kit/no_anchor_soft_overlap_weak_positive_verifier.py`;
- positives are same-stream overlap pairs with high same-frame bbox IoU and
  consistent OSNet/sample/posecolor/colorhist evidence;
- negatives are same-stream overlap pairs that remain hard cannot-link evidence
  despite high visual similarity;
- GT is used only after prediction for metrics.

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_weak_positive_verifier_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_weak_positive_verifier_maxprob_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_weak_positive_verifier_logreg_pair_20260619.json`.

Weak-label evidence:

- positive labels: `1,425`;
- negative labels: `12,000`;
- HGB train AUC/AP: `0.996519 / 0.968380`;
- logreg train AUC/AP: `0.985055 / 0.889719`;
- positive example: median IoU `0.997917`, visual `0.929251`, sample top-mean
  `0.951195`;
- negative example: median IoU `0.080109`, visual `0.974209`, sample top-mean
  `0.859968`.

Metrics:

| branch | pair F1 | precision | recall | accepted edges |
| --- | ---: | ---: | ---: | ---: |
| base assignment | `0.768743` | `0.813273` | `0.728836` | n/a |
| direct soft-overlap rule | `0.768878` | `0.813293` | `0.729063` | `34` |
| weak verifier, HGB top-mean | `0.768873` | `0.813293` | `0.729054` | `27` |
| weak verifier, HGB max-prob | `0.768812` | `0.813083` | `0.729114` | `37` |
| weak verifier, logreg top-mean | `0.768867` | `0.813292` | `0.729044` | `40` |

Verdict:

- The weak-positive source is valid because all learned verifier variants beat
  the raw base assignment.
- It is not better than the direct soft-overlap rule, so the promoted
  model-side best remains `0.768878`.
- The useful lesson is calibration: visually similar overlap pairs can be hard
  negatives when bbox IoU is low.  Future resolvers should expose this as
  evidence and state, not only as a binary merge decision.

## Visual-Edge Delivery Filter Check - 2026-06-19

The agentic visual-edge branch remains useful as model-side evidence:

- pair F1/P/R: `0.768927 / 0.813579 / 0.728921`;
- accepted no-GT visual edges: `16`;
- unfiltered full IDF1: `0.652830`.

Delivery filters were tested on the same assignment:

| config | full IDF1 | HOTA | AssA | DetPr | DetRe |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | `0.652830` | `0.516640` | `0.532301` | `0.755991` | `0.574442` |
| q02 | `0.652144` | `0.515158` | `0.531103` | `0.767645` | `0.566854` |
| q03 | `0.649590` | `0.511597` | `0.527800` | `0.777206` | `0.557972` |
| q04 | `0.644982` | `0.505620` | `0.522148` | `0.787198` | `0.546288` |
| q05 | `0.637395` | `0.496176` | `0.513075` | `0.798600` | `0.530341` |

Conclusion:

- The visual-edge branch is not bottlenecked by low-confidence detection rows.
- The e2e loss is structural: the resolver needs split-before-merge state for
  impure large components, not more delivery filtering.

Cross-assignment reuse check:

- Applying the same no-GT visual decisions directly to the current promoted
  assignment gives pair F1/P/R
  `0.768833 / 0.813275 / 0.728996` with `16` accepted edges.
- This is above the current base but below the verifier-split visual branch
  (`0.768927`) and below the soft-overlap model-side best (`0.768878`).
- Therefore the visual decisions are state-dependent evidence.  They should be
  used inside a split-state resolver, not as a reusable merge patch.

## Latest No-Anchor Model Diagnostic - 2026-06-19

The latest model-side diagnostic winner is now component-scale softcut
splitting, not a production promotion:

- script:
  `kit/no_anchor_assignment_softcut_split_sweep.py`;
- pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_pair_20260619.json`;
- full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_promoted_filters_20260619.json`;
- assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_best_assignments_20260619.csv`.

Mechanism:

- within large current predicted IDs, cluster tracklets into visual modes using
  OSNet s7, DB-view, pose/color, and color histogram evidence;
- treat same-stream temporal conflicts as soft penalties;
- accept splits using no-GT evidence: conflict reduction, visual margin, and
  minimum part size/fraction;
- use GT only after prediction for pair/full metrics.

Metrics:

| branch | pair F1 | precision | recall | full IDF1 | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| current production assignment | `0.768743` | `0.813273` | `0.728836` | `0.654739` q03 | production best |
| direct soft-overlap rule | `0.768878` | `0.813293` | `0.729063` | `0.652884` q03 relaxed full | model-side only |
| NMS-singleton relaxed | `0.768930` | `0.813789` | `0.728759` | `0.652845` q03 | model-side diagnostic only |
| softcut, 1 component | `0.769135` | `0.814237` | `0.728768` | `0.652924` q03 | rejected |
| softcut, 2 components | `0.769171` | `0.814853` | `0.728340` | `0.653050` q03 | rejected |
| softcut relaxed | `0.769668` | `0.815981` | `0.728329` | `0.653205` q03 | former diagnostic best |
| softcut then soft-overlap | `0.769661` | `0.815773` | `0.728482` | `0.653225` q03 | rejected |
| softcut then soft-overlap density selector | n/a | n/a | n/a | `0.655240` | current no-GT full best |
| target-localized repair | `0.769721` | `0.815785` | `0.728581` | `0.653297` density selector | former diagnostic only |
| multi-edge target-localized repair | `0.769760` | `0.815788` | `0.728648` | `0.653311` density selector | latest diagnostic only |
| previous production density selector | n/a | n/a | n/a | `0.654800` | confirms filter heuristic |

Interpretation:

- The global-ID model-only gate is still comfortably above 0.70, and the latest
  diagnostic pair F1 is `0.769760`.
- The end-to-end pipeline is still below 0.70; the best verified production
  artifact is now IDF1 `0.655240`.
- The pair/full mismatch is now reproduced across visual-edge, soft-overlap,
  NMS-singleton, and component-scale softcut branches.  Future work should
  optimize detection-weighted IDF1/HOTA directly, or learn a no-GT split and
  per-video filter acceptor whose reward is closer to full submission behavior
  than pure tracklet-pair F1.
- The softcut+soft-overlap per-video oracle exposed a small delivery-filter
  lift, and the `density_oracle_lite` no-GT selector reproduced that threshold
  pattern from row density and confidence quantiles.  It selects light
  confidence filters on MCAM03 Tc6, MCAM04 Tc6, MCAM06 Tc8, and MCAM08 Tc6
  while leaving the other videos unfiltered.

## AutoResearch Self-Play Continuation - 2026-06-20

The Deli AutoResearch framework was distilled into an execution protocol for
this no-anchor global-ID work:

- keep durable state and artifacts on disk;
- separate proposer, opponent, and evaluator roles;
- execute ready experiments instead of waiting for interaction;
- record negative score movement as evidence, not as failed bookkeeping.

For VLINCS, the mapped self-play loop is:

- proposer: a no-GT identity edit, state policy, or feature source;
- opponent: cannot-link, same-stream temporal conflict, density, and
  cross-view disagreement evidence;
- evaluator: post-hoc pair and full submission metrics, with GT used only
  after prediction for scoring and diagnostics.

Two new branches were executed under that protocol.

| branch | artifact | pair F1 | precision | recall | decision |
| --- | --- | ---: | ---: | ---: | --- |
| edge-table identity-island merge | `no_anchor_edge_table_island_merge_focused_pair_20260620.json` | `0.769698` | `0.815763` | `0.728558` | below diagnostic gate |
| DINOv2-base s1 component merge | `no_anchor_dinobase_s1_component_merge_pair_20260620.json` | `0.769661` | `0.815773` | `0.728482` | no-op |
| fused + DINO 0.03 | `no_anchor_fused_dinobase003_s1_component_merge_pair_20260620.json` | `0.769661` | `0.815773` | `0.728482` | no-op |
| fused + DINO 0.05 | `no_anchor_fused_dinobase005_s1_component_merge_pair_20260620.json` | `0.769661` | `0.815773` | `0.728482` | no-op |
| fused + DINO 0.05 loose sanity | `no_anchor_fused_dinobase005_s1_component_merge_loose_pair_20260620.json` | `0.769661` | `0.815773` | `0.728482` | no useful gain |

The DINOv2-base extraction itself is valid and no-anchor:

- full feature artifact:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_dinov2base_s1_20260620.npz`;
- model: `facebook/dinov2-base`;
- coverage: `9734` tracklets, feature dim `768`;
- missing video rows: `0`;
- missing crop rows: `0`.

No branch passed the current pair gate `> 0.769760`, so no new full submission
artifact was promoted.  The promoted no-anchor production state remains:

- assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`;
- full/e2e zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.zip`;
- full IDF1/HOTA/AssA/DetPr/DetRe:
  `0.655240 / 0.518652 / 0.534359 / 0.763322 / 0.573970`.

Interpretation:

- adding another global visual embedding is not enough in the current resolver;
- small island merging is a local optimum and does not close the e2e gap;
- the next model needs to act at component or identity-island state level:
  predict repairability/admission first, then perform localized split/merge
  decisions and emit committed/provisional/forced states.

## Conflict-State Repair Update - 2026-06-20

Additional no-anchor experiments were run after the DINO/island branch.

New temporal relink script:

- `kit/no_anchor_assignment_video_temporal_relink_sweep.py`.

Temporal relink result:

- strict global-component temporal relink:
  pair F1/P/R `0.769661 / 0.815773 / 0.728482`, `0` accepted edges;
- loose global-component temporal relink:
  pair F1/P/R `0.769661 / 0.815773 / 0.728482`, `1` accepted edge;
- diagnostic video-local output full IDF1:
  `0.366217`, so video-local relabeling is invalid for this scorer.

Conflict-state repair result:

| branch | pair F1 | precision | recall | full IDF1 | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| production softcut+soft-overlap | `0.769661` | `0.815773` | `0.728482` | `0.655240` | keep production |
| singleton forced-conflict split | `0.771670` | `0.822651` | `0.726639` | `0.646655` | model-side only |
| color forced-conflict split | `0.770599` | `0.818962` | `0.727630` | `0.652774` | rejected |
| color split + multiview remerge | `0.770667` | `0.818533` | `0.728090` | `0.652744` | rejected |

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_top1_full_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_color_forced_full_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_state_color_forced_multiview_merge_top1_full_20260620.json`.

Current status:

- The best verified no-anchor global-ID model-side score is now pair F1
  `0.772654`.
- The best verified no-anchor end-to-end artifact remains full IDF1 `0.655240`:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.zip`.
- Direct conflict splitting is not the deployable answer.  It is useful as a
  provisional evidence state, but the final resolver must repair and reconnect
  those conflict parts before forced delivery.

## Conflict Subcluster Reassign Update - 2026-06-20

New no-anchor model-side best:

- script: `kit/no_anchor_assignment_conflict_reassign_sweep.py`;
- pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_pair_20260620.json`;
- full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_top1_full_20260620.json`;
- density-filter artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_density_filter_selector_zip_20260620.json`.

Result:

| branch | pair F1 | precision | recall | full IDF1 | density IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict conflict reassign | `0.772654` | `0.818667` | `0.731538` | `0.653823` | `0.654037` |

This branch is a better global-ID model because it uses conflict split evidence
only as provisional evidence, then reassigns one clean subcluster into an
existing target component.  It is still not the production artifact because the
best no-GT end-to-end delivery remains IDF1 `0.655240`.

## Source-Island G8 Reassign Update - 2026-06-20

The latest no-anchor model-side best is now the loose source-island candidate
search with a strict target gate and a max source island size of 8 tracklets.

Artifacts:

- source audit script:
  `kit/no_anchor_conflict_source_island_audit.py`;
- pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_pair_20260620.json`;
- full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json`;
- assignment artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_top1_assignments_20260620.csv`;
- density-filter artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_density_filter_selector_zip_20260620.json`.

Result:

| branch | pair F1 | precision | recall | full IDF1 | density IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict conflict reassign | `0.772654` | `0.818667` | `0.731538` | `0.653823` | `0.654037` |
| loose source + strict target | `0.772127` | `0.817025` | `0.731906` | not scored | not scored |
| loose source, max group 8 + strict target | `0.775234` | `0.820504` | `0.734698` | `0.653541` | `0.653681` |

Accepted edits:

- accepted reassignments: `4`;
- moved tracklets: `32`;
- candidate edges: `7`;
- source groups:
  `2232, 2270, 2308, 2374, 2415, 2452, 2488, 2553`;
  `4406, 4461, 4513, 4557, 4619, 4658, 4744, 4790`;
  `633, 764, 977, 1006, 1044, 1078, 1102, 1120`;
  `8305, 8540, 8914, 8972, 9013, 9050, 9130, 9247`.

Decision:

- This is the new best global-ID model-side diagnostic result.
- It is still rejected for production because both full scorers stay below the
  standing end-to-end artifact at IDF1 `0.655240`.
- The remaining model gap is delivery-aware admission: pair-score gains are
  real, but full IDF1 is dominated by how the submission prices identity
  fragmentation, detection rows, and per-video row density.

## Delivery-Aware Negative Checks - 2026-06-20

The current model-side best remains:

- `loose_source_island_g8_strict_target_reassign`;
- pair F1 / precision / recall:
  `0.775234 / 0.820504 / 0.734698`;
- pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_pair_20260620.json`.

Additional checks after that model-side win:

| check | best pair F1 | best full IDF1 | verdict |
| --- | ---: | ---: | --- |
| prefix full sensitivity | `0.775234` | `0.653823` at top 1/2 edits, `0.653541` at top 4 | pair gains do not transfer |
| target-quality gate `>=0.75` | `0.775064` | `0.654009` | best branch still below production |
| fixed row filters | n/a | `0.653686` | filter mismatch is not the main gap |
| source-switch explicit policy | n/a | `0.497171` | mixed submission namespaces break IDs |
| aggressive identity-island repair | `0.769692` | not scored | below previous pair gate |

Source-switch artifact:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.json`.

Interpretation:

- The G8 reassign branch is a better global-ID model in the tracklet-pair
  diagnostic sense, but it is not a deployable e2e model.
- Whole-video source switching is explicitly rejected: replacing MCAM04 Tc6
  with `conflict_g8` and MCAM06 Tc8 with `quality` drops full IDF1 to
  `0.497171`, because global ID namespaces are not compatible across sources.
- The next model should not be another globally sorted pair-edge edit list.
  It needs a component/video-level acceptor that predicts full-score side
  effects before committing an ID rewrite, or a broader identity-component
  resolver that recovers large false-split identities inside a single namespace.
- A follow-up aggressive island repair stayed inside one namespace but only
  reached pair F1 `0.769692`, below the narrower edge-target repair
  `0.769760`; larger island attachment is therefore not the next model.

### DINO Edge-Source Target-Repair Negative Check - 2026-06-20

Protocol update:

- Added file-backed AutoResearch state under
  `autoresearch_state/no_anchor_global_id/state/`.
- Patched `kit/no_anchor_edge_table_target_repair_sweep.py` so single-source
  edge tables can use `primary_sim/primary_rank_*` as fallback evidence.

Artifacts:

- edge table:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_acceptor_table_20260620.csv`;
- corrected full scorer:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_fallback_single_full1_20260620.json`;
- assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_fallback_single_assignments_20260620.csv`.

Result:

| model | pair F1 | precision | recall | full IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| current model-side best, G8 strict target | `0.775234` | `0.820504` | `0.734698` | `0.653541` | keep as model-side best |
| fused-DINO target repair | `0.769661` | `0.815126` | `0.729001` | `0.652948` | rejected |

Conclusion:

- Fused-DINO does not currently produce a better no-anchor global-ID model when
  used as a target-repair edge source.
- It slightly increases recall but loses precision, and full IDF1 is lower than
  the standing e2e artifact.
- Next model work should target a repairability/admission model or cached edit
  scoring, not another wider edge threshold sweep.

### Pose2ID/NFC Negative Check - 2026-06-20

New utility:

`kit/make_no_anchor_nfc_features.py`

This script implements a no-anchor Neighbor Feature Centralization pass over
tracklet feature `.npz` files.  It keeps all original metadata, replaces the
`features` array, and stores `nfc_info`.  No GT, anchors, or identity seeds are
read.

NFC feature artifact:

`/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s7_true_nfc_k2_eta05_20260620.npz`

Evaluation artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_nfc_fused_osnet005_s7true_timeagglom_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_nfc_osnet_s7_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_nfc_aux_osnet_s7_pair_20260620.json`.

Results:

| model | pair F1 | precision | recall | verdict |
| --- | ---: | ---: | ---: | --- |
| current model-side best, G8 strict target | `0.775234` | `0.820504` | `0.734698` | keep |
| softcut base before NFC branch | `0.768743` | `0.813273` | `0.728836` | baseline |
| NFC time-agglom | `0.594775` | `0.720286` | `0.506515` | rejected |
| NFC softcut primary | `0.764329` | `0.815670` | `0.719068` | rejected |
| NFC softcut auxiliary | `0.767759` | `0.814164` | `0.726358` | rejected |

Decision:

- Do not promote NFC features into the global-ID model in the current form.
- The current no-anchor model best remains the loose source-island G8 strict
  target reassign result at pair F1 `0.775234`.
- The next model work should estimate edit repairability and full-delivery
  side effects, not smooth visual neighborhoods further.

### SigLIP2 Person-ReID Feature Check - 2026-06-20

Extractor update:

- `kit/extract_tracklet_foundation_features.py` now accepts
  `--processor-model` for HF models whose processor lives in a different repo.
- It also supports `get_image_features()` outputs that return pooled model
  output objects.

Feature artifact:

`/mnt/localssd/vlincs_reid_runs/ds1_tracklet_siglip2_person_reid_s1_20260620.npz`

Feature details:

- model: `MarketaJu/siglip2-person-description-reid`;
- processor: `google/siglip2-base-patch16-224`;
- coverage: `9734 / 9734` tracklets;
- feature dim: `768`;
- no anchors and no GT during extraction.

Pair-gate artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_siglip2_person_reid_s1_timeagglom_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_siglip2p003_s1_component_merge_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_siglip2p005_s1_component_merge_pair_20260620.json`.

Results:

| model | pair F1 | precision | recall | verdict |
| --- | ---: | ---: | ---: | --- |
| current model-side best, G8 strict target | `0.775234` | `0.820504` | `0.734698` | keep |
| SigLIP2 standalone time-agglom | `0.574380` | `0.640377` | `0.520716` | rejected |
| fused + SigLIP2 `0.03` component merge | `0.769661` | `0.815773` | `0.728482` | no-op |
| fused + SigLIP2 `0.05` component merge | `0.769661` | `0.815773` | `0.728482` | no-op |

Decision:

- Do not promote this SigLIP2 feature into the current global-ID model.
- It may still be kept as a future provenance feature for a learned
  repairability proxy, but it does not change identity assignments by itself.

### Weak Metric Projection Check - 2026-06-20

New module:

`kit/make_no_anchor_weak_metric_features.py`

Purpose:

- test the user's proposed weak-label route directly: generate positives from
  same-tracklet crops and short-gap continuation evidence, generate negatives
  from cannot-link overlaps, then train a no-anchor metric projection.
- output a standard `seqs/features` NPZ so the model can be inserted into the
  existing global-ID pipeline.

No-anchor contract:

- no anchors;
- no GT identities during training or feature generation;
- GT used only after predictions for pair/full metrics.

Training artifact:

`/mnt/localssd/vlincs_reid_runs/ds1_tracklet_weakmetric_osnet_s7_fused_20260620.json`

Feature artifacts:

- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620.npz`;
- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p05.npz`.

Training diagnostics:

| metric | value |
| --- | ---: |
| positives | `17555` |
| negatives | `18000` |
| projected positive cosine mean | `0.420363` |
| projected negative cosine mean | `0.096400` |
| train cosine margin | `0.323963` |

Pair gate under the current best conflict-reassign structure:

| model input | pair F1 | precision | recall | verdict |
| --- | ---: | ---: | ---: | --- |
| current G8 strict-target primary | `0.775234` | `0.820504` | `0.734698` | standing model best |
| weakmetric `w=0.02` primary | `0.775234` | `0.820504` | `0.734698` | no-op |
| weakmetric `w=0.05` primary | `0.775234` | `0.820504` | `0.734698` | no-op |

Decision:

- The weak-label metric route is viable as a training procedure, but this
  linear projection does not improve the current no-anchor global-ID model.
- Do not promote it into the model.  Keep the script for future projection or
  contrastive-loss variants.
- Next model work should target large false-split identity components directly,
  rather than adding another auxiliary feature view.

### Large False-Split Structural Probe - 2026-06-20

Tested model behavior:

- Start from the current conflict-reassign delivered assignment.
- Try structural identity edits without anchors or GT:
  pure component merge, then bulk conflict split plus merge.
- Use GT only after predictions for pair/full metrics.

Artifacts:

- input assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_top1_assignments_20260620.csv`;
- pure merge log:
  `local_runs/remote_h100_test_3_20260620/large_false_split_component_merge_best_assignment_20260620.log`;
- split micro artifact:
  `local_runs/remote_h100_test_3_20260620/no_anchor_large_false_split_split_then_merge_micro_best_assignment_20260620.json`.

Metrics:

| model variant | pair F1 | precision | recall | full IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| reloaded current assignment | `0.772727` | `0.818297` | `0.731964` | - | baseline for this probe |
| pure component merge | `0.772727` | `0.818297` | `0.731965` | `0.653541` | no-op |
| bulk conflict split | `0.618505` | `0.848465` | `0.486617` | `0.602445` | rejected |

Model-card decision:

- Do not add bulk component splitting to the current global-ID model.
- Do not rely on pure component merge as the false-split recovery mechanism.
- Keep conflict-reassign as the current model-side best, but the next model
  iteration must use smaller source-group edits and estimate full-score side
  effects before committing an ID change.

### Full-Proxy Surgical Repair Check - 2026-06-20

Model change tested:

- Add a no-GT `full_proxy` ranking mode to the surgical conflict-reassign
  solver.
- The model still uses the same candidate generator; only the expensive full
  candidates are selected by a delivery-risk proxy instead of pair F1.

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json`;
- `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json`;
- assignment output:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_fullproxy_top_assignments_20260620.csv`.

Metrics:

| variant | accepted edits | moved tracklets | pair F1 | full IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| pair-ranked current G8 row | `4` | `32` | `0.775234` | `0.653541` | not promoted |
| full-proxy row | `1` | `8` | `0.772654` | `0.653823` | not promoted |
| standing e2e best | - | - | - | `0.655240` | keep |

Model-card decision:

- Do not promote the proxy-ranked assignment yet.
- Keep the code path because it moves in the right e2e direction, but require
  diversity-aware candidate selection or a learned edit acceptor before the
  next full-score run.

### Unique Full-Proxy Candidate Check - 2026-06-20

Model change tested:

- De-duplicate accepted-edit signatures before full scoring proxy-ranked
  surgical source-group candidates.

Artifact:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_fullproxy_unique_pairfull_20260620.json`.

Metrics:

| variant | accepted edits | moved tracklets | pair F1 | precision | recall | full IDF1 | HOTA | AssA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unique full rank 1 | `1` | `8` | `0.772654` | `0.818667` | `0.731538` | `0.653823` | `0.517789` | `0.533336` |
| unique full rank 2 | `2` | `16` | `0.774082` | `0.819399` | `0.733515` | `0.653823` | `0.517789` | `0.533336` |
| unique full rank 3 | `3` | `24` | `0.774252` | `0.819890` | `0.733427` | `0.653042` | `0.516939` | `0.532540` |
| standing e2e best | - | - | - | - | - | `0.655240` | `0.518652` | `0.534359` |

Model-card decision:

- Do not promote any unique full-proxy candidate.
- Keep `full_proxy` as an analysis/evaluator tool, not as the active global ID
  model.
- The next global-ID model branch should learn an edit acceptor from proposal
  features and posthoc full outcomes, or force source-group diversity before
  spending more DS1 full-score runs.

### Prepared But Not Promoted - Diverse First-Edge Selection

Implementation:

- Added `--full-selection diverse_first_edge` to the conflict-reassign solver.

Model status:

- Not promoted and not scored yet.
- This is a candidate-evaluation strategy only; it does not change the current
  delivered global-ID model until a full DS1 run beats the standing gate.

Blocked verification:

- Local compile/self-test passed.
- h100-test-3 remote verification and run were blocked by Pluto connectivity:
  service connection failed and SSH banner exchange timed out.

### Prepared But Not Scored - Candidate Skip-Family Diversity

Implementation:

- Added `--candidate-skip-first-edge-families` to the conflict-reassign solver.
- Added `kit/run_no_anchor_false_split_diversity.sh` as the reproducible remote
  launcher.

Model status:

- Not promoted.
- Not scored on DS1 yet because all existing Pluto SSH configs currently time
  out during banner exchange and Pluto service status calls fail.
- This is aligned with the model-card objective because it explores
  alternative no-anchor repair families rather than retuning the already
  saturated top edit family.

### Prepared But Not Scored - Learned Full-Proxy Ranking

Implementation:

- Added `kit/analyze_no_anchor_full_proxy_training.py`.
- Exported:
  `local_runs/no_anchor_full_proxy_compact_ridge_model_20260620.json`.
- Added solver support for:
  `--rank-by learned_proxy --learned-proxy-json <model.json>`.
- Updated `kit/run_no_anchor_false_split_diversity.sh` to use learned-proxy
  ranking by default.

Training/evaluation status:

- Training rows are prior no-anchor proposal rows with completed posthoc full
  scoring; oracle rows are excluded from the deployable model.
- It uses no anchors and no GT at prediction time.
- Local audit: `32` rows, `29` compact features, LOOCV corr `0.996050`, MAE
  `0.000913`.

Model status:

- Not promoted.
- Needs a fresh DS1 full run after Pluto connectivity recovers.
