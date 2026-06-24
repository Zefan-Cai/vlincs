# No-anchor visual-positive subcluster CTF top2 p005 gain

- Date: `2026-06-24`
- Pipeline module: `M7+M8+M12`
- Used in pipeline: `yes: promoted as current no-anchor best after direct, density_simple and valid p005_area scoring`
- Status: `gain`
- No-anchor: `True`

## Summary

A broad visual-positive edge 87 -> 86 was previously too risky as a full merge. This iteration keeps the edge but applies a Re-ID-CTF-style top-k filter using DINOv2, SigLIP and weakmetric agreement. Only seq 5550 and 2987 move from component 87 / gid 960000351 to component 86 / gid 960000350. Canonical p005_area improves from 0.668198 to 0.668332.

## Metrics

- Baseline: `0.668198`
- Candidate: `0.668332`
- Delta: `0.000134`
- Metric name: `canonical p005_area IDF1`

## Implementation

The implementation treats generated/visual positive evidence as a proposal, not a forced identity label. A candidate edge first identifies the source and target components. The reviewer ranks source tracklets by agreement across DINOv2, SigLIP and weakmetric similarities to the target centroid. The scorer tests top1, top2, top4 and top8. Top2 is the only promoted setting because it improves direct score and the canonical delivery path; larger top-k settings regress, confirming that the correct operation is a small subcluster move, not a full merge.

## Environment

- `repo=/Users/zcai/Codex/vlincs_reid_by_search`
- `verified_publish_clone=/private/tmp/vlincs-wisc-demo-lfs.TMM8XH`
- `DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `python=/private/tmp/vlincs-wisc-demo-lfs.TMM8XH/.venv-demo/bin/python`
- `tracklets=kit/demo_data/ds1/tracklets/*/tracklets.parquet`
- `no anchors; GT used only by evaluator after assignment materialization`

## Commands

```bash
python kit/export_no_anchor_scheduler_manifest_assignments.py --scheduler-json local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/visual_positive_subcluster_ctf_candidates.json --base-assignment-csv local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/assignments/rank06_component_subset_attach_source_assignments.csv --assignment-out-dir local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/assignments --manifest-json local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/manifest_assignments.json --selection-ranks 1,2,3,4,5,6
```

```bash
PYTHONPATH=$PWD DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/assignments/rank02_visual_positive_subcluster_ctf_topk_source_assignments.csv --fallback singleton --json local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/rank02_subcluster_ctf_full_export_delivery.json --zip-out local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/rank02_subcluster_ctf_full_export.zip
```

```bash
PYTHONPATH=$PWD DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/rank02_subcluster_ctf_full_export.zip --policies density_simple --json local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/rank02_subcluster_ctf_density_simple.json --zip-out local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/rank02_subcluster_ctf_density_simple.zip
```

```bash
PYTHONPATH=$PWD DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/rank02_subcluster_ctf_density_simple.zip --config "$(cat reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/repro/input/p005_area_config.txt)" --json local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/rank02_subcluster_ctf_density_p005_area.json --zip-out local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/rank02_subcluster_ctf_density_p005_area.zip
```

```bash
bash reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/reproduce.sh
```

## Code Paths

- `kit/export_no_anchor_scheduler_manifest_assignments.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/reproduce.sh`

## Artifacts

- `reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/repro/input/rank02_visual_positive_subcluster_ctf_topk_source_assignments.csv`
- `reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/repro/input/p005_area_config.txt`
- `reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/repro/expected/rank02_full_export.json`
- `reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/repro/expected/rank02_density_simple.json`
- `reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/repro/expected/rank02_density_p005_area.json`
- `reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/repro/provenance/visual_positive_subcluster_ctf_candidates.json`
- `reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/repro/provenance/visual_positive_subcluster_ctf_ranked.csv`
- `reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/cases/rank02_top2_visual_positive_subcluster_ctf/rank02_bbox_evidence.png`

## Visual Cases

- Rank02 top2 subcluster repair: component 87 -> 86: Two moved tracklets with three sampled frames each. Shows predicted_global_id 960000351 -> 960000350 and component 87 -> 86.
  - failure: Full visual-positive merge 87 -> 86 was too broad; leaving all tracklets in 87 kept a small compatible island split from 86.
  - improvement: The CTF reviewer keeps only seq 5550 and 2987, improving direct, density and p005 delivery while preserving the rest of component 87.
  - image: `cases/rank02_top2_visual_positive_subcluster_ctf/rank02_bbox_evidence.png`
  - html: `cases/rank02_top2_visual_positive_subcluster_ctf/case.html`
  - json: `cases/rank02_top2_visual_positive_subcluster_ctf/case.json`
- Real-frame context: cross-camera global-id success: Previously extracted real-frame montage showing what an identity evidence chain should look like when raw frames are available.
  - failure: Coordinate-only panels cannot show appearance cues.
  - improvement: The package carries real-frame context next to the new coordinate-level top2 case.
  - image: `cases/context_real_frame_examples/case_success_cross_camera_m0048_real.png`
- Real-frame context: same-tracklet sequence: Previously extracted real-frame tracklet timeline used as visual reference for repeated-frame support.
  - failure: Single-frame diagnostics hide temporal consistency.
  - improvement: Timeline panels illustrate how repeated-frame evidence should be inspected for future raw-frame cases.
  - image: `cases/context_real_frame_examples/case_focus_m0012_tracklet_sequence_real.png`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| full visual-positive merge 87 -> 86 | Move all 147 source tracklets to target 86. | direct IDF1/HOTA/AssA=0.651799/0.517692/0.535597 | killed; broad false merge |
| rank01 top1 CTF | Move only seq 5550 from component 87 to 86. | direct IDF1/HOTA/AssA=0.666008/0.526982/0.537177 | tiny direct change; not enough for promotion |
| rank02 top2 CTF | Move seq 5550 and 2987 from component 87 to 86. | direct=0.666140/0.527110/0.537273; density_simple=0.668227/0.528792/0.539038; p005_area=0.668332/0.528875/0.539163 | promoted |
| rank03 top4 CTF | Move the top4 ranked source tracklets. | direct IDF1/HOTA/AssA=0.665709/0.526762/0.537061 | killed; adding more positives reverses the gain |
| rank04 top8 CTF | Move the top8 ranked source tracklets. | direct IDF1/HOTA/AssA=0.665038/0.526225/0.536759 | killed; too broad |
| rank05 top16 CTF | Start top16 broadening run after top4/top8 regressions. | interrupted before completion after lower top-k variants had already failed | not kept; no evidence to justify extra complexity |

## Upload

- Bitbucket: `pushed to Novateur/vlincs_reid_by_search branch wisc; package path reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/`
- S3: `uploaded: s3://dit-scale-up/zcai/vlincs/no_anchor_gains/20260624_visual_subcluster_ctf_top2/`

## Next

Use the Diffusion-ReID idea more literally: generate or augment positive views only for already CTF-passing small islands, then require an opponent score and direct+density+p005 verification before promotion. Do not use generated samples to justify whole-component merges.
