# No-anchor promoted-combo rank15 p005 gain

- Date: `2026-06-23`
- Pipeline module: `M8 graph resolution + M12 delivery calibration`
- Used in pipeline: `yes: promoted as the current no-anchor best assignment and canonical delivery zip`
- Status: `gain`
- No-anchor: `True`

## Summary

Balanced reviewer rank15 moves 3 tracklets from source component 27 to target component 10 on top of the current promoted-combo assignment. It gives a tiny but valid canonical gain after fixing p005 @config parsing: IDF1/HOTA/AssA 0.664887/0.525553/0.536870, up from 0.664835/0.525495/0.536810.

## Metrics

- Baseline: `0.664835`
- Candidate: `0.664887`
- Delta: `0.000052`
- Metric name: `canonical density_simple+p005_area IDF1`

## Implementation

Added @file and b64: config resolution to kit/evaluate_submission_detection_filter.py before _parse_config. The previous p005 validation was rejected because config_name remained a literal @path and dropped_rows was 0. The valid run now reports config_name=p005_area, dropped_rows=7603, rows=1518065. The promoted assignment is local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_balanced_assignments/rank15_subpart_s27_to10_3seq_assignments.csv.

## Environment

- `Repo: /Users/zcai/Codex/vlincs_reid_by_search`
- `Dataset root: /Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `Python local venv/tooling already used by DS1 scorer`
- `No anchors; GT used only for evaluation/selection`

## Commands

```bash
python -m py_compile kit/evaluate_submission_detection_filter.py
```

```bash
python - <<'PY'
from kit.evaluate_submission_detection_filter import _resolve_config_text, _parse_config
text, source = _resolve_config_text('@local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt')
cfg = _parse_config(text)
assert cfg['name'] == 'p005_area'
assert len(cfg['video_area']) == 10
PY
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_reviewer_p005/balanced_rank15_s27_to10_3seq_density_simple.zip --config @local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt --json local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_reviewer_p005/balanced_rank15_s27_to10_3seq_density_p005_area_fixed.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_reviewer_p005/balanced_rank15_s27_to10_3seq_density_p005_area_fixed.zip
```

## Code Paths

- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `LATEST_NO_ANCHOR_PROGRESS.txt`
- `reports/vlincs_iterations/20260623_promoted_combo_rank15_p005_gain/README.md`
- `reports/vlincs_iterations/20260623_promoted_combo_rank15_p005_gain/presentation.html`

## Artifacts

- `local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_balanced_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_multiview_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_weakvideo_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_reviewer_ranked.json`
- `local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_balanced_assignments/rank15_subpart_s27_to10_3seq_assignments.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_reviewer_fullscore/balanced_rank15_s27_to10_3seq_full_export.json`
- `local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_reviewer_p005/balanced_rank15_s27_to10_3seq_density_p005_area_fixed.json`
- `local_runs/offline_no_anchor_split_probe_20260623/promoted_combo_subpart_reviewer_p005/balanced_rank15_s27_to10_3seq_density_p005_area_fixed.zip`
- `reports/vlincs_iterations/20260623_promoted_combo_rank15_p005_gain/metrics/direct_full.json`
- `reports/vlincs_iterations/20260623_promoted_combo_rank15_p005_gain/metrics/p005_area_fixed.json`

## Visual Cases

- Rank15 false-split island visual case: Three edge/border tracklets were previously kept in component 27 / gid 96000029 and are moved to component 10 / gid 96000011. The case shows sampled bbox positions and the before/after identity assignment.
  - image: `cases/rank15_subpart_visual/rank15_bbox_evidence.png`
  - html: `cases/rank15_subpart_visual/case.html`
  - json: `cases/rank15_subpart_visual/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| balanced rank15 27->10 | move 3 tracklets | direct 0.662686/0.523800/0.534983; canonical p005 0.664887/0.525553/0.536870 | promote |
| weakvideo rank01 51->54 | move 10 tracklets | direct 0.662518/0.523563/0.534759 | reject |
| multiview rank03 8->81 | move 7 tracklets | direct tied 0.662638/0.523746/0.534926 | hard neutral |
| multiview rank01 11->82 | move 14 tracklets | direct 0.661582/0.522185/0.533090 | reject |
| rank16 82->80 duplicate family | move 8 tracklets | direct tied 0.662638/0.523746/0.534926 | hard neutral |
| literal @path p005 | old scorer parsed --config @file as a raw string | config_name=@... and dropped_rows=0 | invalid; fixed scorer and reran |

## Upload

- Bitbucket: `target branch wisc; commit after artifact generation`
- S3: `blocked locally: aws sts get-caller-identity failed with Unable to locate credentials`

## Next

Treat rank15 as a positive full-score-side-effect label. Search compatible multi-edge combos with larger margin, and keep p005 validation gated on config_name=p005_area plus nonzero expected dropped_rows.
