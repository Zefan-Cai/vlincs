# No-Anchor Generated-Positive Best-Crop CTF Probe

Date: 2026-06-25
Branch/commit before commit: `wisc` / `4c95d1f`
Module: M2 real-frame evidence input + M3 generated-positive admission + M5 CTF calibration
Status: no e2e promotion; keep `seq7771` best-crop path as the next GPT-image target

## Hypothesis

Generated-positive probes should not use every sampled crop from a tracklet. For occluded or edge-case tracklets, a low-quality reference crop can make the generated positive drift in DINOv2 even when SigLIP still accepts it. A best-reference-crop admission rule should improve the source CTF pass rate before spending GPT-image budget.

## Method

I added `kit/build_no_anchor_generated_positive_prompt_assets.py`, a reproducible no-anchor asset builder:

- input: current assignment CSV, DS1 tracklet parquets, raw MP4 videos, source seq, counter seqs
- output: real raw frames, bbox crops, contact sheet, `prompt_manifest.json`
- provenance: raw-video sha256, bbox xyxy, crop xyxy, tracklet key, component/global ID from the current assignment
- no anchors and no GT labels; counters are hard negatives only, never positive identity labels

Two current-best-related source tracklets were tested:

| source | video | role in current best | counters | raw-frame package |
|---:|---|---|---|---|
| 5486 | MCAM04 Tc6 | promoted post-combo5 source, comp47/gid96000051 | 4043, 5716 | `local_runs/generated_positive_pixel_inputs_20260625/seq5486_promoted_postcombo5_realframe_ctf/` |
| 7771 | MCAM08 Tc6 | promoted post-combo5 source, comp82/gid960000202 | 8315, 8336 | `local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/` |

## Results

Current canonical best remains:

| metric | value |
|---|---:|
| IDF1 | 0.669466 |
| HOTA | 0.530154 |
| AssA | 0.540397 |

This probe only scores generated-positive admission quality:

| source | generation input | DINOv2 source CTF | SigLIP source CTF | decision |
|---:|---|---:|---:|---|
| 5486 | 3 reference crops | 0/3 | 3/3 | reject for GPT-image; DINO too unstable |
| 5486 | best crop only | 0/3 | 3/3 | reject; best-crop does not rescue |
| 7771 | 3 reference crops | 1/3 | 3/3 | reject multi-crop; bad references cause DINO drift |
| 7771 | best crop only | 3/3 | 3/3 | keep as next GPT-image target |

The useful gain is local and upstream: for `seq7771`, changing the generation input from three mixed crops to the single best crop improves DINOv2 source CTF from `1/3` to `3/3` while preserving SigLIP `3/3`.

## Commands

Build real-frame prompt assets:

```bash
.venv-demo/bin/python kit/build_no_anchor_generated_positive_prompt_assets.py \
  --assignment-csv local_runs/demo_post_combo5_sig14_weak22_verify_20260625/method_reproduction/post_combo5_sig14_weak22_assignments.csv \
  --tracklet-root kit/demo_data/ds1/tracklets \
  --raw-video-root /Users/zcai/Codex/vlincs_reid_by_search/local_runs/raw_videos_s3_20260624 \
  --source-seq 7771 --counter-seq 8315 --counter-seq 8336 \
  --output-dir local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf \
  --frames-per-seq 3 --repo-root "$PWD"
```

Generate deterministic local positives:

```bash
.venv-demo/bin/python kit/generate_no_anchor_tracklet_local_positives.py \
  --prompt-manifest local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/prompt_manifest.json \
  --output-dir local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/local_aug_bestcrop_probe \
  --backend local_aug --include-seq 7771 --variants-per-seq 3 --max-reference-crops 1 \
  --repo-root "$PWD"
```

Audit with DINOv2 and SigLIP:

```bash
PY=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/deep_ctf_venv_20260624/bin/python
$PY kit/audit_no_anchor_generated_positive_ctf.py \
  --prompt-manifest local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/prompt_manifest.json \
  --generated-manifest local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/local_aug_bestcrop_probe/generated_manifest.json \
  --json local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/local_aug_bestcrop_probe/ctf_audit_dinov2_source.json \
  --feature-backend dinov2 --reference-mode source --min-same-sim 0.80 --min-margin 0.05 \
  --repo-root "$PWD"
$PY kit/audit_no_anchor_generated_positive_ctf.py \
  --prompt-manifest local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/prompt_manifest.json \
  --generated-manifest local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/local_aug_bestcrop_probe/generated_manifest.json \
  --json local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/local_aug_bestcrop_probe/ctf_audit_siglip_source.json \
  --feature-backend siglip --reference-mode source --min-same-sim 0.75 --min-margin 0.03 \
  --repo-root "$PWD"
```

## Visual Cases

- `seq5486`: `local_runs/generated_positive_pixel_inputs_20260625/seq5486_promoted_postcombo5_realframe_ctf/source_counter_contact_sheet.png`
- `seq7771`: `local_runs/generated_positive_pixel_inputs_20260625/seq7771_promoted_postcombo5_realframe_ctf/source_counter_contact_sheet.png`

Both are real raw-frame crop sheets, not coordinate-only fallback images.

## Decision

Keep the asset builder and best-crop admission path. Do not promote an assignment or e2e score from this probe. Do not spend GPT-image budget on `seq5486`. When a safe `OPENAI_API_KEY` environment variable is available, run GPT-image only for `seq7771` with `--max-reference-crops 1`, then require both DINOv2 and SigLIP source CTF before using generated positives in any M7/M8 repair proposal.

## Upload

S3 upload is still blocked locally: no local AWS credentials and Pluto SSH credential retrieval was not reachable in the previous S3 attempt. Local artifacts are listed above.
