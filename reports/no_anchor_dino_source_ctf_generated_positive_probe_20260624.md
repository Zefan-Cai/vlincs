# No-Anchor DINO Source-CTF Generated-Positive Probe

Date: 2026-06-24

Status: useful generated-positive admission method, not an end-to-end gain.

## Hypothesis

The previous generated-positive dry run used a color-only CTF audit and rejected
all 8 local augmentations. That gate is too brittle for VLINCS ReID crops. A
Diffusion-ReID-style CTF should compare each generated image against the exact
source crop used to generate it, then require separation from hard negatives.

## Pipeline Module

- M2 Evidence Extraction: DINOv2 embedding for real and generated crops.
- M5 Calibration/Fusion: source-specific similarity plus hard-negative margin.
- M7 Candidate Retrieval: future generated positives should be admitted only
  after this CTF.

No anchors and no GT labels are used. This probe only compares real crops and
tracklet-local generated/local-aug images.

## Method

`kit/audit_no_anchor_generated_positive_ctf.py` now supports:

```bash
--feature-backend dinov2 \
--reference-mode source \
--dinov2-model-id facebook/dinov2-small \
--min-same-sim 0.80 \
--min-margin 0.05
```

The default remains the original lightweight histogram/centroid CTF, so demo
reproduction is unchanged.

## Results

| Gate | Pass | Reject | Note |
| --- | ---: | ---: | --- |
| Previous color-only CTF | 0 | 8 | killed as too brittle |
| DINO same-tracklet mean CTF | 1 | 7 | too strict for viewpoint/occlusion variation |
| DINO source-specific CTF | 7 | 1 | retained as future GPT-image CTF |

## Command

```bash
local_runs/deep_ctf_venv_20260624/bin/python \
  kit/audit_no_anchor_generated_positive_ctf.py \
  --repo-root /Users/zcai/Codex/vlincs_reid_by_search \
  --prompt-manifest local_runs/generated_positive_pixel_inputs_20260624/top2_visual_subcluster_ctf/prompt_manifest.json \
  --generated-manifest local_runs/generated_positive_pixel_inputs_20260624/top2_visual_subcluster_ctf_local_aug_probe/generated_manifest.json \
  --json local_runs/generated_positive_pixel_inputs_20260624/top2_visual_subcluster_ctf_local_aug_probe/ctf_audit_dinov2_source_after_patch.json \
  --feature-backend dinov2 \
  --reference-mode source \
  --min-same-sim 0.80 \
  --min-margin 0.05
```

## Decision

Keep the audit backend. Do not promote any assignment or default pipeline change
from this probe alone.

The next GPT-image run should use tracklet-local source-specific DINO CTF first,
then a second SigLIP/body-part gate before any generated image becomes weak
training evidence.
