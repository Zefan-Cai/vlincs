# No-Anchor Generated-Positive Relocalized CTF Probe

Date: 2026-06-25

Status: infrastructure/refutation record, not an end-to-end gain.

## Hypothesis

The Diffusion-ReID idea should map to VLINCS as:

1. generate same-tracklet identity-preserving positive images;
2. filter generated images with Re-ID confidence threshold filters;
3. use only CTF-passed images as weak evidence for later no-anchor M7/M8 repair proposals.

This probe checks whether the existing prompt/crop assets and CTF gates are
self-contained in the current `wisc` checkout before calling GPT-image.

## Implementation

- Reused `kit/generate_no_anchor_tracklet_local_positives.py`.
- Reused `kit/audit_no_anchor_generated_positive_ctf.py`.
- Relocated the small `seq4043_residual_feature_outlier_ctf` prompt/crop input
  from the older checkout into the current checkout under:
  `local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/`.
- Rewrote crop/frame paths in the prompt manifest so the current checkout can
  rerun the local dry-run without depending on `/Users/zcai/Codex/vlincs_reid_by_search`.
- Did not use anchors or GT identities.
- Did not call GPT-image because `OPENAI_API_KEY` was not present in shell or
  `launchctl` environment. The API key was not written to commands, files,
  repo, S3, shell history, or logs.

## Commands

```bash
python kit/generate_no_anchor_tracklet_local_positives.py \
  --prompt-manifest local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/prompt_manifest.json \
  --output-dir local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/local_aug_relocalized_probe \
  --backend local_aug \
  --variants-per-seq 4 \
  --max-reference-crops 2 \
  --repo-root "$PWD"

python kit/audit_no_anchor_generated_positive_ctf.py \
  --prompt-manifest local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/prompt_manifest.json \
  --generated-manifest local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/local_aug_relocalized_probe/generated_manifest.json \
  --json local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/local_aug_relocalized_probe/ctf_audit_dinov2_source.json \
  --feature-backend dinov2 \
  --reference-mode source \
  --min-same-sim 0.80 \
  --min-margin 0.05 \
  --repo-root "$PWD"

python kit/audit_no_anchor_generated_positive_ctf.py \
  --prompt-manifest local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/prompt_manifest.json \
  --generated-manifest local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/local_aug_relocalized_probe/generated_manifest.json \
  --json local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/local_aug_relocalized_probe/ctf_audit_siglip_source.json \
  --feature-backend siglip \
  --reference-mode source \
  --min-same-sim 0.75 \
  --min-margin 0.03 \
  --repo-root "$PWD"
```

## Results

| Gate | Generated | Pass | Reject | Decision |
|---|---:|---:|---:|---|
| DINOv2 source CTF | 12 | 8 | 4 | useful but rejects all seq4043 local-aug variants at min-same-sim 0.80 |
| SigLIP source CTF | 12 | 12 | 0 | useful second-view gate |

For the intended `seq4043` subject:

| Gate | seq4043 pass | Note |
|---|---:|---|
| DINOv2 source CTF | 0/4 | same-source similarities were 0.625556 to 0.764019, below 0.80 |
| SigLIP source CTF | 4/4 | similarities were 0.894539 to 0.944782 with positive margins |

## Decision

No end-to-end score was run and no best score changed.

The local dry-run confirms the current checkout can reproduce the generation
and CTF plumbing, but it also shows why local augmentation is not enough: the
actual target subject `seq4043` is too weak under DINO source CTF. The next
useful test must call GPT-image from a safe environment variable and then keep
only images that pass both DINO and SigLIP source CTF.

## Next

Run the same prompt manifest with:

```bash
export OPENAI_API_KEY="$(launchctl getenv OPENAI_API_KEY)"
python kit/generate_no_anchor_tracklet_local_positives.py \
  --prompt-manifest local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/prompt_manifest.json \
  --output-dir local_runs/generated_positive_pixel_inputs_20260625/seq4043_residual_feature_outlier_ctf/gpt_image2_probe \
  --backend openai \
  --model gpt-image-2 \
  --variants-per-seq 4 \
  --max-reference-crops 2 \
  --repo-root "$PWD"
```

Then rerun DINOv2 and SigLIP source CTF before any generated image influences a
candidate repair.
