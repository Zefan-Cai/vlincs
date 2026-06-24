# No-anchor generated-positive local-augmentation CTF refutation

Date: 2026-06-24
Module: M2/M5/M10
Status: negative ablation / infrastructure retained

## Hypothesis

Before spending GPT-image-2 calls, validate the generated-positive pipeline on
restored real VLINCS crops: source crops -> generated candidates -> no-anchor CTF
reviewer -> upload/provenance record.

## Implementation

Added two small scripts:

- `kit/generate_no_anchor_tracklet_local_positives.py`
  - `--backend openai` calls `gpt-image-2` through `OPENAI_API_KEY` only.
  - It refuses command-line key literals and does not store secrets.
  - `--backend local_aug` creates deterministic local augmentations as a dry-run.
- `kit/audit_no_anchor_generated_positive_ctf.py`
  - no-anchor CTF reviewer over generated images.
  - current dry-run gate uses upper/lower RGB histogram similarity and a local counter-tracklet margin.
  - no GT labels or anchors are read.

Both scripts accept `--repo-root` so a clean `wisc` checkout can resolve crop
paths from the published local/S3 input manifest without assuming the current
working directory is the original research checkout.

## Result

Input artifact:

- `local_runs/generated_positive_pixel_inputs_20260624/top2_visual_subcluster_ctf/prompt_manifest.json`
- `source_contact_sheet.png`

Dry-run output:

- generated images: `8`
- CTF pass: `0`
- CTF reject: `8`
- gate: `same_seq_color_similarity >= 0.86` and `margin >= 0.025`

The local augmentation path validates the artifact/provenance flow, but the
color-only CTF is too brittle for these crops. Seq 5550 often scores closer to
seq 2987 under color histograms, so color-only CTF is not safe as a promotion
gate.

## Decision

Do not promote. Keep the scripts as infrastructure. The next real experiment
must use GPT-image-2 tracklet-local generation and DINOv2/SigLIP/weakmetric CTF,
preferably on `h100-test-2` or `h100-test-3`.

## S3

Uploaded:

`s3://dit-scale-up/zcai/vlincs/no_anchor_generated_positive_inputs/20260624_top2_visual_subcluster_ctf_local_aug_probe/`

## Security

The OpenAI backend was safety-checked and correctly refused to run because
`OPENAI_API_KEY` was not set in the environment. The API key was not written to
files or command lines.
