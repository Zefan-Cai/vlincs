# No-Anchor Generated-Positive SigLIP CTF Output Fix

Date: 2026-06-24
Module: M2/M5 generated-positive evidence extraction and CTF filtering
Status: infrastructure fix, no end-to-end gain

## Hypothesis

The generated-positive path needs a robust Re-ID confidence threshold filter
before any GPT-image output can become weak evidence. The DINO path works, but
the SigLIP path must also run because generated images can preserve color while
drifting in identity.

## Implementation

`kit/audit_no_anchor_generated_positive_ctf.py` now accepts transformer image
feature outputs that are either tensors, `pooler_output`, or
`last_hidden_state`. This fixes a SigLIP crash where `get_image_features()`
returned `BaseModelOutputWithPooling` instead of a raw tensor.

No default demo path changed. No anchors or GT labels are used. No OpenAI key is
stored, printed, committed, or passed on a command line.

## Dry-Run Input

Prompt manifest:

`/Users/zcai/Codex/vlincs_reid_by_search/local_runs/generated_positive_pixel_inputs_20260624/seq4043_residual_feature_outlier_ctf/prompt_manifest.json`

The manifest has three sequences and nine real VLINCS crops:

- `seq4043`: source-positive subject
- `seq5824`: local counter-tracklet
- `seq2344`: local counter-tracklet

The source contact sheet was visually checked and contains real frame crops, not
fallback boxes or blank images.

## Result

Local augmentation dry-run in the current `wisc_sync` checkout:

`local_runs/generated_positive_gpt_ready_seq4043_dryrun_20260624_abs/`

| Gate | Generated | Pass | Reject | Decision |
| --- | ---: | ---: | ---: | --- |
| DINOv2 source CTF | 12 | 8 | 4 | keep as first generated-positive gate |
| SigLIP source CTF | 12 | 12 | 0 | keep as second generated-positive gate |

This is not an end-to-end gain and does not update the best DS1 score:

`IDF1 / HOTA / AssA = 0.669311 / 0.529982 / 0.540246`

## Next

Once `OPENAI_API_KEY` is safely available as an environment variable, run the
same prompt manifest with `--backend openai --model gpt-image-2`, then apply
DINOv2 and SigLIP source CTF before any M8 repair proposal. Generated images
that fail either CTF gate become hard negatives for the reviewer.
