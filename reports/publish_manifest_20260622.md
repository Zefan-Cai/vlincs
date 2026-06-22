# VLINCS Publish Manifest

Date: 2026-06-22

## Current Standing

- Setting: no-anchor global-id research.
- Global-id pair model: F1 / precision / recall =
  `0.775234 / 0.820504 / 0.734698`.
- End-to-end best: IDF1 / HOTA / AssA =
  `0.656563 / 0.520067 / 0.535717`.
- Goal remains open: end-to-end IDF1 is still below `0.70`.

## Main Recent Promotions

Recall-guarded subpart repair promoted the current no-anchor e2e best.
The promoted edit moved six MCAM04 Tc6 tracklets from component `8` to
component `40`:

`[3886, 3929, 3967, 4002, 4043, 4079]`

Canonical p005-area scoring improved from
`0.656225 / 0.519723 / 0.535329` to
`0.656453 / 0.519993 / 0.535718`.

Then a subpart combo ablation promoted the IDF1 best again.  Combining
`rank01 + rank02 + rank04` moved 11 tracklets and reached:

`IDF1 / HOTA / AssA = 0.656563 / 0.520067 / 0.535717`

## Publish Scope

S3 receives the full current progress snapshot, including:

- source scripts under `kit/`;
- reports under `reports/`;
- durable autoresearch state under `autoresearch_state/`;
- selected local run artifacts under `local_runs/`;
- repository metadata needed for continuity, excluding `.git`, caches, and
  local OS/editor junk.

GitHub receives the core reproducible code and state:

- `kit/`;
- `reports/`;
- `autoresearch_state/`;
- lightweight repo docs/config that already belong to source control.

Large `local_runs/` experiment artifacts are intentionally S3-only.
