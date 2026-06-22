# No-Anchor OSNet Evidence Diagnostic: GT-Box, Retrieval, and NFC

Date: 2026-06-22

## Purpose

The previous BoTSORT sample result showed that stronger crop/person evidence
matters:

- weak crop+bbox BoTSORT sample: identity F1 / pair F1 = `0.191162 / 0.032079`
- OSNet+color BoTSORT sample: identity F1 / pair F1 = `0.380024 / 0.253572`

This round asked two follow-up questions under the same no-anchor rule:

1. If boxes are upper-bound GT boxes, does OSNet+color become much better?
2. Can simple neighbor feature centralization improve the BoTSORT OSNet result?

GT identity is used only for evaluation and diagnostics, never for anchors,
training labels, or feature construction.

## GT-Box OSNet Extraction

Feature extraction used:

- `kit/extract_sample_tracklet_osnet_features.py`
- input: `gtbox_eval.parquet`
- output: `features_gtbox_osnet_color_s3_20260622.npz`
- samples per tracklet: `3`
- feature blocks: `features_osnet` dim `512`, `features_color` dim `82`

Extraction summary:

| field | value |
|---|---:|
| tracklets | `1,887` |
| OSNet valid | `1,887` |
| color valid | `1,887` |
| seen crops | `5,284` |
| missing video / crop | `0 / 0` |

## Sample Sweep Results

| source / transform | best solver | identity F1 | pair F1 | pair P | pair R | purity mean / p10 / min | components | largest |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BoTSORT OSNet+color | `consensus_guard@0.35` | `0.380024` | `0.253572` | `0.315609` | `0.211916` | `0.743022 / 0.356520 / 0.241489` | `172` | `114` |
| GT-box OSNet+color | `consensus_guard@0.25` | `0.272212` | `0.095926` | `0.132012` | `0.075333` | `0.898674 / 0.391758 / 0.170272` | `304` | `120` |
| BoTSORT OSNet+color + NFC k8 eta0.5 | `consensus_attach@0.55` | `0.372728` | `0.255158` | `0.303206` | `0.220255` | `0.792345 / 0.325062 / 0.177128` | `184` | `120` |
| BoTSORT OSNet+color + NFC k16 eta0.5 | `consensus_guard@0.25` | `0.363662` | `0.234999` | `0.268661` | `0.208834` | `0.761812 / 0.317959 / 0.175894` | `166` | `120` |

NFC conclusion:

- k8 gives a tiny pair-F1 gain: `0.253572 -> 0.255158`;
- k8 lowers identity F1: `0.380024 -> 0.372728`;
- k16 degrades both identity and pair F1.

So simple NFC is not a production-worthy route.

## Feature Retrieval Diagnostic

New eval-only tool:

- `kit/analyze_sample_feature_retrieval.py`

It measures whether same-GT-ID tracklets are present in the nearest-neighbor
evidence pool.  This is diagnostic only; it does not write predictions.

| source | AP | ROC AUC | all top-45 hit recall | cross-video top-45 hit recall | cross-video top-positive > top-negative |
|---|---:|---:|---:|---:|---:|
| GT-box OSNet+color | `0.098062` | `0.623248` | `0.965554` | `0.905670` | `0.255962` |
| BoTSORT OSNet+color | `0.306367` | `0.846629` | `0.993133` | `0.976228` | `0.656101` |

Interpretation:

- GT-box OSNet does retrieve at least one true same-ID neighbor for most
  tracklets by top-45, so candidate recall is not zero.
- But the margin is bad: for cross-video retrieval, only `25.6%` of GT-box
  queries have a best positive neighbor above the best negative neighbor.
- BoTSORT has a much healthier `65.6%` on the same check.
- Therefore the GT-box result is not a clean upper bound.  Long GT tracklets
  plus 3-sample mean pooling produce weaker identity evidence than shorter
  BoTSORT fragments.

Similarity quantiles support the same conclusion:

| source | same median | diff median | same p25 | diff p75 |
|---|---:|---:|---:|---:|
| GT-box OSNet+color | `0.638550` | `0.602763` | `0.577623` | `0.652721` |
| BoTSORT OSNet+color | `0.661030` | `0.530491` | `0.596933` | `0.587256` |

For GT-box, a typical negative pair is too close to a true positive pair.  For
BoTSORT, the same/different gap is much larger.

## Decision

Close the hypothesis that GT-box + 3-sample mean OSNet is a useful upper bound.
It is a diagnostic artifact, not a production candidate.

Keep BoTSORT OSNet+color as the best sample evidence source so far, but do not
insert NFC.  The next productive branch is feature aggregation, not graph
policy:

1. extract more crops per long GT tracklet and test robust aggregation;
2. store per-tracklet multi-prototype features instead of a single mean vector;
3. train/evaluate a no-anchor pair scorer with weak positives and hard negatives
   over crop prototypes;
4. only after sample identity F1 moves materially above `0.38` should this be
   inserted back into the full DS1 no-anchor delivery pipeline.

## Artifacts

- `kit/analyze_sample_feature_retrieval.py`
- `reports/feature_retrieval_gtbox_osnet_color_s3_20260622.md`
- `reports/feature_retrieval_botsort_osnet_color_s3_20260622.md`
- `local_runs/remote_h100_test_3_20260621/gtbox_no_anchor_sample_20260618/features_gtbox_osnet_color_s3_20260622.npz`
- `local_runs/remote_h100_test_3_20260621/gtbox_no_anchor_sample_20260618/no_anchor_gtbox_sample_osnet_color_s3_v1.json`
- `local_runs/remote_h100_test_3_20260621/gtbox_no_anchor_sample_20260618/feature_retrieval_gtbox_osnet_color_s3_20260622.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_nfc_k8_eta05.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_nfc_k16_eta05.json`
