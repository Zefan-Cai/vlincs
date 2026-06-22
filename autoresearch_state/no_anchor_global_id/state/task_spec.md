# VLINCS No-Anchor Global ID AutoResearch Task

Goal:

- Build a no-anchor global-ID resolver for VLINCS tracklets.
- The active target is both model-side global-ID quality and end-to-end DS1
  submission quality above 0.70.
- Anchors are not allowed in production policy selection.  Ground truth may be
  used only after predictions are written, for evaluation and bottleneck
  diagnosis.

Current verified best:

- Model-side: `loose_source_island_g8_strict_target_reassign`, pair
  F1/P/R = `0.775234 / 0.820504 / 0.734698`.
- End-to-end: `no_anchor_softcut_then_softoverlap_density_filter_selector_zip`,
  IDF1/HOTA/AssA = `0.655240 / 0.518652 / 0.534359`.
- Oracle diagnostic on the current assignment reaches about `0.706202` IDF1,
  so the remaining gap is dominated by identity resolution, not only detector
  row filtering.

AutoResearch operating rules distilled from Deli_AutoResearch:

- Persist state to files before relying on chat history.
- Ready means execute: when a bounded experiment has a command and validation
  path, run it.
- Separate proposer from evaluator: selection uses no GT, scoring uses GT only
  after prediction.
- Record negative results.  A score drop is a finding if it rules out a branch.
- Prefer direction diversity after repeated local failures.
- Treat the evaluator as a self-play opponent: pair F1 may propose a candidate,
  but full DS1 IDF1/HOTA, no-anchor metadata, and delivery coverage may veto it.
- Repeated implementation or runtime failures become preflight checks:
  `py_compile`, direct `--self-test`, no-anchor metadata, and result-gate
  inspection before spending another full-score run.

Next research shape:

- Stop widening simple thresholds, row filters, zip switching, small-island
  attachment, DINO target-repair edge tables, or NFC feature smoothing unless a
  new evidence signal is introduced.
- Test repairability proxies that can predict whether an edit is delivery-safe
  before committing it in one global-ID namespace.  Useful no-GT signals include
  component size, conflict density, per-video row density, feature margin,
  temporal/camera overlap, source agreement, and tracklet-quality distribution.
- Because `stale_count` is now high, the next executable branch must change a
  structural constraint or evidence source.  The current allowed branch is
  edge-rank target-fragment repair plus tiny-fragment override; pair-only
  winners are explicitly rejected unless they improve full DS1 IDF1/HOTA.
