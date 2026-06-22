# No-Anchor AutoResearch Protocol Audit

This report distills the Deli AutoResearch protocol into the VLINCS no-anchor global-ID loop.

## Distillation

- State is file-backed: `progress.json`, `directions_tried.json`, and append-only findings.
- Proposers generate no-anchor candidate assignments.
- Proxy reviewers rank scarce full-score slots, but are not completion evidence.
- Opponents can veto: result gate, false-split coverage audit, and canonical DS1 scorer.
- GT/oracle data is allowed only after prediction, as evaluator or historical proxy label.

## Audit Summary

- candidate rows: `8`
- rows with hard blockers: `0`
- rows carrying post-hoc labels: `0`
- proxy models: `5`
- proxy models with hard blockers: `0`
- pass protocol: `true`

## Proxy Models

| model | columns | forbidden feature columns | posthoc labels | corr | mae |
| --- | ---: | --- | --- | ---: | ---: |
| `local_runs/no_anchor_full_proxy_autoresearch_refresh_model_20260620.json` | `34` | `-` | `true` | `None` | `None` |
| `local_runs/no_anchor_full_proxy_compact_ridge_model_20260620.json` | `29` | `-` | `true` | `None` | `None` |
| `local_runs/no_anchor_full_proxy_componentaware_ridge_model_20260620.json` | `96` | `-` | `true` | `None` | `None` |
| `local_runs/no_anchor_full_proxy_delivery_ridge_model_20260620.json` | `34` | `-` | `true` | `None` | `None` |
| `local_runs/no_anchor_full_proxy_mass_features_ridge_model_20260620.json` | `34` | `-` | `true` | `None` | `None` |

## Row Findings

- no hard blockers or post-hoc row labels found
