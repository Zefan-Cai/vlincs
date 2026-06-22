# No-Anchor AutoResearch Protocol Audit

This report distills the Deli AutoResearch protocol into the VLINCS no-anchor global-ID loop.

## Distillation

- State is file-backed: `progress.json`, `directions_tried.json`, and append-only findings.
- Proposers generate no-anchor candidate assignments.
- Proxy reviewers rank scarce full-score slots, but are not completion evidence.
- Opponents can veto: result gate, false-split coverage audit, and canonical DS1 scorer.
- GT/oracle data is allowed only after prediction, as evaluator or historical proxy label.

## Audit Summary

- candidate rows: `1`
- rows with hard blockers: `0`
- rows carrying post-hoc labels: `0`
- proxy models: `0`
- proxy models with hard blockers: `0`
- pass protocol: `true`

## Proxy Models

| model | columns | forbidden feature columns | posthoc labels | corr | mae |
| --- | ---: | --- | --- | ---: | ---: |

## Row Findings

- no hard blockers or post-hoc row labels found
