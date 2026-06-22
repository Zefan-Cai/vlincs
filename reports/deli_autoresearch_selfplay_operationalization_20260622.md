# Deli AutoResearch / Self-Play Distillation For No-Anchor VLINCS

Date: 2026-06-22

## Sources Read

- Deli_AutoResearch framework:
  https://victorchen96.github.io/auto_research/framework.html
- Deli AutoResearch papers index:
  https://victorchen96.github.io/auto_research/paper.html
- self-play story:
  https://victorchen96.github.io/blog_self_play_story.html
- related bilevel autoresearch paper:
  https://arxiv.org/abs/2603.23420
- autoresearch guardrail note:
  https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating

## Distilled Mechanism

The useful transfer is an operating protocol, not a new ReID model.
Deli_AutoResearch is a long-horizon research scaffold: persist state to files,
detect cognitive loops and stalls, separate workers from watchdogs, and treat
score drops as evidence. The self-play story adds the key research behavior:
the agent should generate experiments that can falsify its own current claim,
then honestly downgrade when external checks or raw logs contradict the story.

The June 2026 thread adds one stronger lesson: once a candidate research claim
is concrete enough, the agent must turn it into a real experiment, not a
proposal. In Deli's self-play run, the differentiator was not only the paper
loop; it was the autonomous transition from survey claim to 285B GRPO jobs,
including failed submissions, bug fixes, raw-log checks, and score downgrades
when citation evidence was wrong. For VLINCS this maps to: do not stop after a
proxy scheduler says "candidate looks good"; materialize the assignment, run
the canonical DS1 evaluator, then let the result either promote or kill the
family.

For VLINCS no-anchor global ID this means:

1. Keep every iteration in durable state files, not conversation memory.
2. Separate proposer, referee, opponent, executor, and result gate.
3. Never use GT labels, anchor tracklets, or eval metrics as production
   assignment evidence.
4. Allow GT only in eval-only opponent reports and final benchmark scoring.
5. When a family gives repeated small or negative returns, force a structural
   pivot instead of another threshold sweep.
6. Before expensive full-score jobs, require an opponent that can articulate
   likely false merge, false split, namespace drift, or detector-admission
   failure.

## VLINCS-Specific Reviewer Personas

The Deli five-reviewer idea maps cleanly to the no-anchor loop:

| Reviewer | VLINCS job | Rejects |
|---|---|---|
| Experimentalist | checks raw score artifacts and provenance | pair-only wins that fail DS1 full score |
| Theorist | checks identity-resolution assumptions | merges that violate cannot-link or transition logic |
| Perfectionist | checks artifact hygiene | oracle leakage, stale artifacts, duplicate families |
| Synthesizer | checks whether the iteration answers the current bottleneck | local tweaks after stale loops |
| Newcomer | checks report/case readability | single-frame bbox cases that do not show identity evidence |

The median-reviewer lesson is important: a single enthusiastic metric should
not carry the decision. For this task, pair F1 alone is the over-enthusiastic
reviewer; DS1 IDF1/HOTA/AssA plus opponent diagnostics are the median.

## Applied To Current Evidence Branch

Current verified state:

- no-anchor global-id pair model is above target:
  F1/P/R = `0.775234 / 0.820504 / 0.734698`;
- no-anchor end-to-end delivery is still below target:
  IDF1/HOTA/AssA = `0.655911 / 0.519311 / 0.534922`;
- BoTSORT OSNet+color sample evidence improved weak features:
  identity F1 / pair F1 = `0.380024 / 0.253572`;
- GT-box 3-sample mean OSNet did not help:
  identity F1 / pair F1 = `0.272212 / 0.095926`;
- simple NFC is not enough:
  k8 pair F1 `0.255158`, identity F1 `0.372728`.

This creates a concrete AutoResearch decision:

- close more threshold-only GT-box/mean-feature sweeps;
- keep BoTSORT OSNet+color as the current positive evidence source;
- make the next candidate family a structural evidence change:
  multi-prototype crop aggregation, mean+dispersion feature blocks, and a weak
  pair scorer trained from no-anchor weak positives plus hard negatives;
- insert back into DS1 only after sample identity evidence moves materially
  beyond the current `0.38` identity F1 plateau.

## Immediate Research Consequence

The current branch has two active pressure points:

1. Sample identity evidence improved locally, but does not yet explain the DS1
   delivery plateau.
2. There is already a referee-pruned full-score queue whose proxy predicts
   `0.662361-0.666790` IDF1, above the older `0.655240` base but still below
   the `0.70` target.

Following the Deli rule "ready means execute", the next action is to run the
referee-pruned queue on h100-test-3 rather than invent another proxy. The
experiment is still no-anchor: GT is used only by the evaluator, and production
assignment evidence comes from the exported no-anchor assignment CSVs.

Acceptance gate:

- promote only if canonical DS1 IDF1 exceeds the current verified best
  `0.655911`;
- log as a negative result if pair F1/proxy stays high but DS1 IDF1 does not
  move;
- if all four ranks lose, force a structural pivot away from crossqueue
  component rewrites and toward a stronger weak-label curriculum or new visual
  evidence source.

## Next No-Anchor Loop Contract

For the next worker:

1. Rerun OSNet extraction with `features_osnet_std` and `features_color_std`
   enabled.
2. Compare mean-only, mean+std, and prototype-aware pair scoring on the sample
   parquet before touching the DS1 pipeline.
3. Keep GT labels eval-only: use them to report retrieval AP, top-k hit recall,
   precision, recall, false-merge/false-split, and per-video slices.
4. Reject any proposal that improves pair F1 but worsens identity F1 or the
   opponent margin diagnostics.
5. If sample identity F1 remains near `0.38`, pivot to a different evidence
   source rather than another graph solver.

The Deli/self-play translation is therefore: make each experiment attack the
active bottleneck, force opponent evidence into the report, and let bad results
close branches quickly.
