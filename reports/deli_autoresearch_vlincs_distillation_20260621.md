# Deli AutoResearch Distillation For VLINCS

Date: 2026-06-21

## Source Reading

- [Deli_AutoResearch framework](https://victorchen96.github.io/auto_research/framework.html): state files, zero-interaction execution, ready-means-execute, stall detection, guardian/worker separation.
- [Deli AutoResearch papers page](https://victorchen96.github.io/auto_research/paper.html): self-play paper uses multi-round review, production stats, 285B GRPO experiment as autonomous research evidence.
- [Self-play story](https://victorchen96.github.io/blog_self_play_story.html): honest score drops, verifier-noise ablation, median of reviewer personas, theory/experiment bottleneck routing.

## Distilled Operating Rules

- Persist progress in progress.json/findings.jsonl/reports, not chat memory.
- Ready means execute: if a candidate has enough no-GT evidence and budget is available, export/score it rather than asking for confirmation.
- A score drop is evidence, not failure; append it as a refutation with exact artifacts.
- Separate proposer, referee, opponent, and gate. Pair/global-id model metrics do not certify e2e delivery.
- After repeated stale iterations, pivot structure: new evidence source, new admission guard, or new counterfactual family, not threshold retuning. The source framework uses stale_count>=2 as the hard pivot signal; this VLINCS loop also treats the current long stale streak as a structural-pivot requirement.
- Use self-play as proposer/referee opposition: generate candidates, create adversarial counterexamples, then let canonical full-score/opponent veto.
- For VLINCS specifically, same-policy density recheck is required before claiming a candidate beats the standing best.

## Latest Thread Distillation

The open-source announcement does not change the VLINCS objective, but it
sharpens how the research loop should behave:

- Treat AutoResearch as a protocol, not a magic model.  The useful part is the
  durable loop: state files, fresh directions, watchdogs, and explicit
  anti-loop constraints.
- Self-play translates to proposer versus opponent.  For VLINCS, that means a
  candidate generator must be paired with adversarial same-camera/time
  counterexamples, source-retention checks, and canonical full-score veto.
- The DeepSeek 285B GRPO story is relevant because the agent ran real
  experiments, debugged failures, and accepted a score drop after evidence
  checks.  Our equivalent is to keep every below-standing full-score as a
  refutation, not bury it.
- After repeated stale iterations, the loop should pivot structure.  In the
  current no-anchor VLINCS loop, single-tracklet relinks are now quality-gated
  but too low-mass, so the next structural pivot is grouped/component relinking
  with source-retention admission.

## Applied Immediately

- Added side-effect admission audit for scheduler rows.
- Patched target namespace repair through target_top_seqs unique vote.
- Executed repaired rank2 candidate and rejected it after full-score plus density recheck.
- Executed k3 mass-bridge proxy candidate after the Deli refresh and rejected it after canonical full-score dropped to IDF1 0.653963, below standing best 0.655378. This is now recorded as `reports/no_anchor_k3_mass_bridge_proxy_refutation_20260621.md`.

## VLINCS Translation

For our no-anchor global-ID work, AutoResearch is not a paper-writing trick; it is a research scheduler discipline.  The concrete translation is: every candidate gets a provenance trail, a no-GT admission audit, a canonical opponent score, and a durable refutation/promotion entry.  The current branch did not improve the score, but it improved the pipeline by making namespace drift and side-effect risk machine-checkable.

## Refresh: Temporal And Target-Fragment Probes

The latest application of the protocol tested the next suggested branch, `target_fragment_or_local_track_continuity_positive_mining`.

- MCAM04/08 strict temporal continuity found 0 admissible edges.
- MCAM04/08 ultrawide temporal continuity found 1 edge, but canonical full-score was IDF1 0.653210 / HOTA 0.517030 / AssA 0.532678, below the standing best.
- Target-fragment softcut selection found 5 high-confidence small-component merges and improved proxy pair-F1 from 0.769367 to 0.769406, but canonical full-score was IDF1 0.653220 / HOTA 0.517042 / AssA 0.532691, also below the standing best.
- Conservative clothing/conflict source quarantine found a safe 1-node split with unchanged proxy pair-F1, but canonical full-score was IDF1 0.653242 / HOTA 0.517070 / AssA 0.532728, again below the standing best.

Decision: record all three as refutations.  The AutoResearch lesson here is to stop tuning same-family small-fragment thresholds and pivot structure toward a side-effect critic plus component-scale search.  Conservative quarantine remains useful as an admission/safety primitive, not as the primary improvement axis.
