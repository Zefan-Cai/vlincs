# AutoResearch Distillation for VLINCS No-Anchor Global ID

Date: 2026-06-20

Sources read:

- https://victorchen96.github.io/auto_research/framework.html
- https://victorchen96.github.io/auto_research/paper.html
- https://victorchen96.github.io/blog_self_play_story.html

## What matters for this project

Deli AutoResearch is useful here less as a model recipe and more as an operating
protocol for long-horizon research. The relevant pieces are:

- Persist progress in files, not in chat history.
- Treat score drops as evidence, not failure to hide.
- Separate proposal generation from evaluation.
- After repeated stale iterations, change the structural constraint instead of
  continuing the same threshold sweep.
- Use independent review or guardrail stages, especially when the proposer can
  overfit to a proxy metric.
- Prefer diverse candidate directions once a family is saturated.

## VLINCS mapping

Current status:

- Model-side no-anchor pair F1 is already above target:
  F1 `0.775234`, precision `0.820504`, recall `0.734698`.
- End-to-end IDF1 is still below target:
  best promoted full IDF1 `0.655240`.
- Oracle decomposition says the missing mass is recoverable:
  base `0.653071`, oracle majority `0.706202`,
  split-top-40-then-merge-top-40 `0.706008`.

So the loop should not spend the next iteration on a broad visual-threshold
sweep. It should test a structural false-split repair hypothesis:

- Candidate proposer: multiview bridge edges.
- Guardrail: keep normal cannot-link by default.
- Controlled relaxation: allow only high-scoring large-to-tiny component bridges
  through cannot-link under a small-side and size-ratio budget.
- Evaluator: frozen DS1 full-score gate and result gate.
- Scheduler: consume full-score budget on diverse unscored candidates, not
  duplicate solver families.

## Concrete next branch

Added a tiny-fragment cannot-link override to
`kit/no_anchor_assignment_multiview_merge_sweep.py`.

Default behavior is unchanged. When enabled, a forbidden edge can be accepted
only if:

- the smaller original component size is within the swept limit;
- the larger/smaller original component-size ratio clears the swept threshold;
- normal score, rank-vote, sim-vote, max-component-size, and edge-budget gates
  have already passed.

The launcher `kit/run_no_anchor_false_split_budget_merge.sh` now runs:

- `chained`: original budgeted bridge sweep;
- `diverse`: one original component bridge per source/target component;
- `tiny_fragment_override`: narrow grid over small-side sizes `1,2,4` and
  large/small ratios `12,24,48`.

This is a no-anchor branch: GT is not used for training or anchors. Oracle
numbers only motivated the structural hypothesis and are kept in the analysis
log.

## 2026-06-20 thread update: self-play opponent

The new Deli thread adds a sharper lesson for this project: the important
artifact is not only a proposer that writes experiments, but an opponent that
can lower the score when the evidence says so.  In the self-play story, a
review round dropped after an external check found bad references; in our loop,
the analogous failure is a high pair-F1 candidate whose delivered global-ID
namespace collapses under full DS1 scoring.

Concrete VLINCS translation:

- Proposer: no-anchor assignment editors and edge-rank target-fragment repair.
- Opponent: frozen no-anchor result gate, full DS1 IDF1/HOTA, no-anchor
  metadata, delivery coverage, and repeated-direction detection.
- Verdict rule: pair F1/precision/recall above `0.70` is not enough; a candidate
  must beat the standing promoted full IDF1 `0.655240` and eventually the goal
  `0.70`.
- Preflight rule: every remote launcher should run `py_compile`, direct
  `--self-test`, and metadata/result-gate checks before full-score budget is
  spent.

Executable guard added:

- `kit/no_anchor_deli_opponent.py`

Current opponent verdict:

- input gate: `local_runs/no_anchor_result_gate_all_20260620.json`
- output JSON: `local_runs/no_anchor_deli_opponent_20260620.json`
- output report: `reports/no_anchor_deli_opponent_verdict_20260620.md`
- verdict: `pivot`
- reason: best pair/global candidate passes pair metrics but has full IDF1
  `0.085412`; best gated full IDF1 is `0.654009`, still below the standing
  promoted `0.655240`.

Research consequence:

- Reject pair-only winners unless full DS1 IDF1/HOTA improves.
- Continue the structural branch already prepared:
  `kit/run_no_anchor_edge_rank_target_fragment.sh`.
  This tests edge-rank target fragments and tiny-fragment repair rather than
  another broad threshold sweep.

Execution status:

- Remote probes for `h100-test-3`, `h100-test-2`, and `test-video-0` still
  failed on 2026-06-20: Pluto API returned `Failed to connect to Pluto service`
  and SSH dry-runs timed out during banner exchange.
- Treat this as an infrastructure pause.  The branch is prepared and validated,
  but not scored.

## 2026-06-20 evaluator correction

The self-play opponent itself needed a guardrail fix.  The no-anchor result gate
was not reading full-score artifacts whose metrics live in `rows[0]`, and it
could over-admit oracle/audit diagnostics.  That created two bad incentives:
missing the true standing production best and risking a false `pass_joint` from
GT-analysis rows.

Patch:

- `kit/no_anchor_result_gate.py` now parses `rows/full_rows/top_full_rows` and
  maps `idf1/hota/assa` to full metrics.
- It excludes GT-analysis/filter/oracle rows from production eligibility.
- `--self-test` covers this exact format.

Refreshed outcome:

- `local_runs/no_anchor_result_gate_all_20260620.json`
- `pass_joint=false`
- best production e2e `0.655240`
- best joint `0.654009`
- Deli opponent verdict remains `pivot`

## 2026-06-20 missing-mass budget correction

The Deli lesson about honest score drops applies again: a rule can be locally
perfect and still not matter globally.  The previous target-fragment audit found
clean no-GT rules such as `17/17` true DINO-fused tiny-fragment edges, but that
only measured precision.  The new question is coverage of the full-IDF1 gap.

New eval-only guard:

- `kit/analyze_no_anchor_edge_mass_budget.py`
- default audit:
  `local_runs/no_anchor_edge_mass_budget_audit_combined_20260620.json`
- wide audit:
  `local_runs/no_anchor_edge_mass_budget_audit_combined_wide_20260620.json`
- report:
  `reports/no_anchor_edge_mass_budget_audit_combined_20260620.md`

Budget result:

- base full IDF1 `0.653071`;
- oracle full IDF1 `0.706202`;
- missing true-pair mass `9,105,079,781`;
- reaching full IDF1 `0.70` requires about `88.3%` of the oracle gap under the
  linear mass model;
- best high-precision target-fragment rule: `17/17` true edges, precision
  `1.0`, but only `0.0448%` missing-mass coverage;
- wide highest-coverage rule: `0.4329%` coverage, but edge precision only
  `6.9%`.

Research consequence:

- Keep target-fragment repair as a safe cleanup branch.
- Stop treating edge-table tiny-fragment thresholds as the main route to `0.70`.
- Pivot the active research direction to high-mass false-split component
  bridge/split selection with component-level side-effect prediction.

Implementation consequence:

- Fix evaluator fidelity before spending more GPU/remote full-score slots:
  candidate-search rows now replay `accepted_preview` exactly for full scoring
  and assignment export.
- Add a high-mass proposer rather than another threshold-only proposer:
  `mass_bridge_proxy` ranks candidates by moved mass, target size, visual
  consensus, target margin, source acceptor score, and side-effect risk.
- New remote entrypoint:
  `kit/run_no_anchor_high_mass_bridge.sh`.

## 2026-06-20 high-mass component-merge self-play branch

The AutoResearch/self-play lesson also changed the component-merge branch:
candidate rows need provenance about the actual accepted edges, otherwise the
opponent cannot tell whether a full-score slot tested a high-mass false-split
repair or another low-impact local cleanup.

Implementation:

- `kit/no_anchor_component_merge_sweep.py` now records `accepted_preview`,
  edge score means, rank-margin means, weight sums, and mass proxies for accepted
  component bridges.
- `kit/no_anchor_assignment_component_merge_sweep.py` inherits the same
  provenance when starting from an existing no-anchor assignment CSV.
- Both component-merge paths now support `--rank-by mass_proxy`, so expensive
  full scoring can prioritize high-mass bridges without using GT labels.
- `kit/run_no_anchor_high_mass_component_merge.sh` packages the remote run:
  fused-feature and DINO-feature assignment-level component-merge sweeps,
  `--accepted-preview-n 40`, `--rank-by mass_proxy`, and `--full-top-n 12`.
- `kit/no_anchor_fullscore_scheduler.py` now recognizes component-merge
  accepted-preview fields (`source/target` and `source_rep/target_rep`) when
  building family keys, so later manifests can de-duplicate these candidates
  by actual proposed bridge rather than only by artifact name.

Validation:

- `python -m py_compile` passed for the component-merge scripts and current
  gate/opponent scripts.
- `python kit/no_anchor_component_merge_sweep.py --self-test` passed with an
  empty `PYTHONPATH`, confirming the launcher does not depend on implicit local
  import state.
- `python kit/no_anchor_fullscore_scheduler.py --self-test` passed after the
  component-preview family-key update.
- `bash -n kit/run_no_anchor_high_mass_component_merge.sh` passed.

Remote status:

- `CONNECT_TIMEOUT=8 REMOTE_TIMEOUT=20 kit/run_no_anchor_high_mass_component_merge.sh --job h100-test-3 --dry-run`
  reached SSH probing and failed at the known remote blocker:
  `Connection timed out during banner exchange`.
- No new DS1 full-score artifact was produced in this iteration.

## 2026-06-20 edge-table candidate recall audit

The next self-play question was whether the current edge-table proposer has
enough candidate recall to recover the high-mass oracle false splits.  If it
does not, no ranker or threshold policy can make it the main route to e2e
`0.70`.

New eval-only artifact:

- `kit/analyze_no_anchor_edge_table_oracle_coverage.py`
- `local_runs/no_anchor_edge_table_oracle_coverage_20260620.json`
- `local_runs/no_anchor_edge_table_oracle_coverage_20260620.csv`
- `reports/no_anchor_edge_table_oracle_coverage_20260620.md`

Inputs:

- softcut edge table:
  `local_runs/remote_h100_test_3_20260619/no_anchor_softcut_current_edge_acceptor_table_20260619.csv`
- DINO-fused edge table:
  `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_acceptor_table_20260620.csv`
- oracle concentration audit:
  `local_runs/no_anchor_oracle_gap_concentration_20260620.json`

Result:

- combined candidate edges: `5240`;
- union true edges: `47`;
- union true edge mass: `39,412,716`;
- coverage of missing true-pair mass: `0.00432865`;
- top-1 oracle GT coverage: `0.00030661`;
- top-3 oracle GT coverage: `0.00013385`;
- GT `36` and GT `11`, the second and third largest false-split identities,
  have zero true edge coverage in the current edge tables.

Research consequence:

- Edge-table target-fragment repair is now formally closed as a main route to
  `0.70`; it lacks candidate recall, not just ranking quality.
- Keep it as a safe cleanup branch only.
- The main queue should be high-mass component/hub bridge generation:
  `run_no_anchor_high_mass_bridge.sh` and
  `run_no_anchor_high_mass_component_merge.sh`.

## 2026-06-20 mass-proxy bridge provenance

The latest protocol update turns the high-mass bridge branch into a better
AutoResearch unit: the proposer now records what it actually accepted, and the
opponent can rank future full-score slots by likely delivered mass rather than
by local pair-score only.

Implementation:

- `kit/no_anchor_assignment_multiview_merge_sweep.py` now emits
  `accepted_preview`, accepted-edge score aggregates, and no-GT mass proxies for
  multiview bridge rows.
- The same solver now supports `--rank-by mass_proxy`, so candidate rows that
  move larger plausible false-split mass are considered before low-impact local
  cleanups.
- `kit/run_no_anchor_false_split_budget_merge.sh` now uses
  `--rank-by mass_proxy` and `--accepted-preview-n 40`.
- `kit/analyze_no_anchor_full_proxy_training.py` now includes accepted-edge
  provenance and mass features when fitting the compact full-IDF1 proxy.

Refreshed proxy audit:

- artifact:
  `local_runs/no_anchor_full_proxy_training_audit_mass_features_20260620.json`
- ridge model:
  `local_runs/no_anchor_full_proxy_mass_features_ridge_model_20260620.json`
- rows/features: `32 / 34`
- full-IDF1 range: `0.602445..0.654009`
- LOOCV ridge: correlation `0.969076`, MAE `0.001736`, RMSE `0.004556`

Research consequence:

- The current blocker is not model-side pair quality: pair
  `F1/P/R = 0.775234 / 0.820504 / 0.734698` remains above the no-anchor model
  target.
- The end-to-end best is still `0.655240`, below the `0.70` goal.
- The next executable direction is therefore not another edge-table recall
  audit.  It is a mass-ranked multiview/component bridge submission when Pluto
  recovers, followed by the frozen full DS1 gate.

## 2026-06-20 scheduler diversity guard

The full-score scheduler was refreshed with the mass-feature proxy and then
audited as an AutoResearch opponent.  This exposed a budget-allocation issue:
without better family recovery, JSON and CSV views of the same edit could occupy
separate full-score slots.

Patch:

- `kit/no_anchor_fullscore_scheduler.py` now recovers provenance for CSV rows by
  reading the sibling JSON via `_source_rank`, avoiding an O(N^2) artifact scan.
- artifact families are canonicalized by file stem, so `foo.csv` and `foo.json`
  are treated as the same family.
- `--max-per-mode` adds an optional structural diversity cap.  Default behavior
  remains uncapped.

Strict mass-feature manifest:

- artifact:
  `local_runs/no_anchor_fullscore_scheduler_mass_features_diverse_20260620.json`
- report:
  `reports/no_anchor_fullscore_scheduler_mass_features_diverse_20260620.md`
- raw / eligible / selected: `202010 / 696 / 3`
- selected families:
  `component:32->15`, `component:21->19`, and `component:21->0`
- predicted full-IDF1 range among selected rows: `0.655254..0.658302`

Component/multiview exploration:

- artifact:
  `local_runs/no_anchor_fullscore_scheduler_component_multiview_explore_20260620.json`
- report:
  `reports/no_anchor_fullscore_scheduler_component_multiview_explore_20260620.md`
- best old component/multiview predicted full-IDF1: `0.653753`, below the
  standing production best `0.655240`

Research consequence:

- The immediate remote queue should full-score only the three deduplicated
  conflict candidates if Pluto recovers.
- Existing component/multiview artifacts are not strong enough under the strict
  gate; the high-mass component/hub bridge branch still needs newly generated
  rows rather than recycling old low-impact component-merge grids.

## 2026-06-20 Deli AutoResearch distillation and manifest execution

The newly published Deli AutoResearch material is useful here as an operating
protocol, not as a new ReID model.  The parts that transfer cleanly to VLINCS
are:

- Persist every research loop in task/state files rather than chat memory.
- Treat candidate generation, execution, and evaluation as separate roles.
- Use direction diversity after stalls; a new iteration should change a
  structural constraint, not only retune thresholds.
- Let scores move downward when evidence demands it.  A failed full-score run is
  still a finding if it closes a route.
- Run verification between iterations, and keep a watchdog/launcher ready so a
  completed manifest is executed instead of ending at "ready to submit".

Sources read:

- `https://victorchen96.github.io/auto_research/framework.html`
- `https://victorchen96.github.io/auto_research/paper.html`
- `https://victorchen96.github.io/blog_self_play_story.html`

VLINCS mapping:

- Worker/proposer: no-anchor sweep scripts produce rows with
  `accepted_preview`, no-GT mass proxies, and pair metrics.
- Opponent/scheduler: `kit/no_anchor_fullscore_scheduler.py` picks a small,
  structurally diverse full-score manifest without using anchors or GT for
  training/ranking.
- Executor: `kit/export_no_anchor_scheduler_manifest_assignments.py` replays a
  selected row's `accepted_preview` on top of a base assignment CSV and emits a
  canonical assignment CSV for the full scorer.
- Judge: `kit/evaluate_db_assignments_full.py` runs the DS1 scorer and uses GT
  only after prediction.

New executor artifacts:

- `kit/export_no_anchor_scheduler_manifest_assignments.py`
- `kit/run_no_anchor_scheduler_manifest_fullscore.sh`

Local validation:

- `python -m py_compile kit/export_no_anchor_scheduler_manifest_assignments.py kit/no_anchor_fullscore_scheduler.py kit/evaluate_db_assignments_full.py`
- `python kit/export_no_anchor_scheduler_manifest_assignments.py --self-test`
- `bash -n kit/run_no_anchor_scheduler_manifest_fullscore.sh`

Real manifest recovery check:

| rank | source rank | edit | moved tracklets |
| ---: | ---: | --- | ---: |
| 1 | 11 | component `32 -> 15` | 12 |
| 2 | 5 | component `21 -> 19` | 12 |
| 3 | 1 | component `21 -> 0` | 8 |

Remote execution attempt:

- `bash kit/run_no_anchor_scheduler_manifest_fullscore.sh --ranks 1,2,3`
  probed `h100-test-3`, `h100-test-2`, and `test-video-0`.
- All three SSH probes failed during banner exchange.
- Pluto CLI status also failed for all three jobs with
  `Failed to connect to Pluto service`.
- No new full-score artifact was produced.

Research consequence:

- The selected manifest is now executable and reproducible; when Pluto recovers,
  run `kit/run_no_anchor_scheduler_manifest_fullscore.sh --ranks 1,2,3`.
- Because the strict manifest only predicts `0.655254..0.658302`, it is a
  low-cost full-score sanity check, not the likely route to `0.70`.
- The next structural research branch remains new high-mass hub/component bridge
  generation, using Deli-style proposer/opponent/judge separation and preserving
  accepted evidence for every candidate.

## 2026-06-20 accepted-preview evidence audit

To keep the next self-play iteration from overfitting to familiar conflict
reassign rows, I added a preview-level ablation audit.

New artifacts:

- `kit/analyze_no_anchor_preview_evidence.py`
- `local_runs/no_anchor_preview_evidence_audit_20260620.json`
- `local_runs/no_anchor_preview_evidence_audit_20260620.csv`
- `reports/no_anchor_preview_evidence_audit_20260620.md`

Validation:

- `python -m py_compile kit/analyze_no_anchor_preview_evidence.py`
- `python kit/analyze_no_anchor_preview_evidence.py --self-test`

Result:

- preview rows scanned: `5051`;
- full-labelled preview rows: `38`;
- labelled preview rows above current full-IDF1 best `0.655240`: `0`;
- best labelled conflict candidate-search preview row: `0.654009`;
- best labelled conflict reassign preview row: `0.653823`;
- best labelled provisional relink preview row: `0.652265`.

The highest unlabelled pair-mass edits are still mostly
`conflict_subcluster_reassign_candidate_search`, e.g. component `15 -> 14`
with source size `12`, target size `214`, and pair-mass proxy `2568`.

Research consequence:

- Existing full-scored accepted-preview edits are all below the production best.
  This is route-closing evidence for "repeat small conflict reassign harder".
- The unlabelled high-mass conflict rows are worth full-score sanity checks, but
  they are not structurally new enough to be the main path to `0.70`.
- The next proposer should explicitly generate larger hub/component bridge
  candidates and preserve preview evidence so the scheduler can compare them
  against these known conflict baselines.

Follow-up patch:

- `kit/run_no_anchor_high_mass_bridge.sh` now includes large-source source
  groups: `--source-min-group-sizes 2,4,8,12,16` and
  `--source-max-group-sizes 12,24,48`.
- The same launcher now includes `--max-reassignments 24`.

This is the structural pivot suggested by the audit: try larger source islands
and hub bridges instead of repeatedly moving only `8..12`-tracklet fragments.

## 2026-06-20 hub-bridge composite proposer

The previous accepted-preview audit closed the "repeat small conflict reassign"
route as a main path.  The next structural proposer composes multiple
no-anchor accepted-preview edits that point to the same target component,
creating larger hub/component bridge candidates while keeping executable
`source_seqs`.

New artifact:

- `kit/compose_no_anchor_hub_bridge_candidates.py`
- `local_runs/no_anchor_hub_bridge_composite_candidates_20260620.json`
- `local_runs/no_anchor_hub_bridge_composite_candidates_20260620.csv`
- `reports/no_anchor_hub_bridge_composite_candidates_20260620.md`

Validation:

- `python -m py_compile kit/compose_no_anchor_hub_bridge_candidates.py`
- `python kit/compose_no_anchor_hub_bridge_candidates.py --self-test`

Result:

- unique no-anchor preview items after de-duplication: `25`;
- target components represented: `18`;
- composite candidates: `5`;
- strongest composite row: components `4+0 -> 6`, moves `16`
  tracklets, pair-mass proxy `4816`, pair F1 `0.772006`;
- largest composite row: components `19+4+0 -> 6`, moves `28`
  tracklets, pair-mass proxy `4816`, pair F1 `0.772046`.

Combined next full-score queue:

- `local_runs/no_anchor_fullscore_scheduler_next_queue_20260620.json`
- `reports/no_anchor_fullscore_scheduler_next_queue_20260620.md`

Scheduler result:

- raw / eligible / selected: `153 / 147 / 7`;
- selected rows: `4` hub composites + `3` strict conflict rows;
- top predicted full-IDF1: composite `4+0 -> 6`, predicted `0.659003`;
- all selected rows recover executable `accepted_preview` through
  `export_no_anchor_scheduler_manifest_assignments.py`.

Remote command when Pluto recovers:

```bash
bash kit/run_no_anchor_scheduler_manifest_fullscore.sh \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_next_queue_20260620.json \
  --ranks 1,2,3,4,5,6,7 \
  --run-name no_anchor_next_queue_fullscore_20260620
```

Research consequence:

- The next queue now contains structurally new no-anchor candidates rather than
  only single small-island reassignments.
- This still does not prove e2e `0.70`; it creates a higher-value full-score
  queue once the Pluto service/SSH path recovers.

## 2026-06-20 portfolio-level repair proposer

Single selected rows still move too little mass relative to the oracle gap, so
the next proposer composes multiple compatible no-anchor rows into a single
assignment candidate.  Compatibility is conservative:

- no overlapping source seqs;
- no chained component edits, meaning a target component cannot also be a source
  component in the same portfolio.

New artifact:

- `kit/compose_no_anchor_portfolio_candidates.py`
- `local_runs/no_anchor_hub_bridge_portfolio_candidates_20260620.json`
- `local_runs/no_anchor_hub_bridge_portfolio_candidates_20260620.csv`
- `reports/no_anchor_hub_bridge_portfolio_candidates_20260620.md`

Result:

- source scheduler rows: `7`;
- portfolio rows: `30`;
- top portfolio: source ranks `[1, 2, 3]`, targets `15+19+6`,
  moves `40` tracklets, `4` edits, pair-mass proxy `5764`, predicted full
  `0.662663`;
- largest high-ranked portfolio: source ranks `[2, 4, 5, 7]`, targets
  `15+21+55+6`, moves `72` tracklets, `8` edits, pair-mass proxy `12872`,
  predicted full `0.659738`.

Portfolio scheduler:

- `local_runs/no_anchor_fullscore_scheduler_portfolio_next_queue_20260620.json`
- `reports/no_anchor_fullscore_scheduler_portfolio_next_queue_20260620.md`
- raw / eligible / selected: `30 / 30 / 20`;
- top scheduler row: portfolio `[1, 2, 3]`, predicted full `0.662663`,
  scheduler score `0.661267`.

Remote command when Pluto recovers:

```bash
bash kit/run_no_anchor_scheduler_manifest_fullscore.sh \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_portfolio_next_queue_20260620.json \
  --ranks 1,2,3,4,5,6,7,8 \
  --run-name no_anchor_portfolio_next_queue_fullscore_20260620
```

Remote status:

- Dry-run with the command above passed local file/argument checks, then failed
  at the known infrastructure blocker: SSH banner exchange timeout on
  `h100-test-3`.

Research consequence:

- This is the current best no-anchor full-score queue: it tests portfolio-level
  repairs, not just individual hub edits.
- It still needs canonical DS1 full scoring before any claim of e2e progress.

## 2026-06-20 video-focus scheduler correction

The portfolio queue exposed a scheduling problem: the top predicted portfolios
mostly spend edits on videos whose current per-video IDF1 is already above the
`0.70` target.  The actual end-to-end bottlenecks are:

- `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6`: current per-video IDF1 `0.560694`;
- `vlincs_MS01_MC0001_MCAM06_2024-03-Tc6`: current per-video IDF1 `0.608378`;
- `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8`: current per-video IDF1 `0.628615`.

New artifact:

- `kit/rerank_no_anchor_scheduler_by_video_focus.py`
- `local_runs/no_anchor_fullscore_scheduler_portfolio_video_focus_20260620.json`
- `reports/no_anchor_fullscore_scheduler_portfolio_video_focus_20260620.md`

The reranker uses prior full-score per-video metrics only as research-budget
feedback.  It does not create anchors, does not train on GT labels, and does
not alter candidate assignments.  The candidate rows themselves remain
no-anchor `accepted_preview` portfolios.

Result:

- reranked `8` executable portfolio rows;
- the first three rows are original portfolio ranks `19`, `18`, and `20`,
  because they touch many MCAM04 tracklets;
- exporter materialization verified all rank `1..8` rows: `8` assignment CSVs,
  `3..6` preview edits per row, and `28..56` moved tracklets.

Follow-up expansion:

- `local_runs/no_anchor_fullscore_scheduler_portfolio_all30_20260620.json`
  keeps all `30` eligible portfolio rows instead of only the original top `20`;
- `local_runs/no_anchor_fullscore_scheduler_portfolio_all30_video_focus_20260620.json`
  reranks that wider set and selects `10` rows;
- the wider queue pulls original ranks `30`, `29`, `23`, and `21` into the
  top set because they cover MCAM04 and a small amount of MCAM06 Tc6;
- exporter materialization verified all rank `1..10` rows, moving
  `32..72` tracklets per row.

Run this wider queue first when Pluto recovers:

```bash
bash kit/run_no_anchor_scheduler_manifest_fullscore.sh \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_portfolio_all30_video_focus_20260620.json \
  --ranks 1,2,3,4,5,6,7,8,9,10 \
  --run-name no_anchor_portfolio_all30_video_focus_fullscore_20260620
```

Research consequence:

- plain proxy ranking over-prioritized already-good videos;
- scarce full-score slots should now prioritize low-IDF1 / high-row-density
  videos, especially MCAM04;
- no e2e improvement is claimed until canonical DS1 full scoring verifies it.

## 2026-06-20 direct video-focus portfolio proposer

The all30 video-focus queue still only recombines the earlier `7` selected
rows.  A wider proposer now builds portfolios directly from no-anchor
`accepted_preview` items across conflict-reassign artifacts, then filters them
by bottleneck-video coverage.

New artifact:

- `kit/compose_no_anchor_video_focus_portfolio_candidates.py`
- moderate candidates:
  `local_runs/no_anchor_video_focus_portfolio_moderate_candidates_20260620.json`
- moderate scheduler:
  `local_runs/no_anchor_fullscore_scheduler_video_focus_moderate_portfolio_20260620.json`
- aggressive candidates:
  `local_runs/no_anchor_video_focus_portfolio_candidates_20260620.json`
- aggressive scheduler:
  `local_runs/no_anchor_fullscore_scheduler_video_focus_direct_portfolio_20260620.json`

Moderate branch:

- max `6` edits per portfolio;
- `16` candidates, `8` selected by scheduler;
- selected rows move `52..56` tracklets;
- predicted full IDF1 range in the selected scheduler rows:
  `0.658162..0.658897`;
- exporter verified all scheduler ranks `1..8`.

Aggressive branch:

- max `10` edits per portfolio;
- `8` selected rows;
- selected rows move `92..96` tracklets;
- exporter verified all scheduler ranks `1..8`;
- this branch is high-risk and should be scored after the moderate queue.

Run moderate first when Pluto recovers:

```bash
bash kit/run_no_anchor_scheduler_manifest_fullscore.sh \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_video_focus_moderate_portfolio_20260620.json \
  --ranks 1,2,3,4,5,6,7,8 \
  --run-name no_anchor_video_focus_moderate_portfolio_fullscore_20260620
```

Then score the aggressive exploratory queue only if budget remains:

```bash
bash kit/run_no_anchor_scheduler_manifest_fullscore.sh \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_video_focus_direct_portfolio_20260620.json \
  --ranks 1,2,3,4,5,6,7,8 \
  --run-name no_anchor_video_focus_direct_portfolio_fullscore_20260620
```

Research consequence:

- this is a more direct attack on the current e2e bottleneck than plain
  all-video portfolio ranking;
- it still does not prove e2e `0.70`;
- if a direct portfolio improves full IDF1, the next task is to replace the
  eval-feedback scheduler with a no-GT production selector based on video
  density/confidence/tracklet metadata.

## 2026-06-20 Deli distillation and local export executor

The public Deli AutoResearch protocol was distilled into the active VLINCS loop
as an execution discipline: durable state, no stopping at "ready to submit",
proposer/executor/scorer/judge separation, score-drop-as-evidence, and
structural pivots after repeated stale iterations.

New local execution path:

- `kit/run_no_anchor_scheduler_manifest_sample_fullscore.py`
- `kit/evaluate_sample_assignments_full.py --allow-no-gt-export`
- report:
  `reports/no_anchor_deli_distillation_local_export_20260620.md`

The local runner materializes scheduler rows into assignment CSVs, merges them
with DS1 detection-level tracklet parquets, and exports submission zips without
requiring PostgreSQL.  If DS1 GT is mounted locally it also scores through the
canonical evaluator.

Current local limitation:

- this Mac does not have DS1 GT mounted, so local runs produce `gt_available=false`;
- e2e best remains `0.655240` until canonical DS1 scoring verifies a new row.

Artifacts produced:

- recovered local base:
  `local_runs/remote_h100_test_3_20260620/no_anchor_recovered_90family_base_assignments_20260620.csv`
- local moderate top-8 export:
  `local_runs/no_anchor_video_focus_moderate_portfolio_recovered_base_local_export_20260620`

Recovered-base caveat:

- the true remote base CSV is still absent locally;
- the recovered `90`-family base is a majority-vote reconstruction from seven
  returned assignment CSVs, with `40/7487` unstable seqs;
- use it for local materialization and zip generation, not as a canonical
  improvement claim.

## 2026-06-20 cross-queue portfolio composition

Remote scoring was still unavailable, so I tried a structural diversity pivot:
merge selected rows from the hub-bridge portfolio queues and the direct
video-focus portfolio queues, then re-run the no-anchor portfolio composer
across that union.

Artifacts:

- union scheduler:
  `local_runs/no_anchor_union_scheduler_existing_portfolios_20260620.json`
- relaxed cross-queue candidates:
  `local_runs/no_anchor_crossqueue_portfolio_relaxed_candidates_20260620.json`
- cross-queue scheduler:
  `local_runs/no_anchor_fullscore_scheduler_crossqueue_portfolio_20260620.json`
- local export:
  `local_runs/no_anchor_crossqueue_portfolio_recovered_base_local_export_20260620`
- report:
  `reports/no_anchor_crossqueue_portfolio_relaxed_candidates_20260620.md`

Result:

- union rows: `41`;
- compatible cross-queue rows: `7`;
- scheduler selected: `3`;
- selected predicted full IDF1: `0.663182`, `0.661996`, `0.660630`;
- moved tracklets: `56`, `60`, `72`;
- pair F1: `0.769014`, `0.770636`, `0.770646`;
- all three exported as 10-video local submission zips with `gt_available=false`.

Interpretation:

- this is the strongest current no-anchor proxy queue, narrowly above the
  previous hub-bridge portfolio proxy top of `0.662663`;
- still not close enough to claim progress toward `0.70`;
- the next structural step should either obtain canonical scoring for these
  candidates or improve the full-score proxy by learning from more verified
  DS1 full-score rows.
