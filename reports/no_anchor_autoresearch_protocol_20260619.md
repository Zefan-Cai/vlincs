# No-Anchor AutoResearch Protocol - 2026-06-19

Goal:

- Continue no-anchor VLINCS global-ID research until both model-side global-ID
  assignment and end-to-end submission exceed the target gate.
- Current verified state: model-side no-anchor is already above 70; end-to-end
  best remains below 70, so the active bottleneck is resolution/admission plus
  delivery-row structure.

Distilled from Deli AutoResearch:

- Treat long-running research as a protocol, not as a single chat session.
- Persist state in files and reports, because context compaction and remote
  session loss are normal failure modes.
- Separate worker and evaluator roles: a branch can propose or execute a repair,
  but the gate is always quantitative metrics.
- Use anti-loop constraints: a new direction must differ from prior failed
  directions, and every branch needs a measurable exit condition.
- Ready means execute: after a branch has a script, validation, and command, the
  next action is submission/execution when compute is reachable.
- Honest score movement matters: if an external check or metric drops the score,
  record the lower score and pivot from that evidence.

Sources checked:

- `https://victorchen96.github.io/auto_research/framework.html`
- `https://victorchen96.github.io/auto_research/paper.html`
- `https://victorchen96.github.io/blog_self_play_story.html`

Self-play translation for VLINCS:

- Treat each no-anchor identity resolver as both a proposer and an opponent.
  A source that proposes a component is not trusted until competing sources,
  cannot-link evidence, and delivery metrics fail to refute it.
- Prior knowledge is useful for bootstrapping features, but the loop should not
  assume one hand-built prior is the ceiling.  Create diverse opponents:
  time/trajectory agglomeration, OSNet source variants, face/local-grid
  variants, cannot-link splitters, verifier splitters, and admission filters.
- Score movement is allowed to go down.  A lower score after a stricter
  citation/check/evaluator pass is a real finding, not a failed run.
- For this project, "GRPO/self-play" maps to bounded research games rather
  than RL weight updates: proposer generates a no-anchor global-ID assignment;
  opponent searches for false merges/splits; evaluator scores pair metrics and
  full IDF1; the next iteration must either keep the improvement or pivot the
  structural assumption.
- The current active self-play arena is source switching: different no-anchor
  assignment sources compete per video/camera, using GT only for diagnostic
  scoring.  Any winning oracle policy must later be replaced by a no-GT
  selector before it can count as a production model.

Current no-anchor task map:

1. Worker branch A: per-video admission oracle.
   - Script: `kit/no_anchor_submission_pervideo_filter_oracle.py`
   - Purpose: estimate whether delivery-row filtering can close the e2e gap.
   - Gate: full IDF1 materially above `0.654739`.
   - If it does not move, stop spending effort on q-threshold delivery filters.

2. Worker branch B: stateful component resolution.
   - Script: `kit/no_anchor_assignment_state_policy_sweep.py`
   - Purpose: convert forced IDs into evidence states:
     `committed`, `provisional`, `pending`, and `forced_conflict`.
   - New repair: cannot-link graph coloring via `color_forced` and
     `color_pending_forced`.
   - Gate: tracklet-pair F1 and full IDF1 both improve; if only pair F1
     improves, the remaining problem is submission/detection structure.

3. Worker branch C: unsupervised replacement for GT-selected oracles.
   - Trigger: branch A or B shows a meaningful oracle lift.
   - Candidate signals: component state, detection confidence, bbox area,
     stream/camera span, cannot-link rate, tracklet length, feature margin.
   - Gate: same policy chosen without GT keeps most of the oracle gain.

4. Worker branch D: self-play source switching.
   - Script: `kit/no_anchor_assignment_video_switch.py`
   - Local proxy script: `kit/sample_assignment_video_source_switch.py`
   - No-GT selector scripts:
     `kit/sample_assignment_source_selector.py` and
     `kit/no_anchor_assignment_source_selector.py`
   - Purpose: make no-anchor assignment sources compete as proposers/opponents
     and identify whether per-video source diversity can repair weak cameras.
   - Gate: full IDF1 above the current `0.654739` e2e artifact on DS1, or a
     local proxy lift large enough to justify a remote full run.
   - Current local signal: a sparse precision-overlay selector using no GT for
     policy selection chooses `target_d1_c50` and reaches `0.739426` IDF1 on
     the two-video parquet proxy with no anchors, but this is not a DS1
     completion gate.

5. Evaluator branch: external metric gate.
   - Model-side metrics: tracklet-pair precision/recall/F1.
   - End-to-end metrics: IDF1, HOTA, AssA, DetPr, DetRe, IDs.
   - Required reporting: per-video deltas, case examples, command provenance.

6. Local fallback evaluator branch.
   - Script: `kit/sample_assignment_state_policy_sweep.py`
   - Purpose: continue useful no-anchor ablations while Pluto is unreachable.
   - Metric label: `sample_parquet_gt_same_detection_boxes`.
   - Constraint: never treat this proxy as the DS1 end-to-end completion gate.
   - Current signal: full-coverage sample IDF1 can reach `0.731491`, but MCAM08
     remains weak and precision-only state repair collapses recall.

Anti-loop rules for the next iterations:

- Do not run another global threshold sweep unless it introduces a new signal.
- Do not call an oracle result a solution; it is only a bottleneck detector.
- Do not accept higher tracklet-pair F1 as sufficient if full IDF1 regresses.
- If a branch repeats the same failure twice, write the failure pattern and move
  to the next distinct branch.
- When Pluto SSH recovers, execute the prepared remote launcher directly:
  `bash kit/run_no_anchor_remote_recovery_experiments.sh --case all`.
- If Pluto stays unreachable, run only local proxy experiments that change the
  next remote decision.  Do not keep repeating state/color policies after the
  observed precision-recall collapse.

Current prepared execution:

```bash
bash kit/run_no_anchor_remote_recovery_experiments.sh --case all
```

## 2026-06-20 Addendum: Deli AutoResearch Distillation

External sources checked:

- `https://victorchen96.github.io/auto_research/framework.html`
- `https://victorchen96.github.io/auto_research/paper.html`
- `https://victorchen96.github.io/blog_self_play_story.html`

Distilled protocol changes for VLINCS:

- Long-horizon identity research is now treated as a file-backed state machine:
  `progress.json`, `directions_tried.json`, `findings.jsonl`, and reports are
  the source of truth, not chat memory.
- A proposal is not a result until an independent evaluator has either scored
  it or refuted it.  For us, the evaluator is pair metrics plus full DS1 IDF1,
  with delivery coverage as a first-class gate.
- Score decreases are findings.  The Deli self-play run explicitly lowered its
  score after citation/evidence failures; the VLINCS analogue is that a high
  tracklet-pair score can be a negative result if full delivery collapses.
- Structural pivots beat tactical threshold churn.  After a family repeats the
  same first edit or the same precision/recall failure, the next branch must
  change the proposer structure, feature family, candidate-selection rule, or
  acceptance model.
- The self-play mapping is proposer/opponent/evaluator:
  proposer = no-anchor assignment generator;
  opponent = cannot-link, delivery, and side-effect refuter;
  evaluator = pair/full metrics and case-level evidence.

Immediate implementation from this distillation:

- The learned full-proxy acceptor was audited for honesty.  The first compact
  model looked excellent on comparable rows, but over-ranked a state-policy
  branch with `pair F1=0.954691` because that branch delivered only
  `421-431` tracklets and had real full IDF1 near `0.0854`.
- That failure is now a required negative example, not an excluded outlier.
  Delivery-aware training includes low-IDF1 rows and no-GT delivery features:
  `output_tracklets`, `eval_tracklets`, `coverage_ratio`,
  `delivery_tracklets_min`, and `delivery_tracklets_mean`.
- Candidate ranking now excludes derived `pair_candidates` and
  `full_proxy*training_audit` artifacts from harvest input, preventing the
  scorer from reading its own reports as future evidence.
- Candidate ranking also applies a delivery gate only when the candidate row
  already exposes delivery fields.  This filters known low-coverage state
  policies without killing unscored full-assignment repair candidates whose
  delivery fields are not materialized until full evaluation.

Expected artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_20260619.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_state_policy_quality060_best_assignments_20260619.csv`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_quality060_pervideo_conf_oracle_20260619.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_video_switch_quality060_20260619.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_video_switch_quality060_best_assignments_20260619.csv`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_source_selector_quality060_20260619.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_assignment_source_selector_quality060_best_assignments_20260619.csv`
- `/mnt/localssd/vlincs_reid_runs/logs/no_anchor_state_policy_quality060_20260619.log`
- `/mnt/localssd/vlincs_reid_runs/logs/no_anchor_quality060_pervideo_conf_oracle_20260619.log`
- `/mnt/localssd/vlincs_reid_runs/logs/no_anchor_assignment_video_switch_quality060_20260619.log`
- `/mnt/localssd/vlincs_reid_runs/logs/no_anchor_assignment_source_selector_quality060_20260619.log`

## 2026-06-19 Loop Update: Deli Distillation To VLINCS

Additional source reading:

- `https://victorchen96.github.io/auto_research/framework.html`
  describes Deli_AutoResearch as a protocol for long-horizon autonomous work:
  state is persisted to files, workers and evaluators are separated, stalls are
  detected explicitly, and anti-loop constraints force a new direction after
  repeated local failures.
- `https://victorchen96.github.io/auto_research/paper.html` lists Paper #4,
  "Self-Play in the Age of Foundation Models", as the self-play survey.
- `https://victorchen96.github.io/blog_self_play_story.html` adds the key
  operational lesson: the agent ran real 285B GRPO experiments, hit multiple
  submission/runtime failures, fixed them, and accepted score decreases when
  external citation/evidence checks found problems.

VLINCS translation for this loop:

- The proposer is an assignment-source generator, not a chat answer.
- The opponent is a refuter: full DS1 IDF1/HOTA, tracklet-pair metrics,
  cannot-link evidence, and per-video deltas.
- The policy can only count as production if source selection uses no GT,
  no anchors, and no post-hoc hand-picked video switches.
- GT is allowed only after prediction to label a branch as positive, neutral,
  negative, or diagnostic.

Latest no-anchor DS1 outcome:

| branch | selector uses GT | full IDF1 | HOTA | pair F1 | pair P | pair R | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| base current | no | `0.652624` | `0.516393` | `0.768743` | `0.813273` | `0.728836` | still best among these sources |
| conservative sparse overlay, global | no | `0.647694` | `0.511262` | `0.760766` | `0.800733` | `0.724599` | negative |
| conservative sparse overlay, per-video | no | `0.649623` | `0.512975` | `0.763389` | `0.804257` | `0.726474` | negative |
| balanced sparse overlay, global | no | `0.618557` | `0.482834` | `0.719390` | `0.744874` | `0.695591` | negative |
| balanced sparse overlay, per-video | no | `0.623119` | `0.487056` | `0.725458` | `0.752790` | `0.700041` | negative |
| per-video confidence oracle | yes | `0.654800` | `0.518170` | n/a | n/a | n/a | bottleneck detector only |

Negative-result interpretation:

- Full-data target-agglomeration sources are useful diagnostics: the best sparse
  source by post-hoc pair F1 was `target_t640_d10_c0p75`, with pair
  `F1/P/R = 0.732927 / 0.785117 / 0.687244`.
- However, applying those sparse target sources as overlays damages full DS1
  delivery.  The failure is not only that the conservative selector picked
  overly small components; the balanced selector deliberately chose the
  `target_t640_*` family and regressed harder.
- Therefore, the next branch should not keep tuning sparse-overlay source
  selection.  It should generate a new full-delivery assignment source whose
  namespace is stable across all rows, or use target-agglomeration evidence as
  a merge/split verifier inside the current assignment instead of replacing IDs
  on sparse subsets.

Current active remote note:

- The old-source greedy video-switch oracle was stopped on h100-test-3 after
  about 73 minutes.  It had scored all single historical sources at or below
  the base `0.652624` full IDF1, so it was closed as a low-priority oracle
  relative to generating new target/admission evidence.

Cheap-gate example from the next branch:

- A naive full-delivery `target_clusters=640` agglomeration source covered all
  `9734` resolve tracklets with one coherent namespace, but its tracklet-pair
  `F1/P/R` was only `0.486063 / 0.554703 / 0.432540`.
- Because the current base pair is `0.768743 / 0.813273 / 0.728836`, the branch
  was stopped before full HOTA/IDF1 evaluation.
- This is now a protocol rule for VLINCS: dense target clustering must pass a
  cheap pair-evidence gate before it is allowed to consume full DS1 evaluation
  time.

## 2026-06-20 Loop Update: Conflict-Reassign Candidate Search

Branch:

- Convert strict conflict-subcluster reassign from a broad threshold sweep into
  a proposer/evaluator loop.
- The proposer builds a no-GT candidate table of source conflict islands and
  existing clean target components.
- The evaluator only scores greedy top-prefix candidate sets.

Artifacts:

- narrow self-play:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_narrow_selfplay_pair_20260620.json`
- candidate-search smoke:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_tiny_pair_20260620.json`

Outcome:

| branch | pair F1 | precision | recall | verdict |
| --- | ---: | ---: | ---: | --- |
| strict narrow self-play | `0.772654` | `0.818667` | `0.731538` | reproduces current model best |
| candidate-search tiny | `0.772654` | `0.818667` | `0.731538` | validates candidate-table path |

Protocol lesson:

- The known strict reassign edit is stable; repeated local neighborhoods return
  the same 8-tracklet move.
- Broad/source-wide online generation is too expensive relative to its score
  movement and should not be repeated as another threshold grid.
- The next valid branch must change the proposer structure: precompute source
  islands offline, learn a source repairability proxy, or generate stronger
  cross-tracklet positives before attempting another reassign expansion.

## 2026-06-19 Loop Update: Target Teacher Refutation

Branch:

- Use sparse target-agglomeration sources only as weak teachers for component
  merge evidence, not as delivered-ID replacements.

Artifacts:

- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_target_teacher_merge_pair_20260619.json`
- remote CSV:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_target_teacher_merge_pair_20260619.csv`
- local JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_teacher_merge_pair_20260619.json`
- local CSV:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_target_teacher_merge_pair_20260619.csv`

Observed gate:

- base pair:
  `0.768743 / 0.813273 / 0.728836`
- target-teacher edge count:
  `2016`
- component conflicts:
  `1588`
- accepted-edge range across the grid:
  `1..6`
- all `2160` scored rows have the same rounded pair F1:
  `0.768743`.

Protocol decision:

- Mark this branch neutral/negative.  It is not harmful at the rounded pair
  metric, but it does not create a meaningful improvement.
- Do not widen target-teacher thresholds unless a new teacher source changes
  the positive-label semantics.
- The next branch must generate new positive evidence instead of recycling
  assignment namespaces.  Candidate directions:
  intra-tracklet crop augmentation positives, body-part consistency, clothing
  attribute agreement, face-gated positives, and cannot-link hard negatives.

## 2026-06-19 Loop Update: Repair Target And Positive-Generation Refutation

Eval-only repair decomposition:

- remote pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_oracle_repair_decomposition_paironly_20260619.json`
- remote full top-1 JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_oracle_repair_decomposition_full_top1_20260619.json`
- local pair-only JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_oracle_repair_decomposition_paironly_20260619.json`

Observed target shape:

- current production best:
  full IDF1 about `0.654739`;
- all-current-tracklet GT-majority oracle:
  full IDF1 `0.711353`;
- split top 40 false-merge components plus merge top 40 false-split GTs:
  pair `F1/P/R = 0.996040 / 0.999490 / 0.992614`,
  full IDF1 `0.705997`.

Protocol decision:

- The remaining e2e headroom is real but narrow.
- The required model behavior is now specific: recover large false-split
  identities while keeping false-merge precision near current levels.
- Do not spend more time on detection confidence filters as the primary path;
  they have already saturated around `0.6548`.

No-anchor self-play positive check:

- script:
  `kit/no_anchor_assignment_selfplay_component_merge_sweep.py`
- normal pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_selfplay_component_merge_pair_20260619.json`
- low-threshold sanity JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_selfplay_component_merge_lowthr_pair_20260619.json`

Result:

- current-component split positives:
  `158`;
- candidate negatives:
  `4448`;
- normal verifier accepted:
  `0` real edges;
- low-threshold sanity accepted:
  `6` real edges;
- pair metric unchanged at:
  `0.768743 / 0.813273 / 0.728836`.

Protocol decision:

- Current-component self-play positives are refuted as a sufficient positive
  generation mechanism.
- The next branch must use image-grounded positives rather than assignment
  self-consistency: intra-tracklet crop augmentation, part-level clothing
  stability, face-gated positives, or generated/augmented body evidence.

## 2026-06-19 Loop Update: Deli Protocol Applied To Split And Sample Positives

Deli AutoResearch distillation used in this loop:

- ready means execute: once the branch had a script and a scalar gate, it was
  run on h100-test-3 without waiting for more instruction;
- score movement can go down: negative pair/full movement was recorded as a
  useful refutation;
- after repeated split/merge refutations, pivot the structure rather than
  widening the same thresholds.

Split-then-merge branch:

- script:
  `kit/no_anchor_assignment_split_then_merge_sweep.py`
- remote JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_split_then_merge_pair_20260619.json`
- local JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_split_then_merge_pair_20260619.json`

Result:

- base pair:
  `0.768743 / 0.813273 / 0.728836`;
- best split-then-merge pair:
  `0.633061 / 0.840151 / 0.507874`;
- best row split `31` components, rewrote `1957` tracklets, then accepted `9`
  visual merges.

Protocol decision:

- cannot-link coloring is refuted as a no-GT substitute for the oracle split.
  It raises precision but destroys recall.  Future split policies must be
  local/provisional, not whole-component coloring.

Image-grounded sample-positive branch:

- extractor update:
  `kit/extract_tracklet_osnet_features.py --save-sample-features`;
- sample feature artifact:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_osnet_msmt_s7_true_samples_20260619.npz`;
- verifier script:
  `kit/no_anchor_sample_positive_edge_verifier.py`;
- max-probability JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_sample_positive_edge_verifier_pair_20260619.json`;
- top-mean JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_sample_positive_edge_verifier_topmean_pair_20260619.json`.

Result:

- extracted `66324` crops for `9734` tracklets, with `3` missing crops;
- training labels: `9660` intra-tracklet positives and `12000` cannot-link
  negatives, no anchors and no GT;
- max-probability merge best:
  `0.768734 / 0.813252 / 0.728836`, accepted `29` merges;
- top-mean aggregation best:
  `0.768743 / 0.813273 / 0.728836`, accepted `27` merges but no measurable
  pair lift.

Protocol decision:

- intra-tracklet crop positives are too easy.  They train a perfect internal
  verifier but do not transfer into cross-tracklet component merge evidence.
- Face-gated positives are also too sparse at the current component-edge level:
  among the separability top-500, non-forbidden `face_sim >= 0.7` has `0`
  candidate edges; `face_sim >= 0.5` has only `2`, with one impure large edge.
- The next positive-generation branch must create cross-tracklet positives:
  short-gap same-stream continuation, generated pose/clothing variants, or
  attribute-level consistency, then evaluate with the same pair gate.

## 2026-06-19 Loop Update: Cross-Tracklet Continuation Positives

Purpose:

- test the next structural hypothesis after same-tracklet crop positives failed:
  generate no-GT positives from cross-tracklet evidence rather than from a
  single tracklet;
- positives are same-video/same-camera short-gap pairs with plausible spatial
  continuation and high sample-level OSNet agreement;
- the verifier sees only visual sample-pair features, so the temporal/geometric
  rule is label construction, not a feature shortcut.

Artifacts:

- verifier script:
  `kit/no_anchor_continuation_positive_edge_verifier.py`;
- pair JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_continuation_positive_edge_verifier_pair_20260619.json`;
- local JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/no_anchor_continuation_positive_edge_verifier_pair_20260619.json`;
- peel-merge script:
  `kit/no_anchor_continuation_peel_merge_sweep.py`;
- peel-merge JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_continuation_peel_merge_pair_20260619.json`.

Continuation-positive result:

- positive label rule generated `6694` cross-tracklet positives from `62305`
  considered short-gap pairs;
- hard negative pool had `174788` cannot-link overlap candidates; the run used
  the top `12000` hard visual negatives;
- train AUC/AP:
  `0.936908 / 0.906549`;
- base pair:
  `0.768743 / 0.813273 / 0.728836`;
- best continuation-verifier merge:
  `0.768740 / 0.813266 / 0.728837`, accepted `1` merge.

Interpretation:

- Cross-tracklet positives are much healthier than same-tracklet positives:
  the verifier no longer memorizes with perfect train metrics.
- However, the production merge gate is still slightly negative.  The added
  true mass is too small relative to the added predicted-pair mass.

Peel-merge follow-up:

- motivation: high-probability continuation edges often point to true
  false-split candidates but are blocked by cannot-link inside impure large
  components;
- repair: peel only the conflicting nodes from the large component, then merge
  the small component into the compatible remainder.

Result:

- best peel-merge pair:
  `0.768597 / 0.813240 / 0.728601`;
- best row accepted `1` peel repair, peeled `1` node, and still regressed.

Protocol decision:

- edge-guided peel is also too destructive in its current deterministic form.
- Keep the continuation-positive verifier as a reusable diagnostic signal, but
  do not promote it to full scoring.
- The next branch needs either a better cross-tracklet positive source or a
  probabilistic state output that can mark conflicting subgraphs provisional
  without forcibly relabeling them.

## 2026-06-19 Loop Update: Clothing/Body Consistency Refutation

Why this branch:

- Deli AutoResearch says repeated local failures should trigger a structural
  pivot, not a wider threshold sweep.
- Same-tracklet crop positives were too easy, and short-gap continuation
  positives were healthier but still not production-useful.
- The next distinct source was body-part/clothing consistency: use pose/color
  and color-hist agreement as image-grounded evidence for weak cross-tracklet
  positives and for intra-component conflict quarantine.

Artifacts:

- merge verifier:
  `kit/no_anchor_clothing_positive_edge_verifier.py`;
- conflict quarantine:
  `kit/no_anchor_clothing_conflict_quarantine_sweep.py`;
- loose clothing verifier JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_clothing_positive_edge_verifier_pair_20260619.json`;
- strict clothing verifier JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_clothing_positive_edge_verifier_strict_pair_20260619.json`;
- clothing conflict quarantine JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_clothing_conflict_quarantine_pair_20260619.json`;
- OSNet-only conflict quarantine JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_osnet_conflict_quarantine_pair_20260619.json`;
- local copies:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260619/`.

Weak-label setup:

- positives are no-GT same-video/same-camera continuations with plausible
  geometry, OSNet sample agreement, pose/body-part color agreement, and
  color-hist agreement;
- negatives are same-stream overlap cannot-link pairs, selected as hard visual
  negatives;
- the verifier sees visual evidence only: OSNet sample-pair features plus
  posecolor/colorhist similarities.

Pair gate:

| branch | pair F1 | precision | recall | edit count |
| --- | ---: | ---: | ---: | ---: |
| base assignment | `0.768743` | `0.813273` | `0.728836` | `0` |
| clothing verifier, loose | `0.768689` | `0.813145` | `0.728842` | `17` merges |
| clothing verifier, strict | `0.768740` | `0.813266` | `0.728837` | `1` merge |
| clothing conflict quarantine | `0.768743` | `0.813273` | `0.728836` | `0` splits |
| OSNet-only conflict quarantine | `0.768658` | `0.813672` | `0.728364` | `7` splits |

Training/diagnostic details:

- loose clothing positives: `12000` positives from `117939` considered
  continuation pairs; train AUC/AP `0.897736 / 0.909390`;
- strict clothing positives: `6557` positives from `62305` considered pairs;
  train AUC/AP `0.943740 / 0.911267`;
- internal conflict-node centroid diagnostic:
  `3964` conflict nodes were inspected.  Pose/color features did not separate
  them: pose_color_only median centroid similarity was `0.9486`, colorhist
  median was `0.9634`, and blend median was `0.8768`.  OSNet-only separated
  more nodes, but conservative splits still reduced pair F1.

Protocol decision:

- Body-part/color features are useful as auxiliary provenance but are refuted
  as a direct production merge permission or conflict-node quarantine signal.
- The current component errors are not explained by a few low-similarity visual
  outliers inside conflicted components.
- Next structural branch should stop editing the existing components directly
  and instead build a stronger positive generator: generated pose variants,
  cross-video clothing/face agreement with explicit hard-negative mining, or a
  candidate-retrieval model that outputs committed/provisional states before
  forced delivery.

## 2026-06-20 Loop Update: Mass-Ranked Bridge Provenance

Why this branch:

- The edge-table oracle-coverage audit showed that the current edge tables cover
  only `0.00432865` of the missing true-pair mass.  That route is recall-limited
  and cannot be the main path to full IDF1 `0.70`.
- Deli AutoResearch says stale loops need a structural pivot.  The pivot here is
  from edge-table tiny-fragment cleanup to high-mass component/hub bridge
  generation.
- The self-play opponent needs provenance: a row must expose the actual
  accepted bridge edges before a full-score slot is spent.

Implementation:

- `kit/no_anchor_assignment_multiview_merge_sweep.py` now records
  `accepted_preview`, accepted-edge score means, view-vote means, and no-GT
  mass-proxy aggregates for accepted multiview bridges.
- The multiview solver now supports `--rank-by mass_proxy`, ranking candidate
  rows by likely false-split mass before pair-score tie breakers.
- `kit/run_no_anchor_false_split_budget_merge.sh` now passes
  `--rank-by mass_proxy` and `--accepted-preview-n 40`.
- `kit/analyze_no_anchor_full_proxy_training.py` now trains/evaluates with
  accepted-edge provenance and mass features.

Full-proxy audit:

- audit JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_full_proxy_training_audit_mass_features_20260620.json`
- model JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_full_proxy_mass_features_ridge_model_20260620.json`
- row count: `32`
- feature count: `34`
- full-IDF1 range: `0.602445..0.654009`
- LOOCV ridge: correlation `0.969076`, MAE `0.001736`, RMSE `0.004556`

Current gate:

| gate | value | status |
| --- | ---: | --- |
| no-anchor pair F1 | `0.775234` | above `0.70` |
| no-anchor pair precision | `0.820504` | above `0.70` |
| no-anchor pair recall | `0.734698` | above `0.70` |
| best promoted full IDF1 | `0.655240` | below `0.70` |

Protocol decision:

- Do not mark the goal complete: end-to-end is still below target.
- Do not continue edge-table target-fragment audits as the main route.
- When Pluto recovers, run the mass-ranked multiview bridge and high-mass
  component-merge launchers, then evaluate only through the frozen full DS1
  result gate.

## 2026-06-20 Loop Update: Full-Score Scheduler Diversity Guard

Why this branch:

- Full-score budget is scarce while Pluto is unstable.
- The refreshed mass-feature proxy was good enough for ordering candidates, but
  the scheduler still risked spending slots on duplicate JSON/CSV views of the
  same edit family.
- Deli anti-loop rule: if a branch has already saturated, de-duplicate it and
  force structural diversity before spending more expensive evaluation.

Implementation:

- `kit/no_anchor_fullscore_scheduler.py` now recovers CSV-row provenance from
  the sibling JSON using `_source_rank`.
- This avoids scanning a whole JSON for every CSV row and restores component
  family keys such as `conflict_subcluster_reassign:component:21->0`.
- Global artifact families now canonicalize by file stem, so `foo.csv` and
  `foo.json` no longer occupy separate family slots.
- Added `--max-per-mode`; default `0` keeps historical uncapped selection.

Strict scheduler result:

- JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_fullscore_scheduler_mass_features_diverse_20260620.json`
- report:
  `/Users/zcai/Codex/vlincs_reid_by_search/reports/no_anchor_fullscore_scheduler_mass_features_diverse_20260620.md`
- raw candidates: `202010`
- eligible candidates: `696`
- selected unique families: `3`

Selected remote full-score queue:

| rank | family | predicted full | pair F1 | P | R |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `conflict_subcluster_reassign_candidate_search:component:32->15` | `0.658179` | `0.771045` | `0.816632` | `0.730278` |
| 2 | `conflict_subcluster_reassign_candidate_search:component:21->19` | `0.658302` | `0.767329` | `0.814359` | `0.725434` |
| 3 | `conflict_subcluster_reassign:component:21->0` | `0.655254` | `0.772654` | `0.818667` | `0.731538` |

Component/multiview exploratory scheduler:

- JSON:
  `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_fullscore_scheduler_component_multiview_explore_20260620.json`
- report:
  `/Users/zcai/Codex/vlincs_reid_by_search/reports/no_anchor_fullscore_scheduler_component_multiview_explore_20260620.md`
- best existing component/multiview predicted full-IDF1: `0.653753`
- verdict: below current production best `0.655240`, so old component/multiview
  artifacts should not consume strict full-score slots.

Protocol decision:

- When Pluto is reachable, first full-score the three deduplicated conflict
  candidates above.
- In parallel or next, generate new mass-proxy component/hub bridge rows; do not
  recycle the old component/multiview grids as if they were the high-mass branch.

## 2026-06-19 Loop Update: Visual Opponent and Component Surgery

Why this branch:

- The clothing/body branch showed that feature-derived weak labels are useful
  as provenance but too weak as direct merge permissions.
- The next Deli-style self-play move was to make the evaluator more visual:
  export top component-pair candidates as image montages, have a visual
  opponent accept/reject same-person evidence, then let the graph constraints
  challenge those accepted edges.

Artifacts:

- candidate montage exporter:
  `kit/export_no_anchor_candidate_edge_montages.py`;
- constrained/no-forbidden visual edge applier:
  `kit/apply_no_anchor_visual_edge_decisions.py`;
- sampled micro-component surgery:
  `kit/apply_no_anchor_visual_edge_surgery.py`;
- montage directory:
  `/mnt/localssd/vlincs_reid_runs/vlm_edge_montages_split_t040_m16_20260619`;
- montage metadata:
  `/mnt/localssd/vlincs_reid_runs/vlm_edge_montages_split_t040_m16_20260619.json`;
- visual decision JSON:
  `/mnt/localssd/vlincs_reid_runs/codex_visual_edge_decisions_split_t040_m16_20260619.json`.

Setup:

- Start from the verifier-split assignment
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_verifier_split_t040_m16_assignments_20260619.csv`.
- Score component edges using OSNet plus posecolor/colorhist/CLIP-DINO views.
- Export the top `30` candidate component-pair montages, with top row = source
  component samples and bottom row = target component samples.
- Gemini was attempted first, but both configured Gemini API keys returned
  expired-key errors.  The branch therefore used a conservative Codex visual
  contact-sheet verifier and recorded this limitation explicitly.
- GT was not used for visual decisions.  GT entered only after prediction for
  pair/full metrics.

Results:

| branch | pair F1 | precision | recall | full IDF1 | key observation |
| --- | ---: | ---: | ---: | ---: | --- |
| verifier-split base | `0.768837` | `0.813577` | `0.728761` | `0.652806` | starting point |
| multiview merge | `0.768847` | `0.813561` | `0.728792` | `0.652812` | tiny pair/full gain, not enough |
| visual verifier constrained | `0.768837` | `0.813577` | `0.728761` | `0.652806` | `15/16` visual positives blocked by cannot-link |
| visual verifier no-forbidden | `0.768927` | `0.813579` | `0.728921` | `0.652830` | visual signal exists, but weak |
| sampled surgery | `0.767834` | `0.813322` | `0.727164` | not scored | local surgery destroys recall |

Interpretation:

- The visual opponent did find plausible false-split evidence: green sweater,
  pink sweater, gray `67` hoodie, floral outfit, light patterned coat, and
  white/black outfit candidates were visually convincing.
- The graph opponent refuted direct production merge: most visual positives
  hit large components that already contain temporal/cannot-link conflicts.
- Ignoring cannot-link gives a tiny positive diagnostic lift, but this is not a
  safe production rule and still does not approach the e2e gate.
- Sampled surgery confirms that crop-level evidence is too small a unit of
  action.  Pulling a handful of montage tracklets out of a large component
  loses more recall than it gains in purity.

Protocol decision:

- Stop spending iterations on direct component merge, conflict-node peeling,
  or sampled-crop surgery against the same current component graph.
- The next branch should be stateful: represent conflicted large components as
  provisional evidence containers, retrieve candidate subclusters, and only
  commit identities when a whole subcluster has enough positive evidence and no
  hard constraints.
- A stronger verifier can still help, but it must act at tracklet-subcluster
  granularity, not at full-component or single-crop granularity.

## 2026-06-19 Loop Update: Conflict Subcluster Extraction

Why this branch:

- The visual-opponent loop showed that hard positives often point inside impure
  large components.
- The next distinct action was subcluster extraction: audit only conflicted
  large components and extract small internally coherent visual islands, rather
  than splitting the whole component or merging entire components.

Artifacts:

- script:
  `kit/no_anchor_assignment_conflict_subcluster_sweep.py`;
- broad pair result:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_pair_20260619.json`;
- conservative pair result:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_pair_20260619.json`;
- conservative full result:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_full1_20260619.json`;
- detection-filter result:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_subcluster_conservative_detection_filter_20260619.json`.

Results:

| branch | pair F1 | precision | recall | full IDF1 | q03/filter IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base assignment | `0.768743` | `0.813273` | `0.728836` | approx `0.6526` | `0.654739` |
| broad extraction | `0.767262` | `0.818155` | `0.722329` | not scored | not scored |
| conservative extraction | `0.768756` | `0.814231` | `0.728091` | `0.652924` | `0.653081` |

Interpretation:

- The conservative extractor found one credible 4-tracklet island
  (`3171,3442,3491,3547`) inside component `38`.
- This is a useful precision/provenance signal but not a production resolver:
  the tiny pair gain does not survive to a promoted e2e gain.
- The loop again supports stateful resolution.  Suspicious islands should
  become query/evidence nodes with `provisional` state, not immediate forced
  global IDs.

Protocol decision:

- Keep conflict-subcluster extraction as an audit feature for the next model.
- Do not keep sweeping extraction thresholds unless a new verifier can retrieve
  whole compatible subclusters and recover recall.

## 2026-06-19 Loop Update: Provisional Relink and Soft Cannot-Link

Why this branch:

- Conflict subclusters were useful as suspicious evidence, but hard extraction
  lost recall.
- Face/OSNet component-edge diagnostics showed that some visually plausible,
  GT-positive edges were blocked by hard temporal cannot-link (`is_forbidden=1`).
- The next self-play opponent therefore tested whether cannot-link should be
  evidence-calibrated rather than absolute.

Experiments:

1. Provisional relink:
   - take conflict-derived query subclusters;
   - retrieve visually compatible tracklets from other components;
   - peel query plus retrieved neighbors into a new ID.
2. Soft overlap cannot-link:
   - precompute same-stream overlap pair stats from bbox IoU and OSNet visual
     similarity;
   - remove cannot-link only for duplicate-like overlap pairs;
   - run multiview component merge with the softened forbidden set.

Artifacts:

- `kit/no_anchor_assignment_provisional_relink_sweep.py`;
- `kit/no_anchor_assignment_soft_overlap_merge_sweep.py`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_provisional_relink_narrow_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_narrow_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_relaxed_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_relaxed_best_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_narrow_detection_filter_20260619.json`.

Results:

| branch | pair F1 | precision | recall | full/filter IDF1 |
| --- | ---: | ---: | ---: | ---: |
| base assignment | `0.768743` | `0.813273` | `0.728836` | `0.654739` promoted q03 |
| provisional relink | `0.767775` | `0.817525` | `0.723733` | `0.652265` full |
| soft-overlap narrow | `0.768875` | `0.813287` | `0.729062` | `0.652880` hard_q03 |
| soft-overlap relaxed best | `0.768878` | `0.813293` | `0.729063` | not full-scored |

Interpretation:

- Provisional relink is a good adversarial failure case: it improves precision
  but cuts too much true pair mass from large components.
- Soft cannot-link is a genuine positive model-side move.  The best rule
  cached `174,788` overlap pairs and softened `1,978` duplicate-like pairs.
- The e2e gap remains.  Soft overlap should be a building block for future
  retrieval/merge, not the final resolver.

Protocol decision:

- Promote soft-overlap only as an internal evidence-calibration primitive.
- Do not promote its forced delivery assignment because hard_q03 full IDF1 is
  `0.652880`, below the current promoted `0.654739`.
- Next self-play branch should target the oracle repair decomposition directly:
  identify high-mass false-split identities/components with no-GT evidence and
  use soft-overlap constraints during candidate graph construction.

## 2026-06-19 Loop Update: Deli SKILL Distillation and Weak-Positive Refutation

Deli AutoResearch distilled for this turn:

- Long-horizon agent research should run from durable files, not conversation
  memory.  This report, the model card, and the JSON artifacts are the state.
- "Ready means execute": once a script, validation, and remote command exist,
  the experiment is submitted without asking for another go/no-go.
- Honest downward score movement is part of the loop.  A branch that looks
  conceptually attractive but loses the quantitative gate should be recorded as
  a refutation, not silently dropped.
- The self-play analogy for VLINCS is not RL weight training.  It is a research
  game:
  proposer = weak evidence source;
  opponent = cannot-link/conflict/per-video failure evidence;
  evaluator = pair/full metrics and case provenance.

Executed branch:

- proposer: soft-overlap duplicate-like pairs as no-anchor positive labels;
- opponent: visually similar same-stream overlaps with low bbox IoU as hard
  negatives;
- evaluator: current assignment pair metrics;
- script: `kit/no_anchor_soft_overlap_weak_positive_verifier.py`.

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_weak_positive_verifier_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_weak_positive_verifier_maxprob_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_weak_positive_verifier_logreg_pair_20260619.json`;
- local copies under
  `local_runs/remote_h100_test_3_20260619/no_anchor_soft_overlap_weak_positive_verifier*_pair_20260619.json`.

Outcome:

| branch | pair F1 | precision | recall | action |
| --- | ---: | ---: | ---: | --- |
| base | `0.768743` | `0.813273` | `0.728836` | current assignment |
| direct soft-overlap | `0.768878` | `0.813293` | `0.729063` | keep as model-side best |
| HGB top-mean verifier | `0.768873` | `0.813293` | `0.729054` | no full scoring |
| HGB max-prob verifier | `0.768812` | `0.813083` | `0.729114` | no full scoring |
| logreg top-mean verifier | `0.768867` | `0.813292` | `0.729044` | no full scoring |

Refutation:

- Duplicate-overlap is a good positive source but too narrow; it improves over
  base but cannot beat the direct soft-overlap rule.
- Single strongest crop-pair evidence is unsafe: max-prob increases recall but
  damages precision.
- Do not keep sweeping verifier model classes on this same weak-label source.

Next structural direction:

- build a high-mass false-split repair branch;
- use soft-overlap as a constraint/evidence feature;
- choose candidate components by no-GT conflict, margin, component state,
  tracklet length, and cross-view support;
- require a cheap pair gate above `0.768878` before any full scorer run.

## 2026-06-19 Loop Update: Visual-Edge Delivery Filter Opponent

Opponent check:

- The visual-contact-sheet branch had a better pair metric, but weak full
  score.  The next opponent was delivery filtering: if q-threshold filtering
  fixed it, the problem would be row quality rather than identity structure.

Artifact:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_edge_no_forbidden_detection_filter_20260619.json`.

Result:

| config | IDF1 | DetPr | DetRe |
| --- | ---: | ---: | ---: |
| base | `0.652830` | `0.755991` | `0.574442` |
| q02 | `0.652144` | `0.767645` | `0.566854` |
| q03 | `0.649590` | `0.777206` | `0.557972` |
| q04 | `0.644982` | `0.787198` | `0.546288` |
| q05 | `0.637395` | `0.798600` | `0.530341` |

Protocol update:

- Mark detection-threshold repair as negative for the visual-edge branch.
- Future loops should not spend full-scorer time on plain q-filter sweeps unless
  a new assignment source changes the delivered row distribution.
- The next proposer must change identity structure directly: split impure
  high-mass components, then merge verified false-split islands.

Cross-assignment opponent:

- Reapplied the same no-GT visual decisions to the current promoted assignment
  instead of the verifier-split assignment.
- Pair result:
  `0.768833 / 0.813275 / 0.728996`, with `16` accepted edges.
- Verdict: visual decisions are not portable as a standalone merge patch.  They
  need the split-state context that produced the stronger `0.768927` diagnostic.

## 2026-06-19 Loop Update: Split-State Proposers vs Full-Score Opponent

This loop followed the Deli-style proposer/opponent/evaluator discipline:

- proposer A: visual seed subclusters from contact-sheet decisions;
- proposer B: relaxed soft-overlap assignment, sent to full delivery check;
- proposer C: temporal cannot-link NMS-singleton split;
- opponent: full DS1 IDF1/HOTA with the same base/hard_q03/global_q03 delivery
  configs;
- evaluator rule: pair gains are necessary but not sufficient for promotion.

Artifacts:

- visual seed script:
  `kit/no_anchor_visual_seed_subcluster_merge.py`;
- NMS-singleton script:
  `kit/no_anchor_assignment_cannotlink_nms_singleton_sweep.py`;
- visual-seed JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_visual_seed_subcluster_currentbase_pair_20260619.json`;
- soft-overlap relaxed full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_soft_overlap_merge_relaxed_detection_filter_20260619.json`;
- NMS-singleton pair/full JSONs:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_cannotlink_nms_singleton_relaxed_pair_20260619.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_cannotlink_nms_singleton_relaxed_detection_filter_20260619.json`.

Results:

| branch | pair F1 | precision | recall | best full IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| current production | `0.768743` | `0.813273` | `0.728836` | `0.654739` | still production best |
| visual seed subcluster | `0.765444` | `0.812450` | `0.723580` | not scored | negative pair gate |
| soft-overlap relaxed | `0.768878` | `0.813293` | `0.729063` | `0.652884` | pair-positive, full-negative |
| NMS-singleton relaxed | `0.768930` | `0.813789` | `0.728759` | `0.652845` | model-side diagnostic only |

Protocol update:

- Promote `NMS-singleton` as the latest model-side diagnostic best, but not as
  a production artifact.
- Do not spend more full-scorer time on branches that only alter a tiny number
  of tracklets unless they also change detection-weighted IDF1/HOTA risk.
- The next proposer should target the high-mass oracle false-merge components
  directly.  The evidence should be no-GT and component-scale: visual mode
  separability, cannot-link conflict density, per-stream occupancy, and
  cross-view agreement.  Single-edge or few-loser repairs are now a known local
  optimum.

## 2026-06-19 Loop Update: Softcut Split Evidence Game

AutoResearch distillation applied:

- The Deli framework's useful rule for this project is "pivot structure, not
  tactics": after several few-edge repairs failed to move full DS1, the next
  branch had to change the component structure itself.
- The self-play translation is now explicit:
  proposer = component-scale no-GT split policy;
  opponent = full DS1 promoted-filter scorer;
  evaluator = pair metrics plus IDF1/HOTA, with GT only after prediction.
- A score decrease after the stronger opponent is a valid finding.  It marks
  the branch as diagnostic rather than production, even if pair F1 improves.

Executed proposer:

- script: `kit/no_anchor_assignment_softcut_split_sweep.py`;
- base: current promoted no-anchor assignment;
- evidence: OSNet s7, DB-view, pose/color, color histogram, same-stream
  temporal conflict density;
- no anchors and no GT for split selection;
- GT used only after prediction for metrics.

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_one_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_two_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_relaxed_promoted_filters_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_pair_20260619.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_promoted_q03_20260619.json`.

Results:

| branch | pair F1 | precision | recall | full q03 IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| current production | `0.768743` | `0.813273` | `0.728836` | `0.654739` | production best |
| softcut, 1 component | `0.769135` | `0.814237` | `0.728768` | `0.652924` | rejected |
| softcut, 2 components | `0.769171` | `0.814853` | `0.728340` | `0.653050` | rejected |
| softcut relaxed | `0.769668` | `0.815981` | `0.728329` | `0.653205` | diagnostic best |
| softcut then soft-overlap | `0.769661` | `0.815773` | `0.728482` | `0.653225` | rejected |

Per-video filter opponent:

- unfiltered softcut+soft-overlap base:
  IDF1/HOTA `0.653071 / 0.516882`;
- fixed promoted q03 on the same assignment:
  IDF1/HOTA `0.653225 / 0.516860`;
- GT-selected per-video filter oracle:
  IDF1/HOTA `0.655240 / 0.518652`.
- no-GT row-density/confidence selector:
  IDF1/HOTA `0.655240 / 0.518652`.

Oracle filter choices:

- MCAM03 Tc6: `conf_q=0.01`, `conf_thr=0.115188`;
- MCAM04 Tc6: `conf_q=0.03`, `conf_thr=0.136421`;
- MCAM06 Tc8: `conf_q=0.02`, `conf_thr=0.145700`;
- MCAM08 Tc6: `conf_q=0.03`, `conf_thr=0.175449`;
- all other videos: no confidence filtering.

Interpretation:

- This is the strongest no-anchor global-ID model-side result so far, but it
  still fails the end-to-end gate by a wide margin.
- The per-video oracle showed a small full-score opportunity above the previous
  production artifact, and the no-GT `density_oracle_lite` selector recovered
  that lift from row density and confidence quantiles only.
- The split policy is too pair-metric aligned: it improves component purity and
  precision, but loses detection-weighted association mass under the submission
  scorer.
- The new verified full-score best is `0.655240`; this is progress, but the
  remaining gap to 0.70 is identity association, especially weak videos such as
  MCAM04 Tc6 and MCAM03 Tc8, not simple delivery filtering.

Next executable research direction:

1. Build a training table from all attempted softcut candidate components.
   Features: split size, min/max part size, conflict reduction, visual margin,
   within/cross similarity, camera/time occupancy entropy, avg confidence,
   bbox-area distribution, and whether soft-overlap can reconnect after split.
2. Label candidates post-hoc with cheap proxy deltas:
   pair delta, promoted-filter IDF1 delta on a bounded sample, and per-video
   detection-retention deltas.  These labels are diagnostics only, not anchors.
3. Stop treating per-video confidence filtering as the main bottleneck.  The
   no-GT density selector already recovers the small oracle lift.
4. Train a no-GT deployment acceptor that sees only candidate/video features
   and targets high-mass identity association repairs, not only row filtering.
5. Gate the acceptor in this order:
   pair F1 must exceed `0.769760`;
   full IDF1 must exceed `0.655240`;
   per-video MCAM04 Tc6 and MCAM03 Tc8 IDF1 must improve without damaging
   MCAM00/05/08;
   only then spend more time on q05/per-video filter sweeps.

## 2026-06-19 Loop Update: Error-First Relink Refutation

Fresh Deli AutoResearch distillation:

- Treat the framework as a research operating system: durable state, no
  waiting after preparation, quantitative stall checks, and structural pivots
  after repeated local failures.
- For this VLINCS loop, "self-play" means assignment proposers compete against
  a fixed evaluator.  A negative ablation is a valid finding when it rules out
  a tempting evidence source.
- The current structural pivot is from delivery filtering to identity
  association.  The density selector already recovered the small no-GT
  filtering lift; more filter sweeps are now tactical overfitting.

Diagnostic state:

- current deployable no-anchor full best:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.zip`;
- full IDF1/HOTA: `0.655240 / 0.518652`;
- current softcut+soft-overlap assignment pair metrics:
  `0.769661 / 0.815773 / 0.728482`;
- weakest videos:
  MCAM04 Tc6 full IDF1 `0.558658`,
  MCAM06 Tc6 `0.606895`,
  MCAM03 Tc8 `0.626660`;
- largest remaining error mode is mixed: top false split GT `9` has mass
  `1046536325`, while top false merge component `70000074` has mass
  `790173339`.

Relink branch result:

- deployable multiview relink with temporal forbidden on accepted only `1`
  edge and did not move pair F1 beyond `0.769661`;
- diagnostic relink with forbidden disabled accepted `10` edges and increased
  recall to `0.742027`, but precision collapsed to `0.778711` and F1 fell to
  `0.759926`;
- conclusion: hard cannot-link is not the only blocker.  The top multiview
  candidate edges are not calibrated enough; disabling safety merely exposes
  false visual-neighbor merges.

Updated next-step rule:

1. Do not run broader blind multiview merge grids from the current assignment.
   The narrow grid already shows no deployable signal, and the broad grid is
   too slow for this evaluator.
2. Build an edge/candidate acceptor dataset, not another direct rule.  Each
   row should be a proposed component edge or softcut split with no-GT
   features: rank votes, view similarities, component sizes, component-size
   growth, camera/video occupancy overlap, soft-overlap status, temporal
   conflict counts, and confidence/area density features.
3. Use GT only to label diagnostic deltas after prediction: pair precision,
   pair recall, pair F1, and full-score delta on bounded candidate sets.
4. Deployable acceptor input must not include GT labels or GT IDs.  It may use
   the learned relationship between no-GT features and post-hoc deltas as an
   offline ablation, then be frozen and evaluated as a no-anchor rule.
5. The next accepted branch must beat both gates:
   pair F1 `> 0.769760` and full IDF1 `> 0.655240`.

Edge acceptor table generated for the next branch:

- artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_edge_acceptor_table_20260619.json`;
- candidate table:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_edge_acceptor_table_20260619.csv`;
- candidate edges: `3240`;
- post-hoc true edges: `47`;
- true edges in top-40 false-split targets: `47`;
- best no-oracle edge rule: `4 / 28` true edges, precision `0.142857`;
- target-oracle rule: `7 / 11` true edges, precision `0.636364`.

Implication:

- The next branch should not be a direct edge-threshold rule.
- It should be a two-stage acceptor:
  target-localization first, edge-selection second.
- Target-localization features should describe predicted components and GT-free
  evidence of repairability: false-split-like occupancy pattern, component
  fragmentation, video/camera spread, same-stream soft-overlap burden,
  candidate-edge density, and disagreement between view rankings.
- Edge-selection features should then be evaluated only inside predicted
  repairable targets.  This is the only path suggested by the table: the edge
  signal is real, but global thresholding is too impure.

## 2026-06-20 Loop Update: Target-Localized Repair Accepted As Diagnostic Only

Self-play proposer:

- Instead of a global edge-threshold merge, select repairable target components
  first from no-GT edge-density and multi-view rank features.
- Inside each selected target, attach only tiny fragments: `small_size<=2`,
  one edge per target, and at most `16` targets.
- Explicitly strip `gt_*` columns from the edge table before selection.

Evaluator:

- Pair gate: weighted tracklet-pair F1/P/R.
- Full gate: canonical DS1 HOTA/IDF1.
- Delivery gate: no-GT `density_oracle_lite` row-density/confidence selector.
- GT is used only after prediction for metrics.

Artifacts:

- script:
  `kit/no_anchor_edge_table_target_repair_sweep.py`;
- pair-only JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_micro_paironly_20260620.json`;
- top full JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_top1_full_20260620.json`;
- density-filter JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_density_filter_selector_zip_20260620.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_edge_target_repair_*_20260620.*`.

Result:

| branch | pair F1 | precision | recall | full/delivery IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| softcut+soft-overlap base | `0.769661` | `0.815773` | `0.728482` | `0.655240` best density-selected artifact | keep production |
| target-localized repair | `0.769721` | `0.815785` | `0.728581` | `0.653084` unfiltered | diagnostic only |
| target-localized repair + density selector | n/a | n/a | n/a | `0.653297` | rejected |

Interpretation:

- The two-stage proposer worked as a model-side move: it created the best
  no-anchor diagnostic pair F1 so far, but only by `+0.000060`.
- The full-score evaluator rejected it.  The same no-GT density selector that
  made softcut+soft-overlap deployable reaches only `0.653297` here, below the
  current verified best `0.655240`.
- This is an informative failed self-play iteration: target localization is
  safer than global relink, but attaching singleton/tiny fragments is too small
  to change detection-weighted identity association.

Updated rule:

1. Keep target localization as a useful proposer primitive.
2. Stop spending full-score time on one-edge-per-target tiny-fragment repairs.
3. Next proposer should operate on multi-tracklet identity islands and optimize
   a no-GT proxy for full-score deltas: per-video row density, confidence
   retention, component size/weight growth, cross-view agreement, and
   cannot-link burden after edit.
4. The promotion gates are now:
   pair F1 must exceed `0.769760` and no-GT full IDF1 must exceed `0.655240`.

## 2026-06-20 Loop Update: Multi-Edge Repair Still Fails Full Gate

Proposer update:

- Keep target localization.
- Allow all localized targets (`max_targets=0`).
- Allow up to two edges per target.
- Test the same rule with and without `require_forbidden`.

Artifacts:

- pair-only:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_island_paironly_20260620.json`;
- no-forbidden pair-only:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_island_noforbidden_paironly_20260620.json`;
- full:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_island_top1_full_20260620.json`;
- density-filter:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_island_density_filter_selector_zip_20260620.json`;
- current oracle refresh:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_oracle_repair_decomposition_full_top1_20260620.json`.

Result:

| branch | pair F1 | precision | recall | full/delivery IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| one-edge target repair | `0.769721` | `0.815785` | `0.728581` | `0.653297` density selector | former diagnostic |
| multi-edge target repair | `0.769760` | `0.815788` | `0.728648` | `0.653097` unfiltered | diagnostic only |
| multi-edge target repair + density selector | n/a | n/a | n/a | `0.653311` | rejected |
| no-forbidden pair diagnostic | `0.769760` | `0.815788` | `0.728648` | not full-scored | no extra gain |

Current oracle refresh:

- all GT-majority current tracklets under the current scorer: full IDF1
  `0.706202`;
- top-40 split+merge oracle: full IDF1 `0.706008`;
- current deployable best remains `0.655240`.

Interpretation:

- The self-play loop has now found the boundary of target-edge repair: it can
  make small model-side gains, but it does not move delivery-weighted IDF1.
- The current oracle is barely above 0.70, so reaching the goal requires
  recovering most of the oracle association headroom, not incremental pair-F1
  improvements.
- The next proposer should be a large-identity recovery model: first find
  split identity islands like GT `9`, `36`, `11`, and `43` using no-GT features,
  then perform coordinated split+merge edits rather than local edge attachments.

## 2026-06-20 Loop Update: AutoResearch Distillation, Island Gate, and DINO Base Feature Check

Fresh Deli AutoResearch distillation:

- Source pages inspected:
  `https://victorchen96.github.io/auto_research/framework.html`,
  `https://victorchen96.github.io/auto_research/paper.html`, and
  `https://victorchen96.github.io/blog_self_play_story.html`.
- The useful transfer is procedural, not a new model class:
  persist state to files, execute once ready, separate proposer/opponent/
  evaluator roles, and treat score drops as real evidence.
- The Self-play analogy maps cleanly to VLINCS:
  proposer = no-GT identity edit or feature source;
  opponent = temporal/cannot-link/conflict/density checks;
  evaluator = pair/full metrics and provenance.
- Anti-loop rule for this continuation:
  after target-edge repair and soft-overlap both produced only tiny pair gains,
  run either a genuinely different structural proposer or a genuinely new
  feature source.  Do not broaden old grids.

Executed proposer A: edge-table identity-island merge.

- New script:
  `kit/no_anchor_edge_table_island_merge_sweep.py`.
- It consumes the current softcut+soft-overlap assignment and the existing
  no-GT edge acceptor table, strips all `gt_*` columns before selection, and
  forms small component-island merges from high fused multi-view agreement.
- Remote artifacts:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_table_island_merge_focused_pair_20260620.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_table_island_merge_focused_pair_20260620.csv`.
- Local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_edge_table_island_merge_focused_pair_20260620.*`.

Executed proposer B: DINOv2-base as a new no-anchor feature source.

- Full feature artifact:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_dinov2base_s1_20260620.npz`.
- Extraction details:
  model `facebook/dinov2-base`, `samples=1`, `9734` tracklets, feature dim
  `768`, `missing_video_rows=0`, `missing_crop_rows=0`.
- Fused feature artifacts:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_dinobase003_s1_20260620.npz`,
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_dinobase005_s1_20260620.npz`.
- Pair-gate artifacts:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinobase_s1_component_merge_pair_20260620.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_dinobase003_s1_component_merge_pair_20260620.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_dinobase005_s1_component_merge_pair_20260620.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_dinobase005_s1_component_merge_loose_pair_20260620.json`.

Results:

| branch | pair F1 | precision | recall | action | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| softcut+soft-overlap base | `0.769661` | `0.815773` | `0.728482` | none | baseline |
| focused edge-table island merge | `0.769698` | `0.815763` | `0.728558` | `8` accepted edges | below gate |
| DINOv2-base s1 component merge | `0.769661` | `0.815773` | `0.728482` | `0` accepted merges | no-op |
| fused + DINO 0.03 component merge | `0.769661` | `0.815773` | `0.728482` | `0` accepted merges | no-op |
| fused + DINO 0.05 component merge | `0.769661` | `0.815773` | `0.728482` | `0` accepted merges | no-op |
| fused + DINO 0.05 loose sanity | `0.769661` | `0.815773` | `0.728482` | best `1` accepted merge | no useful signal |

Interpretation:

- Focused high-fused island merging is directionally positive but does not beat
  the current diagnostic gate `0.769760`, so it was not full-scored.
- DINOv2-base is a valid extracted feature source, but current component-merge
  mechanics either reject it or produce no weighted pair gain.
- The negative result strengthens the diagnosis: the bottleneck is not another
  global visual embedding or a few more small-fragment edges.  The next
  proposer must predict large-component repairability and split/merge state at
  the identity-island level before edge scoring.

Next structural rule:

- Use the current edge table and error audit as training/evaluation material for
  a no-GT repairability proxy: component conflict density, visual mode
  separability, row density, confidence/area statistics, and cross-view
  disagreement.
- The proxy must emit `committed`, `provisional`, and `forced` states.  Hard
  relabeling of small islands is now a known local optimum.

## 2026-06-20 Continued Loop: Temporal Relink and Conflict-State Repair

Executed proposer C: video-local temporal evidence with global-component output.

- New script:
  `kit/no_anchor_assignment_video_temporal_relink_sweep.py`.
- It builds same-video candidate edges from time gap, bbox endpoint distance,
  and fused appearance similarity, then applies accepted edges only as
  global-component merges with overlap guards.
- Artifacts:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_video_temporal_relink_global_fused_s7_pair_20260620.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_video_temporal_relink_global_fused_s7_loose_pair_20260620.json`.

Temporal result:

| branch | pair F1 | precision | recall | accepted edges | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| strict temporal relink | `0.769661` | `0.815773` | `0.728482` | `0` | no-op |
| loose temporal relink | `0.769661` | `0.815773` | `0.728482` | `1` | no useful gain |

Important correction:

- A diagnostic video-local output scope scored only full IDF1 `0.366217`.
- Therefore full scoring is not independent per video in the way a local MOT
  switcher would assume; deployable output must preserve global component
  continuity.

Executed proposer D: conflict-state repair on the current softcut assignment.

- Pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_pair_20260620.json`.
- Singleton forced-conflict split:
  pair F1/P/R `0.771670 / 0.822651 / 0.726639`.
- Full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_top1_full_20260620.json`.
- Full IDF1/HOTA/AssA/DetPr/DetRe:
  `0.646655 / 0.513889 / 0.534396 / 0.744532 / 0.571522`.

Executed proposer E: color forced-conflict split plus multiview remerge.

- Color split pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_color_forced_pair_20260620.json`.
- Color split full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_current_state_policy_color_forced_full_20260620.json`.
- Split+merge pair artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_state_color_forced_multiview_merge_pair_20260620.json`.
- Split+merge full artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_state_color_forced_multiview_merge_top1_full_20260620.json`.

Results:

| branch | pair F1 | precision | recall | full IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| color forced-conflict split | `0.770599` | `0.818962` | `0.727630` | `0.652774` | e2e negative |
| color split + multiview remerge | `0.770667` | `0.818533` | `0.728090` | `0.652744` | e2e negative |

Interpretation:

- Conflict-state repair is the new model-side best, but it is not deployable:
  pair F1 increased while full IDF1 fell.
- Forced conflict splitting improves pair precision by breaking false merges,
  but the full scorer penalizes the resulting detection-level identity
  fragmentation.
- The next self-play proposer should not deliver split components directly.
  It should keep conflict subgraphs as provisional evidence, then solve a
  local identity-component assignment before emitting forced IDs.

## Deli AutoResearch Distillation Applied - 2026-06-20

The Deli AutoResearch/self-play thread was distilled into a concrete operating
rule for this VLINCS run:

- every proposer must write durable artifacts and a falsifiable gate;
- the opponent is the known failure evidence: same-stream cannot-link,
  component conflict, density, and full-score fragmentation;
- when an edit improves pair score but hurts full score, change the output
  structure instead of sweeping the same thresholds.

Applied pivot:

- prior branch: `forced_conflict` split directly emitted singleton/color IDs;
- observed failure: pair precision improved, but full IDF1 dropped because the
  submission became fragmented;
- new branch: keep conflict subclusters as provisional evidence, then reassign
  them only into an existing target component.  No unresolved split is emitted
  as a new delivery ID.

New script:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py`.

No-anchor contract:

- GT is not used to build states, candidates, targets, or filters;
- source candidates come from conflicted components and multi-view visual
  separability;
- target candidates must be existing components with sufficient fused/pose/color
  evidence;
- strict mode enforces component-level same-stream cannot-link clean targets;
- GT is used only after prediction for pair/full evaluation.

Artifacts:

- pair gate:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_pair_20260620.json`;
- full scorer:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_top1_full_20260620.json`;
- assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_top1_assignments_20260620.csv`;
- density-selected delivery:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_strict_density_filter_selector_zip_20260620.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_*_20260620.*`.

Results:

| branch | pair F1 | precision | recall | full IDF1 | density IDF1 | action |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| production softcut+soft-overlap | `0.769661` | `0.815773` | `0.728482` | n/a | `0.655240` | standing best |
| state singleton forced-conflict | `0.771670` | `0.822651` | `0.726639` | `0.646655` | n/a | rejected |
| non-strict conflict reassign | `0.772399` | `0.818671` | `0.731077` | `0.653724` | `0.653936` | e2e-negative |
| margin 0.03 reassign | `0.770554` | `0.816238` | `0.729712` | `0.652957` | n/a | worse |
| strict conflict reassign | `0.772654` | `0.818667` | `0.731538` | `0.653823` | `0.654037` | new model-side best |

Strict accepted edit:

- accepted reassignments: `1`;
- moved tracklets: `8`;
- source component: `21`;
- target component: `0`;
- source seqs:
  `2232, 2270, 2308, 2374, 2415, 2452, 2488, 2553`;
- target evidence seqs:
  `2656, 2591, 5838`;
- target mean similarity: `0.940898`;
- target best similarity: `0.947318`;
- target view vote: `1.0`;
- target component-level forbidden pairs: `0`.

Decision:

- Promote `strict_conflict_subcluster_reassign` as the latest no-anchor
  global-ID model-side result: pair F1 `0.772654`.
- Do not promote it as the end-to-end artifact: density-selected IDF1
  `0.654037` remains below `0.655240`.
- The next self-play proposer should learn a full-score proxy for edit
  admission.  Pair F1 now rewards the right identity move, but the delivery
  layer still underweights or misprices detection-level side effects.

## 2026-06-20 Continued Loop: Source-Island Audit And G8 Reassign

Fresh Deli AutoResearch distillation applied in this iteration:

- Do not keep widening the same online threshold grid after a stall.
- Materialize state into files, audit the generator offline, then make one
  structural pivot.
- Treat a score drop as evidence.  The loose source generator found more
  oracle-positive islands, but its no-GT ranking over-selected bad 12-tracklet
  islands; that is a source-repairability proxy failure, not just a target-gate
  failure.

New source audit script:

- `kit/no_anchor_conflict_source_island_audit.py`.

No-anchor contract:

- source islands are built from assignment state, conflict evidence, and visual
  views only;
- GT is used only after a candidate source exists, to compute diagnostic oracle
  deltas and source-rank hit rates;
- no anchor tracklets or GT identity seeds are used.

Source-audit artifacts:

- tiny audit:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_source_island_audit_tiny_20260620.json`;
- loose audit:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_source_island_audit_loose1_20260620.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_source_island_audit_*_20260620.*`.

Source-audit result:

| audit | dedup sources | oracle-positive sources | top no-GT source positive? | best oracle delta | lesson |
| --- | ---: | ---: | --- | ---: | --- |
| tiny strict source | `17` | `3` | yes | `+0.002993` pair F1 | reproduces strict 8-tracklet island |
| loose source | `44` | `9` | no | `+0.002409` pair F1 | more candidates, worse no-GT ranking |

Candidate-search artifacts:

- loose target gate:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json`;
- loose source + strict target gate:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_stricttarget_pair_20260620.json`;
- loose source capped at 8 + strict target gate:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_pair_20260620.json`;
- full scorer:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json`;
- assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_top1_assignments_20260620.csv`;
- no-GT density-selected delivery:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_density_filter_selector_zip_20260620.json`.

Results:

| branch | pair F1 | precision | recall | accepted reassignments | moved tracklets | full/density IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strict conflict reassign | `0.772654` | `0.818667` | `0.731538` | `1` | `8` | `0.654037` density | former model best |
| loose source, loose target | `0.767329` | `0.814359` | `0.725434` | `1` | `12` | not scored | bad source selected |
| loose source, strict target | `0.772127` | `0.817025` | `0.731906` | `2` | `24` | not scored | recovers but below strict best |
| loose source, max group 8, strict target | `0.775234` | `0.820504` | `0.734698` | `4` | `32` | `0.653541` full, `0.653681` density | new model-side best, e2e negative |

Accepted source islands in the new model-side best:

- `2232, 2270, 2308, 2374, 2415, 2452, 2488, 2553`;
- `4406, 4461, 4513, 4557, 4619, 4658, 4744, 4790`;
- `633, 764, 977, 1006, 1044, 1078, 1102, 1120`;
- `8305, 8540, 8914, 8972, 9013, 9050, 9130, 9247`.

Decision:

- Promote `loose_source_island_g8_strict_target_reassign` as the latest
  no-anchor global-ID model-side diagnostic result: pair F1 `0.775234`.
- Do not promote it as the end-to-end artifact.  Unfiltered full IDF1
  `0.653541` and density-selected IDF1 `0.653681` remain below the production
  best `0.655240`.
- The new structural lesson is sharp: source-island capping repairs the model
  pair graph, but the delivery layer still prices fragmented/shifted IDs
  differently.  The next proposer should be delivery-aware admission or
  component-state scoring, not a wider source-island grid.

## 2026-06-20 Continued Loop: Delivery-Aware Admission Checks

Branch:

- Test whether the G8 source-island edits fail e2e because of a small number of
  bad edits or because simple detection-row filters are mismatched.
- Add a no-GT target-quality gate to
  `kit/no_anchor_assignment_conflict_reassign_sweep.py`:
  `--min-target-qualities`, default `0.0`, so old behavior is unchanged.
- GT is still used only after prediction for pair/full scoring.

Artifacts:

- prefix full sensitivity:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json`;
- target-quality gate:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json`;
- target-quality assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_top_assignments_20260620.csv`;
- row-filter policy sweep:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_filter_policy_sweep_20260620.json`.

Prefix/full sensitivity:

| admitted edits | pair F1 | precision | recall | full IDF1 | note |
| ---: | ---: | ---: | ---: | ---: | --- |
| `1` | `0.772654` | `0.818667` | `0.731538` | `0.653823` | same strict island |
| `2` | `0.774082` | `0.819399` | `0.733515` | `0.653823` | pair gain, no full gain |
| `3` | `0.774252` | `0.819890` | `0.733427` | `0.653042` | third edit hurts full |
| `4` | `0.775234` | `0.820504` | `0.734698` | `0.653541` | model-side best, e2e negative |

No-GT target-quality gate:

| branch | target-quality gate | pair F1 | precision | recall | full IDF1 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| quality-gated top 3 | `0.75` | `0.775064` | `0.820014` | `0.734786` | `0.653541` | skips low-quality target, no e2e gain |
| quality-gated top 4 | `0.75` | `0.773540` | `0.818424` | `0.733324` | `0.654009` | best in this branch, still below `0.655240` |

Fixed no-GT row-filter policies on the G8 assignment:

| policy | IDF1 | HOTA | AssA | DetPr | DetRe | dropped rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `density_oracle_lite` | `0.653681` | `0.517358` | `0.532902` | `0.760029` | `0.573442` | `34,434` |
| `density_simple` | `0.653686` | `0.517364` | `0.532902` | `0.759920` | `0.573512` | `33,685` |
| `confidence_tail` | `0.653581` | `0.517217` | `0.532757` | `0.760258` | `0.573158` | `36,385` |

Decision:

- No delivery-aware admission/filter candidate beat the standing e2e best
  `0.655240`.
- The failure is not just a bad third edit or an overly weak row-density
  filter.  Full IDF1 is sensitive to which videos/identity components receive
  edits, so the next branch should predict full-score side effects at the
  component/video level before changing IDs.
- Keep the G8 branch as the model-side best only.  For e2e, pivot to a
  component/video admission model or a larger identity-component resolver that
  optimizes delivery structure directly.

## 2026-06-20 Source-Switch Compatibility Diagnostic

Fresh AutoResearch distillation:

- The Deli AutoResearch framework is best used here as an operating protocol:
  persist state to files, run bounded experiments when a hypothesis is ready,
  and preserve negative score movement as evidence.
- The Self-play paper/story maps cleanly onto VLINCS:
  proposer = no-anchor ID edit or source;
  opponent = cannot-link, namespace consistency, and density/admission checks;
  evaluator = pair/full scorer, with GT used only after prediction.
- The anti-loop rule for this continuation: after target-quality gates and row
  filters failed, test source complementarity once, then stop if namespace
  consistency breaks.

Artifacts:

- explicit source-switch JSON:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.json`;
- explicit source-switch zip:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.zip`;
- local JSON:
  `local_runs/remote_h100_test_3_20260620/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.json`.

Setup:

- candidate source pool was restricted to existing no-anchor submission zips;
- per-video scan over existing full JSONs suggested only two source changes:
  MCAM04 Tc6 from `conflict_g8`, MCAM06 Tc8 from `quality`;
- no anchors were used; the policy itself is a diagnostic, not a deployable
  no-GT selector, because the per-video choices came from scorer feedback.

Result:

| policy | IDF1 | HOTA | AssA | DetPr | DetRe | predicted IDs | rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current/base | `0.655240` | `0.518652` | `0.534359` | `0.763322` | `0.573970` | `377` | `1,524,919` |
| explicit switch | `0.497171` | `0.345933` | `0.355785` | `0.801372` | `0.360373` | `379` | `1,525,668` |

Per-video symptoms:

- switched MCAM04 Tc6 collapsed to IDF1 `0.228783`;
- switched MCAM06 Tc8 collapsed to IDF1 `0.000000`;
- even videos left on `current` moved downward, meaning the mixed source
  namespace broke global identity consistency rather than acting as a local
  per-video replacement.

Decision:

- Do not build a video-level source selector over full submission zips.
- Whole-video source switching is not an identity resolver; it violates the
  global-ID namespace unless the sources share calibrated ID semantics.
- Keep `current` as the standing e2e artifact at IDF1 `0.655240`.
- Next proposer must repair identity components in one namespace, not stitch
  independent submission namespaces together.

## 2026-06-20 Aggressive Island Pair Gate

Follow-up after source switching:

- stay inside one namespace;
- expand edge-table island repair beyond tiny fragments;
- stop at pair gate unless it beats the previous edge-target repair.

Artifact:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_table_island_merge_aggressive_pair_20260620.json`.

Result:

| branch | pair F1 | precision | recall | accepted edges | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| current/base | `0.769661` | `0.815773` | `0.728482` | n/a | input |
| edge-target repair | `0.769760` | `0.815788` | `0.728648` | `23` | stronger |
| aggressive island | `0.769692` | `0.815171` | `0.729019` | `17` | rejected |

Decision:

- Bigger small-island attachments are recall-positive but precision-negative.
- Do not full-score this branch.
- The next branch needs a repairability model for larger identity components
  or a stronger pseudo-positive source; "attach more islands" is now a closed
  local direction.

### 2026-06-20 DINO edge-source target-repair challenger

Distillation update:

- The Deli AutoResearch lesson from
  `https://victorchen96.github.io/auto_research/framework.html` is now encoded
  as file-backed state under
  `autoresearch_state/no_anchor_global_id/state/`.
- The self-play translation is: propose a feature-source challenger, evaluate
  it with the same no-GT repair gate, and close it if the opponent metric
  shows precision or delivery damage.

State files added:

- `autoresearch_state/no_anchor_global_id/state/task_spec.md`;
- `autoresearch_state/no_anchor_global_id/state/progress.json`;
- `autoresearch_state/no_anchor_global_id/state/directions_tried.json`;
- `autoresearch_state/no_anchor_global_id/state/findings.jsonl`.

Implementation fix:

- `kit/no_anchor_edge_table_target_repair_sweep.py` now falls back from
  `fused_sim/fused_rank_max/db_rank_min` to `primary_sim/primary_rank_*` when
  an edge table was generated from a single feature source.
- This fixed a silent no-op where the fused-DINO edge table had only
  `primary_sim` and was filtered to `eligible_edges=0`.

Artifacts:

- edge-table diagnostic:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_acceptor_table_20260620.json`;
- edge table:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_acceptor_table_20260620.csv`;
- initial schema-mismatch no-op:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_pair_20260620.json`;
- corrected single target-repair run:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_fallback_single_full1_20260620.json`;
- corrected assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_fallback_single_assignments_20260620.csv`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_*_20260620.*`.

Edge-source diagnostic:

| diagnostic | value |
| --- | ---: |
| candidate edges | `3240` |
| eval-only true edges | `47` |
| eval-only true top-false-split edges | `47` |
| best no-oracle rule | `4 / 23` true edges, precision `0.173913` |
| best target-oracle rule | `1 / 1` true edge, precision `1.000000` |

Corrected single repair result:

| branch | pair F1 | precision | recall | full IDF1 | HOTA | AssA | selected edges | localized targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| softcut+soft-overlap base | `0.769661` | `0.815773` | `0.728482` | `0.655240` standing zip | `0.518652` | `0.534359` | n/a | n/a |
| fused-DINO target repair | `0.769661` | `0.815126` | `0.729001` | `0.652948` | `0.516770` | `0.532397` | `15` | `8` |

Per-video IDF1 for fused-DINO target repair:

| video | IDF1 |
| --- | ---: |
| MCAM00 Tc6 | `0.878682` |
| MCAM00 Tc8 | `0.827908` |
| MCAM03 Tc6 | `0.688387` |
| MCAM03 Tc8 | `0.625107` |
| MCAM04 Tc6 | `0.558659` |
| MCAM05 Tc6 | `0.710965` |
| MCAM05 Tc8 | `0.791599` |
| MCAM06 Tc6 | `0.606972` |
| MCAM06 Tc8 | `0.704160` |
| MCAM08 Tc6 | `0.767157` |

Conclusion:

- Fused-DINO as an edge-source challenger is closed in the current target-repair
  form.  It found candidates but traded precision for recall and hurt full
  delivery.
- The wide fallback grid was interrupted because repeated all-pairs pair-metric
  recomputation was too slow even for small grids.  This is an engineering
  bottleneck, not a model win.
- Next branch should implement cached/incremental edit scoring or a
  repairability proxy before any further target-repair search.

### 2026-06-20 Cached target-repair and Pose2ID/NFC branch

Deli-style loop rule applied:

- after a branch was blocked by engineering cost, fix the bottleneck and rerun
  a bounded experiment once;
- after a feature-source idea is negative in multiple insertion points, close
  it instead of re-tuning the same family.

Implementation:

- `kit/no_anchor_edge_table_target_repair_sweep.py` now caches accepted-edit
  signatures so repeated grid rows do not recompute identical pair metrics.
- Added `kit/make_no_anchor_nfc_features.py`, a no-anchor Neighbor Feature
  Centralization utility inspired by the training-free NFC idea in Pose2ID.
  It reads only feature-space nearest neighbors, writes a replacement `.npz`,
  and records `nfc_info`; it uses no GT and no anchors.

Artifacts:

- cached fused-DINO target-repair grid:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_dinofused_edge_target_repair_cached_pair_20260620.json`;
- old edge-table signature scout:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_edge_target_repair_signature_scout_pair_20260620.json`;
- NFC time-agglom resolver:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_nfc_fused_osnet005_s7true_timeagglom_pair_20260620.json`;
- NFC softcut primary replacement:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_nfc_osnet_s7_pair_20260620.json`;
- NFC softcut auxiliary view:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_nfc_aux_osnet_s7_pair_20260620.json`.

Results:

| branch | pair F1 | precision | recall | verdict |
| --- | ---: | ---: | ---: | --- |
| current model-side best, G8 strict target | `0.775234` | `0.820504` | `0.734698` | standing model best |
| cached fused-DINO target repair | `0.769755` | `0.815757` | `0.728664` | closed |
| old edge-table signature scout | `0.769713` | `0.815146` | `0.729077` | closed |
| NFC time-agglom resolver | `0.594775` | `0.720286` | `0.506515` | closed |
| NFC softcut primary replacement | `0.764329` | `0.815670` | `0.719068` | closed |
| NFC softcut auxiliary view | `0.767759` | `0.814164` | `0.726358` | closed |

Interpretation:

- The cache fixed the runtime waste, but not the modeling issue: DINO target
  repair still selects recall-positive, precision-negative edits.
- NFC/Pose2ID-style smoothing over-smears crowded identity neighborhoods in
  this setting.  As a primary feature it causes large softcut splits
  (`763` tracklets); as an auxiliary view it is less harmful but still below
  the original softcut base.
- The next valid branch is a delivery-aware repairability proxy: predict
  whether an edit is safe before committing it, instead of accepting edges by
  local visual similarity alone.

### 2026-06-20 SigLIP2 person-ReID feature challenger

Why this branch:

- Prior direct weak-label verifiers and DINO/NFC feature-source challengers did
  not improve the delivery gate.
- `MarketaJu/siglip2-person-description-reid` is a person-description /
  re-identification SigLIP2 checkpoint, so it is a structurally different
  visual prior from OSNet, DINO, and generic CLIP.

Implementation:

- Patched `kit/extract_tracklet_foundation_features.py` to support
  `--processor-model`, needed because the model repo does not ship its own
  image processor.
- The extractor now handles `get_image_features()` outputs that return pooled
  model-output objects rather than raw tensors.

Artifacts:

- feature smoke:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_siglip2_person_reid_smoke32_20260620.npz`;
- full feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_siglip2_person_reid_s1_20260620.npz`;
- standalone pair gate:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_siglip2_person_reid_s1_timeagglom_pair_20260620.json`;
- low-weight fusion pair gates:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_siglip2p003_s1_component_merge_pair_20260620.json`,
  `/mnt/localssd/vlincs_reid_runs/no_anchor_fused_siglip2p005_s1_component_merge_pair_20260620.json`.

Result:

| branch | pair F1 | precision | recall | verdict |
| --- | ---: | ---: | ---: | --- |
| SigLIP2-person-ReID s1 only | `0.574380` | `0.640377` | `0.520716` | rejected |
| current base for component merge | `0.769661` | `0.815773` | `0.728482` | baseline |
| fused + SigLIP2 `0.03` | `0.769661` | `0.815773` | `0.728482` | no-op |
| fused + SigLIP2 `0.05` | `0.769661` | `0.815773` | `0.728482` | no-op |

Decision:

- Close this feature challenger in the current form.  It is valid no-anchor
  evidence, but it does not alter the current component-merge pair gate.
- Do not spend full-score time on it unless a future branch uses it as a
  provenance feature inside a learned delivery-aware acceptor.

### 2026-06-20 Deli AutoResearch thread distillation and weak-metric probe

External cue distilled:

- Deli AutoResearch is a protocol, not a reusable model.  The useful rules for
  this VLINCS loop are durable file state, ready-means-execute, evaluator-owned
  scores, and structural pivots after repeated stalls.
- The Self-Play story maps to our setting as proposer versus opponent:
  proposer = no-anchor identity edit / feature / component solver;
  opponent = cannot-link, namespace consistency, density, and full-score
  fragmentation;
  evaluator = pair/full scorer with GT used only after predictions.
- Since stale count is now high, new work must change structure, not tune
  another local threshold around G8 strict-target reassign.

New script:

- `kit/make_no_anchor_weak_metric_features.py`.

No-anchor weak supervision:

- positives:
  same-tracklet crop pairs plus short-gap same-stream visual continuations;
- negatives:
  same-stream temporal-overlap cannot-link hard negatives;
- no anchors or GT identities are read during training.

Training artifact:

- `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_weakmetric_osnet_s7_fused_20260620.json`;
- output features:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620.npz`,
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p05.npz`.

Training stats:

| quantity | value |
| --- | ---: |
| same-tracklet positives | `9000` |
| continuation positives | `8555` |
| cannot-link negatives | `18000` |
| train pairs | `35555` |
| projected positive cosine mean | `0.420363` |
| projected negative cosine mean | `0.096400` |
| projected train margin | `0.323963` |

Downstream G8 strict-target conflict-reassign probe:

| primary feature | pair F1 | precision | recall | accepted reassignments | moved tracklets | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current fused primary | `0.775234` | `0.820504` | `0.734698` | `4` | `32` | standing model best |
| weakmetric `w=0.02` | `0.775234` | `0.820504` | `0.734698` | `4` | `32` | no-op |
| weakmetric `w=0.05` | `0.775234` | `0.820504` | `0.734698` | `4` | `32` | no-op |

Artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_weakmetric_w002_conflict_reassign_g8_pair_20260620.json`;
- `/mnt/localssd/vlincs_reid_runs/no_anchor_weakmetric_w005_conflict_reassign_g8_pair_20260620.json`;
- local copies:
  `local_runs/remote_h100_test_3_20260620/ds1_tracklet_weakmetric_osnet_s7_fused_20260620.json`,
  `local_runs/remote_h100_test_3_20260620/no_anchor_weakmetric_w002_conflict_reassign_g8_pair_20260620.json`,
  `local_runs/remote_h100_test_3_20260620/no_anchor_weakmetric_w005_conflict_reassign_g8_pair_20260620.json`.

Decision:

- The weak labels are not degenerate: the projection learns a measurable
  positive/negative margin.
- But the projection does not improve current identity decisions.  In the
  strongest current branch it changes target evidence ordering slightly, but
  the predicted assignment and pair score remain unchanged.
- Close this as a feature/projection challenger.  The next branch should solve
  large false-split identities in one namespace, rather than adding another
  auxiliary visual feature.

### 2026-06-20 Self-play pivot: large false-split micro probes

Protocol lesson from Deli AutoResearch:

- A failed move is useful if it narrows the action space.
- The proposer here tried two structural edits from the current delivered
  global-ID assignment: pure component merge and bulk conflict split.
- The opponent/evaluator was delivery side effects: pair metrics plus full
  HOTA/IDF1 when a candidate reached the top.

Input assignment:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_top1_assignments_20260620.csv`.

Results:

| proposer move | evidence changed | pair F1 | full IDF1 | evaluator decision |
| --- | --- | ---: | ---: | --- |
| pure component merge | `3240` component edges, rank-1 accepts `1` merge | `0.772727` | `0.653541` | no-op |
| bulk conflict split | split `41` components, rewrite `2252` tracklets | `0.618505` | `0.602445` | destructive |

Artifacts:

- pure-merge partial log:
  `local_runs/remote_h100_test_3_20260620/large_false_split_component_merge_best_assignment_20260620.log`;
- split micro JSON:
  `local_runs/remote_h100_test_3_20260620/no_anchor_large_false_split_split_then_merge_micro_best_assignment_20260620.json`;
- split micro CSV:
  `local_runs/remote_h100_test_3_20260620/no_anchor_large_false_split_split_then_merge_micro_best_assignment_20260620.csv`.

Interpretation:

- Pure merge lacks enough clean evidence to attach high-mass false-split
  identity islands.
- Bulk splitting uses real cannot-link evidence, but it overreacts: `2252`
  tracklets are moved into new parts, causing a recall collapse.
- The next proposer should keep the same large-false-split objective but
  switch action granularity: surgical source groups, one or a few edits at a
  time, with a learned or cached full-score side-effect proxy before full
  scoring.

### 2026-06-20 Proxy-ranked surgical repair

New proposer rule:

- Add `--rank-by full_proxy` to `kit/no_anchor_assignment_conflict_reassign_sweep.py`.
- Use no-GT evidence to rank candidate rows before paying the full-score cost:
  target/source quality, visual consensus, min-view similarity, margin, and a
  penalty for broader edits.

Evaluator result:

| row selector | accepted edits | moved tracklets | pair F1 | full IDF1 | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| pair-ranked G8 | `4` | `32` | `0.775234` | `0.653541` | pair good, e2e weak |
| full-proxy | `1` | `8` | `0.772654` | `0.653823` | slightly better e2e, below gate |

Self-play update:

- The proxy learned the right preference for conservative edits, but it is too
  myopic: the top ranks are duplicate variants of the same source group.
- Next proposer should add diversity-aware source selection or a learned edit
  acceptor trained from prior no-GT proposal features and posthoc full outcomes.

### 2026-06-20 Unique-signature proxy check

Deli AutoResearch/self-play distillation for this rerun:

- The Deli framework page frames long-horizon research as durable state,
  stall detection, and structural pivots rather than repeated tactical tuning.
- The self-play story is useful here because it treats the evaluator as an
  opponent: a proposal only survives if it improves the actual held-out metric,
  and a down-score is evidence rather than failure.
- For VLINCS, the proposer is the no-anchor source-group editor, the opponent
  is full DS1 IDF1/HOTA plus cannot-link fragmentation, and the evaluator is
  allowed to use GT only after prediction for scoring.

Rerun:

- The solver now de-duplicates accepted-edit signatures before spending full
  evaluation slots.
- Artifact:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_conflict_reassign_fullproxy_unique_pairfull_20260620.json`.

Outcome:

| full rank | accepted edits | moved tracklets | pair F1 | full IDF1 | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| `1` | `1` | `8` | `0.772654` | `0.653823` | best unique proxy row, below gate |
| `2` | `2` | `16` | `0.774082` | `0.653823` | no e2e gain from second source group |
| `3` | `3` | `24` | `0.774252` | `0.653042` | broader edit hurts e2e |

AutoResearch state update:

- This counts as a real finding, but not a metric breakthrough.
- The loop should stop trying wider variants of the same surgical repair and
  pivot to a learned edit acceptor / diverse source proposer that predicts
  component-level delivery side effects before full scoring.

### 2026-06-20 Pending proposer: diverse first edit

Prepared next proposer:

- Add `--full-selection diverse_first_edge` to force the evaluator to sample
  different first source-to-target edits.
- This is the direct self-play response to the opponent's finding that
  signature de-duplication still spent all full-score candidates on the same
  first edit family.

Blocked execution:

- Local self-test passed.
- h100-test-3 could not be reached for a trusted run: Pluto service failed to
  connect, and SSH timed out during banner exchange.
- Leave the AutoResearch state active; this is an infrastructure pause, not a
  research conclusion.

### 2026-06-20 Proposer diversity patch

Self-play response:

- The opponent found that full-score de-duplication still evaluated variants of
  the same first edit family.
- The proposer now has a stronger knob:
  `--candidate-skip-first-edge-families`, which skips the top `N` unique
  source-to-target edge families before greedy assignment construction.

Executable artifact:

- `kit/run_no_anchor_false_split_diversity.sh`.

Status:

- Local compile/self-test and launcher syntax checks pass.
- Remote run is pending on Pluto connectivity.  This keeps the research state
  active and gives the next continuation a single command to run instead of a
  vague plan.

### 2026-06-20 Learned opponent-to-proposer transfer

New AutoResearch artifact:

- `kit/analyze_no_anchor_full_proxy_training.py` converts prior full-scored
  proposal outcomes into a compact learned full-proxy model.
- The default training set excludes oracle rows and uses GT only as posthoc
  full-score labels from completed evaluations.

Result:

- no-oracle comparable rows: `32`;
- compact features: `29`;
- full IDF1 range: `0.602445-0.654009`;
- ridge LOOCV: corr `0.996050`, MAE `0.000913`.

Protocol implication:

- The hand-written full proxy was useful as an opponent, but not sufficient as
  a proposer.
- The next executable proposer is now:
  learned full-proxy ranking + candidate first-edge family skipping.
  This is still no-anchor because it does not use labeled anchor tracklets or
  GT global IDs at prediction time.

### 2026-06-20 Deli opponent guard

Distillation from the open Deli AutoResearch/self-play thread:

- The key mechanism to copy is the honest opponent, not GRPO itself.
- For VLINCS, pair F1 is the proposer-side score; full DS1 IDF1/HOTA is the
  opponent.  If they disagree, the opponent wins.
- Repeated runtime or parser failures should become preflight checks, just as
  the Deli run converted a recurring submission bug into a check script.

Executable artifact:

- `kit/no_anchor_deli_opponent.py`.

Current verdict:

- gate input:
  `local_runs/no_anchor_result_gate_all_20260620.json`;
- progress input:
  `autoresearch_state/no_anchor_global_id/state/progress.json`;
- verdict JSON:
  `local_runs/no_anchor_deli_opponent_20260620.json`;
- verdict markdown:
  `reports/no_anchor_deli_opponent_verdict_20260620.md`;
- result: `pivot`;
- blockers:
  best pair/global candidate has pair F1 `0.954691` but full IDF1 `0.085412`;
  best gated full IDF1 is `0.654009`, below standing `0.655240`;
  stale count requires a structural pivot.

Next action remains:

- submit `kit/run_no_anchor_edge_rank_target_fragment.sh` and the tiny-fragment
  branch when Pluto/SSH recovers.

Remote execution probe:

- `h100-test-3`: Pluto status failed with `Failed to connect to Pluto service`;
  SSH dry-run failed during banner exchange.
- `h100-test-2`: same Pluto status failure and SSH banner timeout.
- `test-video-0`: same Pluto status failure and SSH banner timeout.
- No new DS1 full-score artifact was produced in this loop.

### 2026-06-20 Result-gate parser hardening

Evaluator bug found:

- Several production full-score artifacts, including the standing best
  `no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619.json`,
  store metrics under `rows[0].idf1/hota/assa`, not under `top[*].full_idf1`.
- The old gate therefore missed the production best `0.655240` and reported
  the best gated full IDF1 as `0.654009`.
- After reading `rows`, the gate also exposed a second bug: oracle/audit rows
  could be admitted if they lacked explicit GT-selection metadata.

Fix:

- `kit/no_anchor_result_gate.py` now reads `top`, `rows`, `full_rows`,
  `top_full_rows`, `passing_rows`, and `top_rows`.
- It normalizes `idf1/hota/assa` to `full_idf1/full_hota/full_assa`.
- It excludes rows marked or named as GT analysis/selection/oracle diagnostics:
  `uses_gt_for_analysis_only`, `uses_gt_for_filter_selection`,
  `selection_uses_gt_metric`, `with_oracle`, `oracle_repair`, and `oracle_*`
  modes.
- Added `--self-test` for this parser case.

Refreshed gate:

- `local_runs/no_anchor_result_gate_all_20260620.json`;
- `pass_joint=false`;
- best production e2e: `0.655240`;
- best joint candidate: `0.654009`;
- refreshed Deli opponent verdict remains `pivot`.

Post-fix remote probe:

- `h100-test-3` and `h100-test-2` were checked again with the Pluto status
  command and generated SSH config/password fallback dry-run.
- Both status commands returned `Failed to connect to Pluto service`.
- Both SSH probes timed out during banner exchange.
- The edge-rank target-fragment sweeps are still prepared, but not submitted.

### 2026-06-20 Deli AutoResearch to strict full-score admission

Distillation:

- Deli AutoResearch is most useful here as a research-loop operating system:
  file-backed state, anti-loop pivoting, worker/evaluator separation, and
  honest downward score movement.
- Its self-play lesson maps to VLINCS as:
  proposer = no-anchor candidate generator;
  opponent = cannot-link, delivery, side-effect, and stale-family refuter;
  evaluator = pair metrics plus full DS1 IDF1/HOTA.
- Therefore a pair/proxy winner is not allowed to spend full-score budget unless
  it survives the opponent checks and is predicted to beat the current
  production full IDF1.

Implementation:

- `kit/no_anchor_fullscore_scheduler.py` now mirrors the hardened result gate.
- It reads candidate rows from `top`, `rows`, `full_rows`, `top_full_rows`,
  `results`, and `top_rows`.
- It maps full-score rows with `idf1/hota/assa` into
  `known_full_idf1_norm`.
- It excludes oracle/GT-selection diagnostics using:
  `uses_gt_for_analysis_only`, `uses_gt_for_filter_selection`,
  `selection_uses_gt_metric`, path markers such as `with_oracle`,
  `oracle_repair`, `pervideo_filter_oracle`, and `oracle_*` mode names.
- By default it requires `predicted_full_idf1 > 0.655240`.  The old exploratory
  behavior is still available only with `--allow-predicted-below-current`.
- Added a scheduler `--self-test` covering duplicate-family suppression,
  current-best rejection, and oracle rejection.

Strict scheduler example:

```bash
python kit/no_anchor_fullscore_scheduler.py \
  --candidate local_runs/no_anchor_delivery_proxy_pair_candidates_20260620.json \
  --candidate local_runs/no_anchor_delivery_proxy_pair_candidates_with_full_20260620.json \
  --candidate local_runs/no_anchor_delivery_proxy_pair_candidates_20260620.csv \
  --json local_runs/no_anchor_fullscore_scheduler_strict_20260620.json \
  --csv local_runs/no_anchor_fullscore_scheduler_strict_20260620.csv \
  --md reports/no_anchor_fullscore_scheduler_strict_20260620.md
```

Strict scheduler output:

- raw candidate rows: `150`;
- strict eligible rows: `12`;
- selected diverse candidates: `2`;
- selected candidate 1:
  `conflict_subcluster_reassign_candidate_search:source_target:12:19`,
  predicted full `0.656057`, pair F1/P/R
  `0.767329 / 0.814359 / 0.725434`;
- selected candidate 2:
  `louvain:artifact:no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json`,
  predicted full `0.655826`, pair F1/P/R
  `0.764808 / 0.810282 / 0.724168`.

Exploratory comparison:

- With `--allow-predicted-below-current`, the same pool has `69` eligible rows
  and `11` selected candidates.
- This is useful as an ablation queue, but not the first remote budget target.
- The strict-vs-explore gap is now an anti-loop signal: the candidate family is
  saturated near the current full-IDF1 ceiling, so the next structural move
  should remain target-fragment / tiny-fragment repair rather than another broad
  global threshold sweep.

Prepared artifacts:

- `local_runs/no_anchor_fullscore_scheduler_strict_20260620.json`;
- `local_runs/no_anchor_fullscore_scheduler_strict_20260620.csv`;
- `reports/no_anchor_fullscore_scheduler_strict_20260620.md`;
- `local_runs/no_anchor_fullscore_scheduler_explore_20260620.json`;
- `reports/no_anchor_fullscore_scheduler_explore_20260620.md`.

Remote probe after strict scheduling:

- `h100-test-3`: Pluto status returned `Failed to connect to Pluto service`;
  launcher dry-run failed during SSH banner exchange.
- `h100-test-2`: same Pluto status failure and SSH banner timeout.
- No strict-manifest candidate has been DS1 full-scored in this loop.

### 2026-06-20 Artifact provenance guard for broad scheduling

Bug found:

- A broad strict scheduler pass over existing pair and pair-only artifacts read
  `4349` rows and initially selected an `assignment_state_policy` row with
  pair F1 `0.954690`.
- That row referenced
  `local_runs/remote_h100_test_3_20260619/no_anchor_state_policy_quality060_20260619.json`,
  whose own `top` rows already contain full IDF1 `0.085412`, HOTA
  `0.154201`, and only about `431` delivered tracklets.
- The candidate row itself did not inline delivery/full fields, so the
  scheduler had treated it as an unscored high-proxy pair winner.

Fix:

- `kit/no_anchor_fullscore_scheduler.py` now follows candidate artifact
  provenance.
- If a candidate row has a local JSON/CSV `artifact`, the scheduler reads the
  artifact, matches rows by mode and pair metrics, and imports
  `known_full_idf1`, `full_hota/full_assa` context, and delivery fields before
  eligibility checks.
- The scheduler self-test now includes a high-pair/low-full state-policy
  artifact and requires rejection by `known_below_current_best` and
  `low_delivery`.

Broad strict manifest after artifact-provenance fix:

```bash
python kit/no_anchor_fullscore_scheduler.py \
  --candidate 'local_runs/remote_h100_test_3_20260620/*pair*.json' \
  --candidate 'local_runs/remote_h100_test_3_20260620/*paironly*.json' \
  --candidate 'local_runs/remote_h100_test_3_20260619/*pair*.json' \
  --candidate 'local_runs/no_anchor_*pair_candidates*_20260620.json' \
  --json local_runs/no_anchor_fullscore_scheduler_broad_strict_20260620.json \
  --csv local_runs/no_anchor_fullscore_scheduler_broad_strict_20260620.csv \
  --md reports/no_anchor_fullscore_scheduler_broad_strict_20260620.md
```

Output:

- raw rows: `4349`;
- strict eligible rows: `26`;
- selected diverse candidates before family canonicalization: `5`;
- top selected family:
  `conflict_subcluster_reassign_candidate_search:source_target:12:15`,
  predicted full `0.656329`, pair F1/P/R
  `0.771045 / 0.816632 / 0.730278`;
- state-policy false positive is no longer selected.

Additional family canonicalization:

- The first broad strict manifest still spent two slots on the same edit family:
  a learned-candidate row represented it as `source_target:12:19`, while the
  underlying artifact represented it as `component:21->19`.
- `kit/no_anchor_fullscore_scheduler.py` now recovers component labels from
  referenced pair artifacts before falling back to signature/source-target keys.
- After this canonicalization, the same broad strict command keeps `26`
  eligible rows but selects `4` unique candidates.

Current broad strict selected candidates:

| rank | predicted full | pair F1 | pair P | pair R | family |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.656329` | `0.771045` | `0.816632` | `0.730278` | `conflict_subcluster_reassign_candidate_search:component:32->15` |
| 2 | `0.656447` | `0.767329` | `0.814359` | `0.725434` | `conflict_subcluster_reassign_candidate_search:component:21->19` |
| 3 | `0.655404` | `0.770090` | `0.816031` | `0.729045` | `conflict_subcluster_reassign:component:21->0` |
| 4 | `0.655826` | `0.764808` | `0.810282` | `0.724168` | `louvain:artifact:no_anchor_louvain_face005_osnet005_s7true_quality060_pair_grid_20260619.json` |

Remote probe after artifact guard:

- `h100-test-3`: Pluto status still failed with
  `Failed to connect to Pluto service`; SSH dry-run still timed out during
  banner exchange.
- `h100-test-2`: same failure.
- No new DS1 full-score artifact was produced in this loop.

Family-canonicalization probe:

- Validation passed:
  `py_compile`, result-gate self-test, Deli-opponent self-test, scheduler
  self-test, launcher `bash -n`, JSON/JSONL checks.
- Current broad strict manifest:
  `4349` raw rows, `26` eligible rows, `4` selected unique families.
- Remote check after this change:
  `h100-test-3` and `h100-test-2` Pluto status still returned
  `Failed to connect to Pluto service`; dry-run SSH probes still timed out
  during banner exchange.
- No new DS1 full-score artifact was produced in this loop.

### 2026-06-20 Missing-mass budget audit for edge-table repair

Why this audit was added:

- The target-fragment edge-rank audit showed excellent local precision:
  combined softcut+DINO-fused rules can find tiny false-split edges with
  `17/17` to `19/19` true positives.
- Full-score target-fragment repair did not improve IDF1, so the missing
  question was not precision but mass coverage.
- Following the Deli AutoResearch pattern, this is an opponent/evaluator check:
  a candidate family can be rejected as a main path even when a local metric
  looks perfect.

New artifact:

```bash
python kit/analyze_no_anchor_edge_mass_budget.py \
  --oracle-json local_runs/remote_h100_test_3_20260620/no_anchor_softcut_current_oracle_repair_decomposition_full_top1_20260620.json \
  --edge-csv local_runs/remote_h100_test_3_20260619/no_anchor_softcut_current_edge_acceptor_table_20260619.csv \
  --edge-csv local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_acceptor_table_20260620.csv \
  --json local_runs/no_anchor_edge_mass_budget_audit_combined_20260620.json \
  --csv local_runs/no_anchor_edge_mass_budget_audit_combined_20260620.csv \
  --md reports/no_anchor_edge_mass_budget_audit_combined_20260620.md
```

Core numbers:

| item | value |
| --- | ---: |
| base full IDF1 | `0.653071` |
| oracle full IDF1 | `0.706202` |
| missing true-pair mass | `9,105,079,781` |
| estimated oracle-gap coverage needed for IDF1 0.70 | `88.3%` |
| best high-precision rule | `17/17`, precision `1.0` |
| best high-precision missing-mass coverage | `0.0448%` |
| best high-precision estimated full IDF1 | `0.653095` |
| wide highest-coverage rule coverage | `0.4329%` |
| wide highest-coverage edge precision | `6.9%` |
| wide highest-coverage estimated full IDF1 | `0.653301` |

Interpretation:

- Target-fragment repair is real and high-confidence, but it is too small to
  explain the path from `0.653071` to `0.70`.
- Loosening thresholds does not solve the mass gap; it only admits many false
  edges while still covering less than `0.5%` of the missing oracle mass.
- The next no-anchor branch should target high-mass false-split components
  directly: component bridge/split selection with side-effect prediction, not
  another tiny-fragment edge threshold sweep.

Artifacts:

- `local_runs/no_anchor_edge_mass_budget_audit_combined_20260620.json`;
- `local_runs/no_anchor_edge_mass_budget_audit_combined_wide_20260620.json`;
- `reports/no_anchor_edge_mass_budget_audit_combined_20260620.md`;
- `reports/no_anchor_edge_mass_budget_audit_combined_wide_20260620.md`.

### 2026-06-20 Candidate-search fidelity fix and high-mass bridge branch

Important implementation issue found:

- `conflict_subcluster_reassign_candidate_search` creates rows by ranking
  candidate source-target edges, applying prefix / skip-family diversity, and
  storing the chosen edit set in `accepted_preview`.
- The later full-score and assignment-export path reconstructed rows through
  the ordinary `_choose_targets` path instead of directly replaying
  `accepted_preview`.
- That means some candidate-search full-score slots could evaluate a different
  edit than the row selected by the candidate-search ranker.

Patch:

- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now restores
  candidate-search accepted edits from `accepted_preview.source_seqs` to
  `source_indices` for both full scoring and assignment export.
- Added self-test coverage for preview restoration.
- Added `mass_bridge_proxy`, a no-GT ranker that explicitly rewards moved
  mass and target component size, while still using target visual agreement,
  target margin, source acceptor score, and side-effect proxy terms.

New launcher:

- `kit/run_no_anchor_high_mass_bridge.sh`

It runs:

- `--rank-by mass_bridge_proxy`;
- `--candidate-edge-rank-by mass_bridge`;
- `--candidate-search-top-n 512`;
- `--candidate-targets-per-source 2`;
- `--candidate-search-prefixes 8,16,32,64,128,256`;
- `--candidate-skip-first-edge-families 0,1,2,4,8,16,32,64`;
- `--full-selection diverse_first_edge`;
- `--full-top-n 12`.

Validation:

- `python -m py_compile kit/no_anchor_assignment_conflict_reassign_sweep.py`
  passed;
- `python kit/no_anchor_assignment_conflict_reassign_sweep.py --self-test`
  passed;
- `bash -n kit/run_no_anchor_high_mass_bridge.sh` passed.

Follow-up refinement:

- The first high-mass branch still built the candidate-edge pool using the old
  `proposal_score` order before row-level `mass_bridge_proxy` could rank
  prefix candidates.
- `kit/no_anchor_assignment_conflict_reassign_sweep.py` now exposes
  `--candidate-edge-rank-by {proposal,mass_bridge}`.
- The new `mass_bridge` edge pre-ranker is no-GT and favors source size,
  target size, target visual agreement, target margin, source acceptor score,
  and source score before prefix/skip truncation.
- Existing experiments keep the default `proposal`; the high-mass launcher uses
  `mass_bridge`.

Remote status:

- Pluto API still returned `Failed to connect to Pluto service` for
  `h100-test-3` and `h100-test-2`.
- `bash kit/run_no_anchor_high_mass_bridge.sh --job h100-test-3 --dry-run`
  reached SSH probing but timed out during banner exchange.
- No new DS1 full-score artifact was produced.

### 2026-06-20 Oracle gap concentration audit

After the edge-mass audit showed tiny-fragment repair is too low-mass, the next
question was whether the missing true-pair mass is diffuse or concentrated.  If
it is concentrated, a high-mass component bridge/split selector is a better
next experiment than another global threshold sweep.

New artifact:

```bash
python kit/analyze_no_anchor_oracle_gap_concentration.py \
  --oracle-json local_runs/remote_h100_test_3_20260620/no_anchor_softcut_current_oracle_repair_decomposition_full_top1_20260620.json \
  --json local_runs/no_anchor_oracle_gap_concentration_20260620.json \
  --csv local_runs/no_anchor_oracle_gap_concentration_20260620.csv \
  --md reports/no_anchor_oracle_gap_concentration_20260620.md
```

Result:

| prefix | false-split coverage of missing mass | false-merge coverage of base pred-pair mass |
| ---: | ---: | ---: |
| top 1 | `11.5%` | `2.6%` |
| top 3 | `26.3%` | `4.9%` |
| top 5 | `39.2%` | `6.8%` |
| top 10 | `56.8%` | `10.3%` |
| top 20 | `80.9%` | `14.7%` |
| top 30 | `95.8%` | `17.2%` |

Top false-split identities:

- GT `9`: false-split mass `1,046,536,325`, `26` predicted components;
- GT `36`: false-split mass `678,199,057`, `16` predicted components;
- GT `11`: false-split mass `672,659,005`, `25` predicted components;
- GT `43`: false-split mass `595,698,820`, `20` predicted components;
- GT `52`: false-split mass `575,122,667`, `22` predicted components.

Interpretation:

- The false-split gap is extremely concentrated: top-20 identities cover
  `80.9%` of the oracle missing true-pair mass, close to the `88.3%` linear
  coverage estimated as needed for IDF1 `0.70`.
- This strengthens the high-mass bridge branch: the system needs to recover a
  small set of large fragmented identities, not many tiny fragments.
- The next remote full-score queue should run `run_no_anchor_high_mass_bridge.sh`
  before spending more budget on target-fragment cleanup.

### 2026-06-20 High-mass component-merge provenance branch

After the oracle concentration audit, the ordinary component-merge sweep had
one remaining protocol flaw: it proposed component bridges, but did not preserve
which accepted edges were actually responsible for a row.  That made it hard to
audit whether a full-score candidate represented a high-mass false-split repair
or merely another locally safe, low-impact merge.

Patch:

- `kit/no_anchor_component_merge_sweep.py` now emits accepted-edge provenance:
  `accepted_preview`, score/rank-margin means, source/target weight sums,
  size-product sums, and mass proxy sums.
- `kit/no_anchor_assignment_component_merge_sweep.py` inherits the same fields
  when starting from an existing assignment CSV.
- Both paths expose `--rank-by {pair,precision,recall,mass_proxy,mass_then_pair}`.
  The default remains `pair`; the high-mass branch uses `mass_proxy`, which is
  computed from no-GT component weights and accepted edge evidence.
- Direct script invocation now inserts the repo root into `sys.path`, so
  preflight self-tests do not depend on an externally configured `PYTHONPATH`.
- `kit/no_anchor_fullscore_scheduler.py` now understands component-merge
  accepted-preview keys (`source/target`, `source_rep/target_rep`) when it
  builds de-duplication families for future manifests.

New launcher:

- `kit/run_no_anchor_high_mass_component_merge.sh`

It runs two assignment-level component-merge sweeps:

- fused feature:
  `ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_20260619.npz`;
- DINO feature:
  `ds1_tracklet_dinov2base_s1_20260620.npz`.

Both use:

- `--accepted-preview-n 40`;
- `--rank-by mass_proxy`;
- `--full-top-n 12`.

Validation:

- `python -m py_compile kit/no_anchor_component_merge_sweep.py kit/no_anchor_assignment_component_merge_sweep.py`
  passed;
- `python kit/no_anchor_component_merge_sweep.py --self-test` passed;
- `PYTHONPATH= python kit/no_anchor_component_merge_sweep.py --self-test`
  passed;
- `python kit/no_anchor_fullscore_scheduler.py --self-test` passed after the
  component-preview family-key update;
- `bash -n kit/run_no_anchor_high_mass_component_merge.sh` passed.

Remote status:

- `CONNECT_TIMEOUT=8 REMOTE_TIMEOUT=20 kit/run_no_anchor_high_mass_component_merge.sh --job h100-test-3 --dry-run`
  reached SSH probing and failed with the existing infrastructure blocker:
  `Connection timed out during banner exchange`.
- No new DS1 full-score artifact was produced.

### 2026-06-20 Edge-table candidate recall audit against oracle false splits

The edge-mass budget audit showed thresholded edge-table rules had too little
mass.  The stricter follow-up question is candidate recall: maybe the right
edges exist somewhere in the edge table but are not selected by the current
rules.  If even candidate recall is missing, the proposer itself is the
bottleneck.

New eval-only script:

- `kit/analyze_no_anchor_edge_table_oracle_coverage.py`

Command:

```bash
python kit/analyze_no_anchor_edge_table_oracle_coverage.py \
  --oracle-json local_runs/no_anchor_oracle_gap_concentration_20260620.json \
  --edge-csv local_runs/remote_h100_test_3_20260619/no_anchor_softcut_current_edge_acceptor_table_20260619.csv \
  --edge-csv local_runs/remote_h100_test_3_20260620/no_anchor_dinofused_edge_acceptor_table_20260620.csv \
  --json local_runs/no_anchor_edge_table_oracle_coverage_20260620.json \
  --csv local_runs/no_anchor_edge_table_oracle_coverage_20260620.csv \
  --md reports/no_anchor_edge_table_oracle_coverage_20260620.md
```

Result:

- combined candidate edges: `5240`;
- union true edges: `47`;
- union true edge same-ID mass: `39,412,716`;
- coverage of total missing true-pair mass: `0.00432865`;
- top-1 oracle GT edge coverage: `0.00030661`;
- top-3 oracle GT edge coverage: `0.00013385`;
- GT `36` and GT `11`, the second and third largest false-split identities,
  have zero true edge coverage in these edge tables.

Interpretation:

- Edge-table target-fragment repair is not only low-impact after thresholding;
  the candidate tables themselves do not recall the high-mass identities needed
  to approach full IDF1 `0.70`.
- This closes edge-table repair as a main route and upgrades high-mass
  component/hub bridge generation to the active route.
- Next remote queue when Pluto recovers:
  `kit/run_no_anchor_high_mass_bridge.sh` plus
  `kit/run_no_anchor_high_mass_component_merge.sh`.

### 2026-06-20 Deli AutoResearch protocol distillation

The Deli AutoResearch release was distilled into the no-anchor loop as a
research operating protocol:

- state is durable (`progress.json`, `findings.jsonl`,
  `directions_tried.json`);
- proposer, scheduler, executor, and judge are separate;
- stalls trigger structural pivots, not only threshold tuning;
- full-score drops are recorded as route-closing evidence rather than hidden;
- ready manifests are executed by launchers without asking for another manual
  decision.

Current VLINCS implementation of that protocol:

| role | artifact | no-anchor guarantee |
| --- | --- | --- |
| proposer | sweep scripts with `accepted_preview` | no anchors; GT only for optional eval |
| scheduler | `kit/no_anchor_fullscore_scheduler.py` | ranks by proxy/evidence only |
| executor | `kit/export_no_anchor_scheduler_manifest_assignments.py` | replays selected edits on base assignments |
| judge | `kit/evaluate_db_assignments_full.py` | canonical DS1 full scorer; GT eval only |

New execution path:

```bash
python kit/export_no_anchor_scheduler_manifest_assignments.py --self-test

bash kit/run_no_anchor_scheduler_manifest_fullscore.sh --ranks 1,2,3
```

The exporter was validated against the real strict mass-feature manifest:

| rank | provenance | replayed edit |
| ---: | --- | --- |
| 1 | JSON source rank `11` | 12 source seqs from component `32` to target `15` |
| 2 | JSON source rank `5` | 12 source seqs from component `21` to target `19` |
| 3 | CSV source rank `1`, recovered from sibling JSON | 8 source seqs from component `21` to target `0` |

Remote status:

- `run_no_anchor_scheduler_manifest_fullscore.sh --ranks 1,2,3` probed
  `h100-test-3`, `h100-test-2`, and `test-video-0`; all SSH probes timed out
  during banner exchange.
- Pluto CLI `job status ... --project video-world-models` failed for all three
  with `Failed to connect to Pluto service`.
- No full-score run started.

Next direction:

- Execute this manifest as soon as Pluto recovers, because it is now a concrete
  low-cost full-score check.
- Do not over-invest in this manifest: its proxy range is still only
  `0.655254..0.658302`.
- Continue the structural branch that can actually move toward `0.70`: generate
  new high-mass hub/component bridge candidates with accepted evidence and use
  the same proposer/scheduler/executor/judge split.

### 2026-06-20 Accepted-preview evidence audit

New ablation artifact:

```bash
python kit/analyze_no_anchor_preview_evidence.py \
  --json local_runs/no_anchor_preview_evidence_audit_20260620.json \
  --csv local_runs/no_anchor_preview_evidence_audit_20260620.csv \
  --md reports/no_anchor_preview_evidence_audit_20260620.md
```

Result:

- preview rows scanned: `5051`;
- full-labelled preview rows: `38`;
- labelled preview rows above current best `0.655240`: `0`;
- best labelled conflict candidate-search preview row: `0.654009`;
- best labelled conflict reassign preview row: `0.653823`;
- best labelled provisional relink preview row: `0.652265`.

Interpretation:

- The current family of small accepted-preview conflict reassignments has not
  produced a full-score improvement over the production best.
- The highest unlabelled pair-mass rows remain conflict candidate-search edits
  such as component `15 -> 14` with source size `12`, target size `214`, and
  pair-mass proxy `2568`; these are useful full-score candidates, but not a
  sufficient structural pivot.
- The next proposer should target larger hub/component bridge structures rather
  than only more `8..12`-tracklet source islands.

Operational pivot:

- `kit/run_no_anchor_high_mass_bridge.sh` now searches larger source groups:
  `--source-min-group-sizes 2,4,8,12,16`;
  `--source-max-group-sizes 12,24,48`;
  `--max-reassignments 2,4,8,12,24`.
- This preserves the no-anchor candidate-search path but increases the chance
  that the proposer creates true hub/component bridge rows instead of another
  small-island conflict cleanup.

### 2026-06-20 Hub-bridge composite proposer

New proposer:

```bash
python kit/compose_no_anchor_hub_bridge_candidates.py \
  --json local_runs/no_anchor_hub_bridge_composite_candidates_20260620.json \
  --csv local_runs/no_anchor_hub_bridge_composite_candidates_20260620.csv \
  --md reports/no_anchor_hub_bridge_composite_candidates_20260620.md
```

It reads no-anchor conflict accepted-preview rows, groups them by target
component, deduplicates repeated source islands, and emits executable composite
rows with full `accepted_preview` lists.

Generated candidates:

- unique preview items: `25`;
- target components: `18`;
- composite rows: `5`;
- top composite: `4+0 -> 6`, moves `16` tracklets, pair-mass proxy `4816`,
  pair F1 `0.772006`;
- largest composite: `19+4+0 -> 6`, moves `28` tracklets, pair-mass proxy
  `4816`, pair F1 `0.772046`.

Combined next queue:

```bash
python kit/no_anchor_fullscore_scheduler.py \
  --candidate local_runs/no_anchor_hub_bridge_composite_candidates_20260620.json \
  --candidate local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_stricttarget_pair_20260620.json \
  --candidate local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json \
  --candidate local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_narrow_selfplay_pair_20260620.json \
  --proxy-model-json local_runs/no_anchor_full_proxy_mass_features_ridge_model_20260620.json \
  --current-best-full-idf1 0.655240 \
  --min-pair-f1 0.70 --min-pair-precision 0.70 --min-pair-recall 0.70 \
  --min-delivery-tracklets 0 --min-predicted-full 0.650 \
  --allow-predicted-below-current \
  --top-n 10 \
  --json local_runs/no_anchor_fullscore_scheduler_next_queue_20260620.json \
  --csv local_runs/no_anchor_fullscore_scheduler_next_queue_20260620.csv \
  --md reports/no_anchor_fullscore_scheduler_next_queue_20260620.md
```

Result:

- raw / eligible / selected: `153 / 147 / 7`;
- queue composition: `4` hub composites and `3` strict conflict rows;
- best predicted row: `hub_component_bridge_composite:component:4+0->6`,
  predicted full-IDF1 `0.659003`.

Run when Pluto recovers:

```bash
bash kit/run_no_anchor_scheduler_manifest_fullscore.sh \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_next_queue_20260620.json \
  --ranks 1,2,3,4,5,6,7 \
  --run-name no_anchor_next_queue_fullscore_20260620
```

### 2026-06-20 Portfolio-level full-score queue

Individual hub edits still move too little mass, so a second-level no-anchor
proposer composes compatible selected rows into portfolio candidates.

```bash
python kit/compose_no_anchor_portfolio_candidates.py \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_next_queue_20260620.json \
  --json local_runs/no_anchor_hub_bridge_portfolio_candidates_20260620.json \
  --csv local_runs/no_anchor_hub_bridge_portfolio_candidates_20260620.csv \
  --md reports/no_anchor_hub_bridge_portfolio_candidates_20260620.md
```

Compatibility:

- no source-seq overlap;
- no chained source/target component edits.

Result:

- source scheduler rows: `7`;
- portfolio rows: `30`;
- top portfolio: source ranks `[1, 2, 3]`, targets `15+19+6`,
  moves `40` tracklets, `4` edits, pair-mass proxy `5764`, predicted full
  `0.662663`.

Portfolio scheduler:

```bash
python kit/no_anchor_fullscore_scheduler.py \
  --candidate local_runs/no_anchor_hub_bridge_portfolio_candidates_20260620.json \
  --proxy-model-json local_runs/no_anchor_full_proxy_mass_features_ridge_model_20260620.json \
  --current-best-full-idf1 0.655240 \
  --min-pair-f1 0.70 --min-pair-precision 0.70 --min-pair-recall 0.70 \
  --min-delivery-tracklets 0 --min-predicted-full 0.650 \
  --allow-predicted-below-current \
  --top-n 20 \
  --json local_runs/no_anchor_fullscore_scheduler_portfolio_next_queue_20260620.json \
  --csv local_runs/no_anchor_fullscore_scheduler_portfolio_next_queue_20260620.csv \
  --md reports/no_anchor_fullscore_scheduler_portfolio_next_queue_20260620.md
```

Result:

- raw / eligible / selected: `30 / 30 / 20`;
- best scheduled row: `hub_bridge_portfolio:component:0+21+32+4->15+19+6`,
  predicted full `0.662663`, scheduler score `0.661267`.

Run when Pluto recovers:

```bash
bash kit/run_no_anchor_scheduler_manifest_fullscore.sh \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_portfolio_next_queue_20260620.json \
  --ranks 1,2,3,4,5,6,7,8 \
  --run-name no_anchor_portfolio_next_queue_fullscore_20260620
```

Current remote status:

- local launcher checks pass;
- `h100-test-3` still fails at SSH banner exchange timeout, so no full-score run
  has started.

### 2026-06-20 Video-focus full-score budget correction

The portfolio queue is executable, but its top proxy-ranked rows do not
necessarily touch the videos that dominate the e2e miss.  The current production
best has per-video bottlenecks:

- MCAM04 Tc6: IDF1 `0.560694`;
- MCAM06 Tc6: IDF1 `0.608378`;
- MCAM03 Tc8: IDF1 `0.628615`.

Rerank the portfolio queue by bottleneck-video coverage:

```bash
python kit/rerank_no_anchor_scheduler_by_video_focus.py \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_portfolio_next_queue_20260620.json \
  --seq-video-csv local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_top1_assignments_20260620.csv \
  --metric-json local_runs/remote_h100_test_3_20260620/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.json \
  --metric-row-index 0 \
  --target-idf1 0.70 \
  --top-n 8 \
  --json local_runs/no_anchor_fullscore_scheduler_portfolio_video_focus_20260620.json \
  --csv local_runs/no_anchor_fullscore_scheduler_portfolio_video_focus_20260620.csv \
  --md reports/no_anchor_fullscore_scheduler_portfolio_video_focus_20260620.md
```

Result:

- top video-focus rows are original portfolio ranks `19`, `18`, and `20`;
- these rows move `48`, `36`, and `44` tracklets respectively;
- all `8` reranked rows materialize through
  `export_no_anchor_scheduler_manifest_assignments.py`.

Then widen the scheduler to all `30` eligible portfolio rows before reranking:

```bash
python kit/no_anchor_fullscore_scheduler.py \
  --candidate local_runs/no_anchor_hub_bridge_portfolio_candidates_20260620.json \
  --proxy-model-json local_runs/no_anchor_full_proxy_mass_features_ridge_model_20260620.json \
  --current-best-full-idf1 0.655240 \
  --min-pair-f1 0.70 --min-pair-precision 0.70 --min-pair-recall 0.70 \
  --min-delivery-tracklets 0 --min-predicted-full 0.650 \
  --allow-predicted-below-current \
  --top-n 30 \
  --json local_runs/no_anchor_fullscore_scheduler_portfolio_all30_20260620.json \
  --csv local_runs/no_anchor_fullscore_scheduler_portfolio_all30_20260620.csv \
  --md reports/no_anchor_fullscore_scheduler_portfolio_all30_20260620.md

python kit/rerank_no_anchor_scheduler_by_video_focus.py \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_portfolio_all30_20260620.json \
  --seq-video-csv local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_top1_assignments_20260620.csv \
  --metric-json local_runs/remote_h100_test_3_20260620/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.json \
  --metric-row-index 0 \
  --target-idf1 0.70 \
  --top-n 10 \
  --json local_runs/no_anchor_fullscore_scheduler_portfolio_all30_video_focus_20260620.json \
  --csv local_runs/no_anchor_fullscore_scheduler_portfolio_all30_video_focus_20260620.csv \
  --md reports/no_anchor_fullscore_scheduler_portfolio_all30_video_focus_20260620.md
```

All30 video-focus result:

- selected `10` rows;
- top rows are original portfolio ranks `19`, `18`, `30`, `29`, and `23`;
- exporter materialization verified all rank `1..10` rows;
- moved tracklet counts are `[48, 36, 72, 60, 60, 48, 44, 32, 56, 44]`.

Run first when remote execution recovers:

```bash
bash kit/run_no_anchor_scheduler_manifest_fullscore.sh \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_portfolio_all30_video_focus_20260620.json \
  --ranks 1,2,3,4,5,6,7,8,9,10 \
  --run-name no_anchor_portfolio_all30_video_focus_fullscore_20260620
```

This is still no-anchor: GT is not used as labels, anchors, or assignment
evidence.  The per-video full-score feedback is used only to allocate scarce
research full-score budget toward the current e2e bottleneck.

### 2026-06-20 Direct video-focus portfolio proposer

The all30 video-focus queue can only recombine earlier selected scheduler rows.
The direct proposer instead scans no-anchor accepted-preview evidence from
conflict-reassign artifacts and composes cross-target portfolios focused on
current bottleneck videos.

```bash
python kit/compose_no_anchor_video_focus_portfolio_candidates.py \
  --input-glob 'local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign*_20260620.json' \
  --seq-video-csv local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_top1_assignments_20260620.csv \
  --metric-json local_runs/remote_h100_test_3_20260620/no_anchor_submission_switch_current_conflictg8_quality_explicit_20260620.json \
  --metric-row-index 0 \
  --target-idf1 0.70 \
  --min-item-focus 0.05 \
  --min-portfolio-focus 1.5 \
  --min-items 2 \
  --max-items 6 \
  --min-moved-tracklets 40 \
  --max-seed-items 80 \
  --top-n 80 \
  --json local_runs/no_anchor_video_focus_portfolio_moderate_candidates_20260620.json \
  --csv local_runs/no_anchor_video_focus_portfolio_moderate_candidates_20260620.csv \
  --md reports/no_anchor_video_focus_portfolio_moderate_candidates_20260620.md

python kit/no_anchor_fullscore_scheduler.py \
  --candidate local_runs/no_anchor_video_focus_portfolio_moderate_candidates_20260620.json \
  --proxy-model-json local_runs/no_anchor_full_proxy_mass_features_ridge_model_20260620.json \
  --current-best-full-idf1 0.655240 \
  --min-pair-f1 0.70 --min-pair-precision 0.70 --min-pair-recall 0.70 \
  --min-delivery-tracklets 0 --min-predicted-full 0.650 \
  --allow-predicted-below-current \
  --top-n 8 \
  --json local_runs/no_anchor_fullscore_scheduler_video_focus_moderate_portfolio_20260620.json \
  --csv local_runs/no_anchor_fullscore_scheduler_video_focus_moderate_portfolio_20260620.csv \
  --md reports/no_anchor_fullscore_scheduler_video_focus_moderate_portfolio_20260620.md
```

Moderate result:

- `16` candidate rows, `8` selected;
- all selected rows pass pair gates;
- selected rows move `[52,52,52,52,52,52,52,56]` tracklets;
- `export_no_anchor_scheduler_manifest_assignments.py` materializes all
  selected ranks.

Run moderate first:

```bash
bash kit/run_no_anchor_scheduler_manifest_fullscore.sh \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_video_focus_moderate_portfolio_20260620.json \
  --ranks 1,2,3,4,5,6,7,8 \
  --run-name no_anchor_video_focus_moderate_portfolio_fullscore_20260620
```

Aggressive variant:

- `local_runs/no_anchor_fullscore_scheduler_video_focus_direct_portfolio_20260620.json`
- selected rows move `92..96` tracklets;
- use after the moderate queue because the side-effect risk is higher.

### 2026-06-20 Local sample full-score/export runner

Deli-style "ready means execute" exposed a local execution gap: when Pluto is
down, we should still be able to materialize candidate submissions instead of
waiting.  The new local runner bridges scheduler manifests to DS1 tracklet
parquets:

```bash
python kit/run_no_anchor_scheduler_manifest_sample_fullscore.py \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_video_focus_moderate_portfolio_20260620.json \
  --base-assignment-csv local_runs/remote_h100_test_3_20260620/no_anchor_recovered_90family_base_assignments_20260620.csv \
  --run-dir local_runs/no_anchor_video_focus_moderate_portfolio_recovered_base_local_export_20260620 \
  --selection-ranks 1,2,3,4,5,6,7,8 \
  --allow-no-gt-export
```

Results:

- materialized ranks `1..8`;
- each rank exports a 10-video submission zip;
- each rank contains about `1.72M` detection rows after fallback singletons;
- local GT is not mounted, so these are export artifacts, not full-score
  improvements;
- summary report:
  `reports/no_anchor_deli_distillation_local_export_20260620.md`.

The recovered base assignment is a majority-vote local reconstruction over the
seven returned `90xxxxxx`-namespace assignment CSVs:

- `7487` rows;
- `0` missing seqs;
- `40` unstable seqs;
- no anchors or GT used.

### 2026-06-20 Cross-queue portfolio composer

The direct video-focus portfolios and the hub-bridge portfolios are not
identical search neighborhoods.  I merged their selected rows into a union
scheduler and re-ran the no-anchor portfolio composer to find compatible
cross-queue combinations:

```bash
python kit/compose_no_anchor_portfolio_candidates.py \
  --scheduler-json local_runs/no_anchor_union_scheduler_existing_portfolios_20260620.json \
  --current-best-full-idf1 0.655240 \
  --min-items 2 \
  --max-items 3 \
  --min-moved-tracklets 1 \
  --top-n 20 \
  --json local_runs/no_anchor_crossqueue_portfolio_relaxed_candidates_20260620.json \
  --csv local_runs/no_anchor_crossqueue_portfolio_relaxed_candidates_20260620.csv \
  --md reports/no_anchor_crossqueue_portfolio_relaxed_candidates_20260620.md
```

Then scheduled and locally exported:

```bash
python kit/no_anchor_fullscore_scheduler.py \
  --candidate local_runs/no_anchor_crossqueue_portfolio_relaxed_candidates_20260620.json \
  --proxy-model-json local_runs/no_anchor_full_proxy_mass_features_ridge_model_20260620.json \
  --current-best-full-idf1 0.655240 \
  --min-pair-f1 0.70 --min-pair-precision 0.70 --min-pair-recall 0.70 \
  --min-delivery-tracklets 0 --min-predicted-full 0.650 \
  --allow-predicted-below-current \
  --top-n 7 \
  --json local_runs/no_anchor_fullscore_scheduler_crossqueue_portfolio_20260620.json \
  --csv local_runs/no_anchor_fullscore_scheduler_crossqueue_portfolio_20260620.csv \
  --md reports/no_anchor_fullscore_scheduler_crossqueue_portfolio_20260620.md
```

Outcome:

- `41` union scheduler rows;
- `7` compatible cross-queue rows;
- `3` scheduled rows;
- top predicted full IDF1 `0.663182`;
- local export directory:
  `local_runs/no_anchor_crossqueue_portfolio_recovered_base_local_export_20260620`.

This is currently the best unverified no-anchor full-score queue.  It still
requires canonical DS1 scoring before it counts as e2e progress.
