# No-Anchor Counter-Target Opponent Features V2

Date: 2026-06-21

## AutoResearch Distillation Delta

The Deli AutoResearch announcement is useful here as an operating protocol, not
as a replacement model.  The pieces that map directly to VLINCS are:

- persist state in files, not chat;
- ready means execute, but only after a concrete no-GT referee/opponent pass;
- record score drops and blocked candidates as refutations;
- separate proposer, referee, opponent, and delivery gate;
- after repeated stale iterations, change the evidence schema or candidate
  space rather than tuning thresholds inside the same family.

Sources read:

- https://victorchen96.github.io/auto_research/framework.html
- https://victorchen96.github.io/auto_research/paper.html
- https://victorchen96.github.io/blog_self_play_story.html

For this VLINCS loop, the self-play translation is:

- proposer: visual/component candidate generator;
- referee: multiview/crop/body evidence;
- opponent: same-video temporal overlap, local competitor density, namespace
  drift, and weak view agreement;
- gate: materialize only if the opponent cannot produce a concrete
  contradiction.

## What Changed

Tool updated:

- `kit/audit_no_anchor_countertarget_verifier.py`

New no-GT edge features:

- `view_margin_min`, `view_margin_mean`, `view_weak_margin_fraction`,
  `view_non_rank1_fraction`;
- `visual_opponent_risk_score`;
- `same_video_pair_fraction`, `local_pair_fraction_same_video`,
  `overlap_pair_fraction_same_video`;
- `temporal_opponent_risk_score`;
- `combined_opponent_risk_score`;
- source/target video-count metadata and `source_target_video_jaccard`.

The full-score scheduler and proxy trainer were also wired to understand these
fields:

- `kit/no_anchor_fullscore_scheduler.py`
- `kit/analyze_no_anchor_full_proxy_training.py`

Scheduler self-test now covers a visual-pass but temporal-overlap hard negative.

## Experiment

Base assignment:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`

Candidate source:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_countertarget_inputs_20260621/no_anchor_countertarget_compact_edges_20260621.json`

Feature ensemble:

- primary fused feature, weight `1.0`;
- pose/color feature, weight `0.5`;
- color histogram feature, weight `0.5`;
- DINOv2 base feature, weight `0.3`.

Remote artifacts:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_countertarget_compact_opponent_features_v2_20260621/audit_visual_features.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_countertarget_compact_opponent_features_v2_20260621/audit_temporal_features.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_countertarget_compact_opponent_features_v2_20260621/summary.json`

Local mirror:

- `local_runs/remote_h100_test_3_20260621/no_anchor_countertarget_compact_opponent_features_v2_20260621/`
- `local_runs/no_anchor_countertarget_opponent_features_v2_summary_20260621.json`

## Result

| audit | candidate rows | edge rows | accepted | temporal rejected | counter-target rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| visual-only | 10 | 18 | 1 | 0 | 17 |
| visual + temporal opponent | 10 | 18 | 0 | 1 | 17 |

Decision: do not materialize a submission from this batch.  The only
visual-only accepted edge is the known bad edge and is rejected by the no-GT
temporal opponent.

## Hard Negative

The visual-only accepted edge:

- candidate index: `5`
- edge index: `1`
- source component: `13`
- target component: `49`
- source seqs: `3408, 3518, 4311, 4889, 4938, 5026, 5062, 5158`
- source size: `8`
- target size: `155`
- weighted margin: `0.079118`
- rank vote: `0.75`
- margin vote: `0.50`
- view margin min: `-0.002524`
- weak view margin fraction: `0.50`
- visual opponent risk: `0.3625`
- combined opponent risk: `0.823333`
- source/target video jaccard: `0.25`

Temporal contradiction:

- same-video pairs: `720 / 1240`, fraction `0.580645`;
- local same-video pair fraction: `0.134722`;
- overlap pairs: `18`, overlap fraction among same-video pairs `0.025`;
- max same-video overlap: `247` frames;
- median source duration: `300` frames;
- max overlap/source-duration fraction: `0.823333`;
- min same-video gap: `0` frames.

This is exactly the case the AutoResearch self-play framing wants: the proposer
finds an apparently good visual bridge, and the opponent returns a concrete
counterexample in the same camera/time neighborhood.

## Proxy Recheck

Re-running compact full-proxy training after adding the schema produced:

- rows: `80`
- features: `45`
- LOOCV corr: `-0.032403`
- LOOCV MAE: `0.012658`

The feature count did not change because historical full-score rows do not yet
carry the new opponent fields.  This confirms the next action: candidate
generators must propagate opponent fields into output rows before the full-score
critic can learn from them.

## Next Direction

Promote the opponent feature schema into the next candidate generator:

1. score proposed accepted-preview edges with the counter-target opponent before
   writing candidate rows;
2. copy the edge-level risk fields into `accepted_preview`;
3. let `no_anchor_fullscore_scheduler.py --side-effect-risk-weight` penalize
   rows with concrete temporal or multiview contradictions;
4. only full-score candidates where the temporal opponent is clean and visual
   margins are not weak across half the views.

This keeps the no-anchor rule intact: GT appears only after materialization as a
delivery label, not as an anchor or selector.
