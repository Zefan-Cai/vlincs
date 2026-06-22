# VLINCS No-Anchor Global-ID Remote Full-Score Queue Report

Date: 2026-06-20

## Deli AutoResearch Distillation

Read sources:

- https://victorchen96.github.io/auto_research/framework.html
- https://victorchen96.github.io/auto_research/paper.html
- https://victorchen96.github.io/blog_self_play_story.html

The useful import for VLINCS is not a new ReID architecture. It is a research
execution protocol:

- keep proposer, verifier, scorer, gate, and opponent as separate roles;
- persist state to files, not chat memory;
- submit ready candidates to canonical scoring instead of stopping at proxy;
- treat score drops as real evidence;
- force a structural pivot after repeated stale iterations.

For this run, that protocol became:

- proposer: assignment-summary proxy, self-play bridge ablation, adversarial bridge counterfactuals;
- verifier/opponent: no-GT metadata checks plus canonical DS1 scorer;
- scorer: remote `kit/evaluate_db_assignments_full.py` and `kit/no_anchor_pervideo_filter_selector.py`;
- gate: `kit/no_anchor_result_gate.py`;
- opponent verdict: `kit/no_anchor_deli_opponent.py`.

## Remote Execution Status

Pluto recovered on `h100-test-2` and `h100-test-3` through password SSH fallback.
`test-video-0` still failed SSH with `Connection closed by UNKNOWN port 65535`,
so it was not used for scoring.

New reusable launcher:

- `kit/run_no_anchor_assignment_queue_fullscore.sh`

The launcher deploys a selected assignment queue to a Pluto node and runs
canonical DS1 full scoring. It uses no anchors and no GT for prediction; GT is
used only by the evaluator.

## Canonical Full-Score Results

Standing best remains:

| artifact | IDF1 | HOTA | AssA |
| --- | ---: | ---: | ---: |
| `no_anchor_softcut_then_softoverlap_density_filter_selector_zip_20260619` | `0.655240` | `0.518652` | `0.534359` |

New remote-scored candidates:

| group | candidate | IDF1 | HOTA | AssA | decision |
| --- | --- | ---: | ---: | ---: | --- |
| assignment-summary top3 | balanced source selector | `0.653177` | `0.517094` | `0.532626` | reject: below standing |
| assignment-summary top3 | conservative source selector | `0.653175` | `0.517092` | `0.532623` | reject: below standing |
| assignment-summary top3 | edge-table focused | `0.653827` | `0.517796` | `0.533344` | keep as diagnostic only |
| density combo | edge-table + `density_oracle_lite` | `0.654041` | `0.517791` | `0.533451` | best new, still below standing |
| density combo | edge-table + `confidence_tail` | `0.653942` | `0.517651` | `0.533306` | reject |
| density combo | balanced + `density_oracle_lite` | `0.653347` | `0.517035` | `0.532683` | reject |
| density combo | balanced + `confidence_tail` | `0.653247` | `0.516895` | `0.532538` | reject |
| self-play bridge | provisional `44 -> 8` | `0.638369` | `0.508307` | `0.532770` | hard reject |
| self-play bridge | quarantine `17 -> 47` | `0.637193` | `0.505426` | `0.528272` | hard reject |
| adversarial bridge | `25 -> 31` | `0.653177` | `0.517094` | `0.532626` | near no-op |
| adversarial bridge | `24 -> 39` | `0.651455` | `0.515575` | `0.531548` | hard reject |
| adversarial bridge | `31 -> 25` | `0.651993` | `0.516071` | `0.531932` | hard reject |

Machine-readable artifacts:

- `local_runs/no_anchor_remote_fullscore_assignment_queues_summary_20260620.json`
- `local_runs/no_anchor_remote_fullscore_assignment_queues_summary_20260620.csv`
- `local_runs/remote_fullscore_fetch_20260620/`

## Per-Video Bottleneck

The standing best and new best are close overall, but the hard videos did not
move enough.

Standing best per-video IDF1:

| video | IDF1 |
| --- | ---: |
| `MCAM04 Tc6` | `0.560694` |
| `MCAM06 Tc6` | `0.608378` |
| `MCAM03 Tc8` | `0.628615` |
| `MCAM03 Tc6` | `0.690820` |

New edge-table focused full-score per-video IDF1:

| video | IDF1 |
| --- | ---: |
| `MCAM04 Tc6` | `0.560501` |
| `MCAM06 Tc6` | `0.606895` |
| `MCAM03 Tc8` | `0.626660` |
| `MCAM03 Tc6` | `0.688393` |

Interpretation:

- assignment-summary ranking overestimated balanced/conservative source selectors;
- adversarial one-bridge counterfactuals do not repair the hard videos;
- no-GT density filtering gives a small lift to edge-table focused, but not enough;
- the bottleneck is still source-local association in `MCAM04 Tc6` and `MCAM06 Tc6`,
  not just delivery-row confidence filtering.

## Gate And Opponent

Updated result gate:

- `local_runs/no_anchor_result_gate_after_remote_assignment_queues_20260620.json`
- `pass_joint=false`
- best e2e remains `0.655240`
- best joint remains `0.654009`

Updated Deli opponent:

- `local_runs/no_anchor_deli_opponent_after_remote_assignment_queues_20260620.json`
- `reports/no_anchor_deli_opponent_after_remote_assignment_queues_20260620.md`
- verdict: `pivot`

Opponent blockers:

- pair/global metric passes but full DS1 IDF1 is below target;
- best newly gated full IDF1 does not beat standing promoted artifact;
- stale count requires a structural pivot, not another threshold-only sweep.

## Conclusion

This run restored canonical remote full scoring and closed several candidate
families:

- source-selector queue: negative;
- self-play non-committed bridge promotion: strongly negative;
- adversarial one-bridge counterfactuals: no-op or negative;
- density filter on new assignments: small positive but below standing.

Next direction should stop broad bridge/selector tuning and target source-local
repairs in the hard videos:

- `MCAM04 Tc6`: association repair without dropping recall;
- `MCAM06 Tc6`: source-local verifier or temporal relink focused on detection-heavy false splits;
- build a no-GT adversarial verifier that rejects bridges if they degrade per-video association evidence before remote scoring.
