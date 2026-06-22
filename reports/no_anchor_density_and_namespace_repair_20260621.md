# No-Anchor Density Recheck And Namespace Repair Refutation

Date: 2026-06-21

## Purpose

Apply the same no-GT density filter to newly full-scored candidates and test whether strict target namespace repair can recover an old manifest row that failed because the current remote base no longer had the requested target component.

## Results

| candidate | raw IDF1 | best density policy | best density IDF1 | HOTA | AssA | moved | decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `mass_rank1_small_move_64_to_15` | 0.652184 | `density_simple` | 0.654409 | 0.517670 | 0.533220 | 12 | reject; below standing best |
| `referee_rank1_crossqueue` | 0.650714 | `density_simple` | 0.652932 | 0.515943 | 0.531419 | 28 | reject; below standing best |
| `mass_rank2_targetseq_namespace_repair_32_to_30` | 0.652468 | `density_simple` | 0.654638 | 0.518056 | 0.533962 | 12 | reject; below standing best |

Standing best remains IDF1 `0.655240`, HOTA `0.518652`, AssA `0.534359`.

## Namespace Repair

I patched `kit/export_no_anchor_scheduler_manifest_assignments.py` so that when a historical `target_component` is missing in the current base assignment, the exporter can repair it only if `target_top_seqs` uniquely vote for one current component. For rank2 this mapped old target component `19` to current component `30`, moving 12 tracklets from component `32` to predicted global ID `70000030`.

This is an operational fix, not a score win. The repaired rank2 candidate scored raw IDF1 `0.652468`; after the same no-GT density policies its best value was `0.654638`, still below best by `0.000602`.

## Side-Effect Interpretation

- `mass_rank1` and repaired `mass_rank2` are both low-mass single-edge changes, but both remain below best after density filtering.
- `referee_rank1` is rejected more strongly: multi-edge movement creates same-video overlap and broad target-component side effects.
- The side-effect audit now gives a production-side admission guard: replay failure, namespace drift, temporal overlap, and large multi-video targets trigger quarantine before full-score budget is spent.

## Artifacts

- `kit/audit_no_anchor_scheduler_side_effects.py`
- `kit/export_no_anchor_scheduler_manifest_assignments.py`
- `local_runs/no_anchor_scheduler_side_effect_audit_20260621.json`
- `reports/no_anchor_scheduler_side_effect_audit_20260621.md`
- `local_runs/no_anchor_density_and_namespace_repair_summary_20260621.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_candidate_density_filter_recheck/mass_rank1_density_filter.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_candidate_density_filter_recheck/referee_rank1_density_filter.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_mass_rank2_targetseq_repair_fullscore/rank02_targetseq_repair_full.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_mass_rank2_targetseq_repair_fullscore/rank02_targetseq_repair_density_filter.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_mass_rank2_targetseq_repair_fullscore/manifest_assignments.json`
