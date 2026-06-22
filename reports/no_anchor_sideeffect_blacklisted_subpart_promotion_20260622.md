# No-Anchor Side-Effect Blacklisted Subpart Promotion

Date: 2026-06-22

## Standing

- Setting: no-anchor global-ID research; GT is used only for canonical evaluation.
- Previous best before this run: IDF1 / HOTA / AssA = `0.657653 / 0.520723 / 0.535819`.
- New best: IDF1 / HOTA / AssA = `0.657887 / 0.520944 / 0.535983`.
- Improvement: IDF1 `+0.000234`, HOTA `+0.000221`, AssA `+0.000164`.
- E2E target remains open: `0.657887 < 0.70`.

## Method

The previous filter/admission sweep showed `p005_area` is locally optimal, so this run returned to identity evidence. It generated three no-anchor subpart-repair pools from the current best assignment `weakmetric 10 -> 22`:

- `weakmetric_dino`: weakmetric primary feature plus DINO view weight `0.25`.
- `siglip_dino`: SigLIP-fused primary feature plus DINO view weight `0.25`.
- `dino_weak`: DINO primary feature plus weakmetric view weight `0.35`.

Candidate generation used only assignment CSVs, tracklet features, temporal overlap conflicts, component sizes, and focus-video bookkeeping. No anchors or GT labels were used for proposal, filtering, or composition.

The selected candidates were filtered with a hard side-effect blacklist from previous full-score evidence:

`8->40, 9->19, 9->32, 10->9, 10->15, 10->22, 11->40, 19->11, 24->38, 24->41, 31->25, 32->9, 35->60, 40->2329, 55->58, 2329->40`

The three proposal pools produced `201` selected assignment rows before blacklist/diversity filtering; six novel candidates were full-scored. Then the two positive singles and the safe-neutral singles were composed into five combo ablations.

## Single-Candidate Results

| Candidate | Pool | Rank | Moved | Focus hits | IDF1 | HOTA | AssA | MCAM04 | MCAM06 Tc6 | MCAM03 Tc8 | Delta IDF1 | Label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `32->15` | `weakmetric_dino` | 47 | 7 | 3 | 0.657860 | 0.520922 | 0.535961 | 0.565566 | 0.612018 | 0.629337 | +0.000207 | positive |
| `47->2330` | `weakmetric_dino` | 49 | 7 | 3 | 0.657680 | 0.520745 | 0.535841 | 0.565570 | 0.610296 | 0.628528 | +0.000027 | positive |
| `24->31` | `weakmetric_dino` | 54 | 6 | 6 | 0.657653 | 0.520723 | 0.535819 | 0.565566 | 0.610296 | 0.628528 | +0.000000 | tie |
| `31->24` | `siglip_dino` | 65 | 4 | 3 | 0.657653 | 0.520723 | 0.535819 | 0.565566 | 0.610296 | 0.628528 | +0.000000 | tie |
| `44->29` | `siglip_dino` | 62 | 5 | 2 | 0.657291 | 0.520386 | 0.535575 | 0.565566 | 0.610296 | 0.626173 | -0.000362 | negative |
| `11->19` | `siglip_dino` | 58 | 9 | 2 | 0.657268 | 0.520269 | 0.535395 | 0.565566 | 0.604762 | 0.628528 | -0.000385 | negative |

## Combo Results

| Ranks | Component edits | Moved | IDF1 | HOTA | AssA | MCAM04 | MCAM06 Tc6 | MCAM03 Tc8 | Delta vs old | Delta vs best single | Label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `47+49` | `32->15+47->2330` | 14 | 0.657887 | 0.520944 | 0.535983 | 0.565570 | 0.612018 | 0.629337 | +0.000234 | +0.000027 | positive |
| `47+49+65` | `32->15+47->2330+31->24` | 18 | 0.657887 | 0.520944 | 0.535983 | 0.565570 | 0.612018 | 0.629337 | +0.000234 | +0.000027 | positive |
| `47+49+54` | `32->15+47->2330+24->31` | 20 | 0.657887 | 0.520944 | 0.535983 | 0.565570 | 0.612018 | 0.629337 | +0.000234 | +0.000027 | positive |
| `47+65` | `32->15+31->24` | 11 | 0.657860 | 0.520922 | 0.535961 | 0.565566 | 0.612018 | 0.629337 | +0.000207 | +0.000000 | positive |
| `47+54` | `32->15+24->31` | 13 | 0.657860 | 0.520922 | 0.535961 | 0.565566 | 0.612018 | 0.629337 | +0.000207 | +0.000000 | positive |

## Interpretation

- `32 -> 15` is the main new positive: it moved 7 tracklets and improved IDF1 to `0.657860`, mainly through MCAM06 Tc6 (`0.610296 -> 0.612018`) and MCAM03 Tc8 (`0.628528 -> 0.629337`).
- `47 -> 2330` is a tiny compatible positive. Alone it reaches `0.657680`; combined with `32 -> 15` it reaches the new best `0.657887`.
- `24 -> 31` and `31 -> 24` are safe-neutral: alone and in combos they tie the relevant base score without measurable gain.
- `11 -> 19` is a strong negative at `0.657268`, confirming the `11/19` component pair as risky in both directions after earlier `19 -> 11` refutations.
- `44 -> 29` is also negative at `0.657291`; it hurt MCAM03 Tc8 despite no MCAM04 change.
- The current search is still finding only micro-gains. The next useful step is to treat this run as a richer side-effect label bank and search for higher-mass candidates that preserve the `32 -> 15 + 47 -> 2330` gains.

## Promoted Artifact

- Promoted assignment CSV: `local_runs/remote_h100_test_3_20260622/no_anchor_sideeffect_blacklisted_subpart_search_20260622/combos/assignments/subpart_combo_r47_r49_14seq_assignments.csv`
- Remote promoted assignment CSV: `/mnt/localssd/vlincs_reid_runs/no_anchor_sideeffect_blacklisted_subpart_search_20260622/combos/assignments/subpart_combo_r47_r49_14seq_assignments.csv`
- Remote run: `/mnt/localssd/vlincs_reid_runs/no_anchor_sideeffect_blacklisted_subpart_search_20260622`

## Artifacts

- `local_runs/remote_h100_test_3_20260622/no_anchor_sideeffect_blacklisted_subpart_search_20260622/blacklisted_novel_selection.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_sideeffect_blacklisted_subpart_search_20260622/p005_fullscore_summary.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_sideeffect_blacklisted_subpart_search_20260622/p005_combo_fullscore_summary.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_sideeffect_blacklisted_subpart_search_20260622/combos/subpart_combo_manifest.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_sideeffect_blacklisted_subpart_search_20260622_light.tgz`
