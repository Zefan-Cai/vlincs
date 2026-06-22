# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `8`
- audited rows: `8`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `21555371.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1` | `40.0` | `5.0` | `0.6584969549601252` | `0.7666855` | `0.002323` | `1` | `2` | `video_focus_portfolio` |
| 2 | `7` | `48.0` | `5.0` | `0.6578852162040673` | `0.7673818333333333` | `0.000015` | `1` | `3` | `video_focus_portfolio` |
| 3 | `3` | `48.0` | `5.0` | `0.65772749858187` | `0.7703633333333334` | `0.000015` | `1` | `3` | `video_focus_portfolio` |
| 4 | `5` | `48.0` | `5.0` | `0.6574188962451921` | `0.7698841666666666` | `0.000015` | `1` | `3` | `video_focus_portfolio` |
| 5 | `8` | `52.0` | `6.0` | `0.6587678874033294` | `0.7684736666666666` | `0.000000` | `0` | `3` | `video_focus_portfolio` |
| 6 | `2` | `40.0` | `5.0` | `0.6586567422883965` | `0.7660423333333332` | `0.000000` | `0` | `2` | `video_focus_portfolio` |
| 7 | `4` | `48.0` | `5.0` | `0.6581653326205299` | `0.7680338333333333` | `0.000000` | `0` | `2` | `video_focus_portfolio` |
| 8 | `6` | `48.0` | `5.0` | `0.6581545257823227` | `0.7659634999999999` | `0.000000` | `0` | `2` | `video_focus_portfolio` |

## Top Edge Evidence

### Candidate rank 1
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `21` -> `0`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `36` -> `5`: mass `0.000`, best_gt ``, target_gt_count `17.0`
- `13` -> `49`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 7
- `10` -> `26`: mass `0.000`, best_gt ``, target_gt_count `2.0`
- `15` -> `14`: mass `0.000`, best_gt ``, target_gt_count `None`
- `48` -> `9`: mass `0.000`, best_gt ``, target_gt_count `16.0`
- `20` -> `50`: mass `0.000`, best_gt ``, target_gt_count `None`
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`

### Candidate rank 3
- `48` -> `9`: mass `0.000`, best_gt ``, target_gt_count `16.0`
- `20` -> `50`: mass `0.000`, best_gt ``, target_gt_count `None`
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `21` -> `0`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 5
- `15` -> `14`: mass `0.000`, best_gt ``, target_gt_count `None`
- `48` -> `9`: mass `0.000`, best_gt ``, target_gt_count `16.0`
- `20` -> `50`: mass `0.000`, best_gt ``, target_gt_count `None`
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`

### Candidate rank 8
- `40` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `36` -> `5`: mass `0.000`, best_gt ``, target_gt_count `17.0`
- `26` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `13` -> `49`: mass `0.000`, best_gt ``, target_gt_count `None`
- `15` -> `14`: mass `0.000`, best_gt ``, target_gt_count `None`

