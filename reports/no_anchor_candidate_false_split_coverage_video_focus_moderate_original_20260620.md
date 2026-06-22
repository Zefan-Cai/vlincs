# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `8`
- audited rows: `8`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `63863193.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `4` | `52.0` | `6.0` | `0.6582574627126219` | `0.7689035` | `0.002338` | `2` | `4` | `video_focus_portfolio` |
| 2 | `5` | `52.0` | `6.0` | `0.6587369549601252` | `0.7666855` | `0.002323` | `1` | `2` | `video_focus_portfolio` |
| 3 | `7` | `52.0` | `6.0` | `0.658484964490681` | `0.7674851666666668` | `0.002323` | `1` | `3` | `video_focus_portfolio` |
| 4 | `2` | `52.0` | `6.0` | `0.6582860172742488` | `0.7708031666666667` | `0.000015` | `1` | `4` | `video_focus_portfolio` |
| 5 | `8` | `56.0` | `6.0` | `0.6581622184344184` | `0.7694241666666667` | `0.000015` | `1` | `3` | `video_focus_portfolio` |
| 6 | `6` | `52.0` | `6.0` | `0.6588967422883966` | `0.7660423333333332` | `0.000000` | `0` | `2` | `video_focus_portfolio` |
| 7 | `3` | `52.0` | `6.0` | `0.658781458053995` | `0.7664033333333334` | `0.000000` | `0` | `2` | `video_focus_portfolio` |
| 8 | `1` | `52.0` | `6.0` | `0.6587678874033294` | `0.7684736666666666` | `0.000000` | `0` | `3` | `video_focus_portfolio` |

## Top Edge Evidence

### Candidate rank 4
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `21` -> `0`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `36` -> `5`: mass `0.000`, best_gt ``, target_gt_count `17.0`

### Candidate rank 5
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `21` -> `19`: mass `0.000`, best_gt ``, target_gt_count `None`
- `21` -> `0`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `36` -> `5`: mass `0.000`, best_gt ``, target_gt_count `17.0`
- `13` -> `49`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 7
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `21` -> `0`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `36` -> `5`: mass `0.000`, best_gt ``, target_gt_count `17.0`
- `13` -> `49`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 2
- `48` -> `9`: mass `0.000`, best_gt ``, target_gt_count `16.0`
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `21` -> `0`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `36` -> `5`: mass `0.000`, best_gt ``, target_gt_count `17.0`

### Candidate rank 8
- `20` -> `50`: mass `0.000`, best_gt ``, target_gt_count `None`
- `48` -> `9`: mass `0.000`, best_gt ``, target_gt_count `16.0`
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `21` -> `0`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`

