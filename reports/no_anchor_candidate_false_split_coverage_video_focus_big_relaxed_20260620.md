# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `19`
- audited rows: `19`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `273432781.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `19` | `138.0` | `16.0` | `0.6572921886840836` | `0.76887525` | `0.003602` | `5` | `10` | `video_focus_portfolio` |
| 2 | `13` | `130.0` | `15.0` | `0.6576231937669097` | `0.7667360000000001` | `0.003587` | `4` | `7` | `video_focus_portfolio` |
| 3 | `14` | `138.0` | `16.0` | `0.6574613423372365` | `0.7683433749999999` | `0.003587` | `4` | `9` | `video_focus_portfolio` |
| 4 | `18` | `138.0` | `16.0` | `0.6573128403574967` | `0.769587625` | `0.001279` | `4` | `10` | `video_focus_portfolio` |
| 5 | `12` | `128.0` | `14.0` | `0.6579352482597156` | `0.7675976428571428` | `0.001264` | `3` | `8` | `video_focus_portfolio` |
| 6 | `5` | `130.0` | `15.0` | `0.6578264990488146` | `0.7677346` | `0.001264` | `3` | `6` | `video_focus_portfolio` |
| 7 | `6` | `130.0` | `15.0` | `0.6578147992351129` | `0.7670526` | `0.001264` | `3` | `6` | `video_focus_portfolio` |
| 8 | `7` | `126.0` | `15.0` | `0.6577137475872821` | `0.7672597333333333` | `0.001264` | `3` | `6` | `video_focus_portfolio` |
| 9 | `11` | `130.0` | `15.0` | `0.6576404425855168` | `0.7674958666666666` | `0.001264` | `3` | `7` | `video_focus_portfolio` |
| 10 | `15` | `138.0` | `16.0` | `0.6574430433193318` | `0.769423125` | `0.001176` | `3` | `9` | `video_focus_portfolio` |
| 11 | `16` | `138.0` | `16.0` | `0.6574317030698206` | `0.76864675` | `0.001176` | `3` | `9` | `video_focus_portfolio` |
| 12 | `17` | `134.0` | `16.0` | `0.6572966619202131` | `0.769602125` | `0.001176` | `3` | `9` | `video_focus_portfolio` |
| 13 | `1` | `134.0` | `15.0` | `0.6580094389564215` | `0.7670733999999999` | `0.001161` | `2` | `5` | `video_focus_portfolio` |
| 14 | `2` | `134.0` | `15.0` | `0.6579977391427198` | `0.7663914000000001` | `0.001161` | `2` | `5` | `video_focus_portfolio` |
| 15 | `3` | `130.0` | `15.0` | `0.6579218323473518` | `0.7675591333333333` | `0.001161` | `2` | `5` | `video_focus_portfolio` |
| 16 | `4` | `130.0` | `15.0` | `0.6579101325336499` | `0.7668771333333333` | `0.001161` | `2` | `5` | `video_focus_portfolio` |
| 17 | `10` | `130.0` | `15.0` | `0.6577453900783463` | `0.767879` | `0.001161` | `2` | `6` | `video_focus_portfolio` |
| 18 | `8` | `138.0` | `16.0` | `0.6575912429864129` | `0.76889125` | `0.001161` | `2` | `8` | `video_focus_portfolio` |
| 19 | `9` | `138.0` | `16.0` | `0.6575820107877471` | `0.768114875` | `0.001161` | `2` | `8` | `video_focus_portfolio` |

## Top Edge Evidence

### Candidate rank 19
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `24` -> `38`: mass `0.000`, best_gt ``, target_gt_count `None`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `37` -> `7`: mass `10312352.000`, best_gt `52`, target_gt_count `6.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`

### Candidate rank 13
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `24` -> `38`: mass `0.000`, best_gt ``, target_gt_count `None`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `37` -> `7`: mass `10312352.000`, best_gt `52`, target_gt_count `6.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `3` -> `38`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 14
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `24` -> `38`: mass `0.000`, best_gt ``, target_gt_count `None`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `37` -> `7`: mass `10312352.000`, best_gt `52`, target_gt_count `6.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `3` -> `38`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 18
- `48` -> `46`: mass `0.000`, best_gt ``, target_gt_count `None`
- `20` -> `50`: mass `0.000`, best_gt ``, target_gt_count `None`
- `48` -> `9`: mass `0.000`, best_gt ``, target_gt_count `16.0`
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `24` -> `38`: mass `0.000`, best_gt ``, target_gt_count `None`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `37` -> `7`: mass `10312352.000`, best_gt `52`, target_gt_count `6.0`

### Candidate rank 12
- `38` -> `3`: mass `0.000`, best_gt ``, target_gt_count `8.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `37` -> `7`: mass `10312352.000`, best_gt `52`, target_gt_count `6.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `21` -> `19`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`

