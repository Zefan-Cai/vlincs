# Current K3/Density Best Error Decomposition

Date: 2026-06-21

## Context

After the self-play critic rejected the old 20260620 queues, the active branch
is the current k3/density-filter best:

- assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`
- raw assignment full-score:
  `IDF1/HOTA/AssA = 0.653210 / 0.517030 / 0.532678`
- density-filtered standing best:
  `IDF1/HOTA/AssA = 0.655378 / 0.518798 / 0.534546`
- pair F1/P/R on raw assignment:
  `0.769367 / 0.816518 / 0.727364`

This report is eval-only diagnosis.  It does not use GT for training, anchors,
or production selection.

Artifact:

- `local_runs/no_anchor_current_k3_error_decomposition_20260621/k3_assignment_error_decomposition.json`
- remote:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_current_k3_error_decomposition_20260621/k3_assignment_error_decomposition.json`

## Per-Video Pair Metrics

| video | pair F1 | precision | recall | note |
| --- | ---: | ---: | ---: | --- |
| MCAM00 Tc6 | `0.952044` | `0.997391` | `0.910641` | already strong |
| MCAM00 Tc8 | `0.798566` | `0.902392` | `0.716167` | recall limited |
| MCAM03 Tc6 | `0.790401` | `0.838578` | `0.747460` | medium |
| MCAM03 Tc8 | `0.762848` | `0.803523` | `0.726093` | both precision/recall |
| MCAM04 Tc6 | `0.759042` | `0.790199` | `0.730248` | largest mass and lowest full IDF1 |
| MCAM05 Tc6 | `0.819835` | `0.865664` | `0.778615` | small video |
| MCAM05 Tc8 | `0.815112` | `0.855876` | `0.778055` | small video |
| MCAM06 Tc6 | `0.765680` | `0.836400` | `0.705987` | recall limited |
| MCAM06 Tc8 | `0.821171` | `0.879403` | `0.770172` | medium |
| MCAM08 Tc6 | `0.849379` | `0.894526` | `0.808569` | good but large mass |

## Top False-Merge Components

| predicted gid | false-merge mass | dominant GT | dominant frac | GT count | main cameras |
| ---: | ---: | ---: | ---: | ---: | --- |
| `96000035` | `790173339` | `36` | `0.602136` | `6` | MCAM04, MCAM03, MCAM08 |
| `96000048` | `368583737` | `36` | `0.445855` | `30` | MCAM03, MCAM04 |
| `96000021` | `317340419` | `31` | `0.802036` | `19` | MCAM08, MCAM04, MCAM03 |
| `96000026` | `306768505` | `37` | `0.884391` | `12` | MCAM04, MCAM08 |
| `96000007` | `256926684` | `30` | `0.861695` | `16` | MCAM03, MCAM04, MCAM08 |
| `96000005` | `247122795` | `41` | `0.902874` | `14` | MCAM04, MCAM08, MCAM03 |
| `96000003` | `232660326` | `23` | `0.902110` | `14` | MCAM04, MCAM08, MCAM00 |
| `96000008` | `198507968` | `29` | `0.816611` | `12` | MCAM04, MCAM08, MCAM03 |

## Top False-Split GT IDs

| GT id | false-split mass | pred parts | dominant prediction | dominant frac | main cameras |
| ---: | ---: | ---: | ---: | ---: | --- |
| `9` | `1046536325` | `26` | `96000000` | `0.701911` | MCAM04, MCAM03, MCAM08, MCAM00 |
| `36` | `678199057` | `16` | `96000035` | `0.657854` | MCAM03, MCAM04 |
| `11` | `672659005` | `25` | `96000037` | `0.786685` | MCAM08, MCAM04, MCAM03 |
| `52` | `612597459` | `24` | `96002329` | `0.680121` | MCAM04, MCAM08, MCAM03 |
| `43` | `595698820` | `20` | `96000015` | `0.649796` | MCAM04, MCAM08, MCAM03 |
| `31` | `452595928` | `18` | `96000021` | `0.745779` | MCAM04, MCAM08, MCAM03 |
| `24` | `321191884` | `19` | `96000017` | `0.754835` | MCAM04, MCAM08 |
| `20` | `282084003` | `14` | `96000035` | `0.663823` | MCAM04, MCAM03, MCAM08 |

## Diagnosis

The current bottleneck is not a single missing bridge.  The largest errors are
multi-camera identity components where one dominant GT is mixed with many small
impostor fragments, especially in MCAM04 and MCAM08.  The top false split GT
IDs also spread across many predicted components, so a naive merge repair will
increase false merges unless it first quarantines impostor fragments.

This explains why old queue repairs failed:

- large portfolio merges amplify false merges;
- small single-edge moves do not move enough of the dominant split mass;
- density filtering helps, but it filters detections rather than changing the
  identity graph.

## Next Candidate Generator

Build a current-k3 namespace proposer with three stages:

1. For each high false-merge predicted gid, split out low-support minority
   fragments using no-GT features: camera, time, local trajectory, bbox quality,
   clothing/body verifier disagreement, and density anomaly.
2. For each high false-split GT-like pattern, propose only tiny fragment
   reassignment into the dominant component when the opponent finds no same-video
   temporal contradiction.
3. Apply density/detector-quality quarantine before materializing any identity
   edit.  The output should be a committed/provisional split internally, then a
   forced benchmark assignment only after quarantine.

The immediate target slice is MCAM04 plus MCAM08 for predicted gids
`96000035`, `96000048`, `96000021`, and `96000026`, with GT ids used only as
post-hoc diagnostics in this report.
