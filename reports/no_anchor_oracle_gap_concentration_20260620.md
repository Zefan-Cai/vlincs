# No-Anchor Oracle Gap Concentration

Base full IDF1: `0.653071`
Oracle full IDF1: `0.706202`
Missing true-pair mass: `9105079781`
Base predicted pair mass: `29945695165`

This is eval-only analysis over an oracle decomposition artifact. It is
used to decide which no-anchor production branch should be tested next.

## False-Split Concentration

| top N | cumulative missing-mass coverage | cumulative false-split mass |
| ---: | ---: | ---: |
| 1 | 0.115 | 1046536325 |
| 3 | 0.263 | 2397394387 |
| 5 | 0.392 | 3568215874 |
| 10 | 0.568 | 5167619465 |
| 20 | 0.809 | 7366722631 |
| 30 | 0.958 | 8721294723 |

## Top False-Split GT IDs

| rank | gt id | false-split mass | cumulative coverage | components | dominant prediction | dominant frac |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9 | 1046536325 | 0.115 | 26 | 70000001 | 0.701911 |
| 2 | 36 | 678199057 | 0.189 | 16 | 70000074 | 0.657854 |
| 3 | 11 | 672659005 | 0.263 | 25 | 70000076 | 0.786685 |
| 4 | 43 | 595698820 | 0.329 | 20 | 70000024 | 0.649796 |
| 5 | 52 | 575122667 | 0.392 | 22 | 70000007 | 0.703232 |
| 6 | 31 | 452595928 | 0.442 | 18 | 70000032 | 0.745779 |
| 7 | 24 | 321191884 | 0.477 | 19 | 70000027 | 0.754835 |
| 8 | 20 | 282084003 | 0.508 | 14 | 70000074 | 0.663823 |
| 9 | 15 | 277863577 | 0.538 | 19 | 70000190 | 0.858255 |
| 10 | 14 | 265668199 | 0.568 | 15 | 70000155 | 0.761503 |
| 11 | 51 | 254391997 | 0.595 | 16 | 70000101 | 0.72602 |
| 12 | 23 | 254053608 | 0.623 | 18 | 70000004 | 0.893477 |
| 13 | 10 | 232756229 | 0.649 | 16 | 70000005 | 0.83955 |
| 14 | 50 | 229521507 | 0.674 | 18 | 70000002 | 0.869647 |
| 15 | 4 | 215732416 | 0.698 | 17 | 70000068 | 0.895963 |
| 16 | 12 | 211395223 | 0.721 | 20 | 70000016 | 0.811404 |
| 17 | 37 | 208226287 | 0.744 | 12 | 70000039 | 0.916743 |
| 18 | 54 | 205045596 | 0.766 | 11 | 70000021 | 0.900739 |
| 19 | 7 | 196568788 | 0.788 | 12 | 70000015 | 0.909733 |
| 20 | 29 | 191411515 | 0.809 | 16 | 70000010 | 0.822075 |

## False-Merge Concentration

| top N | cumulative base-pred-pair coverage | cumulative false-merge mass |
| ---: | ---: | ---: |
| 1 | 0.026 | 790173339 |
| 3 | 0.049 | 1476097495 |
| 5 | 0.068 | 2039792684 |
| 10 | 0.103 | 3085952551 |
| 20 | 0.147 | 4411638719 |
| 30 | 0.172 | 5138722241 |

## Interpretation

- The false-split gap is highly concentrated: the top few GT identities carry
  a large fraction of the recoverable true-pair mass.
- This supports high-mass component bridge/split selection over further
  tiny-fragment threshold sweeps.
