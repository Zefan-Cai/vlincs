# Deli Opponent Verdict

- verdict: `pivot`
- pass_joint: `false`
- stale_count: `8`
- standing best full IDF1: `0.655240`
- best gated full IDF1: `0.654009`
- pair winner rejected by opponent: `true`

## Blockers
- pair/global metric passes but full DS1 IDF1 is below target
- best newly gated full IDF1 does not beat standing promoted artifact
- stale_count requires a structural pivot, not another threshold-only sweep

## Next Actions
- structural_no_gt_false_split_solver_or_restore_canonical_scorer
- run edge-rank target-fragment launcher when Pluto/SSH recovers
- spend full-score budget only on diverse unscored candidates
- reject pair-only winners unless full DS1 IDF1/HOTA improves
