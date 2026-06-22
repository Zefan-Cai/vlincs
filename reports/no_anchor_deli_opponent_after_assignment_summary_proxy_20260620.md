# Deli Opponent Verdict

- verdict: `pivot`
- pass_joint: `false`
- stale_count: `7`
- standing best full IDF1: `0.655240`
- best gated full IDF1: `0.654009`
- pair winner rejected by opponent: `true`

## Blockers
- pair/global metric passes but full DS1 IDF1 is below target
- best newly gated full IDF1 does not beat standing promoted artifact
- stale_count requires a structural pivot, not another threshold-only sweep

## Next Actions
- recover_canonical_fullscore_then_score_consensus_or_build_assignment_summary_proxy_without_gt_leakage
- run edge-rank target-fragment launcher when Pluto/SSH recovers
- spend full-score budget only on diverse unscored candidates
- reject pair-only winners unless full DS1 IDF1/HOTA improves
