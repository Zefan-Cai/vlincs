# DS1 No-Anchor Feature Artifacts

These NPZ files are production-side tracklet features used by the WISC DS1
method reproduction:

- `ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz`
- `ds1_tracklet_dinov2base_s1_20260620.npz`
- `ds1_tracklet_siglip2_person_reid_s1_20260620.npz`

They contain `seqs` and feature vectors aligned to the committed DS1 assignment
CSV.  They do not contain GT IDs, anchor labels, evaluator labels, or training
targets.  `./demo.sh` uses them to rerun the no-anchor feature-outlier proposer
before scoring.
