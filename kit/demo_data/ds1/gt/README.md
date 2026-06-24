# DS1 Dense GT Bundle

This folder contains the 10 DS1 dense ground-truth parquet files used by the
root `./demo.sh` replay evaluator.

It is not used as production identity evidence. The no-anchor pipeline first
materializes a submission from the committed assignment CSV and DS1 tracklets;
these GT parquet files are then used only to compute and verify IDF1/HOTA/AssA.

Fresh clones must materialize this folder through Git LFS:

```bash
git lfs pull --include="kit/demo_data/ds1/**"
./demo.sh
```

`checksums.sha256` records the expected file digests relative to this directory.
