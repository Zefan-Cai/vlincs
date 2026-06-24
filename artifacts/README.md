# VLINCS Artifacts

## DS1 Dense GT Data Root

`ds1_dense_gt_data_root_20260622.tar.gz` contains the 10 dense DS1 ground-truth
parquet files used to score the WISC no-anchor replay.

SHA256:

```text
cf9d78fd8728d690dfeba5b55fe09af709dc10af211eff2264269d00a88c6381  ds1_dense_gt_data_root_20260622.tar.gz
```

It is evaluation-only data. It is not used as training evidence, identity
anchors, or production resolver input. The replay first creates a submission
from tracklets plus the committed assignment CSV; these GT parquet files are
then used by the evaluator to compute IDF1/HOTA/AssA.

To unpack as a `DATA_ROOT`:

```bash
mkdir -p /tmp/vlincs_ds1_gt
tar -C /tmp/vlincs_ds1_gt -xzf artifacts/ds1_dense_gt_data_root_20260622.tar.gz
DATA_ROOT=/tmp/vlincs_ds1_gt ./demo.sh
```
