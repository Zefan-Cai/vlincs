#!/usr/bin/env python
"""Full-score upper bounds for current predicted tracklets.

This is diagnostic only: it uses GT labels from the eval cache to quantify the
best possible full score if every current tracklet got its GT-majority identity.
It must not be used as a no-anchor model result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _score_full,
        _with_detection_endpoints,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--json", default="/mnt/localssd/vlincs_reid_runs/oracle_current_tracklets_full_upper_20260617.json")
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, _emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": 0.5,
        "min_matches": 1,
        "min_purity": 0.0,
        "n_tracklets": len(records),
        "prediction_rows": int(sum(len(value) for value in pred_by_video.values())),
        "gt_rows": int(sum(len(value) for value in gt_by_video.values())),
    }
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, _weight_by_seq, eval_stats = cached

    oracle_map = {int(seq): int(gid) for seq, gid in gt_by_seq.items()}
    unique_map = {int(record.seq): 90_000_000 + idx for idx, record in enumerate(records)}
    oracle_full = _score_full(pred_by_video, gt_by_video, oracle_map)
    unique_full = _score_full(pred_by_video, gt_by_video, unique_map)
    result = {
        "dbname": args.dbname,
        "role": args.role,
        "n_records": int(len(records)),
        "n_gt_labeled_seqs": int(len(gt_by_seq)),
        "eval_stats": eval_stats,
        "oracle_tracklet_majority_full": oracle_full,
        "unique_tracklet_full": unique_full,
        "uses_gt_for_analysis_only": True,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "json": str(out),
                "oracle_idf1": oracle_full["idf1"],
                "oracle_hota": oracle_full["hota"],
                "oracle_assa": oracle_full["assa"],
                "oracle_detpr": oracle_full["detpr"],
                "oracle_detre": oracle_full["detre"],
                "unique_idf1": unique_full["idf1"],
                "unique_hota": unique_full["hota"],
                "unique_assa": unique_full["assa"],
                "eval_labeled": int(len(gt_by_seq)),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
