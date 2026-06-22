#!/usr/bin/env python
"""Evaluate the existing online assignment gid as a no-anchor baseline."""

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
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _load_online_gid(con) -> dict[int, int]:
    with con.cursor() as cur:
        cur.execute(
            """SELECT seq, gid, COUNT(*) AS n
               FROM assignments
               GROUP BY seq, gid
               ORDER BY seq, n DESC, gid"""
        )
        rows = cur.fetchall()
    out: dict[int, int] = {}
    for seq, gid, _n in rows:
        out.setdefault(int(seq), int(gid))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--json", default="/mnt/localssd/vlincs_reid_runs/online_gid_baseline_full_20260617.json")
    ap.add_argument("--skip-full", action="store_true")
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
    gt_by_seq, weight_by_seq, eval_stats = cached
    online = _load_online_gid(con)
    pair = _pair_metrics([record.seq for record in records], online, gt_by_seq, weight_by_seq)
    full = None if args.skip_full else _score_full(pred_by_video, gt_by_video, online)
    result = {
        "dbname": args.dbname,
        "role": args.role,
        "n_records": int(len(records)),
        "n_online": int(len(online)),
        "eval_stats": eval_stats,
        "online_gid_pair": pair,
        "online_gid_full": full,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {"json": str(out), **pair}
    if full:
        summary.update({"full_idf1": full["idf1"], "full_hota": full["hota"], "full_assa": full["assa"]})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
