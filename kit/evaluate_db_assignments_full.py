#!/usr/bin/env python
"""Evaluate/export a DB-tracklet assignment CSV with the canonical scorer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.evaluate_sample_assignments_full import _export_zip, _metric_dict
from kit.no_anchor_resolve_sweep import _build_comp, _connect, _load_predictions
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


def _load_assignment_csv(path: str, pred_col: str) -> tuple[dict[int, int], Counter[str]]:
    pred: dict[int, int] = {}
    statuses: Counter[str] = Counter()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"seq", pred_col} - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        status_col = "resolution_status" if "resolution_status" in fields else "decision_status" if "decision_status" in fields else ""
        for row in reader:
            seq = int(float(row["seq"]))
            pred[seq] = int(float(row[pred_col]))
            if status_col:
                statuses[str(row.get(status_col, ""))] += 1
    return pred, statuses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--json", required=True)
    ap.add_argument("--zip-out", default="")
    args = ap.parse_args()

    pred_by_seq, statuses = _load_assignment_csv(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    pred_by_video = _load_predictions(con)
    comp = _build_comp(pred_by_video, pred_by_seq)
    gt_by_video = load_ds1_gt_by_video()
    keys = sorted(set(gt_by_video).intersection(comp))
    metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    out = {
        "assignment_csv": str(args.assignment_csv),
        "assigned_tracklets": int(len(pred_by_seq)),
        "assignment_status_counts": dict(statuses),
        "videos_scored": keys,
        **_metric_dict(metrics),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    if args.zip_out:
        out.update(_export_zip(comp, args.zip_out))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
