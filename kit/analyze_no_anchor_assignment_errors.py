#!/usr/bin/env python
"""Diagnose an exported no-anchor assignment CSV against eval-only GT labels.

This script does not construct identities or use GT for model selection.  It
reads a delivered seq -> predicted_global_id CSV and reports where pair recall
and precision are lost, so we can target the next no-anchor resolver change.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
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
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _comb2(sum_w: float, sum_w2: float) -> float:
    return max((sum_w * sum_w - sum_w2) / 2.0, 0.0)


def _bucket_pair_mass(items: list[tuple[int, float]]) -> float:
    total = sum(weight for _key, weight in items)
    total2 = sum(weight * weight for _key, weight in items)
    return _comb2(total, total2)


def _top_items(counter: Counter, n: int = 8) -> list[dict[str, object]]:
    return [{"key": str(key), "value": float(value)} for key, value in counter.most_common(n)]


def _load_assignment_csv(path: str, pred_col: str) -> tuple[dict[int, int], dict[int, dict[str, str]]]:
    pred: dict[int, int] = {}
    rows: dict[int, dict[str, str]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "seq" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing a seq column")
        if pred_col not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing prediction column {pred_col!r}")
        for row in reader:
            seq = int(row["seq"])
            pred[seq] = int(float(row[pred_col]))
            rows[seq] = dict(row)
    return pred, rows


def _summarize_pred_components(records, pred_by_seq, keep_seqs, gt_by_seq, weight_by_seq, top_n: int):
    by_pred: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    seq_to_record = {int(record.seq): record for record in records}
    for seq, pred in pred_by_seq.items():
        seq = int(seq)
        if seq not in keep_seqs or seq not in gt_by_seq or seq not in seq_to_record:
            continue
        by_pred[int(pred)].append((int(gt_by_seq[seq]), float(weight_by_seq.get(seq, 1.0)), seq))

    rows = []
    for pred, triples in by_pred.items():
        total = sum(weight for _gt, weight, _seq in triples)
        total2 = sum(weight * weight for _gt, weight, _seq in triples)
        pred_pair_mass = _comb2(total, total2)
        by_gt: dict[int, list[tuple[int, float]]] = defaultdict(list)
        videos = Counter()
        cameras = Counter()
        time_span = [10**18, -10**18]
        for gt, weight, seq in triples:
            by_gt[int(gt)].append((seq, weight))
            rec = seq_to_record[seq]
            videos[rec.video] += 1
            cameras[rec.camera] += 1
            time_span[0] = min(time_span[0], int(rec.start_abs_ms))
            time_span[1] = max(time_span[1], int(rec.end_abs_ms))
        true_pair_mass = sum(_bucket_pair_mass([(seq, weight) for seq, weight in values]) for values in by_gt.values())
        dominant_gt, dominant_weight = max(
            ((gt, sum(weight for _seq, weight in values)) for gt, values in by_gt.items()),
            key=lambda item: item[1],
        )
        gt_parts = Counter({gt: round(sum(weight for _seq, weight in values), 3) for gt, values in by_gt.items()})
        rows.append(
            {
                "predicted_global_id": int(pred),
                "tracklets": int(len(triples)),
                "weight": round(float(total), 3),
                "pred_pair_mass": round(float(pred_pair_mass), 3),
                "true_pair_mass": round(float(true_pair_mass), 3),
                "false_merge_mass": round(float(pred_pair_mass - true_pair_mass), 3),
                "dominant_gt": int(dominant_gt),
                "dominant_gt_weight_frac": round(float(dominant_weight / max(total, 1e-9)), 6),
                "gt_count": int(len(by_gt)),
                "gt_parts": _top_items(gt_parts, 8),
                "videos": _top_items(videos, 8),
                "cameras": _top_items(cameras, 8),
                "start_abs_ms": int(time_span[0]) if time_span[0] < 10**18 else None,
                "end_abs_ms": int(time_span[1]) if time_span[1] > -10**18 else None,
            }
        )
    rows.sort(key=lambda row: (float(row["false_merge_mass"]), float(row["pred_pair_mass"])), reverse=True)
    return rows[:top_n]


def _summarize_gt_splits(records, pred_by_seq, keep_seqs, gt_by_seq, weight_by_seq, top_n: int):
    by_gt: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    seq_to_record = {int(record.seq): record for record in records}
    for seq, pred in pred_by_seq.items():
        seq = int(seq)
        if seq not in keep_seqs or seq not in gt_by_seq or seq not in seq_to_record:
            continue
        by_gt[int(gt_by_seq[seq])].append((int(pred), float(weight_by_seq.get(seq, 1.0)), seq))

    rows = []
    for gt, triples in by_gt.items():
        total = sum(weight for _pred, weight, _seq in triples)
        total2 = sum(weight * weight for _pred, weight, _seq in triples)
        gt_pair_mass = _comb2(total, total2)
        by_pred: dict[int, list[tuple[int, float]]] = defaultdict(list)
        videos = Counter()
        cameras = Counter()
        time_span = [10**18, -10**18]
        for pred, weight, seq in triples:
            by_pred[int(pred)].append((seq, weight))
            rec = seq_to_record[seq]
            videos[rec.video] += 1
            cameras[rec.camera] += 1
            time_span[0] = min(time_span[0], int(rec.start_abs_ms))
            time_span[1] = max(time_span[1], int(rec.end_abs_ms))
        true_pair_mass = sum(_bucket_pair_mass([(seq, weight) for seq, weight in values]) for values in by_pred.values())
        dominant_pred, dominant_weight = max(
            ((pred, sum(weight for _seq, weight in values)) for pred, values in by_pred.items()),
            key=lambda item: item[1],
        )
        pred_parts = Counter({pred: round(sum(weight for _seq, weight in values), 3) for pred, values in by_pred.items()})
        rows.append(
            {
                "gt_id": int(gt),
                "tracklets": int(len(triples)),
                "weight": round(float(total), 3),
                "gt_pair_mass": round(float(gt_pair_mass), 3),
                "true_pair_mass": round(float(true_pair_mass), 3),
                "false_split_mass": round(float(gt_pair_mass - true_pair_mass), 3),
                "pred_component_count": int(len(by_pred)),
                "dominant_prediction": int(dominant_pred),
                "dominant_prediction_weight_frac": round(float(dominant_weight / max(total, 1e-9)), 6),
                "pred_components": _top_items(pred_parts, 12),
                "videos": _top_items(videos, 8),
                "cameras": _top_items(cameras, 8),
                "start_abs_ms": int(time_span[0]) if time_span[0] < 10**18 else None,
                "end_abs_ms": int(time_span[1]) if time_span[1] > -10**18 else None,
            }
        )
    rows.sort(key=lambda row: (float(row["false_split_mass"]), float(row["gt_pair_mass"])), reverse=True)
    return rows[:top_n]


def _per_video_pair(records, pred_by_seq, gt_by_seq, weight_by_seq):
    rows = []
    by_video: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_video[str(record.video)].append(int(record.seq))
    for video, seqs in sorted(by_video.items()):
        metric = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
        metric["video"] = video
        rows.append(metric)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--output-drop-area-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-area-quantile-by-video", default="")
    ap.add_argument("--output-drop-quality-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-quality-quantile-by-video", default="")
    ap.add_argument("--output-auto-anomaly-admission", action="store_true")
    ap.add_argument("--output-auto-anomaly-metric", default="quality")
    ap.add_argument("--output-auto-anomaly-quantile", type=float, default=0.75)
    ap.add_argument("--output-auto-anomaly-area-ratio", type=float, default=0.60)
    ap.add_argument("--output-auto-anomaly-quality-mad", type=float, default=1.0)
    ap.add_argument("--output-auto-anomaly-min-video-tracklets", type=int, default=20)
    ap.add_argument("--output-auto-anomaly-max-videos", type=int, default=3)
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    pred_by_seq, assignment_rows = _load_assignment_csv(args.assignment_csv, args.pred_col)
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
    keep_seqs, output_info = _output_keep_seqs(records, args)
    pred_by_seq = {int(seq): int(pred) for seq, pred in pred_by_seq.items() if int(seq) in keep_seqs}

    all_seqs = [int(record.seq) for record in records]
    pair = _pair_metrics(all_seqs, pred_by_seq, gt_by_seq, weight_by_seq)
    full = _score_full(pred_by_video, gt_by_video, pred_by_seq) if args.full else None
    result = {
        "assignment_csv": args.assignment_csv,
        "pred_col": args.pred_col,
        "assignment_rows": int(len(assignment_rows)),
        "emitted_assignments_after_output_filter": int(len(pred_by_seq)),
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "pair_metrics": pair,
        "per_video_pair_metrics": _per_video_pair(records, pred_by_seq, gt_by_seq, weight_by_seq),
        "full": full,
        "top_false_merge_components": _summarize_pred_components(
            records, pred_by_seq, keep_seqs, gt_by_seq, weight_by_seq, int(args.top_n)
        ),
        "top_false_split_gt_ids": _summarize_gt_splits(
            records, pred_by_seq, keep_seqs, gt_by_seq, weight_by_seq, int(args.top_n)
        ),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"json": str(out), "pair": pair, "full_idf1": None if full is None else full.get("idf1")}, sort_keys=True))


if __name__ == "__main__":
    main()
