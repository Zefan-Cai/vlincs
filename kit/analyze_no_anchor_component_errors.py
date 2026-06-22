#!/usr/bin/env python
"""Diagnose no-anchor tracklet component errors against eval-only GT labels.

This script reconstructs one resolver configuration, then reports where the
weighted tracklet-pair precision and recall are being lost.  It uses GT labels
only after prediction for analysis; it does not train or choose identities.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _run_resolver,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _run_resolver,
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


def _summarize_pred_components(records, labels, keep_seqs, gt_by_seq, weight_by_seq, top_n: int):
    by_pred: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    seq_to_record = {int(record.seq): record for record in records}
    for record, label in zip(records, labels):
        seq = int(record.seq)
        if seq not in keep_seqs or seq not in gt_by_seq:
            continue
        by_pred[int(label)].append((int(gt_by_seq[seq]), float(weight_by_seq.get(seq, 1.0)), seq))

    rows = []
    for label, triples in by_pred.items():
        total = sum(weight for _gt, weight, _seq in triples)
        total2 = sum(weight * weight for _gt, weight, _seq in triples)
        pred_pair_mass = _comb2(total, total2)
        by_gt: dict[int, list[tuple[int, float]]] = defaultdict(list)
        videos = Counter()
        cameras = Counter()
        n_dets = 0
        area_sum = 0.0
        for gt, weight, seq in triples:
            by_gt[int(gt)].append((seq, weight))
            rec = seq_to_record[seq]
            videos[rec.video] += 1
            cameras[rec.camera] += 1
            n_dets += int(rec.n_dets)
            area_sum += float(rec.width) * float(rec.height)
        true_pair_mass = sum(_bucket_pair_mass([(seq, weight) for seq, weight in values]) for values in by_gt.values())
        dominant_gt, dominant_weight = max(
            ((gt, sum(weight for _seq, weight in values)) for gt, values in by_gt.items()),
            key=lambda item: item[1],
        )
        rows.append(
            {
                "component_label": int(label),
                "tracklets": int(len(triples)),
                "weight": round(float(total), 3),
                "pred_pair_mass": round(float(pred_pair_mass), 3),
                "true_pair_mass": round(float(true_pair_mass), 3),
                "false_merge_mass": round(float(pred_pair_mass - true_pair_mass), 3),
                "dominant_gt": int(dominant_gt),
                "dominant_gt_weight_frac": round(float(dominant_weight / max(total, 1e-9)), 6),
                "gt_count": int(len(by_gt)),
                "videos": _top_items(videos, 5),
                "cameras": _top_items(cameras, 5),
                "mean_n_dets": round(float(n_dets / max(len(triples), 1)), 3),
                "mean_area": round(float(area_sum / max(len(triples), 1)), 3),
            }
        )
    rows.sort(key=lambda row: (float(row["false_merge_mass"]), float(row["pred_pair_mass"])), reverse=True)
    return rows[:top_n]


def _summarize_gt_splits(records, labels, keep_seqs, gt_by_seq, weight_by_seq, top_n: int):
    by_gt: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    seq_to_record = {int(record.seq): record for record in records}
    for record, label in zip(records, labels):
        seq = int(record.seq)
        if seq not in keep_seqs or seq not in gt_by_seq:
            continue
        by_gt[int(gt_by_seq[seq])].append((int(label), float(weight_by_seq.get(seq, 1.0)), seq))

    rows = []
    for gt, triples in by_gt.items():
        total = sum(weight for _label, weight, _seq in triples)
        total2 = sum(weight * weight for _label, weight, _seq in triples)
        gt_pair_mass = _comb2(total, total2)
        by_pred: dict[int, list[tuple[int, float]]] = defaultdict(list)
        videos = Counter()
        cameras = Counter()
        for label, weight, seq in triples:
            by_pred[int(label)].append((seq, weight))
            rec = seq_to_record[seq]
            videos[rec.video] += 1
            cameras[rec.camera] += 1
        true_pair_mass = sum(_bucket_pair_mass([(seq, weight) for seq, weight in values]) for values in by_pred.values())
        dominant_pred, dominant_weight = max(
            ((label, sum(weight for _seq, weight in values)) for label, values in by_pred.items()),
            key=lambda item: item[1],
        )
        pred_parts = Counter({label: round(sum(weight for _seq, weight in values), 3) for label, values in by_pred.items()})
        rows.append(
            {
                "gt_id": int(gt),
                "tracklets": int(len(triples)),
                "weight": round(float(total), 3),
                "gt_pair_mass": round(float(gt_pair_mass), 3),
                "true_pair_mass": round(float(true_pair_mass), 3),
                "false_split_mass": round(float(gt_pair_mass - true_pair_mass), 3),
                "pred_component_count": int(len(by_pred)),
                "dominant_component": int(dominant_pred),
                "dominant_component_weight_frac": round(float(dominant_weight / max(total, 1e-9)), 6),
                "pred_components": _top_items(pred_parts, 8),
                "videos": _top_items(videos, 8),
                "cameras": _top_items(cameras, 8),
            }
        )
    rows.sort(key=lambda row: (float(row["false_split_mass"]), float(row["gt_pair_mass"])), reverse=True)
    return rows[:top_n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--mode", default="time_agglom")
    ap.add_argument("--theta", type=float, default=0.014)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--exclude-same", default="camera")
    ap.add_argument("--temporal-bonus", type=float, default=0.005)
    ap.add_argument("--time-window-ms", type=int, default=1000)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, emb = _load_tracklets(con, args.role)
    if args.feature_npz:
        emb = _load_feature_npz(
            args.feature_npz,
            records,
            emb,
            concat_db=bool(args.concat_db_embedding),
            db_weight=float(args.db_weight),
            feature_weight=float(args.feature_weight),
        )
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

    cfg = ResolveConfig(
        mode=str(args.mode),
        theta=float(args.theta),
        top_k=int(args.top_k),
        min_dets=int(args.min_dets),
        exclude_same=str(args.exclude_same),
        temporal_bonus=float(args.temporal_bonus),
        time_window_ms=int(args.time_window_ms),
    )
    labels, resolve_info = _run_resolver(records, emb, cfg, graph_cache=None)
    pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
    pair = _pair_metrics([record.seq for record in records], pred_by_seq, gt_by_seq, weight_by_seq)
    full = _score_full(pred_by_video, gt_by_video, pred_by_seq) if args.full else None

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "resolve_config": cfg.__dict__,
        "resolve_info": resolve_info,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "pair_metrics": pair,
        "full": full,
        "top_false_merge_components": _summarize_pred_components(
            records, labels, keep_seqs, gt_by_seq, weight_by_seq, int(args.top_n)
        ),
        "top_false_split_gt_ids": _summarize_gt_splits(
            records, labels, keep_seqs, gt_by_seq, weight_by_seq, int(args.top_n)
        ),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "json": str(out),
                **pair,
                "top_false_merge_mass": result["top_false_merge_components"][0]["false_merge_mass"]
                if result["top_false_merge_components"]
                else 0.0,
                "top_false_split_mass": result["top_false_split_gt_ids"][0]["false_split_mass"]
                if result["top_false_split_gt_ids"]
                else 0.0,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
