#!/usr/bin/env python
"""Export the best no-anchor time-agglom global-ID model assignments.

This turns a resolver configuration into a production-shaped tracklet table:
seq -> predicted_global_id, confidence, decision_status, component metadata.
No anchors or GT identities are used to make predictions.  GT can optionally be
loaded after prediction for pair-metric reporting only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _cache_eval_labels,
        _connect,
        _label_tracklets_for_eval,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _time_agglom_resolve,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _cache_eval_labels,
        _connect,
        _label_tracklets_for_eval,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _time_agglom_resolve,
        _with_detection_endpoints,
    )


def _component_confidence(records, emb, labels, keep_seqs: set[int], top_k: int) -> dict[int, dict[str, object]]:
    kept = [i for i, record in enumerate(records) if int(record.seq) in keep_seqs]
    by_label: dict[int, list[int]] = defaultdict(list)
    for i in kept:
        by_label[int(labels[i])].append(i)
    meta: dict[int, dict[str, object]] = {}
    if not kept:
        return meta

    x = emb.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    kept_arr = np.asarray(kept, dtype=np.int64)
    sim = x[kept_arr] @ x[kept_arr].T
    np.fill_diagonal(sim, -2.0)
    k = min(max(int(top_k), 1), max(len(kept) - 1, 1))

    internal_scores: dict[int, list[float]] = defaultdict(list)
    external_max: dict[int, float] = defaultdict(lambda: -1.0)
    for pos, i in enumerate(kept):
        row = sim[pos]
        top = np.argpartition(-row, k - 1)[:k]
        li = int(labels[i])
        for other_pos in top.tolist():
            score = float(row[int(other_pos)])
            j = int(kept_arr[int(other_pos)])
            lj = int(labels[j])
            if li == lj:
                internal_scores[li].append(score)
            else:
                external_max[li] = max(float(external_max[li]), score)

    for label, indices in by_label.items():
        size = int(len(indices))
        internals = internal_scores.get(label, [])
        internal_median = float(np.median(np.asarray(internals, dtype=np.float32))) if internals else 0.0
        internal_min = float(np.min(np.asarray(internals, dtype=np.float32))) if internals else 0.0
        ext = float(external_max.get(label, 0.0))
        margin = internal_median - ext
        if size <= 1:
            confidence = float(np.clip(0.25 * (1.0 - max(ext, 0.0)) + 0.05, 0.05, 0.35))
            status = "forced_singleton"
        else:
            margin_term = float(np.clip(0.5 + margin, 0.0, 1.0))
            confidence = float(np.clip(0.55 * internal_median + 0.35 * margin_term + 0.10 * min(size / 8.0, 1.0), 0.0, 1.0))
            if internal_median >= 0.72 and margin >= 0.08 and len(internals) >= max(1, size - 1):
                status = "committed"
            elif internal_median >= 0.62 or confidence >= 0.65:
                status = "provisional"
            else:
                status = "forced_component"
        meta[label] = {
            "component_size": size,
            "internal_edges": int(len(internals)),
            "internal_score_median": round(internal_median, 6),
            "internal_score_min": round(internal_min, 6),
            "external_score_max": round(ext, 6),
            "margin": round(margin, 6),
            "confidence": round(confidence, 6),
            "decision_status": status,
        }
    return meta


def _write_assignments(path: str, records, labels, keep_seqs: set[int], meta: dict[int, dict[str, object]], offset: int) -> dict[str, object]:
    rows = []
    counts = Counter(int(labels[i]) for i, record in enumerate(records) if int(record.seq) in keep_seqs)
    for i, record in enumerate(records):
        if int(record.seq) not in keep_seqs:
            continue
        label = int(labels[i])
        m = meta.get(label, {})
        rows.append(
            {
                "seq": int(record.seq),
                "tracklet_key": record.tracklet_key,
                "video": record.video,
                "camera": record.camera,
                "start_frame": int(record.start_frame),
                "end_frame": int(record.end_frame),
                "n_dets": int(record.n_dets),
                "avg_conf": round(float(record.avg_conf), 6),
                "predicted_global_id": int(offset + label),
                "component_label": label,
                "component_size": int(counts[label]),
                "prediction_confidence": m.get("confidence", 0.15),
                "decision_status": m.get("decision_status", "forced_component"),
                "component_internal_edges": m.get("internal_edges", 0),
                "component_internal_score_median": m.get("internal_score_median", 0.0),
                "component_external_score_max": m.get("external_score_max", 0.0),
                "component_margin": m.get("margin", 0.0),
            }
        )
    fieldnames = list(rows[0]) if rows else [
        "seq",
        "tracklet_key",
        "video",
        "camera",
        "start_frame",
        "end_frame",
        "n_dets",
        "avg_conf",
        "predicted_global_id",
        "component_label",
        "component_size",
        "prediction_confidence",
        "decision_status",
        "component_internal_edges",
        "component_internal_score_median",
        "component_external_score_max",
        "component_margin",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "assignments_out": str(out),
        "assignment_rows": int(len(rows)),
        "assignment_components": int(len(counts)),
        "largest_assignment_component": int(max(counts.values(), default=0)),
        "assignment_status_counts": dict(Counter(str(row["decision_status"]) for row in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.0155)
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
    ap.add_argument("--assignment-offset", type=int, default=30_000_000)
    ap.add_argument("--assignments-out", required=True)
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
    keep_seqs, output_info = _output_keep_seqs(records, args)

    cfg = ResolveConfig(
        mode="time_agglom",
        theta=float(args.theta),
        top_k=int(args.top_k),
        min_dets=int(args.min_dets),
        exclude_same=str(args.exclude_same),
        temporal_bonus=float(args.temporal_bonus),
        time_window_ms=int(args.time_window_ms),
    )
    labels, resolve_info = _time_agglom_resolve(records, emb, cfg)
    meta = _component_confidence(records, emb, labels, keep_seqs, int(args.top_k))
    assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs, meta, int(args.assignment_offset))

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
    cached = _load_eval_label_cache(args.eval_cache, expected) if args.eval_cache else None
    if cached is None:
        gt_by_seq, weight_by_seq, eval_stats = _label_tracklets_for_eval(
            pred_by_video,
            gt_by_video,
            iou_thr=0.5,
            min_matches=1,
            min_purity=0.0,
        )
        eval_stats.update(expected)
        if args.eval_cache:
            _cache_eval_labels(args.eval_cache, gt_by_seq, weight_by_seq, eval_stats)
    else:
        gt_by_seq, weight_by_seq, eval_stats = cached
    pair_metrics = _pair_metrics(
        [record.seq for record in records],
        _labels_to_seq_map(records, labels, keep_seqs=keep_seqs),
        gt_by_seq,
        weight_by_seq,
    )
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
        "pair_metrics": pair_metrics,
        **assignment_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
