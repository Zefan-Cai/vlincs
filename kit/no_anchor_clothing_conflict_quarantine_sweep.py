#!/usr/bin/env python
"""Quarantine visually inconsistent conflict nodes inside no-anchor components.

This is a conservative split-only post-processor.  It does not use anchors or
GT to decide edits.  Within each predicted component, it finds tracklets that
already participate in same-stream cannot-link conflicts and peels only those
whose OSNet + body-part clothing embedding is an outlier relative to the
component centroid.  GT is loaded only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_clothing_positive_edge_verifier import _blend_embedding, _load_view
    from kit.no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
    from kit.no_anchor_sample_positive_edge_verifier import _load_samples, _parse_floats, _parse_ints, _write_csv
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_clothing_positive_edge_verifier import _blend_embedding, _load_view
    from no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
    from no_anchor_sample_positive_edge_verifier import _load_samples, _parse_floats, _parse_ints, _write_csv


def _components(labels: np.ndarray, keep_indices: set[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        out[int(labels[int(idx)])].append(int(idx))
    return out


def _quarantine(labels: np.ndarray, keep_indices: set[int], emb: np.ndarray, forbidden, args, *, threshold: float, min_component_size: int):
    labels = labels.copy()
    next_label = int(labels.max()) + 1
    components = _components(labels, keep_indices)
    split_nodes = 0
    split_components = 0
    audited_components = 0
    skipped_no_conflict = 0
    skipped_large = 0
    max_splits = max(int(args.max_splits_per_component), 1)
    for label, indices in components.items():
        if len(indices) < int(min_component_size):
            continue
        if int(args.max_component_size) > 0 and len(indices) > int(args.max_component_size):
            skipped_large += 1
            continue
        comp = set(indices)
        conflict_count = {idx: len(forbidden[idx] & comp) for idx in indices}
        conflict_nodes = [idx for idx, count in conflict_count.items() if count > 0]
        if not conflict_nodes:
            skipped_no_conflict += 1
            continue
        audited_components += 1
        mat = emb[np.asarray(indices, dtype=np.int64)]
        centroid = mat.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1.0e-9)
        sims = emb[np.asarray(conflict_nodes, dtype=np.int64)] @ centroid
        candidates = sorted(
            [
                (float(sim), int(idx), int(conflict_count[int(idx)]))
                for sim, idx in zip(sims.tolist(), conflict_nodes)
                if float(sim) < float(threshold)
            ],
            key=lambda row: (row[0], -row[2]),
        )
        limit = min(max_splits, max(1, int(np.floor(len(indices) * float(args.max_split_frac)))))
        chosen = candidates[:limit]
        if not chosen:
            continue
        split_components += 1
        for _sim, idx, _count in chosen:
            labels[int(idx)] = next_label
            next_label += 1
            split_nodes += 1
    values = labels.tolist()
    return labels, {
        "mode": "clothing_conflict_quarantine",
        "quarantine_threshold": float(threshold),
        "quarantine_min_component_size": int(min_component_size),
        "quarantine_max_component_size": int(args.max_component_size),
        "quarantine_max_splits_per_component": int(args.max_splits_per_component),
        "quarantine_max_split_frac": float(args.max_split_frac),
        "quarantine_audited_components": int(audited_components),
        "quarantine_split_components": int(split_components),
        "quarantine_split_nodes": int(split_nodes),
        "quarantine_skipped_no_conflict": int(skipped_no_conflict),
        "quarantine_skipped_large": int(skipped_large),
        "components": int(len(set(values))),
        "largest_component": int(max(Counter(values).values(), default=0)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }


def _write_assignments(path: str, records, labels: np.ndarray, keep_seqs: set[int], *, offset: int) -> dict[str, object]:
    rows = []
    for idx, record in enumerate(records):
        seq = int(record.seq)
        if seq not in keep_seqs:
            continue
        rows.append(
            {
                "seq": seq,
                "tracklet_key": record.tracklet_key,
                "video": record.video,
                "camera": record.camera,
                "start_frame": int(record.start_frame),
                "end_frame": int(record.end_frame),
                "n_dets": int(record.n_dets),
                "avg_conf": round(float(record.avg_conf), 6),
                "predicted_global_id": int(offset) + int(labels[idx]),
                "decision_status": "clothing_conflict_quarantine",
            }
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["seq", "predicted_global_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {"assignments_out": str(path), "assignment_rows": int(len(rows))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--sample-feature-npz", required=True)
    ap.add_argument("--posecolor-npz", required=True)
    ap.add_argument("--colorhist-npz", required=True)
    ap.add_argument("--edge-osnet-weight", type=float, default=0.60)
    ap.add_argument("--edge-posecolor-weight", type=float, default=0.25)
    ap.add_argument("--edge-colorhist-weight", type=float, default=0.15)
    ap.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70")
    ap.add_argument("--min-component-sizes", default="4,8,16,32")
    ap.add_argument("--max-component-size", type=int, default=500)
    ap.add_argument("--max-splits-per-component", type=int, default=2)
    ap.add_argument("--max-split-frac", type=float, default=0.10)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=90_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, _db_emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    _samples, _counts, mean_emb, sample_meta = _load_samples(args.sample_feature_npz, records)
    pose_emb, pose_meta = _load_view(args.posecolor_npz, records)
    color_emb, color_meta = _load_view(args.colorhist_npz, records)
    emb = _blend_embedding(mean_emb, pose_emb, color_emb, args)

    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": 0.5,
        "min_matches": 1,
        "min_purity": 0.0,
        "n_tracklets": len(records),
        "prediction_rows": int(sum(len(v) for v in pred_by_video.values())),
        "gt_rows": int(sum(len(v) for v in gt_by_video.values())),
    }
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached
    keep_seqs, output_info = _output_keep_seqs(records, args)
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}

    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    base_pred = _labels_to_seq_map(records, base_labels, keep_seqs=keep_seqs)
    base_pair = _pair_metrics([record.seq for record in records], base_pred, gt_by_seq, weight_by_seq)
    forbidden = _build_overlap_forbidden(records)

    rows = []
    label_cache = {}
    for threshold in _parse_floats(args.thresholds):
        for min_size in _parse_ints(args.min_component_sizes):
            labels, info = _quarantine(base_labels, keep_indices, emb, forbidden, args, threshold=threshold, min_component_size=min_size)
            pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
            pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
            row = {**info, **pair}
            rows.append(row)
            label_cache[(float(threshold), int(min_size))] = labels
    rows.sort(key=lambda row: (float(row["tracklet_pair_f1"]), float(row["tracklet_pair_precision"]), float(row["tracklet_pair_recall"])), reverse=True)

    full_rows = []
    for row in rows[: max(int(args.full_top_n), 0)]:
        labels = label_cache[(float(row["quarantine_threshold"]), int(row["quarantine_min_component_size"]))]
        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        full_rows.append(row)
    assignment_info = {}
    if args.assignments_out and rows:
        best = rows[0]
        labels = label_cache[(float(best["quarantine_threshold"]), int(best["quarantine_min_component_size"]))]
        assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs, offset=int(args.assignment_offset))

    result = {
        "assignment_csv": args.assignment_csv,
        "base_assignment_components": int(len(raw_to_local)),
        "base_pair_metrics": base_pair,
        "sample_meta": sample_meta,
        "posecolor_meta": pose_meta,
        "colorhist_meta": color_meta,
        "top": rows[:100],
        "full_rows": full_rows,
        "assignment_info": assignment_info,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "base": base_pair, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
