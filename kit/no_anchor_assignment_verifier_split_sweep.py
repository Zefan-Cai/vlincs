#!/usr/bin/env python
"""Split an existing no-anchor assignment with a pseudo-trained pair verifier.

The input assignment is treated as the current forced-delivery identity output.
This script does not retrieve or merge new identities. It only audits edges
inside each delivered component using a no-anchor pair model, then splits weakly
connected components. Ground truth is used only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    import kit.no_anchor_global_id_model as gid_model
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    import no_anchor_global_id_model as gid_model
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _component_members(labels: np.ndarray, keep_indices: set[int]) -> list[list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        by_label[int(labels[idx])].append(int(idx))
    return list(by_label.values())


class _LocalDSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _pair_probability_edges(records, emb, model, pair_feature_views, members: list[list[int]], *, max_size: int):
    x = emb.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)
    forbidden = _build_overlap_forbidden(records)
    out: list[dict[str, object]] = []
    summary = {
        "audited_components": 0,
        "skipped_large_components": 0,
        "evaluated_pairs": 0,
        "forbidden_pairs": 0,
    }
    for component_idx, indices in enumerate(members):
        if len(indices) > int(max_size):
            summary["skipped_large_components"] += 1
            continue
        summary["audited_components"] += 1
        pair_refs: list[tuple[int, int, float, bool]] = []
        feature_rows: list[list[float]] = []
        for left_pos in range(len(indices)):
            i = int(indices[left_pos])
            for right_pos in range(left_pos + 1, len(indices)):
                j = int(indices[right_pos])
                is_forbidden = j in forbidden[i]
                if is_forbidden:
                    summary["forbidden_pairs"] += 1
                visual = float(np.dot(x[i], x[j]))
                extra = gid_model._pair_view_features(pair_feature_views, i, j)
                feature_rows.append(gid_model._pair_feature(records[i], records[j], visual, 1000, extra))
                pair_refs.append((left_pos, right_pos, visual, is_forbidden))
        if not pair_refs:
            continue
        prob = model.predict_proba(np.asarray(feature_rows, dtype=np.float32))[:, 1].astype(np.float32)
        summary["evaluated_pairs"] += int(len(pair_refs))
        out.append(
            {
                "component_idx": int(component_idx),
                "indices": indices,
                "pairs": pair_refs,
                "probabilities": prob,
            }
        )
    return out, summary


def _split_by_threshold(base_labels: np.ndarray, edge_blocks, *, threshold: float, split_min_size: int):
    labels = base_labels.copy()
    next_label = int(labels.max()) + 1
    info = {
        "verifier_threshold": float(threshold),
        "split_min_size": int(split_min_size),
        "split_components": 0,
        "split_produced_parts": 0,
        "split_singleton_parts": 0,
        "kept_internal_edges": 0,
        "blocked_forbidden_edges": 0,
    }
    for block in edge_blocks:
        indices = list(block["indices"])
        if len(indices) < int(split_min_size):
            continue
        dsu = _LocalDSU(len(indices))
        kept = 0
        blocked = 0
        for (left_pos, right_pos, _visual, is_forbidden), prob in zip(block["pairs"], block["probabilities"]):
            if bool(is_forbidden):
                blocked += 1
                continue
            if float(prob) >= float(threshold):
                dsu.union(int(left_pos), int(right_pos))
                kept += 1
        groups: dict[int, list[int]] = defaultdict(list)
        for local_pos, idx in enumerate(indices):
            groups[dsu.find(local_pos)].append(idx)
        if len(groups) <= 1:
            continue
        info["split_components"] += 1
        info["split_produced_parts"] += int(len(groups))
        info["split_singleton_parts"] += int(sum(1 for group in groups.values() if len(group) == 1))
        info["kept_internal_edges"] += int(kept)
        info["blocked_forbidden_edges"] += int(blocked)
        for group in groups.values():
            for idx in group:
                labels[int(idx)] = next_label
            next_label += 1
    keep_labels = labels.tolist()
    info["components"] = int(len(set(keep_labels)))
    info["largest_component"] = int(max(Counter(keep_labels).values(), default=0))
    return labels, info


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list, tuple))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_assignments(path: str, records, labels: np.ndarray, keep_seqs: set[int], *, offset: int) -> dict[str, object]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
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
                "decision_status": "verifier_split_component",
            }
        )
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["seq", "predicted_global_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {"assignments_out": str(out), "assignment_rows": int(len(rows))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--pair-feature-npz", action="append", default=[])
    ap.add_argument("--load-model", required=True)
    ap.add_argument("--thresholds", default="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    ap.add_argument("--split-min-sizes", default="8,16,32,64")
    ap.add_argument("--audit-max-component-size", type=int, default=400)
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
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=70_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    emb = _load_feature_npz(
        args.feature_npz,
        records,
        db_emb,
        concat_db=bool(args.concat_db_embedding),
        db_weight=float(args.db_weight),
        feature_weight=float(args.feature_weight),
    )
    pair_feature_views, pair_feature_names, pair_feature_meta = gid_model._load_pair_feature_views(
        list(args.pair_feature_npz), records
    )
    gid_model.FEATURE_NAMES = list(gid_model.BASE_FEATURE_NAMES) + list(pair_feature_names)
    model = joblib.load(args.load_model)["model"]

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
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}

    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "assignment_components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    members = _component_members(base_labels, keep_indices)
    edge_blocks, audit_info = _pair_probability_edges(
        records,
        emb,
        model,
        pair_feature_views,
        members,
        max_size=int(args.audit_max_component_size),
    )
    print(json.dumps({"stage": "audited_internal_edges", **audit_info}, sort_keys=True), flush=True)

    rows: list[dict[str, object]] = []
    label_cache: dict[tuple[float, int], np.ndarray] = {}
    for split_min_size in _parse_ints(args.split_min_sizes):
        for threshold in _parse_floats(args.thresholds):
            labels, split_info = _split_by_threshold(
                base_labels,
                edge_blocks,
                threshold=float(threshold),
                split_min_size=int(split_min_size),
            )
            pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
            pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
            row = {
                "mode": "assignment_verifier_split",
                "assignment_csv": str(args.assignment_csv),
                **split_info,
                **pair,
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
                "uses_gt_for_evaluation_only": True,
            }
            rows.append(row)
            label_cache[(float(threshold), int(split_min_size))] = labels
            print(json.dumps({"stage": "pair", **row}, sort_keys=True), flush=True)

    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_precision"]),
            float(row["tracklet_pair_recall"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        labels = label_cache[(float(row["verifier_threshold"]), int(row["split_min_size"]))]
        pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        best = rows[0]
        labels = label_cache[(float(best["verifier_threshold"]), int(best["split_min_size"]))]
        assignment_info = _write_assignments(
            args.assignments_out,
            records,
            labels,
            keep_seqs,
            offset=int(args.assignment_offset),
        )
        rows[0].update(assignment_info)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "assignment_csv": str(args.assignment_csv),
        "feature_npz": str(args.feature_npz),
        "pair_feature_views": pair_feature_meta,
        "load_model": str(args.load_model),
        "feature_names": gid_model.FEATURE_NAMES,
        "base_pair_metrics": base_pair,
        "eval_stats": eval_stats,
        "output_admission": output_info,
        "audit_info": audit_info,
        "n_configs": len(rows),
        "top": rows[: max(50, int(args.full_top_n))],
        "assignment_info": assignment_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"stage": "done", "json": str(out), "best": rows[0] if rows else None}, sort_keys=True))


if __name__ == "__main__":
    main()
