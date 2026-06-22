#!/usr/bin/env python
"""Relink conflicted component outliers using no-anchor feature evidence.

This is a lightweight proposer for the current no-anchor assignment regime.  It
does not use anchors or identity ground truth to select edits.  Candidate
tracklets must already sit inside a same-stream cannot-link conflict, must be
weakly supported by their current component centroid, and must have a stronger
feature match to another physically compatible component.  Ground truth is only
loaded after assignments are materialized, for pair/full diagnostics.
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
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
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
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
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


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _components(labels: np.ndarray, keep_indices: set[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        groups[int(labels[int(idx)])].append(int(idx))
    return dict(groups)


def _component_centroids(emb: np.ndarray, groups: dict[int, list[int]]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for label, indices in groups.items():
        if not indices:
            continue
        out[int(label)] = _l2n(emb[np.asarray(indices, dtype=np.int64)].mean(axis=0, keepdims=True))[0]
    return out


def _has_forbidden(idx: int, target_indices: list[int], forbidden: list[set[int]]) -> bool:
    blocked = forbidden[int(idx)]
    return any(int(other) in blocked for other in target_indices)


def _topk_support(emb: np.ndarray, idx: int, indices: list[int], *, k: int) -> dict[str, float | int]:
    others = [int(other) for other in indices if int(other) != int(idx)]
    if not others:
        return {"nn_max": -1.0, "nn_topk_mean": -1.0, "nn_count": 0}
    sims = emb[np.asarray(others, dtype=np.int64)] @ emb[int(idx)]
    sims = np.sort(np.asarray(sims, dtype=np.float32))[::-1]
    kk = max(1, min(int(k), int(len(sims))))
    return {
        "nn_max": float(sims[0]),
        "nn_topk_mean": float(np.mean(sims[:kk])),
        "nn_count": int(len(sims)),
    }


def _frame_gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    if int(a_end) < int(b_start):
        return int(b_start) - int(a_end)
    if int(b_end) < int(a_start):
        return int(a_start) - int(b_end)
    return 0


def _temporal_support(records, idx: int, indices: list[int], *, max_gap_frames: int) -> dict[str, int]:
    rec = records[int(idx)]
    support = 0
    overlap = 0
    for other in indices:
        other = int(other)
        if other == int(idx):
            continue
        item = records[other]
        if str(item.video) != str(rec.video):
            continue
        gap = _frame_gap(int(rec.start_frame), int(rec.end_frame), int(item.start_frame), int(item.end_frame))
        if gap == 0:
            overlap += 1
        if gap <= int(max_gap_frames):
            support += 1
    return {"temporal_support": int(support), "temporal_overlap": int(overlap)}


def _build_candidates(
    records,
    labels: np.ndarray,
    keep_indices: set[int],
    emb: np.ndarray,
    forbidden: list[set[int]],
    *,
    min_source_size: int,
    max_source_size: int,
    min_target_size: int,
    max_target_size: int,
    min_conflict_edges: int,
    max_source_sim: float,
    min_target_sim: float,
    min_margin: float,
    max_candidates_per_source: int,
    support_top_k: int,
    max_temporal_gap_frames: int,
    min_neighbor_margin: float,
    max_source_neighbor_sim: float,
    max_source_temporal_support: int,
    rank_mode: str,
) -> list[dict[str, object]]:
    groups = _components(labels, keep_indices)
    centroids = _component_centroids(emb, groups)
    candidates: list[dict[str, object]] = []
    for source_label, source_indices in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        source_size = len(source_indices)
        if source_size < int(min_source_size):
            continue
        if int(max_source_size) > 0 and source_size > int(max_source_size):
            continue
        source_set = set(source_indices)
        conflict_count = {idx: len(forbidden[int(idx)] & source_set) for idx in source_indices}
        conflict_nodes = [idx for idx, count in conflict_count.items() if count >= int(min_conflict_edges)]
        if not conflict_nodes:
            continue
        source_centroid = centroids[int(source_label)]
        source_rows: list[dict[str, object]] = []
        for idx in conflict_nodes:
            source_sim = float(emb[int(idx)] @ source_centroid)
            if source_sim > float(max_source_sim):
                continue
            source_support = _topk_support(emb, int(idx), source_indices, k=int(support_top_k))
            if float(source_support["nn_max"]) > float(max_source_neighbor_sim):
                continue
            source_time = _temporal_support(records, int(idx), source_indices, max_gap_frames=int(max_temporal_gap_frames))
            if int(source_time["temporal_support"]) > int(max_source_temporal_support):
                continue
            target_rows = []
            for target_label, target_indices in groups.items():
                target_label = int(target_label)
                if target_label == int(source_label):
                    continue
                target_size = len(target_indices)
                if target_size < int(min_target_size):
                    continue
                if int(max_target_size) > 0 and target_size > int(max_target_size):
                    continue
                if _has_forbidden(int(idx), target_indices, forbidden):
                    continue
                target_sim = float(emb[int(idx)] @ centroids[target_label])
                margin = target_sim - source_sim
                if target_sim < float(min_target_sim) or margin < float(min_margin):
                    continue
                target_support = _topk_support(emb, int(idx), target_indices, k=int(support_top_k))
                neighbor_margin = float(target_support["nn_topk_mean"]) - float(source_support["nn_topk_mean"])
                if neighbor_margin < float(min_neighbor_margin):
                    continue
                target_time = _temporal_support(records, int(idx), target_indices, max_gap_frames=int(max_temporal_gap_frames))
                target_rows.append((target_sim, margin, neighbor_margin, target_label, target_size, target_support, target_time))
            if not target_rows:
                continue
            if str(rank_mode) == "source_retention":
                target_key = lambda row: (row[2], row[1], row[0])
            elif str(rank_mode) == "target_neighbor":
                target_key = lambda row: (float(row[5]["nn_topk_mean"]), row[2], row[1], row[0])
            else:
                target_key = lambda row: (row[1], row[0])
            target_sim, margin, neighbor_margin, target_label, target_size, target_support, target_time = max(
                target_rows,
                key=target_key,
            )
            rec = records[int(idx)]
            centroid_score = float(margin + 0.03 * min(conflict_count[int(idx)], 10) + 0.001 * np.log1p(source_size))
            retention_score = float(
                margin
                + 0.75 * neighbor_margin
                + 0.03 * min(conflict_count[int(idx)], 10)
                + 0.02 * min(int(target_time["temporal_support"]), 5)
                - 0.05 * min(int(source_time["temporal_support"]), 5)
                - 0.03 * min(int(source_time["temporal_overlap"]), 5)
                + 0.001 * np.log1p(source_size)
            )
            target_neighbor_score = float(
                float(target_support["nn_topk_mean"])
                + 0.25 * neighbor_margin
                + 0.10 * margin
                + 0.01 * min(conflict_count[int(idx)], 10)
                - 0.01 * min(int(source_time["temporal_overlap"]), 5)
            )
            if str(rank_mode) == "source_retention":
                score = retention_score
            elif str(rank_mode) == "target_neighbor":
                score = target_neighbor_score
            else:
                score = centroid_score
            source_rows.append(
                {
                    "seq": int(rec.seq),
                    "tracklet_key": str(rec.tracklet_key),
                    "video": str(rec.video),
                    "camera": str(rec.camera),
                    "start_frame": int(rec.start_frame),
                    "end_frame": int(rec.end_frame),
                    "source_label": int(source_label),
                    "target_label": int(target_label),
                    "source_size": int(source_size),
                    "target_size": int(target_size),
                    "source_sim": round(float(source_sim), 6),
                    "target_sim": round(float(target_sim), 6),
                    "margin": round(float(margin), 6),
                    "source_nn_max": round(float(source_support["nn_max"]), 6),
                    "source_nn_topk_mean": round(float(source_support["nn_topk_mean"]), 6),
                    "source_nn_count": int(source_support["nn_count"]),
                    "target_nn_max": round(float(target_support["nn_max"]), 6),
                    "target_nn_topk_mean": round(float(target_support["nn_topk_mean"]), 6),
                    "target_nn_count": int(target_support["nn_count"]),
                    "neighbor_margin": round(float(neighbor_margin), 6),
                    "source_temporal_support": int(source_time["temporal_support"]),
                    "source_temporal_overlap": int(source_time["temporal_overlap"]),
                    "target_temporal_support": int(target_time["temporal_support"]),
                    "target_temporal_overlap": int(target_time["temporal_overlap"]),
                    "conflict_edges": int(conflict_count[int(idx)]),
                    "centroid_score": round(float(centroid_score), 6),
                    "retention_score": round(float(retention_score), 6),
                    "target_neighbor_score": round(float(target_neighbor_score), 6),
                    "rank_mode": str(rank_mode),
                    "score": round(float(score), 6),
                }
            )
        source_rows.sort(key=lambda row: (float(row["score"]), float(row["margin"]), int(row["conflict_edges"])), reverse=True)
        candidates.extend(source_rows[: max(int(max_candidates_per_source), 1)])
    candidates.sort(key=lambda row: (float(row["score"]), float(row["margin"]), int(row["conflict_edges"])), reverse=True)
    return candidates


def _apply_candidates(base_labels: np.ndarray, candidates: list[dict[str, object]], seq_to_idx: dict[int, int], *, top_k: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels = base_labels.copy()
    accepted: list[dict[str, object]] = []
    used_seqs: set[int] = set()
    for cand in candidates:
        if len(accepted) >= int(top_k):
            break
        seq = int(cand["seq"])
        idx = seq_to_idx.get(seq)
        if idx is None or seq in used_seqs:
            continue
        if int(labels[int(idx)]) != int(cand["source_label"]):
            continue
        labels[int(idx)] = int(cand["target_label"])
        accepted.append(dict(cand))
        used_seqs.add(seq)
    return labels, accepted


def _write_assignments(path: str, records, labels: np.ndarray, keep_seqs: set[int], *, offset: int, accepted: list[dict[str, object]]) -> dict[str, object]:
    moved = {int(row["seq"]): row for row in accepted}
    rows = []
    for idx, record in enumerate(records):
        seq = int(record.seq)
        if seq not in keep_seqs:
            continue
        status = "outlier_relinked" if seq in moved else "base_component"
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
                "decision_status": status,
            }
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["seq", "predicted_global_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {"assignments_out": str(path), "assignment_rows": int(len(rows)), "accepted_relinks": int(len(accepted))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--min-source-sizes", default="64")
    ap.add_argument("--max-source-size", type=int, default=500)
    ap.add_argument("--min-target-size", type=int, default=4)
    ap.add_argument("--max-target-size", type=int, default=500)
    ap.add_argument("--min-conflict-edges", type=int, default=1)
    ap.add_argument("--max-source-sims", default="0.70")
    ap.add_argument("--min-target-sims", default="0.72")
    ap.add_argument("--min-margins", default="0.03")
    ap.add_argument("--max-candidates-per-source", type=int, default=2)
    ap.add_argument("--support-top-k", type=int, default=3)
    ap.add_argument("--max-temporal-gap-frames", type=int, default=300)
    ap.add_argument("--min-neighbor-margins", default="-1.0")
    ap.add_argument("--max-source-neighbor-sims", default="1.0")
    ap.add_argument("--max-source-temporal-supports", default="1000000")
    ap.add_argument("--rank-modes", default="centroid")
    ap.add_argument("--top-ks", default="1,2,4,8")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--score-pair", action="store_true")
    ap.add_argument("--score-full", action="store_true")
    ap.add_argument("--assignment-offset", type=int, default=96_000_000)
    ap.add_argument("--assignments-dir", default="")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    emb = _load_feature_npz(args.feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0).astype(np.float32)
    emb = _l2n(emb)
    keep_seqs, output_info = _output_keep_seqs(records, args)
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
    forbidden = _build_overlap_forbidden(records)

    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    gt_by_seq = weight_by_seq = eval_stats = None
    if bool(args.score_pair):
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

    rows = []
    candidate_cache: dict[tuple[int, float, float, float, float, float, int, str], list[dict[str, object]]] = {}
    for min_source_size in _parse_ints(args.min_source_sizes):
        for max_source_sim in _parse_floats(args.max_source_sims):
            for min_target_sim in _parse_floats(args.min_target_sims):
                for min_margin in _parse_floats(args.min_margins):
                    for min_neighbor_margin in _parse_floats(args.min_neighbor_margins):
                        for max_source_neighbor_sim in _parse_floats(args.max_source_neighbor_sims):
                            for max_source_temporal_support in _parse_ints(args.max_source_temporal_supports):
                                for rank_mode in [part.strip() for part in str(args.rank_modes).split(",") if part.strip()]:
                                    key = (
                                        int(min_source_size),
                                        float(max_source_sim),
                                        float(min_target_sim),
                                        float(min_margin),
                                        float(min_neighbor_margin),
                                        float(max_source_neighbor_sim),
                                        int(max_source_temporal_support),
                                        str(rank_mode),
                                    )
                                    candidates = _build_candidates(
                                        records,
                                        base_labels,
                                        keep_indices,
                                        emb,
                                        forbidden,
                                        min_source_size=int(min_source_size),
                                        max_source_size=int(args.max_source_size),
                                        min_target_size=int(args.min_target_size),
                                        max_target_size=int(args.max_target_size),
                                        min_conflict_edges=int(args.min_conflict_edges),
                                        max_source_sim=float(max_source_sim),
                                        min_target_sim=float(min_target_sim),
                                        min_margin=float(min_margin),
                                        max_candidates_per_source=int(args.max_candidates_per_source),
                                        support_top_k=int(args.support_top_k),
                                        max_temporal_gap_frames=int(args.max_temporal_gap_frames),
                                        min_neighbor_margin=float(min_neighbor_margin),
                                        max_source_neighbor_sim=float(max_source_neighbor_sim),
                                        max_source_temporal_support=int(max_source_temporal_support),
                                        rank_mode=str(rank_mode),
                                    )
                                    candidate_cache[key] = candidates
                                    for top_k in _parse_ints(args.top_ks):
                                        labels, accepted = _apply_candidates(base_labels, candidates, seq_to_idx, top_k=int(top_k))
                                        pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                        row = {
                                            "mode": "conflict_outlier_relink",
                                            "min_source_size": int(min_source_size),
                                            "max_source_size": int(args.max_source_size),
                                            "min_target_size": int(args.min_target_size),
                                            "max_target_size": int(args.max_target_size),
                                            "min_conflict_edges": int(args.min_conflict_edges),
                                            "max_source_sim": float(max_source_sim),
                                            "min_target_sim": float(min_target_sim),
                                            "min_margin": float(min_margin),
                                            "support_top_k": int(args.support_top_k),
                                            "max_temporal_gap_frames": int(args.max_temporal_gap_frames),
                                            "min_neighbor_margin": float(min_neighbor_margin),
                                            "max_source_neighbor_sim": float(max_source_neighbor_sim),
                                            "max_source_temporal_support": int(max_source_temporal_support),
                                            "rank_mode": str(rank_mode),
                                            "top_k": int(top_k),
                                            "candidate_count": int(len(candidates)),
                                            "accepted_relinks": int(len(accepted)),
                                            "accepted_preview": accepted[:12],
                                            "uses_anchors": False,
                                            "uses_gt_for_training_or_anchors": False,
                                            "uses_gt_for_evaluation_only": True,
                                        }
                                        if bool(args.score_pair) and gt_by_seq is not None and weight_by_seq is not None:
                                            row.update(_pair_metrics([int(record.seq) for record in records], pred, gt_by_seq, weight_by_seq))
                                        if bool(args.score_full):
                                            full = _score_full(pred_by_video, gt_by_video, pred)
                                            row.update({f"full_{name}": value for name, value in full.items() if name != "per_video"})
                                            row["full_per_video"] = full["per_video"]
                                        if args.assignments_dir:
                                            sig = (
                                                f"src{min_source_size}_ss{str(max_source_sim).replace('.', 'p')}"
                                                f"_ts{str(min_target_sim).replace('.', 'p')}_m{str(min_margin).replace('.', 'p')}"
                                                f"_nm{str(min_neighbor_margin).replace('.', 'p').replace('-', 'n')}"
                                                f"_sn{str(max_source_neighbor_sim).replace('.', 'p')}_st{max_source_temporal_support}"
                                                f"_{rank_mode}_k{top_k}"
                                            )
                                            row.update(
                                                _write_assignments(
                                                    str(Path(args.assignments_dir) / f"{sig}.csv"),
                                                    records,
                                                    labels,
                                                    keep_seqs,
                                                    offset=int(args.assignment_offset),
                                                    accepted=accepted,
                                                )
                                            )
                                        rows.append(row)

    sort_keys = (
        (lambda row: (float(row.get("full_IDF1", 0.0)), float(row.get("accepted_relinks", 0.0))))
        if bool(args.score_full)
        else (lambda row: (float(row.get("accepted_relinks", 0.0)), float(row.get("candidate_count", 0.0))))
    )
    rows.sort(key=sort_keys, reverse=True)
    result = {
        "assignment_csv": str(args.assignment_csv),
        "feature_npz": str(args.feature_npz),
        "base_assignment_components": int(len(raw_to_local)),
        "candidate_configs": int(len(candidate_cache)),
        "top": rows[:100],
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
    print(json.dumps({"json": str(out), "rows": len(rows), "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
