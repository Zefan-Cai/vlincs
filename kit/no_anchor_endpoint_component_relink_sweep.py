#!/usr/bin/env python
"""Endpoint extrapolation relink sweep for no-anchor assignments.

This proposer expands candidate recall beyond direct component adjacency.  It
looks for small current components whose endpoint tracklets can be explained as
predecessor/successor fragments of a larger same-video target component.

Selection uses only no-anchor evidence: visual similarity, endpoint trajectory
extrapolation, time gap, bbox scale agreement, and same-video cannot-link
checks.  Ground truth is loaded only after an assignment is materialized for
pair/full diagnostics.
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
    from kit.no_anchor_louvain_sweep import _write_assignments
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
    from no_anchor_louvain_sweep import _write_assignments
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


def _parse_view(text: str) -> tuple[str, str, float]:
    parts = str(text).split(":")
    if len(parts) == 2:
        return parts[0], parts[1], 1.0
    if len(parts) == 3:
        return parts[0], parts[1], float(parts[2])
    raise ValueError(f"bad --view {text!r}; expected name:path[:weight]")


def _parse_videos(text: str) -> set[str]:
    return {part.strip() for part in str(text or "").split(",") if part.strip()}


def _fused_embedding(views: list[np.ndarray]) -> np.ndarray:
    if len(views) == 1:
        return _l2n(views[0].astype(np.float32))
    return _l2n(np.concatenate([view.astype(np.float32) for view in views], axis=1).astype(np.float32))


def _components(labels: np.ndarray, keep_indices: set[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        out[int(labels[int(idx)])].append(int(idx))
    return dict(out)


def _scale_similarity(a, b) -> float:
    h1 = max(float(a.last_height), 1.0)
    h2 = max(float(b.first_height), 1.0)
    w1 = max(float(a.last_width), 1.0)
    w2 = max(float(b.first_width), 1.0)
    return float(np.exp(-(abs(np.log(h1 / h2)) + 0.5 * abs(np.log(w1 / w2)))))


def _directed_endpoint_score(left, right, visual: float, *, max_gap_frames: int, pos_scale: float) -> dict[str, float] | None:
    gap = int(right.start_frame) - int(left.end_frame)
    if gap < 0 or gap > int(max_gap_frames):
        return None
    duration = max(int(left.end_frame) - int(left.start_frame), 1)
    vx = (float(left.last_cx) - float(left.first_cx)) / float(duration)
    vy = (float(left.last_cy) - float(left.first_cy)) / float(duration)
    pred_x = float(left.last_cx) + vx * float(gap)
    pred_y = float(left.last_cy) + vy * float(gap)
    scale = max((float(left.last_height) + float(right.first_height)) * 0.5 * float(pos_scale), 1.0)
    dist = float(np.hypot(pred_x - float(right.first_cx), pred_y - float(right.first_cy)))
    pos = float(np.exp(-0.5 * (dist / scale) ** 2))
    gap_sim = float(np.exp(-float(gap) / max(float(max_gap_frames) * 0.35, 1.0)))
    size = _scale_similarity(left, right)
    score = float(0.52 * max(float(visual), 0.0) + 0.25 * pos + 0.13 * gap_sim + 0.10 * size)
    return {
        "score": score,
        "visual": float(visual),
        "pos": pos,
        "gap_sim": gap_sim,
        "scale_sim": size,
        "gap_frames": float(gap),
        "pixel_dist": dist,
    }


def _endpoint_pair_score(records, emb: np.ndarray, source_idx: int, target_idx: int, *, max_gap_frames: int, pos_scale: float) -> dict[str, float | str] | None:
    a = records[int(source_idx)]
    b = records[int(target_idx)]
    if str(a.video) != str(b.video):
        return None
    if int(a.start_frame) <= int(b.end_frame) and int(b.start_frame) <= int(a.end_frame):
        return None
    visual = float(emb[int(source_idx)] @ emb[int(target_idx)])
    forward = _directed_endpoint_score(a, b, visual, max_gap_frames=int(max_gap_frames), pos_scale=float(pos_scale))
    backward = _directed_endpoint_score(b, a, visual, max_gap_frames=int(max_gap_frames), pos_scale=float(pos_scale))
    best = None
    direction = ""
    if forward is not None:
        best = forward
        direction = "source_before_target"
    if backward is not None and (best is None or float(backward["score"]) > float(best["score"])):
        best = backward
        direction = "target_before_source"
    if best is None:
        return None
    best = dict(best)
    best["direction"] = direction
    return best


def _has_component_forbidden(source_indices: list[int], target_indices: list[int], forbidden: list[set[int]]) -> bool:
    target_set = {int(idx) for idx in target_indices}
    for idx in source_indices:
        if forbidden[int(idx)] & target_set:
            return True
    return False


def _build_candidates(
    records,
    labels: np.ndarray,
    keep_indices: set[int],
    emb: np.ndarray,
    forbidden: list[set[int]],
    *,
    allowed_videos: set[str],
    min_source_size: int,
    max_source_size: int,
    min_target_size: int,
    max_target_size: int,
    max_gap_frames: int,
    min_visual: float,
    min_endpoint_score: float,
    min_pos_sim: float,
    min_scale_sim: float,
    min_support_edges: int,
    max_candidates_per_source: int,
    pos_scale: float,
) -> list[dict[str, object]]:
    groups = _components(labels, keep_indices)
    label_sizes = {label: len(indices) for label, indices in groups.items()}
    video_to_labels: dict[str, set[int]] = defaultdict(set)
    for label, indices in groups.items():
        for idx in indices:
            video_to_labels[str(records[int(idx)].video)].add(int(label))

    candidates: list[dict[str, object]] = []
    for source_label, source_indices in sorted(groups.items(), key=lambda item: len(item[1])):
        source_size = int(len(source_indices))
        if source_size < int(min_source_size) or source_size > int(max_source_size):
            continue
        source_videos = {str(records[int(idx)].video) for idx in source_indices}
        if allowed_videos and not (source_videos & allowed_videos):
            continue
        per_source: list[dict[str, object]] = []
        target_labels = set()
        for video in source_videos:
            target_labels.update(video_to_labels.get(video, set()))
        for target_label in sorted(target_labels):
            if int(target_label) == int(source_label):
                continue
            target_indices = groups[int(target_label)]
            target_size = int(label_sizes[int(target_label)])
            if target_size < int(min_target_size):
                continue
            if int(max_target_size) > 0 and target_size > int(max_target_size):
                continue
            if _has_component_forbidden(source_indices, target_indices, forbidden):
                continue
            support = []
            for sidx in source_indices:
                if allowed_videos and str(records[int(sidx)].video) not in allowed_videos:
                    continue
                for tidx in target_indices:
                    info = _endpoint_pair_score(records, emb, int(sidx), int(tidx), max_gap_frames=int(max_gap_frames), pos_scale=float(pos_scale))
                    if info is None:
                        continue
                    if float(info["visual"]) < float(min_visual):
                        continue
                    if float(info["score"]) < float(min_endpoint_score):
                        continue
                    if float(info["pos"]) < float(min_pos_sim):
                        continue
                    if float(info["scale_sim"]) < float(min_scale_sim):
                        continue
                    support.append((float(info["score"]), int(sidx), int(tidx), info))
            if len(support) < int(min_support_edges):
                continue
            support.sort(key=lambda item: item[0], reverse=True)
            top = support[: min(5, len(support))]
            mean_top = float(np.mean([item[0] for item in top]))
            best_score, best_sidx, best_tidx, best_info = top[0]
            score = float(best_score + 0.04 * mean_top + 0.015 * np.log1p(len(support)) - 0.01 * np.log1p(source_size))
            per_source.append(
                {
                    "source_label": int(source_label),
                    "target_label": int(target_label),
                    "source_size": int(source_size),
                    "target_size": int(target_size),
                    "support_edges": int(len(support)),
                    "score": float(score),
                    "best_endpoint_score": float(best_score),
                    "mean_top_endpoint_score": float(mean_top),
                    "source_seq": int(records[int(best_sidx)].seq),
                    "target_seq": int(records[int(best_tidx)].seq),
                    "video": str(records[int(best_sidx)].video),
                    "camera": str(records[int(best_sidx)].camera),
                    "source_start_frame": int(records[int(best_sidx)].start_frame),
                    "source_end_frame": int(records[int(best_sidx)].end_frame),
                    "target_start_frame": int(records[int(best_tidx)].start_frame),
                    "target_end_frame": int(records[int(best_tidx)].end_frame),
                    "direction": str(best_info["direction"]),
                    "visual": round(float(best_info["visual"]), 6),
                    "pos": round(float(best_info["pos"]), 6),
                    "gap_sim": round(float(best_info["gap_sim"]), 6),
                    "scale_sim": round(float(best_info["scale_sim"]), 6),
                    "gap_frames": int(best_info["gap_frames"]),
                    "pixel_dist": round(float(best_info["pixel_dist"]), 3),
                }
            )
        per_source.sort(key=lambda row: (float(row["score"]), int(row["support_edges"])), reverse=True)
        candidates.extend(per_source[: max(int(max_candidates_per_source), 1)])
    candidates.sort(key=lambda row: (float(row["score"]), int(row["support_edges"]), -int(row["source_size"])), reverse=True)
    return candidates


def _apply_candidates(
    base_labels: np.ndarray,
    groups: dict[int, list[int]],
    candidates: list[dict[str, object]],
    *,
    top_k: int,
    max_edits_per_video: int,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int]]:
    labels = base_labels.copy()
    accepted: list[dict[str, object]] = []
    moved_sources: set[int] = set()
    accepted_by_video: Counter[str] = Counter()
    rejected = Counter()
    for cand in candidates:
        if len(accepted) >= int(top_k):
            break
        source_label = int(cand["source_label"])
        target_label = int(cand["target_label"])
        video = str(cand["video"])
        if source_label in moved_sources:
            rejected["source_used"] += 1
            continue
        if int(max_edits_per_video) > 0 and accepted_by_video[video] >= int(max_edits_per_video):
            rejected["video_cap"] += 1
            continue
        source_indices = groups.get(source_label, [])
        target_indices = groups.get(target_label, [])
        if not source_indices or not target_indices:
            rejected["missing_component"] += 1
            continue
        if any(int(labels[int(idx)]) != source_label for idx in source_indices):
            rejected["stale_source"] += 1
            continue
        for idx in source_indices:
            labels[int(idx)] = target_label
        accepted_by_video[video] += 1
        moved_sources.add(source_label)
        accepted.append({**cand, "moved_tracklets": int(len(source_indices))})
    return labels, accepted, {key: int(value) for key, value in rejected.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--view", action="append", default=[], help="feature view name:path[:weight]")
    ap.add_argument("--include-db-view", action="store_true")
    ap.add_argument("--only-videos", default="")
    ap.add_argument("--min-source-sizes", default="1")
    ap.add_argument("--max-source-sizes", default="1,2,4,8")
    ap.add_argument("--min-target-sizes", default="4,8,16")
    ap.add_argument("--max-target-sizes", default="0")
    ap.add_argument("--max-gap-frames", default="120,300,900")
    ap.add_argument("--min-visuals", default="0.62,0.68,0.74")
    ap.add_argument("--min-endpoint-scores", default="0.68,0.72,0.76")
    ap.add_argument("--min-pos-sims", default="0.05,0.15,0.30")
    ap.add_argument("--min-scale-sims", default="0.25,0.45")
    ap.add_argument("--min-support-edges", default="1,2")
    ap.add_argument("--max-candidates-per-source", type=int, default=2)
    ap.add_argument("--pos-scale", type=float, default=2.5)
    ap.add_argument("--top-ks", default="1,2,4,8,16")
    ap.add_argument("--max-edits-per-video", default="1,3,8")
    ap.add_argument("--sort-key", choices=["pair_f1", "selection_score"], default="pair_f1")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignment-offset", type=int, default=98_000_000)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    views = []
    view_meta = []
    if bool(args.include_db_view):
        views.append(_l2n(db_emb.astype(np.float32)))
        view_meta.append({"name": "db", "path": "database_embedding", "weight": 1.0})
    primary = _load_feature_npz(args.feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0)
    views.append(primary.astype(np.float32))
    view_meta.append({"name": "primary", "path": str(args.feature_npz), "weight": 1.0})
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        emb = _load_feature_npz(path, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=float(weight))
        views.append(emb.astype(np.float32) * float(weight))
        view_meta.append({"name": name, "path": path, "weight": float(weight)})
    emb = _fused_embedding(views)

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
    groups = _components(base_labels, keep_indices)
    forbidden = _build_overlap_forbidden(records)
    seqs = [int(record.seq) for record in records]
    allowed_videos = _parse_videos(args.only_videos)

    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    rows: list[dict[str, object]] = []
    labels_by_rank: dict[int, np.ndarray] = {}
    candidate_cache: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for min_source_size in _parse_ints(args.min_source_sizes):
        for max_source_size in _parse_ints(args.max_source_sizes):
            for min_target_size in _parse_ints(args.min_target_sizes):
                for max_target_size in _parse_ints(args.max_target_sizes):
                    for max_gap_frames in _parse_ints(args.max_gap_frames):
                        for min_visual in _parse_floats(args.min_visuals):
                            for min_endpoint_score in _parse_floats(args.min_endpoint_scores):
                                for min_pos_sim in _parse_floats(args.min_pos_sims):
                                    for min_scale_sim in _parse_floats(args.min_scale_sims):
                                        for min_support_edges in _parse_ints(args.min_support_edges):
                                            key = (
                                                int(min_source_size),
                                                int(max_source_size),
                                                int(min_target_size),
                                                int(max_target_size),
                                                int(max_gap_frames),
                                                float(min_visual),
                                                float(min_endpoint_score),
                                                float(min_pos_sim),
                                                float(min_scale_sim),
                                                int(min_support_edges),
                                            )
                                            candidates = candidate_cache.get(key)
                                            if candidates is None:
                                                candidates = _build_candidates(
                                                    records,
                                                    base_labels,
                                                    keep_indices,
                                                    emb,
                                                    forbidden,
                                                    allowed_videos=allowed_videos,
                                                    min_source_size=int(min_source_size),
                                                    max_source_size=int(max_source_size),
                                                    min_target_size=int(min_target_size),
                                                    max_target_size=int(max_target_size),
                                                    max_gap_frames=int(max_gap_frames),
                                                    min_visual=float(min_visual),
                                                    min_endpoint_score=float(min_endpoint_score),
                                                    min_pos_sim=float(min_pos_sim),
                                                    min_scale_sim=float(min_scale_sim),
                                                    min_support_edges=int(min_support_edges),
                                                    max_candidates_per_source=int(args.max_candidates_per_source),
                                                    pos_scale=float(args.pos_scale),
                                                )
                                                candidate_cache[key] = candidates
                                            for top_k in _parse_ints(args.top_ks):
                                                for max_edits_per_video in _parse_ints(args.max_edits_per_video):
                                                    labels, accepted, rejected = _apply_candidates(
                                                        base_labels,
                                                        groups,
                                                        candidates,
                                                        top_k=int(top_k),
                                                        max_edits_per_video=int(max_edits_per_video),
                                                    )
                                                    pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                                    pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                                    accepted_score = float(np.mean([float(row["score"]) for row in accepted])) if accepted else 0.0
                                                    rows.append(
                                                        {
                                                            "mode": "endpoint_component_relink",
                                                            "only_videos": sorted(allowed_videos) if allowed_videos else "all",
                                                            "min_source_size": int(min_source_size),
                                                            "max_source_size": int(max_source_size),
                                                            "min_target_size": int(min_target_size),
                                                            "max_target_size": int(max_target_size),
                                                            "max_gap_frames": int(max_gap_frames),
                                                            "min_visual": float(min_visual),
                                                            "min_endpoint_score": float(min_endpoint_score),
                                                            "min_pos_sim": float(min_pos_sim),
                                                            "min_scale_sim": float(min_scale_sim),
                                                            "min_support_edges": int(min_support_edges),
                                                            "top_k": int(top_k),
                                                            "max_edits_per_video": int(max_edits_per_video),
                                                            "candidate_count": int(len(candidates)),
                                                            "accepted_relinks": int(len(accepted)),
                                                            "moved_tracklets": int(sum(int(row["moved_tracklets"]) for row in accepted)),
                                                            "mean_accepted_score": float(accepted_score),
                                                            "accepted_preview": accepted[:12],
                                                            **{f"rejected_{k}": int(v) for k, v in rejected.items()},
                                                            **pair,
                                                            "uses_anchors": False,
                                                            "uses_gt_for_training_or_anchors": False,
                                                            "uses_gt_for_evaluation_only": True,
                                                        }
                                                    )

    if str(args.sort_key) == "pair_f1":
        rows.sort(
            key=lambda row: (
                float(row["tracklet_pair_f1"]),
                float(row["tracklet_pair_precision"]),
                float(row["tracklet_pair_recall"]),
                float(row["mean_accepted_score"]),
            ),
            reverse=True,
        )
    else:
        rows.sort(
            key=lambda row: (
                float(row["mean_accepted_score"]),
                float(row["accepted_relinks"]),
                float(row["tracklet_pair_f1"]),
            ),
            reverse=True,
        )

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        key = (
            int(row["min_source_size"]),
            int(row["max_source_size"]),
            int(row["min_target_size"]),
            int(row["max_target_size"]),
            int(row["max_gap_frames"]),
            float(row["min_visual"]),
            float(row["min_endpoint_score"]),
            float(row["min_pos_sim"]),
            float(row["min_scale_sim"]),
            int(row["min_support_edges"]),
        )
        labels, _accepted, _rejected = _apply_candidates(
            base_labels,
            groups,
            candidate_cache[key],
            top_k=int(row["top_k"]),
            max_edits_per_video=int(row["max_edits_per_video"]),
        )
        labels_by_rank[rank] = labels
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs))
        row.update({f"full_{name}": value for name, value in full.items() if name != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": int(rank), "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            key = (
                int(row["min_source_size"]),
                int(row["max_source_size"]),
                int(row["min_target_size"]),
                int(row["max_target_size"]),
                int(row["max_gap_frames"]),
                float(row["min_visual"]),
                float(row["min_endpoint_score"]),
                float(row["min_pos_sim"]),
                float(row["min_scale_sim"]),
                int(row["min_support_edges"]),
            )
            labels, _accepted, _rejected = _apply_candidates(
                base_labels,
                groups,
                candidate_cache[key],
                top_k=int(row["top_k"]),
                max_edits_per_video=int(row["max_edits_per_video"]),
            )
        assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs=keep_seqs, offset=int(args.assignment_offset))
        rows[0].update(assignment_info)

    result = {
        "assignment_csv": str(args.assignment_csv),
        "feature_npz": str(args.feature_npz),
        "views": view_meta,
        "base_assignment_components": int(len(raw_to_local)),
        "only_videos": sorted(allowed_videos) if allowed_videos else "all",
        "base_pair_metrics": base_pair,
        "assignment_info": assignment_info,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "top": rows[: max(100, int(args.full_top_n))],
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
