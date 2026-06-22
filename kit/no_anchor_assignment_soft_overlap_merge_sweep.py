#!/usr/bin/env python
"""Merge no-anchor components with a softened temporal-overlap constraint.

The default resolver treats any same-stream temporal overlap as cannot-link.
That is safe for two different people, but too strict when the detector/tracker
creates duplicate overlapping tracklets for the same person.  This postprocessor
keeps the no-anchor setting and softens only overlap pairs whose same-frame
boxes have high IoU and high visual similarity.  GT is used only after the
assignment is formed for metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _component_members, _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_assignment_multiview_merge_sweep import (
        _centroid_candidate_edges,
        _load_npz_aligned,
        _merge_edges,
        _parse_view,
        _score_edges,
        _view_tables,
    )
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from kit.no_anchor_louvain_sweep import _write_assignments
    from kit.no_anchor_resolve_sweep import (
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
    from no_anchor_assignment_component_merge_sweep import _component_members, _labels_from_assignment, _load_assignment_labels
    from no_anchor_assignment_multiview_merge_sweep import (
        _centroid_candidate_edges,
        _load_npz_aligned,
        _merge_edges,
        _parse_view,
        _score_edges,
        _view_tables,
    )
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from no_anchor_louvain_sweep import _write_assignments
    from no_anchor_resolve_sweep import (
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


def _seq_box_tables(pred_by_video) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for df in pred_by_video.values():
        if df.empty:
            continue
        ordered = df.sort_values(["seq", "frame"], kind="mergesort")
        for seq, group in ordered.groupby("seq", sort=False):
            frames = group["frame"].to_numpy(np.int64)
            boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
            tables[int(seq)] = (frames, boxes)
    return tables


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(float(a[2]) - float(a[0]), 0.0) * max(float(a[3]) - float(a[1]), 0.0)
    area_b = max(float(b[2]) - float(b[0]), 0.0) * max(float(b[3]) - float(b[1]), 0.0)
    return float(inter / max(area_a + area_b - inter, 1.0e-9))


def _overlap_iou_stats(
    a: tuple[np.ndarray, np.ndarray] | None,
    b: tuple[np.ndarray, np.ndarray] | None,
    *,
    max_frames: int,
) -> tuple[int, float, float, float]:
    if a is None or b is None:
        return 0, 0.0, 0.0, 0.0
    af, ab = a
    bf, bb = b
    i = j = 0
    values: list[float] = []
    while i < len(af) and j < len(bf):
        fa = int(af[i])
        fb = int(bf[j])
        if fa == fb:
            values.append(_box_iou(ab[i], bb[j]))
            i += 1
            j += 1
            if len(values) >= int(max_frames):
                break
        elif fa < fb:
            i += 1
        else:
            j += 1
    if not values:
        return 0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return int(len(arr)), float(np.mean(arr)), float(np.median(arr)), float(np.max(arr))


def _build_soft_overlap_forbidden(
    records,
    pred_by_video,
    emb: np.ndarray,
    *,
    min_common_frames: int,
    iou_threshold: float,
    visual_threshold: float,
    iou_stat: str,
    max_iou_frames: int,
) -> tuple[list[set[int]], dict[str, object]]:
    tables = _seq_box_tables(pred_by_video)
    x = _l2n(emb.astype(np.float32))
    forbidden = [set() for _ in records]
    stream_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        stream_to_indices[(str(record.video), str(record.camera))].append(int(idx))
    overlap_pairs = 0
    softened_pairs = 0
    hard_pairs = 0
    missing_box_pairs = 0
    common_counts = []
    softened_ious = []
    hard_ious = []
    for indices in stream_to_indices.values():
        ordered = sorted(indices, key=lambda idx: (int(records[idx].start_frame), int(records[idx].end_frame), idx))
        active: list[int] = []
        for j in ordered:
            active = [i for i in active if int(records[i].end_frame) >= int(records[j].start_frame)]
            for i in active:
                overlap_pairs += 1
                seq_i = int(records[i].seq)
                seq_j = int(records[j].seq)
                count, mean_iou, median_iou, max_iou = _overlap_iou_stats(
                    tables.get(seq_i),
                    tables.get(seq_j),
                    max_frames=int(max_iou_frames),
                )
                common_counts.append(int(count))
                if count <= 0:
                    missing_box_pairs += 1
                value = {"mean": mean_iou, "median": median_iou, "max": max_iou}.get(str(iou_stat), median_iou)
                visual = float(x[i] @ x[j])
                duplicate_like = (
                    int(count) >= int(min_common_frames)
                    and float(value) >= float(iou_threshold)
                    and visual >= float(visual_threshold)
                )
                if duplicate_like:
                    softened_pairs += 1
                    softened_ious.append(float(value))
                    continue
                forbidden[i].add(j)
                forbidden[j].add(i)
                hard_pairs += 1
                hard_ious.append(float(value))
            active.append(j)
    info = {
        "soft_min_common_frames": int(min_common_frames),
        "soft_iou_threshold": float(iou_threshold),
        "soft_visual_threshold": float(visual_threshold),
        "soft_iou_stat": str(iou_stat),
        "soft_max_iou_frames": int(max_iou_frames),
        "overlap_pairs": int(overlap_pairs),
        "softened_pairs": int(softened_pairs),
        "hard_forbidden_pairs": int(hard_pairs),
        "missing_box_pairs": int(missing_box_pairs),
        "mean_common_frames": round(float(np.mean(common_counts)) if common_counts else 0.0, 6),
        "mean_softened_iou": round(float(np.mean(softened_ious)) if softened_ious else 0.0, 6),
        "mean_hard_iou": round(float(np.mean(hard_ious)) if hard_ious else 0.0, 6),
        "uses_ground_truth": False,
    }
    return forbidden, info


def _precompute_overlap_pair_stats(records, pred_by_video, emb: np.ndarray, *, max_iou_frames: int) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    tables = _seq_box_tables(pred_by_video)
    x = _l2n(emb.astype(np.float32))
    stream_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        stream_to_indices[(str(record.video), str(record.camera))].append(int(idx))
    pairs: list[dict[str, float | int]] = []
    missing_box_pairs = 0
    common_counts = []
    for indices in stream_to_indices.values():
        ordered = sorted(indices, key=lambda idx: (int(records[idx].start_frame), int(records[idx].end_frame), idx))
        active: list[int] = []
        for j in ordered:
            active = [i for i in active if int(records[i].end_frame) >= int(records[j].start_frame)]
            for i in active:
                count, mean_iou, median_iou, max_iou = _overlap_iou_stats(
                    tables.get(int(records[i].seq)),
                    tables.get(int(records[j].seq)),
                    max_frames=int(max_iou_frames),
                )
                common_counts.append(int(count))
                if count <= 0:
                    missing_box_pairs += 1
                pairs.append(
                    {
                        "i": int(i),
                        "j": int(j),
                        "common_frames": int(count),
                        "mean_iou": float(mean_iou),
                        "median_iou": float(median_iou),
                        "max_iou": float(max_iou),
                        "visual": float(x[i] @ x[j]),
                    }
                )
            active.append(j)
    info = {
        "overlap_pairs_cached": int(len(pairs)),
        "overlap_missing_box_pairs": int(missing_box_pairs),
        "overlap_mean_common_frames": round(float(np.mean(common_counts)) if common_counts else 0.0, 6),
        "soft_max_iou_frames": int(max_iou_frames),
        "uses_ground_truth": False,
    }
    return pairs, info


def _soft_forbidden_from_stats(
    records,
    pair_stats: list[dict[str, float | int]],
    *,
    min_common_frames: int,
    iou_threshold: float,
    visual_threshold: float,
    iou_stat: str,
    max_iou_frames: int,
) -> tuple[list[set[int]], dict[str, object]]:
    forbidden = [set() for _ in records]
    softened_pairs = 0
    hard_pairs = 0
    softened_ious = []
    hard_ious = []
    key = f"{iou_stat}_iou"
    for row in pair_stats:
        i = int(row["i"])
        j = int(row["j"])
        value = float(row.get(key, row.get("median_iou", 0.0)))
        duplicate_like = (
            int(row["common_frames"]) >= int(min_common_frames)
            and value >= float(iou_threshold)
            and float(row["visual"]) >= float(visual_threshold)
        )
        if duplicate_like:
            softened_pairs += 1
            softened_ious.append(value)
        else:
            forbidden[i].add(j)
            forbidden[j].add(i)
            hard_pairs += 1
            hard_ious.append(value)
    info = {
        "soft_min_common_frames": int(min_common_frames),
        "soft_iou_threshold": float(iou_threshold),
        "soft_visual_threshold": float(visual_threshold),
        "soft_iou_stat": str(iou_stat),
        "soft_max_iou_frames": int(max_iou_frames),
        "overlap_pairs": int(len(pair_stats)),
        "softened_pairs": int(softened_pairs),
        "hard_forbidden_pairs": int(hard_pairs),
        "missing_box_pairs": int(sum(1 for row in pair_stats if int(row["common_frames"]) <= 0)),
        "mean_common_frames": round(float(np.mean([int(row["common_frames"]) for row in pair_stats])) if pair_stats else 0.0, 6),
        "mean_softened_iou": round(float(np.mean(softened_ious)) if softened_ious else 0.0, 6),
        "mean_hard_iou": round(float(np.mean(hard_ious)) if hard_ious else 0.0, 6),
        "uses_ground_truth": False,
    }
    return forbidden, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--primary-feature-npz", required=True)
    ap.add_argument("--view", action="append", default=[], help="feature view name:path[:weight], or name:db[:weight]")
    ap.add_argument("--candidate-top-k", type=int, default=100)
    ap.add_argument("--rank-ks", default="1,3,5")
    ap.add_argument("--sim-thresholds", default="0.62,0.66,0.70")
    ap.add_argument("--score-modes", default="hybrid,mean_min,min_sim")
    ap.add_argument("--merge-thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--min-rank-votes", default="0.0,0.25,0.5")
    ap.add_argument("--min-sim-votes", default="0.0,0.25,0.5")
    ap.add_argument("--max-component-sizes", default="500")
    ap.add_argument("--soft-min-common-frames", default="1,3")
    ap.add_argument("--soft-iou-thresholds", default="0.30,0.50,0.70")
    ap.add_argument("--soft-visual-thresholds", default="0.70,0.80,0.90")
    ap.add_argument("--soft-iou-stat", default="median", choices=["mean", "median", "max"])
    ap.add_argument("--soft-max-iou-frames", type=int, default=20)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
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
    primary_emb = _load_feature_npz(args.primary_feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0)
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
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    reps, members = _component_members(base_labels, keep_indices)
    base_edges, edge_info = _centroid_candidate_edges(records, primary_emb, reps, members, int(args.candidate_top_k))
    view_embeddings: dict[str, np.ndarray] = {"primary": primary_emb.astype(np.float32)}
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        if path.lower() == "db":
            view_embeddings[name] = _l2n(db_emb.astype(np.float32)) * float(weight)
        else:
            view_embeddings[name] = _load_npz_aligned(path, records, weight=float(weight))
    sims, ranks = _view_tables(view_embeddings, members)

    scored_cache: dict[tuple[str, int, float], list[dict[str, object]]] = {}
    for score_mode in [part for part in str(args.score_modes).split(",") if part.strip()]:
        for rank_k in _parse_ints(args.rank_ks):
            for sim_threshold in _parse_floats(args.sim_thresholds):
                scored_cache[(str(score_mode), int(rank_k), float(sim_threshold))] = _score_edges(
                    base_edges,
                    sims,
                    ranks,
                    score_mode=str(score_mode),
                    rank_k=int(rank_k),
                    sim_threshold=float(sim_threshold),
                )

    overlap_pair_stats, overlap_cache_info = _precompute_overlap_pair_stats(
        records,
        pred_by_video,
        primary_emb,
        max_iou_frames=int(args.soft_max_iou_frames),
    )
    print(json.dumps({"stage": "overlap_cache", **overlap_cache_info}, sort_keys=True), flush=True)

    rows: list[dict[str, object]] = []
    labels_by_rank: dict[int, np.ndarray] = {}
    forbidden_cache: dict[tuple[int, float, float], tuple[list[set[int]], dict[str, object]]] = {}
    for min_common in _parse_ints(args.soft_min_common_frames):
        for iou_threshold in _parse_floats(args.soft_iou_thresholds):
            for visual_threshold in _parse_floats(args.soft_visual_thresholds):
                key = (int(min_common), float(iou_threshold), float(visual_threshold))
                forbidden, soft_info = _soft_forbidden_from_stats(
                    records,
                    overlap_pair_stats,
                    min_common_frames=int(min_common),
                    iou_threshold=float(iou_threshold),
                    visual_threshold=float(visual_threshold),
                    iou_stat=str(args.soft_iou_stat),
                    max_iou_frames=int(args.soft_max_iou_frames),
                )
                forbidden_cache[key] = (forbidden, soft_info)
                for score_key, scored in scored_cache.items():
                    score_mode, rank_k, sim_threshold = score_key
                    for max_component_size in _parse_ints(args.max_component_sizes):
                        for threshold in _parse_floats(args.merge_thresholds):
                            for min_rank_vote in _parse_floats(args.min_rank_votes):
                                for min_sim_vote in _parse_floats(args.min_sim_votes):
                                    labels, merge_info = _merge_edges(
                                        records,
                                        base_labels,
                                        scored,
                                        forbidden,
                                        threshold=float(threshold),
                                        min_rank_vote=float(min_rank_vote),
                                        min_sim_vote=float(min_sim_vote),
                                        max_component_size=int(max_component_size),
                                    )
                                    pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                    pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                    rows.append(
                                        {
                                            "mode": "soft_overlap_multiview_merge",
                                            "score_mode": str(score_mode),
                                            "rank_k": int(rank_k),
                                            "sim_threshold": float(sim_threshold),
                                            **soft_info,
                                            **merge_info,
                                            **pair,
                                            "uses_anchors": False,
                                            "uses_gt_for_training_or_anchors": False,
                                            "uses_gt_for_evaluation_only": True,
                                        }
                                    )

    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
            float(row["tracklet_pair_precision"]),
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        forbidden, _soft_info = forbidden_cache[
            (int(row["soft_min_common_frames"]), float(row["soft_iou_threshold"]), float(row["soft_visual_threshold"]))
        ]
        scored = scored_cache[(str(row["score_mode"]), int(row["rank_k"]), float(row["sim_threshold"]))]
        labels, _merge_info = _merge_edges(
            records,
            base_labels,
            scored,
            forbidden,
            threshold=float(row["merge_threshold"]),
            min_rank_vote=float(row["min_rank_vote"]),
            min_sim_vote=float(row["min_sim_vote"]),
            max_component_size=int(row["max_component_size"]),
        )
        labels_by_rank[rank] = labels
        full = _score_full(
            pred_by_video,
            gt_by_video,
            _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs),
        )
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": int(rank), "full": full, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            forbidden, _soft_info = forbidden_cache[
                (int(row["soft_min_common_frames"]), float(row["soft_iou_threshold"]), float(row["soft_visual_threshold"]))
            ]
            scored = scored_cache[(str(row["score_mode"]), int(row["rank_k"]), float(row["sim_threshold"]))]
            labels, _merge_info = _merge_edges(
                records,
                base_labels,
                scored,
                forbidden,
                threshold=float(row["merge_threshold"]),
                min_rank_vote=float(row["min_rank_vote"]),
                min_sim_vote=float(row["min_sim_vote"]),
                max_component_size=int(row["max_component_size"]),
            )
        assignment_info = _write_assignments(
            args.assignments_out,
            records,
            labels,
            keep_seqs=keep_seqs,
            offset=int(args.assignment_offset),
        )
        rows[0].update(assignment_info)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "assignment_csv": str(args.assignment_csv),
        "primary_feature_npz": str(args.primary_feature_npz),
        "views": sorted(view_embeddings),
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "edge_info": edge_info,
        "assignment_info": assignment_info,
        "top": rows[: max(80, int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"base": base_pair, "best": rows[0] if rows else None, "json": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
