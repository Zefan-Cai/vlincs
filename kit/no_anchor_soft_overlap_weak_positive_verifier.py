#!/usr/bin/env python
"""Train a no-anchor verifier from soft-overlap duplicate evidence.

Positive labels are duplicate-like same-stream overlaps: the two tracklets
share frames, their boxes overlap strongly on those frames, and visual/body
evidence agrees.  Negative labels are same-stream overlaps that fail that
duplicate test.  Identity ground truth is loaded only after prediction for
metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_assignment_soft_overlap_merge_sweep import _precompute_overlap_pair_stats, _soft_forbidden_from_stats
    from kit.no_anchor_clothing_positive_edge_verifier import _blend_embedding, _edge_probability, _load_view, _pair_features
    from kit.no_anchor_component_merge_sweep import _candidate_edges, _component_members, _unionfind_from_labels
    from kit.no_anchor_louvain_sweep import _write_assignments
    from kit.no_anchor_resolve_sweep import (
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
    from kit.no_anchor_sample_positive_edge_verifier import (
        _fit_model,
        _load_samples,
        _parse_floats,
        _parse_ints,
        _sample_pair_features,
        _write_csv,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_assignment_soft_overlap_merge_sweep import _precompute_overlap_pair_stats, _soft_forbidden_from_stats
    from no_anchor_clothing_positive_edge_verifier import _blend_embedding, _edge_probability, _load_view, _pair_features
    from no_anchor_component_merge_sweep import _candidate_edges, _component_members, _unionfind_from_labels
    from no_anchor_louvain_sweep import _write_assignments
    from no_anchor_resolve_sweep import (
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
    from no_anchor_sample_positive_edge_verifier import (
        _fit_model,
        _load_samples,
        _parse_floats,
        _parse_ints,
        _sample_pair_features,
        _write_csv,
    )


def _sample_topmean(samples, counts, i: int, j: int) -> float:
    ni = int(counts[i])
    nj = int(counts[j])
    if ni <= 0 or nj <= 0:
        return -1.0
    return float(_sample_pair_features(samples[i, :ni], samples[j, :nj])[1])


def _iou_value(row: dict[str, float | int], stat: str) -> float:
    return float(row.get(f"{stat}_iou", row.get("median_iou", 0.0)))


def _row_evidence(row, samples, counts, pose, color, iou_stat: str) -> dict[str, float | int]:
    i = int(row["i"])
    j = int(row["j"])
    return {
        "i": i,
        "j": j,
        "common_frames": int(row["common_frames"]),
        "iou": float(_iou_value(row, iou_stat)),
        "visual": float(row["visual"]),
        "sample_topmean": _sample_topmean(samples, counts, i, j),
        "posecolor": float(np.dot(pose[i], pose[j])),
        "colorhist": float(np.dot(color[i], color[j])),
    }


def _duplicate_like(evidence: dict[str, float | int], args) -> bool:
    return (
        int(evidence["common_frames"]) >= int(args.positive_min_common_frames)
        and float(evidence["iou"]) >= float(args.positive_min_iou)
        and float(evidence["visual"]) >= float(args.positive_min_visual)
        and float(evidence["sample_topmean"]) >= float(args.positive_min_sample_topmean)
        and float(evidence["posecolor"]) >= float(args.positive_min_posecolor)
        and float(evidence["colorhist"]) >= float(args.positive_min_colorhist)
    )


def _build_training(records, pair_stats, samples, counts, mean, pose, color, args):
    positives = []
    hard_negative_pool = []
    fallback_negative_pool = []
    pos_keys = set()
    for row in pair_stats:
        i = int(row["i"])
        j = int(row["j"])
        if int(counts[i]) <= 0 or int(counts[j]) <= 0:
            continue
        ev = _row_evidence(row, samples, counts, pose, color, str(args.iou_stat))
        key = (min(i, j), max(i, j))
        if _duplicate_like(ev, args):
            positives.append((i, j, ev))
            pos_keys.add(key)
            continue
        if int(ev["common_frames"]) < int(args.negative_min_common_frames):
            continue
        fallback_negative_pool.append((i, j, ev))
        if float(ev["iou"]) <= float(args.negative_max_iou) and float(ev["visual"]) >= float(args.negative_min_visual):
            hard_negative_pool.append((i, j, ev))

    positives.sort(
        key=lambda item: (
            float(item[2]["iou"]),
            float(item[2]["visual"]),
            float(item[2]["sample_topmean"]),
            float(item[2]["posecolor"]) + float(item[2]["colorhist"]),
        ),
        reverse=True,
    )
    hard_negative_pool.sort(key=lambda item: (float(item[2]["visual"]), -float(item[2]["iou"])), reverse=True)
    fallback_negative_pool.sort(key=lambda item: (float(item[2]["visual"]), -float(item[2]["iou"])), reverse=True)

    if int(args.max_positive_pairs) > 0:
        positives = positives[: int(args.max_positive_pairs)]
    negatives = []
    seen = set(pos_keys)
    for i, j, ev in hard_negative_pool + fallback_negative_pool:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        negatives.append((i, j, ev))
        if int(args.max_negative_pairs) > 0 and len(negatives) >= int(args.max_negative_pairs):
            break

    if not positives or not negatives:
        raise RuntimeError(f"need positive and negative pairs, got pos={len(positives)} neg={len(negatives)}")

    X_pos = [_pair_features(samples, counts, mean, pose, color, i, j) for i, j, _ev in positives]
    X_neg = [_pair_features(samples, counts, mean, pose, color, i, j) for i, j, _ev in negatives]
    X = np.asarray(X_pos + X_neg, dtype=np.float32)
    y = np.asarray([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int8)
    return X, y, {
        "positive_source": "same_stream_soft_overlap_duplicate_like",
        "negative_source": "same_stream_overlap_non_duplicate_like",
        "train_positive": int(len(X_pos)),
        "train_negative": int(len(X_neg)),
        "positive_candidates": int(len(positives)),
        "hard_negative_candidates": int(len(hard_negative_pool)),
        "fallback_negative_candidates": int(len(fallback_negative_pool)),
        "positive_examples": [item[2] for item in positives[:20]],
        "negative_examples": [item[2] for item in negatives[:20]],
        "uses_ground_truth": False,
    }


def _merge_by_probability_soft(records, base_labels, edge_rows, forbidden, args, threshold: float, max_size: int):
    uf = _unionfind_from_labels(base_labels)
    accepted = rejected_threshold = rejected_forbidden = rejected_size = rejected_stale = 0
    score_key = str(args.probability_field)
    for row in sorted(edge_rows, key=lambda item: float(item[score_key]), reverse=True):
        if float(row[score_key]) < float(threshold):
            rejected_threshold += 1
            continue
        a = int(row["source_rep"])
        b = int(row["target_rep"])
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            rejected_stale += 1
            continue
        if len(uf.members[ra]) + len(uf.members[rb]) > int(max_size):
            rejected_size += 1
            continue
        if not uf.can_merge(a, b, forbidden, int(max_size)):
            rejected_forbidden += 1
            continue
        uf.merge(a, b)
        accepted += 1
    labels = uf.labels()
    return labels, {
        "sample_probability_field": score_key,
        "sample_probability_threshold": float(threshold),
        "merge_max_component_size": int(max_size),
        "merge_accepted": int(accepted),
        "merge_rejected_threshold": int(rejected_threshold),
        "merge_rejected_forbidden": int(rejected_forbidden),
        "merge_rejected_size": int(rejected_size),
        "merge_rejected_stale": int(rejected_stale),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
    }


def _load_eval(records, pred_by_video, dbname: str, role: str, eval_cache: str):
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {
        "cache_version": 1,
        "dbname": dbname,
        "role": role,
        "iou_thr": 0.5,
        "min_matches": 1,
        "min_purity": 0.0,
        "n_tracklets": len(records),
        "prediction_rows": int(sum(len(v) for v in pred_by_video.values())),
        "gt_rows": int(sum(len(v) for v in gt_by_video.values())),
    }
    cached = _load_eval_label_cache(eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {eval_cache}")
    return gt_by_video, cached


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--sample-feature-npz", required=True)
    ap.add_argument("--posecolor-npz", required=True)
    ap.add_argument("--colorhist-npz", required=True)
    ap.add_argument("--iou-stat", default="median", choices=["mean", "median", "max"])
    ap.add_argument("--positive-min-common-frames", type=int, default=3)
    ap.add_argument("--positive-min-iou", type=float, default=0.55)
    ap.add_argument("--positive-min-visual", type=float, default=0.80)
    ap.add_argument("--positive-min-sample-topmean", type=float, default=0.72)
    ap.add_argument("--positive-min-posecolor", type=float, default=0.50)
    ap.add_argument("--positive-min-colorhist", type=float, default=0.50)
    ap.add_argument("--negative-min-common-frames", type=int, default=1)
    ap.add_argument("--negative-max-iou", type=float, default=0.10)
    ap.add_argument("--negative-min-visual", type=float, default=0.55)
    ap.add_argument("--soft-min-common-frames", default="1,3")
    ap.add_argument("--soft-iou-thresholds", default="0.50,0.60")
    ap.add_argument("--soft-visual-thresholds", default="0.75,0.80,0.85")
    ap.add_argument("--soft-max-iou-frames", type=int, default=20)
    ap.add_argument("--candidate-top-k", type=int, default=160)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weight", type=float, default=0.0)
    ap.add_argument("--edge-pair-topk", type=int, default=48)
    ap.add_argument("--edge-osnet-weight", type=float, default=0.60)
    ap.add_argument("--edge-posecolor-weight", type=float, default=0.25)
    ap.add_argument("--edge-colorhist-weight", type=float, default=0.15)
    ap.add_argument("--probability-field", default="sample_prob_top_mean", choices=["sample_probability", "sample_prob_top_mean", "sample_prob_mean"])
    ap.add_argument("--max-positive-pairs", type=int, default=12000)
    ap.add_argument("--max-negative-pairs", type=int, default=12000)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf", "logreg"])
    ap.add_argument("--random-state", type=int, default=37)
    ap.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95")
    ap.add_argument("--max-component-sizes", default="300,500,800")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=80_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, _db_emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    samples, counts, mean_emb, sample_meta = _load_samples(args.sample_feature_npz, records)
    pose_emb, pose_meta = _load_view(args.posecolor_npz, records)
    color_emb, color_meta = _load_view(args.colorhist_npz, records)

    gt_by_video, cached = _load_eval(records, pred_by_video, args.dbname, args.role, args.eval_cache)
    gt_by_seq, weight_by_seq, eval_stats = cached
    keep_seqs, output_info = _output_keep_seqs(records, args)
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
    seqs = [int(record.seq) for record in records]

    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    overlap_pair_stats, overlap_cache_info = _precompute_overlap_pair_stats(
        records,
        pred_by_video,
        mean_emb,
        max_iou_frames=int(args.soft_max_iou_frames),
    )
    X_train, y_train, train_info = _build_training(records, overlap_pair_stats, samples, counts, mean_emb, pose_emb, color_emb, args)
    model, model_info = _fit_model(X_train, y_train, args)

    candidate_emb = _blend_embedding(mean_emb, pose_emb, color_emb, args)
    reps, members = _component_members(base_labels, keep_indices)
    edges, edge_info = _candidate_edges(
        records,
        candidate_emb,
        reps,
        members,
        candidate_top_k=int(args.candidate_top_k),
        top_edge_k=int(args.top_edge_k),
        centroid_weight=float(args.centroid_weight),
        min_source_size=1,
        max_source_size=1_000_000,
        min_target_size=1,
        max_target_size=1_000_000,
        forbid_camera_overlap=False,
        forbid_video_overlap=False,
    )
    edge_rows = []
    for edge in edges:
        prob, prob_info = _edge_probability(edge, members, samples, counts, mean_emb, pose_emb, color_emb, model, args)
        edge_rows.append({**edge, **prob_info, "sample_probability": float(prob)})

    rows = []
    labels_by_rank: dict[int, np.ndarray] = {}
    forbidden_cache: dict[tuple[int, float, float], tuple[list[set[int]], dict[str, object]]] = {}
    for min_common in _parse_ints(args.soft_min_common_frames):
        for iou_threshold in _parse_floats(args.soft_iou_thresholds):
            for visual_threshold in _parse_floats(args.soft_visual_thresholds):
                cache_key = (int(min_common), float(iou_threshold), float(visual_threshold))
                forbidden, soft_info = _soft_forbidden_from_stats(
                    records,
                    overlap_pair_stats,
                    min_common_frames=int(min_common),
                    iou_threshold=float(iou_threshold),
                    visual_threshold=float(visual_threshold),
                    iou_stat=str(args.iou_stat),
                    max_iou_frames=int(args.soft_max_iou_frames),
                )
                forbidden_cache[cache_key] = (forbidden, soft_info)
                for threshold in _parse_floats(args.thresholds):
                    for max_size in _parse_ints(args.max_component_sizes):
                        labels, info = _merge_by_probability_soft(records, base_labels, edge_rows, forbidden, args, threshold, max_size)
                        pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                        pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                        rows.append(
                            {
                                "mode": "soft_overlap_weak_positive_verifier",
                                **soft_info,
                                **info,
                                **pair,
                                "uses_anchors": False,
                                "uses_gt_for_training_or_anchors": False,
                                "uses_gt_for_evaluation_only": True,
                            }
                        )
    rows.sort(key=lambda row: (float(row["tracklet_pair_f1"]), float(row["tracklet_pair_recall"]), float(row["tracklet_pair_precision"])), reverse=True)

    full_rows = []
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        forbidden, _soft_info = forbidden_cache[(int(row["soft_min_common_frames"]), float(row["soft_iou_threshold"]), float(row["soft_visual_threshold"]))]
        labels, _ = _merge_by_probability_soft(records, base_labels, edge_rows, forbidden, args, float(row["sample_probability_threshold"]), int(row["merge_max_component_size"]))
        labels_by_rank[rank] = labels
        pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        full_rows.append(row)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            forbidden, _soft_info = forbidden_cache[(int(row["soft_min_common_frames"]), float(row["soft_iou_threshold"]), float(row["soft_visual_threshold"]))]
            labels, _ = _merge_by_probability_soft(records, base_labels, edge_rows, forbidden, args, float(row["sample_probability_threshold"]), int(row["merge_max_component_size"]))
        assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs=keep_seqs, offset=int(args.assignment_offset))
        rows[0].update(assignment_info)

    result = {
        "assignment_csv": args.assignment_csv,
        "base_assignment_components": int(len(raw_to_local)),
        "sample_meta": sample_meta,
        "posecolor_meta": pose_meta,
        "colorhist_meta": color_meta,
        "base_pair_metrics": base_pair,
        "overlap_cache_info": overlap_cache_info,
        "train_info": train_info,
        "model_info": model_info,
        "edge_info": edge_info,
        "assignment_info": assignment_info,
        "top_edges_by_probability": sorted(edge_rows, key=lambda row: float(row[str(args.probability_field)]), reverse=True)[:80],
        "top": rows[:100],
        "full_rows": full_rows,
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
    print(json.dumps({"json": str(out), "base": base_pair, "train_info": train_info, "model_info": model_info, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
