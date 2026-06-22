#!/usr/bin/env python
"""No-anchor verifier using cross-tracklet clothing/body consistency.

Positive labels are generated without identity labels: short-gap same-stream
continuations that are plausible geometrically and agree in OSNet sample
features, body-part pose/color features, and color histograms.  Negatives are
hard cannot-link overlaps.  The classifier sees only visual evidence; temporal
and geometric constraints are used only to build weak labels.

Ground truth is loaded only after prediction for pair/full metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _candidate_edges, _component_members
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
    from kit.no_anchor_sample_positive_edge_verifier import (
        _fit_model,
        _l2n,
        _load_samples,
        _merge_by_probability,
        _parse_floats,
        _parse_ints,
        _row_features,
        _sample_pair_features,
        _write_csv,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _candidate_edges, _component_members
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
    from no_anchor_sample_positive_edge_verifier import (
        _fit_model,
        _l2n,
        _load_samples,
        _merge_by_probability,
        _parse_floats,
        _parse_ints,
        _row_features,
        _sample_pair_features,
        _write_csv,
    )


def _load_view(path: str, records) -> tuple[np.ndarray, dict[str, object]]:
    data = np.load(path, allow_pickle=True)
    if "features" not in data or "seqs" not in data:
        raise ValueError(f"{path} must contain seqs and features")
    seqs = [int(seq) for seq in data["seqs"].tolist()]
    by_seq = {seq: pos for pos, seq in enumerate(seqs)}
    features = data["features"].astype(np.float32)
    aligned = np.zeros((len(records), int(features.shape[1])), dtype=np.float32)
    missing = 0
    for idx, record in enumerate(records):
        pos = by_seq.get(int(record.seq))
        if pos is None:
            missing += 1
            continue
        aligned[idx] = features[pos]
    return _l2n(aligned), {"path": str(path), "dim": int(features.shape[1]), "missing_tracklets": int(missing)}


def _tracklet_gap(a, b) -> int:
    return int(b.start_frame) - int(a.end_frame)


def _scale_sim(a, b) -> float:
    h1 = max(float(a.last_height), 1.0)
    h2 = max(float(b.first_height), 1.0)
    return float(np.exp(-abs(np.log(h1 / h2))))


def _center_dist(a, b) -> float:
    denom = max(0.5 * (float(a.last_height) + float(b.first_height)), 1.0)
    dx = float(a.last_cx) - float(b.first_cx)
    dy = float(a.last_cy) - float(b.first_cy)
    return float(np.sqrt(dx * dx + dy * dy) / denom)


def _sample_topmean(samples, counts, i: int, j: int) -> float:
    ni = int(counts[i])
    nj = int(counts[j])
    if ni <= 0 or nj <= 0:
        return -1.0
    return float(_sample_pair_features(samples[i, :ni], samples[j, :nj])[1])


def _pair_features(samples, counts, mean, pose, color, i: int, j: int) -> list[float]:
    pose_sim = float(np.dot(pose[i], pose[j]))
    color_sim = float(np.dot(color[i], color[j]))
    return [
        *_row_features(samples, counts, i, j, mean),
        pose_sim,
        color_sim,
        min(pose_sim, color_sim),
        0.5 * (pose_sim + color_sim),
        abs(pose_sim - color_sim),
    ]


def _positive_pairs(records, samples, counts, pose, color, args):
    by_stream: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        if int(counts[idx]) > 0:
            by_stream[(str(rec.video), str(rec.camera))].append(int(idx))
    out = []
    considered = rejected_gap = rejected_geom = rejected_sample = rejected_clothing = 0
    for indices in by_stream.values():
        ordered = sorted(indices, key=lambda idx: (int(records[idx].start_frame), int(records[idx].end_frame), idx))
        for pos, i in enumerate(ordered):
            a = records[i]
            for j in ordered[pos + 1 :]:
                b = records[j]
                gap = _tracklet_gap(a, b)
                if gap < int(args.positive_min_gap_frames):
                    continue
                if gap > int(args.positive_max_gap_frames):
                    rejected_gap += 1
                    break
                considered += 1
                dist = _center_dist(a, b)
                scale = _scale_sim(a, b)
                if dist > float(args.positive_max_center_dist) or scale < float(args.positive_min_scale_sim):
                    rejected_geom += 1
                    continue
                sample_sim = _sample_topmean(samples, counts, i, j)
                if sample_sim < float(args.positive_min_sample_topmean):
                    rejected_sample += 1
                    continue
                pose_sim = float(np.dot(pose[i], pose[j]))
                color_sim = float(np.dot(color[i], color[j]))
                if pose_sim < float(args.positive_min_posecolor) or color_sim < float(args.positive_min_colorhist):
                    rejected_clothing += 1
                    continue
                out.append((i, j, sample_sim, pose_sim, color_sim, dist, scale, gap))
    out.sort(key=lambda row: (row[2] + row[3] + row[4], -row[5]), reverse=True)
    if int(args.max_positive_pairs) > 0 and len(out) > int(args.max_positive_pairs):
        out = out[: int(args.max_positive_pairs)]
    return [(int(i), int(j)) for i, j, *_rest in out], {
        "positive_source": "same_stream_short_gap_visual_clothing_continuation",
        "positive_pairs": int(len(out)),
        "positive_considered": int(considered),
        "positive_rejected_gap_breaks": int(rejected_gap),
        "positive_rejected_geometry": int(rejected_geom),
        "positive_rejected_sample": int(rejected_sample),
        "positive_rejected_clothing": int(rejected_clothing),
        "positive_min_gap_frames": int(args.positive_min_gap_frames),
        "positive_max_gap_frames": int(args.positive_max_gap_frames),
        "positive_max_center_dist": float(args.positive_max_center_dist),
        "positive_min_scale_sim": float(args.positive_min_scale_sim),
        "positive_min_sample_topmean": float(args.positive_min_sample_topmean),
        "positive_min_posecolor": float(args.positive_min_posecolor),
        "positive_min_colorhist": float(args.positive_min_colorhist),
    }


def _negative_pairs(samples, counts, mean, pose, color, forbidden, args):
    rows = []
    for i, nbrs in enumerate(forbidden):
        if int(counts[i]) <= 0:
            continue
        for j in nbrs:
            if i >= j or int(counts[j]) <= 0:
                continue
            osnet = float(np.dot(mean[i], mean[j]))
            pose_sim = float(np.dot(pose[i], pose[j]))
            color_sim = float(np.dot(color[i], color[j]))
            score = (
                float(args.edge_osnet_weight) * osnet
                + float(args.edge_posecolor_weight) * pose_sim
                + float(args.edge_colorhist_weight) * color_sim
            )
            rows.append((score, int(i), int(j)))
    rows.sort(reverse=True)
    total = len(rows)
    if int(args.max_negative_pairs) > 0:
        rows = rows[: int(args.max_negative_pairs)]
    return [(int(i), int(j)) for _score, i, j in rows], {
        "negative_source": "same_stream_overlap_cannot_link_hard_visual_clothing",
        "negative_pairs": int(len(rows)),
        "negative_candidates": int(total),
    }


def _build_training(records, samples, counts, mean, pose, color, forbidden, args):
    positives, pos_info = _positive_pairs(records, samples, counts, pose, color, args)
    negatives, neg_info = _negative_pairs(samples, counts, mean, pose, color, forbidden, args)
    if not positives or not negatives:
        raise RuntimeError(f"need positive and negative pairs, got pos={len(positives)} neg={len(negatives)}")
    X_pos = [_pair_features(samples, counts, mean, pose, color, i, j) for i, j in positives]
    X_neg = [_pair_features(samples, counts, mean, pose, color, i, j) for i, j in negatives]
    X = np.asarray(X_pos + X_neg, dtype=np.float32)
    y = np.asarray([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int8)
    return X, y, {**pos_info, **neg_info, "train_positive": int(len(X_pos)), "train_negative": int(len(X_neg))}


def _blend_embedding(mean, pose, color, args) -> np.ndarray:
    parts = [
        np.sqrt(max(float(args.edge_osnet_weight), 0.0)) * mean,
        np.sqrt(max(float(args.edge_posecolor_weight), 0.0)) * pose,
        np.sqrt(max(float(args.edge_colorhist_weight), 0.0)) * color,
    ]
    return _l2n(np.concatenate(parts, axis=1).astype(np.float32))


def _edge_probability(edge, members, samples, counts, mean, pose, color, model, args):
    left = np.asarray(members[int(edge["source"])], dtype=np.int64)
    right = np.asarray(members[int(edge["target"])], dtype=np.int64)
    if len(left) == 0 or len(right) == 0:
        return 0.0, {"sample_pair_count": 0, "sample_prob_top_mean": 0.0}
    osnet = mean[left] @ mean[right].T
    pose_sim = pose[left] @ pose[right].T
    color_sim = color[left] @ color[right].T
    blend = (
        float(args.edge_osnet_weight) * osnet
        + float(args.edge_posecolor_weight) * pose_sim
        + float(args.edge_colorhist_weight) * color_sim
    )
    flat = np.argsort(-blend.reshape(-1))[: max(int(args.edge_pair_topk), 1)]
    feature_rows = []
    selected = []
    for pos in flat.tolist():
        li = int(left[pos // len(right)])
        rj = int(right[pos % len(right)])
        if int(counts[li]) <= 0 or int(counts[rj]) <= 0:
            continue
        feature_rows.append(_pair_features(samples, counts, mean, pose, color, li, rj))
        selected.append((float(osnet[pos // len(right), pos % len(right)]), float(pose_sim[pos // len(right), pos % len(right)]), float(color_sim[pos // len(right), pos % len(right)])))
    if not feature_rows:
        return 0.0, {"sample_pair_count": 0, "sample_prob_top_mean": 0.0}
    prob = model.predict_proba(np.asarray(feature_rows, dtype=np.float32))[:, 1].astype(np.float32)
    top = np.sort(prob)[-min(5, len(prob)) :]
    sel = np.asarray(selected, dtype=np.float32)
    return float(prob.max()), {
        "sample_pair_count": int(len(prob)),
        "sample_prob_top_mean": float(top.mean()),
        "sample_prob_mean": float(prob.mean()),
        "selected_osnet_max": float(sel[:, 0].max()),
        "selected_posecolor_max": float(sel[:, 1].max()),
        "selected_colorhist_max": float(sel[:, 2].max()),
        "selected_osnet_mean": float(sel[:, 0].mean()),
        "selected_posecolor_mean": float(sel[:, 1].mean()),
        "selected_colorhist_mean": float(sel[:, 2].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--sample-feature-npz", required=True)
    ap.add_argument("--posecolor-npz", required=True)
    ap.add_argument("--colorhist-npz", required=True)
    ap.add_argument("--positive-min-gap-frames", type=int, default=0)
    ap.add_argument("--positive-max-gap-frames", type=int, default=120)
    ap.add_argument("--positive-max-center-dist", type=float, default=2.0)
    ap.add_argument("--positive-min-scale-sim", type=float, default=0.40)
    ap.add_argument("--positive-min-sample-topmean", type=float, default=0.66)
    ap.add_argument("--positive-min-posecolor", type=float, default=0.58)
    ap.add_argument("--positive-min-colorhist", type=float, default=0.58)
    ap.add_argument("--candidate-top-k", type=int, default=150)
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
    ap.add_argument("--random-state", type=int, default=31)
    ap.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95")
    ap.add_argument("--max-component-sizes", default="300,500,800")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
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
    X_train, y_train, train_info = _build_training(records, samples, counts, mean_emb, pose_emb, color_emb, forbidden, args)
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
    for threshold in _parse_floats(args.thresholds):
        for max_size in _parse_ints(args.max_component_sizes):
            labels, info = _merge_by_probability(records, base_labels, edge_rows, args, threshold, max_size)
            pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
            pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
            rows.append({"mode": "clothing_positive_edge_verifier", **info, **pair, "uses_anchors": False, "uses_gt_for_training_or_anchors": False, "uses_gt_for_evaluation_only": True})
    rows.sort(key=lambda row: (float(row["tracklet_pair_f1"]), float(row["tracklet_pair_recall"]), float(row["tracklet_pair_precision"])), reverse=True)

    full_rows = []
    for row in rows[: max(int(args.full_top_n), 0)]:
        labels, _ = _merge_by_probability(records, base_labels, edge_rows, args, float(row["sample_probability_threshold"]), int(row["merge_max_component_size"]))
        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        full_rows.append(row)

    result = {
        "assignment_csv": args.assignment_csv,
        "base_assignment_components": int(len(raw_to_local)),
        "sample_meta": sample_meta,
        "posecolor_meta": pose_meta,
        "colorhist_meta": color_meta,
        "base_pair_metrics": base_pair,
        "train_info": train_info,
        "model_info": model_info,
        "edge_info": edge_info,
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
