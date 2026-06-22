#!/usr/bin/env python
"""Train a no-anchor verifier from short-gap cross-tracklet continuations.

Positive labels are generated without identity labels: same-video/same-camera
tracklet pairs with a short temporal gap, plausible spatial continuation, and
high sample-level OSNet similarity.  Negatives are same-stream overlaps
(``cannot-link``).  The model only sees visual sample-pair features, so the
temporal/geometric rule is used as label construction rather than a feature.

GT is loaded only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

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
        _edge_probability,
        _fit_model,
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
        _edge_probability,
        _fit_model,
        _load_samples,
        _merge_by_probability,
        _parse_floats,
        _parse_ints,
        _row_features,
        _sample_pair_features,
        _write_csv,
    )

from vlincs_gallery.eval.score import load_ds1_gt_by_video


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
    feats = _sample_pair_features(samples[i, :ni], samples[j, :nj])
    return float(feats[1])


def _continuation_positive_pairs(records, samples, counts, args) -> tuple[list[tuple[int, int, dict[str, float | int]]], dict[str, object]]:
    by_stream: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        if int(counts[idx]) > 0:
            by_stream[(str(rec.video), str(rec.camera))].append(int(idx))
    out: list[tuple[int, int, dict[str, float | int]]] = []
    considered = 0
    rejected_gap = rejected_geom = rejected_sim = 0
    max_gap = int(args.positive_max_gap_frames)
    min_gap = int(args.positive_min_gap_frames)
    for indices in by_stream.values():
        ordered = sorted(indices, key=lambda idx: (int(records[idx].start_frame), int(records[idx].end_frame), idx))
        for pos, i in enumerate(ordered):
            a = records[i]
            for j in ordered[pos + 1 :]:
                b = records[j]
                gap = _tracklet_gap(a, b)
                if gap < min_gap:
                    continue
                if gap > max_gap:
                    rejected_gap += 1
                    break
                considered += 1
                dist = _center_dist(a, b)
                scale = _scale_sim(a, b)
                if dist > float(args.positive_max_center_dist) or scale < float(args.positive_min_scale_sim):
                    rejected_geom += 1
                    continue
                sim = _sample_topmean(samples, counts, i, j)
                if sim < float(args.positive_min_sample_topmean):
                    rejected_sim += 1
                    continue
                out.append(
                    (
                        int(i),
                        int(j),
                        {
                            "gap": int(gap),
                            "center_dist": float(dist),
                            "scale_sim": float(scale),
                            "sample_topmean": float(sim),
                        },
                    )
                )
    out.sort(key=lambda item: (float(item[2]["sample_topmean"]), -float(item[2]["center_dist"])), reverse=True)
    if int(args.max_positive_pairs) > 0 and len(out) > int(args.max_positive_pairs):
        out = out[: int(args.max_positive_pairs)]
    return out, {
        "positive_source": "same_stream_short_gap_continuation",
        "positive_pairs": int(len(out)),
        "positive_considered": int(considered),
        "positive_rejected_gap_breaks": int(rejected_gap),
        "positive_rejected_geometry": int(rejected_geom),
        "positive_rejected_similarity": int(rejected_sim),
        "positive_min_gap_frames": int(min_gap),
        "positive_max_gap_frames": int(max_gap),
        "positive_max_center_dist": float(args.positive_max_center_dist),
        "positive_min_scale_sim": float(args.positive_min_scale_sim),
        "positive_min_sample_topmean": float(args.positive_min_sample_topmean),
    }


def _cannotlink_negative_pairs(samples, counts, mean, forbidden, args) -> tuple[list[tuple[int, int]], dict[str, object]]:
    pairs = [(i, j) for i, nbrs in enumerate(forbidden) for j in nbrs if i < j and int(counts[i]) > 0 and int(counts[j]) > 0]
    scored = [(float(np.dot(mean[i], mean[j])), i, j) for i, j in pairs]
    scored.sort(reverse=True)
    if int(args.max_negative_pairs) > 0:
        scored = scored[: int(args.max_negative_pairs)]
    return [(int(i), int(j)) for _score, i, j in scored], {
        "negative_source": "same_stream_overlap_cannot_link_hard_visual",
        "negative_pairs": int(len(scored)),
        "negative_candidates": int(len(pairs)),
    }


def _build_training(records, samples, counts, mean, forbidden, args):
    positives, pos_info = _continuation_positive_pairs(records, samples, counts, args)
    negatives, neg_info = _cannotlink_negative_pairs(samples, counts, mean, forbidden, args)
    if not positives or not negatives:
        raise RuntimeError(f"need positive and negative pairs, got pos={len(positives)} neg={len(negatives)}")
    X_pos = [_row_features(samples, counts, i, j, mean) for i, j, _info in positives]
    X_neg = [_row_features(samples, counts, i, j, mean) for i, j in negatives]
    X = np.asarray(X_pos + X_neg, dtype=np.float32)
    y = np.asarray([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int8)
    return X, y, {**pos_info, **neg_info, "train_positive": int(len(X_pos)), "train_negative": int(len(X_neg))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--sample-feature-npz", required=True)
    ap.add_argument("--positive-min-gap-frames", type=int, default=0)
    ap.add_argument("--positive-max-gap-frames", type=int, default=60)
    ap.add_argument("--positive-max-center-dist", type=float, default=1.25)
    ap.add_argument("--positive-min-scale-sim", type=float, default=0.50)
    ap.add_argument("--positive-min-sample-topmean", type=float, default=0.72)
    ap.add_argument("--candidate-top-k", type=int, default=100)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weight", type=float, default=0.0)
    ap.add_argument("--edge-pair-topk", type=int, default=32)
    ap.add_argument(
        "--probability-field",
        default="sample_prob_top_mean",
        choices=["sample_probability", "sample_prob_top_mean", "sample_prob_mean"],
    )
    ap.add_argument("--max-positive-pairs", type=int, default=12000)
    ap.add_argument("--max-negative-pairs", type=int, default=12000)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf", "logreg"])
    ap.add_argument("--random-state", type=int, default=29)
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
    X_train, y_train, train_info = _build_training(records, samples, counts, mean_emb, forbidden, args)
    model, model_info = _fit_model(X_train, y_train, args)

    reps, members = _component_members(base_labels, keep_indices)
    edges, edge_info = _candidate_edges(
        records,
        mean_emb,
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
        prob, prob_info = _edge_probability(edge, members, samples, counts, mean_emb, model, max_pairs=int(args.edge_pair_topk))
        edge_rows.append({**edge, **prob_info, "sample_probability": float(prob)})

    rows: list[dict[str, object]] = []
    for threshold in _parse_floats(args.thresholds):
        for max_size in _parse_ints(args.max_component_sizes):
            labels, info = _merge_by_probability(records, base_labels, edge_rows, args, threshold, max_size)
            pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
            pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
            rows.append({"mode": "continuation_positive_edge_verifier", **info, **pair, "uses_anchors": False, "uses_gt_for_training_or_anchors": False, "uses_gt_for_evaluation_only": True})
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
    out = __import__("pathlib").Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "base": base_pair, "train_info": train_info, "model_info": model_info, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
