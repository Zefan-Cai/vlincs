#!/usr/bin/env python
"""Train a no-anchor edge verifier from tracklet sample-level positives.

Positive labels come from different sampled crops of the same tracklet.  Hard
negative labels come from same-stream temporal overlaps, which cannot be the
same identity.  The trained verifier scores component merge candidates from an
existing no-anchor assignment.  GT is used only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _candidate_edges, _component_members, _unionfind_from_labels
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
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _candidate_edges, _component_members, _unionfind_from_labels
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


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1.0e-9)


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _load_samples(path: str, records):
    data = np.load(path, allow_pickle=True)
    if "sample_features" not in data:
        raise ValueError(f"{path} does not contain sample_features; rerun extractor with --save-sample-features")
    seqs = [int(seq) for seq in data["seqs"].tolist()]
    by_seq = {seq: pos for pos, seq in enumerate(seqs)}
    raw = data["sample_features"].astype(np.float32)
    counts = np.asarray(data["sample_counts"], dtype=np.int16)
    dim = int(raw.shape[-1])
    aligned = np.zeros((len(records), raw.shape[1], dim), dtype=np.float32)
    aligned_counts = np.zeros(len(records), dtype=np.int16)
    missing = 0
    for idx, record in enumerate(records):
        pos = by_seq.get(int(record.seq))
        if pos is None:
            missing += 1
            continue
        n = int(counts[pos])
        aligned[idx, :n] = raw[pos, :n]
        aligned_counts[idx] = n
    mean = np.zeros((len(records), dim), dtype=np.float32)
    for idx, n in enumerate(aligned_counts.tolist()):
        if n > 0:
            mean[idx] = _l2n(aligned[idx, :n].mean(axis=0, keepdims=True))[0]
    return _l2n(aligned), aligned_counts, mean, {"sample_feature_path": str(path), "missing_tracklets": int(missing), "sample_feature_dim": dim}


def _sample_pair_features(a: np.ndarray, b: np.ndarray) -> list[float]:
    sim = (a @ b.T).reshape(-1).astype(np.float32)
    if sim.size == 0:
        return [0.0] * 9
    ordered = np.sort(sim)
    top3 = ordered[-min(3, sim.size) :]
    top5 = ordered[-min(5, sim.size) :]
    return [
        float(sim.max()),
        float(top3.mean()),
        float(top5.mean()),
        float(np.quantile(sim, 0.90)),
        float(np.quantile(sim, 0.75)),
        float(sim.mean()),
        float(sim.std()),
        float(sim.min()),
        float(np.log1p(sim.size)),
    ]


def _row_features(samples, counts, i: int, j: int, mean: np.ndarray) -> list[float]:
    ni = int(counts[i])
    nj = int(counts[j])
    visual = _sample_pair_features(samples[i, :ni], samples[j, :nj])
    return [float(np.dot(mean[i], mean[j])), float(min(ni, nj)), float(max(ni, nj)), *visual]


def _build_training(samples, counts, mean, forbidden, *, max_pos: int, max_neg: int, random_state: int):
    rng = np.random.default_rng(int(random_state))
    pos_indices = [idx for idx, n in enumerate(counts.tolist()) if int(n) >= 2]
    if len(pos_indices) > int(max_pos):
        pos_indices = rng.choice(pos_indices, size=int(max_pos), replace=False).tolist()
    X_pos = []
    for idx in pos_indices:
        n = int(counts[idx])
        left = samples[idx, :n:2]
        right = samples[idx, 1:n:2]
        if len(right) == 0:
            left = samples[idx, :1]
            right = samples[idx, 1:2]
        X_pos.append([1.0, float(len(left)), float(len(right)), *_sample_pair_features(left, right)])

    neg_pairs = [(i, j) for i, nbrs in enumerate(forbidden) for j in nbrs if i < j and counts[i] > 0 and counts[j] > 0]
    if len(neg_pairs) > int(max_neg):
        choice = rng.choice(len(neg_pairs), size=int(max_neg), replace=False)
        neg_pairs = [neg_pairs[int(pos)] for pos in choice.tolist()]
    X_neg = [_row_features(samples, counts, int(i), int(j), mean) for i, j in neg_pairs]
    if not X_pos or not X_neg:
        raise RuntimeError(f"need positive and negative training rows, got pos={len(X_pos)} neg={len(X_neg)}")
    X = np.asarray(X_pos + X_neg, dtype=np.float32)
    y = np.asarray([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int8)
    return X, y, {"train_positive": int(len(X_pos)), "train_negative": int(len(X_neg)), "negative_source": "same_stream_overlap_cannot_link"}


def _fit_model(X: np.ndarray, y: np.ndarray, args):
    weights = np.ones(len(y), dtype=np.float32)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    weights[y == 1] = 0.5 * len(y) / max(n_pos, 1)
    weights[y == 0] = 0.5 * len(y) / max(n_neg, 1)
    if args.model_type == "logreg":
        model = LogisticRegression(class_weight="balanced", max_iter=300, solver="liblinear", random_state=int(args.random_state))
        model.fit(X, y)
    elif args.model_type == "rf":
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=int(args.random_state))
        model.fit(X, y)
    else:
        model = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.01, random_state=int(args.random_state))
        model.fit(X, y, sample_weight=weights)
    p = model.predict_proba(X)[:, 1]
    return model, {
        "model_type": str(args.model_type),
        "train_auc": round(float(roc_auc_score(y, p)), 6),
        "train_ap": round(float(average_precision_score(y, p)), 6),
    }


def _edge_probability(edge, members, samples, counts, mean, model, *, max_pairs: int):
    left = np.asarray(members[int(edge["source"])], dtype=np.int64)
    right = np.asarray(members[int(edge["target"])], dtype=np.int64)
    if len(left) == 0 or len(right) == 0:
        return 0.0, {"sample_pair_count": 0, "sample_prob_top_mean": 0.0}
    sim = mean[left] @ mean[right].T
    flat = np.argsort(-sim.reshape(-1))[: max(int(max_pairs), 1)]
    rows = []
    for pos in flat.tolist():
        li = int(left[pos // len(right)])
        rj = int(right[pos % len(right)])
        if int(counts[li]) > 0 and int(counts[rj]) > 0:
            rows.append(_row_features(samples, counts, li, rj, mean))
    if not rows:
        return 0.0, {"sample_pair_count": 0, "sample_prob_top_mean": 0.0}
    prob = model.predict_proba(np.asarray(rows, dtype=np.float32))[:, 1].astype(np.float32)
    top = np.sort(prob)[-min(5, len(prob)) :]
    return float(prob.max()), {"sample_pair_count": int(len(prob)), "sample_prob_top_mean": float(top.mean()), "sample_prob_mean": float(prob.mean())}


def _merge_by_probability(records, base_labels, edge_rows, args, threshold: float, max_size: int):
    uf = _unionfind_from_labels(base_labels)
    forbidden = _build_overlap_forbidden(records)
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


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list, tuple))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--sample-feature-npz", required=True)
    ap.add_argument("--candidate-top-k", type=int, default=100)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weight", type=float, default=0.0)
    ap.add_argument("--edge-pair-topk", type=int, default=32)
    ap.add_argument(
        "--probability-field",
        default="sample_probability",
        choices=["sample_probability", "sample_prob_top_mean", "sample_prob_mean"],
    )
    ap.add_argument("--max-positive-pairs", type=int, default=12000)
    ap.add_argument("--max-negative-pairs", type=int, default=12000)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf", "logreg"])
    ap.add_argument("--random-state", type=int, default=23)
    ap.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90,0.95,0.98")
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
    samples, counts, mean_emb, sample_meta = _load_samples(args.sample_feature_npz, records)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {"cache_version": 1, "dbname": args.dbname, "role": args.role, "iou_thr": 0.5, "min_matches": 1, "min_purity": 0.0, "n_tracklets": len(records), "prediction_rows": int(sum(len(v) for v in pred_by_video.values())), "gt_rows": int(sum(len(v) for v in gt_by_video.values()))}
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
    X_train, y_train, train_info = _build_training(samples, counts, mean_emb, forbidden, max_pos=int(args.max_positive_pairs), max_neg=int(args.max_negative_pairs), random_state=int(args.random_state))
    model, model_info = _fit_model(X_train, y_train, args)

    reps, members = _component_members(base_labels, keep_indices)
    edges, edge_info = _candidate_edges(records, mean_emb, reps, members, candidate_top_k=int(args.candidate_top_k), top_edge_k=int(args.top_edge_k), centroid_weight=float(args.centroid_weight), min_source_size=1, max_source_size=1_000_000, min_target_size=1, max_target_size=1_000_000, forbid_camera_overlap=False, forbid_video_overlap=False)
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
            rows.append({"mode": "sample_positive_edge_verifier", **info, **pair, "uses_anchors": False, "uses_gt_for_training_or_anchors": False, "uses_gt_for_evaluation_only": True})
    rows.sort(key=lambda row: (float(row["tracklet_pair_f1"]), float(row["tracklet_pair_recall"]), float(row["tracklet_pair_precision"])), reverse=True)

    full_rows = []
    for row in rows[: max(int(args.full_top_n), 0)]:
        labels, _ = _merge_by_probability(records, base_labels, edge_rows, args, float(row["sample_probability_threshold"]), int(row["merge_max_component_size"]))
        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        full_rows.append(row)

    result = {"assignment_csv": args.assignment_csv, "base_assignment_components": int(len(raw_to_local)), "sample_meta": sample_meta, "base_pair_metrics": base_pair, "train_info": train_info, "model_info": model_info, "edge_info": edge_info, "top_edges_by_probability": sorted(edge_rows, key=lambda row: float(row["sample_probability"]), reverse=True)[:60], "top": rows[:80], "full_rows": full_rows, "output_admission": output_info, "eval_stats": eval_stats, "uses_anchors": False, "uses_gt_for_training_or_anchors": False, "uses_gt_for_evaluation_only": True}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "base": base_pair, "train_info": train_info, "model_info": model_info, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
