#!/usr/bin/env python
"""Self-play component merge verifier for an existing no-anchor assignment.

The script creates pseudo-positive component edges by splitting current
assignment components without using identity labels.  It uses cannot-link and
low-consensus candidate edges as pseudo negatives, trains an edge verifier, and
then scores real inter-component merge candidates.  GT is loaded only after
prediction for pair/full metrics.
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
    from kit.no_anchor_component_merge_sweep import _candidate_edges
    from kit.no_anchor_component_verifier_sweep import (
        _edge_feature_table,
        _fit_model,
        _load_npz_aligned,
        _merge_by_probability,
        _parse_floats,
        _parse_ints,
        _parse_view,
    )
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
    from no_anchor_component_merge_sweep import _candidate_edges
    from no_anchor_component_verifier_sweep import (
        _edge_feature_table,
        _fit_model,
        _load_npz_aligned,
        _merge_by_probability,
        _parse_floats,
        _parse_ints,
        _parse_view,
    )
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


def _component_members(labels: np.ndarray, keep_indices: set[int]) -> tuple[list[int], list[list[int]]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        if idx in keep_indices:
            by_label[int(label)].append(int(idx))
    reps = []
    members = []
    for _label, indices in sorted(by_label.items(), key=lambda item: min(item[1])):
        reps.append(int(indices[0]))
        members.append([int(idx) for idx in indices])
    return reps, members


def _component_conflict_rate(indices: list[int], forbidden: list[set[int]]) -> float:
    if len(indices) < 2:
        return 0.0
    idx_set = set(indices)
    conflicts = 0
    possible = 0
    for pos, idx in enumerate(indices):
        rest = indices[pos + 1 :]
        possible += len(rest)
        conflicts += len(forbidden[int(idx)] & idx_set)
    return float(conflicts / max(possible, 1))


def _split_groups(records, indices: list[int], *, min_part_size: int) -> list[tuple[list[int], list[int], str]]:
    out: list[tuple[list[int], list[int], str]] = []
    by_camera: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        by_camera[str(records[idx].camera)].append(int(idx))
    camera_parts = sorted(by_camera.values(), key=len, reverse=True)
    if len(camera_parts) >= 2 and len(camera_parts[0]) >= min_part_size:
        left = list(camera_parts[0])
        right = [idx for part in camera_parts[1:] for idx in part]
        if len(right) >= min_part_size:
            out.append((left, right, "camera_largest_vs_rest"))

    by_video: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        by_video[str(records[idx].video)].append(int(idx))
    video_parts = sorted(by_video.values(), key=len, reverse=True)
    if len(video_parts) >= 2 and len(video_parts[0]) >= min_part_size:
        left = list(video_parts[0])
        right = [idx for part in video_parts[1:] for idx in part]
        if len(right) >= min_part_size:
            out.append((left, right, "video_largest_vs_rest"))

    ordered = sorted(indices, key=lambda idx: (records[idx].start_abs_ms or 0, records[idx].start_frame, idx))
    mid = len(ordered) // 2
    if mid >= min_part_size and len(ordered) - mid >= min_part_size:
        out.append((ordered[:mid], ordered[mid:], "temporal_halves"))

    even = ordered[::2]
    odd = ordered[1::2]
    if len(even) >= min_part_size and len(odd) >= min_part_size:
        out.append((even, odd, "even_odd"))
    return out


def _centroid_edge(records, emb: np.ndarray, members: list[list[int]], source: int, target: int, *, reason: str) -> dict[str, object]:
    x = _l2n(emb.astype(np.float32))
    a = np.asarray(members[int(source)], dtype=np.int64)
    b = np.asarray(members[int(target)], dtype=np.int64)
    ca = x[a].mean(axis=0)
    cb = x[b].mean(axis=0)
    ca = ca / (np.linalg.norm(ca) + 1.0e-9)
    cb = cb / (np.linalg.norm(cb) + 1.0e-9)
    sim = float(np.dot(ca, cb))
    return {
        "source": int(source),
        "target": int(target),
        "source_rep": int(members[int(source)][0]),
        "target_rep": int(members[int(target)][0]),
        "source_size": int(len(members[int(source)])),
        "target_size": int(len(members[int(target)])),
        "source_weight": float(sum(max(int(records[idx].n_dets), 1) for idx in members[int(source)])),
        "target_weight": float(sum(max(int(records[idx].n_dets), 1) for idx in members[int(target)])),
        "score": sim,
        "centroid_score": sim,
        "rank_margin": 0.0,
        "source_rank": 1,
        "target_rank": 1,
        "pseudo_reason": reason,
    }


def _selfplay_positive_edges(records, emb, base_members, forbidden, args):
    pseudo_members: list[list[int]] = []
    pseudo_edges: list[dict[str, object]] = []
    reason_counts = Counter()
    skipped_conflict = 0
    skipped_size = 0
    for indices in base_members:
        if len(indices) < int(args.min_positive_component_size) or len(indices) > int(args.max_positive_component_size):
            skipped_size += 1
            continue
        conflict_rate = _component_conflict_rate(indices, forbidden)
        if conflict_rate > float(args.max_positive_conflict_rate):
            skipped_conflict += 1
            continue
        for left, right, reason in _split_groups(records, indices, min_part_size=int(args.min_positive_part_size)):
            if len(pseudo_edges) >= int(args.max_positive_edges):
                break
            source = len(pseudo_members)
            pseudo_members.append(left)
            target = len(pseudo_members)
            pseudo_members.append(right)
            pseudo_edges.append(_centroid_edge(records, emb, pseudo_members, source, target, reason=reason))
            reason_counts[reason] += 1
        if len(pseudo_edges) >= int(args.max_positive_edges):
            break
    return pseudo_members, pseudo_edges, {
        "positive_edges": int(len(pseudo_edges)),
        "positive_reasons": dict(sorted(reason_counts.items())),
        "skipped_positive_components_size": int(skipped_size),
        "skipped_positive_components_conflict": int(skipped_conflict),
    }


def _sample_training_rows(pos_X, neg_X, *, max_neg_per_pos: float, random_state: int):
    n_pos = int(pos_X.shape[0])
    if n_pos <= 0:
        raise RuntimeError("no self-play positive edges were generated")
    max_neg = max(int(round(n_pos * float(max_neg_per_pos))), 1)
    rng = np.random.default_rng(int(random_state))
    if int(neg_X.shape[0]) > max_neg:
        choice = rng.choice(int(neg_X.shape[0]), size=max_neg, replace=False)
        neg_X = neg_X[choice]
    X = np.concatenate([pos_X, neg_X], axis=0)
    y = np.concatenate([np.ones(len(pos_X), dtype=np.int8), np.zeros(len(neg_X), dtype=np.int8)])
    return X, y


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
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--view", action="append", default=[])
    ap.add_argument("--candidate-top-k", type=int, default=50)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weight", type=float, default=0.0)
    ap.add_argument("--limit-candidate-edges", type=int, default=0)
    ap.add_argument("--min-positive-component-size", type=int, default=4)
    ap.add_argument("--max-positive-component-size", type=int, default=500)
    ap.add_argument("--min-positive-part-size", type=int, default=2)
    ap.add_argument("--max-positive-conflict-rate", type=float, default=0.02)
    ap.add_argument("--max-positive-edges", type=int, default=600)
    ap.add_argument("--neg-max-score", type=float, default=0.58)
    ap.add_argument("--neg-max-votes-top10", type=int, default=0)
    ap.add_argument("--max-neg-per-pos", type=float, default=8.0)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf", "logreg"])
    ap.add_argument("--random-state", type=int, default=17)
    ap.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--min-votes-top5-grid", default="0,1,2")
    ap.add_argument("--max-component-size", type=int, default=500)
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
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    primary_emb = _load_feature_npz(
        args.feature_npz,
        records,
        db_emb,
        concat_db=bool(args.concat_db_embedding),
        db_weight=float(args.db_weight),
        feature_weight=float(args.feature_weight),
    )
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
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_seqs = {int(seq) for seq in keep_seqs}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
    base_pred = _labels_to_seq_map(records, base_labels, keep_seqs=keep_seqs)
    base_pair = _pair_metrics([record.seq for record in records], base_pred, gt_by_seq, weight_by_seq)

    reps, members = _component_members(base_labels, keep_indices)
    candidate_edges, edge_info = _candidate_edges(
        records,
        primary_emb,
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
    if int(args.limit_candidate_edges) > 0 and len(candidate_edges) > int(args.limit_candidate_edges):
        candidate_edges = candidate_edges[: int(args.limit_candidate_edges)]
        edge_info["candidate_edges_limited"] = int(len(candidate_edges))

    view_embeddings: dict[str, np.ndarray] = {"primary": primary_emb.astype(np.float32)}
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        if path.lower() == "db":
            view_embeddings[name] = _l2n(db_emb.astype(np.float32)) * float(weight)
        else:
            view_embeddings[name] = _load_npz_aligned(path, records, weight=float(weight))

    forbidden = _build_overlap_forbidden(records)
    pseudo_members, pseudo_edges, pseudo_info = _selfplay_positive_edges(records, primary_emb, members, forbidden, args)
    pos_rows, pos_X, feature_names = _edge_feature_table(records, pseudo_members, pseudo_edges, view_embeddings)
    real_rows, real_X, real_feature_names = _edge_feature_table(records, members, candidate_edges, view_embeddings)
    if feature_names != real_feature_names:
        raise RuntimeError("training and inference feature names do not match")

    neg_indices = [
        idx
        for idx, row in enumerate(real_rows)
        if int(row["is_forbidden"]) > 0
        or int(row["votes_top10"]) <= int(args.neg_max_votes_top10)
        or float(row["score"]) <= float(args.neg_max_score)
    ]
    if not neg_indices:
        raise RuntimeError("no pseudo-negative component edges were generated")
    neg_X = real_X[np.asarray(neg_indices, dtype=np.int64)]
    X_train, y_train = _sample_training_rows(
        pos_X,
        neg_X,
        max_neg_per_pos=float(args.max_neg_per_pos),
        random_state=int(args.random_state),
    )
    fit_args = SimpleNamespace(model_type=args.model_type, random_state=int(args.random_state))
    model, _train_prob, model_stats = _fit_model(X_train, y_train, fit_args)
    probabilities = model.predict_proba(real_X)[:, 1].astype(np.float32)
    for row, prob in zip(real_rows, probabilities):
        row["verifier_probability"] = float(prob)

    rows: list[dict[str, object]] = []
    for threshold in _parse_floats(args.thresholds):
        for min_votes in _parse_ints(args.min_votes_top5_grid):
            labels, info = _merge_by_probability(records, base_labels, real_rows, probabilities, args, threshold, min_votes)
            pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
            pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
            rows.append(
                {
                    "mode": "selfplay_component_merge",
                    **info,
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

    full_rows = []
    for row in rows[: max(int(args.full_top_n), 0)]:
        labels, _info = _merge_by_probability(
            records,
            base_labels,
            real_rows,
            probabilities,
            args,
            float(row["verifier_threshold"]),
            int(row["verifier_min_votes_top5"]),
        )
        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        full_rows.append(row)

    result = {
        "assignment_csv": args.assignment_csv,
        "base_assignment_components": int(len(raw_to_local)),
        "base_pair_metrics": base_pair,
        "edge_info": edge_info,
        "feature_names": feature_names,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "pseudo_info": {
            **pseudo_info,
            "pseudo_negative_edges": int(len(neg_indices)),
            "train_positive": int(np.sum(y_train == 1)),
            "train_negative": int(np.sum(y_train == 0)),
            "negative_rule": {
                "neg_max_score": float(args.neg_max_score),
                "neg_max_votes_top10": int(args.neg_max_votes_top10),
            },
        },
        "model_stats": model_stats,
        "top_edges_by_probability": sorted(real_rows, key=lambda row: float(row["verifier_probability"]), reverse=True)[:50],
        "top": rows[:80],
        "full_rows": full_rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(
        json.dumps(
            {
                "json": str(out),
                "base": base_pair,
                "pseudo_info": result["pseudo_info"],
                "model_stats": model_stats,
                "best": rows[0] if rows else None,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
