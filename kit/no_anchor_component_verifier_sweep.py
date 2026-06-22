#!/usr/bin/env python
"""No-anchor component-edge verifier sweep for VLINCS.

The current no-anchor graph retrieves many true split candidates, but raw
similarity cannot safely choose them.  This script trains a verifier on pseudo
labels only:

- pseudo positives: stable reciprocal component edges across multiple feature
  views;
- pseudo negatives: cannot-link edges and low-consensus candidate edges.

Ground truth labels are loaded only after prediction for pair/full metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_component_merge_sweep import _candidate_edges, _component_members, _unionfind_from_labels
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
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
        _time_agglom_resolve,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_component_merge_sweep import _candidate_edges, _component_members, _unionfind_from_labels
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_resolve_sweep import (
        ResolveConfig,
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
        _time_agglom_resolve,
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _parse_view(text: str) -> tuple[str, str, float]:
    parts = str(text).split(":")
    if len(parts) == 2:
        name, path = parts
        weight = 1.0
    elif len(parts) == 3:
        name, path, weight_text = parts
        weight = float(weight_text)
    else:
        raise ValueError(f"bad --view {text!r}; expected name:path[:weight]")
    if not name:
        raise ValueError(f"view name is empty in {text!r}")
    return name, path, float(weight)


def _load_npz_aligned(path: str, records, *, weight: float = 1.0) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    seqs = [int(seq) for seq in data["seqs"].tolist()]
    features = data["features"].astype(np.float32)
    by_seq = {seq: idx for idx, seq in enumerate(seqs)}
    order = []
    for record in records:
        seq = int(record.seq)
        if seq not in by_seq:
            raise ValueError(f"{path} missing seq={seq}")
        order.append(by_seq[seq])
    return _l2n(features[np.asarray(order, dtype=np.int64)].astype(np.float32)) * float(weight)


def _component_centroids(x: np.ndarray, members: list[list[int]]) -> np.ndarray:
    x = _l2n(x.astype(np.float32))
    cents = []
    for indices in members:
        v = x[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        cents.append(v / (np.linalg.norm(v) + 1.0e-9))
    return np.stack(cents).astype(np.float32)


def _rank_matrix(sim: np.ndarray) -> np.ndarray:
    order = np.argsort(-sim, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(order.shape[1], dtype=np.int32)[None, :] + 1
    return ranks


def _component_sets(records, members: list[list[int]]):
    camera_sets = []
    video_sets = []
    starts = []
    ends = []
    det_weights = []
    areas = []
    confs = []
    for indices in members:
        camera_sets.append({records[idx].camera for idx in indices})
        video_sets.append({records[idx].video for idx in indices})
        starts.append(min(int(records[idx].start_abs_ms or 0) for idx in indices))
        ends.append(max(int(records[idx].end_abs_ms or records[idx].start_abs_ms or 0) for idx in indices))
        det_weights.append(sum(max(int(records[idx].n_dets), 1) for idx in indices))
        areas.append(float(np.mean([float(records[idx].width) * float(records[idx].height) for idx in indices])))
        confs.append(float(np.mean([float(records[idx].avg_conf) for idx in indices])))
    return camera_sets, video_sets, np.asarray(starts), np.asarray(ends), np.asarray(det_weights), np.asarray(areas), np.asarray(confs)


def _centroid_candidate_edges(records, emb: np.ndarray, reps: list[int], members: list[list[int]], candidate_top_k: int):
    C = _component_centroids(emb, members)
    sim = (C @ C.T).astype(np.float32)
    np.fill_diagonal(sim, -2.0)
    n = int(sim.shape[0])
    k = min(max(int(candidate_top_k), 1), max(n - 1, 1))
    sizes = np.asarray([len(indices) for indices in members], dtype=np.int64)
    weights = np.asarray([sum(max(int(records[idx].n_dets), 1) for idx in indices) for indices in members], dtype=np.float32)
    best: dict[tuple[int, int], dict[str, float | int]] = {}
    directed_rank: dict[tuple[int, int], int] = {}
    directed_second: dict[int, float] = {}
    for src in range(n):
        row = sim[src]
        top = np.argpartition(-row, min(k - 1, n - 1))[:k]
        top = sorted([int(t) for t in top if int(t) != src], key=lambda t: float(row[t]), reverse=True)
        second = float(row[top[1]]) if len(top) > 1 else -1.0
        directed_second[src] = second
        for rank, tgt in enumerate(top, start=1):
            directed_rank[(src, tgt)] = int(rank)
            a, b = (src, tgt) if src < tgt else (tgt, src)
            key = (a, b)
            score = float(row[tgt])
            if key not in best or score > float(best[key]["score"]):
                best[key] = {
                    "source": int(a),
                    "target": int(b),
                    "source_rep": int(reps[a]),
                    "target_rep": int(reps[b]),
                    "source_size": int(sizes[a]),
                    "target_size": int(sizes[b]),
                    "source_weight": float(weights[a]),
                    "target_weight": float(weights[b]),
                    "score": float(score),
                    "second_score": float(second),
                    "margin": float(score - second),
                    "centroid_score": float(sim[a, b]),
                    "top_edge_mean": float(score),
                    "rank_margin": float(score - second),
                }
    edges = []
    for (a, b), edge in best.items():
        edge["source_rank"] = int(directed_rank.get((a, b), 1_000_000))
        edge["target_rank"] = int(directed_rank.get((b, a), 1_000_000))
        edge["rank_margin"] = float(edge["score"]) - max(float(directed_second.get(a, -1.0)), float(directed_second.get(b, -1.0)))
        edges.append(edge)
    edges.sort(key=lambda row: float(row["score"]), reverse=True)
    return edges, {
        "components": int(n),
        "eligible_sources": int(n),
        "eligible_targets": int(n),
        "candidate_edges": int(len(edges)),
        "candidate_top_k": int(candidate_top_k),
        "top_edge_k": 0,
        "centroid_weight": 1.0,
        "candidate_centroid_only": True,
    }


def _forbidden_candidate_pairs(records, members: list[list[int]], edges) -> set[tuple[int, int]]:
    forbidden = _build_overlap_forbidden(records)
    out: set[tuple[int, int]] = set()
    member_sets = [set(indices) for indices in members]
    for edge in edges:
        i = int(edge["source"])
        j = int(edge["target"])
        key = (i, j) if i < j else (j, i)
        if key in out:
            continue
        mi = member_sets[i]
        mj = member_sets[j]
        small, large = (mi, mj) if len(mi) <= len(mj) else (mj, mi)
        for node in small:
            if forbidden[node] & large:
                out.add(key)
                break
    return out


def _edge_feature_table(records, members, edges, view_embeddings: dict[str, np.ndarray]):
    camera_sets, video_sets, starts, ends, det_weights, areas, confs = _component_sets(records, members)
    forbidden_pairs = _forbidden_candidate_pairs(records, members, edges)
    view_sims: dict[str, np.ndarray] = {}
    view_ranks: dict[str, np.ndarray] = {}
    for name, emb in view_embeddings.items():
        C = _component_centroids(emb, members)
        sim = (C @ C.T).astype(np.float32)
        np.fill_diagonal(sim, -2.0)
        view_sims[name] = sim
        view_ranks[name] = _rank_matrix(sim)

    rows: list[dict[str, float | int]] = []
    feature_names: list[str] = []
    feature_rows: list[list[float]] = []
    for edge in edges:
        a = int(edge["source"])
        b = int(edge["target"])
        key = (a, b) if a < b else (b, a)
        overlap_start = max(int(starts[a]), int(starts[b]))
        overlap_end = min(int(ends[a]), int(ends[b]))
        abs_gap = max(0, max(int(starts[a]) - int(ends[b]), int(starts[b]) - int(ends[a])))
        base: dict[str, float | int] = {
            "source": a,
            "target": b,
            "source_rep": int(edge["source_rep"]),
            "target_rep": int(edge["target_rep"]),
            "score": float(edge["score"]),
            "centroid_score": float(edge["centroid_score"]),
            "rank_margin": float(edge["rank_margin"]),
            "source_rank": int(edge.get("source_rank", 1_000_000)),
            "target_rank": int(edge.get("target_rank", 1_000_000)),
            "source_size": int(edge["source_size"]),
            "target_size": int(edge["target_size"]),
            "source_weight": float(edge["source_weight"]),
            "target_weight": float(edge["target_weight"]),
            "camera_overlap": int(bool(camera_sets[a] & camera_sets[b])),
            "video_overlap": int(bool(video_sets[a] & video_sets[b])),
            "time_overlap": int(overlap_start <= overlap_end and overlap_start > 0),
            "abs_gap_log": float(np.log1p(abs_gap)),
            "area_ratio_sim": float(np.exp(-abs(np.log(max(areas[a], 1.0) / max(areas[b], 1.0))))),
            "conf_min": float(min(confs[a], confs[b])),
            "conf_mean": float(0.5 * (confs[a] + confs[b])),
            "is_forbidden": int(key in forbidden_pairs),
        }
        votes_top1 = 0
        votes_top3 = 0
        votes_top5 = 0
        votes_top10 = 0
        for name in sorted(view_embeddings):
            sim = float(view_sims[name][a, b])
            ra = int(view_ranks[name][a, b])
            rb = int(view_ranks[name][b, a])
            rmax = max(ra, rb)
            base[f"{name}_sim"] = sim
            base[f"{name}_rank_min"] = min(ra, rb)
            base[f"{name}_rank_max"] = rmax
            if rmax <= 1:
                votes_top1 += 1
            if rmax <= 3:
                votes_top3 += 1
            if rmax <= 5:
                votes_top5 += 1
            if rmax <= 10:
                votes_top10 += 1
        base["votes_top1"] = int(votes_top1)
        base["votes_top3"] = int(votes_top3)
        base["votes_top5"] = int(votes_top5)
        base["votes_top10"] = int(votes_top10)
        if not feature_names:
            feature_names = [
                key
                for key, value in base.items()
                if key not in {"source", "target", "source_rep", "target_rep"}
                and isinstance(value, (int, float))
            ]
        rows.append(base)
        feature_rows.append([float(base[name]) for name in feature_names])
    return rows, np.asarray(feature_rows, dtype=np.float32), feature_names


def _pseudo_labels(rows, args) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.full(len(rows), -1, dtype=np.int8)
    for idx, row in enumerate(rows):
        forbidden = int(row["is_forbidden"]) > 0
        votes5 = int(row["votes_top5"])
        votes10 = int(row["votes_top10"])
        score = float(row["score"])
        rmax = max(int(row["source_rank"]), int(row["target_rank"]))
        if (not forbidden) and votes5 >= int(args.pos_min_votes_top5) and score >= float(args.pos_min_score) and rmax <= int(args.pos_max_primary_rank):
            labels[idx] = 1
        elif forbidden or votes10 <= int(args.neg_max_votes_top10) or score <= float(args.neg_max_score):
            labels[idx] = 0
    counts = Counter(labels.tolist())
    return labels, {
        "pseudo_positive": int(counts.get(1, 0)),
        "pseudo_negative": int(counts.get(0, 0)),
        "pseudo_unlabeled": int(counts.get(-1, 0)),
    }


def _fit_model(X: np.ndarray, y: np.ndarray, args):
    keep = y >= 0
    X_train = X[keep]
    y_train = y[keep].astype(np.int64)
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError(f"need both pseudo classes, got {Counter(y_train.tolist())}")
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    weights = np.ones(len(y_train), dtype=np.float32)
    if n_pos > 0 and n_neg > 0:
        weights[y_train == 1] = 0.5 * len(y_train) / n_pos
        weights[y_train == 0] = 0.5 * len(y_train) / n_neg
    if args.model_type == "logreg":
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=300,
            solver="liblinear",
            random_state=int(args.random_state),
        )
        model.fit(X_train, y_train)
    elif args.model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=int(args.random_state),
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
    else:
        model = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=int(args.random_state),
        )
        model.fit(X_train, y_train, sample_weight=weights)
    proba = model.predict_proba(X)[:, 1].astype(np.float32)
    train_proba = proba[keep]
    stats = {
        "model_type": str(args.model_type),
        "train_edges": int(len(y_train)),
        "train_positive": n_pos,
        "train_negative": n_neg,
    }
    try:
        stats["pseudo_train_auc"] = round(float(roc_auc_score(y_train, train_proba)), 6)
    except Exception:
        stats["pseudo_train_auc"] = None
    try:
        stats["pseudo_train_ap"] = round(float(average_precision_score(y_train, train_proba)), 6)
    except Exception:
        stats["pseudo_train_ap"] = None
    return model, proba, stats


def _merge_by_probability(records, base_labels, edge_rows, probabilities: np.ndarray, args, threshold: float, min_votes_top5: int):
    uf = _unionfind_from_labels(base_labels)
    forbidden = _build_overlap_forbidden(records)
    order = np.argsort(-probabilities)
    accepted = 0
    rejected_threshold = 0
    rejected_votes = 0
    rejected_forbidden = 0
    rejected_size = 0
    rejected_stale = 0
    for idx in order.tolist():
        prob = float(probabilities[idx])
        if prob < float(threshold):
            rejected_threshold += 1
            continue
        row = edge_rows[idx]
        if int(row["votes_top5"]) < int(min_votes_top5):
            rejected_votes += 1
            continue
        a = int(row["source_rep"])
        b = int(row["target_rep"])
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            rejected_stale += 1
            continue
        if len(uf.members[ra]) + len(uf.members[rb]) > int(args.max_component_size):
            rejected_size += 1
            continue
        if not uf.can_merge(a, b, forbidden, int(args.max_component_size)):
            rejected_forbidden += 1
            continue
        uf.merge(a, b)
        accepted += 1
    labels = uf.labels()
    return labels, {
        "verifier_threshold": float(threshold),
        "verifier_min_votes_top5": int(min_votes_top5),
        "verifier_accepted": int(accepted),
        "verifier_rejected_threshold": int(rejected_threshold),
        "verifier_rejected_votes": int(rejected_votes),
        "verifier_rejected_forbidden": int(rejected_forbidden),
        "verifier_rejected_size": int(rejected_size),
        "verifier_rejected_stale": int(rejected_stale),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
    }


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list, tuple))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _default_admission_args(args):
    return SimpleNamespace(
        output_min_dets=int(args.output_min_dets),
        output_min_conf=float(args.output_min_conf),
        output_min_area=float(args.output_min_area),
        output_min_quality=float(args.output_min_quality),
        output_min_area_by_video=str(args.output_min_area_by_video),
        output_drop_area_quantile=0.0,
        output_drop_area_quantile_by_video="",
        output_drop_quality_quantile=0.0,
        output_drop_quality_quantile_by_video="",
        output_auto_anomaly_admission=False,
        output_auto_anomaly_metric="quality",
        output_auto_anomaly_quantile=0.75,
        output_auto_anomaly_area_ratio=0.60,
        output_auto_anomaly_quality_mad=1.0,
        output_auto_anomaly_min_video_tracklets=20,
        output_auto_anomaly_max_videos=3,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument(
        "--assignment-csv",
        default="",
        help="optional no-anchor assignment CSV to use as the starting component graph instead of time-agglom",
    )
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--view", action="append", default=[], help="extra verifier view name:path[:weight], or name:db[:weight]")
    ap.add_argument("--theta", type=float, default=0.014)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--exclude-same", default="camera")
    ap.add_argument("--temporal-bonus", type=float, default=0.005)
    ap.add_argument("--time-window-ms", type=int, default=1000)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--candidate-top-k", type=int, default=50)
    ap.add_argument("--limit-candidate-edges", type=int, default=0)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weight", type=float, default=0.0)
    ap.add_argument("--candidate-centroid-only", action="store_true")
    ap.add_argument("--pos-min-votes-top5", type=int, default=2)
    ap.add_argument("--pos-min-score", type=float, default=0.70)
    ap.add_argument("--pos-max-primary-rank", type=int, default=10)
    ap.add_argument("--neg-max-votes-top10", type=int, default=0)
    ap.add_argument("--neg-max-score", type=float, default=0.58)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf", "logreg"])
    ap.add_argument("--random-state", type=int, default=17)
    ap.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--min-votes-top5-grid", default="0,1,2")
    ap.add_argument("--max-component-size", type=int, default=500)
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    print(json.dumps({"stage": "loaded_tracklets", "n_tracklets": len(records), "db_dim": int(db_emb.shape[1])}), flush=True)
    primary_emb = _load_feature_npz(
        args.feature_npz,
        records,
        db_emb,
        concat_db=bool(args.concat_db_embedding),
        db_weight=float(args.db_weight),
        feature_weight=float(args.feature_weight),
    )
    print(json.dumps({"stage": "loaded_primary_feature", "dim": int(primary_emb.shape[1])}), flush=True)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    print(json.dumps({"stage": "loaded_predictions", "videos": len(pred_by_video), "rows": int(sum(len(value) for value in pred_by_video.values()))}), flush=True)
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
    print(json.dumps({"stage": "loaded_eval_cache", "eval_tracklets": int(len(gt_by_seq))}), flush=True)
    keep_seqs, output_info = _output_keep_seqs(records, _default_admission_args(args))
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}

    cfg = ResolveConfig(
        mode="time_agglom",
        theta=float(args.theta),
        top_k=int(args.top_k),
        min_dets=int(args.min_dets),
        exclude_same=str(args.exclude_same),
        temporal_bonus=float(args.temporal_bonus),
        time_window_ms=int(args.time_window_ms),
    )
    if str(args.assignment_csv).strip():
        pred_input = _load_assignment_labels(str(args.assignment_csv), str(args.pred_col))
        base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
        keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
        keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
        base_info = {
            "mode": "assignment_base",
            "assignment_csv": str(args.assignment_csv),
            "assignment_rows": int(len(pred_input)),
            "assignment_components": int(len(raw_to_local)),
            "uses_ground_truth": False,
        }
    else:
        base_labels, base_info = _time_agglom_resolve(records, primary_emb, cfg)
    base_pred = _labels_to_seq_map(records, base_labels, keep_seqs=keep_seqs)
    base_pair = _pair_metrics([record.seq for record in records], base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base_resolved", **base_pair}), flush=True)
    reps, members = _component_members(base_labels, keep_indices)
    if bool(args.candidate_centroid_only):
        edges, edge_info = _centroid_candidate_edges(records, primary_emb, reps, members, int(args.candidate_top_k))
    else:
        edges, edge_info = _candidate_edges(
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
    if int(args.limit_candidate_edges) > 0 and len(edges) > int(args.limit_candidate_edges):
        edges = edges[: int(args.limit_candidate_edges)]
        edge_info["candidate_edges_limited"] = int(len(edges))
    print(json.dumps({"stage": "candidate_edges", **edge_info}), flush=True)

    view_embeddings: dict[str, np.ndarray] = {"primary": primary_emb.astype(np.float32)}
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        if path.lower() == "db":
            view_embeddings[name] = _l2n(db_emb.astype(np.float32)) * float(weight)
        else:
            view_embeddings[name] = _load_npz_aligned(path, records, weight=float(weight))
    print(json.dumps({"stage": "loaded_views", "views": sorted(view_embeddings.keys())}), flush=True)

    edge_rows, X, feature_names = _edge_feature_table(records, members, edges, view_embeddings)
    pseudo_y, pseudo_counts = _pseudo_labels(edge_rows, args)
    print(json.dumps({"stage": "built_edge_features", "edges": int(len(edge_rows)), "features": int(X.shape[1]), **pseudo_counts}), flush=True)
    model, probabilities, model_stats = _fit_model(X, pseudo_y, args)
    print(json.dumps({"stage": "fit_verifier", **model_stats}), flush=True)
    for row, prob in zip(edge_rows, probabilities):
        row["verifier_probability"] = float(prob)
    for idx, row in enumerate(edge_rows):
        row["pseudo_label"] = int(pseudo_y[idx])

    rows: list[dict[str, object]] = []
    for threshold in _parse_floats(args.thresholds):
        for min_votes in _parse_ints(args.min_votes_top5_grid):
            labels, info = _merge_by_probability(records, base_labels, edge_rows, probabilities, args, threshold, min_votes)
            pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
            pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
            rows.append(
                {
                    "mode": "component_verifier",
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
            edge_rows,
            probabilities,
            args,
            float(row["verifier_threshold"]),
            int(row["verifier_min_votes_top5"]),
        )
        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full"] = full
        full_rows.append(row)

    edge_preview = sorted(edge_rows, key=lambda row: float(row["verifier_probability"]), reverse=True)[:50]
    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "views": sorted(view_embeddings.keys()),
        "resolve_config": cfg.__dict__,
        "base_resolve_info": base_info,
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "edge_info": edge_info,
        "feature_names": feature_names,
        "pseudo_counts": pseudo_counts,
        "model_stats": model_stats,
        "top_edges_by_probability": edge_preview,
        "top": rows[:50],
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
    print(json.dumps({"base": base_pair, "pseudo_counts": pseudo_counts, "model_stats": model_stats, "best": rows[0] if rows else None, "json": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
