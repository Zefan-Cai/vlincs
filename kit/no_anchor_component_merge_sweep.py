#!/usr/bin/env python
"""Second-stage no-anchor component merge sweep for VLINCS tracklets.

The base resolver is the label-free time-aware agglomeration used by the current
best no-anchor pipeline.  This script then builds component-level candidate
edges from visual evidence only and sweeps conservative merge gates.  Ground
truth is loaded only after prediction for metrics and optional full scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _UnionFind,
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
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _UnionFind,
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


def _row_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _unionfind_from_labels(labels: np.ndarray) -> _UnionFind:
    uf = _UnionFind(int(len(labels)))
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        groups[int(label)].append(int(idx))
    for indices in groups.values():
        head = int(indices[0])
        for idx in indices[1:]:
            uf.merge(head, int(idx))
    return uf


def _component_members(labels: np.ndarray, keep_indices: set[int]) -> tuple[list[int], list[list[int]]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        if idx in keep_indices:
            by_label[int(label)].append(int(idx))
    reps: list[int] = []
    members: list[list[int]] = []
    for _label, indices in sorted(by_label.items(), key=lambda item: min(item[1])):
        reps.append(int(indices[0]))
        members.append(indices)
    return reps, members


def _component_stats(records, reps: list[int], members: list[list[int]]) -> tuple[list[set[str]], list[set[str]], np.ndarray]:
    camera_sets: list[set[str]] = []
    video_sets: list[set[str]] = []
    weights = np.zeros(len(members), dtype=np.float32)
    for comp_idx, indices in enumerate(members):
        cameras = {records[idx].camera for idx in indices}
        videos = {records[idx].video for idx in indices}
        camera_sets.append(cameras)
        video_sets.append(videos)
        weights[comp_idx] = float(sum(max(int(records[idx].n_dets), 1) for idx in indices))
    return camera_sets, video_sets, weights


def _candidate_edges(
    records,
    emb: np.ndarray,
    reps: list[int],
    members: list[list[int]],
    *,
    candidate_top_k: int,
    top_edge_k: int,
    centroid_weight: float,
    min_source_size: int,
    max_source_size: int,
    min_target_size: int,
    max_target_size: int,
    forbid_camera_overlap: bool,
    forbid_video_overlap: bool,
) -> tuple[list[dict[str, float | int]], dict[str, int | float]]:
    x = _l2n(emb.astype(np.float32))
    cents = []
    for indices in members:
        v = x[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        cents.append(v / (np.linalg.norm(v) + 1.0e-9))
    C = np.stack(cents).astype(np.float32)
    centroid_sim = C @ C.T
    np.fill_diagonal(centroid_sim, -2.0)
    sizes = np.asarray([len(indices) for indices in members], dtype=np.int64)
    camera_sets, video_sets, weights = _component_stats(records, reps, members)

    allowed = np.ones_like(centroid_sim, dtype=bool)
    np.fill_diagonal(allowed, False)
    if forbid_camera_overlap:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if camera_sets[i] & camera_sets[j]:
                    allowed[i, j] = False
                    allowed[j, i] = False
    if forbid_video_overlap:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if video_sets[i] & video_sets[j]:
                    allowed[i, j] = False
                    allowed[j, i] = False

    source_ok = (sizes >= int(min_source_size)) & (sizes <= int(max_source_size))
    target_ok = (sizes >= int(min_target_size)) & (sizes <= int(max_target_size))
    if not np.any(source_ok) or not np.any(target_ok):
        return [], {
            "components": int(len(members)),
            "eligible_sources": int(np.count_nonzero(source_ok)),
            "eligible_targets": int(np.count_nonzero(target_ok)),
            "candidate_edges": 0,
        }

    all_scores = np.full_like(centroid_sim, -2.0, dtype=np.float32)
    source_rank_margin = np.zeros(len(members), dtype=np.float32)
    directed_rank: dict[tuple[int, int], int] = {}
    k = min(max(int(candidate_top_k), 1), max(len(members) - 1, 1))
    top_edge_k = max(int(top_edge_k), 1)
    centroid_weight = float(centroid_weight)

    for src, src_members in enumerate(members):
        if not source_ok[src]:
            continue
        mask = allowed[src] & target_ok
        mask[src] = False
        if not np.any(mask):
            continue
        row = np.where(mask, centroid_sim[src], -2.0)
        top = np.argpartition(-row, min(k - 1, len(row) - 1))[:k]
        src_x = x[np.asarray(src_members, dtype=np.int64)]
        scored: list[tuple[float, int]] = []
        for tgt in top.tolist():
            tgt = int(tgt)
            if not mask[tgt]:
                continue
            tgt_x = x[np.asarray(members[tgt], dtype=np.int64)]
            edge_scores = (src_x @ tgt_x.T).reshape(-1)
            if edge_scores.size == 0:
                continue
            kk = min(top_edge_k, int(edge_scores.size))
            top_scores = np.partition(edge_scores, -kk)[-kk:]
            top_mean = float(np.mean(top_scores))
            score = centroid_weight * float(centroid_sim[src, tgt]) + (1.0 - centroid_weight) * top_mean
            all_scores[src, tgt] = float(score)
            scored.append((float(score), tgt))
        scored.sort(reverse=True, key=lambda item: item[0])
        for rank, (_score, tgt) in enumerate(scored, start=1):
            directed_rank[(int(src), int(tgt))] = int(rank)
        if scored:
            best = scored[0][0]
            second = scored[1][0] if len(scored) > 1 else -1.0
            source_rank_margin[src] = float(best - second)

    pair_scores: dict[tuple[int, int], float] = {}
    pair_centroid: dict[tuple[int, int], float] = {}
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            score = max(float(all_scores[i, j]), float(all_scores[j, i]))
            if score <= -1.0:
                continue
            pair_scores[(i, j)] = score
            pair_centroid[(i, j)] = float(centroid_sim[i, j])

    edges: list[dict[str, float | int]] = []
    for (i, j), score in pair_scores.items():
        margin = min(float(source_rank_margin[i]), float(source_rank_margin[j]))
        edges.append(
            {
                "source": int(i),
                "target": int(j),
                "source_rep": int(reps[i]),
                "target_rep": int(reps[j]),
                "source_size": int(sizes[i]),
                "target_size": int(sizes[j]),
                "source_weight": float(weights[i]),
                "target_weight": float(weights[j]),
                "score": float(score),
                "centroid_score": float(pair_centroid[(i, j)]),
                "rank_margin": float(margin),
                "source_rank": int(directed_rank.get((i, j), 1000000)),
                "target_rank": int(directed_rank.get((j, i), 1000000)),
            }
        )
    edges.sort(key=lambda row: float(row["score"]), reverse=True)
    return edges, {
        "components": int(len(members)),
        "eligible_sources": int(np.count_nonzero(source_ok)),
        "eligible_targets": int(np.count_nonzero(target_ok)),
        "candidate_edges": int(len(edges)),
        "candidate_top_k": int(candidate_top_k),
        "top_edge_k": int(top_edge_k),
        "centroid_weight": float(centroid_weight),
        "min_source_size": int(min_source_size),
        "max_source_size": int(max_source_size),
        "min_target_size": int(min_target_size),
        "max_target_size": int(max_target_size),
        "forbid_camera_overlap": bool(forbid_camera_overlap),
        "forbid_video_overlap": bool(forbid_video_overlap),
    }


def _merge_edges(records, base_labels: np.ndarray, edges: list[dict[str, float | int]], args, threshold: float, margin: float):
    uf = _unionfind_from_labels(base_labels)
    forbidden = _build_overlap_forbidden(records)
    accepted = 0
    accepted_preview: list[dict[str, float | int]] = []
    accepted_score_sum = 0.0
    accepted_rank_margin_sum = 0.0
    accepted_mass_proxy_sum = 0.0
    accepted_pair_mass_proxy_sum = 0.0
    accepted_size_product_sum = 0.0
    accepted_min_weight_sum = 0.0
    accepted_max_weight_sum = 0.0
    accepted_source_weight_sum = 0.0
    accepted_target_weight_sum = 0.0
    rejected_threshold = 0
    rejected_margin = 0
    rejected_stale = 0
    rejected_forbidden = 0
    rejected_size = 0
    rejected_rank = 0
    mutual_top_k = int(getattr(args, "mutual_top_k", 0))
    accepted_preview_n = max(int(getattr(args, "accepted_preview_n", 20)), 0)
    for edge in edges:
        if float(edge["score"]) < float(threshold):
            rejected_threshold += 1
            continue
        if float(edge["rank_margin"]) < float(margin):
            rejected_margin += 1
            continue
        if mutual_top_k > 0 and (
            int(edge.get("source_rank", 1000000)) > mutual_top_k
            or int(edge.get("target_rank", 1000000)) > mutual_top_k
        ):
            rejected_rank += 1
            continue
        src = int(edge["source_rep"])
        tgt = int(edge["target_rep"])
        src_root = uf.find(src)
        tgt_root = uf.find(tgt)
        if src_root == tgt_root:
            rejected_stale += 1
            continue
        merged_size = len(uf.members[src_root]) + len(uf.members[tgt_root])
        if merged_size > int(args.max_component_size):
            rejected_size += 1
            continue
        if not uf.can_merge(src, tgt, forbidden, int(args.max_component_size)):
            rejected_forbidden += 1
            continue
        src_weight = float(edge.get("source_weight", max(int(edge.get("source_size", 1)), 1)))
        tgt_weight = float(edge.get("target_weight", max(int(edge.get("target_size", 1)), 1)))
        src_size = int(edge.get("source_size", 1))
        tgt_size = int(edge.get("target_size", 1))
        bridge_mass_proxy = float(np.sqrt(max(src_weight, 1.0) * max(tgt_weight, 1.0)))
        pair_mass_proxy = float(max(src_weight, 1.0) * max(tgt_weight, 1.0))
        accepted_score_sum += float(edge.get("score", 0.0))
        accepted_rank_margin_sum += float(edge.get("rank_margin", 0.0))
        accepted_mass_proxy_sum += bridge_mass_proxy
        accepted_pair_mass_proxy_sum += pair_mass_proxy
        accepted_size_product_sum += float(max(src_size, 1) * max(tgt_size, 1))
        accepted_min_weight_sum += min(src_weight, tgt_weight)
        accepted_max_weight_sum += max(src_weight, tgt_weight)
        accepted_source_weight_sum += src_weight
        accepted_target_weight_sum += tgt_weight
        if len(accepted_preview) < accepted_preview_n:
            accepted_preview.append(
                {
                    "accepted_order": int(accepted + 1),
                    "source": int(edge["source"]),
                    "target": int(edge["target"]),
                    "source_rep": int(src),
                    "target_rep": int(tgt),
                    "source_size": int(src_size),
                    "target_size": int(tgt_size),
                    "source_weight": float(src_weight),
                    "target_weight": float(tgt_weight),
                    "pre_merge_source_root_size": int(len(uf.members[src_root])),
                    "pre_merge_target_root_size": int(len(uf.members[tgt_root])),
                    "pre_merge_merged_size": int(merged_size),
                    "score": float(edge.get("score", 0.0)),
                    "centroid_score": float(edge.get("centroid_score", 0.0)),
                    "rank_margin": float(edge.get("rank_margin", 0.0)),
                    "source_rank": int(edge.get("source_rank", 1000000)),
                    "target_rank": int(edge.get("target_rank", 1000000)),
                    "bridge_mass_proxy": float(bridge_mass_proxy),
                    "pair_mass_proxy": float(pair_mass_proxy),
                }
            )
        uf.merge(src, tgt)
        accepted += 1
    labels = uf.labels()
    accepted_denom = float(max(accepted, 1))
    return labels, {
        "merge_threshold": float(threshold),
        "merge_margin": float(margin),
        "merge_accepted": int(accepted),
        "merge_rejected_threshold": int(rejected_threshold),
        "merge_rejected_margin": int(rejected_margin),
        "merge_rejected_stale": int(rejected_stale),
        "merge_rejected_forbidden": int(rejected_forbidden),
        "merge_rejected_size": int(rejected_size),
        "merge_rejected_rank": int(rejected_rank),
        "mutual_top_k": int(mutual_top_k),
        "accepted_preview": accepted_preview,
        "accepted_preview_n": int(accepted_preview_n),
        "merge_score_mean": float(accepted_score_sum / accepted_denom),
        "merge_rank_margin_mean": float(accepted_rank_margin_sum / accepted_denom),
        "merge_mass_proxy_sum": float(accepted_mass_proxy_sum),
        "merge_pair_mass_proxy_sum": float(accepted_pair_mass_proxy_sum),
        "merge_size_product_sum": float(accepted_size_product_sum),
        "merge_min_weight_sum": float(accepted_min_weight_sum),
        "merge_max_weight_sum": float(accepted_max_weight_sum),
        "merge_source_weight_sum": float(accepted_source_weight_sum),
        "merge_target_weight_sum": float(accepted_target_weight_sum),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "max_component_size": int(args.max_component_size),
        "uses_ground_truth": False,
    }


def _row_sort_key(row: dict[str, object], rank_by: str) -> tuple[float, ...]:
    if rank_by == "precision":
        return (
            _row_float(row, "tracklet_pair_precision"),
            _row_float(row, "tracklet_pair_f1"),
            _row_float(row, "tracklet_pair_recall"),
        )
    if rank_by == "recall":
        return (
            _row_float(row, "tracklet_pair_recall"),
            _row_float(row, "tracklet_pair_f1"),
            _row_float(row, "tracklet_pair_precision"),
        )
    if rank_by == "mass_proxy":
        return (
            _row_float(row, "merge_pair_mass_proxy_sum"),
            _row_float(row, "merge_mass_proxy_sum"),
            _row_float(row, "merge_accepted"),
            _row_float(row, "merge_score_mean"),
            _row_float(row, "tracklet_pair_f1"),
        )
    if rank_by == "mass_then_pair":
        return (
            _row_float(row, "merge_pair_mass_proxy_sum"),
            _row_float(row, "tracklet_pair_f1"),
            _row_float(row, "tracklet_pair_precision"),
            _row_float(row, "tracklet_pair_recall"),
        )
    return (
        _row_float(row, "tracklet_pair_f1"),
        _row_float(row, "tracklet_pair_recall"),
        _row_float(row, "tracklet_pair_precision"),
    )


def _self_test() -> None:
    records = [
        SimpleNamespace(seq=10, video="v", camera="c0", start_frame=0, end_frame=5, n_dets=10),
        SimpleNamespace(seq=11, video="v", camera="c1", start_frame=0, end_frame=5, n_dets=20),
        SimpleNamespace(seq=12, video="v", camera="c2", start_frame=10, end_frame=15, n_dets=40),
    ]
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    edges = [
        {
            "source": 0,
            "target": 1,
            "source_rep": 0,
            "target_rep": 1,
            "source_size": 1,
            "target_size": 1,
            "source_weight": 10.0,
            "target_weight": 20.0,
            "score": 0.90,
            "centroid_score": 0.88,
            "rank_margin": 0.10,
            "source_rank": 1,
            "target_rank": 1,
        },
        {
            "source": 1,
            "target": 2,
            "source_rep": 1,
            "target_rep": 2,
            "source_size": 1,
            "target_size": 1,
            "source_weight": 20.0,
            "target_weight": 40.0,
            "score": 0.89,
            "centroid_score": 0.85,
            "rank_margin": 0.10,
            "source_rank": 3,
            "target_rank": 1,
        },
    ]
    args = SimpleNamespace(max_component_size=10, mutual_top_k=1, accepted_preview_n=1)
    merged, info = _merge_edges(records, labels, edges, args, threshold=0.80, margin=0.0)
    assert int(info["merge_accepted"]) == 1, info
    assert int(info["merge_rejected_rank"]) == 1, info
    assert len(info["accepted_preview"]) == 1, info
    assert info["accepted_preview"][0]["source_rep"] == 0, info
    assert _row_float(info, "merge_pair_mass_proxy_sum") == 200.0, info
    assert len(set(merged.tolist())) == 2, merged
    pair_top = {"tracklet_pair_f1": 0.90, "tracklet_pair_precision": 0.91, "tracklet_pair_recall": 0.89, "merge_pair_mass_proxy_sum": 100.0}
    mass_top = {"tracklet_pair_f1": 0.80, "tracklet_pair_precision": 0.82, "tracklet_pair_recall": 0.78, "merge_pair_mass_proxy_sum": 1000.0}
    assert _row_sort_key(mass_top, "mass_proxy") > _row_sort_key(pair_top, "mass_proxy")
    assert _row_sort_key(pair_top, "pair") > _row_sort_key(mass_top, "pair")
    print(json.dumps({"stage": "self_test", "status": "ok"}, sort_keys=True))


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list, tuple))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
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
    ap.add_argument("--candidate-top-k", type=int, default=30)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weights", default="0.2")
    ap.add_argument("--min-source-size", type=int, default=1)
    ap.add_argument("--max-source-size", type=int, default=1000000)
    ap.add_argument("--min-target-size", type=int, default=1)
    ap.add_argument("--max-target-size", type=int, default=1000000)
    ap.add_argument("--forbid-camera-overlap", action="store_true")
    ap.add_argument("--forbid-video-overlap", action="store_true")
    ap.add_argument("--max-component-sizes", default="500")
    ap.add_argument("--mutual-top-ks", default="0")
    ap.add_argument("--accepted-preview-n", type=int, default=20)
    ap.add_argument("--rank-by", default="pair", choices=["pair", "precision", "recall", "mass_proxy", "mass_then_pair"])
    ap.add_argument("--thresholds", default="0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74")
    ap.add_argument("--margins", default="-1.0,0.0,0.02,0.04")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.json:
        raise SystemExit("--json is required unless --self-test is used")

    con = _connect(args.dbname)
    records, emb = _load_tracklets(con, args.role)
    if args.feature_npz:
        emb = _load_feature_npz(
            args.feature_npz,
            records,
            emb,
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
    base_labels, base_info = _time_agglom_resolve(records, emb, cfg)
    base_pred = _labels_to_seq_map(records, base_labels, keep_seqs=keep_seqs)
    base_pair = _pair_metrics([record.seq for record in records], base_pred, gt_by_seq, weight_by_seq)

    rows: list[dict[str, object]] = []
    edge_summaries: list[dict[str, object]] = []
    edges_by_centroid_weight: dict[float, list[dict[str, float | int]]] = {}
    for centroid_weight in _parse_floats(args.centroid_weights):
        reps, members = _component_members(base_labels, keep_indices)
        edges, edge_info = _candidate_edges(
            records,
            emb,
            reps,
            members,
            candidate_top_k=int(args.candidate_top_k),
            top_edge_k=int(args.top_edge_k),
            centroid_weight=float(centroid_weight),
            min_source_size=int(args.min_source_size),
            max_source_size=int(args.max_source_size),
            min_target_size=int(args.min_target_size),
            max_target_size=int(args.max_target_size),
            forbid_camera_overlap=bool(args.forbid_camera_overlap),
            forbid_video_overlap=bool(args.forbid_video_overlap),
        )
        edges_by_centroid_weight[float(centroid_weight)] = edges
        edge_summaries.append(edge_info)
        for max_component_size in _parse_ints(args.max_component_sizes):
            args.max_component_size = int(max_component_size)
            for mutual_top_k in _parse_ints(args.mutual_top_ks):
                args.mutual_top_k = int(mutual_top_k)
                for threshold in _parse_floats(args.thresholds):
                    for margin in _parse_floats(args.margins):
                        labels, merge_info = _merge_edges(records, base_labels, edges, args, threshold, margin)
                        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
                        pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
                        row = {
                            "mode": "component_merge",
                            "centroid_weight": float(centroid_weight),
                            **edge_info,
                            **merge_info,
                            **pair,
                            "uses_anchors": False,
                            "uses_gt_for_training_or_anchors": False,
                            "uses_gt_for_evaluation_only": True,
                        }
                        rows.append(row)

    rows.sort(key=lambda row: _row_sort_key(row, str(args.rank_by)), reverse=True)

    full_rows = []
    for row in rows[: max(int(args.full_top_n), 0)]:
        full_edges = edges_by_centroid_weight[float(row["centroid_weight"])]
        args.max_component_size = int(row["max_component_size"])
        args.mutual_top_k = int(row.get("mutual_top_k", 0))
        labels, _merge_info = _merge_edges(
            records,
            base_labels,
            full_edges,
            args,
            float(row["merge_threshold"]),
            float(row["merge_margin"]),
        )
        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update(
            {
                "full_idf1": full.get("idf1"),
                "full_hota": full.get("hota"),
                "full_assa": full.get("assa"),
                "full_det_re": full.get("det_re"),
                "full_det_pr": full.get("det_pr"),
                "full": full,
            }
        )
        full_rows.append(row)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "resolve_config": cfg.__dict__,
        "base_resolve_info": base_info,
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "edge_summaries": edge_summaries,
        "rank_by": str(args.rank_by),
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
    print(json.dumps({"base": base_pair, "best": rows[0] if rows else None, "json": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
