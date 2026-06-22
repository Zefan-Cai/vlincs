#!/usr/bin/env python
"""Video-local temporal relink sweep for no-anchor global-ID assignments.

The current full-score gap is mostly within-video ID continuity, while the
standing assignment already has high tracklet-pair quality.  This script keeps
the input assignment label-free, projects it into per-video identity nodes, and
links temporally adjacent fragments in the same video using only no-GT evidence:
appearance similarity, time gap, and bbox endpoint distance.

No anchors or identity GT are used to build candidates or select edges.  GT is
loaded only after prediction for pair/full metrics.
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


class _NodeUF:
    def __init__(self, n: int, node_intervals: list[list[tuple[float, float]]], node_sizes: list[int]):
        self.parent = list(range(n))
        self.members = [[i] for i in range(n)]
        self.intervals = [list(items) for items in node_intervals]
        self.sizes = [int(size) for size in node_sizes]

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    @staticmethod
    def _overlaps(a: tuple[float, float], b: tuple[float, float], slack_ms: float) -> bool:
        return max(float(a[0]), float(b[0])) <= min(float(a[1]), float(b[1])) + float(slack_ms)

    def can_merge(self, a: int, b: int, *, max_tracklets: int, overlap_slack_ms: float) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if int(max_tracklets) > 0 and self.sizes[ra] + self.sizes[rb] > int(max_tracklets):
            return False
        for ia in self.intervals[ra]:
            for ib in self.intervals[rb]:
                if self._overlaps(ia, ib, float(overlap_slack_ms)):
                    return False
        return True

    def merge(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].extend(self.members[rb])
        self.intervals[ra].extend(self.intervals[rb])
        self.sizes[ra] += self.sizes[rb]
        return True


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


def _video_nodes(records, base_labels: np.ndarray, keep_indices: set[int], emb: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    key_to_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        key_to_indices[(str(records[idx].video), int(base_labels[idx]))].append(int(idx))

    nodes: list[dict[str, object]] = []
    node_label_by_idx = np.full(len(records), -1, dtype=np.int64)
    x = _l2n(emb.astype(np.float32))
    for node_id, ((video, base_label), indices) in enumerate(sorted(key_to_indices.items())):
        starts = [float(records[idx].start_abs_ms) for idx in indices]
        ends = [float(records[idx].end_abs_ms) for idx in indices]
        first_idx = min(indices, key=lambda idx: (float(records[idx].start_abs_ms), int(idx)))
        last_idx = max(indices, key=lambda idx: (float(records[idx].end_abs_ms), int(idx)))
        center = x[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        center = center / (np.linalg.norm(center) + 1.0e-9)
        mean_height = float(np.mean([max(float(records[idx].height), 1.0) for idx in indices]))
        mean_width = float(np.mean([max(float(records[idx].width), 1.0) for idx in indices]))
        internal_overlaps = 0
        for pos, a in enumerate(indices):
            for b in indices[pos + 1 :]:
                if max(float(records[a].start_abs_ms), float(records[b].start_abs_ms)) <= min(
                    float(records[a].end_abs_ms), float(records[b].end_abs_ms)
                ):
                    internal_overlaps += 1
        for idx in indices:
            node_label_by_idx[int(idx)] = int(node_id)
        nodes.append(
            {
                "node_id": int(node_id),
                "video": video,
                "base_label": int(base_label),
                "indices": [int(idx) for idx in indices],
                "tracklets": int(len(indices)),
                "total_dets": int(sum(int(records[idx].n_dets) for idx in indices)),
                "start_abs_ms": float(min(starts)),
                "end_abs_ms": float(max(ends)),
                "first_cx": float(records[first_idx].first_cx),
                "first_cy": float(records[first_idx].first_cy),
                "first_w": float(records[first_idx].first_width),
                "first_h": float(records[first_idx].first_height),
                "last_cx": float(records[last_idx].last_cx),
                "last_cy": float(records[last_idx].last_cy),
                "last_w": float(records[last_idx].last_width),
                "last_h": float(records[last_idx].last_height),
                "mean_height": mean_height,
                "mean_width": mean_width,
                "internal_overlaps": int(internal_overlaps),
                "embedding": center.astype(np.float32),
            }
        )
    return nodes, node_label_by_idx


def _node_edges(
    nodes: list[dict[str, object]],
    *,
    max_gap_ms: int,
    min_app_sim: float,
    max_center_dist_norm: float,
    skip_dirty_nodes: bool,
    allowed_videos: set[str],
) -> list[dict[str, object]]:
    by_video: dict[str, list[int]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        by_video[str(node["video"])].append(int(idx))
    edges: list[dict[str, object]] = []
    for video, node_ids in by_video.items():
        if allowed_videos and str(video) not in allowed_videos:
            continue
        ordered = sorted(node_ids, key=lambda idx: (float(nodes[idx]["start_abs_ms"]), float(nodes[idx]["end_abs_ms"])))
        for pos, a in enumerate(ordered):
            na = nodes[a]
            if skip_dirty_nodes and int(na["internal_overlaps"]) > 0:
                continue
            for b in ordered[pos + 1 :]:
                nb = nodes[b]
                if skip_dirty_nodes and int(nb["internal_overlaps"]) > 0:
                    continue
                gap = float(nb["start_abs_ms"]) - float(na["end_abs_ms"])
                if gap < 0:
                    continue
                if gap > float(max_gap_ms):
                    break
                app = float(np.dot(na["embedding"], nb["embedding"]))
                if app < float(min_app_sim):
                    continue
                dx = float(nb["first_cx"]) - float(na["last_cx"])
                dy = float(nb["first_cy"]) - float(na["last_cy"])
                scale = max((float(na["mean_height"]) + float(nb["mean_height"])) * 0.5, 1.0)
                dist_norm = float((dx * dx + dy * dy) ** 0.5 / scale)
                if dist_norm > float(max_center_dist_norm):
                    continue
                size_balance = min(float(na["tracklets"]), float(nb["tracklets"])) / max(
                    float(na["tracklets"]), float(nb["tracklets"]), 1.0
                )
                gap_bonus = float(np.exp(-gap / max(float(max_gap_ms), 1.0)))
                geom_bonus = float(np.exp(-dist_norm / max(float(max_center_dist_norm), 1.0e-6)))
                score = float(0.70 * app + 0.16 * gap_bonus + 0.12 * geom_bonus + 0.02 * size_balance)
                edges.append(
                    {
                        "source": int(a),
                        "target": int(b),
                        "video": video,
                        "score": score,
                        "app_sim": app,
                        "gap_ms": gap,
                        "center_dist_norm": dist_norm,
                        "source_tracklets": int(na["tracklets"]),
                        "target_tracklets": int(nb["tracklets"]),
                        "source_internal_overlaps": int(na["internal_overlaps"]),
                        "target_internal_overlaps": int(nb["internal_overlaps"]),
                    }
                )
    edges.sort(key=lambda row: float(row["score"]), reverse=True)
    return edges


def _apply_edges(
    records,
    nodes: list[dict[str, object]],
    node_label_by_idx: np.ndarray,
    edges: list[dict[str, object]],
    *,
    min_score: float,
    max_tracklets_per_chain: int,
    overlap_slack_ms: float,
    max_edges_per_video: int,
) -> tuple[np.ndarray, dict[str, object]]:
    intervals = [[(float(node["start_abs_ms"]), float(node["end_abs_ms"]))] for node in nodes]
    sizes = [int(node["tracklets"]) for node in nodes]
    uf = _NodeUF(len(nodes), intervals, sizes)
    accepted: list[dict[str, object]] = []
    rejected_score = rejected_overlap = rejected_video_cap = rejected_stale = 0
    accepted_by_video: Counter[str] = Counter()
    for edge in edges:
        if float(edge["score"]) < float(min_score):
            rejected_score += 1
            continue
        video = str(edge["video"])
        if int(max_edges_per_video) > 0 and accepted_by_video[video] >= int(max_edges_per_video):
            rejected_video_cap += 1
            continue
        a, b = int(edge["source"]), int(edge["target"])
        if uf.find(a) == uf.find(b):
            rejected_stale += 1
            continue
        if not uf.can_merge(
            a,
            b,
            max_tracklets=int(max_tracklets_per_chain),
            overlap_slack_ms=float(overlap_slack_ms),
        ):
            rejected_overlap += 1
            continue
        uf.merge(a, b)
        accepted_by_video[video] += 1
        accepted.append({k: v for k, v in edge.items() if k != "embedding"})

    root_to_label: dict[int, int] = {}
    next_label = 0
    labels = np.full(len(records), -1, dtype=np.int64)
    for idx, node_label in enumerate(node_label_by_idx.tolist()):
        if int(node_label) < 0:
            labels[idx] = next_label
            next_label += 1
            continue
        root = int(uf.find(int(node_label)))
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[idx] = int(root_to_label[root])
    kept_labels = [int(labels[idx]) for idx, node_label in enumerate(node_label_by_idx.tolist()) if int(node_label) >= 0]
    mean_score = float(np.mean([float(row["score"]) for row in accepted])) if accepted else 0.0
    mean_app = float(np.mean([float(row["app_sim"]) for row in accepted])) if accepted else 0.0
    info = {
        "accepted_edges": int(len(accepted)),
        "accepted_by_video": dict(accepted_by_video),
        "accepted_preview": accepted[:20],
        "rejected_score": int(rejected_score),
        "rejected_overlap": int(rejected_overlap),
        "rejected_video_cap": int(rejected_video_cap),
        "rejected_stale": int(rejected_stale),
        "mean_accepted_score": mean_score,
        "mean_accepted_app_sim": mean_app,
        "components": int(len(set(kept_labels))),
        "largest_component": int(max(Counter(kept_labels).values(), default=0)),
        "policy_score": float(len(accepted) * mean_score * max(mean_app, 0.0)),
    }
    return labels, info


def _base_component_intervals(records, base_labels: np.ndarray, keep_indices: set[int]) -> tuple[list[list[tuple[float, float]]], list[int]]:
    max_label = int(base_labels.max()) if len(base_labels) else -1
    intervals: list[list[tuple[float, float]]] = [[] for _ in range(max_label + 1)]
    sizes = [0 for _ in range(max_label + 1)]
    for idx in keep_indices:
        label = int(base_labels[int(idx)])
        intervals[label].append((float(records[int(idx)].start_abs_ms), float(records[int(idx)].end_abs_ms)))
        sizes[label] += 1
    return intervals, sizes


def _apply_edges_global_components(
    records,
    base_labels: np.ndarray,
    keep_indices: set[int],
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    *,
    min_score: float,
    max_tracklets_per_chain: int,
    overlap_slack_ms: float,
    max_edges_per_video: int,
) -> tuple[np.ndarray, dict[str, object]]:
    intervals, sizes = _base_component_intervals(records, base_labels, keep_indices)
    uf = _NodeUF(len(intervals), intervals, sizes)
    accepted: list[dict[str, object]] = []
    rejected_score = rejected_overlap = rejected_video_cap = rejected_stale = rejected_empty = 0
    accepted_by_video: Counter[str] = Counter()
    for edge in edges:
        if float(edge["score"]) < float(min_score):
            rejected_score += 1
            continue
        video = str(edge["video"])
        if int(max_edges_per_video) > 0 and accepted_by_video[video] >= int(max_edges_per_video):
            rejected_video_cap += 1
            continue
        a_node = nodes[int(edge["source"])]
        b_node = nodes[int(edge["target"])]
        a = int(a_node["base_label"])
        b = int(b_node["base_label"])
        if a == b or uf.find(a) == uf.find(b):
            rejected_stale += 1
            continue
        if not intervals[a] or not intervals[b]:
            rejected_empty += 1
            continue
        if not uf.can_merge(
            a,
            b,
            max_tracklets=int(max_tracklets_per_chain),
            overlap_slack_ms=float(overlap_slack_ms),
        ):
            rejected_overlap += 1
            continue
        uf.merge(a, b)
        accepted_by_video[video] += 1
        accepted.append({**edge, "source_base_label": int(a), "target_base_label": int(b)})

    root_to_label: dict[int, int] = {}
    next_label = 0
    labels = np.full(len(records), -1, dtype=np.int64)
    for idx, base_label in enumerate(base_labels.tolist()):
        root = int(uf.find(int(base_label)))
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[idx] = int(root_to_label[root])
    kept_labels = [int(labels[idx]) for idx in keep_indices]
    mean_score = float(np.mean([float(row["score"]) for row in accepted])) if accepted else 0.0
    mean_app = float(np.mean([float(row["app_sim"]) for row in accepted])) if accepted else 0.0
    info = {
        "accepted_edges": int(len(accepted)),
        "accepted_by_video": dict(accepted_by_video),
        "accepted_preview": accepted[:20],
        "rejected_score": int(rejected_score),
        "rejected_overlap": int(rejected_overlap),
        "rejected_video_cap": int(rejected_video_cap),
        "rejected_stale": int(rejected_stale),
        "rejected_empty": int(rejected_empty),
        "mean_accepted_score": mean_score,
        "mean_accepted_app_sim": mean_app,
        "components": int(len(set(kept_labels))),
        "largest_component": int(max(Counter(kept_labels).values(), default=0)),
        "policy_score": float(len(accepted) * mean_score * max(mean_app, 0.0)),
    }
    return labels, info


def _sort_key(row: dict[str, object], key: str) -> tuple[float, float, float, float]:
    return (
        float(row.get(key, 0.0)),
        float(row.get("tracklet_pair_f1", 0.0)),
        float(row.get("mean_accepted_app_sim", 0.0)),
        float(row.get("accepted_edges", 0.0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--view", action="append", default=[], help="feature view name:path[:weight]")
    ap.add_argument("--include-db-view", action="store_true")
    ap.add_argument("--max-gap-ms", default="1000,2000,5000,10000,30000")
    ap.add_argument("--min-app-sims", default="0.72,0.76,0.80,0.84")
    ap.add_argument("--max-center-dist-norms", default="1.0,2.0,4.0,8.0")
    ap.add_argument("--only-videos", default="", help="comma-separated video names whose temporal edges may be added")
    ap.add_argument("--min-scores", default="0.72,0.76,0.80,0.84")
    ap.add_argument("--max-tracklets-per-chain", default="16,32,64,128,0")
    ap.add_argument("--overlap-slack-ms", type=float, default=0.0)
    ap.add_argument("--max-edges-per-video", default="0,25,50,100")
    ap.add_argument("--skip-dirty-nodes", action="store_true")
    ap.add_argument("--output-scope", choices=["global_component", "video_local"], default="global_component")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--sort-key", default="policy_score")
    ap.add_argument("--assignment-offset", type=int, default=97_000_000)
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
        if path.lower() == "db":
            emb = _l2n(db_emb.astype(np.float32)) * float(weight)
        else:
            emb = _load_feature_npz(path, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=float(weight))
        views.append(emb.astype(np.float32))
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
    seqs = [int(record.seq) for record in records]
    allowed_videos = _parse_videos(args.only_videos)

    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    nodes, node_label_by_idx = _video_nodes(records, base_labels, keep_indices, emb)
    video_local_pred = _labels_to_seq_map(records, node_label_by_idx, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    video_local_pair = _pair_metrics(seqs, video_local_pred, gt_by_seq, weight_by_seq)
    print(
        json.dumps(
            {
                "stage": "base",
                "raw_components": len(raw_to_local),
                "video_nodes": len(nodes),
                "input_pair": base_pair,
                "video_local_pair": video_local_pair,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    edge_cache: dict[tuple[int, float, float, bool], list[dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    labels_by_rank: dict[int, np.ndarray] = {}
    for max_gap_ms in _parse_ints(args.max_gap_ms):
        for min_app_sim in _parse_floats(args.min_app_sims):
            for max_center_dist_norm in _parse_floats(args.max_center_dist_norms):
                edge_key = (
                    int(max_gap_ms),
                    float(min_app_sim),
                    float(max_center_dist_norm),
                    bool(args.skip_dirty_nodes),
                )
                edges = edge_cache.get(edge_key)
                if edges is None:
                    edges = _node_edges(
                        nodes,
                        max_gap_ms=int(max_gap_ms),
                        min_app_sim=float(min_app_sim),
                        max_center_dist_norm=float(max_center_dist_norm),
                        skip_dirty_nodes=bool(args.skip_dirty_nodes),
                        allowed_videos=allowed_videos,
                    )
                    edge_cache[edge_key] = edges
                for min_score in _parse_floats(args.min_scores):
                    for max_tracklets in _parse_ints(args.max_tracklets_per_chain):
                        for max_edges_per_video in _parse_ints(args.max_edges_per_video):
                            if args.output_scope == "video_local":
                                labels, info = _apply_edges(
                                    records,
                                    nodes,
                                    node_label_by_idx,
                                    edges,
                                    min_score=float(min_score),
                                    max_tracklets_per_chain=int(max_tracklets),
                                    overlap_slack_ms=float(args.overlap_slack_ms),
                                    max_edges_per_video=int(max_edges_per_video),
                                )
                            else:
                                labels, info = _apply_edges_global_components(
                                    records,
                                    base_labels,
                                    keep_indices,
                                    nodes,
                                    edges,
                                    min_score=float(min_score),
                                    max_tracklets_per_chain=int(max_tracklets),
                                    overlap_slack_ms=float(args.overlap_slack_ms),
                                    max_edges_per_video=int(max_edges_per_video),
                                )
                            pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                            pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                            rows.append(
                                {
                                    "mode": "video_temporal_relink",
                                    "max_gap_ms": int(max_gap_ms),
                                    "min_app_sim": float(min_app_sim),
                                    "max_center_dist_norm": float(max_center_dist_norm),
                                    "min_score": float(min_score),
                                    "max_tracklets_per_chain": int(max_tracklets),
                                    "max_edges_per_video": int(max_edges_per_video),
                                    "candidate_edges": int(len(edges)),
                                    "skip_dirty_nodes": bool(args.skip_dirty_nodes),
                                    "output_scope": str(args.output_scope),
                                    **info,
                                    **pair,
                                    "uses_anchors": False,
                                    "uses_gt_for_training_or_anchors": False,
                                    "uses_gt_for_evaluation_only": True,
                                }
                            )
    rows.sort(key=lambda row: _sort_key(row, str(args.sort_key)), reverse=True)

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        edges = edge_cache[
            (
                int(row["max_gap_ms"]),
                float(row["min_app_sim"]),
                float(row["max_center_dist_norm"]),
                bool(row["skip_dirty_nodes"]),
            )
        ]
        if args.output_scope == "video_local":
            labels, _info = _apply_edges(
                records,
                nodes,
                node_label_by_idx,
                edges,
                min_score=float(row["min_score"]),
                max_tracklets_per_chain=int(row["max_tracklets_per_chain"]),
                overlap_slack_ms=float(args.overlap_slack_ms),
                max_edges_per_video=int(row["max_edges_per_video"]),
            )
        else:
            labels, _info = _apply_edges_global_components(
                records,
                base_labels,
                keep_indices,
                nodes,
                edges,
                min_score=float(row["min_score"]),
                max_tracklets_per_chain=int(row["max_tracklets_per_chain"]),
                overlap_slack_ms=float(args.overlap_slack_ms),
                max_edges_per_video=int(row["max_edges_per_video"]),
            )
        labels_by_rank[rank] = labels
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs))
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": int(rank), "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            edges = edge_cache[
                (
                    int(row["max_gap_ms"]),
                    float(row["min_app_sim"]),
                    float(row["max_center_dist_norm"]),
                    bool(row["skip_dirty_nodes"]),
                )
            ]
            if args.output_scope == "video_local":
                labels, _info = _apply_edges(
                    records,
                    nodes,
                    node_label_by_idx,
                    edges,
                    min_score=float(row["min_score"]),
                    max_tracklets_per_chain=int(row["max_tracklets_per_chain"]),
                    overlap_slack_ms=float(args.overlap_slack_ms),
                    max_edges_per_video=int(row["max_edges_per_video"]),
                )
            else:
                labels, _info = _apply_edges_global_components(
                    records,
                    base_labels,
                    keep_indices,
                    nodes,
                    edges,
                    min_score=float(row["min_score"]),
                    max_tracklets_per_chain=int(row["max_tracklets_per_chain"]),
                    overlap_slack_ms=float(args.overlap_slack_ms),
                    max_edges_per_video=int(row["max_edges_per_video"]),
                )
        assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs=keep_seqs, offset=int(args.assignment_offset))
        rows[0].update(assignment_info)

    result = {
        "assignment_csv": str(args.assignment_csv),
        "feature_npz": str(args.feature_npz),
        "views": view_meta,
        "base_assignment_components": int(len(raw_to_local)),
        "video_nodes": int(len(nodes)),
        "only_videos": sorted(allowed_videos) if allowed_videos else "all",
        "base_pair_metrics": base_pair,
        "video_local_pair_metrics": video_local_pair,
        "output_scope": str(args.output_scope),
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
    print(json.dumps({"json": str(out), "base": base_pair, "video_local": video_local_pair, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
