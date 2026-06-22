#!/usr/bin/env python
"""No-anchor provisional-subcluster retrieval/relink sweep.

This stage treats small conflict-derived subclusters as provisional query
nodes.  Each query can retrieve visually compatible tracklets from other current
components, then the query plus retrieved neighbors are peeled into a new
predicted ID.  The rules use only tracklet evidence and temporal cannot-link
constraints; GT is loaded only after prediction for metrics.
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
    from kit.no_anchor_assignment_conflict_subcluster_sweep import _build_candidates, _candidate_stats
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
        _tracklet_quality_score,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_assignment_conflict_subcluster_sweep import _build_candidates, _candidate_stats
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
        _tracklet_quality_score,
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_view(text: str) -> tuple[str, str, float]:
    parts = str(text).split(":")
    if len(parts) == 2:
        name, path = parts
        weight = 1.0
    elif len(parts) == 3:
        name, path, weight = parts
    else:
        raise ValueError(f"bad --view {text!r}; expected name:path[:weight]")
    return name, path, float(weight)


def _load_npz_aligned(path: str, records) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    seqs = [int(seq) for seq in data["seqs"].tolist()]
    features = data["features"].astype(np.float32)
    by_seq = {seq: idx for idx, seq in enumerate(seqs)}
    missing = [int(record.seq) for record in records if int(record.seq) not in by_seq]
    if missing:
        raise ValueError(f"{path} missing seq={missing[0]} ({len(missing)} total)")
    order = np.asarray([by_seq[int(record.seq)] for record in records], dtype=np.int64)
    return _l2n(features[order].astype(np.float32))


def _component_groups(labels: np.ndarray, keep_indices: set[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        groups[int(labels[int(idx)])].append(int(idx))
    return dict(groups)


def _fused_local_similarity(views: list[dict[str, object]], indices: list[int]) -> np.ndarray:
    mats = []
    for view in views:
        emb = view["emb"]
        weight = float(view["weight"])
        x = _l2n(emb[np.asarray(indices, dtype=np.int64)].astype(np.float32))
        mats.append((x @ x.T).astype(np.float32) * weight)
    denom = max(sum(float(view["weight"]) for view in views), 1.0e-9)
    sim = (np.sum(np.stack(mats, axis=0), axis=0) / denom).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


def _component_conflict_count(indices: list[int], forbidden: list[set[int]]) -> tuple[int, int]:
    idx_set = set(int(idx) for idx in indices)
    edges = 0
    nodes: set[int] = set()
    for idx in indices:
        for nbr in forbidden[int(idx)] & idx_set:
            if int(idx) < int(nbr):
                edges += 1
                nodes.add(int(idx))
                nodes.add(int(nbr))
    return int(edges), int(len(nodes))


def _source_candidates(
    records,
    labels: np.ndarray,
    keep_indices: set[int],
    views: list[dict[str, object]],
    forbidden: list[set[int]],
    *,
    min_component_size: int,
    max_component_size: int,
    seed_sim: float,
    expand_sim: float,
    top_k: int,
    min_group_size: int,
    max_group_size: int,
    min_conflicts_to_rest: int,
    min_margin: float,
    max_groups_per_component: int,
    max_total_groups: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups = _component_groups(labels, keep_indices)
    out: list[dict[str, object]] = []
    audited_components = 0
    conflicted_components = 0
    conflict_edges_total = 0
    conflict_nodes_total = 0
    rejected_overlap = 0
    for component_label, indices in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        if len(indices) < int(min_component_size) or len(indices) > int(max_component_size):
            continue
        audited_components += 1
        conflict_edges, conflict_nodes = _component_conflict_count(indices, forbidden)
        if conflict_edges <= 0:
            continue
        conflicted_components += 1
        conflict_edges_total += int(conflict_edges)
        conflict_nodes_total += int(conflict_nodes)
        sim = _fused_local_similarity(views, indices)
        candidates, _cinfo = _build_candidates(
            records,
            indices,
            sim,
            forbidden,
            seed_sim=float(seed_sim),
            expand_sim=float(expand_sim),
            top_k=int(top_k),
            min_group_size=int(min_group_size),
            max_group_size=int(max_group_size),
            min_conflicts_to_rest=int(min_conflicts_to_rest),
            min_margin=float(min_margin),
        )
        used: set[int] = set()
        accepted = 0
        for cand in candidates:
            cand_indices = [int(idx) for idx in cand["indices"]]
            if any(idx in used for idx in cand_indices):
                rejected_overlap += 1
                continue
            if accepted >= int(max_groups_per_component) or len(out) >= int(max_total_groups):
                break
            used.update(cand_indices)
            accepted += 1
            quality = float(np.mean([_tracklet_quality_score(records[idx]) for idx in cand_indices]))
            out.append(
                {
                    "source_component_label": int(component_label),
                    "source_indices": cand_indices,
                    "source_seqs": [int(records[idx].seq) for idx in cand_indices],
                    "source_size": int(len(cand_indices)),
                    "source_internal_sim": float(cand["internal_sim"]),
                    "source_cross_mean_sim": float(cand["cross_mean_sim"]),
                    "source_cross_max_sim": float(cand["cross_max_sim"]),
                    "source_margin_mean": float(cand["margin_mean"]),
                    "source_margin_max": float(cand["margin_max"]),
                    "source_conflicts_to_rest": int(cand["conflicts_to_rest"]),
                    "source_quality": float(quality),
                    "source_score": float(cand["score"]),
                }
            )
        if len(out) >= int(max_total_groups):
            break
    out.sort(
        key=lambda row: (
            float(row["source_score"]),
            float(row["source_margin_mean"]),
            int(row["source_conflicts_to_rest"]),
        ),
        reverse=True,
    )
    return out, {
        "source_min_component_size": int(min_component_size),
        "source_max_component_size": int(max_component_size),
        "source_seed_sim": float(seed_sim),
        "source_expand_sim": float(expand_sim),
        "source_top_k": int(top_k),
        "source_min_group_size": int(min_group_size),
        "source_max_group_size": int(max_group_size),
        "source_min_conflicts_to_rest": int(min_conflicts_to_rest),
        "source_min_margin": float(min_margin),
        "source_max_groups_per_component": int(max_groups_per_component),
        "source_max_total_groups": int(max_total_groups),
        "source_audited_components": int(audited_components),
        "source_conflicted_components": int(conflicted_components),
        "source_conflict_edges": int(conflict_edges_total),
        "source_conflict_nodes": int(conflict_nodes_total),
        "source_rejected_overlap": int(rejected_overlap),
        "source_candidates": int(len(out)),
    }


def _weighted_query_scores(
    views: list[dict[str, object]],
    source_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    values = []
    weights = []
    preview = {}
    for view in views:
        emb = view["emb"]
        weight = float(view["weight"])
        centroid = emb[np.asarray(source_indices, dtype=np.int64)].mean(axis=0)
        centroid = centroid / (float(np.linalg.norm(centroid)) + 1.0e-9)
        sims = (emb @ centroid).astype(np.float32)
        values.append(sims)
        weights.append(weight)
        preview[str(view["name"])] = float(np.mean(sims[np.asarray(source_indices, dtype=np.int64)]))
    stack = np.stack(values, axis=0).astype(np.float32)
    w = np.asarray(weights, dtype=np.float32)
    mean = (stack * w[:, None]).sum(axis=0) / max(float(w.sum()), 1.0e-9)
    return mean.astype(np.float32), stack.min(axis=0).astype(np.float32), stack, preview


def _has_forbidden(idx: int, group: set[int], forbidden: list[set[int]]) -> bool:
    return bool(forbidden[int(idx)] & {int(x) for x in group})


def _build_relink_groups(
    records,
    base_labels: np.ndarray,
    keep_indices: set[int],
    views: list[dict[str, object]],
    forbidden: list[set[int]],
    sources: list[dict[str, object]],
    *,
    attach_threshold: float,
    attach_min_view_vote: float,
    attach_view_sim_threshold: float,
    attach_min_view_min: float,
    attach_pool_k: int,
    attach_min_added: int,
    attach_max_added: int,
    attach_max_per_component: int,
    allow_source_component: bool,
    max_accepted_relinks: int,
) -> tuple[np.ndarray, dict[str, object]]:
    labels = base_labels.copy()
    next_label = int(labels.max()) + 1
    used: set[int] = set()
    accepted_groups: list[dict[str, object]] = []
    rejected_overlap = 0
    rejected_no_targets = 0
    rejected_forbidden = 0
    rejected_threshold = 0
    keep_arr = np.asarray(sorted(keep_indices), dtype=np.int64)
    keep_mask = np.zeros(len(records), dtype=bool)
    keep_mask[keep_arr] = True

    for source in sources:
        if len(accepted_groups) >= int(max_accepted_relinks):
            break
        source_indices = [int(idx) for idx in source["source_indices"]]
        if any(idx in used for idx in source_indices):
            rejected_overlap += 1
            continue
        source_component = int(source["source_component_label"])
        mean_sim, min_sim, view_stack, source_self = _weighted_query_scores(views, source_indices)
        eligible = keep_mask.copy()
        eligible[np.asarray(source_indices, dtype=np.int64)] = False
        if not bool(allow_source_component):
            eligible &= base_labels != source_component
        vote = (view_stack >= float(attach_view_sim_threshold)).mean(axis=0)
        eligible &= mean_sim >= float(attach_threshold)
        eligible &= min_sim >= float(attach_min_view_min)
        eligible &= vote >= float(attach_min_view_vote)
        if not np.any(eligible):
            rejected_no_targets += 1
            continue
        candidates = np.where(eligible)[0]
        if len(candidates) > int(attach_pool_k):
            top_local = np.argpartition(-mean_sim[candidates], int(attach_pool_k) - 1)[: int(attach_pool_k)]
            candidates = candidates[top_local]
        candidates = sorted(
            [int(idx) for idx in candidates],
            key=lambda idx: (
                -float(mean_sim[idx]),
                -float(vote[idx]),
                -_tracklet_quality_score(records[idx]),
                int(idx),
            ),
        )
        group = set(source_indices)
        per_component = Counter()
        added: list[int] = []
        threshold_rejects = 0
        forbidden_rejects = 0
        for idx in candidates:
            if idx in used or idx in group:
                continue
            component = int(base_labels[idx])
            if int(attach_max_per_component) > 0 and per_component[component] >= int(attach_max_per_component):
                continue
            if _has_forbidden(idx, group, forbidden):
                forbidden_rejects += 1
                continue
            group.add(idx)
            added.append(idx)
            per_component[component] += 1
            if len(added) >= int(attach_max_added):
                break
        if len(added) < int(attach_min_added):
            rejected_no_targets += 1
            rejected_forbidden += int(forbidden_rejects)
            rejected_threshold += int(threshold_rejects)
            continue
        group_sorted = sorted(group)
        new_label = int(next_label)
        next_label += 1
        for idx in group_sorted:
            labels[idx] = new_label
            used.add(idx)
        added_scores = [float(mean_sim[idx]) for idx in added]
        added_votes = [float(vote[idx]) for idx in added]
        accepted_groups.append(
            {
                **{k: v for k, v in source.items() if k != "source_indices"},
                "new_label": int(new_label),
                "group_size": int(len(group_sorted)),
                "added_tracklets": int(len(added)),
                "added_seqs": [int(records[idx].seq) for idx in added],
                "added_components": int(len(set(int(base_labels[idx]) for idx in added))),
                "mean_added_sim": float(np.mean(added_scores)) if added_scores else 0.0,
                "min_added_sim": float(np.min(added_scores)) if added_scores else 0.0,
                "mean_added_vote": float(np.mean(added_votes)) if added_votes else 0.0,
                "source_self_view_sim": source_self,
            }
        )
        rejected_forbidden += int(forbidden_rejects)

    keep_labels = [int(labels[idx]) for idx in keep_indices]
    return labels, {
        "attach_threshold": float(attach_threshold),
        "attach_min_view_vote": float(attach_min_view_vote),
        "attach_view_sim_threshold": float(attach_view_sim_threshold),
        "attach_min_view_min": float(attach_min_view_min),
        "attach_pool_k": int(attach_pool_k),
        "attach_min_added": int(attach_min_added),
        "attach_max_added": int(attach_max_added),
        "attach_max_per_component": int(attach_max_per_component),
        "allow_source_component": bool(allow_source_component),
        "max_accepted_relinks": int(max_accepted_relinks),
        "accepted_relinks": int(len(accepted_groups)),
        "rewritten_tracklets": int(sum(int(row["group_size"]) for row in accepted_groups)),
        "retrieved_tracklets": int(sum(int(row["added_tracklets"]) for row in accepted_groups)),
        "rejected_overlap": int(rejected_overlap),
        "rejected_no_targets": int(rejected_no_targets),
        "rejected_forbidden": int(rejected_forbidden),
        "rejected_threshold": int(rejected_threshold),
        "components": int(len(set(keep_labels))),
        "largest_component": int(max(Counter(keep_labels).values(), default=0)),
        "accepted_preview": accepted_groups[:20],
        "uses_ground_truth": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--primary-feature-npz", required=True)
    ap.add_argument("--view", action="append", default=[], help="feature view name:path[:weight]")
    ap.add_argument("--source-min-component-sizes", default="64,128")
    ap.add_argument("--source-max-component-sizes", default="1000000")
    ap.add_argument("--source-seed-sims", default="0.74,0.78")
    ap.add_argument("--source-expand-sims", default="0.70,0.74")
    ap.add_argument("--source-top-ks", default="4,8")
    ap.add_argument("--source-min-group-sizes", default="3,4")
    ap.add_argument("--source-max-group-sizes", default="8,16")
    ap.add_argument("--source-min-conflicts-to-rest", default="1,2")
    ap.add_argument("--source-min-margins", default="0.03,0.06")
    ap.add_argument("--source-max-groups-per-component", default="1,2")
    ap.add_argument("--source-max-total-groups", default="8,16,32")
    ap.add_argument("--attach-thresholds", default="0.74,0.78,0.82")
    ap.add_argument("--attach-min-view-votes", default="0.5,0.75")
    ap.add_argument("--attach-view-sim-thresholds", default="0.70,0.74")
    ap.add_argument("--attach-min-view-mins", default="0.45,0.55")
    ap.add_argument("--attach-pool-ks", default="100,250")
    ap.add_argument("--attach-min-added", default="1,2")
    ap.add_argument("--attach-max-added", default="4,8")
    ap.add_argument("--attach-max-per-component", default="1,2")
    ap.add_argument("--allow-source-component", action="store_true")
    ap.add_argument("--max-accepted-relinks", default="4,8,16")
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
    primary = _load_feature_npz(args.primary_feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0)
    views: list[dict[str, object]] = [{"name": "primary", "path": str(args.primary_feature_npz), "weight": 1.0, "emb": _l2n(primary.astype(np.float32))}]
    view_meta = [{"name": "primary", "path": str(args.primary_feature_npz), "weight": 1.0}]
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        views.append({"name": name, "path": path, "weight": float(weight), "emb": _load_npz_aligned(path, records)})
        view_meta.append({"name": name, "path": path, "weight": float(weight)})

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

    forbidden = _build_overlap_forbidden(records)
    rows: list[dict[str, object]] = []
    labels_by_rank: dict[int, np.ndarray] = {}
    source_cache: dict[tuple[object, ...], tuple[list[dict[str, object]], dict[str, object]]] = {}
    for min_component_size in _parse_ints(args.source_min_component_sizes):
        for max_component_size in _parse_ints(args.source_max_component_sizes):
            for seed_sim in _parse_floats(args.source_seed_sims):
                for expand_sim in _parse_floats(args.source_expand_sims):
                    if float(expand_sim) > float(seed_sim):
                        continue
                    for top_k in _parse_ints(args.source_top_ks):
                        for min_group_size in _parse_ints(args.source_min_group_sizes):
                            for max_group_size in _parse_ints(args.source_max_group_sizes):
                                if int(max_group_size) < int(min_group_size):
                                    continue
                                for min_conflicts in _parse_ints(args.source_min_conflicts_to_rest):
                                    for min_margin in _parse_floats(args.source_min_margins):
                                        for max_groups_per_component in _parse_ints(args.source_max_groups_per_component):
                                            for max_total_groups in _parse_ints(args.source_max_total_groups):
                                                source_key = (
                                                    int(min_component_size),
                                                    int(max_component_size),
                                                    float(seed_sim),
                                                    float(expand_sim),
                                                    int(top_k),
                                                    int(min_group_size),
                                                    int(max_group_size),
                                                    int(min_conflicts),
                                                    float(min_margin),
                                                    int(max_groups_per_component),
                                                    int(max_total_groups),
                                                )
                                                if source_key not in source_cache:
                                                    source_cache[source_key] = _source_candidates(
                                                        records,
                                                        base_labels,
                                                        keep_indices,
                                                        views,
                                                        forbidden,
                                                        min_component_size=int(min_component_size),
                                                        max_component_size=int(max_component_size),
                                                        seed_sim=float(seed_sim),
                                                        expand_sim=float(expand_sim),
                                                        top_k=int(top_k),
                                                        min_group_size=int(min_group_size),
                                                        max_group_size=int(max_group_size),
                                                        min_conflicts_to_rest=int(min_conflicts),
                                                        min_margin=float(min_margin),
                                                        max_groups_per_component=int(max_groups_per_component),
                                                        max_total_groups=int(max_total_groups),
                                                    )
                                                sources, source_info = source_cache[source_key]
                                                if not sources:
                                                    continue
                                                for attach_threshold in _parse_floats(args.attach_thresholds):
                                                    for min_view_vote in _parse_floats(args.attach_min_view_votes):
                                                        for view_sim_threshold in _parse_floats(args.attach_view_sim_thresholds):
                                                            for min_view_min in _parse_floats(args.attach_min_view_mins):
                                                                for pool_k in _parse_ints(args.attach_pool_ks):
                                                                    for min_added in _parse_ints(args.attach_min_added):
                                                                        for max_added in _parse_ints(args.attach_max_added):
                                                                            if int(max_added) < int(min_added):
                                                                                continue
                                                                            for max_per_component in _parse_ints(args.attach_max_per_component):
                                                                                for max_relinks in _parse_ints(args.max_accepted_relinks):
                                                                                    labels, relink_info = _build_relink_groups(
                                                                                        records,
                                                                                        base_labels,
                                                                                        keep_indices,
                                                                                        views,
                                                                                        forbidden,
                                                                                        sources,
                                                                                        attach_threshold=float(attach_threshold),
                                                                                        attach_min_view_vote=float(min_view_vote),
                                                                                        attach_view_sim_threshold=float(view_sim_threshold),
                                                                                        attach_min_view_min=float(min_view_min),
                                                                                        attach_pool_k=int(pool_k),
                                                                                        attach_min_added=int(min_added),
                                                                                        attach_max_added=int(max_added),
                                                                                        attach_max_per_component=int(max_per_component),
                                                                                        allow_source_component=bool(args.allow_source_component),
                                                                                        max_accepted_relinks=int(max_relinks),
                                                                                    )
                                                                                    pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                                                                    pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                                                                    rows.append(
                                                                                        {
                                                                                            "mode": "provisional_subcluster_relink",
                                                                                            **{k: v for k, v in source_info.items() if k != "source_preview"},
                                                                                            **{k: v for k, v in relink_info.items() if k != "accepted_preview"},
                                                                                            "accepted_preview": relink_info["accepted_preview"],
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
        source_key = (
            int(row["source_min_component_size"]),
            int(row["source_max_component_size"]),
            float(row["source_seed_sim"]),
            float(row["source_expand_sim"]),
            int(row["source_top_k"]),
            int(row["source_min_group_size"]),
            int(row["source_max_group_size"]),
            int(row["source_min_conflicts_to_rest"]),
            float(row["source_min_margin"]),
            int(row["source_max_groups_per_component"]),
            int(row["source_max_total_groups"]),
        )
        sources, _source_info = source_cache[source_key]
        labels, _relink_info = _build_relink_groups(
            records,
            base_labels,
            keep_indices,
            views,
            forbidden,
            sources,
            attach_threshold=float(row["attach_threshold"]),
            attach_min_view_vote=float(row["attach_min_view_vote"]),
            attach_view_sim_threshold=float(row["attach_view_sim_threshold"]),
            attach_min_view_min=float(row["attach_min_view_min"]),
            attach_pool_k=int(row["attach_pool_k"]),
            attach_min_added=int(row["attach_min_added"]),
            attach_max_added=int(row["attach_max_added"]),
            attach_max_per_component=int(row["attach_max_per_component"]),
            allow_source_component=bool(row["allow_source_component"]),
            max_accepted_relinks=int(row["max_accepted_relinks"]),
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
            source_key = (
                int(row["source_min_component_size"]),
                int(row["source_max_component_size"]),
                float(row["source_seed_sim"]),
                float(row["source_expand_sim"]),
                int(row["source_top_k"]),
                int(row["source_min_group_size"]),
                int(row["source_max_group_size"]),
                int(row["source_min_conflicts_to_rest"]),
                float(row["source_min_margin"]),
                int(row["source_max_groups_per_component"]),
                int(row["source_max_total_groups"]),
            )
            sources, _source_info = source_cache[source_key]
            labels, _relink_info = _build_relink_groups(
                records,
                base_labels,
                keep_indices,
                views,
                forbidden,
                sources,
                attach_threshold=float(row["attach_threshold"]),
                attach_min_view_vote=float(row["attach_min_view_vote"]),
                attach_view_sim_threshold=float(row["attach_view_sim_threshold"]),
                attach_min_view_min=float(row["attach_min_view_min"]),
                attach_pool_k=int(row["attach_pool_k"]),
                attach_min_added=int(row["attach_min_added"]),
                attach_max_added=int(row["attach_max_added"]),
                attach_max_per_component=int(row["attach_max_per_component"]),
                allow_source_component=bool(row["allow_source_component"]),
                max_accepted_relinks=int(row["max_accepted_relinks"]),
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
        "views": view_meta,
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "source_cache_size": int(len(source_cache)),
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
