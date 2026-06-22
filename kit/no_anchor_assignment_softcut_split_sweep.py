#!/usr/bin/env python
"""Soft-cut large conflicted assignment components with no-GT evidence.

The oracle decomposition says the current no-anchor assignment is limited by
large false-merge components.  Prior hard graph-coloring split too much; tiny
subcluster extraction split too little.  This sweep tries the middle ground:
inside large components, cluster visual modes while treating same-stream
temporal overlaps as a soft distance penalty.  A split is accepted only when it
reduces internal cannot-link conflicts and has a no-GT visual margin.

No anchors or identity GT are used for selecting components, clusters, or
thresholds.  GT is loaded only after prediction for pair/full metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
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
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
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
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_view(text: str) -> tuple[str, str, float]:
    parts = str(text).split(":")
    if len(parts) == 2:
        return parts[0], parts[1], 1.0
    if len(parts) == 3:
        return parts[0], parts[1], float(parts[2])
    raise ValueError(f"bad --view {text!r}; expected name:path[:weight]")


def _component_members(labels: np.ndarray, keep_indices: set[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        out[int(labels[int(idx)])].append(int(idx))
    return dict(out)


def _fused_embedding(views: list[np.ndarray]) -> np.ndarray:
    if len(views) == 1:
        return _l2n(views[0].astype(np.float32))
    return _l2n(np.concatenate([view.astype(np.float32) for view in views], axis=1).astype(np.float32))


def _conflict_pairs(indices: list[int], forbidden: list[set[int]]) -> list[tuple[int, int]]:
    pos = {int(idx): i for i, idx in enumerate(indices)}
    out = []
    for a_i, a_idx in enumerate(indices):
        for b_idx in forbidden[int(a_idx)]:
            b_i = pos.get(int(b_idx))
            if b_i is None or b_i <= a_i:
                continue
            out.append((int(a_i), int(b_i)))
    return out


def _split_stats(sim: np.ndarray, local_labels: np.ndarray, conflict_pairs: list[tuple[int, int]]) -> dict[str, float | int]:
    n = int(sim.shape[0])
    labels = np.asarray(local_labels, dtype=np.int64)
    parts = [np.where(labels == label)[0] for label in sorted(set(labels.tolist()))]
    sizes = [int(len(part)) for part in parts]
    before = int(len(conflict_pairs))
    after = int(sum(1 for a, b in conflict_pairs if int(labels[a]) == int(labels[b])))
    within_values = []
    cross_values = []
    for i in range(n):
        for j in range(i + 1, n):
            if int(labels[i]) == int(labels[j]):
                within_values.append(float(sim[i, j]))
            else:
                cross_values.append(float(sim[i, j]))
    within = float(np.mean(within_values)) if within_values else 1.0
    cross = float(np.mean(cross_values)) if cross_values else 0.0
    conflict_reduction = float((before - after) / max(before, 1))
    return {
        "parts": int(len(parts)),
        "min_part_size": int(min(sizes) if sizes else 0),
        "max_part_size": int(max(sizes) if sizes else 0),
        "min_part_frac": float((min(sizes) / max(n, 1)) if sizes else 0.0),
        "conflict_edges_before": int(before),
        "conflict_edges_after": int(after),
        "conflict_reduction": float(conflict_reduction),
        "within_sim": float(within),
        "cross_sim": float(cross),
        "visual_margin": float(within - cross),
    }


def _cluster_component(sim: np.ndarray, conflict_pairs: list[tuple[int, int]], *, k: int, penalty: float) -> np.ndarray:
    dist = 1.0 - sim.astype(np.float32)
    np.clip(dist, 0.0, 2.0, out=dist)
    for a, b in conflict_pairs:
        dist[int(a), int(b)] = max(float(dist[int(a), int(b)]), 1.0 + float(penalty))
        dist[int(b), int(a)] = dist[int(a), int(b)]
    np.fill_diagonal(dist, 0.0)
    clustered = AgglomerativeClustering(n_clusters=int(k), metric="precomputed", linkage="average").fit_predict(dist)
    _, relabeled = np.unique(clustered, return_inverse=True)
    return relabeled.astype(np.int64)


_CANDIDATE_CACHE: dict[tuple[int, float, int, int, int], list[dict[str, object]]] = {}


def _build_candidates(
    records,
    base_labels: np.ndarray,
    keep_indices: set[int],
    emb: np.ndarray,
    forbidden: list[set[int]],
    *,
    split_k: int,
    penalty: float,
    min_component_size: int,
    max_component_size: int,
    min_conflict_edges: int,
) -> list[dict[str, object]]:
    cache_key = (int(split_k), float(penalty), int(min_component_size), int(max_component_size), int(min_conflict_edges))
    if cache_key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[cache_key]
    candidates: list[dict[str, object]] = []
    members_by_label = _component_members(base_labels, keep_indices)
    for component_label, indices in sorted(members_by_label.items(), key=lambda item: len(item[1]), reverse=True):
        if len(indices) < int(min_component_size) or len(indices) > int(max_component_size):
            continue
        conflicts = _conflict_pairs(indices, forbidden)
        if len(conflicts) < int(min_conflict_edges):
            continue
        if len(indices) <= int(split_k):
            continue
        x = _l2n(emb[np.asarray(indices, dtype=np.int64)].astype(np.float32))
        sim = (x @ x.T).astype(np.float32)
        np.fill_diagonal(sim, 1.0)
        local_labels = _cluster_component(sim, conflicts, k=int(split_k), penalty=float(penalty))
        stats = _split_stats(sim, local_labels, conflicts)
        if int(stats["parts"]) < 2:
            continue
        score = (
            2.0 * float(stats["conflict_reduction"])
            + 1.5 * float(stats["visual_margin"])
            + 0.25 * float(stats["within_sim"])
            + 0.02 * float(np.log1p(len(indices)))
            - 0.50 * max(0.0, 0.08 - float(stats["min_part_frac"]))
        )
        candidates.append(
            {
                "component_label": int(component_label),
                "indices": [int(idx) for idx in indices],
                "seqs": [int(records[int(idx)].seq) for idx in indices],
                "local_labels": [int(x) for x in local_labels.tolist()],
                "split_k": int(split_k),
                "penalty": float(penalty),
                "score": float(score),
                **stats,
            }
        )
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    _CANDIDATE_CACHE[cache_key] = candidates
    return candidates


def _apply_softcuts(
    records,
    base_labels: np.ndarray,
    keep_indices: set[int],
    emb: np.ndarray,
    forbidden: list[set[int]],
    *,
    split_k: int,
    penalty: float,
    min_component_size: int,
    max_component_size: int,
    min_conflict_edges: int,
    min_conflict_reduction: float,
    min_visual_margin: float,
    min_part_size: int,
    min_part_frac: float,
    max_split_components: int,
) -> tuple[np.ndarray, dict[str, object]]:
    labels = base_labels.copy()
    next_label = int(labels.max()) + 1
    accepted = []
    rejected_conflict = rejected_margin = rejected_size = 0
    candidates = _build_candidates(
        records,
        base_labels,
        keep_indices,
        emb,
        forbidden,
        split_k=int(split_k),
        penalty=float(penalty),
        min_component_size=int(min_component_size),
        max_component_size=int(max_component_size),
        min_conflict_edges=int(min_conflict_edges),
    )
    for cand in candidates:
        if int(max_split_components) > 0 and len(accepted) >= int(max_split_components):
            break
        if float(cand["conflict_reduction"]) < float(min_conflict_reduction):
            rejected_conflict += 1
            continue
        if float(cand["visual_margin"]) < float(min_visual_margin):
            rejected_margin += 1
            continue
        if int(cand["min_part_size"]) < int(min_part_size) or float(cand["min_part_frac"]) < float(min_part_frac):
            rejected_size += 1
            continue
        component_indices = [int(idx) for idx in cand["indices"]]
        local_labels = [int(x) for x in cand["local_labels"]]
        local_to_global: dict[int, int] = {}
        for local_label in sorted(set(local_labels)):
            local_to_global[int(local_label)] = int(next_label)
            next_label += 1
        for idx, local_label in zip(component_indices, local_labels):
            labels[int(idx)] = int(local_to_global[int(local_label)])
        accepted.append(
            {
                key: value
                for key, value in cand.items()
                if key not in {"indices", "local_labels"}
            }
        )
    keep_labels = [int(labels[int(idx)]) for idx in keep_indices]
    return labels, {
        "mode": "assignment_softcut_split",
        "split_k": int(split_k),
        "penalty": float(penalty),
        "min_component_size": int(min_component_size),
        "max_component_size": int(max_component_size),
        "min_conflict_edges": int(min_conflict_edges),
        "min_conflict_reduction": float(min_conflict_reduction),
        "min_visual_margin": float(min_visual_margin),
        "min_part_size": int(min_part_size),
        "min_part_frac": float(min_part_frac),
        "max_split_components": int(max_split_components),
        "candidate_components": int(len(candidates)),
        "split_components": int(len(accepted)),
        "split_tracklets": int(sum(len(row["seqs"]) for row in accepted)),
        "split_parts": int(sum(int(row["parts"]) for row in accepted)),
        "accepted_preview": accepted[:12],
        "rejected_conflict_reduction": int(rejected_conflict),
        "rejected_visual_margin": int(rejected_margin),
        "rejected_size": int(rejected_size),
        "components": int(len(set(keep_labels))),
        "largest_component": int(max(Counter(keep_labels).values(), default=0)),
        "uses_ground_truth": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--view", action="append", default=[], help="feature view name:path[:weight]")
    ap.add_argument("--include-db-view", action="store_true")
    ap.add_argument("--split-ks", default="2,3")
    ap.add_argument("--penalties", default="0.25,0.50,1.00")
    ap.add_argument("--min-component-sizes", default="128,160,192")
    ap.add_argument("--max-component-sizes", default="1000000")
    ap.add_argument("--min-conflict-edges", default="10,25,50")
    ap.add_argument("--min-conflict-reductions", default="0.20,0.40,0.60")
    ap.add_argument("--min-visual-margins", default="-0.02,0.00,0.02,0.05")
    ap.add_argument("--min-part-sizes", default="8,16,24")
    ap.add_argument("--min-part-fracs", default="0.03,0.05,0.08")
    ap.add_argument("--max-split-components", default="1,2,4,8")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=96_000_000)
    ap.add_argument("--rank-by", choices=["pair", "no_gt_evidence"], default="pair")
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
        emb = _load_feature_npz(path, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=float(weight))
        views.append(emb.astype(np.float32) * float(weight))
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
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    forbidden = _build_overlap_forbidden(records)
    rows = []
    labels_by_rank: dict[int, np.ndarray] = {}
    for split_k in _parse_ints(args.split_ks):
        for penalty in _parse_floats(args.penalties):
            for min_component_size in _parse_ints(args.min_component_sizes):
                for max_component_size in _parse_ints(args.max_component_sizes):
                    for min_conflict_edges in _parse_ints(args.min_conflict_edges):
                        for min_conflict_reduction in _parse_floats(args.min_conflict_reductions):
                            for min_visual_margin in _parse_floats(args.min_visual_margins):
                                for min_part_size in _parse_ints(args.min_part_sizes):
                                    for min_part_frac in _parse_floats(args.min_part_fracs):
                                        for max_split_components in _parse_ints(args.max_split_components):
                                            labels, info = _apply_softcuts(
                                                records,
                                                base_labels,
                                                keep_indices,
                                                emb,
                                                forbidden,
                                                split_k=int(split_k),
                                                penalty=float(penalty),
                                                min_component_size=int(min_component_size),
                                                max_component_size=int(max_component_size),
                                                min_conflict_edges=int(min_conflict_edges),
                                                min_conflict_reduction=float(min_conflict_reduction),
                                                min_visual_margin=float(min_visual_margin),
                                                min_part_size=int(min_part_size),
                                                min_part_frac=float(min_part_frac),
                                                max_split_components=int(max_split_components),
                                            )
                                            pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                            pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                            accepted_preview = info["accepted_preview"]
                                            evidence_scores = [
                                                float(item.get("score", 0.0))
                                                for item in accepted_preview
                                                if isinstance(item, dict)
                                            ]
                                            conflict_reductions = [
                                                float(item.get("conflict_reduction", 0.0))
                                                for item in accepted_preview
                                                if isinstance(item, dict)
                                            ]
                                            visual_margins = [
                                                float(item.get("visual_margin", 0.0))
                                                for item in accepted_preview
                                                if isinstance(item, dict)
                                            ]
                                            no_gt_evidence_score = (
                                                float(np.mean(evidence_scores)) if evidence_scores else 0.0
                                            )
                                            no_gt_evidence_score += 0.75 * (
                                                float(np.mean(conflict_reductions)) if conflict_reductions else 0.0
                                            )
                                            no_gt_evidence_score += 1.25 * (
                                                float(np.mean(visual_margins)) if visual_margins else 0.0
                                            )
                                            no_gt_evidence_score += 0.02 * float(info["split_tracklets"])
                                            rows.append(
                                                {
                                                    **{key: value for key, value in info.items() if key != "accepted_preview"},
                                                    "accepted_preview": accepted_preview,
                                                    "no_gt_evidence_score": float(no_gt_evidence_score),
                                                    "no_gt_mean_candidate_score": float(np.mean(evidence_scores)) if evidence_scores else 0.0,
                                                    "no_gt_mean_conflict_reduction": float(np.mean(conflict_reductions))
                                                    if conflict_reductions
                                                    else 0.0,
                                                    "no_gt_mean_visual_margin": float(np.mean(visual_margins))
                                                    if visual_margins
                                                    else 0.0,
                                                    **pair,
                                                    "uses_anchors": False,
                                                    "uses_gt_for_training_or_anchors": False,
                                                    "uses_gt_for_evaluation_only": True,
                                                }
                                            )
    if args.rank_by == "no_gt_evidence":
        rows.sort(
            key=lambda row: (
                float(row["no_gt_evidence_score"]),
                int(row["split_components"]),
                float(row["no_gt_mean_conflict_reduction"]),
                float(row["no_gt_mean_visual_margin"]),
                float(row["tracklet_pair_f1"]),
            ),
            reverse=True,
        )
    else:
        rows.sort(key=lambda row: (float(row["tracklet_pair_f1"]), float(row["tracklet_pair_precision"])), reverse=True)

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        labels, _info = _apply_softcuts(
            records,
            base_labels,
            keep_indices,
            emb,
            forbidden,
            split_k=int(row["split_k"]),
            penalty=float(row["penalty"]),
            min_component_size=int(row["min_component_size"]),
            max_component_size=int(row["max_component_size"]),
            min_conflict_edges=int(row["min_conflict_edges"]),
            min_conflict_reduction=float(row["min_conflict_reduction"]),
            min_visual_margin=float(row["min_visual_margin"]),
            min_part_size=int(row["min_part_size"]),
            min_part_frac=float(row["min_part_frac"]),
            max_split_components=int(row["max_split_components"]),
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
            labels, _info = _apply_softcuts(
                records,
                base_labels,
                keep_indices,
                emb,
                forbidden,
                split_k=int(row["split_k"]),
                penalty=float(row["penalty"]),
                min_component_size=int(row["min_component_size"]),
                max_component_size=int(row["max_component_size"]),
                min_conflict_edges=int(row["min_conflict_edges"]),
                min_conflict_reduction=float(row["min_conflict_reduction"]),
                min_visual_margin=float(row["min_visual_margin"]),
                min_part_size=int(row["min_part_size"]),
                min_part_frac=float(row["min_part_frac"]),
                max_split_components=int(row["max_split_components"]),
            )
        assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs=keep_seqs, offset=int(args.assignment_offset))
        rows[0].update(assignment_info)

    result = {
        "assignment_csv": str(args.assignment_csv),
        "feature_npz": str(args.feature_npz),
        "views": view_meta,
        "base_assignment_components": int(len(raw_to_local)),
        "base_pair_metrics": base_pair,
        "rank_by": str(args.rank_by),
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
    print(json.dumps({"json": str(out), "base": base_pair, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
