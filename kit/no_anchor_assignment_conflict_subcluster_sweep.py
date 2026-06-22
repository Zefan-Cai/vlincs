#!/usr/bin/env python
"""Extract visual subclusters from conflicted no-anchor components.

This is a no-anchor resolver audit.  It starts from an existing assignment CSV,
looks only inside components with same-stream cannot-link conflicts, and extracts
small visual neighborhoods as new provisional IDs.  GT is used only after the
new assignment is formed, for pair/full metrics.
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
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from no_anchor_louvain_sweep import _write_assignments
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


def _load_npz_aligned(path: str, records, *, weight: float = 1.0) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    seqs = [int(seq) for seq in data["seqs"].tolist()]
    features = data["features"].astype(np.float32)
    by_seq = {seq: idx for idx, seq in enumerate(seqs)}
    missing = [int(record.seq) for record in records if int(record.seq) not in by_seq]
    if missing:
        raise ValueError(f"{path} missing seq={missing[0]} ({len(missing)} total)")
    order = np.asarray([by_seq[int(record.seq)] for record in records], dtype=np.int64)
    return _l2n(features[order].astype(np.float32)) * float(weight)


def _component_members(labels: np.ndarray, keep_indices: set[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        groups[int(labels[idx])].append(int(idx))
    return dict(groups)


def _fused_component_similarity(view_embeddings: list[np.ndarray], indices: list[int]) -> np.ndarray:
    sims = []
    for emb in view_embeddings:
        x = _l2n(emb[np.asarray(indices, dtype=np.int64)].astype(np.float32))
        sims.append((x @ x.T).astype(np.float32))
    sim = np.mean(np.stack(sims, axis=0), axis=0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


def _component_conflict_edges(indices: list[int], forbidden: list[set[int]]) -> tuple[list[tuple[int, int]], dict[int, int]]:
    pos_by_idx = {int(idx): pos for pos, idx in enumerate(indices)}
    edges: list[tuple[int, int]] = []
    degree = {int(idx): 0 for idx in indices}
    for pos, i in enumerate(indices):
        for j in indices[pos + 1 :]:
            if int(j) in forbidden[int(i)]:
                edges.append((int(i), int(j)))
                degree[int(i)] += 1
                degree[int(j)] += 1
    return edges, degree


def _candidate_stats(candidate_pos: set[int], sim: np.ndarray, conflict_pairs_pos: set[tuple[int, int]]) -> dict[str, float | int]:
    cand = sorted(candidate_pos)
    rest = [idx for idx in range(sim.shape[0]) if idx not in candidate_pos]
    internal_values = []
    for a_i, a in enumerate(cand):
        for b in cand[a_i + 1 :]:
            internal_values.append(float(sim[a, b]))
    internal = float(np.mean(internal_values)) if internal_values else 1.0
    cross_values = [float(sim[a, b]) for a in cand for b in rest]
    cross_mean = float(np.mean(cross_values)) if cross_values else 0.0
    cross_max = float(np.max(cross_values)) if cross_values else 0.0
    conflicts_to_rest = 0
    for a in cand:
        for b in rest:
            key = (a, b) if a < b else (b, a)
            if key in conflict_pairs_pos:
                conflicts_to_rest += 1
    return {
        "size": int(len(cand)),
        "internal_sim": internal,
        "cross_mean_sim": cross_mean,
        "cross_max_sim": cross_max,
        "margin_mean": float(internal - cross_mean),
        "margin_max": float(internal - cross_max),
        "conflicts_to_rest": int(conflicts_to_rest),
    }


def _build_candidates(
    records,
    indices: list[int],
    sim: np.ndarray,
    forbidden: list[set[int]],
    *,
    seed_sim: float,
    expand_sim: float,
    top_k: int,
    min_group_size: int,
    max_group_size: int,
    min_conflicts_to_rest: int,
    min_margin: float,
):
    conflict_edges, degree_by_idx = _component_conflict_edges(indices, forbidden)
    if not conflict_edges:
        return [], {"conflict_edges": 0, "conflict_nodes": 0}
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(indices)}
    conflict_pairs_pos = {
        (min(idx_to_pos[a], idx_to_pos[b]), max(idx_to_pos[a], idx_to_pos[b])) for a, b in conflict_edges
    }
    seeds = sorted(
        [idx for idx, degree in degree_by_idx.items() if degree > 0],
        key=lambda idx: (-degree_by_idx[idx], -_tracklet_quality_score(records[int(idx)]), int(idx)),
    )
    candidates = []
    seen: set[tuple[int, ...]] = set()
    for seed_idx in seeds:
        seed = idx_to_pos[int(seed_idx)]
        order = np.argsort(-sim[seed]).tolist()
        group = {int(seed)}
        for pos in order:
            pos = int(pos)
            if pos == seed:
                continue
            if len(group) >= int(max_group_size):
                break
            if float(sim[seed, pos]) < float(seed_sim):
                continue
            if any(indices[pos] in forbidden[indices[member]] for member in group):
                continue
            group.add(pos)
            if len(group) >= int(top_k) + 1:
                break
        changed = True
        while changed and len(group) < int(max_group_size):
            changed = False
            for pos in order:
                pos = int(pos)
                if pos in group:
                    continue
                if any(indices[pos] in forbidden[indices[member]] for member in group):
                    continue
                avg = float(np.mean([sim[pos, member] for member in group]))
                if avg >= float(expand_sim):
                    group.add(pos)
                    changed = True
                    if len(group) >= int(max_group_size):
                        break
        stats = _candidate_stats(group, sim, conflict_pairs_pos)
        if int(stats["size"]) < int(min_group_size):
            continue
        if int(stats["conflicts_to_rest"]) < int(min_conflicts_to_rest):
            continue
        if float(stats["margin_mean"]) < float(min_margin):
            continue
        key = tuple(sorted(indices[pos] for pos in group))
        if key in seen:
            continue
        seen.add(key)
        quality = float(np.mean([_tracklet_quality_score(records[indices[pos]]) for pos in group]))
        score = (
            2.0 * float(stats["margin_mean"])
            + 0.5 * float(stats["internal_sim"])
            + 0.08 * float(np.log1p(int(stats["size"])))
            + 0.05 * float(stats["conflicts_to_rest"])
            + 0.01 * quality
        )
        row = {
            "indices": [int(indices[pos]) for pos in sorted(group)],
            "seqs": [int(records[indices[pos]].seq) for pos in sorted(group)],
            "score": float(score),
            **stats,
        }
        candidates.append(row)
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    return candidates, {"conflict_edges": len(conflict_edges), "conflict_nodes": sum(1 for degree in degree_by_idx.values() if degree > 0)}


def _extract_subclusters(
    records,
    base_labels: np.ndarray,
    keep_indices: set[int],
    view_embeddings: list[np.ndarray],
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
):
    labels = base_labels.copy()
    next_label = int(labels.max()) + 1
    component_groups = _component_members(base_labels, keep_indices)
    selected_total = 0
    selected_tracklets = 0
    audited_components = 0
    conflicted_components = 0
    rejected_overlap = 0
    extracted: list[dict[str, object]] = []
    conflict_edges_total = 0
    conflict_nodes_total = 0
    for component_label, indices in sorted(component_groups.items(), key=lambda item: len(item[1]), reverse=True):
        if len(indices) < int(min_component_size) or len(indices) > int(max_component_size):
            continue
        audited_components += 1
        sim = _fused_component_similarity(view_embeddings, indices)
        candidates, cinfo = _build_candidates(
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
        if not candidates:
            continue
        conflicted_components += 1
        conflict_edges_total += int(cinfo["conflict_edges"])
        conflict_nodes_total += int(cinfo["conflict_nodes"])
        used: set[int] = set()
        accepted_for_component = 0
        for cand in candidates:
            cand_indices = [int(idx) for idx in cand["indices"]]
            if any(idx in used for idx in cand_indices):
                rejected_overlap += 1
                continue
            if accepted_for_component >= int(max_groups_per_component):
                break
            if selected_total >= int(max_total_groups):
                break
            new_label = int(next_label)
            next_label += 1
            for idx in cand_indices:
                labels[int(idx)] = new_label
                used.add(int(idx))
            selected_total += 1
            accepted_for_component += 1
            selected_tracklets += len(cand_indices)
            extracted.append(
                {
                    "component_label": int(component_label),
                    "new_label": new_label,
                    **{k: v for k, v in cand.items() if k != "indices"},
                }
            )
        if selected_total >= int(max_total_groups):
            break
    keep_labels = [int(labels[idx]) for idx in keep_indices]
    return labels, {
        "min_component_size": int(min_component_size),
        "max_component_size": int(max_component_size),
        "seed_sim": float(seed_sim),
        "expand_sim": float(expand_sim),
        "top_k": int(top_k),
        "min_group_size": int(min_group_size),
        "max_group_size": int(max_group_size),
        "min_conflicts_to_rest": int(min_conflicts_to_rest),
        "min_margin": float(min_margin),
        "max_groups_per_component": int(max_groups_per_component),
        "max_total_groups": int(max_total_groups),
        "audited_components": int(audited_components),
        "conflicted_components": int(conflicted_components),
        "conflict_edges": int(conflict_edges_total),
        "conflict_nodes": int(conflict_nodes_total),
        "selected_groups": int(selected_total),
        "selected_tracklets": int(selected_tracklets),
        "rejected_overlap": int(rejected_overlap),
        "components": int(len(set(keep_labels))),
        "largest_component": int(max(Counter(keep_labels).values(), default=0)),
        "extracted_preview": extracted[:20],
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
    ap.add_argument("--min-component-sizes", default="64,128,192")
    ap.add_argument("--max-component-sizes", default="1000000")
    ap.add_argument("--seed-sims", default="0.72,0.76,0.80")
    ap.add_argument("--expand-sims", default="0.68,0.72,0.76")
    ap.add_argument("--top-ks", default="4,8,16")
    ap.add_argument("--min-group-sizes", default="2,3,4")
    ap.add_argument("--max-group-sizes", default="8,16,32")
    ap.add_argument("--min-conflicts-to-rest", default="1,2")
    ap.add_argument("--min-margins", default="0.00,0.03,0.06")
    ap.add_argument("--max-groups-per-component", default="1,2,4")
    ap.add_argument("--max-total-groups", default="4,8,16,32")
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
    view_embeddings = [_l2n(primary.astype(np.float32))]
    view_meta = [{"name": "primary", "path": str(args.primary_feature_npz), "weight": 1.0}]
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        emb = _load_npz_aligned(path, records, weight=float(weight))
        view_embeddings.append(emb.astype(np.float32))
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
    for min_component_size in _parse_ints(args.min_component_sizes):
        for max_component_size in _parse_ints(args.max_component_sizes):
            for seed_sim in _parse_floats(args.seed_sims):
                for expand_sim in _parse_floats(args.expand_sims):
                    if float(expand_sim) > float(seed_sim):
                        continue
                    for top_k in _parse_ints(args.top_ks):
                        for min_group_size in _parse_ints(args.min_group_sizes):
                            for max_group_size in _parse_ints(args.max_group_sizes):
                                if int(max_group_size) < int(min_group_size):
                                    continue
                                for min_conflicts in _parse_ints(args.min_conflicts_to_rest):
                                    for min_margin in _parse_floats(args.min_margins):
                                        for max_groups_per_component in _parse_ints(args.max_groups_per_component):
                                            for max_total_groups in _parse_ints(args.max_total_groups):
                                                labels, info = _extract_subclusters(
                                                    records,
                                                    base_labels,
                                                    keep_indices,
                                                    view_embeddings,
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
                                                pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                                pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                                rows.append(
                                                    {
                                                        "mode": "conflict_subcluster_extract",
                                                        **{k: v for k, v in info.items() if k != "extracted_preview"},
                                                        "extracted_preview": info["extracted_preview"],
                                                        **pair,
                                                        "uses_anchors": False,
                                                        "uses_gt_for_training_or_anchors": False,
                                                        "uses_gt_for_evaluation_only": True,
                                                    }
                                                )
    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_precision"]),
            float(row["tracklet_pair_recall"]),
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        labels, _info = _extract_subclusters(
            records,
            base_labels,
            keep_indices,
            view_embeddings,
            forbidden,
            min_component_size=int(row["min_component_size"]),
            max_component_size=int(row["max_component_size"]),
            seed_sim=float(row["seed_sim"]),
            expand_sim=float(row["expand_sim"]),
            top_k=int(row["top_k"]),
            min_group_size=int(row["min_group_size"]),
            max_group_size=int(row["max_group_size"]),
            min_conflicts_to_rest=int(row["min_conflicts_to_rest"]),
            min_margin=float(row["min_margin"]),
            max_groups_per_component=int(row["max_groups_per_component"]),
            max_total_groups=int(row["max_total_groups"]),
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
            labels, _info = _extract_subclusters(
                records,
                base_labels,
                keep_indices,
                view_embeddings,
                forbidden,
                min_component_size=int(row["min_component_size"]),
                max_component_size=int(row["max_component_size"]),
                seed_sim=float(row["seed_sim"]),
                expand_sim=float(row["expand_sim"]),
                top_k=int(row["top_k"]),
                min_group_size=int(row["min_group_size"]),
                max_group_size=int(row["max_group_size"]),
                min_conflicts_to_rest=int(row["min_conflicts_to_rest"]),
                min_margin=float(row["min_margin"]),
                max_groups_per_component=int(row["max_groups_per_component"]),
                max_total_groups=int(row["max_total_groups"]),
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
