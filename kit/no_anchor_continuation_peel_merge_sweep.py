#!/usr/bin/env python
"""Edge-guided peel-and-merge repair for no-anchor assignments.

High-confidence continuation-verifier edges often connect a small component to
the correct large identity component, but cannot-link prevents the merge
because the large component contains a few temporally conflicting tracklets.
This script tries a local repair: peel only the conflicting nodes out of the
large component, then merge the small component into the remaining compatible
body.  GT is used only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _candidate_edges, _component_members
    from kit.no_anchor_continuation_positive_edge_verifier import _build_training
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
        _parse_floats,
        _parse_ints,
        _write_csv,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _candidate_edges, _component_members
    from no_anchor_continuation_positive_edge_verifier import _build_training
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
        _parse_floats,
        _parse_ints,
        _write_csv,
    )

from vlincs_gallery.eval.score import load_ds1_gt_by_video


def _components(labels: np.ndarray, keep_indices: set[int]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for idx in keep_indices:
        out[int(labels[int(idx)])].add(int(idx))
    return out


def _can_merge_sets(a: set[int], b: set[int], forbidden: list[set[int]]) -> bool:
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    for node in small:
        if forbidden[node] & large:
            return False
    return True


def _peel_merge(
    base_labels: np.ndarray,
    keep_indices: set[int],
    edge_rows: list[dict[str, object]],
    forbidden: list[set[int]],
    *,
    probability_field: str,
    threshold: float,
    min_edge_score: float,
    max_small_size: int,
    max_large_size: int,
    max_conflict_nodes: int,
    max_conflict_frac: float,
    min_compatible_size: int,
    max_repairs: int,
) -> tuple[np.ndarray, dict[str, object]]:
    labels = base_labels.copy()
    next_label = int(labels.max()) + 1
    accepted_direct = 0
    accepted_peel = 0
    peeled_nodes = 0
    rejected_threshold = rejected_score = rejected_stale = 0
    rejected_size = rejected_conflict = rejected_compatible = rejected_budget = 0
    processed = 0
    rows = sorted(edge_rows, key=lambda row: float(row[probability_field]), reverse=True)
    for row in rows:
        if int(max_repairs) > 0 and accepted_peel + accepted_direct >= int(max_repairs):
            rejected_budget += 1
            continue
        if float(row[probability_field]) < float(threshold):
            rejected_threshold += 1
            continue
        if float(row["score"]) < float(min_edge_score):
            rejected_score += 1
            continue
        processed += 1
        a_rep = int(row["source_rep"])
        b_rep = int(row["target_rep"])
        la = int(labels[a_rep])
        lb = int(labels[b_rep])
        if la == lb:
            rejected_stale += 1
            continue
        comps = _components(labels, keep_indices)
        ca = comps.get(la, set())
        cb = comps.get(lb, set())
        if not ca or not cb:
            rejected_stale += 1
            continue
        if _can_merge_sets(ca, cb, forbidden):
            if len(ca) + len(cb) > int(max_large_size):
                rejected_size += 1
                continue
            target_label = la if len(ca) >= len(cb) else lb
            source_set = cb if target_label == la else ca
            for idx in source_set:
                labels[int(idx)] = target_label
            accepted_direct += 1
            continue

        small_label, large_label = (la, lb) if len(ca) <= len(cb) else (lb, la)
        small = comps[small_label]
        large = comps[large_label]
        if len(small) > int(max_small_size) or len(large) > int(max_large_size):
            rejected_size += 1
            continue
        conflict_nodes = {idx for idx in large if forbidden[idx] & small}
        if not conflict_nodes:
            rejected_conflict += 1
            continue
        if len(conflict_nodes) > int(max_conflict_nodes):
            rejected_conflict += 1
            continue
        if len(conflict_nodes) / max(float(len(large)), 1.0) > float(max_conflict_frac):
            rejected_conflict += 1
            continue
        compatible = large - conflict_nodes
        if len(compatible) < int(min_compatible_size):
            rejected_compatible += 1
            continue
        if not _can_merge_sets(compatible, small, forbidden):
            rejected_compatible += 1
            continue
        conflict_label = next_label
        next_label += 1
        for idx in conflict_nodes:
            labels[int(idx)] = conflict_label
        for idx in small:
            labels[int(idx)] = large_label
        accepted_peel += 1
        peeled_nodes += int(len(conflict_nodes))

    return labels, {
        "probability_field": str(probability_field),
        "peel_threshold": float(threshold),
        "peel_min_edge_score": float(min_edge_score),
        "peel_max_small_size": int(max_small_size),
        "peel_max_large_size": int(max_large_size),
        "peel_max_conflict_nodes": int(max_conflict_nodes),
        "peel_max_conflict_frac": float(max_conflict_frac),
        "peel_min_compatible_size": int(min_compatible_size),
        "peel_max_repairs": int(max_repairs),
        "peel_processed_edges": int(processed),
        "peel_accepted_direct": int(accepted_direct),
        "peel_accepted_peel": int(accepted_peel),
        "peel_peeled_nodes": int(peeled_nodes),
        "peel_rejected_threshold": int(rejected_threshold),
        "peel_rejected_score": int(rejected_score),
        "peel_rejected_stale": int(rejected_stale),
        "peel_rejected_size": int(rejected_size),
        "peel_rejected_conflict": int(rejected_conflict),
        "peel_rejected_compatible": int(rejected_compatible),
        "peel_rejected_budget": int(rejected_budget),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
    }


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
    ap.add_argument("--probability-field", default="sample_prob_top_mean")
    ap.add_argument("--max-positive-pairs", type=int, default=12000)
    ap.add_argument("--max-negative-pairs", type=int, default=12000)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf", "logreg"])
    ap.add_argument("--random-state", type=int, default=29)
    ap.add_argument("--thresholds", default="0.80,0.90,0.95,0.98")
    ap.add_argument("--min-edge-scores", default="0.55,0.65,0.75")
    ap.add_argument("--max-small-sizes", default="1,3,8,20")
    ap.add_argument("--max-large-sizes", default="300,500")
    ap.add_argument("--max-conflict-nodes", default="1,2,4,8")
    ap.add_argument("--max-conflict-fracs", default="0.01,0.03,0.05")
    ap.add_argument("--min-compatible-sizes", default="8,32,64")
    ap.add_argument("--max-repairs", default="1,3,8,16")
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
    label_cache: dict[tuple, np.ndarray] = {}
    for threshold in _parse_floats(args.thresholds):
        for min_edge_score in _parse_floats(args.min_edge_scores):
            for max_small in _parse_ints(args.max_small_sizes):
                for max_large in _parse_ints(args.max_large_sizes):
                    for max_conflict in _parse_ints(args.max_conflict_nodes):
                        for max_frac in _parse_floats(args.max_conflict_fracs):
                            for min_compat in _parse_ints(args.min_compatible_sizes):
                                for max_repairs in _parse_ints(args.max_repairs):
                                    labels, info = _peel_merge(
                                        base_labels,
                                        keep_indices,
                                        edge_rows,
                                        forbidden,
                                        probability_field=str(args.probability_field),
                                        threshold=float(threshold),
                                        min_edge_score=float(min_edge_score),
                                        max_small_size=int(max_small),
                                        max_large_size=int(max_large),
                                        max_conflict_nodes=int(max_conflict),
                                        max_conflict_frac=float(max_frac),
                                        min_compatible_size=int(min_compat),
                                        max_repairs=int(max_repairs),
                                    )
                                    pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
                                    pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
                                    key = (
                                        float(threshold),
                                        float(min_edge_score),
                                        int(max_small),
                                        int(max_large),
                                        int(max_conflict),
                                        float(max_frac),
                                        int(min_compat),
                                        int(max_repairs),
                                    )
                                    label_cache[key] = labels
                                    rows.append({"mode": "continuation_peel_merge", **info, **pair, "uses_anchors": False, "uses_gt_for_training_or_anchors": False, "uses_gt_for_evaluation_only": True})
    rows.sort(key=lambda row: (float(row["tracklet_pair_f1"]), float(row["tracklet_pair_recall"]), float(row["tracklet_pair_precision"])), reverse=True)

    full_rows = []
    for row in rows[: max(int(args.full_top_n), 0)]:
        key = (
            float(row["peel_threshold"]),
            float(row["peel_min_edge_score"]),
            int(row["peel_max_small_size"]),
            int(row["peel_max_large_size"]),
            int(row["peel_max_conflict_nodes"]),
            float(row["peel_max_conflict_frac"]),
            int(row["peel_min_compatible_size"]),
            int(row["peel_max_repairs"]),
        )
        labels = label_cache[key]
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
        "top": rows[:120],
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
