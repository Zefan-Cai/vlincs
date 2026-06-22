#!/usr/bin/env python
"""Diagnose whether no-anchor component candidates can recover false splits.

This is an eval-only analysis.  It reconstructs the base no-anchor resolver,
builds component-level candidate edges from no-GT evidence, then uses cached GT
labels only to measure whether same-identity split components are retrieved in
the candidate graph.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_component_merge_sweep import _candidate_edges, _component_members
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _time_agglom_resolve,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_component_merge_sweep import _candidate_edges, _component_members
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _time_agglom_resolve,
        _with_detection_endpoints,
    )


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _component_gt_tables(records, members, gt_by_seq, weight_by_seq):
    comp_gt_weight: list[dict[int, float]] = []
    comp_total_weight: list[float] = []
    comp_dominant: list[int | None] = []
    comp_purity: list[float] = []
    for indices in members:
        by_gt: dict[int, float] = defaultdict(float)
        total = 0.0
        for idx in indices:
            seq = int(records[idx].seq)
            if seq not in gt_by_seq:
                continue
            w = float(weight_by_seq.get(seq, 1.0))
            by_gt[int(gt_by_seq[seq])] += w
            total += w
        comp_gt_weight.append(dict(by_gt))
        comp_total_weight.append(float(total))
        if by_gt and total > 0:
            gt, weight = max(by_gt.items(), key=lambda item: item[1])
            comp_dominant.append(int(gt))
            comp_purity.append(float(weight / max(total, 1.0e-9)))
        else:
            comp_dominant.append(None)
            comp_purity.append(0.0)
    return comp_gt_weight, np.asarray(comp_total_weight, dtype=np.float64), comp_dominant, np.asarray(comp_purity, dtype=np.float64)


def _split_pairs_by_gt(comp_gt_weight):
    by_gt: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for comp_idx, table in enumerate(comp_gt_weight):
        for gt, weight in table.items():
            if weight > 0:
                by_gt[int(gt)].append((int(comp_idx), float(weight)))
    pairs: list[tuple[int, int, int, float]] = []
    total = 0.0
    by_gt_mass: dict[int, float] = defaultdict(float)
    for gt, items in by_gt.items():
        if len(items) < 2:
            continue
        for a_pos in range(len(items)):
            a, wa = items[a_pos]
            for b, wb in items[a_pos + 1 :]:
                mass = float(wa * wb)
                if mass <= 0:
                    continue
                pairs.append((int(gt), int(a), int(b), mass))
                total += mass
                by_gt_mass[int(gt)] += mass
    return pairs, float(total), by_gt_mass


def _edge_lookup(edges):
    scores: dict[tuple[int, int], float] = {}
    neighbors: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for edge in edges:
        a = int(edge["source"])
        b = int(edge["target"])
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        score = float(edge["score"])
        if score > scores.get(key, -2.0):
            scores[key] = score
    for (a, b), score in scores.items():
        neighbors[a].append((score, b))
        neighbors[b].append((score, a))
    for values in neighbors.values():
        values.sort(reverse=True, key=lambda item: item[0])
    return scores, neighbors


def _retrieval_summary(split_pairs, scores, neighbors, ks: list[int], thresholds: list[float]):
    total = sum(mass for _gt, _a, _b, mass in split_pairs)
    retrieved_at_k = {int(k): 0.0 for k in ks}
    retrieved_at_thr = {float(thr): 0.0 for thr in thresholds}
    scored_mass = 0.0
    score_values = []
    for _gt, a, b, mass in split_pairs:
        key = (a, b) if a < b else (b, a)
        score = scores.get(key)
        if score is not None:
            scored_mass += mass
            score_values.append(float(score))
        for k in ks:
            top_a = {idx for _score, idx in neighbors.get(a, [])[: int(k)]}
            top_b = {idx for _score, idx in neighbors.get(b, [])[: int(k)]}
            if b in top_a or a in top_b:
                retrieved_at_k[int(k)] += mass
        if score is not None:
            for thr in thresholds:
                if float(score) >= float(thr):
                    retrieved_at_thr[float(thr)] += mass
    return {
        "false_split_pair_mass": round(float(total), 3),
        "candidate_scored_mass": round(float(scored_mass), 3),
        "candidate_scored_fraction": round(float(scored_mass / total), 6) if total > 0 else 0.0,
        "retrieved_at_k": {
            str(k): {
                "mass": round(float(value), 3),
                "fraction": round(float(value / total), 6) if total > 0 else 0.0,
            }
            for k, value in retrieved_at_k.items()
        },
        "retrieved_at_threshold": {
            str(thr): {
                "mass": round(float(value), 3),
                "fraction": round(float(value / total), 6) if total > 0 else 0.0,
            }
            for thr, value in retrieved_at_thr.items()
        },
        "same_gt_score_quantiles": {
            str(q): round(float(np.quantile(score_values, q)), 6) if score_values else None
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]
        },
    }


def _edge_precision_summary(edges, comp_dominant, comp_purity, comp_gt_weight, thresholds: list[float], min_purity: float):
    rows = []
    for thr in thresholds:
        cand = [edge for edge in edges if float(edge["score"]) >= float(thr)]
        tp = 0
        fp = 0
        weighted_tp = 0.0
        weighted_fp = 0.0
        for edge in cand:
            a = int(edge["source"])
            b = int(edge["target"])
            dom_a = comp_dominant[a]
            dom_b = comp_dominant[b]
            mass = 0.0
            common = set(comp_gt_weight[a]) & set(comp_gt_weight[b])
            for gt in common:
                mass += float(comp_gt_weight[a][gt] * comp_gt_weight[b][gt])
            pair_weight = max(
                float(sum(comp_gt_weight[a].values()) * sum(comp_gt_weight[b].values())),
                1.0,
            )
            is_tp = dom_a is not None and dom_a == dom_b and comp_purity[a] >= min_purity and comp_purity[b] >= min_purity
            if is_tp:
                tp += 1
                weighted_tp += mass
            else:
                fp += 1
                weighted_fp += max(pair_weight - mass, 0.0)
        rows.append(
            {
                "threshold": float(thr),
                "candidate_edges": int(len(cand)),
                "dominant_pure_tp_edges": int(tp),
                "dominant_pure_fp_edges": int(fp),
                "edge_precision": round(float(tp / len(cand)), 6) if cand else 0.0,
                "weighted_tp_mass": round(float(weighted_tp), 3),
                "weighted_fp_proxy": round(float(weighted_fp), 3),
            }
        )
    return rows


def _top_gt_rows(split_pairs, by_gt_mass, scores, neighbors, top_n: int):
    by_gt_pairs: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for gt, a, b, mass in split_pairs:
        by_gt_pairs[int(gt)].append((int(a), int(b), float(mass)))
    rows = []
    for gt, pairs in by_gt_pairs.items():
        total = float(by_gt_mass[gt])
        scored = 0.0
        r10 = 0.0
        best_score = -2.0
        comps = set()
        for a, b, mass in pairs:
            comps.add(a)
            comps.add(b)
            key = (a, b) if a < b else (b, a)
            score = scores.get(key)
            if score is not None:
                scored += mass
                best_score = max(best_score, float(score))
            top_a = {idx for _score, idx in neighbors.get(a, [])[:10]}
            top_b = {idx for _score, idx in neighbors.get(b, [])[:10]}
            if b in top_a or a in top_b:
                r10 += mass
        rows.append(
            {
                "gt_id": int(gt),
                "component_count": int(len(comps)),
                "false_split_mass": round(total, 3),
                "candidate_scored_fraction": round(float(scored / total), 6) if total > 0 else 0.0,
                "retrieved_at_10_fraction": round(float(r10 / total), 6) if total > 0 else 0.0,
                "best_same_gt_candidate_score": round(float(best_score), 6) if best_score > -1.5 else None,
            }
        )
    rows.sort(key=lambda row: float(row["false_split_mass"]), reverse=True)
    return rows[: int(top_n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--assignment-csv", default="", help="optional existing assignment CSV to diagnose instead of rebuilding resolver labels")
    ap.add_argument("--pred-col", default="predicted_global_id")
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
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weight", type=float, default=0.0)
    ap.add_argument("--forbid-camera-overlap", action="store_true")
    ap.add_argument("--forbid-video-overlap", action="store_true")
    ap.add_argument("--retrieval-ks", default="1,5,10,20,50")
    ap.add_argument("--thresholds", default="0.54,0.58,0.62,0.66,0.70,0.74")
    ap.add_argument("--min-purity", type=float, default=0.75)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

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
    if args.assignment_csv:
        pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
        keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}

    cfg = ResolveConfig(
        mode="assignment_csv" if args.assignment_csv else "time_agglom",
        theta=float(args.theta),
        top_k=int(args.top_k),
        min_dets=int(args.min_dets),
        exclude_same=str(args.exclude_same),
        temporal_bonus=float(args.temporal_bonus),
        time_window_ms=int(args.time_window_ms),
    )
    if args.assignment_csv:
        labels, raw_to_local = _labels_from_assignment(records, pred_input)
        resolve_info = {
            "mode": "assignment_csv",
            "assignment_csv": str(args.assignment_csv),
            "pred_col": str(args.pred_col),
            "input_components": int(len(raw_to_local)),
            "input_assigned_tracklets": int(len(pred_input)),
        }
    else:
        labels, resolve_info = _time_agglom_resolve(records, emb, cfg)
    reps, members = _component_members(labels, keep_indices)
    edges, edge_info = _candidate_edges(
        records,
        emb,
        reps,
        members,
        candidate_top_k=int(args.candidate_top_k),
        top_edge_k=int(args.top_edge_k),
        centroid_weight=float(args.centroid_weight),
        min_source_size=1,
        max_source_size=1000000,
        min_target_size=1,
        max_target_size=1000000,
        forbid_camera_overlap=bool(args.forbid_camera_overlap),
        forbid_video_overlap=bool(args.forbid_video_overlap),
    )
    comp_gt_weight, comp_total, comp_dominant, comp_purity = _component_gt_tables(records, members, gt_by_seq, weight_by_seq)
    split_pairs, total_split_mass, by_gt_mass = _split_pairs_by_gt(comp_gt_weight)
    scores, neighbors = _edge_lookup(edges)
    ks = _parse_ints(args.retrieval_ks)
    thresholds = _parse_floats(args.thresholds)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "assignment_csv": str(args.assignment_csv) if args.assignment_csv else None,
        "pred_col": str(args.pred_col),
        "resolve_config": cfg.__dict__,
        "resolve_info": resolve_info,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "edge_info": edge_info,
        "component_eval": {
            "components": int(len(members)),
            "gt_labeled_components": int(np.count_nonzero(comp_total > 0)),
            "mean_component_purity": round(float(np.mean(comp_purity[comp_total > 0])) if np.any(comp_total > 0) else 0.0, 6),
            "pure_components_ge_min_purity": int(np.count_nonzero(comp_purity >= float(args.min_purity))),
            "false_split_pair_count": int(len(split_pairs)),
            "false_split_pair_mass": round(float(total_split_mass), 3),
        },
        "retrieval": _retrieval_summary(split_pairs, scores, neighbors, ks, thresholds),
        "edge_precision": _edge_precision_summary(
            edges,
            comp_dominant,
            comp_purity,
            comp_gt_weight,
            thresholds,
            float(args.min_purity),
        ),
        "top_gt_false_splits": _top_gt_rows(split_pairs, by_gt_mass, scores, neighbors, int(args.top_n)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"json": str(out), "retrieval": result["retrieval"], "edge_precision": result["edge_precision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
