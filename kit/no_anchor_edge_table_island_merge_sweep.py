#!/usr/bin/env python
"""No-anchor component-island merge sweep from an edge feature table.

This is a deliberately narrow self-play proposer.  It consumes an existing
assignment plus a no-GT component edge table, selects small identity-island
merge candidates from no-GT evidence only, and evaluates the resulting global
ID assignment after the fact.

The selection path ignores all ``gt_*`` columns.  Ground truth is loaded only
after the new labels are formed, for pair/full metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _unionfind_from_labels, _write_csv
    from kit.no_anchor_edge_table_target_repair_sweep import _load_edge_rows
    from kit.no_anchor_louvain_sweep import _write_assignments
    from kit.no_anchor_resolve_sweep import (
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
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _unionfind_from_labels, _write_csv
    from no_anchor_edge_table_target_repair_sweep import _load_edge_rows
    from no_anchor_louvain_sweep import _write_assignments
    from no_anchor_resolve_sweep import (
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


def _forbidden_ok(value: int, mode: str) -> bool:
    if mode == "any":
        return True
    if mode == "on":
        return int(value) > 0
    if mode == "off":
        return int(value) <= 0
    raise ValueError(f"unknown forbidden mode: {mode!r}")


def _edge_score(row: dict[str, object]) -> float:
    """No-GT priority score.  High fused agreement is the main signal."""

    fused = float(row["fused_sim"])
    score = float(row["score"])
    db = float(row["db_sim"])
    primary = float(row["primary_sim"])
    votes = float(row["votes_top5"])
    rank = float(row["fused_rank_max"])
    small = float(min(int(row["source_size"]), int(row["target_size"])))
    size_penalty = 0.012 * max(small - 1.0, 0.0)
    return float(1.60 * fused + 0.25 * score + 0.10 * db + 0.05 * primary + 0.025 * votes - 0.003 * rank - size_penalty)


def _eligible_edges(
    rows: list[dict[str, object]],
    *,
    min_score: float,
    max_score: float,
    min_fused_sim: float,
    max_fused_rank: int,
    max_db_rank_min: int,
    min_votes_top5: int,
    max_small_size: int,
    min_large_size: int,
    forbidden_mode: str,
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if float(row["score"]) < float(min_score) or float(row["score"]) > float(max_score):
            continue
        if float(row["fused_sim"]) < float(min_fused_sim):
            continue
        if int(row["fused_rank_max"]) > int(max_fused_rank):
            continue
        if int(row["db_rank_min"]) > int(max_db_rank_min):
            continue
        if int(row["votes_top5"]) < int(min_votes_top5):
            continue
        if min(int(row["source_size"]), int(row["target_size"])) > int(max_small_size):
            continue
        if max(int(row["source_size"]), int(row["target_size"])) < int(min_large_size):
            continue
        if not _forbidden_ok(int(row["is_forbidden"]), forbidden_mode):
            continue
        out.append(row)
    out.sort(key=_edge_score, reverse=True)
    return out


def _apply_edges(
    base_labels: np.ndarray,
    keep_indices: set[int],
    edges: list[dict[str, object]],
    *,
    max_new_size: int,
    max_growth_ratio: float,
    max_edges: int,
    max_degree: int,
) -> tuple[np.ndarray, dict[str, object]]:
    uf = _unionfind_from_labels(base_labels)
    root_size = Counter(int(uf.find(idx)) for idx in keep_indices)
    base_degree: Counter[int] = Counter()
    accepted: list[dict[str, object]] = []
    rejected_stale = 0
    rejected_size = 0
    rejected_growth = 0
    rejected_degree = 0
    rejected_budget = 0

    for row in edges:
        if int(max_edges) > 0 and len(accepted) >= int(max_edges):
            rejected_budget += 1
            break
        source = int(row["source"])
        target = int(row["target"])
        if int(max_degree) > 0 and (base_degree[source] >= int(max_degree) or base_degree[target] >= int(max_degree)):
            rejected_degree += 1
            continue
        a = int(row["source_rep"])
        b = int(row["target_rep"])
        ra = int(uf.find(a))
        rb = int(uf.find(b))
        if ra == rb:
            rejected_stale += 1
            continue
        size_a = int(root_size.get(ra, 0))
        size_b = int(root_size.get(rb, 0))
        new_size = int(size_a + size_b)
        if int(max_new_size) > 0 and new_size > int(max_new_size):
            rejected_size += 1
            continue
        growth = float(min(size_a, size_b) / max(max(size_a, size_b), 1))
        if growth > float(max_growth_ratio):
            rejected_growth += 1
            continue
        uf.merge(a, b)
        new_root = int(uf.find(a))
        root_size[new_root] = new_size
        root_size.pop(ra, None)
        root_size.pop(rb, None)
        base_degree[source] += 1
        base_degree[target] += 1
        accepted.append(
            {
                "source": source,
                "target": target,
                "source_size": int(row["source_size"]),
                "target_size": int(row["target_size"]),
                "new_size": int(new_size),
                "growth_ratio": round(growth, 6),
                "edge_score": round(_edge_score(row), 6),
                "score": round(float(row["score"]), 6),
                "fused_sim": round(float(row["fused_sim"]), 6),
                "db_sim": round(float(row["db_sim"]), 6),
                "primary_sim": round(float(row["primary_sim"]), 6),
                "fused_rank_max": int(row["fused_rank_max"]),
                "votes_top5": int(row["votes_top5"]),
                "is_forbidden": int(row["is_forbidden"]),
            }
        )
    labels = uf.labels()
    keep_labels = [int(labels[idx]) for idx in keep_indices]
    return labels, {
        "accepted_edges": int(len(accepted)),
        "rewritten_components": int(len({x for row in accepted for x in (row["source"], row["target"])})),
        "largest_component": int(max(Counter(keep_labels).values(), default=0)),
        "components": int(len(set(keep_labels))),
        "rejected_stale": int(rejected_stale),
        "rejected_size": int(rejected_size),
        "rejected_growth": int(rejected_growth),
        "rejected_degree": int(rejected_degree),
        "rejected_budget": int(rejected_budget),
        "accepted_preview": accepted[:30],
        "uses_ground_truth": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--edge-csv", required=True)
    ap.add_argument("--min-scores", default="0.45,0.55,0.65")
    ap.add_argument("--max-scores", default="0.86,0.90,1.01")
    ap.add_argument("--min-fused-sims", default="0.78,0.80,0.82")
    ap.add_argument("--max-fused-ranks", default="1,3,5,10,20")
    ap.add_argument("--max-db-rank-mins", default="1,5,10")
    ap.add_argument("--min-votes-top5", default="0,1,2")
    ap.add_argument("--max-small-sizes", default="1,2,4,8")
    ap.add_argument("--min-large-sizes", default="32,64,96")
    ap.add_argument("--forbidden-modes", default="any,on")
    ap.add_argument("--max-new-sizes", default="256,320,500")
    ap.add_argument("--max-growth-ratios", default="0.02,0.05,0.10")
    ap.add_argument("--max-edges", default="0,8,16,32")
    ap.add_argument("--max-degrees", default="1,2")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=99_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    edge_rows = _load_edge_rows(args.edge_csv)
    con = _connect(args.dbname)
    records, _db_emb = _load_tracklets(con, args.role)
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

    rows: list[dict[str, object]] = []
    label_cache: dict[int, np.ndarray] = {}
    for min_score in _parse_floats(args.min_scores):
        for max_score in _parse_floats(args.max_scores):
            if float(max_score) < float(min_score):
                continue
            for min_fused_sim in _parse_floats(args.min_fused_sims):
                for max_fused_rank in _parse_ints(args.max_fused_ranks):
                    for max_db_rank_min in _parse_ints(args.max_db_rank_mins):
                        for min_votes_top5 in _parse_ints(args.min_votes_top5):
                            for max_small_size in _parse_ints(args.max_small_sizes):
                                for min_large_size in _parse_ints(args.min_large_sizes):
                                    for forbidden_mode in [part.strip() for part in str(args.forbidden_modes).split(",") if part.strip()]:
                                        eligible = _eligible_edges(
                                            edge_rows,
                                            min_score=float(min_score),
                                            max_score=float(max_score),
                                            min_fused_sim=float(min_fused_sim),
                                            max_fused_rank=int(max_fused_rank),
                                            max_db_rank_min=int(max_db_rank_min),
                                            min_votes_top5=int(min_votes_top5),
                                            max_small_size=int(max_small_size),
                                            min_large_size=int(min_large_size),
                                            forbidden_mode=str(forbidden_mode),
                                        )
                                        if not eligible:
                                            continue
                                        for max_new_size in _parse_ints(args.max_new_sizes):
                                            for max_growth_ratio in _parse_floats(args.max_growth_ratios):
                                                for max_edges in _parse_ints(args.max_edges):
                                                    for max_degree in _parse_ints(args.max_degrees):
                                                        labels, info = _apply_edges(
                                                            base_labels,
                                                            keep_indices,
                                                            eligible,
                                                            max_new_size=int(max_new_size),
                                                            max_growth_ratio=float(max_growth_ratio),
                                                            max_edges=int(max_edges),
                                                            max_degree=int(max_degree),
                                                        )
                                                        pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                                        pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                                        row = {
                                                            "mode": "edge_table_island_merge",
                                                            "min_score": float(min_score),
                                                            "max_score": float(max_score),
                                                            "min_fused_sim": float(min_fused_sim),
                                                            "max_fused_rank": int(max_fused_rank),
                                                            "max_db_rank_min": int(max_db_rank_min),
                                                            "min_votes_top5": int(min_votes_top5),
                                                            "max_small_size": int(max_small_size),
                                                            "min_large_size": int(min_large_size),
                                                            "forbidden_mode": str(forbidden_mode),
                                                            "max_new_size": int(max_new_size),
                                                            "max_growth_ratio": float(max_growth_ratio),
                                                            "max_edges": int(max_edges),
                                                            "max_degree": int(max_degree),
                                                            "eligible_edges": int(len(eligible)),
                                                            **{key: value for key, value in info.items() if key != "accepted_preview"},
                                                            "accepted_preview": info["accepted_preview"],
                                                            **pair,
                                                            "uses_anchors": False,
                                                            "uses_gt_for_training_or_anchors": False,
                                                            "uses_gt_for_evaluation_only": True,
                                                        }
                                                        rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
            float(row["tracklet_pair_precision"]),
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        eligible = _eligible_edges(
            edge_rows,
            min_score=float(row["min_score"]),
            max_score=float(row["max_score"]),
            min_fused_sim=float(row["min_fused_sim"]),
            max_fused_rank=int(row["max_fused_rank"]),
            max_db_rank_min=int(row["max_db_rank_min"]),
            min_votes_top5=int(row["min_votes_top5"]),
            max_small_size=int(row["max_small_size"]),
            min_large_size=int(row["min_large_size"]),
            forbidden_mode=str(row["forbidden_mode"]),
        )
        labels, _info = _apply_edges(
            base_labels,
            keep_indices,
            eligible,
            max_new_size=int(row["max_new_size"]),
            max_growth_ratio=float(row["max_growth_ratio"]),
            max_edges=int(row["max_edges"]),
            max_degree=int(row["max_degree"]),
        )
        label_cache[int(rank)] = labels
        full = _score_full(
            pred_by_video,
            gt_by_video,
            _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs),
        )
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": int(rank), "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = label_cache.get(1)
        if labels is None:
            row = rows[0]
            eligible = _eligible_edges(
                edge_rows,
                min_score=float(row["min_score"]),
                max_score=float(row["max_score"]),
                min_fused_sim=float(row["min_fused_sim"]),
                max_fused_rank=int(row["max_fused_rank"]),
                max_db_rank_min=int(row["max_db_rank_min"]),
                min_votes_top5=int(row["min_votes_top5"]),
                max_small_size=int(row["max_small_size"]),
                min_large_size=int(row["min_large_size"]),
                forbidden_mode=str(row["forbidden_mode"]),
            )
            labels, _info = _apply_edges(
                base_labels,
                keep_indices,
                eligible,
                max_new_size=int(row["max_new_size"]),
                max_growth_ratio=float(row["max_growth_ratio"]),
                max_edges=int(row["max_edges"]),
                max_degree=int(row["max_degree"]),
            )
        assignment_info = _write_assignments(
            args.assignments_out,
            records,
            labels,
            keep_seqs=keep_seqs,
            offset=int(args.assignment_offset),
        )
        rows[0].update(assignment_info)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "assignment_csv": str(args.assignment_csv),
        "edge_csv": str(args.edge_csv),
        "base_pair_metrics": base_pair,
        "base_assignment_components": int(len(raw_to_local)),
        "edge_rows_loaded": int(len(edge_rows)),
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "assignment_info": assignment_info,
        "top": rows[: max(100, int(args.full_top_n))],
        "selection_ignored_gt_columns": True,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "base": base_pair, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
