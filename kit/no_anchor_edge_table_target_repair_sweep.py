#!/usr/bin/env python
"""Two-stage no-anchor repair sweep from a component edge feature table.

This script consumes a component-edge table whose candidate edges were generated
without identity ground truth.  It intentionally ignores all ``gt_*`` columns in
the table when selecting repair targets and merge edges.

Stage 1: localize repairable large components from no-GT edge density patterns.
Stage 2: merge only small fragments into those localized targets.

Ground truth is loaded only after the new assignment is formed, for pair/full
metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, KIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _unionfind_from_labels, _write_csv
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


_GT_PREFIXES = ("gt_",)


def _raw_value(row: dict[str, str], key: str, *fallback_keys: str):
    for name in (key, *fallback_keys):
        value = row.get(name, "")
        if value != "" and value is not None:
            return value
    return ""


def _as_float(row: dict[str, str], key: str, default: float = 0.0, *fallback_keys: str) -> float:
    value = _raw_value(row, key, *fallback_keys)
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_int(row: dict[str, str], key: str, default: int = 0, *fallback_keys: str) -> int:
    return int(round(_as_float(row, key, float(default), *fallback_keys)))


def _load_edge_rows(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"source", "target", "source_rep", "target_rep", "source_size", "target_size", "score"}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for raw in reader:
            # Do not copy GT columns into the candidate object used for selection.
            row = {key: value for key, value in raw.items() if not key.startswith(_GT_PREFIXES)}
            source = _as_int(row, "source")
            target = _as_int(row, "target")
            source_size = _as_int(row, "source_size")
            target_size = _as_int(row, "target_size")
            if source_size >= target_size:
                large, small = source, target
                large_rep, small_rep = _as_int(row, "source_rep"), _as_int(row, "target_rep")
                large_size, small_size = source_size, target_size
                large_weight, small_weight = _as_float(row, "source_weight"), _as_float(row, "target_weight")
            else:
                large, small = target, source
                large_rep, small_rep = _as_int(row, "target_rep"), _as_int(row, "source_rep")
                large_size, small_size = target_size, source_size
                large_weight, small_weight = _as_float(row, "target_weight"), _as_float(row, "source_weight")
            row.update(
                {
                    "source": source,
                    "target": target,
                    "source_rep": _as_int(row, "source_rep"),
                    "target_rep": _as_int(row, "target_rep"),
                    "source_size": source_size,
                    "target_size": target_size,
                    "large_component": int(large),
                    "small_component": int(small),
                    "large_rep": int(large_rep),
                    "small_rep": int(small_rep),
                    "large_size": int(large_size),
                    "small_size": int(small_size),
                    "large_weight": float(large_weight),
                    "small_weight": float(small_weight),
                    "score": _as_float(row, "score"),
                    "fused_sim": _as_float(row, "fused_sim", -1.0, "primary_sim", "score"),
                    "db_sim": _as_float(row, "db_sim", -1.0, "primary_sim", "score"),
                    "primary_sim": _as_float(row, "primary_sim", -1.0),
                    "posecolor_sim": _as_float(row, "posecolor_sim", -1.0),
                    "colorhist_sim": _as_float(row, "colorhist_sim", -1.0),
                    "fused_rank_max": _as_int(row, "fused_rank_max", 1_000_000, "primary_rank_max"),
                    "db_rank_min": _as_int(row, "db_rank_min", 1_000_000, "primary_rank_min"),
                    "primary_rank_min": _as_int(row, "primary_rank_min", 1_000_000),
                    "source_rank": _as_int(row, "source_rank", 1_000_000),
                    "target_rank": _as_int(row, "target_rank", 1_000_000),
                    "edge_rank_max": max(
                        _as_int(row, "source_rank", 1_000_000),
                        _as_int(row, "target_rank", 1_000_000),
                    ),
                    "votes_top5": _as_int(row, "votes_top5", 0),
                    "is_forbidden": _as_int(row, "is_forbidden", 0),
                    "rank_margin": _as_float(row, "rank_margin", 0.0),
                    "area_ratio_sim": _as_float(row, "area_ratio_sim", 0.0),
                    "conf_min": _as_float(row, "conf_min", 0.0),
                }
            )
            rows.append(row)
    return rows


def _candidate_score(row: dict[str, object], *, prefer_mid_score: bool) -> float:
    score = float(row["score"])
    if prefer_mid_score:
        score -= max(score - 0.88, 0.0) * 2.0
    view_agree = 0.35 * float(row["fused_sim"]) + 0.20 * float(row["db_sim"]) + 0.10 * float(row["primary_sim"])
    rank_bonus = 0.05 * float(row["votes_top5"]) - 0.002 * float(row["fused_rank_max"])
    small_bonus = 0.03 / max(float(row["small_size"]), 1.0)
    return float(0.45 * score + view_agree + rank_bonus + small_bonus)


def _eligible_edges(
    rows: list[dict[str, object]],
    *,
    min_large_size: int,
    max_small_size: int,
    min_score: float,
    max_score: float,
    min_fused_sim: float,
    max_fused_rank: int,
    max_edge_rank: int,
    max_db_rank_min: int,
    min_votes_top5: int,
    require_forbidden: bool,
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if int(row["large_size"]) < int(min_large_size):
            continue
        if int(row["small_size"]) > int(max_small_size):
            continue
        if float(row["score"]) < float(min_score) or float(row["score"]) > float(max_score):
            continue
        if float(row["fused_sim"]) < float(min_fused_sim):
            continue
        if int(row["fused_rank_max"]) > int(max_fused_rank):
            continue
        if int(row["edge_rank_max"]) > int(max_edge_rank):
            continue
        if int(row["db_rank_min"]) > int(max_db_rank_min):
            continue
        if int(row["votes_top5"]) < int(min_votes_top5):
            continue
        if require_forbidden and int(row["is_forbidden"]) <= 0:
            continue
        out.append(row)
    return out


def _select_edges(
    rows: list[dict[str, object]],
    *,
    min_target_edges: int,
    max_targets: int,
    max_edges_per_target: int,
    prefer_mid_score: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_large: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_large[int(row["large_component"])].append(row)
    target_rows = []
    for large, edges in by_large.items():
        if len(edges) < int(min_target_edges):
            continue
        edges = sorted(edges, key=lambda row: _candidate_score(row, prefer_mid_score=prefer_mid_score), reverse=True)
        target_score = float(
            len(edges)
            + 0.50 * np.mean([float(row["fused_sim"]) for row in edges])
            + 0.25 * np.mean([float(row["db_sim"]) for row in edges])
            + 0.05 * sum(float(row["small_size"]) <= 2 for row in edges)
        )
        target_rows.append(
            {
                "large_component": int(large),
                "target_score": target_score,
                "candidate_edges": int(len(edges)),
                "large_size": int(edges[0]["large_size"]),
                "preview": edges[: int(max_edges_per_target)],
            }
        )
    target_rows.sort(key=lambda row: (float(row["target_score"]), int(row["candidate_edges"])), reverse=True)
    if int(max_targets) > 0:
        target_rows = target_rows[: int(max_targets)]
    selected = []
    used_small: set[int] = set()
    for target in target_rows:
        for edge in target["preview"]:
            small = int(edge["small_component"])
            if small in used_small:
                continue
            selected.append(edge)
            used_small.add(small)
    selected_scores = [_candidate_score(edge, prefer_mid_score=prefer_mid_score) for edge in selected]
    selected_weights = [
        float(edge["score"]) * max(float(edge["fused_sim"]), 0.0) * (1.0 + 0.05 * float(edge["votes_top5"]))
        for edge in selected
    ]
    info = {
        "localized_targets": int(len(target_rows)),
        "selected_edges": int(len(selected)),
        "selection_score": round(float(sum(selected_scores)), 6),
        "selection_score_mean": round(float(np.mean(selected_scores)) if selected_scores else 0.0, 6),
        "selection_weighted_evidence": round(float(sum(selected_weights)), 6),
        "target_preview": [
            {
                "large_component": int(row["large_component"]),
                "target_score": round(float(row["target_score"]), 6),
                "candidate_edges": int(row["candidate_edges"]),
                "large_size": int(row["large_size"]),
                "small_components": [int(edge["small_component"]) for edge in row["preview"]],
            }
            for row in target_rows[:12]
        ],
    }
    return selected, info


def _sort_key(row: dict[str, object], key: str) -> tuple[float, float, float, float]:
    if key == "selection_score":
        return (
            float(row.get("selection_score", 0.0)),
            float(row.get("selection_weighted_evidence", 0.0)),
            float(row.get("selected_edges", 0.0)),
            float(row.get("tracklet_pair_f1", 0.0)),
        )
    if key == "selected_edges":
        return (
            float(row.get("selected_edges", 0.0)),
            float(row.get("selection_score", 0.0)),
            float(row.get("selection_weighted_evidence", 0.0)),
            float(row.get("tracklet_pair_f1", 0.0)),
        )
    if key == "pair_f1":
        return (
            float(row.get("tracklet_pair_f1", 0.0)),
            float(row.get("tracklet_pair_recall", 0.0)),
            float(row.get("tracklet_pair_precision", 0.0)),
            float(row.get("selection_score", 0.0)),
        )
    raise ValueError(f"unknown sort key: {key}")


def _apply_edges(base_labels: np.ndarray, edges: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, object]]:
    uf = _unionfind_from_labels(base_labels)
    accepted = rejected_stale = 0
    for edge in edges:
        a = int(edge["large_rep"])
        b = int(edge["small_rep"])
        if uf.find(a) == uf.find(b):
            rejected_stale += 1
            continue
        uf.merge(a, b)
        accepted += 1
    labels = uf.labels()
    return labels, {"accepted_edges": int(accepted), "rejected_stale": int(rejected_stale)}


def _selected_signature(edges: list[dict[str, object]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(edge["large_rep"]), int(edge["small_rep"])) for edge in edges))


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "edges.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source",
                    "target",
                    "source_rep",
                    "target_rep",
                    "source_size",
                    "target_size",
                    "source_rank",
                    "target_rank",
                    "score",
                    "fused_sim",
                    "db_sim",
                    "primary_sim",
                    "fused_rank_max",
                    "db_rank_min",
                    "votes_top5",
                    "is_forbidden",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source": 0,
                    "target": 1,
                    "source_rep": 10,
                    "target_rep": 11,
                    "source_size": 180,
                    "target_size": 2,
                    "source_rank": 2,
                    "target_rank": 3,
                    "score": 0.80,
                    "fused_sim": 0.75,
                    "db_sim": 0.70,
                    "primary_sim": 0.80,
                    "fused_rank_max": 9,
                    "db_rank_min": 4,
                    "votes_top5": 1,
                    "is_forbidden": 1,
                }
            )
            writer.writerow(
                {
                    "source": 2,
                    "target": 3,
                    "source_rep": 12,
                    "target_rep": 13,
                    "source_size": 180,
                    "target_size": 2,
                    "source_rank": 5,
                    "target_rank": 6,
                    "score": 0.82,
                    "fused_sim": 0.75,
                    "db_sim": 0.70,
                    "primary_sim": 0.80,
                    "fused_rank_max": 9,
                    "db_rank_min": 4,
                    "votes_top5": 1,
                    "is_forbidden": 1,
                }
            )
        rows = _load_edge_rows(str(path))
        assert rows[0]["edge_rank_max"] == 3
        eligible = _eligible_edges(
            rows,
            min_large_size=1,
            max_small_size=2,
            min_score=0.75,
            max_score=1.0,
            min_fused_sim=0.7,
            max_fused_rank=10,
            max_edge_rank=3,
            max_db_rank_min=5,
            min_votes_top5=1,
            require_forbidden=True,
        )
        assert len(eligible) == 1
        assert eligible[0]["source"] == 0
        print(json.dumps({"stage": "self_test", "status": "ok", "eligible_edges": len(eligible)}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", default="")
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--edge-csv", default="")
    ap.add_argument("--min-large-sizes", default="96,128,160")
    ap.add_argument("--max-small-sizes", default="1,2,4")
    ap.add_argument("--min-scores", default="0.55,0.65,0.75")
    ap.add_argument("--max-scores", default="0.86,0.90,1.01")
    ap.add_argument("--min-fused-sims", default="0.40,0.55,0.65")
    ap.add_argument("--max-fused-ranks", default="5,10,20")
    ap.add_argument("--max-edge-ranks", default="1000000")
    ap.add_argument("--max-db-rank-mins", default="1,5,10")
    ap.add_argument("--min-votes-top5", default="0,1,2")
    ap.add_argument("--min-target-edges", default="1,2")
    ap.add_argument("--max-targets", default="0,4,8,16")
    ap.add_argument("--max-edges-per-target", default="1,2,3")
    ap.add_argument("--require-forbidden", action="store_true")
    ap.add_argument("--prefer-mid-score", action="store_true")
    ap.add_argument("--sort-key", default="selection_score", choices=["selection_score", "selected_edges", "pair_f1"])
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=98_000_000)
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.assignment_csv:
        ap.error("--assignment-csv is required unless --self-test is used")
    if not args.edge_csv:
        ap.error("--edge-csv is required unless --self-test is used")
    if not args.json:
        ap.error("--json is required unless --self-test is used")

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
    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    rows = []
    labels_by_rank: dict[int, np.ndarray] = {}
    edit_metric_cache: dict[tuple[tuple[int, int], ...], tuple[np.ndarray, dict[str, object], dict[str, float]]] = {}
    for min_large_size in _parse_ints(args.min_large_sizes):
        for max_small_size in _parse_ints(args.max_small_sizes):
            for min_score in _parse_floats(args.min_scores):
                for max_score in _parse_floats(args.max_scores):
                    if float(max_score) < float(min_score):
                        continue
                    for min_fused_sim in _parse_floats(args.min_fused_sims):
                        for max_fused_rank in _parse_ints(args.max_fused_ranks):
                            for max_edge_rank in _parse_ints(args.max_edge_ranks):
                                for max_db_rank_min in _parse_ints(args.max_db_rank_mins):
                                    for min_votes_top5 in _parse_ints(args.min_votes_top5):
                                        eligible = _eligible_edges(
                                            edge_rows,
                                            min_large_size=int(min_large_size),
                                            max_small_size=int(max_small_size),
                                            min_score=float(min_score),
                                            max_score=float(max_score),
                                            min_fused_sim=float(min_fused_sim),
                                            max_fused_rank=int(max_fused_rank),
                                            max_edge_rank=int(max_edge_rank),
                                            max_db_rank_min=int(max_db_rank_min),
                                            min_votes_top5=int(min_votes_top5),
                                            require_forbidden=bool(args.require_forbidden),
                                        )
                                        for min_target_edges in _parse_ints(args.min_target_edges):
                                            for max_targets in _parse_ints(args.max_targets):
                                                for max_edges_per_target in _parse_ints(args.max_edges_per_target):
                                                    selected, target_info = _select_edges(
                                                        eligible,
                                                        min_target_edges=int(min_target_edges),
                                                        max_targets=int(max_targets),
                                                        max_edges_per_target=int(max_edges_per_target),
                                                        prefer_mid_score=bool(args.prefer_mid_score),
                                                    )
                                                    signature = _selected_signature(selected)
                                                    cached_edit = edit_metric_cache.get(signature)
                                                    if cached_edit is None:
                                                        labels, apply_info = _apply_edges(base_labels, selected)
                                                        pred = _labels_to_seq_map(
                                                            records,
                                                            labels,
                                                            offset=int(args.assignment_offset),
                                                            keep_seqs=keep_seqs,
                                                        )
                                                        pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                                        edit_metric_cache[signature] = (labels, apply_info, pair)
                                                    else:
                                                        _labels, apply_info, pair = cached_edit
                                                    rows.append(
                                                        {
                                                            "mode": "edge_table_target_repair",
                                                            "min_large_size": int(min_large_size),
                                                            "max_small_size": int(max_small_size),
                                                            "min_score": float(min_score),
                                                            "max_score": float(max_score),
                                                            "min_fused_sim": float(min_fused_sim),
                                                            "max_fused_rank": int(max_fused_rank),
                                                            "max_edge_rank": int(max_edge_rank),
                                                            "max_db_rank_min": int(max_db_rank_min),
                                                            "min_votes_top5": int(min_votes_top5),
                                                            "min_target_edges": int(min_target_edges),
                                                            "max_targets": int(max_targets),
                                                            "max_edges_per_target": int(max_edges_per_target),
                                                            "require_forbidden": bool(args.require_forbidden),
                                                            "prefer_mid_score": bool(args.prefer_mid_score),
                                                            "eligible_edges": int(len(eligible)),
                                                            **target_info,
                                                            **apply_info,
                                                            **pair,
                                                            "uses_anchors": False,
                                                            "uses_gt_for_training_or_anchors": False,
                                                            "uses_gt_for_evaluation_only": True,
                                                        }
                                                    )
    rows.sort(key=lambda row: _sort_key(row, str(args.sort_key)), reverse=True)

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        eligible = _eligible_edges(
            edge_rows,
            min_large_size=int(row["min_large_size"]),
            max_small_size=int(row["max_small_size"]),
            min_score=float(row["min_score"]),
            max_score=float(row["max_score"]),
            min_fused_sim=float(row["min_fused_sim"]),
            max_fused_rank=int(row["max_fused_rank"]),
            max_edge_rank=int(row.get("max_edge_rank", 1_000_000)),
            max_db_rank_min=int(row["max_db_rank_min"]),
            min_votes_top5=int(row["min_votes_top5"]),
            require_forbidden=bool(row["require_forbidden"]),
        )
        selected, _target_info = _select_edges(
            eligible,
            min_target_edges=int(row["min_target_edges"]),
            max_targets=int(row["max_targets"]),
            max_edges_per_target=int(row["max_edges_per_target"]),
            prefer_mid_score=bool(row["prefer_mid_score"]),
        )
        signature = _selected_signature(selected)
        cached_edit = edit_metric_cache.get(signature)
        if cached_edit is None:
            labels, apply_info = _apply_edges(base_labels, selected)
            pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
            pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
            edit_metric_cache[signature] = (labels, apply_info, pair)
        else:
            labels, _apply_info, _pair = cached_edit
        labels_by_rank[int(rank)] = labels
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
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            eligible = _eligible_edges(
                edge_rows,
                min_large_size=int(row["min_large_size"]),
                max_small_size=int(row["max_small_size"]),
                min_score=float(row["min_score"]),
                max_score=float(row["max_score"]),
                min_fused_sim=float(row["min_fused_sim"]),
                max_fused_rank=int(row["max_fused_rank"]),
                max_edge_rank=int(row.get("max_edge_rank", 1_000_000)),
                max_db_rank_min=int(row["max_db_rank_min"]),
                min_votes_top5=int(row["min_votes_top5"]),
                require_forbidden=bool(row["require_forbidden"]),
            )
            selected, _target_info = _select_edges(
                eligible,
                min_target_edges=int(row["min_target_edges"]),
                max_targets=int(row["max_targets"]),
                max_edges_per_target=int(row["max_edges_per_target"]),
                prefer_mid_score=bool(row["prefer_mid_score"]),
            )
            signature = _selected_signature(selected)
            cached_edit = edit_metric_cache.get(signature)
            if cached_edit is None:
                labels, apply_info = _apply_edges(base_labels, selected)
                pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                edit_metric_cache[signature] = (labels, apply_info, pair)
            else:
                labels, _apply_info, _pair = cached_edit
        assignment_info = _write_assignments(
            args.assignments_out,
            records,
            labels,
            keep_seqs=keep_seqs,
            offset=int(args.assignment_offset),
        )
        rows[0].update(assignment_info)

    result = {
        "assignment_csv": str(args.assignment_csv),
        "edge_csv": str(args.edge_csv),
        "base_pair_metrics": base_pair,
        "base_assignment_components": int(len(raw_to_local)),
        "edge_rows_loaded": int(len(edge_rows)),
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "assignment_info": assignment_info,
        "edit_metric_cache_entries": int(len(edit_metric_cache)),
        "top": rows[: max(100, int(args.full_top_n))],
        "selection_ignored_gt_columns": True,
        "sort_key": str(args.sort_key),
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
