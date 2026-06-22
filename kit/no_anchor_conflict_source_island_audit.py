#!/usr/bin/env python
"""Audit no-anchor conflict-source islands before target reassignment.

The construction path is no-anchor: source islands are generated from current
assignment components, visual evidence, and same-stream cannot-link conflicts.
GT labels are loaded only after candidates exist to diagnose whether the source
generator contains repairable identity islands.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, KIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from kit.no_anchor_assignment_conflict_reassign_sweep import _component_groups
    from kit.no_anchor_assignment_provisional_relink_sweep import (
        _l2n,
        _load_npz_aligned,
        _parse_view,
        _source_candidates,
    )
    from kit.no_anchor_assignment_state_policy_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from kit.no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_conflict_reassign_sweep import _component_groups
    from no_anchor_assignment_provisional_relink_sweep import (
        _l2n,
        _load_npz_aligned,
        _parse_view,
        _source_candidates,
    )
    from no_anchor_assignment_state_policy_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _with_detection_endpoints,
    )


def _admission_args(args) -> SimpleNamespace:
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


def _gt_weight_by_group(indices: list[int], records, gt_by_seq, weight_by_seq) -> tuple[Counter, float]:
    by_gt: Counter = Counter()
    total = 0.0
    for idx in indices:
        seq = int(records[int(idx)].seq)
        if seq not in gt_by_seq:
            continue
        weight = float(weight_by_seq.get(seq, 1.0))
        by_gt[int(gt_by_seq[seq])] += weight
        total += weight
    return by_gt, float(total)


def _dominant(counter: Counter, total: float) -> tuple[int | None, float, int]:
    if not counter or total <= 0:
        return None, 0.0, 0
    key, value = counter.most_common(1)[0]
    return int(key), float(value / max(total, 1e-9)), int(len(counter))


def _sub_counter(a: Counter, b: Counter) -> Counter:
    out: Counter = Counter()
    for key, value in a.items():
        remain = float(value) - float(b.get(key, 0.0))
        if remain > 1e-9:
            out[key] = remain
    return out


def _metric_from_masses(true_pair_mass: float, pred_pair_mass: float, gt_pair_mass: float) -> dict[str, float]:
    precision = true_pair_mass / pred_pair_mass if pred_pair_mass > 0 else 0.0
    recall = true_pair_mass / gt_pair_mass if gt_pair_mass > 0 else 0.0
    f1 = 2.0 * true_pair_mass / (pred_pair_mass + gt_pair_mass) if (pred_pair_mass + gt_pair_mass) > 0 else 0.0
    return {
        "tracklet_pair_f1": round(float(f1), 6),
        "tracklet_pair_precision": round(float(precision), 6),
        "tracklet_pair_recall": round(float(recall), 6),
    }


def _oracle_best_reassign(
    source_gt: Counter,
    source_total: float,
    source_component: int,
    component_gt: dict[int, Counter],
    component_total: dict[int, float],
    *,
    base_true_pair_mass: float,
    base_pred_pair_mass: float,
    gt_pair_mass: float,
    min_target_weight: float,
) -> dict[str, object]:
    source_component_gt = component_gt[int(source_component)]
    source_component_total = float(component_total[int(source_component)])
    rest_gt = _sub_counter(source_component_gt, source_gt)
    rest_total = max(source_component_total - float(source_total), 0.0)
    best = {
        "oracle_target_component": None,
        "oracle_delta_true_pair_mass": 0.0,
        "oracle_delta_pred_pair_mass": 0.0,
        "oracle_pair_f1": _metric_from_masses(base_true_pair_mass, base_pred_pair_mass, gt_pair_mass)["tracklet_pair_f1"],
        "oracle_delta_pair_f1": 0.0,
        "oracle_pair_precision": _metric_from_masses(base_true_pair_mass, base_pred_pair_mass, gt_pair_mass)[
            "tracklet_pair_precision"
        ],
        "oracle_pair_recall": _metric_from_masses(base_true_pair_mass, base_pred_pair_mass, gt_pair_mass)["tracklet_pair_recall"],
    }
    base_f1 = float(best["oracle_pair_f1"])
    for target_component, target_gt in component_gt.items():
        target_component = int(target_component)
        if target_component == int(source_component):
            continue
        target_total = float(component_total[target_component])
        if target_total < float(min_target_weight):
            continue
        delta_true = 0.0
        for gt, weight in source_gt.items():
            delta_true += float(weight) * (float(target_gt.get(gt, 0.0)) - float(rest_gt.get(gt, 0.0)))
        delta_pred = float(source_total) * (target_total - rest_total)
        true_pair_mass = float(base_true_pair_mass) + delta_true
        pred_pair_mass = float(base_pred_pair_mass) + delta_pred
        metric = _metric_from_masses(true_pair_mass, pred_pair_mass, gt_pair_mass)
        if float(metric["tracklet_pair_f1"]) > float(best["oracle_pair_f1"]):
            best = {
                "oracle_target_component": int(target_component),
                "oracle_delta_true_pair_mass": round(float(delta_true), 3),
                "oracle_delta_pred_pair_mass": round(float(delta_pred), 3),
                "oracle_pair_f1": float(metric["tracklet_pair_f1"]),
                "oracle_delta_pair_f1": round(float(metric["tracklet_pair_f1"]) - base_f1, 6),
                "oracle_pair_precision": float(metric["tracklet_pair_precision"]),
                "oracle_pair_recall": float(metric["tracklet_pair_recall"]),
            }
    return best


def _topk_summary(rows: list[dict[str, object]], ks: list[int]) -> list[dict[str, object]]:
    out = []
    for k in ks:
        subset = rows[: min(int(k), len(rows))]
        if not subset:
            continue
        positive = [row for row in subset if float(row.get("oracle_delta_pair_f1", 0.0)) > 0.0]
        out.append(
            {
                "top_k": int(k),
                "rows": int(len(subset)),
                "oracle_positive": int(len(positive)),
                "oracle_positive_frac": round(float(len(positive) / max(len(subset), 1)), 6),
                "max_oracle_delta_pair_f1": max(float(row.get("oracle_delta_pair_f1", 0.0)) for row in subset),
                "sum_positive_oracle_delta_pair_f1": round(
                    float(sum(max(float(row.get("oracle_delta_pair_f1", 0.0)), 0.0) for row in subset)),
                    6,
                ),
            }
        )
    return out


def _self_test() -> None:
    base = _metric_from_masses(10.0, 20.0, 25.0)
    assert base["tracklet_pair_f1"] == 0.444444, base
    assert _sub_counter(Counter({1: 3.0, 2: 1.0}), Counter({1: 2.0})) == Counter({1: 1.0, 2: 1.0})
    print(json.dumps({"stage": "self_test", "status": "ok"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", default="")
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--primary-feature-npz", default="")
    ap.add_argument("--view", action="append", default=[], help="feature view name:path[:weight]")
    ap.add_argument("--source-min-component-sizes", default="64,128")
    ap.add_argument("--source-max-component-sizes", default="1000000")
    ap.add_argument("--source-seed-sims", default="0.74,0.76,0.78,0.80")
    ap.add_argument("--source-expand-sims", default="0.70,0.72,0.74")
    ap.add_argument("--source-top-ks", default="8")
    ap.add_argument("--source-min-group-sizes", default="2,3")
    ap.add_argument("--source-max-group-sizes", default="8,12")
    ap.add_argument("--source-min-conflicts-to-rest", default="1")
    ap.add_argument("--source-min-margins", default="0.03")
    ap.add_argument("--source-max-groups-per-component", default="1")
    ap.add_argument("--source-max-total-groups", default="16,32,64")
    ap.add_argument("--dedup-keep", default="score", choices=["score", "oracle"])
    ap.add_argument("--oracle-min-target-weight", type=float, default=1.0)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.assignment_csv or not args.primary_feature_npz or not args.json:
        raise SystemExit("--assignment-csv, --primary-feature-npz, and --json are required")

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    primary = _load_feature_npz(args.primary_feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0)
    views: list[dict[str, object]] = [
        {"name": "primary", "path": str(args.primary_feature_npz), "weight": 1.0, "emb": _l2n(primary.astype(np.float32))}
    ]
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
    keep_seqs, output_info = _output_keep_seqs(records, _admission_args(args))
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    base_labels, keep_indices, raw_to_local = _labels_from_assignment(records, pred_input)
    keep_indices = {idx for idx in keep_indices if int(records[idx].seq) in keep_seqs}
    base_pred = {
        int(record.seq): int(base_labels[idx])
        for idx, record in enumerate(records)
        if int(record.seq) in keep_seqs
    }
    seqs = [int(record.seq) for record in records]
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    groups = _component_groups(base_labels, keep_indices)
    component_gt: dict[int, Counter] = {}
    component_total: dict[int, float] = {}
    for label, indices in groups.items():
        by_gt, total = _gt_weight_by_group(indices, records, gt_by_seq, weight_by_seq)
        component_gt[int(label)] = by_gt
        component_total[int(label)] = total

    forbidden = _build_overlap_forbidden(records)
    dedup: dict[tuple[int, ...], dict[str, object]] = {}
    source_grid = list(
        product(
            _parse_ints(args.source_min_component_sizes),
            _parse_ints(args.source_max_component_sizes),
            _parse_floats(args.source_seed_sims),
            _parse_floats(args.source_expand_sims),
            _parse_ints(args.source_top_ks),
            _parse_ints(args.source_min_group_sizes),
            _parse_ints(args.source_max_group_sizes),
            _parse_ints(args.source_min_conflicts_to_rest),
            _parse_floats(args.source_min_margins),
            _parse_ints(args.source_max_groups_per_component),
            _parse_ints(args.source_max_total_groups),
        )
    )
    grid_rows = 0
    for source_params in source_grid:
        (
            min_component_size,
            max_component_size,
            seed_sim,
            expand_sim,
            top_k,
            min_group_size,
            max_group_size,
            min_conflicts,
            min_margin,
            max_groups_per_component,
            max_total_groups,
        ) = source_params
        if float(expand_sim) > float(seed_sim) or int(max_group_size) < int(min_group_size):
            continue
        grid_rows += 1
        sources, source_info = _source_candidates(
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
        for source in sources:
            source_indices = [int(idx) for idx in source["source_indices"]]
            source_gt, source_total = _gt_weight_by_group(source_indices, records, gt_by_seq, weight_by_seq)
            source_dom, source_dom_frac, source_gt_count = _dominant(source_gt, source_total)
            component_label = int(source["source_component_label"])
            comp_dom, comp_dom_frac, comp_gt_count = _dominant(component_gt[component_label], component_total[component_label])
            oracle = _oracle_best_reassign(
                source_gt,
                source_total,
                component_label,
                component_gt,
                component_total,
                base_true_pair_mass=float(base_pair["true_pair_mass"]),
                base_pred_pair_mass=float(base_pair["pred_pair_mass"]),
                gt_pair_mass=float(base_pair["gt_pair_mass"]),
                min_target_weight=float(args.oracle_min_target_weight),
            )
            row = {
                **{k: v for k, v in source.items() if k != "source_indices"},
                **{k: v for k, v in source_info.items() if k != "source_preview"},
                "source_eval_weight": round(float(source_total), 3),
                "source_majority_gt": source_dom,
                "source_majority_gt_frac": round(float(source_dom_frac), 6),
                "source_gt_count": int(source_gt_count),
                "component_eval_weight": round(float(component_total[component_label]), 3),
                "component_majority_gt": comp_dom,
                "component_majority_gt_frac": round(float(comp_dom_frac), 6),
                "component_gt_count": int(comp_gt_count),
                "source_differs_from_component_majority": bool(source_dom is not None and comp_dom is not None and source_dom != comp_dom),
                "source_rank_score": round(
                    float(source["source_score"])
                    + 0.15 * float(source["source_margin_mean"])
                    + 0.05 * min(np.log1p(int(source["source_conflicts_to_rest"])) / np.log(64.0), 1.0)
                    + 0.05 * float(source["source_quality"]),
                    6,
                ),
                **oracle,
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
                "uses_gt_for_evaluation_only": True,
            }
            key = tuple(int(seq) for seq in row["source_seqs"])
            prev = dedup.get(key)
            keep = prev is None
            if prev is not None and args.dedup_keep == "score":
                keep = float(row["source_rank_score"]) > float(prev["source_rank_score"])
            if prev is not None and args.dedup_keep == "oracle":
                keep = float(row["oracle_delta_pair_f1"]) > float(prev["oracle_delta_pair_f1"])
            if keep:
                dedup[key] = row

    rows = list(dedup.values())
    rows.sort(
        key=lambda row: (
            float(row["source_rank_score"]),
            float(row["source_score"]),
            float(row["source_margin_mean"]),
        ),
        reverse=True,
    )
    oracle_sorted = sorted(rows, key=lambda row: float(row["oracle_delta_pair_f1"]), reverse=True)
    result = {
        "assignment_csv": str(args.assignment_csv),
        "primary_feature_npz": str(args.primary_feature_npz),
        "views": view_meta,
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "base_assignment_components": int(len(raw_to_local)),
        "source_grid_rows": int(grid_rows),
        "dedup_sources": int(len(rows)),
        "oracle_positive_sources": int(sum(float(row["oracle_delta_pair_f1"]) > 0.0 for row in rows)),
        "top_by_source_rank": rows[:100],
        "top_by_oracle_delta": oracle_sorted[:100],
        "topk_source_rank_summary": _topk_summary(rows, [10, 20, 50, 100, 200, 500]),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(
        json.dumps(
            {
                "stage": "done",
                "json": str(out),
                "dedup_sources": len(rows),
                "oracle_positive_sources": result["oracle_positive_sources"],
                "best_source_rank": rows[0] if rows else None,
                "best_oracle": oracle_sorted[0] if oracle_sorted else None,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
