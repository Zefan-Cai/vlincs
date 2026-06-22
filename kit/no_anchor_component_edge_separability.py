#!/usr/bin/env python
"""Eval-only separability analysis for component merge edges.

This diagnostic labels candidate component edges with GT only after candidate
generation, then measures whether no-anchor visual/rank features can separate
false-split repair edges from false merge edges.  It never trains or selects a
production policy from GT labels.
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
    from kit.no_anchor_component_merge_sweep import _candidate_edges
    from kit.no_anchor_component_verifier_sweep import _edge_feature_table, _load_npz_aligned, _parse_view
    from kit.no_anchor_resolve_sweep import (
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
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _candidate_edges
    from no_anchor_component_verifier_sweep import _edge_feature_table, _load_npz_aligned, _parse_view
    from no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _component_members(labels: np.ndarray, keep_indices: set[int]) -> tuple[list[int], list[list[int]]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        if idx in keep_indices:
            by_label[int(label)].append(int(idx))
    reps = []
    members = []
    for _label, indices in sorted(by_label.items(), key=lambda item: min(item[1])):
        reps.append(int(indices[0]))
        members.append([int(idx) for idx in indices])
    return reps, members


def _component_gt(records, members, gt_by_seq, weight_by_seq):
    tables = []
    dominant = []
    purity = []
    total_weight = []
    for indices in members:
        by_gt: dict[int, float] = defaultdict(float)
        for idx in indices:
            seq = int(records[idx].seq)
            if seq not in gt_by_seq:
                continue
            by_gt[int(gt_by_seq[seq])] += float(weight_by_seq.get(seq, 1.0))
        total = float(sum(by_gt.values()))
        total_weight.append(total)
        tables.append(dict(by_gt))
        if by_gt and total > 0:
            gid, weight = max(by_gt.items(), key=lambda item: item[1])
            dominant.append(int(gid))
            purity.append(float(weight / max(total, 1.0e-9)))
        else:
            dominant.append(None)
            purity.append(0.0)
    return tables, dominant, np.asarray(purity, dtype=np.float32), np.asarray(total_weight, dtype=np.float64)


def _rank_false_split_gts(records, labels, keep_indices, gt_by_seq, weight_by_seq, top_n: int) -> list[int]:
    by_gt: dict[int, dict[int, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for idx in sorted(keep_indices):
        seq = int(records[idx].seq)
        if seq not in gt_by_seq:
            continue
        by_gt[int(gt_by_seq[seq])][int(labels[idx])].append((seq, float(weight_by_seq.get(seq, 1.0))))
    rows = []
    for gid, by_comp in by_gt.items():
        comp_weights = [sum(w for _seq, w in values) for values in by_comp.values()]
        total = sum(comp_weights)
        total2 = sum(w * w for w in comp_weights)
        gt_pair_mass = max((total * total - total2) / 2.0, 0.0)
        true_pair_mass = 0.0
        for values in by_comp.values():
            ws = [w for _seq, w in values]
            subtotal = sum(ws)
            subtotal2 = sum(w * w for w in ws)
            true_pair_mass += max((subtotal * subtotal - subtotal2) / 2.0, 0.0)
        rows.append((float(gt_pair_mass - true_pair_mass), int(gid)))
    rows.sort(reverse=True)
    return [gid for _mass, gid in rows[: int(top_n)]]


def _annotate_edges(rows, comp_tables, comp_dominant, comp_purity, comp_weight, top_gt_set: set[int], min_purity: float):
    for row in rows:
        a = int(row["source"])
        b = int(row["target"])
        common = set(comp_tables[a]) & set(comp_tables[b])
        same_mass = sum(float(comp_tables[a][gid] * comp_tables[b][gid]) for gid in common)
        all_mass = float(comp_weight[a] * comp_weight[b])
        dom_same = comp_dominant[a] is not None and comp_dominant[a] == comp_dominant[b]
        pure = float(comp_purity[a]) >= float(min_purity) and float(comp_purity[b]) >= float(min_purity)
        label = int(bool(dom_same and pure))
        row["gt_edge_label"] = label
        row["gt_edge_same_mass"] = float(same_mass)
        row["gt_edge_all_mass"] = float(all_mass)
        row["gt_edge_same_frac"] = float(same_mass / max(all_mass, 1.0e-9))
        row["gt_dominant_same"] = bool(dom_same)
        row["gt_both_pure"] = bool(pure)
        row["gt_dominant_id"] = int(comp_dominant[a]) if dom_same and comp_dominant[a] is not None else None
        row["gt_top_false_split_target"] = bool(dom_same and comp_dominant[a] in top_gt_set)
    return rows


def _rule_rows(rows, *, thresholds: list[float], min_votes: list[int], min_sizes: list[int], top_gt_only: bool):
    out = []
    labeled = [row for row in rows if bool(row["gt_both_pure"])]
    for score_thr in thresholds:
        for votes in min_votes:
            for min_size in min_sizes:
                cand = []
                for row in rows:
                    if top_gt_only and not bool(row["gt_top_false_split_target"]):
                        continue
                    if int(row["source_size"]) < int(min_size) or int(row["target_size"]) < int(min_size):
                        continue
                    if int(row["is_forbidden"]) > 0:
                        continue
                    if int(row["votes_top5"]) < int(votes):
                        continue
                    if float(row["score"]) < float(score_thr):
                        continue
                    cand.append(row)
                if not cand:
                    continue
                tp = [row for row in cand if int(row["gt_edge_label"]) == 1]
                tp_top = [row for row in tp if bool(row["gt_top_false_split_target"])]
                same_mass = sum(float(row["gt_edge_same_mass"]) for row in tp)
                fp_proxy = sum(max(float(row["gt_edge_all_mass"]) - float(row["gt_edge_same_mass"]), 0.0) for row in cand if int(row["gt_edge_label"]) == 0)
                out.append(
                    {
                        "score_threshold": float(score_thr),
                        "min_votes_top5": int(votes),
                        "min_component_size": int(min_size),
                        "top_gt_only_oracle_filter": bool(top_gt_only),
                        "candidate_edges": int(len(cand)),
                        "true_edges": int(len(tp)),
                        "true_top_false_split_edges": int(len(tp_top)),
                        "edge_precision": round(float(len(tp) / max(len(cand), 1)), 6),
                        "same_mass": round(float(same_mass), 3),
                        "fp_proxy_mass": round(float(fp_proxy), 3),
                    }
                )
    out.sort(
        key=lambda row: (
            float(row["edge_precision"]),
            float(row["same_mass"]),
            int(row["true_top_false_split_edges"]),
        ),
        reverse=True,
    )
    return out


def _feature_quantiles(rows, feature_names: list[str], *, label_key: str):
    out = {}
    for name in feature_names:
        vals_pos = [float(row[name]) for row in rows if bool(row.get(label_key)) and isinstance(row.get(name), (int, float))]
        vals_neg = [float(row[name]) for row in rows if not bool(row.get(label_key)) and isinstance(row.get(name), (int, float))]
        if not vals_pos or not vals_neg:
            continue
        out[name] = {
            "positive_n": int(len(vals_pos)),
            "negative_n": int(len(vals_neg)),
            "positive_q50": round(float(np.quantile(vals_pos, 0.5)), 6),
            "positive_q90": round(float(np.quantile(vals_pos, 0.9)), 6),
            "negative_q50": round(float(np.quantile(vals_neg, 0.5)), 6),
            "negative_q90": round(float(np.quantile(vals_neg, 0.9)), 6),
        }
    return out


def _write_edge_csv(path: str, rows: list[dict[str, object]], limit: int) -> None:
    rows = rows[: max(int(limit), 0)]
    if not rows:
        return
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list, tuple))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--view", action="append", default=[])
    ap.add_argument("--candidate-top-k", type=int, default=100)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weight", type=float, default=0.0)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--min-purity", type=float, default=0.75)
    ap.add_argument("--top-false-split-gt", type=int, default=40)
    ap.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90,0.94")
    ap.add_argument("--min-votes-top5", default="0,1,2,3,4")
    ap.add_argument("--min-component-sizes", default="1,2,4,8,16,32")
    ap.add_argument("--edge-csv", default="")
    ap.add_argument("--edge-csv-limit", type=int, default=500)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    emb = _load_feature_npz(
        args.feature_npz,
        records,
        db_emb,
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
    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}

    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    base_pred = {int(seq): int(pred) for seq, pred in pred_input.items() if int(seq) in keep_seqs}
    base_pair = _pair_metrics([record.seq for record in records], base_pred, gt_by_seq, weight_by_seq)
    reps, members = _component_members(base_labels, keep_indices)
    edges, edge_info = _candidate_edges(
        records,
        emb,
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
    view_embeddings = {"primary": emb.astype(np.float32)}
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        if path.lower() == "db":
            view_embeddings[name] = _l2n(db_emb.astype(np.float32)) * float(weight)
        else:
            view_embeddings[name] = _load_npz_aligned(path, records, weight=float(weight))
    edge_rows, _X, feature_names = _edge_feature_table(records, members, edges, view_embeddings)
    comp_tables, comp_dominant, comp_purity, comp_weight = _component_gt(records, members, gt_by_seq, weight_by_seq)
    top_gts = set(_rank_false_split_gts(records, base_labels, keep_indices, gt_by_seq, weight_by_seq, int(args.top_false_split_gt)))
    edge_rows = _annotate_edges(edge_rows, comp_tables, comp_dominant, comp_purity, comp_weight, top_gts, float(args.min_purity))
    edge_rows.sort(key=lambda row: float(row["score"]), reverse=True)

    thresholds = _parse_floats(args.thresholds)
    votes = _parse_ints(args.min_votes_top5)
    sizes = _parse_ints(args.min_component_sizes)
    rules = _rule_rows(edge_rows, thresholds=thresholds, min_votes=votes, min_sizes=sizes, top_gt_only=False)
    oracle_rules = _rule_rows(edge_rows, thresholds=thresholds, min_votes=votes, min_sizes=sizes, top_gt_only=True)
    pos = [row for row in edge_rows if int(row["gt_edge_label"]) == 1]
    pos_top = [row for row in pos if bool(row["gt_top_false_split_target"])]
    result = {
        "assignment_csv": args.assignment_csv,
        "base_assignment_components": int(len(raw_to_local)),
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "edge_info": edge_info,
        "views": sorted(view_embeddings),
        "feature_names": feature_names,
        "top_false_split_gt_ids": sorted(top_gts),
        "labeled_edge_counts": {
            "candidate_edges": int(len(edge_rows)),
            "true_edges": int(len(pos)),
            "true_top_false_split_edges": int(len(pos_top)),
            "forbidden_edges": int(sum(1 for row in edge_rows if int(row["is_forbidden"]) > 0)),
        },
        "feature_quantiles_true_edge": _feature_quantiles(edge_rows, feature_names, label_key="gt_edge_label"),
        "feature_quantiles_top_false_split_edge": _feature_quantiles(edge_rows, feature_names, label_key="gt_top_false_split_target"),
        "top_rules": rules[:50],
        "top_oracle_target_rules": oracle_rules[:50],
        "top_true_edges": sorted(pos, key=lambda row: float(row["score"]), reverse=True)[:50],
        "top_false_edges": sorted([row for row in edge_rows if int(row["gt_edge_label"]) == 0], key=lambda row: float(row["score"]), reverse=True)[:50],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_analysis_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.edge_csv:
        preview = sorted(edge_rows, key=lambda row: (int(row["gt_edge_label"]), float(row["score"])), reverse=True)
        _write_edge_csv(args.edge_csv, preview, int(args.edge_csv_limit))
    print(
        json.dumps(
            {
                "json": str(out),
                "base_pair": base_pair,
                "edge_counts": result["labeled_edge_counts"],
                "best_rule": rules[0] if rules else None,
                "best_oracle_target_rule": oracle_rules[0] if oracle_rules else None,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
