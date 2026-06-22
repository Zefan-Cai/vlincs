#!/usr/bin/env python
"""Use no-GT visual decisions as seeds for local subcluster merge.

The visual verifier can identify plausible same-person component edges, but
whole-component merges are often blocked by impure large components.  This
postprocessor uses the verifier's sampled montage tracklets as seeds: inside
each current component, it extracts only members visually close to the seed
tracklets, then assigns the extracted islands from both sides to a new ID.

No anchors or identity GT are used to select seeds or subclusters.  GT is loaded
only after prediction for pair/full metrics.
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


def _load_decision_rows(decisions_json: str, montage_json: str) -> list[dict[str, object]]:
    decisions_doc = json.loads(Path(decisions_json).read_text())
    if bool(decisions_doc.get("uses_gt_for_decision", False)):
        raise ValueError("refusing visual decisions marked as GT-derived")
    montage_doc = json.loads(Path(montage_json).read_text())
    by_edge = {int(row["edge_id"]): row for row in montage_doc.get("rows", [])}
    rows = []
    for row in decisions_doc.get("rows", []):
        edge_id = int(row["edge_id"])
        merged = dict(by_edge.get(edge_id, {}))
        merged.update(row)
        if "source_rep" not in merged or "target_rep" not in merged:
            raise ValueError(f"edge {edge_id} missing source_rep/target_rep")
        rows.append(merged)
    if not rows:
        raise ValueError(f"no visual decision rows in {decisions_json}")
    return rows


def _members_by_label(labels: np.ndarray, allowed: set[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        if int(idx) in allowed:
            out[int(label)].append(int(idx))
    return out


def _seed_indices(row: dict[str, object], seq_to_idx: dict[int, int], label_by_idx: dict[int, int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for seq in list(row.get("left_seqs", [])) + list(row.get("right_seqs", [])):
        idx = seq_to_idx.get(int(seq))
        if idx is not None:
            out[int(label_by_idx[int(idx)])].append(int(idx))
    for key in ("source_rep", "target_rep"):
        idx = int(row[key])
        out[int(label_by_idx[idx])].append(idx)
    deduped = {}
    for label, values in out.items():
        seen = set()
        clean = []
        for idx in values:
            if int(idx) in seen:
                continue
            seen.add(int(idx))
            clean.append(int(idx))
        deduped[int(label)] = clean
    return deduped


def _select_near_seed(
    emb: np.ndarray,
    members: list[int],
    seeds: list[int],
    *,
    sim_threshold: float,
    max_fraction: float,
    max_selected: int,
    include_whole_below: int,
    moved: set[int],
) -> tuple[list[int], dict[str, object]]:
    available = [int(idx) for idx in members if int(idx) not in moved]
    seed_set = {int(idx) for idx in seeds if int(idx) in available}
    if not available or not seed_set:
        return [], {"available": int(len(available)), "seed_count": int(len(seed_set)), "selected": 0}
    if int(include_whole_below) > 0 and len(available) <= int(include_whole_below):
        return available, {
            "available": int(len(available)),
            "seed_count": int(len(seed_set)),
            "selected": int(len(available)),
            "whole_small_component": True,
            "min_selected_sim": 1.0,
        }
    seed_arr = np.asarray(sorted(seed_set), dtype=np.int64)
    member_arr = np.asarray(available, dtype=np.int64)
    sims = emb[member_arr] @ emb[seed_arr].T
    score = sims.max(axis=1)
    order = np.argsort(-score)
    selected: list[int] = []
    max_by_frac = int(np.ceil(float(max_fraction) * len(available))) if float(max_fraction) > 0 else len(available)
    limit = min(len(available), max(max_by_frac, len(seed_set)))
    if int(max_selected) > 0:
        limit = min(limit, int(max_selected))
    for pos in order.tolist():
        idx = int(member_arr[pos])
        if idx in seed_set or float(score[pos]) >= float(sim_threshold):
            selected.append(idx)
        if len(selected) >= limit:
            break
    for idx in sorted(seed_set):
        if idx not in selected:
            selected.append(int(idx))
    selected = selected[:limit] if limit > 0 else selected
    selected_scores = [float(score[np.where(member_arr == idx)[0][0]]) for idx in selected if idx in set(member_arr.tolist())]
    return selected, {
        "available": int(len(available)),
        "seed_count": int(len(seed_set)),
        "selected": int(len(selected)),
        "whole_small_component": False,
        "min_selected_sim": round(float(min(selected_scores)) if selected_scores else 0.0, 6),
        "mean_selected_sim": round(float(np.mean(selected_scores)) if selected_scores else 0.0, 6),
    }


def _apply_seed_subclusters(records, base_labels, keep_indices, emb, decision_rows, args, *, confidence_threshold: float, sim_threshold: float, max_fraction: float, max_selected: int, min_group_size: int, include_whole_below: int, max_groups: int):
    labels = base_labels.copy()
    members_by_label = _members_by_label(base_labels, keep_indices)
    label_by_idx = {int(idx): int(base_labels[int(idx)]) for idx in range(len(base_labels))}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    next_label = int(labels.max()) + 1
    moved: set[int] = set()
    accepted_groups = 0
    accepted_tracklets = 0
    rejected_not_same = rejected_confidence = rejected_same_label = 0
    rejected_group_size = rejected_no_seed = rejected_max_groups = 0
    groups_out = []
    accepted_edge_ids = []

    rows = sorted(decision_rows, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    for row in rows:
        if not bool(row.get("same_person", False)):
            rejected_not_same += 1
            continue
        if float(row.get("confidence", 0.0)) < float(confidence_threshold):
            rejected_confidence += 1
            continue
        if int(max_groups) > 0 and accepted_groups >= int(max_groups):
            rejected_max_groups += 1
            continue
        a = int(row["source_rep"])
        b = int(row["target_rep"])
        la = int(base_labels[a])
        lb = int(base_labels[b])
        if la == lb:
            rejected_same_label += 1
            continue
        seed_by_label = _seed_indices(row, seq_to_idx, label_by_idx)
        selected_all: list[int] = []
        side_info = {}
        for label in (la, lb):
            seeds = seed_by_label.get(int(label), [])
            if not seeds:
                rejected_no_seed += 1
                continue
            selected, info = _select_near_seed(
                emb,
                members_by_label.get(int(label), []),
                seeds,
                sim_threshold=float(sim_threshold),
                max_fraction=float(max_fraction),
                max_selected=int(max_selected),
                include_whole_below=int(include_whole_below),
                moved=moved,
            )
            side_info[str(label)] = info
            selected_all.extend(selected)
        selected_all = sorted({int(idx) for idx in selected_all if int(idx) not in moved})
        if len(selected_all) < int(min_group_size) or len({int(base_labels[idx]) for idx in selected_all}) < 2:
            rejected_group_size += 1
            continue
        label = int(next_label)
        next_label += 1
        for idx in selected_all:
            labels[int(idx)] = label
            moved.add(int(idx))
        accepted_groups += 1
        accepted_tracklets += len(selected_all)
        accepted_edge_ids.append(int(row.get("edge_id", -1)))
        groups_out.append(
            {
                "edge_id": int(row.get("edge_id", -1)),
                "confidence": float(row.get("confidence", 0.0)),
                "source_label": int(la),
                "target_label": int(lb),
                "label": label,
                "seqs": [int(records[idx].seq) for idx in selected_all],
                "size": int(len(selected_all)),
                "side_info": side_info,
            }
        )
    counts = Counter(labels.tolist())
    return labels, {
        "confidence_threshold": float(confidence_threshold),
        "seed_sim_threshold": float(sim_threshold),
        "seed_max_fraction": float(max_fraction),
        "seed_max_selected_per_side": int(max_selected),
        "seed_min_group_size": int(min_group_size),
        "seed_include_whole_below": int(include_whole_below),
        "seed_max_groups": int(max_groups),
        "accepted_groups": int(accepted_groups),
        "accepted_tracklets": int(accepted_tracklets),
        "accepted_edge_ids": accepted_edge_ids,
        "sample_groups": groups_out,
        "rejected_not_same": int(rejected_not_same),
        "rejected_confidence": int(rejected_confidence),
        "rejected_same_label": int(rejected_same_label),
        "rejected_no_seed": int(rejected_no_seed),
        "rejected_group_size": int(rejected_group_size),
        "rejected_max_groups": int(rejected_max_groups),
        "components": int(len(counts)),
        "largest_component": int(max(counts.values(), default=0)),
        "uses_ground_truth": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--decisions-json", required=True)
    ap.add_argument("--montage-json", required=True)
    ap.add_argument("--confidence-thresholds", default="0.80,0.85,0.90,0.95")
    ap.add_argument("--seed-sim-thresholds", default="0.84,0.88,0.92,0.95")
    ap.add_argument("--seed-max-fractions", default="0.05,0.10,0.20")
    ap.add_argument("--seed-max-selected-per-side", default="4,8,16,32")
    ap.add_argument("--seed-min-group-sizes", default="2,3,4")
    ap.add_argument("--seed-include-whole-below", default="2,4,8")
    ap.add_argument("--seed-max-groups", default="4,8,16,0")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--skip-pair-grid-metrics", action="store_true")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=90_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    decision_rows = _load_decision_rows(args.decisions_json, args.montage_json)
    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    emb = _l2n(_load_feature_npz(args.feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0).astype(np.float32))
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
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    rows = []
    labels_by_rank: dict[int, np.ndarray] = {}
    cache: dict[tuple[float, float, float, int, int, int, int], tuple[np.ndarray, dict[str, object]]] = {}
    for conf in _parse_floats(args.confidence_thresholds):
        for sim in _parse_floats(args.seed_sim_thresholds):
            for frac in _parse_floats(args.seed_max_fractions):
                for max_selected in _parse_ints(args.seed_max_selected_per_side):
                    for min_group in _parse_ints(args.seed_min_group_sizes):
                        for include_whole in _parse_ints(args.seed_include_whole_below):
                            for max_groups in _parse_ints(args.seed_max_groups):
                                key = (float(conf), float(sim), float(frac), int(max_selected), int(min_group), int(include_whole), int(max_groups))
                                labels, info = _apply_seed_subclusters(
                                    records,
                                    base_labels,
                                    keep_indices,
                                    emb,
                                    decision_rows,
                                    args,
                                    confidence_threshold=float(conf),
                                    sim_threshold=float(sim),
                                    max_fraction=float(frac),
                                    max_selected=int(max_selected),
                                    min_group_size=int(min_group),
                                    include_whole_below=int(include_whole),
                                    max_groups=int(max_groups),
                                )
                                cache[key] = (labels, info)
                                if bool(args.skip_pair_grid_metrics):
                                    pair = {
                                        "eval_tracklets": int(base_pair.get("eval_tracklets", 0)),
                                        "gt_pair_mass": float(base_pair.get("gt_pair_mass", 0.0)),
                                        "pred_pair_mass": None,
                                        "tracklet_pair_f1": -1.0,
                                        "tracklet_pair_precision": -1.0,
                                        "tracklet_pair_recall": -1.0,
                                        "true_pair_mass": None,
                                        "pair_metrics_skipped": True,
                                    }
                                else:
                                    pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                    pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                rows.append(
                                    {
                                        "mode": "visual_seed_subcluster_merge",
                                        **info,
                                        **pair,
                                        "decision_rows": int(len(decision_rows)),
                                        "positive_decisions": int(sum(1 for row in decision_rows if bool(row.get("same_person", False)))),
                                        "uses_anchors": False,
                                        "uses_gt_for_training_or_anchors": False,
                                        "uses_gt_for_evaluation_only": True,
                                    }
                                )
    if bool(args.skip_pair_grid_metrics):
        rows.sort(
            key=lambda row: (
                int(row["accepted_groups"]),
                int(row["accepted_tracklets"]),
                float(row["confidence_threshold"]),
                float(row["seed_sim_threshold"]),
            ),
            reverse=True,
        )
    else:
        rows.sort(key=lambda row: (float(row["tracklet_pair_f1"]), float(row["tracklet_pair_recall"]), float(row["tracklet_pair_precision"])), reverse=True)

    full_rows = []
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        key = (
            float(row["confidence_threshold"]),
            float(row["seed_sim_threshold"]),
            float(row["seed_max_fraction"]),
            int(row["seed_max_selected_per_side"]),
            int(row["seed_min_group_size"]),
            int(row["seed_include_whole_below"]),
            int(row["seed_max_groups"]),
        )
        labels = cache[key][0]
        labels_by_rank[rank] = labels
        pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        full_rows.append(dict(row))

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            key = (
                float(row["confidence_threshold"]),
                float(row["seed_sim_threshold"]),
                float(row["seed_max_fraction"]),
                int(row["seed_max_selected_per_side"]),
                int(row["seed_min_group_size"]),
                int(row["seed_include_whole_below"]),
                int(row["seed_max_groups"]),
            )
            labels = cache[key][0]
        assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs=keep_seqs, offset=int(args.assignment_offset))
        rows[0].update(assignment_info)

    result = {
        "assignment_csv": args.assignment_csv,
        "feature_npz": args.feature_npz,
        "decisions_json": args.decisions_json,
        "montage_json": args.montage_json,
        "base_assignment_components": int(len(raw_to_local)),
        "base_pair_metrics": base_pair,
        "assignment_info": assignment_info,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "top": rows[:100],
        "full_rows": full_rows,
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
