#!/usr/bin/env python
"""Create micro-components from visually verified sampled edge tracklets.

This is a no-anchor surgical splitter: for accepted visual verifier edges, it
does not merge the full source/target components.  Instead it extracts the
sampled tracklets shown in the montage into small new components.  GT is used
only after prediction for diagnostics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _parse_floats, _write_csv
    from kit.no_anchor_louvain_sweep import _write_assignments
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
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _parse_floats, _write_csv
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
        _with_detection_endpoints,
    )


def _load_decision_rows(decisions_json: str, montage_json: str) -> list[dict[str, object]]:
    decisions_doc = json.loads(Path(decisions_json).read_text())
    if bool(decisions_doc.get("uses_gt_for_decision", False)):
        raise ValueError("refusing verifier decisions marked as GT-derived")
    montage_doc = json.loads(Path(montage_json).read_text())
    by_edge = {int(row["edge_id"]): row for row in montage_doc.get("rows", [])}
    rows = []
    for row in decisions_doc.get("rows", []):
        edge_id = int(row["edge_id"])
        merged = dict(by_edge.get(edge_id, {}))
        merged.update(row)
        if "left_seqs" not in merged or "right_seqs" not in merged:
            raise ValueError(f"edge {edge_id} missing sampled seqs")
        rows.append(merged)
    if not rows:
        raise ValueError(f"no decision rows in {decisions_json}")
    return rows


def _can_add(idx: int, group_indices: list[int], forbidden: list[set[int]]) -> bool:
    return all(int(other) not in forbidden[int(idx)] for other in group_indices)


def _surgery_labels(
    records,
    base_labels,
    decision_rows,
    forbidden,
    *,
    confidence_threshold: float,
    min_positive_edges_per_group: int,
    min_group_size: int,
    max_groups: int,
):
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    labels = base_labels.copy()
    next_label = int(labels.max()) + 1
    grouped: dict[tuple[int, tuple[int, ...]], list[dict[str, object]]] = defaultdict(list)
    rejected_not_same = 0
    rejected_confidence = 0
    for row in decision_rows:
        if not bool(row.get("same_person", False)):
            rejected_not_same += 1
            continue
        if float(row.get("confidence", 0.0)) < float(confidence_threshold):
            rejected_confidence += 1
            continue
        key = (int(row.get("source", -1)), tuple(sorted(int(seq) for seq in row.get("left_seqs", []))))
        grouped[key].append(row)

    candidates = []
    for key, rows in grouped.items():
        if len(rows) < int(min_positive_edges_per_group):
            continue
        seqs: list[int] = []
        for row in sorted(rows, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
            seqs.extend(int(seq) for seq in row.get("left_seqs", []))
            seqs.extend(int(seq) for seq in row.get("right_seqs", []))
        deduped = []
        seen = set()
        for seq in seqs:
            if seq in seen or seq not in seq_to_idx:
                continue
            seen.add(seq)
            deduped.append(seq)
        candidates.append((max(float(row.get("confidence", 0.0)) for row in rows), key, rows, deduped))
    candidates.sort(key=lambda item: (item[0], len(item[3])), reverse=True)

    accepted_groups = 0
    accepted_tracklets = 0
    rejected_group_size = 0
    rejected_sample_forbidden = 0
    accepted_edge_ids: list[int] = []
    groups_out: list[dict[str, object]] = []
    already_moved: set[int] = set()
    for _score, key, rows, seqs in candidates[: max(int(max_groups), 0)]:
        group_indices: list[int] = []
        kept_seqs: list[int] = []
        for seq in seqs:
            idx = int(seq_to_idx[seq])
            if idx in already_moved:
                continue
            if not _can_add(idx, group_indices, forbidden):
                rejected_sample_forbidden += 1
                continue
            group_indices.append(idx)
            kept_seqs.append(int(seq))
        if len(group_indices) < int(min_group_size):
            rejected_group_size += 1
            continue
        label = int(next_label)
        next_label += 1
        for idx in group_indices:
            labels[int(idx)] = label
            already_moved.add(int(idx))
        accepted_groups += 1
        accepted_tracklets += len(group_indices)
        edge_ids = [int(row["edge_id"]) for row in rows]
        accepted_edge_ids.extend(edge_ids)
        groups_out.append(
            {
                "group_key": [int(key[0]), list(key[1])],
                "edge_ids": edge_ids,
                "seqs": kept_seqs,
                "size": int(len(kept_seqs)),
                "label": label,
            }
        )

    return labels, {
        "confidence_threshold": float(confidence_threshold),
        "min_positive_edges_per_group": int(min_positive_edges_per_group),
        "min_group_size": int(min_group_size),
        "max_groups": int(max_groups),
        "accepted_groups": int(accepted_groups),
        "accepted_tracklets": int(accepted_tracklets),
        "accepted_edge_ids": accepted_edge_ids,
        "sample_groups": groups_out,
        "rejected_not_same": int(rejected_not_same),
        "rejected_confidence": int(rejected_confidence),
        "rejected_group_size": int(rejected_group_size),
        "rejected_sample_forbidden": int(rejected_sample_forbidden),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "uses_ground_truth": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--decisions-json", required=True)
    ap.add_argument("--montage-json", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--confidence-thresholds", default="0.80,0.85,0.90,0.95")
    ap.add_argument("--min-positive-edges-per-group", default="1,2")
    ap.add_argument("--min-group-sizes", default="2,3,4")
    ap.add_argument("--max-groups", default="4,8,16")
    ap.add_argument("--disable-sample-forbidden", action="store_true")
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

    decision_rows = _load_decision_rows(args.decisions_json, args.montage_json)
    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
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

    forbidden = [set() for _ in records] if bool(args.disable_sample_forbidden) else _build_overlap_forbidden(records)
    rows: list[dict[str, object]] = []
    labels_by_rank: dict[int, object] = {}
    for max_groups in [int(x) for x in str(args.max_groups).split(",") if str(x).strip()]:
        for min_group_size in [int(x) for x in str(args.min_group_sizes).split(",") if str(x).strip()]:
            for min_edges in [int(x) for x in str(args.min_positive_edges_per_group).split(",") if str(x).strip()]:
                for confidence_threshold in _parse_floats(args.confidence_thresholds):
                    labels, info = _surgery_labels(
                        records,
                        base_labels,
                        decision_rows,
                        forbidden,
                        confidence_threshold=float(confidence_threshold),
                        min_positive_edges_per_group=int(min_edges),
                        min_group_size=int(min_group_size),
                        max_groups=int(max_groups),
                    )
                    pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                    pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                    rows.append(
                        {
                            "mode": "visual_edge_sample_surgery",
                            **info,
                            **pair,
                            "decision_rows": int(len(decision_rows)),
                            "positive_decisions": int(sum(1 for row in decision_rows if bool(row.get("same_person", False)))),
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
        labels, _info = _surgery_labels(
            records,
            base_labels,
            decision_rows,
            forbidden,
            confidence_threshold=float(row["confidence_threshold"]),
            min_positive_edges_per_group=int(row["min_positive_edges_per_group"]),
            min_group_size=int(row["min_group_size"]),
            max_groups=int(row["max_groups"]),
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
            labels, _info = _surgery_labels(
                records,
                base_labels,
                decision_rows,
                forbidden,
                confidence_threshold=float(row["confidence_threshold"]),
                min_positive_edges_per_group=int(row["min_positive_edges_per_group"]),
                min_group_size=int(row["min_group_size"]),
                max_groups=int(row["max_groups"]),
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
        "decisions_json": str(args.decisions_json),
        "montage_json": str(args.montage_json),
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "disable_sample_forbidden": bool(args.disable_sample_forbidden),
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
