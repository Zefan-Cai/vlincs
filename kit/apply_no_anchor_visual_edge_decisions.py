#!/usr/bin/env python
"""Apply visual verifier component-edge decisions to a no-anchor assignment.

The verifier decisions are produced without anchors or GT.  Ground truth is
loaded only after the merged assignment is formed, for diagnostics/ablation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from kit.no_anchor_global_id_model import _UnionFind
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
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from no_anchor_global_id_model import _UnionFind
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


def _merge_decisions(records, base_labels, decisions, forbidden, *, confidence_threshold: float, max_component_size: int):
    uf = _UnionFind(len(base_labels))
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(base_labels.tolist()):
        groups[int(label)].append(int(idx))
    for indices in groups.values():
        head = int(indices[0])
        for idx in indices[1:]:
            uf.merge(head, int(idx))

    accepted = 0
    rejected_not_same = 0
    rejected_confidence = 0
    rejected_forbidden = 0
    rejected_size = 0
    rejected_stale = 0
    accepted_edge_ids: list[int] = []
    for row in sorted(decisions, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        if not bool(row.get("same_person", False)):
            rejected_not_same += 1
            continue
        if float(row.get("confidence", 0.0)) < float(confidence_threshold):
            rejected_confidence += 1
            continue
        a = int(row["source_rep"])
        b = int(row["target_rep"])
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            rejected_stale += 1
            continue
        if len(uf.members[ra]) + len(uf.members[rb]) > int(max_component_size):
            rejected_size += 1
            continue
        if not uf.can_merge(a, b, forbidden, int(max_component_size)):
            rejected_forbidden += 1
            continue
        uf.merge(a, b)
        accepted += 1
        accepted_edge_ids.append(int(row.get("edge_id", -1)))

    labels = uf.labels()
    return labels, {
        "confidence_threshold": float(confidence_threshold),
        "max_component_size": int(max_component_size),
        "accepted_edges": int(accepted),
        "accepted_edge_ids": accepted_edge_ids,
        "rejected_not_same": int(rejected_not_same),
        "rejected_confidence": int(rejected_confidence),
        "rejected_forbidden": int(rejected_forbidden),
        "rejected_size": int(rejected_size),
        "rejected_stale": int(rejected_stale),
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
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--confidence-thresholds", default="0.70,0.80,0.85,0.90,0.95")
    ap.add_argument("--max-component-sizes", default="300,500,800")
    ap.add_argument("--disable-forbidden", action="store_true")
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

    decisions_doc = json.loads(Path(args.decisions_json).read_text())
    decisions = list(decisions_doc.get("rows", []))
    if not decisions:
        raise ValueError(f"no decision rows in {args.decisions_json}")
    if bool(decisions_doc.get("uses_gt_for_decision", False)):
        raise ValueError("refusing verifier decisions marked as GT-derived")

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

    forbidden = [set() for _ in records] if bool(args.disable_forbidden) else _build_overlap_forbidden(records)
    rows: list[dict[str, object]] = []
    labels_by_rank: dict[int, object] = {}
    for max_component_size in _parse_ints(args.max_component_sizes):
        for confidence_threshold in _parse_floats(args.confidence_thresholds):
            labels, info = _merge_decisions(
                records,
                base_labels,
                decisions,
                forbidden,
                confidence_threshold=float(confidence_threshold),
                max_component_size=int(max_component_size),
            )
            pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
            pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
            rows.append(
                {
                    "mode": "visual_edge_decision_merge",
                    **info,
                    **pair,
                    "decision_rows": int(len(decisions)),
                    "positive_decisions": int(sum(1 for row in decisions if bool(row.get("same_person", False)))),
                    "uses_anchors": False,
                    "uses_gt_for_training_or_anchors": False,
                    "uses_gt_for_evaluation_only": True,
                }
            )
    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
            float(row["tracklet_pair_precision"]),
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        labels, _info = _merge_decisions(
            records,
            base_labels,
            decisions,
            forbidden,
            confidence_threshold=float(row["confidence_threshold"]),
            max_component_size=int(row["max_component_size"]),
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
            labels, _info = _merge_decisions(
                records,
                base_labels,
                decisions,
                forbidden,
                confidence_threshold=float(row["confidence_threshold"]),
                max_component_size=int(row["max_component_size"]),
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
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "disable_forbidden": bool(args.disable_forbidden),
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
