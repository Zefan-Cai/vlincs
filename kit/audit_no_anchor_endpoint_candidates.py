#!/usr/bin/env python
"""Audit endpoint relink candidates against evaluation labels.

This is an evaluation-only diagnostic.  It explains whether an endpoint pair is
locally correct and whether the whole target component is safe to merge into.
No output from this script should be used as a no-anchor training signal.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _write_csv
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _write_csv
    from no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
    )


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _load_endpoint_candidates(path: str, ranks: set[int], limit: int) -> list[dict[str, object]]:
    data = json.loads(Path(path).read_text())
    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for rank, row in enumerate(data.get("top", []), start=1):
        if ranks and rank not in ranks:
            continue
        for cand in row.get("accepted_preview", []) or []:
            key = (
                cand.get("video"),
                int(cand.get("source_seq")),
                int(cand.get("target_seq")),
                int(cand.get("source_label")),
                int(cand.get("target_label")),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append({**cand, "source_result_rank": int(rank)})
            if limit > 0 and len(candidates) >= limit:
                return candidates
    return candidates


def _component_gt_summary(component_indices, records, gt_by_seq, weight_by_seq) -> dict[str, object]:
    counts: Counter[int] = Counter()
    weights: Counter[int] = Counter()
    for idx in component_indices:
        seq = int(records[int(idx)].seq)
        if seq not in gt_by_seq:
            continue
        gid = int(gt_by_seq[seq])
        counts[gid] += 1
        weights[gid] += float(weight_by_seq.get(seq, 1.0))
    dominant_gt = None
    dominant_count = 0
    if counts:
        dominant_gt, dominant_count = counts.most_common(1)[0]
    total = int(sum(counts.values()))
    total_weight = float(sum(weights.values()))
    dominant_weight = float(weights.get(int(dominant_gt), 0.0)) if dominant_gt is not None else 0.0
    return {
        "eval_labeled_tracklets": int(total),
        "eval_labeled_weight": round(total_weight, 6),
        "dominant_gt": int(dominant_gt) if dominant_gt is not None else None,
        "dominant_count": int(dominant_count),
        "dominant_frac": round(float(dominant_count / total), 6) if total else 0.0,
        "dominant_weight": round(dominant_weight, 6),
        "dominant_weight_frac": round(float(dominant_weight / total_weight), 6) if total_weight else 0.0,
        "top_gt_counts": [{"gt": int(k), "count": int(v)} for k, v in counts.most_common(8)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--endpoint-json", required=True)
    ap.add_argument("--top-ranks", default="1,2,3,4,5")
    ap.add_argument("--candidate-limit", type=int, default=50)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, _emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
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

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    keep_seqs, output_info = _output_keep_seqs(records, args)
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    keep_indices = {idx for idx, record in enumerate(records) if int(record.seq) in keep_seqs}
    labels, raw_to_local = _labels_from_assignment(records, pred_input)
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}

    ranks = set(_parse_ints(args.top_ranks))
    candidates = _load_endpoint_candidates(args.endpoint_json, ranks, int(args.candidate_limit))
    rows: list[dict[str, object]] = []
    for cand in candidates:
        source_seq = int(cand["source_seq"])
        target_seq = int(cand["target_seq"])
        if source_seq not in seq_to_idx or target_seq not in seq_to_idx:
            continue
        source_idx = int(seq_to_idx[source_seq])
        target_idx = int(seq_to_idx[target_seq])
        source_label = int(labels[source_idx])
        target_label = int(labels[target_idx])
        source_indices = [int(idx) for idx in keep_indices if int(labels[int(idx)]) == source_label]
        target_indices = [int(idx) for idx in keep_indices if int(labels[int(idx)]) == target_label]
        source_gt = gt_by_seq.get(source_seq)
        target_gt = gt_by_seq.get(target_seq)
        source_summary = _component_gt_summary(source_indices, records, gt_by_seq, weight_by_seq)
        target_summary = _component_gt_summary(target_indices, records, gt_by_seq, weight_by_seq)
        source_gt_in_target_count = 0
        target_gt_in_target_count = 0
        if source_gt is not None:
            source_gt_in_target_count = sum(1 for idx in target_indices if gt_by_seq.get(int(records[int(idx)].seq)) == int(source_gt))
        if target_gt is not None:
            target_gt_in_target_count = sum(1 for idx in target_indices if gt_by_seq.get(int(records[int(idx)].seq)) == int(target_gt))
        target_labeled = int(target_summary["eval_labeled_tracklets"])
        rows.append(
            {
                **cand,
                "actual_source_label": int(source_label),
                "actual_target_label": int(target_label),
                "source_gt": int(source_gt) if source_gt is not None else None,
                "target_seq_gt": int(target_gt) if target_gt is not None else None,
                "source_eq_target_seq_gt": bool(source_gt is not None and target_gt is not None and int(source_gt) == int(target_gt)),
                "source_component_size_eval": int(len(source_indices)),
                "target_component_size_eval": int(len(target_indices)),
                "target_component_dominant_gt": target_summary["dominant_gt"],
                "target_component_dominant_count": target_summary["dominant_count"],
                "target_component_dominant_frac": target_summary["dominant_frac"],
                "source_gt_count_in_target_component": int(source_gt_in_target_count),
                "source_gt_frac_in_target_component": round(float(source_gt_in_target_count / target_labeled), 6) if target_labeled else 0.0,
                "target_seq_gt_count_in_target_component": int(target_gt_in_target_count),
                "target_seq_gt_frac_in_target_component": round(float(target_gt_in_target_count / target_labeled), 6) if target_labeled else 0.0,
                "target_component_top_gt_counts": target_summary["top_gt_counts"],
                "source_component_top_gt_counts": source_summary["top_gt_counts"],
                "failure_mode": (
                    "local_true_but_target_component_impure"
                    if source_gt is not None
                    and target_gt is not None
                    and int(source_gt) == int(target_gt)
                    and target_summary["dominant_gt"] != int(source_gt)
                    else "local_false_pair"
                    if source_gt is not None and target_gt is not None and int(source_gt) != int(target_gt)
                    else "unlabeled_endpoint"
                ),
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
                "uses_gt_for_evaluation_only": True,
            }
        )

    summary = {
        "endpoint_json": str(args.endpoint_json),
        "assignment_csv": str(args.assignment_csv),
        "candidate_count": int(len(candidates)),
        "audited_count": int(len(rows)),
        "failure_mode_counts": dict(Counter(str(row["failure_mode"]) for row in rows)),
        "output_admission": output_info,
        "base_assignment_components": int(len(raw_to_local)),
        "eval_stats": eval_stats,
        "rows": rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "failure_mode_counts": summary["failure_mode_counts"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
