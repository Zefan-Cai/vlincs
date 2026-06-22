#!/usr/bin/env python
"""Reassign visually verified edge samples into existing components.

This is a no-anchor local relink ablation.  It consumes visual verifier
support pairs, then tries conservative one-sided or sampled two-sided moves
into an already existing predicted-ID component.  GT is loaded only after the
candidate assignment is formed, for metrics and ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
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
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
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


def _load_assignment(path: str, pred_col: str, component_col: str) -> tuple[dict[int, int], dict[int, int]]:
    pred_by_seq: dict[int, int] = {}
    component_by_seq: dict[int, int] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for required in ("seq", pred_col, component_col):
            if required not in fields:
                raise ValueError(f"{path} is missing {required}")
        for row in reader:
            seq = int(float(row["seq"]))
            pred_by_seq[seq] = int(float(row[pred_col]))
            component_by_seq[seq] = int(float(row[component_col]))
    return pred_by_seq, component_by_seq


def _labels_from_component_assignment(records, component_by_seq: dict[int, int]) -> np.ndarray:
    labels = np.full(len(records), -1, dtype=np.int64)
    next_label = max(component_by_seq.values(), default=-1) + 1
    for idx, record in enumerate(records):
        seq = int(record.seq)
        if seq in component_by_seq:
            labels[idx] = int(component_by_seq[seq])
        else:
            labels[idx] = int(next_label)
            next_label += 1
    return labels


def _load_rows(decisions_json: str) -> list[dict[str, object]]:
    doc = json.loads(Path(decisions_json).read_text())
    if bool(doc.get("uses_gt_for_decision", False)):
        raise ValueError("refusing decisions marked as GT-derived")
    rows = list(doc.get("rows", []))
    if not rows:
        raise ValueError(f"no rows in {decisions_json}")
    return rows


def _unique_ints(values) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _support_seqs(row: dict[str, object], top_pairs: int) -> tuple[list[int], list[int]]:
    if int(top_pairs) <= 0:
        return _unique_ints(row.get("left_seqs", [])), _unique_ints(row.get("right_seqs", []))
    pairs = list(row.get("support_pair_rows", []))[: int(top_pairs)]
    left = _unique_ints(pair.get("left_seq", pair.get("left_index")) for pair in pairs)
    right = _unique_ints(pair.get("right_seq", pair.get("right_index")) for pair in pairs)
    return left, right


def _moves_for_direction(
    row: dict[str, object],
    *,
    direction: str,
    top_pairs: int,
) -> tuple[list[int], int]:
    left, right = _support_seqs(row, int(top_pairs))
    source_label = int(row["source"])
    target_label = int(row["target"])
    if direction == "left_to_target":
        return left, target_label
    if direction == "right_to_source":
        return right, source_label
    if direction == "both_to_source":
        return _unique_ints([*left, *right]), source_label
    if direction == "both_to_target":
        return _unique_ints([*left, *right]), target_label
    raise ValueError(f"unknown direction {direction!r}")


def _component_indices(labels: np.ndarray, label: int, keep_indices: set[int]) -> list[int]:
    return [int(idx) for idx in sorted(keep_indices) if int(labels[int(idx)]) == int(label)]


def _check_move(
    *,
    labels: np.ndarray,
    seq_to_idx: dict[int, int],
    keep_indices: set[int],
    forbidden: list[set[int]],
    move_seqs: list[int],
    dest_label: int,
    disable_forbidden: bool,
) -> tuple[bool, str, list[int]]:
    missing = [int(seq) for seq in move_seqs if int(seq) not in seq_to_idx]
    if missing:
        return False, "missing_seq", []
    move_indices = [int(seq_to_idx[int(seq)]) for seq in move_seqs]
    move_indices = [idx for idx in move_indices if idx in keep_indices]
    if not move_indices:
        return False, "empty_move", []
    if all(int(labels[idx]) == int(dest_label) for idx in move_indices):
        return False, "noop", move_indices
    dest_indices = _component_indices(labels, int(dest_label), keep_indices)
    if not dest_indices:
        return False, "missing_destination", move_indices
    if bool(disable_forbidden):
        return True, "accepted", move_indices

    move_set = set(move_indices)
    dest_set = set(dest_indices)
    new_group = (dest_set | move_set)
    for idx in sorted(move_set):
        if bool(forbidden[int(idx)] & (new_group - {int(idx)})):
            return False, "forbidden", move_indices
    return True, "accepted", move_indices


def _apply_variant(
    base_labels: np.ndarray,
    *,
    seq_to_idx: dict[int, int],
    keep_indices: set[int],
    forbidden: list[set[int]],
    rows: list[dict[str, object]],
    direction: str,
    top_pairs: int,
    confidence_threshold: float,
    edge_id: int | None,
    disable_forbidden: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    labels = base_labels.copy()
    accepted: list[dict[str, object]] = []
    rejected = Counter()
    candidate_rows = []
    for row in sorted(rows, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        if edge_id is not None and int(row.get("edge_id", -1)) != int(edge_id):
            continue
        candidate_rows.append(row)
    for row in candidate_rows:
        if not bool(row.get("same_person", False)):
            rejected["not_same"] += 1
            continue
        confidence = float(row.get("confidence", 0.0))
        if confidence < float(confidence_threshold):
            rejected["confidence"] += 1
            continue
        move_seqs, dest_label = _moves_for_direction(row, direction=direction, top_pairs=int(top_pairs))
        ok, reason, move_indices = _check_move(
            labels=labels,
            seq_to_idx=seq_to_idx,
            keep_indices=keep_indices,
            forbidden=forbidden,
            move_seqs=move_seqs,
            dest_label=int(dest_label),
            disable_forbidden=bool(disable_forbidden),
        )
        if not ok:
            rejected[reason] += 1
            continue
        moved = []
        idx_to_seq = {
            int(seq_to_idx[int(seq)]): int(seq)
            for seq in move_seqs
            if int(seq) in seq_to_idx and int(seq_to_idx[int(seq)]) in keep_indices
        }
        for idx in move_indices:
            old_label = int(labels[int(idx)])
            if old_label == int(dest_label):
                continue
            labels[int(idx)] = int(dest_label)
            moved.append(
                {
                    "seq": int(idx_to_seq[int(idx)]),
                    "old_label": int(old_label),
                    "new_label": int(dest_label),
                }
            )
        if moved:
            accepted.append(
                {
                    "edge_id": int(row.get("edge_id", -1)),
                    "source": int(row["source"]),
                    "target": int(row["target"]),
                    "confidence": float(confidence),
                    "direction": str(direction),
                    "top_pairs": int(top_pairs),
                    "destination_label": int(dest_label),
                    "moved": moved,
                }
            )
        else:
            rejected["noop"] += 1

    keep_labels = [int(labels[idx]) for idx in keep_indices]
    info = {
        "direction": str(direction),
        "top_pairs": int(top_pairs),
        "confidence_threshold": float(confidence_threshold),
        "edge_id": int(edge_id) if edge_id is not None else -1,
        "policy": "combined" if edge_id is None else "single_edge",
        "accepted_reassignments": int(len(accepted)),
        "moved_tracklets": int(sum(len(row["moved"]) for row in accepted)),
        "accepted_preview": accepted[:20],
        "rejected_not_same": int(rejected["not_same"]),
        "rejected_confidence": int(rejected["confidence"]),
        "rejected_forbidden": int(rejected["forbidden"]),
        "rejected_noop": int(rejected["noop"]),
        "rejected_missing_destination": int(rejected["missing_destination"]),
        "rejected_empty_move": int(rejected["empty_move"]),
        "rejected_missing_seq": int(rejected["missing_seq"]),
        "components": int(len(set(keep_labels))),
        "largest_component": int(max(Counter(keep_labels).values(), default=0)),
        "uses_ground_truth": False,
    }
    return labels, info


def _assignment_signature(row: dict[str, object]) -> str:
    moved = []
    for item in row.get("accepted_preview", []):
        for move in item.get("moved", []):
            moved.append((int(move["seq"]), int(move["old_label"]), int(move["new_label"])))
    if not moved:
        return "noop"
    return json.dumps(sorted(moved), separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--decisions-json", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--component-col", default="component_label")
    ap.add_argument("--confidence-thresholds", default="0.80,0.90,0.95")
    ap.add_argument("--top-pairs", default="1,2,3,0")
    ap.add_argument("--directions", default="left_to_target,right_to_source,both_to_source,both_to_target")
    ap.add_argument("--policies", default="single_edge,combined")
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

    decision_rows = _load_rows(args.decisions_json)
    pred_input, component_input = _load_assignment(args.assignment_csv, args.pred_col, args.component_col)
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
    base_labels = _labels_from_component_assignment(records, component_input)
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(set(base_labels.tolist())), **base_pair}, sort_keys=True), flush=True)

    forbidden = [set() for _ in records] if bool(args.disable_forbidden) else _build_overlap_forbidden(records)
    positive_edge_ids = [int(row["edge_id"]) for row in decision_rows if bool(row.get("same_person", False))]
    directions = [text.strip() for text in str(args.directions).split(",") if text.strip()]
    policies = {text.strip() for text in str(args.policies).split(",") if text.strip()}
    rows: list[dict[str, object]] = []
    labels_by_key: dict[tuple[object, ...], np.ndarray] = {}
    for confidence_threshold in _parse_floats(args.confidence_thresholds):
        for top_pairs in _parse_ints(args.top_pairs):
            for direction in directions:
                edge_ids: list[int | None] = []
                if "single_edge" in policies:
                    edge_ids.extend(positive_edge_ids)
                if "combined" in policies:
                    edge_ids.append(None)
                for edge_id in edge_ids:
                    key = (float(confidence_threshold), int(top_pairs), str(direction), int(edge_id) if edge_id is not None else -1)
                    labels, info = _apply_variant(
                        base_labels,
                        seq_to_idx=seq_to_idx,
                        keep_indices=keep_indices,
                        forbidden=forbidden,
                        rows=decision_rows,
                        direction=str(direction),
                        top_pairs=int(top_pairs),
                        confidence_threshold=float(confidence_threshold),
                        edge_id=edge_id,
                        disable_forbidden=bool(args.disable_forbidden),
                    )
                    pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                    pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                    row_out = {
                        "mode": "visual_edge_sample_reassignment",
                        **info,
                        **pair,
                        "decision_rows": int(len(decision_rows)),
                        "positive_decisions": int(len(positive_edge_ids)),
                        "disable_forbidden": bool(args.disable_forbidden),
                        "uses_anchors": False,
                        "uses_gt_for_training_or_anchors": False,
                        "uses_gt_for_evaluation_only": True,
                    }
                    rows.append(row_out)
                    labels_by_key[key] = labels
    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
            float(row["tracklet_pair_precision"]),
        ),
        reverse=True,
    )
    deduped_rows: list[dict[str, object]] = []
    seen_signatures: set[str] = set()
    duplicate_assignments = 0
    for row in rows:
        signature = _assignment_signature(row)
        if signature in seen_signatures:
            duplicate_assignments += 1
            continue
        seen_signatures.add(signature)
        row["assignment_signature"] = signature
        deduped_rows.append(row)
    rows = deduped_rows

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        key = (
            float(row["confidence_threshold"]),
            int(row["top_pairs"]),
            str(row["direction"]),
            int(row["edge_id"]),
        )
        labels = labels_by_key[key]
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
        best = rows[0]
        key = (
            float(best["confidence_threshold"]),
            int(best["top_pairs"]),
            str(best["direction"]),
            int(best["edge_id"]),
        )
        assignment_info = _write_assignments(
            args.assignments_out,
            records,
            labels_by_key[key],
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
        "assignment_info": assignment_info,
        "candidate_assignments_after_dedup": int(len(rows)),
        "duplicate_assignments_removed": int(duplicate_assignments),
        "top": rows[: max(120, int(args.full_top_n))],
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
