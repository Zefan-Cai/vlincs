#!/usr/bin/env python
"""Split temporal-NMS losers into singleton IDs without dropping detections.

This is the delivery-preserving counterpart to the cannot-link NMS admission
test.  Inside each current predicted ID, high-quality non-overlapping tracklets
keep the original ID; lower-quality tracklets that temporally overlap the kept
core are assigned fresh singleton IDs.  No anchors or GT labels are used to
choose losers.  GT is loaded only after prediction for pair/full metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_cannotlink_split_sweep import _load_assignment, _overlap, _parse_floats, _parse_ints
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_cannotlink_split_sweep import _load_assignment, _overlap, _parse_floats, _parse_ints
    from no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _base_pred(rows: dict[int, dict[str, object]], pred_col: str, keep_seqs: set[int]) -> dict[int, int]:
    return {int(seq): int(row[pred_col]) for seq, row in rows.items() if int(seq) in keep_seqs}


def _quality(row: dict[str, object], score_mode: str) -> float:
    n_dets = float(row.get("n_dets") or 1.0)
    avg_conf = float(row.get("avg_conf") or 0.0)
    pred_conf = float(row.get("prediction_confidence") or 0.0)
    if score_mode == "n_dets":
        return n_dets
    if score_mode == "avg_conf":
        return avg_conf
    if score_mode == "n_dets_conf":
        return n_dets * (0.25 + avg_conf)
    if score_mode == "n_dets_conf_margin":
        margin = float(row.get("component_margin_prob") or 0.0)
        return n_dets * (0.25 + avg_conf) * (0.50 + pred_conf + margin)
    return n_dets * (0.25 + avg_conf) * (0.50 + pred_conf)


def _has_conflict(seq: int, kept: list[int], rows: dict[int, dict[str, object]], *, min_overlap_frames: int, overlap_slack: int) -> bool:
    row = rows[int(seq)]
    stream = (str(row.get("video") or ""), str(row.get("camera") or ""))
    for other in kept:
        other_row = rows[int(other)]
        other_stream = (str(other_row.get("video") or ""), str(other_row.get("camera") or ""))
        if other_stream != stream:
            continue
        if _overlap(row, other_row, int(overlap_slack)) >= int(min_overlap_frames):
            return True
    return False


def _component_conflict_count(seqs: list[int], rows: dict[int, dict[str, object]], *, min_overlap_frames: int, overlap_slack: int) -> int:
    by_stream: dict[tuple[str, str], list[int]] = defaultdict(list)
    for seq in seqs:
        row = rows[int(seq)]
        by_stream[(str(row.get("video") or ""), str(row.get("camera") or ""))].append(int(seq))
    edges = 0
    for group in by_stream.values():
        group = sorted(group, key=lambda seq: (int(rows[seq]["start_frame"]), int(rows[seq]["end_frame"]), int(seq)))
        for pos, a_seq in enumerate(group):
            a = rows[int(a_seq)]
            for b_seq in group[pos + 1 :]:
                b = rows[int(b_seq)]
                if int(b["start_frame"]) > int(a["end_frame"]) + int(overlap_slack):
                    break
                if _overlap(a, b, int(overlap_slack)) >= int(min_overlap_frames):
                    edges += 1
    return int(edges)


_PLAN_CACHE: dict[tuple[int, int, str, int, int], dict[int, dict[str, object]]] = {}


def _component_plans(
    rows: dict[int, dict[str, object]],
    pred: dict[int, int],
    *,
    min_overlap_frames: int,
    overlap_slack: int,
    score_mode: str,
) -> dict[int, dict[str, object]]:
    cache_key = (int(min_overlap_frames), int(overlap_slack), str(score_mode), len(rows), len(pred))
    if cache_key in _PLAN_CACHE:
        return _PLAN_CACHE[cache_key]
    by_gid: dict[int, list[int]] = defaultdict(list)
    for seq, gid in pred.items():
        by_gid[int(gid)].append(int(seq))
    plans: dict[int, dict[str, object]] = {}
    for gid, seqs in sorted(by_gid.items()):
        conflict_edges = _component_conflict_count(
            seqs,
            rows,
            min_overlap_frames=int(min_overlap_frames),
            overlap_slack=int(overlap_slack),
        )
        ordered = sorted(
            seqs,
            key=lambda seq: (
                -_quality(rows[int(seq)], str(score_mode)),
                int(rows[int(seq)]["start_frame"]),
                int(seq),
            ),
        )
        kept: list[int] = []
        losers: list[int] = []
        for seq in ordered:
            if _has_conflict(
                int(seq),
                kept,
                rows,
                min_overlap_frames=int(min_overlap_frames),
                overlap_slack=int(overlap_slack),
            ):
                losers.append(int(seq))
            else:
                kept.append(int(seq))
        plans[int(gid)] = {
            "seqs": [int(seq) for seq in seqs],
            "conflict_edges": int(conflict_edges),
            "kept": kept,
            "losers": losers,
        }
    _PLAN_CACHE[cache_key] = plans
    return plans


def _nms_singleton_pred(
    rows: dict[int, dict[str, object]],
    pred_col: str,
    keep_seqs: set[int],
    *,
    min_component_size: int,
    min_overlap_frames: int,
    overlap_slack: int,
    max_loser_fraction: float,
    max_losers_per_component: int,
    min_conflict_edges: int,
    score_mode: str,
    new_id_offset: int,
) -> tuple[dict[int, int], dict[str, object]]:
    pred = _base_pred(rows, pred_col, keep_seqs)
    plans = _component_plans(
        rows,
        pred,
        min_overlap_frames=int(min_overlap_frames),
        overlap_slack=int(overlap_slack),
        score_mode=str(score_mode),
    )
    out = dict(pred)
    next_gid = int(new_id_offset)
    touched_components = 0
    skipped_too_many_losers = 0
    skipped_no_conflict = 0
    singleton_losers = 0
    conflict_edges_total = 0
    max_component_losers = 0

    for gid, plan in sorted(plans.items()):
        seqs = list(plan["seqs"])
        if len(seqs) < int(min_component_size):
            continue
        conflict_edges = int(plan["conflict_edges"])
        conflict_edges_total += int(conflict_edges)
        if conflict_edges < int(min_conflict_edges):
            skipped_no_conflict += 1
            continue
        losers = list(plan["losers"])
        if not losers:
            continue
        max_component_losers = max(max_component_losers, len(losers))
        loser_fraction = float(len(losers) / max(len(seqs), 1))
        if float(max_loser_fraction) > 0 and loser_fraction > float(max_loser_fraction):
            skipped_too_many_losers += 1
            continue
        if int(max_losers_per_component) > 0 and len(losers) > int(max_losers_per_component):
            skipped_too_many_losers += 1
            continue
        touched_components += 1
        for seq in losers:
            out[int(seq)] = next_gid
            next_gid += 1
            singleton_losers += 1

    return out, {
        "mode": "cannotlink_nms_singleton",
        "min_component_size": int(min_component_size),
        "min_overlap_frames": int(min_overlap_frames),
        "overlap_slack": int(overlap_slack),
        "max_loser_fraction": float(max_loser_fraction),
        "max_losers_per_component": int(max_losers_per_component),
        "min_conflict_edges": int(min_conflict_edges),
        "score_mode": str(score_mode),
        "touched_components": int(touched_components),
        "singleton_losers": int(singleton_losers),
        "skipped_too_many_losers": int(skipped_too_many_losers),
        "skipped_no_conflict": int(skipped_no_conflict),
        "conflict_edges_total": int(conflict_edges_total),
        "max_component_losers": int(max_component_losers),
        "assignment_components": int(len(set(out.values()))),
        "output_tracklets": int(len(out)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }


def _write_assignments(path: str, rows: dict[int, dict[str, object]], fields: list[str], pred_by_seq: dict[int, int], pred_col: str) -> dict[str, object]:
    out_fields = list(fields)
    if pred_col not in out_fields:
        out_fields.append(pred_col)
    if "decision_status" not in out_fields:
        out_fields.append("decision_status")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    status_counts: Counter[str] = Counter()
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        for seq in sorted(pred_by_seq):
            row = {field: rows[int(seq)].get(field, "") for field in out_fields}
            row["seq"] = int(seq)
            base_gid = int(rows[int(seq)][pred_col])
            new_gid = int(pred_by_seq[int(seq)])
            row[pred_col] = new_gid
            status = str(row.get("decision_status") or "")
            if new_gid != base_gid:
                status = f"{status}|cannotlink_nms_singleton" if status else "cannotlink_nms_singleton"
            row["decision_status"] = status
            status_counts[status or ""] += 1
            writer.writerow(row)
    return {
        "assignments_out": str(path),
        "assignment_rows": int(len(pred_by_seq)),
        "assignment_components": int(len(set(pred_by_seq.values()))),
        "assignment_status_counts": dict(status_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--min-component-sizes", default="32,64,96,128")
    ap.add_argument("--min-overlap-frames", default="1,5,15,30")
    ap.add_argument("--overlap-slacks", default="0,5")
    ap.add_argument("--max-loser-fractions", default="0.01,0.02,0.05,0.10,0.20,1.0")
    ap.add_argument("--max-losers-per-component", default="1,2,4,8,16,0")
    ap.add_argument("--min-conflict-edges", default="1,5,10")
    ap.add_argument("--score-modes", default="n_dets_conf,n_dets_conf_pred,n_dets_conf_margin")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--new-id-offset", type=int, default=95_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    rows, fields = _load_assignment(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, _emb = _load_tracklets(con, args.role)
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
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in rows}
    seqs = [int(record.seq) for record in records]
    base_pred = _base_pred(rows, args.pred_col, keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", **base_pair}, sort_keys=True), flush=True)

    scored: list[tuple[dict[str, object], dict[int, int]]] = []
    for min_component_size in _parse_ints(args.min_component_sizes):
        for min_overlap_frames in _parse_ints(args.min_overlap_frames):
            for overlap_slack in _parse_ints(args.overlap_slacks):
                for max_loser_fraction in _parse_floats(args.max_loser_fractions):
                    for max_losers in _parse_ints(args.max_losers_per_component):
                        for min_conflict_edges in _parse_ints(args.min_conflict_edges):
                            for score_mode in [part.strip() for part in str(args.score_modes).split(",") if part.strip()]:
                                pred, info = _nms_singleton_pred(
                                    rows,
                                    args.pred_col,
                                    keep_seqs,
                                    min_component_size=int(min_component_size),
                                    min_overlap_frames=int(min_overlap_frames),
                                    overlap_slack=int(overlap_slack),
                                    max_loser_fraction=float(max_loser_fraction),
                                    max_losers_per_component=int(max_losers),
                                    min_conflict_edges=int(min_conflict_edges),
                                    score_mode=str(score_mode),
                                    new_id_offset=int(args.new_id_offset),
                                )
                                pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                scored.append(({**info, **pair}, pred))
    scored.sort(key=lambda item: (float(item[0]["tracklet_pair_f1"]), float(item[0]["tracklet_pair_precision"])), reverse=True)

    full_rows = []
    for rank, (row, pred) in enumerate(scored[: max(int(args.full_top_n), 0)], start=1):
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        full_rows.append(dict(row))
        print(json.dumps({"stage": "full", "rank": rank, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and scored:
        assignment_info = _write_assignments(args.assignments_out, rows, fields, scored[0][1], args.pred_col)
        scored[0][0].update(assignment_info)

    result = {
        "assignment_csv": args.assignment_csv,
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "assignment_info": assignment_info,
        "top": [row for row, _pred in scored[:100]],
        "full_rows": full_rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        keys = sorted(key for row, _pred in scored for key, value in row.items() if not isinstance(value, (dict, list, tuple)))
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row, _pred in scored:
                writer.writerow({key: row.get(key) for key in keys})
    print(json.dumps({"json": str(out), "base": base_pair, "best": scored[0][0] if scored else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
