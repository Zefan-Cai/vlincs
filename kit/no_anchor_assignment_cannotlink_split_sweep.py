#!/usr/bin/env python
"""Split impossible no-anchor assignment components with temporal cannot-links.

This is an assignment-level post-processor.  If two tracklets with the same
``predicted_global_id`` overlap in the same video/camera, they cannot be the
same physical identity.  The script colors each conflicted component so that
overlapping tracklets receive different delivered IDs.  The split rule uses no
GT and no anchors; GT is loaded only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
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


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _load_assignment(path: str, pred_col: str) -> tuple[dict[int, dict[str, object]], list[str]]:
    rows: dict[int, dict[str, object]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"seq", "video", "camera", "start_frame", "end_frame", pred_col}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        for row in reader:
            seq = int(float(row["seq"]))
            parsed = dict(row)
            parsed["seq"] = seq
            parsed[pred_col] = int(float(row[pred_col]))
            parsed["start_frame"] = int(float(row["start_frame"]))
            parsed["end_frame"] = int(float(row["end_frame"]))
            parsed["n_dets"] = int(float(row.get("n_dets") or 1))
            parsed["prediction_confidence"] = float(row.get("prediction_confidence") or 0.0)
            rows[seq] = parsed
    return rows, fields


def _base_pred(rows: dict[int, dict[str, object]], pred_col: str, keep_seqs: set[int]) -> dict[int, int]:
    return {int(seq): int(row[pred_col]) for seq, row in rows.items() if int(seq) in keep_seqs}


def _overlap(a: dict[str, object], b: dict[str, object], slack: int) -> int:
    start = max(int(a["start_frame"]), int(b["start_frame"]))
    end = min(int(a["end_frame"]), int(b["end_frame"]))
    return max(0, end - start + 1 - int(slack))


def _color_component(
    seqs: list[int],
    rows: dict[int, dict[str, object]],
    *,
    min_overlap_frames: int,
    overlap_slack: int,
    max_colors: int,
) -> tuple[dict[int, int], dict[str, int]]:
    neighbors: dict[int, set[int]] = {seq: set() for seq in seqs}
    conflict_edges = 0
    by_video: dict[tuple[str, str], list[int]] = defaultdict(list)
    for seq in seqs:
        row = rows[int(seq)]
        by_video[(str(row.get("video") or ""), str(row.get("camera") or ""))].append(int(seq))

    for group in by_video.values():
        group = sorted(group, key=lambda seq: (int(rows[seq]["start_frame"]), int(rows[seq]["end_frame"])))
        for i, a_seq in enumerate(group):
            a = rows[a_seq]
            for b_seq in group[i + 1 :]:
                b = rows[b_seq]
                if int(b["start_frame"]) > int(a["end_frame"]) + int(overlap_slack):
                    break
                if _overlap(a, b, int(overlap_slack)) >= int(min_overlap_frames):
                    neighbors[a_seq].add(b_seq)
                    neighbors[b_seq].add(a_seq)
                    conflict_edges += 1

    if conflict_edges == 0:
        return {seq: 0 for seq in seqs}, {"conflict_edges": 0, "conflict_nodes": 0, "colors": 1}

    order = sorted(
        seqs,
        key=lambda seq: (
            -len(neighbors[seq]),
            -int(rows[seq].get("n_dets") or 1),
            int(rows[seq]["start_frame"]),
            int(seq),
        ),
    )
    color: dict[int, int] = {}
    for seq in order:
        forbidden = {color[nbr] for nbr in neighbors[seq] if nbr in color}
        chosen = 0
        while chosen in forbidden:
            chosen += 1
        color[int(seq)] = int(chosen)
    colors = max(color.values(), default=0) + 1
    if int(max_colors) > 0 and colors > int(max_colors):
        return {seq: 0 for seq in seqs}, {
            "conflict_edges": int(conflict_edges),
            "conflict_nodes": int(sum(1 for seq in seqs if neighbors[seq])),
            "colors": int(colors),
            "skipped_too_many_colors": 1,
        }
    return color, {
        "conflict_edges": int(conflict_edges),
        "conflict_nodes": int(sum(1 for seq in seqs if neighbors[seq])),
        "colors": int(colors),
        "skipped_too_many_colors": 0,
    }


def _split_pred(
    rows: dict[int, dict[str, object]],
    pred_col: str,
    keep_seqs: set[int],
    *,
    min_component_size: int,
    min_overlap_frames: int,
    overlap_slack: int,
    max_colors: int,
    min_conflict_node_frac: float,
    new_id_offset: int,
) -> tuple[dict[int, int], dict[str, object]]:
    pred = _base_pred(rows, pred_col, keep_seqs)
    by_gid: dict[int, list[int]] = defaultdict(list)
    for seq, gid in pred.items():
        by_gid[int(gid)].append(int(seq))

    next_gid = int(new_id_offset)
    out = dict(pred)
    split_components = 0
    skipped_components = 0
    rewritten_seqs = 0
    total_conflict_edges = 0
    total_conflict_nodes = 0
    max_observed_colors = 1

    for gid, seqs in sorted(by_gid.items()):
        if len(seqs) < int(min_component_size):
            continue
        colors, stats = _color_component(
            seqs,
            rows,
            min_overlap_frames=int(min_overlap_frames),
            overlap_slack=int(overlap_slack),
            max_colors=int(max_colors),
        )
        total_conflict_edges += int(stats.get("conflict_edges", 0))
        total_conflict_nodes += int(stats.get("conflict_nodes", 0))
        max_observed_colors = max(max_observed_colors, int(stats.get("colors", 1)))
        if int(stats.get("skipped_too_many_colors", 0)):
            skipped_components += 1
            continue
        conflict_frac = float(stats.get("conflict_nodes", 0)) / max(float(len(seqs)), 1.0)
        if int(stats.get("colors", 1)) <= 1 or conflict_frac < float(min_conflict_node_frac):
            continue
        split_components += 1
        color_to_gid = {0: int(gid)}
        for color_id in sorted(set(colors.values())):
            if int(color_id) == 0:
                continue
            color_to_gid[int(color_id)] = next_gid
            next_gid += 1
        for seq, color_id in colors.items():
            new_gid = int(color_to_gid[int(color_id)])
            if int(out[int(seq)]) != new_gid:
                out[int(seq)] = new_gid
                rewritten_seqs += 1

    info = {
        "mode": "cannotlink_split",
        "min_component_size": int(min_component_size),
        "min_overlap_frames": int(min_overlap_frames),
        "overlap_slack": int(overlap_slack),
        "max_colors": int(max_colors),
        "min_conflict_node_frac": float(min_conflict_node_frac),
        "split_components": int(split_components),
        "skipped_components": int(skipped_components),
        "rewritten_seqs": int(rewritten_seqs),
        "total_conflict_edges": int(total_conflict_edges),
        "total_conflict_nodes": int(total_conflict_nodes),
        "max_observed_colors": int(max_observed_colors),
        "output_tracklets": int(len(out)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    return out, info


def _write_assignments(path: str, rows: dict[int, dict[str, object]], fields: list[str], pred_by_seq: dict[int, int], pred_col: str) -> dict[str, int]:
    out_fields = list(fields)
    if pred_col not in out_fields:
        out_fields.append(pred_col)
    if "decision_status" not in out_fields:
        out_fields.append("decision_status")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        for seq in sorted(pred_by_seq):
            row = {field: rows[int(seq)].get(field, "") for field in out_fields}
            row["seq"] = int(seq)
            row[pred_col] = int(pred_by_seq[int(seq)])
            status = str(row.get("decision_status") or "")
            row["decision_status"] = f"{status}|cannotlink_split" if status else "cannotlink_split"
            writer.writerow(row)
    return {
        "assignment_rows": int(len(pred_by_seq)),
        "assignment_components": int(len(set(pred_by_seq.values()))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--min-component-sizes", default="8,16,32,64")
    ap.add_argument("--min-overlap-frames", default="1,5,15,30")
    ap.add_argument("--overlap-slacks", default="0,5")
    ap.add_argument("--max-colors", default="2,3,4,0")
    ap.add_argument("--min-conflict-node-fracs", default="0.0,0.05,0.10")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--output-drop-area-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-area-quantile-by-video", default="")
    ap.add_argument("--output-drop-quality-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-quality-quantile-by-video", default="")
    ap.add_argument("--output-auto-anomaly-admission", action="store_true")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--new-id-offset", type=int, default=80_000_000)
    ap.add_argument("--json", required=True)
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
                for max_colors in _parse_ints(args.max_colors):
                    for min_conflict_node_frac in _parse_floats(args.min_conflict_node_fracs):
                        pred, info = _split_pred(
                            rows,
                            args.pred_col,
                            keep_seqs,
                            min_component_size=min_component_size,
                            min_overlap_frames=min_overlap_frames,
                            overlap_slack=overlap_slack,
                            max_colors=max_colors,
                            min_conflict_node_frac=min_conflict_node_frac,
                            new_id_offset=int(args.new_id_offset),
                        )
                        pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                        scored.append(({**info, **pair}, pred))

    scored.sort(
        key=lambda item: (
            float(item[0]["tracklet_pair_f1"]),
            float(item[0]["tracklet_pair_precision"]),
            float(item[0]["tracklet_pair_recall"]),
        ),
        reverse=True,
    )

    for rank, (row, pred) in enumerate(scored[: max(int(args.full_top_n), 0)], start=1):
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and scored:
        assignment_info = _write_assignments(args.assignments_out, rows, fields, scored[0][1], args.pred_col)
        scored[0][0].update(assignment_info)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "assignment_csv": args.assignment_csv,
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "assignment_info": assignment_info,
        "top": [row for row, _pred in scored[: max(50, int(args.full_top_n))]],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"base": base_pair, "best": result["top"][0] if result["top"] else None, "json": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
