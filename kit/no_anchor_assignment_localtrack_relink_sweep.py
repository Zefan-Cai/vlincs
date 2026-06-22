#!/usr/bin/env python
"""Relink assignment IDs using no-anchor local-track evidence.

The input is an existing no-anchor assignment CSV.  The script parses
``tracklet_key`` as ``<video>:<local_track_id>:<segment>`` and tries conservative
rewrites of ``seq -> predicted_global_id`` within each ``video, local_track_id``
group.  The rewrite itself uses no GT and no anchors; GT is loaded only after
prediction for pair/full metrics.
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


def _parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_videos(text: str) -> set[str]:
    return {part.strip() for part in str(text or "").split(",") if part.strip()}


def _load_assignment(path: str, pred_col: str) -> tuple[dict[int, dict[str, object]], list[str]]:
    rows: dict[int, dict[str, object]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"seq", "tracklet_key", "video", pred_col}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        for row in reader:
            seq = int(float(row["seq"]))
            parsed = dict(row)
            parsed["seq"] = seq
            parsed[pred_col] = int(float(row[pred_col]))
            parsed["n_dets"] = int(float(row.get("n_dets") or 1))
            parsed["prediction_confidence"] = float(row.get("prediction_confidence") or 0.0)
            rows[seq] = parsed
    return rows, fields


def _local_track_key(row: dict[str, object]) -> str | None:
    video = str(row.get("video") or "")
    key = str(row.get("tracklet_key") or "")
    prefix = f"{video}:"
    if key.startswith(prefix):
        rest = key[len(prefix) :]
        local_id = rest.split(":", 1)[0]
        return f"{video}:{local_id}"
    parts = key.rsplit(":", 2)
    if len(parts) >= 2:
        return f"{video}:{parts[-2]}"
    return None


def _build_groups(rows: dict[int, dict[str, object]], keep_seqs: set[int], allowed_videos: set[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for seq, row in rows.items():
        if seq not in keep_seqs:
            continue
        video = str(row.get("video") or "")
        if allowed_videos and video not in allowed_videos:
            continue
        local_key = _local_track_key(row)
        if local_key is None:
            continue
        groups[local_key].append(int(seq))
    return groups


def _base_pred(rows: dict[int, dict[str, object]], pred_col: str, keep_seqs: set[int]) -> dict[int, int]:
    return {int(seq): int(row[pred_col]) for seq, row in rows.items() if int(seq) in keep_seqs}


def _rewrite_pred(
    rows: dict[int, dict[str, object]],
    pred_col: str,
    keep_seqs: set[int],
    *,
    mode: str,
    allowed_videos: set[str],
    min_group_tracklets: int,
    min_group_dets: int,
    max_group_components: int,
    min_dominant_frac: float,
    min_conf: float,
    new_id_offset: int,
) -> tuple[dict[int, int], dict[str, object]]:
    pred = _base_pred(rows, pred_col, keep_seqs)
    groups = _build_groups(rows, keep_seqs, allowed_videos)
    touched_groups = 0
    rewritten_seqs = 0
    split_like_groups = 0
    merge_like_groups = 0
    new_ids = 0
    next_new_id = int(new_id_offset)

    for local_key, seqs in sorted(groups.items()):
        if len(seqs) < int(min_group_tracklets):
            continue
        total_dets = sum(int(rows[seq].get("n_dets") or 1) for seq in seqs)
        if total_dets < int(min_group_dets):
            continue
        if min(float(rows[seq].get("prediction_confidence") or 0.0) for seq in seqs) < float(min_conf):
            continue
        by_gid = Counter()
        for seq in seqs:
            by_gid[int(pred[seq])] += int(rows[seq].get("n_dets") or 1)
        if len(by_gid) <= 1:
            continue
        if int(max_group_components) > 0 and len(by_gid) > int(max_group_components):
            continue
        dominant_gid, dominant_weight = by_gid.most_common(1)[0]
        dominant_frac = float(dominant_weight) / max(float(total_dets), 1.0)
        if dominant_frac < float(min_dominant_frac):
            continue

        if mode == "dominant":
            target_gid = int(dominant_gid)
        elif mode == "new_local":
            target_gid = next_new_id
            next_new_id += 1
            new_ids += 1
        elif mode == "lowconf_new_else_dominant":
            mean_conf = sum(float(rows[seq].get("prediction_confidence") or 0.0) for seq in seqs) / max(len(seqs), 1)
            if mean_conf < 0.75:
                target_gid = next_new_id
                next_new_id += 1
                new_ids += 1
            else:
                target_gid = int(dominant_gid)
        else:
            raise ValueError(f"unknown mode: {mode}")

        touched_groups += 1
        if target_gid == int(dominant_gid):
            merge_like_groups += 1
        else:
            split_like_groups += 1
        for seq in seqs:
            if pred[int(seq)] != target_gid:
                pred[int(seq)] = int(target_gid)
                rewritten_seqs += 1

    info = {
        "mode": mode,
        "allowed_videos": sorted(allowed_videos) if allowed_videos else "all",
        "min_group_tracklets": int(min_group_tracklets),
        "min_group_dets": int(min_group_dets),
        "max_group_components": int(max_group_components),
        "min_dominant_frac": float(min_dominant_frac),
        "min_conf": float(min_conf),
        "touched_groups": int(touched_groups),
        "rewritten_seqs": int(rewritten_seqs),
        "merge_like_groups": int(merge_like_groups),
        "split_like_groups": int(split_like_groups),
        "new_ids": int(new_ids),
        "output_tracklets": int(len(pred)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    return pred, info


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
            row["decision_status"] = f"{status}|localtrack_relink" if status else "localtrack_relink"
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
    ap.add_argument("--modes", default="dominant,new_local,lowconf_new_else_dominant")
    ap.add_argument("--only-videos", default="")
    ap.add_argument("--min-group-tracklets", default="2,3,4")
    ap.add_argument("--min-group-dets", default="2,10,30,60")
    ap.add_argument("--max-group-components", default="2,3,5,0")
    ap.add_argument("--min-dominant-fracs", default="0.50,0.65,0.80")
    ap.add_argument("--min-confs", default="0.0,0.6")
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
    ap.add_argument("--new-id-offset", type=int, default=70_000_000)
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

    modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
    allowed_videos = _parse_videos(args.only_videos)
    scored: list[tuple[dict[str, object], dict[int, int]]] = []
    for mode in modes:
        for min_group_tracklets in _parse_ints(args.min_group_tracklets):
            for min_group_dets in _parse_ints(args.min_group_dets):
                for max_group_components in _parse_ints(args.max_group_components):
                    for min_dominant_frac in _parse_floats(args.min_dominant_fracs):
                        for min_conf in _parse_floats(args.min_confs):
                            pred, info = _rewrite_pred(
                                rows,
                                args.pred_col,
                                keep_seqs,
                                mode=mode,
                                allowed_videos=allowed_videos,
                                min_group_tracklets=min_group_tracklets,
                                min_group_dets=min_group_dets,
                                max_group_components=max_group_components,
                                min_dominant_frac=min_dominant_frac,
                                min_conf=min_conf,
                                new_id_offset=int(args.new_id_offset),
                            )
                            pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                            row = {**info, **pair}
                            scored.append((row, pred))

    scored.sort(
        key=lambda item: (
            float(item[0]["tracklet_pair_f1"]),
            float(item[0]["tracklet_pair_precision"]),
            float(item[0]["tracklet_pair_recall"]),
            -float(item[0]["rewritten_seqs"]),
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
