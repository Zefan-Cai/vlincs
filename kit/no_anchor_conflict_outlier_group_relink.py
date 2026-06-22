#!/usr/bin/env python
"""Group conflict-outlier relinks into higher-mass no-anchor edits.

The single-tracklet source-retention critic can identify true relinks, but the
strict gate only yields a couple of edits.  This proposer first builds the same
no-GT single-tracklet evidence, then groups compatible source->target moves and
scores the grouped edit.  Ground truth is used only if pair/full scoring is
requested after materialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from kit.no_anchor_conflict_outlier_relink import _build_candidates, _l2n
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
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
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from no_anchor_conflict_outlier_relink import _build_candidates, _l2n
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
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


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _min(values: list[float]) -> float:
    return float(np.min(values)) if values else 0.0


def _group_key(row: dict[str, object], mode: str) -> tuple[object, ...]:
    base = (int(row["source_label"]), int(row["target_label"]))
    if mode == "source_target_camera":
        return (*base, str(row["camera"]))
    if mode == "source_target_video":
        return (*base, str(row["video"]))
    return base


def _build_groups(
    candidates: list[dict[str, object]],
    records_by_seq: dict[int, object],
    *,
    group_key_mode: str,
    max_group_size: int,
    min_group_size: int,
    min_group_dets: int,
    min_group_mean_score: float,
    min_group_mean_margin: float,
    min_group_mean_neighbor_margin: float,
    min_group_min_neighbor_margin: float,
    max_group_source_overlap: int,
) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        buckets[_group_key(row, group_key_mode)].append(row)

    groups: list[dict[str, object]] = []
    for key, rows in buckets.items():
        rows = sorted(
            rows,
            key=lambda row: (float(row["score"]), float(row["neighbor_margin"]), float(row["margin"])),
            reverse=True,
        )[: max(1, int(max_group_size))]
        dets = 0
        conf_weighted = 0.0
        source_overlap = 0
        videos = set()
        cameras = set()
        for row in rows:
            rec = records_by_seq.get(int(row["seq"]))
            if rec is None:
                continue
            n_dets = int(getattr(rec, "n_dets", 1))
            dets += n_dets
            conf_weighted += n_dets * float(getattr(rec, "avg_conf", 0.0))
            source_overlap += int(row.get("source_temporal_overlap", 0))
            videos.add(str(row["video"]))
            cameras.add(str(row["camera"]))
        scores = [float(row["score"]) for row in rows]
        margins = [float(row["margin"]) for row in rows]
        neighbor_margins = [float(row["neighbor_margin"]) for row in rows]
        target_sims = [float(row["target_sim"]) for row in rows]
        source_nn_max = [float(row["source_nn_max"]) for row in rows]
        if len(rows) < int(min_group_size):
            continue
        if dets < int(min_group_dets):
            continue
        if _mean(scores) < float(min_group_mean_score):
            continue
        if _mean(margins) < float(min_group_mean_margin):
            continue
        if _mean(neighbor_margins) < float(min_group_mean_neighbor_margin):
            continue
        if _min(neighbor_margins) < float(min_group_min_neighbor_margin):
            continue
        if source_overlap > int(max_group_source_overlap):
            continue
        mean_conf = conf_weighted / max(dets, 1)
        mass_bonus = 0.025 * min(math.log1p(dets) / math.log(2000.0), 1.0)
        size_bonus = 0.025 * min(math.log1p(len(rows)) / math.log(16.0), 1.0)
        purity_bonus = 0.08 * _mean(neighbor_margins) + 0.03 * _mean(target_sims)
        source_penalty = 0.03 * max(0.0, _mean(source_nn_max) - 0.75)
        group_score = _mean(scores) + mass_bonus + size_bonus + purity_bonus + 0.02 * mean_conf - source_penalty
        group = {
            "mode": "conflict_outlier_group_relink",
            "group_key_mode": str(group_key_mode),
            "source_label": int(rows[0]["source_label"]),
            "target_label": int(rows[0]["target_label"]),
            "group_size": int(len(rows)),
            "group_dets": int(dets),
            "group_mean_conf": round(float(mean_conf), 6),
            "group_score": round(float(group_score), 6),
            "group_mean_single_score": round(_mean(scores), 6),
            "group_mean_margin": round(_mean(margins), 6),
            "group_min_margin": round(_min(margins), 6),
            "group_mean_neighbor_margin": round(_mean(neighbor_margins), 6),
            "group_min_neighbor_margin": round(_min(neighbor_margins), 6),
            "group_mean_target_sim": round(_mean(target_sims), 6),
            "group_mean_source_nn_max": round(_mean(source_nn_max), 6),
            "group_source_temporal_overlap": int(source_overlap),
            "group_video_count": int(len(videos)),
            "group_camera_count": int(len(cameras)),
            "accepted_preview": rows,
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": False,
        }
        groups.append(group)
    groups.sort(
        key=lambda row: (
            float(row["group_score"]),
            int(row["group_dets"]),
            int(row["group_size"]),
            float(row["group_mean_neighbor_margin"]),
        ),
        reverse=True,
    )
    return groups


def _apply_groups(
    base_labels: np.ndarray,
    groups: list[dict[str, object]],
    seq_to_idx: dict[int, int],
    *,
    top_k: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels = base_labels.copy()
    accepted: list[dict[str, object]] = []
    used_seqs: set[int] = set()
    for group in groups:
        if len(accepted) >= int(top_k):
            break
        moved = []
        for row in group["accepted_preview"]:
            seq = int(row["seq"])
            idx = seq_to_idx.get(seq)
            if idx is None or seq in used_seqs:
                continue
            if int(labels[int(idx)]) != int(row["source_label"]):
                continue
            labels[int(idx)] = int(row["target_label"])
            moved.append(dict(row))
            used_seqs.add(seq)
        if moved:
            kept = dict(group)
            kept["accepted_preview"] = moved
            kept["applied_group_size"] = int(len(moved))
            accepted.append(kept)
    return labels, accepted


def _write_assignments(path: str, records, labels: np.ndarray, keep_seqs: set[int], *, offset: int, accepted_groups: list[dict[str, object]]) -> dict[str, object]:
    moved = {}
    for group in accepted_groups:
        for row in group["accepted_preview"]:
            moved[int(row["seq"])] = row
    rows = []
    for idx, record in enumerate(records):
        seq = int(record.seq)
        if seq not in keep_seqs:
            continue
        rows.append(
            {
                "seq": seq,
                "tracklet_key": record.tracklet_key,
                "video": record.video,
                "camera": record.camera,
                "start_frame": int(record.start_frame),
                "end_frame": int(record.end_frame),
                "n_dets": int(record.n_dets),
                "avg_conf": round(float(record.avg_conf), 6),
                "predicted_global_id": int(offset) + int(labels[idx]),
                "decision_status": "group_outlier_relinked" if seq in moved else "base_component",
            }
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["seq", "predicted_global_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "assignments_out": str(path),
        "assignment_rows": int(len(rows)),
        "accepted_groups": int(len(accepted_groups)),
        "accepted_relinks": int(len(moved)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--min-source-sizes", default="32")
    ap.add_argument("--max-source-size", type=int, default=500)
    ap.add_argument("--min-target-size", type=int, default=4)
    ap.add_argument("--max-target-size", type=int, default=500)
    ap.add_argument("--min-conflict-edges", type=int, default=1)
    ap.add_argument("--max-source-sims", default="0.78")
    ap.add_argument("--min-target-sims", default="0.66")
    ap.add_argument("--min-margins", default="0.00")
    ap.add_argument("--max-candidates-per-source", type=int, default=256)
    ap.add_argument("--support-top-k", type=int, default=3)
    ap.add_argument("--max-temporal-gap-frames", type=int, default=300)
    ap.add_argument("--min-neighbor-margins", default="0.08,0.12,0.16")
    ap.add_argument("--max-source-neighbor-sims", default="0.82,0.88")
    ap.add_argument("--max-source-temporal-supports", default="1000000")
    ap.add_argument("--rank-modes", default="source_retention,target_neighbor")
    ap.add_argument("--group-key-modes", default="source_target_camera,source_target_video")
    ap.add_argument("--max-group-size", type=int, default=24)
    ap.add_argument("--min-group-sizes", default="2,3")
    ap.add_argument("--min-group-dets", type=int, default=20)
    ap.add_argument("--min-group-mean-scores", default="-10")
    ap.add_argument("--min-group-mean-margins", default="0.03,0.08")
    ap.add_argument("--min-group-mean-neighbor-margins", default="0.12,0.16")
    ap.add_argument("--min-group-min-neighbor-margins", default="-0.05,0.02")
    ap.add_argument("--max-group-source-overlap", type=int, default=8)
    ap.add_argument("--top-group-ks", default="1,2,4")
    ap.add_argument("--top-groups-out", type=int, default=100)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--score-pair", action="store_true")
    ap.add_argument("--score-full", action="store_true")
    ap.add_argument("--assignment-offset", type=int, default=96_000_000)
    ap.add_argument("--assignments-dir", default="")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    emb = _load_feature_npz(args.feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0).astype(np.float32)
    emb = _l2n(emb)
    keep_seqs, output_info = _output_keep_seqs(records, args)
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    records_by_seq = {int(record.seq): record for record in records}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
    forbidden = _build_overlap_forbidden(records)

    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    gt_by_seq = weight_by_seq = eval_stats = None
    if bool(args.score_pair):
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

    rows = []
    group_cache: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for min_source_size in _parse_ints(args.min_source_sizes):
        for max_source_sim in _parse_floats(args.max_source_sims):
            for min_target_sim in _parse_floats(args.min_target_sims):
                for min_margin in _parse_floats(args.min_margins):
                    for min_neighbor_margin in _parse_floats(args.min_neighbor_margins):
                        for max_source_neighbor_sim in _parse_floats(args.max_source_neighbor_sims):
                            for max_source_temporal_support in _parse_ints(args.max_source_temporal_supports):
                                for rank_mode in [part.strip() for part in str(args.rank_modes).split(",") if part.strip()]:
                                    candidates = _build_candidates(
                                        records,
                                        base_labels,
                                        keep_indices,
                                        emb,
                                        forbidden,
                                        min_source_size=int(min_source_size),
                                        max_source_size=int(args.max_source_size),
                                        min_target_size=int(args.min_target_size),
                                        max_target_size=int(args.max_target_size),
                                        min_conflict_edges=int(args.min_conflict_edges),
                                        max_source_sim=float(max_source_sim),
                                        min_target_sim=float(min_target_sim),
                                        min_margin=float(min_margin),
                                        max_candidates_per_source=int(args.max_candidates_per_source),
                                        support_top_k=int(args.support_top_k),
                                        max_temporal_gap_frames=int(args.max_temporal_gap_frames),
                                        min_neighbor_margin=float(min_neighbor_margin),
                                        max_source_neighbor_sim=float(max_source_neighbor_sim),
                                        max_source_temporal_support=int(max_source_temporal_support),
                                        rank_mode=str(rank_mode),
                                    )
                                    for group_key_mode in [part.strip() for part in str(args.group_key_modes).split(",") if part.strip()]:
                                        for min_group_size in _parse_ints(args.min_group_sizes):
                                            for min_group_mean_score in _parse_floats(args.min_group_mean_scores):
                                                for min_group_mean_margin in _parse_floats(args.min_group_mean_margins):
                                                    for min_group_mean_neighbor_margin in _parse_floats(args.min_group_mean_neighbor_margins):
                                                        for min_group_min_neighbor_margin in _parse_floats(args.min_group_min_neighbor_margins):
                                                            cache_key = (
                                                                int(min_source_size),
                                                                float(max_source_sim),
                                                                float(min_target_sim),
                                                                float(min_margin),
                                                                float(min_neighbor_margin),
                                                                float(max_source_neighbor_sim),
                                                                int(max_source_temporal_support),
                                                                str(rank_mode),
                                                                str(group_key_mode),
                                                                int(min_group_size),
                                                                float(min_group_mean_score),
                                                                float(min_group_mean_margin),
                                                                float(min_group_mean_neighbor_margin),
                                                                float(min_group_min_neighbor_margin),
                                                            )
                                                            groups = _build_groups(
                                                                candidates,
                                                                records_by_seq,
                                                                group_key_mode=str(group_key_mode),
                                                                max_group_size=int(args.max_group_size),
                                                                min_group_size=int(min_group_size),
                                                                min_group_dets=int(args.min_group_dets),
                                                                min_group_mean_score=float(min_group_mean_score),
                                                                min_group_mean_margin=float(min_group_mean_margin),
                                                                min_group_mean_neighbor_margin=float(min_group_mean_neighbor_margin),
                                                                min_group_min_neighbor_margin=float(min_group_min_neighbor_margin),
                                                                max_group_source_overlap=int(args.max_group_source_overlap),
                                                            )
                                                            group_cache[cache_key] = groups
                                                            for top_k in _parse_ints(args.top_group_ks):
                                                                labels, accepted_groups = _apply_groups(base_labels, groups, seq_to_idx, top_k=int(top_k))
                                                                pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                                                accepted_relinks = sum(int(group.get("applied_group_size", 0)) for group in accepted_groups)
                                                                row = {
                                                                    "mode": "conflict_outlier_group_relink",
                                                                    "min_source_size": int(min_source_size),
                                                                    "max_source_sim": float(max_source_sim),
                                                                    "min_target_sim": float(min_target_sim),
                                                                    "min_margin": float(min_margin),
                                                                    "min_neighbor_margin": float(min_neighbor_margin),
                                                                    "max_source_neighbor_sim": float(max_source_neighbor_sim),
                                                                    "rank_mode": str(rank_mode),
                                                                    "group_key_mode": str(group_key_mode),
                                                                    "min_group_size": int(min_group_size),
                                                                    "min_group_mean_score": float(min_group_mean_score),
                                                                    "min_group_mean_margin": float(min_group_mean_margin),
                                                                    "min_group_mean_neighbor_margin": float(min_group_mean_neighbor_margin),
                                                                    "min_group_min_neighbor_margin": float(min_group_min_neighbor_margin),
                                                                    "top_group_k": int(top_k),
                                                                    "candidate_count": int(len(candidates)),
                                                                    "group_count": int(len(groups)),
                                                                    "accepted_groups": int(len(accepted_groups)),
                                                                    "accepted_relinks": int(accepted_relinks),
                                                                    "accepted_preview": accepted_groups[:8],
                                                                    "uses_anchors": False,
                                                                    "uses_gt_for_training_or_anchors": False,
                                                                    "uses_gt_for_evaluation_only": bool(args.score_pair or args.score_full),
                                                                }
                                                                if bool(args.score_pair) and gt_by_seq is not None and weight_by_seq is not None:
                                                                    row.update(_pair_metrics([int(record.seq) for record in records], pred, gt_by_seq, weight_by_seq))
                                                                if bool(args.score_full):
                                                                    full = _score_full(pred_by_video, gt_by_video, pred)
                                                                    row.update({f"full_{name}": value for name, value in full.items() if name != "per_video"})
                                                                    row["full_per_video"] = full["per_video"]
                                                                if args.assignments_dir:
                                                                    sig = (
                                                                        f"src{min_source_size}_ss{str(max_source_sim).replace('.', 'p')}"
                                                                        f"_ts{str(min_target_sim).replace('.', 'p')}_m{str(min_margin).replace('.', 'p')}"
                                                                        f"_nm{str(min_neighbor_margin).replace('.', 'p').replace('-', 'n')}"
                                                                        f"_sn{str(max_source_neighbor_sim).replace('.', 'p')}_{rank_mode}_{group_key_mode}"
                                                                        f"_gs{min_group_size}_gm{str(min_group_mean_margin).replace('.', 'p')}"
                                                                        f"_gn{str(min_group_mean_neighbor_margin).replace('.', 'p')}_gmin{str(min_group_min_neighbor_margin).replace('.', 'p').replace('-', 'n')}"
                                                                        f"_k{top_k}"
                                                                    )
                                                                    row.update(
                                                                        _write_assignments(
                                                                            str(Path(args.assignments_dir) / f"{sig}.csv"),
                                                                            records,
                                                                            labels,
                                                                            keep_seqs,
                                                                            offset=int(args.assignment_offset),
                                                                            accepted_groups=accepted_groups,
                                                                        )
                                                                    )
                                                                rows.append(row)

    sort_keys = (
        (lambda row: (float(row.get("full_IDF1", 0.0)), float(row.get("accepted_relinks", 0.0))))
        if bool(args.score_full)
        else (lambda row: (float(row.get("accepted_relinks", 0.0)), float(row.get("group_count", 0.0)), float(row.get("candidate_count", 0.0))))
    )
    rows.sort(key=sort_keys, reverse=True)
    result = {
        "assignment_csv": str(args.assignment_csv),
        "feature_npz": str(args.feature_npz),
        "base_assignment_components": int(len(raw_to_local)),
        "candidate_configs": int(len(group_cache)),
        "top": rows[: int(args.top_groups_out)],
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": bool(args.score_pair or args.score_full),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "rows": len(rows), "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
