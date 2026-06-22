#!/usr/bin/env python
"""Endpoint-pair micro-surgery for no-anchor assignments.

Unlike endpoint_component_relink, this materializes only the local endpoint
evidence: the source component and the best target endpoint tracklet become a
new small component, or the target endpoint is peeled as a singleton.  The
candidate source is no-GT endpoint evidence; ground truth is used only for
diagnostic scoring after materialization.
"""

from __future__ import annotations

import argparse
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
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from no_anchor_louvain_sweep import _write_assignments
    from no_anchor_resolve_sweep import (
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
    )


def _parse_modes(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _load_endpoint_candidates(path: str, max_rows: int, max_candidates: int) -> list[dict[str, object]]:
    data = json.loads(Path(path).read_text())
    rows = data.get("top", [])
    if max_rows > 0:
        rows = rows[:max_rows]
    out: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for rank, row in enumerate(rows, start=1):
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
            out.append({**cand, "source_result_rank": int(rank)})
            if max_candidates > 0 and len(out) >= max_candidates:
                return out
    out.sort(
        key=lambda row: (
            float(row.get("score", 0.0)),
            int(row.get("support_edges", 0)),
            float(row.get("visual", 0.0)),
        ),
        reverse=True,
    )
    return out


def _components(labels: np.ndarray, keep_indices: set[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        groups[int(labels[int(idx)])].append(int(idx))
    return dict(groups)


def _filter_candidates(
    candidates: list[dict[str, object]],
    *,
    min_score: float,
    min_visual: float,
    min_pos: float,
    min_scale: float,
    min_support_edges: int,
    max_source_size: int,
    max_target_size: int,
) -> list[dict[str, object]]:
    out = []
    for cand in candidates:
        if float(cand.get("score", 0.0)) < float(min_score):
            continue
        if float(cand.get("visual", 0.0)) < float(min_visual):
            continue
        if float(cand.get("pos", 0.0)) < float(min_pos):
            continue
        if float(cand.get("scale_sim", 0.0)) < float(min_scale):
            continue
        if int(cand.get("support_edges", 0)) < int(min_support_edges):
            continue
        if int(max_source_size) > 0 and int(cand.get("source_size", 0)) > int(max_source_size):
            continue
        if int(max_target_size) > 0 and int(cand.get("target_size", 0)) > int(max_target_size):
            continue
        out.append(cand)
    return out


def _apply_micro_surgery(
    records,
    base_labels: np.ndarray,
    groups: dict[int, list[int]],
    keep_indices: set[int],
    seq_to_idx: dict[int, int],
    candidates: list[dict[str, object]],
    *,
    mode: str,
    top_k: int,
    max_edits_per_video: int,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int]]:
    labels = base_labels.copy()
    accepted: list[dict[str, object]] = []
    used_sources: set[int] = set()
    moved_targets: set[int] = set()
    accepted_by_video: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    next_label = int(labels.max()) + 1
    for cand in candidates:
        if len(accepted) >= int(top_k):
            break
        video = str(cand.get("video"))
        if int(max_edits_per_video) > 0 and accepted_by_video[video] >= int(max_edits_per_video):
            rejected["video_cap"] += 1
            continue
        source_seq = int(cand["source_seq"])
        target_seq = int(cand["target_seq"])
        if source_seq not in seq_to_idx or target_seq not in seq_to_idx:
            rejected["missing_seq"] += 1
            continue
        source_idx = int(seq_to_idx[source_seq])
        target_idx = int(seq_to_idx[target_seq])
        source_label = int(base_labels[source_idx])
        target_label = int(base_labels[target_idx])
        if source_label in used_sources:
            rejected["source_used"] += 1
            continue
        if target_idx in moved_targets:
            rejected["target_moved"] += 1
            continue
        if int(labels[source_idx]) != source_label or int(labels[target_idx]) != target_label:
            rejected["stale_label"] += 1
            continue
        source_indices = [idx for idx in groups.get(source_label, []) if int(idx) in keep_indices]
        if not source_indices:
            rejected["empty_source"] += 1
            continue
        if str(mode) == "source_plus_target_new":
            moved_indices = sorted(set(source_indices + [target_idx]))
            for idx in moved_indices:
                labels[int(idx)] = next_label
            next_label += 1
            used_sources.add(source_label)
            moved_targets.add(target_idx)
        elif str(mode) == "target_endpoint_singleton":
            moved_indices = [target_idx]
            labels[target_idx] = next_label
            next_label += 1
            moved_targets.add(target_idx)
        else:
            raise ValueError(f"unknown mode: {mode}")
        accepted_by_video[video] += 1
        accepted.append(
            {
                **cand,
                "mode": str(mode),
                "actual_source_label": int(source_label),
                "actual_target_label": int(target_label),
                "moved_tracklets": int(len(moved_indices)),
                "moved_seqs": [int(records[int(idx)].seq) for idx in moved_indices],
            }
        )
    return labels, accepted, {key: int(value) for key, value in rejected.items()}


def _action_signature(accepted: list[dict[str, object]]) -> str:
    parts = []
    for row in accepted:
        moved = ",".join(str(int(seq)) for seq in row.get("moved_seqs", []))
        parts.append(f"{row.get('mode')}:{moved}")
    return "|".join(sorted(parts)) if parts else "noop"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--endpoint-json", required=True)
    ap.add_argument("--candidate-row-limit", type=int, default=100)
    ap.add_argument("--candidate-limit", type=int, default=100)
    ap.add_argument("--modes", default="source_plus_target_new,target_endpoint_singleton")
    ap.add_argument("--min-scores", default="0.0,0.80,0.86")
    ap.add_argument("--min-visuals", default="0.0,0.70,0.80")
    ap.add_argument("--min-poss", default="0.0,0.90")
    ap.add_argument("--min-scales", default="0.0,0.80")
    ap.add_argument("--min-support-edges", default="1,2")
    ap.add_argument("--max-source-sizes", default="1,2,4")
    ap.add_argument("--max-target-sizes", default="0")
    ap.add_argument("--top-ks", default="1,2,3,5")
    ap.add_argument("--max-edits-per-video", default="1,2,4")
    ap.add_argument("--sort-key", choices=["pair_f1", "selection_score"], default="pair_f1")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignment-offset", type=int, default=99_000_000)
    ap.add_argument("--assignments-out", default="")
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
    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    groups = _components(base_labels, keep_indices)
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    raw_candidates = _load_endpoint_candidates(args.endpoint_json, int(args.candidate_row_limit), int(args.candidate_limit))
    print(json.dumps({"stage": "base", "raw_candidates": len(raw_candidates), **base_pair}, sort_keys=True), flush=True)

    rows: list[dict[str, object]] = []
    labels_by_rank: dict[int, np.ndarray] = {}
    cache: dict[tuple[object, ...], tuple[np.ndarray, list[dict[str, object]], dict[str, int], list[dict[str, object]]]] = {}
    for mode in _parse_modes(args.modes):
        for min_score in _parse_floats(args.min_scores):
            for min_visual in _parse_floats(args.min_visuals):
                for min_pos in _parse_floats(args.min_poss):
                    for min_scale in _parse_floats(args.min_scales):
                        for min_support_edges in _parse_ints(args.min_support_edges):
                            for max_source_size in _parse_ints(args.max_source_sizes):
                                for max_target_size in _parse_ints(args.max_target_sizes):
                                    filtered = _filter_candidates(
                                        raw_candidates,
                                        min_score=float(min_score),
                                        min_visual=float(min_visual),
                                        min_pos=float(min_pos),
                                        min_scale=float(min_scale),
                                        min_support_edges=int(min_support_edges),
                                        max_source_size=int(max_source_size),
                                        max_target_size=int(max_target_size),
                                    )
                                    for top_k in _parse_ints(args.top_ks):
                                        for max_edits in _parse_ints(args.max_edits_per_video):
                                            key = (
                                                str(mode),
                                                float(min_score),
                                                float(min_visual),
                                                float(min_pos),
                                                float(min_scale),
                                                int(min_support_edges),
                                                int(max_source_size),
                                                int(max_target_size),
                                                int(top_k),
                                                int(max_edits),
                                            )
                                            labels, accepted, rejected = _apply_micro_surgery(
                                                records,
                                                base_labels,
                                                groups,
                                                keep_indices,
                                                seq_to_idx,
                                                filtered,
                                                mode=str(mode),
                                                top_k=int(top_k),
                                                max_edits_per_video=int(max_edits),
                                            )
                                            cache[key] = (labels, accepted, rejected, filtered)
                                            pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                                            pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                            mean_score = float(np.mean([float(row.get("score", 0.0)) for row in accepted])) if accepted else 0.0
                                            action_signature = _action_signature(accepted)
                                            rows.append(
                                                {
                                                    "mode": str(mode),
                                                    "min_score": float(min_score),
                                                    "min_visual": float(min_visual),
                                                    "min_pos": float(min_pos),
                                                    "min_scale": float(min_scale),
                                                    "min_support_edges": int(min_support_edges),
                                                    "max_source_size": int(max_source_size),
                                                    "max_target_size": int(max_target_size),
                                                    "top_k": int(top_k),
                                                    "max_edits_per_video": int(max_edits),
                                                    "raw_candidate_count": int(len(raw_candidates)),
                                                    "filtered_candidate_count": int(len(filtered)),
                                                    "accepted_surgeries": int(len(accepted)),
                                                    "moved_tracklets": int(sum(int(row.get("moved_tracklets", 0)) for row in accepted)),
                                                    "mean_accepted_score": float(mean_score),
                                                    "action_signature": action_signature,
                                                    "accepted_preview": accepted[:12],
                                                    **{f"rejected_{k}": int(v) for k, v in rejected.items()},
                                                    **pair,
                                                    "uses_anchors": False,
                                                    "uses_gt_for_training_or_anchors": False,
                                                    "uses_gt_for_evaluation_only": True,
                                                }
                                            )

    if str(args.sort_key) == "pair_f1":
        rows.sort(
            key=lambda row: (
                float(row["tracklet_pair_f1"]),
                float(row["tracklet_pair_precision"]),
                float(row["tracklet_pair_recall"]),
                float(row["mean_accepted_score"]),
            ),
            reverse=True,
        )
    else:
        rows.sort(
            key=lambda row: (
                float(row["mean_accepted_score"]),
                float(row["accepted_surgeries"]),
                float(row["tracklet_pair_f1"]),
            ),
            reverse=True,
        )

    seen_full_signatures: set[str] = set()
    full_scored_rows: list[dict[str, object]] = []
    full_rank = 0
    for row in rows:
        if full_rank >= max(int(args.full_top_n), 0):
            break
        signature = str(row.get("action_signature", ""))
        if signature in seen_full_signatures:
            continue
        seen_full_signatures.add(signature)
        full_rank += 1
        key = (
            str(row["mode"]),
            float(row["min_score"]),
            float(row["min_visual"]),
            float(row["min_pos"]),
            float(row["min_scale"]),
            int(row["min_support_edges"]),
            int(row["max_source_size"]),
            int(row["max_target_size"]),
            int(row["top_k"]),
            int(row["max_edits_per_video"]),
        )
        labels = cache[key][0]
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs))
        row.update({f"full_{name}": value for name, value in full.items() if name != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(full_rank)
        labels_by_rank[full_rank] = labels
        full_scored_rows.append(dict(row))
        print(json.dumps({"stage": "full", "rank": int(full_rank), "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            key = (
                str(row["mode"]),
                float(row["min_score"]),
                float(row["min_visual"]),
                float(row["min_pos"]),
                float(row["min_scale"]),
                int(row["min_support_edges"]),
                int(row["max_source_size"]),
                int(row["max_target_size"]),
                int(row["top_k"]),
                int(row["max_edits_per_video"]),
            )
            labels = cache[key][0]
        assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs=keep_seqs, offset=int(args.assignment_offset))
        rows[0].update(assignment_info)

    result = {
        "endpoint_json": str(args.endpoint_json),
        "assignment_csv": str(args.assignment_csv),
        "base_assignment_components": int(len(raw_to_local)),
        "base_pair_metrics": base_pair,
        "raw_candidate_count": int(len(raw_candidates)),
        "assignment_info": assignment_info,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "full_scored_rows": full_scored_rows,
        "top": rows[: max(100, int(args.full_top_n))],
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
