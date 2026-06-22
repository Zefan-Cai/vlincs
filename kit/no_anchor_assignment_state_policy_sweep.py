#!/usr/bin/env python
"""Stateful policy sweep for an existing no-anchor global-ID assignment.

This script treats the input assignment as a forced-delivery identity output and
adds an evidence-calibrated state layer over its components.  It does not use
anchors or ground-truth IDs to build states.  Component states come from
tracklet metadata, detection quality, and same-stream temporal cannot-link
conflicts.  Ground truth is used only after prediction for pair/full metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, KIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_split_sweep import _labels_from_assignment, _load_assignment_labels
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
        _tracklet_quality_score,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_split_sweep import _labels_from_assignment, _load_assignment_labels
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
        _tracklet_quality_score,
        _with_detection_endpoints,
    )


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _parse_strings(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _admission_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        output_min_dets=int(args.output_min_dets),
        output_min_conf=float(args.output_min_conf),
        output_min_area=float(args.output_min_area),
        output_min_quality=float(args.output_min_quality),
        output_min_area_by_video=str(args.output_min_area_by_video),
        output_drop_area_quantile=float(args.output_drop_area_quantile),
        output_drop_area_quantile_by_video=str(args.output_drop_area_quantile_by_video),
        output_drop_quality_quantile=float(args.output_drop_quality_quantile),
        output_drop_quality_quantile_by_video=str(args.output_drop_quality_quantile_by_video),
        output_auto_anomaly_admission=bool(args.output_auto_anomaly_admission),
        output_auto_anomaly_metric=str(args.output_auto_anomaly_metric),
        output_auto_anomaly_quantile=float(args.output_auto_anomaly_quantile),
        output_auto_anomaly_area_ratio=float(args.output_auto_anomaly_area_ratio),
        output_auto_anomaly_quality_mad=float(args.output_auto_anomaly_quality_mad),
        output_auto_anomaly_min_video_tracklets=int(args.output_auto_anomaly_min_video_tracklets),
        output_auto_anomaly_max_videos=int(args.output_auto_anomaly_max_videos),
    )


def _component_members(labels: np.ndarray, keep_indices: set[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        out[int(labels[idx])].append(int(idx))
    return dict(out)


def _component_stats(records, labels: np.ndarray, keep_indices: set[int]) -> dict[int, dict[str, object]]:
    forbidden = _build_overlap_forbidden(records)
    members = _component_members(labels, keep_indices)
    stats: dict[int, dict[str, object]] = {}
    for label, indices in members.items():
        n = len(indices)
        pair_count = n * (n - 1) // 2
        conflict_pairs = 0
        conflict_edges: list[tuple[int, int]] = []
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                if int(j) in forbidden[int(i)]:
                    conflict_pairs += 1
                    conflict_edges.append((int(i), int(j)))
        qualities = [_tracklet_quality_score(records[idx]) for idx in indices]
        areas = [float(records[idx].width) * float(records[idx].height) for idx in indices]
        confs = [float(records[idx].avg_conf) for idx in indices]
        dets = [int(records[idx].n_dets) for idx in indices]
        videos = sorted({records[idx].video for idx in indices})
        cameras = sorted({records[idx].camera for idx in indices})
        stats[int(label)] = {
            "component_label": int(label),
            "size": int(n),
            "pair_count": int(pair_count),
            "conflict_pairs": int(conflict_pairs),
            "conflict_rate": round(float(conflict_pairs / max(pair_count, 1)), 8),
            "videos": videos,
            "n_videos": int(len(videos)),
            "cameras": cameras,
            "n_cameras": int(len(cameras)),
            "total_dets": int(sum(dets)),
            "mean_dets": round(float(np.mean(dets)) if dets else 0.0, 6),
            "min_dets": int(min(dets) if dets else 0),
            "mean_conf": round(float(np.mean(confs)) if confs else 0.0, 6),
            "min_conf": round(float(min(confs)) if confs else 0.0, 6),
            "mean_area": round(float(np.mean(areas)) if areas else 0.0, 3),
            "min_area": round(float(min(areas)) if areas else 0.0, 3),
            "mean_quality": round(float(np.mean(qualities)) if qualities else 0.0, 6),
            "min_quality": round(float(min(qualities)) if qualities else 0.0, 6),
            "start_frame": int(min(records[idx].start_frame for idx in indices)),
            "end_frame": int(max(records[idx].end_frame for idx in indices)),
            "conflict_edges": conflict_edges,
            "indices": indices,
            "seqs": [int(records[idx].seq) for idx in indices],
        }
    return stats


def _assign_states(
    stats: dict[int, dict[str, object]],
    *,
    committed_min_size: int,
    pending_max_size: int,
    conflict_rate_threshold: float,
    min_quality_quantile: float,
) -> dict[int, str]:
    quality_values = [float(item["mean_quality"]) for item in stats.values()]
    quality_threshold = (
        float(np.quantile(np.asarray(quality_values, dtype=np.float32), float(min_quality_quantile)))
        if quality_values and float(min_quality_quantile) > 0.0
        else -1.0e18
    )
    out: dict[int, str] = {}
    for label, item in stats.items():
        size = int(item["size"])
        conflict_rate = float(item["conflict_rate"])
        mean_quality = float(item["mean_quality"])
        if size <= int(pending_max_size):
            state = "pending"
        elif conflict_rate > float(conflict_rate_threshold):
            state = "forced_conflict"
        elif size >= int(committed_min_size) and mean_quality >= quality_threshold and int(item["conflict_pairs"]) == 0:
            state = "committed"
        else:
            state = "provisional"
        out[int(label)] = state
    return out


def _color_conflict_groups(indices: list[int], conflict_edges: list[tuple[int, int]]) -> list[list[int]]:
    if not conflict_edges:
        return [list(indices)]
    adjacency: dict[int, set[int]] = {int(idx): set() for idx in indices}
    for left, right in conflict_edges:
        left = int(left)
        right = int(right)
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    color_by_idx: dict[int, int] = {}
    ordered = sorted(indices, key=lambda idx: (-len(adjacency.get(int(idx), set())), int(idx)))
    for idx in ordered:
        used = {color_by_idx[nbr] for nbr in adjacency.get(int(idx), set()) if nbr in color_by_idx}
        color = 0
        while color in used:
            color += 1
        color_by_idx[int(idx)] = color
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(indices):
        groups[int(color_by_idx[int(idx)])].append(int(idx))
    return [group for _color, group in sorted(groups.items())]


def _labels_for_policy(
    records,
    base_labels: np.ndarray,
    keep_indices: set[int],
    stats: dict[int, dict[str, object]],
    states: dict[int, str],
    policy: str,
) -> tuple[np.ndarray, set[int], dict[str, object]]:
    out = np.full(len(base_labels), -1, dtype=np.int64)
    keep_out: set[int] = set()
    next_label = 0
    singleton_states = set()
    drop_states = set()
    color_states = set()
    if policy == "keep_all":
        pass
    elif policy == "singleton_forced":
        singleton_states = {"forced_conflict"}
    elif policy == "color_forced":
        color_states = {"forced_conflict"}
    elif policy == "drop_forced":
        drop_states = {"forced_conflict"}
    elif policy == "singleton_pending_forced":
        singleton_states = {"pending", "forced_conflict"}
    elif policy == "color_pending_forced":
        color_states = {"pending", "forced_conflict"}
    elif policy == "drop_pending_forced":
        drop_states = {"pending", "forced_conflict"}
    elif policy == "singleton_pending":
        singleton_states = {"pending"}
    elif policy == "color_pending":
        color_states = {"pending"}
    elif policy == "drop_pending":
        drop_states = {"pending"}
    else:
        raise ValueError(f"unknown policy: {policy}")

    policy_info = Counter()
    for label, item in sorted(stats.items()):
        state = states[int(label)]
        indices = [idx for idx in item["indices"] if int(idx) in keep_indices]
        if state in drop_states:
            policy_info[f"dropped_{state}"] += len(indices)
            continue
        if state in singleton_states:
            for idx in indices:
                out[int(idx)] = next_label
                next_label += 1
                keep_out.add(int(records[int(idx)].seq))
                policy_info[f"singleton_{state}"] += len(indices)
            continue
        if state in color_states:
            groups = _color_conflict_groups(indices, list(item.get("conflict_edges", [])))
            for group in groups:
                for idx in group:
                    out[int(idx)] = next_label
                    keep_out.add(int(records[int(idx)].seq))
                next_label += 1
            policy_info[f"colored_{state}"] += len(indices)
            policy_info[f"color_parts_{state}"] += len(groups)
            continue
        for idx in indices:
            out[int(idx)] = next_label
            keep_out.add(int(records[int(idx)].seq))
        next_label += 1
        policy_info[f"kept_{state}"] += len(indices)

    for idx in range(len(base_labels)):
        if out[idx] < 0:
            out[idx] = next_label
            next_label += 1
    kept_labels = [int(out[idx]) for idx in range(len(out)) if int(records[idx].seq) in keep_out]
    return out, keep_out, {
        "policy": policy,
        "output_tracklets": int(len(keep_out)),
        "components": int(len(set(kept_labels))),
        "largest_component": int(max(Counter(kept_labels).values(), default=0)),
        "policy_counts": dict(policy_info),
    }


def _write_component_states(path: str, stats: dict[int, dict[str, object]], states: dict[int, str]) -> dict[str, object]:
    rows = []
    for label, item in sorted(stats.items()):
        row = {
            key: value
            for key, value in item.items()
            if key not in {"indices", "seqs", "videos", "cameras", "conflict_edges"}
        }
        row["state"] = states[int(label)]
        row["videos"] = "|".join(item["videos"])
        row["cameras"] = "|".join(item["cameras"])
        row["seqs"] = "|".join(str(seq) for seq in item["seqs"])
        rows.append(row)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"component_states_out": str(out), "component_state_rows": int(len(rows))}


def _write_assignments(
    path: str,
    records,
    labels: np.ndarray,
    keep_seqs: set[int],
    states_by_base_label: dict[int, str],
    base_labels: np.ndarray,
    *,
    offset: int,
) -> dict[str, object]:
    rows = []
    comp_sizes = Counter(int(labels[idx]) for idx, record in enumerate(records) if int(record.seq) in keep_seqs)
    for idx, record in enumerate(records):
        seq = int(record.seq)
        if seq not in keep_seqs:
            continue
        base_label = int(base_labels[idx])
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
                "component_label": int(labels[idx]),
                "component_size": int(comp_sizes[int(labels[idx])]),
                "base_component_label": base_label,
                "decision_status": str(states_by_base_label.get(base_label, "unknown")),
                "resolution_status": str(states_by_base_label.get(base_label, "unknown")),
            }
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["seq", "predicted_global_id"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"assignments_out": str(out), "assignment_rows": int(len(rows))}


def _sort_key(row: dict[str, object], key: str) -> tuple[float, float, float, int]:
    return (
        float(row.get(key, 0.0)),
        float(row.get("tracklet_pair_f1", 0.0)),
        float(row.get("tracklet_pair_precision", 0.0)),
        int(row.get("output_tracklets", 0)),
    )


def _self_test() -> None:
    from kit.no_anchor_resolve_sweep import TrackletRecord

    def rec(seq: int, start: int, end: int, *, video: str = "v", camera: str = "c", n: int = 5, conf: float = 0.9):
        return TrackletRecord(
            seq=seq,
            tracklet_key=f"t{seq}",
            video=video,
            camera=camera,
            start_frame=start,
            end_frame=end,
            start_abs_ms=start,
            end_abs_ms=end,
            n_dets=n,
            avg_conf=conf,
            cx=0.0,
            cy=0.0,
            width=20.0,
            height=50.0,
            first_cx=0.0,
            first_cy=0.0,
            first_width=20.0,
            first_height=50.0,
            last_cx=0.0,
            last_cy=0.0,
            last_width=20.0,
            last_height=50.0,
        )

    records = [
        rec(1, 0, 10),
        rec(2, 5, 15),
        rec(3, 20, 30),
        rec(4, 40, 50, video="v2"),
        rec(5, 60, 70, n=1, conf=0.2),
    ]
    base_labels = np.asarray([0, 0, 0, 1, 2], dtype=np.int64)
    keep_indices = set(range(len(records)))
    stats = _component_stats(records, base_labels, keep_indices)
    states = _assign_states(
        stats,
        committed_min_size=2,
        pending_max_size=1,
        conflict_rate_threshold=0.0,
        min_quality_quantile=0.0,
    )
    labels, keep, info = _labels_for_policy(records, base_labels, keep_indices, stats, states, "singleton_forced")
    color_labels, color_keep, color_info = _labels_for_policy(
        records, base_labels, keep_indices, stats, states, "color_forced"
    )
    assert stats[0]["conflict_pairs"] == 1, stats
    assert stats[0]["conflict_edges"] == [(0, 1)], stats
    assert states[0] == "forced_conflict", states
    assert states[1] == "pending", states
    assert len(set(labels[:3].tolist())) == 3, labels.tolist()
    assert sorted(keep) == [1, 2, 3, 4, 5], keep
    assert info["largest_component"] == 1, info
    assert len(set(color_labels[:3].tolist())) == 2, color_labels.tolist()
    assert sorted(color_keep) == [1, 2, 3, 4, 5], color_keep
    print(
        json.dumps(
            {
                "stage": "self_test",
                "status": "ok",
                "states": states,
                "singleton_policy_info": info,
                "color_policy_info": color_info,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", default="")
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--committed-min-sizes", default="4,8,16,32")
    ap.add_argument("--pending-max-sizes", default="0,1,2,4")
    ap.add_argument("--conflict-rate-thresholds", default="0,0.0005,0.001,0.003,0.01")
    ap.add_argument("--min-quality-quantiles", default="0")
    ap.add_argument(
        "--policies",
        default="keep_all,color_forced,singleton_forced,drop_forced,color_pending_forced,singleton_pending_forced,drop_pending_forced",
    )
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
    ap.add_argument("--output-auto-anomaly-metric", default="quality")
    ap.add_argument("--output-auto-anomaly-quantile", type=float, default=0.75)
    ap.add_argument("--output-auto-anomaly-area-ratio", type=float, default=0.60)
    ap.add_argument("--output-auto-anomaly-quality-mad", type=float, default=1.0)
    ap.add_argument("--output-auto-anomaly-min-video-tracklets", type=int, default=20)
    ap.add_argument("--output-auto-anomaly-max-videos", type=int, default=3)
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--json-top-n", type=int, default=100)
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--assignment-offset", type=int, default=80_000_000)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--component-states-out", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.assignment_csv:
        raise SystemExit("--assignment-csv is required unless --self-test is set")
    if not args.json:
        raise SystemExit("--json is required unless --self-test is set")

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
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

    keep_seqs, output_info = _output_keep_seqs(records, _admission_args(args))
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    base_labels, keep_indices, raw_to_local = _labels_from_assignment(records, pred_input)
    keep_indices = {idx for idx in keep_indices if int(records[idx].seq) in keep_seqs}
    stats = _component_stats(records, base_labels, keep_indices)
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(
        json.dumps(
            {
                "stage": "base",
                "components": len(stats),
                "raw_components": len(raw_to_local),
                "output_tracklets": len(keep_seqs),
                **base_pair,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rows: list[dict[str, object]] = []
    cache: dict[tuple[int, int, float, float], dict[int, str]] = {}
    labels_for_rank: dict[int, tuple[np.ndarray, set[int], dict[int, str]]] = {}
    for committed_min_size in _parse_ints(args.committed_min_sizes):
        for pending_max_size in _parse_ints(args.pending_max_sizes):
            for conflict_rate_threshold in _parse_floats(args.conflict_rate_thresholds):
                for min_quality_quantile in _parse_floats(args.min_quality_quantiles):
                    state_key = (
                        int(committed_min_size),
                        int(pending_max_size),
                        float(conflict_rate_threshold),
                        float(min_quality_quantile),
                    )
                    states = cache.get(state_key)
                    if states is None:
                        states = _assign_states(
                            stats,
                            committed_min_size=int(committed_min_size),
                            pending_max_size=int(pending_max_size),
                            conflict_rate_threshold=float(conflict_rate_threshold),
                            min_quality_quantile=float(min_quality_quantile),
                        )
                        cache[state_key] = states
                    state_counts = Counter(states.values())
                    for policy in _parse_strings(args.policies):
                        labels, policy_keep, policy_info = _labels_for_policy(
                            records,
                            base_labels,
                            keep_indices,
                            stats,
                            states,
                            str(policy),
                        )
                        pred = _labels_to_seq_map(
                            records,
                            labels,
                            offset=int(args.assignment_offset),
                            keep_seqs=policy_keep,
                        )
                        pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                        row = {
                            "mode": "assignment_state_policy",
                            "committed_min_size": int(committed_min_size),
                            "pending_max_size": int(pending_max_size),
                            "conflict_rate_threshold": float(conflict_rate_threshold),
                            "min_quality_quantile": float(min_quality_quantile),
                            "state_counts": dict(state_counts),
                            **policy_info,
                            **pair,
                            "uses_anchors": False,
                            "uses_gt_for_training_or_anchors": False,
                            "uses_gt_for_evaluation_only": True,
                        }
                        row["coverage_ratio"] = float(row["output_tracklets"]) / max(len(records), 1)
                        row["coverage_pair_score"] = float(row["tracklet_pair_f1"]) * float(row["coverage_ratio"])
                        row["min_precision_recall"] = min(
                            float(row["tracklet_pair_precision"]),
                            float(row["tracklet_pair_recall"]),
                        )
                        rows.append(row)

    rows.sort(key=lambda row: _sort_key(row, str(args.sort_key)), reverse=True)
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        state_key = (
            int(row["committed_min_size"]),
            int(row["pending_max_size"]),
            float(row["conflict_rate_threshold"]),
            float(row["min_quality_quantile"]),
        )
        states = cache[state_key]
        labels, policy_keep, _policy_info = _labels_for_policy(
            records,
            base_labels,
            keep_indices,
            stats,
            states,
            str(row["policy"]),
        )
        labels_for_rank[rank] = (labels, policy_keep, states)
        full = _score_full(
            pred_by_video,
            gt_by_video,
            _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=policy_keep),
        )
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "full": full, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    component_info = None
    if rows:
        best = rows[0]
        state_key = (
            int(best["committed_min_size"]),
            int(best["pending_max_size"]),
            float(best["conflict_rate_threshold"]),
            float(best["min_quality_quantile"]),
        )
        states = cache[state_key]
        labels, policy_keep, _policy_info = labels_for_rank.get(
            1,
            _labels_for_policy(records, base_labels, keep_indices, stats, states, str(best["policy"])),
        )
        if args.assignments_out:
            assignment_info = _write_assignments(
                args.assignments_out,
                records,
                labels,
                policy_keep,
                states,
                base_labels,
                offset=int(args.assignment_offset),
            )
            rows[0].update(assignment_info)
        if args.component_states_out:
            component_info = _write_component_states(args.component_states_out, stats, states)
            rows[0].update(component_info)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "assignment_csv": str(args.assignment_csv),
        "pred_col": str(args.pred_col),
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "assignment_info": assignment_info,
        "component_info": component_info,
        "n_rows": int(len(rows)),
        "sort_key": str(args.sort_key),
        "top": rows[: max(int(args.json_top_n), int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "done", "json": str(out), "best": rows[0] if rows else None}, sort_keys=True))


if __name__ == "__main__":
    main()
