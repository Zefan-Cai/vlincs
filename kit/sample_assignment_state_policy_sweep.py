#!/usr/bin/env python
"""Offline state-policy sweep for no-anchor sample/parquet assignments.

This is the local sample counterpart of ``no_anchor_assignment_state_policy_sweep``.
It does not use anchors.  Ground-truth columns in the sample parquet are used
only for post-hoc evaluation.  The optional HOTA/IDF1 score is a
``sample_parquet_gt`` diagnostic: reference rows are built from the same
detection boxes plus their parquet ``gt_id`` labels, so it is an identity and
admission proxy rather than the DS1 leaderboard score.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.evaluate_sample_assignments_full import _build_comp, _metric_dict
from kit.sample_assignment_admission_grid import _pair_metrics
from vlincs_gallery.eval.score import evaluate


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_strings(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_parquets(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        df["_source_parquet"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("at least one --tracklet-parquet is required")
    df = pd.concat(frames, ignore_index=True)
    rename = {"video_key": "video", "frame_idx": "frame", "score": "confidence", "coco_cls": "object_type"}
    df = df.rename(columns={old: new for old, new in rename.items() if old in df.columns})
    required = {"video", "frame", "tracklet_key", "x1", "y1", "x2", "y2"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"tracklet parquet is missing columns: {missing}")
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    if "object_type" not in df.columns:
        df["object_type"] = 0
    return df


def _camera_from_video(video: str) -> str:
    match = re.search(r"MCAM\d+", str(video))
    return match.group(0) if match else str(video)


def _tracklet_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["area"] = (work["x2"] - work["x1"]).clip(lower=0) * (work["y2"] - work["y1"]).clip(lower=0)
    agg = {
        "video": ("video", "first"),
        "n_dets": ("frame", "size"),
        "start_frame": ("frame", "min"),
        "end_frame": ("frame", "max"),
        "avg_conf": ("confidence", "mean"),
        "area_median": ("area", "median"),
    }
    if "tracklet_majority_gt_id" in work.columns:
        agg["gt_id"] = ("tracklet_majority_gt_id", "first")
    elif "gt_id" in work.columns:
        agg["gt_id"] = ("gt_id", lambda col: str(col.astype(str).value_counts().idxmax()))
    if "tracklet_majority_gt_fraction" in work.columns:
        agg["gt_fraction"] = ("tracklet_majority_gt_fraction", "first")
    table = work.groupby("tracklet_key", sort=False).agg(**agg).reset_index()
    if "gt_id" not in table.columns:
        table["gt_id"] = ""
    if "gt_fraction" not in table.columns:
        table["gt_fraction"] = 1.0
    table = table.sort_values(["video", "start_frame", "end_frame", "tracklet_key"], kind="mergesort").reset_index(drop=True)
    table["seq"] = np.arange(len(table), dtype=np.int64)
    table["camera"] = table["video"].map(_camera_from_video)
    table["quality"] = (
        0.45 * np.clip(table["n_dets"].to_numpy(np.float32) / 80.0, 0.0, 1.0)
        + 0.25 * np.clip(table["avg_conf"].to_numpy(np.float32) / 0.70, 0.0, 1.0)
        + 0.30 * np.clip(table["area_median"].to_numpy(np.float32) / 12000.0, 0.0, 1.0)
    )
    return table


def _load_assignments(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tracklet_key", "predicted_global_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"assignment CSV is missing columns: {missing}")
    df["tracklet_key"] = df["tracklet_key"].astype(str)
    df["predicted_global_id"] = df["predicted_global_id"].astype(np.int64)
    return df.drop_duplicates("tracklet_key", keep="first")


def _gt_mapping(table: pd.DataFrame, *, min_gt_fraction: float, min_rows: int) -> tuple[dict[int, int], dict[int, float], dict[str, object]]:
    mapping: dict[str, int] = {}
    gt_by_seq: dict[int, int] = {}
    weight_by_seq: dict[int, float] = {}
    rejected = 0
    for row in table.itertuples(index=False):
        gt_id = str(row.gt_id)
        if gt_id in {"", "nan", "None", "-1"} or float(row.gt_fraction) < min_gt_fraction or int(row.n_dets) < min_rows:
            rejected += 1
            continue
        mapping.setdefault(gt_id, len(mapping) + 1)
        gt_by_seq[int(row.seq)] = int(mapping[gt_id])
        weight_by_seq[int(row.seq)] = float(row.n_dets)
    return gt_by_seq, weight_by_seq, {
        "eval_labeled_tracklets": int(len(gt_by_seq)),
        "eval_rejected_tracklets": int(rejected),
        "eval_unique_gt_ids": int(len(set(gt_by_seq.values()))),
        "eval_min_gt_fraction": float(min_gt_fraction),
        "eval_min_rows": int(min_rows),
    }


def _base_labels(table: pd.DataFrame, assignments: pd.DataFrame) -> tuple[np.ndarray, set[int], dict[int, int]]:
    by_key = assignments.set_index("tracklet_key")["predicted_global_id"].to_dict()
    labels = np.full(len(table), -1, dtype=np.int64)
    raw_to_local: dict[int, int] = {}
    keep_indices: set[int] = set()
    for idx, row in enumerate(table.itertuples(index=False)):
        raw = by_key.get(str(row.tracklet_key))
        if raw is None:
            continue
        raw = int(raw)
        if raw not in raw_to_local:
            raw_to_local[raw] = len(raw_to_local)
        labels[idx] = raw_to_local[raw]
        keep_indices.add(int(idx))
    return labels, keep_indices, raw_to_local


def _build_overlap_forbidden(table: pd.DataFrame) -> dict[int, set[int]]:
    forbidden: dict[int, set[int]] = defaultdict(set)
    for _video, group in table.reset_index().groupby("video", sort=False):
        rows = sorted(group.itertuples(index=False), key=lambda row: (int(row.start_frame), int(row.end_frame)))
        for pos, left in enumerate(rows):
            for right in rows[pos + 1 :]:
                if int(right.start_frame) > int(left.end_frame):
                    break
                if int(left.start_frame) <= int(right.end_frame) and int(right.start_frame) <= int(left.end_frame):
                    forbidden[int(left.index)].add(int(right.index))
                    forbidden[int(right.index)].add(int(left.index))
    return forbidden


def _component_stats(table: pd.DataFrame, labels: np.ndarray, keep_indices: set[int]) -> dict[int, dict[str, object]]:
    forbidden = _build_overlap_forbidden(table)
    members: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        members[int(labels[idx])].append(int(idx))
    stats: dict[int, dict[str, object]] = {}
    for label, indices in members.items():
        pair_count = len(indices) * (len(indices) - 1) // 2
        conflict_edges: list[tuple[int, int]] = []
        for pos, left in enumerate(indices):
            for right in indices[pos + 1 :]:
                if int(right) in forbidden[int(left)]:
                    conflict_edges.append((int(left), int(right)))
        rows = table.iloc[indices]
        stats[int(label)] = {
            "component_label": int(label),
            "size": int(len(indices)),
            "pair_count": int(pair_count),
            "conflict_pairs": int(len(conflict_edges)),
            "conflict_rate": float(len(conflict_edges) / max(pair_count, 1)),
            "videos": sorted(rows["video"].astype(str).unique().tolist()),
            "n_videos": int(rows["video"].nunique()),
            "cameras": sorted(rows["camera"].astype(str).unique().tolist()),
            "n_cameras": int(rows["camera"].nunique()),
            "mean_conf": float(rows["avg_conf"].mean()),
            "min_conf": float(rows["avg_conf"].min()),
            "mean_area": float(rows["area_median"].mean()),
            "min_area": float(rows["area_median"].min()),
            "mean_quality": float(rows["quality"].mean()),
            "min_quality": float(rows["quality"].min()),
            "total_dets": int(rows["n_dets"].sum()),
            "mean_dets": float(rows["n_dets"].mean()),
            "min_dets": int(rows["n_dets"].min()),
            "start_frame": int(rows["start_frame"].min()),
            "end_frame": int(rows["end_frame"].max()),
            "conflict_edges": conflict_edges,
            "indices": indices,
            "tracklet_keys": rows["tracklet_key"].astype(str).tolist(),
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
    qualities = [float(item["mean_quality"]) for item in stats.values()]
    quality_threshold = (
        float(np.quantile(np.asarray(qualities, dtype=np.float32), float(min_quality_quantile)))
        if qualities and float(min_quality_quantile) > 0.0
        else -1.0e18
    )
    states: dict[int, str] = {}
    for label, item in stats.items():
        if int(item["size"]) <= int(pending_max_size):
            state = "pending"
        elif float(item["conflict_rate"]) > float(conflict_rate_threshold):
            state = "forced_conflict"
        elif int(item["size"]) >= int(committed_min_size) and float(item["mean_quality"]) >= quality_threshold and int(item["conflict_pairs"]) == 0:
            state = "committed"
        else:
            state = "provisional"
        states[int(label)] = state
    return states


def _color_conflict_groups(indices: list[int], conflict_edges: list[tuple[int, int]]) -> list[list[int]]:
    if not conflict_edges:
        return [list(indices)]
    adjacency: dict[int, set[int]] = {int(idx): set() for idx in indices}
    for left, right in conflict_edges:
        if int(left) in adjacency and int(right) in adjacency:
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    color_by_idx: dict[int, int] = {}
    for idx in sorted(indices, key=lambda item: (-len(adjacency.get(int(item), set())), int(item))):
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
    labels: np.ndarray,
    keep_indices: set[int],
    stats: dict[int, dict[str, object]],
    states: dict[int, str],
    policy: str,
) -> tuple[np.ndarray, set[int], dict[str, object]]:
    out = np.full(len(labels), -1, dtype=np.int64)
    keep_out: set[int] = set()
    next_label = 0
    singleton_states: set[str] = set()
    drop_states: set[str] = set()
    color_states: set[str] = set()
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
    elif policy == "drop_pending":
        drop_states = {"pending"}
    else:
        raise ValueError(f"unknown policy: {policy}")

    policy_counts = Counter()
    for label, item in sorted(stats.items()):
        state = states[int(label)]
        indices = [int(idx) for idx in item["indices"] if int(idx) in keep_indices]
        if state in drop_states:
            policy_counts[f"dropped_{state}"] += len(indices)
            continue
        if state in singleton_states:
            for idx in indices:
                out[idx] = next_label
                keep_out.add(idx)
                next_label += 1
            policy_counts[f"singleton_{state}"] += len(indices)
            continue
        if state in color_states:
            groups = _color_conflict_groups(indices, list(item.get("conflict_edges", [])))
            for group in groups:
                for idx in group:
                    out[idx] = next_label
                    keep_out.add(idx)
                next_label += 1
            policy_counts[f"colored_{state}"] += len(indices)
            policy_counts[f"color_parts_{state}"] += len(groups)
            continue
        for idx in indices:
            out[idx] = next_label
            keep_out.add(idx)
        next_label += 1
        policy_counts[f"kept_{state}"] += len(indices)
    for idx in range(len(out)):
        if out[idx] < 0:
            out[idx] = next_label
            next_label += 1
    kept_labels = [int(out[idx]) for idx in keep_out]
    return out, keep_out, {
        "policy": policy,
        "output_tracklets": int(len(keep_out)),
        "components": int(len(set(kept_labels))),
        "largest_component": int(max(Counter(kept_labels).values(), default=0)),
        "policy_counts": dict(policy_counts),
    }


def _pred_by_seq(table: pd.DataFrame, labels: np.ndarray, keep_indices: set[int], offset: int) -> dict[int, int]:
    return {int(table.iloc[idx].seq): int(offset) + int(labels[idx]) for idx in keep_indices}


def _assignments_from_labels(
    table: pd.DataFrame,
    labels: np.ndarray,
    keep_indices: set[int],
    states: dict[int, str],
    base_labels: np.ndarray,
    *,
    offset: int,
) -> pd.DataFrame:
    rows = []
    comp_sizes = Counter(int(labels[idx]) for idx in keep_indices)
    for idx in sorted(keep_indices):
        row = table.iloc[idx]
        base_label = int(base_labels[idx])
        rows.append(
            {
                "tracklet_key": str(row.tracklet_key),
                "video": str(row.video),
                "camera": str(row.camera),
                "start_frame": int(row.start_frame),
                "end_frame": int(row.end_frame),
                "n_dets": int(row.n_dets),
                "avg_conf": round(float(row.avg_conf), 6),
                "predicted_global_id": int(offset) + int(labels[idx]),
                "component_label": int(labels[idx]),
                "component_size": int(comp_sizes[int(labels[idx])]),
                "base_component_label": base_label,
                "decision_status": str(states.get(base_label, "unknown")),
                "resolution_status": str(states.get(base_label, "unknown")),
            }
        )
    return pd.DataFrame(rows)


def _sample_parquet_gt_score(df: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, object]:
    def dedupe_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        if frame.empty:
            return frame, 0
        work = frame.copy()
        work["_area_for_dedupe"] = (work["x2"] - work["x1"]).clip(lower=0) * (work["y2"] - work["y1"]).clip(lower=0)
        before = len(work)
        work = (
            work.sort_values(["video", "frame", "id", "confidence", "_area_for_dedupe"], ascending=[True, True, True, False, False])
            .drop_duplicates(["video", "frame", "id"], keep="first")
            .drop(columns=["_area_for_dedupe"])
        )
        return work, int(before - len(work))

    valid = ~df["gt_id"].astype(str).isin(["", "nan", "None", "-1"])
    ref = df.loc[valid].copy()
    if ref.empty:
        return {"sample_full_error": "no valid parquet gt_id rows"}
    gt_map = {key: idx + 1 for idx, key in enumerate(sorted(ref["gt_id"].astype(str).unique()))}
    ref["id"] = ref["gt_id"].astype(str).map(gt_map).astype(np.int64)
    ref, ref_deduped_rows = dedupe_rows(ref)
    ref_comp = {}
    for video, group in ref.groupby("video", sort=True):
        tmp = group[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
        ref_comp[str(video)] = tmp
    pred = df.merge(assignments[["tracklet_key", "predicted_global_id"]], on="tracklet_key", how="inner")
    pred = pred.loc[~pred["gt_id"].astype(str).isin(["", "nan", "None", "-1"])].copy()
    pred["predicted_global_id"] = pred["predicted_global_id"].astype(np.int64)
    pred = pred.rename(columns={"predicted_global_id": "id"})
    pred, pred_deduped_rows = dedupe_rows(pred)
    pred = pred.rename(columns={"id": "predicted_global_id"})
    comp = _build_comp(pred)
    keys = sorted(set(ref_comp).intersection(comp))
    if not keys:
        return {"sample_full_error": "no overlapping videos after assignment filtering"}
    try:
        metrics = evaluate({key: ref_comp[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    except ModuleNotFoundError as exc:
        if exc.name == "reid_hota":
            return {
                "sample_full_error": "missing reid_hota",
                "sample_full_videos_scored": keys,
                "sample_full_reference": "parquet_gt_same_detection_boxes",
                "sample_full_scored_rows": int(len(pred)),
                "sample_full_ref_rows": int(sum(len(ref_comp[key]) for key in keys)),
            }
        raise
    return {
        **{f"sample_full_{key}": value for key, value in _metric_dict(metrics).items() if key != "per_video"},
        "sample_full_per_video": _metric_dict(metrics)["per_video"],
        "sample_full_videos_scored": keys,
        "sample_full_reference": "parquet_gt_same_detection_boxes",
        "sample_full_scored_rows": int(len(pred)),
        "sample_full_ref_rows": int(sum(len(ref_comp[key]) for key in keys)),
        "sample_full_ref_deduped_rows": int(ref_deduped_rows),
        "sample_full_pred_deduped_rows": int(pred_deduped_rows),
    }


def _write_component_states(path: str, stats: dict[int, dict[str, object]], states: dict[int, str]) -> dict[str, object]:
    rows = []
    for label, item in sorted(stats.items()):
        row = {
            key: value
            for key, value in item.items()
            if key not in {"indices", "tracklet_keys", "videos", "cameras", "conflict_edges"}
        }
        row["state"] = states[int(label)]
        row["videos"] = "|".join(item["videos"])
        row["cameras"] = "|".join(item["cameras"])
        row["tracklet_keys"] = "|".join(item["tracklet_keys"])
        rows.append(row)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"component_states_out": str(out), "component_state_rows": int(len(rows))}


def _sort_tuple(row: dict[str, object], key: str) -> tuple[float, float, float, int]:
    return (
        float(row.get(key, 0.0)),
        float(row.get("tracklet_pair_f1", 0.0)),
        float(row.get("tracklet_pair_precision", 0.0)),
        int(row.get("output_tracklets", 0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracklet-parquet", nargs="+", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--committed-min-sizes", default="4,8,16,32")
    ap.add_argument("--pending-max-sizes", default="0,1,2,4")
    ap.add_argument("--conflict-rate-thresholds", default="0,0.0005,0.001,0.003,0.01")
    ap.add_argument("--min-quality-quantiles", default="0")
    ap.add_argument(
        "--policies",
        default="keep_all,color_forced,singleton_forced,drop_forced,color_pending_forced,singleton_pending_forced,drop_pending_forced",
    )
    ap.add_argument("--eval-min-gt-fraction", type=float, default=0.5)
    ap.add_argument("--eval-min-rows", type=int, default=1)
    ap.add_argument("--assignment-offset", type=int, default=80_000_000)
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--json-top-n", type=int, default=100)
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--component-states-out", default="")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    df = _load_parquets(args.tracklet_parquet)
    table = _tracklet_table(df)
    assignments = _load_assignments(args.assignments)
    base_labels, keep_indices, raw_to_local = _base_labels(table, assignments)
    stats = _component_stats(table, base_labels, keep_indices)
    gt_by_seq, weight_by_seq, eval_info = _gt_mapping(
        table,
        min_gt_fraction=float(args.eval_min_gt_fraction),
        min_rows=int(args.eval_min_rows),
    )
    seqs = [int(row.seq) for row in table.itertuples(index=False)]
    base_pair = _pair_metrics(seqs, _pred_by_seq(table, base_labels, keep_indices, int(args.assignment_offset)), gt_by_seq, weight_by_seq)

    rows: list[dict[str, object]] = []
    state_cache: dict[tuple[int, int, float, float], dict[int, str]] = {}
    labels_for_rank: dict[int, tuple[np.ndarray, set[int], dict[int, str]]] = {}
    for committed_min_size in _parse_ints(args.committed_min_sizes):
        for pending_max_size in _parse_ints(args.pending_max_sizes):
            for conflict_rate_threshold in _parse_floats(args.conflict_rate_thresholds):
                for min_quality_quantile in _parse_floats(args.min_quality_quantiles):
                    key = (int(committed_min_size), int(pending_max_size), float(conflict_rate_threshold), float(min_quality_quantile))
                    states = state_cache.get(key)
                    if states is None:
                        states = _assign_states(
                            stats,
                            committed_min_size=key[0],
                            pending_max_size=key[1],
                            conflict_rate_threshold=key[2],
                            min_quality_quantile=key[3],
                        )
                        state_cache[key] = states
                    state_counts = Counter(states.values())
                    for policy in _parse_strings(args.policies):
                        labels, policy_keep, policy_info = _labels_for_policy(base_labels, keep_indices, stats, states, policy)
                        pred = _pred_by_seq(table, labels, policy_keep, int(args.assignment_offset))
                        pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                        row = {
                            "mode": "sample_assignment_state_policy",
                            "committed_min_size": key[0],
                            "pending_max_size": key[1],
                            "conflict_rate_threshold": key[2],
                            "min_quality_quantile": key[3],
                            "state_counts": dict(state_counts),
                            **policy_info,
                            **pair,
                            "coverage_ratio": round(float(len(policy_keep) / max(len(table), 1)), 6),
                            "min_precision_recall": round(float(min(pair["tracklet_pair_precision"], pair["tracklet_pair_recall"])), 6),
                            "uses_anchors": False,
                            "uses_gt_for_training_or_anchors": False,
                            "uses_gt_for_evaluation_only": True,
                        }
                        rows.append(row)

    rows.sort(key=lambda row: _sort_tuple(row, str(args.sort_key)), reverse=True)
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        key = (
            int(row["committed_min_size"]),
            int(row["pending_max_size"]),
            float(row["conflict_rate_threshold"]),
            float(row["min_quality_quantile"]),
        )
        states = state_cache[key]
        labels, policy_keep, _info = _labels_for_policy(base_labels, keep_indices, stats, states, str(row["policy"]))
        labels_for_rank[rank] = (labels, policy_keep, states)
        assign_df = _assignments_from_labels(table, labels, policy_keep, states, base_labels, offset=int(args.assignment_offset))
        row.update(_sample_parquet_gt_score(df, assign_df))
        row["full_rank"] = rank
        print(json.dumps({"stage": "sample_full", "rank": rank, "row": _jsonable(row)}, sort_keys=True), flush=True)

    assignment_info = None
    component_info = None
    if rows:
        best = rows[0]
        key = (
            int(best["committed_min_size"]),
            int(best["pending_max_size"]),
            float(best["conflict_rate_threshold"]),
            float(best["min_quality_quantile"]),
        )
        states = state_cache[key]
        labels, policy_keep, _info = labels_for_rank.get(
            1,
            _labels_for_policy(base_labels, keep_indices, stats, states, str(best["policy"])),
        )
        if args.assignments_out:
            assign_df = _assignments_from_labels(table, labels, policy_keep, states, base_labels, offset=int(args.assignment_offset))
            out = Path(args.assignments_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            assign_df.to_csv(out, index=False)
            assignment_info = {"assignments_out": str(out), "assignment_rows": int(len(assign_df))}
            best.update(assignment_info)
        if args.component_states_out:
            component_info = _write_component_states(args.component_states_out, stats, states)
            best.update(component_info)

    result = {
        "tracklet_parquet": [str(path) for path in args.tracklet_parquet],
        "assignment_csv": str(args.assignments),
        "n_rows": int(len(df)),
        "n_tracklets": int(len(table)),
        "assigned_tracklets": int(len(keep_indices)),
        "raw_components": int(len(raw_to_local)),
        "eval_stats": eval_info,
        "base_pair_metrics": base_pair,
        "sort_key": str(args.sort_key),
        "grid_rows": int(len(rows)),
        "assignment_info": assignment_info,
        "component_info": component_info,
        "top": rows[: max(int(args.json_top_n), int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "sample_full_reference": "parquet_gt_same_detection_boxes",
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "done", "json": str(out), "best": _jsonable(rows[0] if rows else {})}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
