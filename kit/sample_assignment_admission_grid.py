#!/usr/bin/env python
"""Sweep no-anchor admission thresholds for sample-parquet assignments.

This is the sample/parquet counterpart of ``no_anchor_assignment_admission_grid``.
It keeps predicted global IDs fixed and changes only which tracklets are
delivered.  Ground truth in the sample parquet is used only for post-hoc
ranking and optional full scoring.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.evaluate_sample_assignments_full import _build_comp, _export_zip, _metric_dict
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


def _parse_grid(text: str, cast=float) -> list:
    return [cast(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _parse_fixed_video_values(text: str, cast=float) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(text or "").split(","):
        if not part.strip():
            continue
        video, value = part.rsplit(":", 1)
        out[str(video)] = cast(value)
    return out


def _parse_video_grid(text: str, cast=float) -> list[tuple[str, list[float]]]:
    out: list[tuple[str, list[float]]] = []
    for part in str(text or "").split(";"):
        if not part.strip():
            continue
        video, values = part.rsplit(":", 1)
        out.append((str(video), [cast(value) for value in values.split("|") if value.strip()]))
    return out


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


def _load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rename = {"video_key": "video", "frame_idx": "frame", "score": "confidence"}
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
        "area_mean": ("area", "mean"),
    }
    if "tracklet_majority_gt_id" in work.columns:
        agg["gt_id"] = ("tracklet_majority_gt_id", "first")
    if "tracklet_majority_gt_fraction" in work.columns:
        agg["gt_fraction"] = ("tracklet_majority_gt_fraction", "first")
    table = work.groupby("tracklet_key", sort=False).agg(**agg).reset_index()
    if "gt_id" not in table.columns:
        table["gt_id"] = ""
    if "gt_fraction" not in table.columns:
        table["gt_fraction"] = 0.0
    table = table.sort_values(["video", "start_frame", "end_frame", "tracklet_key"], kind="mergesort").reset_index(drop=True)
    table["seq"] = np.arange(len(table), dtype=np.int64)
    table["duration"] = (table["end_frame"] - table["start_frame"] + 1).astype(np.int64)
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
    return df


def _gt_mapping(table: pd.DataFrame, *, min_gt_fraction: float, min_rows: int) -> tuple[dict[int, int], dict[int, float], dict[str, object]]:
    mapping: dict[str, int] = {}
    gt_by_seq: dict[int, int] = {}
    weight_by_seq: dict[int, float] = {}
    rejected = 0
    for row in table.itertuples(index=False):
        gt_id = str(row.gt_id)
        if gt_id in {"", "nan", "None"} or float(row.gt_fraction) < min_gt_fraction or int(row.n_dets) < min_rows:
            rejected += 1
            continue
        if gt_id not in mapping:
            mapping[gt_id] = len(mapping) + 1
        gt_by_seq[int(row.seq)] = int(mapping[gt_id])
        weight_by_seq[int(row.seq)] = float(row.n_dets)
    return gt_by_seq, weight_by_seq, {
        "eval_labeled_tracklets": int(len(gt_by_seq)),
        "eval_rejected_tracklets": int(rejected),
        "eval_unique_gt_ids": int(len(set(gt_by_seq.values()))),
        "eval_min_gt_fraction": float(min_gt_fraction),
        "eval_min_rows": int(min_rows),
    }


def _pair_metrics(seqs: list[int], pred_by_seq: dict[int, int], gt_by_seq: dict[int, int], weight_by_seq: dict[int, float]) -> dict[str, object]:
    pred_totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    gt_totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    cross_totals: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    n_eval = 0
    for seq in seqs:
        if seq not in gt_by_seq:
            continue
        weight = float(weight_by_seq.get(seq, 1.0))
        gt = int(gt_by_seq[seq])
        gt_totals[gt][0] += weight
        gt_totals[gt][1] += weight * weight
        pred = pred_by_seq.get(seq)
        if pred is None:
            continue
        pred = int(pred)
        pred_totals[pred][0] += weight
        pred_totals[pred][1] += weight * weight
        cross_totals[(pred, gt)][0] += weight
        cross_totals[(pred, gt)][1] += weight * weight
        n_eval += 1
    pred_pairs = sum(max((v[0] * v[0] - v[1]) / 2.0, 0.0) for v in pred_totals.values())
    gt_pairs = sum(max((v[0] * v[0] - v[1]) / 2.0, 0.0) for v in gt_totals.values())
    true_pairs = sum(max((v[0] * v[0] - v[1]) / 2.0, 0.0) for v in cross_totals.values())
    precision = true_pairs / pred_pairs if pred_pairs > 0 else 0.0
    recall = true_pairs / gt_pairs if gt_pairs > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "tracklet_pair_f1": round(float(f1), 6),
        "tracklet_pair_precision": round(float(precision), 6),
        "tracklet_pair_recall": round(float(recall), 6),
        "eval_tracklets": int(n_eval),
        "pred_pair_mass": round(float(pred_pairs), 3),
        "gt_pair_mass": round(float(gt_pairs), 3),
        "true_pair_mass": round(float(true_pairs), 3),
    }


def _quality_thresholds(table: pd.DataFrame, video_quantiles: dict[str, float]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for video, quantile in video_quantiles.items():
        values = table.loc[table["video"].astype(str) == str(video), "quality"].to_numpy(np.float32)
        thresholds[str(video)] = float(np.quantile(values, float(quantile))) if values.size else -1.0e9
    return thresholds


def _keep_tracklets(
    table: pd.DataFrame,
    *,
    min_dets: int,
    min_conf: float,
    min_area: float,
    min_quality: float,
    video_min_area: dict[str, float],
    video_quality_quantile: dict[str, float],
) -> tuple[set[str], dict[str, object]]:
    quality_thresholds = _quality_thresholds(table, video_quality_quantile)
    keep: set[str] = set()
    drop_reasons: Counter[str] = Counter()
    for row in table.itertuples(index=False):
        key = str(row.tracklet_key)
        video = str(row.video)
        area_thr = max(float(min_area), float(video_min_area.get(video, -math.inf)))
        quality_thr = max(float(min_quality), float(quality_thresholds.get(video, -math.inf)))
        if int(row.n_dets) < int(min_dets):
            drop_reasons["min_dets"] += 1
            continue
        if float(row.avg_conf) < float(min_conf):
            drop_reasons["min_conf"] += 1
            continue
        if float(row.area_median) < area_thr:
            drop_reasons["min_area"] += 1
            continue
        if float(row.quality) < quality_thr:
            drop_reasons["min_quality"] += 1
            continue
        keep.add(key)
    return keep, {
        "output_min_dets": int(min_dets),
        "output_min_conf": float(min_conf),
        "output_min_area": float(min_area),
        "output_min_quality": float(min_quality),
        "video_min_area": video_min_area,
        "video_quality_quantile": video_quality_quantile,
        "video_quality_thresholds": {k: round(v, 6) for k, v in quality_thresholds.items()},
        "output_kept_tracklets": int(len(keep)),
        "output_dropped_tracklets": int(len(table) - len(keep)),
        "output_drop_reasons": dict(drop_reasons),
    }


def _filtered_assignments(assignments: pd.DataFrame, keep_keys: set[str]) -> pd.DataFrame:
    return assignments[assignments["tracklet_key"].astype(str).isin(keep_keys)].copy()


def _pred_by_seq(table: pd.DataFrame, assignments: pd.DataFrame, keep_keys: set[str]) -> dict[int, int]:
    seq_by_key = {str(row.tracklet_key): int(row.seq) for row in table.itertuples(index=False)}
    out: dict[int, int] = {}
    for row in assignments.itertuples(index=False):
        key = str(row.tracklet_key)
        if key not in keep_keys or key not in seq_by_key:
            continue
        out[int(seq_by_key[key])] = int(row.predicted_global_id)
    return out


def _write_assignments(path: str, assignments: pd.DataFrame) -> dict[str, object]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out, index=False)
    return {"assignments_out": str(out), "assignment_rows": int(len(assignments))}


def _full_score(df: pd.DataFrame, assignments: pd.DataFrame, *, zip_out: str = "") -> dict[str, object]:
    work = df.merge(assignments[["tracklet_key", "predicted_global_id"]], on="tracklet_key", how="inner")
    work["predicted_global_id"] = work["predicted_global_id"].astype(np.int64)
    comp = _build_comp(work)
    gt_by_video = load_ds1_gt_by_video()
    keys = sorted(set(gt_by_video).intersection(comp))
    metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    out = {
        **_metric_dict(metrics),
        "videos_scored": keys,
        "scored_rows": int(len(work)),
        "predicted_ids": int(work["predicted_global_id"].nunique()) if len(work) else 0,
    }
    if zip_out:
        out.update(_export_zip(comp, zip_out))
    return out


def _sort_tuple(row: dict[str, object], key: str) -> tuple[float, float, float, float, int]:
    return (
        float(row.get(key, 0.0)),
        float(row.get("tracklet_pair_f1", 0.0)),
        float(row.get("tracklet_pair_precision", 0.0)),
        float(row.get("tracklet_pair_recall", 0.0)),
        int(row.get("output_kept_tracklets", 0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracklet-parquet", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--min-dets-grid", default="1")
    ap.add_argument("--min-conf-grid", default="0.0")
    ap.add_argument("--min-area-grid", default="0")
    ap.add_argument("--min-quality-grid", default="-1000000000")
    ap.add_argument("--fixed-video-min-area", default="")
    ap.add_argument("--video-area-grid", default="")
    ap.add_argument("--fixed-video-quality-quantile", default="")
    ap.add_argument("--video-quality-quantile-grid", default="")
    ap.add_argument("--eval-min-gt-fraction", type=float, default=0.5)
    ap.add_argument("--eval-min-rows", type=int, default=1)
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--json-top-n", type=int, default=200)
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--zip-out", default="")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    df = _load_parquet(args.tracklet_parquet)
    table = _tracklet_table(df)
    assignments = _load_assignments(args.assignments)
    gt_by_seq, weight_by_seq, eval_info = _gt_mapping(table, min_gt_fraction=args.eval_min_gt_fraction, min_rows=args.eval_min_rows)
    seqs = [int(row.seq) for row in table.itertuples(index=False)]

    fixed_area = _parse_fixed_video_values(args.fixed_video_min_area, float)
    area_grid = _parse_video_grid(args.video_area_grid, float) or [("__none__", [0.0])]
    fixed_quality_quantile = _parse_fixed_video_values(args.fixed_video_quality_quantile, float)
    quality_grid = _parse_video_grid(args.video_quality_quantile_grid, float) or [("__none__", [0.0])]
    rows: list[dict[str, object]] = []
    keep_by_rank_key: dict[str, set[str]] = {}

    for min_dets in _parse_grid(args.min_dets_grid, int):
        for min_conf in _parse_grid(args.min_conf_grid, float):
            for min_area in _parse_grid(args.min_area_grid, float):
                for min_quality in _parse_grid(args.min_quality_grid, float):
                    for area_values in itertools.product(*[item[1] for item in area_grid]):
                        video_area = dict(fixed_area)
                        for idx, (video, _choices) in enumerate(area_grid):
                            if video != "__none__":
                                video_area[video] = float(area_values[idx])
                        for quality_values in itertools.product(*[item[1] for item in quality_grid]):
                            video_quality = dict(fixed_quality_quantile)
                            for idx, (video, _choices) in enumerate(quality_grid):
                                if video != "__none__":
                                    video_quality[video] = float(quality_values[idx])
                            keep, admission = _keep_tracklets(
                                table,
                                min_dets=int(min_dets),
                                min_conf=float(min_conf),
                                min_area=float(min_area),
                                min_quality=float(min_quality),
                                video_min_area=video_area,
                                video_quality_quantile=video_quality,
                            )
                            filtered = _filtered_assignments(assignments, keep)
                            pred = _pred_by_seq(table, filtered, keep)
                            metrics = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                            rank_key = f"r{len(rows)}"
                            keep_by_rank_key[rank_key] = keep
                            row = {
                                "rank_key": rank_key,
                                "assignment_csv": str(args.assignments),
                                "assigned_tracklets": int(len(filtered)),
                                "coverage_ratio": round(float(len(filtered) / max(len(table), 1)), 6),
                                **admission,
                                **metrics,
                                "min_precision_recall": round(float(min(metrics["tracklet_pair_precision"], metrics["tracklet_pair_recall"])), 6),
                                "uses_anchors": False,
                                "uses_gt_for_training_or_anchors": False,
                                "uses_gt_for_evaluation_only": True,
                            }
                            rows.append(row)

    rows.sort(key=lambda row: _sort_tuple(row, str(args.sort_key)), reverse=True)
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        keep = keep_by_rank_key[str(row["rank_key"])]
        filtered = _filtered_assignments(assignments, keep)
        zip_out = ""
        if args.zip_out and rank == 1:
            zip_out = str(args.zip_out)
        full = _full_score(df, filtered, zip_out=zip_out)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = rank
        print(json.dumps({"stage": "full", "rank": rank, "row": _jsonable(row)}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        keep = keep_by_rank_key[str(rows[0]["rank_key"])]
        assignment_info = _write_assignments(args.assignments_out, _filtered_assignments(assignments, keep))
        rows[0].update(assignment_info)

    result = {
        "tracklet_parquet": str(args.tracklet_parquet),
        "assignment_csv": str(args.assignments),
        "n_rows": int(len(df)),
        "n_tracklets": int(len(table)),
        "eval_stats": eval_info,
        "sort_key": str(args.sort_key),
        "grid_rows": int(len(rows)),
        "assignment_info": assignment_info,
        "top": rows[: max(int(args.json_top_n), int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "done", "json": str(out), "n_rows": len(rows), "best": _jsonable(rows[0] if rows else {})}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
