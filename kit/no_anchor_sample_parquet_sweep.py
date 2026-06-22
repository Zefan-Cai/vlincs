#!/usr/bin/env python
"""No-anchor global-ID model sweep on local sample parquet + feature NPZ files.

This is the parquet/NPZ sibling of ``no_anchor_global_id_model.py``.  It is
designed for cached sample runs where we have YOLO tracklet boxes and external
tracklet features, but not the full PostgreSQL gallery DB.  Ground-truth IDs
stored in the parquet/NPZ metadata are used only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

try:
    from kit.no_anchor_global_id_model import (
        BASE_FEATURE_NAMES,
        FEATURE_NAMES,
        PairModelConfig,
        _component_assignment_metadata,
        _fit_model,
        _load_pair_feature_views,
        _pseudo_training_pairs,
        _resolve_with_model,
        _write_assignments,
    )
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        TrackletRecord,
        _labels_to_seq_map,
        _pair_metrics,
        _time_agglom_resolve,
    )
    from vlincs_gallery.feature_centralization import neighbor_feature_centralization
except ModuleNotFoundError:
    from no_anchor_global_id_model import (
        BASE_FEATURE_NAMES,
        FEATURE_NAMES,
        PairModelConfig,
        _component_assignment_metadata,
        _fit_model,
        _load_pair_feature_views,
        _pseudo_training_pairs,
        _resolve_with_model,
        _write_assignments,
    )
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        TrackletRecord,
        _labels_to_seq_map,
        _pair_metrics,
        _time_agglom_resolve,
    )
    from vlincs_gallery.feature_centralization import neighbor_feature_centralization


DEFAULT_SAMPLE_ROOT = (
    "/Users/zcai/Codex/videolincs/local_runs/"
    "yolo_reference_labels_iou050_mcam04_08_frame12000_18000_20260616"
)
_CAM_RE = re.compile(r"(MCAM\d+)")
_GT_NUM_RE = re.compile(r"(\d+)$")


def _parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_weight_map(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not text:
        return weights
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            raise ValueError(f"bad feature weight entry {part!r}; expected key=value")
        key = key.strip()
        if not key:
            raise ValueError(f"bad feature weight entry {part!r}; empty key")
        weights[key] = float(value)
    return weights


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        denom = float(np.linalg.norm(x)) + 1.0e-9
        return (x / denom).astype(np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)).astype(np.float32)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _camera_from_video(video_key: str) -> str:
    match = _CAM_RE.search(str(video_key))
    return match.group(1) if match else "UNKNOWN"


def _gt_numeric_id(gt_id: object, mapping: dict[str, int]) -> int:
    key = str(gt_id)
    if key in mapping:
        return mapping[key]
    match = _GT_NUM_RE.search(key)
    if match:
        candidate = int(match.group(1))
        if candidate > 0 and candidate not in mapping.values():
            mapping[key] = candidate
            return candidate
    mapping[key] = max(mapping.values(), default=0) + 1
    return mapping[key]


def _resolve_parquet_paths(sample_root: Path, explicit: list[str]) -> list[Path]:
    paths: list[Path] = []
    if explicit:
        for item in explicit:
            item_path = Path(item)
            if any(ch in item for ch in "*?[]"):
                paths.extend(Path(path) for path in sorted(glob.glob(item)))
            elif item_path.is_dir():
                paths.extend(sorted(item_path.glob("*eval.parquet")))
            else:
                paths.append(item_path)
    else:
        paths = sorted(sample_root.glob("*eval.parquet"))
    out = [path for path in paths if path.exists()]
    if not out:
        raise FileNotFoundError(f"no parquet files found under {sample_root}")
    return out


def _resolve_feature_paths(sample_root: Path, explicit: list[str]) -> list[Path]:
    if explicit:
        paths = [Path(path) for path in explicit]
    else:
        paths = sorted(sample_root.glob("features*.npz"))
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise FileNotFoundError(f"no feature npz files found under {sample_root}")
    return paths


def _load_sample_parquets(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        df["_source_parquet"] = str(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    required = {
        "video_key",
        "frame_idx",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "tracklet_key",
        "tracklet_majority_gt_id",
        "tracklet_majority_gt_fraction",
    }
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"sample parquet is missing required columns: {missing}")
    return out


def _build_records(df: pd.DataFrame, *, fps: float) -> tuple[list[TrackletRecord], pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    ordered_keys = (
        df.groupby("tracklet_key", sort=False)
        .agg(video_key=("video_key", "first"), start_frame=("frame_idx", "min"), end_frame=("frame_idx", "max"))
        .reset_index()
        .sort_values(["video_key", "start_frame", "end_frame", "tracklet_key"], kind="mergesort")
    )
    key_to_seq = {str(key): idx for idx, key in enumerate(ordered_keys["tracklet_key"].tolist())}
    for key, group in df.sort_values(["tracklet_key", "frame_idx"], kind="mergesort").groupby("tracklet_key", sort=False):
        key = str(key)
        seq = int(key_to_seq[key])
        first = group.iloc[0]
        last = group.iloc[-1]
        video = str(first["video_key"])
        camera = _camera_from_video(video)
        x1 = group["x1"].astype(float)
        y1 = group["y1"].astype(float)
        x2 = group["x2"].astype(float)
        y2 = group["y2"].astype(float)
        widths = np.maximum(x2.to_numpy(np.float32) - x1.to_numpy(np.float32), 0.0)
        heights = np.maximum(y2.to_numpy(np.float32) - y1.to_numpy(np.float32), 0.0)
        row = {
            "seq": seq,
            "tracklet_key": key,
            "video": video,
            "camera": camera,
            "start_frame": int(group["frame_idx"].min()),
            "end_frame": int(group["frame_idx"].max()),
            "start_abs_ms": int(round(float(group["frame_idx"].min()) * 1000.0 / max(float(fps), 1.0e-6))),
            "end_abs_ms": int(round(float(group["frame_idx"].max()) * 1000.0 / max(float(fps), 1.0e-6))),
            "n_dets": int(len(group)),
            "avg_conf": float(group["score"].astype(float).mean()),
            "cx": float(((x1 + x2) * 0.5).mean()),
            "cy": float(((y1 + y2) * 0.5).mean()),
            "width": float(widths.mean()),
            "height": float(heights.mean()),
            "first_cx": float((float(first["x1"]) + float(first["x2"])) * 0.5),
            "first_cy": float((float(first["y1"]) + float(first["y2"])) * 0.5),
            "first_width": float(max(float(first["x2"]) - float(first["x1"]), 0.0)),
            "first_height": float(max(float(first["y2"]) - float(first["y1"]), 0.0)),
            "last_cx": float((float(last["x1"]) + float(last["x2"])) * 0.5),
            "last_cy": float((float(last["y1"]) + float(last["y2"])) * 0.5),
            "last_width": float(max(float(last["x2"]) - float(last["x1"]), 0.0)),
            "last_height": float(max(float(last["y2"]) - float(last["y1"]), 0.0)),
            "gt_id": str(first["tracklet_majority_gt_id"]),
            "gt_fraction": float(first["tracklet_majority_gt_fraction"]),
            "area_median": float(np.median(widths * heights)) if len(widths) else 0.0,
        }
        rows.append(row)
    rows.sort(key=lambda row: int(row["seq"]))
    table = pd.DataFrame(rows)
    records = [
        TrackletRecord(
            seq=int(row.seq),
            tracklet_key=str(row.tracklet_key),
            video=str(row.video),
            camera=str(row.camera),
            start_frame=int(row.start_frame),
            end_frame=int(row.end_frame),
            start_abs_ms=int(row.start_abs_ms),
            end_abs_ms=int(row.end_abs_ms),
            n_dets=int(row.n_dets),
            avg_conf=float(row.avg_conf),
            cx=float(row.cx),
            cy=float(row.cy),
            width=float(row.width),
            height=float(row.height),
            first_cx=float(row.first_cx),
            first_cy=float(row.first_cy),
            first_width=float(row.first_width),
            first_height=float(row.first_height),
            last_cx=float(row.last_cx),
            last_cy=float(row.last_cy),
            last_width=float(row.last_width),
            last_height=float(row.last_height),
        )
        for row in table.itertuples(index=False)
    ]
    return records, table, key_to_seq


def _feature_keys_for_npz(data: np.lib.npyio.NpzFile, requested: str, include_trajectory: bool) -> list[str]:
    if requested and requested != "auto":
        return [key for key in _parse_csv_strings(requested) if key in data.files]
    keys = []
    for key in data.files:
        if not key.startswith("features_"):
            continue
        if key == "features_trajectory" and not include_trajectory:
            continue
        keys.append(key)
    return sorted(keys)


def _load_npz_metadata(data: np.lib.npyio.NpzFile, path: Path) -> dict[str, object]:
    if "metadata" not in data.files:
        raise ValueError(f"{path} is missing metadata")
    return json.loads(str(data["metadata"].item()))


def _feature_matrix_from_npzs(
    paths: list[Path],
    records: list[TrackletRecord],
    *,
    feature_keys: str,
    include_trajectory: bool,
    feature_weights: dict[str, float],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    blocks: list[np.ndarray] = []
    block_info: list[dict[str, object]] = []
    tracklet_keys = [record.tracklet_key for record in records]
    for path in paths:
        data = np.load(path, allow_pickle=True)
        metadata = _load_npz_metadata(data, path)
        meta_records = metadata.get("records", [])
        index_by_key = {str(row["tracklet_key"]): int(row["index"]) for row in meta_records}
        missing = [key for key in tracklet_keys if key not in index_by_key]
        if missing:
            raise ValueError(f"{path} is missing {len(missing)} tracklets; first missing={missing[0]}")
        indices = np.asarray([index_by_key[key] for key in tracklet_keys], dtype=np.int64)
        for key in _feature_keys_for_npz(data, feature_keys, include_trajectory):
            feats = data[key].astype(np.float32)
            if feats.ndim != 2:
                continue
            mat = feats[indices].astype(np.float32)
            valid_key = "valid_" + key[len("features_") :]
            valid = np.ones((len(indices),), dtype=bool)
            if valid_key in data.files:
                valid = data[valid_key].astype(bool)[indices]
                mat[~valid] = 0.0
            if key == "features_trajectory":
                mean = mat.mean(axis=0, keepdims=True)
                std = mat.std(axis=0, keepdims=True) + 1.0e-6
                mat = (mat - mean) / std
            mat = _l2n(mat)
            weight = float(feature_weights.get(key, feature_weights.get(key.removeprefix("features_"), 1.0)))
            mat = (mat * weight).astype(np.float32)
            blocks.append(mat)
            block_info.append(
                {
                    "path": str(path),
                    "key": key,
                    "dim": int(mat.shape[1]),
                    "weight": weight,
                    "valid_rows": int(valid.sum()),
                    "metadata_model": metadata.get("model"),
                }
            )
    if not blocks:
        raise RuntimeError("no usable feature blocks loaded")
    emb = _l2n(np.concatenate(blocks, axis=1).astype(np.float32))
    return emb, block_info


def _apply_nfc_transform(
    emb: np.ndarray,
    records: list[TrackletRecord],
    *,
    k1: int,
    k2: int,
    eta: float,
    exclude_same_camera: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    if int(k1) <= 0:
        return emb, {
            "feature_transform": "none",
            "nfc_enabled": False,
        }
    camera_to_code = {camera: idx for idx, camera in enumerate(sorted({record.camera for record in records}))}
    group_codes = np.asarray([camera_to_code[record.camera] for record in records], dtype=np.int32)
    out, info = neighbor_feature_centralization(
        emb,
        k1=int(k1),
        k2=int(k2),
        eta=float(eta),
        group_codes=group_codes,
        exclude_same_group=bool(exclude_same_camera),
    )
    return out.astype(np.float32), {
        "feature_transform": "neighbor_feature_centralization",
        "nfc_enabled": True,
        "nfc_k1": int(k1),
        "nfc_k2": int(k2),
        "nfc_eta": float(eta),
        "nfc_exclude_same_camera": bool(exclude_same_camera),
        "nfc_info": asdict(info),
    }


def _eval_labels(
    tracklet_table: pd.DataFrame,
    *,
    min_gt_fraction: float,
    min_rows: int,
) -> tuple[dict[int, int], dict[int, float], dict[str, object]]:
    mapping: dict[str, int] = {}
    gt_by_seq: dict[int, int] = {}
    weight_by_seq: dict[int, float] = {}
    rejected = 0
    for row in tracklet_table.itertuples(index=False):
        if float(row.gt_fraction) < float(min_gt_fraction) or int(row.n_dets) < int(min_rows):
            rejected += 1
            continue
        gt_by_seq[int(row.seq)] = _gt_numeric_id(row.gt_id, mapping)
        weight_by_seq[int(row.seq)] = float(row.n_dets)
    return (
        gt_by_seq,
        weight_by_seq,
        {
            "eval_labeled_tracklets": int(len(gt_by_seq)),
            "eval_rejected_tracklets": int(rejected),
            "eval_min_gt_fraction": float(min_gt_fraction),
            "eval_min_rows": int(min_rows),
            "eval_unique_gt_ids": int(len(set(gt_by_seq.values()))),
            "uses_gt_for_evaluation_only": True,
        },
    )


def _output_keep_seqs(tracklet_table: pd.DataFrame, *, min_dets: int, min_conf: float, min_area: float) -> tuple[set[int], dict[str, object]]:
    keep: set[int] = set()
    for row in tracklet_table.itertuples(index=False):
        if int(row.n_dets) < int(min_dets):
            continue
        if float(row.avg_conf) < float(min_conf):
            continue
        if float(row.area_median) < float(min_area):
            continue
        keep.add(int(row.seq))
    return keep, {
        "output_min_dets": int(min_dets),
        "output_min_conf": float(min_conf),
        "output_min_area": float(min_area),
        "output_kept_tracklets": int(len(keep)),
        "output_dropped_tracklets": int(len(tracklet_table) - len(keep)),
    }


def _identity_row_metrics(
    df: pd.DataFrame,
    key_to_seq: dict[str, int],
    pred_by_seq: dict[int, int],
    *,
    gt_col: str,
) -> dict[str, float | int]:
    if gt_col not in df.columns:
        gt_col = "tracklet_majority_gt_id"
    work = df[["tracklet_key", gt_col]].copy()
    work["seq"] = work["tracklet_key"].map(lambda key: key_to_seq.get(str(key), -1)).astype(int)
    work = work[work["seq"] >= 0].copy()
    work["gt"] = work[gt_col].astype(str)
    work = work[work["gt"].notna() & (work["gt"] != "nan") & (work["gt"] != "None")].copy()
    total_gt_rows = int(len(work))
    work["pred"] = work["seq"].map(lambda seq: pred_by_seq.get(int(seq)))
    pred = work[work["pred"].notna()].copy()
    pred_rows = int(len(pred))
    if pred_rows == 0 or total_gt_rows == 0:
        return {
            "identity_precision": 0.0,
            "identity_recall": 0.0,
            "identity_f1": 0.0,
            "identity_matched_rows": 0,
            "identity_pred_rows": pred_rows,
            "identity_gt_rows": total_gt_rows,
            "identity_pred_ids": 0,
            "identity_gt_ids": int(work["gt"].nunique()) if total_gt_rows else 0,
        }
    pred["pred"] = pred["pred"].astype(int)
    confusion = pred.groupby(["pred", "gt"], sort=False).size().unstack(fill_value=0)
    matrix = confusion.to_numpy(np.float64)
    row_ind, col_ind = linear_sum_assignment(-matrix)
    matched = int(matrix[row_ind, col_ind].sum())
    precision = matched / max(pred_rows, 1)
    recall = matched / max(total_gt_rows, 1)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "identity_precision": round(float(precision), 6),
        "identity_recall": round(float(recall), 6),
        "identity_f1": round(float(f1), 6),
        "identity_matched_rows": int(matched),
        "identity_pred_rows": int(pred_rows),
        "identity_gt_rows": int(total_gt_rows),
        "identity_pred_ids": int(confusion.shape[0]),
        "identity_gt_ids": int(confusion.shape[1]),
    }


def _component_stats(
    tracklet_table: pd.DataFrame,
    pred_by_seq: dict[int, int],
    gt_by_seq: dict[int, int],
    weight_by_seq: dict[int, float],
) -> dict[str, object]:
    comp_to_items: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for seq, gid in gt_by_seq.items():
        pred = pred_by_seq.get(int(seq))
        if pred is None:
            continue
        comp_to_items[int(pred)].append((int(gid), float(weight_by_seq.get(int(seq), 1.0))))
    purities = []
    sizes = []
    for items in comp_to_items.values():
        counts: Counter[int] = Counter()
        total = 0.0
        for gid, weight in items:
            counts[int(gid)] += float(weight)
            total += float(weight)
        if total <= 0:
            continue
        purities.append(max(counts.values()) / total)
        sizes.append(len(items))
    return {
        "pred_components_with_eval_gt": int(len(comp_to_items)),
        "component_purity_mean": round(float(np.mean(purities)) if purities else 0.0, 6),
        "component_purity_p10": round(float(np.quantile(purities, 0.10)) if purities else 0.0, 6),
        "component_purity_min": round(float(np.min(purities)) if purities else 0.0, 6),
        "component_size_mean": round(float(np.mean(sizes)) if sizes else 0.0, 6),
        "component_size_max": int(max(sizes, default=0)),
    }


def _prefixed_metrics(prefix: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _pair_metrics_against_full(
    seqs: Iterable[int],
    pred_by_seq: dict[int, int],
    gt_by_seq: dict[int, int],
    weight_by_seq: dict[int, float],
) -> dict[str, float | int]:
    pred_totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    gt_totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    cross_totals: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    n_eval = 0
    for seq in seqs:
        seq = int(seq)
        if seq not in gt_by_seq:
            continue
        w = float(weight_by_seq.get(seq, 1.0))
        g = int(gt_by_seq[seq])
        gt_totals[g][0] += w
        gt_totals[g][1] += w * w
        if seq not in pred_by_seq:
            continue
        p = int(pred_by_seq[seq])
        pred_totals[p][0] += w
        pred_totals[p][1] += w * w
        cross_totals[(p, g)][0] += w
        cross_totals[(p, g)][1] += w * w
        n_eval += 1

    pred_pairs = sum(max((v[0] * v[0] - v[1]) / 2.0, 0.0) for v in pred_totals.values())
    gt_pairs = sum(max((v[0] * v[0] - v[1]) / 2.0, 0.0) for v in gt_totals.values())
    true_pairs = sum(max((v[0] * v[0] - v[1]) / 2.0, 0.0) for v in cross_totals.values())
    precision = true_pairs / pred_pairs if pred_pairs > 0 else 0.0
    recall = true_pairs / gt_pairs if gt_pairs > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "tracklet_pair_precision_full": round(float(precision), 6),
        "tracklet_pair_recall_full": round(float(recall), 6),
        "tracklet_pair_f1_full": round(float(f1), 6),
        "eval_tracklets_full": int(n_eval),
        "pred_pair_mass_full": round(float(pred_pairs), 3),
        "gt_pair_mass_full": round(float(gt_pairs), 3),
        "true_pair_mass_full": round(float(true_pairs), 3),
    }


def _component_centroid_bundle(
    records: list[TrackletRecord],
    emb: np.ndarray,
    labels: np.ndarray,
    keep_seqs: set[int],
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, object]], dict[int, int]]:
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (record, label) in enumerate(zip(records, labels)):
        if int(record.seq) not in keep_seqs:
            continue
        label_to_indices[int(label)].append(idx)

    centroids: dict[int, np.ndarray] = {}
    component_meta: dict[int, dict[str, object]] = {}
    counts = {label: len(indices) for label, indices in label_to_indices.items()}
    ordered_labels = sorted(label_to_indices)
    if ordered_labels:
        centroid_matrix = []
        for label in ordered_labels:
            centroid = _l2n(emb[label_to_indices[label]].mean(axis=0))
            centroids[int(label)] = centroid
            centroid_matrix.append(centroid)
        centroid_matrix_np = _l2n(np.vstack(centroid_matrix).astype(np.float32))
        centroid_sim = centroid_matrix_np @ centroid_matrix_np.T
    else:
        centroid_sim = np.zeros((0, 0), dtype=np.float32)

    for pos, label in enumerate(ordered_labels):
        indices = label_to_indices[label]
        centroid = centroids[int(label)]
        member_sims = np.asarray(emb[indices] @ centroid, dtype=np.float32)
        external_max = 0.0
        if len(ordered_labels) > 1:
            row = centroid_sim[pos].copy()
            row[pos] = -1.0
            external_max = float(row.max())
        internal_median = float(np.median(member_sims)) if member_sims.size else 0.0
        internal_min = float(np.min(member_sims)) if member_sims.size else 0.0
        component_meta[int(label)] = {
            "component_label": int(label),
            "component_size": int(counts[label]),
            "member_centroid_sim_median": round(internal_median, 6),
            "member_centroid_sim_min": round(internal_min, 6),
            "nearest_external_centroid_sim": round(external_max, 6),
            "centroid_margin": round(float(internal_median - external_max), 6),
        }
    return centroids, component_meta, counts


def _resolution_metadata(
    records: list[TrackletRecord],
    emb: np.ndarray,
    labels: np.ndarray,
    keep_seqs: set[int],
    *,
    commit_min_dets: int,
    commit_min_conf: float,
    commit_min_area: float,
    commit_min_component_size: int,
    commit_min_member_sim: float,
    commit_min_margin: float,
    provisional_min_dets: int,
    provisional_min_conf: float,
    tracklet_table: pd.DataFrame,
) -> tuple[dict[int, dict[str, object]], dict[int, dict[str, object]], dict[int, np.ndarray]]:
    centroids, component_meta, counts = _component_centroid_bundle(records, emb, labels, keep_seqs)
    area_by_seq = {int(row.seq): float(row.area_median) for row in tracklet_table.itertuples(index=False)}
    seq_to_label = {int(record.seq): int(label) for record, label in zip(records, labels)}
    seq_meta: dict[int, dict[str, object]] = {}
    status_counts: Counter[str] = Counter()
    for record, label in zip(records, labels):
        seq = int(record.seq)
        if seq not in keep_seqs:
            continue
        label = int(label)
        comp = component_meta.get(label, {})
        component_size = int(counts.get(label, 1))
        area = float(area_by_seq.get(seq, 0.0))
        quality_pass = (
            int(record.n_dets) >= int(commit_min_dets)
            and float(record.avg_conf) >= float(commit_min_conf)
            and area >= float(commit_min_area)
        )
        provisional_quality = (
            int(record.n_dets) >= int(provisional_min_dets)
            and float(record.avg_conf) >= float(provisional_min_conf)
            and area >= float(commit_min_area)
        )
        internal_ok = float(comp.get("member_centroid_sim_median", 0.0)) >= float(commit_min_member_sim)
        margin_ok = float(comp.get("centroid_margin", -1.0)) >= float(commit_min_margin)
        if component_size <= 1:
            status = "forced_singleton"
        elif quality_pass and component_size >= int(commit_min_component_size) and internal_ok and margin_ok:
            status = "committed"
        elif provisional_quality:
            status = "provisional"
        else:
            status = "forced_component"
        quality_term = min(float(record.n_dets) / max(float(commit_min_dets), 1.0), 1.0)
        conf_term = min(float(record.avg_conf) / max(float(commit_min_conf), 1.0e-6), 1.0)
        sim_term = float(np.clip((float(comp.get("member_centroid_sim_median", 0.0)) + 1.0) * 0.5, 0.0, 1.0))
        margin_term = float(np.clip((float(comp.get("centroid_margin", -1.0)) + 1.0) * 0.5, 0.0, 1.0))
        confidence = float(np.clip(0.30 * quality_term + 0.25 * conf_term + 0.25 * sim_term + 0.20 * margin_term, 0.0, 1.0))
        if status.startswith("forced"):
            confidence = min(confidence, 0.49)
        elif status == "provisional":
            confidence = min(max(confidence, 0.50), 0.79)
        else:
            confidence = max(confidence, 0.80)
        seq_meta[seq] = {
            "resolution_status": status,
            "prediction_confidence": round(confidence, 6),
            "tracklet_quality_pass": bool(quality_pass),
            "area_median": round(area, 3),
            **comp,
        }
        status_counts[status] += 1

    for comp in component_meta.values():
        label = int(comp["component_label"])
        comp_statuses = [
            str(meta["resolution_status"])
            for seq, meta in seq_meta.items()
            if int(seq_to_label.get(int(seq), -1)) == label
        ]
        comp["status_counts"] = dict(Counter(comp_statuses))
    return component_meta, seq_meta, centroids


def _status_pred_map(
    records: list[TrackletRecord],
    labels: np.ndarray,
    seq_meta: dict[int, dict[str, object]],
    statuses: set[str],
    *,
    offset: int = 30_000_000,
) -> dict[int, int]:
    out: dict[int, int] = {}
    for record, label in zip(records, labels):
        seq = int(record.seq)
        meta = seq_meta.get(seq)
        if not meta or str(meta.get("resolution_status")) not in statuses:
            continue
        out[seq] = int(offset + int(label))
    return out


def _resolution_status_metrics(
    labels: np.ndarray,
    *,
    records: list[TrackletRecord],
    emb: np.ndarray,
    df: pd.DataFrame,
    tracklet_table: pd.DataFrame,
    key_to_seq: dict[str, int],
    gt_by_seq: dict[int, int],
    weight_by_seq: dict[int, float],
    keep_seqs: set[int],
    gt_col: str,
    commit_min_dets: int,
    commit_min_conf: float,
    commit_min_area: float,
    commit_min_component_size: int,
    commit_min_member_sim: float,
    commit_min_margin: float,
    provisional_min_dets: int,
    provisional_min_conf: float,
) -> tuple[dict[str, object], dict[int, dict[str, object]], dict[int, dict[str, object]], dict[int, np.ndarray]]:
    component_meta, seq_meta, centroids = _resolution_metadata(
        records,
        emb,
        labels,
        keep_seqs,
        commit_min_dets=commit_min_dets,
        commit_min_conf=commit_min_conf,
        commit_min_area=commit_min_area,
        commit_min_component_size=commit_min_component_size,
        commit_min_member_sim=commit_min_member_sim,
        commit_min_margin=commit_min_margin,
        provisional_min_dets=provisional_min_dets,
        provisional_min_conf=provisional_min_conf,
        tracklet_table=tracklet_table,
    )
    status_counts = Counter(str(meta["resolution_status"]) for meta in seq_meta.values())
    seqs = [int(record.seq) for record in records]
    out: dict[str, object] = {
        "resolution_committed_tracklets": int(status_counts.get("committed", 0)),
        "resolution_provisional_tracklets": int(status_counts.get("provisional", 0)),
        "resolution_forced_component_tracklets": int(status_counts.get("forced_component", 0)),
        "resolution_forced_singleton_tracklets": int(status_counts.get("forced_singleton", 0)),
        "resolution_status_counts": dict(status_counts),
        "commit_min_dets": int(commit_min_dets),
        "commit_min_conf": float(commit_min_conf),
        "commit_min_area": float(commit_min_area),
        "commit_min_component_size": int(commit_min_component_size),
        "commit_min_member_sim": float(commit_min_member_sim),
        "commit_min_margin": float(commit_min_margin),
        "provisional_min_dets": int(provisional_min_dets),
        "provisional_min_conf": float(provisional_min_conf),
    }
    for prefix, statuses in (
        ("committed", {"committed"}),
        ("committed_or_provisional", {"committed", "provisional"}),
        ("deliverable", {"committed", "provisional", "forced_component", "forced_singleton"}),
    ):
        pred = _status_pred_map(records, labels, seq_meta, statuses)
        subset_pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
        full_pair = _pair_metrics_against_full(seqs, pred, gt_by_seq, weight_by_seq)
        identity = _identity_row_metrics(df, key_to_seq, pred, gt_col=gt_col)
        out.update(_prefixed_metrics(prefix, subset_pair))
        out.update(_prefixed_metrics(prefix, full_pair))
        out.update(_prefixed_metrics(prefix, identity))
        out[f"{prefix}_tracklet_coverage"] = round(float(len(pred)) / max(float(len(keep_seqs)), 1.0), 6)
    return out, component_meta, seq_meta, centroids


def _write_status_assignments(
    path: Path,
    records: list[TrackletRecord],
    labels: np.ndarray,
    *,
    keep_seqs: set[int],
    seq_meta: dict[int, dict[str, object]],
    offset: int = 30_000_000,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(int(label) for record, label in zip(records, labels) if int(record.seq) in keep_seqs)
    rows = []
    for record, label in zip(records, labels):
        seq = int(record.seq)
        if seq not in keep_seqs:
            continue
        label = int(label)
        meta = seq_meta.get(seq, {})
        status = str(meta.get("resolution_status", "forced_singleton" if int(counts[label]) == 1 else "forced_component"))
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
                "predicted_global_id": int(offset + label),
                "component_label": label,
                "component_size": int(counts[label]),
                "prediction_confidence": meta.get("prediction_confidence", 0.15 if int(counts[label]) == 1 else 0.35),
                "decision_status": status,
                "resolution_status": status,
                "tracklet_quality_pass": bool(meta.get("tracklet_quality_pass", False)),
                "area_median": meta.get("area_median", 0.0),
                "member_centroid_sim_median": meta.get("member_centroid_sim_median", 0.0),
                "member_centroid_sim_min": meta.get("member_centroid_sim_min", 0.0),
                "nearest_external_centroid_sim": meta.get("nearest_external_centroid_sim", 0.0),
                "centroid_margin": meta.get("centroid_margin", 0.0),
            }
        )
    fieldnames = [
        "seq",
        "tracklet_key",
        "video",
        "camera",
        "start_frame",
        "end_frame",
        "n_dets",
        "avg_conf",
        "predicted_global_id",
        "component_label",
        "component_size",
        "prediction_confidence",
        "decision_status",
        "resolution_status",
        "tracklet_quality_pass",
        "area_median",
        "member_centroid_sim_median",
        "member_centroid_sim_min",
        "nearest_external_centroid_sim",
        "centroid_margin",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    status_counts = Counter(str(row["resolution_status"]) for row in rows)
    return {
        "assignments_out": str(path),
        "assignment_rows": int(len(rows)),
        "assignment_components": int(len(counts)),
        "largest_assignment_component": int(max(counts.values(), default=0)),
        "assignment_status_counts": dict(status_counts),
    }


def _slice_metrics(
    df: pd.DataFrame,
    records: list[TrackletRecord],
    tracklet_table: pd.DataFrame,
    key_to_seq: dict[str, int],
    pred_by_seq: dict[int, int],
    gt_by_seq: dict[int, int],
    weight_by_seq: dict[int, float],
    *,
    gt_col: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    table_by_seq = tracklet_table.set_index("seq")
    for kind, col in (("video", "video_key"), ("camera", "camera")):
        if col == "camera":
            work = df.assign(camera=df["video_key"].map(_camera_from_video))
        else:
            work = df
        for value, group in work.groupby(col, sort=True):
            seqs = sorted({key_to_seq[str(key)] for key in group["tracklet_key"].unique() if str(key) in key_to_seq})
            if not seqs:
                continue
            pair = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
            row = {
                "slice_kind": kind,
                "slice_value": str(value),
                "rows": int(len(group)),
                "tracklets": int(len(seqs)),
                "gt_ids": int(table_by_seq.loc[seqs, "gt_id"].nunique()) if seqs else 0,
                **pair,
                **_identity_row_metrics(group, key_to_seq, pred_by_seq, gt_col=gt_col),
            }
            rows.append(row)
    return rows


def _evaluate_labels(
    labels: np.ndarray,
    *,
    mode: str,
    records: list[TrackletRecord],
    df: pd.DataFrame,
    tracklet_table: pd.DataFrame,
    key_to_seq: dict[str, int],
    gt_by_seq: dict[int, int],
    weight_by_seq: dict[int, float],
    keep_seqs: set[int],
    gt_col: str,
) -> tuple[dict[str, object], dict[int, int], list[dict[str, object]]]:
    pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
    pair = _pair_metrics([record.seq for record in records], pred_by_seq, gt_by_seq, weight_by_seq)
    identity = _identity_row_metrics(df, key_to_seq, pred_by_seq, gt_col=gt_col)
    row = {
        "mode": mode,
        **pair,
        **identity,
        **_component_stats(tracklet_table, pred_by_seq, gt_by_seq, weight_by_seq),
    }
    slices = _slice_metrics(df, records, tracklet_table, key_to_seq, pred_by_seq, gt_by_seq, weight_by_seq, gt_col=gt_col)
    return row, pred_by_seq, slices


def _target_agglom_labels(emb: np.ndarray, n_clusters: int) -> np.ndarray:
    n_clusters = min(max(int(n_clusters), 1), int(len(emb)))
    sim = np.clip(emb @ emb.T, -1.0, 1.0).astype(np.float32)
    dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float32)
    return AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average").fit_predict(dist)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if not isinstance(value, (dict, list, tuple))
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _pair_config_from_args(args: argparse.Namespace) -> PairModelConfig:
    return PairModelConfig(
        train_top_k=int(args.train_top_k),
        infer_top_k=int(args.infer_top_k),
        min_dets=int(args.min_dets),
        max_component_size=int(args.max_component_size),
        min_merge_support=int(args.min_merge_support),
        min_merge_support_ratio=float(args.min_merge_support_ratio),
        exclude_same=str(args.exclude_same),
        pseudo_theta=float(args.pseudo_theta),
        pseudo_top_k=int(args.pseudo_top_k),
        pseudo_temporal_bonus=float(args.pseudo_temporal_bonus),
        pseudo_time_window_ms=int(args.pseudo_time_window_ms),
        pseudo_ensemble=bool(args.pseudo_ensemble),
        pseudo_ensemble_thetas=str(args.pseudo_ensemble_thetas),
        pseudo_ensemble_min_votes=int(args.pseudo_ensemble_min_votes),
        pseudo_ensemble_max_neg_votes=int(args.pseudo_ensemble_max_neg_votes),
        pseudo_consensus_pos_min_votes=int(args.pseudo_consensus_pos_min_votes),
        pseudo_consensus_pos_min_sim=float(args.pseudo_consensus_pos_min_sim),
        pseudo_pos_min_sim=float(args.pseudo_pos_min_sim),
        pseudo_strong_pos_sim=float(args.pseudo_strong_pos_sim),
        pseudo_neg_max_sim=float(args.pseudo_neg_max_sim),
        candidate_time_bonus=float(args.candidate_time_bonus),
        affinity_time_bonus=float(args.affinity_time_bonus),
        attach_threshold=float(args.attach_threshold),
        attach_margin=float(args.attach_margin),
        attach_model_weight=float(args.attach_model_weight),
        attach_max_source_size=int(args.attach_max_source_size),
        attach_min_target_size=int(args.attach_min_target_size),
        attach_top_k=int(args.attach_top_k),
        attach_min_edge_support=int(args.attach_min_edge_support),
        attach_score_agg=str(args.attach_score_agg),
        attach_top_mean_k=int(args.attach_top_mean_k),
        max_neg_per_pos=float(args.max_neg_per_pos),
        random_negatives=int(args.random_negatives),
        model_type=str(args.model_type),
        random_state=int(args.random_state),
    )


def _sort_rows(rows: Iterable[dict[str, object]], sort_key: str) -> list[dict[str, object]]:
    def score(row: dict[str, object]) -> float:
        try:
            return float(row.get(sort_key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    return sorted(rows, key=score, reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-root", default=DEFAULT_SAMPLE_ROOT)
    ap.add_argument("--tracklet-parquet", action="append", default=[])
    ap.add_argument("--feature-npz", action="append", default=[])
    ap.add_argument("--feature-keys", default="auto")
    ap.add_argument(
        "--pair-feature-npz",
        action="append",
        default=[],
        help="optional name:path NPZ views with seqs/features[/valid] for pair-model-only evidence",
    )
    ap.add_argument(
        "--feature-key-weights",
        default="",
        help="optional comma list such as features_full_body=1.0,features_trajectory=0.05",
    )
    ap.add_argument("--include-trajectory", action="store_true")
    ap.add_argument("--nfc-k1", type=int, default=0, help="enable no-anchor mutual-neighbor feature centralization")
    ap.add_argument("--nfc-k2", type=int, default=2, help="reciprocal-neighbor depth for --nfc-k1")
    ap.add_argument("--nfc-eta", type=float, default=1.0, help="neighbor feature weight for --nfc-k1")
    ap.add_argument("--nfc-exclude-same-camera", action="store_true", help="exclude same-camera neighbors during NFC")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--gt-col", default="gt_id")
    ap.add_argument("--eval-min-gt-fraction", type=float, default=0.50)
    ap.add_argument("--eval-min-rows", type=int, default=1)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--commit-min-dets", type=int, default=400)
    ap.add_argument("--commit-min-conf", type=float, default=0.60)
    ap.add_argument("--commit-min-area", type=float, default=0.0)
    ap.add_argument("--commit-min-component-size", type=int, default=2)
    ap.add_argument("--commit-min-member-sim", type=float, default=0.0)
    ap.add_argument("--commit-min-margin", type=float, default=-1.0)
    ap.add_argument("--provisional-min-dets", type=int, default=80)
    ap.add_argument("--provisional-min-conf", type=float, default=0.20)
    ap.add_argument("--exclude-same", default="camera", choices=["camera", "stream", "video", "none"])
    ap.add_argument("--baseline-thetas", default="0.014,0.016,0.018,0.020,0.022,0.025,0.030,0.035")
    ap.add_argument("--baseline-top-ks", default="15,30,45")
    ap.add_argument("--target-clusters", default="", help="optional comma list for no-anchor fixed-N agglomerative clustering ablation")
    ap.add_argument("--train-top-k", type=int, default=45)
    ap.add_argument("--infer-top-k", type=int, default=45)
    ap.add_argument("--max-component-size", type=int, default=120)
    ap.add_argument("--min-merge-support", type=int, default=1)
    ap.add_argument("--min-merge-support-ratio", type=float, default=0.0)
    ap.add_argument("--pseudo-theta", type=float, default=0.018)
    ap.add_argument("--pseudo-top-k", type=int, default=24)
    ap.add_argument("--pseudo-temporal-bonus", type=float, default=0.005)
    ap.add_argument("--pseudo-time-window-ms", type=int, default=1000)
    ap.add_argument("--pseudo-ensemble", dest="pseudo_ensemble", action="store_true", default=True)
    ap.add_argument("--no-pseudo-ensemble", dest="pseudo_ensemble", action="store_false")
    ap.add_argument("--pseudo-ensemble-thetas", default="")
    ap.add_argument("--pseudo-ensemble-min-votes", type=int, default=2)
    ap.add_argument("--pseudo-ensemble-max-neg-votes", type=int, default=0)
    ap.add_argument("--pseudo-consensus-pos-min-votes", type=int, default=3)
    ap.add_argument("--pseudo-consensus-pos-min-sim", type=float, default=0.66)
    ap.add_argument("--pseudo-pos-min-sim", type=float, default=0.62)
    ap.add_argument("--pseudo-strong-pos-sim", type=float, default=0.76)
    ap.add_argument("--pseudo-neg-max-sim", type=float, default=0.42)
    ap.add_argument("--candidate-time-bonus", type=float, default=0.0)
    ap.add_argument("--affinity-time-bonus", type=float, default=0.005)
    ap.add_argument("--attach-threshold", type=float, default=0.72)
    ap.add_argument("--attach-margin", type=float, default=0.06)
    ap.add_argument("--attach-model-weight", type=float, default=0.65)
    ap.add_argument("--attach-max-source-size", type=int, default=2)
    ap.add_argument("--attach-min-target-size", type=int, default=2)
    ap.add_argument("--attach-top-k", type=int, default=60)
    ap.add_argument("--attach-min-edge-support", type=int, default=1)
    ap.add_argument("--attach-score-agg", default="max", choices=["max", "top_mean", "hybrid"])
    ap.add_argument("--attach-top-mean-k", type=int, default=3)
    ap.add_argument("--max-neg-per-pos", type=float, default=4.0)
    ap.add_argument("--random-negatives", type=int, default=60000)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf"])
    ap.add_argument("--solvers", default="consensus_guard,consensus_attach")
    ap.add_argument("--skip-pair-model", action="store_true")
    ap.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--blends", default="0.00")
    ap.add_argument("--sort-key", default="identity_f1")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--slice-csv", default=None)
    ap.add_argument("--assignments-out", default=None)
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--global-model-out", default=None)
    ap.add_argument("--random-state", type=int, default=17)
    args = ap.parse_args()

    sample_root = Path(args.sample_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else sample_root / f"no_anchor_sample_model_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = Path(args.json) if args.json else out_dir / "no_anchor_sample_model_summary.json"
    csv_out = Path(args.csv) if args.csv else out_dir / "no_anchor_sample_model_sweep.csv"
    slice_csv_out = Path(args.slice_csv) if args.slice_csv else out_dir / "no_anchor_sample_model_slices.csv"
    assignments_out = Path(args.assignments_out) if args.assignments_out else out_dir / "no_anchor_sample_model_assignments.csv"
    model_out = Path(args.model_out) if args.model_out else out_dir / "no_anchor_sample_pair_model.joblib"
    global_model_out = Path(args.global_model_out) if args.global_model_out else out_dir / "no_anchor_sample_global_id_model.joblib"

    parquet_paths = _resolve_parquet_paths(sample_root, args.tracklet_parquet)
    feature_paths = _resolve_feature_paths(sample_root, args.feature_npz)
    feature_weights = _parse_weight_map(str(args.feature_key_weights))
    df = _load_sample_parquets(parquet_paths)
    records, tracklet_table, key_to_seq = _build_records(df, fps=float(args.fps))
    emb, feature_blocks = _feature_matrix_from_npzs(
        feature_paths,
        records,
        feature_keys=str(args.feature_keys),
        include_trajectory=bool(args.include_trajectory),
        feature_weights=feature_weights,
    )
    emb, feature_transform = _apply_nfc_transform(
        emb,
        records,
        k1=int(args.nfc_k1),
        k2=int(args.nfc_k2),
        eta=float(args.nfc_eta),
        exclude_same_camera=bool(args.nfc_exclude_same_camera),
    )
    pair_feature_views, pair_feature_names, pair_feature_meta = _load_pair_feature_views(
        list(args.pair_feature_npz),
        records,
    )
    global FEATURE_NAMES
    FEATURE_NAMES = list(BASE_FEATURE_NAMES) + list(pair_feature_names)
    _fit_model.__globals__["FEATURE_NAMES"] = FEATURE_NAMES
    _pseudo_training_pairs.__globals__["FEATURE_NAMES"] = FEATURE_NAMES
    _resolve_with_model.__globals__["FEATURE_NAMES"] = FEATURE_NAMES
    gt_by_seq, weight_by_seq, eval_info = _eval_labels(
        tracklet_table,
        min_gt_fraction=float(args.eval_min_gt_fraction),
        min_rows=int(args.eval_min_rows),
    )
    keep_seqs, output_info = _output_keep_seqs(
        tracklet_table,
        min_dets=int(args.output_min_dets),
        min_conf=float(args.output_min_conf),
        min_area=float(args.output_min_area),
    )
    print(
        json.dumps(
            {
                "stage": "loaded_sample",
                "parquets": [str(path) for path in parquet_paths],
                "feature_npz": [str(path) for path in feature_paths],
                "rows": int(len(df)),
                "tracklets": int(len(records)),
                "embedding_dim": int(emb.shape[1]),
                "feature_key_weights": feature_weights,
                "pair_feature_views": pair_feature_meta,
                "pair_feature_names": pair_feature_names,
                **feature_transform,
                **eval_info,
                **output_info,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    label_cache: dict[tuple[str, float, float, int], np.ndarray] = {}
    info_cache: dict[tuple[str, float, float, int], dict[str, object]] = {}
    best_pred_by_seq: dict[int, int] = {}
    best_labels: np.ndarray | None = None
    best_key: tuple[str, float, float, int] | None = None

    def add_status_metrics(row: dict[str, object], labels: np.ndarray) -> None:
        status_metrics, _component_meta, _seq_meta, _centroids = _resolution_status_metrics(
            labels,
            records=records,
            emb=emb,
            df=df,
            tracklet_table=tracklet_table,
            key_to_seq=key_to_seq,
            gt_by_seq=gt_by_seq,
            weight_by_seq=weight_by_seq,
            keep_seqs=keep_seqs,
            gt_col=str(args.gt_col),
            commit_min_dets=int(args.commit_min_dets),
            commit_min_conf=float(args.commit_min_conf),
            commit_min_area=float(args.commit_min_area),
            commit_min_component_size=int(args.commit_min_component_size),
            commit_min_member_sim=float(args.commit_min_member_sim),
            commit_min_margin=float(args.commit_min_margin),
            provisional_min_dets=int(args.provisional_min_dets),
            provisional_min_conf=float(args.provisional_min_conf),
        )
        row.update(status_metrics)

    for top_k in _parse_csv_ints(args.baseline_top_ks):
        for theta in _parse_csv_floats(args.baseline_thetas):
            cfg = ResolveConfig(
                mode="time_agglom",
                theta=float(theta),
                top_k=int(top_k),
                min_dets=int(args.min_dets),
                exclude_same=str(args.exclude_same),
                temporal_bonus=float(args.pseudo_temporal_bonus),
                time_window_ms=int(args.pseudo_time_window_ms),
                max_component_size=int(args.max_component_size),
            )
            labels, info = _time_agglom_resolve(records, emb, cfg)
            row, pred_by_seq, slices = _evaluate_labels(
                labels,
                mode="time_agglom",
                records=records,
                df=df,
                tracklet_table=tracklet_table,
                key_to_seq=key_to_seq,
                gt_by_seq=gt_by_seq,
                weight_by_seq=weight_by_seq,
                keep_seqs=keep_seqs,
                gt_col=str(args.gt_col),
            )
            row.update({"theta": float(theta), "top_k": int(top_k), **info, **output_info})
            row.update(feature_transform)
            add_status_metrics(row, labels)
            rows.append(row)
            key = ("time_agglom", float(theta), 0.0, int(top_k))
            label_cache[key] = labels
            info_cache[key] = info
            for slice_row in slices:
                slice_row.update({"mode": "time_agglom", "theta": float(theta), "top_k": int(top_k)})
                slice_rows.append(slice_row)
            print(json.dumps({"stage": "baseline", "theta": theta, "top_k": top_k, **row}, sort_keys=True), flush=True)

    for target_clusters in _parse_csv_ints(args.target_clusters):
        labels = _target_agglom_labels(emb, int(target_clusters))
        row, pred_by_seq, slices = _evaluate_labels(
            labels,
            mode="target_agglom",
            records=records,
            df=df,
            tracklet_table=tracklet_table,
            key_to_seq=key_to_seq,
            gt_by_seq=gt_by_seq,
            weight_by_seq=weight_by_seq,
            keep_seqs=keep_seqs,
            gt_col=str(args.gt_col),
        )
        row.update(
            {
                "target_clusters": int(target_clusters),
                "components": int(len(set(labels.tolist()))),
                "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
                **output_info,
                "uses_ground_truth": False,
                **feature_transform,
            }
        )
        add_status_metrics(row, labels)
        rows.append(row)
        key = ("target_agglom", float(target_clusters), 0.0, 0)
        label_cache[key] = labels
        info_cache[key] = {"target_clusters": int(target_clusters)}
        for slice_row in slices:
            slice_row.update({"mode": "target_agglom", "target_clusters": int(target_clusters)})
            slice_rows.append(slice_row)
        print(json.dumps({"stage": "target_agglom", "target_clusters": int(target_clusters), **row}, sort_keys=True), flush=True)

    pair_cfg = _pair_config_from_args(args)
    solvers = _parse_csv_strings(args.solvers)
    model = None
    pseudo_info: dict[str, object] = {"skipped": True, "reason": "no pair-model solvers requested"}
    train_info: dict[str, object] = {"skipped": True}
    if (not bool(args.skip_pair_model)) and solvers:
        X, y, pseudo_info = _pseudo_training_pairs(records, emb, {}, pair_cfg, pair_feature_views)
        print(json.dumps({"stage": "pseudo_training_pairs", **pseudo_info}, sort_keys=True), flush=True)
        model, train_info = _fit_model(X, y, pair_cfg)
        print(json.dumps({"stage": "fit_pair_model", **train_info}, sort_keys=True), flush=True)

        for solver in solvers:
            for threshold in _parse_csv_floats(args.thresholds):
                for blend in _parse_csv_floats(args.blends):
                    labels, info = _resolve_with_model(
                        records,
                        emb,
                        model,
                        pair_cfg,
                        threshold=float(threshold),
                        blend=float(blend),
                        solver=solver,
                        pair_feature_views=pair_feature_views,
                    )
                    row, pred_by_seq, slices = _evaluate_labels(
                        labels,
                        mode="pair_model",
                        records=records,
                        df=df,
                        tracklet_table=tracklet_table,
                        key_to_seq=key_to_seq,
                        gt_by_seq=gt_by_seq,
                        weight_by_seq=weight_by_seq,
                        keep_seqs=keep_seqs,
                        gt_col=str(args.gt_col),
                    )
                    row.update(
                        {
                            "solver": solver,
                            "threshold": float(threshold),
                            "blend": float(blend),
                            **asdict(pair_cfg),
                            **info,
                            **output_info,
                            **feature_transform,
                        }
                    )
                    add_status_metrics(row, labels)
                    rows.append(row)
                    key = (solver, float(threshold), float(blend), int(pair_cfg.infer_top_k))
                    label_cache[key] = labels
                    info_cache[key] = info
                    for slice_row in slices:
                        slice_row.update({"mode": "pair_model", "solver": solver, "threshold": float(threshold), "blend": float(blend)})
                        slice_rows.append(slice_row)
                    print(json.dumps({"stage": "pair_model", "solver": solver, "threshold": threshold, "blend": blend, **row}, sort_keys=True), flush=True)

    rows = _sort_rows(rows, str(args.sort_key))
    if rows:
        best = rows[0]
        if best.get("mode") == "time_agglom":
            best_key = ("time_agglom", float(best.get("theta", 0.0)), 0.0, int(best.get("top_k", 0)))
        elif best.get("mode") == "target_agglom":
            best_key = ("target_agglom", float(best.get("target_clusters", 0.0)), 0.0, 0)
        else:
            best_key = (str(best.get("solver")), float(best.get("threshold", 0.0)), float(best.get("blend", 0.0)), int(best.get("infer_top_k", pair_cfg.infer_top_k)))
        best_labels = label_cache.get(best_key)
        if best_labels is not None:
            best_pred_by_seq = _labels_to_seq_map(records, best_labels, keep_seqs=keep_seqs)
            status_metrics, component_meta, seq_meta, centroids = _resolution_status_metrics(
                best_labels,
                records=records,
                emb=emb,
                df=df,
                tracklet_table=tracklet_table,
                key_to_seq=key_to_seq,
                gt_by_seq=gt_by_seq,
                weight_by_seq=weight_by_seq,
                keep_seqs=keep_seqs,
                gt_col=str(args.gt_col),
                commit_min_dets=int(args.commit_min_dets),
                commit_min_conf=float(args.commit_min_conf),
                commit_min_area=float(args.commit_min_area),
                commit_min_component_size=int(args.commit_min_component_size),
                commit_min_member_sim=float(args.commit_min_member_sim),
                commit_min_margin=float(args.commit_min_margin),
                provisional_min_dets=int(args.provisional_min_dets),
                provisional_min_conf=float(args.provisional_min_conf),
            )
            best.update(status_metrics)
            assignment_info = _write_status_assignments(
                assignments_out,
                records,
                best_labels,
                keep_seqs=keep_seqs,
                seq_meta=seq_meta,
            )
            best.update(assignment_info)
            ordered_component_labels = sorted(centroids)
            component_centroids = (
                np.vstack([centroids[label] for label in ordered_component_labels]).astype(np.float32)
                if ordered_component_labels
                else np.zeros((0, int(emb.shape[1])), dtype=np.float32)
            )
            assignment_rows = [
                {
                    "seq": int(record.seq),
                    "tracklet_key": record.tracklet_key,
                    "predicted_global_id": int(30_000_000 + int(label)),
                    **seq_meta.get(int(record.seq), {}),
                }
                for record, label in zip(records, best_labels)
                if int(record.seq) in keep_seqs
            ]
            joblib.dump(
                {
                    "schema_version": 1,
                    "kind": "no_anchor_transductive_global_id_model",
                    "created_at": stamp,
                    "sample_root": str(sample_root),
                    "best_config": _jsonable(best),
                    "best_key": list(best_key) if best_key is not None else None,
                    "feature_blocks": feature_blocks,
                    "feature_transform": feature_transform,
                    "feature_keys": str(args.feature_keys),
                    "feature_key_weights": feature_weights,
                    "pair_feature_views": pair_feature_meta,
                    "include_trajectory": bool(args.include_trajectory),
                    "embedding_dim": int(emb.shape[1]),
                    "component_labels": np.asarray(ordered_component_labels, dtype=np.int64),
                    "component_centroids": component_centroids,
                    "component_meta": _jsonable(component_meta),
                    "labels_by_seq": {int(record.seq): int(label) for record, label in zip(records, best_labels)},
                    "assignments": _jsonable(assignment_rows),
                    "output_admission": output_info,
                    "resolution_policy": {
                        "commit_min_dets": int(args.commit_min_dets),
                        "commit_min_conf": float(args.commit_min_conf),
                        "commit_min_area": float(args.commit_min_area),
                        "commit_min_component_size": int(args.commit_min_component_size),
                        "commit_min_member_sim": float(args.commit_min_member_sim),
                        "commit_min_margin": float(args.commit_min_margin),
                        "provisional_min_dets": int(args.provisional_min_dets),
                        "provisional_min_conf": float(args.provisional_min_conf),
                    },
                    "uses_anchors": False,
                    "uses_gt_for_training_or_anchors": False,
                    "uses_gt_for_evaluation_only": True,
                },
                global_model_out,
            )
            best["global_model_out"] = str(global_model_out)

    if model is not None:
        joblib.dump(
            {
                "model": model,
                "pair_model_config": asdict(pair_cfg),
                "feature_names": FEATURE_NAMES,
                "feature_blocks": feature_blocks,
                "feature_key_weights": feature_weights,
                "pair_feature_views": pair_feature_meta,
                "pseudo_training": pseudo_info,
                "model_training": train_info,
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
            },
            model_out,
        )

    _write_csv(csv_out, rows)
    _write_csv(slice_csv_out, slice_rows)
    result = {
        "created_at": stamp,
        "sample_root": str(sample_root),
        "parquet_paths": [str(path) for path in parquet_paths],
        "feature_npz": [str(path) for path in feature_paths],
        "feature_keys": str(args.feature_keys),
        "feature_key_weights": feature_weights,
        "include_trajectory": bool(args.include_trajectory),
        "feature_blocks": feature_blocks,
        "feature_transform": feature_transform,
        "pair_feature_views": pair_feature_meta,
        "feature_names": FEATURE_NAMES,
        "n_rows": int(len(df)),
        "n_tracklets": int(len(records)),
        "embedding_dim": int(emb.shape[1]),
        "eval_stats": eval_info,
        "output_admission": output_info,
        "pair_model_config": asdict(pair_cfg),
        "pseudo_training": pseudo_info,
        "model_training": train_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "sort_key": str(args.sort_key),
        "top": rows[:20],
        "best_key": list(best_key) if best_key is not None else None,
        "best_predicted_tracklets": int(len(best_pred_by_seq)),
        "csv": str(csv_out),
        "slice_csv": str(slice_csv_out),
        "assignments_out": str(assignments_out),
        "model_out": str(model_out) if model is not None else None,
        "global_model_out": str(global_model_out) if best_labels is not None else None,
    }
    json_out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"stage": "done", "json": str(json_out), "csv": str(csv_out), "best": rows[0] if rows else None}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
