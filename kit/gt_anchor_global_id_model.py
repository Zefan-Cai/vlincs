#!/usr/bin/env python
"""Anchor-supervised global-ID model on DS1 GT boxes.

This is an upper-bound research driver, not a deployment path. It uses DS1
reference boxes as the tracklet source, fixes a budget of anchor tracklets with
known global IDs, trains a closed-world identity model, then scores the full
GT-box submission with the canonical VLINCS scorer.

The script deliberately separates:

* anchors: allowed supervised evidence nodes;
* non-anchors: the model/propagation target;
* GT boxes: detection upper bound;
* optional GT geo columns: trajectory upper bound unless an equivalent camera
  calibration pipeline supplies them from boxes at inference time.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from submit import _box_hash
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


_GT_ID_RE = re.compile(r"(\d+)$")
_CAMERAS = ("MCAM00", "MCAM03", "MCAM04", "MCAM05", "MCAM06", "MCAM08")
_CARDS = ("Tc6", "Tc8")


@dataclass(frozen=True)
class TrackletRecord:
    key: str
    video: str
    camera: str
    card: str
    source_id: str
    numeric_id: int
    tracklet_id: int
    n_rows: int
    start_frame: int
    end_frame: int
    feature: np.ndarray


@dataclass(frozen=True)
class LoadedTracklets:
    records: list[TrackletRecord]
    gt_by_video: dict[str, pd.DataFrame]
    source_by_video: dict[str, pd.DataFrame]
    input_kind: str
    uses_geo_features: bool
    label_stats: dict[str, object]


def _numeric_id(value: object, mapping: dict[str, int]) -> int:
    key = str(value)
    if key in mapping:
        return mapping[key]
    match = _GT_ID_RE.search(key)
    if match:
        candidate = int(match.group(1))
        if candidate > 0 and candidate not in mapping.values():
            mapping[key] = candidate
            return candidate
    mapping[key] = max(mapping.values(), default=0) + 1
    return mapping[key]


def _camera_from_video(video: str) -> str:
    return next((part for part in str(video).split("_") if part.startswith("MCAM")), "CAM")


def _card_from_video(video: str) -> str:
    return str(video).split("_")[-1]


def _canonical_video_key(video: str) -> str:
    return str(video).split("__", 1)[0]


def _safe_arr(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()


def _iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    px1, py1, px2, py2 = [pred_boxes[:, i][:, None] for i in range(4)]
    gx1, gy1, gx2, gy2 = [gt_boxes[:, i][None, :] for i in range(4)]
    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    p_area = np.maximum(px2 - px1, 0) * np.maximum(py2 - py1, 0)
    g_area = np.maximum(gx2 - gx1, 0) * np.maximum(gy2 - gy1, 0)
    return (inter / np.maximum(p_area + g_area - inter, 1e-9)).astype(np.float32)


def _add_iou_gt_labels(df: pd.DataFrame, gt: pd.DataFrame, iou_thr: float) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df.copy().reset_index(drop=True)
    labels = np.full(len(out), "", dtype=object)
    gt_by_frame = {int(frame): rows for frame, rows in gt.groupby("frame", sort=False)}
    matched = 0
    for _frame, pframe in out.groupby("frame_idx", sort=False):
        gframe = gt_by_frame.get(int(_frame))
        if gframe is None or gframe.empty:
            continue
        pboxes = pframe[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        gboxes = gframe[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        ious = _iou_matrix(pboxes, gboxes)
        best = ious.argmax(axis=1)
        best_iou = ious[np.arange(len(pframe)), best]
        gt_ids = gframe["id"].astype(str).to_numpy()
        for row_index, gt_index, score in zip(pframe.index.to_numpy(), best, best_iou):
            if float(score) >= iou_thr:
                labels[int(row_index)] = str(gt_ids[int(gt_index)])
                matched += 1
    out["gt_id"] = labels
    stats = {
        "iou_thr": float(iou_thr),
        "rows": int(len(out)),
        "matched_rows": int(matched),
        "matched_row_fraction": round(float(matched / max(len(out), 1)), 6),
        "label_source": "iou_majority_from_reference",
    }
    return out, stats


def _valid_label_series(series: pd.Series) -> pd.Series:
    clean = series.astype(str)
    lowered = clean.str.lower()
    return clean[(clean != "") & (lowered != "nan") & (lowered != "none") & (lowered != "unknown")]


def _stats(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [0.0] * 7
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return [
        float(np.nanmedian(arr)),
        float(np.nanmean(arr)),
        float(np.nanstd(arr)),
        float(np.nanmin(arr)),
        float(np.nanmax(arr)),
        float(arr[0]),
        float(arr[-1]),
    ]


def _one_hot(value: str, choices: Iterable[str]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _base_feature(group: pd.DataFrame, video: str, *, use_geo: bool) -> np.ndarray:
    g = group.sort_values("frame")
    x1, y1, x2, y2 = [_safe_arr(g[col]) for col in ("x1", "y1", "x2", "y2")]
    frames = _safe_arr(g["frame"])
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    width = np.maximum(x2 - x1, 1.0)
    height = np.maximum(y2 - y1, 1.0)
    area = width * height
    aspect = height / np.maximum(width, 1.0)
    camera = _camera_from_video(video)
    card = _card_from_video(video)
    feat: list[float] = [
        math.log1p(float(len(g))),
        float(frames[0]),
        float(frames[-1]),
        float(frames[-1] - frames[0] + 1.0),
        float((frames[0] + frames[-1]) * 0.5),
        float(cx[-1] - cx[0]),
        float(cy[-1] - cy[0]),
        float(width[-1] - width[0]),
        float(height[-1] - height[0]),
    ]
    for arr in (cx, cy, width, height, area, aspect):
        feat.extend(_stats(arr))
    feat.extend(_one_hot(camera, _CAMERAS))
    feat.extend(_one_hot(card, _CARDS))
    if use_geo:
        for col in ("lat", "lon", "depth", "alt"):
            if col in g.columns:
                feat.extend(_stats(_safe_arr(g[col])))
            else:
                feat.extend([0.0] * 7)
    return np.nan_to_num(np.asarray(feat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _sample_frames(frames: np.ndarray, n: int) -> list[int]:
    vals = sorted(set(int(x) for x in frames.tolist()))
    if not vals or n <= 0:
        return []
    if len(vals) <= n:
        return vals
    idx = np.linspace(0, len(vals) - 1, n).round().astype(int)
    return [vals[int(i)] for i in idx]


class CropFeatureReader:
    """Lazy OpenCV crop-hist extractor."""

    crop_dim = 82

    def __init__(
        self,
        video_root: str | None,
        samples: int,
        margin: float,
        *,
        cache_path: str | None = None,
        refresh_cache: bool = False,
    ) -> None:
        self.root = Path(video_root).expanduser() if video_root else None
        self.samples = int(samples)
        self.margin = float(margin)
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.refresh_cache = bool(refresh_cache)
        self._cache: dict[str, np.ndarray] = {}
        self._cache_dirty = False
        self._cv2 = None
        self._caps: dict[str, object] = {}
        self._paths: dict[str, str | None] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path is None or self.refresh_cache or not self.cache_path.exists():
            return
        data = np.load(self.cache_path, allow_pickle=True)
        keys = data["keys"].astype(str).tolist()
        features = data["features"].astype(np.float32)
        if len(keys) != len(features):
            raise ValueError(f"crop cache {self.cache_path} has {len(keys)} keys but {len(features)} features")
        self._cache = {key: features[i] for i, key in enumerate(keys)}

    def _save_cache(self) -> None:
        if self.cache_path is None or not self._cache_dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.asarray(sorted(self._cache), dtype=object)
        features = np.stack([self._cache[str(key)] for key in keys]).astype(np.float32) if len(keys) else np.zeros((0, self.crop_dim), dtype=np.float32)
        np.savez_compressed(
            self.cache_path,
            keys=keys,
            features=features,
            samples=np.asarray([self.samples], dtype=np.int32),
            margin=np.asarray([self.margin], dtype=np.float32),
            crop_dim=np.asarray([self.crop_dim], dtype=np.int32),
        )

    @property
    def enabled(self) -> bool:
        if self.root is None or self.samples <= 0:
            return False
        if self._cv2 is not None:
            return True
        try:
            import cv2  # type: ignore
        except Exception:
            return False
        self._cv2 = cv2
        return True

    def close(self) -> None:
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._caps.clear()
        self._save_cache()

    def _path(self, video: str) -> str | None:
        if video in self._paths:
            return self._paths[video]
        if self.root is None:
            self._paths[video] = None
            return None
        direct = self.root / f"{video}.mp4"
        if direct.exists():
            self._paths[video] = str(direct)
            return str(direct)
        hits = sorted(self.root.rglob(f"{video}.mp4"))
        path = str(hits[0]) if hits else None
        self._paths[video] = path
        return path

    def _cap(self, video: str):
        path = self._path(video)
        if not path:
            return None
        cap = self._caps.get(path)
        if cap is None:
            cap = self._cv2.VideoCapture(path)
            self._caps[path] = cap
        return cap

    def _read_crop(self, video: str, frame_idx: int, box: tuple[float, float, float, float]):
        if not self.enabled:
            return None
        cap = self._cap(video)
        if cap is None:
            return None
        cap.set(self._cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = [float(x) for x in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        x1 -= self.margin * bw
        x2 += self.margin * bw
        y1 -= self.margin * bh
        y2 += self.margin * bh
        xi1, yi1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
        xi2, yi2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
        if xi2 <= xi1 or yi2 <= yi1:
            return None
        return bgr[yi1:yi2, xi1:xi2]

    def feature(self, video: str, group: pd.DataFrame, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key].astype(np.float32)
        if not self.enabled:
            return np.zeros(self.crop_dim, dtype=np.float32) if self.samples > 0 else np.zeros(0, dtype=np.float32)
        g = group.sort_values("frame")
        by_frame = {int(row.frame): row for row in g.itertuples(index=False)}
        feats: list[np.ndarray] = []
        for frame in _sample_frames(g["frame"].astype(int).to_numpy(), self.samples):
            row = by_frame.get(int(frame))
            if row is None:
                continue
            crop = self._read_crop(video, int(frame), (row.x1, row.y1, row.x2, row.y2))
            if crop is None or crop.size == 0:
                continue
            parts = [crop[: max(1, crop.shape[0] // 2)], crop[max(1, crop.shape[0] // 2) :]]
            frame_feats = []
            for part in parts:
                if part.size == 0:
                    part = crop
                hsv = self._cv2.cvtColor(part, self._cv2.COLOR_BGR2HSV)
                h_hist = self._cv2.calcHist([hsv], [0], None, [16], [0, 180]).reshape(-1)
                s_hist = self._cv2.calcHist([hsv], [1], None, [8], [0, 256]).reshape(-1)
                v_hist = self._cv2.calcHist([hsv], [2], None, [8], [0, 256]).reshape(-1)
                hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
                hist /= float(hist.sum() + 1e-6)
                flat = part.reshape(-1, 3).astype(np.float32)
                color = np.concatenate([flat.mean(axis=0), flat.std(axis=0), np.median(flat, axis=0)])
                frame_feats.append(np.concatenate([hist, color.astype(np.float32)]))
            feats.append(np.concatenate(frame_feats))
        if not feats:
            feature = np.zeros(self.crop_dim, dtype=np.float32)
        else:
            feature = np.nan_to_num(np.mean(np.stack(feats), axis=0).astype(np.float32), nan=0.0)
        self._cache[key] = feature
        self._cache_dirty = self.cache_path is not None
        return feature


def _standard_source_rows(video: str, group: pd.DataFrame, key: str, tracklet_id: int) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "frame": group["frame"].astype(int).to_numpy(),
            "tracklet_key": key,
            "tracklet_id": int(tracklet_id),
            "x1": _safe_arr(group["x1"]),
            "y1": _safe_arr(group["y1"]),
            "x2": _safe_arr(group["x2"]),
            "y2": _safe_arr(group["y2"]),
            "object_type": 0,
            "confidence": _safe_arr(group["score"]) if "score" in group.columns else np.ones(len(group), dtype=np.float32),
        }
    )
    out["video"] = video
    return out


def _iter_tracklet_parquets(tracklets_root: str | Path) -> list[Path]:
    root = Path(tracklets_root).expanduser()
    if root.is_file():
        return [root]
    direct = root / "tracklets.parquet"
    if direct.exists():
        return [direct]
    return sorted(root.rglob("tracklets.parquet"))


def load_gt_records(
    *,
    use_geo: bool,
    video_root: str | None,
    crop_samples: int,
    crop_margin: float,
    crop_cache: str | None,
    refresh_crop_cache: bool,
) -> LoadedTracklets:
    gt_by_video = load_ds1_gt_by_video()
    id_map: dict[str, int] = {}
    reader = CropFeatureReader(
        video_root,
        crop_samples,
        crop_margin,
        cache_path=crop_cache,
        refresh_cache=refresh_crop_cache,
    )
    records: list[TrackletRecord] = []
    source_chunks: dict[str, list[pd.DataFrame]] = {}
    n_rows = 0
    try:
        for video, df in sorted(gt_by_video.items()):
            camera = _camera_from_video(video)
            card = _card_from_video(video)
            for (gid, tracklet_id), group in df.groupby(["id", "tracklet_id"], sort=True):
                group = group.sort_values("frame")
                key = f"{video}::tracklet:{int(tracklet_id)}"
                base = _base_feature(group, video, use_geo=use_geo)
                crop = reader.feature(video, group, key)
                feature = np.concatenate([base, crop]).astype(np.float32)
                source_id = str(gid)
                numeric = _numeric_id(source_id, id_map)
                records.append(
                    TrackletRecord(
                        key=key,
                        video=video,
                        camera=camera,
                        card=card,
                        source_id=source_id,
                        numeric_id=numeric,
                        tracklet_id=int(tracklet_id),
                        n_rows=int(len(group)),
                        start_frame=int(group["frame"].min()),
                        end_frame=int(group["frame"].max()),
                        feature=feature,
                    )
                )
                source_chunks.setdefault(video, []).append(_standard_source_rows(video, group, key, int(tracklet_id)))
                n_rows += int(len(group))
    finally:
        reader.close()
    source_by_video = {video: pd.concat(chunks, ignore_index=True) for video, chunks in source_chunks.items()}
    return LoadedTracklets(
        records=records,
        gt_by_video=gt_by_video,
        source_by_video=source_by_video,
        input_kind="reference_gt_dataframe",
        uses_geo_features=bool(use_geo),
        label_stats={
            "label_source": "reference_gt_dataframe",
            "rows": int(n_rows),
            "matched_rows": int(n_rows),
            "matched_row_fraction": 1.0,
            "tracklets": int(len(records)),
            "labeled_tracklets": int(len(records)),
            "quarantined_tracklets": 0,
        },
    )


def load_tracklet_root_records(
    tracklets_root: str | Path,
    *,
    use_geo: bool,
    video_root: str | None,
    crop_samples: int,
    crop_margin: float,
    crop_cache: str | None,
    refresh_crop_cache: bool,
    tracklet_label_iou_thr: float,
    min_tracklet_label_fraction: float,
) -> LoadedTracklets:
    files = _iter_tracklet_parquets(tracklets_root)
    if not files:
        raise FileNotFoundError(f"no tracklets.parquet files found under {tracklets_root}")
    gt_by_video_all = load_ds1_gt_by_video()
    id_map: dict[str, int] = {}
    reader = CropFeatureReader(
        video_root,
        crop_samples,
        crop_margin,
        cache_path=crop_cache,
        refresh_cache=refresh_crop_cache,
    )
    records: list[TrackletRecord] = []
    source_chunks: dict[str, list[pd.DataFrame]] = {}
    uses_geo = False
    label_stats: dict[str, object] = {
        "label_source": "unknown",
        "iou_thr": float(tracklet_label_iou_thr),
        "min_tracklet_label_fraction": float(min_tracklet_label_fraction),
        "files": int(len(files)),
        "rows": 0,
        "matched_rows": 0,
        "matched_row_fraction": 0.0,
        "tracklets": 0,
        "labeled_tracklets": 0,
        "quarantined_tracklets": 0,
        "quarantine_reasons": {},
    }
    label_sources: set[str] = set()

    def add_quarantine(reason: str) -> None:
        label_stats["quarantined_tracklets"] = int(label_stats["quarantined_tracklets"]) + 1
        reasons = dict(label_stats["quarantine_reasons"])
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        label_stats["quarantine_reasons"] = reasons

    try:
        for pq in files:
            df = pd.read_parquet(pq)
            if "frame_idx" not in df.columns:
                raise ValueError(f"{pq} is missing frame_idx")
            raw_video = str(df["video_key"].iloc[0]) if "video_key" in df.columns else pq.parent.name
            video = _canonical_video_key(raw_video)
            if video not in gt_by_video_all:
                continue
            if "gt_id" not in df.columns:
                df, stats = _add_iou_gt_labels(df, gt_by_video_all[video], tracklet_label_iou_thr)
                label_sources.add(str(stats["label_source"]))
                label_stats["matched_rows"] = int(label_stats["matched_rows"]) + int(stats["matched_rows"])
            else:
                df = df.copy().reset_index(drop=True)
                valid = _valid_label_series(df["gt_id"])
                label_sources.add("gt_id_column")
                label_stats["matched_rows"] = int(label_stats["matched_rows"]) + int(len(valid))
            label_stats["rows"] = int(label_stats["rows"]) + int(len(df))
            if "tracklet_key" not in df.columns:
                if "local_track_id" not in df.columns:
                    raise ValueError(f"{pq} has neither tracklet_key nor local_track_id")
                df = df.copy()
                df["tracklet_key"] = [f"{raw_video}:local:{int(x)}" for x in df["local_track_id"]]
            camera = _camera_from_video(video)
            card = _card_from_video(video)
            for key, group in df.groupby("tracklet_key", sort=True):
                label_stats["tracklets"] = int(label_stats["tracklets"]) + 1
                valid_labels = _valid_label_series(group["gt_id"])
                if valid_labels.empty:
                    add_quarantine("no_iou_or_gt_label")
                    continue
                counts = valid_labels.value_counts(dropna=True)
                source_id = str(counts.index[0])
                majority_fraction = float(counts.iloc[0] / max(len(group), 1))
                if majority_fraction < min_tracklet_label_fraction:
                    add_quarantine("low_majority_fraction")
                    continue
                group = group.sort_values("frame_idx").rename(columns={"frame_idx": "frame"})
                tracklet_id = int(group["local_track_id"].iloc[0]) if "local_track_id" in group.columns else len(records)
                numeric = _numeric_id(source_id, id_map)
                group_uses_geo = bool(use_geo)
                uses_geo = uses_geo or group_uses_geo
                base = _base_feature(group, video, use_geo=group_uses_geo)
                crop = reader.feature(video, group, str(key))
                feature = np.concatenate([base, crop]).astype(np.float32)
                records.append(
                    TrackletRecord(
                        key=str(key),
                        video=video,
                        camera=camera,
                        card=card,
                        source_id=source_id,
                        numeric_id=numeric,
                        tracklet_id=tracklet_id,
                        n_rows=int(len(group)),
                        start_frame=int(group["frame"].min()),
                        end_frame=int(group["frame"].max()),
                        feature=feature,
                    )
                )
                source_chunks.setdefault(video, []).append(_standard_source_rows(video, group, str(key), tracklet_id))
                label_stats["labeled_tracklets"] = int(label_stats["labeled_tracklets"]) + 1
    finally:
        reader.close()
    source_by_video = {video: pd.concat(chunks, ignore_index=True) for video, chunks in source_chunks.items()}
    gt_by_video = {key: value for key, value in gt_by_video_all.items() if key in source_by_video}
    total_rows = int(label_stats["rows"])
    label_stats["matched_row_fraction"] = round(float(int(label_stats["matched_rows"]) / max(total_rows, 1)), 6)
    label_stats["label_source"] = "+".join(sorted(label_sources)) if label_sources else "none"
    return LoadedTracklets(
        records=records,
        gt_by_video=gt_by_video,
        source_by_video=source_by_video,
        input_kind="tracklets_root",
        uses_geo_features=uses_geo,
        label_stats=label_stats,
    )


def load_records(
    *,
    tracklets_root: str | None,
    use_geo: bool,
    video_root: str | None,
    crop_samples: int,
    crop_margin: float,
    crop_cache: str | None,
    refresh_crop_cache: bool,
    tracklet_label_iou_thr: float,
    min_tracklet_label_fraction: float,
) -> LoadedTracklets:
    if tracklets_root:
        return load_tracklet_root_records(
            tracklets_root,
            use_geo=use_geo,
            video_root=video_root,
            crop_samples=crop_samples,
            crop_margin=crop_margin,
            crop_cache=crop_cache,
            refresh_crop_cache=refresh_crop_cache,
            tracklet_label_iou_thr=tracklet_label_iou_thr,
            min_tracklet_label_fraction=min_tracklet_label_fraction,
        )
    return load_gt_records(
        use_geo=use_geo,
        video_root=video_root,
        crop_samples=crop_samples,
        crop_margin=crop_margin,
        crop_cache=crop_cache,
        refresh_crop_cache=refresh_crop_cache,
    )


def select_anchors(records: list[TrackletRecord], budget: int, strategy: str) -> np.ndarray:
    if budget <= 0:
        raise ValueError("anchor budget must be positive")
    budget = min(int(budget), len(records))
    weights = np.asarray([record.n_rows for record in records], dtype=np.float64)
    labels = np.asarray([record.numeric_id for record in records], dtype=np.int64)
    cameras = np.asarray([record.camera for record in records], dtype=object)
    order = np.argsort(-weights)
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(index: int) -> None:
        if len(selected) < budget and int(index) not in selected_set:
            selected.append(int(index))
            selected_set.add(int(index))

    # Every known global identity must have at least one supervised seed.
    for label in sorted(set(labels.tolist())):
        candidates = [i for i in order.tolist() if int(labels[i]) == int(label)]
        if candidates:
            add(candidates[0])

    if strategy in {"active_longest", "identity_camera"}:
        for label in sorted(set(labels.tolist())):
            for camera in _CAMERAS:
                candidates = [i for i in order.tolist() if int(labels[i]) == int(label) and str(cameras[i]) == camera]
                if candidates:
                    add(candidates[0])

    if strategy in {"active_longest", "longest", "identity_camera"}:
        for index in order.tolist():
            add(index)
            if len(selected) >= budget:
                break
    elif strategy == "round_robin":
        by_label: dict[int, list[int]] = {}
        for index in order.tolist():
            by_label.setdefault(int(labels[index]), []).append(int(index))
        while len(selected) < budget:
            before = len(selected)
            for label in sorted(by_label):
                if by_label[label]:
                    add(by_label[label].pop(0))
                if len(selected) >= budget:
                    break
            if len(selected) == before:
                break
    else:
        raise ValueError(f"unknown anchor strategy {strategy!r}")
    return np.asarray(selected, dtype=np.int64)


def _make_model(name: str, seed: int):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "knn1":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1, weights="distance"))
    if name == "knn5":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=800,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=1,
        )
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced"),
        )
    if name == "soft_vote":
        return VotingClassifier(
            estimators=[
                ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))),
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        random_state=seed,
                        n_jobs=-1,
                        class_weight="balanced",
                        min_samples_leaf=1,
                    ),
                ),
                (
                    "lr",
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced"),
                    ),
                ),
            ],
            voting="soft",
            weights=[1.0, 1.5, 0.8],
        )
    raise ValueError(f"unknown model {name!r}")


def _topk_from_proba(model, x: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        kk = min(int(k), proba.shape[1])
        top = np.argpartition(-proba, kk - 1, axis=1)[:, :kk]
        order = np.take_along_axis(proba, top, axis=1).argsort(axis=1)[:, ::-1]
        return classes[np.take_along_axis(top, order, axis=1)]
    pred = model.predict(x)
    return pred[:, None]


def _prediction_confidence(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.max(model.predict_proba(x), axis=1)
    return np.ones(len(x), dtype=np.float32)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, anchors: np.ndarray, topk: dict[int, np.ndarray]) -> dict[str, object]:
    from sklearn.metrics import precision_recall_fscore_support

    anchor_mask = np.zeros(len(y_true), dtype=bool)
    anchor_mask[anchors] = True
    non = ~anchor_mask

    def acc(mask: np.ndarray, weighted: bool) -> float:
        if not np.any(mask):
            return 0.0
        good = y_true[mask] == y_pred[mask]
        if weighted:
            return float(weights[mask][good].sum() / max(weights[mask].sum(), 1e-9))
        return float(np.mean(good))

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out: dict[str, object] = {
        "tracklet_accuracy": round(acc(np.ones(len(y_true), dtype=bool), False), 6),
        "row_weighted_tracklet_accuracy": round(acc(np.ones(len(y_true), dtype=bool), True), 6),
        "non_anchor_tracklet_accuracy": round(acc(non, False), 6),
        "non_anchor_row_weighted_accuracy": round(acc(non, True), 6),
        "macro_precision": round(float(p), 6),
        "macro_recall": round(float(r), 6),
        "macro_f1": round(float(f1), 6),
    }
    for k, top in topk.items():
        hit = np.array([y_true[i] in set(top[i].tolist()) for i in range(len(y_true))], dtype=bool)
        out[f"candidate_recall_at_{k}"] = round(float(hit.mean()), 6)
        out[f"non_anchor_candidate_recall_at_{k}"] = round(float(hit[non].mean()) if np.any(non) else 0.0, 6)
    return out


def build_comp(source_by_video: dict[str, pd.DataFrame], records: list[TrackletRecord], y_pred_numeric: np.ndarray) -> dict[str, pd.DataFrame]:
    pred_of: dict[str, int] = {}
    for record, pred in zip(records, y_pred_numeric):
        pred_of[record.key] = int(pred)
    comp: dict[str, pd.DataFrame] = {}
    for video, source in source_by_video.items():
        out = source[["frame", "tracklet_key", "tracklet_id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
        out["id"] = [
            pred_of.get(str(tracklet_key), 99_000_000 + int(tracklet_id))
            for tracklet_key, tracklet_id in zip(out["tracklet_key"], out["tracklet_id"])
        ]
        comp[video] = out[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
    return comp


def export_zip(comp: dict[str, pd.DataFrame], out_zip: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="vlincs_gt_anchor_submit_"))
    written = []
    for video, df in comp.items():
        out = df.copy()
        out["frame"] = out["frame"].astype("uint32")
        out["id"] = out["id"].astype("uint32")
        for col in ("x1", "y1", "x2", "y2"):
            out[col] = out[col].clip(lower=0).astype("uint32")
        out["object_type"] = 0
        out["object_type"] = out["object_type"].astype("uint8")
        out["confidence"] = out["confidence"].astype("float32")
        out["box_hash"] = [_box_hash(r.x1, r.y1, r.x2, r.y2) for r in out.itertuples()]
        for col in ("lat", "long", "alt"):
            out[col] = np.float64("nan")
        path = tmp / f"{video}.parquet"
        out[["frame", "id", "x1", "y1", "x2", "y2", "box_hash", "object_type", "confidence", "lat", "long", "alt"]].to_parquet(path)
        written.append(path)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in written:
            zf.write(path, path.name)


def write_csv(path: str | None, records: list[TrackletRecord], y_pred: np.ndarray, confidence: np.ndarray, anchor_mask: np.ndarray) -> None:
    if not path:
        return
    rows = []
    for i, record in enumerate(records):
        rows.append(
            {
                "tracklet_key": record.key,
                "video": record.video,
                "camera": record.camera,
                "tracklet_id": record.tracklet_id,
                "gt_global_id": record.source_id,
                "gt_numeric_id": record.numeric_id,
                "pred_numeric_id": int(y_pred[i]),
                "is_anchor": bool(anchor_mask[i]),
                "confidence": round(float(confidence[i]), 6),
                "n_rows": record.n_rows,
                "start_frame": record.start_frame,
                "end_frame": record.end_frame,
                "correct": bool(int(y_pred[i]) == int(record.numeric_id)),
            }
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def write_model_bundle(
    path: str | None,
    *,
    model,
    records: list[TrackletRecord],
    anchors: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    result: dict[str, object],
) -> None:
    if not path:
        return
    import joblib

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anchor_set = set(int(x) for x in anchors.tolist())
    record_rows = []
    for i, record in enumerate(records):
        record_rows.append(
            {
                "index": int(i),
                "tracklet_key": record.key,
                "video": record.video,
                "camera": record.camera,
                "tracklet_id": int(record.tracklet_id),
                "source_global_id": record.source_id,
                "source_numeric_id": int(y_true[i]),
                "pred_numeric_id": int(y_pred[i]),
                "confidence": float(confidence[i]),
                "is_anchor": int(i) in anchor_set,
                "n_rows": int(record.n_rows),
                "start_frame": int(record.start_frame),
                "end_frame": int(record.end_frame),
            }
        )
    bundle = {
        "model": model,
        "result": result,
        "anchors": anchors.astype(np.int64),
        "records": record_rows,
    }
    joblib.dump(bundle, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracklets-root", default=None, help="optional tracklets.parquet root; gt_id is used only for anchors/eval")
    ap.add_argument("--tracklet-label-iou-thr", type=float, default=0.5, help="IoU threshold for labeling unlabeled tracklet-root rows from reference boxes")
    ap.add_argument("--min-tracklet-label-fraction", type=float, default=0.5, help="quarantine tracklets whose majority GT label covers less than this fraction of rows")
    ap.add_argument("--anchor-budget", type=int, default=600)
    ap.add_argument("--anchor-strategy", choices=["active_longest", "longest", "identity_camera", "round_robin"], default="active_longest")
    ap.add_argument("--model", choices=["knn1", "knn5", "extra_trees", "random_forest", "logreg", "soft_vote"], default="soft_vote")
    ap.add_argument("--no-geo", action="store_true", help="disable GT lat/lon/depth/alt trajectory features")
    ap.add_argument("--video-root", default=None, help="optional root containing DS1 MP4s for crop-color features")
    ap.add_argument("--crop-samples", type=int, default=0, help="sampled frames per tracklet for crop hist features")
    ap.add_argument("--crop-margin", type=float, default=0.05)
    ap.add_argument("--crop-cache", default=None, help="optional NPZ cache for per-tracklet crop-color features")
    ap.add_argument("--refresh-crop-cache", action="store_true", help="ignore and overwrite an existing crop cache")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--submit", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pred-csv", default=None)
    ap.add_argument("--model-out", default=None, help="optional joblib bundle for the fitted identity classifier and provenance")
    ap.add_argument("--skip-hota", action="store_true", help="only run classification/candidate metrics")
    args = ap.parse_args()

    np.random.seed(args.seed)
    loaded = load_records(
        tracklets_root=args.tracklets_root,
        use_geo=not args.no_geo,
        video_root=args.video_root,
        crop_samples=args.crop_samples,
        crop_margin=args.crop_margin,
        crop_cache=args.crop_cache,
        refresh_crop_cache=args.refresh_crop_cache,
        tracklet_label_iou_thr=args.tracklet_label_iou_thr,
        min_tracklet_label_fraction=args.min_tracklet_label_fraction,
    )
    records = loaded.records
    gt_by_video = loaded.gt_by_video
    source_by_video = loaded.source_by_video
    if not records:
        raise SystemExit("no GT tracklet records loaded; check DATA_ROOT")
    x = np.stack([record.feature for record in records]).astype(np.float32)
    y = np.asarray([record.numeric_id for record in records], dtype=np.int64)
    weights = np.asarray([record.n_rows for record in records], dtype=np.float64)

    anchors = select_anchors(records, args.anchor_budget, args.anchor_strategy)
    anchor_mask = np.zeros(len(records), dtype=bool)
    anchor_mask[anchors] = True
    model = _make_model(args.model, args.seed)
    model.fit(x[anchors], y[anchors])
    y_pred = np.asarray(model.predict(x), dtype=np.int64)
    confidence = _prediction_confidence(model, x)
    # Anchors are fixed evidence nodes.
    y_pred[anchors] = y[anchors]
    confidence[anchors] = 1.0
    classes = np.asarray(getattr(model, "classes_", sorted(set(y.tolist()))), dtype=np.int64)
    topk = {k: _topk_from_proba(model, x, classes, k) for k in (1, 3, 5)}
    tracklets_root_text = str(args.tracklets_root or "")
    uses_gt_boxes = loaded.input_kind == "reference_gt_dataframe" or "from_ground_truth" in tracklets_root_text

    result = {
        "dataset": "ds1",
        "method": "anchor_supervised_global_id_model",
        "input_kind": loaded.input_kind,
        "tracklets_root": args.tracklets_root,
        "model": args.model,
        "anchor_budget": int(args.anchor_budget),
        "anchor_strategy": args.anchor_strategy,
        "n_tracklets": len(records),
        "n_global_ids": int(len(set(y.tolist()))),
        "n_anchors": int(len(anchors)),
        "anchor_tracklet_fraction": round(float(len(anchors) / len(records)), 6),
        "anchor_row_fraction": round(float(weights[anchors].sum() / weights.sum()), 6),
        "feature_dim": int(x.shape[1]),
        "uses_gt_boxes": bool(uses_gt_boxes),
        "box_source_hint": "reference_or_from_ground_truth" if uses_gt_boxes else "external_tracklet_root",
        "uses_gt_identity_for_anchors": True,
        "uses_gt_identity_for_all": False,
        "uses_tracklet_gt_id_column": "gt_id_column" in str(loaded.label_stats.get("label_source", "")),
        "uses_reference_iou_labels": "iou_majority_from_reference" in str(loaded.label_stats.get("label_source", "")),
        "tracklet_label_iou_thr": float(args.tracklet_label_iou_thr),
        "min_tracklet_label_fraction": float(args.min_tracklet_label_fraction),
        "label_stats": loaded.label_stats,
        "uses_gt_geo_features": bool(loaded.uses_geo_features),
        "uses_video_crop_features": bool((args.video_root or args.crop_cache) and args.crop_samples > 0),
        "crop_cache": args.crop_cache,
        "crop_samples": int(args.crop_samples),
        "classification": _classification_metrics(y, y_pred, weights, anchors, topk),
    }
    comp = None
    if not args.skip_hota:
        comp = build_comp(source_by_video, records, y_pred)
        metrics = evaluate(gt_by_video, comp, dense=False, n_workers=1)
        result.update(
            {
                "idf1": round(metrics.idf1, 6),
                "hota": round(metrics.hota, 6),
                "assa": round(metrics.assa, 6),
                "deta": round(metrics.deta, 6),
                "detre": round(metrics.detre, 6),
                "detpr": round(metrics.detpr, 6),
                "unmatched_fp": int(metrics.unmatched_fp),
                "per_video": {
                    key: {metric: round(float(value), 6) for metric, value in vals.items()}
                    for key, vals in sorted(metrics.per_video.items())
                },
            }
        )
    if args.submit:
        if comp is None:
            comp = build_comp(source_by_video, records, y_pred)
        export_zip(comp, args.submit)
        result["submission"] = args.submit
    if args.pred_csv:
        write_csv(args.pred_csv, records, y_pred, confidence, anchor_mask)
        result["pred_csv"] = args.pred_csv
    if args.model_out:
        write_model_bundle(
            args.model_out,
            model=model,
            records=records,
            anchors=anchors,
            y_true=y,
            y_pred=y_pred,
            confidence=confidence,
            result=result,
        )
        result["model_out"] = args.model_out
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
