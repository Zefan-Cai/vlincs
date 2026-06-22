"""Generate no-GT weak labels from tracklet boxes and optional video crops.

The output is a CSV keyed by ``tracklet_key`` that can be fed directly into
``kit/cli.py import-weak-labels`` or ``demo --weak-label-csv``. This module is
deliberately label-free: it never reads reference annotations or identity
columns, only tracker boxes, timing, detector confidence, and optional RGB crop
statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import colorsys
import csv
import glob
import json
import math
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeakLabelGenerationConfig:
    """Knobs for bbox/crop weak-label extraction."""

    sample_frames: int = 3
    min_tracklet_dets: int = 1
    max_tracklets: int | None = None
    include_crop_colors: bool = True
    crop_margin: float = 0.05


def _camera_from_video(video: str) -> str:
    return next((part for part in str(video).split("_") if part.startswith("MCAM")), "CAM")


def _token_payload(tokens: dict[str, str | int | float]) -> str:
    clean = {
        str(k): str(v)
        for k, v in tokens.items()
        if v is not None and str(v) and str(v).lower() not in {"nan", "none", "unknown"}
    }
    return json.dumps(clean, sort_keys=True)


def _bucket(value: float, cuts: list[tuple[float, str]], default: str) -> str:
    for cutoff, name in cuts:
        if value < cutoff:
            return name
    return default


def _motion_bucket(delta: float, scale: float) -> str:
    if scale <= 1e-6 or abs(delta) / scale < 0.04:
        return "stable"
    return "right" if delta > 0 else "left"


def _vertical_motion_bucket(delta: float, scale: float) -> str:
    if scale <= 1e-6 or abs(delta) / scale < 0.04:
        return "stable"
    return "down" if delta > 0 else "up"


def _rgb_color_name(rgb: np.ndarray) -> str:
    """Map a median RGB color to a coarse clothing-like color name."""
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return "unknown"
    r, g, b = [float(np.clip(x, 0, 255)) for x in arr[:3]]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 45:
        return "black"
    if mx > 220 and mx - mn < 28:
        return "white"
    if mx - mn < 30:
        return "gray"
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue = h * 360.0
    if 20 <= hue < 50 and v < 0.65:
        return "brown"
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 75:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 250:
        return "blue"
    if hue < 310:
        return "purple"
    return "pink" if s < 0.55 or v > 0.65 else "red"


def _sample_frame_indices(frames: list[int], n: int) -> list[int]:
    if not frames:
        return []
    if n <= 1:
        return [int(frames[len(frames) // 2])]
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    return [int(frames[i]) for i in sorted(set(idx.tolist()))]


class _VideoFrames:
    """Lazy OpenCV frame reader; absent when cv2 or the source video is unavailable."""

    def __init__(self, video_root: str | None):
        self.video_root = Path(video_root).expanduser() if video_root else None
        self._cv2 = None
        self._path_cache: dict[str, str | None] = {}
        self._cap_cache: dict[str, object] = {}

    @property
    def available(self) -> bool:
        if self.video_root is None:
            return False
        if self._cv2 is not None:
            return True
        try:
            import cv2  # type: ignore
        except Exception:
            return False
        self._cv2 = cv2
        return True

    def _path(self, video: str) -> str | None:
        if video in self._path_cache:
            return self._path_cache[video]
        if self.video_root is None:
            self._path_cache[video] = None
            return None
        direct = self.video_root / f"{video}.mp4"
        if direct.exists():
            self._path_cache[video] = str(direct)
            return str(direct)
        parts = str(video).split("_")
        candidates = []
        if len(parts) >= 5:
            site, cluster = parts[1], parts[2]
            card = "_".join(parts[4:])
            candidates.append(self.video_root / site / cluster / card / f"{video}.mp4")
        for cand in candidates:
            if cand.exists():
                self._path_cache[video] = str(cand)
                return str(cand)
        hits = glob.glob(str(self.video_root / "**" / f"{video}.mp4"), recursive=True)
        path = hits[0] if hits else None
        self._path_cache[video] = path
        return path

    def read(self, video: str, frame_idx: int):
        if not self.available:
            return None
        path = self._path(video)
        if not path:
            return None
        cap = self._cap_cache.get(path)
        if cap is None:
            cap = self._cv2.VideoCapture(path)
            self._cap_cache[path] = cap
        n = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame = max(0, int(frame_idx))
        if n > 0:
            frame = min(frame, n - 1)
        cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None
        return bgr[:, :, ::-1]

    def close(self) -> None:
        for cap in self._cap_cache.values():
            try:
                cap.release()
            except Exception:
                pass
        self._cap_cache.clear()


def _crop_rgb(frame, box, margin: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= margin * bw
    x2 += margin * bw
    y1 -= margin * bh
    y2 += margin * bh
    xi1, yi1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    xi2, yi2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return frame[yi1:yi2, xi1:xi2]


def _crop_color_tokens(video: str, group: pd.DataFrame, reader: _VideoFrames, cfg: WeakLabelGenerationConfig) -> dict[str, str]:
    if not reader.available:
        return {}
    frames = group["frame_idx"].astype(int).tolist()
    if not frames:
        return {}
    by_frame = {int(r.frame_idx): r for r in group.itertuples(index=False)}
    upper_rgbs, lower_rgbs = [], []
    for frame_idx in _sample_frame_indices(frames, cfg.sample_frames):
        row = by_frame.get(frame_idx)
        if row is None:
            continue
        frame = reader.read(video, frame_idx)
        if frame is None:
            continue
        box = [row.x1, row.y1, row.x2, row.y2]
        crop = _crop_rgb(frame, box, cfg.crop_margin)
        if crop is None or crop.size == 0:
            continue
        mid = max(1, crop.shape[0] // 2)
        upper = crop[:mid]
        lower = crop[mid:] if crop.shape[0] > mid else crop
        upper_rgbs.append(np.median(upper.reshape(-1, 3), axis=0))
        lower_rgbs.append(np.median(lower.reshape(-1, 3), axis=0))
    if not upper_rgbs:
        return {}
    return {
        "upper_color": _rgb_color_name(np.median(np.stack(upper_rgbs), axis=0)),
        "lower_color": _rgb_color_name(np.median(np.stack(lower_rgbs), axis=0)),
    }


def _bbox_tokens(group: pd.DataFrame) -> tuple[dict[str, str], dict[str, float | int]]:
    g = group.sort_values("frame_idx")
    x1 = g["x1"].to_numpy(float)
    y1 = g["y1"].to_numpy(float)
    x2 = g["x2"].to_numpy(float)
    y2 = g["y2"].to_numpy(float)
    widths = np.maximum(1.0, x2 - x1)
    heights = np.maximum(1.0, y2 - y1)
    areas = widths * heights
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    frame_span = int(g["frame_idx"].max() - g["frame_idx"].min() + 1)
    mean_h = float(np.median(heights))
    mean_area = float(np.median(areas))
    aspect = float(np.median(heights / np.maximum(widths, 1e-6)))
    conf = float(np.nanmedian(g["score"].to_numpy(float))) if "score" in g else 0.0
    scale_x = float(np.nanmax(cx) - np.nanmin(cx) + np.nanmedian(widths))
    scale_y = float(np.nanmax(cy) - np.nanmin(cy) + np.nanmedian(heights))
    tokens = {
        "bbox_height": _bucket(mean_h, [(80, "tiny"), (150, "small"), (260, "medium"), (420, "large")], "xlarge"),
        "bbox_area": _bucket(mean_area, [(6_000, "tiny"), (18_000, "small"), (45_000, "medium"), (90_000, "large")], "xlarge"),
        "bbox_aspect": _bucket(aspect, [(1.7, "wide"), (2.5, "body"), (3.4, "tall")], "very_tall"),
        "track_duration": _bucket(frame_span, [(8, "very_short"), (24, "short"), (90, "medium"), (240, "long")], "very_long"),
        "motion_x": _motion_bucket(float(cx[-1] - cx[0]), scale_x),
        "motion_y": _vertical_motion_bucket(float(cy[-1] - cy[0]), scale_y),
        "det_conf": _bucket(conf, [(0.35, "low"), (0.55, "medium"), (0.75, "high")], "very_high"),
    }
    stats = {
        "start_frame": int(g["frame_idx"].min()),
        "end_frame": int(g["frame_idx"].max()),
        "n_dets": int(len(g)),
        "confidence": round(float(conf), 4),
        "mean_box_area": round(mean_area, 2),
        "mean_box_height": round(mean_h, 2),
    }
    return tokens, stats


def weak_tokens_from_tracklet(frames, boxes, confs=None) -> dict[str, str]:
    """Return no-GT bbox/time weak tokens from one in-memory tracklet.

    This is the same bbox-only evidence used by ``generate_weak_labels``. It is
    intentionally identity-agnostic and does not look at tracklet IDs, reference
    labels, or previous assignments.
    """
    if len(frames) != len(boxes):
        raise ValueError("frames and boxes must have the same length")
    data = {
        "frame_idx": [int(x) for x in frames],
        "x1": [float(b[0]) for b in boxes],
        "y1": [float(b[1]) for b in boxes],
        "x2": [float(b[2]) for b in boxes],
        "y2": [float(b[3]) for b in boxes],
    }
    if confs is not None and len(confs) == len(frames):
        data["score"] = [float(x) for x in confs]
    tokens, _stats = _bbox_tokens(pd.DataFrame(data))
    return tokens


def iter_tracklet_parquets(tracklets_root: str | Path) -> list[Path]:
    """Return tracklet parquet files from a file, a video dir, or a dataset root."""
    root = Path(tracklets_root).expanduser()
    if root.is_file():
        return [root]
    direct = root / "tracklets.parquet"
    if direct.exists():
        return [direct]
    return sorted(root.rglob("tracklets.parquet"))


def generate_weak_labels(
    tracklets_root: str | Path,
    out_csv: str | Path,
    *,
    video_root: str | None = None,
    cfg: WeakLabelGenerationConfig | None = None,
) -> dict:
    """Generate a weak-label CSV from tracker boxes and optional video crops."""
    cfg = cfg or WeakLabelGenerationConfig()
    files = iter_tracklet_parquets(tracklets_root)
    if not files:
        raise FileNotFoundError(f"no tracklets.parquet files found under {tracklets_root}")
    reader = _VideoFrames(video_root) if cfg.include_crop_colors else _VideoFrames(None)
    rows = []
    try:
        for pq in files:
            try:
                df = pd.read_parquet(pq)
            except ImportError as e:
                raise ImportError(
                    f"reading {pq} requires pyarrow or fastparquet. Install kit/requirements.txt, "
                    "run inside the kit container, or use the bundled Codex runtime for local smoke tests."
                ) from e
            if "tracklet_key" not in df.columns:
                if "local_track_id" not in df.columns:
                    raise ValueError(f"{pq} has neither tracklet_key nor local_track_id")
                video_guess = df["video_key"].iloc[0] if "video_key" in df.columns else pq.parent.name
                df = df.copy()
                df["tracklet_key"] = [f"{video_guess}:local:{x}" for x in df["local_track_id"]]
            video = str(df["video_key"].iloc[0]) if "video_key" in df.columns else pq.parent.name
            camera = _camera_from_video(video)
            for key, group in df.groupby("tracklet_key", sort=True):
                if len(group) < cfg.min_tracklet_dets:
                    continue
                tokens, stats = _bbox_tokens(group)
                if cfg.include_crop_colors:
                    tokens.update(_crop_color_tokens(video, group, reader, cfg))
                row = {
                    "tracklet_key": str(key),
                    "video": video,
                    "camera": camera,
                    "start_frame": stats["start_frame"],
                    "end_frame": stats["end_frame"],
                    "n_dets": stats["n_dets"],
                    "confidence": stats["confidence"],
                    "weak_tokens": _token_payload(tokens),
                    "mean_box_area": stats["mean_box_area"],
                    "mean_box_height": stats["mean_box_height"],
                }
                rows.append(row)
                if cfg.max_tracklets is not None and len(rows) >= cfg.max_tracklets:
                    break
            if cfg.max_tracklets is not None and len(rows) >= cfg.max_tracklets:
                break
    finally:
        reader.close()
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tracklet_key", "video", "camera", "start_frame", "end_frame", "n_dets",
        "confidence", "weak_tokens", "mean_box_area", "mean_box_height",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return {
        "tracklet_files": len(files),
        "rows": len(rows),
        "out_csv": str(out),
        "video_root": str(video_root) if video_root else None,
        "crop_colors_enabled": bool(cfg.include_crop_colors and reader.available),
    }


__all__ = [
    "WeakLabelGenerationConfig",
    "generate_weak_labels",
    "iter_tracklet_parquets",
    "weak_tokens_from_tracklet",
]
