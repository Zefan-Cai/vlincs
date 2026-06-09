"""Within-camera motion (IoU) tracklet linker — the gallery's tracklet entry unit (option a).

Greedy per-frame IoU association turns the hot per-detection stream into short within-camera tracks
(a person's consecutive-frame boxes), internally consistent by MOTION. The gallery then associates
these tracklets cross-camera/cross-time by appearance — restoring the within-camera continuity the
pure per-detection appearance matcher lacked (MCAM00: 0.48 vs 0.94 oracle ceiling). Cheap, CPU-only,
deterministic; reuses the detections we already have (no re-detect, no Kalman for the first cut).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _iou_mat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """IoU of every box in A (n,4) against every box in B (m,4) -> (n,m)."""
    ax1, ay1, ax2, ay2 = A[:, 0:1], A[:, 1:2], A[:, 2:3], A[:, 3:4]
    bx1, by1, bx2, by2 = B[:, 0][None], B[:, 1][None], B[:, 2][None], B[:, 3][None]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / np.maximum(aa + bb - inter, 1e-9)


def link_camera(df: pd.DataFrame, camera: str, iou_thresh: float = 0.5, max_gap: int = 10) -> dict[str, str]:
    """Greedy IoU tracklet linking within one camera. Returns {det_id: tracklet_key}.

    Per frame: prune tracks idle longer than max_gap, greedily match this frame's detections to
    active tracks by descending IoU (>= iou_thresh, one det per track), start new tracks for the rest.
    """
    df = df.sort_values("frame_idx", kind="stable")
    active: list[dict] = []
    assign: dict[str, str] = {}
    nt = 0
    for frame, grp in df.groupby("frame_idx", sort=True):
        active = [t for t in active if frame - t["last_frame"] <= max_gap]
        boxes = grp[["x1", "y1", "x2", "y2"]].to_numpy(float)
        dids = grp["det_id"].to_numpy(str)
        matched: dict[int, dict] = {}
        if active and len(boxes):
            M = _iou_mat(boxes, np.stack([t["box"] for t in active]))
            pairs = sorted(((M[i, j], i, j) for i in range(len(boxes)) for j in range(len(active))
                            if M[i, j] >= iou_thresh), reverse=True)
            du, tu = set(), set()
            for _s, i, j in pairs:
                if i in du or j in tu:
                    continue
                du.add(i); tu.add(j); matched[i] = active[j]
        for i in range(len(boxes)):
            if i in matched:
                t = matched[i]; t["last_frame"] = int(frame); t["box"] = boxes[i]
                assign[dids[i]] = t["tid"]
            else:
                t = {"tid": f"{camera}:T{nt}", "last_frame": int(frame), "box": boxes[i]}
                nt += 1; active.append(t); assign[dids[i]] = t["tid"]
    return assign


def link_tracklets(meta: pd.DataFrame, iou_thresh: float = 0.5, max_gap: int = 10) -> dict[str, str]:
    """Link tracklets within each camera. Returns {det_id: tracklet_key} across all cameras."""
    out: dict[str, str] = {}
    for cam, df in meta.groupby("camera"):
        out.update(link_camera(df, str(cam), iou_thresh, max_gap))
    return out
