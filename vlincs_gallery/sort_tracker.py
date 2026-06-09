"""Self-contained SORT/ByteTrack-style motion tracker over EXISTING detections (no re-detect).

Consumes our hot per-detection boxes (meta.parquet) and links them into clean within-camera tracklets
with a constant-velocity Kalman filter + IoU association against the PREDICTED box (the piece the
greedy-IoU linker lacked) + ByteTrack two-stage matching (high-conf dets drive tracks, low-conf dets
sustain them). Keeps the hot detector's recall (we track OUR detections, not a tracker's own) while
restoring the motion continuity within-camera association needs. Signature-compatible drop-in for
vlincs_gallery.tracklets.link_tracklets. numpy-only, deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vlincs_gallery.tracklets import _iou_mat


def _xyxy_to_z(b: np.ndarray) -> np.ndarray:
    w = b[2] - b[0]; h = b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)], dtype=float)


def _z_to_xyxy(x: np.ndarray) -> np.ndarray:
    s, r = x[2], x[3]
    w = np.sqrt(max(s * r, 1e-6)); h = s / max(w, 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBox:
    """Canonical SORT 7-state constant-velocity box Kalman filter (u, v, s, r, u', v', s')."""

    def __init__(self, bbox: np.ndarray):
        self.F = np.eye(7); self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7)); self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.R = np.eye(4); self.R[2:, 2:] *= 10.0
        self.P = np.eye(7); self.P[4:, 4:] *= 1000.0; self.P *= 10.0
        self.Q = np.eye(7); self.Q[-1, -1] *= 0.01; self.Q[4:, 4:] *= 0.01
        self.x = np.zeros(7); self.x[:4] = _xyxy_to_z(bbox)
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return _z_to_xyxy(self.x)

    def update(self, bbox: np.ndarray) -> None:
        z = _xyxy_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0


def _greedy(iou: np.ndarray, track_idx: list[int], det_idx: list[int], thr: float):
    pairs = sorted(((iou[a, b], a, b) for a in range(len(track_idx)) for b in range(len(det_idx))
                    if iou[a, b] >= thr), reverse=True)
    tu, du, res = set(), set(), []
    for _s, a, b in pairs:
        if a in tu or b in du:
            continue
        tu.add(a); du.add(b); res.append((track_idx[a], det_idx[b]))
    return res


def track_camera(df: pd.DataFrame, camera: str, *, iou_thresh: float = 0.3, max_age: int = 30,
                 high_thresh: float = 0.5, new_thresh: float = 0.6) -> dict[str, str]:
    df = df.sort_values("frame_idx", kind="stable")
    tracks: list[dict] = []          # {kf, key}
    assign: dict[str, str] = {}
    nkey = 0
    for frame, grp in df.groupby("frame_idx", sort=True):
        boxes = grp[["x1", "y1", "x2", "y2"]].to_numpy(float)
        confs = grp["conf"].to_numpy(float)
        dids = grp["det_id"].to_numpy(str)
        pred = [t["kf"].predict() for t in tracks]
        hi = [i for i in range(len(boxes)) if confs[i] >= high_thresh]
        lo = [i for i in range(len(boxes)) if confs[i] < high_thresh]
        det_track: dict[int, int] = {}
        if tracks:
            P = np.stack(pred)
            m1 = _greedy(_iou_mat(P, boxes[hi]) if hi else np.zeros((len(tracks), 0)),
                         list(range(len(tracks))), hi, iou_thresh)
            for ti, di in m1:
                det_track[di] = ti
            rem_t = [ti for ti in range(len(tracks)) if ti not in {t for t, _ in m1}]
            m2 = _greedy(_iou_mat(P[rem_t], boxes[lo]) if (rem_t and lo) else np.zeros((len(rem_t), 0)),
                         rem_t, lo, iou_thresh)
            for ti, di in m2:
                det_track[di] = ti
        for di, ti in det_track.items():
            tracks[ti]["kf"].update(boxes[di])
        for di in hi:                                   # spawn tracks from confident unmatched dets
            if di not in det_track and confs[di] >= new_thresh:
                kf = KalmanBox(boxes[di]); key = f"{camera}:K{nkey}"; nkey += 1
                tracks.append({"kf": kf, "key": key}); det_track[di] = len(tracks) - 1
        for di in range(len(boxes)):                    # keep every detection (singleton if untracked)
            if di in det_track:
                assign[dids[di]] = tracks[det_track[di]]["key"]
            else:
                assign[dids[di]] = f"{camera}:S{nkey}"; nkey += 1
        tracks = [t for t in tracks if t["kf"].time_since_update <= max_age]
    return assign


def sort_tracklets(meta: pd.DataFrame, iou_thresh: float = 0.3, max_gap: int = 30,
                   high_thresh: float = 0.5, new_thresh: float = 0.6) -> dict[str, str]:
    """ByteTrack-style linking per camera (Kalman + two-stage high/low-conf IoU). {det_id: tracklet_key}."""
    out: dict[str, str] = {}
    for cam, df in meta.groupby("camera"):
        out.update(track_camera(df, str(cam), iou_thresh=iou_thresh, max_age=max_gap,
                                 high_thresh=high_thresh, new_thresh=new_thresh))
    return out


# --- OC-SORT: observation-centric momentum (OCM) + recovery (OCR) + re-update (ORU) -----------------

def _center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _greedy_gated(score: np.ndarray, gate: np.ndarray, track_idx: list[int], det_idx: list[int], gate_thr: float):
    pairs = sorted(((score[a, b], a, b) for a in range(len(track_idx)) for b in range(len(det_idx))
                    if gate[a, b] >= gate_thr), reverse=True)
    tu, du, res = set(), set(), []
    for _s, a, b in pairs:
        if a in tu or b in du:
            continue
        tu.add(a); du.add(b); res.append((track_idx[a], det_idx[b]))
    return res


def _ocsort_camera(df: pd.DataFrame, camera: str, *, iou_thresh: float = 0.3, max_age: int = 30,
                   new_thresh: float = 0.6, delta_t: int = 3, vdc: float = 0.2) -> dict[str, str]:
    df = df.sort_values("frame_idx", kind="stable")
    tracks: list[dict] = []
    assign: dict[str, str] = {}
    nkey = 0
    for frame, grp in df.groupby("frame_idx", sort=True):
        boxes = grp[["x1", "y1", "x2", "y2"]].to_numpy(float)
        confs = grp["conf"].to_numpy(float)
        dids = grp["det_id"].to_numpy(str)
        pred = [t["kf"].predict() for t in tracks]
        det_track: dict[int, int] = {}
        all_d = list(range(len(boxes)))
        if tracks and all_d:                                   # stage 1: IoU(pred) + OCM momentum
            IOU = _iou_mat(np.stack(pred), boxes)
            cost = IOU.copy()
            for a, t in enumerate(tracks):
                if t["vel"] is not None and np.linalg.norm(t["vel"]) > 1e-6:
                    tc = _center(t["last_obs"])
                    for b in all_d:
                        dv = _center(boxes[b]) - tc; n = np.linalg.norm(dv)
                        if n > 1e-6:
                            cost[a, b] += vdc * float(np.dot(dv, t["vel"]) / (n * np.linalg.norm(t["vel"])))
            for ti, di in _greedy_gated(cost, IOU, list(range(len(tracks))), all_d, iou_thresh):
                det_track[di] = ti
        matched = set(det_track.values())                      # stage 2 OCR: last-observation recovery
        rem_t = [ti for ti in range(len(tracks)) if ti not in matched]
        rem_d = [di for di in all_d if di not in det_track]
        if rem_t and rem_d:
            IOU2 = _iou_mat(np.stack([tracks[ti]["last_obs"] for ti in rem_t]), boxes[rem_d])
            for a, b in _greedy_gated(IOU2, IOU2, list(range(len(rem_t))), list(range(len(rem_d))), iou_thresh):
                det_track[rem_d[b]] = rem_t[a]
        for di, ti in det_track.items():
            t = tracks[ti]; gap = t["kf"].time_since_update
            if gap > 1:                                        # ORU: re-update along virtual trajectory
                last = t["last_obs"]; new = boxes[di]
                for k in range(1, gap):
                    t["kf"].update(last + (new - last) * (k / gap))
            t["kf"].update(boxes[di])
            t["hist"].append(_center(boxes[di])); t["hist"] = t["hist"][-(delta_t + 1):]
            t["vel"] = (t["hist"][-1] - t["hist"][0]) if len(t["hist"]) >= 2 else None
            t["last_obs"] = boxes[di].copy()
        for di in all_d:
            if di not in det_track and confs[di] >= new_thresh:
                kf = KalmanBox(boxes[di]); key = f"{camera}:K{nkey}"; nkey += 1
                tracks.append({"kf": kf, "key": key, "last_obs": boxes[di].copy(),
                               "hist": [_center(boxes[di])], "vel": None})
                det_track[di] = len(tracks) - 1
        for di in range(len(boxes)):
            if di in det_track:
                assign[dids[di]] = tracks[det_track[di]]["key"]
            else:
                assign[dids[di]] = f"{camera}:S{nkey}"; nkey += 1
        tracks = [t for t in tracks if t["kf"].time_since_update <= max_age]
    return assign


def ocsort_tracklets(meta: pd.DataFrame, iou_thresh: float = 0.3, max_gap: int = 30,
                     new_thresh: float = 0.6, delta_t: int = 3, vdc: float = 0.2) -> dict[str, str]:
    """OC-SORT linking per camera (OCM + OCR + ORU). Signature-compatible with sort_tracklets."""
    out: dict[str, str] = {}
    for cam, df in meta.groupby("camera"):
        out.update(_ocsort_camera(df, str(cam), iou_thresh=iou_thresh, max_age=max_gap,
                                   new_thresh=new_thresh, delta_t=delta_t, vdc=vdc))
    return out
