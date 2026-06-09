"""Common absolute clock for multi-camera / multi-card reasoning.

`build_inputs` stores `wall_clock_ms = frame_idx/fps*1000` — RELATIVE to each video's own frame 0,
so every video starts at t=0 and cross-camera time is meaningless. Each shipped per-video
`*_camera_extrinsics_*.parquet` carries the absolute start time (its single `time` row = absolute
time of frame 0) plus the static camera geo-pose (lat/lon/alt/roll/pitch/yaw).

Epochs are keyed by VIDEO STEM (not camera): the same physical camera appears in DS1 Tc6 and Tc8 as
two different recordings with different start times, so they need distinct epochs. `absolute_ms`
maps (video, frame_idx) -> ms on one shared timeline for DB storage, cross-camera stream order, and
time+geo cannot-link. DS1 Tc6 cameras start within ~0.6s of each other (synchronized).
"""
from __future__ import annotations
import glob
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd

_SUFFIX = "_camera_extrinsics"
_M_PER_DEG = 111320.0                                       # local flat-earth lat/lon scale (good <0.1% over a site)


def _ms_since_midnight(t: str) -> float:
    h, m, s = str(t).split(":")
    return (int(h) * 3600 + int(m) * 60 + float(s)) * 1000.0


def _cam_token(stem: str) -> str | None:
    """The MCAMxxx token from a video stem (the camera the matcher/veto keys on)."""
    return next((p for p in stem.split("_") if p.startswith("MCAM")), None)


def _mp4_epochs(card_dir: Path) -> dict[str, float]:
    """MS02-style frame-0 epoch from the .mp4 filename's trailing HH-MM-SS (no extrinsics parquet ships)."""
    out: dict[str, float] = {}
    for mp4 in sorted(glob.glob(str(card_dir / "*.mp4"))):
        stem = Path(mp4).stem
        m = re.search(r"(\d{2})-(\d{2})-(\d{2})$", stem)    # ..._2018-03-15_15-00-06 -> 15:00:06
        if m:
            h, mi, s = map(int, m.groups())
            out[stem] = (h * 3600 + mi * 60 + s) * 1000.0
    return out


def _params_latlon(j: dict) -> dict:
    """Camera lat/lon/alt from an MS02 *_camera_params.json: the camera CENTRE is C = -Rᵀ·t in the ENU
    frame (metres from enu_extrinsics.enu), converted to geodetic by a local flat-earth offset from that
    origin. (enu lat/lon alone is the ORIGIN, not the camera — the two MS02 cameras share one origin.)"""
    R = np.asarray(j["extrinsics"]["R"], dtype=float)
    t = np.asarray(j["extrinsics"]["t"], dtype=float)
    enu = j["enu_extrinsics"]["enu"]
    C = -R.T @ t                                            # camera centre in ENU metres (E, N, U)
    lat0, lon0, alt0 = enu["latitude"], enu["longitude"], enu["altitude"]
    return {"lat": float(lat0 + C[1] / _M_PER_DEG),
            "lon": float(lon0 + C[0] / (_M_PER_DEG * math.cos(math.radians(lat0)))),
            "alt": float(alt0 + C[2])}


def _params_geo(card_dir: Path) -> dict[str, dict]:
    """{video_stem -> {lat,lon,alt,camera}} from MCAMxxx_camera_params.json, joined to each video stem
    by the MCAM token in the .mp4 name (MS02 ships these instead of an extrinsics parquet)."""
    params = {}
    for f in sorted(glob.glob(str(card_dir / "*_camera_params.json"))):
        cam = Path(f).name.split("_camera_params")[0]       # MCAM310
        with open(f) as fh:
            params[cam] = _params_latlon(json.load(fh))
    out: dict[str, dict] = {}
    if not params:
        return out
    for mp4 in sorted(glob.glob(str(card_dir / "*.mp4"))):
        stem = Path(mp4).stem
        cam = _cam_token(stem)
        if cam in params:
            out[stem] = {**params[cam], "camera": cam}
    return out


def video_epochs(*card_dirs: str) -> dict[str, float]:
    """{video_stem -> absolute start-ms (frame 0)} from each video's extrinsics across card dirs.

    video_stem matches the `video` column build_inputs writes (the .mp4 stem). ms-since-midnight
    (recordings are same-day; switch to epoch ms if a card spans midnight). Cards that ship the
    DS1/DS2 `_camera_extrinsics_*.parquet` read its `time`; MS02 (no parquet) falls back to the
    .mp4 filename's HH-MM-SS so the cross-camera timeline (and the simultaneity/travel veto) is real.
    """
    out: dict[str, float] = {}
    for d in card_dirs:
        pq = sorted(glob.glob(str(Path(d) / f"*{_SUFFIX}*.parquet")))
        if pq:
            for f in pq:
                name = Path(f).name
                stem = name[: name.index(_SUFFIX)]           # strip _camera_extrinsics_v…parquet
                e = pd.read_parquet(f, columns=["time"])
                out[stem] = _ms_since_midnight(e["time"].iloc[0])
        else:
            out.update(_mp4_epochs(Path(d)))
    return out


def camera_geo(*card_dirs: str) -> dict[str, dict]:
    """{video_stem -> {lat,lon,alt,roll,pitch,yaw,camera}} static camera pose (basis for the geo veto).

    DS1/DS2 read the surveyed `_camera_extrinsics_*.parquet`; MS02 (which ships MCAMxxx_camera_params.json
    instead) falls back to deriving the camera centre from R,t. Every entry carries `camera` (the MCAM
    token) so cam_xy/dist are keyed by the SAME camera string the pipeline passes to add_tracklet — without
    it the travel/simultaneity veto could never match a camera and silently never fired.
    """
    out: dict[str, dict] = {}
    cols = ["lat", "lon", "alt", "roll", "pitch", "yaw"]
    for d in card_dirs:
        pq = sorted(glob.glob(str(Path(d) / f"*{_SUFFIX}*.parquet")))
        if pq:
            for f in pq:
                name = Path(f).name
                stem = name[: name.index(_SUFFIX)]
                e = pd.read_parquet(f)
                g = {c: float(e[c].iloc[0]) for c in cols if c in e.columns}
                cam = _cam_token(stem)
                if cam:
                    g["camera"] = cam
                out[stem] = g
        else:
            out.update(_params_geo(Path(d)))
    return out


def absolute_ms(meta: pd.DataFrame, epochs: dict[str, float], fps: float = 30.0) -> pd.Series:
    """Per-detection absolute ms = video_epoch + frame_idx/fps*1000. Falls back to epoch 0."""
    vid = str(meta["video"].values[0])
    base = epochs.get(vid, 0.0)
    return base + meta["frame_idx"].astype(float) / fps * 1000.0
