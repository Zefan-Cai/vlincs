"""Common absolute clock for multi-camera / multi-card reasoning.

`build_inputs` stores `wall_clock_ms = frame_idx/fps*1000` - RELATIVE to each video's own frame 0,
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


def _ms_since_midnight(t) -> float:
    """Absolute ms-since-midnight from an extrinsics `time` value - a "HH:MM:SS.fff" string (v1.1.x,
    single-row) OR a pandas Timestamp / datetime64 (v2.0.x, full date+time)."""
    if isinstance(t, str):
        h, m, s = t.split(":")
        return (int(h) * 3600 + int(m) * 60 + float(s)) * 1000.0
    ts = pd.Timestamp(t)
    return float((ts - ts.normalize()).total_seconds() * 1000.0)


def _times_to_ms(s: pd.Series) -> np.ndarray:
    """Vectorized ms-since-midnight for a whole `time` column (datetime64 v2.0.x, or "HH:MM:SS" strings)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s)
        return ((dt - dt.dt.normalize()).dt.total_seconds() * 1000.0).to_numpy(dtype=float)
    return s.map(_ms_since_midnight).to_numpy(dtype=float)


def _version(name: str) -> tuple:
    m = re.search(r"_v(\d+(?:\.\d+)*)\.parquet$", name)
    return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)


def _extrinsics_by_stem(dirs) -> dict[str, str]:
    """Best (highest-version) `*_camera_extrinsics_*.parquet` per video stem across dirs - so the newer
    PER-FRAME v2.0.x wins over the single-row v1.1.x when both sit in input_extrinsics/."""
    best: dict[str, tuple] = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / f"*{_SUFFIX}*.parquet")):
            name = Path(f).name
            stem = name[: name.index(_SUFFIX)]
            ver = _version(name)
            if stem not in best or ver > best[stem][0]:
                best[stem] = (ver, f)
    return {stem: f for stem, (_ver, f) in best.items()}


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
    origin. (enu lat/lon alone is the ORIGIN, not the camera - the two MS02 cameras share one origin.)"""
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
            j = json.load(fh)
        # full CV intrinsics/extrinsics ship only on MS02 -> persisted to cameras.intrinsics (NULL elsewhere)
        intr = {"K": j["intrinsics"]["K"], "D": j["intrinsics"]["D"],
                "R": j["extrinsics"]["R"], "t": j["extrinsics"]["t"],
                "enu_origin": j["enu_extrinsics"]["enu"]}
        params[cam] = {**_params_latlon(j), "intrinsics": intr}
    out: dict[str, dict] = {}
    if not params:
        return out
    for mp4 in sorted(glob.glob(str(card_dir / "*.mp4"))):
        stem = Path(mp4).stem
        cam = _cam_token(stem)
        if cam in params:
            out[stem] = {**params[cam], "camera": cam}
    return out


def load_clock(*dirs: str):
    """Read each video's best (highest-version) extrinsics ONCE and return (epochs, geo, frame_abs):

      epochs[stem]    -> absolute frame-0 ms-since-midnight (captures the cross-camera START offset, e.g.
                         DS1 Tc6 cameras start ~0.64s apart).
      geo[stem]       -> {lat,lon,alt,roll,pitch,yaw,camera} static pose (frame 0) - the geo-veto basis.
      frame_abs[stem] -> np.ndarray indexed by frame_idx of absolute ms, ONLY for PER-FRAME (v2.0.x)
                         extrinsics. The frame intervals are NON-uniform, so this is the EXACT clock; videos
                         with single-row (v1.1.x) extrinsics or MS02 (camera_params.json + .mp4, no parquet)
                         are absent here, so the caller falls back to the old `epoch + frame/fps` clock.

    video_stem matches the `video` column the pipeline pushes (the .mp4 stem). ms-since-midnight
    (recordings are same-day). MS02 has no extrinsics parquet -> epoch from the .mp4 filename HH-MM-SS,
    geo from MCAMxxx_camera_params.json (camera centre from R,t)."""
    epochs: dict[str, float] = {}
    geo: dict[str, dict] = {}
    frame_abs: dict[str, np.ndarray] = {}
    pose_cols = ("lat", "lon", "alt", "roll", "pitch", "yaw")
    for stem, f in _extrinsics_by_stem(dirs).items():
        e = pd.read_parquet(f)
        epochs[stem] = _ms_since_midnight(e["time"].iloc[0])
        g = {c: float(e[c].iloc[0]) for c in pose_cols if c in e.columns}
        mver = re.search(r"_v(\d+(?:\.\d+)*)", Path(f).name)   # the extrinsics file version (cameras.extrinsics_version)
        if mver:
            g["extrinsics_version"] = "v" + mver.group(1)
        cam = _cam_token(stem)
        if cam:
            g["camera"] = cam
        geo[stem] = g
        if len(e) > 1 and "frame" in e.columns:              # per-frame timing (v2.0.x) -> the exact clock
            fr = e["frame"].to_numpy()
            arr = np.full(int(fr.max()) + 1, np.nan)
            arr[fr] = _times_to_ms(e["time"])
            frame_abs[stem] = arr
    for d in dirs:                                           # MS02 fallback: no extrinsics parquet in the dir
        if not glob.glob(str(Path(d) / f"*{_SUFFIX}*.parquet")):
            epochs.update(_mp4_epochs(Path(d)))
            geo.update(_params_geo(Path(d)))
    return epochs, geo, frame_abs


def video_epochs(*dirs: str) -> dict[str, float]:
    """{video_stem -> absolute frame-0 ms}. Thin wrapper over load_clock (kept for external callers)."""
    return load_clock(*dirs)[0]


def camera_geo(*dirs: str) -> dict[str, dict]:
    """{video_stem -> static pose dict}. Thin wrapper over load_clock (kept for external callers)."""
    return load_clock(*dirs)[1]
