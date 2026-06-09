"""Flat-ground world projection for the gallery's cross-camera geo gate.

Projects a detection's bbox foot-point (bottom-center) to WGS84 (lat,lon) using the per-camera
flat-ground calibration recovered from the geo_b608d03 submission (see runs/ds1_tc6_geo_calib.json).
origin = per-camera, so outputs are absolute WGS84 and directly comparable across cameras.

Geodesy is inlined (copied from vlincs_geo_flatground/src/model/geodesy.py) so this module has no
cross-repo path dependency. Used by the gallery matcher (vlincs_gallery.gallery) to give each identity a
world track, so the consolidation merge can be GATED on world-position consistency (independent of appearance).
"""
from __future__ import annotations
import json
import numpy as np

_A = 6378137.0
_F = 1.0 / 298.257223563
_B = _A * (1.0 - _F)
_E2 = 1.0 - (_B / _A) ** 2


def _g2e(lat, lon, alt):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    s, c = np.sin(la), np.cos(la)
    N = _A / np.sqrt(1.0 - _E2 * s * s)
    return np.array([(N + alt) * c * np.cos(lo), (N + alt) * c * np.sin(lo), (N * (1.0 - _E2) + alt) * s])


def _e2enu(p, olat, olon, oalt):
    o = _g2e(olat, olon, oalt); d = np.asarray(p) - o
    la, lo = np.deg2rad(olat), np.deg2rad(olon)
    sla, cla, slo, clo = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    R = np.array([[-slo, clo, 0], [-sla * clo, -sla * slo, cla], [cla * clo, cla * slo, sla]])
    return R @ d


def _enu2g(p, olat, olon, oalt):
    la, lo = np.deg2rad(olat), np.deg2rad(olon)
    sla, cla, slo, clo = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    RT = np.array([[-slo, -sla * clo, cla * clo], [clo, -sla * slo, cla * slo], [0, cla, sla]])
    pe = _g2e(olat, olon, oalt) + RT @ np.asarray(p)
    x, y, z = pe; lon = np.arctan2(y, x); pp = np.hypot(x, y); lat = np.arctan2(z, pp * (1.0 - _E2))
    for _ in range(5):
        s = np.sin(lat); N = _A / np.sqrt(1.0 - _E2 * s * s)
        alt = pp / np.cos(lat) - N; lat = np.arctan2(z, pp * (1.0 - _E2 * N / (N + alt)))
    return float(np.rad2deg(lat)), float(np.rad2deg(lon))


def _euler_R(roll, pitch, yaw):
    r, p, y = np.deg2rad([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def load_calib(path):
    return json.load(open(path))


def project_feet(cam, pose, U, V, calib, W=1920, H=1080):
    """Vectorized: foot pixels (U,V arrays) for one camera -> (lat[], lon[]) WGS84; NaN where invalid."""
    conv = calib["convention"][cam]
    foc = float(calib["focal_per_camera"][cam]); gz = float(calib["ground_z_per_camera"][cam])
    R = _euler_R(pose["roll"], pose["pitch"], pose["yaw"])
    Rc, sx, sy, sz = conv[0], conv[1], conv[2], conv[3]
    M = (R.T if Rc == "R.T" else R) @ np.diag([sx, sy, sz]).astype(float)
    olat, olon, oalt = pose["lat"], pose["lon"], pose["alt"]   # per-camera origin => camera at ENU 0
    U = np.asarray(U, float); V = np.asarray(V, float); n = len(U)
    cx, cy = W / 2.0, H / 2.0
    d_cam = np.stack([(U - cx) / foc, (V - cy) / foc, np.ones(n)])      # 3xn
    d_world = M @ d_cam
    dz = d_world[2]
    t = np.where(np.abs(dz) > 0.05, (gz - 0.0) / dz, np.nan)
    t = np.where(t > 0, t, np.nan)
    P = t[None, :] * d_world                                            # camera at ENU origin
    lat = np.full(n, np.nan); lon = np.full(n, np.nan)
    for i in np.nonzero(np.isfinite(t))[0]:
        lat[i], lon[i] = _enu2g(P[:, i], olat, olon, oalt)
    return lat, lon


def meters(lat1, lon1, lat2, lon2, lat0=38.9209):
    """Fast local planar distance in meters (good for the few-meter gate at this latitude)."""
    k = 111320.0
    dy = (lat1 - lat2) * k
    dx = (lon1 - lon2) * k * np.cos(np.deg2rad(lat0))
    return float(np.hypot(dx, dy))
