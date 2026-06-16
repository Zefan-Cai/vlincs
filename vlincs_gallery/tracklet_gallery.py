"""Tracklet-level gallery: cross-camera / cross-time identity association over within-camera tracklets.

The dual of IdentityGallery, one level up: the entry unit is a motion-linked tracklet (already
internally consistent), so the gallery only has to make CROSS-tracklet decisions - far fewer, each on
a pooled (stable) appearance, and the cannot-link is now the correct tracklet-level one: two tracklets
that temporally OVERLAP in the same camera are different people and may not merge. Cross-camera
same-instant tracklets may merge (that is cross-camera ReID). Append-only, deterministic.
"""

from __future__ import annotations

import numpy as np


class TrackletGallery:
    def __init__(self, dim: int, cfg):
        self.dim = dim
        self.cfg = cfg
        self.bank_arrs: list[np.ndarray] = []                 # per-gid (r, dim) L2-normed exemplars
        self.cam_intervals: list[dict[str, list]] = []         # per-gid: camera -> [(start_frame, end_frame)]
        self.n = 0
        self._mat: np.ndarray | None = None
        self._gid: np.ndarray | None = None
        self._dirty = True

    def _rebuild(self) -> None:
        if self.bank_arrs:
            self._mat = np.concatenate(self.bank_arrs, 0)
            self._gid = np.concatenate([np.full(len(b), g) for g, b in enumerate(self.bank_arrs)])
        else:
            self._mat = np.zeros((0, self.dim), np.float32)
            self._gid = np.zeros((0,), int)
        self._dirty = False

    def _cap(self, exemplars: np.ndarray) -> np.ndarray:
        # keep up to max_reps evenly-spaced exemplars from the tracklet (cheap diversity proxy)
        if len(exemplars) <= self.cfg.max_reps:
            return exemplars.astype(np.float32)
        idx = np.linspace(0, len(exemplars) - 1, self.cfg.max_reps).astype(int)
        return exemplars[idx].astype(np.float32)

    def _overlap(self, gid: int, camera: str, s: int, e: int, tol: int) -> bool:
        # cannot-link: the gid already owns a same-camera tracklet temporally overlapping [s, e]
        for (a, b) in self.cam_intervals[gid].get(camera, ()):
            if s <= b + tol and a <= e + tol:
                return True
        return False

    def add_or_match(self, exemplars: np.ndarray, centroid: np.ndarray, camera: str,
                     s: int, e: int, tau: float, tol: int = 0) -> tuple[int, str, float]:
        if self._dirty:
            self._rebuild()
        if self._mat.shape[0] == 0:
            return self._expand(exemplars, camera, s, e), "expand", 0.0
        sims = self._mat @ centroid                       # tracklet centroid vs gid exemplars
        best = np.full(self.n, -1.0)
        np.maximum.at(best, self._gid, sims)
        best_overall = float(best.max())
        for gid in np.argsort(best)[::-1]:
            if best[gid] < tau:
                break
            if self._overlap(int(gid), camera, s, e, tol):
                continue
            self._add(int(gid), exemplars, camera, s, e)
            return int(gid), "match", float(best[gid])
        return self._expand(exemplars, camera, s, e), "expand", best_overall

    def _expand(self, exemplars: np.ndarray, camera: str, s: int, e: int) -> int:
        gid = self.n
        self.bank_arrs.append(self._cap(exemplars))
        self.cam_intervals.append({camera: [(s, e)]})
        self.n += 1
        self._dirty = True
        return gid

    def _add(self, gid: int, exemplars: np.ndarray, camera: str, s: int, e: int) -> None:
        self.cam_intervals[gid].setdefault(camera, []).append((s, e))
        b = self.bank_arrs[gid]
        for v in exemplars:                                # diversity-gated admission, capped
            if len(b) >= self.cfg.max_reps:
                break
            if float((b @ v).max()) < self.cfg.admit_tau:
                b = np.concatenate([b, v[None, :].astype(np.float32)], 0)
        self.bank_arrs[gid] = b
        self._dirty = True

    def stats(self) -> dict:
        return dict(n_gids=self.n,
                    mean_reps=round(float(np.mean([len(b) for b in self.bank_arrs])), 2) if self.n else 0.0)
