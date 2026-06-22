"""IdentityGallery - the canonical online match-or-expand matcher (in-memory core).

This is the single source of truth for the gallery's match/expand/consolidate logic. The matcher
core is tracklet-level: ``match_mode`` ∈ {max, centroid, retrieval}, ``coherence_floor``, a
tracklet-level ``match_or_expand(video, cam, frames, boxes, t0, t1, pooled, ...)``, geo-aware
``consolidate(merge_tau)``, FAISS retrieval, ``assign_det`` / ``apply_remap``. The revisability API
(``split_low_coherence`` / ``revise`` / ``camera_span_stats``) is expressed against this core's
``rep_mat`` / ``rep_gid`` state.

DB / pgvector backing is NOT baked in here - this class is pure, in-memory, and depends only on numpy
(+ FAISS only when ``match_mode="retrieval"``). The durable pgvector system-of-record the kit needs
sits ON TOP of this matcher (``kit/online.py`` persists boxes/ids/decisions to Postgres while driving
this same matcher), so the matcher stays embedder-agnostic and easy to unit-test.

Cannot-link policy (configurable; ``cam_xy`` / ``dist`` / ``overlaps`` feed the cross-camera vetoes):
  * same (video, frame): two SPATIALLY-DISTINCT boxes are two people -> veto (a duplicate-track box
    that heavily overlaps is the SAME person -> allowed).
  * cross-camera SIMULTANEITY within ``sim_window_ms`` for a non-overlapping camera pair -> veto.
  * impossible travel-time (camera distance / Δt > ``max_speed``) -> veto.
  * optional flat-ground geo gate on ``consolidate`` (``geo_max_m``).

Determinism: tracklets are processed in a fixed order by the drivers and all operations are
deterministic, so a run is reproducible from (inputs, config) alone.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from vlincs_gallery import geo



# --------------------------------------------------------------------------------------------------
# module-level helpers (importable by the experiment drivers)
# --------------------------------------------------------------------------------------------------
def _normed(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _haversine_m(a, b):
    R = 6371000.0
    la1, lo1 = math.radians(a[0]), math.radians(a[1])
    la2, lo2 = math.radians(b[0]), math.radians(b[1])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return R * 2 * math.asin(math.sqrt(h))


def iou(a, B):
    ix1 = np.maximum(a[0], B[:, 0]); iy1 = np.maximum(a[1], B[:, 1])
    ix2 = np.minimum(a[2], B[:, 2]); iy2 = np.minimum(a[3], B[:, 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None); inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa + bb - inter + 1e-9)


def iou1(a, b):
    return float(iou(np.asarray(a, float), np.asarray(b, float)[None, :])[0])


def _iou_pairwise(A, B):
    """Element-wise IoU of two (S,4) box arrays (vectorized equivalent of ``[iou1(A[i], B[i]) for i]``).
    Used to collapse the consolidate same-frame occ check from S separate iou1 calls into one numpy op."""
    ix1 = np.maximum(A[:, 0], B[:, 0]); iy1 = np.maximum(A[:, 1], B[:, 1])
    ix2 = np.minimum(A[:, 2], B[:, 2]); iy2 = np.minimum(A[:, 3], B[:, 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None); inter = iw * ih
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa + bb - inter + 1e-9)


def _fast_consolidate(g, merge_tau, return_events=True):
    """Fast equivalent of IdentityGallery._consolidate_ref. Returns the same `remap` and `events`
    for every input. Merge -only consolidation of live entities. """
    rep_mat = g.rep_mat
    rep_gid = g.rep_gid
    groups = {}
    # Get the exemplars for each group
    for gid in sorted(set(rep_gid.tolist())):
        groups[gid] = dict(
            gids={gid},
            reps=rep_mat[rep_gid == gid],
            occ=dict(g.occ.get(gid, {})),
            crange={c: list(r) for c, r in g.crange.get(gid, {}).items()},
            world={s: (float(np.median([p[0] for p in lst])), float(np.median([p[1] for p in lst])))
                   for s, lst in g.world.get(gid, {}).items()},
        )
    same_box_iou = g.same_box_iou; geo_max_m = g.geo_max_m; merge_free_xcam = g.merge_free_xcam
    sw = g.sw; overlaps = g.overlaps; dist = g.dist; max_speed = g.max_speed

    def cannot_merge(gid_a, gid_b):
        """Decide whether two gallery groups are provably different identities.

        Vetoes a merge - regardless of appearance similarity - when any of three hard structural
        constraints is violated. The checks run in order and short-circuit:

        1. Co-visibility: at any ``(video, frame)`` where both groups have a box, boxes that barely
           overlap (median same-frame IoU < ``same_box_iou``) are two distinct people, not one
           person's duplicate track.
        2. Geo (only when ``geo_max_m`` > 0): at any shared second, flat-ground world positions more
           than ``geo_max_m`` metres apart put one body in two places at once.
        3. Cross-camera travel feasibility (skipped when ``merge_free_xcam``): for every pair of
           per-camera presence intervals in different cameras, a transition is impossible if the
           intervals are near-simultaneous (gap <= ``sw`` ms) without the cameras sharing a field of
           view, or if covering the inter-camera distance in that time would exceed ``max_speed`` m/s.

        Args:
            gid_a: Root gid of the first group (keys into the enclosing ``groups`` dict).
            gid_b: Root gid of the second group.

        Returns:
            ``True`` if the two groups must never be merged (a hard veto fired); ``False`` if the
            merge is structurally allowed, leaving appearance vs ``merge_tau`` to decide. Mirrors
            ``IdentityGallery._consolidate_ref`` exactly.
        """
        # 1. co-visibility: spatially distinct boxes in a shared frame => two people
        occ_a, occ_b = groups[gid_a]["occ"], groups[gid_b]["occ"]
        shared_frames = list(occ_a.keys() & occ_b.keys())
        if shared_frames:
            # one vectorized same-frame IoU over all shared frames (not len() separate iou1 calls)
            boxes_a = np.array([occ_a[fr] for fr in shared_frames], float)
            boxes_b = np.array([occ_b[fr] for fr in shared_frames], float)
            if float(np.median(_iou_pairwise(boxes_a, boxes_b))) < same_box_iou:
                return True

        # 2. geo: same second, far apart on the ground => one identity can't be in both places
        if geo_max_m > 0:
            world_a, world_b = groups[gid_a]["world"], groups[gid_b]["world"]
            shared_secs = set(world_a) & set(world_b)
            if shared_secs:
                sep_m = [geo.meters(world_a[sec][0], world_a[sec][1], world_b[sec][0], world_b[sec][1])
                         for sec in shared_secs]
                if float(np.median(sep_m)) > geo_max_m:
                    return True

        # 3. cross-camera travel feasibility
        if merge_free_xcam:
            return False
        for cam_a, (lo_a, hi_a) in groups[gid_a]["crange"].items():
            for cam_b, (lo_b, hi_b) in groups[gid_b]["crange"].items():
                if cam_a == cam_b:
                    continue
                gap_ms = max(0, lo_b - hi_a, lo_a - hi_b)           # 0 if the two intervals overlap in time
                if gap_ms <= sw:                                    # near-simultaneous (within the slop window)
                    if frozenset((cam_a, cam_b)) not in overlaps:  # ...but the cameras don't share a FOV
                        return True
                else:
                    cam_dist_m = dist.get(frozenset((cam_a, cam_b)))
                    if cam_dist_m is not None and cam_dist_m / (gap_ms / 1000.0) > max_speed:
                        return True                                # would have to move faster than max_speed
        return False

    events = []
    if len(groups) <= 1:
        return ({gid: root for root, grp in groups.items() for gid in grp["gids"]}, events) if return_events \
            else {gid: root for root, grp in groups.items() for gid in grp["gids"]}

    roots = list(groups)                                  # live insertion order (ascending gid at seed)
    n = len(roots)
    cents = [_normed(groups[r]["reps"].mean(0)) for r in roots]   # per-group normed centroid: mean then /(norm+1e-9)
    cosM = np.full((n, n), -np.inf, dtype=np.float64)     # cosM[i,j] (i<j) = float(cents[i] @ cents[j])
    vetoM = np.zeros((n, n), dtype=bool)                  # vetoM[i,j] (i<j) = cannot_merge(roots[i], roots[j])
    for i in range(n):
        ci = cents[i]
        for j in range(i + 1, n):
            cosM[i, j] = float(ci @ cents[j])
            vetoM[i, j] = cannot_merge(roots[i], roots[j])

    while n > 1:
        iu = np.triu_indices(n, 1)                        # row-major (i asc, then j asc)
        cos_flat = cosM[iu]; veto_flat = vetoM[iu]
        valid = (~veto_flat) & (cos_flat >= merge_tau)
        if not valid.any():
            # No valid merge candidates
            break
        cmax = cos_flat[valid].max()
        eq = valid & (cos_flat == cmax)                   # exact-float equality to the max
        pos = np.nonzero(eq)[0][-1]                        # LAST in row-major: `>=` last-wins tie-break
        i = int(iu[0][pos]); j = int(iu[1][pos])
        a, b = roots[i], roots[j]
        events.append((int(a), int(b), round(float(cmax), 4)))
        groups[a]["gids"] |= groups[b]["gids"]
        groups[a]["reps"] = np.vstack([groups[a]["reps"], groups[b]["reps"]])
        groups[a]["occ"].update(groups[b]["occ"])
        for c, (l, h) in groups[b]["crange"].items():
            if c in groups[a]["crange"]:
                groups[a]["crange"][c][0] = min(groups[a]["crange"][c][0], l)
                groups[a]["crange"][c][1] = max(groups[a]["crange"][c][1], h)
            else:
                groups[a]["crange"][c] = [l, h]
        for s, ll in groups[b]["world"].items():
            if s in groups[a]["world"]:
                groups[a]["world"][s] = ((groups[a]["world"][s][0] + ll[0]) / 2,
                                         (groups[a]["world"][s][1] + ll[1]) / 2)
            else:
                groups[a]["world"][s] = ll
        del groups[b]
        roots.pop(j); cents.pop(j)                         # drop position j (the absorbed group)
        cosM = np.delete(np.delete(cosM, j, axis=0), j, axis=1)
        vetoM = np.delete(np.delete(vetoM, j, axis=0), j, axis=1)
        n -= 1
        cents[i] = _normed(groups[a]["reps"].mean(0))      # survivor at position i (j>i removed): refresh its row
        for k in range(n):
            if k == i:
                continue
            lo, hi = (i, k) if i < k else (k, i)
            cosM[lo, hi] = float(cents[lo] @ cents[hi])
            vetoM[lo, hi] = cannot_merge(roots[lo], roots[hi])

    remap = {gid: root for root, grp in groups.items() for gid in grp["gids"]}
    return (remap, events) if return_events else remap


def self_coherence(reds):
    """Within-tracklet coherence = mean pairwise cosine of the (normalized) per-frame reds. Low =>
    the tracklet's appearance is internally inconsistent (ID-switch / occlusion / mixed people)."""
    R = np.stack(reds).astype(np.float32)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    if len(R) < 2:
        return 1.0
    return float((R @ R.T)[np.triu_indices(len(R), 1)].mean())


def _greedy(cost):
    pairs, ur, uc = [], set(), set()
    order = sorted((cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1]) if cost[i, j] < 1e5)
    for _c, i, j in order:
        if i not in ur and j not in uc:
            pairs.append((i, j)); ur.add(i); uc.add(j)
    return pairs


def kalman_app_tracklets(vr, *, window, iou_gate, high_conf, max_age, app_weight, ema=0.9):
    """BoT-SORT-lite within-camera tracker: Kalman motion prediction + appearance-EMA + two-stage
    (high/low conf) association, with OUR max-length cap (split a track into <=window-frame
    tracklets, keeping Kalman/appearance state across the cut). Input vr = [(did, frame, abs_ms,
    box4, red64, conf), ...] for ONE video, frame-sorted. Returns [dict(dids, reds, frames, boxes,
    confs, t0, t1)] capped tracklets that feed the cross-camera gallery (confs = per-det detector conf)."""
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:                              # pragma: no cover
        linear_sum_assignment = None
    from vlincs_gallery.sort_tracker import KalmanBox

    frames = {}
    for r in vr:
        frames.setdefault(r[1], []).append(r)
    tracks, out = [], []

    def emit(t):
        if t["dids"]:
            out.append(dict(dids=t["dids"], reds=t["reds"], frames=t["frames"], boxes=t["boxes"],
                            confs=t["confs"], t0=t["t0"], t1=t["t1"]))

    def fresh(t):
        t["dids"], t["reds"], t["frames"], t["boxes"], t["confs"], t["t0"], t["t1"], t["sf"] = \
            [], [], [], [], [], None, None, None

    def extend(t, d):
        t["kbox"].update(d[3])
        t["app"] = ema * t["app"] + (1 - ema) * d[4]; t["app"] /= (np.linalg.norm(t["app"]) + 1e-9)
        if not t["dids"]:
            t["t0"], t["sf"] = d[2], d[1]
        t["dids"].append(d[0]); t["reds"].append(d[4]); t["frames"].append(d[1]); t["boxes"].append(d[3])
        t["confs"].append(d[5]); t["t1"] = d[2]; t["age"] = 0

    for f in sorted(frames):
        dets = frames[f]
        for t in tracks:
            t["pbox"] = t["kbox"].predict(); t["age"] += 1
        hi = [d for d in dets if d[5] >= high_conf]; lo = [d for d in dets if d[5] < high_conf]
        mt, md = set(), set()
        if tracks and hi:
            HB = np.stack([d[3] for d in hi]); cost = np.full((len(tracks), len(hi)), 1e6)
            for i, t in enumerate(tracks):
                ii = iou(t["pbox"], HB)
                for j in range(len(hi)):
                    if ii[j] >= iou_gate:
                        cost[i, j] = 1.0 - (ii[j] + app_weight * float(t["app"] @ hi[j][4]))
            pairs = ([(i, j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e5]
                     if linear_sum_assignment is not None else _greedy(cost))
            for i, j in pairs:
                extend(tracks[i], hi[j]); mt.add(i); md.add(j)
        ut = [i for i in range(len(tracks)) if i not in mt]
        if ut and lo:
            LB = np.stack([d[3] for d in lo]); cost = np.full((len(ut), len(lo)), 1e6)
            for a_, i in enumerate(ut):
                ii = iou(tracks[i]["pbox"], LB)
                for b_ in range(len(lo)):
                    if ii[b_] >= iou_gate:
                        cost[a_, b_] = 1.0 - ii[b_]
            pairs = ([(i, j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e5]
                     if linear_sum_assignment is not None else _greedy(cost))
            for a_, j in pairs:
                extend(tracks[ut[a_]], lo[j])
        for t in tracks:                       # OUR max-length cap
            if t["sf"] is not None and (f - t["sf"] + 1) >= window:
                emit(t); fresh(t)
        for j, d in enumerate(hi):             # birth on unmatched high-conf
            if j not in md:
                t = dict(kbox=KalmanBox(d[3]), app=d[4] / (np.linalg.norm(d[4]) + 1e-9), age=0,
                         dids=[d[0]], reds=[d[4]], frames=[d[1]], boxes=[d[3]], confs=[d[5]],
                         t0=d[2], t1=d[2], sf=d[1])
                tracks.append(t)
        keep = []                              # death
        for t in tracks:
            if t["age"] > max_age:
                emit(t)
            else:
                keep.append(t)
        tracks = keep
    for t in tracks:
        emit(t)
    return out


class IdentityGallery:
    """The in-memory match-or-expand matcher.

    Construct directly with the positional signature for experiment / kit use::

        g = IdentityGallery(cam_xy, dist, overlaps, tau, max_speed, sim_window_ms, admit_tau,
                            max_reps, match_mode="centroid", coherence_floor=0.4, emb_dim=64)

    or via :meth:`from_config` for the ``PolicyConfig``-driven shakeout drivers.
    """

    def __init__(self, cam_xy, dist, overlaps, tau, max_speed, sim_window_ms, admit_tau, max_reps,
                 same_box_iou=0.35, coherence_floor=0.0, tracklet_coh_min=0.0, match_mode="max",
                 emb_dim=64, *, cfg=None, disc_k=20):
        self.cam_xy, self.dist, self.overlaps = cam_xy, dist, overlaps
        self.tau, self.max_speed, self.sw = tau, max_speed, sim_window_ms
        self.admit_tau, self.max_reps = admit_tau, max_reps
        self.same_box_iou = same_box_iou   # two boxes in one (video,frame) with IoU>=this are the SAME person (duplicate track), not two people
        self.coherence_floor = coherence_floor   # reject an exemplar whose MIN cosine to the id's bank < this (anti-accretion)
        self.tracklet_coh_min = tracklet_coh_min # quarantine tracklets with internal self-coherence < this ("do nothing")
        self.match_mode = match_mode             # candidate scoring: "max" over exemplars, id "centroid", or "retrieval"
        self.knn = 10                            # retrieval mode: top-k neighbors to vote over (set from --knn)
        self._faiss = None                       # retrieval mode: FAISS IndexFlatIP over the gallery exemplars (lazy)
        self.expand_block_sim = 0.0              # if a would-be expand had a blocked candidate at sim>=this, quarantine instead (don't confidently seed)
        self.match_min_floor = 0.0               # MATCH gate: require MIN cosine to ALL of an id's exemplars >= this (consistency, not just nearest)
        self.merge_free_xcam = False             # if True, consolidate() drops the cross-camera simultaneity/travel veto (keeps within-cam same-frame guard)
        self.geo_max_m = 0.0                     # cross-cam merge GATE: block merging two ids whose world tracks differ by > this (m) at shared seconds (0=off)
        self.world = {}                          # gid -> {sec: [(lat,lon), ...]} flat-ground world track (for the geo gate)
        self.next_gid = 1
        self.rep_mat = np.zeros((0, emb_dim), np.float32)
        self.rep_gid = np.zeros((0,), np.int64)
        # centroid-mode cache: {gid -> normed centroid}. Recomputing every gid's centroid from rep_mat on
        # EVERY tracklet is the dominant ingest cost at scale (O(gids*reps)/tracklet). The centroid only
        # changes when a gid's bank changes, so we keep it incrementally (update one gid on admit) and
        # invalidate (-> None, lazy full rebuild) on apply_remap / reset.
        # GUARD: rep_mat/rep_gid must only change via this class's methods (which keep the cache in sync).
        # `_cent_sig` is a cheap signature of the bank that `_candidates` re-checks; if the bank was mutated
        # out-of-band (e.g. someone reassigned/relabeled rep_gid directly) the cache self-heals (rebuild) and
        # warns once. Direct editors should call `invalidate_centroid_cache()`.
        self._cent: dict | None = None
        self._cent_sig: tuple | None = None
        self._cent_warned = False
        self.occ = {}           # gid -> {(video, frame): box}  (box kept so same_frame can tell a duplicate-track from two people)
        self.crange = {}        # gid -> {camera: [lo_ms, hi_ms]}
        self.n_reps = {}        # gid -> count
        self.decisions = []     # per-tracklet decision trail (the "why"); -> decision_log
        self._seq = 0
        # revisability-A bookkeeping (camera-span gate, discriminability diagnostic)
        self.cfg = cfg
        self.disc_k = getattr(cfg, "disc_k", disc_k) if cfg is not None else disc_k

    # ---- legacy PolicyConfig constructor (day-1 shakeout drivers) -------------------------------
    @classmethod
    def from_config(cls, dim: int, cfg) -> "IdentityGallery":
        """Build from a :class:`vlincs_gallery.config.PolicyConfig` (MS02/DS1 shakeout drivers).

        Defaults: no geo/cannot-link wiring (the per-detection callers pass an empty camera map),
        ``match_mode="max"``, no coherence floor.
        """
        return cls({}, {}, set(), cfg.match_tau, 3.0, 200, cfg.admit_tau, cfg.max_reps,
                   match_mode="max", coherence_floor=0.0, emb_dim=dim, cfg=cfg)

    # ---- introspection -------------------------------------------------------------------------
    @property
    def n(self) -> int:
        """Number of live identities (distinct gids carrying at least one exemplar)."""
        return int(np.unique(self.rep_gid).size) if self.rep_gid.size else 0

    def _gids(self):
        return [int(g) for g in np.unique(self.rep_gid)] if self.rep_gid.size else []

    def centroids(self) -> np.ndarray:
        gids = self._gids()
        if not gids:
            return np.zeros((0, self.rep_mat.shape[1]), np.float32)
        return np.stack([_normed(self.rep_mat[self.rep_gid == g].mean(0)) for g in gids])

    def discriminability(self) -> float:
        try:
            from vlincs_sdk.gallery import discriminability_ratio  # SDK helper (lazy, OPTIONAL)
        except ImportError:
            return float("nan")   # diagnostic snapshot only; the base kit (no vlincs-sdk) skips it -> disc_ratio NULL
        c = self.centroids()
        return discriminability_ratio(c, k=self.disc_k) if len(c) >= 2 else float("nan")

    # --------------------------------------------------------------------------------------------
    # MATCHER CORE
    # --------------------------------------------------------------------------------------------
    def _bank_sig(self) -> tuple:
        """Cheap signature of the exemplar bank, used to detect out-of-band mutation of the centroid cache.
        Catches the realistic mistakes - reassigning rep_mat/rep_gid (object id changes), growing/shrinking
        the bank (length changes), or relabeling gids in place (rep_gid sum changes). It does NOT detect an
        in-place edit of an existing exemplar's VECTOR (same id/len/gids); call `invalidate_centroid_cache()`
        after any such edit."""
        g = self.rep_gid
        return (g.shape[0], id(self.rep_mat), id(g), int(g.sum()) if g.size else 0)

    def invalidate_centroid_cache(self) -> None:
        """Drop the centroid cache. Call this if you mutate rep_mat/rep_gid directly (outside
        match_or_expand / apply_remap) - otherwise centroid-mode matching may use stale centroids."""
        self._cent = None

    def _candidates(self, v):
        if self.rep_mat.shape[0] == 0:
            return []
        if self.match_mode == "centroid":
            # score by id centroid - a spread/garbage bank has a blurred centroid that attracts less.
            # Centroids are cached (see self._cent): rebuilt from rep_mat only when invalid, then one dot per
            # gid.
            sig = self._bank_sig()
            if self._cent is not None and sig != self._cent_sig:
                # the bank changed without going through this class's methods -> the cache is stale. Self-heal
                # (rebuild below) so the RESULT stays correct, and warn once so the caller fixes their code.
                if not self._cent_warned:
                    warnings.warn(
                        "IdentityGallery: rep_mat/rep_gid changed outside match_or_expand/apply_remap, so the "
                        "centroid cache was stale; it has been rebuilt. Mutate the bank only via the matcher's "
                        "methods, or call invalidate_centroid_cache() after a direct edit.",
                        RuntimeWarning, stacklevel=2)
                    self._cent_warned = True
                self._cent = None
            if self._cent is None:
                self._cent = {int(g): _normed(self.rep_mat[self.rep_gid == g].mean(0))
                              for g in np.unique(self.rep_gid)}
                self._cent_sig = sig
            # per-gid dot (NOT a single matmul) + sorted gids for a stable score order
            best = {g: float(self._cent[g] @ v) for g in sorted(self._cent)}
            return sorted(best.items(), key=lambda kv: -kv[1])
        if self.match_mode == "retrieval":
            # RETRIEVAL-STYLE: FAISS k-NN over ALL gallery exemplars, then VOTE - a gid's score is the
            # mean cosine of its neighbors within the query's top-k (so an id with several close exemplars
            # beats one with a single lucky exemplar). The IndexFlatIP is rebuilt lazily when the bank grew.
            import faiss
            n = self.rep_mat.shape[0]
            if self._faiss is None or self._faiss.ntotal != n:
                self._faiss = faiss.IndexFlatIP(self.rep_mat.shape[1])
                self._faiss.add(np.ascontiguousarray(self.rep_mat))
            k = min(self.knn, n)
            D, I = self._faiss.search(np.ascontiguousarray(v[None].astype(np.float32)), k)
            agg = {}
            for j, idx in enumerate(I[0]):
                agg.setdefault(int(self.rep_gid[idx]), []).append(float(D[0][j]))
            return sorted(((g, float(np.mean(s))) for g, s in agg.items()), key=lambda kv: -kv[1])
        sims = self.rep_mat @ v
        order = np.argsort(-sims, kind="stable")
        best = {}
        for k in order:
            g = int(self.rep_gid[k])
            if g not in best:
                best[g] = float(sims[k])
        return sorted(best.items(), key=lambda kv: -kv[1])   # [(gid, sim)] desc

    def _cannot_link(self, gid, video, cam, frames, boxes, t0, t1):
        # same (video, frame): a conflict ONLY if the two boxes are spatially DISTINCT (two different
        # people coexisting in that frame). If they heavily overlap, it's the SAME person the
        # within-camera tracker duplicated -> not a conflict; allow the match/merge.
        occ = self.occ.get(gid)
        if occ:
            ov = [iou1(bx, occ[(video, f)]) for f, bx in zip(frames, boxes) if (video, f) in occ]
            if ov and float(np.median(ov)) < self.same_box_iou:
                return "same_frame"
        for c2, (lo, hi) in self.crange.get(gid, {}).items():
            if c2 == cam:
                continue
            gap = max(0, lo - t1, t0 - hi)               # ms between the two time-ranges (0 if overlap)
            if gap <= self.sw:                            # simultaneous (within slop)
                if frozenset((cam, c2)) not in self.overlaps:
                    return f"simultaneity:{c2}"           # non-overlapping cameras can't share a person now
            else:                                         # travel-time
                d = self.dist.get(frozenset((cam, c2)))
                if d is not None and d / (gap / 1000.0) > self.max_speed:
                    return f"travel:{c2}"
        return None

    def match_or_expand(self, video, cam, frames, boxes, t0, t1, pooled, rep_did=None,
                        self_coh=1.0, force_expand=False, world=None):
        """Tracklet-level match-or-expand (the canonical signature).

        ``frames``/``boxes`` are the tracklet's per-detection frame indices and boxes (same camera),
        ``t0``/``t1`` its absolute-clock span (ms), ``pooled`` its L2-normed pooled appearance vector.
        Returns ``(gid, sim, decision_type, pruned, admitted)``.

        For a per-detection call, pass a one-element tracklet:
        ``match_or_expand(video, camera, [frame], [box], t, t, vec)``.
        """
        # "do nothing" guard: a low-self-coherence (mixed-person / messy) tracklet is QUARANTINED - it
        # gets its own id but is NEVER admitted to the matchable bank, so it can neither seed nor
        # pollute an identity. force_expand (warmup batch-seed) skips matching but DOES admit.
        quarantine = self_coh < self.tracklet_coh_min
        qreason = "low_coherence" if quarantine else ""
        chosen, sim, dtype, pruned, cands = None, 0.0, "expand", [], []
        if not (force_expand or quarantine):
            for g, s in self._candidates(pooled):
                if s < self.tau:
                    cands.append((g, round(float(s), 4), "below_tau")); break   # best candidate, but below threshold
                cl = self._cannot_link(g, video, cam, frames, boxes, t0, t1)
                cands.append((g, round(float(s), 4), cl))
                if cl:
                    pruned.append(g); continue
                # consistency gate: close to ONE exemplar (max>=tau) is not enough - require closeness
                # to the WHOLE bank, else it's likely a different person resembling one exemplar.
                if self.match_min_floor > 0:
                    same = self.rep_mat[self.rep_gid == g]
                    if same.shape[0] >= 2 and float((same @ pooled).min()) < self.match_min_floor:
                        cands[-1] = (g, round(float(s), 4), "inconsistent"); continue
                chosen, sim, dtype = g, s, "match"; break
        if chosen is None:
            # would EXPAND. But if a high-sim candidate was only blocked by cannot-link, a confident new
            # identity is premature (it clearly resembles an existing one) -> QUARANTINE instead of seeding.
            if not (force_expand or quarantine) and self.expand_block_sim > 0:
                best_blocked = max((c[1] for c in cands if c[2] and c[2] != "below_tau"), default=0.0)
                if best_blocked >= self.expand_block_sim:
                    quarantine = True; qreason = "ambiguous_expand"
            chosen = self.next_gid; self.next_gid += 1
            if quarantine:
                dtype = "quarantine"
        # update state
        occ = self.occ.setdefault(chosen, {})
        for f, bx in zip(frames, boxes):
            occ[(video, f)] = np.asarray(bx, float)
        cr = self.crange.setdefault(chosen, {})
        rng = cr.get(cam)
        cr[cam] = [min(rng[0], t0), max(rng[1], t1)] if rng else [t0, t1]
        if world:                                 # accumulate the tracklet's world-by-second track
            w = self.world.setdefault(chosen, {})
            for s, ll in world.items():
                w.setdefault(s, []).append(ll)
        # exemplar admission keeps banks COHERENT (the anti-accretion gate). Reject if the new exemplar
        # is (a) a near-duplicate of an existing one (diversity gate, admit_tau) or (b) too FAR from the
        # bank (coherence_floor) - (b) is what stops an id from accreting other people into a spread,
        # "matches-everything" attractor. Quarantined tracklets never admit.
        admit = not quarantine
        # diagnostics: capture WHICH gate decided + the deciding cosine numbers.
        admit_reason = "added" if admit else "quarantine"
        admit_sim = admit_min = None        # ss.max() / ss.min() when a bank existed to compare against
        if admit and self.n_reps.get(chosen, 0) >= self.max_reps:
            admit = False
            admit_reason = "bank_full"
        elif admit and self.rep_mat.shape[0]:
            same = self.rep_mat[self.rep_gid == chosen]
            if same.shape[0]:
                ss = same @ pooled
                admit_sim, admit_min = float(ss.max()), float(ss.min())
                if float(ss.max()) >= self.admit_tau:
                    admit = False
                    admit_reason = "redundant"
                elif self.coherence_floor > 0 and float(ss.min()) < self.coherence_floor:
                    admit = False
                    admit_reason = "incoherent"
        if admit:
            self.rep_mat = np.vstack([self.rep_mat, pooled[None, :]])
            self.rep_gid = np.concatenate([self.rep_gid, [chosen]])
            self.n_reps[chosen] = self.n_reps.get(chosen, 0) + 1
            if self._cent is not None:    # keep the centroid cache in sync: only `chosen`'s bank changed
                self._cent[int(chosen)] = _normed(self.rep_mat[self.rep_gid == int(chosen)].mean(0))
                self._cent_sig = self._bank_sig()    # re-stamp the signature for this in-API change
        if rep_did is not None:        # per-tracklet decision trail: the why (candidates/scores/vetoes/choice) + added-vs-rejected (admitted)
            top = cands[:6]
            if qreason == "low_coherence":     # no candidates examined; report the self-coherence
                sc, vr = [round(float(self_coh), 4)], ["low_coherence"]
            else:                              # match / clean expand / ambiguous_expand: report candidate sims + vetoes
                sc, vr = [c[1] for c in top], [(c[2] or "") for c in top]
            self.decisions.append(dict(
                seq=self._seq, det_id=rep_did, chosen_gid=int(chosen), decision_type=dtype,
                threshold=round(float(self.tau), 4),
                candidate_gids=[int(c[0]) for c in top], scores=sc,
                cannot_link_pruned=[int(p) for p in pruned], veto_reasons=vr, admitted=bool(admit),
                # exemplar-admission diagnostics (which gate fired + the deciding numbers + the gate
                # cutoffs) so the viz can show WHY a matched tracklet was not added.
                admit_reason=admit_reason,
                admit_sim=(round(admit_sim, 4) if admit_sim is not None else None),
                admit_min=(round(admit_min, 4) if admit_min is not None else None),
                admit_tau=round(float(self.admit_tau), 4),
                coherence_floor=round(float(self.coherence_floor), 4),
                max_reps=int(self.max_reps)))
            self._seq += 1
        return chosen, sim, dtype, pruned, admit

    def assign_det(self, video, cam, frame, box, red, t, floor):
        """Per-DETECTION assignment against the (already-ripe) gallery: return (gid, True) for the nearest
        non-cannot-link identity whose cosine clears `floor`, else (-1, False). Does NOT admit or extend
        the gallery - used to decompose a low-coherence tracklet into detections that each find their own
        home (recovers within-tracklet ID-switches that pooling blends away). floor<=0 => nearest-always
        (no junk, no fragmentation). Banks stay frozen (clean pass-1)."""
        for g, s in self._candidates(red):
            if s < floor:
                break
            if self._cannot_link(g, video, cam, [frame], [box], t, t) is None:
                return int(g), True
        return -1, False

    def consolidate(self, merge_tau, *, return_events=True):
        """Agglomeratively merge over-split gids by exemplar-centroid cosine, never merging a cannot-link
        pair. Delegates to `_fast_consolidate`; `_consolidate_ref` is the reference implementation."""
        return _fast_consolidate(self, merge_tau, return_events=return_events)

    def _consolidate_ref(self, merge_tau, *, return_events=True):
        """Deferred revision (merge side): agglomeratively merge over-split gids by exemplar-centroid
        cosine in the reduced space, but NEVER merge a pair that violates cannot-link (same
        (video,frame), cross-camera simultaneity in non-overlapping FOVs, an impossible travel-time,
        or - if ``geo_max_m`` - a flat-ground world-track disagreement).

        Returns ``(remap, events)`` where ``remap`` is ``{old_gid -> surviving_gid}`` and ``events`` is
        the per-merge ``(survivor, absorbed, centroid_cosine)`` trail. Pass ``return_events=False`` to
        get just the ``remap`` dict.
        """
        groups = {}
        for g in sorted(set(self.rep_gid.tolist())):
            groups[g] = dict(gids={g}, reps=self.rep_mat[self.rep_gid == g],
                             occ=dict(self.occ.get(g, {})),
                             crange={c: list(r) for c, r in self.crange.get(g, {}).items()},
                             # per-second world position (median of the second's projections)
                             world={s: (float(np.median([p[0] for p in lst])), float(np.median([p[1] for p in lst])))
                                    for s, lst in self.world.get(g, {}).items()})

        def cent(grp):
            c = grp["reps"].mean(0); return c / (np.linalg.norm(c) + 1e-9)

        def cannot_merge(a, b):
            oa, ob = groups[a]["occ"], groups[b]["occ"]
            shared = set(oa) & set(ob)                              # shared (video,frame)s (within-camera)
            if shared:
                ov = [iou1(oa[k], ob[k]) for k in shared]
                if float(np.median(ov)) < self.same_box_iou:        # spatially distinct => two people
                    return True
            # GEO GATE: two ids can't be one person if their flat-ground world tracks disagree at
            # shared seconds (independent of appearance - prunes the far-apart look-alikes cosine can't).
            if self.geo_max_m > 0:
                wa, wb = groups[a]["world"], groups[b]["world"]
                sh = set(wa) & set(wb)
                if sh:
                    ds = [geo.meters(wa[s][0], wa[s][1], wb[s][0], wb[s][1]) for s in sh]
                    if float(np.median(ds)) > self.geo_max_m:
                        return True
            # cross-camera simultaneity / travel-time veto. These BLOCK unifying a person's same-time
            # fragments across overlapping cameras (the IDFN recall-recovery merge). Disabled under
            # merge_free_xcam so the global consolidation can do its job; the within-camera same-frame
            # guard above (different videos => no shared key => never fires cross-cam) still protects merges.
            if self.merge_free_xcam:
                return False
            for ca, (la, ha) in groups[a]["crange"].items():
                for cb, (lb, hb) in groups[b]["crange"].items():
                    if ca == cb:
                        continue
                    gap = max(0, lb - ha, la - hb)
                    if gap <= self.sw:                              # simultaneous
                        if frozenset((ca, cb)) not in self.overlaps:
                            return True
                    else:                                           # impossible travel
                        d = self.dist.get(frozenset((ca, cb)))
                        if d is not None and d / (gap / 1000.0) > self.max_speed:
                            return True
            return False

        changed = True
        events = []   # (survivor_gid, absorbed_gid, centroid_cosine) - for the merge decision trail
        while changed and len(groups) > 1:
            changed = False
            roots = list(groups); cents = {r: cent(groups[r]) for r in roots}
            best, bestcos = None, merge_tau
            for i in range(len(roots)):
                ci = cents[roots[i]]
                for j in range(i + 1, len(roots)):
                    cos = float(ci @ cents[roots[j]])
                    if cos >= bestcos and not cannot_merge(roots[i], roots[j]):
                        best, bestcos = (roots[i], roots[j]), cos
            if best:
                a, b = best
                events.append((int(a), int(b), round(float(bestcos), 4)))
                groups[a]["gids"] |= groups[b]["gids"]
                groups[a]["reps"] = np.vstack([groups[a]["reps"], groups[b]["reps"]])
                groups[a]["occ"].update(groups[b]["occ"])
                for c, (l, h) in groups[b]["crange"].items():
                    if c in groups[a]["crange"]:
                        groups[a]["crange"][c][0] = min(groups[a]["crange"][c][0], l)
                        groups[a]["crange"][c][1] = max(groups[a]["crange"][c][1], h)
                    else:
                        groups[a]["crange"][c] = [l, h]
                for s, ll in groups[b]["world"].items():
                    if s in groups[a]["world"]:
                        groups[a]["world"][s] = ((groups[a]["world"][s][0] + ll[0]) / 2, (groups[a]["world"][s][1] + ll[1]) / 2)
                    else:
                        groups[a]["world"][s] = ll
                del groups[b]; changed = True
        remap = {}
        for root, grp in groups.items():
            for g in grp["gids"]:
                remap[g] = root
        return (remap, events) if return_events else remap

    def apply_remap(self, remap):
        """Mutate live gallery state so merged gids are unified for subsequent ONLINE matching (used by
        the warmup batch-seed): relabel exemplars, union occ/crange, recompute n_reps."""
        if not remap:
            return
        self._cent = None        # gids relabeled to survivors -> centroid cache invalid (lazy full rebuild)
        self.rep_gid = np.array([remap.get(int(g), int(g)) for g in self.rep_gid.tolist()], np.int64)
        newocc = {}
        for g, o in self.occ.items():
            newocc.setdefault(remap.get(g, g), {}).update(o)
        self.occ = newocc
        newcr = {}
        for g, cr in self.crange.items():
            d = newcr.setdefault(remap.get(g, g), {})
            for c, (l, h) in cr.items():
                if c in d:
                    d[c] = [min(d[c][0], l), max(d[c][1], h)]
                else:
                    d[c] = [l, h]
        self.crange = newcr
        neww = {}
        for g, w in self.world.items():
            d = neww.setdefault(remap.get(g, g), {})
            for s, lst in w.items():
                d.setdefault(s, []).extend(lst)
        self.world = neww
        self.n_reps = {}
        for g in self.rep_gid.tolist():
            self.n_reps[int(g)] = self.n_reps.get(int(g), 0) + 1

    def resolve(self, theta, *, resolve_mode="global_agglom", top_k=30, exclude_same_cam=True,
                cannot_link=False, return_remap=True):
        """GLOBAL re-partition (training-free): re-cluster ALL live identities from scratch over their
        pooled exemplar centroids, replacing the streamed greedy partition with a fresh kNN-sparse
        cosine + average-linkage agglomerative partition at a single global ``theta``.

        Unlike :meth:`consolidate` (which only MERGES greedy gid-centroids and cannot undo a greedy
        over-merge), ``resolve`` discards the greedy boundaries and re-partitions the points, so a
        greedy over-split AND a greedy over-merge are both recoverable.
        Mutates live state via :meth:`apply_remap`; returns ``{old_gid: new_gid}``
        (or the per-gid label array if ``return_remap=False``). Each live gid is one resolve item; its
        item embedding is its L2-normed pooled exemplar centroid (== :meth:`centroids`)."""
        if resolve_mode != "global_agglom":
            raise ValueError(f"resolve: unknown resolve_mode {resolve_mode!r}")
        from vlincs_gallery.resolve import global_agglom_resolve
        gids = self._gids()
        if len(gids) <= 1:
            return {g: g for g in gids} if return_remap else np.zeros((len(gids),), np.int64)
        cents = self.centroids()                                  # aligned with _gids(), L2-normed
        # one camera code per gid: the first camera in its crange (or "" -> shared code 0 if unknown)
        cam_names = [sorted(self.crange.get(g, {}).keys())[0] if self.crange.get(g) else ""
                     for g in gids]
        uniq = {c: i for i, c in enumerate(sorted(set(cam_names)))}
        cam_codes = np.array([uniq[c] for c in cam_names], np.int64)
        cl_pairs = None
        if cannot_link:
            def _iou(a, b):
                ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                return inter / ua if ua > 0 else 0.0
            cl_pairs = []
            for ai in range(len(gids)):
                oa = self.occ.get(gids[ai], {})
                if not oa:
                    continue
                for bj in range(ai + 1, len(gids)):
                    ob = self.occ.get(gids[bj], {})
                    shared = set(oa) & set(ob)
                    if shared and float(np.median([_iou(oa[k], ob[k]) for k in shared])) < self.same_box_iou:
                        cl_pairs.append((ai, bj))
        rr = global_agglom_resolve(cents, cam_codes, theta, top_k=top_k,
                                   exclude_same_cam=exclude_same_cam, cannot_link_pairs=cl_pairs)
        lab = rr.labels
        label_to_gids = {}
        for k_, g in enumerate(gids):
            label_to_gids.setdefault(int(lab[k_]), []).append(g)
        remap = {}
        for members in label_to_gids.values():
            survivor = min(members)
            for g in members:
                remap[g] = survivor
        self.apply_remap(remap)
        return remap if return_remap else lab


    def camera_span_stats(self) -> dict:
        """GT-free fragmentation diagnostic over the LIVE gallery (per-detection analog of
        ``vlincs_sdk.clustering.gid_camera_span_stats``). Reads the authoritative per-gid camera set
        out of ``crange`` and the per-gid exemplar count out of ``rep_gid``. Returns ``n_gids``,
        ``mean_cameras_per_gid``, ``frac_gids_ge3_cams``, ``max_cameras_per_gid``,
        ``singleton_gid_fraction`` (singleton == bank holds a single exemplar)."""
        gids = self._gids()
        if not gids:
            return {"n_gids": 0, "mean_cameras_per_gid": 0.0, "frac_gids_ge3_cams": 0.0,
                    "max_cameras_per_gid": 0, "singleton_gid_fraction": 0.0}
        cams_per_gid = [len(self.crange.get(g, {})) for g in gids]
        singletons = sum(1 for g in gids if int((self.rep_gid == g).sum()) == 1)
        n = len(gids)
        return {
            "n_gids": n,
            "mean_cameras_per_gid": sum(cams_per_gid) / n,
            "frac_gids_ge3_cams": sum(1 for c in cams_per_gid if c >= 3) / n,
            "max_cameras_per_gid": max(cams_per_gid),
            "singleton_gid_fraction": singletons / n,
        }

    def split_low_coherence(
        self,
        coherence_threshold: float,
        det_assign: dict[str, int],
        det_embs: dict[str, np.ndarray],
        *,
        coherence_metric: str = "avg",
        strategy: str = "spectral_2way",
        min_reps_to_check: int = 2,
        max_passes: int = 4,
    ) -> dict[str, int]:
        """Revisability-A (SPLIT side): the DUAL of :meth:`consolidate`.

        For each live gid, measure within-gid appearance coherence over its diversity-gated EXEMPLAR
        BANK (the rows of ``rep_mat`` carrying that gid) using the SDK's ``_coherence`` reducer
        (``avg``/``min``/``p25`` of upper-triangle pairwise cosine). If
        ``coherence < coherence_threshold`` the gid is over-merged: cut its exemplar bank in two via
        the SDK's ``_spectral_2way`` and reassign every owned DETECTION to the nearest resulting
        sub-centroid (MAX-cosine - the matcher's own rule). Iterates up to ``max_passes`` so a gid
        welded from >2 prototypes peels apart one cut per pass.

        **TODO(unify):** lift a ``split_assignment_low_coherence(assign, embeddings, ...)`` primitive
        into ``vlincs_sdk.clustering`` so both this gallery and the batch pipeline share it.

        Returns ``{det_id -> new_gid}`` - detections of unsplit gids keep their gid; detections of
        split gids are routed to a fresh sub-gid by nearest sub-centroid.
        """
        from vlincs_sdk.clustering import (  # SDK helpers (lazy)
            _coherence as _sdk_coherence,
            _spectral_2way as _sdk_spectral_2way,
        )
        if strategy not in ("atomic", "spectral_2way"):
            raise ValueError(f"unknown strategy: {strategy}")
        gids = self._gids()
        if not gids:
            return dict(det_assign)

        # Group detections by current gid (only those we have embeddings for).
        dets_of: dict[int, list[str]] = {}
        for did, gid in det_assign.items():
            if did in det_embs:
                dets_of.setdefault(int(gid), []).append(did)

        # Per-gid "seed centroids": start from the live exemplar bank rows.
        seeds: dict[int, np.ndarray] = {g: self.rep_mat[self.rep_gid == g].astype(np.float32) for g in gids}
        next_id = (max(det_assign.values()) if det_assign else max(gids)) + 1
        out = dict(det_assign)

        for gid in gids:
            bank = seeds[gid]
            if len(bank) < min_reps_to_check:
                continue
            dets = dets_of.get(gid, [])
            if len(dets) < 2:
                continue  # nothing to fan out even if the bank is incoherent

            # Iteratively peel the bank into coherent sub-banks. Each entry is a (sub_id, vecs) pair
            # where sub_id is the gid the sub-bank's detections will carry.
            work: list[tuple[int, np.ndarray]] = [(gid, bank)]
            final: list[tuple[int, np.ndarray]] = []
            for _pass in range(max_passes):
                still: list[tuple[int, np.ndarray]] = []
                changed = False
                for sub_id, vecs in work:
                    if len(vecs) < min_reps_to_check or _sdk_coherence(vecs, coherence_metric) >= coherence_threshold:
                        final.append((sub_id, vecs))
                        continue
                    if strategy == "atomic":
                        for k, v in enumerate(vecs):
                            sid = sub_id if k == 0 else next_id
                            if k > 0:
                                next_id += 1
                            final.append((sid, v[None, :]))
                        changed = True
                        continue
                    labels = _sdk_spectral_2way(vecs)
                    if int(labels.max()) != 1:  # sklearn-missing fallback returns arange, not {0,1}
                        final.append((sub_id, vecs))  # no usable 2-way cut → leave the bank intact
                        continue
                    a = vecs[labels == 0]
                    b = vecs[labels == 1]
                    if len(a) == 0 or len(b) == 0:  # degenerate cut → no split
                        final.append((sub_id, vecs))
                        continue
                    still.append((sub_id, a))          # sub-label 0 keeps the id
                    still.append((next_id, b))         # sub-label 1 gets a fresh id
                    next_id += 1
                    changed = True
                work = still
                if not changed:
                    break
            final.extend(work)  # anything still pending after max_passes

            if len(final) <= 1:
                continue  # no actual split happened

            # Reassign each detection of this gid to the nearest sub-centroid (MAX-cosine).
            sub_ids = [sid for sid, _ in final]
            sub_cents = np.stack([_normed(v.mean(0)) for _, v in final])  # (S, dim)
            for did in dets:
                v = det_embs[did]
                best = int(np.argmax(sub_cents @ v))
                out[did] = sub_ids[best]

        return out

    def revise(
        self,
        det_assign: dict[str, int],
        det_embs: dict[str, np.ndarray],
        *,
        auto: bool = True,
        mode: str | None = None,
        merge_tau: float | None = None,
        split_tau: float | None = None,
        camera_span_gate: float = 2.0,
        **split_kwargs,
    ) -> tuple[dict[str, int], str]:
        """Single revisability-A entry point: GT-free auto-gate that picks merge vs split.

        Uses :meth:`camera_span_stats`: ``mean_cameras_per_gid > camera_span_gate`` ⇒ over-merged ⇒
        **split**; ``<= gate`` ⇒ over-fragmented ⇒ **merge** (consolidate). When ``auto=False`` the
        direction is taken from ``mode`` (``"merge"`` | ``"split"`` | ``"none"``).

        Returns ``(new_det_assign, applied)`` where ``applied`` ∈ {``"merge"``, ``"split"``,
        ``"none"``}.
        """
        if auto:
            span = self.camera_span_stats()["mean_cameras_per_gid"]
            mode = "split" if span > camera_span_gate else "merge"
        elif mode is None:
            raise ValueError("revise: pass auto=True or mode in {merge,split,none}")

        if mode == "merge":
            mt = merge_tau if merge_tau is not None else self.tau
            remap = self.consolidate(mt, return_events=False)
            return ({did: remap[g] for did, g in det_assign.items()}, "merge")
        if mode == "split":
            st = split_tau if split_tau is not None else 0.55
            return (self.split_low_coherence(st, det_assign, det_embs, **split_kwargs), "split")
        if mode == "none":
            return (dict(det_assign), "none")
        raise ValueError(f"revise: unknown mode {mode!r}")

    def stats(self) -> dict:
        gids = self._gids()
        mean_reps = round(float(np.mean([int((self.rep_gid == g).sum()) for g in gids])), 2) if gids else 0.0
        return dict(n_gids=self.n, mean_reps=mean_reps, disc_ratio=round(self.discriminability(), 3))
