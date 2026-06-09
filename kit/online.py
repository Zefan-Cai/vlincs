"""Online cross-camera identity gallery — the kit's core.

The colleague's pipeline drives this LIVE, on THEIR cadence: push a detection or a tracklet (their
detector/tracker/embedder — their problem), get a global id back, periodically `resolve()`. ALL state
lives in the DB (pgvector, the durable system-of-record) + a hot in-process index (FAISS-equivalent:
exact inner-product cosine over the identity exemplar bank). No files are produced until `export_submission()`.
No weights, no models — the gallery matches on whatever embedding you push.

    g = OnlineGallery("ds1")                            # connects to the EMPTY db; loads camera geo from the dataset
    for v, cam, fr, box, emb in their_pipeline():       # any cadence
        gid = g.add_detection(v, cam, fr, box, emb)     # match / expand / do-nothing -> gid, persisted live
    g.resolve()                                         # periodic consolidation (call it whenever they like)
    print(g.score())                                    # IDF1 from the DB (ms02/ds1; ds2 has no GT)
    g.export_submission("out.zip")                      # the ONLY file ever written

Tracklet form (their tracker): g.add_tracklet(video, camera, frames, boxes, embedding[, confs]).
The match/expand/do-nothing + consolidate logic is the validated matcher `vlincs_gallery.gallery.
IdentityGallery` (the ONE matcher, imported straight from the package — no scripts/ path-hack); this
wraps it as a streaming, DB-persisting service and adds camera geo + abs-clock from the dataset's
shipped extrinsics (so the cross-camera simultaneity/travel vetoes work without you supplying timing).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

# Heavy/DB deps (psycopg, pgvector, the vlincs_gallery package matcher) are loaded LAZILY —
# `import online` then works without them (e.g. to read CARDDIRS or run example.py --help), and
# they're only required when you actually build an OnlineGallery (which needs a live DB). Keeps the module
# importable for inspection and makes the missing-deps failure mode an actionable message, not an
# opaque ImportError on the package matcher import.
psycopg = register_vector = _Matcher = self_coherence = _haversine_m = None
gdb = video_epochs = camera_geo = evaluate = load_ds1_gt_by_video = load_ms02_gt = None


def _load_deps():
    """Import the runtime/DB deps on first OnlineGallery construction; raise an actionable error if absent."""
    global psycopg, register_vector, _Matcher, self_coherence, _haversine_m
    global gdb, video_epochs, camera_geo, evaluate, load_ds1_gt_by_video, load_ms02_gt
    if psycopg is not None:
        return
    try:
        import psycopg as _psycopg
        from pgvector.psycopg import register_vector as _register_vector
        # the ONE matcher now lives in the importable package (no scripts/ path-hack needed)
        from vlincs_gallery.gallery import (IdentityGallery as _gallery,
                                            self_coherence as _self_coh, _haversine_m as _haversine)
        from vlincs_gallery import db as _gdb
        from vlincs_gallery.clock import video_epochs as _video_epochs, camera_geo as _camera_geo
        from vlincs_gallery.eval.score import (evaluate as _evaluate,
                                               load_ds1_gt_by_video as _load_ds1, load_ms02_gt as _load_ms02)
    except ImportError as e:
        raise ImportError(
            f"online.OnlineGallery needs the kit's runtime deps ({e}). Run inside the kit container "
            "(`docker compose ...`), or `pip install -r kit/requirements.txt` in your environment. "
            "(`import online` itself works without them — they're only needed to build a Gallery.)"
        ) from e
    psycopg, register_vector = _psycopg, _register_vector
    _Matcher, self_coherence, _haversine_m = _gallery, _self_coh, _haversine
    gdb, video_epochs, camera_geo = _gdb, _video_epochs, _camera_geo
    evaluate, load_ds1_gt_by_video, load_ms02_gt = _evaluate, _load_ds1, _load_ms02


# Data locations live in ONE place (vlincs_gallery.paths). Re-exported here so the CLI keeps importing
# them from `online` unchanged. paths is dependency-free, so this doesn't break online's import-light design.
from vlincs_gallery.paths import DATA, CARDDIRS, HAS_GT   # noqa: E402  (re-export for kit/cli.py)

# --- SQL the ingest/score path runs, named here so the methods below read as one-liners ---
_INSERT_DETECTION = """
    INSERT INTO detections (det_id, video, camera, frame_idx, wall_clock_ms, abs_ms,
        x1, y1, x2, y2, conf, object_type, embedding, embedding_red)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (det_id) DO NOTHING"""

_UPSERT_ASSIGNMENT = """
    INSERT INTO assignments (det_id, gid, score, decision_type, seq)
    VALUES (%s,%s,%s,%s,%s) ON CONFLICT (det_id) DO UPDATE SET gid=EXCLUDED.gid"""

_INSERT_DECISION_LOG = """
    INSERT INTO decision_log (seq, det_id, chosen_gid, decision_type, admitted,
        candidate_gids, scores, cannot_link_pruned, veto_reasons, threshold,
        admit_reason, admit_sim, admit_min, admit_tau, coherence_floor, max_reps)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (seq) DO NOTHING"""

_INSERT_IDENTITY_REP = """
    INSERT INTO identity_reps (rep_id, gid, det_id, embedding, embedding_red)
    VALUES (%s,%s,%s,%s,%s) ON CONFLICT (rep_id) DO NOTHING"""

# Recompute one identity's span/cameras/member-count from its committed assignments (see _refresh_identity_spans).
_REFRESH_IDENTITY_SPANS = """
    UPDATE identities i SET n_members=COALESCE(s.n,0), first_wall_ms=s.lo,
        last_wall_ms=s.hi, cameras=COALESCE(s.cams,'{}')
    FROM (SELECT count(*) n, min(d.abs_ms) lo, max(d.abs_ms) hi, array_agg(DISTINCT d.camera) cams
          FROM assignments a JOIN detections d ON d.det_id=a.det_id WHERE a.gid=%s) s
    WHERE i.gid=%s"""

_SELECT_ASSIGNMENTS = """
    SELECT d.video, d.camera, d.frame_idx, a.gid, d.x1, d.y1, d.x2, d.y2
    FROM detections d JOIN assignments a ON a.det_id=d.det_id"""

_INSERT_MERGE = "INSERT INTO merges (at_seq, old_gid, new_gid, score) VALUES (%s, %s, %s, %s)"


class OnlineGallery:
    """Streaming, DB-backed online identity gallery service. One per dataset.

    This is the kit's user-facing SERVICE that wraps the matcher (`vlincs_gallery.gallery.IdentityGallery`,
    stored as `self.m`): it adds pgvector DB persistence, camera geo + absolute-clock from the dataset's
    shipped extrinsics, and the streaming API. The match/expand/consolidate logic itself lives in the
    matcher, not here. (Named `OnlineGallery` to keep it distinct from the matcher class.)"""

    def __init__(self, dataset: str, *, fps: float = 30.0, truncate: bool = True,
                 tau: float = 0.60, merge_tau: float = 0.35, coherence_floor: float = 0.4,
                 match_mode: str = "centroid",  # locked whole-bank-mean fix: DS1 0.4677 -> 0.5446
                 tracklet_coh_min: float = 0.0, admit_tau: float = 0.9, max_reps: int = 16,
                 max_speed: float = 3.0, sim_window_ms: int = 200, same_box_iou: float = 0.35,
                 cannot_link: bool = True, overlaps=None, batch_commit: int = 1):
        _load_deps()  # import the DB deps now; clear, actionable error if they're absent
        d = dataset.lower()
        self.dataset = ("ms02" if d.startswith(("ms02", "ds0000")) else
                        "ds1" if d.startswith(("ds1", "ds0001")) else
                        "ds2" if d.startswith(("ds2", "ds0002")) else d)
        self.fps = fps
        self.merge_tau = merge_tau
        gdb.ensure_db(self.dataset)                       # create db + apply schema (empty)
        self.con = psycopg.connect(gdb.dsn(self.dataset), autocommit=False)
        register_vector(self.con)
        cur = self.con.cursor()
        if truncate:
            cur.execute("TRUNCATE detections, identities, identity_reps, assignments, decision_log, "
                        "merges, models, cameras RESTART IDENTITY CASCADE;")
        # camera geo + per-video start clock from the dataset's SHIPPED extrinsics (we provide this)
        self.epochs = video_epochs(*CARDDIRS.get(self.dataset, []))
        geo = camera_geo(*CARDDIRS.get(self.dataset, []))
        self.video_cam = {stem: g.get("camera", "CAM") for stem, g in geo.items()}
        cam_xy = {}
        for stem, g in geo.items():
            cam = g.get("camera", stem)
            cur.execute("""INSERT INTO cameras (video, camera, lat, lon, alt, roll, pitch, yaw, start_ms, fps)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (video) DO NOTHING""",
                        (stem, cam, g.get("lat"), g.get("lon"), g.get("alt"), g.get("roll"),
                         g.get("pitch"), g.get("yaw"), int(round(self.epochs.get(stem, 0.0))), fps))
            if g.get("lat") is not None:
                cam_xy[cam] = (g["lat"], g["lon"])
        dist = {frozenset((a, b)): _haversine_m(cam_xy[a], cam_xy[b])
                for a in cam_xy for b in cam_xy if a < b}
        self.con.commit()
        self.m = _Matcher(cam_xy, dist, set(overlaps or []), tau, max_speed, sim_window_ms,
                          admit_tau, max_reps, same_box_iou=same_box_iou,
                          coherence_floor=coherence_floor, tracklet_coh_min=tracklet_coh_min,
                          match_mode=match_mode)
        if not cannot_link:
            # ablation (default is ON): drop ALL vetoes — same_frame (two distinct boxes in one frame
            # can't share an id), simultaneity + travel (cross-camera). OFF reproduces the old
            # appearance-only "best DS1 config"; ON is the physically-correct default.
            self.m._cannot_link = lambda *a, **k: None
            self.m.merge_free_xcam = True
        # Persist the decision config (the knobs this run was built with) as a models row so the viz can
        # SHOW what the current DB state reflects — otherwise "is cannot_link on?" is unanswerable from
        # the data. role='gallery' fits the models table's "settings provenance" purpose; /meta returns it.
        self.config = {"cannot_link": bool(cannot_link), "match_mode": match_mode,
                       "tau": tau, "merge_tau": merge_tau, "coherence_floor": coherence_floor,
                       "tracklet_coh_min": tracklet_coh_min, "admit_tau": admit_tau, "max_reps": max_reps,
                       "max_speed": max_speed, "sim_window_ms": sim_window_ms, "same_box_iou": same_box_iou}
        gdb.upsert_model(cur, "gallery", "online-gallery", params=self.config)
        self.con.commit()
        self._emb_dim = None
        self._pending = 0
        self._batch = max(1, batch_commit)
        self._ndet = 0
        self._last_seq = -1        # the most recent decision seq committed (the step a resolve() merge takes effect)
        self._trk_seq = 0          # monotonic per-tracklet counter -> auto-generated det_ids never collide
        # ADDITIVE pgvector persistence (viz + hnsw ANN path only — the matcher's rep_mat is untouched).
        # The schema pins vector(64)/(1024); we persist embedding_red only when the pushed dim == 64 and
        # the raw embedding only when dim == 1024 (the kit is embedder-agnostic, so a different dim just
        # leaves the columns NULL — logged once — and the matcher/viz still work, minus the embedding view).
        self._red_dim, self._raw_dim = 64, 1024
        self._dim_warned = False

    def _vec_if_dim(self, vec, want_dim):
        """Return a float32 numpy vector for pgvector iff `vec` already has exactly `want_dim` dims, else
        None (so the column persists NULL). NO truncation/padding — pgvector vector(N) is fixed-width and a
        silently reshaped embedding would be a lie. The first time a pushed dim matches neither the 64-d nor
        the 1024-d schema column, log a single embedder-agnostic note and never crash."""
        if int(vec.shape[0]) == int(want_dim):
            return np.ascontiguousarray(vec, dtype=np.float32)
        if (not self._dim_warned and self._emb_dim is not None
                and self._emb_dim not in (self._red_dim, self._raw_dim)):
            print(f"[online] note: pushed embedding dim={self._emb_dim} matches neither vector(64) nor "
                  f"vector(1024); detections.embedding/embedding_red + identity_reps left NULL "
                  f"(matcher + score unaffected; the embedding-projection viz will be empty).")
            self._dim_warned = True
        return None

    def _refresh_identity_spans(self, cur, gids):
        """Recompute identities.{n_members,first_wall_ms,last_wall_ms,cameras} from the committed
        assignments for the given gids. A batch persist would fill these; the streaming path
        otherwise leaves them at insert defaults (first_wall_ms NULL, cameras '{}'), which makes the viz
        /state query — `WHERE i.first_wall_ms <= t` — return NOTHING (NULL <= t is UNKNOWN), so the
        "identities so far" / "cross-camera" / "dets assigned" KPIs and the identity list never populate.
        A gid with no remaining assignments (e.g. merged away in resolve) recomputes to n=0 /
        first_wall_ms=NULL and so drops back out of /state. Cheap (asn_gid-indexed, scoped per gid)."""
        for gid in {int(x) for x in gids}:
            cur.execute(_REFRESH_IDENTITY_SPANS, (gid, gid))

    # ---- the streaming API (their cadence) ----
    def add_detection(self, video: str, camera: str, frame: int, box, embedding, conf: float = 1.0,
                      object_type: int = 0, det_id: str | None = None) -> int:
        """Match-or-expand a single detection. Returns its global id."""
        return self.add_tracklet(video, camera, [frame], [box], embedding, confs=[conf],
                                 object_type=object_type, det_ids=[det_id] if det_id else None)

    def add_tracklet(self, video: str, camera: str, frames, boxes, embedding, confs=None,
                     object_type: int = 0, det_ids=None) -> int:
        """Match-or-expand a tracklet (their tracker's output: same-camera dets + one pooled embedding).
        `embedding` may be (D,) pooled or (n,D) per-detection (mean-pooled here). Returns its global id."""
        emb = np.asarray(embedding, np.float32)
        per_det = emb.ndim == 2
        pooled = emb.mean(0) if per_det else emb
        pooled = pooled / (np.linalg.norm(pooled) + 1e-9)
        if self._emb_dim is None:                          # first add fixes the index dim (any D)
            self._emb_dim = pooled.shape[0]
            self.m.rep_mat = np.zeros((0, self._emb_dim), np.float32)
        n_dets = len(frames)
        if det_ids is None:                                # the {i} is a WITHIN-tracklet index, so two
            self._trk_seq += 1                             # tracklets with a det at the same (video,frame)
            det_ids = [f"{video}::{camera}:{int(frames[i])}:{i}:t{self._trk_seq}"   # would collide on det_id
                       for i in range(n_dets)]             # (PK) and silently drop -> tag with a per-tracklet seq.
        epoch_ms = int(round(self.epochs.get(video, 0.0)))
        det_abs_ms = [epoch_ms + int(round(int(frames[i]) / self.fps * 1000.0)) for i in range(n_dets)]
        boxes = [np.asarray(b, float) for b in boxes]
        confs = list(confs) if confs is not None else [1.0] * n_dets
        self_coh = self_coherence(list(emb)) if per_det else 1.0
        gid, score, _dtype, _pruned, _admitted = self.m.match_or_expand(
            video, camera, [int(f) for f in frames], boxes, min(det_abs_ms), max(det_abs_ms),
            pooled.astype(np.float32), rep_did=det_ids[0], self_coh=self_coh)
        # persist live: detections + assignment + the decision (the viz reads this)
        cur = self.con.cursor()
        decision = self.m.decisions[-1]
        # The matcher's appearance state still lives ONLY in the in-process index (rep_mat = FAISS-equivalent);
        # the columns below are an ADDITIVE copy for the viz + the hnsw ANN path. We persist the pooled
        # match-space vector the gallery actually scored: into embedding_red when it's 64-d (the schema's
        # reduced match column), into the raw embedding when it's 1024-d. Any other dim -> NULL (logged once).
        red_vec = self._vec_if_dim(pooled, self._red_dim)     # 64-d -> vector(64), else None
        raw_vec = self._vec_if_dim(pooled, self._raw_dim)     # 1024-d -> vector(1024), else None
        decision_type = decision["decision_type"] if decision["decision_type"] in ("match", "expand") else "revised"
        for i in range(n_dets):
            box = boxes[i]
            wall_clock_ms = int(round(int(frames[i]) / self.fps * 1000.0))
            cur.execute(_INSERT_DETECTION,
                        (det_ids[i], video, camera, int(frames[i]), wall_clock_ms, det_abs_ms[i],
                         float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                         float(confs[i]), int(object_type), raw_vec, red_vec))
            cur.execute("INSERT INTO identities (gid, n_members) VALUES (%s,0) ON CONFLICT (gid) DO NOTHING",
                        (int(gid),))
            cur.execute(_UPSERT_ASSIGNMENT,
                        (det_ids[i], int(gid), float(score), decision_type, decision["seq"]))
        cur.execute(_INSERT_DECISION_LOG,
                    (decision["seq"], det_ids[0], int(gid), decision["decision_type"], bool(decision["admitted"]),
                     decision["candidate_gids"], decision["scores"], decision["cannot_link_pruned"],
                     decision["veto_reasons"], decision["threshold"],
                     decision.get("admit_reason"), decision.get("admit_sim"), decision.get("admit_min"),
                     decision.get("admit_tau"), decision.get("coherence_floor"), decision.get("max_reps")))
        # If the matcher ADMITTED this tracklet's pooled vector to its exemplar bank (a row appended to
        # rep_mat/rep_gid), mirror it into identity_reps so the viz/ANN see the same bank. rep_id = the
        # decision seq (stable, never reused within a run). embedding_red is NOT NULL in the schema, so a
        # rep row is only writable when the pushed dim == 64; for other dims we skip it (logged once).
        if decision["admitted"] and red_vec is not None:
            cur.execute(_INSERT_IDENTITY_REP,
                        (int(decision["seq"]), int(gid), det_ids[0], raw_vec, red_vec))
        # keep this identity's span/cameras/member-count current so the viz /state (and its KPIs) reflect
        # the gallery as of t — otherwise these columns stay NULL and /state returns nothing.
        self._refresh_identity_spans(cur, [gid])
        self._last_seq = int(decision["seq"])     # the step a subsequent resolve()'s merges take effect at
        self._ndet += n_dets
        self._pending += 1
        if self._pending >= self._batch:
            self.con.commit()
            self._pending = 0
        return int(gid)

    def resolve(self) -> dict:
        """Periodic resolve (consolidation): merge over-split identities; update the DB AND the live
        in-process index so subsequent online matches use the merged ids. Call on any cadence."""
        remap, events = self.m.consolidate(self.merge_tau)
        self.m.apply_remap(remap)                          # unify live gallery state for future adds
        real = [(old, new) for old, new in remap.items() if old != new]
        if real:
            cur = self.con.cursor()
            touched = set()
            for old, new in real:                          # move assignments + bank to the FINAL survivor
                cur.execute("UPDATE assignments SET gid=%s WHERE gid=%s", (int(new), int(old)))
                # keep the persisted exemplar bank's gids in sync with the merged in-process bank so the
                # embedding-projection viz colours reps by the SAME (post-consolidation) gid as assignments
                cur.execute("UPDATE identity_reps SET gid=%s WHERE gid=%s", (int(new), int(old)))
                touched.update((int(old), int(new)))
            # record each merge STEP from the consolidate trail (survivor, absorbed, centroid_cosine) at the
            # current ingest step. The cosine is the WHY (>= merge_tau); the step is the WHEN; the pair is the
            # WHERE. This also lets decision-order replay reconstruct the exact pre-merge state at any step.
            for survivor, absorbed, cosine in (events or []):
                cur.execute(_INSERT_MERGE, (self._last_seq, int(absorbed), int(survivor), float(cosine)))
                touched.update((int(absorbed), int(survivor)))
            # re-derive spans for both survivors (grew) and merged-away gids (now empty -> drop from /state)
            self._refresh_identity_spans(cur, touched)
            self.con.commit()
        return {"merges": len(real), "events": len(events) if events else 0}

    def score(self) -> dict:
        """IDF1 (canonical reid_hota, leaderboard config) from the live DB. ds2 has no GT -> None."""
        if not HAS_GT.get(self.dataset):
            return {"dataset": self.dataset, "idf1": None, "note": "no GT shipped for this dataset (leaderboard only)"}
        self.con.commit()
        with self.con.cursor() as cur:
            cur.execute(_SELECT_ASSIGNMENTS)
            rows = cur.fetchall()
        import pandas as pd
        # DS1 reuses camera names across Tc6/Tc8, so it's scored per VIDEO; MS02 per CAMERA.
        score_per = "video" if self.dataset == "ds1" else "camera"
        pred_rows_by_key: dict = {}
        for video_stem, camera, frame, gid, x1, y1, x2, y2 in rows:
            key = video_stem if score_per == "video" else camera
            pred_rows_by_key.setdefault(key, []).append((frame, gid, x1, y1, x2, y2, 0))
        pred_by_key = {key: pd.DataFrame(rows, columns=["frame", "id", "x1", "y1", "x2", "y2", "object_type"])
                       for key, rows in pred_rows_by_key.items()}
        gt = load_ms02_gt() if self.dataset == "ms02" else load_ds1_gt_by_video()
        gt = {key: df for key, df in gt.items() if key in pred_by_key}
        if not gt:
            return {"dataset": self.dataset, "idf1": None,
                    "note": f"no GT overlap with pushed keys {sorted(pred_by_key)} — check the dataset mount (DATA_ROOT)"}
        metrics = evaluate(gt, {key: pred_by_key[key] for key in gt}, dense=False, n_workers=1)   # n_workers=1: the canonical call
        n_ids = len({gid for df in pred_by_key.values() for gid in df["id"]})
        return {"dataset": self.dataset, "idf1": round(metrics.idf1, 4), "assa": round(metrics.assa, 4),
                "detre": round(metrics.detre, 4), "n_ids": n_ids}

    def export_submission(self, path: str) -> str:
        """The ONLY file this kit writes: a canonical TA1 submission from the live DB assignments."""
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from submit import export as _export
        return _export(self.con, self.dataset, path)

    def close(self):
        self.con.commit()
        self.con.close()
