"""Online cross-camera identity gallery — the kit's core.

The colleague's pipeline drives this live, on their cadence: push a detection or a tracklet (their
detector/tracker/embedder), get a global id back, periodically `resolve()`. All state
lives in the DB (pgvector, the durable system-of-record) + a hot in-process index (exact
inner-product cosine over the identity exemplar bank). No files are produced until `export_submission()`.
No weights, no models — the gallery matches on whatever embedding you push.

    g = OnlineGallery("ds1")                            # connects to the empty db; loads camera geo from the dataset
    for v, cam, fr, box, emb in their_pipeline():       # any cadence
        gid = g.add_detection(v, cam, fr, box, emb)     # match / expand / do-nothing -> gid, persisted live
    g.resolve()                                         # periodic consolidation
    print(g.score())                                    # IDF1 from the DB (ms02/ds1; ds2 has no GT)
    g.export_submission("out.zip")                      # the only file ever written

Tracklet form (their tracker): g.add_tracklet(video, camera, frames, boxes, embedding[, confs]).
The match/expand/do-nothing + consolidate logic is the matcher `vlincs_gallery.gallery.
IdentityGallery`; this wraps it as a streaming, DB-persisting service and adds camera geo + abs-clock
from the dataset's shipped extrinsics (so the cross-camera simultaneity/travel vetoes work without you
supplying timing).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

# Heavy/DB deps (psycopg, pgvector, the vlincs_gallery package matcher) are loaded lazily —
# `import online` works without them (e.g. to read CARDDIRS or run example.py --help); they're
# only required when you actually build an OnlineGallery (which needs a live DB). Keeps the module
# importable for inspection and makes the missing-deps failure mode an actionable message.
psycopg = register_vector = _Matcher = self_coherence = _haversine_m = None
gdb = load_clock = evaluate = load_ds1_gt_by_video = load_ms02_gt = None


def _load_deps():
    """Import the runtime/DB deps on first OnlineGallery construction; raise an actionable error if absent."""
    global psycopg, register_vector, _Matcher, self_coherence, _haversine_m
    global gdb, load_clock, evaluate, load_ds1_gt_by_video, load_ms02_gt
    if psycopg is not None:
        return
    try:
        import psycopg as _psycopg
        from pgvector.psycopg import register_vector as _register_vector
        from vlincs_gallery.gallery import (IdentityGallery as _gallery,
                                            self_coherence as _self_coh, _haversine_m as _haversine)
        from vlincs_gallery import db as _gdb
        from vlincs_gallery.clock import load_clock as _load_clock
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
    gdb, load_clock = _gdb, _load_clock
    evaluate, load_ds1_gt_by_video, load_ms02_gt = _evaluate, _load_ds1, _load_ms02


# Data locations live in one place (vlincs_gallery.paths). Re-exported here so the CLI keeps importing
# them from `online` unchanged. paths is dependency-free, so this doesn't break online's import-light design.
from vlincs_gallery.paths import DATA, CARDDIRS, HAS_GT, EXTRINSICS_DIRS   # noqa: E402  (re-export for kit/cli.py)

# --- SQL the ingest/score path runs, named here so the methods below read as one-liners ---
_INSERT_DETECTION = """
    INSERT INTO detections (det_id, video, camera, frame_idx, wall_clock_ms, abs_ms,
        x1, y1, x2, y2, conf, object_type)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (det_id) DO NOTHING"""

# Embeddings go to the polymorphic `embeddings` table (any dim, multi-model). The gallery's unit of work
# is the tracklet, so this is one row per tracklet — the pooled match-space vector the gallery scored —
# not one per detection (detections + assignments carry the per-detection rows). It records the birth gid +
# the decision seq (the tracklet id / as-of-step replay key) + is_rep (admitted to the exemplar bank: the
# viz "bank" subset). vec is unconstrained, so any embedder dim just stores.
_INSERT_EMBEDDING = """
    INSERT INTO embeddings (entity_kind, entity_id, model_id, role, dim, vec, gid, seq, is_rep)
    VALUES ('tracklet', %s, %s, 'match', %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""

# A second per-tracklet vector under role='resolve' (distinct model_id) — the gallery stores both the match
# space and the resolve space; resolve_global() reads these back. PK (entity_kind,entity_id,model_id,role)
# lets it coexist with the role='match' row on the same rep det.
_INSERT_RESOLVE_EMB = """
    INSERT INTO embeddings (entity_kind, entity_id, model_id, role, dim, vec, gid, seq, is_rep)
    VALUES ('tracklet', %s, %s, 'resolve', %s, %s, %s, %s, false) ON CONFLICT DO NOTHING"""

_UPSERT_ASSIGNMENT = """
    INSERT INTO assignments (det_id, gid, score, decision_type, seq)
    VALUES (%s,%s,%s,%s,%s) ON CONFLICT (det_id) DO UPDATE SET gid=EXCLUDED.gid"""

_INSERT_DECISION_LOG = """
    INSERT INTO decision_log (seq, det_id, chosen_gid, decision_type, admitted,
        candidate_gids, scores, cannot_link_pruned, veto_reasons, threshold,
        admit_reason, admit_sim, admit_min, admit_tau, coherence_floor, max_reps)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (seq) DO NOTHING"""

# Recompute one identity's span/cameras/member-count from its committed assignments (see _refresh_identity_spans).
_REFRESH_IDENTITY_SPANS = """
    UPDATE identities i SET n_members=COALESCE(s.n,0), first_wall_ms=s.lo,
        last_wall_ms=s.hi, cameras=COALESCE(s.cams,'{}')
    FROM (SELECT count(*) n, min(d.abs_ms) lo, max(d.abs_ms) hi, array_agg(DISTINCT d.camera) cams
          FROM assignments a JOIN detections d ON d.det_id=a.det_id WHERE a.gid=%s) s
    WHERE i.gid=%s"""

# Incremental O(1) span extend used on the per-tracklet streaming add (the full recompute above re-JOINs the
# gid's whole — growing — assignment set every tracklet). Adding a tracklet only extends its gid's span:
# count += n_dets, first/last_wall = min/max with the tracklet's range, cameras |= {camera}. Yields the same
# result as the full recompute for the streaming add path (unique det_ids, one assignment each); merges still
# go through the full recompute at resolve() (survivor = union, absorbed -> 0).
_EXTEND_IDENTITY_SPAN = """
    UPDATE identities SET
        n_members     = n_members + %s,
        first_wall_ms = LEAST(COALESCE(first_wall_ms, %s), %s),
        last_wall_ms  = GREATEST(COALESCE(last_wall_ms, %s), %s),
        cameras       = ARRAY(SELECT DISTINCT unnest(cameras || %s::text[]) ORDER BY 1)
    WHERE gid=%s"""

_SELECT_ASSIGNMENTS = """
    SELECT d.video, d.camera, d.frame_idx, a.gid, d.x1, d.y1, d.x2, d.y2
    FROM detections d JOIN assignments a ON a.det_id=d.det_id"""

_INSERT_MERGE = "INSERT INTO merges (at_seq, old_gid, new_gid, score) VALUES (%s, %s, %s, %s)"


class OnlineGallery:
    """Streaming, DB-backed online identity gallery service. One per dataset.

    This is the kit's user-facing service that wraps the matcher (`vlincs_gallery.gallery.IdentityGallery`,
    stored as `self.m`): it adds pgvector DB persistence, camera geo + absolute-clock from the dataset's
    shipped extrinsics, and the streaming API. The match/expand/consolidate logic itself lives in the
    matcher, not here."""

    def __init__(self, dataset: str, *, fps: float = 30.0, truncate: bool = True,
                 tau: float = 0.60, merge_tau: float = 0.35, coherence_floor: float = 0.4,
                 match_mode: str = "centroid",  # whole-bank-mean match space
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
            cur.execute("TRUNCATE detections, identities, assignments, decision_log, "
                        "merges, embeddings, models, cameras RESTART IDENTITY CASCADE;")
            # Ephemeral fast-ingest (the demo / any truncate-before-ingest run): this DB is rebuilt from
            # scratch every run, so durability buys nothing. Turn off WAL — UNLOGGED tables skip WAL entirely
            # (the tables are empty right after TRUNCATE, so SET UNLOGGED is instant) and synchronous_commit=off
            # drops the per-commit fsync. On an unclean crash these tables truncate to empty — fine, the next
            # run rebuilds them.
            # Order matters: a referenced table may go UNLOGGED only after its referencers (a permanent table
            # cannot reference an unlogged one; unlogged->logged is allowed). FKs: assignments->{detections,
            # identities}; {detections,embeddings}->models. So referencers first, models last.
            cur.execute("SET synchronous_commit = off")
            for _t in ("assignments", "embeddings", "detections", "identities",
                       "decision_log", "merges", "cameras", "models"):
                cur.execute(f"ALTER TABLE {_t} SET UNLOGGED")
            self.con.commit()
        # camera geo + per-video clock from the dataset's shipped extrinsics. DS1/DS2 use the NEW per-frame
        # v2.0.x extrinsics (frame_abs = exact per-frame absolute ms, since the frame intervals are
        # non-uniform); MS02 (no extrinsics parquet) gets epoch+frame/fps via the load_clock fallback.
        self.epochs, geo, self.frame_abs = load_clock(*EXTRINSICS_DIRS.get(self.dataset, []))
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
            # ablation (default is on): drop all vetoes — same_frame (two distinct boxes in one frame
            # can't share an id), simultaneity + travel (cross-camera). Off is appearance-only; on is the
            # physically-correct default.
            self.m._cannot_link = lambda *a, **k: None
            self.m.merge_free_xcam = True
        # Persist the decision config (the knobs this run was built with) as a models row so the viz can
        # show what the current DB state reflects. role='gallery' fits the models table's "settings
        # provenance" purpose; /meta returns it.
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
        # Additive pgvector persistence (viz + DB-side ANN path only — the matcher's rep_mat is untouched).
        # The pushed match vector goes to the polymorphic `embeddings` table at whatever dim it is; the
        # embedder is registered once and its per-model partial HNSW index is created lazily (db.enable_ann)
        # so DB-side ANN works alongside the in-process index.
        self._emb_model_id = None   # the registered embedder model_id (set on the first push)
        self._resolve_model_id = None   # the role='resolve' embedder model_id (set on the first resolve_emb push)
        self._emb_type = None       # 'vector' (<=2000) | 'halfvec' (<=4000) | None (>4000: storage+FAISS only)

    def _register_embedder(self, cur, name=None):
        """On the first push, register the embedder (name + the now-known dim) and lazily create its
        per-model DB-ANN index. The matcher (FAISS/rep_mat) is unaffected; this only enables the pgvector
        ANN path + records provenance. Multi-model callers pass a distinct `name` per embedder."""
        self._emb_model_id, self._emb_type = gdb.register_embedder(
            cur, name or f"embedding-{self._emb_dim}d", self._emb_dim)
        self.con.commit()           # commit the model row so the embeddings FK + the ANN index can see it
        try:
            gdb.enable_ann(self.dataset, self._emb_model_id, self._emb_dim, self._emb_type)
        except Exception as e:       # DB-ANN is opt-in/best-effort; never block ingest on it
            print(f"[online] DB-ANN index not created ({e}); FAISS path + storage unaffected.")

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
    def _abs_ms(self, fa, epoch_ms: int, fr: int) -> int:
        """Absolute ms for one frame: the EXACT per-frame time (DS1/DS2 v2.0.x extrinsics, `fa`) when
        available, else the old uniform-fps clock (epoch + frame/fps) — for MS02, single-row extrinsics,
        or frames a tracker reports beyond the extrinsics table."""
        if fa is not None and 0 <= fr < len(fa) and not np.isnan(fa[fr]):
            return int(round(float(fa[fr])))
        return epoch_ms + int(round(fr / self.fps * 1000.0))

    def add_detection(self, video: str, camera: str, frame: int, box, embedding, conf: float = 1.0,
                      object_type: int = 0, det_id: str | None = None) -> int:
        """Match-or-expand a single detection. Returns its global id."""
        return self.add_tracklet(video, camera, [frame], [box], embedding, confs=[conf],
                                 object_type=object_type, det_ids=[det_id] if det_id else None)

    def add_tracklet(self, video: str, camera: str, frames, boxes, embedding, confs=None,
                     object_type: int = 0, det_ids=None, model: str | None = None, resolve_emb=None) -> int:
        """Match-or-expand a tracklet (their tracker's output: same-camera dets + one pooled embedding).
        `embedding` may be (D,) pooled or (n,D) per-detection (mean-pooled here). `model` names the embedder
        (default 'embedding-<D>d'); pass a distinct name to store multiple models' vectors. Returns the gid.

        `resolve_emb` (optional): a SECOND per-tracklet vector (e.g. osnet-xcam) stored in the polymorphic
        embeddings table under role='resolve' — the gallery is the system of record for BOTH the match space
        and the resolve space; `resolve_global()` re-clusters on it (read back from the DB). Not used to match."""
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
        fa = self.frame_abs.get(video)                     # per-frame absolute ms (v2.0.x extrinsics), else None
        det_abs_ms = [self._abs_ms(fa, epoch_ms, int(frames[i])) for i in range(n_dets)]
        boxes = [np.asarray(b, float) for b in boxes]
        # Representative det for display (the decision row, the exemplar crop, the embeddings key) — not
        # det_ids[0]. The first detection is the tracker's acquisition moment (person entering: edge of frame,
        # occluded, small, blurred), reliably the worst crop in the tracklet. Prefer the highest-confidence
        # detection when real confs vary (DS1); else the median-frame det (temporally central -> usually fully
        # in view) for uniform-conf inputs. Display-only: the matcher still matches on `pooled` (the mean over
        # all dets), and the score reads per-det assignments — neither uses rep_did.
        _c = np.asarray(confs, float) if (confs is not None and len(confs) == n_dets) else None
        rep_idx = int(np.argmax(_c)) if (_c is not None and np.ptp(_c) > 0) else int(np.argsort(frames)[n_dets // 2])
        rep_did = det_ids[rep_idx]
        confs = list(confs) if confs is not None else [1.0] * n_dets
        self_coh = self_coherence(list(emb)) if per_det else 1.0
        gid, score, _dtype, _pruned, _admitted = self.m.match_or_expand(
            video, camera, [int(f) for f in frames], boxes, min(det_abs_ms), max(det_abs_ms),
            pooled.astype(np.float32), rep_did=rep_did, self_coh=self_coh)
        # persist live: detections + assignment + the decision (the viz reads this)
        cur = self.con.cursor()
        decision = self.m.decisions[-1]
        if self._emb_model_id is None:                        # first push: register the embedder + its DB-ANN index
            self._register_embedder(cur, model)
        # The matcher's appearance state lives only in the in-process index (rep_mat). We additively persist
        # the pooled match-space vector (the gallery scored it) into the polymorphic `embeddings` table at
        # whatever dim — the viz + the DB-side ANN read it from there.
        vec = np.ascontiguousarray(pooled, dtype=np.float32)  # unconstrained `vector` column: any dim
        decision_type = decision["decision_type"] if decision["decision_type"] in ("match", "expand") else "revised"
        # Batched per-detection writes: build the tracklet's det + assignment rows and executemany them
        # (one pipelined round-trip each) instead of ~n_dets single-row executes. DS1 tracklets are dense
        # (~hundreds of dets each). The identity row is the same gid for every det, so it's inserted once
        # (not per det).
        det_rows = [(det_ids[i], video, camera, int(frames[i]), det_abs_ms[i] - epoch_ms, det_abs_ms[i],
                     float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3]),
                     float(confs[i]), int(object_type)) for i in range(n_dets)]
        asn_rows = [(det_ids[i], int(gid), float(score), decision_type, int(decision["seq"]))
                    for i in range(n_dets)]
        cur.execute("INSERT INTO identities (gid, n_members) VALUES (%s,0) ON CONFLICT (gid) DO NOTHING",
                    (int(gid),))
        cur.executemany(_INSERT_DETECTION, det_rows)
        # Re-push guard: _INSERT_DETECTION is ON CONFLICT DO NOTHING, so rowcount = dets actually inserted.
        # If < n_dets, some det_ids already existed (a re-push / shared det_id across tracklets). The O(1) span
        # extend below only adds, so it would over-count n_members; capture the gids those dets are leaving so
        # we can fall back to the exact full recompute for them + this gid. Normal single-push ingest never
        # trips this (all dets new -> repush=False -> fast path).
        new_dets = cur.rowcount
        repush = isinstance(new_dets, int) and 0 <= new_dets < n_dets
        old_gids: set[int] = set()
        if repush:
            cur.execute("SELECT DISTINCT gid FROM assignments WHERE det_id = ANY(%s)",
                        ([r[0] for r in asn_rows],))      # the dets' CURRENT gids, BEFORE the upsert moves them
            old_gids = {int(r[0]) for r in cur.fetchall()}
        cur.executemany(_UPSERT_ASSIGNMENT, asn_rows)
        cur.execute(_INSERT_DECISION_LOG,
                    (decision["seq"], rep_did, int(gid), decision["decision_type"], bool(decision["admitted"]),
                     decision["candidate_gids"], decision["scores"], decision["cannot_link_pruned"],
                     decision["veto_reasons"], decision["threshold"],
                     decision.get("admit_reason"), decision.get("admit_sim"), decision.get("admit_min"),
                     decision.get("admit_tau"), decision.get("coherence_floor"), decision.get("max_reps")))
        # One embeddings row per tracklet (the unit of work): the pooled match vector, keyed by the
        # representative det, carrying the birth gid + decision seq + is_rep (whether the matcher admitted it
        # to the exemplar bank — the viz "bank" subset). The matcher's live bank is rep_mat in memory; this is
        # the additive copy the viz + DB-side ANN read. Any dim.
        cur.execute(_INSERT_EMBEDDING, (rep_did, self._emb_model_id, self._emb_dim, vec,
                                        int(gid), int(decision["seq"]), bool(decision["admitted"])))
        if resolve_emb is not None:                          # the SECOND (resolve-space) vector for this tracklet
            rv = np.ascontiguousarray(resolve_emb, dtype=np.float32)
            if self._resolve_model_id is None:               # register the resolve embedder once (no ANN index needed)
                self._resolve_model_id, _ = gdb.register_embedder(cur, "resolve-embedder", int(rv.shape[0]))
                self.con.commit()
            cur.execute(_INSERT_RESOLVE_EMB, (rep_did, self._resolve_model_id, int(rv.shape[0]), rv,
                                              int(gid), int(decision["seq"])))
        # keep this identity's span/cameras/member-count current so the viz /state (and its KPIs) reflect
        # the gallery as of t. Incremental extend (O(1)) — not the full re-JOIN recompute, which scales with
        # the matched gid's whole assignment set. (resolve() still does the full recompute for merge-touched
        # gids.)
        if repush:                          # re-push / re-assignment: extend would mis-count -> exact recompute
            self._refresh_identity_spans(cur, {int(gid)} | old_gids)
        else:
            cur.execute(_EXTEND_IDENTITY_SPAN,
                        (n_dets, min(det_abs_ms), min(det_abs_ms), max(det_abs_ms), max(det_abs_ms), [camera], int(gid)))
        self._last_seq = int(decision["seq"])     # the step a subsequent resolve()'s merges take effect at
        self._ndet += n_dets
        self._pending += 1
        if self._pending >= self._batch:
            self.con.commit()
            self._pending = 0
        return int(gid)

    def resolve(self) -> dict:
        """Periodic resolve (consolidation): merge over-split identities; update the DB and the live
        in-process index so subsequent online matches use the merged ids. Call on any cadence.

        The DB bookkeeping after the merges is done in bulk: one batched assignment relabel, a batched
        merge-trail insert, and an incremental span union (survivor span = union of its absorbed gids' spans;
        absorbed -> 0) instead of a per-gid full re-JOIN recompute."""
        remap, events = self.m.consolidate(self.merge_tau)
        self.m.apply_remap(remap)                          # unify live gallery state for future adds
        real = [(int(old), int(new)) for old, new in remap.items() if old != new]
        if real:
            cur = self.con.cursor()
            olds = [o for o, _ in real]; news = [n for _, n in real]
            # 1) One batched relabel: move every absorbed gid's committed assignments to its final survivor.
            # remap is direct/unchained, so a single parallel-unnest join-update == the per-merge UPDATE loop.
            # embeddings.gid (rep rows) keeps the immutable birth gid; the viz derives the post-merge gid from
            # assignments (wall) or the merges replay (decision order).
            cur.execute("UPDATE assignments a SET gid = r.new "
                        "FROM unnest(%s::bigint[], %s::bigint[]) AS r(old, new) WHERE a.gid = r.old",
                        (olds, news))
            # 2) batched merge trail (survivor, absorbed, centroid_cosine) at the current ingest step.
            if events:
                cur.executemany(_INSERT_MERGE, [(self._last_seq, int(absorbed), int(survivor), float(cosine))
                                                for survivor, absorbed, cosine in events])
            # 3) Incremental span union (no re-JOIN scan): read each absorbed gid's current span, fold it into
            # its survivor (n += , LEAST/GREATEST span, camera-union), then zero the absorbed (now empty ->
            # drops from /state). Equals a full recompute because the pre-merge spans are already exact.
            cur.execute("SELECT gid, n_members, first_wall_ms, last_wall_ms, cameras FROM identities "
                        "WHERE gid = ANY(%s)", (olds,))
            agg: dict[int, list] = {}
            for gid, n, lo, hi, cams in cur.fetchall():
                a = agg.setdefault(int(remap[gid]), [0, None, None, set()])
                a[0] += int(n or 0)
                if lo is not None: a[1] = lo if a[1] is None else min(a[1], lo)
                if hi is not None: a[2] = hi if a[2] is None else max(a[2], hi)
                a[3].update(cams or [])
            if agg:                                        # LEAST/GREATEST ignore NULL params -> safe when absorbed span empty
                cur.executemany(
                    "UPDATE identities SET n_members = n_members + %s, first_wall_ms = LEAST(first_wall_ms, %s), "
                    "last_wall_ms = GREATEST(last_wall_ms, %s), "
                    "cameras = ARRAY(SELECT DISTINCT unnest(cameras || %s::text[]) ORDER BY 1) WHERE gid = %s",
                    [(add_n, lo, hi, sorted(cams), surv) for surv, (add_n, lo, hi, cams) in agg.items()])
            cur.execute("UPDATE identities SET n_members=0, first_wall_ms=NULL, last_wall_ms=NULL, cameras='{}' "
                        "WHERE gid = ANY(%s)", (olds,))
            self.con.commit()
        return {"merges": len(real), "events": len(events) if events else 0}

    def resolve_global(self, theta: float, *, top_k: int = 15, min_dets: int = 20) -> dict:
        """GLOBAL re-partition from the gallery's OWN stored resolve embeddings (role='resolve', pushed per
        tracklet during the stream — the gallery is the system of record). Reads them back seq-ordered + each
        tracklet's camera + detection count, runs vlincs_gallery.resolve.global_agglom_resolve (kNN-sparse
        cross-camera cosine + average-linkage at distance_threshold = 1 - theta), then OVERWRITES every
        assignment's gid with its tracklet's new cluster — recovers greedy over-split AND over-merge (unlike
        resolve()'s gid-merge). Per-tracklet, top_k=15, theta=0.02, and only tracklets with >= `min_dets`
        detections are re-partitioned (shorter, low-evidence tracklets keep their own singleton id). Rebuilds
        the identities table; the greedy decision_log/merges feed is left intact (the online stream that
        preceded the re-partition). New gids are OFFSET clear of the greedy ones."""
        from vlincs_gallery.resolve import global_agglom_resolve
        cur = self.con.cursor()
        cur.execute("SELECT e.seq, d.camera, e.vec, n.cnt FROM embeddings e "
                    "JOIN detections d ON e.entity_id = d.det_id "
                    "JOIN (SELECT seq, COUNT(*) cnt FROM assignments GROUP BY seq) n ON n.seq = e.seq "
                    "WHERE e.role = 'resolve' ORDER BY e.seq")
        rows = cur.fetchall()
        if not rows:
            raise RuntimeError("resolve_global: no role='resolve' embeddings — pass resolve_emb to add_tracklet first")
        seqs = [int(r[0]) for r in rows]
        keep = [i for i in range(len(rows)) if int(rows[i][3]) >= min_dets]   # only re-partition >= min_dets-det
        cams = sorted({rows[i][1] for i in keep}); ccode = {c: j for j, c in enumerate(cams)}
        cam_codes = np.array([ccode[rows[i][1]] for i in keep], np.int64)
        emb = np.stack([np.asarray(rows[i][2], np.float32) for i in keep])
        res = global_agglom_resolve(emb, cam_codes, theta=theta, top_k=top_k, exclude_same_cam=True)
        OFFSET = 10_000_000
        gid_of_seq = {seqs[i]: int(res.labels[k]) + OFFSET for k, i in enumerate(keep)}   # clustered -> cluster id
        sing = OFFSET + int(res.n_clusters) + 1
        for s in seqs:                                                         # short tracklets -> own singleton id
            if s not in gid_of_seq:
                gid_of_seq[s] = sing; sing += 1
        new_gids = [gid_of_seq[s] for s in seqs]
        cur.executemany("INSERT INTO identities (gid, n_members) VALUES (%s, 0) ON CONFLICT (gid) DO NOTHING",
                        [(g,) for g in sorted(set(new_gids))])
        cur.execute("UPDATE assignments a SET gid = r.g "                       # relabel every det by its seq
                    "FROM unnest(%s::bigint[], %s::bigint[]) AS r(s, g) WHERE a.seq = r.s", (seqs, new_gids))
        cur.execute("UPDATE identities i SET n_members=s.n, first_wall_ms=s.lo, last_wall_ms=s.hi, cameras=s.cams "
                    "FROM (SELECT a.gid, COUNT(DISTINCT a.seq) n, MIN(d.wall_clock_ms) lo, MAX(d.wall_clock_ms) hi, "
                    "             ARRAY_AGG(DISTINCT d.camera ORDER BY d.camera) cams "
                    "      FROM assignments a JOIN detections d ON a.det_id = d.det_id GROUP BY a.gid) s "
                    "WHERE i.gid = s.gid")
        cur.execute("UPDATE identities SET n_members=0, first_wall_ms=NULL, last_wall_ms=NULL, cameras='{}' "
                    "WHERE gid < %s", (OFFSET,))            # drop the now-empty greedy identities from /state
        self.con.commit()
        return {"clusters": int(res.n_clusters), "theta": float(theta), "min_dets": min_dets,
                "n_tracklets": len(seqs), "n_clustered": len(keep), "n_singleton": len(seqs) - len(keep)}

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
        # dense=False is the canonical / leaderboard scorer setting for sparse GT (don't punish the detector
        # as a false positive for finding a real person the GT didn't annotate). We do not additionally
        # `restrict_to_gt_matched` — dropping your own predictions to delete the IDFPs they incur is optimistic
        # and not submission-honest (you can't drop detections at submission time). idf1 here is the honest number.
        metrics = evaluate(gt, {key: pred_by_key[key] for key in gt}, dense=False, n_workers=1)
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
