"""FastAPI backend over the per-dataset gallery DB - see how/when/why the gallery is what it is at
time t, with detection crops served on demand.

Run:  GALLERY_DATASET=ds1 uvicorn vlincs_gallery.viz.app:app --port 8077 --reload
(from the repo root with PYTHONPATH=. , using the gallery venv).

Endpoints (all read live from gallery_<dataset>):
  /meta                          dataset, cameras, time span, counts
  /state?t=<abs_ms>              identities that EXIST as of t + per-t member counts + decision tallies
  /detections?t=&window=        boxes active near t (playback), with gid + admitted
  /decisions?from=&to=&limit=    decision trail in a time window: candidates, scores, veto reasons, match/expand, admitted (the WHY/WHEN)
  /tracklet/{seq}                one tracklet's dets (the within-camera unit), its decision, admitted, evenly-subsampled det_ids for display
  /identity/{gid}                an identity's tracklets (added vs rejected), exemplars, cameras, span
  /crop/{det_id}                 JPEG crop of that detection from the source video frame (ON DEMAND)
  /embedding_projection?mode=    the match space in 2D: bank exemplars (bank) or per-det cloud (det), by gid
  /decision_geometry/{det_id}    one decision's geometry: query vector + candidate exemplars + tau
"""
from __future__ import annotations
import glob
import math
import os, threading
from collections import OrderedDict
import cv2
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from fastapi import FastAPI, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from vlincs_gallery import db as gdb
from vlincs_gallery import paths

DATASET = os.environ.get("GALLERY_DATASET", "ds1")     # the startup DEFAULT dataset (used when no ?dataset=…)
DSN = gdb.dsn(DATASET)

# Per-request dataset switch: the UI sends ?dataset=<key> on EVERY call (incl. <img src> crops), so one viz
# process serves every gallery_<key> DB without a restart. The validated key is stashed in a contextvar set by
# a pure-ASGI middleware (pure ASGI sets it in the request's context, which anyio copies into the
# sync-endpoint threadpool).
import contextvars
import re as _re
from urllib.parse import parse_qs as _parse_qs

_REQ_DATASET = contextvars.ContextVar("gallery_dataset", default=DATASET)
_DS_RE = _re.compile(r"^[a-z0-9_-]{1,32}$")             # sanitize: the key lands in a dbname (gallery_<key>)


def _active_dataset() -> str:
    return _REQ_DATASET.get()


def _dsn() -> str:                                     # the DSN for THIS request's dataset
    return gdb.dsn(_active_dataset())


def _db_to_key(db: str) -> str:                        # gallery_ms02 -> ms02 (inverse of gdb.dataset_db)
    return db[len("gallery_"):] if db.startswith("gallery_") else db


class _DatasetMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        ds = _parse_qs(scope.get("query_string", b"").decode()).get("dataset", [None])[0]
        token = _REQ_DATASET.set(ds if (ds and _DS_RE.match(ds)) else DATASET)
        try:
            await self.app(scope, receive, send)
        finally:
            _REQ_DATASET.reset(token)


app = FastAPI(title="VLINCS Gallery viz")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(_DatasetMiddleware)

# Crop serving: cv2.VideoCapture is NOT thread-safe, so hold a PER-VIDEO lock (so different videos -
# a bank's exemplars usually span several cameras - decode in parallel) and cache encoded JPEGs (crops
# are immutable, so re-opening an identity is instant).
_caps: dict[str, cv2.VideoCapture] = {}
_caps_lock = threading.Lock()
_vlocks: dict[str, threading.Lock] = {}
_cache_lock = threading.Lock()
_crop_cache: "OrderedDict[str, bytes]" = OrderedDict()    # "det_id:max_w" -> jpeg
_frame_cache: "OrderedDict[str, bytes]" = OrderedDict()    # "camera:frame:w" -> jpeg
_CROP_MAX, _FRAME_MAX = 8000, 1200


def _vlock(video: str) -> threading.Lock:
    with _caps_lock:
        lk = _vlocks.get(video)
        if lk is None:
            lk = _vlocks[video] = threading.Lock()
        return lk


def _read_frame(video: str, frame_idx: int):
    """Decode one BGR frame from video at frame_idx (per-video locked - cap is not thread-safe).
    frame_idx is CLAMPED to [0, frame_count-1]: an out-of-range seek otherwise returns None -> 404 and the
    canvas background never paints. Clamping makes it degrade to the last real frame instead."""
    with _vlock(video):
        cp = _caps.get(video)
        if cp is None:
            cp = _caps[video] = cv2.VideoCapture(_video_path(video))
        fi = max(0, int(frame_idx))
        n = int(cp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 0 and fi >= n:
            fi = n - 1
        cp.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cp.read()
    return frame if ok else None


def _video_dims(video: str):
    """Real (width, height) of the source video, for correct canvas aspect + box scaling. Reuses the
    per-video cap (cached). Returns None if the video can't be opened."""
    try:
        with _vlock(video):
            cp = _caps.get(video)
            if cp is None:
                cp = _caps[video] = cv2.VideoCapture(_video_path(video))
            w = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (w, h) if w > 0 and h > 0 else None
    except Exception:
        return None


def _cache_get(cache, key):
    with _cache_lock:
        b = cache.get(key)
        if b is not None:
            cache.move_to_end(key)
        return b


def _cache_put(cache, key, b, cap):
    with _cache_lock:
        cache[key] = b
        cache.move_to_end(key)
        while len(cache) > cap:
            cache.popitem(last=False)


def _encode(img, max_w, q):
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else None


def _q(sql, args=(), one=False):
    with psycopg.connect(_dsn()) as c, c.cursor() as cur:
        cur.execute(sql, args)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    out = [dict(zip(cols, r)) for r in rows]
    return (out[0] if out else None) if one else out


def _vecq(sql, args=()):
    """Like _q but with pgvector registered, so vector() columns return numpy arrays (not strings)."""
    with psycopg.connect(_dsn()) as c:
        register_vector(c)
        with c.cursor() as cur:
            cur.execute(sql, args)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]


def _project_2d(mat):
    """Deterministic 2D projection of an (n, d) float matrix. PCA (sklearn, already a dep) - seeded,
    no heavy extra dep. UMAP only if it happens to be installed (better cluster separation when present).
    Returns (n, 2) float array; for n<3 falls back to the raw first two dims (PCA needs >=2 samples)."""
    mat = np.asarray(mat, dtype=np.float64)
    n, d = mat.shape
    if n == 0:
        return np.zeros((0, 2), np.float64)
    if n < 3 or d < 2:
        out = np.zeros((n, 2), np.float64)
        out[:, :min(2, d)] = mat[:, :min(2, d)]
        return out
    try:                                   # UMAP gives nicer separation but is optional
        import umap
        return umap.UMAP(n_components=2, random_state=42).fit_transform(mat)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(mat)


def _video_path(stem: str) -> str:
    # The card SUBDIR isn't reliably encoded in the stem: DS1 stems end in the card (..._2024-03-Tc6),
    # but MS02 stems end in a timestamp (..._2018-03-15_15-00-06) while the real dir is 2018-03-Tc85.
    # So glob for the (unique) stem under any card dir of this site/cluster; fall back to the
    # card-from-stem guess only if the glob misses (e.g. an unusual mount layout).
    p = stem.split("_")            # vlincs_<SITE>_<CLUSTER>_<MCAMxx>_<...>
    site, cluster = p[1], p[2]
    root = paths.root_for_site(site)   # MS02 lives in the -selected tree; ds1/ds2 in the Box export
    hits = glob.glob(f"{root}/{site}/{cluster}/*/{stem}.mp4")
    if hits:
        return hits[0]
    return f"{root}/{site}/{cluster}/{'_'.join(p[4:])}/{stem}.mp4"


# Recursive CTE: for each birth gid (decision_log.chosen_gid) compute its CANONICAL gid after applying every
# merge in effect by a given step (merges.at_seq <= %s). This lets decision-order queries reconstruct the
# identity a detection had AS OF that step - not its final post-merge gid - so stepping replays the gallery's
# true state (no "future" merges leaking in). Splice _CANON_CTE before the SELECT; its single %s is the step,
# and join `_canon ON _canon.orig = <birth gid>` (decision_log.chosen_gid of the placing decision).
_CANON_CTE = """
WITH RECURSIVE _mrg AS (
    SELECT old_gid, new_gid FROM merges WHERE at_seq <= %s
),
_walk AS (
    SELECT g AS orig, g AS cur FROM (SELECT DISTINCT chosen_gid AS g FROM decision_log WHERE chosen_gid IS NOT NULL) s0
    UNION ALL
    SELECT w.orig, m.new_gid FROM _walk w JOIN _mrg m ON m.old_gid = w.cur
),
_canon AS (
    SELECT orig, cur FROM _walk w WHERE NOT EXISTS (SELECT 1 FROM _mrg m WHERE m.old_gid = w.cur)
)"""


def _card_filter(card: str, col: str = "d.video"):
    """Return (card_clause, card_params): a SQL `AND <col> LIKE %s` fragment + its params tuple restricting a
    query to one test card (video suffix, e.g. 2024-03-Tc6), or ("", ()) for all cards. Splice card_clause
    into the WHERE and unpack card_params into the query args. Tc6/Tc8 reuse camera names and are ~2.5h apart;
    the per-card toggle un-conflates the two sessions."""
    return (f" AND {col} LIKE %s", (f"%{card}",)) if card else ("", ())


@app.get("/datasets")
def datasets():
    """List the gallery_<key> databases that exist (with identity counts) so the UI can offer a switch.
    `current` is the dataset THIS request resolved to (from ?dataset=…, else the startup default)."""
    try:
        with psycopg.connect(gdb.ADMIN_DSN) as c, c.cursor() as cur:
            cur.execute("SELECT datname FROM pg_database WHERE datname LIKE %s AND datname <> 'gallery' ORDER BY datname",
                        ("gallery\\_%",))
            dbs = [r[0] for r in cur.fetchall()]
    except Exception:
        dbs = [gdb.dataset_db(DATASET)]                  # fall back to at least the default DB
    out = []
    for db in dbs:
        key = _db_to_key(db)
        n = -1
        try:
            with psycopg.connect(gdb.dsn(key)) as c2, c2.cursor() as cur2:
                cur2.execute("SELECT count(*) FROM identities WHERE n_members > 0")
                n = int(cur2.fetchone()[0])
        except Exception:
            n = -1                                       # DB present but unreadable / no schema yet
        out.append({"key": key, "db": db, "n_identities": n})
    return {"datasets": out, "current": _active_dataset(), "default": DATASET}


@app.get("/meta")
def meta(card: str = ""):
    card_clause, card_params = _card_filter(card, "video")   # restrict everything to one card when the toggle is set
    card_clause_d, _ = _card_filter(card, "d.video")         # same filter, for the detections-aliased canvas query
    # One canvas per VIDEO (DS1 reuses camera names across Tc6/Tc8). Derive the canvas list from the
    # DETECTIONS (the videos that actually have data), LEFT JOIN the cameras table for geo - so canvases
    # render even when no camera-geo metadata was loaded (e.g. MS02 ships camera_params.json, not the
    # extrinsics parquet camera_geo() reads, so its cameras table is empty).
    cams = _q(f"""SELECT DISTINCT d.camera, d.video, c.lat, c.lon, c.start_ms
                  FROM detections d LEFT JOIN cameras c ON c.video = d.video
                  WHERE TRUE{card_clause_d} ORDER BY d.camera, d.video""", card_params)
    wh = {r["video"]: r for r in _q(f"SELECT video, max(x2) AS w, max(y2) AS h FROM detections WHERE TRUE{card_clause} GROUP BY video", card_params)}
    for c in cams:                                   # canvas dims: REAL video resolution (correct aspect +
        dims = _video_dims(c["video"])               # box scaling), falling back to detection-extent only
        if dims:                                     # if the video can't be opened.
            c["w"], c["h"] = dims
        else:
            c["w"] = (wh.get(c["video"]) or {}).get("w")
            c["h"] = (wh.get(c["video"]) or {}).get("h")
    span = _q(f"SELECT min(abs_ms) t0, max(abs_ms) t1 FROM detections WHERE TRUE{card_clause}", card_params, one=True) or {}
    # time quantiles so the scrubber maps over ACTUAL detection time. With a card selected this covers only
    # that session (full resolution, no gap); with 'all' it skips the ~2.5h Tc6/Tc8 dead gap.
    qr = _q(f"SELECT percentile_cont(%s) WITHIN GROUP (ORDER BY abs_ms) AS q FROM detections WHERE TRUE{card_clause}",
            ([i / 200 for i in range(201)], *card_params), one=True)
    t_quantiles = [int(x) for x in (qr.get("q") or [])] if qr else []
    # seq quantiles + range so the scrubber can also map over INGEST/DECISION order (one step per tracklet
    # decision), not just wall-clock - the timeline the online gallery was actually built on.
    sq = _q(f"""SELECT percentile_cont(%s) WITHIN GROUP (ORDER BY dl.seq) AS q
                FROM decision_log dl JOIN detections d ON d.det_id=dl.det_id WHERE TRUE{card_clause_d}""",
            ([i / 200 for i in range(201)], *card_params), one=True)
    seq_quantiles = [int(x) for x in (sq.get("q") or [])] if sq and sq.get("q") else []
    seqspan = _q(f"""SELECT min(dl.seq) s0, max(dl.seq) s1, count(*) n
                     FROM decision_log dl JOIN detections d ON d.det_id=dl.det_id WHERE TRUE{card_clause_d}""",
                 card_params, one=True) or {}
    # The resolve writes a terminal merge WAVE at at_seq = (global last decision seq)+1 (see
    # OnlineGallery._record_resolve_merges). Extend the decision scrubber by that one step so dragging to the
    # end applies the resolve merges and the per-video locals "pop" into the final resolved identities.
    rsq = _q("SELECT max(at_seq) r FROM merges WHERE at_seq > (SELECT COALESCE(max(seq),0) FROM decision_log)", one=True)
    resolve_seq = (rsq or {}).get("r")
    if resolve_seq is not None and seqspan.get("s1") is not None:
        seqspan["s1"] = max(int(seqspan["s1"]), int(resolve_seq))
    cards = [r["card"] for r in _q("SELECT DISTINCT split_part(video,'_',5) AS card FROM cameras ORDER BY 1")]
    tallies = _q("SELECT decision_type, admitted, count(*) n FROM decision_log GROUP BY 1,2")
    n = _q("""SELECT (SELECT count(*) FROM identities) gids,
                     (SELECT count(*) FROM detections) dets,
                     (SELECT count(*) FROM decision_log) tracklets,
                     (SELECT count(*) FROM identities WHERE array_length(cameras,1)>1) cross_cam""", one=True)
    models = _q("SELECT role, name, params FROM models ORDER BY model_id")
    return {"dataset": gdb.dataset_db(_active_dataset()), "cameras": cams, "t0": span.get("t0"), "t1": span.get("t1"),
            "t_quantiles": t_quantiles, "seq_quantiles": seq_quantiles,
            "seq0": seqspan.get("s0"), "seq1": seqspan.get("s1"), "n_decisions": seqspan.get("n"),
            "resolve_seq": resolve_seq,    # the terminal step where the resolve "pops" (NULL if not resolved)
            "cards": cards, "card": card,
            "counts": n, "decision_tallies": tallies, "models": models}


@app.get("/next")
def next_decision(t: int, dir: int = 1, card: str = ""):
    """The next (dir>=0) or previous (dir<0) decision relative to time t - for stepping decision-by-decision."""
    card_clause, card_params = _card_filter(card, "d.video")
    op, order = (">", "ORDER BY d.abs_ms") if dir >= 0 else ("<", "ORDER BY d.abs_ms DESC")
    r = _q(f"""SELECT dl.seq, d.abs_ms, dl.chosen_gid, dl.decision_type FROM decision_log dl
               JOIN detections d ON d.det_id=dl.det_id WHERE d.abs_ms {op} %s{card_clause} {order} LIMIT 1""",
           (t, *card_params), one=True)
    return r or {}


@app.get("/state")
def state(t: int = -1, card: str = "", by: str = "wall", step: int = -1):
    """The gallery AS OF a point on the timeline, in one of two senses:
      by=wall (default): AS OF wall-clock time t - identities with a detection by t (a PLAYBACK view).
      by=decision: AS OF ingest/decision step - identities decided by step, "seen" = dets committed by step.
        This is the timeline the online gallery was actually built on (one step per tracklet decision).
    Returns identities + per-point "seen so far" + decision tallies so far."""
    card_clause, card_params = _card_filter(card, "d.video")
    if by == "decision":
        # TRUE replay: each committed det's identity = canonical(birth gid) after merges in effect by `step`.
        # "seen"/"total" are both the dets committed by step (everything shown is as-of-step; no future total),
        # cameras + span are as-of-step, identities ordered by birth (min seq). No final-state leakage.
        ids = _q(f"""{_CANON_CTE}
                     SELECT _canon.cur AS gid, min(d.abs_ms) AS first_wall_ms, max(d.abs_ms) AS last_wall_ms,
                            array_agg(DISTINCT d.camera) AS cameras, count(*) AS total, count(*) AS seen,
                            min(a.seq) AS born_seq
                     FROM assignments a
                     JOIN decision_log dl ON dl.seq = a.seq
                     JOIN detections d ON d.det_id = a.det_id
                     JOIN _canon ON _canon.orig = dl.chosen_gid
                     WHERE a.seq <= %s{card_clause}
                     GROUP BY _canon.cur ORDER BY born_seq""", (step, step, *card_params))
        tal = _q(f"""SELECT dl.decision_type, dl.admitted, count(*) n FROM decision_log dl
                     JOIN detections d ON d.det_id=dl.det_id WHERE dl.seq<=%s{card_clause} GROUP BY 1,2""", (step, *card_params))
        return {"step": step, "n_identities": len(ids), "identities": ids, "decision_tallies_so_far": tal}
    exists = (f" AND EXISTS (SELECT 1 FROM assignments a JOIN detections d ON d.det_id=a.det_id "
              f"WHERE a.gid=i.gid{card_clause})") if card else ""
    ids = _q(f"""SELECT i.gid, i.first_wall_ms, i.last_wall_ms, i.cameras, i.n_members AS total
                 FROM identities i WHERE i.first_wall_ms<=%s{exists} ORDER BY i.first_wall_ms""", (t, *card_params))
    # ONE grouped pass for "seen so far": a single indexed pass over dets<=t, instead of a per-identity
    # correlated subquery that full-scans assignments once per identity.
    seen = {r["gid"]: r["seen"] for r in _q(
        f"""SELECT a.gid, count(*) AS seen FROM assignments a JOIN detections d ON d.det_id=a.det_id
            WHERE d.abs_ms<=%s{card_clause} GROUP BY a.gid""", (t, *card_params))}
    for r in ids:
        r["seen"] = seen.get(r["gid"], 0)
    tal = _q(f"""SELECT dl.decision_type, dl.admitted, count(*) n FROM decision_log dl
                 JOIN detections d ON d.det_id=dl.det_id WHERE d.abs_ms<=%s{card_clause} GROUP BY 1,2""", (t, *card_params))
    return {"t": t, "n_identities": len(ids), "identities": ids, "decision_tallies_so_far": tal}


@app.get("/detections")
def detections(t: int = -1, window: int = 400, card: str = "", by: str = "wall", step: int = -1):
    """Boxes to draw on the camera canvases. by=wall: dets in the trail window before t (playback).
    by=decision: the boxes of the tracklet decided at `step` (the current decision) - its full path in its
    camera, so you see WHAT was just matched/expanded rather than a wall-clock frame."""
    card_clause, card_params = _card_filter(card, "d.video")
    if by == "decision":
        # the tracklet decided at `step`, coloured by the identity it had AS OF this step (canonical birth gid
        # after merges<=step) - NOT its final gid, so the box colour matches the decision being made.
        rows = _q(f"""{_CANON_CTE}
                      SELECT d.det_id, d.camera, d.video, d.frame_idx, d.abs_ms, d.x1,d.y1,d.x2,d.y2, _canon.cur AS gid
                      FROM detections d JOIN assignments a ON a.det_id=d.det_id
                      JOIN decision_log dl ON dl.seq = a.seq
                      JOIN _canon ON _canon.orig = dl.chosen_gid
                      WHERE a.seq=%s{card_clause} ORDER BY d.video, d.frame_idx""", (step, step, *card_params))
        return {"step": step, "n": len(rows), "dets": rows}
    rows = _q(f"""SELECT d.det_id, d.camera, d.video, d.frame_idx, d.abs_ms, d.x1,d.y1,d.x2,d.y2, a.gid
                  FROM detections d JOIN assignments a ON a.det_id=d.det_id
                  WHERE d.abs_ms BETWEEN %s AND %s{card_clause} ORDER BY d.video, d.frame_idx""", (t - window, t, *card_params))
    return {"t": t, "window": window, "n": len(rows), "dets": rows}


@app.get("/decisions")
def decisions(frm: int = Query(0, alias="from"), to: int = Query(2**62), limit: int = 200, card: str = ""):
    card_clause, card_params = _card_filter(card, "d.video")
    return _q(f"""SELECT dl.seq, dl.det_id, d.camera, d.abs_ms, dl.chosen_gid, dl.decision_type, dl.admitted,
                         dl.candidate_gids, dl.scores, dl.cannot_link_pruned, dl.veto_reasons, dl.threshold,
                         dl.admit_reason, dl.admit_sim, dl.admit_min, dl.admit_tau, dl.coherence_floor,
                         dl.max_reps
                  FROM decision_log dl JOIN detections d ON d.det_id=dl.det_id
                  WHERE d.abs_ms BETWEEN %s AND %s{card_clause} ORDER BY d.abs_ms LIMIT %s""", (frm, to, *card_params, limit))


@app.get("/merges")
def merges():
    """All consolidation merge events for the decision-order feed: at_seq (the ingest step the merge takes
    effect), old_gid merged into new_gid. Decision-order only - a merge has no detection/wall-clock time.
    Card-agnostic (gids are global); the frontend filters by at_seq<=step alongside the decisions."""
    return _q("SELECT merge_id, at_seq, old_gid, new_gid, score FROM merges ORDER BY merge_id")


# Frozen-policy thresholds the cannot-link vetoes use (gallery.IdentityGallery defaults - same_box_iou=0.35,
# max_speed=3.0). The viz is read-only; it does NOT recompute the veto, only the supporting numbers that
# explain one the matcher ALREADY made, so surfacing the deciding thresholds next to them is enough.
_SAME_BOX_IOU = 0.35
_MAX_SPEED_MS = 3.0


def _veto_explain(seq: int):
    """For a decision `seq`, compute ON-DEMAND the supporting numbers behind each VETOED candidate gid.

    Read-only: the matcher already recorded which candidates were vetoed (decision_log.veto_reasons,
    aligned with candidate_gids/scores); this reconstructs the evidence from the live tables so a vetoed
    line in the UI is self-explanatory instead of just "same_frame" / "travel:CAM". A candidate is VETOED
    when its veto_reason is non-empty and not 'below_tau' (below_tau is a threshold miss, not a veto).

      same_frame: n_shared (video,frame_idx) between THIS seq's tracklet and gid G's occupancy; the MEDIAN
        box-IoU over those shared frames (< same_box_iou => spatially distinct => two people); plus G's
        ATTRACTOR profile (distinct tracklet/seq count, frame_idx span, distinct camera count) so an
        over-merged "matches-everything" id is visible.
      travel: G's LAST detection on the candidate's OTHER camera(s) before this track, the haversine
        distance between the two cameras (cameras.lat/lon), the time gap, and the implied speed vs max_speed.
    """
    dec = _q("""SELECT seq, det_id, chosen_gid, decision_type, candidate_gids, scores, veto_reasons
                FROM decision_log WHERE seq=%s""", (seq,), one=True)
    if not dec:
        return []
    cand = list(dec.get("candidate_gids") or [])
    scores = list(dec.get("scores") or [])
    vetoes = list(dec.get("veto_reasons") or [])
    # this tracklet's own dets (the within-camera unit being decided): frame count + the camera(s) it spans
    track = _q("""SELECT d.video, d.camera, d.frame_idx, d.abs_ms, d.x1,d.y1,d.x2,d.y2
                  FROM assignments a JOIN detections d ON d.det_id=a.det_id WHERE a.seq=%s""", (seq,))
    n_track = len(track)
    track_cams = sorted({r["camera"] for r in track})
    out = []
    for i, g in enumerate(cand):
        reason = vetoes[i] if i < len(vetoes) else ""
        if not reason or reason == "below_tau":
            continue
        g = int(g)
        score = scores[i] if i < len(scores) else None
        kind = reason.split(":", 1)[0]
        item = {"gid": g, "veto_reason": reason, "kind": kind, "score": score,
                "n_track": n_track, "track_cameras": track_cams}
        if kind == "same_frame":
            # n_shared + median box-IoU over (video,frame_idx) this track shares with gid G's occupancy.
            # IoU computed in SQL: intersection / union over x1,y1,x2,y2; ::numeric so round() applies.
            # One box per (video,frame_idx) per side (DISTINCT ON) so a frame contributes exactly one IoU -
            # matching the matcher, whose occ[(video,frame)] holds a single box. Median over the shared frames.
            ov = _q("""
                WITH tk AS (
                  SELECT DISTINCT ON (d.video, d.frame_idx) d.video, d.frame_idx, d.x1,d.y1,d.x2,d.y2
                  FROM detections d JOIN assignments a ON a.det_id=d.det_id AND a.seq=%s
                  ORDER BY d.video, d.frame_idx, d.det_id
                ),
                gd AS (
                  SELECT DISTINCT ON (d.video, d.frame_idx) d.video, d.frame_idx, d.x1,d.y1,d.x2,d.y2
                  FROM detections d JOIN assignments a ON a.det_id=d.det_id AND a.gid=%s
                  ORDER BY d.video, d.frame_idx, d.det_id
                )
                SELECT count(*) AS n_shared, percentile_cont(0.5) WITHIN GROUP (ORDER BY iou) AS median_iou
                FROM (
                  SELECT (CASE WHEN inter <= 0 THEN 0.0
                               ELSE inter / NULLIF(ta + ga - inter, 0)::numeric END) AS iou
                  FROM (
                    SELECT GREATEST(0, LEAST(tk.x2,gd.x2)-GREATEST(tk.x1,gd.x1))
                         * GREATEST(0, LEAST(tk.y2,gd.y2)-GREATEST(tk.y1,gd.y1)) AS inter,
                           (tk.x2-tk.x1)*(tk.y2-tk.y1) AS ta, (gd.x2-gd.x1)*(gd.y2-gd.y1) AS ga
                    FROM tk JOIN gd ON gd.video=tk.video AND gd.frame_idx=tk.frame_idx
                  ) p
                ) q""", (seq, g), one=True) or {}
            # G's attractor profile: how many distinct tracklets, what frame span, how many cameras it owns
            prof = _q("""SELECT count(DISTINCT a.seq) AS n_seq, min(d.frame_idx) AS f0, max(d.frame_idx) AS f1,
                                count(DISTINCT d.camera) AS n_cam
                         FROM assignments a JOIN detections d ON d.det_id=a.det_id WHERE a.gid=%s""",
                      (g,), one=True) or {}
            med = ov.get("median_iou")
            item.update({"n_shared": int(ov.get("n_shared") or 0),
                         "median_box_iou": (round(float(med), 4) if med is not None else None),
                         "same_box_iou": _SAME_BOX_IOU,
                         "gid_n_tracklets": int(prof.get("n_seq") or 0),
                         "gid_frame_min": prof.get("f0"), "gid_frame_max": prof.get("f1"),
                         "gid_n_cameras": int(prof.get("n_cam") or 0)})
        elif kind == "travel":
            # the OTHER camera the veto named (travel:<CAM>); fall back to any non-track camera if unparsed
            other_cam = reason.split(":", 1)[1] if ":" in reason else None
            t0 = min((r["abs_ms"] for r in track if r["abs_ms"] is not None), default=None)
            # G's LAST detection on that other camera BEFORE this track started (the position it must travel from)
            last = _q("""SELECT d.camera, d.frame_idx, d.abs_ms
                         FROM assignments a JOIN detections d ON d.det_id=a.det_id
                         WHERE a.gid=%s AND d.camera=%s AND (%s IS NULL OR d.abs_ms <= %s)
                         ORDER BY d.abs_ms DESC LIMIT 1""", (g, other_cam, t0, t0), one=True)
            cam_geo = {r["camera"]: r for r in _q(
                "SELECT DISTINCT camera, lat, lon FROM cameras WHERE camera = ANY(%s)",
                ([other_cam] + track_cams,))} if other_cam else {}
            this_cam = track_cams[0] if track_cams else None
            dist = dt = speed = None
            a = cam_geo.get(other_cam) if other_cam else None
            b = cam_geo.get(this_cam) if this_cam else None
            if a and b and a.get("lat") is not None and b.get("lat") is not None:
                dist = _haversine_m((float(a["lat"]), float(a["lon"])),
                                    (float(b["lat"]), float(b["lon"])))
            if last and last.get("abs_ms") is not None and t0 is not None:
                dt = abs(t0 - last["abs_ms"]) / 1000.0
            if dist is not None and dt is not None and dt > 0:
                speed = dist / dt
            item.update({
                "from_camera": other_cam, "to_camera": this_cam,
                "from_frame": (last or {}).get("frame_idx"), "to_frame":
                    (min((r["frame_idx"] for r in track), default=None)),
                "dist_m": (round(dist, 2) if dist is not None else None),
                "dt_s": (round(dt, 2) if dt is not None else None),
                "speed_ms": (round(speed, 2) if speed is not None else None),
                "max_speed": _MAX_SPEED_MS})
        out.append(item)
    return out


def _haversine_m(a, b):
    """Great-circle metres between (lat, lon) a and b - same formula as gallery._haversine_m, duplicated
    here so the read-only viz has no import dependency on the matcher module."""
    R = 6371000.0
    la1, lo1 = math.radians(a[0]), math.radians(a[1])
    la2, lo2 = math.radians(b[0]), math.radians(b[1])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return R * 2 * math.asin(math.sqrt(h))


@app.get("/veto_explain/{seq}")
def veto_explain(seq: int):
    """Standalone access to the per-vetoed-candidate supporting numbers for a decision (also embedded on
    /tracklet/{seq} as `veto_explain`)."""
    return {"seq": seq, "veto_explain": _veto_explain(seq)}


@app.get("/tracklet/{seq}")
def tracklet(seq: int, n: int = 12):
    dec = _q("SELECT * FROM decision_log WHERE seq=%s", (seq,), one=True)
    dets = _q("""SELECT d.det_id, d.camera, d.frame_idx, d.abs_ms, d.conf, d.x1,d.y1,d.x2,d.y2
                 FROM assignments a JOIN detections d ON d.det_id=a.det_id WHERE a.seq=%s
                 ORDER BY d.frame_idx""", (seq,))
    # evenly subsample n det_ids for display (this is "what we'd subsample")
    strip = [dets[i] for i in (sorted({int(k*(len(dets)-1)/max(1, n-1)) for k in range(min(n, len(dets)))}))] if dets else []
    return {"seq": seq, "decision": dec, "n_dets": len(dets), "strip": strip, "dets": dets,
            "veto_explain": _veto_explain(seq)}


@app.get("/identity/{gid}")
def identity(gid: int, by: str = "wall", step: int = -1, t: int = -1):
    """An identity's tracklets + bank, AS OF the timeline cursor so it agrees with the embedding panel/KPIs:
      by=decision: tracklets decided by `step` whose identity AS OF step (canonical birth gid after
        merges<=step) is this gid -> exemplars_added == the bank the embedding shows at this step.
      by=wall: FINAL-gid membership (post-merge); t>=0 limits to tracklets whose representative det is
        committed by t. t<0 (or absent) = the full/final identity."""
    info = _q("SELECT * FROM identities WHERE gid=%s", (gid,), one=True)
    cols = """dl.seq, d.camera, d.abs_ms, dl.decision_type, dl.admitted, dl.det_id AS rep_det,
              dl.admit_reason, dl.admit_sim, dl.admit_min, dl.admit_tau, dl.coherence_floor, dl.max_reps,
              (SELECT count(*) FROM assignments a WHERE a.seq=dl.seq) AS n_dets"""
    if by == "decision":
        tracks = _q(f"""{_CANON_CTE}
                        SELECT {cols}
                        FROM decision_log dl JOIN detections d ON d.det_id=dl.det_id
                        JOIN _canon ON _canon.orig = dl.chosen_gid
                        WHERE dl.seq <= %s AND _canon.cur = %s
                        ORDER BY d.abs_ms""", (step, step, gid))
    else:
        tfilter = " AND d.abs_ms <= %s" if t >= 0 else ""
        args = (gid, t) if t >= 0 else (gid,)
        tracks = _q(f"""SELECT {cols}
                        FROM decision_log dl JOIN detections d ON d.det_id=dl.det_id
                        WHERE dl.seq IN (SELECT DISTINCT a.seq FROM assignments a WHERE a.gid=%s){tfilter}
                        ORDER BY d.abs_ms""", args)
    _warm_bank([r["rep_det"] for r in tracks])   # background pre-decode of this id's crops, seek-ordered
    return {"gid": gid, "info": info,
            "n_tracklets": len(tracks),
            "exemplars_added": sum(1 for r in tracks if r["admitted"]),
            "tracklets": tracks}


def _active_model(model: int) -> int:
    """Resolve the embedder model_id the projection reads. <0 -> the model with the most stored vectors
    (the single embedder of a run; under multi-model the caller passes an explicit model_id). -1 if empty."""
    if model >= 0:
        return model
    row = _q("""SELECT model_id FROM embeddings WHERE role='match'
                GROUP BY model_id ORDER BY count(*) DESC, model_id DESC LIMIT 1""", one=True)
    return int(row["model_id"]) if row else -1


@app.get("/embedding_projection")
def embedding_projection(card: str = "", mode: str = "bank", limit: int = 4000, t: int = -1,
                         by: str = "wall", step: int = -1, model: int = -1):
    """The matcher's match space, projected to 2D - "what the index/matcher actually sees".

    Reads the polymorphic `embeddings` table for one embedder `model` (default: the run's only embedder),
    so it works at ANY dim (64/1024/2048/…) - the PCA/UMAP projects whatever D it gets.
    mode=bank (default): one point per LIVE exemplar (embeddings role='rep') - the bank the matcher scores
      against. mode=det: one point per DETECTION (role='match') coloured by assigned gid (subsampled to `limit`).

    `t` (abs_ms; <0 or absent = full/final state): when t>=0, only rows committed by t are RETURNED. The
    PCA is always FIT on the FULL set so the 2D layout stays fixed and points accumulate as the scrubber moves.

    Returns {mode, n, dim, t, points:[{x,y,gid,n_exemplars,cameras,rep_det_id,det_id}]}. Empty (n=0) when no
    embeddings are stored for this model yet."""
    card_clause, card_params = _card_filter(card, "d.video")
    m = _active_model(model)
    if m < 0:
        return {"mode": mode, "n": 0, "dim": 0, "t": t, "points": []}
    # by=decision colours each point by its identity AS OF `step` (canonical birth gid after merges<=step);
    # by=wall colours by birth gid. (Bank seq == the admitting decision seq; e.gid == the birth gid.)
    if mode == "det" and by == "decision":
        rows = _vecq(
            f"""{_CANON_CTE}
                SELECT d.det_id, _canon.cur AS gid, a.seq, d.camera, d.abs_ms, e.vec AS v
                FROM embeddings e JOIN detections d ON d.det_id=e.entity_id
                JOIN assignments a ON a.det_id=d.det_id
                JOIN decision_log dl ON dl.seq = a.seq
                JOIN _canon ON _canon.orig = dl.chosen_gid
                WHERE e.entity_kind='tracklet' AND e.role='match' AND e.model_id=%s{card_clause}
                ORDER BY a.seq LIMIT %s""", (step, m, *card_params, max(1, limit)))
    elif mode == "det":
        rows = _vecq(
            f"""SELECT d.det_id, a.gid, a.seq, d.camera, d.abs_ms, e.vec AS v
                FROM embeddings e JOIN detections d ON d.det_id=e.entity_id
                JOIN assignments a ON a.det_id=d.det_id
                WHERE e.entity_kind='tracklet' AND e.role='match' AND e.model_id=%s{card_clause}
                ORDER BY d.abs_ms LIMIT %s""", (m, *card_params, max(1, limit)))
    elif by == "decision":
        rows = _vecq(
            f"""{_CANON_CTE}
                SELECT e.seq AS rep_id, _canon.cur AS gid, e.entity_id AS det_id, d.camera, d.abs_ms, e.vec AS v
                FROM embeddings e JOIN detections d ON d.det_id=e.entity_id
                JOIN _canon ON _canon.orig = e.gid
                WHERE e.entity_kind='tracklet' AND e.is_rep AND e.model_id=%s{card_clause}
                ORDER BY e.gid, e.seq""", (step, m, *card_params))
    else:
        # wall: colour reps by their FINAL (post-merge) gid - the rep's det carries the live assignment
        # (assignments.gid is moved to the survivor on merge). Its abs_ms also time-gates the exemplar.
        rows = _vecq(
            f"""SELECT e.seq AS rep_id, a.gid, e.entity_id AS det_id, d.camera, d.abs_ms, e.vec AS v
                FROM embeddings e JOIN detections d ON d.det_id=e.entity_id
                JOIN assignments a ON a.det_id=e.entity_id
                WHERE e.entity_kind='tracklet' AND e.is_rep AND e.model_id=%s{card_clause}
                ORDER BY a.gid, e.seq""", (m, *card_params))
    if not rows:
        return {"mode": mode, "n": 0, "dim": 0, "t": t, "points": []}
    # FIT PCA on the FULL set so coords are stable across scrubs, then keep only rows committed by the cursor.
    xy_all = _project_2d(np.stack([r["v"] for r in rows]))
    if by == "decision":                     # gate on ingest order: det -> assignment seq, bank -> rep_id
        seq_key = "seq" if mode == "det" else "rep_id"
        keep = [i for i, r in enumerate(rows) if r.get(seq_key) is not None and r[seq_key] <= step]
    elif t >= 0:                             # t>=0 is a real time (the scrubber start IS t=0) -> filter to it;
        keep = [i for i, r in enumerate(rows) if r["abs_ms"] is not None and r["abs_ms"] <= t]
    else:                                    # t<0 (default/absent) = full state, for direct API callers
        keep = list(range(len(rows)))
    if not keep:
        return {"mode": mode, "n": 0, "dim": int(rows[0]["v"].shape[0]), "t": t, "points": []}
    sub = [rows[i] for i in keep]
    xy = [xy_all[i] for i in keep]
    # per-gid aggregates (exemplar count + camera span + a representative det_id for a crop) so a hovered
    # point can show identity info without a second round-trip - computed over the RETURNED (t-filtered) set
    per_gid = {}
    for r in sub:
        g = per_gid.setdefault(int(r["gid"]), {"n": 0, "cams": set(), "rep_det": r["det_id"]})
        g["n"] += 1
        g["cams"].add(r["camera"])
    pts = []
    for r, (x, y) in zip(sub, xy):
        g = per_gid[int(r["gid"])]
        pts.append({"x": float(x), "y": float(y), "gid": int(r["gid"]),
                    "det_id": r["det_id"], "rep_det_id": g["rep_det"],
                    "n_exemplars": g["n"], "cameras": sorted(g["cams"])})
    return {"mode": mode, "n": len(pts), "dim": int(rows[0]["v"].shape[0]), "t": t, "points": pts}


@app.get("/decision_geometry/{det_id}")
def decision_geometry(det_id: str):
    """The why-it-matched-vs-expanded picture for one tracklet decision, as 2D geometry: the query
    detection's embedding + the candidate exemplars it scored against (the candidate gids' live bank
    exemplars) + tau - all co-projected so the threshold circle / nearest-candidate is visible.

    Returns {det_id, tau, decision_type, chosen_gid, query:{x,y}, candidates:[...], exemplars:[...]}."""
    dec = _q("""SELECT dl.seq, dl.chosen_gid, dl.decision_type, dl.threshold, dl.candidate_gids,
                       dl.scores, dl.veto_reasons, dl.admitted
                FROM decision_log dl WHERE dl.det_id=%s ORDER BY dl.seq DESC LIMIT 1""", (det_id,), one=True)
    m = _active_model(-1)
    qrow = _vecq("""SELECT e.entity_id AS det_id, e.vec AS v, d.camera
                    FROM embeddings e JOIN detections d ON d.det_id=e.entity_id
                    WHERE e.entity_kind='tracklet' AND e.role='match' AND e.model_id=%s AND e.entity_id=%s""",
                 (m, det_id)) if m >= 0 else []
    if not qrow or qrow[0]["v"] is None:
        return {"det_id": det_id, "tau": (dec or {}).get("threshold"), "decision": dec,
                "query": None, "candidates": [], "exemplars": [], "note": "no persisted embedding for this det"}
    cand_gids = list((dec or {}).get("candidate_gids") or [])
    scores = list((dec or {}).get("scores") or [])
    vetoes = list((dec or {}).get("veto_reasons") or [])
    # the live bank exemplars of every candidate gid the matcher scored (these are the vectors the cosine
    # was computed against) - co-projected with the query so distances/threshold are spatially meaningful
    exrows = []
    if cand_gids:
        exrows = _vecq("""SELECT e.seq AS rep_id, e.gid, e.entity_id AS det_id, d.camera, e.vec AS v
                          FROM embeddings e JOIN detections d ON d.det_id=e.entity_id
                          WHERE e.entity_kind='tracklet' AND e.is_rep AND e.model_id=%s AND e.gid = ANY(%s)""",
                       (m, cand_gids))
    mats = [qrow[0]["v"]] + [r["v"] for r in exrows]
    xy = _project_2d(np.stack(mats))
    query = {"x": float(xy[0][0]), "y": float(xy[0][1]), "det_id": det_id, "camera": qrow[0]["camera"]}
    exemplars = [{"x": float(xy[i + 1][0]), "y": float(xy[i + 1][1]), "gid": int(r["gid"]),
                  "det_id": r["det_id"], "camera": r["camera"]} for i, r in enumerate(exrows)]
    candidates = [{"gid": int(g), "score": (scores[i] if i < len(scores) else None),
                   "veto": (vetoes[i] if i < len(vetoes) else "")} for i, g in enumerate(cand_gids)]
    return {"det_id": det_id, "tau": (dec or {}).get("threshold"),
            "decision_type": (dec or {}).get("decision_type"), "chosen_gid": (dec or {}).get("chosen_gid"),
            "admitted": (dec or {}).get("admitted"),
            "query": query, "candidates": candidates, "exemplars": exemplars}


# no-store: crops/frames are keyed by det_id/frame but the DB behind them is re-ingestable (truncate +
# re-run), so a browser cache could pin a STALE image next to a correct det_id caption. The server keeps
# its own in-process cache, so this only costs a re-request, not a re-decode.
_JPEG_HDR = {"Cache-Control": "no-store"}


@app.get("/crop/{det_id}")
def crop(det_id: str, max_w: int = 160):
    key = f"{det_id}:{max_w}"
    b = _cache_get(_crop_cache, key)
    if b is not None:
        return Response(b, media_type="image/jpeg", headers=_JPEG_HDR)
    r = _q("SELECT video, frame_idx, x1,y1,x2,y2 FROM detections WHERE det_id=%s", (det_id,), one=True)
    if not r:
        return Response(status_code=404)
    frame = _read_frame(r["video"], r["frame_idx"])
    if frame is None:
        return Response(status_code=404)
    H, W = frame.shape[:2]
    x1 = max(0, int(r["x1"])); y1 = max(0, int(r["y1"])); x2 = min(W, int(r["x2"])); y2 = min(H, int(r["y2"]))
    if x2 <= x1 or y2 <= y1:
        return Response(status_code=404)
    b = _encode(frame[y1:y2, x1:x2], max_w, 85)
    if b is None:
        return Response(status_code=404)
    _cache_put(_crop_cache, key, b, _CROP_MAX)
    return Response(b, media_type="image/jpeg", headers=_JPEG_HDR)


@app.get("/frame/{video}")
def frame(video: str, frame: int, w: int = 420):
    """Full source video frame at frame_idx, downscaled - the canvas background. Keyed by VIDEO (not the
    camera name) so same-named cameras across test cards (Tc6/Tc8) show their OWN footage, not a collision."""
    key = f"{video}:{int(frame)}:{w}"
    b = _cache_get(_frame_cache, key)
    if b is not None:
        return Response(b, media_type="image/jpeg", headers=_JPEG_HDR)
    img = _read_frame(video, frame)
    if img is None:
        return Response(status_code=404)
    b = _encode(img, w, 75)
    if b is None:
        return Response(status_code=404)
    _cache_put(_frame_cache, key, b, _FRAME_MAX)
    return Response(b, media_type="image/jpeg", headers=_JPEG_HDR)


def _warm_bank(det_ids, max_w: int = 160):
    """Pre-decode an identity's bank crops into the cache in seek order, in the background, so the
    per-img /crop requests hit cache instead of each triggering its own (uncached) seek+decode."""
    todo = [d for d in det_ids if d and _cache_get(_crop_cache, f"{d}:{max_w}") is None]
    if not todo:
        return

    def run():
        rows = _q("SELECT det_id, video, frame_idx, x1,y1,x2,y2 FROM detections WHERE det_id = ANY(%s)", (todo,))
        for r in sorted(rows, key=lambda r: (r["video"], int(r["frame_idx"]))):  # forward seeks per video
            key = f'{r["det_id"]}:{max_w}'
            if _cache_get(_crop_cache, key) is not None:
                continue
            fr = _read_frame(r["video"], r["frame_idx"])
            if fr is None:
                continue
            H, W = fr.shape[:2]
            x1 = max(0, int(r["x1"])); y1 = max(0, int(r["y1"])); x2 = min(W, int(r["x2"])); y2 = min(H, int(r["y2"]))
            if x2 <= x1 or y2 <= y1:
                continue
            b = _encode(fr[y1:y2, x1:x2], max_w, 85)
            if b:
                _cache_put(_crop_cache, key, b, _CROP_MAX)

    threading.Thread(target=run, daemon=True).start()
