-- vlincs_gallery system-of-record schema (PROTOCOL.md §6).
-- One DATABASE PER DATASET (gallery_ms02 / gallery_ds1 / gallery_ds2 / gallery_ds3) so ingestion
-- runs are isolated and truncate-before-ingest gives clean, uncontaminated performance numbers.
-- The DB is the durable record AND the cannot-link / time-window / geo query layer; the reduced
-- embedding_red column (+ hnsw cosine index) is the hot match/ANN space, the raw vector(1024) is
-- kept for audit/replay.

CREATE EXTENSION IF NOT EXISTS vector;

-- Model/settings provenance. Every artifact-producing component (detector, embedder, reducer,
-- tracker, …) gets a row; detections reference the ones that produced them, so you can always
-- look up "what detector + settings made this detection". Dedup on (role,name,weights,params).
CREATE TABLE IF NOT EXISTS models (
    model_id   BIGSERIAL PRIMARY KEY,
    role       TEXT NOT NULL,                 -- detector | embedder | reducer | tracker | …
    name       TEXT NOT NULL,                 -- e.g. yolo26x, solider, pca64
    weights    TEXT,                          -- weights/checkpoint path or MLflow URI
    params     JSONB NOT NULL DEFAULT '{}',   -- imgsz, conf, iou, dim, fps, …
    sha256     TEXT,                          -- of the weights, when available
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (role, name, weights, params)
);

-- Camera parameters per video (static pose from *_camera_extrinsics_*.parquet for all datasets;
-- full intrinsics K/D/R/t/ENU from MCAMxxx_camera_params.json on MS02). Enables geo filters:
-- camera-to-camera distance + travel-time cannot-link now, per-detection ground projection (filling
-- detections.lat/lon) next. start_ms here is the absolute clock epoch used for that video.
CREATE TABLE IF NOT EXISTS cameras (
    video      TEXT PRIMARY KEY,
    camera     TEXT NOT NULL,
    lat DOUBLE PRECISION, lon DOUBLE PRECISION, alt DOUBLE PRECISION,
    roll DOUBLE PRECISION, pitch DOUBLE PRECISION, yaw DOUBLE PRECISION,
    start_ms   BIGINT,                        -- absolute ms-since-midnight of frame 0
    fps        REAL,
    intrinsics JSONB,                         -- {K,D,R,t,enu_origin} when shipped (MS02); else NULL
    extrinsics_version TEXT
);
CREATE INDEX IF NOT EXISTS cam_camera ON cameras (camera);

-- Every detection (the ~1M DS1 rows live here; the exemplar bank holds only the few-hundred reps).
CREATE TABLE IF NOT EXISTS detections (
    det_id        TEXT PRIMARY KEY,           -- "<video>::<camera:frame_idx:box_idx>" — globally unique within a dataset DB (Tc6/Tc8 of one camera don't collide)
    video         TEXT NOT NULL,
    camera        TEXT NOT NULL,
    frame_idx     INTEGER NOT NULL,
    wall_clock_ms BIGINT NOT NULL,            -- RELATIVE ms (frame_idx/fps*1000), within-video
    abs_ms        BIGINT,                     -- ABSOLUTE common-clock ms (per-video extrinsics start + frame/fps): the cross-camera/cross-card timeline
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    box_hash      TEXT,
    conf          REAL,
    object_type   SMALLINT,                   -- 0=person, 3=vehicle
    lat DOUBLE PRECISION, lon DOUBLE PRECISION, alt DOUBLE PRECISION,  -- NULL => geo unavailable => veto inactive (per-detection ground projection is a follow-on)
    triage_score  REAL,                        -- NULL until the triage pass runs
    detector_id   BIGINT REFERENCES models(model_id),
    embedder_id   BIGINT REFERENCES models(model_id),
    reducer_id    BIGINT REFERENCES models(model_id),
    embedding     vector(1024),                -- raw embedder output (audit/replay)
    embedding_red vector(64)                   -- reduced (warmup-PCA) matching space; threshold + ANN operate here
);
CREATE INDEX IF NOT EXISTS det_cam_frame ON detections (camera, frame_idx);
CREATE INDEX IF NOT EXISTS det_cam_wall  ON detections (camera, wall_clock_ms);
CREATE INDEX IF NOT EXISTS det_abs_ms    ON detections (abs_ms);

-- Identities (galleries). centroid = confidence-adaptive EMA of member exemplars.
CREATE TABLE IF NOT EXISTS identities (
    gid           BIGINT PRIMARY KEY,
    centroid      vector(1024),
    centroid_red  vector(64),
    n_members     INTEGER NOT NULL DEFAULT 0,
    cameras       TEXT[]  NOT NULL DEFAULT '{}',
    first_wall_ms BIGINT,
    last_wall_ms  BIGINT,
    coherence     REAL,
    created_seq   BIGINT,
    updated_seq   BIGINT
);

-- The exemplar bank. Diversity-gated admission; cap=16. embedding_red is the cosine-ANN match space.
CREATE TABLE IF NOT EXISTS identity_reps (
    rep_id        BIGINT PRIMARY KEY,         -- stable; never reused within a run (replay safety)
    gid           BIGINT NOT NULL REFERENCES identities(gid),
    det_id        TEXT   NOT NULL REFERENCES detections(det_id),
    embedding     vector(1024),
    embedding_red vector(64) NOT NULL
);
CREATE INDEX IF NOT EXISTS rep_gid ON identity_reps (gid);
CREATE INDEX IF NOT EXISTS rep_red_hnsw ON identity_reps USING hnsw (embedding_red vector_cosine_ops);

-- The committed per-detection assignment (det_id -> gid).
CREATE TABLE IF NOT EXISTS assignments (
    det_id        TEXT PRIMARY KEY REFERENCES detections(det_id),
    gid           BIGINT NOT NULL REFERENCES identities(gid),
    score         REAL,
    decision_type TEXT   NOT NULL CHECK (decision_type IN ('match','expand','revised')),
    disc_ratio    REAL,
    seq           BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS asn_gid ON assignments (gid);
-- seq is filtered per-tracklet by the viz (/identity counts dets per tracklet; /tracklet joins on seq).
-- Without this every such query full-scans assignments -> /identity for a 561-tracklet id took 65s.
CREATE INDEX IF NOT EXISTS asn_seq ON assignments (seq);

-- Append-only event record -> the gallery state is a deterministic fold over this log (replay).
CREATE TABLE IF NOT EXISTS decision_log (
    seq                BIGINT PRIMARY KEY,
    det_id             TEXT NOT NULL,
    candidate_gids     BIGINT[],
    scores             REAL[],
    cannot_link_pruned BIGINT[],
    veto_reasons       TEXT[],                 -- aligned with candidate_gids: '' or 'same_frame'/'simultaneity:CAM'/'travel:CAM'/'below_tau' (the "why")
    chosen_gid         BIGINT,
    decision_type      TEXT,
    threshold          REAL,
    admitted           BOOLEAN,                -- was this tracklet's pooled vector ADDED to the exemplar bank (vs diversity-gate rejected)?
    admit_reason       TEXT,                   -- WHICH gate decided: added | redundant | incoherent | bank_full | quarantine
    admit_sim          REAL,                   -- closest-exemplar cosine ss.max() (the REDUNDANT number); NULL if no bank yet
    admit_min          REAL,                   -- nearest-bank cosine ss.min() (the INCOHERENT number);  NULL if no bank yet
    admit_tau          REAL,                   -- redundancy cutoff in effect (default 0.9)
    coherence_floor    REAL,                   -- too-far cutoff in effect (default 0.4)
    max_reps           INTEGER,                -- exemplar bank cap in effect (default 16)
    disc_ratio         REAL,
    code_sha           TEXT,
    seed               INTEGER
);
CREATE INDEX IF NOT EXISTS dlog_chosen ON decision_log (chosen_gid);

-- CREATE TABLE IF NOT EXISTS above is a no-op against an ALREADY-created decision_log (e.g. an existing
-- gallery_ms02 from a prior `up`), so it would NOT add the diversity-gate diagnostic columns. These
-- idempotent ALTERs backfill them so the viz picks them up WITHOUT a `down -v` / volume wipe.
ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS admit_reason    TEXT;
ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS admit_sim       REAL;
ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS admit_min       REAL;
ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS admit_tau       REAL;
ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS coherence_floor REAL;
ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS max_reps        INTEGER;

-- Consolidation/merge events from the periodic resolve(): at ingest step at_seq, old_gid was merged into
-- new_gid. decision_log records match/expand; this records the merges resolve() applies in place. Together
-- they are the COMPLETE event log, so the viz can REPLAY the gallery's exact identity state as of any step
-- (decision-order mode) instead of only the final post-merge state. Without it, a tracklet that later merges
-- would show its FINAL identity even at the step it was born ("a decision from the future").
CREATE TABLE IF NOT EXISTS merges (
    merge_id  BIGSERIAL PRIMARY KEY,           -- also the apply order within one resolve (monotonic)
    at_seq    BIGINT NOT NULL,                 -- last decision seq committed when this merge ran (the step it takes effect)
    old_gid   BIGINT NOT NULL,                 -- merged away
    new_gid   BIGINT NOT NULL,                 -- survivor
    score     REAL                             -- exemplar-centroid cosine that triggered it (>= merge_tau) — the WHY
);
CREATE INDEX IF NOT EXISTS merges_at_seq ON merges (at_seq);
ALTER TABLE merges ADD COLUMN IF NOT EXISTS score REAL;     -- backfill for an already-created merges table

-- Haversine metres for the geo cannot-link veto. NULL-propagating: if either point lacks geo the
-- result is NULL, so a `haversine_m(...) > X` predicate is FALSE/UNKNOWN and never blocks a match.
CREATE OR REPLACE FUNCTION haversine_m(lat1 DOUBLE PRECISION, lon1 DOUBLE PRECISION,
                                       lat2 DOUBLE PRECISION, lon2 DOUBLE PRECISION)
RETURNS DOUBLE PRECISION LANGUAGE sql IMMUTABLE AS $$
    SELECT 6371000.0 * 2 * asin(sqrt(
        power(sin(radians(lat2 - lat1) / 2), 2) +
        cos(radians(lat1)) * cos(radians(lat2)) * power(sin(radians(lon2 - lon1) / 2), 2)
    ));
$$;
