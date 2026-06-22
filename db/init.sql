-- vlincs_gallery system-of-record schema.
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
-- Embedder/reducer rows also record their output dim + how DB-side ANN indexes them (see `embeddings`).
ALTER TABLE models ADD COLUMN IF NOT EXISTS emb_dim  INTEGER;   -- embedding dimensionality
ALTER TABLE models ADD COLUMN IF NOT EXISTS emb_type TEXT;      -- DB-ANN cast: 'vector' (≤2000) | 'halfvec' (≤4000) | NULL (no DB ANN)

-- Camera parameters per video (static pose from *_camera_extrinsics_*.parquet for all datasets;
-- full intrinsics K/D/R/t/ENU from MCAMxxx_camera_params.json on MS02). Enables geo filters:
-- camera-to-camera distance + travel-time cannot-link. start_ms here is the absolute clock epoch used for that video.
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
    lat DOUBLE PRECISION, lon DOUBLE PRECISION, alt DOUBLE PRECISION,  -- NULL => geo unavailable => veto inactive
    triage_score  REAL,                        -- NULL until the triage pass runs
    detector_id   BIGINT REFERENCES models(model_id),
    embedder_id   BIGINT REFERENCES models(model_id),
    reducer_id    BIGINT REFERENCES models(model_id)
    -- embeddings live in the polymorphic `embeddings` table (any dim, multi-model), NOT inline here
);
CREATE INDEX IF NOT EXISTS det_cam_frame ON detections (camera, frame_idx);
CREATE INDEX IF NOT EXISTS det_cam_wall  ON detections (camera, wall_clock_ms);
CREATE INDEX IF NOT EXISTS det_abs_ms    ON detections (abs_ms);

-- Identities (galleries). The exemplar bank lives as role='rep' rows in `embeddings`; identity vectors
-- (centroids) are held in the in-memory matcher, not persisted.
CREATE TABLE IF NOT EXISTS identities (
    gid           BIGINT PRIMARY KEY,
    n_members     INTEGER NOT NULL DEFAULT 0,
    cameras       TEXT[]  NOT NULL DEFAULT '{}',
    first_wall_ms BIGINT,
    last_wall_ms  BIGINT,
    coherence     REAL,
    created_seq   BIGINT,
    updated_seq   BIGINT
);

-- ── Polymorphic embedding store: variable-dim, multi-model ──────────────────────────
-- ONE table for every embedding of every entity, at ANY dim, from ANY model. The `vec` column is
-- UNCONSTRAINED (`vector`, no typmod) so a 64-, 1024-, or 2048-d embedder all just store — nothing is
-- enforced upfront. DB-side ANN is opt-in PER MODEL via a PARTIAL CAST-EXPRESSION HNSW index (see
-- db.enable_ann), which pins the dim ONLY inside that index — reconciling "no dim upfront" with pgvector's
-- fixed-dim index requirement. FAISS (the live matcher) reads the same `vec`. Multi-model = >1 row per
-- entity (different model_id). role: 'match' (the scored vector) | 'rep' (exemplar bank) | 'raw' | 'centroid'.
-- ann_search() is the ONLY place ANN queries are built — a cast/predicate that doesn't EXACTLY match the
-- index silently falls back to a seq-scan. pgvector HNSW dim ceilings are hard: vector ≤2000, halfvec ≤4000.
-- The gallery's UNIT OF WORK is the TRACKLET (one pooled match vector per tracklet) — so there is ONE
-- 'match' row per tracklet (NOT one per detection; detections + assignments stay per-detection for scoring).
-- `is_rep` marks the subset admitted to the live exemplar bank. entity_id = the tracklet's representative
-- det_id (joins to detections for camera/abs_ms); seq = the tracklet's decision seq (its id + the replay key).
CREATE TABLE IF NOT EXISTS embeddings (
    entity_kind TEXT    NOT NULL,                   -- 'tracklet' (the gallery unit) | 'detection' | 'identity'
    entity_id   TEXT    NOT NULL,                   -- the tracklet's representative det_id
    model_id    BIGINT  NOT NULL REFERENCES models(model_id),
    role        TEXT    NOT NULL DEFAULT 'match',    -- 'match' (the scored vector) | 'raw' | 'centroid'
    dim         INTEGER NOT NULL,                    -- == vector_dims(vec) (denormalized for the index cast)
    vec         vector  NOT NULL,                    -- UNCONSTRAINED: any dim
    gid         BIGINT,                              -- the identity it was assigned to at birth (pre-merge)
    seq         BIGINT,                              -- the tracklet's decision seq (id + as-of-step replay key)
    is_rep      BOOLEAN NOT NULL DEFAULT false,      -- admitted to the live exemplar bank (the viz "bank" subset)
    PRIMARY KEY (entity_kind, entity_id, model_id, role)
);
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS is_rep BOOLEAN NOT NULL DEFAULT false;
CREATE INDEX IF NOT EXISTS emb_model_role ON embeddings (model_id, role);
CREATE INDEX IF NOT EXISTS emb_entity     ON embeddings (entity_id);
CREATE INDEX IF NOT EXISTS emb_rep        ON embeddings (model_id, seq) WHERE is_rep;

-- Tracklet metadata at the gallery unit of work. entity_id is the representative detection id used by
-- embeddings.entity_id; tracklet_key is the caller's stable external key when one exists. This lets
-- offline weak-label CSVs keyed by tracker output join back to the live DB without using GT labels.
CREATE TABLE IF NOT EXISTS tracklets (
    seq          BIGINT PRIMARY KEY,
    entity_id    TEXT NOT NULL REFERENCES detections(det_id) ON DELETE CASCADE,
    tracklet_key TEXT NOT NULL,
    video        TEXT NOT NULL,
    camera       TEXT NOT NULL,
    start_frame  INTEGER NOT NULL,
    end_frame    INTEGER NOT NULL,
    n_dets       INTEGER NOT NULL,
    UNIQUE (entity_id),
    UNIQUE (tracklet_key)
);
CREATE INDEX IF NOT EXISTS tracklets_key ON tracklets (tracklet_key);

-- Drop dead fixed-dim embedding columns / table / hnsw index from older per-dataset DBs (no-op on fresh
-- DBs). Run AFTER the `embeddings` CREATE.
ALTER TABLE detections DROP COLUMN IF EXISTS embedding;
ALTER TABLE detections DROP COLUMN IF EXISTS embedding_red;
ALTER TABLE identities DROP COLUMN IF EXISTS centroid;
ALTER TABLE identities DROP COLUMN IF EXISTS centroid_red;
DROP TABLE IF EXISTS identity_reps;

-- The committed per-detection assignment (det_id -> gid).
CREATE TABLE IF NOT EXISTS assignments (
    det_id        TEXT PRIMARY KEY REFERENCES detections(det_id),
    gid           BIGINT NOT NULL REFERENCES identities(gid),
    score         REAL,
    decision_type TEXT   NOT NULL CHECK (decision_type IN ('match','expand','quarantine','revised','forced')),
    seq           BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS asn_gid ON assignments (gid);
-- seq is filtered per-tracklet by the viz (/identity counts dets per tracklet; /tracklet joins on seq).
CREATE INDEX IF NOT EXISTS asn_seq ON assignments (seq);
-- Migration (idempotent, re-applied on every ensure_db): widen the decision_type CHECK to allow
-- 'quarantine' (match_or_expand emits match/expand/quarantine; decision_log.decision_type is free-text
-- and already stores the raw value). Self-heals DBs created before 'quarantine' was allowed. 'revised'
-- is retained for forward-compat (currently unused).
ALTER TABLE assignments DROP CONSTRAINT IF EXISTS assignments_decision_type_check;
ALTER TABLE assignments ADD CONSTRAINT assignments_decision_type_check
    CHECK (decision_type IN ('match','expand','quarantine','revised','forced'));
-- disc_ratio is a WHOLE-GALLERY scalar (mean kNN centroid separation), so a per-detection column was the
-- wrong grain - every det of a tracklet shared one gid and so one value. The decision-time snapshot lives
-- on decision_log.disc_ratio (one row per decision = a real time-series); drop the dead per-det column.
ALTER TABLE assignments DROP COLUMN IF EXISTS disc_ratio;

-- Optional weak evidence attached to the tracklet unit of work. entity_id is the representative
-- detection id used by embeddings.entity_id; tokens are language/VLM/CV attributes such as
-- {"upper_color":"black","hat":"no"} or ["upper_color:black", "hat:no"]. These are evidence,
-- never identity labels.
CREATE TABLE IF NOT EXISTS weak_tracklet_labels (
    entity_id  TEXT   NOT NULL REFERENCES detections(det_id) ON DELETE CASCADE,
    seq        BIGINT NOT NULL,
    source     TEXT   NOT NULL DEFAULT 'weak',
    tokens     JSONB  NOT NULL DEFAULT '{}',
    confidence REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entity_id, source)
);
CREATE INDEX IF NOT EXISTS weak_tracklet_labels_seq ON weak_tracklet_labels (seq);

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

-- Backfill the diversity-gate diagnostic columns into decision_log on already-created DBs.
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
ALTER TABLE merges ADD COLUMN IF NOT EXISTS score REAL;

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
