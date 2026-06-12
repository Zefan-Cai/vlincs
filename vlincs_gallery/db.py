"""Per-dataset pgvector databases + model provenance helpers.

One Postgres database per DATASET (gallery_ms02 / gallery_ds1 / gallery_ds2 / gallery_ds3) in the
shared `vlincs_gallery_pg` container, so ingestion runs are isolated and a truncate-before-ingest
gives clean performance numbers with zero cross-dataset contamination. `ensure_db` creates the DB
(if missing) and applies db/init.sql. `upsert_model` records detector/embedder/reducer settings and
returns a stable model_id to stamp onto detections.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import psycopg

# env-overridable so the same code runs against the local dev container (defaults) OR the kit's compose
# `db` service (PGHOST=db PGPORT=5432). Defaults preserve the existing local setup unchanged.
_HOST = os.environ.get("PGHOST", "localhost"); _PORT = int(os.environ.get("PGPORT", "55433"))
_USER = os.environ.get("PGUSER", "gallery"); _PW = os.environ.get("PGPASSWORD", "gallery")
ADMIN_DSN = f"postgresql://{_USER}:{_PW}@{_HOST}:{_PORT}/gallery"  # maintenance/default DB
INIT_SQL = str(Path(__file__).resolve().parents[1] / "db" / "init.sql")


def dataset_db(dataset: str) -> str:
    d = dataset.lower()
    if d.startswith("ms02") or d.startswith("ds0000"): return "gallery_ms02"
    if d.startswith("ds1") or d.startswith("ds0001"):  return "gallery_ds1"
    if d.startswith("ds2") or d.startswith("ds0002"):  return "gallery_ds2"
    if d.startswith("ds3") or d.startswith("ds0003"):  return "gallery_ds3"
    return "gallery_" + d.replace("-", "_")


def dsn(dataset: str) -> str:
    return f"postgresql://{_USER}:{_PW}@{_HOST}:{_PORT}/{dataset_db(dataset)}"


def ensure_db(dataset: str, init_sql: str = INIT_SQL) -> str:
    """Create the per-dataset DB if absent and (idempotently) apply the schema. Returns the dbname."""
    db = dataset_db(dataset)
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (db,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{db}"')
    with psycopg.connect(dsn(dataset), autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(Path(init_sql).read_text())
    return db


def upsert_model(cur, role: str, name: str, weights: str | None = None, params: dict | None = None) -> int:
    """Insert-or-get a model row; returns model_id. Dedup on (role,name,weights,params)."""
    cur.execute(
        """INSERT INTO models (role, name, weights, params)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (role, name, weights, params) DO UPDATE SET role = EXCLUDED.role
           RETURNING model_id""",
        (role, name, weights or "", json.dumps(params or {}, sort_keys=True)))
    return int(cur.fetchone()[0])


# ── Polymorphic embeddings: registry + per-model DB-side ANN (pgvector) alongside FAISS ──────────────
# The `embeddings` table stores any-dim vectors from any model (see db/init.sql). DB-side ANN is opt-in
# per model via a PARTIAL CAST-EXPRESSION HNSW index, which pins the dim only inside the index — so
# variable-dim storage and a real (Index Scan) pgvector ANN coexist. The dim ceilings are hard pgvector
# limits: HNSW supports vector ≤2000 dims, halfvec ≤4000; a >4000-d model gets storage + FAISS + viz but
# no DB-side index. `ann_search` MUST be the only place ANN queries are built — a cast/predicate that
# doesn't EXACTLY match the index silently falls back to a seq-scan with no error.

def emb_index_type(dim: int) -> str | None:
    """The pgvector index cast for a given dim: 'vector' (≤2000) | 'halfvec' (≤4000) | None (no DB ANN)."""
    return "vector" if dim <= 2000 else "halfvec" if dim <= 4000 else None


def register_embedder(cur, name: str, dim: int, *, weights: str | None = None, params: dict | None = None):
    """Upsert an 'embedder' model row carrying its output dim + DB-ANN index type. Returns (model_id, emb_type)."""
    emb_type = emb_index_type(int(dim))
    cur.execute(
        """INSERT INTO models (role, name, weights, params, emb_dim, emb_type)
           VALUES ('embedder', %s, %s, %s, %s, %s)
           ON CONFLICT (role, name, weights, params)
           DO UPDATE SET emb_dim = EXCLUDED.emb_dim, emb_type = EXCLUDED.emb_type
           RETURNING model_id""",
        (name, weights or "", json.dumps(params or {}, sort_keys=True), int(dim), emb_type))
    return int(cur.fetchone()[0]), emb_type


def enable_ann(dataset: str, model_id: int, dim: int, emb_type: str | None, role: str = "match") -> bool:
    """Create this model's partial cast-expression HNSW index so DB-side ANN works for its `role` vectors.
    Runs in its OWN autocommit connection — CREATE INDEX takes locks and must not sit inside the per-row
    ingest txn. dim>4000 (emb_type None) -> no DB ANN; returns whether an index was built."""
    if not emb_type:
        return False
    rlit = "'" + str(role).replace("'", "''") + "'"
    opclass = "vector_cosine_ops" if emb_type == "vector" else "halfvec_cosine_ops"
    sql = (f"CREATE INDEX IF NOT EXISTS emb_ann_{int(model_id)} ON embeddings "
           f"USING hnsw ((vec::{emb_type}({int(dim)})) {opclass}) "
           f"WHERE model_id = {int(model_id)} AND role = {rlit}")
    with psycopg.connect(dsn(dataset), autocommit=True) as c, c.cursor() as cur:
        cur.execute(sql)
    return True


def ann_search(con, model_id: int, q, k: int = 10, role: str = "match"):
    """DB-side ANN over a model's `role` vectors via its partial HNSW index. Reads the model's emb_dim/
    emb_type so the cast+predicate EXACTLY match the index (a mismatch silently falls back to seq-scan).
    `con` must have pgvector.register_vector applied. Returns [(entity_id, cosine_distance), ...].
    FAISS stays the live matcher's hot path; this is the DB path for replay / large-K / ad-hoc queries."""
    cur = con.cursor()
    cur.execute("SELECT emb_dim, emb_type FROM models WHERE model_id=%s", (model_id,))
    row = cur.fetchone()
    if not row or not row[1]:
        return []                         # no DB-side ANN for this model (dim>4000 or unregistered)
    dim, etype = int(row[0]), row[1]
    cast = f"::{etype}({dim})"
    cur.execute(
        f"""SELECT entity_id, (vec{cast} <=> %s{cast}) AS dist
            FROM embeddings WHERE model_id=%s AND role=%s
            ORDER BY vec{cast} <=> %s{cast} LIMIT %s""",
        (q, model_id, role, q, k))
    return [(r[0], float(r[1])) for r in cur.fetchall()]


def active_emb_model(cur, role: str = "match"):
    """The model_id whose `role` embeddings the viz reads by default — the embedder with the most rows
    (the single embedder of a run; under multi-model the caller passes an explicit model_id). None if empty."""
    cur.execute("""SELECT model_id FROM embeddings WHERE role=%s
                   GROUP BY model_id ORDER BY count(*) DESC, model_id DESC LIMIT 1""", (role,))
    row = cur.fetchone()
    return int(row[0]) if row else None
