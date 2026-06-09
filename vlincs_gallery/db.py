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
