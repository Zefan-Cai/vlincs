#!/usr/bin/env python
"""Round-trip smoke test for the vlincs_gallery pgvector system-of-record.

Verifies the container brought up from db/docker-compose.yml is usable end-to-end:
  - connects to localhost:55433 (postgresql://gallery:gallery@localhost:55433/gallery)
  - registers the pgvector type adapter on the connection
  - INSERTs a `detections` row carrying a real 1024-d embedding
  - SELECTs it back and asserts the vector round-trips bit-exactly
  - runs a cosine-distance (`<=>`) query against two stored 1024-d embeddings and
    asserts the operator returns the expected ordering / magnitudes
  - calls the schema's `haversine_m()` SQL function on two known lat/lon points and
    asserts the great-circle distance matches the analytic value within tolerance

The test cleans up the rows it inserts (idempotent: safe to re-run). It does NOT
query embeddings on the hot path in production (FAISS does) — this only exercises
the durable system-of-record so we know the container + schema + adapter all work.

Run with the dedicated venv:
    PYTHONPATH=<repo> vlincs_gallery/.venv/bin/python vlincs_gallery/db/smoke_test.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DSN = "postgresql://gallery:gallery@localhost:55433/gallery"
DIM = 1024

# Deterministic det_ids in the documented camera:frame_idx:box_idx form so a re-run
# overwrites cleanly and never collides with real ingest data.
DET_A = "SMOKE:0:0"
DET_B = "SMOKE:0:1"
DET_Q = "SMOKE:0:2"


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def main() -> int:
    rng = np.random.default_rng(0)

    # Three L2-normalized 1024-d embeddings. vec_a and the query are intentionally
    # close (small perturbation); vec_b is independent random => far in cosine space.
    vec_a = _unit(rng.standard_normal(DIM).astype(np.float32))
    vec_b = _unit(rng.standard_normal(DIM).astype(np.float32))
    vec_q = _unit(vec_a + 0.05 * rng.standard_normal(DIM).astype(np.float32))

    with psycopg.connect(DSN) as conn:
        # The pgvector adapter must be registered so numpy arrays bind to vector(1024)
        # and the DB sends vectors back as numpy arrays.
        register_vector(conn)
        with conn.cursor() as cur:
            # Clean slate for these smoke rows (assignments/reps reference detections;
            # drop dependents first just in case a prior run left them).
            cur.execute(
                "DELETE FROM detections WHERE det_id = ANY(%s)",
                ([DET_A, DET_B, DET_Q],),
            )

            # --- INSERT a detection row with a 1024-d embedding ---------------------
            cur.execute(
                """
                INSERT INTO detections
                    (det_id, video, camera, frame_idx, wall_clock_ms,
                     x1, y1, x2, y2, conf, object_type,
                     lat, lon, alt, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s)
                """,
                (
                    DET_A, "SMOKEVID", "SMOKECAM", 0, 1000,
                    10.0, 20.0, 30.0, 40.0, 0.9, 0,
                    38.8895, -77.0353, 205.0, vec_a,
                ),
            )
            # A second + query det so the cosine ordering test has something to rank.
            cur.execute(
                """
                INSERT INTO detections
                    (det_id, video, camera, frame_idx, wall_clock_ms, object_type, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (DET_B, "SMOKEVID", "SMOKECAM", 0, 1001, 0, vec_b),
            )
            cur.execute(
                """
                INSERT INTO detections
                    (det_id, video, camera, frame_idx, wall_clock_ms, object_type, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (DET_Q, "SMOKEVID", "SMOKECAM", 0, 1002, 0, vec_q),
            )

            # --- SELECT the embedding back and assert exact round-trip --------------
            cur.execute(
                "SELECT embedding FROM detections WHERE det_id = %s", (DET_A,)
            )
            (got,) = cur.fetchone()
            assert isinstance(got, np.ndarray), f"expected numpy array, got {type(got)}"
            assert got.shape == (DIM,), f"bad shape {got.shape}"
            max_abs_err = float(np.max(np.abs(got.astype(np.float32) - vec_a)))
            # pgvector stores float4 => exact for our float32 source.
            assert max_abs_err < 1e-6, f"embedding round-trip drift {max_abs_err}"
            print(f"[ok] 1024-d embedding INSERT/SELECT round-trip (max_abs_err={max_abs_err:.2e})")

            # --- cosine distance (<=>) query against the query embedding ------------
            # <=> is cosine distance in [0,2]; 0 == identical direction.
            cur.execute(
                """
                SELECT det_id, (embedding <=> %s) AS cos_dist
                FROM detections
                WHERE det_id = ANY(%s)
                ORDER BY cos_dist ASC
                """,
                (vec_q, [DET_A, DET_B]),
            )
            ranked = cur.fetchall()
            ids = [r[0] for r in ranked]
            dists = {r[0]: float(r[1]) for r in ranked}
            assert ids[0] == DET_A, f"nearest should be DET_A, got {ids}"
            # Cross-check pgvector's cosine distance against numpy.
            np_cos_a = 1.0 - float(np.dot(vec_q, vec_a))
            np_cos_b = 1.0 - float(np.dot(vec_q, vec_b))
            assert abs(dists[DET_A] - np_cos_a) < 1e-5, (dists[DET_A], np_cos_a)
            assert abs(dists[DET_B] - np_cos_b) < 1e-5, (dists[DET_B], np_cos_b)
            assert dists[DET_A] < dists[DET_B], (dists[DET_A], dists[DET_B])
            print(
                f"[ok] cosine <=> query: nearest={ids[0]} "
                f"d(A)={dists[DET_A]:.4f} < d(B)={dists[DET_B]:.4f} "
                f"(numpy A={np_cos_a:.4f} B={np_cos_b:.4f})"
            )

            # --- haversine_m() on two known points ---------------------------------
            # Reference pair: roughly 1 deg of latitude apart at the equator-ish line.
            # Use a well-known leg: Washington DC area two points ~ measurable distance.
            lat1, lon1 = 38.8895, -77.0353   # ~ National Mall
            lat2, lon2 = 38.8977, -77.0365   # ~ White House
            cur.execute(
                "SELECT haversine_m(%s, %s, %s, %s)", (lat1, lon1, lat2, lon2)
            )
            (hav,) = cur.fetchone()
            hav = float(hav)
            # Independent analytic haversine to validate the SQL function.
            R = 6371000.0
            dphi = math.radians(lat2 - lat1)
            dlam = math.radians(lon2 - lon1)
            a = (math.sin(dphi / 2) ** 2
                 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
                 * math.sin(dlam / 2) ** 2)
            ref = R * 2 * math.asin(math.sqrt(a))
            assert abs(hav - ref) < 1e-3, f"haversine_m {hav} vs analytic {ref}"
            assert 900.0 < hav < 1000.0, f"unexpected distance {hav} m"
            print(f"[ok] haversine_m() = {hav:.3f} m (analytic {ref:.3f} m)")

            # --- NULL-propagation guarantee for the geo veto -----------------------
            cur.execute("SELECT haversine_m(%s, %s, NULL, NULL)", (lat1, lon1))
            (hav_null,) = cur.fetchone()
            assert hav_null is None, f"NULL geo should propagate NULL, got {hav_null}"
            # A `> X` predicate over NULL must be NULL (treated as not-true) so it
            # never blocks a match — the documented NaN-safe veto behavior.
            cur.execute("SELECT (haversine_m(%s, %s, NULL, NULL) > 50.0)", (lat1, lon1))
            (pred,) = cur.fetchone()
            assert pred is None, f"veto predicate over NULL should be NULL, got {pred}"
            print("[ok] haversine_m() NULL-propagation: NULL geo => veto predicate inactive")

            # Clean up the smoke rows so the table is left as init.sql created it.
            cur.execute(
                "DELETE FROM detections WHERE det_id = ANY(%s)",
                ([DET_A, DET_B, DET_Q],),
            )
        conn.commit()

    print("\nSMOKE TEST PASSED: pgvector container + schema + adapter round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
