"""Export a TA1 submission from the live gallery DB - the ONLY file the kit ever writes.

One parquet per video in the canonical takehome schema (the exact columns the leaderboard scorer
expects), zipped. box_hash is recomputed per row from the bbox (the documented formula), so the zip is
self-consistent. Confidence/geo are placeholders (TA1 scores identity, not geo). This is a convenience
export so a colleague can submit what they built; verify with the official scorer before upload.
"""
from __future__ import annotations
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _box_hash(x1: float, y1: float, x2: float, y2: float) -> str:
    """SHA-256 of the bbox corners rounded to 6 dp - the documented takehome ``box_hash`` formula."""
    return hashlib.sha256(json.dumps((round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6))).encode()).hexdigest()


def export(con, dataset: str, out_zip: str) -> str:
    """Write a TA1 submission zip from the live gallery DB.

    Joins ``detections`` to ``assignments`` (unassigned detections get id 0), groups by video, and
    writes one ``<video>.parquet`` per video in the canonical takehome schema (with a freshly
    recomputed ``box_hash`` per row), then zips them.

    Args:
        con: An open psycopg connection to the dataset's gallery DB.
        dataset: Dataset key, recorded for context; the rows come from ``con``.
        out_zip: Path to write the ``.zip`` to.

    Returns:
        str: The ``out_zip`` path written.
    """
    with con.cursor() as cur:
        cur.execute("""SELECT d.video, d.frame_idx, COALESCE(a.gid, 0) AS id, d.x1, d.y1, d.x2, d.y2,
                              d.object_type, d.conf
                       FROM detections d LEFT JOIN assignments a ON a.det_id = d.det_id
                       ORDER BY d.video, d.frame_idx""")
        rows = cur.fetchall()
    out_zip = str(out_zip)
    by_video: dict[str, list] = {}
    for video, frame, gid, x1, y1, x2, y2, ot, conf in rows:
        by_video.setdefault(video, []).append((frame, gid, x1, y1, x2, y2, ot, conf))
    tmp = Path(tempfile.mkdtemp(prefix="vlincs_submit_"))
    written = []
    for video, recs in by_video.items():
        df = pd.DataFrame(recs, columns=["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"])
        df["frame"] = df["frame"].astype("uint32")
        df["id"] = df["id"].fillna(0).astype("uint32")
        for c in ("x1", "y1", "x2", "y2"):
            df[c] = df[c].clip(lower=0).astype("uint32")
        df["object_type"] = df["object_type"].astype("uint8")
        df["confidence"] = df["confidence"].astype("float32")
        df["box_hash"] = [_box_hash(r.x1, r.y1, r.x2, r.y2) for r in df.itertuples()]
        for c in ("lat", "long", "alt"):
            df[c] = np.float64("nan")
        p = tmp / f"{video}.parquet"
        df[["frame", "id", "x1", "y1", "x2", "y2", "box_hash", "object_type", "confidence",
            "lat", "long", "alt"]].to_parquet(p)
        written.append(p)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in written:
            z.write(p, p.name)
    print(f"[submit] wrote {out_zip}  ({len(written)} videos, {len(rows)} detections)")
    return out_zip
