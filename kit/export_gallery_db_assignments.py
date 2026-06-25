#!/usr/bin/env python3
"""Export gallery DB output as a tracklet-level assignment CSV.

This bridges the shipped gallery pipeline and the no-anchor repair tools.  The
gallery DB is the system of record for ``seq -> gid`` after ``kit/demo.py`` has
run, while DS1's stable ``tracklet_key`` and metadata live in the parquet/LFS
bundle.  This script joins those two production-side sources without reading
anchors or GT labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _connect(dbname: str):
    host = os.environ.get("PGHOST", "localhost")
    port = int(os.environ.get("PGPORT", "55433"))
    user = os.environ.get("PGUSER", "gallery")
    password = os.environ.get("PGPASSWORD", "gallery")
    return psycopg.connect(host=host, port=port, user=user, password=password, dbname=dbname)


def _db_gid_by_seq(dbname: str) -> tuple[dict[int, int], dict[str, Any]]:
    with _connect(dbname) as con, con.cursor() as cur:
        cur.execute(
            """
            WITH counts AS (
                SELECT seq, gid, COUNT(*) AS n
                FROM assignments
                GROUP BY seq, gid
            ),
            ranked AS (
                SELECT seq, gid, n,
                       ROW_NUMBER() OVER (PARTITION BY seq ORDER BY n DESC, gid ASC) AS rn,
                       COUNT(*) OVER (PARTITION BY seq) AS gid_choices
                FROM counts
            )
            SELECT seq, gid, n, gid_choices
            FROM ranked
            WHERE rn = 1
            ORDER BY seq
            """
        )
        rows = cur.fetchall()
    gid_by_seq = {int(seq): int(gid) for seq, gid, _n, _choices in rows}
    ambiguous = sum(1 for _seq, _gid, _n, choices in rows if int(choices) > 1)
    meta = {
        "dbname": dbname,
        "db_tracklets": int(len(gid_by_seq)),
        "ambiguous_seq_gid_votes": int(ambiguous),
        "min_db_seq": int(min(gid_by_seq)) if gid_by_seq else None,
        "max_db_seq": int(max(gid_by_seq)) if gid_by_seq else None,
        "predicted_ids": int(len(set(gid_by_seq.values()))),
    }
    return gid_by_seq, meta


def _embedding_keys(path: Path) -> set[str] | None:
    npz = path / "embeddings.npz"
    if not npz.is_file():
        return None
    import numpy as np

    data = np.load(npz, allow_pickle=True)
    return {str(x) for x in data["track_ids"].tolist()}


def _load_tracklet_rows(tracklet_root: Path, embedding_root: Path | None) -> list[dict[str, Any]]:
    if embedding_root is not None and embedding_root.is_dir():
        vdirs = sorted(p for p in embedding_root.iterdir() if p.is_dir())
    else:
        vdirs = sorted(p for p in tracklet_root.iterdir() if p.is_dir())
    rows: list[dict[str, Any]] = []
    for vdir in vdirs:
        video = vdir.name
        pq = tracklet_root / video / "tracklets.parquet"
        if not pq.is_file():
            continue
        valid_keys = _embedding_keys(vdir) if embedding_root is not None else None
        camera = video.split("_")[3] if len(video.split("_")) > 3 else ""
        df = pd.read_parquet(pq)
        for tracklet_key, group in df.groupby("tracklet_key"):
            key = str(tracklet_key)
            if valid_keys is not None and key not in valid_keys:
                continue
            group = group.sort_values("frame_idx")
            rows.append(
                {
                    "tracklet_key": key,
                    "video": str(group["video_key"].iloc[0]) if "video_key" in group.columns else video,
                    "camera": camera,
                    "start_frame": int(group["frame_idx"].min()),
                    "end_frame": int(group["frame_idx"].max()),
                    "n_dets": int(len(group)),
                    "avg_conf": float(group["score"].astype(float).mean()) if "score" in group.columns else 0.0,
                }
            )
    return rows


def _component_label(gid: int, mode: str, offset: int, dense_map: dict[int, int]) -> int:
    if mode == "gid":
        return int(gid)
    if mode == "dense":
        return int(dense_map[int(gid)])
    label = int(gid) - int(offset)
    return int(label if label >= 0 else dense_map[int(gid)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--tracklet-root", default=str(REPO_ROOT / "kit" / "demo_data" / "ds1" / "tracklets"))
    ap.add_argument("--embedding-root", default=str(REPO_ROOT / "kit" / "demo_data" / "ds1" / "embeddings"))
    ap.add_argument("--assignment-out", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--seq-origin", choices=["zero", "db"], default="zero")
    ap.add_argument("--component-label-mode", choices=["offset", "dense", "gid"], default="offset")
    ap.add_argument("--component-offset", type=int, default=100_000_000)
    ap.add_argument("--decision-status", default="gallery_db_component")
    ap.add_argument("--prediction-confidence", type=float, default=0.7)
    args = ap.parse_args()

    gid_by_db_seq, db_meta = _db_gid_by_seq(str(args.dbname))
    records = _load_tracklet_rows(Path(args.tracklet_root), Path(args.embedding_root) if args.embedding_root else None)
    if len(records) != len(gid_by_db_seq):
        raise SystemExit(
            f"tracklet bundle has {len(records)} joined rows but DB has {len(gid_by_db_seq)} seqs; "
            "check --tracklet-root/--embedding-root and the gallery DB"
        )

    dense_map = {gid: i for i, gid in enumerate(sorted(set(gid_by_db_seq.values())))}
    component_sizes = Counter(
        _component_label(gid, str(args.component_label_mode), int(args.component_offset), dense_map)
        for gid in gid_by_db_seq.values()
    )
    out_rows = []
    for pos, record in enumerate(records, start=1):
        gid = int(gid_by_db_seq[pos])
        label = _component_label(gid, str(args.component_label_mode), int(args.component_offset), dense_map)
        seq = pos - 1 if args.seq_origin == "zero" else pos
        out_rows.append(
            {
                "seq": int(seq),
                "db_seq": int(pos),
                **record,
                "predicted_global_id": int(gid),
                "component_label": int(label),
                "component_size": int(component_sizes[label]),
                "prediction_confidence": f"{float(args.prediction_confidence):.6f}",
                "decision_status": str(args.decision_status),
                "component_internal_edges": 0,
                "component_internal_prob_median": 0.0,
                "component_internal_score_median": 0.0,
                "component_external_prob_max": 0.0,
                "component_margin_prob": 0.0,
            }
        )

    args.assignment_out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else ["seq", "predicted_global_id"]
    with args.assignment_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    report = {
        "assignment_csv": str(args.assignment_out),
        "assignment_sha256": _sha256(args.assignment_out),
        "tracklet_root": str(args.tracklet_root),
        "embedding_root": str(args.embedding_root),
        "rows": int(len(out_rows)),
        "seq_origin": str(args.seq_origin),
        "component_label_mode": str(args.component_label_mode),
        "component_offset": int(args.component_offset),
        "decision_status": str(args.decision_status),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        **db_meta,
    }
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
