#!/usr/bin/env python
"""Tracklet-level oracle upper bound for DS1 global-ID resolution.

This script is an evaluation probe, not a production method. It uses GT only to
label each existing gallery tracklet by the majority IoU-matched GT identity,
then scores the resulting submission with the canonical VLINCS scorer. If this
upper bound cannot reach the target, a better identity model alone cannot reach
it with the current detector/tracklet boxes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from submit import _box_hash
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


_GT_ID_RE = re.compile(r"(\d+)$")


def _connect(dbname: str):
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "55433"),
        user=os.environ.get("PGUSER", "gallery"),
        password=os.environ.get("PGPASSWORD", "gallery"),
        dbname=dbname,
    )


def _load_predictions(con) -> dict[str, pd.DataFrame]:
    with con.cursor() as cur:
        cur.execute(
            """SELECT d.video, d.frame_idx AS frame, a.seq, d.x1, d.y1, d.x2, d.y2,
                      d.object_type, d.conf
               FROM detections d JOIN assignments a ON a.det_id = d.det_id
               ORDER BY d.video, d.frame_idx, a.seq"""
        )
        rows = cur.fetchall()
    df = pd.DataFrame(
        rows,
        columns=["video", "frame", "seq", "x1", "y1", "x2", "y2", "object_type", "confidence"],
    )
    out: dict[str, pd.DataFrame] = {}
    for video, g in df.groupby("video", sort=True):
        out[str(video)] = g.drop(columns=["video"]).reset_index(drop=True)
    return out


def _iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    px1, py1, px2, py2 = [pred_boxes[:, i][:, None] for i in range(4)]
    gx1, gy1, gx2, gy2 = [gt_boxes[:, i][None, :] for i in range(4)]
    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    p_area = np.maximum(px2 - px1, 0) * np.maximum(py2 - py1, 0)
    g_area = np.maximum(gx2 - gx1, 0) * np.maximum(gy2 - gy1, 0)
    return (inter / np.maximum(p_area + g_area - inter, 1e-9)).astype(np.float32)


def _gt_numeric_id(gt_id: object, mapping: dict[str, int]) -> int:
    key = str(gt_id)
    if key in mapping:
        return mapping[key]
    match = _GT_ID_RE.search(key)
    if match:
        candidate = int(match.group(1))
        if candidate > 0 and candidate not in mapping.values():
            mapping[key] = candidate
            return candidate
    mapping[key] = max(mapping.values(), default=0) + 1
    return mapping[key]


def _label_tracklets(
    pred_by_video: dict[str, pd.DataFrame],
    gt_by_video: dict[str, pd.DataFrame],
    *,
    iou_thr: float,
    min_matches: int,
    min_purity: float,
    singleton_offset: int,
) -> tuple[dict[int, int], dict[str, object]]:
    seq_counts: dict[int, Counter] = defaultdict(Counter)
    seq_total = Counter()
    gt_id_map: dict[str, int] = {}
    matched_rows = 0
    total_rows = 0

    for video, pred in pred_by_video.items():
        gt = gt_by_video.get(video)
        if gt is None or pred.empty:
            continue
        gt_frame_groups = {int(frame): rows for frame, rows in gt.groupby("frame", sort=False)}
        for frame, pframe in pred.groupby("frame", sort=False):
            total_rows += len(pframe)
            gframe = gt_frame_groups.get(int(frame))
            if gframe is None or gframe.empty:
                continue
            pboxes = pframe[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
            gboxes = gframe[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
            ious = _iou_matrix(pboxes, gboxes)
            best = ious.argmax(axis=1)
            best_iou = ious[np.arange(len(pframe)), best]
            gt_ids = gframe["id"].to_numpy()
            for row, j, score in zip(pframe.itertuples(index=False), best, best_iou):
                seq = int(row.seq)
                seq_total[seq] += 1
                if float(score) >= iou_thr:
                    seq_counts[seq][_gt_numeric_id(gt_ids[int(j)], gt_id_map)] += 1
                    matched_rows += 1

    seq_to_id: dict[int, int] = {}
    accepted = 0
    rejected = 0
    purities = []
    for seq, total in seq_total.items():
        counts = seq_counts.get(seq)
        if counts:
            gid, count = counts.most_common(1)[0]
            purity = count / max(sum(counts.values()), 1)
            purities.append(purity)
            if count >= min_matches and purity >= min_purity:
                seq_to_id[int(seq)] = int(gid)
                accepted += 1
                continue
        seq_to_id[int(seq)] = int(singleton_offset + int(seq))
        rejected += 1

    stats = {
        "iou_thr": float(iou_thr),
        "min_matches": int(min_matches),
        "min_purity": float(min_purity),
        "n_tracklets": len(seq_total),
        "accepted_gt_tracklets": accepted,
        "singleton_tracklets": rejected,
        "matched_detection_rows": matched_rows,
        "total_detection_rows": total_rows,
        "matched_detection_fraction": round(matched_rows / max(total_rows, 1), 6),
        "mean_gt_purity_of_matched_tracklets": round(float(np.mean(purities)) if purities else 0.0, 6),
        "uses_gt_for_oracle_upper_bound": True,
    }
    return seq_to_id, stats


def _build_comp(pred_by_video: dict[str, pd.DataFrame], seq_to_id: dict[int, int]) -> dict[str, pd.DataFrame]:
    comp: dict[str, pd.DataFrame] = {}
    for video, pred in pred_by_video.items():
        out = pred.copy()
        out["id"] = [int(seq_to_id.get(int(seq), 90_000_000 + int(seq))) for seq in out["seq"]]
        comp[video] = out[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
    return comp


def _export_zip(comp: dict[str, pd.DataFrame], out_zip: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="vlincs_oracle_submit_"))
    written = []
    for video, df in comp.items():
        out = df.copy()
        out["frame"] = out["frame"].astype("uint32")
        out["id"] = out["id"].astype("uint32")
        for col in ("x1", "y1", "x2", "y2"):
            out[col] = out[col].clip(lower=0).astype("uint32")
        out["object_type"] = out["object_type"].astype("uint8")
        out["confidence"] = out["confidence"].fillna(1.0).astype("float32")
        out["box_hash"] = [_box_hash(r.x1, r.y1, r.x2, r.y2) for r in out.itertuples()]
        for col in ("lat", "long", "alt"):
            out[col] = np.float64("nan")
        path = tmp / f"{video}.parquet"
        out[["frame", "id", "x1", "y1", "x2", "y2", "box_hash", "object_type", "confidence", "lat", "long", "alt"]].to_parquet(path)
        written.append(path)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in written:
            zf.write(path, path.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1", choices=["ds1"])
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--min-matches", type=int, default=1)
    ap.add_argument("--min-purity", type=float, default=0.0)
    ap.add_argument("--singleton-offset", type=int, default=90_000_000)
    ap.add_argument("--submit", default=None)
    ap.add_argument("--json", default=None, help="optional metrics/stats JSON path")
    args = ap.parse_args()

    con = _connect(args.dbname)
    pred_by_video = _load_predictions(con)
    gt_by_video = load_ds1_gt_by_video()
    gt_by_video = {key: value for key, value in gt_by_video.items() if key in pred_by_video}
    seq_to_id, stats = _label_tracklets(
        pred_by_video,
        gt_by_video,
        iou_thr=args.iou_thr,
        min_matches=args.min_matches,
        min_purity=args.min_purity,
        singleton_offset=args.singleton_offset,
    )
    comp = _build_comp(pred_by_video, seq_to_id)
    metrics = evaluate(gt_by_video, {key: comp[key] for key in gt_by_video}, dense=False, n_workers=1)
    result = {
        "dataset": args.dataset,
        "idf1": round(metrics.idf1, 6),
        "hota": round(metrics.hota, 6),
        "assa": round(metrics.assa, 6),
        "deta": round(metrics.deta, 6),
        "detre": round(metrics.detre, 6),
        "detpr": round(metrics.detpr, 6),
        "unmatched_fp": int(metrics.unmatched_fp),
        "per_video": {
            key: {metric: round(float(value), 6) for metric, value in vals.items()}
            for key, vals in sorted(metrics.per_video.items())
        },
        "oracle_stats": stats,
    }
    if args.submit:
        _export_zip(comp, args.submit)
        result["submission"] = args.submit
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
