#!/usr/bin/env python
"""Export relinked VLINCS tracklets as no-anchor sample parquet/features.

This bridges the alternative within-camera linkers to the existing
``no_anchor_sample_parquet_sweep.py`` resolver.  It reads detections and the
already-computed no-GT gallery embeddings from PostgreSQL, relinks detections
into synthetic tracklets, and writes:

* a per-detection parquet with tracklet keys and GT-majority columns for eval;
* a feature NPZ aligned by tracklet_key;
* an optional weak baseline assignment that replays the current gallery gid.

Ground truth is used only to populate evaluation columns and optional oracle
diagnostics.  No anchors or identity labels are used in any feature block.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    import psycopg
except ModuleNotFoundError as exc:  # pragma: no cover - remote venv has psycopg.
    raise RuntimeError("psycopg is required for gallery DB export") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlincs_gallery.eval.score import load_ds1_gt_by_video
from vlincs_gallery.sort_tracker import ocsort_tracklets, sort_tracklets
from vlincs_gallery.tracklets import link_tracklets

try:
    from kit.no_anchor_resolve_sweep import _label_tracklets_for_eval, _score_full
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import _label_tracklets_for_eval, _score_full


TrackerFn = Callable[..., dict[str, str]]


def _connect(dbname: str):
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "55433"),
        user=os.environ.get("PGUSER", "gallery"),
        password=os.environ.get("PGPASSWORD", "gallery"),
        dbname=dbname,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return (x / (float(np.linalg.norm(x)) + 1.0e-9)).astype(np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)).astype(np.float32)


def _vec_array(value: object) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return np.fromstring(text, sep=",", dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _standardize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    mean = mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True) + 1.0e-6
    return ((mat - mean) / std).astype(np.float32)


def _load_detections(con, limit_videos: set[str] | None) -> pd.DataFrame:
    where = ""
    params: tuple[object, ...] = ()
    if limit_videos:
        where = "WHERE d.video = ANY(%s)"
        params = (sorted(limit_videos),)
    with con.cursor() as cur:
        cur.execute(
            f"""SELECT d.det_id, d.video, d.camera, d.frame_idx, d.abs_ms,
                       d.x1, d.y1, d.x2, d.y2, d.object_type, d.conf
                FROM detections d
                {where}
                ORDER BY d.video, d.camera, d.frame_idx, d.det_id""",
            params,
        )
        rows = cur.fetchall()
    df = pd.DataFrame(
        rows,
        columns=[
            "det_id",
            "video",
            "camera",
            "frame_idx",
            "abs_ms",
            "x1",
            "y1",
            "x2",
            "y2",
            "object_type",
            "conf",
        ],
    )
    if df.empty:
        raise RuntimeError("no detections loaded")
    df["det_id"] = df["det_id"].astype(str)
    df["video"] = df["video"].astype(str)
    df["camera"] = df["camera"].astype(str)
    df["frame_idx"] = df["frame_idx"].astype(np.int64)
    df["abs_ms"] = df["abs_ms"].fillna(0).astype(np.int64)
    df["object_type"] = df["object_type"].fillna(0).astype(np.int64)
    for col in ("x1", "y1", "x2", "y2", "conf"):
        df[col] = df[col].astype(float)
    return df


def _load_current_assignments(con) -> pd.DataFrame:
    with con.cursor() as cur:
        cur.execute(
            """SELECT det_id, seq, gid, score
               FROM assignments
               ORDER BY det_id"""
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["det_id", "current_seq", "current_gid", "current_score"])
    df["det_id"] = df["det_id"].astype(str)
    df["current_seq"] = df["current_seq"].astype(np.int64)
    df["current_gid"] = df["current_gid"].astype(np.int64)
    df["current_score"] = df["current_score"].fillna(0.0).astype(float)
    return df


def _load_seq_embeddings(con, role: str) -> dict[int, np.ndarray]:
    with con.cursor() as cur:
        cur.execute(
            """SELECT DISTINCT ON (seq) seq, vec
               FROM embeddings
               WHERE role = %s
               ORDER BY seq, entity_id""",
            (role,),
        )
        rows = cur.fetchall()
    out = {int(seq): _l2n(_vec_array(vec)) for seq, vec in rows}
    if not out:
        raise RuntimeError(f"no role={role!r} embeddings found")
    return out


def _current_gid_centroids(assignments: pd.DataFrame, seq_emb: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    by_gid: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in assignments.groupby(["current_seq", "current_gid"], sort=False).size().reset_index(name="n").itertuples(index=False):
        emb = seq_emb.get(int(row.current_seq))
        if emb is None:
            continue
        by_gid[int(row.current_gid)].append(emb * float(row.n))
    return {gid: _l2n(np.sum(vecs, axis=0)) for gid, vecs in by_gid.items() if vecs}


def _parse_linker_kwargs(args: argparse.Namespace) -> tuple[str, TrackerFn, dict[str, object]]:
    kind = str(args.linker)
    if kind == "greedy":
        return kind, link_tracklets, {"iou_thresh": float(args.iou_thresh), "max_gap": int(args.max_gap)}
    if kind == "sort":
        return (
            kind,
            sort_tracklets,
            {
                "iou_thresh": float(args.iou_thresh),
                "max_gap": int(args.max_gap),
                "high_thresh": float(args.high_thresh),
                "new_thresh": float(args.new_thresh),
            },
        )
    if kind == "ocsort":
        return (
            kind,
            ocsort_tracklets,
            {
                "iou_thresh": float(args.iou_thresh),
                "max_gap": int(args.max_gap),
                "new_thresh": float(args.new_thresh),
                "delta_t": int(args.delta_t),
                "vdc": float(args.vdc),
            },
        )
    raise ValueError(f"unknown linker: {kind}")


def _run_linker(meta: pd.DataFrame, fn: TrackerFn, kwargs: dict[str, object]) -> dict[str, str]:
    det_to_key: dict[str, str] = {}
    for video, video_df in meta.groupby("video", sort=True):
        local_assign = fn(video_df.copy(), **kwargs)
        for det_id, key in local_assign.items():
            det_to_key[str(det_id)] = f"{video}:{key}"
    if len(det_to_key) != len(meta):
        missing = set(meta["det_id"].astype(str)) - set(det_to_key)
        raise RuntimeError(f"linker missed {len(missing)} detections; first={next(iter(missing), None)}")
    return det_to_key


def _attach_tracklet_keys(meta: pd.DataFrame, det_to_key: dict[str, str]) -> tuple[pd.DataFrame, dict[str, int]]:
    work = meta.copy()
    work["tracklet_key"] = work["det_id"].map(det_to_key).astype(str)
    order = (
        work.groupby("tracklet_key", sort=False)
        .agg(video=("video", "first"), start_frame=("frame_idx", "min"), end_frame=("frame_idx", "max"))
        .reset_index()
        .sort_values(["video", "start_frame", "end_frame", "tracklet_key"], kind="mergesort")
    )
    key_to_seq = {str(key): idx + 1 for idx, key in enumerate(order["tracklet_key"].tolist())}
    work["seq"] = work["tracklet_key"].map(key_to_seq).astype(np.int64)
    return work, key_to_seq


def _pred_by_video(work: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pred = work.rename(columns={"frame_idx": "frame", "conf": "confidence"})
    pred = pred[["video", "frame", "seq", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
    return {
        str(video): group.drop(columns=["video"]).reset_index(drop=True)
        for video, group in pred.groupby("video", sort=True)
    }


def _write_parquet(work: pd.DataFrame, gt_by_seq: dict[int, int], out_path: Path) -> dict[str, object]:
    out = work[
        [
            "video",
            "camera",
            "frame_idx",
            "x1",
            "y1",
            "x2",
            "y2",
            "object_type",
            "conf",
            "det_id",
            "tracklet_key",
            "seq",
            "current_seq",
            "current_gid",
            "current_score",
        ]
    ].copy()
    out = out.rename(columns={"video": "video_key", "conf": "score"})
    out["tracklet_majority_gt_id"] = out["seq"].map(lambda seq: f"M{int(gt_by_seq[int(seq)]):04d}" if int(seq) in gt_by_seq else np.nan)
    out["tracklet_majority_gt_fraction"] = out["seq"].map(lambda seq: 1.0 if int(seq) in gt_by_seq else 0.0).astype(float)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return {
        "parquet_out": str(out_path),
        "parquet_rows": int(len(out)),
        "parquet_tracklets": int(out["tracklet_key"].nunique()),
    }


def _tracklet_feature_rows(work: pd.DataFrame, seq_emb: dict[int, np.ndarray], gid_centroids: dict[int, np.ndarray]) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    if not seq_emb:
        raise RuntimeError("seq_emb is empty")
    dim = len(next(iter(seq_emb.values())))
    gid_dim = len(next(iter(gid_centroids.values()))) if gid_centroids else dim
    records: list[dict[str, object]] = []
    db_features: list[np.ndarray] = []
    gid_features: list[np.ndarray] = []
    geometry_features: list[list[float]] = []
    quality_features: list[list[float]] = []
    valid_db: list[bool] = []
    valid_gid: list[bool] = []
    grouped = work.sort_values(["seq", "frame_idx"], kind="mergesort").groupby("seq", sort=True)
    for index, (seq, group) in enumerate(grouped):
        first = group.iloc[0]
        last = group.iloc[-1]
        counts = Counter(int(x) for x in group["current_seq"].tolist())
        db_vecs = [seq_emb[cur_seq] * float(n) for cur_seq, n in counts.items() if cur_seq in seq_emb]
        if db_vecs:
            db_feat = _l2n(np.sum(db_vecs, axis=0))
            valid_db.append(True)
        else:
            db_feat = np.zeros((dim,), dtype=np.float32)
            valid_db.append(False)
        gid_counts = Counter(int(x) for x in group["current_gid"].tolist())
        gid_vecs = [gid_centroids[gid] * float(n) for gid, n in gid_counts.items() if gid in gid_centroids]
        if gid_vecs:
            gid_feat = _l2n(np.sum(gid_vecs, axis=0))
            valid_gid.append(True)
        else:
            gid_feat = np.zeros((gid_dim,), dtype=np.float32)
            valid_gid.append(False)
        x1 = group["x1"].astype(float)
        y1 = group["y1"].astype(float)
        x2 = group["x2"].astype(float)
        y2 = group["y2"].astype(float)
        w = np.maximum(x2.to_numpy(np.float32) - x1.to_numpy(np.float32), 0.0)
        h = np.maximum(y2.to_numpy(np.float32) - y1.to_numpy(np.float32), 0.0)
        cx = ((x1 + x2) * 0.5).to_numpy(np.float32)
        cy = ((y1 + y2) * 0.5).to_numpy(np.float32)
        start = int(group["frame_idx"].min())
        end = int(group["frame_idx"].max())
        duration = max(end - start, 1)
        frames = group["frame_idx"].to_numpy(np.float32)
        vx = float((cx[-1] - cx[0]) / duration)
        vy = float((cy[-1] - cy[0]) / duration)
        areas = w * h
        dominant_seq_frac = max(counts.values()) / max(float(len(group)), 1.0)
        dominant_gid_frac = max(gid_counts.values()) / max(float(len(group)), 1.0)
        records.append(
            {
                "index": int(index),
                "seq": int(seq),
                "tracklet_key": str(first.tracklet_key),
                "video": str(first.video),
                "camera": str(first.camera),
                "start_frame": start,
                "end_frame": end,
                "n_dets": int(len(group)),
                "dominant_current_gid": int(gid_counts.most_common(1)[0][0]),
                "dominant_current_gid_fraction": round(float(dominant_gid_frac), 6),
            }
        )
        db_features.append(db_feat)
        gid_features.append(gid_feat)
        geometry_features.append(
            [
                float(start),
                float(end),
                float(duration),
                float(len(group)),
                float(np.mean(cx)),
                float(np.mean(cy)),
                float(np.mean(w)),
                float(np.mean(h)),
                float(np.median(areas)) if len(areas) else 0.0,
                float(cx[0]),
                float(cy[0]),
                float(cx[-1]),
                float(cy[-1]),
                vx,
                vy,
                float(np.std(cx)) if len(cx) > 1 else 0.0,
                float(np.std(cy)) if len(cy) > 1 else 0.0,
                float(np.mean(np.diff(frames))) if len(frames) > 1 else 0.0,
            ]
        )
        quality_features.append(
            [
                float(len(group)),
                float(group["conf"].astype(float).mean()),
                float(group["conf"].astype(float).min()),
                float(group["conf"].astype(float).max()),
                float(dominant_seq_frac),
                float(dominant_gid_frac),
                float(len(counts)),
                float(len(gid_counts)),
                float(np.median(areas)) if len(areas) else 0.0,
                float(np.quantile(areas, 0.10)) if len(areas) else 0.0,
            ]
        )
    arrays = {
        "features_dbresolve": np.vstack(db_features).astype(np.float32),
        "features_current_gid_centroid": np.vstack(gid_features).astype(np.float32),
        "features_geometry": _standardize(np.asarray(geometry_features, dtype=np.float32)),
        "features_quality": _standardize(np.asarray(quality_features, dtype=np.float32)),
        "valid_dbresolve": np.asarray(valid_db, dtype=bool),
        "valid_current_gid_centroid": np.asarray(valid_gid, dtype=bool),
    }
    return records, arrays


def _write_features(work: pd.DataFrame, seq_emb: dict[int, np.ndarray], gid_centroids: dict[int, np.ndarray], out_path: Path, metadata: dict[str, object]) -> dict[str, object]:
    records, arrays = _tracklet_feature_rows(work, seq_emb, gid_centroids)
    feature_metadata = {
        **metadata,
        "records": records,
        "feature_blocks": {key: list(value.shape) for key, value in arrays.items() if key.startswith("features_")},
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, metadata=json.dumps(feature_metadata, sort_keys=True), **arrays)
    return {
        "feature_out": str(out_path),
        "feature_tracklets": int(len(records)),
        "feature_blocks": feature_metadata["feature_blocks"],
        "valid_dbresolve": int(arrays["valid_dbresolve"].sum()),
        "valid_current_gid_centroid": int(arrays["valid_current_gid_centroid"].sum()),
    }


def _write_current_gid_assignments(work: pd.DataFrame, out_path: Path) -> dict[str, object]:
    rows = []
    for seq, group in work.sort_values(["seq", "frame_idx"], kind="mergesort").groupby("seq", sort=True):
        first = group.iloc[0]
        counts = Counter(int(x) for x in group["current_gid"].tolist())
        gid, n = counts.most_common(1)[0]
        frac = float(n) / max(float(len(group)), 1.0)
        rows.append(
            {
                "seq": int(seq),
                "tracklet_key": str(first.tracklet_key),
                "video": str(first.video),
                "camera": str(first.camera),
                "start_frame": int(group["frame_idx"].min()),
                "end_frame": int(group["frame_idx"].max()),
                "n_dets": int(len(group)),
                "avg_conf": round(float(group["conf"].astype(float).mean()), 6),
                "predicted_global_id": int(gid),
                "component_label": int(gid),
                "component_size": int(0),
                "prediction_confidence": round(frac, 6),
                "decision_status": "weak_current_gid_replay",
                "resolution_status": "weak_current_gid_replay",
                "tracklet_quality_pass": True,
                "area_median": round(float(np.median((group["x2"] - group["x1"]) * (group["y2"] - group["y1"]))), 3),
                "member_centroid_sim_median": 0.0,
                "member_centroid_sim_min": 0.0,
                "nearest_external_centroid_sim": 0.0,
                "centroid_margin": 0.0,
            }
        )
    comp_sizes = Counter(int(row["predicted_global_id"]) for row in rows)
    for row in rows:
        row["component_size"] = int(comp_sizes[int(row["predicted_global_id"])])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "current_gid_assignments_out": str(out_path),
        "current_gid_assignment_rows": int(len(rows)),
        "current_gid_assignment_components": int(len(comp_sizes)),
        "largest_current_gid_component": int(max(comp_sizes.values(), default=0)),
    }


def _current_gid_full_metrics(work: pd.DataFrame, pred_by_video: dict[str, pd.DataFrame], gt_by_video: dict[str, pd.DataFrame]) -> dict[str, object]:
    pred_by_seq = {}
    for seq, group in work.groupby("seq", sort=True):
        gid = Counter(int(x) for x in group["current_gid"].tolist()).most_common(1)[0][0]
        pred_by_seq[int(seq)] = int(gid)
    return _score_full(pred_by_video, gt_by_video, pred_by_seq)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default="gallery_ds1")
    parser.add_argument("--linker", choices=["greedy", "sort", "ocsort"], default="greedy")
    parser.add_argument("--linker-name", default="")
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--high-thresh", type=float, default=0.50)
    parser.add_argument("--new-thresh", type=float, default=0.60)
    parser.add_argument("--delta-t", type=int, default=3)
    parser.add_argument("--vdc", type=float, default=0.20)
    parser.add_argument("--embedding-role", default="resolve")
    parser.add_argument("--limit-videos", default="")
    parser.add_argument("--eval-iou-thr", type=float, default=0.50)
    parser.add_argument("--eval-min-matches", type=int, default=1)
    parser.add_argument("--eval-min-purity", type=float, default=0.0)
    parser.add_argument("--parquet-out", required=True)
    parser.add_argument("--feature-out", required=True)
    parser.add_argument("--current-gid-assignments-out", default="")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    limit_videos = {part.strip() for part in str(args.limit_videos).split(",") if part.strip()} or None
    linker_kind, linker_fn, linker_kwargs = _parse_linker_kwargs(args)
    linker_name = str(args.linker_name).strip() or (
        f"{linker_kind}_iou{int(round(float(args.iou_thresh) * 100)):03d}_gap{int(args.max_gap)}"
    )
    con = _connect(str(args.dbname))
    meta = _load_detections(con, limit_videos)
    current = _load_current_assignments(con)
    seq_emb = _load_seq_embeddings(con, str(args.embedding_role))
    gid_centroids = _current_gid_centroids(current, seq_emb)
    work = meta.merge(current, on="det_id", how="left")
    if work["current_seq"].isna().any():
        missing = int(work["current_seq"].isna().sum())
        raise RuntimeError(f"{missing} detections have no current assignment")
    for col in ("current_seq", "current_gid"):
        work[col] = work[col].astype(np.int64)
    work["current_score"] = work["current_score"].fillna(0.0).astype(float)
    det_to_key = _run_linker(work, linker_fn, linker_kwargs)
    work, key_to_seq = _attach_tracklet_keys(work, det_to_key)
    pred_by_video = _pred_by_video(work)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    if not gt_by_video:
        raise RuntimeError("no DS1 GT videos matched relinked detections")
    gt_by_seq, weight_by_seq, eval_stats = _label_tracklets_for_eval(
        pred_by_video,
        gt_by_video,
        iou_thr=float(args.eval_iou_thr),
        min_matches=int(args.eval_min_matches),
        min_purity=float(args.eval_min_purity),
    )

    metadata = {
        "linker_name": linker_name,
        "linker_kind": linker_kind,
        "linker_kwargs": linker_kwargs,
        "dbname": str(args.dbname),
        "embedding_role": str(args.embedding_role),
        "limit_videos": sorted(limit_videos) if limit_videos else None,
        "n_detections": int(len(work)),
        "n_tracklets": int(len(key_to_seq)),
        "n_current_seq_embeddings": int(len(seq_emb)),
        "n_current_gid_centroids": int(len(gid_centroids)),
        "uses_current_gallery_gid_as_weak_evidence": True,
    }
    parquet_info = _write_parquet(work, gt_by_seq, Path(args.parquet_out))
    feature_info = _write_features(work, seq_emb, gid_centroids, Path(args.feature_out), metadata)
    baseline_info: dict[str, object] = {}
    if args.current_gid_assignments_out:
        baseline_info.update(_write_current_gid_assignments(work, Path(args.current_gid_assignments_out)))
        baseline_info["current_gid_replay_full"] = _current_gid_full_metrics(work, pred_by_video, gt_by_video)

    out = {
        **metadata,
        **parquet_info,
        **feature_info,
        **baseline_info,
        "eval_stats": eval_stats,
        "eval_gt_weight_sum": round(float(sum(weight_by_seq.values())), 3),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
