#!/usr/bin/env python
"""No-anchor geo-time support sweep for VLINCS tracklet identity resolution.

This extends the current time-aware agglomeration with a small world-geometry
support term from detection lat/lon.  It uses no identity labels for graph
construction; GT is loaded only after prediction for pair/full metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from vlincs_gallery.eval.score import load_ds1_gt_by_video
from vlincs_gallery.resolve import _knn_sparse_affinity

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _group_codes,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _time_support_matrix,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _group_codes,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _time_support_matrix,
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _load_tracklet_geo(con) -> dict[int, dict[str, float]]:
    with con.cursor() as cur:
        cur.execute(
            """WITH ordered AS (
                   SELECT a.seq, d.frame_idx, d.abs_ms, d.lat, d.lon,
                          ROW_NUMBER() OVER (PARTITION BY a.seq ORDER BY d.frame_idx ASC, d.det_id ASC) AS rn_first,
                          ROW_NUMBER() OVER (PARTITION BY a.seq ORDER BY d.frame_idx DESC, d.det_id DESC) AS rn_last
                   FROM assignments a JOIN detections d ON d.det_id = a.det_id
                   WHERE d.lat IS NOT NULL AND d.lon IS NOT NULL
               ),
               agg AS (
                   SELECT seq,
                          AVG(lat) AS mean_lat,
                          AVG(lon) AS mean_lon,
                          MIN(abs_ms) AS start_abs_ms,
                          MAX(abs_ms) AS end_abs_ms,
                          COUNT(*) AS n_geo
                   FROM ordered
                   GROUP BY seq
               ),
               firsts AS (
                   SELECT seq, lat AS first_lat, lon AS first_lon
                   FROM ordered
                   WHERE rn_first = 1
               ),
               lasts AS (
                   SELECT seq, lat AS last_lat, lon AS last_lon
                   FROM ordered
                   WHERE rn_last = 1
               )
               SELECT agg.seq, mean_lat, mean_lon, first_lat, first_lon,
                      last_lat, last_lon, start_abs_ms, end_abs_ms, n_geo
               FROM agg
               JOIN firsts USING (seq)
               JOIN lasts USING (seq)
               ORDER BY agg.seq"""
        )
        rows = cur.fetchall()
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        out[int(row[0])] = {
            "mean_lat": float(row[1]),
            "mean_lon": float(row[2]),
            "first_lat": float(row[3]),
            "first_lon": float(row[4]),
            "last_lat": float(row[5]),
            "last_lon": float(row[6]),
            "start_abs_ms": float(row[7] or 0.0),
            "end_abs_ms": float(row[8] or row[7] or 0.0),
            "n_geo": float(row[9] or 0.0),
        }
    return out


def _meters_matrix(lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    k = 111320.0
    lat0 = 38.9209
    dy = (lat_a - lat_b) * k
    dx = (lon_a - lon_b) * k * np.cos(np.deg2rad(lat0))
    return np.hypot(dx, dy).astype(np.float32)


def _geo_support_matrix(
    records,
    geo_by_seq: dict[int, dict[str, float]],
    indices: list[int],
    *,
    radius_m: float,
    window_ms: int,
    max_speed_mps: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n = len(indices)
    support = np.zeros((n, n), dtype=np.float32)
    if n == 0:
        return support, {"geo_valid_tracklets": 0, "geo_positive_pairs": 0}
    starts = np.asarray([records[i].start_abs_ms for i in indices], dtype=np.float64)
    ends = np.asarray([records[i].end_abs_ms for i in indices], dtype=np.float64)
    mean_lat = np.full(n, np.nan, dtype=np.float64)
    mean_lon = np.full(n, np.nan, dtype=np.float64)
    first_lat = np.full(n, np.nan, dtype=np.float64)
    first_lon = np.full(n, np.nan, dtype=np.float64)
    last_lat = np.full(n, np.nan, dtype=np.float64)
    last_lon = np.full(n, np.nan, dtype=np.float64)
    for pos, idx in enumerate(indices):
        row = geo_by_seq.get(int(records[idx].seq))
        if not row:
            continue
        mean_lat[pos] = row["mean_lat"]
        mean_lon[pos] = row["mean_lon"]
        first_lat[pos] = row["first_lat"]
        first_lon[pos] = row["first_lon"]
        last_lat[pos] = row["last_lat"]
        last_lon[pos] = row["last_lon"]
    valid = np.isfinite(mean_lat) & np.isfinite(mean_lon)
    valid_count = int(np.count_nonzero(valid))
    window_ms = int(window_ms)
    radius_m = max(float(radius_m), 1.0e-6)
    max_speed_mps = float(max_speed_mps)

    valid_pair = valid[:, None] & valid[None, :]
    gap = np.maximum(0.0, np.maximum(starts[:, None] - ends[None, :], starts[None, :] - ends[:, None])).astype(np.float32)
    a_before_b = ends[:, None] <= starts[None, :]
    b_before_a = ends[None, :] <= starts[:, None]
    dist_ab = _meters_matrix(last_lat[:, None], last_lon[:, None], first_lat[None, :], first_lon[None, :])
    dist_ba = _meters_matrix(first_lat[:, None], first_lon[:, None], last_lat[None, :], last_lon[None, :])
    dist_overlap = _meters_matrix(mean_lat[:, None], mean_lon[:, None], mean_lat[None, :], mean_lon[None, :])
    dist = np.where(a_before_b, dist_ab, np.where(b_before_a, dist_ba, dist_overlap)).astype(np.float32)
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(valid_pair, False)
    near_time = gap <= float(window_ms)
    overlap = gap <= 1.0
    overlap_ok = (~overlap) | (dist <= radius_m)
    if max_speed_mps > 0.0:
        speed = dist / np.maximum(gap / 1000.0, 1.0e-6)
        speed_ok = overlap | (speed <= max_speed_mps)
        speed_term = np.where(overlap, 1.0, np.exp(-speed / max(max_speed_mps, 1.0))).astype(np.float32)
    else:
        speed_ok = np.ones_like(valid_pair, dtype=bool)
        speed_term = np.ones_like(dist, dtype=np.float32)
    mask = valid_pair & near_time & overlap_ok & speed_ok & np.isfinite(dist)
    support = (
        np.exp(-dist / radius_m).astype(np.float32)
        * np.exp(-gap / max(float(window_ms), 1.0)).astype(np.float32)
        * speed_term
    )
    support[~mask] = 0.0
    np.fill_diagonal(support, 0.0)
    upper = np.triu(np.ones_like(mask, dtype=bool), k=1)
    positive = int(np.count_nonzero((support > 0) & upper))
    speed_rejected = int(np.count_nonzero(valid_pair & near_time & overlap_ok & (~speed_ok) & upper))
    far_overlap_rejected = int(np.count_nonzero(valid_pair & near_time & overlap & (~overlap_ok) & upper))
    return support, {
        "geo_valid_tracklets": valid_count,
        "geo_positive_pairs": positive,
        "geo_speed_rejected": speed_rejected,
        "geo_far_overlap_rejected": far_overlap_rejected,
    }


def _geo_time_resolve(records, emb: np.ndarray, geo_by_seq, cfg: ResolveConfig, args, geo_bonus: float, radius_m: float, geo_window_ms: int, max_speed_mps: float):
    keep = [i for i, record in enumerate(records) if record.n_dets >= cfg.min_dets]
    labels = np.full(len(records), -1, dtype=np.int64)
    if keep:
        group_codes = _group_codes(records, cfg.exclude_same, keep)
        x = _l2n(emb[np.asarray(keep, dtype=np.int64)].astype(np.float32))
        S = (x @ x.T).astype(np.float32)
        time_support = _time_support_matrix(records, keep, int(cfg.time_window_ms))
        if cfg.exclude_same != "none":
            same = group_codes[:, None] == group_codes[None, :]
            time_support[same] = 0.0
        S += float(cfg.temporal_bonus) * time_support
        geo_support, geo_info = _geo_support_matrix(
            records,
            geo_by_seq,
            keep,
            radius_m=float(radius_m),
            window_ms=int(geo_window_ms),
            max_speed_mps=float(max_speed_mps),
        )
        if cfg.exclude_same != "none":
            geo_support[group_codes[:, None] == group_codes[None, :]] = 0.0
        S += float(geo_bonus) * geo_support
        if cfg.exclude_same != "none":
            S[group_codes[:, None] == group_codes[None, :]] = -2.0
        np.fill_diagonal(S, -2.0)
        A, n_edges = _knn_sparse_affinity(S, int(cfg.top_k))
        D = 1.0 - A
        np.clip(D, 0.0, None, out=D)
        clustered = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(1.0 - cfg.theta),
            metric="precomputed",
            linkage="average",
        ).fit_predict(D)
        _, clustered = np.unique(clustered, return_inverse=True)
        for k, i in enumerate(keep):
            labels[i] = int(clustered[k])
        next_label = int(clustered.max()) + 1 if clustered.size else 0
        positive_time_edges = int(np.count_nonzero(np.triu(time_support > 0, k=1)))
    else:
        next_label = 0
        n_edges = 0
        positive_time_edges = 0
        geo_info = {"geo_valid_tracklets": 0, "geo_positive_pairs": 0, "geo_speed_rejected": 0, "geo_far_overlap_rejected": 0}
    for i in range(len(records)):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
    return labels, {
        "candidate_edges": int(n_edges),
        "time_candidate_pairs": int(positive_time_edges),
        "geo_bonus": float(geo_bonus),
        "geo_radius_m": float(radius_m),
        "geo_window_ms": int(geo_window_ms),
        "geo_max_speed_mps": float(max_speed_mps),
        **geo_info,
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "uses_ground_truth": False,
    }


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    keys = sorted(key for row in rows for key, value in row.items() if not isinstance(value, (dict, list, tuple)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.014)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--exclude-same", default="camera")
    ap.add_argument("--temporal-bonus", type=float, default=0.005)
    ap.add_argument("--time-window-ms", type=int, default=1000)
    ap.add_argument("--geo-bonuses", default="0.0,0.002,0.004,0.006,0.008")
    ap.add_argument("--geo-radii-m", default="1.5,3,6,12")
    ap.add_argument("--geo-window-ms", default="500,1000,2000")
    ap.add_argument("--geo-max-speed-mps", default="1.5,3,5")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, emb = _load_tracklets(con, args.role)
    if args.feature_npz:
        emb = _load_feature_npz(
            args.feature_npz,
            records,
            emb,
            concat_db=bool(args.concat_db_embedding),
            db_weight=float(args.db_weight),
            feature_weight=float(args.feature_weight),
        )
    geo_by_seq = _load_tracklet_geo(con)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": 0.5,
        "min_matches": 1,
        "min_purity": 0.0,
        "n_tracklets": len(records),
        "prediction_rows": int(sum(len(value) for value in pred_by_video.values())),
        "gt_rows": int(sum(len(value) for value in gt_by_video.values())),
    }
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached
    keep_seqs, output_info = _output_keep_seqs(records, args)
    cfg = ResolveConfig(
        mode="geo_time_agglom",
        theta=float(args.theta),
        top_k=int(args.top_k),
        min_dets=int(args.min_dets),
        exclude_same=str(args.exclude_same),
        temporal_bonus=float(args.temporal_bonus),
        time_window_ms=int(args.time_window_ms),
    )

    rows: list[dict[str, object]] = []
    total = (
        len(_parse_floats(args.geo_bonuses))
        * len(_parse_floats(args.geo_radii_m))
        * len(_parse_ints(args.geo_window_ms))
        * len(_parse_floats(args.geo_max_speed_mps))
    )
    progress = 0
    for geo_bonus in _parse_floats(args.geo_bonuses):
        for radius_m in _parse_floats(args.geo_radii_m):
            for geo_window_ms in _parse_ints(args.geo_window_ms):
                for max_speed_mps in _parse_floats(args.geo_max_speed_mps):
                    progress += 1
                    labels, info = _geo_time_resolve(records, emb, geo_by_seq, cfg, args, geo_bonus, radius_m, geo_window_ms, max_speed_mps)
                    pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
                    pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
                    row = {
                        "progress": int(progress),
                        "total": int(total),
                        **asdict(cfg),
                        **info,
                        **{key: value for key, value in output_info.items() if not isinstance(value, dict)},
                        **pair,
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "stage": "config",
                                "progress": progress,
                                "total": total,
                                "geo_bonus": geo_bonus,
                                "geo_radius_m": radius_m,
                                "geo_window_ms": geo_window_ms,
                                "geo_max_speed_mps": max_speed_mps,
                                "tracklet_pair_f1": pair["tracklet_pair_f1"],
                                "tracklet_pair_precision": pair["tracklet_pair_precision"],
                                "tracklet_pair_recall": pair["tracklet_pair_recall"],
                                "components": info["components"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    rows.sort(
        key=lambda row: (
            float(row.get(args.sort_key, 0.0)),
            float(row.get("tracklet_pair_f1", 0.0)),
            float(row.get("tracklet_pair_recall", 0.0)),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        labels, _info = _geo_time_resolve(
            records,
            emb,
            geo_by_seq,
            cfg,
            args,
            float(row["geo_bonus"]),
            float(row["geo_radius_m"]),
            int(row["geo_window_ms"]),
            float(row["geo_max_speed_mps"]),
        )
        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "full_idf1": full["idf1"], "row": row}, sort_keys=True), flush=True)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "base_config": asdict(cfg),
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "geo_tracklets": int(len(geo_by_seq)),
        "n_configs": int(total),
        "sort_key": str(args.sort_key),
        "top": rows[: max(50, int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
