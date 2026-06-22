#!/usr/bin/env python
"""Oracle upper-bound sweep for alternative detection-to-tracklet linkers.

This diagnostic never trains on identities and never writes the gallery DB.  It
reuses the existing DS1 detections, relinks them into synthetic within-camera
tracklets, assigns each synthetic tracklet its GT-majority identity only for
oracle scoring, and reports the full VLINCS metrics.  Use it to decide whether
the current end-to-end ceiling is limited by the tracklet linker before spending
more time on global-ID resolution.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    import psycopg
except ModuleNotFoundError as exc:  # pragma: no cover - remote venv has psycopg.
    raise RuntimeError("psycopg is required for gallery DB diagnostics") from exc

from vlincs_gallery.eval.score import load_ds1_gt_by_video
from vlincs_gallery.sort_tracker import ocsort_tracklets, sort_tracklets
from vlincs_gallery.tracklets import link_tracklets

try:
    from kit.no_anchor_resolve_sweep import (
        TrackletRecord,
        _label_tracklets_for_eval,
        _load_predictions,
        _score_full,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        TrackletRecord,
        _label_tracklets_for_eval,
        _load_predictions,
        _score_full,
    )


TrackerFn = Callable[..., dict[str, str]]


def _connect(dbname: str):
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "55433"),
        user=os.environ.get("PGUSER", "gallery"),
        password=os.environ.get("PGPASSWORD", "gallery"),
        dbname=dbname,
    )


def _load_detections(con, limit_videos: set[str] | None = None) -> pd.DataFrame:
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
    for col in ("x1", "y1", "x2", "y2", "conf"):
        df[col] = df[col].astype(float)
    return df


def _current_db_records(pred_by_video: dict[str, pd.DataFrame]) -> list[TrackletRecord]:
    rows: list[TrackletRecord] = []
    for video, pred in sorted(pred_by_video.items()):
        if pred.empty:
            continue
        for seq, group in pred.sort_values(["seq", "frame"], kind="mergesort").groupby("seq", sort=False):
            group = group.sort_values("frame", kind="mergesort")
            first = group.iloc[0]
            last = group.iloc[-1]
            rows.append(
                TrackletRecord(
                    seq=int(seq),
                    tracklet_key=f"current:{int(seq)}",
                    video=str(video),
                    camera=str(video).split("_")[-2] if "_" in str(video) else "",
                    start_frame=int(group["frame"].min()),
                    end_frame=int(group["frame"].max()),
                    start_abs_ms=0,
                    end_abs_ms=0,
                    n_dets=int(len(group)),
                    avg_conf=float(group["confidence"].mean()),
                    cx=float(((group["x1"] + group["x2"]) * 0.5).mean()),
                    cy=float(((group["y1"] + group["y2"]) * 0.5).mean()),
                    width=float((group["x2"] - group["x1"]).mean()),
                    height=float((group["y2"] - group["y1"]).mean()),
                    first_cx=float((first.x1 + first.x2) * 0.5),
                    first_cy=float((first.y1 + first.y2) * 0.5),
                    first_width=float(first.x2 - first.x1),
                    first_height=float(first.y2 - first.y1),
                    last_cx=float((last.x1 + last.x2) * 0.5),
                    last_cy=float((last.y1 + last.y2) * 0.5),
                    last_width=float(last.x2 - last.x1),
                    last_height=float(last.y2 - last.y1),
                )
            )
    return rows


def _seq_records_from_meta(meta: pd.DataFrame, det_to_key: dict[str, str]) -> tuple[dict[str, pd.DataFrame], list[TrackletRecord]]:
    key_to_seq: dict[str, int] = {}
    seqs: list[int] = []
    keys: list[str] = []
    for det_id in meta["det_id"].astype(str).tolist():
        key = det_to_key[str(det_id)]
        if key not in key_to_seq:
            key_to_seq[key] = len(key_to_seq) + 1
        keys.append(key)
        seqs.append(key_to_seq[key])

    work = meta.copy()
    work["seq"] = np.asarray(seqs, dtype=np.int64)
    work["tracklet_key"] = keys
    pred = work.rename(columns={"frame_idx": "frame", "conf": "confidence"})
    pred = pred[["video", "frame", "seq", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
    pred_by_video = {
        str(video): group.drop(columns=["video"]).reset_index(drop=True)
        for video, group in pred.groupby("video", sort=True)
    }

    records: list[TrackletRecord] = []
    for seq, group in work.sort_values(["seq", "frame_idx"], kind="mergesort").groupby("seq", sort=False):
        first = group.iloc[0]
        last = group.iloc[-1]
        records.append(
            TrackletRecord(
                seq=int(seq),
                tracklet_key=str(first.tracklet_key),
                video=str(first.video),
                camera=str(first.camera),
                start_frame=int(group["frame_idx"].min()),
                end_frame=int(group["frame_idx"].max()),
                start_abs_ms=int(group["abs_ms"].min()),
                end_abs_ms=int(group["abs_ms"].max()),
                n_dets=int(len(group)),
                avg_conf=float(group["conf"].mean()),
                cx=float(((group["x1"] + group["x2"]) * 0.5).mean()),
                cy=float(((group["y1"] + group["y2"]) * 0.5).mean()),
                width=float((group["x2"] - group["x1"]).mean()),
                height=float((group["y2"] - group["y1"]).mean()),
                first_cx=float((first.x1 + first.x2) * 0.5),
                first_cy=float((first.y1 + first.y2) * 0.5),
                first_width=float(first.x2 - first.x1),
                first_height=float(first.y2 - first.y1),
                last_cx=float((last.x1 + last.x2) * 0.5),
                last_cy=float((last.y1 + last.y2) * 0.5),
                last_width=float(last.x2 - last.x1),
                last_height=float(last.y2 - last.y1),
            )
        )
    return pred_by_video, records


def _run_linker(meta: pd.DataFrame, fn: TrackerFn, kwargs: dict[str, object]) -> tuple[dict[str, pd.DataFrame], list[TrackletRecord]]:
    det_to_key: dict[str, str] = {}
    for video, video_df in meta.groupby("video", sort=True):
        local_assign = fn(video_df.copy(), **kwargs)
        for det_id, key in local_assign.items():
            det_to_key[str(det_id)] = f"{video}:{key}"
    if len(det_to_key) != len(meta):
        missing = set(meta["det_id"].astype(str)) - set(det_to_key)
        raise RuntimeError(f"linker missed {len(missing)} detections; first={next(iter(missing), None)}")
    return _seq_records_from_meta(meta, det_to_key)


def _size_summary(records: list[TrackletRecord]) -> dict[str, object]:
    sizes = np.asarray([record.n_dets for record in records], dtype=np.float32)
    by_video = Counter(record.video for record in records)
    return {
        "n_tracklets": int(len(records)),
        "n_detections": int(sizes.sum()) if sizes.size else 0,
        "mean_dets": round(float(sizes.mean()) if sizes.size else 0.0, 6),
        "median_dets": round(float(np.median(sizes)) if sizes.size else 0.0, 6),
        "p90_dets": round(float(np.quantile(sizes, 0.90)) if sizes.size else 0.0, 6),
        "max_dets": int(sizes.max()) if sizes.size else 0,
        "tracklets_by_video": dict(sorted((str(k), int(v)) for k, v in by_video.items())),
    }


def _metric_summary(metrics: dict[str, object] | None) -> dict[str, object] | None:
    if metrics is None:
        return None
    return {
        key: metrics[key]
        for key in ("idf1", "hota", "assa", "deta", "detre", "detpr", "unmatched_fp")
        if key in metrics
    }


def _evaluate_config(
    name: str,
    pred_by_video: dict[str, pd.DataFrame],
    records: list[TrackletRecord],
    gt_by_video: dict[str, pd.DataFrame],
    *,
    iou_thr: float,
    min_matches: int,
    min_purity: float,
    skip_full: bool,
    only_labeled_full: bool,
) -> dict[str, object]:
    gt_by_seq, weight_by_seq, eval_stats = _label_tracklets_for_eval(
        pred_by_video,
        gt_by_video,
        iou_thr=iou_thr,
        min_matches=min_matches,
        min_purity=min_purity,
    )
    labeled_oracle = {int(seq): int(gid) for seq, gid in gt_by_seq.items()}
    plus_singletons = dict(labeled_oracle)
    next_singleton = 90_000_000
    for record in records:
        seq = int(record.seq)
        if seq not in plus_singletons:
            plus_singletons[seq] = next_singleton
            next_singleton += 1
    unique = {int(record.seq): 80_000_000 + i for i, record in enumerate(records)}

    labeled_full = None if skip_full else _score_full(pred_by_video, gt_by_video, labeled_oracle)
    plus_singletons_full = None if skip_full or only_labeled_full else _score_full(pred_by_video, gt_by_video, plus_singletons)
    unique_full = None if skip_full or only_labeled_full else _score_full(pred_by_video, gt_by_video, unique)
    return {
        "config": name,
        "tracklet_summary": _size_summary(records),
        "eval_stats": eval_stats,
        "oracle_labeled_full": labeled_full,
        "oracle_plus_singletons_full": plus_singletons_full,
        "unique_tracklet_full": unique_full,
        "oracle_labeled_summary": _metric_summary(labeled_full),
        "oracle_plus_singletons_summary": _metric_summary(plus_singletons_full),
        "unique_summary": _metric_summary(unique_full),
        "n_gt_labeled_seqs": int(len(gt_by_seq)),
        "gt_pair_mass": round(float(sum(weight_by_seq.values())), 3),
        "uses_gt_for_analysis_only": True,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
    }


def _preset_configs(preset: str) -> list[tuple[str, str, dict[str, object]]]:
    tiny = [
        ("current_db", "current", {}),
        ("greedy_iou050_gap10", "greedy", {"iou_thresh": 0.50, "max_gap": 10}),
        ("sort_iou030_gap30", "sort", {"iou_thresh": 0.30, "max_gap": 30, "high_thresh": 0.50, "new_thresh": 0.60}),
        ("ocsort_iou030_gap30", "ocsort", {"iou_thresh": 0.30, "max_gap": 30, "new_thresh": 0.60, "delta_t": 3, "vdc": 0.20}),
    ]
    if preset == "tiny":
        return tiny
    default = tiny + [
        ("greedy_iou040_gap15", "greedy", {"iou_thresh": 0.40, "max_gap": 15}),
        ("greedy_iou030_gap30", "greedy", {"iou_thresh": 0.30, "max_gap": 30}),
        ("sort_iou020_gap30", "sort", {"iou_thresh": 0.20, "max_gap": 30, "high_thresh": 0.45, "new_thresh": 0.55}),
        ("sort_iou040_gap30", "sort", {"iou_thresh": 0.40, "max_gap": 30, "high_thresh": 0.50, "new_thresh": 0.60}),
        ("ocsort_iou020_gap30", "ocsort", {"iou_thresh": 0.20, "max_gap": 30, "new_thresh": 0.55, "delta_t": 3, "vdc": 0.20}),
        ("ocsort_iou040_gap30", "ocsort", {"iou_thresh": 0.40, "max_gap": 30, "new_thresh": 0.60, "delta_t": 3, "vdc": 0.20}),
    ]
    if preset == "default":
        return default
    return default + [
        ("greedy_iou020_gap60", "greedy", {"iou_thresh": 0.20, "max_gap": 60}),
        ("greedy_iou060_gap10", "greedy", {"iou_thresh": 0.60, "max_gap": 10}),
        ("sort_iou020_gap60", "sort", {"iou_thresh": 0.20, "max_gap": 60, "high_thresh": 0.45, "new_thresh": 0.55}),
        ("sort_iou050_gap15", "sort", {"iou_thresh": 0.50, "max_gap": 15, "high_thresh": 0.50, "new_thresh": 0.60}),
        ("ocsort_iou020_gap60", "ocsort", {"iou_thresh": 0.20, "max_gap": 60, "new_thresh": 0.55, "delta_t": 5, "vdc": 0.20}),
        ("ocsort_iou050_gap15", "ocsort", {"iou_thresh": 0.50, "max_gap": 15, "new_thresh": 0.60, "delta_t": 3, "vdc": 0.20}),
    ]


def _parse_configs(text: str) -> list[tuple[str, str, dict[str, object]]]:
    """Parse explicit configs as name=kind:k=v,k=v entries."""
    out: list[tuple[str, str, dict[str, object]]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        name, sep, rest = item.partition("=")
        if not sep:
            raise ValueError(f"bad config {item!r}; expected name=kind:k=v,k=v")
        kind, _colon, arg_text = rest.partition(":")
        kwargs: dict[str, object] = {}
        for part in arg_text.split(","):
            part = part.strip()
            if not part:
                continue
            key, eq, value = part.partition("=")
            if not eq:
                raise ValueError(f"bad config arg {part!r}; expected k=v")
            try:
                if "." in value:
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = int(value)
            except ValueError:
                kwargs[key] = value
        out.append((name.strip(), kind.strip(), kwargs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--json", default="/mnt/localssd/vlincs_reid_runs/relink_tracklet_oracle_sweep_20260617.json")
    ap.add_argument("--preset", choices=["tiny", "default", "wide"], default="tiny")
    ap.add_argument("--configs", default="", help="Optional explicit semicolon-separated config list.")
    ap.add_argument("--limit-videos", default="", help="Comma-separated video stems for a quick slice.")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--min-matches", type=int, default=1)
    ap.add_argument("--min-purity", type=float, default=0.0)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument(
        "--only-labeled-full",
        action="store_true",
        help="Score only the oracle-labeled submission; skip plus-singletons and unique-tracklet full scores.",
    )
    args = ap.parse_args()

    limit_videos = {part.strip() for part in args.limit_videos.split(",") if part.strip()} or None
    con = _connect(args.dbname)
    meta = _load_detections(con, limit_videos)
    current_pred = _load_predictions(con)
    if limit_videos:
        current_pred = {video: df for video, df in current_pred.items() if video in limit_videos}
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in set(meta["video"])}
    if not gt_by_video:
        raise RuntimeError("no DS1 GT videos matched loaded detections")

    configs = _parse_configs(args.configs) if args.configs else _preset_configs(args.preset)
    fn_by_kind: dict[str, TrackerFn] = {
        "greedy": link_tracklets,
        "sort": sort_tracklets,
        "ocsort": ocsort_tracklets,
    }

    results: list[dict[str, object]] = []
    for name, kind, kwargs in configs:
        print(json.dumps({"event": "start_config", "config": name, "kind": kind, "kwargs": kwargs}), flush=True)
        if kind == "current":
            pred_by_video = current_pred
            records = _current_db_records(pred_by_video)
        else:
            if kind not in fn_by_kind:
                raise ValueError(f"unknown tracker kind {kind!r}")
            pred_by_video, records = _run_linker(meta, fn_by_kind[kind], kwargs)
        result = _evaluate_config(
            name,
            pred_by_video,
            records,
            gt_by_video,
            iou_thr=args.iou_thr,
            min_matches=args.min_matches,
            min_purity=args.min_purity,
            skip_full=args.skip_full,
            only_labeled_full=args.only_labeled_full,
        )
        result["tracker_kind"] = kind
        result["tracker_kwargs"] = kwargs
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "done_config",
                    "config": name,
                    "tracklets": result["tracklet_summary"]["n_tracklets"],
                    "labeled": result["n_gt_labeled_seqs"],
                    "matched_detection_fraction": result["eval_stats"]["matched_detection_fraction"],
                    "oracle_labeled": result["oracle_labeled_summary"],
                    "oracle_plus_singletons": result["oracle_plus_singletons_summary"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def sort_key(row: dict[str, object]) -> float:
        summary = row.get("oracle_labeled_summary") or {}
        if isinstance(summary, dict):
            return float(summary.get("idf1", 0.0))
        return 0.0

    ranked = sorted(results, key=sort_key, reverse=True)
    out_obj = {
        "dbname": args.dbname,
        "preset": args.preset,
        "limit_videos": sorted(limit_videos) if limit_videos else None,
        "n_detections": int(len(meta)),
        "n_gt_rows": int(sum(len(df) for df in gt_by_video.values())),
        "configs": [name for name, _kind, _kwargs in configs],
        "topline": [
            {
                "rank": i + 1,
                "config": row["config"],
                "tracker_kind": row["tracker_kind"],
                "tracker_kwargs": row["tracker_kwargs"],
                "tracklets": row["tracklet_summary"]["n_tracklets"],
                "labeled_tracklets": row["n_gt_labeled_seqs"],
                "matched_detection_fraction": row["eval_stats"]["matched_detection_fraction"],
                "oracle_labeled": row["oracle_labeled_summary"],
                "oracle_plus_singletons": row["oracle_plus_singletons_summary"],
                "unique": row["unique_summary"],
            }
            for i, row in enumerate(ranked)
        ],
        "results": results,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_obj, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"json": str(out), "best": out_obj["topline"][0] if out_obj["topline"] else None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
