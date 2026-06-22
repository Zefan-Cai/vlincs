#!/usr/bin/env python
"""Export sparse no-anchor target-agglomeration assignment sources for DS1.

This is the full-data counterpart of the sample ``target_d*_c*`` sources.  It
clusters tracklet embeddings without anchors or identity labels, then applies
an unsupervised output-admission rule (min detections/confidence/area/quality)
to create sparse precision-overlay assignment CSVs.  Ground truth is loaded
only after predictions are fixed, for diagnostics and optional scoring.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.export_no_anchor_time_agglom_model import _component_confidence, _write_assignments
    from kit.no_anchor_resolve_sweep import (
        _cache_eval_labels,
        _connect,
        _label_tracklets_for_eval,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from export_no_anchor_time_agglom_model import _component_confidence, _write_assignments
    from no_anchor_resolve_sweep import (
        _cache_eval_labels,
        _connect,
        _label_tracklets_for_eval,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _slug_float(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return re.sub(r"[^0-9a-zA-Z]+", "p", text)


def _target_agglom_labels(emb: np.ndarray, n_clusters: int) -> np.ndarray:
    n_clusters = min(max(int(n_clusters), 1), int(len(emb)))
    sim = np.clip(emb @ emb.T, -1.0, 1.0).astype(np.float32)
    dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float32)
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    ).fit_predict(dist)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--target-clusters", default="640,960,1280")
    ap.add_argument("--output-min-dets-list", default="1,5,10")
    ap.add_argument("--output-min-conf-list", default="0.5")
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--output-drop-area-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-area-quantile-by-video", default="")
    ap.add_argument("--output-drop-quality-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-quality-quantile-by-video", default="")
    ap.add_argument("--output-auto-anomaly-admission", action="store_true")
    ap.add_argument("--output-auto-anomaly-metric", default="quality", choices=["area", "quality", "both"])
    ap.add_argument("--output-auto-anomaly-quantile", type=float, default=0.75)
    ap.add_argument("--output-auto-anomaly-area-ratio", type=float, default=0.60)
    ap.add_argument("--output-auto-anomaly-quality-mad", type=float, default=1.0)
    ap.add_argument("--output-auto-anomaly-min-video-tracklets", type=int, default=20)
    ap.add_argument("--output-auto-anomaly-max-videos", type=int, default=3)
    ap.add_argument("--confidence-top-k", type=int, default=30)
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--assignment-offset", type=int, default=30_000_000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="no_anchor_target_agglom")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--json", required=True)
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
    cached = _load_eval_label_cache(args.eval_cache, expected) if args.eval_cache else None
    if cached is None:
        gt_by_seq, weight_by_seq, eval_stats = _label_tracklets_for_eval(
            pred_by_video,
            gt_by_video,
            iou_thr=0.5,
            min_matches=1,
            min_purity=0.0,
        )
        eval_stats.update(expected)
        if args.eval_cache:
            _cache_eval_labels(args.eval_cache, gt_by_seq, weight_by_seq, eval_stats)
    else:
        gt_by_seq, weight_by_seq, eval_stats = cached

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    pred_cache: dict[tuple[int, int, float], dict[int, int]] = {}
    keep_cache: dict[tuple[int, float], set[int]] = {}
    labels_by_target: dict[int, np.ndarray] = {}
    output_info_by_admission: dict[tuple[int, float], dict[str, object]] = {}
    seqs = [int(record.seq) for record in records]

    for target_clusters in _parse_ints(args.target_clusters):
        labels = _target_agglom_labels(emb, int(target_clusters))
        labels_by_target[int(target_clusters)] = labels
        component_sizes = defaultdict(int)
        for label in labels.tolist():
            component_sizes[int(label)] += 1
        resolve_info = {
            "mode": "target_agglom",
            "target_clusters": int(target_clusters),
            "components": int(len(component_sizes)),
            "largest_component": int(max(component_sizes.values(), default=0)),
            "uses_ground_truth": False,
        }
        print(json.dumps({"stage": "clustered", **resolve_info}, sort_keys=True), flush=True)

        for output_min_dets in _parse_ints(args.output_min_dets_list):
            for output_min_conf in _parse_floats(args.output_min_conf_list):
                admission_key = (int(output_min_dets), float(output_min_conf))
                if admission_key not in keep_cache:
                    admission_args = SimpleNamespace(**vars(args))
                    admission_args.output_min_dets = int(output_min_dets)
                    admission_args.output_min_conf = float(output_min_conf)
                    keep_seqs, output_info = _output_keep_seqs(records, admission_args)
                    keep_cache[admission_key] = keep_seqs
                    output_info_by_admission[admission_key] = output_info
                keep_seqs = keep_cache[admission_key]
                output_info = output_info_by_admission[admission_key]

                source_name = (
                    f"target_t{int(target_clusters)}"
                    f"_d{int(output_min_dets)}"
                    f"_c{_slug_float(float(output_min_conf))}"
                )
                assignments_out = out_dir / f"{args.prefix}_{source_name}_assignments.csv"
                meta = _component_confidence(records, emb, labels, keep_seqs, int(args.confidence_top_k))
                assignment_info = _write_assignments(
                    str(assignments_out),
                    records,
                    labels,
                    keep_seqs,
                    meta,
                    int(args.assignment_offset) + int(target_clusters) * 100_000,
                )
                pred_by_seq = _labels_to_seq_map(
                    records,
                    labels,
                    offset=int(args.assignment_offset) + int(target_clusters) * 100_000,
                    keep_seqs=keep_seqs,
                )
                pred_cache[(int(target_clusters), int(output_min_dets), float(output_min_conf))] = pred_by_seq
                pair = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
                row = {
                    "source_name": source_name,
                    **resolve_info,
                    **{key: value for key, value in output_info.items() if not isinstance(value, dict)},
                    **pair,
                    **assignment_info,
                    "uses_anchors": False,
                    "uses_gt_for_training_or_anchors": False,
                    "uses_gt_for_evaluation_only": True,
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "stage": "exported",
                            "source_name": source_name,
                            "assignment_rows": row["assignment_rows"],
                            "tracklet_pair_f1": row["tracklet_pair_f1"],
                            "tracklet_pair_precision": row["tracklet_pair_precision"],
                            "tracklet_pair_recall": row["tracklet_pair_recall"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    def sort_score(row: dict[str, object]) -> float:
        try:
            return float(row.get(str(args.sort_key), 0.0))
        except (TypeError, ValueError):
            return 0.0

    rows.sort(key=sort_score, reverse=True)
    for rank, row in enumerate(rows[: max(0, int(args.full_top_n))], start=1):
        key = (int(row["target_clusters"]), int(row["output_min_dets"]), float(row["output_min_conf"]))
        full = _score_full(pred_by_video, gt_by_video, pred_cache[key])
        row.update({f"full_{metric}": value for metric, value in full.items() if metric != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full_scored", "full_rank": rank, "source_name": row["source_name"], "full": full}, sort_keys=True), flush=True)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "n_tracklets": int(len(records)),
        "target_clusters": _parse_ints(args.target_clusters),
        "output_min_dets_list": _parse_ints(args.output_min_dets_list),
        "output_min_conf_list": _parse_floats(args.output_min_conf_list),
        "eval_stats": eval_stats,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "top": rows,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
