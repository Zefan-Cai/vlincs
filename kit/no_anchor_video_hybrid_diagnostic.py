#!/usr/bin/env python
"""No-anchor video-level hybrid between two identity resolvers.

This is a diagnostic resolver ensemble.  It builds two no-anchor labelings:

* A: time-aware agglomeration over fused tracklet embeddings.
* B: a loaded no-anchor pair-link model.

B component labels are aligned into A's label space by tracklet-overlap majority
without using identity labels.  A comma-separated video allowlist then selects
which videos should use A; all other videos use aligned B.  GT is used only for
post-hoc metrics and optional full scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_global_id_model import PairModelConfig, _resolve_with_model, _write_assignments
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _time_agglom_resolve,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_global_id_model import PairModelConfig, _resolve_with_model, _write_assignments
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _time_agglom_resolve,
        _with_detection_endpoints,
    )


def _parse_csv(text: str) -> set[str]:
    return {part.strip() for part in str(text).split(",") if part.strip()}


def _align_labels_by_overlap(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    keep_indices: list[int],
) -> tuple[np.ndarray, dict[str, object]]:
    counts: dict[int, Counter] = defaultdict(Counter)
    for idx in keep_indices:
        counts[int(labels_b[idx])][int(labels_a[idx])] += 1

    next_label = int(max(int(labels_a.max()), int(labels_b.max()))) + 1000
    b_to_a: dict[int, int] = {}
    overlap_rows = []
    for b_label, counter in counts.items():
        a_label, overlap = counter.most_common(1)[0]
        total = sum(counter.values())
        b_to_a[int(b_label)] = int(a_label)
        overlap_rows.append(
            {
                "b_label": int(b_label),
                "a_label": int(a_label),
                "overlap": int(overlap),
                "b_kept_size": int(total),
                "purity": round(float(overlap) / max(float(total), 1.0), 6),
            }
        )

    for b_label in set(int(x) for x in labels_b.tolist()):
        if b_label not in b_to_a:
            b_to_a[b_label] = next_label
            next_label += 1

    aligned = np.asarray([b_to_a[int(label)] for label in labels_b.tolist()], dtype=np.int64)
    purities = [float(row["purity"]) for row in overlap_rows]
    info = {
        "b_components": int(len(set(int(x) for x in labels_b.tolist()))),
        "aligned_components": int(len(set(int(x) for x in aligned.tolist()))),
        "mapped_components": int(len(overlap_rows)),
        "new_components": int(next_label - (int(max(int(labels_a.max()), int(labels_b.max()))) + 1000)),
        "mean_overlap_purity": round(float(np.mean(purities)) if purities else 0.0, 6),
        "median_overlap_purity": round(float(np.median(purities)) if purities else 0.0, 6),
    }
    return aligned, info


def _args_proxy(args):
    class OutputArgs:
        output_min_dets = int(args.output_min_dets)
        output_min_conf = float(args.output_min_conf)
        output_min_area = float(args.output_min_area)
        output_min_quality = float(args.output_min_quality)
        output_min_area_by_video = str(args.output_min_area_by_video)
        output_drop_area_quantile = float(args.output_drop_area_quantile)
        output_drop_area_quantile_by_video = str(args.output_drop_area_quantile_by_video)
        output_drop_quality_quantile = float(args.output_drop_quality_quantile)
        output_drop_quality_quantile_by_video = str(args.output_drop_quality_quantile_by_video)
        output_auto_anomaly_admission = bool(args.output_auto_anomaly_admission)
        output_auto_anomaly_metric = str(args.output_auto_anomaly_metric)
        output_auto_anomaly_quantile = float(args.output_auto_anomaly_quantile)
        output_auto_anomaly_area_ratio = float(args.output_auto_anomaly_area_ratio)
        output_auto_anomaly_quality_mad = float(args.output_auto_anomaly_quality_mad)
        output_auto_anomaly_min_video_tracklets = int(args.output_auto_anomaly_min_video_tracklets)
        output_auto_anomaly_max_videos = int(args.output_auto_anomaly_max_videos)

    return OutputArgs()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=0.32)
    ap.add_argument("--pair-model", required=True)
    ap.add_argument("--a-theta", type=float, default=0.014)
    ap.add_argument("--a-top-k", type=int, default=15)
    ap.add_argument("--b-threshold", type=float, default=0.03)
    ap.add_argument("--b-blend", type=float, default=0.5)
    ap.add_argument("--infer-top-k", type=int, default=30)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--max-component-size", type=int, default=120)
    ap.add_argument("--exclude-same", default="camera", choices=["camera", "stream", "video", "none"])
    ap.add_argument("--temporal-bonus", type=float, default=0.005)
    ap.add_argument("--time-window-ms", type=int, default=1000)
    ap.add_argument("--a-videos", default="", help="comma list of videos to keep from A; other videos use overlap-aligned B")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
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
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-score", action="store_true")
    ap.add_argument("--assignments-out", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    emb = _load_feature_npz(
        args.feature_npz,
        records,
        db_emb,
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
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached
    keep_seqs, output_info = _output_keep_seqs(records, _args_proxy(args))
    keep_indices = [idx for idx, record in enumerate(records) if int(record.seq) in keep_seqs]

    cfg_a = ResolveConfig(
        mode="time_agglom",
        theta=float(args.a_theta),
        top_k=int(args.a_top_k),
        min_dets=int(args.min_dets),
        exclude_same=str(args.exclude_same),
        temporal_bonus=float(args.temporal_bonus),
        time_window_ms=int(args.time_window_ms),
    )
    labels_a, info_a = _time_agglom_resolve(records, emb, cfg_a)

    bundle = joblib.load(args.pair_model)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    cfg_b = PairModelConfig(
        infer_top_k=int(args.infer_top_k),
        min_dets=int(args.min_dets),
        max_component_size=int(args.max_component_size),
        exclude_same=str(args.exclude_same),
        pseudo_temporal_bonus=float(args.temporal_bonus),
        pseudo_time_window_ms=int(args.time_window_ms),
        affinity_time_bonus=float(args.temporal_bonus),
    )
    labels_b, info_b = _resolve_with_model(
        records,
        emb,
        model,
        cfg_b,
        threshold=float(args.b_threshold),
        blend=float(args.b_blend),
        solver="agglom",
        runtime_cache={},
        pair_feature_views=None,
    )
    labels_b_aligned, align_info = _align_labels_by_overlap(labels_a, labels_b, keep_indices)

    a_videos = _parse_csv(args.a_videos)
    hybrid = np.asarray(
        [labels_a[idx] if records[idx].video in a_videos else labels_b_aligned[idx] for idx in range(len(records))],
        dtype=np.int64,
    )
    pred_by_seq = _labels_to_seq_map(records, hybrid, keep_seqs=keep_seqs)
    pair = _pair_metrics([record.seq for record in records], pred_by_seq, gt_by_seq, weight_by_seq)
    full = _score_full(pred_by_video, gt_by_video, pred_by_seq) if args.full_score else None
    assignment_info = _write_assignments(args.assignments_out, records, hybrid, keep_seqs=keep_seqs)

    row = {
        "mode": "video_hybrid_overlap_aligned",
        "solver": "time_agglom_plus_pair_model",
        "a_videos": sorted(a_videos),
        "threshold": float(args.b_threshold),
        "blend": float(args.b_blend),
        **pair,
        "components": assignment_info["assignment_components"],
        "largest_component": assignment_info["largest_assignment_component"],
    }
    if full is not None:
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]

    result = {
        "dataset": "ds1",
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "pair_model": args.pair_model,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "a_config": asdict(cfg_a),
        "b_config": asdict(cfg_b),
        "alignment": align_info,
        "base_info": {"a_time_agglom": info_a, "b_pair_model": info_b},
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "top": [row],
        **assignment_info,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"json": args.json, "assignments": args.assignments_out, **row}, sort_keys=True))


if __name__ == "__main__":
    main()
