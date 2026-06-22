#!/usr/bin/env python
"""No-anchor scoped resolver sweep for VLINCS tracklet global IDs.

This driver tests whether identity resolution should be scoped by one camera
video, by a synchronized scene group such as Tc6/Tc8, or by the whole dataset.
It uses only tracklet evidence for resolving identities; GT is loaded only for
evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _cache_eval_labels,
        _connect,
        _graph_resolve,
        _label_tracklets_for_eval,
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
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _cache_eval_labels,
        _connect,
        _graph_resolve,
        _label_tracklets_for_eval,
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


_TC_RE = re.compile(r"_(Tc\d+)$")


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _scope_key(video: str, scope: str) -> str:
    if scope == "all":
        return "all"
    if scope == "video":
        return video
    if scope == "scene":
        match = _TC_RE.search(video)
        return match.group(1) if match else video
    raise ValueError(f"unknown scope: {scope}")


def _subset_config(cfg: ResolveConfig, mode: str) -> ResolveConfig:
    if mode == "graph":
        return replace(cfg, mode="graph")
    if mode == "time_agglom":
        return replace(cfg, mode="time_agglom")
    raise ValueError(f"unknown mode: {mode}")


def _resolve_scoped(records, emb, cfg: ResolveConfig, *, scope: str, mode: str):
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[_scope_key(record.video, scope)].append(idx)

    labels = [-1] * len(records)
    next_label = 0
    info = {
        "scope_groups": len(groups),
        "scope_mode": scope,
        "scope_resolver": mode,
        "scope_largest_group": max((len(v) for v in groups.values()), default=0),
        "scope_total_candidate_edges": 0,
        "scope_total_components": 0,
    }

    sub_cfg = _subset_config(cfg, mode)
    for _key, indices in sorted(groups.items()):
        sub_records = [records[i] for i in indices]
        sub_emb = emb[indices]
        if mode == "graph":
            sub_labels, sub_info = _graph_resolve(sub_records, sub_emb, sub_cfg, cache=None)
        else:
            sub_labels, sub_info = _time_agglom_resolve(sub_records, sub_emb, sub_cfg)
        label_map: dict[int, int] = {}
        for local_pos, global_idx in enumerate(indices):
            local_label = int(sub_labels[local_pos])
            if local_label not in label_map:
                label_map[local_label] = next_label
                next_label += 1
            labels[global_idx] = label_map[local_label]
        info["scope_total_candidate_edges"] += int(sub_info.get("candidate_edges", 0))
        info["scope_total_components"] += int(sub_info.get("components", 0))

    return labels, info


def _admission_namespace(args):
    return SimpleNamespace(
        output_min_dets=int(args.output_min_dets),
        output_min_conf=float(args.output_min_conf),
        output_min_area=float(args.output_min_area),
        output_min_quality=float(args.output_min_quality),
        output_min_area_by_video=str(args.output_min_area_by_video),
        output_drop_area_quantile=0.0,
        output_drop_area_quantile_by_video="",
        output_drop_quality_quantile=0.0,
        output_drop_quality_quantile_by_video="",
        output_auto_anomaly_admission=False,
        output_auto_anomaly_metric="quality",
        output_auto_anomaly_quantile=0.75,
        output_auto_anomaly_area_ratio=0.60,
        output_auto_anomaly_quality_mad=1.0,
        output_auto_anomaly_min_video_tracklets=20,
        output_auto_anomaly_max_videos=3,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--scopes", default="scene,video,all")
    ap.add_argument("--modes", default="graph,time_agglom")
    ap.add_argument("--thetas", default="0.014,0.015,0.016,0.017,0.018")
    ap.add_argument("--top-ks", default="15,30")
    ap.add_argument("--min-dets", default="10")
    ap.add_argument("--cross-thrs", default="0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--intra-thrs", default="0.45,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--max-gaps", default="120,300,600,1200")
    ap.add_argument("--max-component-sizes", default="40,80,160")
    ap.add_argument("--temporal-bonuses", default="0.0,0.005,0.02")
    ap.add_argument("--time-windows-ms", default="500,1000")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--full-top-n", type=int, default=8)
    ap.add_argument("--json-top-n", type=int, default=50)
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

    keep, output_info = _output_keep_seqs(records, _admission_namespace(args))
    seqs = [record.seq for record in records]
    rows = []
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    scopes = [part.strip() for part in args.scopes.split(",") if part.strip()]
    for scope in scopes:
        for mode in modes:
            if mode == "graph":
                for cross_thr in _parse_floats(args.cross_thrs):
                    for intra_thr in _parse_floats(args.intra_thrs):
                        for top_k in _parse_ints(args.top_ks):
                            for min_dets in _parse_ints(args.min_dets):
                                for max_gap in _parse_ints(args.max_gaps):
                                    for max_component_size in _parse_ints(args.max_component_sizes):
                                        for temporal_bonus in _parse_floats(args.temporal_bonuses):
                                            cfg = ResolveConfig(
                                                mode=mode,
                                                top_k=top_k,
                                                min_dets=min_dets,
                                                cross_thr=cross_thr,
                                                intra_thr=intra_thr,
                                                max_gap=max_gap,
                                                max_component_size=max_component_size,
                                                temporal_bonus=temporal_bonus,
                                            )
                                            labels, info = _resolve_scoped(records, emb, cfg, scope=scope, mode=mode)
                                            pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=keep)
                                            metrics = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
                                            rows.append({"scope": scope, **asdict(cfg), **info, **metrics})
            elif mode == "time_agglom":
                for theta in _parse_floats(args.thetas):
                    for top_k in _parse_ints(args.top_ks):
                        for min_dets in _parse_ints(args.min_dets):
                            for temporal_bonus in _parse_floats(args.temporal_bonuses):
                                for time_window_ms in _parse_ints(args.time_windows_ms):
                                    cfg = ResolveConfig(
                                        mode=mode,
                                        theta=theta,
                                        top_k=top_k,
                                        min_dets=min_dets,
                                        exclude_same="none",
                                        temporal_bonus=temporal_bonus,
                                        time_window_ms=time_window_ms,
                                    )
                                    labels, info = _resolve_scoped(records, emb, cfg, scope=scope, mode=mode)
                                    pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=keep)
                                    metrics = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
                                    rows.append({"scope": scope, **asdict(cfg), **info, **metrics})
            else:
                raise ValueError(f"unknown mode: {mode}")

    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            min(float(row["tracklet_pair_precision"]), float(row["tracklet_pair_recall"])),
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        cfg = ResolveConfig(**{key: row[key] for key in ResolveConfig.__dataclass_fields__ if key in row})
        labels, _info = _resolve_scoped(records, emb, cfg, scope=str(row["scope"]), mode=str(row["mode"]))
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, keep_seqs=keep))
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = rank
        print(json.dumps({"stage": "full", "rank": rank, "row": row}, sort_keys=True), flush=True)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "eval_stats": eval_stats,
        "output_admission": output_info,
        "n_rows": len(rows),
        "top": rows[: max(int(args.json_top_n), int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "done", "json": str(out), "top10": rows[:10]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
