#!/usr/bin/env python
"""No-GT source selector for full DS1 no-anchor assignment CSVs.

The selector chooses a sparse precision overlay source from assignment metadata
only, then scores the selected policy after the fact.  GT is not used for
policy selection, anchors, or training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_video_switch import (
        _align_sources,
        _load_assignment,
        _write_assignments,
        _parse_source,
    )
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_video_switch import (
        _align_sources,
        _load_assignment,
        _write_assignments,
        _parse_source,
    )
    from no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"seq", "video", "predicted_global_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    keep = [
        col
        for col in [
            "seq",
            "video",
            "predicted_global_id",
            "component_size",
            "prediction_confidence",
            "avg_conf",
            "n_dets",
            "member_centroid_sim_median",
            "member_centroid_sim_min",
            "nearest_external_centroid_sim",
            "centroid_margin",
            "component_internal_prob_median",
            "component_external_prob_max",
            "component_margin_prob",
            "decision_status",
        ]
        if col in df.columns
    ]
    out = df[keep].copy()
    out["seq"] = out["seq"].astype(int)
    out["video"] = out["video"].astype(str)
    out["predicted_global_id"] = out["predicted_global_id"].astype(int)
    return out.drop_duplicates("seq", keep="first")


def _num_median(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or len(df) == 0:
        return float(default)
    value = pd.to_numeric(df[col], errors="coerce").median()
    if pd.isna(value):
        return float(default)
    return float(value)


def _num_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or len(df) == 0:
        return float(default)
    value = pd.to_numeric(df[col], errors="coerce").mean()
    if pd.isna(value):
        return float(default)
    return float(value)


def _stats(raw: dict[str, pd.DataFrame], videos: list[str]) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, dict[str, object]]]]:
    totals = Counter()
    for df in raw.values():
        for video, n in df.groupby("video").size().items():
            totals[str(video)] = max(totals[str(video)], int(n))
    total_rows = max(1, sum(totals.get(video, 0) for video in videos))

    overall: dict[str, dict[str, object]] = {}
    per_video: dict[str, dict[str, dict[str, object]]] = {}
    for name, df in raw.items():
        source_per_video: dict[str, dict[str, object]] = {}
        for video in videos:
            g = df.loc[df["video"].astype(str) == str(video)].copy()
            component_counts = g.groupby("predicted_global_id")["seq"].nunique() if len(g) else pd.Series(dtype=float)
            reported_median = _num_median(
                g,
                "component_size",
                default=float(component_counts.median()) if len(component_counts) else 0.0,
            )
            source_per_video[str(video)] = {
                "rows": int(len(g)),
                "coverage": float(len(g) / max(1, totals.get(str(video), len(g)))),
                "components": int(g["predicted_global_id"].nunique()) if len(g) else 0,
                "median_component_size": float(reported_median),
                "prediction_confidence_mean": _num_mean(g, "prediction_confidence"),
                "centroid_margin_median": _num_median(g, "centroid_margin"),
            }
        per_video[name] = source_per_video

        component_counts = df.groupby("predicted_global_id")["seq"].nunique() if len(df) else pd.Series(dtype=float)
        reported_median = _num_median(
            df,
            "component_size",
            default=float(component_counts.median()) if len(component_counts) else 0.0,
        )
        overall[name] = {
            "rows": int(len(df)),
            "coverage": float(len(df) / total_rows),
            "components": int(df["predicted_global_id"].nunique()) if len(df) else 0,
            "median_component_size": float(reported_median),
            "prediction_confidence_mean": _num_mean(df, "prediction_confidence"),
            "centroid_margin_median": _num_median(df, "centroid_margin"),
        }
    return overall, per_video


def _choose(
    candidates: dict[str, dict[str, object]],
    *,
    base_source: str,
    min_coverage: float,
    max_coverage: float,
    max_component_ratio: float,
    min_component_ratio: float,
    target_component_ratio: float,
    strategy: str,
) -> str:
    base = candidates[base_source]
    base_median = max(1.0, float(base["median_component_size"]))
    eligible: list[tuple[float, ...] | tuple[float, float, float, str]] = []
    for name, stats in candidates.items():
        if name == base_source:
            continue
        coverage = float(stats["coverage"])
        median_size = float(stats["median_component_size"])
        if coverage < min_coverage or coverage > max_coverage:
            continue
        ratio = median_size / base_median
        if median_size <= 0.0 or ratio > max_component_ratio:
            continue
        if strategy == "balanced" and ratio < min_component_ratio:
            continue
        if strategy == "balanced":
            eligible.append(
                (
                    abs(ratio - target_component_ratio),
                    -coverage,
                    -median_size,
                    -float(stats["prediction_confidence_mean"]),
                    -float(stats["centroid_margin_median"]),
                    name,
                )
            )
        else:
            eligible.append((median_size, -coverage, -float(stats["components"]), name))
    if not eligible:
        return base_source
    eligible.sort()
    return str(eligible[0][-1])


def _overlay_policy_pred(
    aligned: dict[str, dict[int, tuple[str, int]]],
    policy: dict[str, str],
    *,
    base_source: str,
    videos: list[str],
) -> dict[int, int]:
    """Apply a sparse source as an overlay, falling back to base elsewhere."""

    out: dict[int, int] = {}
    allowed = set(videos)
    base_rows = aligned[base_source]
    for seq, (video, gid) in base_rows.items():
        if video in allowed:
            out[int(seq)] = int(gid)
    for video in videos:
        source = policy[video]
        if source == base_source:
            continue
        for seq, (row_video, gid) in aligned[source].items():
            if row_video == video and row_video in allowed:
                out[int(seq)] = int(gid)
    return out


def _write_overlay_assignments(
    path: str,
    pred_by_seq: dict[int, int],
    aligned_rows: dict[str, dict[int, tuple[str, int]]],
    policy: dict[str, str],
    *,
    base_source: str,
    videos: list[str],
) -> dict[str, int]:
    by_seq_meta: dict[int, tuple[str, str]] = {}
    allowed = set(videos)
    for seq, (video, _gid) in aligned_rows[base_source].items():
        if video in allowed:
            by_seq_meta[int(seq)] = (video, base_source)
    for video in videos:
        source = policy[video]
        if source == base_source:
            continue
        for seq, (row_video, _gid) in aligned_rows[source].items():
            if row_video == video and row_video in allowed:
                by_seq_meta[int(seq)] = (row_video, source)
    return _write_assignments(path, pred_by_seq, {base_source: by_seq_meta}, {video: base_source for video in videos}, videos)


def _score_overlay_policy(
    name: str,
    policy: dict[str, str],
    *,
    aligned: dict[str, dict[int, tuple[str, int]]],
    base_source: str,
    videos: list[str],
    records,
    pred_by_video,
    gt_by_video,
    gt_by_seq,
    weight_by_seq,
) -> dict[str, object]:
    pred = _overlay_policy_pred(aligned, policy, base_source=base_source, videos=videos)
    # Keep the scoring implementation explicit here so sparse overlay semantics
    # cannot accidentally regress to source-only delivery.
    pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
    full = _score_full(pred_by_video, gt_by_video, pred)
    counts = Counter(policy.values())
    return {
        "policy_name": name,
        "policy": dict(sorted(policy.items())),
        "source_counts": dict(sorted(counts.items())),
        "output_tracklets": int(len(pred)),
        "sparse_overlay_base_source": base_source,
        **pair,
        **{f"full_{key}": value for key, value in full.items() if key != "per_video"},
        "full_per_video": full["per_video"],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--source", action="append", required=True, help="name:/path/to/assignments.csv")
    ap.add_argument("--reference-source", required=True)
    ap.add_argument("--base-source", required=True)
    ap.add_argument("--sparse-min-coverage", type=float, default=0.05)
    ap.add_argument("--sparse-max-coverage", type=float, default=0.60)
    ap.add_argument("--sparse-max-component-ratio", type=float, default=0.60)
    ap.add_argument("--sparse-min-component-ratio", type=float, default=0.20)
    ap.add_argument("--sparse-target-component-ratio", type=float, default=0.45)
    ap.add_argument("--selector-strategy", choices=["conservative", "balanced"], default="conservative")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--assignments-out", default=None)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    sources = dict(_parse_source(spec) for spec in args.source)
    if len(sources) != len(args.source):
        raise ValueError("duplicate source names")
    if args.reference_source not in sources:
        raise ValueError("--reference-source must be one of --source")
    if args.base_source not in sources:
        raise ValueError("--base-source must be one of --source")

    raw = {name: _load_assignment(path) for name, path in sources.items()}
    raw_meta = {name: _read_raw(path) for name, path in sources.items()}
    aligned, align_stats = _align_sources(raw, str(args.reference_source))

    con = _connect(args.dbname)
    records, _emb = _load_tracklets(con, args.role)
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
    videos = sorted(pred_by_video)

    overall_stats, per_video_stats = _stats(raw_meta, videos)
    global_source = _choose(
        overall_stats,
        base_source=str(args.base_source),
        min_coverage=float(args.sparse_min_coverage),
        max_coverage=float(args.sparse_max_coverage),
        max_component_ratio=float(args.sparse_max_component_ratio),
        min_component_ratio=float(args.sparse_min_component_ratio),
        target_component_ratio=float(args.sparse_target_component_ratio),
        strategy=str(args.selector_strategy),
    )
    global_policy = {video: global_source for video in videos}
    per_video_policy = {
        video: _choose(
            {name: stats[str(video)] for name, stats in per_video_stats.items()},
            base_source=str(args.base_source),
            min_coverage=float(args.sparse_min_coverage),
            max_coverage=float(args.sparse_max_coverage),
            max_component_ratio=float(args.sparse_max_component_ratio),
            min_component_ratio=float(args.sparse_min_component_ratio),
            target_component_ratio=float(args.sparse_target_component_ratio),
            strategy=str(args.selector_strategy),
        )
        for video in videos
    }
    base_policy = {video: str(args.base_source) for video in videos}

    selector_prefix = (
        "selector_sparse_overlay"
        if str(args.selector_strategy) == "conservative"
        else f"selector_{args.selector_strategy}_sparse_overlay"
    )
    rows = []
    for name, policy in [
        ("base", base_policy),
        (f"{selector_prefix}_global", global_policy),
        (f"{selector_prefix}_per_video", per_video_policy),
    ]:
        row = _score_overlay_policy(
            name,
            policy,
            aligned=aligned,
            base_source=str(args.base_source),
            videos=videos,
            records=records,
            pred_by_video=pred_by_video,
            gt_by_video=gt_by_video,
            gt_by_seq=gt_by_seq,
            weight_by_seq=weight_by_seq,
        )
        row["selector_uses_gt"] = False
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "policy",
                    "policy_name": name,
                    "full_idf1": row["full_idf1"],
                    "full_hota": row["full_hota"],
                    "pair_f1": row["tracklet_pair_f1"],
                    "selector_uses_gt": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rows.sort(
        key=lambda row: (
            float(row["full_idf1"]),
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
        ),
        reverse=True,
    )
    best = rows[0]
    assignment_info = {}
    if args.assignments_out:
        pred = _overlay_policy_pred(aligned, best["policy"], base_source=str(args.base_source), videos=videos)
        assignment_info = _write_overlay_assignments(
            args.assignments_out,
            pred,
            aligned,
            best["policy"],
            base_source=str(args.base_source),
            videos=videos,
        )

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "sources": {name: str(path) for name, path in sorted(sources.items())},
        "reference_source": str(args.reference_source),
        "base_source": str(args.base_source),
        "selector": {
            "name": f"{args.selector_strategy}_sparse_overlay",
            "selector_strategy": str(args.selector_strategy),
            "sparse_min_coverage": float(args.sparse_min_coverage),
            "sparse_max_coverage": float(args.sparse_max_coverage),
            "sparse_max_component_ratio": float(args.sparse_max_component_ratio),
            "sparse_min_component_ratio": float(args.sparse_min_component_ratio),
            "sparse_target_component_ratio": float(args.sparse_target_component_ratio),
            "global_selected_source": global_source,
            "per_video_selected_sources": dict(sorted(per_video_policy.items())),
            "uses_gt": False,
        },
        "alignment": align_stats,
        "source_stats_overall": overall_stats,
        "source_stats_per_video": per_video_stats,
        "eval_stats": eval_stats,
        "rows": rows,
        "best": best,
        "assignment_info": assignment_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "done", "json": str(out), "selector": result["selector"], "best": best}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
