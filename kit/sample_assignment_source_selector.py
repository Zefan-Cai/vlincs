#!/usr/bin/env python
"""No-GT source selector for local no-anchor sample assignments.

This is the production-facing counterpart to
``sample_assignment_video_source_switch.py``.  It may inspect assignment CSV
metadata and source/source overlap, but it does not use GT to choose a policy.
The sample parquet GT is loaded only after policy selection to score the chosen
policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.sample_assignment_admission_grid import _pair_metrics
from kit.sample_assignment_state_policy_sweep import (
    _gt_mapping,
    _jsonable,
    _load_parquets,
    _sample_parquet_gt_score,
    _tracklet_table,
)
from kit.sample_assignment_video_source_switch import (
    _align_sources,
    _build_policy_assignments,
    _load_assignment,
    _parse_source,
    _pred_by_seq,
)


def _raw_assignment(path: Path, table: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "tracklet_key" not in raw.columns or "predicted_global_id" not in raw.columns:
        raise ValueError(f"{path} must contain tracklet_key and predicted_global_id")
    keep = [
        col
        for col in [
            "tracklet_key",
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
        if col in raw.columns
    ]
    raw = raw[keep].copy()
    raw["tracklet_key"] = raw["tracklet_key"].astype(str)
    raw = raw.drop_duplicates("tracklet_key", keep="first")
    meta = table[["tracklet_key", "video"]].copy()
    meta["tracklet_key"] = meta["tracklet_key"].astype(str)
    return raw.merge(meta, on="tracklet_key", how="inner")


def _numeric_median(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or len(df) == 0:
        return float(default)
    value = pd.to_numeric(df[col], errors="coerce").median()
    if pd.isna(value):
        return float(default)
    return float(value)


def _numeric_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or len(df) == 0:
        return float(default)
    value = pd.to_numeric(df[col], errors="coerce").mean()
    if pd.isna(value):
        return float(default)
    return float(value)


def _source_stats(
    raw_meta: dict[str, pd.DataFrame],
    table: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    video_totals = table.groupby("video")["tracklet_key"].nunique().astype(int).to_dict()
    per_video: dict[str, dict[str, dict[str, float]]] = {}
    overall: dict[str, dict[str, float]] = {}
    total_tracklets = int(table["tracklet_key"].nunique())

    for name, df in raw_meta.items():
        source_video: dict[str, dict[str, float]] = {}
        for video, total in sorted(video_totals.items()):
            g = df.loc[df["video"].astype(str) == str(video)].copy()
            comp_sizes = g.groupby("predicted_global_id")["tracklet_key"].nunique() if len(g) else pd.Series(dtype=float)
            median_component_size = _numeric_median(g, "component_size", default=float(comp_sizes.median()) if len(comp_sizes) else 0.0)
            p90_component_size = (
                float(pd.to_numeric(g["component_size"], errors="coerce").quantile(0.9))
                if "component_size" in g.columns and len(g)
                else (float(comp_sizes.quantile(0.9)) if len(comp_sizes) else 0.0)
            )
            source_video[str(video)] = {
                "rows": int(len(g)),
                "coverage": float(len(g) / max(1, int(total))),
                "components": int(g["predicted_global_id"].nunique()) if len(g) else 0,
                "median_component_size": float(median_component_size),
                "p90_component_size": float(p90_component_size),
                "largest_component_fraction": float(comp_sizes.max() / max(1, len(g))) if len(comp_sizes) else 0.0,
                "prediction_confidence_mean": _numeric_mean(g, "prediction_confidence"),
                "centroid_margin_median": _numeric_median(g, "centroid_margin"),
                "avg_conf_median": _numeric_median(g, "avg_conf"),
                "n_dets_median": _numeric_median(g, "n_dets"),
            }
        per_video[name] = source_video

        comp_sizes = df.groupby("predicted_global_id")["tracklet_key"].nunique() if len(df) else pd.Series(dtype=float)
        status_counts = Counter(str(x) for x in df.get("decision_status", pd.Series(dtype=object)).dropna().tolist())
        median_component_size = _numeric_median(df, "component_size", default=float(comp_sizes.median()) if len(comp_sizes) else 0.0)
        p90_component_size = (
            float(pd.to_numeric(df["component_size"], errors="coerce").quantile(0.9))
            if "component_size" in df.columns and len(df)
            else (float(comp_sizes.quantile(0.9)) if len(comp_sizes) else 0.0)
        )
        overall[name] = {
            "rows": int(len(df)),
            "coverage": float(len(df) / max(1, total_tracklets)),
            "components": int(df["predicted_global_id"].nunique()) if len(df) else 0,
            "median_component_size": float(median_component_size),
            "p90_component_size": float(p90_component_size),
            "largest_component_fraction": float(comp_sizes.max() / max(1, len(df))) if len(comp_sizes) else 0.0,
            "prediction_confidence_mean": _numeric_mean(df, "prediction_confidence"),
            "centroid_margin_median": _numeric_median(df, "centroid_margin"),
            "avg_conf_median": _numeric_median(df, "avg_conf"),
            "n_dets_median": _numeric_median(df, "n_dets"),
            "decision_status_forced_fraction": float(status_counts.get("forced_component", 0) / max(1, len(df))),
        }
    return per_video, overall


def _choose_sparse_overlay(
    *,
    candidates: dict[str, dict[str, float]],
    base_source: str,
    min_coverage: float,
    max_coverage: float,
    max_component_ratio: float,
) -> str:
    base = candidates[base_source]
    base_median = max(1.0, float(base["median_component_size"]))
    eligible: list[tuple[float, float, float, str]] = []
    for name, stats in candidates.items():
        if name == base_source:
            continue
        coverage = float(stats["coverage"])
        median_size = float(stats["median_component_size"])
        if coverage < min_coverage or coverage > max_coverage:
            continue
        if median_size <= 0.0 or median_size > base_median * max_component_ratio:
            continue
        # Prefer a precision overlay with smaller components, then more
        # coverage.  This avoids GT-tuned thresholds and keeps the rule
        # interpretable.
        eligible.append((median_size, -coverage, -float(stats["components"]), name))
    if not eligible:
        return base_source
    eligible.sort()
    return eligible[0][3]


def _score_policy(
    *,
    name: str,
    policy: dict[str, str],
    aligned: dict[str, pd.DataFrame],
    base_source: str,
    missing_mode: str,
    df: pd.DataFrame,
    table: pd.DataFrame,
    gt_by_seq: dict[int, int],
    weight_by_seq: dict[int, float],
) -> dict[str, object]:
    assignments = _build_policy_assignments(aligned, policy, base_source=base_source, missing_mode=missing_mode)
    seqs = [int(row.seq) for row in table.itertuples(index=False)]
    pred = _pred_by_seq(table, assignments)
    pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
    full = _sample_parquet_gt_score(df, assignments)
    return {
        "policy_name": name,
        "policy": dict(sorted(policy.items())),
        "source_counts": dict(sorted(Counter(policy.values()).items())),
        "output_tracklets": int(assignments["tracklet_key"].nunique()),
        **pair,
        **full,
        "selector_uses_gt": False,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracklet-parquet", nargs="+", required=True)
    ap.add_argument("--source", action="append", required=True, help="name:/path/to/assignments.csv")
    ap.add_argument("--reference-source", required=True)
    ap.add_argument("--base-source", required=True)
    ap.add_argument("--missing-mode", choices=["drop", "base_fallback"], default="base_fallback")
    ap.add_argument("--sparse-min-coverage", type=float, default=0.05)
    ap.add_argument("--sparse-max-coverage", type=float, default=0.60)
    ap.add_argument("--sparse-max-component-ratio", type=float, default=0.60)
    ap.add_argument("--eval-min-gt-fraction", type=float, default=0.5)
    ap.add_argument("--eval-min-rows", type=int, default=1)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    source_paths = dict(_parse_source(spec) for spec in args.source)
    if len(source_paths) != len(args.source):
        raise ValueError("duplicate source names")
    if args.reference_source not in source_paths:
        raise ValueError("--reference-source must be one of --source")
    if args.base_source not in source_paths:
        raise ValueError("--base-source must be one of --source")

    df = _load_parquets(args.tracklet_parquet)
    table = _tracklet_table(df)
    videos = sorted(table["video"].astype(str).unique().tolist())
    raw = {name: _load_assignment(path, table) for name, path in source_paths.items()}
    raw_meta = {name: _raw_assignment(path, table) for name, path in source_paths.items()}
    aligned, align_stats = _align_sources(raw, str(args.reference_source))
    per_video_stats, overall_stats = _source_stats(raw_meta, table)

    gt_by_seq, weight_by_seq, eval_info = _gt_mapping(
        table,
        min_gt_fraction=float(args.eval_min_gt_fraction),
        min_rows=int(args.eval_min_rows),
    )

    base_policy = {video: str(args.base_source) for video in videos}
    global_source = _choose_sparse_overlay(
        candidates=overall_stats,
        base_source=str(args.base_source),
        min_coverage=float(args.sparse_min_coverage),
        max_coverage=float(args.sparse_max_coverage),
        max_component_ratio=float(args.sparse_max_component_ratio),
    )
    global_policy = {video: global_source for video in videos}
    per_video_policy = {}
    for video in videos:
        per_video_candidates = {name: stats[str(video)] for name, stats in per_video_stats.items()}
        per_video_policy[video] = _choose_sparse_overlay(
            candidates=per_video_candidates,
            base_source=str(args.base_source),
            min_coverage=float(args.sparse_min_coverage),
            max_coverage=float(args.sparse_max_coverage),
            max_component_ratio=float(args.sparse_max_component_ratio),
        )

    rows = [
        _score_policy(
            name="base",
            policy=base_policy,
            aligned=aligned,
            base_source=str(args.base_source),
            missing_mode=str(args.missing_mode),
            df=df,
            table=table,
            gt_by_seq=gt_by_seq,
            weight_by_seq=weight_by_seq,
        ),
        _score_policy(
            name="selector_sparse_overlay_global",
            policy=global_policy,
            aligned=aligned,
            base_source=str(args.base_source),
            missing_mode=str(args.missing_mode),
            df=df,
            table=table,
            gt_by_seq=gt_by_seq,
            weight_by_seq=weight_by_seq,
        ),
        _score_policy(
            name="selector_sparse_overlay_per_video",
            policy=per_video_policy,
            aligned=aligned,
            base_source=str(args.base_source),
            missing_mode=str(args.missing_mode),
            df=df,
            table=table,
            gt_by_seq=gt_by_seq,
            weight_by_seq=weight_by_seq,
        ),
    ]
    rows.sort(key=lambda row: (float(row["sample_full_idf1"]), float(row["tracklet_pair_f1"])), reverse=True)

    result = {
        "tracklet_parquet": [str(p) for p in args.tracklet_parquet],
        "sources": {name: str(path) for name, path in sorted(source_paths.items())},
        "reference_source": str(args.reference_source),
        "base_source": str(args.base_source),
        "missing_mode": str(args.missing_mode),
        "selector": {
            "name": "sparse_precision_overlay",
            "sparse_min_coverage": float(args.sparse_min_coverage),
            "sparse_max_coverage": float(args.sparse_max_coverage),
            "sparse_max_component_ratio": float(args.sparse_max_component_ratio),
            "global_selected_source": global_source,
            "per_video_selected_sources": dict(sorted(per_video_policy.items())),
            "uses_gt": False,
        },
        "align_stats": align_stats,
        "source_stats_overall": overall_stats,
        "source_stats_per_video": per_video_stats,
        "eval_stats": eval_info,
        "rows": rows,
        "best": rows[0],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable({"json": str(out), "selector": result["selector"], "best": rows[0]}), sort_keys=True))


if __name__ == "__main__":
    main()
