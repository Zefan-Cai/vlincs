#!/usr/bin/env python
"""Analyze full-score outcomes for no-anchor proposal rows.

This is an offline AutoResearch utility.  It harvests already full-scored
no-anchor proposal rows from local JSON artifacts and asks whether no-GT row
features can predict the expensive full IDF1 outcome.  Oracle rows are excluded
by default; GT appears only as post-hoc labels from completed full scoring.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_KEYS = [
    "assigned_tracklets",
    "output_tracklets",
    "eval_tracklets",
    "coverage_ratio",
    "delivery_tracklets_min",
    "delivery_tracklets_max",
    "delivery_tracklets_mean",
    "tracklet_pair_f1",
    "tracklet_pair_precision",
    "tracklet_pair_recall",
    "full_side_effect_proxy",
    "accepted_reassignments",
    "moved_tracklets",
    "target_components_used",
    "candidate_search_prefix",
    "candidate_edges",
    "candidate_total_edges_before_skip",
    "candidate_skip_first_edge_families",
    "candidate_first_edge_family_rank",
    "max_sources_per_target",
    "max_reassignments",
    "source_candidates",
    "source_conflicted_components",
    "source_conflict_edges",
    "source_conflict_nodes",
    "source_conflicts_to_rest",
    "source_internal_sim",
    "source_cross_mean_sim",
    "source_cross_max_sim",
    "source_margin_mean",
    "source_margin_max",
    "source_quality",
    "source_score",
    "source_seed_sim",
    "source_expand_sim",
    "source_top_k",
    "source_size",
    "source_min_group_size",
    "source_max_group_size",
    "source_min_conflicts_to_rest",
    "source_min_margin",
    "source_max_groups_per_component",
    "source_max_total_groups",
    "min_target_size",
    "target_top_k",
    "min_target_best_sim",
    "min_target_mean_sim",
    "min_target_view_vote",
    "min_target_quality",
    "target_view_sim_threshold",
    "min_target_margin",
    "target_best_sim",
    "target_mean_sim",
    "target_min_view_sim",
    "target_view_vote",
    "target_quality",
    "target_score",
    "target_size",
    "target_margin",
    "target_rank_for_source",
    "target_forbidden_pairs",
    "committed_min_size",
    "pending_max_size",
    "conflict_rate_threshold",
    "accepted_edges",
    "accepted_score_mean",
    "accepted_view_mean_sim_mean",
    "accepted_view_min_sim_mean",
    "accepted_mass_proxy_sum",
    "accepted_pair_mass_proxy_sum",
    "accepted_min_weight_sum",
    "accepted_max_weight_sum",
    "accepted_size_product_sum",
    "merge_score_mean",
    "merge_rank_margin_mean",
    "merge_mass_proxy_sum",
    "merge_pair_mass_proxy_sum",
    "merge_size_product_sum",
    "merge_min_weight_sum",
    "merge_max_weight_sum",
    "view_margin_min",
    "view_margin_mean",
    "view_weak_margin_fraction",
    "view_non_rank1_fraction",
    "visual_opponent_risk_score",
    "combined_opponent_risk_score",
    "max_same_video_overlap_frames",
    "same_video_pair_fraction",
    "local_pair_fraction_same_video",
    "overlap_pair_fraction_same_video",
    "temporal_opponent_risk_score",
    "max_overlap_source_duration_fraction",
    "source_target_video_jaccard",
]

COMPACT_FEATURE_KEYS = [
    "assigned_tracklets",
    "output_tracklets",
    "eval_tracklets",
    "coverage_ratio",
    "delivery_tracklets_min",
    "delivery_tracklets_mean",
    "tracklet_pair_f1",
    "tracklet_pair_precision",
    "tracklet_pair_recall",
    "full_side_effect_proxy",
    "accepted_reassignments",
    "moved_tracklets",
    "target_components_used",
    "candidate_search_prefix",
    "candidate_skip_first_edge_families",
    "candidate_first_edge_family_rank",
    "max_sources_per_target",
    "max_reassignments",
    "source_quality",
    "source_score",
    "source_size",
    "source_margin_mean",
    "target_best_sim",
    "target_mean_sim",
    "target_min_view_sim",
    "target_view_vote",
    "target_quality",
    "target_score",
    "target_size",
    "target_margin",
    "preview_mean_target_mean_sim",
    "preview_mean_target_best_sim",
    "preview_mean_target_view_vote",
    "preview_min_target_min_view_sim",
    "preview_mean_source_quality",
    "preview_mean_source_score",
    "preview_mean_source_size",
    "accepted_edges",
    "accepted_score_mean",
    "accepted_view_mean_sim_mean",
    "accepted_view_min_sim_mean",
    "accepted_mass_proxy_sum",
    "accepted_pair_mass_proxy_sum",
    "accepted_min_weight_sum",
    "accepted_max_weight_sum",
    "accepted_size_product_sum",
    "merge_score_mean",
    "merge_rank_margin_mean",
    "merge_mass_proxy_sum",
    "merge_pair_mass_proxy_sum",
    "merge_size_product_sum",
    "merge_min_weight_sum",
    "merge_max_weight_sum",
    "preview_mean_score",
    "preview_mean_view_mean_sim",
    "preview_mean_view_min_sim",
    "preview_mean_view_rank_vote",
    "preview_mean_view_sim_vote",
    "preview_mean_bridge_mass_proxy",
    "preview_mean_pair_mass_proxy",
    "preview_mean_source_weight",
    "preview_mean_target_weight",
    "preview_max_pair_mass_proxy",
    "preview_min_view_margin_min",
    "preview_mean_view_weak_margin_fraction",
    "preview_mean_view_non_rank1_fraction",
    "preview_mean_visual_opponent_risk_score",
    "preview_max_combined_opponent_risk_score",
    "preview_max_max_same_video_overlap_frames",
    "preview_mean_same_video_pair_fraction",
    "preview_mean_local_pair_fraction_same_video",
    "preview_mean_overlap_pair_fraction_same_video",
    "preview_mean_temporal_opponent_risk_score",
    "preview_max_max_overlap_source_duration_fraction",
    "preview_mean_source_target_video_jaccard",
]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("top", "rows", "results", "top_pair_rows", "top_full_rows"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    return rows


def _row_features(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in FEATURE_KEYS:
        val = _as_float(row.get(key))
        if val is not None:
            values[key] = val
    delivery_vals = [
        _as_float(row.get(key))
        for key in ("assigned_tracklets", "output_tracklets", "eval_tracklets")
    ]
    delivery_vals = [value for value in delivery_vals if value is not None]
    if delivery_vals:
        values["delivery_tracklets_min"] = float(np.min(delivery_vals))
        values["delivery_tracklets_max"] = float(np.max(delivery_vals))
        values["delivery_tracklets_mean"] = float(np.mean(delivery_vals))
    preview = row.get("accepted_preview")
    if isinstance(preview, list) and preview:
        for key in (
            "target_mean_sim",
            "target_best_sim",
            "target_view_vote",
            "target_min_view_sim",
            "target_quality",
            "target_margin",
            "source_quality",
            "source_score",
            "source_size",
            "source_margin_mean",
            "source_internal_sim",
            "source_cross_mean_sim",
            "score",
            "view_mean_sim",
            "view_min_sim",
            "view_rank_vote",
            "view_sim_vote",
            "bridge_mass_proxy",
            "pair_mass_proxy",
            "source_weight",
            "target_weight",
            "view_margin_min",
            "view_margin_mean",
            "view_weak_margin_fraction",
            "view_non_rank1_fraction",
            "visual_opponent_risk_score",
            "combined_opponent_risk_score",
            "max_same_video_overlap_frames",
            "same_video_pair_fraction",
            "local_pair_fraction_same_video",
            "overlap_pair_fraction_same_video",
            "temporal_opponent_risk_score",
            "max_overlap_source_duration_fraction",
            "source_target_video_jaccard",
        ):
            vals = [_as_float(item.get(key)) for item in preview if isinstance(item, dict)]
            vals = [val for val in vals if val is not None]
            if vals:
                values[f"preview_mean_{key}"] = float(np.mean(vals))
                values[f"preview_min_{key}"] = float(np.min(vals))
                values[f"preview_max_{key}"] = float(np.max(vals))
    return values


def _collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.local_runs)
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in sorted(root.rglob("*.json")):
        if "full_proxy" in path.name and "training_audit" in path.name:
            continue
        if "pair_candidates" in path.name:
            continue
        for row in _iter_rows(path):
            y = _as_float(row.get("full_idf1"))
            if y is None:
                continue
            if y < float(args.min_full_idf1):
                continue
            if not args.include_oracle and row.get("uses_gt_for_analysis_only") is True:
                continue
            features = _row_features(row)
            if not features:
                continue
            signature = (
                row.get("mode") or row.get("name"),
                round(y, 6),
                tuple(sorted((key, round(val, 6)) for key, val in features.items())),
            )
            if args.dedup and signature in seen:
                continue
            seen.add(signature)
            out.append(
                {
                    "artifact": str(path),
                    "mode": row.get("mode") or row.get("name") or "",
                    "full_idf1": y,
                    "features": features,
                    "pair_f1": _as_float(row.get("tracklet_pair_f1")),
                    "pair_precision": _as_float(row.get("tracklet_pair_precision")),
                    "pair_recall": _as_float(row.get("tracklet_pair_recall")),
                    "accepted_reassignments": _as_float(row.get("accepted_reassignments")),
                    "moved_tracklets": _as_float(row.get("moved_tracklets")),
                }
            )
    return out


def _matrix(rows: list[dict[str, Any]], *, feature_mode: str) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    keys = sorted({key for row in rows for key in row["features"]})
    if feature_mode == "compact":
        keys = [key for key in COMPACT_FEATURE_KEYS if key in keys]
    x = np.zeros((len(rows), len(keys)), dtype=np.float64)
    present = np.zeros_like(x)
    for i, row in enumerate(rows):
        feats = row["features"]
        for j, key in enumerate(keys):
            if key in feats:
                x[i, j] = float(feats[key])
                present[i, j] = 1.0
    cols = []
    arrs = []
    fills: dict[str, float] = {}
    for j, key in enumerate(keys):
        vals = x[:, j]
        mask = present[:, j] > 0
        if mask.sum() < 3:
            continue
        fill = float(np.median(vals[mask]))
        fills[key] = fill
        vals = vals.copy()
        vals[~mask] = fill
        arrs.append(vals)
        cols.append(key)
    if not arrs:
        return np.zeros((len(rows), 0)), np.asarray([row["full_idf1"] for row in rows], dtype=np.float64), [], {}
    x2 = np.stack(arrs, axis=1)
    return x2, np.asarray([row["full_idf1"] for row in rows], dtype=np.float64), cols, fills


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1.0e-9] = 1.0
    return (x - mu) / sigma, mu, sigma


def _ridge_fit_predict_loocv(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    preds = np.zeros_like(y)
    for i in range(len(y)):
        train = np.ones(len(y), dtype=bool)
        train[i] = False
        xtr, ytr = x[train], y[train]
        xte = x[~train]
        xtr_s, mu, sigma = _standardize(xtr)
        xte_s = (xte - mu) / sigma
        design = np.concatenate([np.ones((xtr_s.shape[0], 1)), xtr_s], axis=1)
        reg = np.eye(design.shape[1]) * float(alpha)
        reg[0, 0] = 0.0
        coef = np.linalg.pinv(design.T @ design + reg) @ design.T @ ytr
        preds[i] = float((np.concatenate([np.ones((1, 1)), xte_s], axis=1) @ coef).item())
    return preds


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    x_s, mu, sigma = _standardize(x)
    design = np.concatenate([np.ones((x_s.shape[0], 1)), x_s], axis=1)
    reg = np.eye(design.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + reg) @ design.T @ y
    return {
        "intercept": float(coef[0]),
        "coef": [float(value) for value in coef[1:]],
        "mean": [float(value) for value in mu],
        "scale": [float(value) for value in sigma],
        "alpha": float(alpha),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or float(np.std(a)) < 1.0e-12 or float(np.std(b)) < 1.0e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _feature_correlations(x: np.ndarray, y: np.ndarray, cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for j, key in enumerate(cols):
        corr = _corr(x[:, j], y)
        if corr is None:
            continue
        rows.append({"feature": key, "corr_with_full_idf1": corr})
    rows.sort(key=lambda item: abs(float(item["corr_with_full_idf1"])), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-runs", default="local_runs")
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--dedup", action="store_true", default=True)
    parser.add_argument("--min-full-idf1", type=float, default=0.55)
    parser.add_argument("--feature-mode", choices=["compact", "all"], default="compact")
    parser.add_argument("--json", required=True)
    parser.add_argument("--md", default="")
    parser.add_argument("--model-json", default="")
    args = parser.parse_args()

    rows = _collect(args)
    x, y, cols, fills = _matrix(rows, feature_mode=str(args.feature_mode))
    result: dict[str, Any] = {
        "row_count": len(rows),
        "feature_count": len(cols),
        "include_oracle": bool(args.include_oracle),
        "dedup": bool(args.dedup),
        "min_full_idf1": float(args.min_full_idf1),
        "feature_mode": str(args.feature_mode),
        "full_idf1_min": float(np.min(y)) if len(y) else None,
        "full_idf1_max": float(np.max(y)) if len(y) else None,
        "full_idf1_mean": float(np.mean(y)) if len(y) else None,
        "rows": rows,
        "feature_correlations": [],
        "ridge_loocv": {},
        "ridge_model": {},
        "top_rows_by_full": sorted(rows, key=lambda row: float(row["full_idf1"]), reverse=True)[:10],
    }
    if len(rows) >= 6 and x.shape[1] > 0:
        result["feature_correlations"] = _feature_correlations(x, y, cols)[:30]
        ridge_grid = []
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            ridge = _ridge_fit_predict_loocv(x, y, alpha=alpha)
            ridge_grid.append(
                {
                    "alpha": alpha,
                    "corr": _corr(ridge, y),
                    "mae": float(np.mean(np.abs(ridge - y))),
                    "rmse": float(np.sqrt(np.mean((ridge - y) ** 2))),
                }
            )
        result["ridge_loocv"] = min(ridge_grid, key=lambda item: float(item["mae"]))
        result["ridge_loocv_grid"] = ridge_grid
        fit = _ridge_fit(x, y, alpha=float(result["ridge_loocv"]["alpha"]))
        result["ridge_model"] = {
            "model_type": "compact_ridge_full_idf1_proxy",
            "columns": cols,
            "fill_values": fills,
            **fit,
            "row_count": len(rows),
            "feature_mode": str(args.feature_mode),
            "min_full_idf1": float(args.min_full_idf1),
            "include_oracle": bool(args.include_oracle),
            "target": "full_idf1",
            "target_min": float(np.min(y)),
            "target_max": float(np.max(y)),
            "target_mean": float(np.mean(y)),
            "target_clamp_margin": 0.001,
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_posthoc_full_score_labels": True,
        }
        for key in ("pair_f1", "pair_precision", "pair_recall"):
            vals = np.asarray([row[key] if row[key] is not None else np.nan for row in rows], dtype=np.float64)
            mask = np.isfinite(vals)
            if mask.sum() >= 3:
                result[f"{key}_corr"] = _corr(vals[mask], y[mask])
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True))
    if args.model_json:
        Path(args.model_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.model_json).write_text(json.dumps(result.get("ridge_model", {}), indent=2, sort_keys=True))
    if args.md:
        lines = [
            "# No-Anchor Full-Proxy Training Audit",
            "",
            f"- rows: `{result['row_count']}`",
            f"- features: `{result['feature_count']}`",
            f"- include oracle: `{result['include_oracle']}`",
            f"- min full IDF1: `{result['min_full_idf1']}`",
            f"- feature mode: `{result['feature_mode']}`",
            f"- full IDF1 range: `{result['full_idf1_min']}` - `{result['full_idf1_max']}`",
            "",
            "## Ridge LOOCV",
            "",
            json.dumps(result["ridge_loocv"], indent=2, sort_keys=True),
            "",
            "## Top Feature Correlations",
            "",
            "| feature | corr |",
            "| --- | ---: |",
        ]
        for item in result["feature_correlations"][:20]:
            lines.append(f"| `{item['feature']}` | `{item['corr_with_full_idf1']:.6f}` |")
        lines.extend(["", "## Top Full Rows", "", "| full IDF1 | pair F1 | mode | artifact |", "| ---: | ---: | --- | --- |"])
        for row in result["top_rows_by_full"]:
            lines.append(
                f"| `{row['full_idf1']:.6f}` | `{row['pair_f1'] if row['pair_f1'] is not None else ''}` | "
                f"`{row['mode']}` | `{row['artifact']}` |"
            )
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n")
    print(json.dumps({k: result[k] for k in ("row_count", "feature_count", "full_idf1_min", "full_idf1_max", "ridge_loocv")}, sort_keys=True))


if __name__ == "__main__":
    main()
