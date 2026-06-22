#!/usr/bin/env python
"""Audit whether no-GT source-island features can rank useful repairs.

This is an eval-only AutoResearch diagnostic.  It reads
no_anchor_conflict_source_island_audit JSON files, uses only no-GT numeric
source features as inputs, and uses oracle_delta_pair_f1 only as a posthoc
label.  The output tells us whether a source-island repairability acceptor is
promising enough to justify a larger remote sweep.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXCLUDE_FEATURE_KEYS = {
    "component_eval_weight",
    "component_gt_count",
    "component_majority_gt",
    "component_majority_gt_frac",
    "oracle_delta_pair_f1",
    "oracle_delta_pred_pair_mass",
    "oracle_delta_true_pair_mass",
    "oracle_pair_f1",
    "oracle_pair_precision",
    "oracle_pair_recall",
    "oracle_target_component",
    "source_differs_from_component_majority",
    "source_component_label",
    "source_eval_weight",
    "source_gt_count",
    "source_majority_gt",
    "source_majority_gt_frac",
    "source_seqs",
    "uses_anchors",
    "uses_gt_for_evaluation_only",
    "uses_gt_for_training_or_anchors",
}


PREFERRED_FEATURE_KEYS = [
    "source_rank_score",
    "source_score",
    "source_quality",
    "source_size",
    "source_conflicts_to_rest",
    "source_internal_sim",
    "source_cross_mean_sim",
    "source_cross_max_sim",
    "source_margin_mean",
    "source_margin_max",
    "source_conflict_edges",
    "source_conflict_nodes",
    "source_conflicted_components",
    "source_rejected_overlap",
    "source_seed_sim",
    "source_expand_sim",
    "source_top_k",
    "source_min_group_size",
    "source_max_group_size",
    "source_min_conflicts_to_rest",
    "source_min_margin",
    "source_min_component_size",
    "source_max_component_size",
    "source_max_groups_per_component",
    "source_max_total_groups",
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
    return out if math.isfinite(out) else None


def _load_rows(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for match in matches:
            path = Path(match)
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            source_rows = data.get("top_by_source_rank")
            if not isinstance(source_rows, list):
                source_rows = data.get("top_by_oracle_delta")
            if not isinstance(source_rows, list):
                continue
            for rank, row in enumerate(source_rows, start=1):
                if not isinstance(row, dict):
                    continue
                sig = (
                    str(path),
                    row.get("source_component_label"),
                    tuple(row.get("source_seqs", []) or []),
                    row.get("source_seed_sim"),
                    row.get("source_expand_sim"),
                    row.get("source_top_k"),
                )
                if sig in seen:
                    continue
                seen.add(sig)
                y = _as_float(row.get("oracle_delta_pair_f1"))
                feats: dict[str, float] = {}
                for key, value in row.items():
                    if key in EXCLUDE_FEATURE_KEYS:
                        continue
                    if not key.startswith("source_"):
                        continue
                    val = _as_float(value)
                    if val is not None:
                        feats[key] = val
                if not feats or y is None:
                    continue
                rows.append(
                    {
                        "artifact": str(path),
                        "source_rank": int(rank),
                        "source_component_label": row.get("source_component_label"),
                        "oracle_delta_pair_f1": float(y),
                        "oracle_positive": bool(y > 0.0),
                        "features": feats,
                        "row": row,
                    }
                )
    return rows


def _matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    present_keys = {key for row in rows for key in row["features"]}
    keys = [key for key in PREFERRED_FEATURE_KEYS if key in present_keys]
    keys.extend(sorted(key for key in present_keys if key not in set(keys)))
    x = np.zeros((len(rows), len(keys)), dtype=np.float64)
    mask = np.zeros_like(x, dtype=bool)
    for i, row in enumerate(rows):
        for j, key in enumerate(keys):
            if key in row["features"]:
                x[i, j] = float(row["features"][key])
                mask[i, j] = True
    fills: dict[str, float] = {}
    keep_cols: list[str] = []
    arrs: list[np.ndarray] = []
    for j, key in enumerate(keys):
        if int(mask[:, j].sum()) < 3:
            continue
        vals = x[:, j].copy()
        fill = float(np.median(vals[mask[:, j]]))
        vals[~mask[:, j]] = fill
        fills[key] = fill
        keep_cols.append(key)
        arrs.append(vals)
    x2 = np.stack(arrs, axis=1) if arrs else np.zeros((len(rows), 0), dtype=np.float64)
    y = np.asarray([1.0 if row["oracle_positive"] else 0.0 for row in rows], dtype=np.float64)
    return x2, y, keep_cols, fills


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1.0e-9] = 1.0
    return (x - mu) / sigma, mu, sigma


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _ridge_logit_fit(x: np.ndarray, y: np.ndarray, alpha: float, steps: int = 500, lr: float = 0.08) -> np.ndarray:
    if x.shape[1] == 0:
        return np.asarray([float(y.mean())], dtype=np.float64)
    x_s, _mu, _sigma = _standardize(x)
    design = np.concatenate([np.ones((x_s.shape[0], 1)), x_s], axis=1)
    w = np.zeros(design.shape[1], dtype=np.float64)
    # Balanced weights keep the tiny positive class from being ignored.
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    for _ in range(int(steps)):
        p = _sigmoid(design @ w)
        grad = (design.T @ ((p - y) * weights)) / max(len(y), 1)
        grad[1:] += float(alpha) * w[1:] / max(len(y), 1)
        w -= float(lr) * grad
    return w


def _ridge_logit_loocv(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    preds = np.zeros(len(y), dtype=np.float64)
    for i in range(len(y)):
        train = np.ones(len(y), dtype=bool)
        train[i] = False
        xtr, ytr = x[train], y[train]
        xtr_s, mu, sigma = _standardize(xtr)
        design = np.concatenate([np.ones((xtr_s.shape[0], 1)), xtr_s], axis=1)
        w = _ridge_logit_fit(xtr, ytr, alpha=alpha)
        xte = (x[~train] - mu) / sigma
        preds[i] = float(_sigmoid(np.concatenate([np.ones((1, 1)), xte], axis=1) @ w).item())
    return preds


def _ridge_logit_fit_bundle(
    x: np.ndarray,
    y: np.ndarray,
    cols: list[str],
    fills: dict[str, float],
    *,
    alpha: float,
) -> dict[str, Any]:
    if x.shape[1] == 0:
        return {}
    _x_s, mu, sigma = _standardize(x)
    coef = _ridge_logit_fit(x, y, alpha=alpha)
    return {
        "model_type": "source_island_ridge_logit_acceptor",
        "target": "oracle_delta_pair_f1_positive",
        "columns": cols,
        "fill_values": fills,
        "intercept": float(coef[0]),
        "coef": [float(value) for value in coef[1:]],
        "mean": [float(value) for value in mu],
        "scale": [float(value) for value in sigma],
        "alpha": float(alpha),
        "row_count": int(len(y)),
        "positive_count": int(y.sum()),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_posthoc_oracle_delta_labels": True,
        "excluded_inputs": sorted(EXCLUDE_FEATURE_KEYS),
    }


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    positives = float(y.sum())
    if positives <= 0:
        return 0.0
    hit = 0.0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if y[idx] > 0.5:
            hit += 1.0
            total += hit / float(rank)
    return float(total / positives)


def _topk(rows: list[dict[str, Any]], scores: np.ndarray, ks: list[int]) -> dict[str, Any]:
    y = np.asarray([1.0 if row["oracle_positive"] else 0.0 for row in rows], dtype=np.float64)
    order = np.argsort(-scores)
    out = {}
    for k in ks:
        idxs = order[: min(int(k), len(order))]
        deltas = [float(rows[int(idx)]["oracle_delta_pair_f1"]) for idx in idxs]
        out[str(k)] = {
            "positives": int(y[idxs].sum()) if len(idxs) else 0,
            "positive_fraction": round(float(y[idxs].mean()), 6) if len(idxs) else 0.0,
            "sum_positive_delta_pair_f1": round(float(sum(delta for delta in deltas if delta > 0)), 6),
            "max_delta_pair_f1": round(float(max(deltas, default=0.0)), 6),
        }
    return out


def _feature_corrs(x: np.ndarray, y: np.ndarray, cols: list[str]) -> list[dict[str, Any]]:
    out = []
    for j, key in enumerate(cols):
        if float(np.std(x[:, j])) < 1.0e-12:
            continue
        corr = float(np.corrcoef(x[:, j], y)[0, 1])
        if math.isfinite(corr):
            out.append({"feature": key, "corr_with_positive": corr})
    out.sort(key=lambda item: abs(float(item["corr_with_positive"])), reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", required=True)
    parser.add_argument("--md", default="")
    parser.add_argument("--model-json", default="")
    args = parser.parse_args()

    rows = _load_rows(list(args.paths))
    x, y, cols, fills = _matrix(rows)
    rank_score = np.asarray([row["features"].get("source_rank_score", 0.0) for row in rows], dtype=np.float64)
    source_score = np.asarray([row["features"].get("source_score", 0.0) for row in rows], dtype=np.float64)
    quality = np.asarray([row["features"].get("source_quality", 0.0) for row in rows], dtype=np.float64)
    result: dict[str, Any] = {
        "row_count": len(rows),
        "positive_count": int(y.sum()),
        "feature_count": len(cols),
        "columns": cols,
        "fill_values": fills,
        "baseline_average_precision": {
            "source_rank_score": _average_precision(y, rank_score) if len(rows) else 0.0,
            "source_score": _average_precision(y, source_score) if len(rows) else 0.0,
            "source_quality": _average_precision(y, quality) if len(rows) else 0.0,
        },
        "baseline_topk": {
            "source_rank_score": _topk(rows, rank_score, [5, 10, 20, 50]),
            "source_score": _topk(rows, source_score, [5, 10, 20, 50]),
            "source_quality": _topk(rows, quality, [5, 10, 20, 50]),
        },
        "feature_correlations": _feature_corrs(x, y, cols)[:20] if len(rows) else [],
        "ridge_logit": {},
        "top_rows_by_oracle_delta": sorted(rows, key=lambda row: float(row["oracle_delta_pair_f1"]), reverse=True)[:12],
    }
    if len(rows) >= 12 and int(y.sum()) >= 2 and x.shape[1] > 0:
        grid = []
        best_score = None
        best_alpha = None
        for alpha in (0.01, 0.1, 1.0, 10.0):
            score = _ridge_logit_loocv(x, y, alpha=alpha)
            ap = _average_precision(y, score)
            grid.append({"alpha": alpha, "average_precision": ap, "topk": _topk(rows, score, [5, 10, 20, 50])})
            if best_score is None or ap > float(max(item["average_precision"] for item in grid[:-1] or [{"average_precision": -1.0}])):
                best_score = score
                best_alpha = alpha
        result["ridge_logit"] = {
            "best_alpha": best_alpha,
            "loocv_average_precision": _average_precision(y, best_score) if best_score is not None else None,
            "grid": grid,
            "model": _ridge_logit_fit_bundle(x, y, cols, fills, alpha=float(best_alpha)),
            "top_rows": [
                {
                    "rank": rank,
                    "score": float(best_score[idx]) if best_score is not None else None,
                    "oracle_delta_pair_f1": float(rows[int(idx)]["oracle_delta_pair_f1"]),
                    "source_component_label": rows[int(idx)]["source_component_label"],
                    "artifact": rows[int(idx)]["artifact"],
                    "features": rows[int(idx)]["features"],
                }
                for rank, idx in enumerate(np.argsort(-(best_score if best_score is not None else rank_score))[:12], start=1)
            ],
        }
        result["ridge_logit"]["model"]["loocv_average_precision"] = result["ridge_logit"]["loocv_average_precision"]
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.model_json:
        model = result.get("ridge_logit", {}).get("model", {})
        Path(args.model_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.model_json).write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    if args.md:
        lines = [
            "# Source-Island Acceptor Audit",
            "",
            f"- rows: `{result['row_count']}`",
            f"- positives: `{result['positive_count']}`",
            f"- features: `{result['feature_count']}`",
            "",
            "## Average Precision",
            "",
            "| ranker | AP |",
            "| --- | ---: |",
        ]
        for key, value in result["baseline_average_precision"].items():
            lines.append(f"| `{key}` | `{float(value):.6f}` |")
        ridge = result.get("ridge_logit", {})
        if ridge:
            lines.append(f"| `ridge_logit_loocv` | `{float(ridge['loocv_average_precision']):.6f}` |")
        lines.extend(["", "## Top-k Positives", "", "| ranker | k | positives | positive frac | sum positive delta |", "| --- | ---: | ---: | ---: | ---: |"])
        for ranker, table in result["baseline_topk"].items():
            for k, item in table.items():
                lines.append(
                    f"| `{ranker}` | `{k}` | `{item['positives']}` | `{item['positive_fraction']:.6f}` | `{item['sum_positive_delta_pair_f1']:.6f}` |"
                )
        if ridge:
            best_grid = max(ridge["grid"], key=lambda item: float(item["average_precision"]))
            for k, item in best_grid["topk"].items():
                lines.append(
                    f"| `ridge_logit_loocv` | `{k}` | `{item['positives']}` | `{item['positive_fraction']:.6f}` | `{item['sum_positive_delta_pair_f1']:.6f}` |"
                )
        lines.extend(["", "## Top Feature Correlations", "", "| feature | corr |", "| --- | ---: |"])
        for item in result["feature_correlations"][:12]:
            lines.append(f"| `{item['feature']}` | `{float(item['corr_with_positive']):.6f}` |")
        lines.extend(["", "## Top Oracle Delta Rows", "", "| delta pair F1 | source label | source rank | artifact |", "| ---: | ---: | ---: | --- |"])
        for row in result["top_rows_by_oracle_delta"][:8]:
            lines.append(
                f"| `{float(row['oracle_delta_pair_f1']):.6f}` | `{row['source_component_label']}` | `{row['source_rank']}` | `{row['artifact']}` |"
            )
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "rows": result["row_count"],
                "positives": result["positive_count"],
                "source_rank_ap": result["baseline_average_precision"]["source_rank_score"],
                "ridge_ap": result.get("ridge_logit", {}).get("loocv_average_precision"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
