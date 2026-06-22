#!/usr/bin/env python3
"""Rank no-anchor subpart repair candidates from prior full-score labels.

The input labels are post-hoc full-score metrics.  They are used only to
calibrate the scheduler/referee, not to create anchors or train identity labels.
Candidate evidence is taken from no-anchor subpart manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "rank",
    "score",
    "moved_tracklets",
    "focus_video_hits",
    "focus_hit_frac",
    "source_component_size",
    "target_component_size",
    "log_source_component_size",
    "log_target_component_size",
    "source_component_conflict_edges",
    "conflicts_to_rest",
    "conflicts_per_moved",
    "target_rank_for_group",
    "target_sim",
    "target_margin",
    "abs_target_margin",
    "positive_target_margin",
    "group_internal_sim",
    "source_rest_cross_mean",
    "source_rest_cross_max",
    "source_rest_margin_mean",
    "source_rest_margin_max",
    "source_rest_margin_positive",
    "source_video_count",
    "source_video_dominance",
    "source_video_entropy",
    "small_move",
    "medium_move",
    "large_move",
]


def _load_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _csv_stem(path: Any) -> str:
    text = str(path or "")
    return Path(text).name


def _canonical_stem(value: Any) -> str:
    stem = _csv_stem(value)
    for suffix in (
        "_density_p005_area.json",
        "_density_simple.json",
        "_full_export.json",
        "_assignments.csv",
        ".csv",
        ".json",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _seq_signature(row: dict[str, Any]) -> str:
    seqs = row.get("source_seqs")
    if seqs is None and isinstance(row.get("moved_preview"), list):
        seqs = [item.get("seq") for item in row["moved_preview"]]
    if not isinstance(seqs, list):
        return ""
    try:
        values = sorted(_as_int(v) for v in seqs)
    except TypeError:
        return ""
    return ",".join(str(v) for v in values)


def _candidate_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        _as_int(row.get("source_component")),
        _as_int(row.get("target_component")),
        _seq_signature(row),
    )


def _video_stats(row: dict[str, Any]) -> tuple[float, float, float]:
    videos = row.get("source_videos")
    if not isinstance(videos, dict) or not videos:
        moved = max(_as_float(row.get("moved_tracklets")), 1.0)
        return 0.0, 0.0, 0.0
    counts = np.asarray([max(_as_float(v), 0.0) for v in videos.values()], dtype=float)
    total = float(counts.sum())
    if total <= 0.0:
        return float(len(counts)), 0.0, 0.0
    probs = counts / total
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    return float(len(counts)), float(counts.max() / total), entropy


def _feature_row(row: dict[str, Any]) -> dict[str, float]:
    moved = max(_as_float(row.get("moved_tracklets")), 0.0)
    focus_hits = _as_float(row.get("focus_video_hits"))
    source_size = _as_float(row.get("source_component_size"))
    target_size = _as_float(row.get("target_component_size"))
    conflicts = _as_float(row.get("conflicts_to_rest"))
    video_count, video_dom, video_entropy = _video_stats(row)
    target_margin = _as_float(row.get("target_margin"))
    source_rest_margin = _as_float(row.get("source_rest_margin_mean"))
    return {
        "rank": _as_float(row.get("rank")),
        "score": _as_float(row.get("score")),
        "moved_tracklets": moved,
        "focus_video_hits": focus_hits,
        "focus_hit_frac": focus_hits / max(moved, 1.0),
        "source_component_size": source_size,
        "target_component_size": target_size,
        "log_source_component_size": math.log1p(max(source_size, 0.0)),
        "log_target_component_size": math.log1p(max(target_size, 0.0)),
        "source_component_conflict_edges": _as_float(row.get("source_component_conflict_edges")),
        "conflicts_to_rest": conflicts,
        "conflicts_per_moved": conflicts / max(moved, 1.0),
        "target_rank_for_group": _as_float(row.get("target_rank_for_group")),
        "target_sim": _as_float(row.get("target_sim")),
        "target_margin": target_margin,
        "abs_target_margin": abs(target_margin),
        "positive_target_margin": 1.0 if target_margin > 0 else 0.0,
        "group_internal_sim": _as_float(row.get("group_internal_sim")),
        "source_rest_cross_mean": _as_float(row.get("source_rest_cross_mean")),
        "source_rest_cross_max": _as_float(row.get("source_rest_cross_max")),
        "source_rest_margin_mean": source_rest_margin,
        "source_rest_margin_max": _as_float(row.get("source_rest_margin_max")),
        "source_rest_margin_positive": 1.0 if source_rest_margin > 0 else 0.0,
        "source_video_count": video_count,
        "source_video_dominance": video_dom,
        "source_video_entropy": video_entropy,
        "small_move": 1.0 if moved <= 8 else 0.0,
        "medium_move": 1.0 if 8 < moved <= 20 else 0.0,
        "large_move": 1.0 if moved > 20 else 0.0,
    }


def _heuristic(row: dict[str, Any], baseline: float) -> float:
    f = _feature_row(row)
    raw = (
        0.00020 * max(f["target_sim"] - 0.60, 0.0)
        + 0.00012 * max(f["source_rest_margin_mean"], 0.0)
        + 0.00006 * max(f["group_internal_sim"] - 0.50, 0.0)
        - 0.00008 * max(f["abs_target_margin"] - 0.15, 0.0)
        - 0.00009 * f["large_move"]
        - 0.00003 * max(f["source_video_dominance"] - 0.85, 0.0)
    )
    return float(baseline + raw)


def _candidate_from_manifest(path: str | Path, row: dict[str, Any], pool: str, source_index: int) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("pool", pool)
    out["_manifest"] = str(path)
    out["_source_index"] = int(source_index)
    out["_csv_stem"] = _csv_stem(out.get("assignment_csv"))
    out["_candidate_key"] = list(_candidate_key(out))
    out["_seq_signature"] = _seq_signature(out)
    return out


def _load_candidates(manifest_paths: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, str]]] = set()
    for path in manifest_paths:
        data = _load_json(path)
        pool = str(Path(path).stem).replace("_manifest", "")
        rows = data.get("selected", [])
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cand = _candidate_from_manifest(path, row, str(row.get("pool") or pool), index)
            key = (str(cand.get("assignment_csv") or ""), _candidate_key(cand))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
    return candidates


def _load_summary_rows(summary_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        data = _load_json(path)
        current = data.get("results") or data.get("rows") or []
        if isinstance(current, dict):
            current = [current]
        if not isinstance(current, list):
            continue
        for row in current:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            if "assignment_csv" not in item and not item.get("stem") and not item.get("json"):
                continue
            item["_summary"] = str(path)
            item["_csv_stem"] = _canonical_stem(item.get("assignment_csv") or item.get("stem") or item.get("json"))
            rows.append(item)
    return rows


def _attach_labels(candidates: list[dict[str, Any]], summary_rows: list[dict[str, Any]], baseline: float) -> list[dict[str, Any]]:
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_simple: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        full_path = str(row.get("assignment_csv") or "")
        if full_path:
            by_path[full_path].append(row)
        stem = str(row.get("_csv_stem") or _canonical_stem(full_path))
        if stem:
            by_stem[stem].append(row)
        simple = (
            _as_int(row.get("source_component")),
            _as_int(row.get("target_component")),
            _as_int(row.get("moved_tracklets")),
        )
        if simple[0] or simple[1] or simple[2]:
            by_simple[simple].append(row)

    labeled: list[dict[str, Any]] = []
    for cand in candidates:
        matches = by_path.get(str(cand.get("assignment_csv") or ""), [])
        if not matches:
            matches = by_stem.get(_canonical_stem(cand.get("_csv_stem") or cand.get("assignment_csv")), [])
        if not matches:
            simple = (
                _as_int(cand.get("source_component")),
                _as_int(cand.get("target_component")),
                _as_int(cand.get("moved_tracklets")),
            )
            matches = by_simple.get(simple, [])
        if not matches:
            continue
        idf1_values = [_as_float(row.get("idf1", row.get("full_idf1")), float("nan")) for row in matches]
        idf1_values = [v for v in idf1_values if not math.isnan(v)]
        if not idf1_values:
            continue
        out = dict(cand)
        out["_label_idf1"] = float(np.mean(idf1_values))
        out["_label_delta_vs_baseline"] = float(out["_label_idf1"] - baseline)
        out["_label_count"] = int(len(idf1_values))
        out["_label_summaries"] = sorted({str(row.get("_summary")) for row in matches})
        labeled.append(out)
    return labeled


def _matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(_feature_row(row)[name]) for name in NUMERIC_FEATURES] for row in rows], dtype=float)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _fit_model(labeled: list[dict[str, Any]], baseline: float) -> tuple[Any | None, dict[str, Any]]:
    if len(labeled) < 4:
        return None, {"status": "too_few_labels", "labeled_count": len(labeled)}
    x = _matrix(labeled)
    y = np.asarray([_as_float(row.get("_label_idf1"), baseline) for row in labeled], dtype=float)
    alphas = np.asarray([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=None))
    model.fit(x, y)
    fit_pred = np.asarray(model.predict(x), dtype=float)

    loo_pred = []
    for index in range(len(labeled)):
        train_idx = [i for i in range(len(labeled)) if i != index]
        sub = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=None))
        sub.fit(x[train_idx], y[train_idx])
        loo_pred.append(float(sub.predict(x[index : index + 1])[0]))
    loo = np.asarray(loo_pred, dtype=float)

    coefs = model.named_steps["ridgecv"].coef_.tolist()
    feature_weights = sorted(
        [{"feature": name, "coef": float(coef)} for name, coef in zip(NUMERIC_FEATURES, coefs)],
        key=lambda item: abs(float(item["coef"])),
        reverse=True,
    )
    diagnostics = {
        "status": "fit",
        "labeled_count": int(len(labeled)),
        "baseline_idf1": float(baseline),
        "label_idf1_min": float(y.min()),
        "label_idf1_mean": float(y.mean()),
        "label_idf1_max": float(y.max()),
        "fit_rmse": float(np.sqrt(np.mean((fit_pred - y) ** 2))),
        "loocv_rmse": float(np.sqrt(np.mean((loo - y) ** 2))),
        "loocv_corr": _corr(loo, y),
        "loocv_rank_corr": _corr(_rankdata(loo), _rankdata(y)),
        "alpha": float(model.named_steps["ridgecv"].alpha_),
        "feature_weights": feature_weights[:12],
    }
    return model, diagnostics


def _score_candidates(candidates: list[dict[str, Any]], model: Any | None, baseline: float) -> list[dict[str, Any]]:
    if model is not None and candidates:
        pred = np.asarray(model.predict(_matrix(candidates)), dtype=float)
    else:
        pred = np.asarray([_heuristic(row, baseline) for row in candidates], dtype=float)
    scored: list[dict[str, Any]] = []
    for row, model_pred in zip(candidates, pred.tolist()):
        heuristic = _heuristic(row, baseline)
        conservative = min(float(model_pred), float(heuristic) + 0.00012)
        out = dict(row)
        out["predicted_idf1"] = float(model_pred)
        out["heuristic_idf1"] = float(heuristic)
        out["referee_score"] = float(0.65 * conservative + 0.35 * heuristic)
        out["predicted_delta_vs_baseline"] = float(out["referee_score"] - baseline)
        out["is_labeled"] = "_label_idf1" in out
        scored.append(out)
    scored.sort(
        key=lambda row: (
            float(row["referee_score"]),
            float(row.get("score", 0.0)),
            -float(row.get("rank", 0.0)),
        ),
        reverse=True,
    )
    return scored


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "rank",
        "pool",
        "source_component",
        "target_component",
        "moved_tracklets",
        "focus_video_hits",
        "score",
        "target_sim",
        "target_margin",
        "group_internal_sim",
        "source_rest_margin_mean",
        "source_rest_cross_max",
        "conflicts_to_rest",
        "assignment_csv",
        "predicted_idf1",
        "heuristic_idf1",
        "referee_score",
        "predicted_delta_vs_baseline",
        "is_labeled",
        "_label_idf1",
        "_label_delta_vs_baseline",
        "_manifest",
    ]
    return {key: row[key] for key in keep if key in row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Candidate manifest JSON. Repeatable.")
    parser.add_argument("--label-summary", action="append", required=True, help="Full-score summary JSON. Repeatable.")
    parser.add_argument("--baseline-idf1", type=float, default=0.657887)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    candidates = _load_candidates(args.manifest)
    summary_rows = _load_summary_rows(args.label_summary)
    labeled = _attach_labels(candidates, summary_rows, float(args.baseline_idf1))
    model, diagnostics = _fit_model(labeled, float(args.baseline_idf1))
    scored = _score_candidates(candidates, model, float(args.baseline_idf1))
    unlabeled_ranked = [row for row in scored if not row.get("is_labeled")]
    labeled_ranked = [row for row in scored if row.get("is_labeled")]

    result = {
        "candidate_count": int(len(candidates)),
        "summary_label_rows": int(len(summary_rows)),
        "matched_labeled_candidates": int(len(labeled)),
        "baseline_idf1": float(args.baseline_idf1),
        "feature_names": NUMERIC_FEATURES,
        "diagnostics": diagnostics,
        "top_unlabeled": [_compact(row) for row in unlabeled_ranked[: max(0, int(args.top_n))]],
        "top_labeled": [_compact(row) for row in labeled_ranked[: max(0, min(int(args.top_n), 40))]],
        "all_ranked_compact": [_compact(row) for row in scored],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "label_usage_note": "Full-score labels calibrate the no-anchor scheduler/referee only.",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "order",
            "is_labeled",
            "rank",
            "pool",
            "source_component",
            "target_component",
            "moved_tracklets",
            "focus_video_hits",
            "score",
            "target_sim",
            "target_margin",
            "referee_score",
            "predicted_delta_vs_baseline",
            "_label_idf1",
            "assignment_csv",
            "_manifest",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(scored, start=1):
                compact = _compact(row)
                writer.writerow({field: compact.get(field, index if field == "order" else "") for field in fields})

    print(
        json.dumps(
            {
                "output_json": str(out),
                "candidate_count": len(candidates),
                "matched_labeled_candidates": len(labeled),
                "top_unlabeled": len(unlabeled_ranked[: max(0, int(args.top_n))]),
                "diagnostics": diagnostics,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
