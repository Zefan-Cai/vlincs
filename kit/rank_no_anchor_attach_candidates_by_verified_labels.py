#!/usr/bin/env python
"""Rank no-anchor attach candidates from previously verified local-edit labels.

This is an AutoResearch scheduler helper.  It never uses anchors or GT labels
as assignment evidence.  The optional labels are previous canonical full-score
outcomes used to learn which no-GT candidate features correlated with safe
local edits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_KEYS = [
    "scheduler_score",
    "critic_score",
    "source_count",
    "source_component_size",
    "target_count_in_time",
    "target_component_size",
    "time_component_size",
    "target_dominance",
    "source_fraction",
    "weak_video_source_fraction",
    "same_video_overlap_frames_max",
    "same_camera_overlap_frames_max",
    "source_avg_conf_mean",
    "source_n_dets_mean",
    "source_end_rank_frac_mean",
    "source_start_rank_frac_mean",
    "source_terminal_fraction",
    "source_prev_gap_min",
    "source_next_gap_min",
    "target_prev_gap_min",
    "target_next_gap_min",
    "target_gap_min",
    "target_same_video_fraction",
    "source_same_video_fraction",
    "source_video_entropy_norm",
]

LOG_FEATURES = {
    "source_component_size",
    "target_count_in_time",
    "target_component_size",
    "time_component_size",
    "source_n_dets_mean",
    "source_prev_gap_min",
    "source_next_gap_min",
    "target_prev_gap_min",
    "target_next_gap_min",
    "target_gap_min",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("rows")
    if not isinstance(rows, list):
        rows = data.get("selected")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has neither rows[] nor selected[]")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _preview_signature(row: dict[str, Any]) -> str:
    preview = row.get("accepted_preview")
    if not isinstance(preview, list) or not preview:
        return ""
    parts = []
    for item in preview:
        if not isinstance(item, dict):
            continue
        seqs = item.get("source_seqs")
        if not isinstance(seqs, list) or not seqs:
            continue
        target = item.get("target_component")
        if target in (None, ""):
            continue
        seq_text = "+".join(str(int(float(seq))) for seq in sorted(seqs, key=lambda value: int(float(value))))
        parts.append(f"{seq_text}->{int(float(target))}")
    return "|".join(sorted(parts))


def _parse_label(text: str) -> tuple[str, float]:
    left, sep, value = text.partition(":")
    if not sep:
        raise ValueError(f"bad --label {text!r}; expected signature:delta")
    return left.strip(), float(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _feature_value(row: dict[str, Any], key: str) -> float:
    value = _as_float(row.get(key), -1.0 if key.endswith("_gap_min") else 0.0)
    if key in LOG_FEATURES:
        if value < 0:
            return -1.0
        return math.log1p(value)
    return value


def _features(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([_feature_value(row, key) for key in FEATURE_KEYS], dtype=np.float64)


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std
    design = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return coef, mean, std


def _predict(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / std
    design = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    return design @ coef


def _loocv(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, Any]:
    preds = []
    for idx in range(len(y)):
        keep = np.asarray([j for j in range(len(y)) if j != idx], dtype=np.int64)
        coef, mean, std = _fit_ridge(x[keep], y[keep], ridge)
        pred = float(_predict(x[idx : idx + 1], coef, mean, std)[0])
        preds.append(pred)
    pred_arr = np.asarray(preds, dtype=np.float64)
    mae = float(np.mean(np.abs(pred_arr - y))) if len(y) else 0.0
    corr = float(np.corrcoef(pred_arr, y)[0, 1]) if len(y) >= 3 and np.std(pred_arr) > 1e-12 and np.std(y) > 1e-12 else 0.0
    return {
        "loocv_predictions": [round(float(v), 9) for v in pred_arr.tolist()],
        "loocv_mae": round(mae, 9),
        "loocv_corr": round(corr, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-json", required=True, help="candidate JSON to score")
    ap.add_argument("--training-candidate-json", action="append", required=True)
    ap.add_argument("--label", action="append", required=True, help="signature:delta_idf1, e.g. 4973->55:0.000037")
    ap.add_argument("--ridge", type=float, default=0.5)
    ap.add_argument("--selected-top-n", type=int, default=12)
    ap.add_argument("--include-labelled", action="store_true")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    labels = dict(_parse_label(text) for text in args.label)
    training_rows: dict[str, dict[str, Any]] = {}
    for path_text in args.training_candidate_json:
        for row in _load_rows(Path(path_text)):
            sig = _preview_signature(row)
            if sig and sig not in training_rows:
                training_rows[sig] = row
    missing = sorted(sig for sig in labels if sig not in training_rows)
    if missing:
        raise ValueError(f"labels not found in training rows: {missing}")
    train_sigs = sorted(labels)
    x_train = np.stack([_features(training_rows[sig]) for sig in train_sigs], axis=0)
    y_train = np.asarray([float(labels[sig]) for sig in train_sigs], dtype=np.float64)
    coef, mean, std = _fit_ridge(x_train, y_train, float(args.ridge))
    loocv = _loocv(x_train, y_train, float(args.ridge))

    rows = _load_rows(Path(args.candidate_json))
    scored = []
    seen = set()
    for row in rows:
        sig = _preview_signature(row)
        if not sig or sig in seen:
            continue
        seen.add(sig)
        if not args.include_labelled and sig in labels:
            continue
        pred = float(_predict(_features(row)[None, :], coef, mean, std)[0])
        out = dict(row)
        out["signature"] = sig
        out["learned_delta_idf1"] = round(pred, 9)
        out["learned_rank_score"] = round(pred, 9)
        out["ranker_feature_version"] = "tiny_attach_ridge_v1"
        scored.append(out)
    scored.sort(
        key=lambda row: (
            float(row.get("learned_rank_score", 0.0)),
            float(row.get("critic_score", 0.0)),
            float(row.get("scheduler_score", 0.0)),
        ),
        reverse=True,
    )
    selected = scored[: max(int(args.selected_top_n), 0)]
    for idx, row in enumerate(scored, start=1):
        row["learned_rank"] = int(idx)
    out = {
        "candidate_json": str(args.candidate_json),
        "training_candidate_jsons": [str(path) for path in args.training_candidate_json],
        "feature_keys": FEATURE_KEYS,
        "labels": {sig: round(float(value), 9) for sig, value in sorted(labels.items())},
        "training_signatures": train_sigs,
        "ridge": float(args.ridge),
        "coef": [round(float(v), 9) for v in coef.tolist()],
        **loocv,
        "rows": scored,
        "selected": selected,
        "selected_count": int(len(selected)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "note": "Full-score labels calibrate the scheduler only; assignment evidence remains no-anchor/no-GT.",
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        import csv

        fieldnames = [
            "learned_rank",
            "signature",
            "learned_delta_idf1",
            "critic_score",
            "scheduler_score",
            "source_component_label",
            "target_component",
            "source_count",
            "source_component_size",
            "target_count_in_time",
            "target_component_size",
            "target_dominance",
            "target_gap_min",
            "target_same_video_fraction",
            "source_avg_conf_mean",
            "source_terminal_fraction",
            "family",
        ]
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in scored:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(json.dumps({"json": str(out_path), "rows": len(scored), "selected_count": len(selected), **loocv}, sort_keys=True))


if __name__ == "__main__":
    main()
