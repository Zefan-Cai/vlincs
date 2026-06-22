#!/usr/bin/env python
"""Rank no-anchor attach candidates with global and per-video side-effect labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_KEYS = [
    "scheduler_score",
    "critic_score",
    "candidate_rank",
    "source_count",
    "source_component_size",
    "target_count_in_time",
    "target_component_size",
    "time_component_size",
    "target_dominance",
    "source_fraction",
    "weak_video_source_fraction",
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
    "candidate_rank",
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

WEAK_VIDEOS = [
    "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6",
    "vlincs_MS01_MC0001_MCAM06_2024-03-Tc6",
    "vlincs_MS01_MC0001_MCAM03_2024-03-Tc8",
    "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6",
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _feature_value(row: dict[str, Any], key: str) -> float:
    default = -1.0 if key.endswith("_gap_min") else 0.0
    value = _as_float(row.get(key), default)
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
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
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
        preds.append(float(_predict(x[idx : idx + 1], coef, mean, std)[0]))
    pred_arr = np.asarray(preds, dtype=np.float64)
    mae = float(np.mean(np.abs(pred_arr - y))) if len(y) else 0.0
    corr = float(np.corrcoef(pred_arr, y)[0, 1]) if len(y) >= 3 and np.std(pred_arr) > 1e-12 and np.std(y) > 1e-12 else 0.0
    return {
        "loocv_mae": round(mae, 9),
        "loocv_corr": round(corr, 6),
        "loocv_predictions": [round(float(v), 9) for v in pred_arr.tolist()],
    }


def _load_label_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        labels = data.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"{path} has no labels[]")
        rows.extend(dict(row) for row in labels if isinstance(row, dict))
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        sig = str(row.get("signature") or "")
        if sig:
            dedup[sig] = row
    return list(dedup.values())


def _load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("rows")
    if not isinstance(rows, list):
        rows = data.get("selected")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has neither rows[] nor selected[]")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _preview_signature(row: dict[str, Any]) -> str:
    sig = row.get("signature")
    if isinstance(sig, str) and sig:
        return sig
    preview = row.get("accepted_preview")
    if not isinstance(preview, list):
        return ""
    parts = []
    for item in preview:
        if not isinstance(item, dict):
            continue
        seqs = item.get("source_seqs")
        target = item.get("target_component")
        if not isinstance(seqs, list) or not seqs or target in (None, ""):
            continue
        seq_text = "+".join(str(int(float(seq))) for seq in sorted(seqs, key=lambda value: int(float(value))))
        parts.append(f"{seq_text}->{int(float(target))}")
    return "|".join(sorted(parts))


def _target_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    if key == "delta_idf1":
        return np.asarray([_as_float(row.get("delta_idf1")) for row in rows], dtype=np.float64)
    return np.asarray([
        _as_float(row.get("per_video_delta_idf1", {}).get(key)) if isinstance(row.get("per_video_delta_idf1"), dict) else 0.0
        for row in rows
    ], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-json", required=True)
    ap.add_argument("--label-json", action="append", required=True)
    ap.add_argument("--ridge", type=float, default=0.5)
    ap.add_argument("--selected-top-n", type=int, default=12)
    ap.add_argument("--global-weight", type=float, default=1.0)
    ap.add_argument("--weak-video-weight", type=float, default=0.35)
    ap.add_argument("--mcam04-weight", type=float, default=0.75)
    ap.add_argument("--negative-weak-penalty", type=float, default=0.75)
    ap.add_argument("--max-candidate-rank", type=int, default=0)
    ap.add_argument("--min-critic-score", type=float, default=-999.0)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    label_rows = _load_label_rows([Path(path) for path in args.label_json])
    if len(label_rows) < 4:
        raise ValueError("need at least four side-effect labels")
    x = np.stack([_features(row) for row in label_rows], axis=0)
    targets = {"delta_idf1": _target_array(label_rows, "delta_idf1")}
    for video in WEAK_VIDEOS:
        targets[video] = _target_array(label_rows, video)

    models = {}
    diagnostics = {}
    for key, y in targets.items():
        coef, mean, std = _fit_ridge(x, y, float(args.ridge))
        models[key] = (coef, mean, std)
        diagnostics[key] = _loocv(x, y, float(args.ridge))

    rows = _load_candidate_rows(Path(args.candidate_json))
    scored = []
    seen = set()
    for row in rows:
        sig = _preview_signature(row)
        if not sig or sig in seen:
            continue
        seen.add(sig)
        if int(args.max_candidate_rank) > 0 and _as_float(row.get("candidate_rank"), 999999.0) > int(args.max_candidate_rank):
            continue
        if _as_float(row.get("critic_score"), 0.0) < float(args.min_critic_score):
            continue
        feat = _features(row)[None, :]
        pred_global = float(_predict(feat, *models["delta_idf1"])[0])
        pred_videos = {video: float(_predict(feat, *models[video])[0]) for video in WEAK_VIDEOS}
        weak_sum = sum(pred_videos.values())
        neg_weak = sum(min(0.0, value) for value in pred_videos.values())
        score = (
            float(args.global_weight) * pred_global
            + float(args.weak_video_weight) * weak_sum
            + float(args.mcam04_weight) * pred_videos[WEAK_VIDEOS[0]]
            + float(args.negative_weak_penalty) * neg_weak
        )
        out = dict(row)
        out["signature"] = sig
        out["side_effect_rank_score"] = round(score, 9)
        out["pred_delta_idf1"] = round(pred_global, 9)
        for video, value in pred_videos.items():
            out[f"pred_delta_idf1__{video}"] = round(value, 9)
        out["ranker_feature_version"] = "tiny_attach_side_effect_ridge_v1"
        scored.append(out)

    scored.sort(
        key=lambda row: (
            float(row.get("side_effect_rank_score", 0.0)),
            float(row.get("pred_delta_idf1", 0.0)),
            float(row.get("critic_score", 0.0)),
            float(row.get("scheduler_score", 0.0)),
        ),
        reverse=True,
    )
    for idx, row in enumerate(scored, start=1):
        row["side_effect_rank"] = idx
    selected = scored[: max(int(args.selected_top_n), 0)]
    payload = {
        "candidate_json": args.candidate_json,
        "label_jsons": args.label_json,
        "feature_keys": FEATURE_KEYS,
        "weak_videos": WEAK_VIDEOS,
        "label_count": len(label_rows),
        "ridge": float(args.ridge),
        "diagnostics": diagnostics,
        "rows": scored,
        "selected": selected,
        "selected_count": len(selected),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "note": "Metric-derived side-effect labels calibrate scheduler only; assignments remain no-anchor/no-GT.",
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.csv:
        fieldnames = [
            "side_effect_rank",
            "signature",
            "side_effect_rank_score",
            "pred_delta_idf1",
            *(f"pred_delta_idf1__{video}" for video in WEAK_VIDEOS),
            "candidate_rank",
            "critic_score",
            "scheduler_score",
            "source_component_label",
            "target_component",
            "source_count",
            "source_component_size",
            "target_component_size",
            "weak_video_source_fraction",
            "target_same_video_fraction",
            "source_same_video_fraction",
            "target_gap_min",
            "family",
        ]
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in scored:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(json.dumps({"json": str(out_path), "selected_count": len(selected), "label_count": len(label_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()

