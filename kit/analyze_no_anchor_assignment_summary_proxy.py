#!/usr/bin/env python
"""Train and apply a no-GT assignment-summary full-IDF1 proxy.

The proxy uses only assignment CSV summary features: component counts, component
size distribution, detector confidence summaries, status counts, video-level
imbalance, and optional delta to a base assignment.  Full-IDF1 labels come only
from already completed scoring artifacts and are treated as post-hoc labels.

This is a scheduler/referee aid, not completion evidence.  Oracle/GT-selected
rows are excluded from training by default.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.no_anchor_fullscore_scheduler import _oracle_or_gt_selection_reason


METRIC_KEYS = ("full_idf1", "idf1")
ROW_KEYS = ("top", "rows", "full_rows", "top_full_rows", "outputs")
FORBIDDEN_TEXT = ("oracle", "with_gt", "pervideo_filter_oracle")


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _metric(row: dict[str, Any]) -> float | None:
    for key in METRIC_KEYS:
        value = _as_float(row.get(key))
        if value is not None:
            return value / 100.0 if value > 1.5 else value
    return None


def _find_local_csv(path_text: Any) -> str | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.is_file():
        return str(path)
    name = path.name
    if not name:
        return None
    hits = sorted(Path("local_runs").glob(f"**/{name}"))
    return str(hits[0]) if hits else None


def _row_assignment_path(row: dict[str, Any], *, allow_top_level_assignment: bool) -> str | None:
    keys = ("assignments_out", "output_csv", "assignment_csv")
    for key in keys:
        local = _find_local_csv(row.get(key))
        if local:
            return local
    if allow_top_level_assignment:
        for key in keys:
            local = _find_local_csv(row.get(key))
            if local:
                return local
    return None


def _blocked(row: dict[str, Any], source_path: Path) -> str | None:
    reason = _oracle_or_gt_selection_reason(row)
    if reason:
        return str(reason)
    mode = str(row.get("mode") or row.get("policy_name") or "").lower()
    text = f"{source_path} {mode}".lower()
    if any(token in text for token in FORBIDDEN_TEXT):
        return "oracle_or_gt_path_or_mode"
    return None


def _iter_label_rows(paths: list[str]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for pattern in paths:
        if any(ch in pattern for ch in "*?[]"):
            matches = [Path(item) for item in sorted(glob.glob(pattern, recursive=True))]
        else:
            matches = [Path(pattern)]
        for path in matches:
            if not path.is_file() or path.suffix.lower() != ".json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            top_metric = _metric(data)
            if top_metric is not None and not _blocked(data, path):
                local = _row_assignment_path(data, allow_top_level_assignment=True)
                if local:
                    labels.append(
                        {
                            "assignment_csv": local,
                            "full_idf1": float(top_metric),
                            "source_json": str(path),
                            "source_kind": "top_level",
                            "mode": data.get("mode"),
                        }
                    )
            for key in ROW_KEYS:
                rows = data.get(key)
                if not isinstance(rows, list):
                    continue
                for rank, row in enumerate(rows, start=1):
                    if not isinstance(row, dict):
                        continue
                    metric = _metric(row)
                    if metric is None or _blocked(row, path):
                        continue
                    local = _row_assignment_path(row, allow_top_level_assignment=False)
                    if local:
                        labels.append(
                            {
                                "assignment_csv": local,
                                "full_idf1": float(metric),
                                "source_json": str(path),
                                "source_kind": key,
                                "source_rank": rank,
                                "mode": row.get("mode"),
                            }
                        )
    return labels


def _series_stat(values: pd.Series, name: str, out: dict[str, float]) -> None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) == 0:
        out[f"{name}_mean"] = 0.0
        out[f"{name}_median"] = 0.0
        out[f"{name}_p90"] = 0.0
        return
    out[f"{name}_mean"] = float(vals.mean())
    out[f"{name}_median"] = float(vals.median())
    out[f"{name}_p90"] = float(vals.quantile(0.9))


def _features(path: str, base_df: pd.DataFrame | None = None) -> dict[str, float]:
    df = pd.read_csv(path)
    if "tracklet_key" not in df.columns:
        df["tracklet_key"] = df.get("seq", pd.Series(range(len(df)))).astype(str)
    if "video" not in df.columns:
        df["video"] = ""
    df["tracklet_key"] = df["tracklet_key"].astype(str)
    df["video"] = df["video"].astype(str)
    gid = pd.to_numeric(df["predicted_global_id"], errors="coerce").fillna(-1).astype(int)
    counts = gid.value_counts()
    video_sizes = df.groupby("video")["tracklet_key"].nunique()
    video_components = df.assign(_gid=gid).groupby("video")["_gid"].nunique()
    out: dict[str, float] = {
        "rows": float(len(df)),
        "tracklets": float(df["tracklet_key"].nunique()),
        "videos": float(df["video"].nunique()),
        "predicted_ids": float(gid.nunique()),
        "component_size_mean_actual": float(counts.mean()) if len(counts) else 0.0,
        "component_size_median_actual": float(counts.median()) if len(counts) else 0.0,
        "component_size_p90_actual": float(counts.quantile(0.9)) if len(counts) else 0.0,
        "largest_component": float(counts.max()) if len(counts) else 0.0,
        "largest_component_ratio": float(counts.max() / max(len(df), 1)) if len(counts) else 0.0,
        "singleton_component_fraction": float((counts == 1).mean()) if len(counts) else 0.0,
        "singleton_tracklet_ratio": float(counts[counts == 1].sum() / max(len(df), 1)) if len(counts) else 0.0,
        "video_rows_std": float(video_sizes.std(ddof=0)) if len(video_sizes) else 0.0,
        "video_rows_min": float(video_sizes.min()) if len(video_sizes) else 0.0,
        "video_rows_max": float(video_sizes.max()) if len(video_sizes) else 0.0,
        "video_components_std": float(video_components.std(ddof=0)) if len(video_components) else 0.0,
        "video_components_max": float(video_components.max()) if len(video_components) else 0.0,
    }
    for col in (
        "component_size",
        "avg_conf",
        "n_dets",
        "prediction_confidence",
        "component_internal_prob_median",
        "component_internal_score_median",
        "component_external_prob_max",
        "component_margin_prob",
    ):
        if col in df.columns:
            _series_stat(df[col], col, out)
    statuses = Counter(str(x) for x in df.get("decision_status", pd.Series(dtype=object)).fillna("").tolist())
    total = max(len(df), 1)
    for status in (
        "forced_component",
        "forced_singleton",
        "manifest_reassign",
        "pervideo_source_select",
        "base_fallback",
    ):
        out[f"status_{status}_ratio"] = float(statuses.get(status, 0) / total)

    if base_df is not None and len(base_df):
        base = base_df.copy()
        base["tracklet_key"] = base["tracklet_key"].astype(str)
        base_gid = dict(
            zip(
                base["tracklet_key"],
                pd.to_numeric(base["predicted_global_id"], errors="coerce").fillna(-1).astype(int),
            )
        )
        changed = 0
        overlap = 0
        for key, value in zip(df["tracklet_key"], gid):
            if key in base_gid:
                overlap += 1
                changed += int(base_gid[key] != int(value))
        out["base_overlap_ratio"] = float(overlap / max(len(df), 1))
        out["changed_ratio_from_base"] = float(changed / max(overlap, 1))
        out["predicted_id_delta_from_base"] = float(gid.nunique() - base["predicted_global_id"].nunique())
    else:
        out["base_overlap_ratio"] = 0.0
        out["changed_ratio_from_base"] = 0.0
        out["predicted_id_delta_from_base"] = 0.0
    return out


def _candidate_paths(patterns: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            matches = [Path(item) for item in sorted(glob.glob(pattern, recursive=True))]
        else:
            matches = [Path(pattern)]
        for path in matches:
            if path.is_file() and path.suffix.lower() == ".csv":
                text = str(path)
                if text not in seen:
                    seen.add(text)
                    out.append(text)
    return out


def _fit_ridge(rows: list[dict[str, Any]], columns: list[str], alpha: float) -> dict[str, Any]:
    x = np.asarray([[float(row["features"].get(col, 0.0)) for col in columns] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["full_idf1"]) for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    xs = (x - mean) / scale
    design = np.concatenate([np.ones((len(xs), 1)), xs], axis=1)
    reg = np.eye(design.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    pred = design @ coef
    return {
        "alpha": float(alpha),
        "columns": columns,
        "intercept": float(coef[0]),
        "coef": [float(v) for v in coef[1:]],
        "mean": [float(v) for v in mean],
        "scale": [float(v) for v in scale],
        "train_mae": float(np.mean(np.abs(pred - y))),
        "train_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
    }


def _predict(features: dict[str, float], model: dict[str, Any]) -> float:
    score = float(model["intercept"])
    for col, coef, mean, scale in zip(model["columns"], model["coef"], model["mean"], model["scale"]):
        score += float(coef) * ((float(features.get(col, 0.0)) - float(mean)) / max(float(scale), 1.0e-9))
    return float(score)


def _loocv(rows: list[dict[str, Any]], columns: list[str], alphas: list[float]) -> dict[str, Any]:
    if len(rows) < 3:
        return {"alpha": alphas[0], "mae": None, "rmse": None, "corr": None, "predictions": []}
    best: dict[str, Any] | None = None
    for alpha in alphas:
        preds = []
        labels = []
        for idx in range(len(rows)):
            train = [row for j, row in enumerate(rows) if j != idx]
            model = _fit_ridge(train, columns, alpha)
            preds.append(_predict(rows[idx]["features"], model))
            labels.append(float(rows[idx]["full_idf1"]))
        pred_arr = np.asarray(preds)
        label_arr = np.asarray(labels)
        mae = float(np.mean(np.abs(pred_arr - label_arr)))
        rmse = float(np.sqrt(np.mean((pred_arr - label_arr) ** 2)))
        corr = float(np.corrcoef(pred_arr, label_arr)[0, 1]) if np.std(pred_arr) > 1e-12 and np.std(label_arr) > 1e-12 else 0.0
        item = {"alpha": float(alpha), "mae": mae, "rmse": rmse, "corr": corr, "predictions": preds}
        if best is None or (mae, rmse) < (best["mae"], best["rmse"]):
            best = item
    return best or {"alpha": alphas[0], "mae": None, "rmse": None, "corr": None, "predictions": []}


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_df = pd.read_csv(args.base_assignment_csv) if args.base_assignment_csv else None
    labels_raw = _iter_label_rows(args.label_json)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels_raw:
        grouped[row["assignment_csv"]].append(row)
    label_rows: list[dict[str, Any]] = []
    for path, vals in sorted(grouped.items()):
        # Use the best observed non-oracle score for identical assignment bytes.
        best = max(vals, key=lambda row: float(row["full_idf1"]))
        feats = _features(path, base_df)
        label_rows.append({**best, "features": feats, "label_observations": len(vals)})

    if not label_rows:
        raise RuntimeError("no labeled assignment rows resolved")
    columns = sorted(set().union(*(row["features"].keys() for row in label_rows)))
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    cv = _loocv(label_rows, columns, alphas)
    model = _fit_ridge(label_rows, columns, float(cv["alpha"]))

    candidate_paths = _candidate_paths(args.candidate_assignment)
    candidates = []
    labeled_by_path = {row["assignment_csv"]: row for row in label_rows}
    skipped_candidates = []
    for path in candidate_paths:
        feats = _features(path, base_df)
        if feats["tracklets"] < float(args.min_candidate_tracklets) or feats["videos"] < float(args.min_candidate_videos):
            skipped_candidates.append(
                {
                    "assignment_csv": path,
                    "tracklets": feats["tracklets"],
                    "videos": feats["videos"],
                    "reason": "below_full_ds1_coverage",
                }
            )
            continue
        pred = _predict(feats, model)
        label = labeled_by_path.get(path)
        candidates.append(
            {
                "assignment_csv": path,
                "predicted_full_idf1": float(pred),
                "known_full_idf1": float(label["full_idf1"]) if label else None,
                "is_labeled": bool(label),
                "features": feats,
            }
        )
    candidates.sort(key=lambda row: float(row["predicted_full_idf1"]), reverse=True)
    result = {
        "label_json_inputs": args.label_json,
        "candidate_assignment_inputs": args.candidate_assignment,
        "base_assignment_csv": args.base_assignment_csv,
        "label_observations_raw": len(labels_raw),
        "training_rows": len(label_rows),
        "candidate_rows": len(candidates),
        "skipped_candidate_rows": len(skipped_candidates),
        "skipped_candidates": skipped_candidates[: int(args.top_n)],
        "min_candidate_tracklets": float(args.min_candidate_tracklets),
        "min_candidate_videos": float(args.min_candidate_videos),
        "columns": columns,
        "ridge_loocv": cv,
        "ridge_model": model,
        "training": [
            {
                "assignment_csv": row["assignment_csv"],
                "full_idf1": row["full_idf1"],
                "mode": row.get("mode"),
                "source_json": row.get("source_json"),
                "label_observations": row.get("label_observations"),
                "predicted_full_idf1": _predict(row["features"], model),
            }
            for row in label_rows
        ],
        "top_candidates": candidates[: int(args.top_n)],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_posthoc_full_score_labels": True,
        "note": "Assignment-summary proxy only; selected rows still require canonical full scoring.",
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), candidates[: int(args.top_n)])
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "assignment_csv",
        "predicted_full_idf1",
        "known_full_idf1",
        "is_labeled",
        "features_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "assignment_csv": row["assignment_csv"],
                    "predicted_full_idf1": row["predicted_full_idf1"],
                    "known_full_idf1": row["known_full_idf1"],
                    "is_labeled": row["is_labeled"],
                    "features_json": json.dumps(row["features"], sort_keys=True),
                }
            )


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Assignment Summary Proxy",
        "",
        "This is a no-GT scheduler/referee aid. It is not completion evidence.",
        "",
        f"- training rows: `{result['training_rows']}`",
        f"- raw label observations: `{result['label_observations_raw']}`",
        f"- candidate assignments: `{result['candidate_rows']}`",
        f"- skipped candidate assignments: `{result['skipped_candidate_rows']}`",
        f"- min candidate tracklets/videos: `{result['min_candidate_tracklets']}` / `{result['min_candidate_videos']}`",
        f"- LOOCV alpha: `{result['ridge_loocv']['alpha']}`",
        f"- LOOCV MAE: `{result['ridge_loocv']['mae']}`",
        f"- LOOCV RMSE: `{result['ridge_loocv']['rmse']}`",
        f"- LOOCV corr: `{result['ridge_loocv']['corr']}`",
        "",
        "## Top Candidates",
        "",
        "| rank | predicted full IDF1 | known full IDF1 | labeled | assignment |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for idx, row in enumerate(result["top_candidates"][:30], start=1):
        lines.append(
            f"| {idx} | `{row['predicted_full_idf1']:.6f}` | `{row.get('known_full_idf1')}` | "
            f"`{str(row['is_labeled']).lower()}` | `{row['assignment_csv']}` |"
        )
    lines.extend(["", "## Training Labels", ""])
    for row in sorted(result["training"], key=lambda item: float(item["full_idf1"]), reverse=True):
        lines.append(
            f"- `{row['full_idf1']:.6f}` predicted `{row['predicted_full_idf1']:.6f}` "
            f"from `{row['assignment_csv']}` ({row.get('mode')})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base.csv"
        one = root / "one.csv"
        two = root / "two.csv"
        rows = [
            {"seq": 1, "tracklet_key": "a", "video": "v", "predicted_global_id": 1, "avg_conf": 0.9, "decision_status": "forced_component"},
            {"seq": 2, "tracklet_key": "b", "video": "v", "predicted_global_id": 2, "avg_conf": 0.8, "decision_status": "forced_component"},
            {"seq": 3, "tracklet_key": "c", "video": "v", "predicted_global_id": 3, "avg_conf": 0.7, "decision_status": "forced_singleton"},
        ]
        pd.DataFrame(rows).to_csv(base, index=False)
        pd.DataFrame(rows).to_csv(one, index=False)
        alt = [dict(row) for row in rows]
        alt[1]["predicted_global_id"] = 1
        pd.DataFrame(alt).to_csv(two, index=False)
        labels = root / "labels.json"
        labels.write_text(
            json.dumps(
                {
                    "top": [
                        {"assignments_out": str(one), "full_idf1": 0.61, "mode": "clean"},
                        {"assignments_out": str(two), "full_idf1": 0.63, "mode": "merge"},
                        {"assignments_out": str(two), "full_idf1": 0.99, "mode": "oracle_bad"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = run(
            argparse.Namespace(
                label_json=[str(labels)],
                candidate_assignment=[str(root / "*.csv")],
                base_assignment_csv=str(base),
                alphas="0.1,1.0",
                min_candidate_tracklets=1,
                min_candidate_videos=1,
                top_n=5,
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
                md=str(root / "out.md"),
            )
        )
        assert out["training_rows"] == 2, out
        assert out["candidate_rows"] == 3, out
        assert "Top Candidates" in Path(root / "out.md").read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-json", action="append", default=["local_runs/remote_h100_test_3_20260620/*.json", "local_runs/remote_h100_test_3_20260619/*.json"])
    ap.add_argument("--candidate-assignment", action="append", default=["local_runs/**/*assignments*.csv"])
    ap.add_argument("--base-assignment-csv", default="local_runs/remote_h100_test_3_20260620/no_anchor_recovered_softcut_then_softoverlap_base_assignments_20260620.csv")
    ap.add_argument("--alphas", default="0.01,0.1,1.0,10.0,100.0")
    ap.add_argument("--min-candidate-tracklets", type=float, default=7000.0)
    ap.add_argument("--min-candidate-videos", type=float, default=10.0)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--json", default="local_runs/no_anchor_assignment_summary_proxy_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_assignment_summary_proxy_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_assignment_summary_proxy_20260620.md")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    result = run(args)
    print(
        json.dumps(
            {
                "training_rows": result["training_rows"],
                "candidate_rows": result["candidate_rows"],
                "top_predicted_full_idf1": result["top_candidates"][0]["predicted_full_idf1"] if result["top_candidates"] else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
