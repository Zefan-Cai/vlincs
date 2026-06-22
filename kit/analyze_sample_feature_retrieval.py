#!/usr/bin/env python
"""Eval-only feature retrieval diagnostics for no-anchor sample tracklets.

This script does not train a model and does not create predictions.  It uses
GT labels only after feature extraction to answer whether same-identity
tracklets are present in the nearest-neighbor evidence pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - keeps the diagnostic usable in lean envs.
    average_precision_score = None
    roc_auc_score = None


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_weight_map(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in _parse_csv(text):
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            raise ValueError(f"bad feature weight entry {part!r}; expected key=value")
        out[key.strip()] = float(value)
    return out


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)).astype(np.float32)


def _majority_label(group: pd.DataFrame, gt_col: str) -> tuple[str, float, int]:
    labels = group[gt_col].astype(str)
    counts = labels.value_counts(dropna=True)
    if counts.empty:
        return "UNKNOWN", 0.0, int(len(group))
    return str(counts.index[0]), float(counts.iloc[0] / max(len(group), 1)), int(len(group))


def _load_labels(parquet_path: Path, gt_col: str, eval_min_gt_fraction: float, eval_min_rows: int) -> dict[str, dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    if "tracklet_key" not in df.columns:
        raise ValueError(f"{parquet_path} missing tracklet_key")
    if "tracklet_majority_gt_id" in df.columns:
        label_col = "tracklet_majority_gt_id"
        frac_col = "tracklet_majority_gt_fraction"
        agg = (
            df.groupby("tracklet_key", sort=False)
            .agg(
                label=(label_col, "first"),
                frac=(frac_col, "first"),
                rows=("frame_idx", "size"),
                video_key=("video_key", "first"),
            )
            .reset_index()
        )
    else:
        if gt_col not in df.columns:
            raise ValueError(f"{parquet_path} missing {gt_col!r} and tracklet_majority_gt_id")
        rows = []
        for key, group in df.groupby("tracklet_key", sort=False):
            label, frac, n_rows = _majority_label(group, gt_col)
            rows.append(
                {
                    "tracklet_key": str(key),
                    "label": label,
                    "frac": frac,
                    "rows": n_rows,
                    "video_key": str(group["video_key"].iloc[0]),
                }
            )
        agg = pd.DataFrame(rows)
    out: dict[str, dict[str, Any]] = {}
    for row in agg.itertuples(index=False):
        label = str(row.label)
        frac = float(row.frac)
        n_rows = int(row.rows)
        if label in {"", "UNKNOWN", "nan", "None"}:
            continue
        if frac < float(eval_min_gt_fraction) or n_rows < int(eval_min_rows):
            continue
        out[str(row.tracklet_key)] = {
            "label": label,
            "frac": frac,
            "rows": n_rows,
            "video_key": str(row.video_key),
        }
    return out


def _load_features(npz_path: Path, feature_keys: list[str], weights: dict[str, float]) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(data["metadata"].item()))
    records = list(meta["records"])
    blocks = []
    valid = np.ones((len(records),), dtype=bool)
    for key in feature_keys:
        if key not in data.files:
            raise KeyError(f"{npz_path} missing feature key {key!r}; has {data.files}")
        block = _l2n(data[key].astype(np.float32)) * float(weights.get(key, 1.0))
        blocks.append(block)
        valid_key = "valid_" + key.replace("features_", "")
        if valid_key in data.files:
            valid &= data[valid_key].astype(bool)
    emb = _l2n(np.concatenate(blocks, axis=1).astype(np.float32))
    valid &= np.isfinite(emb).all(axis=1)
    return emb[valid], [record for record, keep in zip(records, valid) if bool(keep)], meta


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return {f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in qs}


def _topk_metrics(sim: np.ndarray, labels: np.ndarray, videos: np.ndarray, ks: list[int], *, exclude_same_video: bool) -> dict[str, Any]:
    n = int(len(labels))
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    allowed = np.ones((n, n), dtype=bool)
    np.fill_diagonal(allowed, False)
    if exclude_same_video:
        allowed &= videos[:, None] != videos[None, :]
    positive_allowed = same & allowed
    has_positive = positive_allowed.any(axis=1)
    candidate_counts = allowed.sum(axis=1)
    order_scores = sim.copy()
    order_scores[~allowed] = -np.inf
    order = np.argsort(-order_scores, axis=1)
    out: dict[str, Any] = {
        "queries": n,
        "queries_with_positive": int(has_positive.sum()),
        "exclude_same_video": bool(exclude_same_video),
        "directed_true_pairs": int(positive_allowed.sum()),
    }
    top_pos_sim = np.full((n,), np.nan, dtype=np.float32)
    top_neg_sim = np.full((n,), np.nan, dtype=np.float32)
    for i in range(n):
        pos_mask = positive_allowed[i]
        neg_mask = allowed[i] & (~same[i])
        if pos_mask.any():
            top_pos_sim[i] = float(np.max(sim[i, pos_mask]))
        if neg_mask.any():
            top_neg_sim[i] = float(np.max(sim[i, neg_mask]))
    valid_margin = np.isfinite(top_pos_sim) & np.isfinite(top_neg_sim)
    if valid_margin.any():
        margins = top_pos_sim[valid_margin] - top_neg_sim[valid_margin]
        out["top_positive_minus_top_negative"] = {
            "mean": float(np.mean(margins)),
            "median": float(np.median(margins)),
            "positive_fraction": float(np.mean(margins > 0.0)),
            "quantiles": _quantiles(margins),
        }
    for k in ks:
        hits = []
        true_seen = 0
        pred_seen = 0
        for i in range(n):
            take = min(int(k), int(candidate_counts[i]))
            if take <= 0:
                continue
            top = order[i, :take]
            true_flags = positive_allowed[i, top]
            pred_seen += int(take)
            true_seen += int(true_flags.sum())
            if has_positive[i]:
                hits.append(bool(true_flags.any()))
        out[f"hit_recall_at_{k}"] = float(np.mean(hits)) if hits else None
        out[f"directed_pair_recall_at_{k}"] = float(true_seen / max(int(positive_allowed.sum()), 1))
        out[f"precision_at_{k}"] = float(true_seen / max(pred_seen, 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracklet-parquet", required=True)
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--feature-keys", default="features_osnet,features_color")
    ap.add_argument("--feature-key-weights", default="features_osnet=1.0,features_color=0.1")
    ap.add_argument("--gt-col", default="gt_id")
    ap.add_argument("--eval-min-gt-fraction", type=float, default=0.50)
    ap.add_argument("--eval-min-rows", type=int, default=1)
    ap.add_argument("--top-ks", default="1,5,10,20,45,100")
    ap.add_argument("--max-pair-samples", type=int, default=1000000)
    ap.add_argument("--random-state", type=int, default=17)
    ap.add_argument("--json", required=True)
    ap.add_argument("--report-md", default=None)
    args = ap.parse_args()

    feature_keys = _parse_csv(args.feature_keys)
    weights = _parse_weight_map(args.feature_key_weights)
    ks = [int(k) for k in _parse_csv(args.top_ks)]
    emb, records, meta = _load_features(Path(args.feature_npz), feature_keys, weights)
    labels_by_key = _load_labels(Path(args.tracklet_parquet), args.gt_col, float(args.eval_min_gt_fraction), int(args.eval_min_rows))

    keep = []
    labels = []
    videos = []
    rows = []
    for i, record in enumerate(records):
        key = str(record["tracklet_key"])
        label_info = labels_by_key.get(key)
        if not label_info:
            continue
        keep.append(i)
        labels.append(str(label_info["label"]))
        videos.append(str(label_info["video_key"]))
        rows.append(int(label_info["rows"]))
    emb = emb[np.asarray(keep, dtype=np.int64)]
    labels_arr = np.asarray(labels, dtype=object)
    videos_arr = np.asarray(videos, dtype=object)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)

    n = len(labels_arr)
    iu = np.triu_indices(n, k=1)
    pair_labels = labels_arr[iu[0]] == labels_arr[iu[1]]
    pair_scores = sim[iu]
    finite = np.isfinite(pair_scores)
    pair_labels = pair_labels[finite]
    pair_scores = pair_scores[finite]
    rng = np.random.default_rng(int(args.random_state))
    if len(pair_scores) > int(args.max_pair_samples):
        idx = rng.choice(len(pair_scores), size=int(args.max_pair_samples), replace=False)
        pair_labels_eval = pair_labels[idx]
        pair_scores_eval = pair_scores[idx]
        pair_sampled = True
    else:
        pair_labels_eval = pair_labels
        pair_scores_eval = pair_scores
        pair_sampled = False

    result: dict[str, Any] = {
        "tracklet_parquet": str(args.tracklet_parquet),
        "feature_npz": str(args.feature_npz),
        "feature_keys": feature_keys,
        "feature_key_weights": weights,
        "eval_tracklets": int(n),
        "unique_gt_ids": int(len(set(labels))),
        "rows_total": int(sum(rows)),
        "feature_metadata_model": meta.get("model"),
        "uses_anchors": bool(meta.get("uses_anchors", False)),
        "uses_gt_for_training_or_anchors": bool(meta.get("uses_gt_for_training_or_anchors", False)),
        "same_pair_count": int(pair_labels.sum()),
        "diff_pair_count": int((~pair_labels).sum()),
        "pair_scores_sampled": bool(pair_sampled),
        "pair_scores_eval_count": int(len(pair_scores_eval)),
        "same_similarity_quantiles": _quantiles(pair_scores_eval[pair_labels_eval]),
        "diff_similarity_quantiles": _quantiles(pair_scores_eval[~pair_labels_eval]),
        "topk_all": _topk_metrics(sim, labels_arr, videos_arr, ks, exclude_same_video=False),
        "topk_cross_video": _topk_metrics(sim, labels_arr, videos_arr, ks, exclude_same_video=True),
    }
    if average_precision_score is not None and len(np.unique(pair_labels_eval.astype(int))) == 2:
        result["pair_average_precision"] = float(average_precision_score(pair_labels_eval.astype(int), pair_scores_eval))
    if roc_auc_score is not None and len(np.unique(pair_labels_eval.astype(int))) == 2:
        result["pair_roc_auc"] = float(roc_auc_score(pair_labels_eval.astype(int), pair_scores_eval))

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.report_md:
        lines = [
            "# Sample Feature Retrieval Diagnostic",
            "",
            f"- tracklets: `{result['eval_tracklets']}`",
            f"- unique GT IDs: `{result['unique_gt_ids']}`",
            f"- pair AP/AUC: `{result.get('pair_average_precision')}` / `{result.get('pair_roc_auc')}`",
            f"- top-45 hit recall: `{result['topk_all'].get('hit_recall_at_45')}`",
            f"- cross-video top-45 hit recall: `{result['topk_cross_video'].get('hit_recall_at_45')}`",
            f"- cross-video top-positive-minus-top-negative positive fraction: `{result['topk_cross_video'].get('top_positive_minus_top_negative', {}).get('positive_fraction')}`",
            "",
            "## Same Similarity Quantiles",
            "",
            json.dumps(result["same_similarity_quantiles"], indent=2, sort_keys=True),
            "",
            "## Different Similarity Quantiles",
            "",
            json.dumps(result["diff_similarity_quantiles"], indent=2, sort_keys=True),
        ]
        Path(args.report_md).write_text("\n".join(lines) + "\n")
    print(json.dumps({k: result[k] for k in ["eval_tracklets", "unique_gt_ids", "pair_average_precision", "pair_roc_auc"] if k in result}, sort_keys=True))


if __name__ == "__main__":
    main()
