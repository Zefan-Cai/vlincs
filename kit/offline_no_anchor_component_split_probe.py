#!/usr/bin/env python
"""Offline no-anchor component split probe.

Generates feature-only component splits from an assignment CSV and evaluates
their tracklet-pair effect with an eval cache.  GT labels are never used to
generate split transforms; they are used only for ablation metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_components(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _parse_ks(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _load_eval_cache(path: str):
    z = np.load(path, allow_pickle=True)
    gt = {int(seq): int(gid) for seq, gid in zip(z["seqs"].tolist(), z["gids"].tolist())}
    weight = {int(seq): float(w) for seq, w in zip(z["seqs"].tolist(), z["weights"].tolist())}
    return gt, weight


def _load_feature(spec: str):
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path = spec
        name = Path(path).stem
    z = np.load(path, allow_pickle=True)
    x = _l2n(z["features"].astype(np.float32))
    return name, {int(seq): x[idx] for idx, seq in enumerate(z["seqs"].tolist())}


def _pair_metrics(df: pd.DataFrame, pred_by_seq: dict[int, int], gt_by_seq: dict[int, int], weight_by_seq: dict[int, float]) -> dict[str, float]:
    rows = []
    for seq in df["seq"].astype(int).tolist():
        if seq in gt_by_seq:
            rows.append((seq, int(pred_by_seq[seq]), int(gt_by_seq[seq]), float(weight_by_seq[seq])))

    def mass(groups: dict[object, list[float]]) -> float:
        out = 0.0
        for values in groups.values():
            s = float(sum(values))
            out += 0.5 * max(s * s - sum(float(v) * float(v) for v in values), 0.0)
        return out

    by_gt: dict[int, list[float]] = defaultdict(list)
    by_pred: dict[int, list[float]] = defaultdict(list)
    by_both: dict[tuple[int, int], list[float]] = defaultdict(list)
    for _seq, pred, gid, weight in rows:
        by_gt[gid].append(weight)
        by_pred[pred].append(weight)
        by_both[(pred, gid)].append(weight)
    gt_mass = mass(by_gt)
    pred_mass = mass(by_pred)
    true_mass = mass(by_both)
    precision = true_mass / max(pred_mass, 1.0e-9)
    recall = true_mass / max(gt_mass, 1.0e-9)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-9)
    return {
        "tracklet_pair_f1": float(f1),
        "tracklet_pair_precision": float(precision),
        "tracklet_pair_recall": float(recall),
        "gt_pair_mass": float(gt_mass),
        "pred_pair_mass": float(pred_mass),
        "true_pair_mass": float(true_mass),
        "eval_tracklets": int(len(rows)),
    }


def _component_eval_context(df: pd.DataFrame, comp: int, gt_by_seq: dict[int, int], weight_by_seq: dict[int, float]) -> dict[str, object]:
    by_gt: dict[int, float] = defaultdict(float)
    for seq in df.loc[df["predicted_global_id"].astype(int) == int(comp), "seq"].astype(int).tolist():
        if seq in gt_by_seq:
            by_gt[int(gt_by_seq[seq])] += float(weight_by_seq[seq])
    total = float(sum(by_gt.values()))
    if by_gt and total > 0:
        gid, weight = max(by_gt.items(), key=lambda item: item[1])
        return {"dominant_gt_eval_only": int(gid), "purity_eval_only": float(weight / max(total, 1.0e-9))}
    return {"dominant_gt_eval_only": None, "purity_eval_only": 0.0}


def _labels_for_component(df: pd.DataFrame, comp: int, feature_by_seq: dict[int, np.ndarray], k: int, seed: int):
    seqs = [
        int(seq)
        for seq in df.loc[df["predicted_global_id"].astype(int) == int(comp), "seq"].astype(int).tolist()
        if int(seq) in feature_by_seq
    ]
    if len(seqs) <= int(k):
        return None
    x = np.vstack([feature_by_seq[seq] for seq in seqs])
    labels = KMeans(n_clusters=int(k), random_state=int(seed), n_init=20).fit_predict(x)
    counts = np.bincount(labels, minlength=int(k))
    if int(counts.min()) <= 0:
        return None
    sil = float(silhouette_score(x, labels, metric="cosine")) if len(set(labels.tolist())) > 1 else 0.0
    return {int(seq): int(label) for seq, label in zip(seqs, labels.tolist())}, sil, [int(v) for v in counts.tolist()]


def _apply_split(pred: dict[int, int], comp: int, labels: dict[int, int]) -> dict[int, int]:
    out = dict(pred)
    for seq, label in labels.items():
        out[int(seq)] = int(comp) * 10 + int(label)
    return out


def _write_assignment(df: pd.DataFrame, pred: dict[int, int], path: str, status: str) -> None:
    out = df.copy()
    out["predicted_global_id"] = [int(pred.get(int(seq), int(cur))) for seq, cur in zip(out["seq"].tolist(), out["predicted_global_id"].tolist())]
    out["decision_status"] = status
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--feature", action="append", required=True)
    ap.add_argument("--components", default="", help="comma-separated component IDs; empty means components above --min-size")
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--ks", default="2,3,4,5,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy-limit", type=int, default=10)
    ap.add_argument("--greedy-assignment-out", default="")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.assignment_csv)
    df["seq"] = df["seq"].astype(int)
    df["predicted_global_id"] = df["predicted_global_id"].astype(int)
    gt_by_seq, weight_by_seq = _load_eval_cache(args.eval_cache)
    base_pred = {int(row.seq): int(row.predicted_global_id) for row in df.itertuples(index=False)}
    base_pair = _pair_metrics(df, base_pred, gt_by_seq, weight_by_seq)

    if args.components:
        components = _parse_components(args.components)
    else:
        counts = df.groupby("predicted_global_id")["seq"].nunique()
        components = [int(comp) for comp, size in counts.items() if int(size) >= int(args.min_size)]

    features = dict(_load_feature(spec) for spec in args.feature)
    rows = []
    split_cache: dict[tuple[int, str, int], dict[int, int]] = {}
    for comp in components:
        base_context = _component_eval_context(df, comp, gt_by_seq, weight_by_seq)
        for view_name, feature_by_seq in features.items():
            for k in _parse_ks(args.ks):
                result = _labels_for_component(df, comp, feature_by_seq, k, int(args.seed))
                if result is None:
                    continue
                labels, silhouette, cluster_sizes = result
                trial = _apply_split(base_pred, comp, labels)
                pair = _pair_metrics(df, trial, gt_by_seq, weight_by_seq)
                split_cache[(int(comp), str(view_name), int(k))] = labels
                rows.append(
                    {
                        "component": int(comp),
                        "view": str(view_name),
                        "k": int(k),
                        "component_size": int(len(labels)),
                        "silhouette": float(silhouette),
                        "cluster_sizes": cluster_sizes,
                        "pair_delta_f1": float(pair["tracklet_pair_f1"] - base_pair["tracklet_pair_f1"]),
                        **{f"pair_{key}": value for key, value in pair.items() if key.startswith("tracklet_pair_")},
                        **base_context,
                    }
                )
    rows.sort(key=lambda row: float(row["pair_delta_f1"]), reverse=True)

    greedy_pred = dict(base_pred)
    greedy_selected = []
    current_pair = dict(base_pair)
    used_components = set()
    for row in rows:
        comp = int(row["component"])
        if comp in used_components:
            continue
        key = (comp, str(row["view"]), int(row["k"]))
        trial = _apply_split(greedy_pred, comp, split_cache[key])
        pair = _pair_metrics(df, trial, gt_by_seq, weight_by_seq)
        delta = float(pair["tracklet_pair_f1"] - current_pair["tracklet_pair_f1"])
        if delta <= 1.0e-9:
            continue
        greedy_pred = trial
        current_pair = pair
        used_components.add(comp)
        greedy_selected.append({**row, "greedy_delta_from_previous": delta, **{f"greedy_{k}": v for k, v in pair.items() if k.startswith("tracklet_pair_")}})
        if len(greedy_selected) >= int(args.greedy_limit):
            break

    if args.greedy_assignment_out:
        _write_assignment(df, greedy_pred, args.greedy_assignment_out, "offline_component_split_greedy_candidate")

    out = {
        "assignment_csv": args.assignment_csv,
        "uses_anchors": False,
        "uses_gt_for_candidate_generation": False,
        "uses_gt_for_evaluation_only": True,
        "base_pair": base_pair,
        "components_considered": int(len(components)),
        "feature_views": sorted(features),
        "rows": rows,
        "greedy_selected": greedy_selected,
        "greedy_pair": current_pair,
        "greedy_assignment_out": args.greedy_assignment_out,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({"base_pair": base_pair, "top": rows[:3], "greedy_pair": current_pair, "greedy_selected_n": len(greedy_selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
