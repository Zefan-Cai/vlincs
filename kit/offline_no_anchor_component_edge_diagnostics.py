#!/usr/bin/env python
"""Offline no-anchor component-edge diagnostics from assignment + feature NPZs.

This avoids the Postgres dependency used by the full DS1 pipeline.  Candidate
edges are generated from no-GT component feature similarity.  Eval cache labels
are used only after edge generation to measure whether the no-GT evidence can
separate false-split repairs from false merges.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _load_eval_cache(path: str) -> tuple[dict[int, int], dict[int, float], dict[str, object]]:
    z = np.load(path, allow_pickle=True)
    gt_by_seq = {int(seq): int(gid) for seq, gid in zip(z["seqs"].tolist(), z["gids"].tolist())}
    weight_by_seq = {int(seq): float(weight) for seq, weight in zip(z["seqs"].tolist(), z["weights"].tolist())}
    stats_raw = z["stats"].item() if "stats" in z.files else "{}"
    try:
        stats = json.loads(str(stats_raw))
    except Exception:
        stats = {"raw": str(stats_raw)}
    return gt_by_seq, weight_by_seq, stats


def _load_feature_npz(spec: str) -> tuple[str, dict[int, np.ndarray]]:
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path = spec
        name = Path(path).stem
    z = np.load(path, allow_pickle=True)
    if "seqs" not in z.files or "features" not in z.files:
        raise ValueError(f"{path} must contain seqs/features")
    feats = z["features"].astype(np.float32)
    feats = _l2n(feats)
    out = {int(seq): feats[idx] for idx, seq in enumerate(z["seqs"].tolist())}
    return str(name), out


def _component_tables(df: pd.DataFrame, gt_by_seq: dict[int, int], weight_by_seq: dict[int, float], pred_col: str):
    groups = []
    for pred_id, sub in df.groupby(pred_col, sort=True):
        seqs = [int(v) for v in sub["seq"].tolist()]
        gt_weight: dict[int, float] = defaultdict(float)
        for seq in seqs:
            if seq in gt_by_seq:
                gt_weight[int(gt_by_seq[seq])] += float(weight_by_seq.get(seq, 1.0))
        total_gt_weight = float(sum(gt_weight.values()))
        if gt_weight and total_gt_weight > 0:
            dom_gt, dom_weight = max(gt_weight.items(), key=lambda item: item[1])
            purity = float(dom_weight / max(total_gt_weight, 1.0e-9))
        else:
            dom_gt = None
            purity = 0.0
        intervals = defaultdict(list)
        for row in sub.itertuples(index=False):
            intervals[str(row.video)].append((int(row.start_frame), int(row.end_frame)))
        for video in intervals:
            intervals[video].sort()
        groups.append(
            {
                "component_index": len(groups),
                "predicted_global_id": int(pred_id),
                "seqs": seqs,
                "size": int(len(seqs)),
                "n_dets": int(sub["n_dets"].sum()) if "n_dets" in sub.columns else int(len(seqs)),
                "videos": sorted(str(v) for v in sub["video"].dropna().unique().tolist()),
                "cameras": sorted(str(v) for v in sub["camera"].dropna().unique().tolist()),
                "intervals": dict(intervals),
                "gt_weight": dict(gt_weight),
                "gt_total_weight": total_gt_weight,
                "dominant_gt": dom_gt,
                "purity": purity,
            }
        )
    return groups


def _component_centroids(groups: list[dict[str, object]], feature_by_seq: dict[int, np.ndarray], weight_by_seq: dict[int, float] | None = None):
    dim = len(next(iter(feature_by_seq.values())))
    out = np.zeros((len(groups), dim), dtype=np.float32)
    valid = np.zeros(len(groups), dtype=bool)
    for group in groups:
        rows = []
        weights = []
        for seq in group["seqs"]:
            seq = int(seq)
            feat = feature_by_seq.get(seq)
            if feat is None:
                continue
            rows.append(feat)
            weights.append(float(weight_by_seq.get(seq, 1.0)) if weight_by_seq else 1.0)
        idx = int(group["component_index"])
        if rows:
            X = np.vstack(rows).astype(np.float32)
            w = np.asarray(weights, dtype=np.float32)
            center = (X * w[:, None]).sum(axis=0) / max(float(w.sum()), 1.0e-9)
            out[idx] = center / (np.linalg.norm(center) + 1.0e-9)
            valid[idx] = True
    return out, valid


def _has_interval_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    i = j = 0
    while i < len(a) and j < len(b):
        a0, a1 = a[i]
        b0, b1 = b[j]
        if max(a0, b0) <= min(a1, b1):
            return True
        if a1 < b1:
            i += 1
        else:
            j += 1
    return False


def _edge_constraint(a: dict[str, object], b: dict[str, object]) -> tuple[int, int, int]:
    same_cameras = len(set(a["cameras"]) & set(b["cameras"]))
    same_videos = set(a["intervals"]) & set(b["intervals"])
    overlap = 0
    for video in same_videos:
        if _has_interval_overlap(a["intervals"][video], b["intervals"][video]):
            overlap += 1
    return int(same_cameras), int(len(same_videos)), int(overlap)


def _top_edges_for_view(name: str, centroids: np.ndarray, valid: np.ndarray, top_k: int) -> dict[tuple[int, int], dict[str, float]]:
    sims = centroids @ centroids.T
    np.fill_diagonal(sims, -2.0)
    sims[~valid, :] = -2.0
    sims[:, ~valid] = -2.0
    edge_data: dict[tuple[int, int], dict[str, float]] = {}
    for i in range(sims.shape[0]):
        if not valid[i]:
            continue
        order = np.argsort(-sims[i])[: int(top_k)]
        for rank, j in enumerate(order.tolist(), start=1):
            if i == j or not valid[j]:
                continue
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            score = float(sims[i, j])
            cur = edge_data.get((a, b))
            if cur is None or score > float(cur[f"{name}_sim"]):
                edge_data[(a, b)] = {f"{name}_sim": score, f"{name}_rank": float(rank)}
    return edge_data


def _merge_edge_views(view_edges: dict[str, dict[tuple[int, int], dict[str, float]]], top_k: int) -> dict[tuple[int, int], dict[str, object]]:
    merged: dict[tuple[int, int], dict[str, object]] = {}
    for view_name, edges in view_edges.items():
        for key, vals in edges.items():
            row = merged.setdefault(key, {"source": key[0], "target": key[1]})
            row.update(vals)
    for row in merged.values():
        sims = []
        votes_topk = 0
        votes_top5 = 0
        for view_name in view_edges:
            sim_key = f"{view_name}_sim"
            rank_key = f"{view_name}_rank"
            if sim_key in row:
                sims.append(float(row[sim_key]))
                if float(row[rank_key]) <= float(top_k):
                    votes_topk += 1
                if float(row[rank_key]) <= 5:
                    votes_top5 += 1
        row["view_count"] = int(len(sims))
        row["votes_topk"] = int(votes_topk)
        row["votes_top5"] = int(votes_top5)
        row["mean_sim"] = float(np.mean(sims)) if sims else -2.0
        row["max_sim"] = float(np.max(sims)) if sims else -2.0
        row["min_sim"] = float(np.min(sims)) if sims else -2.0
    return merged


def _false_split_mass_by_gt(groups: list[dict[str, object]]) -> dict[int, float]:
    by_gt: dict[int, list[float]] = defaultdict(list)
    for group in groups:
        for gid, weight in group["gt_weight"].items():
            if float(weight) > 0:
                by_gt[int(gid)].append(float(weight))
    out = {}
    for gid, weights in by_gt.items():
        total = float(sum(weights))
        total2 = float(sum(w * w for w in weights))
        split_mass = max((total * total - total2) / 2.0, 0.0)
        out[int(gid)] = split_mass
    return out


def _annotate_edges(rows: list[dict[str, object]], groups: list[dict[str, object]], min_purity: float, top_false_split_gts: set[int]):
    for row in rows:
        a = groups[int(row["source"])]
        b = groups[int(row["target"])]
        same_cameras, same_videos, overlap = _edge_constraint(a, b)
        common = set(a["gt_weight"]) & set(b["gt_weight"])
        same_mass = sum(float(a["gt_weight"][gid]) * float(b["gt_weight"][gid]) for gid in common)
        all_mass = float(a["gt_total_weight"]) * float(b["gt_total_weight"])
        dom_same = a["dominant_gt"] is not None and a["dominant_gt"] == b["dominant_gt"]
        both_pure = float(a["purity"]) >= float(min_purity) and float(b["purity"]) >= float(min_purity)
        row.update(
            {
                "source_predicted_global_id": int(a["predicted_global_id"]),
                "target_predicted_global_id": int(b["predicted_global_id"]),
                "source_size": int(a["size"]),
                "target_size": int(b["size"]),
                "source_n_dets": int(a["n_dets"]),
                "target_n_dets": int(b["n_dets"]),
                "source_dominant_gt": a["dominant_gt"],
                "target_dominant_gt": b["dominant_gt"],
                "source_purity": float(a["purity"]),
                "target_purity": float(b["purity"]),
                "same_camera_count": same_cameras,
                "same_video_count": same_videos,
                "temporal_overlap_video_count": overlap,
                "is_cannot_link": int(overlap > 0),
                "gt_same_mass": float(same_mass),
                "gt_all_mass": float(all_mass),
                "gt_same_frac": float(same_mass / max(all_mass, 1.0e-9)),
                "gt_dominant_same": bool(dom_same),
                "gt_both_pure": bool(both_pure),
                "gt_edge_label": int(bool(dom_same and both_pure)),
                "gt_top_false_split_target": bool(dom_same and a["dominant_gt"] in top_false_split_gts),
            }
        )
    return rows


def _rule_ablation(rows: list[dict[str, object]], thresholds: list[float], votes: list[int], min_sizes: list[int]) -> list[dict[str, object]]:
    out = []
    for thr in thresholds:
        for vote in votes:
            for min_size in min_sizes:
                cand = [
                    row
                    for row in rows
                    if int(row["is_cannot_link"]) == 0
                    and int(row["votes_topk"]) >= int(vote)
                    and float(row["mean_sim"]) >= float(thr)
                    and min(int(row["source_size"]), int(row["target_size"])) >= int(min_size)
                ]
                if not cand:
                    continue
                tp = [row for row in cand if int(row["gt_edge_label"]) == 1]
                same_mass = sum(float(row["gt_same_mass"]) for row in tp)
                fp_proxy = sum(max(float(row["gt_all_mass"]) - float(row["gt_same_mass"]), 0.0) for row in cand if int(row["gt_edge_label"]) == 0)
                out.append(
                    {
                        "mean_sim_threshold": float(thr),
                        "min_votes_topk": int(vote),
                        "min_component_size": int(min_size),
                        "candidate_edges": int(len(cand)),
                        "true_edges": int(len(tp)),
                        "edge_precision": round(float(len(tp) / max(len(cand), 1)), 6),
                        "same_mass": round(float(same_mass), 3),
                        "fp_proxy_mass": round(float(fp_proxy), 3),
                        "top_false_split_edges": int(sum(1 for row in tp if bool(row["gt_top_false_split_target"]))),
                    }
                )
    out.sort(key=lambda r: (float(r["edge_precision"]), float(r["same_mass"]), int(r["candidate_edges"])), reverse=True)
    return out


def _retrieval_summary(rows: list[dict[str, object]], groups: list[dict[str, object]], top_gts: list[int], top_ks: list[int]) -> dict[str, object]:
    edge_keys = {(int(row["source"]), int(row["target"])) for row in rows}
    neighbors: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        a = int(row["source"])
        b = int(row["target"])
        score = float(row["mean_sim"])
        neighbors[a].append((score, b))
        neighbors[b].append((score, a))
    for vals in neighbors.values():
        vals.sort(reverse=True)
    out = {}
    for gid in top_gts:
        comps = [(idx, float(group["gt_weight"].get(gid, 0.0))) for idx, group in enumerate(groups) if float(group["gt_weight"].get(gid, 0.0)) > 0]
        total = 0.0
        union_mass = 0.0
        by_k = {int(k): 0.0 for k in top_ks}
        for i in range(len(comps)):
            a, wa = comps[i]
            for b, wb in comps[i + 1 :]:
                mass = float(wa * wb)
                total += mass
                key = (a, b) if a < b else (b, a)
                if key in edge_keys:
                    union_mass += mass
                for k in top_ks:
                    top_a = {idx for _score, idx in neighbors.get(a, [])[: int(k)]}
                    top_b = {idx for _score, idx in neighbors.get(b, [])[: int(k)]}
                    if b in top_a or a in top_b:
                        by_k[int(k)] += mass
        out[str(gid)] = {
            "component_count": int(len(comps)),
            "false_split_pair_mass": round(float(total), 3),
            "candidate_union_fraction": round(float(union_mass / max(total, 1.0e-9)), 6),
            "retrieved_at_k": {
                str(k): round(float(v / max(total, 1.0e-9)), 6)
                for k, v in by_k.items()
            },
        }
    return out


def _write_csv(path: str, rows: list[dict[str, object]], limit: int = 0) -> None:
    rows = rows[: int(limit)] if int(limit) > 0 else rows
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list, tuple))})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_candidate_assignments(df: pd.DataFrame, rows: list[dict[str, object]], groups: list[dict[str, object]], out_dir: str, limit: int) -> list[dict[str, object]]:
    if not out_dir or int(limit) <= 0:
        return []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    written = []
    for rank, row in enumerate(rows[: int(limit)], start=1):
        a = groups[int(row["source"])]
        b = groups[int(row["target"])]
        if int(a["size"]) <= int(b["size"]):
            src, dst = a, b
        else:
            src, dst = b, a
        out = df.copy()
        src_id = int(src["predicted_global_id"])
        dst_id = int(dst["predicted_global_id"])
        mask = out["predicted_global_id"].astype(int) == src_id
        out.loc[mask, "predicted_global_id"] = dst_id
        out.loc[mask, "decision_status"] = "offline_component_edge_merge_candidate"
        name = f"rank{rank:03d}_merge_{src_id}_to_{dst_id}_{int(mask.sum())}seq_assignments.csv"
        path = str(Path(out_dir) / name)
        out.to_csv(path, index=False)
        written.append(
            {
                "rank": int(rank),
                "assignment_csv": path,
                "source_predicted_global_id": src_id,
                "target_predicted_global_id": dst_id,
                "moved_tracklets": int(mask.sum()),
                "mean_sim": round(float(row["mean_sim"]), 6),
                "votes_topk": int(row["votes_topk"]),
                "is_cannot_link": int(row["is_cannot_link"]),
                "gt_edge_label_eval_only": int(row["gt_edge_label"]),
                "gt_same_mass_eval_only": round(float(row["gt_same_mass"]), 3),
            }
        )
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--feature", action="append", required=True, help="name=/path/to/features.npz with seqs/features")
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--min-votes", default="1,2,3")
    ap.add_argument("--min-component-sizes", default="1,2,4,8,16,32")
    ap.add_argument("--min-purity", type=float, default=0.75)
    ap.add_argument("--top-false-split-gt", type=int, default=20)
    ap.add_argument("--candidate-assignments-dir", default="")
    ap.add_argument("--candidate-limit", type=int, default=0)
    ap.add_argument("--edge-csv", default="")
    ap.add_argument("--edge-csv-limit", type=int, default=1000)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.assignment_csv)
    if "seq" not in df.columns or args.pred_col not in df.columns:
        raise ValueError("assignment CSV must contain seq and pred column")
    df["seq"] = df["seq"].astype(int)
    gt_by_seq, weight_by_seq, eval_stats = _load_eval_cache(args.eval_cache)
    groups = _component_tables(df, gt_by_seq, weight_by_seq, args.pred_col)

    view_edges = {}
    view_valid_components = {}
    for spec in args.feature:
        name, feature_by_seq = _load_feature_npz(spec)
        centroids, valid = _component_centroids(groups, feature_by_seq)
        view_edges[name] = _top_edges_for_view(name, centroids, valid, int(args.top_k))
        view_valid_components[name] = int(valid.sum())

    merged = _merge_edge_views(view_edges, int(args.top_k))
    rows = list(merged.values())
    split_by_gt = _false_split_mass_by_gt(groups)
    top_gts = [gid for gid, _mass in sorted(split_by_gt.items(), key=lambda item: item[1], reverse=True)[: int(args.top_false_split_gt)]]
    rows = _annotate_edges(rows, groups, float(args.min_purity), set(top_gts))
    rows.sort(
        key=lambda row: (
            int(row["is_cannot_link"]) == 0,
            int(row["votes_topk"]),
            float(row["mean_sim"]),
            min(int(row["source_size"]), int(row["target_size"])),
        ),
        reverse=True,
    )

    rule_rows = _rule_ablation(rows, _parse_floats(args.thresholds), _parse_ints(args.min_votes), _parse_ints(args.min_component_sizes))
    retrieval = _retrieval_summary(rows, groups, top_gts[:10], [1, 3, 5, 10, int(args.top_k)])
    no_gt_candidates = [
        row
        for row in rows
        if int(row["is_cannot_link"]) == 0
        and int(row["votes_topk"]) >= max(1, min(2, len(view_edges)))
        and min(int(row["source_size"]), int(row["target_size"])) >= 2
    ]
    candidate_assignments = _write_candidate_assignments(df, no_gt_candidates, groups, args.candidate_assignments_dir, int(args.candidate_limit))
    if args.edge_csv:
        _write_csv(args.edge_csv, rows, int(args.edge_csv_limit))

    summary = {
        "assignment_csv": args.assignment_csv,
        "uses_anchors": False,
        "uses_gt_for_candidate_generation": False,
        "uses_gt_for_evaluation_only": True,
        "assignment_rows": int(len(df)),
        "components": int(len(groups)),
        "eval_labeled_assignment_rows": int(sum(1 for seq in df["seq"].tolist() if int(seq) in gt_by_seq)),
        "eval_cache_stats": eval_stats,
        "feature_views": {name: {"candidate_edges": int(len(edges)), "valid_components": int(view_valid_components[name])} for name, edges in view_edges.items()},
        "union_edges": int(len(rows)),
        "top_false_split_gt_ids": [
            {"gt_id": int(gid), "false_split_mass": round(float(split_by_gt[gid]), 3)}
            for gid in top_gts[: int(args.top_false_split_gt)]
        ],
        "retrieval": retrieval,
        "best_rules": rule_rows[:30],
        "top_no_gt_edges": rows[:50],
        "candidate_assignments": candidate_assignments,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: summary[k] for k in ["assignment_rows", "components", "union_edges", "feature_views"]}, sort_keys=True), flush=True)
    if rule_rows:
        print(json.dumps({"best_rule": rule_rows[0]}, sort_keys=True), flush=True)
    if candidate_assignments:
        print(json.dumps({"candidate_assignments": len(candidate_assignments), "first": candidate_assignments[0]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
