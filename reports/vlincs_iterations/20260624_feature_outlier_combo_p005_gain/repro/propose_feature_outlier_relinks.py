#!/usr/bin/env python3
"""Scratch no-anchor proposer for current-best feature-outlier relinks.

Inputs are the current assignment CSV and feature NPZ files only.  No GT,
anchors, or evaluator labels are read here; scoring happens in a separate step.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _load_view(spec: str, seqs: np.ndarray) -> dict[str, object]:
    name, path, weight = spec.split(":", 2)
    data = np.load(path, allow_pickle=True)
    npz_seqs = data["seqs"].astype(np.int64)
    feats = _l2n(data["features"].astype(np.float32))
    pos = {int(seq): i for i, seq in enumerate(npz_seqs.tolist())}
    aligned = np.zeros((len(seqs), feats.shape[1]), dtype=np.float32)
    missing = []
    for out_i, seq in enumerate(seqs.tolist()):
        src_i = pos.get(int(seq))
        if src_i is None:
            missing.append(int(seq))
            continue
        aligned[out_i] = feats[src_i]
    if missing:
        raise ValueError(f"{name} missing {len(missing)} assignment seqs, first={missing[:5]}")
    return {"name": name, "path": path, "weight": float(weight), "features": aligned}


def _interval_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return int(a_start) <= int(b_end) and int(b_start) <= int(a_end)


def _build_forbidden(df: pd.DataFrame) -> list[set[int]]:
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, video in enumerate(df["video"].astype(str).tolist()):
        by_video[video].append(i)
    forbidden = [set() for _ in range(len(df))]
    starts = df["start_frame"].astype(int).to_numpy()
    ends = df["end_frame"].astype(int).to_numpy()
    for indices in by_video.values():
        order = sorted(indices, key=lambda i: (starts[i], ends[i], i))
        for pos, i in enumerate(order):
            for j in order[pos + 1 :]:
                if starts[j] > ends[i]:
                    break
                if _interval_overlaps(starts[i], ends[i], starts[j], ends[j]):
                    forbidden[i].add(j)
                    forbidden[j].add(i)
    return forbidden


def _topk_mean(view: np.ndarray, idx: int, members: list[int], k: int) -> float:
    others = [m for m in members if int(m) != int(idx)]
    if not others:
        return -1.0
    sims = view[np.asarray(others, dtype=np.int64)] @ view[int(idx)]
    sims = np.sort(sims)[::-1]
    return float(np.mean(sims[: max(1, min(int(k), len(sims)))]))


def _centroids(view: np.ndarray, groups: dict[int, list[int]]) -> dict[int, np.ndarray]:
    out = {}
    for label, members in groups.items():
        out[int(label)] = _l2n(view[np.asarray(members, dtype=np.int64)].mean(axis=0, keepdims=True))[0]
    return out


def _component_gid(df: pd.DataFrame, members: list[int]) -> int:
    counts = Counter(df.iloc[members]["predicted_global_id"].astype(int).tolist())
    return int(counts.most_common(1)[0][0])


def _parse_skip_pairs(raw: str) -> set[tuple[int, int]]:
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("->", 1)
        out.add((int(a), int(b)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--feature", action="append", required=True, help="name:path:weight")
    ap.add_argument("--assignments-dir", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--min-source-size", type=int, default=8)
    ap.add_argument("--min-target-size", type=int, default=6)
    ap.add_argument("--max-target-size", type=int, default=600)
    ap.add_argument("--max-source-centroid", type=float, default=0.66)
    ap.add_argument("--min-target-centroid", type=float, default=0.62)
    ap.add_argument("--min-centroid-margin", type=float, default=0.04)
    ap.add_argument("--min-neighbor-margin", type=float, default=0.04)
    ap.add_argument("--min-view-votes", type=float, default=0.67)
    ap.add_argument("--view-vote-threshold", type=float, default=0.58)
    ap.add_argument("--topk-neighbors", type=int, default=5)
    ap.add_argument("--max-single-candidates", type=int, default=500)
    ap.add_argument("--emit-top-groups", type=int, default=18)
    ap.add_argument("--group-sizes", default="1,2,4,8")
    ap.add_argument("--skip-pairs", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.assignment_csv)
    df = df.reset_index(drop=True)
    required = {"seq", "tracklet_key", "video", "start_frame", "end_frame", "component_label", "predicted_global_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"assignment missing columns: {missing}")
    seqs = df["seq"].astype(np.int64).to_numpy()
    views = [_load_view(spec, seqs) for spec in args.feature]
    total_weight = sum(float(v["weight"]) for v in views)
    labels = df["component_label"].astype(int).to_numpy()
    groups: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels.tolist()):
        groups[int(label)].append(i)
    forbidden = _build_forbidden(df)
    skip_pairs = _parse_skip_pairs(args.skip_pairs)

    cent = {str(v["name"]): _centroids(v["features"], groups) for v in views}
    candidates = []
    for source_label, source_members in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        if len(source_members) < int(args.min_source_size):
            continue
        source_set = set(source_members)
        for idx in source_members:
            same_component_conflicts = len(forbidden[int(idx)] & source_set)
            per_view_source = []
            per_view_source_nn = []
            for view in views:
                feat = view["features"]
                name = str(view["name"])
                per_view_source.append(float(feat[int(idx)] @ cent[name][int(source_label)]))
                per_view_source_nn.append(_topk_mean(feat, int(idx), source_members, int(args.topk_neighbors)))
            source_centroid = sum(float(v["weight"]) * s for v, s in zip(views, per_view_source)) / total_weight
            source_nn = sum(float(v["weight"]) * s for v, s in zip(views, per_view_source_nn)) / total_weight
            if source_centroid > float(args.max_source_centroid) and same_component_conflicts == 0:
                continue
            best = None
            for target_label, target_members in groups.items():
                target_label = int(target_label)
                if target_label == int(source_label):
                    continue
                if (int(source_label), target_label) in skip_pairs:
                    continue
                if len(target_members) < int(args.min_target_size) or len(target_members) > int(args.max_target_size):
                    continue
                if forbidden[int(idx)] & set(target_members):
                    continue
                per_view_target = []
                per_view_target_nn = []
                votes = 0
                for view in views:
                    feat = view["features"]
                    name = str(view["name"])
                    sim = float(feat[int(idx)] @ cent[name][target_label])
                    nn = _topk_mean(feat, int(idx), target_members, int(args.topk_neighbors))
                    per_view_target.append(sim)
                    per_view_target_nn.append(nn)
                    if sim >= float(args.view_vote_threshold) or nn >= float(args.view_vote_threshold):
                        votes += 1
                target_centroid = sum(float(v["weight"]) * s for v, s in zip(views, per_view_target)) / total_weight
                target_nn = sum(float(v["weight"]) * s for v, s in zip(views, per_view_target_nn)) / total_weight
                centroid_margin = target_centroid - source_centroid
                neighbor_margin = target_nn - source_nn
                view_vote = votes / max(len(views), 1)
                if target_centroid < float(args.min_target_centroid):
                    continue
                if centroid_margin < float(args.min_centroid_margin):
                    continue
                if neighbor_margin < float(args.min_neighbor_margin):
                    continue
                if view_vote < float(args.min_view_votes):
                    continue
                score = (
                    target_centroid
                    + 0.65 * centroid_margin
                    + 0.35 * neighbor_margin
                    + 0.02 * min(same_component_conflicts, 5)
                    + 0.01 * min(np.log1p(len(target_members)) / np.log(256.0), 1.0)
                )
                row = {
                    "seq": int(seqs[int(idx)]),
                    "tracklet_key": str(df.at[int(idx), "tracklet_key"]),
                    "video": str(df.at[int(idx), "video"]),
                    "camera": str(df.at[int(idx), "camera"]) if "camera" in df.columns else "",
                    "start_frame": int(df.at[int(idx), "start_frame"]),
                    "end_frame": int(df.at[int(idx), "end_frame"]),
                    "source_label": int(source_label),
                    "target_label": int(target_label),
                    "source_gid": _component_gid(df, source_members),
                    "target_gid": _component_gid(df, target_members),
                    "source_size": int(len(source_members)),
                    "target_size": int(len(target_members)),
                    "source_centroid": round(float(source_centroid), 6),
                    "target_centroid": round(float(target_centroid), 6),
                    "centroid_margin": round(float(centroid_margin), 6),
                    "source_neighbor": round(float(source_nn), 6),
                    "target_neighbor": round(float(target_nn), 6),
                    "neighbor_margin": round(float(neighbor_margin), 6),
                    "same_component_conflicts": int(same_component_conflicts),
                    "view_vote": round(float(view_vote), 6),
                    "score": round(float(score), 6),
                    "per_view_source": [round(float(x), 6) for x in per_view_source],
                    "per_view_target": [round(float(x), 6) for x in per_view_target],
                    "per_view_source_nn": [round(float(x), 6) for x in per_view_source_nn],
                    "per_view_target_nn": [round(float(x), 6) for x in per_view_target_nn],
                    "uses_anchors": False,
                    "uses_gt_for_training_or_anchors": False,
                }
                if best is None or float(row["score"]) > float(best["score"]):
                    best = row
            if best is not None:
                candidates.append(best)
    candidates.sort(key=lambda row: (float(row["score"]), float(row["centroid_margin"]), float(row["neighbor_margin"])), reverse=True)
    candidates = candidates[: int(args.max_single_candidates)]

    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        grouped[(int(row["source_label"]), int(row["target_label"]))].append(row)
    group_rows = []
    for (source_label, target_label), rows in grouped.items():
        rows = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
        for group_size in [int(x) for x in str(args.group_sizes).split(",") if x.strip()]:
            selected = rows[:group_size]
            if len(selected) < group_size:
                continue
            group_rows.append(
                {
                    "source_label": int(source_label),
                    "target_label": int(target_label),
                    "source_gid": int(selected[0]["source_gid"]),
                    "target_gid": int(selected[0]["target_gid"]),
                    "group_size": int(len(selected)),
                    "seqs": [int(row["seq"]) for row in selected],
                    "tracklet_keys": [str(row["tracklet_key"]) for row in selected],
                    "videos": sorted({str(row["video"]) for row in selected}),
                    "mean_score": round(float(np.mean([float(row["score"]) for row in selected])), 6),
                    "min_score": round(float(np.min([float(row["score"]) for row in selected])), 6),
                    "mean_centroid_margin": round(float(np.mean([float(row["centroid_margin"]) for row in selected])), 6),
                    "mean_neighbor_margin": round(float(np.mean([float(row["neighbor_margin"]) for row in selected])), 6),
                    "sum_conflicts": int(sum(int(row["same_component_conflicts"]) for row in selected)),
                    "accepted_preview": selected,
                }
            )
    group_rows.sort(
        key=lambda row: (
            float(row["mean_score"]),
            int(row["sum_conflicts"]),
            int(row["group_size"]),
            float(row["mean_neighbor_margin"]),
        ),
        reverse=True,
    )
    group_rows = group_rows[: int(args.emit_top_groups)]

    out_dir = Path(args.assignments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emitted = []
    for rank, group in enumerate(group_rows, start=1):
        work = df.copy()
        mask = work["seq"].astype(int).isin(set(int(seq) for seq in group["seqs"]))
        work.loc[mask, "predicted_global_id"] = int(group["target_gid"])
        work.loc[mask, "component_label"] = int(group["target_label"])
        if "decision_status" in work.columns:
            work.loc[mask, "decision_status"] = "feature_outlier_relink"
        out_path = out_dir / f"rank{rank:02d}_feature_outlier_s{group['source_label']}_to{group['target_label']}_{group['group_size']}seq_assignments.csv"
        work.to_csv(out_path, index=False)
        emitted.append({**group, "rank": rank, "assignments_out": str(out_path)})

    with open(args.summary_csv, "w", newline="") as handle:
        fields = [
            "rank",
            "source_label",
            "target_label",
            "source_gid",
            "target_gid",
            "group_size",
            "seqs",
            "videos",
            "mean_score",
            "min_score",
            "mean_centroid_margin",
            "mean_neighbor_margin",
            "sum_conflicts",
            "assignments_out",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in emitted:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), list) else row.get(key) for key in fields})
    summary = {
        "assignment_csv": str(args.assignment_csv),
        "features": [{k: v for k, v in view.items() if k != "features"} for view in views],
        "params": vars(args),
        "single_candidates": int(len(candidates)),
        "groups_emitted": int(len(emitted)),
        "top_groups": emitted,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"single_candidates": len(candidates), "groups_emitted": len(emitted), "summary_json": args.summary_json}, sort_keys=True))


if __name__ == "__main__":
    main()
