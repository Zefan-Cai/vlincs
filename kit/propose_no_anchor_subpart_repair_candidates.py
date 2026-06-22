#!/usr/bin/env python
"""Propose no-anchor subpart repair assignment candidates from CSV + features.

This is intentionally production-side: it reads an existing assignment CSV and
tracklet feature NPZ files, then materializes small subpart-to-target repair
candidates.  It does not load GT, anchors, or evaluation labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _intish(value: Any) -> int:
    return int(float(value))


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (slug or "candidate")[:max_len]


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _load_assignment(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = {"seq", "video", "start_frame", "end_frame", "component_label", "predicted_global_id"} - set(fieldnames)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def _load_npz_features(path: Path, seqs: list[int], weight: float) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    npz_seqs = [_intish(seq) for seq in data["seqs"].tolist()]
    features = data["features"].astype(np.float32)
    by_seq = {seq: idx for idx, seq in enumerate(npz_seqs)}
    missing = [seq for seq in seqs if seq not in by_seq]
    if missing:
        raise ValueError(f"{path} missing seq {missing[0]} ({len(missing)} total)")
    order = np.asarray([by_seq[seq] for seq in seqs], dtype=np.int64)
    return _l2n(features[order]) * float(weight)


def _parse_view(text: str) -> tuple[str, Path, float]:
    parts = str(text).split(":")
    if len(parts) == 2:
        name, path = parts
        weight = 1.0
    elif len(parts) == 3:
        name, path, weight = parts
    else:
        raise ValueError(f"bad --view {text!r}; expected name:path[:weight]")
    return name, Path(path), float(weight)


def _load_fused_features(primary: Path, rows: list[dict[str, str]], views: list[str], primary_weight: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    seqs = [_intish(row["seq"]) for row in rows]
    blocks = [_load_npz_features(primary, seqs, primary_weight)]
    meta: list[dict[str, Any]] = [{"name": "primary", "path": str(primary), "weight": float(primary_weight)}]
    for spec in views:
        name, path, weight = _parse_view(spec)
        blocks.append(_load_npz_features(path, seqs, weight))
        meta.append({"name": name, "path": str(path), "weight": float(weight)})
    return _l2n(np.concatenate(blocks, axis=1).astype(np.float32)), meta


def _component_rows(rows: list[dict[str, str]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[_intish(row["component_label"])].append(idx)
    return dict(groups)


def _overlap(a: dict[str, str], b: dict[str, str], margin_frames: int) -> bool:
    if a["video"] != b["video"]:
        return False
    return max(_intish(a["start_frame"]), _intish(b["start_frame"])) <= min(
        _intish(a["end_frame"]), _intish(b["end_frame"])
    ) + int(margin_frames)


def _conflict_graph(rows: list[dict[str, str]], indices: list[int], margin_frames: int) -> tuple[dict[int, set[int]], int]:
    graph = {idx: set() for idx in indices}
    edges = 0
    for pos, i in enumerate(indices):
        for j in indices[pos + 1 :]:
            if _overlap(rows[i], rows[j], margin_frames):
                graph[i].add(j)
                graph[j].add(i)
                edges += 1
    return graph, edges


def _mean_vec(x: np.ndarray, indices: list[int]) -> np.ndarray:
    vec = x[np.asarray(indices, dtype=np.int64)].mean(axis=0)
    return vec / (np.linalg.norm(vec) + 1.0e-9)


def _component_pred_gid(rows: list[dict[str, str]], indices: list[int]) -> int:
    counts = Counter(_intish(rows[idx]["predicted_global_id"]) for idx in indices)
    return counts.most_common(1)[0][0]


def _group_stats(x: np.ndarray, component_indices: list[int], group: list[int], graph: dict[int, set[int]]) -> dict[str, Any]:
    group_set = set(group)
    rest = [idx for idx in component_indices if idx not in group_set]
    if len(group) > 1:
        gx = x[np.asarray(group, dtype=np.int64)]
        sim = gx @ gx.T
        tri = sim[np.triu_indices(len(group), k=1)]
        internal = float(np.mean(tri)) if tri.size else 1.0
    else:
        internal = 1.0
    cross_values = (x[np.asarray(group, dtype=np.int64)] @ x[np.asarray(rest, dtype=np.int64)].T).reshape(-1) if rest else np.asarray([], dtype=np.float32)
    cross_mean = float(np.mean(cross_values)) if cross_values.size else 0.0
    cross_max = float(np.max(cross_values)) if cross_values.size else 0.0
    conflicts_to_rest = 0
    for idx in group:
        conflicts_to_rest += sum(1 for nbr in graph[idx] if nbr not in group_set)
    return {
        "group_internal_sim": internal,
        "source_rest_cross_mean": cross_mean,
        "source_rest_cross_max": cross_max,
        "source_rest_margin_mean": float(internal - cross_mean),
        "source_rest_margin_max": float(internal - cross_max),
        "conflicts_to_rest": int(conflicts_to_rest),
    }


def _group_has_target_overlap(rows: list[dict[str, str]], group: list[int], target_indices: list[int], margin_frames: int) -> bool:
    for i in group:
        for j in target_indices:
            if _overlap(rows[i], rows[j], margin_frames):
                return True
    return False


def _quality(row: dict[str, str]) -> float:
    return math.log1p(max(0, _intish(row.get("n_dets", 0)))) + _floatish(row.get("avg_conf", 0.0))


def _build_candidates(rows: list[dict[str, str]], x: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    comps = _component_rows(rows)
    comp_centroid = {label: _mean_vec(x, indices) for label, indices in comps.items()}
    comp_gid = {label: _component_pred_gid(rows, indices) for label, indices in comps.items()}
    focus_videos = {video.strip() for video in str(args.focus_videos).split(",") if video.strip()}
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()

    for source_label, indices in sorted(comps.items(), key=lambda item: len(item[1]), reverse=True):
        if len(indices) < int(args.min_source_component_size) or len(indices) > int(args.max_source_component_size):
            continue
        graph, conflict_edges = _conflict_graph(rows, indices, int(args.overlap_margin_frames))
        if conflict_edges < int(args.min_component_conflict_edges):
            continue
        seeds = sorted(
            [idx for idx, nbrs in graph.items() if nbrs],
            key=lambda idx: (-len(graph[idx]), -_quality(rows[idx]), _intish(rows[idx]["seq"])),
        )[: int(args.max_seeds_per_component)]
        local = np.asarray(indices, dtype=np.int64)
        source_sim = x[local] @ x[local].T
        pos_by_idx = {idx: pos for pos, idx in enumerate(indices)}
        for seed_idx in seeds:
            seed_pos = pos_by_idx[seed_idx]
            order = [indices[pos] for pos in np.argsort(-source_sim[seed_pos]).tolist()]
            group = [seed_idx]
            group_set = {seed_idx}
            for idx in order:
                if idx == seed_idx or idx in group_set:
                    continue
                if len(group) >= int(args.max_group_size):
                    break
                if float(source_sim[seed_pos, pos_by_idx[idx]]) < float(args.seed_sim):
                    continue
                if any(idx in graph[member] for member in group):
                    continue
                group.append(idx)
                group_set.add(idx)
            if len(group) < int(args.min_group_size):
                continue
            stats = _group_stats(x, indices, group, graph)
            if int(stats["conflicts_to_rest"]) < int(args.min_conflicts_to_rest):
                continue
            if float(stats["source_rest_margin_mean"]) < float(args.min_source_margin):
                continue

            group_vec = _mean_vec(x, group)
            target_rows = []
            for target_label, target_indices in comps.items():
                if int(target_label) == int(source_label):
                    continue
                if len(target_indices) < int(args.min_target_component_size) or len(target_indices) > int(args.max_target_component_size):
                    continue
                if _group_has_target_overlap(rows, group, target_indices, int(args.overlap_margin_frames)):
                    continue
                target_sim = float(group_vec @ comp_centroid[target_label])
                target_margin = float(target_sim - max(float(stats["source_rest_cross_max"]), float(stats["source_rest_cross_mean"])))
                if target_sim < float(args.min_target_sim):
                    continue
                if target_margin < float(args.min_target_margin):
                    continue
                target_rows.append((target_sim, target_margin, target_label, target_indices))
            target_rows.sort(reverse=True, key=lambda item: (item[0], item[1], len(item[3])))
            for target_rank, (target_sim, target_margin, target_label, target_indices) in enumerate(target_rows[: int(args.targets_per_group)], start=1):
                seqs = tuple(sorted(_intish(rows[idx]["seq"]) for idx in group))
                key = (seqs, int(target_label))
                if key in seen:
                    continue
                seen.add(key)
                video_counts = Counter(rows[idx]["video"] for idx in group)
                focus_hits = sum(count for video, count in video_counts.items() if video in focus_videos)
                score = (
                    2.5 * target_sim
                    + 1.0 * target_margin
                    + 0.7 * float(stats["source_rest_margin_mean"])
                    + 0.12 * math.log1p(len(group))
                    + 0.08 * math.log1p(len(target_indices))
                    + 0.05 * int(stats["conflicts_to_rest"])
                    + 0.04 * focus_hits
                )
                candidates.append(
                    {
                        "source_component": int(source_label),
                        "target_component": int(target_label),
                        "source_predicted_global_id": int(comp_gid[source_label]),
                        "target_predicted_global_id": int(comp_gid[target_label]),
                        "source_component_size": int(len(indices)),
                        "target_component_size": int(len(target_indices)),
                        "moved_tracklets": int(len(group)),
                        "source_component_conflict_edges": int(conflict_edges),
                        "target_rank_for_group": int(target_rank),
                        "source_seqs": list(seqs),
                        "source_videos": dict(video_counts),
                        "focus_video_hits": int(focus_hits),
                        "target_sim": float(target_sim),
                        "target_margin": float(target_margin),
                        "score": float(score),
                        **stats,
                        "uses_anchors": False,
                        "uses_gt_for_training_or_anchors": False,
                        "uses_gt_for_evaluation_only": False,
                    }
                )
    candidates.sort(
        key=lambda row: (
            float(row["score"]),
            float(row["target_sim"]),
            float(row["target_margin"]),
            int(row["moved_tracklets"]),
        ),
        reverse=True,
    )
    return candidates


def _refresh_component_sizes(rows: list[dict[str, str]]) -> None:
    counts = Counter(_intish(row["component_label"]) for row in rows)
    for row in rows:
        label = _intish(row["component_label"])
        size = counts[label]
        if "component_size" in row:
            row["component_size"] = str(int(size))
        if "decision_status" in row and row.get("decision_status") != "subpart_reassign":
            row["decision_status"] = "forced_singleton" if size == 1 else "forced_component"
        if "prediction_confidence" in row and row.get("decision_status") != "subpart_reassign":
            confidence = 0.15 if size == 1 else min(0.85, 0.30 + 0.02 * min(size, 20))
            row["prediction_confidence"] = f"{confidence:.6f}"


def _write_candidate_csv(path: Path, base_rows: list[dict[str, str]], fieldnames: list[str], cand: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in base_rows]
    by_seq = {_intish(row["seq"]): row for row in rows}
    target_component = int(cand["target_component"])
    target_gid = int(cand["target_predicted_global_id"])
    moved = []
    for seq in cand["source_seqs"]:
        row = by_seq[_intish(seq)]
        moved.append(
            {
                "seq": _intish(seq),
                "from_component": _intish(row["component_label"]),
                "from_predicted_global_id": _intish(row["predicted_global_id"]),
                "to_component": int(target_component),
                "to_predicted_global_id": int(target_gid),
            }
        )
        row["component_label"] = str(int(target_component))
        row["predicted_global_id"] = str(int(target_gid))
        if "decision_status" in row:
            row["decision_status"] = "subpart_reassign"
        if "prediction_confidence" in row:
            row["prediction_confidence"] = f"{max(_floatish(row.get('prediction_confidence'), 0.7), 0.72):.6f}"
    _refresh_component_sizes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"assignment_csv": str(path), "moved_preview": moved}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--primary-weight", type=float, default=1.0)
    ap.add_argument("--view", action="append", default=[])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--min-source-component-size", type=int, default=64)
    ap.add_argument("--max-source-component-size", type=int, default=1000000)
    ap.add_argument("--min-target-component-size", type=int, default=4)
    ap.add_argument("--max-target-component-size", type=int, default=300)
    ap.add_argument("--min-component-conflict-edges", type=int, default=1)
    ap.add_argument("--max-seeds-per-component", type=int, default=16)
    ap.add_argument("--min-group-size", type=int, default=2)
    ap.add_argument("--max-group-size", type=int, default=8)
    ap.add_argument("--seed-sim", type=float, default=0.74)
    ap.add_argument("--min-conflicts-to-rest", type=int, default=1)
    ap.add_argument("--min-source-margin", type=float, default=0.02)
    ap.add_argument("--min-target-sim", type=float, default=0.55)
    ap.add_argument("--min-target-margin", type=float, default=0.02)
    ap.add_argument("--targets-per-group", type=int, default=2)
    ap.add_argument("--overlap-margin-frames", type=int, default=0)
    ap.add_argument("--focus-videos", default="")
    args = ap.parse_args()

    assignment_csv = Path(args.assignment_csv)
    rows, fieldnames = _load_assignment(assignment_csv)
    x, feature_meta = _load_fused_features(Path(args.feature_npz), rows, list(args.view), float(args.primary_weight))
    candidates = _build_candidates(rows, x, args)

    output_dir = Path(args.output_dir)
    selected = []
    for rank, cand in enumerate(candidates[: max(0, int(args.top_n))], start=1):
        stem = (
            f"rank{rank:02d}_subpart_s{cand['source_component']}_to{cand['target_component']}"
            f"_{cand['moved_tracklets']}seq_assignments.csv"
        )
        info = _write_candidate_csv(output_dir / _safe_slug(stem, 120), rows, fieldnames, cand)
        selected.append({"rank": int(rank), **cand, **info})

    result = {
        "assignment_csv": str(assignment_csv),
        "feature_npz": str(args.feature_npz),
        "feature_views": feature_meta,
        "output_dir": str(output_dir),
        "candidate_count": int(len(candidates)),
        "selected": selected,
        "top_candidates": candidates[: max(80, int(args.top_n))],
        "params": vars(args),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"json": str(out), "candidate_count": len(candidates), "selected": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
