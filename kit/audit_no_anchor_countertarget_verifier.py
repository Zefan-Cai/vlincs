#!/usr/bin/env python
"""No-GT counter-target verifier for accepted-preview identity edits.

This is an admission/referee audit, not an identity solver.  Given the current
assignment and candidate rows with ``accepted_preview`` entries, it asks whether
each proposed source island is more similar to its proposed target component
than to the nearest alternative component under the current assignment.

Ground truth is not loaded or used.  The script only reads assignment labels,
tracklet metadata, and feature NPZs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_resolve_sweep import _connect, _load_feature_npz, _load_tracklets
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_resolve_sweep import _connect, _load_feature_npz, _load_tracklets


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_feature_specs(values: list[str]) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for item in values:
        parts = str(item).split(":")
        if len(parts) == 1:
            out.append((Path(parts[0]).stem, parts[0], 1.0))
        elif len(parts) == 2:
            out.append((parts[0], parts[1], 1.0))
        elif len(parts) == 3:
            out.append((parts[0], parts[1], float(parts[2])))
        else:
            raise ValueError(f"bad feature spec {item!r}; expected name:path[:weight]")
    return out


def _iter_candidate_rows(doc: Any, *, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(doc, list):
        pools = [("list", doc)]
    elif isinstance(doc, dict):
        pools = []
        for key in ("selected", "top", "rows", "full_rows", "top_full_rows"):
            value = doc.get(key)
            if isinstance(value, list):
                pools.append((key, value))
    else:
        pools = []
    for pool_name, values in pools:
        for idx, row in enumerate(values):
            if not isinstance(row, dict):
                continue
            preview = row.get("accepted_preview")
            if isinstance(preview, str):
                try:
                    preview = json.loads(preview)
                except Exception:
                    preview = []
            if not isinstance(preview, list) or not preview:
                continue
            clean = dict(row)
            clean["_source_file"] = source_name
            clean["_candidate_pool"] = pool_name
            clean["_candidate_index"] = int(idx)
            clean["accepted_preview"] = preview
            rows.append(clean)
    return rows


def _load_candidates(paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        doc = json.loads(p.read_text())
        out.extend(_iter_candidate_rows(doc, source_name=str(p)))
    return out


def _members_by_label(labels: np.ndarray, assigned_indices: set[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        if int(idx) in assigned_indices:
            out[int(label)].append(int(idx))
    return dict(out)


def _component_representatives(emb: np.ndarray, members: dict[int, list[int]], *, max_members: int) -> dict[int, np.ndarray]:
    reps: dict[int, np.ndarray] = {}
    for label, indices in members.items():
        arr = np.asarray(indices, dtype=np.int64)
        if len(arr) > int(max_members):
            centroid = _l2n(emb[arr].mean(axis=0, keepdims=True))[0]
            sims = emb[arr] @ centroid
            keep = arr[np.argsort(-sims)[: int(max_members)]]
        else:
            keep = arr
        reps[int(label)] = np.asarray(keep, dtype=np.int64)
    return reps


def _score_component(source_vecs: np.ndarray, target_vecs: np.ndarray, *, top_k: int) -> float:
    if len(source_vecs) == 0 or len(target_vecs) == 0:
        return -1.0
    sims = source_vecs @ target_vecs.T
    k = max(1, min(int(top_k), sims.shape[1]))
    top = np.partition(sims, -k, axis=1)[:, -k:]
    return float(np.mean(top))


def _top_target_seqs(records, emb: np.ndarray, source_idx: np.ndarray, target_idx: np.ndarray, *, n: int) -> list[int]:
    if len(source_idx) == 0 or len(target_idx) == 0:
        return []
    src = emb[source_idx]
    tgt = emb[target_idx]
    score = (src @ tgt.T).max(axis=0)
    order = np.argsort(-score)[: int(n)]
    return [int(records[int(target_idx[pos])].seq) for pos in order.tolist()]


def _video_counts(records, indices: np.ndarray | list[int]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for idx in list(indices):
        counts[str(records[int(idx)].video)] += 1
    return dict(sorted(counts.items()))


def _dominant_fraction(counts: dict[str, int]) -> float:
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return 0.0
    return float(max(int(v) for v in counts.values()) / total)


def _video_jaccard(left: dict[str, int], right: dict[str, int]) -> float:
    left_keys = set(left)
    right_keys = set(right)
    union = left_keys | right_keys
    if not union:
        return 0.0
    return float(len(left_keys & right_keys) / len(union))


def _safe_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _interval_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(int(end_a), int(end_b)) - max(int(start_a), int(start_b)) + 1)


def _interval_gap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    if int(end_a) < int(start_b):
        return int(start_b) - int(end_a) - 1
    if int(end_b) < int(start_a):
        return int(start_a) - int(end_b) - 1
    return 0


def _temporal_opponent(
    records,
    *,
    source_idx: np.ndarray,
    target_idx: list[int],
    args,
) -> dict[str, Any]:
    overlap_pairs: list[dict[str, Any]] = []
    overlap_pair_count = 0
    local_pairs = 0
    same_video_pairs = 0
    min_gap: int | None = None
    max_overlap = 0
    source_durations = []
    total_pairs = int(len(source_idx) * len(target_idx))
    for s_idx in source_idx.tolist():
        s = records[int(s_idx)]
        source_durations.append(max(1, int(s.end_frame) - int(s.start_frame) + 1))
    for s_idx in source_idx.tolist():
        s = records[int(s_idx)]
        for t_idx in target_idx:
            t = records[int(t_idx)]
            if str(s.video) != str(t.video):
                continue
            same_video_pairs += 1
            overlap = _interval_overlap(int(s.start_frame), int(s.end_frame), int(t.start_frame), int(t.end_frame))
            gap = _interval_gap(int(s.start_frame), int(s.end_frame), int(t.start_frame), int(t.end_frame))
            if min_gap is None or gap < min_gap:
                min_gap = int(gap)
            if gap <= int(args.temporal_local_window_frames):
                local_pairs += 1
            if overlap > 0:
                overlap_pair_count += 1
                max_overlap = max(max_overlap, int(overlap))
                if len(overlap_pairs) < int(args.temporal_overlap_examples):
                    overlap_pairs.append(
                        {
                            "source_seq": int(s.seq),
                            "target_seq": int(t.seq),
                            "video": str(s.video),
                            "source_span": [int(s.start_frame), int(s.end_frame)],
                            "target_span": [int(t.start_frame), int(t.end_frame)],
                            "overlap_frames": int(overlap),
                        }
                    )
    max_allowed = int(args.max_same_video_overlap_frames)
    rejects = max_allowed >= 0 and max_overlap > max_allowed
    same_video_pair_fraction = float(same_video_pairs / max(1, total_pairs))
    local_pair_fraction_total = float(local_pairs / max(1, total_pairs))
    local_pair_fraction_same_video = float(local_pairs / max(1, same_video_pairs))
    overlap_pair_fraction_total = float(overlap_pair_count / max(1, total_pairs))
    overlap_pair_fraction_same_video = float(overlap_pair_count / max(1, same_video_pairs))
    median_source_duration = float(np.median(np.asarray(source_durations, dtype=np.float64))) if source_durations else 0.0
    max_overlap_source_duration_fraction = float(max_overlap / max(1.0, median_source_duration))
    temporal_risk = max(
        overlap_pair_fraction_same_video,
        min(1.0, max_overlap_source_duration_fraction),
        0.5 * local_pair_fraction_same_video,
    )
    return {
        "enabled": max_allowed >= 0,
        "max_same_video_overlap_frames": int(max_overlap),
        "same_video_pair_count": int(same_video_pairs),
        "total_pair_count": int(total_pairs),
        "local_pair_count": int(local_pairs),
        "min_same_video_gap_frames": None if min_gap is None else int(min_gap),
        "overlap_pair_count": int(overlap_pair_count),
        "same_video_pair_fraction": same_video_pair_fraction,
        "local_pair_fraction_total": local_pair_fraction_total,
        "local_pair_fraction_same_video": local_pair_fraction_same_video,
        "overlap_pair_fraction_total": overlap_pair_fraction_total,
        "overlap_pair_fraction_same_video": overlap_pair_fraction_same_video,
        "median_source_duration_frames": median_source_duration,
        "max_overlap_source_duration_fraction": max_overlap_source_duration_fraction,
        "temporal_opponent_risk_score": float(min(1.0, temporal_risk)),
        "overlap_examples": overlap_pairs,
        "rejects": bool(rejects),
    }


def _edge_from_preview(
    *,
    preview: dict[str, Any],
    row: dict[str, Any],
    edge_index: int,
    records,
    seq_to_idx: dict[int, int],
    label_by_idx: dict[int, int],
    component_members: dict[int, list[int]],
    component_reps_by_view: dict[str, dict[int, np.ndarray]],
    emb_by_view: dict[str, np.ndarray],
    view_weights: dict[str, float],
    args,
) -> dict[str, Any] | None:
    source_seqs = [int(x) for x in preview.get("source_seqs", []) if str(x).strip()]
    source_idx = [seq_to_idx[seq] for seq in source_seqs if seq in seq_to_idx]
    target_component = preview.get("target_component", row.get("target_component"))
    if target_component is None:
        return None
    target_component = int(float(target_component))
    if not source_idx or target_component not in component_members:
        return None
    source_idx_arr = np.asarray(sorted(set(source_idx)), dtype=np.int64)
    source_component_counts: dict[int, int] = defaultdict(int)
    for idx in source_idx_arr.tolist():
        source_component_counts[int(label_by_idx[int(idx)])] += 1
    source_component = max(source_component_counts.items(), key=lambda item: item[1])[0]

    view_details: dict[str, dict[str, Any]] = {}
    weighted_target_score = 0.0
    weighted_best_alt_score = 0.0
    total_weight = 0.0
    target_rank_votes = 0
    margin_votes = 0
    for name, emb in emb_by_view.items():
        reps_by_label = component_reps_by_view[name]
        source_vecs = emb[source_idx_arr]
        scores = []
        for label, target_idx in reps_by_label.items():
            if int(label) == int(source_component):
                continue
            if len(component_members.get(int(label), [])) < int(args.min_counter_component_size):
                continue
            score = _score_component(source_vecs, emb[target_idx], top_k=int(args.top_k))
            scores.append((score, int(label)))
        scores.sort(reverse=True)
        target_score = None
        for score, label in scores:
            if int(label) == int(target_component):
                target_score = float(score)
                break
        if target_score is None:
            continue
        target_rank = next((rank for rank, (_score, label) in enumerate(scores, start=1) if label == target_component), 10**9)
        alternatives = [(score, label) for score, label in scores if label != target_component]
        best_alt_score, best_alt_label = alternatives[0] if alternatives else (-1.0, -1)
        margin = float(target_score - best_alt_score)
        weight = float(view_weights.get(name, 1.0))
        total_weight += weight
        weighted_target_score += weight * target_score
        weighted_best_alt_score += weight * float(best_alt_score)
        if int(target_rank) <= int(args.max_accept_rank):
            target_rank_votes += 1
        if margin >= float(args.min_accept_margin):
            margin_votes += 1
        view_details[name] = {
            "target_score": float(target_score),
            "target_rank": int(target_rank),
            "best_alt_score": float(best_alt_score),
            "best_alt_component": int(best_alt_label),
            "margin": float(margin),
            "best_alt_top_seqs": _top_target_seqs(records, emb, source_idx_arr, reps_by_label.get(int(best_alt_label), np.asarray([], dtype=np.int64)), n=int(args.top_seq_n)),
            "target_top_seqs": _top_target_seqs(records, emb, source_idx_arr, reps_by_label[int(target_component)], n=int(args.top_seq_n)),
        }
    if total_weight <= 0.0 or not view_details:
        return None
    score = weighted_target_score / total_weight
    alt = weighted_best_alt_score / total_weight
    margin = score - alt
    view_count = len(view_details)
    rank_vote = target_rank_votes / max(1, view_count)
    margin_vote = margin_votes / max(1, view_count)
    view_margins = [float(detail["margin"]) for detail in view_details.values()]
    view_target_scores = [float(detail["target_score"]) for detail in view_details.values()]
    view_ranks = [int(detail["target_rank"]) for detail in view_details.values()]
    margin_stats = _safe_stats(view_margins)
    target_score_stats = _safe_stats(view_target_scores)
    weak_margin_count = sum(1 for value in view_margins if value < float(args.min_accept_margin))
    negative_margin_count = sum(1 for value in view_margins if value < 0.0)
    non_rank1_count = sum(1 for value in view_ranks if value > int(args.max_accept_rank))
    weak_margin_fraction = float(weak_margin_count / max(1, view_count))
    negative_margin_fraction = float(negative_margin_count / max(1, view_count))
    non_rank1_fraction = float(non_rank1_count / max(1, view_count))
    margin_deficit = max(0.0, float(args.min_weighted_margin) - float(margin)) / max(1.0e-9, float(args.min_weighted_margin))
    visual_risk = min(
        1.0,
        0.45 * weak_margin_fraction
        + 0.30 * non_rank1_fraction
        + 0.25 * negative_margin_fraction
        + 0.25 * min(1.0, margin_deficit),
    )
    visual_accept = (
        rank_vote >= float(args.min_rank_vote)
        and margin_vote >= float(args.min_margin_vote)
        and margin >= float(args.min_weighted_margin)
    )
    target_idx_arr = np.asarray(component_members.get(int(target_component), []), dtype=np.int64)
    source_video_counts = _video_counts(records, source_idx_arr)
    target_video_counts = _video_counts(records, target_idx_arr)
    video_jaccard = _video_jaccard(source_video_counts, target_video_counts)
    source_dominant_video_fraction = _dominant_fraction(source_video_counts)
    target_dominant_video_fraction = _dominant_fraction(target_video_counts)
    temporal = _temporal_opponent(
        records,
        source_idx=source_idx_arr,
        target_idx=target_idx_arr.tolist(),
        args=args,
    )
    if visual_accept and temporal["rejects"]:
        verdict = "reject_temporal_overlap"
    elif visual_accept:
        verdict = "accept"
    else:
        verdict = "reject_countertarget"
    return {
        "source_file": row.get("_source_file"),
        "candidate_pool": row.get("_candidate_pool"),
        "candidate_index": int(row.get("_candidate_index", -1)),
        "edge_index": int(edge_index),
        "source_component": int(source_component),
        "source_seqs": source_seqs,
        "source_size": int(len(source_idx_arr)),
        "target_component": int(target_component),
        "target_size": int(len(component_members.get(int(target_component), []))),
        "weighted_target_score": float(score),
        "weighted_best_alt_score": float(alt),
        "weighted_margin": float(margin),
        "rank_vote": float(rank_vote),
        "margin_vote": float(margin_vote),
        "view_count": int(view_count),
        "view_margin_min": margin_stats["min"],
        "view_margin_mean": margin_stats["mean"],
        "view_margin_max": margin_stats["max"],
        "view_margin_std": margin_stats["std"],
        "view_target_score_min": target_score_stats["min"],
        "view_target_score_mean": target_score_stats["mean"],
        "view_target_score_max": target_score_stats["max"],
        "view_target_score_std": target_score_stats["std"],
        "view_target_rank_max": int(max(view_ranks) if view_ranks else 0),
        "view_weak_margin_count": int(weak_margin_count),
        "view_negative_margin_count": int(negative_margin_count),
        "view_non_rank1_count": int(non_rank1_count),
        "view_weak_margin_fraction": weak_margin_fraction,
        "view_negative_margin_fraction": negative_margin_fraction,
        "view_non_rank1_fraction": non_rank1_fraction,
        "visual_opponent_risk_score": float(visual_risk),
        "combined_opponent_risk_score": float(max(float(visual_risk), float(temporal["temporal_opponent_risk_score"]))),
        "source_video_count": int(len(source_video_counts)),
        "target_video_count": int(len(target_video_counts)),
        "shared_video_count": int(len(set(source_video_counts) & set(target_video_counts))),
        "source_target_video_jaccard": float(video_jaccard),
        "source_dominant_video_fraction": float(source_dominant_video_fraction),
        "target_dominant_video_fraction": float(target_dominant_video_fraction),
        "verdict": verdict,
        "preview_target_mean_sim": preview.get("target_mean_sim"),
        "preview_target_best_sim": preview.get("target_best_sim"),
        "preview_target_view_vote": preview.get("target_view_vote"),
        "preview_target_margin": preview.get("target_margin"),
        "row_full_idf1": row.get("full_idf1"),
        "row_predicted_full_idf1": row.get("predicted_full_idf1", row.get("learned_proxy_full_idf1")),
        "source_video_counts": source_video_counts,
        "target_video_counts": target_video_counts,
        "temporal_opponent": temporal,
        "view_details": view_details,
    }


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    scalar_keys = sorted({k for row in rows for k, v in row.items() if not isinstance(v, (dict, list, tuple))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--candidate-json", action="append", required=True)
    ap.add_argument("--feature", action="append", required=True, help="name:path[:weight]")
    ap.add_argument("--max-component-reps", type=int, default=64)
    ap.add_argument("--min-counter-component-size", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--top-seq-n", type=int, default=3)
    ap.add_argument("--max-accept-rank", type=int, default=1)
    ap.add_argument("--min-accept-margin", type=float, default=0.03)
    ap.add_argument("--min-weighted-margin", type=float, default=0.03)
    ap.add_argument("--min-rank-vote", type=float, default=0.75)
    ap.add_argument("--min-margin-vote", type=float, default=0.50)
    ap.add_argument(
        "--max-same-video-overlap-frames",
        type=int,
        default=-1,
        help="Enable temporal cannot-link opponent when >=0; reject accepted edges with same-video source/target overlap above this frame count.",
    )
    ap.add_argument("--temporal-local-window-frames", type=int, default=1500)
    ap.add_argument("--temporal-overlap-examples", type=int, default=8)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    labels, raw_to_local = _labels_from_assignment(records, pred_input)
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    assigned_indices = {idx for idx, record in enumerate(records) if int(record.seq) in pred_input}
    label_by_idx = {idx: int(labels[idx]) for idx in range(len(labels))}
    members = _members_by_label(labels, assigned_indices)

    emb_by_view: dict[str, np.ndarray] = {}
    view_weights: dict[str, float] = {}
    component_reps_by_view: dict[str, dict[int, np.ndarray]] = {}
    for name, path, weight in _parse_feature_specs(args.feature):
        emb = _load_feature_npz(path, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0).astype(np.float32)
        emb = _l2n(emb)
        emb_by_view[name] = emb
        view_weights[name] = float(weight)
        component_reps_by_view[name] = _component_representatives(emb, members, max_members=int(args.max_component_reps))

    candidate_rows = _load_candidates(args.candidate_json)
    edge_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        preview = row.get("accepted_preview", [])
        if not isinstance(preview, list):
            continue
        for edge_index, item in enumerate(preview):
            if not isinstance(item, dict):
                continue
            out = _edge_from_preview(
                preview=item,
                row=row,
                edge_index=edge_index,
                records=records,
                seq_to_idx=seq_to_idx,
                label_by_idx=label_by_idx,
                component_members=members,
                component_reps_by_view=component_reps_by_view,
                emb_by_view=emb_by_view,
                view_weights=view_weights,
                args=args,
            )
            if out is not None:
                edge_rows.append(out)

    by_candidate: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edge_rows:
        by_candidate[(str(edge["source_file"]), str(edge["candidate_pool"]), int(edge["candidate_index"]))].append(edge)
    candidate_summaries = []
    for key, edges in by_candidate.items():
        accepted = [edge for edge in edges if edge["verdict"] == "accept"]
        temporal_rejected = [edge for edge in edges if edge["verdict"] == "reject_temporal_overlap"]
        countertarget_rejected = [edge for edge in edges if edge["verdict"] == "reject_countertarget"]
        if len(accepted) == len(edges) and edges:
            candidate_verdict = "accept"
        elif temporal_rejected:
            candidate_verdict = "reject_temporal_overlap"
        else:
            candidate_verdict = "reject_countertarget"
        candidate_summaries.append(
            {
                "source_file": key[0],
                "candidate_pool": key[1],
                "candidate_index": key[2],
                "edges": int(len(edges)),
                "accepted_edges": int(len(accepted)),
                "temporal_rejected_edges": int(len(temporal_rejected)),
                "countertarget_rejected_edges": int(len(countertarget_rejected)),
                "accepted_fraction": float(len(accepted) / max(1, len(edges))),
                "min_weighted_margin": float(min(float(edge["weighted_margin"]) for edge in edges)),
                "mean_weighted_margin": float(np.mean([float(edge["weighted_margin"]) for edge in edges])),
                "mean_rank_vote": float(np.mean([float(edge["rank_vote"]) for edge in edges])),
                "min_view_margin_min": float(min(float(edge["view_margin_min"]) for edge in edges)),
                "mean_view_margin_min": float(np.mean([float(edge["view_margin_min"]) for edge in edges])),
                "mean_view_weak_margin_fraction": float(np.mean([float(edge["view_weak_margin_fraction"]) for edge in edges])),
                "mean_view_non_rank1_fraction": float(np.mean([float(edge["view_non_rank1_fraction"]) for edge in edges])),
                "mean_visual_opponent_risk_score": float(np.mean([float(edge["visual_opponent_risk_score"]) for edge in edges])),
                "max_temporal_opponent_risk_score": float(max(float(edge["temporal_opponent"]["temporal_opponent_risk_score"]) for edge in edges)),
                "max_combined_opponent_risk_score": float(max(float(edge["combined_opponent_risk_score"]) for edge in edges)),
                "max_same_video_overlap_frames": int(max(int(edge["temporal_opponent"]["max_same_video_overlap_frames"]) for edge in edges)),
                "mean_same_video_pair_fraction": float(np.mean([float(edge["temporal_opponent"]["same_video_pair_fraction"]) for edge in edges])),
                "mean_local_pair_fraction_same_video": float(np.mean([float(edge["temporal_opponent"]["local_pair_fraction_same_video"]) for edge in edges])),
                "mean_source_target_video_jaccard": float(np.mean([float(edge["source_target_video_jaccard"]) for edge in edges])),
                "verdict": candidate_verdict,
            }
        )
    candidate_summaries.sort(key=lambda row: (row["verdict"] == "accept", row["mean_weighted_margin"], row["accepted_fraction"]), reverse=True)
    result = {
        "assignment_csv": args.assignment_csv,
        "candidate_json": args.candidate_json,
        "features": [{"name": name, "path": path, "weight": weight} for name, path, weight in _parse_feature_specs(args.feature)],
        "assignment_components": int(len(raw_to_local)),
        "assigned_tracklets": int(len(assigned_indices)),
        "candidate_rows": int(len(candidate_rows)),
        "edge_rows": edge_rows,
        "candidate_summaries": candidate_summaries,
        "thresholds": {
            "max_accept_rank": int(args.max_accept_rank),
            "min_accept_margin": float(args.min_accept_margin),
            "min_weighted_margin": float(args.min_weighted_margin),
            "min_rank_vote": float(args.min_rank_vote),
            "min_margin_vote": float(args.min_margin_vote),
            "max_same_video_overlap_frames": int(args.max_same_video_overlap_frames),
            "temporal_local_window_frames": int(args.temporal_local_window_frames),
        },
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, edge_rows)
    print(json.dumps({"json": str(out), "edges": len(edge_rows), "candidates": len(candidate_summaries), "accepted_edges": sum(1 for row in edge_rows if row["verdict"] == "accept")}, sort_keys=True))


if __name__ == "__main__":
    main()
