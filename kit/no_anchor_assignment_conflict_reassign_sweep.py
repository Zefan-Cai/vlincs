#!/usr/bin/env python
"""No-anchor conflict-subcluster reassignment sweep.

This is a conservative follow-up to conflict-state splitting.  It treats visual
subclusters inside conflicted components as provisional evidence, but only emits
a final ID change when that subcluster has a strong existing target component.
No new delivery IDs are created for unresolved subclusters.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, KIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_provisional_relink_sweep import (
        _l2n,
        _load_npz_aligned,
        _parse_view,
        _source_candidates,
        _weighted_query_scores,
    )
    from kit.no_anchor_assignment_state_policy_sweep import (
        _assign_states,
        _component_stats,
        _labels_from_assignment,
        _load_assignment_labels,
    )
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from kit.no_anchor_louvain_sweep import _write_assignments
    from kit.no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _tracklet_quality_score,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_provisional_relink_sweep import (
        _l2n,
        _load_npz_aligned,
        _parse_view,
        _source_candidates,
        _weighted_query_scores,
    )
    from no_anchor_assignment_state_policy_sweep import (
        _assign_states,
        _component_stats,
        _labels_from_assignment,
        _load_assignment_labels,
    )
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from no_anchor_louvain_sweep import _write_assignments
    from no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _tracklet_quality_score,
        _with_detection_endpoints,
    )


def _admission_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        output_min_dets=int(args.output_min_dets),
        output_min_conf=float(args.output_min_conf),
        output_min_area=float(args.output_min_area),
        output_min_quality=float(args.output_min_quality),
        output_min_area_by_video=str(args.output_min_area_by_video),
        output_drop_area_quantile=0.0,
        output_drop_area_quantile_by_video="",
        output_drop_quality_quantile=0.0,
        output_drop_quality_quantile_by_video="",
        output_auto_anomaly_admission=False,
        output_auto_anomaly_metric="quality",
        output_auto_anomaly_quantile=0.75,
        output_auto_anomaly_area_ratio=0.60,
        output_auto_anomaly_quality_mad=1.0,
        output_auto_anomaly_min_video_tracklets=20,
        output_auto_anomaly_max_videos=3,
    )


def _component_groups(labels: np.ndarray, keep_indices: set[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        groups[int(labels[int(idx)])].append(int(idx))
    return dict(groups)


def _target_score_for_component(
    records,
    target_indices: list[int],
    source_group: set[int],
    forbidden: list[set[int]],
    mean_sim: np.ndarray,
    min_sim: np.ndarray,
    view_stack: np.ndarray,
    *,
    view_sim_threshold: float,
    min_view_vote: float,
    top_k: int,
    max_forbidden_pairs: int,
) -> dict[str, object] | None:
    clean = []
    forbidden_pairs = 0
    source_set = {int(idx) for idx in source_group}
    for idx in target_indices:
        idx = int(idx)
        overlap = len(forbidden[idx] & source_set)
        forbidden_pairs += int(overlap)
        if overlap == 0:
            clean.append(idx)
    if forbidden_pairs > int(max_forbidden_pairs):
        return None
    pool = clean if int(max_forbidden_pairs) == 0 else [int(idx) for idx in target_indices]
    if not pool:
        return None
    pool_arr = np.asarray(pool, dtype=np.int64)
    vote = (view_stack[:, pool_arr] >= float(view_sim_threshold)).mean(axis=0)
    vote_by_idx = {int(idx): float(vote[pos]) for pos, idx in enumerate(pool)}
    ranked = sorted(
        pool,
        key=lambda idx: (
            -float(mean_sim[idx]),
            -float(min_sim[idx]),
            -float(vote_by_idx[int(idx)]),
            -_tracklet_quality_score(records[idx]),
            int(idx),
        ),
    )
    top = ranked[: max(int(top_k), 1)]
    top_vote = (view_stack[:, np.asarray(top, dtype=np.int64)] >= float(view_sim_threshold)).mean(axis=0)
    vote_mean = float(np.mean(top_vote)) if len(top) else 0.0
    if vote_mean < float(min_view_vote):
        return None
    return {
        "target_indices": top,
        "target_best_sim": float(max(mean_sim[idx] for idx in top)),
        "target_mean_sim": float(np.mean([mean_sim[idx] for idx in top])),
        "target_min_view_sim": float(np.min([min_sim[idx] for idx in top])),
        "target_view_vote": vote_mean,
        "target_forbidden_pairs": int(forbidden_pairs),
        "target_top_seqs": [int(records[idx].seq) for idx in top],
        "target_quality": float(np.mean([_tracklet_quality_score(records[idx]) for idx in top])),
    }


def _choose_targets(
    records,
    base_labels: np.ndarray,
    keep_indices: set[int],
    sources: list[dict[str, object]],
    views: list[dict[str, object]],
    forbidden: list[set[int]],
    states: dict[int, str],
    *,
    target_states: set[str],
    min_target_size: int,
    target_top_k: int,
    min_target_best_sim: float,
    min_target_mean_sim: float,
    min_target_view_vote: float,
    min_target_quality: float,
    target_view_sim_threshold: float,
    min_target_margin: float,
    max_forbidden_pairs: int,
    max_sources_per_target: int,
    max_reassignments: int,
    query_score_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups = _component_groups(base_labels, keep_indices)
    accepted: list[dict[str, object]] = []
    target_use = Counter()
    used_source: set[int] = set()
    rejected_overlap = rejected_no_target = rejected_margin = rejected_threshold = 0

    for source in sources:
        if len(accepted) >= int(max_reassignments):
            break
        source_indices = [int(idx) for idx in source["source_indices"]]
        source_set = set(source_indices)
        if source_set & used_source:
            rejected_overlap += 1
            continue
        source_component = int(source["source_component_label"])
        source_key = tuple(source_indices)
        if query_score_cache is not None and source_key in query_score_cache:
            mean_sim, min_sim, view_stack, source_self = query_score_cache[source_key]
        else:
            mean_sim, min_sim, view_stack, source_self = _weighted_query_scores(views, source_indices)
            if query_score_cache is not None:
                query_score_cache[source_key] = (mean_sim, min_sim, view_stack, source_self)
        candidates = []
        for label, target_indices in groups.items():
            label = int(label)
            if label == source_component:
                continue
            if states.get(label, "provisional") not in target_states:
                continue
            if len(target_indices) < int(min_target_size):
                continue
            if int(max_sources_per_target) > 0 and target_use[label] >= int(max_sources_per_target):
                continue
            stats = _target_score_for_component(
                records,
                target_indices,
                source_set,
                forbidden,
                mean_sim,
                min_sim,
                view_stack,
                view_sim_threshold=float(target_view_sim_threshold),
                min_view_vote=float(min_target_view_vote),
                top_k=int(target_top_k),
                max_forbidden_pairs=int(max_forbidden_pairs),
            )
            if stats is None:
                continue
            if float(stats["target_best_sim"]) < float(min_target_best_sim):
                rejected_threshold += 1
                continue
            if float(stats["target_mean_sim"]) < float(min_target_mean_sim):
                rejected_threshold += 1
                continue
            if float(stats["target_quality"]) < float(min_target_quality):
                rejected_threshold += 1
                continue
            score = (
                0.50 * float(stats["target_mean_sim"])
                + 0.25 * float(stats["target_best_sim"])
                + 0.15 * float(stats["target_view_vote"])
                + 0.05 * min(np.log1p(len(target_indices)) / np.log(256.0), 1.0)
                + 0.05 * float(stats["target_quality"])
            )
            candidates.append({**stats, "target_component": label, "target_size": len(target_indices), "target_score": score})
        if not candidates:
            rejected_no_target += 1
            continue
        candidates.sort(key=lambda row: float(row["target_score"]), reverse=True)
        best = candidates[0]
        second = candidates[1]["target_score"] if len(candidates) > 1 else -1.0e9
        margin = float(best["target_score"]) - float(second)
        if margin < float(min_target_margin):
            rejected_margin += 1
            continue
        target_label = int(best["target_component"])
        accepted.append(
            {
                **{k: v for k, v in source.items() if k != "source_indices"},
                **best,
                "target_margin": float(margin),
                "source_self_view_sim": source_self,
            }
        )
        target_use[target_label] += 1
        used_source.update(source_set)

    return accepted, {
        "accepted_reassignments": int(len(accepted)),
        "moved_tracklets": int(sum(int(row["source_size"]) for row in accepted)),
        "target_components_used": int(len(target_use)),
        "rejected_overlap": int(rejected_overlap),
        "rejected_no_target": int(rejected_no_target),
        "rejected_margin": int(rejected_margin),
        "rejected_threshold": int(rejected_threshold),
        "accepted_preview": accepted[:20],
    }


def _apply_reassignments(base_labels: np.ndarray, accepted: list[dict[str, object]]) -> np.ndarray:
    labels = base_labels.copy()
    for row in accepted:
        target = int(row["target_component"])
        for idx in row.get("source_indices", []):
            labels[int(idx)] = target
    return labels


def _restore_source_indices(accepted: list[dict[str, object]], source_map: dict[tuple[int, ...], list[int]]) -> list[dict[str, object]]:
    restored = []
    for row in accepted:
        seq_key = tuple(int(seq) for seq in row.get("source_seqs", []))
        restored.append({**row, "source_indices": source_map[seq_key]})
    return restored


def _restore_preview_indices(row: dict[str, object], seq_to_idx: dict[int, int]) -> list[dict[str, object]]:
    restored = []
    preview = row.get("accepted_preview", [])
    if not isinstance(preview, list):
        return restored
    for item in preview:
        if not isinstance(item, dict):
            continue
        source_seqs = [int(seq) for seq in item.get("source_seqs", [])]
        source_indices = [int(seq_to_idx[seq]) for seq in source_seqs if int(seq) in seq_to_idx]
        if not source_indices:
            continue
        restored.append({**item, "source_indices": source_indices})
    return restored


def _accepted_signature(accepted: list[dict[str, object]]) -> tuple[tuple[tuple[int, ...], int], ...]:
    return tuple(
        sorted(
            (
                tuple(int(seq) for seq in row.get("source_seqs", [])),
                int(row.get("target_component", -1)),
            )
            for row in accepted
        )
    )


def _row_signature(row: dict[str, object]) -> tuple[tuple[tuple[int, ...], int], ...]:
    preview = row.get("accepted_preview", [])
    if not isinstance(preview, list):
        return tuple()
    return _accepted_signature([item for item in preview if isinstance(item, dict)])


def _source_target_key(row: dict[str, object]) -> tuple[tuple[int, ...], int]:
    return (
        tuple(int(seq) for seq in row.get("source_seqs", [])),
        int(row.get("target_component", -1)),
    )


def _row_first_edge_key(row: dict[str, object]) -> tuple[tuple[int, ...], int]:
    preview = row.get("accepted_preview", [])
    if not isinstance(preview, list) or not preview or not isinstance(preview[0], dict):
        return (tuple(), -1)
    return _source_target_key(preview[0])


def _skip_first_edge_families(edges: list[dict[str, object]], skip_families: int) -> list[dict[str, object]]:
    if int(skip_families) <= 0:
        return edges
    banned: set[tuple[tuple[int, ...], int]] = set()
    for row in edges:
        key = _source_target_key(row)
        if key in banned:
            continue
        banned.add(key)
        if len(banned) >= int(skip_families):
            break
    if not banned:
        return edges
    return [row for row in edges if _source_target_key(row) not in banned]


def _first_edge_family_rank(edges: list[dict[str, object]], row: dict[str, object]) -> int:
    target = _source_target_key(row)
    seen: set[tuple[tuple[int, ...], int]] = set()
    for edge in edges:
        key = _source_target_key(edge)
        if key in seen:
            continue
        seen.add(key)
        if key == target:
            return len(seen) - 1
    return -1


def _model_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _source_acceptor_score(row: dict[str, object], model: dict[str, object] | None) -> float:
    if not model:
        return 0.0
    columns = model.get("columns", [])
    coefs = model.get("coef", [])
    means = model.get("mean", [])
    scales = model.get("scale", [])
    fills = model.get("fill_values", {})
    if not isinstance(columns, list) or not isinstance(coefs, list):
        return 0.0
    z = _model_float(model.get("intercept"), 0.0)
    for idx, key in enumerate(columns):
        if idx >= len(coefs) or not isinstance(key, str):
            break
        value = row.get(key)
        if value is None and isinstance(fills, dict):
            value = fills.get(key)
        raw = _model_float(value, 0.0)
        mean = _model_float(means[idx] if idx < len(means) else 0.0, 0.0)
        scale = _model_float(scales[idx] if idx < len(scales) else 1.0, 1.0)
        if abs(scale) < 1.0e-9:
            scale = 1.0
        z += _model_float(coefs[idx], 0.0) * ((raw - mean) / scale)
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0))))


def _build_candidate_edges(
    records,
    base_labels: np.ndarray,
    keep_indices: set[int],
    sources: list[dict[str, object]],
    views: list[dict[str, object]],
    forbidden: list[set[int]],
    states: dict[int, str],
    *,
    target_states: set[str],
    min_target_size: int,
    target_top_k: int,
    min_target_best_sim: float,
    min_target_mean_sim: float,
    min_target_view_vote: float,
    min_target_quality: float,
    target_view_sim_threshold: float,
    min_target_margin: float,
    max_forbidden_pairs: int,
    targets_per_source: int,
    query_score_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]],
    source_acceptor_model: dict[str, object] | None = None,
    source_context: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    groups = _component_groups(base_labels, keep_indices)
    dedup: dict[tuple[tuple[int, ...], int], dict[str, object]] = {}
    for source in sources:
        source_context = source_context or {}
        source_acceptor_score = _source_acceptor_score({**source_context, **source}, source_acceptor_model)
        source_indices = [int(idx) for idx in source["source_indices"]]
        source_set = set(source_indices)
        source_component = int(source["source_component_label"])
        source_key = tuple(source_indices)
        if source_key in query_score_cache:
            mean_sim, min_sim, view_stack, source_self = query_score_cache[source_key]
        else:
            mean_sim, min_sim, view_stack, source_self = _weighted_query_scores(views, source_indices)
            query_score_cache[source_key] = (mean_sim, min_sim, view_stack, source_self)
        candidates = []
        for label, target_indices in groups.items():
            label = int(label)
            if label == source_component:
                continue
            if states.get(label, "provisional") not in target_states:
                continue
            if len(target_indices) < int(min_target_size):
                continue
            stats = _target_score_for_component(
                records,
                target_indices,
                source_set,
                forbidden,
                mean_sim,
                min_sim,
                view_stack,
                view_sim_threshold=float(target_view_sim_threshold),
                min_view_vote=float(min_target_view_vote),
                top_k=int(target_top_k),
                max_forbidden_pairs=int(max_forbidden_pairs),
            )
            if stats is None:
                continue
            if float(stats["target_best_sim"]) < float(min_target_best_sim):
                continue
            if float(stats["target_mean_sim"]) < float(min_target_mean_sim):
                continue
            if float(stats["target_quality"]) < float(min_target_quality):
                continue
            target_score = (
                0.50 * float(stats["target_mean_sim"])
                + 0.25 * float(stats["target_best_sim"])
                + 0.15 * float(stats["target_view_vote"])
                + 0.05 * min(np.log1p(len(target_indices)) / np.log(256.0), 1.0)
                + 0.05 * float(stats["target_quality"])
            )
            candidates.append({**stats, "target_component": label, "target_size": len(target_indices), "target_score": target_score})
        if not candidates:
            continue
        candidates.sort(key=lambda row: float(row["target_score"]), reverse=True)
        second = float(candidates[1]["target_score"]) if len(candidates) > 1 else -1.0e9
        for rank, target in enumerate(candidates[: max(int(targets_per_source), 1)], start=1):
            margin = float(target["target_score"]) - second if rank == 1 else float(target["target_score"] - candidates[0]["target_score"])
            if rank == 1 and margin < float(min_target_margin):
                continue
            proposal_score = (
                float(target["target_score"])
                + 0.45 * float(source_acceptor_score)
                + 0.20 * float(source.get("source_score", 0.0))
                + 0.08 * min(np.log1p(int(source.get("source_size", 1))) / np.log(16.0), 1.0)
                + 0.04 * float(source.get("source_quality", 0.0))
                + 0.03 * float(source.get("source_margin_mean", 0.0))
            )
            row = {
                **{k: v for k, v in source.items() if k != "source_indices"},
                **target,
                "source_indices": source_indices,
                "target_margin": float(margin),
                "target_rank_for_source": int(rank),
                "source_self_view_sim": source_self,
                "source_acceptor_score": float(source_acceptor_score),
                "proposal_score": float(proposal_score),
            }
            sig = (tuple(int(seq) for seq in row.get("source_seqs", [])), int(row["target_component"]))
            prev = dedup.get(sig)
            if prev is None or float(row["proposal_score"]) > float(prev["proposal_score"]):
                dedup[sig] = row
    edges = list(dedup.values())
    edges.sort(
        key=lambda row: (
            float(row["proposal_score"]),
            float(row["target_score"]),
            float(row.get("source_score", 0.0)),
            int(row.get("source_size", 0)),
        ),
        reverse=True,
    )
    return edges


def _greedy_from_candidate_edges(
    edges: list[dict[str, object]],
    *,
    prefix: int,
    max_sources_per_target: int,
    max_reassignments: int,
) -> list[dict[str, object]]:
    accepted: list[dict[str, object]] = []
    used_source: set[int] = set()
    target_use = Counter()
    for row in edges[: max(int(prefix), 0)]:
        if len(accepted) >= int(max_reassignments):
            break
        source_indices = [int(idx) for idx in row.get("source_indices", [])]
        source_set = set(source_indices)
        if source_set & used_source:
            continue
        target = int(row["target_component"])
        if int(max_sources_per_target) > 0 and target_use[target] >= int(max_sources_per_target):
            continue
        accepted.append(row)
        used_source.update(source_set)
        target_use[target] += 1
    return accepted


def _edge_mass_bridge_score(row: dict[str, object]) -> float:
    source_size = _safe_float(row.get("source_size"), 0.0)
    target_size = _safe_float(row.get("target_size"), 0.0)
    moved_bonus = min(np.log1p(max(source_size, 0.0)) / np.log(64.0), 1.0)
    target_bonus = min(np.log1p(max(target_size, 0.0)) / np.log(256.0), 1.0)
    return float(
        0.28 * moved_bonus
        + 0.18 * target_bonus
        + 0.18 * _safe_float(row.get("target_mean_sim"), 0.0)
        + 0.12 * _safe_float(row.get("target_best_sim"), 0.0)
        + 0.10 * _safe_float(row.get("target_view_vote"), 0.0)
        + 0.08 * min(_safe_float(row.get("target_margin"), 0.0), 1.0)
        + 0.06 * _safe_float(row.get("source_acceptor_score"), 0.0)
        + 0.04 * _safe_float(row.get("source_score"), 0.0)
        - 0.03 * max(_safe_float(row.get("target_forbidden_pairs"), 0.0), 0.0)
    )


def _sort_candidate_edges(edges: list[dict[str, object]], rank_by: str) -> None:
    for row in edges:
        row["edge_mass_bridge_score"] = _edge_mass_bridge_score(row)
    if rank_by == "mass_bridge":
        edges.sort(
            key=lambda row: (
                float(row.get("edge_mass_bridge_score", 0.0)),
                float(row.get("proposal_score", 0.0)),
                float(row.get("target_score", 0.0)),
                int(row.get("source_size", 0)),
            ),
            reverse=True,
        )
    else:
        edges.sort(
            key=lambda row: (
                float(row["proposal_score"]),
                float(row["target_score"]),
                float(row.get("source_score", 0.0)),
            ),
            reverse=True,
        )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _preview_mean(row: dict[str, object], key: str, default: float = 0.0) -> float:
    preview = row.get("accepted_preview", [])
    if not isinstance(preview, list) or not preview:
        return float(default)
    values = [_safe_float(item.get(key), default) for item in preview if isinstance(item, dict)]
    return float(np.mean(values)) if values else float(default)


def _preview_min(row: dict[str, object], key: str, default: float = 0.0) -> float:
    preview = row.get("accepted_preview", [])
    if not isinstance(preview, list) or not preview:
        return float(default)
    values = [_safe_float(item.get(key), default) for item in preview if isinstance(item, dict)]
    return float(min(values)) if values else float(default)


def _full_side_effect_proxy(row: dict[str, object]) -> float:
    """No-GT proxy for choosing a few expensive full-score candidates.

    Pair F1 often over-rewards broader edits that later hurt delivery IDF1.
    This proxy prefers high-quality, high-consensus surgical edits and penalizes
    large moves before the GT-backed full scorer is called.
    """

    accepted = _safe_float(row.get("accepted_reassignments"), 0.0)
    moved = _safe_float(row.get("moved_tracklets"), 0.0)
    target_mean = _preview_mean(row, "target_mean_sim")
    target_best = _preview_mean(row, "target_best_sim")
    target_vote = _preview_mean(row, "target_view_vote")
    target_quality = _preview_mean(row, "target_quality")
    source_quality = _preview_mean(row, "source_quality")
    min_view = _preview_min(row, "target_min_view_sim")
    margin = min(_preview_mean(row, "target_margin"), 2.0)
    source_self_primary = _preview_mean(row, "source_self_view_sim", 0.0)
    if source_self_primary == 0.0:
        preview = row.get("accepted_preview", [])
        vals = []
        if isinstance(preview, list):
            for item in preview:
                if isinstance(item, dict) and isinstance(item.get("source_self_view_sim"), dict):
                    vals.append(_safe_float(item["source_self_view_sim"].get("primary"), 0.0))
        source_self_primary = float(np.mean(vals)) if vals else 0.0
    return float(
        0.40 * target_mean
        + 0.18 * target_best
        + 0.18 * target_vote
        + 0.10 * min_view
        + 0.06 * target_quality
        + 0.04 * source_quality
        + 0.04 * source_self_primary
        + 0.03 * margin
        - 0.018 * max(accepted - 2.0, 0.0)
        - 0.004 * max(moved - 16.0, 0.0)
    )


def _learned_proxy_feature(row: dict[str, object], key: str) -> float | None:
    if key.startswith("preview_mean_"):
        return _preview_mean(row, key.removeprefix("preview_mean_"))
    if key.startswith("preview_min_"):
        return _preview_min(row, key.removeprefix("preview_min_"))
    if key.startswith("preview_max_"):
        preview = row.get("accepted_preview", [])
        if not isinstance(preview, list) or not preview:
            return None
        base_key = key.removeprefix("preview_max_")
        values = [_safe_float(item.get(base_key), float("nan")) for item in preview if isinstance(item, dict)]
        values = [value for value in values if np.isfinite(value)]
        return float(max(values)) if values else None
    value = _safe_float(row.get(key), float("nan"))
    return float(value) if np.isfinite(value) else None


def _learned_proxy_score(row: dict[str, object], model: dict[str, object] | None) -> float:
    if not model:
        return 0.0
    columns = model.get("columns", [])
    coefs = model.get("coef", [])
    means = model.get("mean", [])
    scales = model.get("scale", [])
    fills = model.get("fill_values", {})
    if not isinstance(columns, list) or not isinstance(coefs, list):
        return 0.0
    score = _safe_float(model.get("intercept"), 0.0)
    for idx, key in enumerate(columns):
        if idx >= len(coefs):
            break
        if not isinstance(key, str):
            continue
        value = _learned_proxy_feature(row, key)
        if value is None:
            value = _safe_float(fills.get(key) if isinstance(fills, dict) else None, 0.0)
        mean = _safe_float(means[idx] if idx < len(means) else 0.0, 0.0)
        scale = _safe_float(scales[idx] if idx < len(scales) else 1.0, 1.0)
        if abs(scale) < 1.0e-9:
            scale = 1.0
        score += _safe_float(coefs[idx], 0.0) * ((float(value) - mean) / scale)
    return float(score)


def _source_acceptor_rank_score(row: dict[str, object]) -> float:
    vals = []
    direct = _safe_float(row.get("source_acceptor_score"), float("nan"))
    if np.isfinite(direct):
        vals.append(float(direct))
    preview = row.get("accepted_preview", [])
    if isinstance(preview, list):
        for item in preview:
            if not isinstance(item, dict):
                continue
            val = _safe_float(item.get("source_acceptor_score"), float("nan"))
            if np.isfinite(val):
                vals.append(float(val))
    return float(np.mean(vals)) if vals else 0.0


def _committee_proxy_score(
    row: dict[str, object],
    *,
    source_acceptor_weight: float,
    source_acceptor_floor: float,
) -> float:
    source_score = _source_acceptor_rank_score(row)
    source_bonus = max(float(source_score) - float(source_acceptor_floor), 0.0)
    # Keep this as a ranking nudge, not a replacement for the delivery-aware
    # full proxy.  The full-score labels are scarce; the source acceptor should
    # mainly break ties among similarly safe delivery candidates.
    return float(row.get("learned_full_proxy", 0.0)) + float(source_acceptor_weight) * source_bonus


def _mass_bridge_proxy_score(row: dict[str, object]) -> float:
    moved = _safe_float(row.get("moved_tracklets"), 0.0)
    accepted = _safe_float(row.get("accepted_reassignments"), 0.0)
    target_mean = _preview_mean(row, "target_mean_sim")
    target_best = _preview_mean(row, "target_best_sim")
    target_vote = _preview_mean(row, "target_view_vote")
    target_margin = min(_preview_mean(row, "target_margin"), 1.0)
    source_score = _source_acceptor_rank_score(row)
    target_size = _preview_mean(row, "target_size")
    moved_bonus = min(np.log1p(max(moved, 0.0)) / np.log(64.0), 1.0)
    target_bonus = min(np.log1p(max(target_size, 0.0)) / np.log(256.0), 1.0)
    return float(
        0.30 * moved_bonus
        + 0.18 * target_bonus
        + 0.16 * target_mean
        + 0.12 * target_best
        + 0.10 * target_vote
        + 0.08 * target_margin
        + 0.08 * source_score
        + 0.05 * float(row.get("full_side_effect_proxy", 0.0))
        - 0.03 * max(accepted - 4.0, 0.0)
    )


def _sort_rows(
    rows: list[dict[str, object]],
    rank_by: str,
    learned_proxy_model: dict[str, object] | None = None,
    *,
    source_acceptor_rank_weight: float = 0.0,
    source_acceptor_rank_floor: float = 0.5,
) -> None:
    for row in rows:
        row["full_side_effect_proxy"] = _full_side_effect_proxy(row)
        row["learned_full_proxy"] = _learned_proxy_score(row, learned_proxy_model)
        row["source_acceptor_rank_score"] = _source_acceptor_rank_score(row)
        row["committee_full_proxy"] = _committee_proxy_score(
            row,
            source_acceptor_weight=float(source_acceptor_rank_weight),
            source_acceptor_floor=float(source_acceptor_rank_floor),
        )
        row["mass_bridge_proxy"] = _mass_bridge_proxy_score(row)
    if rank_by == "full_proxy":
        rows.sort(
            key=lambda row: (
                float(row.get("full_side_effect_proxy", 0.0)),
                float(row["tracklet_pair_precision"]),
                float(row["tracklet_pair_f1"]),
                float(row["tracklet_pair_recall"]),
            ),
            reverse=True,
        )
    elif rank_by == "learned_proxy":
        rows.sort(
            key=lambda row: (
                float(row.get("learned_full_proxy", 0.0)),
                float(row.get("full_side_effect_proxy", 0.0)),
                float(row["tracklet_pair_f1"]),
            ),
            reverse=True,
        )
    elif rank_by == "committee_proxy":
        rows.sort(
            key=lambda row: (
                float(row.get("committee_full_proxy", 0.0)),
                float(row.get("learned_full_proxy", 0.0)),
                float(row.get("source_acceptor_rank_score", 0.0)),
                float(row.get("full_side_effect_proxy", 0.0)),
                float(row["tracklet_pair_f1"]),
            ),
            reverse=True,
        )
    elif rank_by == "mass_bridge_proxy":
        rows.sort(
            key=lambda row: (
                float(row.get("mass_bridge_proxy", 0.0)),
                float(row.get("committee_full_proxy", 0.0)),
                float(row.get("source_acceptor_rank_score", 0.0)),
                float(row["tracklet_pair_precision"]),
                float(row["tracklet_pair_f1"]),
            ),
            reverse=True,
        )
    elif rank_by == "precision":
        rows.sort(
            key=lambda row: (
                float(row["tracklet_pair_precision"]),
                float(row["tracklet_pair_f1"]),
                float(row["tracklet_pair_recall"]),
            ),
            reverse=True,
        )
    elif rank_by == "recall":
        rows.sort(
            key=lambda row: (
                float(row["tracklet_pair_recall"]),
                float(row["tracklet_pair_f1"]),
                float(row["tracklet_pair_precision"]),
            ),
            reverse=True,
        )
    else:
        rows.sort(
            key=lambda row: (
                float(row["tracklet_pair_f1"]),
                float(row["tracklet_pair_precision"]),
                float(row["tracklet_pair_recall"]),
            ),
            reverse=True,
        )


def _self_test() -> None:
    labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    accepted = [{"target_component": 1, "source_indices": [0, 2]}]
    out = _apply_reassignments(labels, accepted)
    assert out.tolist() == [1, 0, 1, 1, 1], out.tolist()
    sig = _accepted_signature([{"source_seqs": [3, 1], "target_component": 2}])
    assert sig == (((3, 1), 2),), sig
    row = {"accepted_preview": [{"source_seqs": [9, 8], "target_component": 5}]}
    assert _row_first_edge_key(row) == ((9, 8), 5), _row_first_edge_key(row)
    edge_rows = [
        {"source_seqs": [1], "target_component": 2},
        {"source_seqs": [1], "target_component": 2},
        {"source_seqs": [3], "target_component": 4},
    ]
    skipped = _skip_first_edge_families(edge_rows, 1)
    assert [_source_target_key(item) for item in skipped] == [((3,), 4)], skipped
    assert _first_edge_family_rank(edge_rows, edge_rows[-1]) == 1
    model = {
        "columns": ["tracklet_pair_f1", "preview_mean_target_mean_sim"],
        "fill_values": {"tracklet_pair_f1": 0.5, "preview_mean_target_mean_sim": 0.7},
        "mean": [0.5, 0.7],
        "scale": [0.1, 0.1],
        "coef": [0.1, 0.2],
        "intercept": 0.6,
    }
    scored_row = {
        "tracklet_pair_f1": 0.6,
        "accepted_preview": [{"target_mean_sim": 0.8}],
        "tracklet_pair_precision": 0.7,
        "tracklet_pair_recall": 0.6,
    }
    assert abs(_learned_proxy_score(scored_row, model) - 0.9) < 1.0e-9
    source_model = {
        "columns": ["source_cross_mean_sim"],
        "fill_values": {"source_cross_mean_sim": 0.5},
        "mean": [0.5],
        "scale": [0.1],
        "coef": [-1.0],
        "intercept": 0.0,
    }
    low_cross = _source_acceptor_score({"source_cross_mean_sim": 0.3}, source_model)
    high_cross = _source_acceptor_score({"source_cross_mean_sim": 0.9}, source_model)
    assert low_cross > high_cross, (low_cross, high_cross)
    rows = [
        {
            "tracklet_pair_f1": 0.7,
            "tracklet_pair_precision": 0.7,
            "tracklet_pair_recall": 0.7,
            "accepted_preview": [{"source_acceptor_score": 0.2}],
        },
        {
            "tracklet_pair_f1": 0.7,
            "tracklet_pair_precision": 0.7,
            "tracklet_pair_recall": 0.7,
            "accepted_preview": [{"source_acceptor_score": 0.9}],
        },
    ]
    _sort_rows(rows, "committee_proxy", None, source_acceptor_rank_weight=0.1, source_acceptor_rank_floor=0.5)
    assert rows[0]["source_acceptor_rank_score"] > rows[1]["source_acceptor_rank_score"], rows
    rows = [
        {
            "tracklet_pair_f1": 0.7,
            "tracklet_pair_precision": 0.7,
            "tracklet_pair_recall": 0.7,
            "moved_tracklets": 2,
            "accepted_reassignments": 1,
            "accepted_preview": [{"target_size": 8, "target_mean_sim": 0.8, "target_best_sim": 0.9, "target_view_vote": 1.0, "target_margin": 0.1}],
        },
        {
            "tracklet_pair_f1": 0.7,
            "tracklet_pair_precision": 0.7,
            "tracklet_pair_recall": 0.7,
            "moved_tracklets": 32,
            "accepted_reassignments": 1,
            "accepted_preview": [{"target_size": 128, "target_mean_sim": 0.8, "target_best_sim": 0.9, "target_view_vote": 1.0, "target_margin": 0.1}],
        },
    ]
    _sort_rows(rows, "mass_bridge_proxy", None)
    assert rows[0]["moved_tracklets"] > rows[1]["moved_tracklets"], rows
    edges = [
        {"proposal_score": 1.0, "target_score": 1.0, "target_mean_sim": 0.8, "target_best_sim": 0.9, "target_view_vote": 1.0, "target_margin": 0.1, "source_size": 1, "target_size": 8},
        {"proposal_score": 0.9, "target_score": 0.9, "target_mean_sim": 0.8, "target_best_sim": 0.9, "target_view_vote": 1.0, "target_margin": 0.1, "source_size": 32, "target_size": 128},
    ]
    _sort_candidate_edges(edges, "mass_bridge")
    assert edges[0]["source_size"] > edges[1]["source_size"], edges
    restored = _restore_preview_indices(
        {"accepted_preview": [{"source_seqs": [10, 12], "target_component": 3}]},
        {10: 0, 12: 2},
    )
    assert restored[0]["source_indices"] == [0, 2], restored
    picked = _greedy_from_candidate_edges(
        [
            {"source_indices": [0], "target_component": 7},
            {"source_indices": [0], "target_component": 8},
            {"source_indices": [2], "target_component": 7},
        ],
        prefix=3,
        max_sources_per_target=1,
        max_reassignments=3,
    )
    assert [(row["source_indices"], row["target_component"]) for row in picked] == [([0], 7)], picked
    print(json.dumps({"stage": "self_test", "status": "ok"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", default="")
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--primary-feature-npz", default="")
    ap.add_argument("--view", action="append", default=[], help="feature view name:path[:weight]")
    ap.add_argument("--committed-min-sizes", default="8,16")
    ap.add_argument("--pending-max-sizes", default="0")
    ap.add_argument("--conflict-rate-thresholds", default="0.003,0.01")
    ap.add_argument("--source-min-component-sizes", default="64,128")
    ap.add_argument("--source-max-component-sizes", default="1000000")
    ap.add_argument("--source-seed-sims", default="0.74,0.78,0.82")
    ap.add_argument("--source-expand-sims", default="0.70,0.74,0.78")
    ap.add_argument("--source-top-ks", default="4,8")
    ap.add_argument("--source-min-group-sizes", default="2,3,4")
    ap.add_argument("--source-max-group-sizes", default="6,12")
    ap.add_argument("--source-min-conflicts-to-rest", default="1,2")
    ap.add_argument("--source-min-margins", default="0.03,0.06,0.10")
    ap.add_argument("--source-max-groups-per-component", default="1,2")
    ap.add_argument("--source-max-total-groups", default="8,16,32")
    ap.add_argument("--target-states", default="committed,provisional")
    ap.add_argument("--min-target-sizes", default="8,16,32,64")
    ap.add_argument("--target-top-ks", default="3,5,8")
    ap.add_argument("--min-target-best-sims", default="0.78,0.82,0.86")
    ap.add_argument("--min-target-mean-sims", default="0.72,0.76,0.80")
    ap.add_argument("--min-target-view-votes", default="0.5,0.75")
    ap.add_argument("--min-target-qualities", default="0.0")
    ap.add_argument("--target-view-sim-thresholds", default="0.70,0.74")
    ap.add_argument("--min-target-margins", default="0.00,0.03,0.06")
    ap.add_argument("--max-forbidden-pairs", default="0")
    ap.add_argument("--max-sources-per-target", default="1,2")
    ap.add_argument("--max-reassignments", default="2,4,8,16")
    ap.add_argument("--candidate-search-top-n", type=int, default=0)
    ap.add_argument("--candidate-search-prefixes", default="16,32,64,128")
    ap.add_argument("--candidate-targets-per-source", type=int, default=1)
    ap.add_argument("--candidate-edge-rank-by", default="proposal", choices=["proposal", "mass_bridge"])
    ap.add_argument(
        "--candidate-skip-first-edge-families",
        default="0",
        help="comma-separated counts of top unique source->target edge families to skip before greedy candidate search",
    )
    ap.add_argument("--rank-by", default="pair", choices=["pair", "precision", "recall", "full_proxy", "learned_proxy", "committee_proxy", "mass_bridge_proxy"])
    ap.add_argument("--learned-proxy-json", default="")
    ap.add_argument("--source-acceptor-json", default="", help="optional no-GT source-island acceptor model used to rank candidate-search edges")
    ap.add_argument("--source-acceptor-rank-weight", type=float, default=0.0)
    ap.add_argument("--source-acceptor-rank-floor", type=float, default=0.5)
    ap.add_argument("--full-selection", default="unique_signature", choices=["none", "unique_signature", "diverse_first_edge"])
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=90_000_000)
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.assignment_csv or not args.primary_feature_npz or not args.json:
        raise SystemExit("--assignment-csv, --primary-feature-npz, and --json are required")

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    primary = _load_feature_npz(args.primary_feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0)
    views: list[dict[str, object]] = [
        {"name": "primary", "path": str(args.primary_feature_npz), "weight": 1.0, "emb": _l2n(primary.astype(np.float32))}
    ]
    view_meta = [{"name": "primary", "path": str(args.primary_feature_npz), "weight": 1.0}]
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        views.append({"name": name, "path": path, "weight": float(weight), "emb": _load_npz_aligned(path, records)})
        view_meta.append({"name": name, "path": path, "weight": float(weight)})

    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": 0.5,
        "min_matches": 1,
        "min_purity": 0.0,
        "n_tracklets": len(records),
        "prediction_rows": int(sum(len(value) for value in pred_by_video.values())),
        "gt_rows": int(sum(len(value) for value in gt_by_video.values())),
    }
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached
    keep_seqs, output_info = _output_keep_seqs(records, _admission_args(args))
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    base_labels, keep_indices, raw_to_local = _labels_from_assignment(records, pred_input)
    keep_indices = {idx for idx in keep_indices if int(records[idx].seq) in keep_seqs}
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    seq_to_idx = {int(record.seq): int(idx) for idx, record in enumerate(records)}
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    forbidden = _build_overlap_forbidden(records)
    stats = _component_stats(records, base_labels, keep_indices)
    rows: list[dict[str, object]] = []
    labels_by_rank: dict[int, np.ndarray] = {}
    target_states = {part.strip() for part in str(args.target_states).split(",") if part.strip()}
    source_cache: dict[tuple[object, ...], tuple[list[dict[str, object]], dict[str, object], dict[tuple[int, ...], list[int]]]] = {}
    query_score_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = {}
    pair_metric_cache: dict[tuple[tuple[tuple[int, ...], int], ...], dict[str, object]] = {}
    source_acceptor_model = None
    if args.source_acceptor_json:
        source_acceptor_model = json.loads(Path(args.source_acceptor_json).read_text())

    state_grid = product(
        _parse_ints(args.committed_min_sizes),
        _parse_ints(args.pending_max_sizes),
        _parse_floats(args.conflict_rate_thresholds),
    )
    source_grid = list(
        product(
            _parse_ints(args.source_min_component_sizes),
            _parse_ints(args.source_max_component_sizes),
            _parse_floats(args.source_seed_sims),
            _parse_floats(args.source_expand_sims),
            _parse_ints(args.source_top_ks),
            _parse_ints(args.source_min_group_sizes),
            _parse_ints(args.source_max_group_sizes),
            _parse_ints(args.source_min_conflicts_to_rest),
            _parse_floats(args.source_min_margins),
            _parse_ints(args.source_max_groups_per_component),
            _parse_ints(args.source_max_total_groups),
        )
    )
    target_grid = list(
        product(
            _parse_ints(args.min_target_sizes),
            _parse_ints(args.target_top_ks),
            _parse_floats(args.min_target_best_sims),
            _parse_floats(args.min_target_mean_sims),
            _parse_floats(args.min_target_view_votes),
            _parse_floats(args.min_target_qualities),
            _parse_floats(args.target_view_sim_thresholds),
            _parse_floats(args.min_target_margins),
            _parse_ints(args.max_forbidden_pairs),
            _parse_ints(args.max_sources_per_target),
            _parse_ints(args.max_reassignments),
        )
    )

    if int(args.candidate_search_top_n) > 0:
        search_rows: list[dict[str, object]] = []
        candidate_edges: list[dict[str, object]] = []
        state_grid_values = list(state_grid)
        min_target_size = min(_parse_ints(args.min_target_sizes))
        target_top_k = _parse_ints(args.target_top_ks)[0]
        min_target_best_sim = min(_parse_floats(args.min_target_best_sims))
        min_target_mean_sim = min(_parse_floats(args.min_target_mean_sims))
        min_target_view_vote = min(_parse_floats(args.min_target_view_votes))
        min_target_quality = min(_parse_floats(args.min_target_qualities))
        target_view_sim_threshold = min(_parse_floats(args.target_view_sim_thresholds))
        min_target_margin = min(_parse_floats(args.min_target_margins))
        max_forbidden_pairs = min(_parse_ints(args.max_forbidden_pairs))
        max_sources_per_target_values = _parse_ints(args.max_sources_per_target)
        max_reassignment_values = _parse_ints(args.max_reassignments)
        prefixes = _parse_ints(args.candidate_search_prefixes)
        skip_first_edge_values = _parse_ints(args.candidate_skip_first_edge_families)

        for committed_min_size, pending_max_size, conflict_rate_threshold in state_grid_values:
            states = _assign_states(
                stats,
                committed_min_size=int(committed_min_size),
                pending_max_size=int(pending_max_size),
                conflict_rate_threshold=float(conflict_rate_threshold),
                min_quality_quantile=0.0,
            )
            state_counts = Counter(states.values())
            for source_params in source_grid:
                (
                    min_component_size,
                    max_component_size,
                    seed_sim,
                    expand_sim,
                    top_k,
                    min_group_size,
                    max_group_size,
                    min_conflicts,
                    min_margin,
                    max_groups_per_component,
                    max_total_groups,
                ) = source_params
                if float(expand_sim) > float(seed_sim) or int(max_group_size) < int(min_group_size):
                    continue
                source_key = (
                    int(min_component_size),
                    int(max_component_size),
                    float(seed_sim),
                    float(expand_sim),
                    int(top_k),
                    int(min_group_size),
                    int(max_group_size),
                    int(min_conflicts),
                    float(min_margin),
                    int(max_groups_per_component),
                    int(max_total_groups),
                )
                if source_key not in source_cache:
                    sources, source_info = _source_candidates(
                        records,
                        base_labels,
                        keep_indices,
                        views,
                        forbidden,
                        min_component_size=int(min_component_size),
                        max_component_size=int(max_component_size),
                        seed_sim=float(seed_sim),
                        expand_sim=float(expand_sim),
                        top_k=int(top_k),
                        min_group_size=int(min_group_size),
                        max_group_size=int(max_group_size),
                        min_conflicts_to_rest=int(min_conflicts),
                        min_margin=float(min_margin),
                        max_groups_per_component=int(max_groups_per_component),
                        max_total_groups=int(max_total_groups),
                    )
                    source_map = {
                        tuple(int(seq) for seq in row["source_seqs"]): [int(idx) for idx in row["source_indices"]]
                        for row in sources
                    }
                    source_cache[source_key] = (sources, source_info, source_map)
                sources, source_info, _source_map = source_cache[source_key]
                if not sources:
                    continue
                edges = _build_candidate_edges(
                    records,
                    base_labels,
                    keep_indices,
                    sources,
                    views,
                    forbidden,
                    states,
                    target_states=target_states,
                    min_target_size=int(min_target_size),
                    target_top_k=int(target_top_k),
                    min_target_best_sim=float(min_target_best_sim),
                    min_target_mean_sim=float(min_target_mean_sim),
                    min_target_view_vote=float(min_target_view_vote),
                    min_target_quality=float(min_target_quality),
                    target_view_sim_threshold=float(target_view_sim_threshold),
                    min_target_margin=float(min_target_margin),
                    max_forbidden_pairs=int(max_forbidden_pairs),
                    targets_per_source=int(args.candidate_targets_per_source),
                    query_score_cache=query_score_cache,
                    source_acceptor_model=source_acceptor_model,
                    source_context={k: v for k, v in source_info.items() if k != "source_preview"},
                )
                for edge in edges:
                    candidate_edges.append(
                        {
                            **edge,
                            "committed_min_size": int(committed_min_size),
                            "pending_max_size": int(pending_max_size),
                            "conflict_rate_threshold": float(conflict_rate_threshold),
                            "state_counts": dict(state_counts),
                            **{k: v for k, v in source_info.items() if k != "source_preview"},
                            "target_states": ",".join(sorted(target_states)),
                            "min_target_size": int(min_target_size),
                            "target_top_k": int(target_top_k),
                            "min_target_best_sim": float(min_target_best_sim),
                            "min_target_mean_sim": float(min_target_mean_sim),
                            "min_target_view_vote": float(min_target_view_vote),
                            "min_target_quality": float(min_target_quality),
                            "target_view_sim_threshold": float(target_view_sim_threshold),
                            "min_target_margin": float(min_target_margin),
                            "max_forbidden_pairs": int(max_forbidden_pairs),
                        }
                    )
        _sort_candidate_edges(candidate_edges, str(args.candidate_edge_rank_by))
        dedup_edges = []
        seen_edge: set[tuple[tuple[int, ...], int]] = set()
        for row in candidate_edges:
            sig = (tuple(int(seq) for seq in row.get("source_seqs", [])), int(row["target_component"]))
            if sig in seen_edge:
                continue
            seen_edge.add(sig)
            dedup_edges.append(row)
            if len(dedup_edges) >= int(args.candidate_search_top_n):
                break
        candidate_edges = dedup_edges
        for skip_first_edge_families in skip_first_edge_values:
            scoped_candidate_edges = _skip_first_edge_families(candidate_edges, int(skip_first_edge_families))
            for prefix in prefixes:
                for max_sources_per_target in max_sources_per_target_values:
                    for max_reassignments in max_reassignment_values:
                        accepted = _greedy_from_candidate_edges(
                            scoped_candidate_edges,
                            prefix=int(prefix),
                            max_sources_per_target=int(max_sources_per_target),
                            max_reassignments=int(max_reassignments),
                        )
                        first_family_rank = _first_edge_family_rank(candidate_edges, accepted[0]) if accepted else -1
                        signature = _accepted_signature(accepted)
                        cached_pair = pair_metric_cache.get(signature)
                        if cached_pair is None:
                            labels = _apply_reassignments(base_labels, accepted)
                            pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                            cached_pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                            pair_metric_cache[signature] = cached_pair
                        pair = cached_pair
                        preview = [{k: v for k, v in row.items() if k != "source_indices"} for row in accepted[:20]]
                        search_rows.append(
                            {
                                "mode": "conflict_subcluster_reassign_candidate_search",
                                "candidate_search_top_n": int(args.candidate_search_top_n),
                                "candidate_search_prefix": int(prefix),
                                "candidate_targets_per_source": int(args.candidate_targets_per_source),
                                "candidate_edge_rank_by": str(args.candidate_edge_rank_by),
                                "candidate_edges": int(len(scoped_candidate_edges)),
                                "candidate_total_edges_before_skip": int(len(candidate_edges)),
                                "candidate_skip_first_edge_families": int(skip_first_edge_families),
                                "candidate_first_edge_family_rank": int(first_family_rank),
                                "max_sources_per_target": int(max_sources_per_target),
                                "max_reassignments": int(max_reassignments),
                                "accepted_reassignments": int(len(accepted)),
                                "moved_tracklets": int(sum(int(row.get("source_size", 0)) for row in accepted)),
                                "target_components_used": int(len({int(row["target_component"]) for row in accepted})),
                                "accepted_preview": preview,
                                **(scoped_candidate_edges[0] if scoped_candidate_edges else {}),
                                **pair,
                                "uses_anchors": False,
                                "uses_gt_for_training_or_anchors": False,
                                "uses_gt_for_evaluation_only": True,
                            }
                        )
        rows = search_rows
    else:
        state_grid = product(
            _parse_ints(args.committed_min_sizes),
            _parse_ints(args.pending_max_sizes),
            _parse_floats(args.conflict_rate_thresholds),
        )

        for committed_min_size, pending_max_size, conflict_rate_threshold in state_grid:
            states = _assign_states(
                stats,
                committed_min_size=int(committed_min_size),
                pending_max_size=int(pending_max_size),
                conflict_rate_threshold=float(conflict_rate_threshold),
                min_quality_quantile=0.0,
            )
            state_counts = Counter(states.values())
            for source_params in source_grid:
                (
                    min_component_size,
                    max_component_size,
                    seed_sim,
                    expand_sim,
                    top_k,
                    min_group_size,
                    max_group_size,
                    min_conflicts,
                    min_margin,
                    max_groups_per_component,
                    max_total_groups,
                ) = source_params
                if float(expand_sim) > float(seed_sim) or int(max_group_size) < int(min_group_size):
                    continue
                source_key = (
                    int(min_component_size),
                    int(max_component_size),
                    float(seed_sim),
                    float(expand_sim),
                    int(top_k),
                    int(min_group_size),
                    int(max_group_size),
                    int(min_conflicts),
                    float(min_margin),
                    int(max_groups_per_component),
                    int(max_total_groups),
                )
                if source_key not in source_cache:
                    sources, source_info = _source_candidates(
                        records,
                        base_labels,
                        keep_indices,
                        views,
                        forbidden,
                        min_component_size=int(min_component_size),
                        max_component_size=int(max_component_size),
                        seed_sim=float(seed_sim),
                        expand_sim=float(expand_sim),
                        top_k=int(top_k),
                        min_group_size=int(min_group_size),
                        max_group_size=int(max_group_size),
                        min_conflicts_to_rest=int(min_conflicts),
                        min_margin=float(min_margin),
                        max_groups_per_component=int(max_groups_per_component),
                        max_total_groups=int(max_total_groups),
                    )
                    source_map = {
                        tuple(int(seq) for seq in row["source_seqs"]): [int(idx) for idx in row["source_indices"]]
                        for row in sources
                    }
                    source_cache[source_key] = (sources, source_info, source_map)
                sources, source_info, source_map = source_cache[source_key]
                if not sources:
                    continue
                for target_params in target_grid:
                    (
                        min_target_size,
                        target_top_k,
                        min_target_best_sim,
                        min_target_mean_sim,
                        min_target_view_vote,
                        min_target_quality,
                        target_view_sim_threshold,
                        min_target_margin,
                        max_forbidden_pairs,
                        max_sources_per_target,
                        max_reassignments,
                    ) = target_params
                    accepted, info = _choose_targets(
                        records,
                        base_labels,
                        keep_indices,
                        sources,
                        views,
                        forbidden,
                        states,
                        target_states=target_states,
                        min_target_size=int(min_target_size),
                        target_top_k=int(target_top_k),
                        min_target_best_sim=float(min_target_best_sim),
                        min_target_mean_sim=float(min_target_mean_sim),
                        min_target_view_vote=float(min_target_view_vote),
                        min_target_quality=float(min_target_quality),
                        target_view_sim_threshold=float(target_view_sim_threshold),
                        min_target_margin=float(min_target_margin),
                        max_forbidden_pairs=int(max_forbidden_pairs),
                        max_sources_per_target=int(max_sources_per_target),
                        max_reassignments=int(max_reassignments),
                        query_score_cache=query_score_cache,
                    )
                    accepted = _restore_source_indices(accepted, source_map)
                    signature = _accepted_signature(accepted)
                    cached_pair = pair_metric_cache.get(signature)
                    if cached_pair is None:
                        labels = _apply_reassignments(base_labels, accepted)
                        pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                        cached_pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                        pair_metric_cache[signature] = cached_pair
                    pair = cached_pair
                    rows.append(
                        {
                            "mode": "conflict_subcluster_reassign",
                            "committed_min_size": int(committed_min_size),
                            "pending_max_size": int(pending_max_size),
                            "conflict_rate_threshold": float(conflict_rate_threshold),
                            "state_counts": dict(state_counts),
                            **{k: v for k, v in source_info.items() if k != "source_preview"},
                            "target_states": ",".join(sorted(target_states)),
                            "min_target_size": int(min_target_size),
                            "target_top_k": int(target_top_k),
                            "min_target_best_sim": float(min_target_best_sim),
                            "min_target_mean_sim": float(min_target_mean_sim),
                            "min_target_view_vote": float(min_target_view_vote),
                            "min_target_quality": float(min_target_quality),
                            "target_view_sim_threshold": float(target_view_sim_threshold),
                            "min_target_margin": float(min_target_margin),
                            "max_forbidden_pairs": int(max_forbidden_pairs),
                            "max_sources_per_target": int(max_sources_per_target),
                            "max_reassignments": int(max_reassignments),
                            **{k: v for k, v in info.items() if k != "accepted_preview"},
                            "accepted_preview": info["accepted_preview"],
                            **pair,
                            "uses_anchors": False,
                            "uses_gt_for_training_or_anchors": False,
                            "uses_gt_for_evaluation_only": True,
                        }
                    )

    learned_proxy_model = None
    if args.learned_proxy_json:
        learned_proxy_model = json.loads(Path(args.learned_proxy_json).read_text())
    _sort_rows(
        rows,
        str(args.rank_by),
        learned_proxy_model,
        source_acceptor_rank_weight=float(args.source_acceptor_rank_weight),
        source_acceptor_rank_floor=float(args.source_acceptor_rank_floor),
    )
    full_candidates: list[dict[str, object]] = []
    seen_full_keys: set[tuple[object, ...]] = set()
    for row in rows:
        if str(args.full_selection) == "unique_signature":
            key = ("signature", _row_signature(row))
        elif str(args.full_selection) == "diverse_first_edge":
            key = ("first_edge", _row_first_edge_key(row))
        else:
            key = ("row", len(full_candidates), id(row))
        if str(args.full_selection) != "none":
            if key in seen_full_keys:
                continue
            seen_full_keys.add(key)
        full_candidates.append(row)
        if len(full_candidates) >= max(int(args.full_top_n), 0):
            break

    for rank, row in enumerate(full_candidates, start=1):
        if str(row.get("mode")) == "conflict_subcluster_reassign_candidate_search":
            accepted = _restore_preview_indices(row, seq_to_idx)
        else:
            source_key = (
                int(row["source_min_component_size"]),
                int(row["source_max_component_size"]),
                float(row["source_seed_sim"]),
                float(row["source_expand_sim"]),
                int(row["source_top_k"]),
                int(row["source_min_group_size"]),
                int(row["source_max_group_size"]),
                int(row["source_min_conflicts_to_rest"]),
                float(row["source_min_margin"]),
                int(row["source_max_groups_per_component"]),
                int(row["source_max_total_groups"]),
            )
            sources, _source_info, source_map = source_cache[source_key]
            states = _assign_states(
                stats,
                committed_min_size=int(row["committed_min_size"]),
                pending_max_size=int(row["pending_max_size"]),
                conflict_rate_threshold=float(row["conflict_rate_threshold"]),
                min_quality_quantile=0.0,
            )
            accepted, _info = _choose_targets(
                records,
                base_labels,
                keep_indices,
                sources,
                views,
                forbidden,
                states,
                target_states=target_states,
                min_target_size=int(row["min_target_size"]),
                target_top_k=int(row["target_top_k"]),
                min_target_best_sim=float(row["min_target_best_sim"]),
                min_target_mean_sim=float(row["min_target_mean_sim"]),
                min_target_view_vote=float(row["min_target_view_vote"]),
                min_target_quality=float(row.get("min_target_quality", 0.0)),
                target_view_sim_threshold=float(row["target_view_sim_threshold"]),
                min_target_margin=float(row["min_target_margin"]),
                max_forbidden_pairs=int(row["max_forbidden_pairs"]),
                max_sources_per_target=int(row["max_sources_per_target"]),
                max_reassignments=int(row["max_reassignments"]),
                query_score_cache=query_score_cache,
            )
            accepted = _restore_source_indices(accepted, source_map)
        labels = _apply_reassignments(base_labels, accepted)
        labels_by_rank[rank] = labels
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs))
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            if str(row.get("mode")) == "conflict_subcluster_reassign_candidate_search":
                accepted = _restore_preview_indices(row, seq_to_idx)
            else:
                source_key = (
                    int(row["source_min_component_size"]),
                    int(row["source_max_component_size"]),
                    float(row["source_seed_sim"]),
                    float(row["source_expand_sim"]),
                    int(row["source_top_k"]),
                    int(row["source_min_group_size"]),
                    int(row["source_max_group_size"]),
                    int(row["source_min_conflicts_to_rest"]),
                    float(row["source_min_margin"]),
                    int(row["source_max_groups_per_component"]),
                    int(row["source_max_total_groups"]),
                )
                sources, _source_info, source_map = source_cache[source_key]
                states = _assign_states(stats, committed_min_size=int(row["committed_min_size"]), pending_max_size=int(row["pending_max_size"]), conflict_rate_threshold=float(row["conflict_rate_threshold"]), min_quality_quantile=0.0)
                accepted, _info = _choose_targets(
                    records, base_labels, keep_indices, sources, views, forbidden, states,
                    target_states=target_states, min_target_size=int(row["min_target_size"]), target_top_k=int(row["target_top_k"]),
                    min_target_best_sim=float(row["min_target_best_sim"]), min_target_mean_sim=float(row["min_target_mean_sim"]),
                    min_target_view_vote=float(row["min_target_view_vote"]), min_target_quality=float(row.get("min_target_quality", 0.0)),
                    target_view_sim_threshold=float(row["target_view_sim_threshold"]),
                    min_target_margin=float(row["min_target_margin"]), max_forbidden_pairs=int(row["max_forbidden_pairs"]),
                    max_sources_per_target=int(row["max_sources_per_target"]), max_reassignments=int(row["max_reassignments"]),
                    query_score_cache=query_score_cache)
                accepted = _restore_source_indices(accepted, source_map)
            labels = _apply_reassignments(base_labels, accepted)
        assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs=keep_seqs, offset=int(args.assignment_offset))
        rows[0].update(assignment_info)

    result = {
        "assignment_csv": str(args.assignment_csv),
        "primary_feature_npz": str(args.primary_feature_npz),
        "views": view_meta,
        "base_pair_metrics": base_pair,
        "base_assignment_components": int(len(raw_to_local)),
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "assignment_info": assignment_info,
        "query_score_cache_entries": int(len(query_score_cache)),
        "pair_metric_cache_entries": int(len(pair_metric_cache)),
        "full_selection": str(args.full_selection),
        "rank_by": str(args.rank_by),
        "candidate_edge_rank_by": str(args.candidate_edge_rank_by),
        "source_acceptor_rank_weight": float(args.source_acceptor_rank_weight),
        "source_acceptor_rank_floor": float(args.source_acceptor_rank_floor),
        "learned_proxy_json": str(args.learned_proxy_json),
        "learned_proxy_model": {
            key: learned_proxy_model.get(key)
            for key in ("model_type", "row_count", "alpha", "feature_mode", "min_full_idf1")
        } if isinstance(learned_proxy_model, dict) else None,
        "top": rows[: max(100, int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "base": base_pair, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
