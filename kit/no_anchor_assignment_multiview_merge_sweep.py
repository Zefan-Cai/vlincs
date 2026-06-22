#!/usr/bin/env python
"""Merge current no-anchor components using direct multi-view edge consensus.

This is a no-anchor post-processor: it starts from an existing assignment CSV,
builds component-level candidate edges, scores them from feature-view agreement,
and evaluates merge thresholds.  Ground truth is used only after prediction for
diagnostic metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, KIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _component_members, _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _candidate_edges, _parse_floats, _parse_ints, _write_csv
    from kit.no_anchor_global_id_model import _UnionFind
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
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _component_members, _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _candidate_edges, _parse_floats, _parse_ints, _write_csv
    from no_anchor_global_id_model import _UnionFind
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
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_view(text: str) -> tuple[str, str, float]:
    parts = str(text).split(":")
    if len(parts) == 2:
        name, path = parts
        weight = 1.0
    elif len(parts) == 3:
        name, path, weight_text = parts
        weight = float(weight_text)
    else:
        raise ValueError(f"bad --view {text!r}; expected name:path[:weight]")
    if not name:
        raise ValueError(f"empty view name in {text!r}")
    return name, path, float(weight)


def _row_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _load_npz_aligned(path: str, records, *, weight: float = 1.0) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    seqs = [int(seq) for seq in data["seqs"].tolist()]
    features = data["features"].astype(np.float32)
    by_seq = {seq: idx for idx, seq in enumerate(seqs)}
    missing = [int(record.seq) for record in records if int(record.seq) not in by_seq]
    if missing:
        raise ValueError(f"{path} missing seq={missing[0]} ({len(missing)} total)")
    order = np.asarray([by_seq[int(record.seq)] for record in records], dtype=np.int64)
    return _l2n(features[order].astype(np.float32)) * float(weight)


def _component_centroids(x: np.ndarray, members: list[list[int]]) -> np.ndarray:
    x = _l2n(x.astype(np.float32))
    cents = []
    for indices in members:
        v = x[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        cents.append(v / (np.linalg.norm(v) + 1.0e-9))
    return np.stack(cents).astype(np.float32)


def _rank_matrix(sim: np.ndarray) -> np.ndarray:
    order = np.argsort(-sim, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(order.shape[1], dtype=np.int32)[None, :] + 1
    return ranks


def _view_tables(view_embeddings: dict[str, np.ndarray], members: list[list[int]]):
    sims: dict[str, np.ndarray] = {}
    ranks: dict[str, np.ndarray] = {}
    for name, emb in view_embeddings.items():
        cent = _component_centroids(emb, members)
        sim = (cent @ cent.T).astype(np.float32)
        np.fill_diagonal(sim, -2.0)
        sims[name] = sim
        ranks[name] = _rank_matrix(sim)
    return sims, ranks


def _centroid_candidate_edges(records, emb: np.ndarray, reps: list[int], members: list[list[int]], candidate_top_k: int):
    cent = _component_centroids(emb, members)
    sim = (cent @ cent.T).astype(np.float32)
    np.fill_diagonal(sim, -2.0)
    sizes = np.asarray([len(indices) for indices in members], dtype=np.int64)
    weights = np.asarray([sum(max(int(records[idx].n_dets), 1) for idx in indices) for indices in members], dtype=np.float32)
    n = int(sim.shape[0])
    k = min(max(int(candidate_top_k), 1), max(n - 1, 1))
    best: dict[tuple[int, int], dict[str, float | int]] = {}
    directed_rank: dict[tuple[int, int], int] = {}
    directed_second: dict[int, float] = {}
    for src in range(n):
        row = sim[src]
        top = np.argpartition(-row, min(k - 1, n - 1))[:k]
        top = sorted([int(t) for t in top if int(t) != src], key=lambda idx: float(row[idx]), reverse=True)
        directed_second[src] = float(row[top[1]]) if len(top) > 1 else -1.0
        for rank, tgt in enumerate(top, start=1):
            directed_rank[(src, tgt)] = int(rank)
            a, b = (src, tgt) if src < tgt else (tgt, src)
            key = (a, b)
            score = float(row[tgt])
            if key not in best or score > float(best[key]["score"]):
                best[key] = {
                    "source": int(a),
                    "target": int(b),
                    "source_rep": int(reps[a]),
                    "target_rep": int(reps[b]),
                    "source_size": int(sizes[a]),
                    "target_size": int(sizes[b]),
                    "source_weight": float(weights[a]),
                    "target_weight": float(weights[b]),
                    "score": float(score),
                    "centroid_score": float(sim[a, b]),
                    "top_edge_mean": float(score),
                    "rank_margin": float(score - directed_second[src]),
                }
    edges = []
    for (a, b), edge in best.items():
        edge["source_rank"] = int(directed_rank.get((a, b), 1_000_000))
        edge["target_rank"] = int(directed_rank.get((b, a), 1_000_000))
        edges.append(edge)
    edges.sort(key=lambda row: float(row["score"]), reverse=True)
    return edges, {
        "components": int(n),
        "candidate_edges": int(len(edges)),
        "candidate_top_k": int(candidate_top_k),
        "candidate_centroid_only": True,
    }


def _score_edges(edges, sims, ranks, *, score_mode: str, rank_k: int, sim_threshold: float):
    out = []
    names = sorted(sims)
    for edge in edges:
        a = int(edge["source"])
        b = int(edge["target"])
        values = np.asarray([float(sims[name][a, b]) for name in names], dtype=np.float32)
        rank_maxes = np.asarray([max(int(ranks[name][a, b]), int(ranks[name][b, a])) for name in names], dtype=np.float32)
        vote_rank = float(np.mean(rank_maxes <= int(rank_k))) if len(names) else 0.0
        vote_sim = float(np.mean(values >= float(sim_threshold))) if len(names) else 0.0
        mean_sim = float(values.mean()) if len(values) else 0.0
        min_sim = float(values.min()) if len(values) else 0.0
        max_sim = float(values.max()) if len(values) else 0.0
        std_sim = float(values.std()) if len(values) else 0.0
        if score_mode == "rank_vote":
            score = vote_rank
        elif score_mode == "sim_vote":
            score = vote_sim
        elif score_mode == "min_sim":
            score = min_sim
        elif score_mode == "mean_min":
            score = 0.5 * mean_sim + 0.5 * min_sim
        else:
            score = 0.45 * mean_sim + 0.25 * min_sim + 0.20 * vote_rank + 0.10 * vote_sim - 0.05 * std_sim
        row = dict(edge)
        row.update(
            {
                "multiview_score": float(score),
                "view_mean_sim": mean_sim,
                "view_min_sim": min_sim,
                "view_max_sim": max_sim,
                "view_std_sim": std_sim,
                "view_rank_vote": vote_rank,
                "view_sim_vote": vote_sim,
                "view_count": int(len(names)),
                "rank_k": int(rank_k),
                "sim_threshold": float(sim_threshold),
                "score_mode": str(score_mode),
            }
        )
        out.append(row)
    out.sort(key=lambda row: float(row["multiview_score"]), reverse=True)
    return out


def _merge_edges(
    records,
    base_labels,
    edges,
    forbidden,
    *,
    threshold: float,
    min_rank_vote: float,
    min_sim_vote: float,
    max_component_size: int,
    max_accepted_edges: int = 0,
    one_edge_per_component: bool = False,
    max_edge_rank: int = 1_000_000,
    forbidden_override_small_side: int = 0,
    forbidden_override_min_size_ratio: float = 0.0,
    accepted_preview_n: int = 20,
):
    uf = _UnionFind(len(base_labels))
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(base_labels.tolist()):
        groups[int(label)].append(int(idx))
    for indices in groups.values():
        head = int(indices[0])
        for idx in indices[1:]:
            uf.merge(head, int(idx))
    accepted = 0
    rejected_threshold = 0
    rejected_vote = 0
    rejected_forbidden = 0
    rejected_size = 0
    rejected_stale = 0
    rejected_budget = 0
    rejected_diversity = 0
    rejected_edge_rank = 0
    forbidden_overrides = 0
    accepted_preview: list[dict[str, float | int | bool | str]] = []
    accepted_score_sum = 0.0
    accepted_mean_sim_sum = 0.0
    accepted_min_sim_sum = 0.0
    accepted_mass_proxy_sum = 0.0
    accepted_pair_mass_proxy_sum = 0.0
    accepted_min_weight_sum = 0.0
    accepted_max_weight_sum = 0.0
    accepted_size_product_sum = 0.0
    used_components: set[int] = set()
    max_accepted_edges = max(int(max_accepted_edges), 0)
    max_edge_rank = max(int(max_edge_rank), 1)
    forbidden_override_small_side = max(int(forbidden_override_small_side), 0)
    forbidden_override_min_size_ratio = max(float(forbidden_override_min_size_ratio), 0.0)
    accepted_preview_n = max(int(accepted_preview_n), 0)
    for edge in sorted(edges, key=lambda row: float(row["multiview_score"]), reverse=True):
        if float(edge["multiview_score"]) < float(threshold):
            rejected_threshold += 1
            continue
        if float(edge["view_rank_vote"]) < float(min_rank_vote) or float(edge["view_sim_vote"]) < float(min_sim_vote):
            rejected_vote += 1
            continue
        if max(int(edge.get("source_rank", 1_000_000)), int(edge.get("target_rank", 1_000_000))) > max_edge_rank:
            rejected_edge_rank += 1
            continue
        if max_accepted_edges and accepted >= max_accepted_edges:
            rejected_budget += 1
            continue
        source_component = int(edge["source"])
        target_component = int(edge["target"])
        if bool(one_edge_per_component) and (
            source_component in used_components or target_component in used_components
        ):
            rejected_diversity += 1
            continue
        a = int(edge["source_rep"])
        b = int(edge["target_rep"])
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            rejected_stale += 1
            continue
        if len(uf.members[ra]) + len(uf.members[rb]) > int(max_component_size):
            rejected_size += 1
            continue
        if not uf.can_merge(a, b, forbidden, int(max_component_size)):
            source_size = int(edge.get("source_size", len(uf.members[ra])))
            target_size = int(edge.get("target_size", len(uf.members[rb])))
            small_side = max(min(source_size, target_size), 1)
            large_side = max(max(source_size, target_size), 1)
            can_override_forbidden = (
                forbidden_override_small_side > 0
                and small_side <= forbidden_override_small_side
                and float(large_side / small_side) >= forbidden_override_min_size_ratio
            )
            if not can_override_forbidden:
                rejected_forbidden += 1
                continue
            forbidden_overrides += 1
        source_weight = float(edge.get("source_weight", max(int(edge.get("source_size", 1)), 1)))
        target_weight = float(edge.get("target_weight", max(int(edge.get("target_size", 1)), 1)))
        source_size = int(edge.get("source_size", 1))
        target_size = int(edge.get("target_size", 1))
        bridge_mass_proxy = float(np.sqrt(max(source_weight, 1.0) * max(target_weight, 1.0)))
        pair_mass_proxy = float(max(source_weight, 1.0) * max(target_weight, 1.0))
        accepted_score_sum += float(edge.get("multiview_score", 0.0))
        accepted_mean_sim_sum += float(edge.get("view_mean_sim", 0.0))
        accepted_min_sim_sum += float(edge.get("view_min_sim", 0.0))
        accepted_mass_proxy_sum += bridge_mass_proxy
        accepted_pair_mass_proxy_sum += pair_mass_proxy
        accepted_min_weight_sum += min(source_weight, target_weight)
        accepted_max_weight_sum += max(source_weight, target_weight)
        accepted_size_product_sum += float(max(source_size, 1) * max(target_size, 1))
        if len(accepted_preview) < accepted_preview_n:
            accepted_preview.append(
                {
                    "accepted_order": int(accepted + 1),
                    "source": int(source_component),
                    "target": int(target_component),
                    "source_rep": int(a),
                    "target_rep": int(b),
                    "source_size": int(source_size),
                    "target_size": int(target_size),
                    "source_weight": float(source_weight),
                    "target_weight": float(target_weight),
                    "pre_merge_source_root_size": int(len(uf.members[ra])),
                    "pre_merge_target_root_size": int(len(uf.members[rb])),
                    "score": float(edge.get("multiview_score", 0.0)),
                    "view_mean_sim": float(edge.get("view_mean_sim", 0.0)),
                    "view_min_sim": float(edge.get("view_min_sim", 0.0)),
                    "view_rank_vote": float(edge.get("view_rank_vote", 0.0)),
                    "view_sim_vote": float(edge.get("view_sim_vote", 0.0)),
                    "source_rank": int(edge.get("source_rank", 1_000_000)),
                    "target_rank": int(edge.get("target_rank", 1_000_000)),
                    "forbidden_override": bool(not uf.can_merge(a, b, forbidden, int(max_component_size))),
                    "bridge_mass_proxy": float(bridge_mass_proxy),
                    "pair_mass_proxy": float(pair_mass_proxy),
                }
            )
        uf.merge(a, b)
        accepted += 1
        used_components.add(source_component)
        used_components.add(target_component)
    labels = uf.labels()
    accepted_denom = float(max(accepted, 1))
    return labels, {
        "merge_threshold": float(threshold),
        "min_rank_vote": float(min_rank_vote),
        "min_sim_vote": float(min_sim_vote),
        "max_component_size": int(max_component_size),
        "max_accepted_edges": int(max_accepted_edges),
        "one_edge_per_component": bool(one_edge_per_component),
        "max_edge_rank": int(max_edge_rank),
        "forbidden_override_small_side": int(forbidden_override_small_side),
        "forbidden_override_min_size_ratio": float(forbidden_override_min_size_ratio),
        "accepted_edges": int(accepted),
        "forbidden_overrides": int(forbidden_overrides),
        "accepted_preview": accepted_preview,
        "accepted_preview_n": int(accepted_preview_n),
        "accepted_score_mean": float(accepted_score_sum / accepted_denom),
        "accepted_view_mean_sim_mean": float(accepted_mean_sim_sum / accepted_denom),
        "accepted_view_min_sim_mean": float(accepted_min_sim_sum / accepted_denom),
        "accepted_mass_proxy_sum": float(accepted_mass_proxy_sum),
        "accepted_pair_mass_proxy_sum": float(accepted_pair_mass_proxy_sum),
        "accepted_min_weight_sum": float(accepted_min_weight_sum),
        "accepted_max_weight_sum": float(accepted_max_weight_sum),
        "accepted_size_product_sum": float(accepted_size_product_sum),
        "rejected_threshold": int(rejected_threshold),
        "rejected_vote": int(rejected_vote),
        "rejected_forbidden": int(rejected_forbidden),
        "rejected_size": int(rejected_size),
        "rejected_stale": int(rejected_stale),
        "rejected_budget": int(rejected_budget),
        "rejected_diversity": int(rejected_diversity),
        "rejected_edge_rank": int(rejected_edge_rank),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "uses_ground_truth": False,
    }


def _row_sort_key(row: dict[str, object], rank_by: str) -> tuple[float, ...]:
    if rank_by == "precision":
        return (
            _row_float(row, "tracklet_pair_precision"),
            _row_float(row, "tracklet_pair_f1"),
            _row_float(row, "tracklet_pair_recall"),
        )
    if rank_by == "recall":
        return (
            _row_float(row, "tracklet_pair_recall"),
            _row_float(row, "tracklet_pair_f1"),
            _row_float(row, "tracklet_pair_precision"),
        )
    if rank_by == "mass_proxy":
        return (
            _row_float(row, "accepted_pair_mass_proxy_sum"),
            _row_float(row, "accepted_mass_proxy_sum"),
            _row_float(row, "accepted_edges"),
            _row_float(row, "accepted_score_mean"),
            _row_float(row, "tracklet_pair_f1"),
        )
    if rank_by == "mass_then_pair":
        return (
            _row_float(row, "accepted_pair_mass_proxy_sum"),
            _row_float(row, "tracklet_pair_f1"),
            _row_float(row, "tracklet_pair_precision"),
            _row_float(row, "tracklet_pair_recall"),
        )
    return (
        _row_float(row, "tracklet_pair_f1"),
        _row_float(row, "tracklet_pair_recall"),
        _row_float(row, "tracklet_pair_precision"),
    )


def _self_test() -> None:
    base = np.asarray([0, 1, 2, 3], dtype=np.int64)
    forbidden = [set() for _ in range(4)]
    edges = [
        {
            "source": 0,
            "target": 1,
            "source_rep": 0,
            "target_rep": 1,
            "source_size": 1,
            "target_size": 1,
            "source_rank": 1,
            "target_rank": 1,
            "multiview_score": 0.90,
            "view_rank_vote": 1.0,
            "view_sim_vote": 1.0,
        },
        {
            "source": 1,
            "target": 2,
            "source_rep": 1,
            "target_rep": 2,
            "source_size": 1,
            "target_size": 1,
            "source_rank": 1,
            "target_rank": 1,
            "multiview_score": 0.80,
            "view_rank_vote": 1.0,
            "view_sim_vote": 1.0,
        },
        {
            "source": 2,
            "target": 3,
            "source_rep": 2,
            "target_rep": 3,
            "source_size": 1,
            "target_size": 1,
            "source_rank": 1,
            "target_rank": 1,
            "multiview_score": 0.70,
            "view_rank_vote": 1.0,
            "view_sim_vote": 1.0,
        },
    ]
    _labels, budget = _merge_edges(
        None,
        base,
        edges,
        forbidden,
        threshold=0.0,
        min_rank_vote=0.0,
        min_sim_vote=0.0,
        max_component_size=10,
        max_accepted_edges=2,
        one_edge_per_component=False,
        accepted_preview_n=1,
    )
    assert budget["accepted_edges"] == 2
    assert budget["rejected_budget"] == 1
    assert len(budget["accepted_preview"]) == 1
    assert budget["accepted_pair_mass_proxy_sum"] > 0.0
    labels, diverse = _merge_edges(
        None,
        base,
        edges,
        forbidden,
        threshold=0.0,
        min_rank_vote=0.0,
        min_sim_vote=0.0,
        max_component_size=10,
        max_accepted_edges=2,
        one_edge_per_component=True,
    )
    assert diverse["accepted_edges"] == 2
    assert diverse["rejected_diversity"] == 1
    assert labels.tolist() == [0, 0, 1, 1]
    pair_top = {"tracklet_pair_f1": 0.90, "accepted_pair_mass_proxy_sum": 100.0}
    mass_top = {"tracklet_pair_f1": 0.80, "accepted_pair_mass_proxy_sum": 1000.0}
    assert _row_sort_key(mass_top, "mass_proxy") > _row_sort_key(pair_top, "mass_proxy")
    assert _row_sort_key(pair_top, "pair") > _row_sort_key(mass_top, "pair")
    forbidden_tiny = [{1}, {0}, set(), set()]
    tiny_edge = [
        {
            "source": 0,
            "target": 1,
            "source_rep": 0,
            "target_rep": 1,
            "source_size": 24,
            "target_size": 1,
            "source_rank": 1,
            "target_rank": 1,
            "multiview_score": 0.95,
            "view_rank_vote": 1.0,
            "view_sim_vote": 1.0,
        }
    ]
    _blocked_labels, blocked = _merge_edges(
        None,
        base,
        tiny_edge,
        forbidden_tiny,
        threshold=0.0,
        min_rank_vote=0.0,
        min_sim_vote=0.0,
        max_component_size=10,
    )
    assert blocked["accepted_edges"] == 0
    assert blocked["rejected_forbidden"] == 1
    override_labels, override = _merge_edges(
        None,
        base,
        tiny_edge,
        forbidden_tiny,
        threshold=0.0,
        min_rank_vote=0.0,
        min_sim_vote=0.0,
        max_component_size=10,
        forbidden_override_small_side=1,
        forbidden_override_min_size_ratio=12.0,
    )
    assert override["accepted_edges"] == 1
    assert override["forbidden_overrides"] == 1
    assert override_labels.tolist() == [0, 0, 1, 2]
    large_forbidden = [dict(tiny_edge[0], source_size=24, target_size=12)]
    _large_labels, large_blocked = _merge_edges(
        None,
        base,
        large_forbidden,
        forbidden_tiny,
        threshold=0.0,
        min_rank_vote=0.0,
        min_sim_vote=0.0,
        max_component_size=10,
        forbidden_override_small_side=1,
        forbidden_override_min_size_ratio=2.0,
    )
    assert large_blocked["accepted_edges"] == 0
    assert large_blocked["rejected_forbidden"] == 1
    rank_blocked_edge = [dict(tiny_edge[0], source_rank=4, target_rank=4)]
    _rank_labels, rank_blocked = _merge_edges(
        None,
        base,
        rank_blocked_edge,
        [set() for _ in range(4)],
        threshold=0.0,
        min_rank_vote=0.0,
        min_sim_vote=0.0,
        max_component_size=10,
        max_edge_rank=3,
    )
    assert rank_blocked["accepted_edges"] == 0
    assert rank_blocked["rejected_edge_rank"] == 1
    print(
        json.dumps(
            {
                "stage": "self_test",
                "status": "ok",
                "budget": budget,
                "diverse": diverse,
                "forbidden_override": override,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", default="")
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--primary-feature-npz", default="")
    ap.add_argument("--view", action="append", default=[], help="feature view name:path[:weight], or name:db[:weight]")
    ap.add_argument("--candidate-top-k", type=int, default=100)
    ap.add_argument("--top-edge-k", type=int, default=10)
    ap.add_argument("--centroid-weight", type=float, default=0.0)
    ap.add_argument("--candidate-centroid-only", action="store_true")
    ap.add_argument("--rank-ks", default="1,3,5,10")
    ap.add_argument("--sim-thresholds", default="0.58,0.62,0.66,0.70,0.74")
    ap.add_argument("--score-modes", default="hybrid,rank_vote,sim_vote,mean_min,min_sim")
    ap.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    ap.add_argument("--min-rank-votes", default="0.0,0.25,0.5,0.75")
    ap.add_argument("--min-sim-votes", default="0.0,0.25,0.5,0.75")
    ap.add_argument("--max-component-sizes", default="500")
    ap.add_argument(
        "--max-accepted-edges-grid",
        default="0",
        help="0 means uncapped. Positive values cap accepted component bridges, matching oracle top-k repair diagnostics.",
    )
    ap.add_argument(
        "--one-edge-per-component",
        action="store_true",
        help="accept at most one original component bridge per source/target component for diversity.",
    )
    ap.add_argument(
        "--edge-rank-maxes",
        default="1000000",
        help="Maximum max(source_rank,target_rank) from the original candidate edge table.",
    )
    ap.add_argument(
        "--forbidden-override-small-side-sizes",
        default="0",
        help="0 disables. Positive values allow high-scoring bridges through cannot-link only when the smaller original component is this size or smaller.",
    )
    ap.add_argument(
        "--forbidden-override-min-size-ratios",
        default="0",
        help="Minimum larger/smaller original component-size ratio for the tiny-fragment cannot-link override.",
    )
    ap.add_argument("--accepted-preview-n", type=int, default=20)
    ap.add_argument("--rank-by", default="pair", choices=["pair", "precision", "recall", "mass_proxy", "mass_then_pair"])
    ap.add_argument("--disable-forbidden", action="store_true", help="fast diagnostic: skip temporal cannot-link safety checks")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=70_000_000)
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.assignment_csv:
        ap.error("--assignment-csv is required unless --self-test is used")
    if not args.primary_feature_npz:
        ap.error("--primary-feature-npz is required unless --self-test is used")
    if not args.json:
        ap.error("--json is required unless --self-test is used")

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    primary_emb = _load_feature_npz(args.primary_feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0)
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
    keep_seqs, output_info = _output_keep_seqs(records, args)
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    reps, members = _component_members(base_labels, keep_indices)
    if bool(args.disable_forbidden):
        forbidden = [set() for _ in records]
    else:
        forbidden = _build_overlap_forbidden(records)
    if bool(args.candidate_centroid_only):
        base_edges, edge_info = _centroid_candidate_edges(records, primary_emb, reps, members, int(args.candidate_top_k))
    else:
        base_edges, edge_info = _candidate_edges(
            records,
            primary_emb,
            reps,
            members,
            candidate_top_k=int(args.candidate_top_k),
            top_edge_k=int(args.top_edge_k),
            centroid_weight=float(args.centroid_weight),
            min_source_size=1,
            max_source_size=1_000_000,
            min_target_size=1,
            max_target_size=1_000_000,
            forbid_camera_overlap=False,
            forbid_video_overlap=False,
        )
    view_embeddings: dict[str, np.ndarray] = {"primary": primary_emb.astype(np.float32)}
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        if path.lower() == "db":
            view_embeddings[name] = _l2n(db_emb.astype(np.float32)) * float(weight)
        else:
            view_embeddings[name] = _load_npz_aligned(path, records, weight=float(weight))
    sims, ranks = _view_tables(view_embeddings, members)
    rows: list[dict[str, object]] = []
    edge_previews: list[dict[str, object]] = []
    scored_cache: dict[tuple[str, int, float], list[dict[str, object]]] = {}
    for score_mode in [part for part in str(args.score_modes).split(",") if part.strip()]:
        for rank_k in _parse_ints(args.rank_ks):
            for sim_threshold in _parse_floats(args.sim_thresholds):
                scored = _score_edges(
                    base_edges,
                    sims,
                    ranks,
                    score_mode=str(score_mode),
                    rank_k=int(rank_k),
                    sim_threshold=float(sim_threshold),
                )
                scored_cache[(str(score_mode), int(rank_k), float(sim_threshold))] = scored
                if len(edge_previews) < 5:
                    edge_previews.extend(scored[:5])
                for max_component_size in _parse_ints(args.max_component_sizes):
                    for max_accepted_edges in _parse_ints(args.max_accepted_edges_grid):
                        for max_edge_rank in _parse_ints(args.edge_rank_maxes):
                            for threshold in _parse_floats(args.thresholds):
                                for min_rank_vote in _parse_floats(args.min_rank_votes):
                                    for min_sim_vote in _parse_floats(args.min_sim_votes):
                                        for forbidden_override_small_side in _parse_ints(
                                            args.forbidden_override_small_side_sizes
                                        ):
                                            for forbidden_override_min_size_ratio in _parse_floats(
                                                args.forbidden_override_min_size_ratios
                                            ):
                                                labels, info = _merge_edges(
                                                    records,
                                                    base_labels,
                                                    scored,
                                                    forbidden,
                                                    threshold=float(threshold),
                                                    min_rank_vote=float(min_rank_vote),
                                                    min_sim_vote=float(min_sim_vote),
                                                    max_component_size=int(max_component_size),
                                                    max_accepted_edges=int(max_accepted_edges),
                                                    one_edge_per_component=bool(args.one_edge_per_component),
                                                    max_edge_rank=int(max_edge_rank),
                                                    forbidden_override_small_side=int(forbidden_override_small_side),
                                                    forbidden_override_min_size_ratio=float(
                                                        forbidden_override_min_size_ratio
                                                    ),
                                                    accepted_preview_n=int(args.accepted_preview_n),
                                                )
                                                pred = _labels_to_seq_map(
                                                    records,
                                                    labels,
                                                    offset=int(args.assignment_offset),
                                                    keep_seqs=keep_seqs,
                                                )
                                                pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                                                rows.append(
                                                    {
                                                        "mode": "assignment_multiview_merge",
                                                        "score_mode": str(score_mode),
                                                        "rank_k": int(rank_k),
                                                        "sim_threshold": float(sim_threshold),
                                                        **info,
                                                        **pair,
                                                        "uses_anchors": False,
                                                        "uses_gt_for_training_or_anchors": False,
                                                        "uses_gt_for_evaluation_only": True,
                                                    }
                                                )
    rows.sort(key=lambda row: _row_sort_key(row, str(args.rank_by)), reverse=True)

    labels_by_rank: dict[int, np.ndarray] = {}
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        scored = scored_cache[(str(row["score_mode"]), int(row["rank_k"]), float(row["sim_threshold"]))]
        labels, _info = _merge_edges(
            records,
            base_labels,
            scored,
            forbidden,
            threshold=float(row["merge_threshold"]),
            min_rank_vote=float(row["min_rank_vote"]),
            min_sim_vote=float(row["min_sim_vote"]),
            max_component_size=int(row["max_component_size"]),
            max_accepted_edges=int(row.get("max_accepted_edges", 0)),
            one_edge_per_component=bool(row.get("one_edge_per_component", False)),
            max_edge_rank=int(row.get("max_edge_rank", 1_000_000)),
            forbidden_override_small_side=int(row.get("forbidden_override_small_side", 0)),
            forbidden_override_min_size_ratio=float(row.get("forbidden_override_min_size_ratio", 0.0)),
            accepted_preview_n=int(args.accepted_preview_n),
        )
        labels_by_rank[rank] = labels
        full = _score_full(
            pred_by_video,
            gt_by_video,
            _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs),
        )
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": int(rank), "full": full, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            scored = scored_cache[(str(row["score_mode"]), int(row["rank_k"]), float(row["sim_threshold"]))]
            labels, _info = _merge_edges(
                records,
                base_labels,
                scored,
                forbidden,
                threshold=float(row["merge_threshold"]),
                min_rank_vote=float(row["min_rank_vote"]),
                min_sim_vote=float(row["min_sim_vote"]),
                max_component_size=int(row["max_component_size"]),
                max_accepted_edges=int(row.get("max_accepted_edges", 0)),
                one_edge_per_component=bool(row.get("one_edge_per_component", False)),
                max_edge_rank=int(row.get("max_edge_rank", 1_000_000)),
                forbidden_override_small_side=int(row.get("forbidden_override_small_side", 0)),
                forbidden_override_min_size_ratio=float(row.get("forbidden_override_min_size_ratio", 0.0)),
                accepted_preview_n=int(args.accepted_preview_n),
            )
        assignment_info = _write_assignments(
            args.assignments_out,
            records,
            labels,
            keep_seqs=keep_seqs,
            offset=int(args.assignment_offset),
        )
        rows[0].update(assignment_info)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "assignment_csv": str(args.assignment_csv),
        "primary_feature_npz": str(args.primary_feature_npz),
        "views": sorted(view_embeddings),
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "edge_info": edge_info,
        "disable_forbidden": bool(args.disable_forbidden),
        "edge_previews": edge_previews[:20],
        "assignment_info": assignment_info,
        "rank_by": str(args.rank_by),
        "top": rows[: max(80, int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"base": base_pair, "best": rows[0] if rows else None, "json": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
