#!/usr/bin/env python
"""No-anchor pseudo-supervised global-ID model for VLINCS tracklets.

The model is a pair-link calibrator: it predicts whether two tracklets should
share a forced global ID from label-free evidence only. Pseudo positives come
from agreement between independent no-GT resolvers / very strong visual links;
pseudo negatives come from physical cannot-link constraints and low-similarity
background pairs. Ground truth, when provided, is used only for reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        TrackletRecord,
        _UnionFind,
        _build_overlap_forbidden,
        _cache_eval_labels,
        _connect,
        _graph_cache,
        _label_tracklets_for_eval,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _parse_float_list,
        _parse_int_list,
        _score_full,
        _time_agglom_resolve,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        TrackletRecord,
        _UnionFind,
        _build_overlap_forbidden,
        _cache_eval_labels,
        _connect,
        _graph_cache,
        _label_tracklets_for_eval,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _parse_float_list,
        _parse_int_list,
        _score_full,
        _time_agglom_resolve,
        _with_detection_endpoints,
    )

from vlincs_gallery.eval.score import load_ds1_gt_by_video


BASE_FEATURE_NAMES = [
    "visual_cosine",
    "same_camera",
    "same_video",
    "same_stream",
    "stream_overlap",
    "frame_gap_log",
    "abs_ms_gap_log",
    "time_support",
    "height_similarity",
    "width_similarity",
    "area_similarity",
    "same_stream_center_similarity",
    "min_dets_log",
    "max_dets_log",
    "min_conf",
    "mean_conf",
]
FEATURE_NAMES = list(BASE_FEATURE_NAMES)


@dataclass(frozen=True)
class PairFeatureView:
    name: str
    features: np.ndarray
    valid: np.ndarray | None = None


@dataclass(frozen=True)
class PairModelConfig:
    train_top_k: int = 30
    infer_top_k: int = 30
    min_dets: int = 10
    max_component_size: int = 120
    min_merge_support: int = 1
    min_merge_support_ratio: float = 0.0
    exclude_same: str = "camera"
    pseudo_theta: float = 0.018
    pseudo_top_k: int = 15
    pseudo_temporal_bonus: float = 0.005
    pseudo_time_window_ms: int = 1000
    pseudo_ensemble: bool = False
    pseudo_ensemble_thetas: str = ""
    pseudo_ensemble_min_votes: int = 2
    pseudo_ensemble_max_neg_votes: int = 0
    pseudo_consensus_pos_min_votes: int = 0
    pseudo_consensus_pos_min_sim: float = 0.70
    pseudo_pos_min_sim: float = 0.64
    pseudo_strong_pos_sim: float = 0.78
    pseudo_neg_max_sim: float = 0.42
    candidate_time_bonus: float = 0.0
    affinity_time_bonus: float = 0.005
    attach_threshold: float = 0.72
    attach_margin: float = 0.06
    attach_model_weight: float = 0.65
    attach_max_source_size: int = 2
    attach_min_target_size: int = 2
    attach_top_k: int = 60
    attach_min_edge_support: int = 1
    attach_score_agg: str = "max"
    attach_top_mean_k: int = 3
    max_neg_per_pos: float = 4.0
    random_negatives: int = 60000
    model_type: str = "hgb"
    random_state: int = 17


def _cfg_theta_grid(cfg: PairModelConfig) -> list[float]:
    if str(cfg.pseudo_ensemble_thetas).strip():
        base = _parse_float_list(str(cfg.pseudo_ensemble_thetas))
    else:
        base = [0.016, 0.018, 0.020, 0.022]
    out: list[float] = []
    for theta in [float(cfg.pseudo_theta), *base]:
        if all(abs(float(theta) - existing) > 1.0e-9 for existing in out):
            out.append(float(theta))
    return out


def _load_online_gids(con) -> dict[int, int]:
    with con.cursor() as cur:
        cur.execute(
            """SELECT seq, gid, COUNT(*) AS n
               FROM assignments
               GROUP BY seq, gid
               ORDER BY seq, n DESC, gid"""
        )
        rows = cur.fetchall()
    out: dict[int, int] = {}
    for seq, gid, _n in rows:
        out.setdefault(int(seq), int(gid))
    return out


def _group_values(records: list[TrackletRecord], exclude_same: str) -> list[str]:
    values: list[str] = []
    for idx, record in enumerate(records):
        if exclude_same == "none":
            values.append(f"node:{idx}")
        elif exclude_same == "video":
            values.append(record.video)
        elif exclude_same == "stream":
            values.append(f"{record.video}:{record.camera}")
        else:
            values.append(record.camera)
    return values


def _gap_frames(a: TrackletRecord, b: TrackletRecord) -> int:
    if a.start_frame <= b.end_frame and b.start_frame <= a.end_frame:
        return 0
    return max(0, b.start_frame - a.end_frame, a.start_frame - b.end_frame)


def _gap_ms(a: TrackletRecord, b: TrackletRecord) -> int:
    if not (a.start_abs_ms and a.end_abs_ms and b.start_abs_ms and b.end_abs_ms):
        return 0
    if a.start_abs_ms <= b.end_abs_ms and b.start_abs_ms <= a.end_abs_ms:
        return 0
    return max(0, b.start_abs_ms - a.end_abs_ms, a.start_abs_ms - b.end_abs_ms)


def _ratio_similarity(left: float, right: float) -> float:
    left = max(float(left), 1.0)
    right = max(float(right), 1.0)
    return float(np.exp(-abs(np.log(left / right))))


def _parse_pair_feature_source(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"pair feature source must be name:path, got {text!r}")
    name, path = text.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"pair feature source has empty name: {text!r}")
    return name, path


def _load_pair_feature_views(specs: list[str], records: list[TrackletRecord]) -> tuple[list[PairFeatureView], list[str], list[dict[str, object]]]:
    views: list[PairFeatureView] = []
    names: list[str] = []
    meta: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for spec in specs:
        name, path = _parse_pair_feature_source(spec)
        if name in seen_names:
            raise ValueError(f"duplicate pair feature name {name!r}")
        seen_names.add(name)
        data = np.load(path, allow_pickle=True)
        if "seqs" not in data or "features" not in data:
            raise ValueError(f"{path} must contain seqs and features")
        seqs = [int(x) for x in data["seqs"].tolist()]
        features = np.asarray(data["features"], dtype=np.float32)
        by_seq = {seq: features[i] for i, seq in enumerate(seqs)}
        valid_by_seq: dict[int, np.ndarray | bool] | None = None
        if "valid" in data:
            valid_arr = np.asarray(data["valid"]).astype(bool)
            if int(valid_arr.shape[0]) != len(seqs):
                raise ValueError(f"{path} valid mask has {len(valid_arr)} rows but {len(seqs)} seqs")
            valid_by_seq = {seq: valid_arr[i] for i, seq in enumerate(seqs)}
        missing = [record.seq for record in records if record.seq not in by_seq]
        if missing:
            raise ValueError(f"{path} is missing {len(missing)} DB tracklets; first missing seq={missing[0]}")
        aligned = np.stack([by_seq[record.seq] for record in records]).astype(np.float32)
        if aligned.ndim == 2:
            aligned = aligned / (np.linalg.norm(aligned, axis=1, keepdims=True) + 1.0e-9)
        elif aligned.ndim == 3:
            aligned = aligned / (np.linalg.norm(aligned, axis=2, keepdims=True) + 1.0e-9)
        else:
            raise ValueError(f"{path} features must be 2D or 3D, got shape {aligned.shape}")
        aligned_valid = None
        if valid_by_seq is not None:
            aligned_valid = np.stack([valid_by_seq[record.seq] for record in records]).astype(bool)
        views.append(PairFeatureView(name=name, features=aligned, valid=aligned_valid))
        if aligned.ndim == 2:
            names.append(f"{name}_cosine")
            if aligned_valid is not None:
                names.extend([f"{name}_both_valid", f"{name}_either_valid"])
        else:
            names.extend(
                [
                    f"{name}_proto_max",
                    f"{name}_proto_top2_mean",
                    f"{name}_proto_top3_mean",
                    f"{name}_proto_mean",
                    f"{name}_proto_min",
                    f"{name}_proto_std",
                    f"{name}_proto_valid_pairs",
                ]
            )
        meta_item = {"name": name, "path": path, "shape": list(features.shape), "n_tracklets": int(len(seqs))}
        if features.ndim == 2:
            meta_item["dim"] = int(features.shape[1])
        elif features.ndim == 3:
            meta_item["slots"] = int(features.shape[1])
            meta_item["dim"] = int(features.shape[2])
        if aligned_valid is not None:
            meta_item["valid_tracklets"] = int(aligned_valid.sum())
        meta.append(meta_item)
    if len(views) >= 2:
        names.extend(["pair_view_cosine_mean", "pair_view_cosine_min", "pair_view_cosine_max", "pair_view_cosine_std"])
    return views, names, meta


def _pair_view_features(pair_feature_views: list[PairFeatureView] | None, i: int, j: int) -> list[float]:
    if not pair_feature_views:
        return []
    features: list[float] = []
    cosine_values: list[float] = []
    for view in pair_feature_views:
        if view.features.ndim == 2:
            cosine = float(np.dot(view.features[i], view.features[j]))
            cosine_values.append(cosine)
            features.append(cosine)
            if view.valid is not None:
                left_valid = bool(view.valid[i])
                right_valid = bool(view.valid[j])
                features.extend([float(left_valid and right_valid), float(left_valid or right_valid)])
            continue
        if view.features.ndim != 3:
            raise ValueError(f"pair feature view {view.name!r} has unsupported shape {view.features.shape}")
        left = view.features[i]
        right = view.features[j]
        sim = (left @ right.T).astype(np.float32)
        if view.valid is not None:
            left_valid = np.asarray(view.valid[i], dtype=bool)
            right_valid = np.asarray(view.valid[j], dtype=bool)
            mask = left_valid[:, None] & right_valid[None, :]
            values = sim[mask]
        else:
            values = sim.reshape(-1)
        if values.size == 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            cosine_values.append(0.0)
            continue
        values = np.sort(values.astype(np.float32))[::-1]
        top2 = values[: min(2, len(values))]
        top3 = values[: min(3, len(values))]
        max_value = float(values[0])
        cosine_values.append(max_value)
        features.extend(
            [
                max_value,
                float(np.mean(top2)),
                float(np.mean(top3)),
                float(np.mean(values)),
                float(values[-1]),
                float(np.std(values)),
                float(len(values)),
            ]
        )
    if len(cosine_values) >= 2:
        arr = np.asarray(cosine_values, dtype=np.float32)
        features.extend([float(arr.mean()), float(arr.min()), float(arr.max()), float(arr.std())])
    return features


def _pair_feature(
    record_a: TrackletRecord,
    record_b: TrackletRecord,
    visual: float,
    time_window_ms: int,
    extra_features: list[float] | None = None,
) -> list[float]:
    same_camera = float(record_a.camera == record_b.camera)
    same_video = float(record_a.video == record_b.video)
    same_stream = float(record_a.video == record_b.video and record_a.camera == record_b.camera)
    stream_overlap = float(
        same_stream
        and record_a.start_frame <= record_b.end_frame
        and record_b.start_frame <= record_a.end_frame
    )
    frame_gap = _gap_frames(record_a, record_b) if same_stream else 1_000_000
    abs_gap = _gap_ms(record_a, record_b)
    time_support = 0.0 if abs_gap <= 0 else float(np.exp(-float(abs_gap) / max(float(time_window_ms), 1.0)))
    if abs_gap == 0 and not stream_overlap:
        time_support = 1.0
    area_a = max(float(record_a.width) * float(record_a.height), 1.0)
    area_b = max(float(record_b.width) * float(record_b.height), 1.0)
    if same_stream:
        scale = max(0.5 * (float(record_a.height) + float(record_b.height)), 1.0)
        center_dist = float(np.hypot(float(record_a.cx) - float(record_b.cx), float(record_a.cy) - float(record_b.cy)))
        center_sim = float(np.exp(-0.5 * (center_dist / scale) ** 2))
    else:
        center_sim = 0.0
    return [
        float(visual),
        same_camera,
        same_video,
        same_stream,
        stream_overlap,
        float(np.log1p(frame_gap)),
        float(np.log1p(abs_gap)),
        time_support,
        _ratio_similarity(record_a.height, record_b.height),
        _ratio_similarity(record_a.width, record_b.width),
        _ratio_similarity(area_a, area_b),
        center_sim,
        float(np.log1p(min(record_a.n_dets, record_b.n_dets))),
        float(np.log1p(max(record_a.n_dets, record_b.n_dets))),
        float(min(record_a.avg_conf, record_b.avg_conf)),
        float(0.5 * (record_a.avg_conf + record_b.avg_conf)),
        *(extra_features or []),
    ]


def _candidate_pairs(
    records: list[TrackletRecord],
    emb: np.ndarray,
    top_k: int,
    exclude_same: str,
    *,
    time_bonus: float = 0.0,
    time_window_ms: int = 1000,
) -> dict[tuple[int, int], float]:
    x = emb.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    sim = (x @ x.T).astype(np.float32)
    if float(time_bonus) != 0.0:
        starts = np.asarray([record.start_abs_ms for record in records], dtype=np.int64)
        ends = np.asarray([record.end_abs_ms for record in records], dtype=np.int64)
        valid = (starts > 0) & (ends > 0)
        gap = np.maximum(0, np.maximum(starts[:, None] - ends[None, :], starts[None, :] - ends[:, None]))
        support = np.exp(-gap.astype(np.float32) / max(float(time_window_ms), 1.0)).astype(np.float32)
        support[gap > int(time_window_ms)] = 0.0
        support[~(valid[:, None] & valid[None, :])] = 0.0
        sim += float(time_bonus) * support
    group_values = _group_values(records, exclude_same)
    if exclude_same != "none":
        groups = np.asarray(group_values, dtype=object)
        sim[groups[:, None] == groups[None, :]] = -2.0
    np.fill_diagonal(sim, -2.0)
    pairs: dict[tuple[int, int], float] = {}
    k = min(max(int(top_k), 1), max(len(records) - 1, 1))
    for i in range(len(records)):
        top = np.argpartition(-sim[i], k - 1)[:k]
        top = top[np.argsort(-sim[i, top])]
        for j in top.tolist():
            j = int(j)
            score = float(sim[i, j])
            if score <= -1.5:
                continue
            pair = (i, j) if i < j else (j, i)
            pairs[pair] = max(score, pairs.get(pair, -1.0))
    return pairs


def _cached_candidate_pairs(
    runtime_cache: dict[str, object] | None,
    records: list[TrackletRecord],
    emb: np.ndarray,
    top_k: int,
    exclude_same: str,
    *,
    time_bonus: float = 0.0,
    time_window_ms: int = 1000,
) -> dict[tuple[int, int], float]:
    if runtime_cache is None:
        return _candidate_pairs(
            records,
            emb,
            top_k,
            exclude_same,
            time_bonus=time_bonus,
            time_window_ms=time_window_ms,
        )
    key = (
        "candidate_pairs",
        int(top_k),
        str(exclude_same),
        round(float(time_bonus), 8),
        int(time_window_ms),
        int(len(records)),
        int(emb.shape[1]),
    )
    cached = runtime_cache.get(key)
    if cached is None:
        cached = _candidate_pairs(
            records,
            emb,
            top_k,
            exclude_same,
            time_bonus=time_bonus,
            time_window_ms=time_window_ms,
        )
        runtime_cache[key] = cached
    return cached  # type: ignore[return-value]


def _sample_random_negatives(
    records: list[TrackletRecord],
    emb: np.ndarray,
    existing: set[tuple[int, int]],
    forbidden: list[set[int]],
    *,
    n_samples: int,
    exclude_same: str,
    rng: np.random.Generator,
) -> dict[tuple[int, int], float]:
    n = len(records)
    group_values = _group_values(records, exclude_same)
    out: dict[tuple[int, int], float] = {}
    attempts = 0
    max_attempts = max(n_samples * 20, 1000)
    while len(out) < n_samples and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i
        if (i, j) in existing or (i, j) in out:
            continue
        if exclude_same != "none" and group_values[i] == group_values[j]:
            continue
        if j in forbidden[i]:
            continue
        out[(i, j)] = float(emb[i] @ emb[j])
    return out


def _pseudo_training_pairs(
    records: list[TrackletRecord],
    emb: np.ndarray,
    online_gids: dict[int, int],
    cfg: PairModelConfig,
    pair_feature_views: list[PairFeatureView] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(int(cfg.random_state))
    ensemble_thetas = _cfg_theta_grid(cfg) if bool(cfg.pseudo_ensemble) else [float(cfg.pseudo_theta)]

    pseudo_label_sets: list[np.ndarray] = []
    pseudo_sizes_sets: list[Counter] = []
    pseudo_infos: list[dict[str, object]] = []
    for theta in ensemble_thetas:
        pseudo_cfg = ResolveConfig(
            mode="time_agglom",
            theta=float(theta),
            top_k=int(cfg.pseudo_top_k),
            min_dets=int(cfg.min_dets),
            exclude_same=str(cfg.exclude_same),
            temporal_bonus=float(cfg.pseudo_temporal_bonus),
            time_window_ms=int(cfg.pseudo_time_window_ms),
        )
        pseudo_labels, pseudo_info = _time_agglom_resolve(records, emb, pseudo_cfg)
        pseudo_label_sets.append(pseudo_labels)
        pseudo_sizes_sets.append(Counter(pseudo_labels.tolist()))
        pseudo_infos.append({"theta": float(theta), **pseudo_info})
    online_sizes = Counter(online_gids.get(record.seq, -1) for record in records)
    forbidden = _build_overlap_forbidden(records)
    pairs = _candidate_pairs(
        records,
        emb,
        max(int(cfg.train_top_k), int(cfg.pseudo_top_k)),
        cfg.exclude_same,
        time_bonus=float(cfg.candidate_time_bonus),
        time_window_ms=int(cfg.pseudo_time_window_ms),
    )
    positives: dict[tuple[int, int], float] = {}
    negatives: dict[tuple[int, int], float] = {}
    positive_vote_counts = Counter()
    negative_vote_counts = Counter()
    positive_source_counts = Counter()

    for (i, j), score in pairs.items():
        votes = 0
        for labels, sizes in zip(pseudo_label_sets, pseudo_sizes_sets):
            if labels[i] == labels[j] and sizes[int(labels[i])] > 1:
                votes += 1
        min_votes = int(cfg.pseudo_ensemble_min_votes) if bool(cfg.pseudo_ensemble) else 1
        max_neg_votes = int(cfg.pseudo_ensemble_max_neg_votes) if bool(cfg.pseudo_ensemble) else 0
        same_pseudo = votes >= min_votes
        gid_i = online_gids.get(records[i].seq)
        gid_j = online_gids.get(records[j].seq)
        same_online = gid_i is not None and gid_i == gid_j and online_sizes.get(gid_i, 0) > 1
        consensus_positive = (
            int(cfg.pseudo_consensus_pos_min_votes) > 0
            and votes >= int(cfg.pseudo_consensus_pos_min_votes)
            and score >= float(cfg.pseudo_consensus_pos_min_sim)
        )
        cannot = j in forbidden[i]
        if cannot:
            negatives[(i, j)] = score
            negative_vote_counts[votes] += 1
            continue
        positive_source = None
        if same_pseudo and same_online and score >= float(cfg.pseudo_pos_min_sim):
            positive_source = "pseudo_online_agree"
        elif score >= float(cfg.pseudo_strong_pos_sim) and same_pseudo:
            positive_source = "strong_visual_pseudo"
        elif consensus_positive:
            positive_source = "consensus_votes"
        if positive_source is not None:
            positives[(i, j)] = score
            positive_vote_counts[votes] += 1
            positive_source_counts[positive_source] += 1
        elif votes <= max_neg_votes and (not same_online) and score <= float(cfg.pseudo_neg_max_sim):
            negatives[(i, j)] = score
            negative_vote_counts[votes] += 1

    random_negs = _sample_random_negatives(
        records,
        emb,
        set(pairs) | set(positives) | set(negatives),
        forbidden,
        n_samples=int(cfg.random_negatives),
        exclude_same=cfg.exclude_same,
        rng=rng,
    )
    negatives.update(random_negs)

    max_neg = int(max(len(positives) * float(cfg.max_neg_per_pos), 1))
    if len(negatives) > max_neg:
        neg_items = list(negatives.items())
        order = rng.permutation(len(neg_items))[:max_neg]
        negatives = {neg_items[int(k)][0]: float(neg_items[int(k)][1]) for k in order}

    rows: list[list[float]] = []
    labels: list[int] = []
    for (i, j), score in positives.items():
        rows.append(
            _pair_feature(
                records[i],
                records[j],
                score,
                cfg.pseudo_time_window_ms,
                _pair_view_features(pair_feature_views, i, j),
            )
        )
        labels.append(1)
    for (i, j), score in negatives.items():
        rows.append(
            _pair_feature(
                records[i],
                records[j],
                score,
                cfg.pseudo_time_window_ms,
                _pair_view_features(pair_feature_views, i, j),
            )
        )
        labels.append(0)
    if not rows or len(set(labels)) < 2:
        raise RuntimeError(f"pseudo training failed: positives={len(positives)} negatives={len(negatives)}")
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    info = {
        "pseudo_resolver": {
            "mode": "time_agglom",
            "top_k": int(cfg.pseudo_top_k),
            "min_dets": int(cfg.min_dets),
            "exclude_same": str(cfg.exclude_same),
            "temporal_bonus": float(cfg.pseudo_temporal_bonus),
            "time_window_ms": int(cfg.pseudo_time_window_ms),
            "ensemble": bool(cfg.pseudo_ensemble),
            "ensemble_thetas": [float(theta) for theta in ensemble_thetas],
            "ensemble_min_votes": int(cfg.pseudo_ensemble_min_votes),
            "ensemble_max_neg_votes": int(cfg.pseudo_ensemble_max_neg_votes),
            "base_runs": pseudo_infos,
        },
        "candidate_pairs": int(len(pairs)),
        "pseudo_positive_pairs": int(len(positives)),
        "pseudo_negative_pairs": int(len(negatives)),
        "pseudo_positive_vote_counts": {str(key): int(value) for key, value in sorted(positive_vote_counts.items())},
        "pseudo_negative_vote_counts": {str(key): int(value) for key, value in sorted(negative_vote_counts.items())},
        "pseudo_positive_source_counts": dict(sorted(positive_source_counts.items())),
        "pseudo_consensus_pos_min_votes": int(cfg.pseudo_consensus_pos_min_votes),
        "pseudo_consensus_pos_min_sim": float(cfg.pseudo_consensus_pos_min_sim),
        "random_negative_pairs": int(len(random_negs)),
        "uses_ground_truth": False,
    }
    return X, y, info


def _fit_model(X: np.ndarray, y: np.ndarray, cfg: PairModelConfig) -> tuple[object, dict[str, object]]:
    stratify = y if min(Counter(y.tolist()).values()) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=int(cfg.random_state),
        stratify=stratify,
    )
    if cfg.model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=180,
            max_depth=16,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=int(cfg.random_state),
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.055,
            l2_regularization=0.02,
            max_leaf_nodes=31,
            random_state=int(cfg.random_state),
        )
    sample_weight = np.where(y_train > 0, 1.0, max(float((y_train == 1).sum()) / max(float((y_train == 0).sum()), 1.0), 0.05))
    model.fit(X_train, y_train, sample_weight=sample_weight)
    prob = model.predict_proba(X_val)[:, 1]
    train_info: dict[str, object] = {
        "model_type": cfg.model_type,
        "train_pairs": int(len(y_train)),
        "validation_pairs": int(len(y_val)),
        "train_positive_fraction": round(float(np.mean(y_train)), 6),
        "validation_positive_fraction": round(float(np.mean(y_val)), 6),
        "pseudo_validation_auc": round(float(roc_auc_score(y_val, prob)), 6) if len(set(y_val.tolist())) > 1 else None,
        "pseudo_validation_ap": round(float(average_precision_score(y_val, prob)), 6) if len(set(y_val.tolist())) > 1 else None,
    }
    if hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_")
        train_info["feature_importances"] = {
            name: round(float(value), 6)
            for name, value in sorted(zip(FEATURE_NAMES, importances), key=lambda item: -float(item[1]))
        }
    return model, train_info


def _resolve_with_model(
    records: list[TrackletRecord],
    emb: np.ndarray,
    model: object,
    cfg: PairModelConfig,
    *,
    threshold: float,
    blend: float,
    solver: str,
    runtime_cache: dict[str, object] | None = None,
    pair_feature_views: list[PairFeatureView] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    keep = [i for i, record in enumerate(records) if record.n_dets >= int(cfg.min_dets)]
    keep_set = set(keep)
    labels = np.full(len(records), -1, dtype=np.int64)
    if len(keep) < 2:
        next_label = 0
        for i in range(len(records)):
            labels[i] = next_label
            next_label += 1
        return labels, {"candidate_edges": 0, "accepted_edges": 0, "components": next_label, "uses_ground_truth": False}

    if solver == "consensus":
        return _resolve_consensus(
            records,
            emb,
            model,
            cfg,
            threshold=threshold,
            guarded=False,
            runtime_cache=runtime_cache,
            pair_feature_views=pair_feature_views,
        )
    if solver == "consensus_guard":
        return _resolve_consensus(
            records,
            emb,
            model,
            cfg,
            threshold=threshold,
            guarded=True,
            runtime_cache=runtime_cache,
            pair_feature_views=pair_feature_views,
        )
    if solver == "consensus_attach":
        return _resolve_consensus_attach(
            records,
            emb,
            model,
            cfg,
            core_threshold=threshold,
            blend=blend,
            runtime_cache=runtime_cache,
            pair_feature_views=pair_feature_views,
        )

    all_pairs = _cached_candidate_pairs(
        runtime_cache,
        records,
        emb,
        int(cfg.infer_top_k),
        cfg.exclude_same,
        time_bonus=float(cfg.candidate_time_bonus),
        time_window_ms=int(cfg.pseudo_time_window_ms),
    )
    pairs = [(i, j, score) for (i, j), score in all_pairs.items() if i in keep_set and j in keep_set]
    if not pairs:
        labels = np.arange(len(records), dtype=np.int64)
        return labels, {"candidate_edges": 0, "accepted_edges": 0, "components": int(len(records)), "uses_ground_truth": False}

    X = np.asarray(
        [
            _pair_feature(
                records[i],
                records[j],
                score,
                cfg.pseudo_time_window_ms,
                _pair_view_features(pair_feature_views, i, j),
            )
            for i, j, score in pairs
        ],
        dtype=np.float32,
    )
    prob = model.predict_proba(X)[:, 1].astype(np.float32)
    raw = np.asarray([score for _i, _j, score in pairs], dtype=np.float32)
    time_support = X[:, FEATURE_NAMES.index("time_support")].astype(np.float32)
    edge_scores = float(blend) * prob + (1.0 - float(blend)) * np.clip(raw, 0.0, 1.0)
    edge_scores = np.clip(edge_scores + float(cfg.affinity_time_bonus) * time_support, 0.0, 1.0)

    if solver == "agglom":
        keep_pos = {idx: pos for pos, idx in enumerate(keep)}
        m = len(keep)
        A = np.zeros((m, m), dtype=np.float32)
        used_edges = 0
        for (i, j, _raw_score), score in zip(pairs, edge_scores):
            pi = keep_pos.get(i)
            pj = keep_pos.get(j)
            if pi is None or pj is None:
                continue
            if float(score) > float(A[pi, pj]):
                A[pi, pj] = float(score)
                A[pj, pi] = float(score)
            used_edges += 1
        np.fill_diagonal(A, 1.0)
        D = 1.0 - A
        np.clip(D, 0.0, None, out=D)
        clustered = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(1.0 - threshold),
            metric="precomputed",
            linkage="average",
        ).fit_predict(D)
        _, clustered = np.unique(clustered, return_inverse=True)
        for pos, i in enumerate(keep):
            labels[i] = int(clustered[pos])
        next_label = int(clustered.max()) + 1 if clustered.size else 0
        for i in range(len(records)):
            if labels[i] < 0:
                labels[i] = next_label
                next_label += 1
        return labels, {
            "candidate_edges": int(len(pairs)),
            "affinity_edges": int(used_edges),
            "accepted_edges": int(np.count_nonzero(np.triu(A > float(threshold), k=1))),
            "components": int(next_label),
            "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
            "n_tracklets": int(len(records)),
            "n_clustered": int(len(keep)),
            "n_singleton": int(len(records) - len(keep)),
            "solver": solver,
            "uses_ground_truth": False,
        }

    edge_lookup: dict[tuple[int, int], float] = {}
    for (i, j, _raw_score), score in zip(pairs, edge_scores):
        pair = (i, j) if i < j else (j, i)
        edge_lookup[pair] = max(float(score), edge_lookup.get(pair, -1.0))

    order = np.argsort(-edge_scores)
    forbidden = _build_overlap_forbidden(records)
    uf = _UnionFind(len(records))
    accepted = 0
    rejected_threshold = 0
    rejected_forbidden = 0
    rejected_size = 0
    rejected_support = 0
    support_sum = 0
    for idx in order.tolist():
        score = float(edge_scores[int(idx)])
        if score < float(threshold):
            rejected_threshold += 1
            continue
        i, j, _raw_score = pairs[int(idx)]
        if j in forbidden[i]:
            rejected_forbidden += 1
            continue
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if len(uf.members[ri]) + len(uf.members[rj]) > int(cfg.max_component_size):
            rejected_size += 1
            continue
        if not uf.can_merge(i, j, forbidden, int(cfg.max_component_size)):
            rejected_forbidden += 1
            continue
        if solver == "support":
            left = uf.members[ri]
            right = uf.members[rj]
            support = 0
            possible = 0
            for a in left:
                for b in right:
                    if b in forbidden[a]:
                        continue
                    possible += 1
                    pair = (a, b) if a < b else (b, a)
                    if edge_lookup.get(pair, -1.0) >= float(threshold):
                        support += 1
            if support < int(cfg.min_merge_support):
                rejected_support += 1
                continue
            if possible > 0 and support / possible < float(cfg.min_merge_support_ratio):
                rejected_support += 1
                continue
            support_sum += int(support)
        uf.merge(i, j)
        accepted += 1
    labels = uf.labels()
    return labels, {
        "candidate_edges": int(len(pairs)),
        "accepted_edges": int(accepted),
        "rejected_threshold": int(rejected_threshold),
        "rejected_forbidden": int(rejected_forbidden),
        "rejected_size": int(rejected_size),
        "rejected_support": int(rejected_support),
        "mean_merge_support": round(float(support_sum) / max(accepted, 1), 6),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "solver": solver,
        "uses_ground_truth": False,
    }


def _consensus_base_partitions(
    records: list[TrackletRecord],
    emb: np.ndarray,
    model: object,
    cfg: PairModelConfig,
    runtime_cache: dict[str, object] | None,
    pair_feature_views: list[PairFeatureView] | None,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    key = (
        "consensus_base_partitions",
        int(cfg.infer_top_k),
        int(cfg.min_dets),
        str(cfg.exclude_same),
        round(float(cfg.candidate_time_bonus), 8),
        round(float(cfg.pseudo_temporal_bonus), 8),
        int(cfg.pseudo_time_window_ms),
        str(cfg.pseudo_ensemble_thetas),
        round(float(cfg.pseudo_theta), 8),
    )
    if runtime_cache is not None and key in runtime_cache:
        cached = runtime_cache[key]
        return cached  # type: ignore[return-value]

    base_labels: list[np.ndarray] = []
    base_infos: list[dict[str, object]] = []
    theta_grid = _cfg_theta_grid(cfg)
    for theta in theta_grid:
        raw_cfg = ResolveConfig(
            mode="time_agglom",
            theta=float(theta),
            top_k=int(cfg.infer_top_k),
            min_dets=int(cfg.min_dets),
            exclude_same=str(cfg.exclude_same),
            temporal_bonus=float(cfg.candidate_time_bonus or cfg.pseudo_temporal_bonus),
            time_window_ms=int(cfg.pseudo_time_window_ms),
        )
        lab, info = _time_agglom_resolve(records, emb, raw_cfg)
        base_labels.append(lab)
        base_infos.append({"kind": "time_agglom", "theta": theta, "components": info.get("components")})

    base_thresholds: list[float] = []
    for theta in theta_grid:
        if all(abs(float(theta) - existing) > 1.0e-9 for existing in base_thresholds):
            base_thresholds.append(float(theta))
    if len(base_thresholds) == 1:
        base_thresholds.append(float(base_thresholds[0]))
    for base_blend, base_thr in (
        (0.25, base_thresholds[0]),
        (0.25, base_thresholds[1]),
        (0.50, base_thresholds[0]),
        (0.75, base_thresholds[0]),
    ):
        lab, info = _resolve_with_model(
            records,
            emb,
            model,
            cfg,
            threshold=float(base_thr),
            blend=float(base_blend),
            solver="agglom",
            runtime_cache=runtime_cache,
            pair_feature_views=pair_feature_views,
        )
        base_labels.append(lab)
        base_infos.append(
            {
                "kind": "pair_model_agglom",
                "threshold": base_thr,
                "blend": base_blend,
                "components": info.get("components"),
            }
        )

    out = (base_labels, base_infos)
    if runtime_cache is not None:
        runtime_cache[key] = out
    return out


def _resolve_consensus(
    records: list[TrackletRecord],
    emb: np.ndarray,
    model: object,
    cfg: PairModelConfig,
    *,
    threshold: float,
    guarded: bool = False,
    runtime_cache: dict[str, object] | None = None,
    pair_feature_views: list[PairFeatureView] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    keep = [i for i, record in enumerate(records) if record.n_dets >= int(cfg.min_dets)]
    labels = np.full(len(records), -1, dtype=np.int64)
    solver_name = "consensus_guard" if guarded else "consensus"
    if len(keep) < 2:
        return np.arange(len(records), dtype=np.int64), {
            "candidate_edges": 0,
            "accepted_edges": 0,
            "components": int(len(records)),
            "solver": solver_name,
            "uses_ground_truth": False,
        }

    base_labels, base_infos = _consensus_base_partitions(records, emb, model, cfg, runtime_cache, pair_feature_views)

    pairs = _cached_candidate_pairs(
        runtime_cache,
        records,
        emb,
        max(int(cfg.infer_top_k), 30),
        cfg.exclude_same,
        time_bonus=float(cfg.candidate_time_bonus),
        time_window_ms=int(cfg.pseudo_time_window_ms),
    )
    keep_pos = {idx: pos for pos, idx in enumerate(keep)}
    m = len(keep)
    A = np.zeros((m, m), dtype=np.float32)
    scored_edges: list[tuple[float, int, int]] = []
    candidate_edges = 0
    for (i, j), _raw_score in pairs.items():
        pi = keep_pos.get(i)
        pj = keep_pos.get(j)
        if pi is None or pj is None:
            continue
        votes = sum(1 for lab in base_labels if int(lab[i]) == int(lab[j]))
        score = float(votes) / max(float(len(base_labels)), 1.0)
        if score > A[pi, pj]:
            A[pi, pj] = score
            A[pj, pi] = score
        scored_edges.append((score, i, j))
        candidate_edges += 1

    if guarded:
        forbidden = _build_overlap_forbidden(records)
        uf = _UnionFind(len(records))
        accepted = 0
        rejected_threshold = 0
        rejected_forbidden = 0
        rejected_size = 0
        for score, i, j in sorted(scored_edges, reverse=True):
            if float(score) < float(threshold):
                rejected_threshold += 1
                continue
            ri, rj = uf.find(i), uf.find(j)
            if ri == rj:
                continue
            if len(uf.members[ri]) + len(uf.members[rj]) > int(cfg.max_component_size):
                rejected_size += 1
                continue
            if not uf.can_merge(i, j, forbidden, int(cfg.max_component_size)):
                rejected_forbidden += 1
                continue
            uf.merge(i, j)
            accepted += 1
        labels = uf.labels()
        return labels, {
            "candidate_edges": int(candidate_edges),
            "accepted_edges": int(accepted),
            "rejected_threshold": int(rejected_threshold),
            "rejected_forbidden": int(rejected_forbidden),
            "rejected_size": int(rejected_size),
            "components": int(len(set(labels.tolist()))),
            "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
            "n_tracklets": int(len(records)),
            "n_clustered": int(len(keep)),
            "n_singleton": int(len(records) - len(keep)),
            "consensus_base_partitions": int(len(base_labels)),
            "consensus_base_infos": base_infos,
            "solver": solver_name,
            "uses_ground_truth": False,
        }

    np.fill_diagonal(A, 1.0)
    D = 1.0 - A
    np.clip(D, 0.0, None, out=D)
    clustered = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(1.0 - threshold),
        metric="precomputed",
        linkage="average",
    ).fit_predict(D)
    _, clustered = np.unique(clustered, return_inverse=True)
    for pos, i in enumerate(keep):
        labels[i] = int(clustered[pos])
    next_label = int(clustered.max()) + 1 if clustered.size else 0
    for i in range(len(records)):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
    return labels, {
        "candidate_edges": int(candidate_edges),
        "accepted_edges": int(np.count_nonzero(np.triu(A >= float(threshold), k=1))),
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "consensus_base_partitions": int(len(base_labels)),
        "consensus_base_infos": base_infos,
        "solver": solver_name,
        "uses_ground_truth": False,
    }


def _attach_scored_pairs(
    records: list[TrackletRecord],
    emb: np.ndarray,
    model: object,
    cfg: PairModelConfig,
    runtime_cache: dict[str, object] | None,
    pair_feature_views: list[PairFeatureView] | None = None,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    key = (
        "attach_scored_pairs",
        max(int(cfg.infer_top_k), int(cfg.attach_top_k)),
        str(cfg.exclude_same),
        round(float(cfg.candidate_time_bonus), 8),
        int(cfg.pseudo_time_window_ms),
        round(float(cfg.attach_model_weight), 8),
        round(float(cfg.affinity_time_bonus), 8),
    )
    if runtime_cache is not None and key in runtime_cache:
        cached = runtime_cache[key]
        return cached  # type: ignore[return-value]

    keep = [i for i, record in enumerate(records) if record.n_dets >= int(cfg.min_dets)]
    keep_set = set(keep)
    pairs_dict = _cached_candidate_pairs(
        runtime_cache,
        records,
        emb,
        max(int(cfg.infer_top_k), int(cfg.attach_top_k)),
        cfg.exclude_same,
        time_bonus=float(cfg.candidate_time_bonus),
        time_window_ms=int(cfg.pseudo_time_window_ms),
    )
    pairs = [(i, j, score) for (i, j), score in pairs_dict.items() if i in keep_set and j in keep_set]
    if not pairs:
        out = (pairs, np.asarray([], dtype=np.float32))
        if runtime_cache is not None:
            runtime_cache[key] = out
        return out

    X = np.asarray(
        [
            _pair_feature(
                records[i],
                records[j],
                score,
                cfg.pseudo_time_window_ms,
                _pair_view_features(pair_feature_views, i, j),
            )
            for i, j, score in pairs
        ],
        dtype=np.float32,
    )
    prob = model.predict_proba(X)[:, 1].astype(np.float32)
    raw = np.asarray([score for _i, _j, score in pairs], dtype=np.float32)
    time_support = X[:, FEATURE_NAMES.index("time_support")].astype(np.float32)
    w = float(np.clip(cfg.attach_model_weight, 0.0, 1.0))
    attach_scores = w * prob + (1.0 - w) * np.clip(raw, 0.0, 1.0)
    attach_scores = np.clip(attach_scores + float(cfg.affinity_time_bonus) * time_support, 0.0, 1.0)
    out = (pairs, attach_scores.astype(np.float32))
    if runtime_cache is not None:
        runtime_cache[key] = out
    return out


def _unionfind_from_labels(labels: np.ndarray) -> _UnionFind:
    uf = _UnionFind(int(len(labels)))
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        groups[int(label)].append(int(idx))
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        head = int(indices[0])
        for idx in indices[1:]:
            uf.merge(head, int(idx))
    return uf


def _resolve_consensus_attach(
    records: list[TrackletRecord],
    emb: np.ndarray,
    model: object,
    cfg: PairModelConfig,
    *,
    core_threshold: float,
    blend: float,
    runtime_cache: dict[str, object] | None = None,
    pair_feature_views: list[PairFeatureView] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    core_labels, core_info = _resolve_consensus(
        records,
        emb,
        model,
        cfg,
        threshold=float(core_threshold),
        guarded=True,
        runtime_cache=runtime_cache,
        pair_feature_views=pair_feature_views,
    )
    keep = [i for i, record in enumerate(records) if record.n_dets >= int(cfg.min_dets)]
    if len(keep) < 2:
        return core_labels, {**core_info, "solver": "consensus_attach", "attach_edges": 0, "attach_accepted": 0}

    pairs, attach_scores = _attach_scored_pairs(records, emb, model, cfg, runtime_cache, pair_feature_views)
    if not pairs:
        return core_labels, {**core_info, "solver": "consensus_attach", "attach_edges": 0, "attach_accepted": 0}

    uf = _unionfind_from_labels(core_labels)
    forbidden = _build_overlap_forbidden(records)
    root_sizes = {root: len(members) for root, members in uf.members.items()}
    best_by_source: dict[int, tuple[float, int, int, int]] = {}
    second_by_source: dict[int, float] = defaultdict(float)
    eligible_edges = 0
    candidate_groups: dict[tuple[int, int], list[tuple[float, int, int]]] = defaultdict(list)

    for (i, j, _raw_score), score in zip(pairs, attach_scores):
        score = float(score)
        if score < float(cfg.attach_threshold):
            continue
        ri = uf.find(i)
        rj = uf.find(j)
        if ri == rj:
            continue
        si = int(root_sizes.get(ri, len(uf.members.get(ri, ()))))
        sj = int(root_sizes.get(rj, len(uf.members.get(rj, ()))))
        candidates: list[tuple[int, int]] = []
        if si <= int(cfg.attach_max_source_size) and sj >= int(cfg.attach_min_target_size):
            candidates.append((ri, rj))
        if sj <= int(cfg.attach_max_source_size) and si >= int(cfg.attach_min_target_size):
            candidates.append((rj, ri))
        if not candidates:
            continue
        eligible_edges += 1
        for source, target in candidates:
            candidate_groups[(source, target)].append((score, int(i), int(j)))

    supported_groups = 0
    for (source, target), edges in candidate_groups.items():
        if len(edges) < int(cfg.attach_min_edge_support):
            continue
        supported_groups += 1
        ordered = sorted(edges, key=lambda item: -float(item[0]))
        top_scores = [float(score) for score, _i, _j in ordered[: max(1, int(cfg.attach_top_mean_k))]]
        if cfg.attach_score_agg == "top_mean":
            score = float(np.mean(np.asarray(top_scores, dtype=np.float32)))
        elif cfg.attach_score_agg == "hybrid":
            score = float(0.5 * top_scores[0] + 0.5 * np.mean(np.asarray(top_scores, dtype=np.float32)))
        else:
            score = float(top_scores[0])
        _best_score, i, j = ordered[0]
        previous = best_by_source.get(source)
        if previous is None or score > previous[0]:
            if previous is not None and previous[1] != target:
                second_by_source[source] = max(float(second_by_source[source]), previous[0])
            best_by_source[source] = (score, target, int(i), int(j))
        elif previous[1] != target:
            second_by_source[source] = max(float(second_by_source[source]), score)

    accepted = 0
    rejected_margin = 0
    rejected_forbidden = 0
    rejected_size = 0
    stale_sources = 0
    for source, (score, target, i, j) in sorted(best_by_source.items(), key=lambda item: -float(item[1][0])):
        if float(score) - float(second_by_source.get(source, 0.0)) < float(cfg.attach_margin):
            rejected_margin += 1
            continue
        current_source = uf.find(source)
        current_target = uf.find(target)
        if current_source != source or current_source == current_target:
            stale_sources += 1
            continue
        if len(uf.members[current_source]) > int(cfg.attach_max_source_size):
            stale_sources += 1
            continue
        if len(uf.members[current_source]) + len(uf.members[current_target]) > int(cfg.max_component_size):
            rejected_size += 1
            continue
        if not uf.can_merge(i, j, forbidden, int(cfg.max_component_size)):
            rejected_forbidden += 1
            continue
        uf.merge(i, j)
        accepted += 1

    labels = uf.labels()
    return labels, {
        **{key: value for key, value in core_info.items() if key not in {"solver", "components", "largest_component"}},
        "solver": "consensus_attach",
        "core_solver": core_info.get("solver"),
        "core_components": int(core_info.get("components", len(set(core_labels.tolist())))),
        "core_largest_component": int(core_info.get("largest_component", max(Counter(core_labels.tolist()).values(), default=0))),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "attach_candidate_edges": int(len(pairs)),
        "attach_eligible_edges": int(eligible_edges),
        "attach_candidate_groups": int(len(candidate_groups)),
        "attach_supported_groups": int(supported_groups),
        "attach_sources": int(len(best_by_source)),
        "attach_accepted": int(accepted),
        "attach_rejected_margin": int(rejected_margin),
        "attach_rejected_forbidden": int(rejected_forbidden),
        "attach_rejected_size": int(rejected_size),
        "attach_stale_sources": int(stale_sources),
        "attach_threshold": float(cfg.attach_threshold),
        "attach_margin": float(cfg.attach_margin),
        "attach_model_weight": float(cfg.attach_model_weight),
        "attach_max_source_size": int(cfg.attach_max_source_size),
        "attach_min_target_size": int(cfg.attach_min_target_size),
        "attach_top_k": int(cfg.attach_top_k),
        "attach_min_edge_support": int(cfg.attach_min_edge_support),
        "attach_score_agg": str(cfg.attach_score_agg),
        "attach_top_mean_k": int(cfg.attach_top_mean_k),
        "uses_ground_truth": False,
    }


def _config_from_args(args) -> PairModelConfig:
    return PairModelConfig(
        train_top_k=int(args.train_top_k),
        infer_top_k=int(args.infer_top_k),
        min_dets=int(args.min_dets),
        max_component_size=int(args.max_component_size),
        min_merge_support=int(args.min_merge_support),
        min_merge_support_ratio=float(args.min_merge_support_ratio),
        exclude_same=str(args.exclude_same),
        pseudo_theta=float(args.pseudo_theta),
        pseudo_top_k=int(args.pseudo_top_k),
        pseudo_temporal_bonus=float(args.pseudo_temporal_bonus),
        pseudo_time_window_ms=int(args.pseudo_time_window_ms),
        pseudo_ensemble=bool(args.pseudo_ensemble),
        pseudo_ensemble_thetas=str(getattr(args, "pseudo_ensemble_thetas", "")),
        pseudo_ensemble_min_votes=int(args.pseudo_ensemble_min_votes),
        pseudo_ensemble_max_neg_votes=int(args.pseudo_ensemble_max_neg_votes),
        pseudo_consensus_pos_min_votes=int(args.pseudo_consensus_pos_min_votes),
        pseudo_consensus_pos_min_sim=float(args.pseudo_consensus_pos_min_sim),
        pseudo_pos_min_sim=float(args.pseudo_pos_min_sim),
        pseudo_strong_pos_sim=float(args.pseudo_strong_pos_sim),
        pseudo_neg_max_sim=float(args.pseudo_neg_max_sim),
        candidate_time_bonus=float(args.candidate_time_bonus),
        affinity_time_bonus=float(args.affinity_time_bonus),
        attach_threshold=float(args.attach_threshold),
        attach_margin=float(args.attach_margin),
        attach_model_weight=float(args.attach_model_weight),
        attach_max_source_size=int(args.attach_max_source_size),
        attach_min_target_size=int(args.attach_min_target_size),
        attach_top_k=int(args.attach_top_k),
        attach_min_edge_support=int(args.attach_min_edge_support),
        attach_score_agg=str(args.attach_score_agg),
        attach_top_mean_k=int(args.attach_top_mean_k),
        max_neg_per_pos=float(args.max_neg_per_pos),
        random_negatives=int(args.random_negatives),
        model_type=str(args.model_type),
        random_state=int(args.random_state),
    )


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if not isinstance(value, (dict, list, tuple))
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def _row_selection_score(row: dict[str, object], key: str) -> float:
    def val(name: str) -> float:
        try:
            return float(row.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0

    if key == "pair_pr_min":
        return min(val("tracklet_pair_precision"), val("tracklet_pair_recall"))
    if key == "pair_gate_margin":
        return min(val("tracklet_pair_f1"), val("tracklet_pair_precision"), val("tracklet_pair_recall"))
    if key == "pair_mean_f1_pr":
        return (val("tracklet_pair_f1") + val("tracklet_pair_precision") + val("tracklet_pair_recall")) / 3.0
    return val(key)


def _select_full_score_rows(rows: list[dict[str, object]], full_n: int, sort_key: str, selection_keys: str) -> list[dict[str, object]]:
    limit = max(int(full_n), 0)
    if limit <= 0 or not rows:
        return []
    keys = [part.strip() for part in str(selection_keys or "").split(",") if part.strip()]
    if not keys:
        keys = [str(sort_key)]
    if str(sort_key) not in keys:
        keys.insert(0, str(sort_key))

    order_by_key = [
        sorted(range(len(rows)), key=lambda idx, key=key: _row_selection_score(rows[idx], key), reverse=True)
        for key in keys
    ]
    selected: list[int] = []
    seen: set[int] = set()
    max_len = max((len(order) for order in order_by_key), default=0)
    for rank in range(max_len):
        for order in order_by_key:
            if rank >= len(order):
                continue
            idx = int(order[rank])
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)
            if len(selected) >= limit:
                return [rows[idx] for idx in selected]
    return [rows[idx] for idx in selected]


def _component_assignment_metadata(
    records: list[TrackletRecord],
    emb: np.ndarray,
    model: object,
    cfg: PairModelConfig,
    labels: np.ndarray,
    *,
    keep_seqs: set[int],
    threshold: float,
    blend: float,
    pair_feature_views: list[PairFeatureView] | None = None,
) -> dict[int, dict[str, object]]:
    kept_indices = [i for i, record in enumerate(records) if int(record.seq) in keep_seqs]
    counts = Counter(int(labels[i]) for i in kept_indices)
    meta: dict[int, dict[str, object]] = {
        int(label): {
            "component_size": int(size),
            "internal_edges": 0,
            "internal_prob_median": 0.0,
            "internal_score_median": 0.0,
            "internal_score_min": 0.0,
            "external_prob_max": 0.0,
            "external_score_max": 0.0,
            "margin_prob": 0.0,
            "confidence": 0.15 if int(size) == 1 else 0.35,
            "decision_status": "forced_singleton" if int(size) == 1 else "forced_component",
        }
        for label, size in counts.items()
    }
    if not kept_indices:
        return meta

    keep_seq_set = {int(seq) for seq in keep_seqs}
    pairs = _candidate_pairs(
        records,
        emb,
        int(cfg.infer_top_k),
        cfg.exclude_same,
        time_bonus=float(cfg.candidate_time_bonus),
        time_window_ms=int(cfg.pseudo_time_window_ms),
    )
    usable_pairs = [
        (i, j, score)
        for (i, j), score in pairs.items()
        if int(records[i].seq) in keep_seq_set and int(records[j].seq) in keep_seq_set
    ]
    if not usable_pairs:
        return meta

    X = np.asarray(
        [
            _pair_feature(
                records[i],
                records[j],
                score,
                cfg.pseudo_time_window_ms,
                _pair_view_features(pair_feature_views, i, j),
            )
            for i, j, score in usable_pairs
        ],
        dtype=np.float32,
    )
    prob = model.predict_proba(X)[:, 1].astype(np.float32)
    raw = np.asarray([score for _i, _j, score in usable_pairs], dtype=np.float32)
    time_support = X[:, FEATURE_NAMES.index("time_support")].astype(np.float32)
    score_values = float(blend) * prob + (1.0 - float(blend)) * np.clip(raw, 0.0, 1.0)
    score_values = np.clip(score_values + float(cfg.affinity_time_bonus) * time_support, 0.0, 1.0)

    internal_prob: dict[int, list[float]] = defaultdict(list)
    internal_score: dict[int, list[float]] = defaultdict(list)
    external_prob_max: dict[int, float] = defaultdict(float)
    external_score_max: dict[int, float] = defaultdict(float)
    for (i, j, _raw_score), p, score in zip(usable_pairs, prob, score_values):
        li = int(labels[i])
        lj = int(labels[j])
        if li == lj:
            internal_prob[li].append(float(p))
            internal_score[li].append(float(score))
        else:
            external_prob_max[li] = max(float(external_prob_max[li]), float(p))
            external_prob_max[lj] = max(float(external_prob_max[lj]), float(p))
            external_score_max[li] = max(float(external_score_max[li]), float(score))
            external_score_max[lj] = max(float(external_score_max[lj]), float(score))

    for label, value in meta.items():
        probs = internal_prob.get(label, [])
        scores = internal_score.get(label, [])
        ext_prob = float(external_prob_max.get(label, 0.0))
        ext_score = float(external_score_max.get(label, 0.0))
        if probs:
            p_med = float(np.median(np.asarray(probs, dtype=np.float32)))
            s_med = float(np.median(np.asarray(scores, dtype=np.float32)))
            s_min = float(np.min(np.asarray(scores, dtype=np.float32)))
            margin = p_med - ext_prob
            margin_term = float(np.clip(0.5 + margin, 0.0, 1.0))
            confidence = float(np.clip(0.50 * p_med + 0.20 * s_med + 0.30 * margin_term, 0.0, 1.0))
            size = int(value["component_size"])
            if size >= 2 and p_med >= 0.70 and margin >= 0.15 and len(probs) >= min(size - 1, 2):
                status = "committed"
            elif size >= 2 and (p_med >= 0.55 or s_med >= float(threshold)):
                status = "provisional"
            else:
                status = "forced_component"
            value.update(
                {
                    "internal_edges": int(len(probs)),
                    "internal_prob_median": round(p_med, 6),
                    "internal_score_median": round(s_med, 6),
                    "internal_score_min": round(s_min, 6),
                    "external_prob_max": round(ext_prob, 6),
                    "external_score_max": round(ext_score, 6),
                    "margin_prob": round(float(margin), 6),
                    "confidence": round(confidence, 6),
                    "decision_status": status,
                }
            )
        else:
            singleton_conf = float(np.clip(0.30 * (1.0 - ext_prob) + 0.05, 0.05, 0.35))
            value.update(
                {
                    "external_prob_max": round(ext_prob, 6),
                    "external_score_max": round(ext_score, 6),
                    "margin_prob": round(float(-ext_prob), 6),
                    "confidence": round(singleton_conf, 6),
                }
            )
    return meta


def _write_assignments(
    path: str,
    records: list[TrackletRecord],
    labels: np.ndarray,
    *,
    keep_seqs: set[int],
    component_meta: dict[int, dict[str, object]] | None = None,
    offset: int = 30_000_000,
) -> dict[str, object]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(int(label) for record, label in zip(records, labels) if int(record.seq) in keep_seqs)
    component_meta = component_meta or {}
    rows = []
    for record, label in zip(records, labels):
        if int(record.seq) not in keep_seqs:
            continue
        label = int(label)
        meta = component_meta.get(label, {})
        rows.append(
            {
                "seq": int(record.seq),
                "tracklet_key": record.tracklet_key,
                "video": record.video,
                "camera": record.camera,
                "start_frame": int(record.start_frame),
                "end_frame": int(record.end_frame),
                "n_dets": int(record.n_dets),
                "avg_conf": round(float(record.avg_conf), 6),
                "predicted_global_id": int(offset + label),
                "component_label": label,
                "component_size": int(counts[label]),
                "prediction_confidence": meta.get("confidence", 0.15 if int(counts[label]) == 1 else 0.35),
                "decision_status": meta.get("decision_status", "forced_singleton" if int(counts[label]) == 1 else "forced_component"),
                "component_internal_edges": meta.get("internal_edges", 0),
                "component_internal_prob_median": meta.get("internal_prob_median", 0.0),
                "component_internal_score_median": meta.get("internal_score_median", 0.0),
                "component_external_prob_max": meta.get("external_prob_max", 0.0),
                "component_margin_prob": meta.get("margin_prob", 0.0),
            }
        )
    fieldnames = [
        "seq",
        "tracklet_key",
        "video",
        "camera",
        "start_frame",
        "end_frame",
        "n_dets",
        "avg_conf",
        "predicted_global_id",
        "component_label",
        "component_size",
        "prediction_confidence",
        "decision_status",
        "component_internal_edges",
        "component_internal_prob_median",
        "component_internal_score_median",
        "component_external_prob_max",
        "component_margin_prob",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    status_counts = Counter(str(row["decision_status"]) for row in rows)
    return {
        "assignments_out": str(path),
        "assignment_rows": int(len(rows)),
        "assignment_components": int(len(counts)),
        "largest_assignment_component": int(max(counts.values(), default=0)),
        "assignment_status_counts": dict(status_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1", choices=["ds1"])
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve", choices=["resolve", "match"])
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument(
        "--pair-feature-npz",
        action="append",
        default=[],
        help="extra verifier view as name:path; appends per-view cosine features to the pair model",
    )
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--train-top-k", type=int, default=30)
    ap.add_argument("--infer-top-k", type=int, default=30)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--max-component-size", type=int, default=120)
    ap.add_argument("--min-merge-support", type=int, default=1)
    ap.add_argument("--min-merge-support-ratio", type=float, default=0.0)
    ap.add_argument("--exclude-same", default="camera", choices=["camera", "stream", "video", "none"])
    ap.add_argument("--pseudo-theta", type=float, default=0.018)
    ap.add_argument("--pseudo-top-k", type=int, default=15)
    ap.add_argument("--pseudo-temporal-bonus", type=float, default=0.005)
    ap.add_argument("--pseudo-time-window-ms", type=int, default=1000)
    ap.add_argument("--pseudo-ensemble", action="store_true", help="train pseudo labels from agreement across multiple no-GT resolver settings")
    ap.add_argument("--pseudo-ensemble-thetas", default="", help="optional comma list overriding the default pseudo ensemble theta grid")
    ap.add_argument("--pseudo-ensemble-min-votes", type=int, default=2)
    ap.add_argument("--pseudo-ensemble-max-neg-votes", type=int, default=0)
    ap.add_argument("--pseudo-consensus-pos-min-votes", type=int, default=0)
    ap.add_argument("--pseudo-consensus-pos-min-sim", type=float, default=0.70)
    ap.add_argument("--pseudo-pos-min-sim", type=float, default=0.64)
    ap.add_argument("--pseudo-strong-pos-sim", type=float, default=0.78)
    ap.add_argument("--pseudo-neg-max-sim", type=float, default=0.42)
    ap.add_argument("--candidate-time-bonus", type=float, default=0.0)
    ap.add_argument("--affinity-time-bonus", type=float, default=0.005)
    ap.add_argument("--attach-threshold", type=float, default=0.72)
    ap.add_argument("--attach-margin", type=float, default=0.06)
    ap.add_argument("--attach-model-weight", type=float, default=0.65)
    ap.add_argument("--attach-max-source-size", type=int, default=2)
    ap.add_argument("--attach-min-target-size", type=int, default=2)
    ap.add_argument("--attach-top-k", type=int, default=60)
    ap.add_argument("--attach-min-edge-support", type=int, default=1)
    ap.add_argument("--attach-score-agg", default="max", choices=["max", "top_mean", "hybrid"])
    ap.add_argument("--attach-top-mean-k", type=int, default=3)
    ap.add_argument("--max-neg-per-pos", type=float, default=4.0)
    ap.add_argument("--random-negatives", type=int, default=60000)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf"])
    ap.add_argument("--load-model", default=None, help="optional joblib bundle from a prior run; skips pseudo-label training")
    ap.add_argument("--solver", default="agglom", choices=["agglom", "cc", "support", "consensus", "consensus_guard", "consensus_attach"])
    ap.add_argument("--thresholds", default="0.00,0.005,0.01,0.015,0.018,0.02,0.03,0.04,0.06")
    ap.add_argument("--blends", default="0.50,0.75,1.00")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--eval-min-matches", type=int, default=1)
    ap.add_argument("--eval-min-purity", type=float, default=0.0)
    ap.add_argument("--eval-cache", default=None)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-drop-area-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-area-quantile-by-video", default="")
    ap.add_argument("--output-drop-quality-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-quality-quantile-by-video", default="")
    ap.add_argument("--output-auto-anomaly-admission", action="store_true")
    ap.add_argument("--output-auto-anomaly-metric", default="quality", choices=["area", "quality", "both"])
    ap.add_argument("--output-auto-anomaly-quantile", type=float, default=0.75)
    ap.add_argument("--output-auto-anomaly-area-ratio", type=float, default=0.60)
    ap.add_argument("--output-auto-anomaly-quality-mad", type=float, default=1.0)
    ap.add_argument("--output-auto-anomaly-min-video-tracklets", type=int, default=20)
    ap.add_argument("--output-auto-anomaly-max-videos", type=int, default=3)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--full-top-n", type=int, default=5)
    ap.add_argument(
        "--full-selection-keys",
        default="",
        help="comma list of fast metrics used round-robin to choose configs for full scoring; supports pair_pr_min, pair_gate_margin, pair_mean_f1_pr",
    )
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--assignments-out", default=None, help="optional CSV of best-config seq -> predicted_global_id assignments")
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--random-state", type=int, default=17)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, emb = _load_tracklets(con, args.role)
    print(json.dumps({"stage": "loaded_tracklets", "n_tracklets": len(records), "emb_dim": int(emb.shape[1])}), flush=True)
    if args.feature_npz:
        emb = _load_feature_npz(
            args.feature_npz,
            records,
            emb,
            concat_db=bool(args.concat_db_embedding),
            db_weight=float(args.db_weight),
            feature_weight=float(args.feature_weight),
        )
        print(
            json.dumps(
                {
                    "stage": "loaded_feature_npz",
                    "feature_npz": args.feature_npz,
                    "concat_db_embedding": bool(args.concat_db_embedding),
                    "db_weight": float(args.db_weight),
                    "feature_weight": float(args.feature_weight),
                    "emb_dim": int(emb.shape[1]),
                }
            ),
            flush=True,
        )
    pair_feature_views, pair_feature_names, pair_feature_meta = _load_pair_feature_views(list(args.pair_feature_npz), records)
    global FEATURE_NAMES
    FEATURE_NAMES = list(BASE_FEATURE_NAMES) + list(pair_feature_names)
    if pair_feature_views:
        print(
            json.dumps(
                {
                    "stage": "loaded_pair_feature_views",
                    "pair_feature_views": pair_feature_meta,
                    "feature_names": FEATURE_NAMES,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    print(json.dumps({"stage": "loaded_predictions", "videos": len(pred_by_video), "rows": int(sum(len(v) for v in pred_by_video.values()))}), flush=True)
    cfg = _config_from_args(args)
    if args.load_model:
        bundle = joblib.load(args.load_model)
        model = bundle["model"]
        pseudo_info = dict(bundle.get("pseudo_training", {}))
        pseudo_info.update({"loaded_model": str(args.load_model), "uses_ground_truth": False})
        train_info = dict(bundle.get("model_training", {}))
        train_info.update({"loaded_model": str(args.load_model)})
        print(json.dumps({"stage": "loaded_pair_model", **train_info}, sort_keys=True), flush=True)
    else:
        online_gids = _load_online_gids(con)
        X, y, pseudo_info = _pseudo_training_pairs(records, emb, online_gids, cfg, pair_feature_views)
        print(json.dumps({"stage": "pseudo_training_pairs", **pseudo_info}, sort_keys=True), flush=True)
        model, train_info = _fit_model(X, y, cfg)
        print(json.dumps({"stage": "fit_pair_model", **train_info}, sort_keys=True), flush=True)

    gt_by_video = load_ds1_gt_by_video()
    gt_by_video = {key: value for key, value in gt_by_video.items() if key in pred_by_video}
    eval_cache_expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": float(args.iou_thr),
        "min_matches": int(args.eval_min_matches),
        "min_purity": float(args.eval_min_purity),
        "n_tracklets": int(len(records)),
        "prediction_rows": int(sum(len(v) for v in pred_by_video.values())),
        "gt_rows": int(sum(len(v) for v in gt_by_video.values())),
    }
    cached_eval = _load_eval_label_cache(args.eval_cache, eval_cache_expected) if args.eval_cache else None
    if cached_eval is not None:
        gt_by_seq, weight_by_seq, eval_stats = cached_eval
    else:
        gt_by_seq, weight_by_seq, eval_stats = _label_tracklets_for_eval(
            pred_by_video,
            gt_by_video,
            iou_thr=float(args.iou_thr),
            min_matches=int(args.eval_min_matches),
            min_purity=float(args.eval_min_purity),
        )
        eval_stats.update(eval_cache_expected)
        if args.eval_cache:
            _cache_eval_labels(args.eval_cache, gt_by_seq, weight_by_seq, eval_stats)
    print(json.dumps({"stage": "labeled_tracklets_for_eval", **eval_stats}), flush=True)
    output_keep_seqs, output_info = _output_keep_seqs(records, args)
    print(json.dumps({"stage": "output_admission", **output_info}, sort_keys=True), flush=True)

    seqs = [record.seq for record in records]
    rows: list[dict[str, object]] = []
    thresholds = _parse_float_list(args.thresholds)
    blends = _parse_float_list(args.blends)
    total = len(thresholds) * len(blends)
    progress = 0
    label_cache: dict[tuple[float, float], np.ndarray] = {}
    info_cache: dict[tuple[float, float], dict[str, object]] = {}
    runtime_cache: dict[str, object] = {}
    for blend in blends:
        for threshold in thresholds:
            progress += 1
            labels, info = _resolve_with_model(
                records,
                emb,
                model,
                cfg,
                threshold=threshold,
                blend=blend,
                solver=args.solver,
                runtime_cache=runtime_cache,
                pair_feature_views=pair_feature_views,
            )
            label_cache[(threshold, blend)] = labels
            info_cache[(threshold, blend)] = info
            pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=output_keep_seqs)
            metrics = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
            row = {
                "rank_input_order": progress,
                "mode": "pair_model",
                "threshold": float(threshold),
                "blend": float(blend),
                "solver": args.solver,
                **asdict(cfg),
                **info,
                **{key: value for key, value in output_info.items() if not isinstance(value, dict)},
                **metrics,
            }
            rows.append(row)
            print(json.dumps({"progress": progress, "total": total, "threshold": threshold, "blend": blend, **metrics}, sort_keys=True), flush=True)

    rows.sort(key=lambda row: float(row.get(args.sort_key, 0.0)), reverse=True)
    full_rows = _select_full_score_rows(rows, int(args.full_top_n), args.sort_key, args.full_selection_keys)
    for rank, row in enumerate(full_rows, start=1):
        key = (float(row["threshold"]), float(row["blend"]))
        labels = label_cache[key]
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, keep_seqs=output_keep_seqs))
        row.update({f"full_{name}": value for name, value in full.items() if name != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = rank
        print(
            json.dumps(
                {
                    "full_rank": rank,
                    "threshold": key[0],
                    "blend": key[1],
                    "full_selection_keys": args.full_selection_keys or args.sort_key,
                    "full": full,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    assignment_info = None
    if args.assignments_out and rows:
        best = rows[0]
        key = (float(best["threshold"]), float(best["blend"]))
        component_meta = _component_assignment_metadata(
            records,
            emb,
            model,
            cfg,
            label_cache[key],
            keep_seqs=output_keep_seqs,
            threshold=key[0],
            blend=key[1],
            pair_feature_views=pair_feature_views,
        )
        assignment_info = _write_assignments(
            args.assignments_out,
            records,
            label_cache[key],
            keep_seqs=output_keep_seqs,
            component_meta=component_meta,
        )

    result = {
        "dataset": args.dataset,
        "dbname": args.dbname,
        "role": args.role,
        "n_tracklets": len(records),
        "feature_npz": args.feature_npz,
        "pair_feature_views": pair_feature_meta,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "pair_model_config": asdict(cfg),
        "feature_names": FEATURE_NAMES,
        "pseudo_training": pseudo_info,
        "model_training": train_info,
        "eval_stats": eval_stats,
        "output_admission": output_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "sort_key": args.sort_key,
        "full_selection_keys": args.full_selection_keys or args.sort_key,
        "n_configs": len(rows),
        "top": rows[: max(int(args.full_top_n), 20)],
    }
    if args.load_model:
        result["load_model"] = str(args.load_model)
    if assignment_info is not None:
        result.update(assignment_info)

    if args.model_out:
        out = Path(args.model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_names": FEATURE_NAMES,
                "pair_feature_views": pair_feature_meta,
                "pair_model_config": asdict(cfg),
                "pseudo_training": pseudo_info,
                "model_training": train_info,
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
            },
            out,
        )
        result["model_out"] = str(out)
    if args.csv:
        _write_csv(args.csv, rows)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
