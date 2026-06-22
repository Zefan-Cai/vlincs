"""Weak-supervision graph resolver for tracklet-level global IDs.

This module is deliberately label-free: the input schema has no ground-truth or
reference identity field. It consumes tracklet evidence produced by the normal
pipeline - pooled visual embeddings, camera/time metadata, and weak language
tokens such as clothing or face-visibility attributes - then returns a forced
delivery ID plus auditable candidate/accepted edges.

The intent is to provide an M2/M4/M7/M8 bridge that can be mounted under the
existing gallery pipeline:

* M2 evidence: visual embedding + weak tokens.
* M4 constraints: same-camera temporal overlap cannot-link.
* M7 retrieval: recall-oriented visual/token/trajectory candidate union.
* M8 delivery: weak component IDs marked ``forced`` unless a later calibrated
  resolver explicitly commits them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Iterable, Mapping, Sequence

import numpy as np


WeakTokens = Mapping[str, object] | Iterable[object] | str | None


@dataclass(frozen=True)
class TrackletEvidence:
    """Label-free evidence for one tracklet.

    ``weak_tokens`` may be a dict (``{"upper_color": "black"}``), a pipe-delimited
    string (``"upper_color:black|hat:no"``), or an iterable of token strings.
    """

    tracklet_key: str
    embedding: Sequence[float] | np.ndarray
    video: str = ""
    camera: str = ""
    start_frame: int = 0
    end_frame: int = 0
    weak_tokens: WeakTokens = None


@dataclass(frozen=True)
class WeakGraphConfig:
    """Knobs for no-GT weak graph construction and delivery."""

    visual_top_k: int = 30
    visual_candidate_threshold: float = 0.45
    visual_strong_threshold: float = 0.72
    weak_token_threshold: float = 0.24
    temporal_threshold: float = 0.72
    edge_threshold: float = 0.64
    visual_weight: float = 0.60
    weak_weight: float = 0.32
    trajectory_weight: float = 0.08
    min_common_tokens: int = 1
    max_token_df: int | None = 120
    temporal_window_frames: int = 300
    temporal_scale_frames: float = 120.0
    allow_same_camera_overlap: bool = False
    max_component_size: int = 80
    lp_iterations: int = 8
    lp_size_penalty_alpha: float = 0.25
    self_label_weight: float = 0.05
    forced_prefix: str = "W"
    force_output: bool = True


@dataclass(frozen=True)
class CandidateEdge:
    left: int
    right: int
    score: float
    visual: float
    weak_token: float
    trajectory: float
    common_tokens: int
    reasons: tuple[str, ...]
    accepted: bool
    cannot_link: bool = False


@dataclass(frozen=True)
class WeakGraphAssignment:
    tracklet_key: str
    predicted_global_id: str
    decision_status: str
    confidence: float
    component_size: int
    weak_tokens: tuple[str, ...]


@dataclass(frozen=True)
class WeakGraphResult:
    assignments: list[WeakGraphAssignment]
    candidate_edges: list[CandidateEdge]
    accepted_edges: list[CandidateEdge]
    labels: np.ndarray
    summary: dict[str, object] = field(default_factory=dict)


def _tokenize(tokens: WeakTokens) -> tuple[str, ...]:
    if tokens is None:
        return ()
    if isinstance(tokens, str):
        raw = [part.strip() for part in tokens.replace(",", "|").split("|")]
    elif isinstance(tokens, Mapping):
        raw = []
        for key, value in tokens.items():
            if value is None:
                continue
            text = str(value).strip().lower()
            if not text or text in {"unknown", "none", "nan"}:
                continue
            raw.append(f"{str(key).strip().lower()}:{text}")
    else:
        raw = [str(part).strip() for part in tokens]
    clean = []
    seen = set()
    for token in raw:
        token = token.strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        clean.append(token)
    return tuple(clean)


def _l2n_matrix(vectors: Sequence[Sequence[float] | np.ndarray]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be a 2-D array, got shape {arr.shape}")
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return int(a0) <= int(b1) and int(b0) <= int(a1)


def _same_camera_overlap(a: TrackletEvidence, b: TrackletEvidence) -> bool:
    return bool(a.video and a.video == b.video and a.camera and a.camera == b.camera and _interval_overlap(a.start_frame, a.end_frame, b.start_frame, b.end_frame))


def _trajectory_similarity(a: TrackletEvidence, b: TrackletEvidence, cfg: WeakGraphConfig) -> float:
    if not (a.video and a.video == b.video and a.camera and a.camera == b.camera):
        return 0.0
    if _interval_overlap(a.start_frame, a.end_frame, b.start_frame, b.end_frame):
        return 1.0
    gap = max(0, int(b.start_frame) - int(a.end_frame), int(a.start_frame) - int(b.end_frame))
    if gap > cfg.temporal_window_frames:
        return 0.0
    return float(exp(-gap / max(float(cfg.temporal_scale_frames), 1.0)))


def _weak_similarity(left: set[str], right: set[str]) -> tuple[float, int]:
    if not left or not right:
        return 0.0, 0
    common = left & right
    if not common:
        return 0.0, 0
    union = left | right
    return float(len(common) / max(len(union), 1)), len(common)


def _edge_score(visual: float, weak: float, trajectory: float, cfg: WeakGraphConfig) -> float:
    return float(
        cfg.visual_weight * max(visual, 0.0)
        + cfg.weak_weight * max(weak, 0.0)
        + cfg.trajectory_weight * max(trajectory, 0.0)
    )


def build_candidate_edges(records: Sequence[TrackletEvidence], cfg: WeakGraphConfig) -> tuple[list[CandidateEdge], list[tuple[str, ...]]]:
    """Build M7 candidate edges without reading identity labels."""

    n = len(records)
    if n == 0:
        return [], []

    embs = _l2n_matrix([record.embedding for record in records])
    sim = embs @ embs.T
    token_rows = [_tokenize(record.weak_tokens) for record in records]

    token_df: dict[str, int] = {}
    for row in token_rows:
        for token in row:
            token_df[token] = token_df.get(token, 0) + 1
    if cfg.max_token_df is not None:
        token_sets = [set(t for t in row if token_df.get(t, 0) <= cfg.max_token_df) for row in token_rows]
    else:
        token_sets = [set(row) for row in token_rows]

    pair_reasons: dict[tuple[int, int], set[str]] = {}

    # Visual retrieval: per-row top-k shortlist, recall-oriented union.
    k = min(max(int(cfg.visual_top_k), 1), max(n - 1, 1))
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        top = np.argpartition(-row, k - 1)[:k] if n > 1 else []
        for j in top:
            j = int(j)
            if row[j] < cfg.visual_candidate_threshold:
                continue
            pair = (i, j) if i < j else (j, i)
            pair_reasons.setdefault(pair, set()).add("visual")

    # Weak-token retrieval: all pairs sharing at least min_common_tokens filtered tokens.
    token_to_indices: dict[str, list[int]] = {}
    for i, toks in enumerate(token_sets):
        for token in toks:
            token_to_indices.setdefault(token, []).append(i)
    for indices in token_to_indices.values():
        if len(indices) < 2:
            continue
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                pair = (i, j) if i < j else (j, i)
                pair_reasons.setdefault(pair, set()).add("weak_tokens")

    # Local trajectory retrieval: useful for adjacent fragments in the same stream.
    # Enumerate only same-video/camera neighbours whose frame ranges fall inside the
    # temporal window; a full all-pairs scan is wasteful at DS1 scale.
    stream_to_indices: dict[tuple[str, str], list[int]] = {}
    for i, record in enumerate(records):
        if record.video and record.camera:
            stream_to_indices.setdefault((record.video, record.camera), []).append(i)
    for indices in stream_to_indices.values():
        ordered = sorted(indices, key=lambda idx: (int(records[idx].start_frame), int(records[idx].end_frame), idx))
        for pos, i in enumerate(ordered):
            right_limit = int(records[i].end_frame) + int(cfg.temporal_window_frames)
            for j in ordered[pos + 1 :]:
                if int(records[j].start_frame) > right_limit:
                    break
                traj = _trajectory_similarity(records[i], records[j], cfg)
                if traj >= cfg.temporal_threshold:
                    pair = (i, j) if i < j else (j, i)
                    pair_reasons.setdefault(pair, set()).add("trajectory")

    edges: list[CandidateEdge] = []
    for (i, j), reasons in sorted(pair_reasons.items()):
        left_tokens, right_tokens = token_sets[i], token_sets[j]
        weak, common = _weak_similarity(left_tokens, right_tokens)
        if "weak_tokens" in reasons and common < cfg.min_common_tokens:
            continue
        visual = float(sim[i, j])
        trajectory = _trajectory_similarity(records[i], records[j], cfg)
        cannot_link = (not cfg.allow_same_camera_overlap) and _same_camera_overlap(records[i], records[j])
        score = _edge_score(visual, weak, trajectory, cfg)
        edge_reasons: list[str] = []
        if visual >= cfg.visual_strong_threshold:
            edge_reasons.append("visual_strong")
        elif "visual" in reasons:
            edge_reasons.append("visual_candidate")
        if weak >= cfg.weak_token_threshold:
            edge_reasons.append("weak_tokens")
        if trajectory >= cfg.temporal_threshold:
            edge_reasons.append("trajectory")
        if cannot_link:
            edge_reasons.append("cannot_link")
        accepted = (not cannot_link) and score >= cfg.edge_threshold and bool(edge_reasons)
        edges.append(
            CandidateEdge(
                left=i,
                right=j,
                score=score,
                visual=visual,
                weak_token=weak,
                trajectory=trajectory,
                common_tokens=common,
                reasons=tuple(edge_reasons),
                accepted=accepted,
                cannot_link=cannot_link,
            )
        )
    edges.sort(key=lambda edge: (-edge.score, edge.left, edge.right))
    return edges, [tuple(sorted(tokens)) for tokens in token_sets]


def _label_components(labels: np.ndarray) -> dict[int, set[int]]:
    components: dict[int, set[int]] = {}
    for index, label in enumerate(labels.tolist()):
        components.setdefault(int(label), set()).add(index)
    return components


def _renumber_labels(labels: np.ndarray) -> np.ndarray:
    first_seen: dict[int, int] = {}
    out = np.zeros_like(labels, dtype=np.int64)
    for i, label in enumerate(labels.tolist()):
        if int(label) not in first_seen:
            first_seen[int(label)] = len(first_seen)
        out[i] = first_seen[int(label)]
    return out


def weighted_label_propagation(
    n_nodes: int,
    accepted_edges: Sequence[CandidateEdge],
    cannot_link_pairs: set[tuple[int, int]],
    cfg: WeakGraphConfig,
) -> np.ndarray:
    """Small deterministic M8-style graph solve over M7 accepted edges."""

    labels = np.arange(n_nodes, dtype=np.int64)
    if n_nodes <= 1 or not accepted_edges:
        return labels

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for edge in accepted_edges:
        adjacency[edge.left].append((edge.right, edge.score))
        adjacency[edge.right].append((edge.left, edge.score))

    for _ in range(max(int(cfg.lp_iterations), 1)):
        changed = False
        components = _label_components(labels)
        next_labels = labels.copy()
        for i in range(n_nodes):
            current = int(labels[i])
            scores: dict[int, float] = {current: float(cfg.self_label_weight)}
            for j, weight in adjacency[i]:
                label = int(labels[j])
                size = max(len(components.get(label, ())), 1)
                scores[label] = scores.get(label, 0.0) + float(weight) / (size ** float(cfg.lp_size_penalty_alpha))
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            chosen = current
            for label, _score in ranked:
                members = components.get(label, set())
                if label != current and len(members) >= cfg.max_component_size:
                    continue
                if any((min(i, member), max(i, member)) in cannot_link_pairs for member in members):
                    continue
                chosen = label
                break
            if chosen != current:
                next_labels[i] = chosen
                changed = True
        labels = next_labels
        if not changed:
            break
    return _renumber_labels(labels)


def resolve_weak_graph(records: Sequence[TrackletEvidence], cfg: WeakGraphConfig | None = None) -> WeakGraphResult:
    """Resolve tracklets into weak component IDs without ground-truth labels."""

    cfg = cfg or WeakGraphConfig()
    records = list(records)
    n = len(records)
    edges, token_rows = build_candidate_edges(records, cfg)
    accepted = [edge for edge in edges if edge.accepted]
    cannot_link_pairs = {
        (min(edge.left, edge.right), max(edge.left, edge.right))
        for edge in edges
        if edge.cannot_link
    }
    labels = weighted_label_propagation(n, accepted, cannot_link_pairs, cfg)
    components = _label_components(labels)
    label_to_gid = {
        label: f"{cfg.forced_prefix}{index + 1:04d}"
        for index, label in enumerate(sorted(components, key=lambda lab: min(components[lab])))
    }

    incident_total = np.zeros(n, dtype=np.float32)
    incident_same = np.zeros(n, dtype=np.float32)
    incident_count = np.zeros(n, dtype=np.int32)
    for edge in accepted:
        incident_total[edge.left] += edge.score
        incident_total[edge.right] += edge.score
        incident_count[edge.left] += 1
        incident_count[edge.right] += 1
        if labels[edge.left] == labels[edge.right]:
            incident_same[edge.left] += edge.score
            incident_same[edge.right] += edge.score

    assignments: list[WeakGraphAssignment] = []
    for i, record in enumerate(records):
        label = int(labels[i])
        if incident_count[i] == 0:
            confidence = 0.0
        elif incident_total[i] > 0:
            confidence = float(incident_same[i] / incident_total[i])
        else:
            confidence = 0.0
        status = "forced" if cfg.force_output else "provisional"
        assignments.append(
            WeakGraphAssignment(
                tracklet_key=record.tracklet_key,
                predicted_global_id=label_to_gid[label],
                decision_status=status,
                confidence=round(confidence, 6),
                component_size=len(components[label]),
                weak_tokens=token_rows[i],
            )
        )

    summary = {
        "tracklets": n,
        "candidate_edges": len(edges),
        "accepted_edges": len(accepted),
        "cannot_link_edges": len(cannot_link_pairs),
        "components": len(components),
        "largest_component": max((len(members) for members in components.values()), default=0),
        "decision_status": "forced" if cfg.force_output else "provisional",
        "uses_ground_truth": False,
    }
    return WeakGraphResult(
        assignments=assignments,
        candidate_edges=edges,
        accepted_edges=accepted,
        labels=labels,
        summary=summary,
    )


__all__ = [
    "CandidateEdge",
    "TrackletEvidence",
    "WeakGraphAssignment",
    "WeakGraphConfig",
    "WeakGraphResult",
    "build_candidate_edges",
    "resolve_weak_graph",
    "weighted_label_propagation",
]
