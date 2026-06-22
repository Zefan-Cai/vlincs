"""Training-free feature centralization utilities for ReID evidence.

The NFC routine follows the Pose2ID idea of using mutual nearest neighbours as
hidden positives, but keeps the implementation numpy-only so it can run inside
the public VLINCS kit without adding a torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


@dataclass(frozen=True)
class CentralizationInfo:
    n_items: int
    k1: int
    k2: int
    eta: float
    n_mutual_edges: int
    mean_mutuals_per_item: float


def neighbor_feature_centralization(
    features: np.ndarray,
    *,
    k1: int = 2,
    k2: int = 2,
    eta: float = 1.0,
    group_codes: np.ndarray | None = None,
    exclude_same_group: bool = False,
) -> tuple[np.ndarray, CentralizationInfo]:
    """Centralize features with mutual nearest neighbours.

    Args:
        features: ``(N, D)`` feature matrix.
        k1: Number of nearest neighbours to collect for each item.
        k2: Reciprocal-neighbour depth used to accept a neighbour.
        eta: Weight on the summed mutual-neighbour feature.
        group_codes: Optional source group per item, for example camera code.
        exclude_same_group: If true, same-group neighbours are not candidates.

    Returns:
        ``(centralized_features, info)``. Output features are L2-normalized.
    """

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {x.shape}")
    n = x.shape[0]
    if n <= 1 or int(k1) <= 0:
        out = _l2n(x.copy())
        return out, CentralizationInfo(n, max(int(k1), 0), max(int(k2), 0), float(eta), 0, 0.0)

    x = _l2n(x)
    k1 = min(max(int(k1), 1), n - 1)
    k2 = min(max(int(k2), 1), n - 1)
    sim = x @ x.T
    np.fill_diagonal(sim, -2.0)
    if exclude_same_group:
        if group_codes is None:
            raise ValueError("group_codes is required when exclude_same_group=True")
        group_codes = np.asarray(group_codes)
        sim[group_codes[:, None] == group_codes[None, :]] = -2.0

    k = max(k1, k2)
    rank = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    row_scores = np.take_along_axis(sim, rank, axis=1)
    order = np.argsort(-row_scores, axis=1)
    rank = np.take_along_axis(rank, order, axis=1)
    top1 = rank[:, :k1]
    top2_sets = [set(map(int, rank[i, :k2])) for i in range(n)]

    out = x.copy()
    n_mutual = 0
    counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        mutual = [int(j) for j in top1[i] if sim[i, int(j)] > -1.5 and i in top2_sets[int(j)]]
        if not mutual:
            continue
        out[i] = x[i] + float(eta) * np.sum(x[mutual], axis=0)
        counts[i] = len(mutual)
        n_mutual += len(mutual)
    out = _l2n(out.astype(np.float32))
    info = CentralizationInfo(
        n_items=n,
        k1=int(k1),
        k2=int(k2),
        eta=float(eta),
        n_mutual_edges=int(n_mutual),
        mean_mutuals_per_item=float(np.mean(counts)) if n else 0.0,
    )
    return out, info


__all__ = ["CentralizationInfo", "neighbor_feature_centralization"]
