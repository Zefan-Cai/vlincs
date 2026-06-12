"""Global-agglomerative resolve() for the online gallery — the training-free re-partition.

After the online match-or-expand pass has produced a (fragmented / greedy) partition, ``resolve()``
re-clusters ALL items from scratch over their pooled per-item embeddings. It does NOT merge
gid-centroids (that is :meth:`IdentityGallery.consolidate`); it throws away the greedy partition's
boundaries and re-partitions the raw points, so a greedy over-split AND a greedy over-merge are both
recoverable.

Mechanism:

  1. L2-normalize the per-item embeddings (osnet-xcam cosine space).
  2. Build a kNN-SPARSE cross-camera cosine affinity: for each item take its top-k (~30) nearest
     neighbours, EXCLUDING same-camera neighbours (same-camera links are within-tracker work the
     online pass already did; the resolve's job is the cross-camera identity join). Symmetrize.
  3. AVERAGE-linkage agglomerative clustering on the dense distance matrix ``D = 1 - affinity`` with
     ``distance_threshold = 1 - theta`` (sklearn ``metric="precomputed", linkage="average"``). A
     single GLOBAL theta. Non-neighbour pairs get affinity 0 (distance 1) so they only merge
     transitively through the linkage, never directly.
  4. (optional) cannot-link: forbid two items in the same (video, frame) whose boxes are spatially
     distinct from landing in one cluster — wired as a flag.
  5. Relabel the agglomerative clusters to contiguous integer gids.

The function is pure (numpy + sklearn) and dataset-agnostic; the gallery wires it as
``resolve_mode="global_agglom"`` via :meth:`IdentityGallery.resolve`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _l2n(E: np.ndarray) -> np.ndarray:
    return E / (np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9)


@dataclass
class ResolveResult:
    labels: np.ndarray          # per-item cluster label (contiguous 0..n_clusters-1)
    n_clusters: int
    theta: float
    n_cand_edges: int
    n_items: int


def build_knn_cosine_affinity(
    emb: np.ndarray,
    cam_codes: np.ndarray,
    *,
    top_k: int = 30,
    exclude_same_cam: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """kNN-sparse, symmetric cross-camera cosine affinity over L2-normed ``emb``.

    Returns (A, Ix, Jx): the dense ``A`` (n,n) affinity (diag 1, non-neighbour entries 0) plus the
    candidate edge endpoints (Ix, Jx) for diagnostics. ``cam_codes`` is an integer per-item camera
    label; with ``exclude_same_cam`` same-camera neighbours are dropped from the shortlist.
    """
    osn = _l2n(emb.astype(np.float32))
    K = osn.shape[0]
    S = osn @ osn.T
    if exclude_same_cam:
        same = cam_codes[:, None] == cam_codes[None, :]
        S[same] = -2.0
    np.fill_diagonal(S, -2.0)
    k = min(top_k, max(1, K - 1))
    topidx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    pair_set = set()
    for i in range(K):
        for j in topidx[i]:
            j = int(j)
            if S[i, j] <= -1.5:
                continue
            pair_set.add((i, j) if i < j else (j, i))
    if pair_set:
        pairs = np.array(sorted(pair_set), dtype=np.int64)
        Ix, Jx = pairs[:, 0], pairs[:, 1]
        cos_e = np.clip(np.einsum("ij,ij->i", osn[Ix], osn[Jx]), 0.0, 1.0).astype(np.float32)
    else:
        Ix = Jx = np.zeros((0,), np.int64)
        cos_e = np.zeros((0,), np.float32)
    A = np.zeros((K, K), np.float32)
    A[Ix, Jx] = cos_e
    A[Jx, Ix] = cos_e
    np.fill_diagonal(A, 1.0)
    return A, Ix, Jx


def global_agglom_resolve(
    emb: np.ndarray,
    cam_codes: np.ndarray,
    theta: float,
    *,
    top_k: int = 30,
    exclude_same_cam: bool = True,
    cannot_link_pairs: list[tuple[int, int]] | None = None,
) -> ResolveResult:
    """Re-partition ALL items by kNN-sparse cosine + average-linkage agglomerative at a global theta.

    ``theta`` is the merge affinity threshold (cosine); the agglomerative distance_threshold is
    ``1 - theta``. ``cannot_link_pairs`` (optional) is a list of (i, j) item pairs that must NOT
    share a cluster — enforced by setting their pairwise distance to +inf before linkage.
    """
    K = emb.shape[0]
    if K == 0:
        return ResolveResult(np.zeros((0,), np.int64), 0, theta, 0, 0)
    if K == 1:
        return ResolveResult(np.zeros((1,), np.int64), 1, theta, 0, 1)
    A, Ix, Jx = build_knn_cosine_affinity(
        emb, cam_codes, top_k=top_k, exclude_same_cam=exclude_same_cam
    )
    D = 1.0 - A
    np.clip(D, 0.0, None, out=D)
    if cannot_link_pairs:
        BIG = 1.0e6
        for i, j in cannot_link_pairs:
            D[i, j] = BIG
            D[j, i] = BIG
    lab = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(1.0 - theta),
        metric="precomputed",
        linkage="average",
    ).fit_predict(D)

    _, lab = np.unique(lab, return_inverse=True)
    return ResolveResult(
        labels=lab.astype(np.int64),
        n_clusters=int(lab.max()) + 1 if lab.size else 0,
        theta=float(theta),
        n_cand_edges=int(len(Ix)),
        n_items=int(K),
    )
