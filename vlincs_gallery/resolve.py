"""Global-agglomerative resolve() for the online gallery - the training-free re-partition.

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
     distinct from landing in one cluster - wired as a flag.
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


def _knn_sparse_affinity(S: np.ndarray, top_k: int) -> tuple[np.ndarray, int]:
    """kNN-sparsify a precomputed similarity matrix ``S`` into a dense symmetric affinity ``A``.

    ``S`` must already have forbidden entries (same-group, self/diagonal) masked to <= -1.5. Keeps each
    row's top-k neighbours, symmetrizes, clips to [0, 1], and sets the diagonal to 1. Returns
    ``(A, n_cand_edges)``. This is the S->A step shared by the per-item (centroid) and per-bank resolves.
    """
    K = S.shape[0]
    k = min(top_k, max(1, K - 1))
    topidx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    pair_set = set()
    for i in range(K):
        for j in topidx[i]:
            j = int(j)
            if S[i, j] <= -1.5:
                continue
            pair_set.add((i, j) if i < j else (j, i))
    A = np.zeros((K, K), np.float32)
    if pair_set:
        pairs = np.array(sorted(pair_set), dtype=np.int64)
        Ix, Jx = pairs[:, 0], pairs[:, 1]
        a = np.clip(S[Ix, Jx], 0.0, 1.0).astype(np.float32)
        A[Ix, Jx] = a
        A[Jx, Ix] = a
    np.fill_diagonal(A, 1.0)
    return A, len(pair_set)


def _bank_max_cosine(banks: list[np.ndarray], group_codes: np.ndarray) -> np.ndarray:
    """``L x L`` cross-bank MAX cosine: ``S[i, j] = max`` over (a in bank_i, b in bank_j) of cos(a, b).

    Same-group pairs (``group_codes[i] == group_codes[j]``) and the diagonal are masked to -2.0
    (forbidden / self). Vectorized via two segment-max (``np.maximum.reduceat``) passes over the full
    exemplar Gram matrix, so it is O(T^2) in the total exemplar count T = sum of bank sizes, not the
    O(L^2 * bank^2) of a naive per-pair loop.
    """
    normed = [_l2n(b.astype(np.float32)) for b in banks]
    sizes = [len(b) for b in normed]
    E = np.concatenate(normed, axis=0)                       # (T, D)
    offsets = np.cumsum([0] + sizes)[:-1]                    # per-local segment starts
    G = E @ E.T                                              # (T, T) exemplar Gram
    R = np.maximum.reduceat(G, offsets, axis=0)             # (L, T) max over each local's rows
    S = np.maximum.reduceat(R, offsets, axis=1).astype(np.float32)   # (L, L) then over its cols
    same = group_codes[:, None] == group_codes[None, :]
    S[same] = -2.0
    np.fill_diagonal(S, -2.0)
    return S


def two_tier_resolve(
    banks: list[np.ndarray],
    video_codes: np.ndarray,
    theta: float,
    *,
    mode: str = "bank",
    top_k: int = 30,
    link: str = "average",
) -> ResolveResult:
    """Cross-video agglomerative resolve over PER-LOCAL exemplar banks - the two-tier global step.

    Each local identity (the output of a per-video gallery, after within-video consolidation) carries
    its diversity-gated exemplar bank. Locals from the SAME video are forbidden to merge: within one
    camera they are already-distinct people. The clustering re-partitions the locals across videos.

    Args:
        banks: one ``(n_i, D)`` exemplar-bank array per local identity.
        video_codes: ``(L,)`` int source-video code per local (same-code pairs cannot merge).
        theta: merge-affinity (cosine) threshold; agglomerative ``distance_threshold = 1 - theta``.
        mode: ``"bank"`` -> local<->local affinity is the MAX cosine over exemplar pairs (multi-modal
            appearance preserved - the lever that avoids the single-centroid blur that cost ~0.03 IDF1
            in the offline path-C test). ``"centroid"`` -> reduce each bank to its L2-normed mean, then
            the standard per-item kNN cosine resolve.
        top_k: kNN shortlist size per local.
        link: agglomerative linkage (``"average"`` default, ``"complete"`` for a stricter join).

    Returns:
        ResolveResult: ``labels`` are per-LOCAL global cluster ids (0..n_clusters-1).
    """
    L = len(banks)
    if L == 0:
        return ResolveResult(np.zeros((0,), np.int64), 0, theta, 0, 0)
    if L == 1:
        return ResolveResult(np.zeros((1,), np.int64), 1, theta, 0, 1)
    video_codes = np.asarray(video_codes)
    if mode == "centroid":
        cents = np.stack([_l2n(b.astype(np.float32).mean(0)) for b in banks])
        return global_agglom_resolve(cents, video_codes, theta, top_k=top_k, exclude_same_cam=True)
    if mode != "bank":
        raise ValueError(f"unknown mode: {mode!r} (expected 'bank' or 'centroid')")
    S = _bank_max_cosine(banks, video_codes)
    A, n_edges = _knn_sparse_affinity(S, top_k)
    D = 1.0 - A
    np.clip(D, 0.0, None, out=D)
    lab = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(1.0 - theta),
        metric="precomputed",
        linkage=link,
    ).fit_predict(D)
    _, lab = np.unique(lab, return_inverse=True)
    return ResolveResult(
        labels=lab.astype(np.int64),
        n_clusters=int(lab.max()) + 1 if lab.size else 0,
        theta=float(theta),
        n_cand_edges=int(n_edges),
        n_items=int(L),
    )


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
    share a cluster - enforced by setting their pairwise distance to +inf before linkage.
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


def mustlink_resolve(
    emb: np.ndarray,
    cam_codes: np.ndarray,
    local_ids: np.ndarray,
    theta: float,
    *,
    top_k: int = 30,
) -> ResolveResult:
    """Global resolve over ALL raw items (tracklets, NOT summarized banks) with per-video MUST-LINK.

    The two-tier alternative to summarizing each per-video local into one bank vector: keep every tracklet
    as its own node so the cross-video linkage has the full redundant kNN edge set (the thing that makes
    the flat resolve robust), but FORCE tracklets sharing a ``local_id`` into the same cluster. Concretely:
    build the standard kNN-sparse cross-camera cosine affinity, then set every same-local pair's affinity
    to 1 (distance 0) so average-linkage merges those blocks first; cross-video merges still come from the
    kNN edges at ``distance_threshold = 1 - theta``. Recovers the per-video grouping (the DS2 lever) without
    the bank-summarization information loss that capped two_tier_resolve.

    Args:
        emb: ``(K, D)`` per-item (per-tracklet) embeddings in the resolve space.
        cam_codes: ``(K,)`` int camera/video code per item (same-cam kNN neighbours excluded).
        local_ids: ``(K,)`` int per-video local-identity id per item (same id => must-link).
        theta: cross-video merge threshold; ``distance_threshold = 1 - theta``.
        top_k: kNN shortlist size.

    Returns:
        ResolveResult: ``labels`` are per-ITEM global cluster ids.
    """
    K = emb.shape[0]
    if K == 0:
        return ResolveResult(np.zeros((0,), np.int64), 0, theta, 0, 0)
    if K == 1:
        return ResolveResult(np.zeros((1,), np.int64), 1, theta, 0, 1)
    A, Ix, Jx = build_knn_cosine_affinity(emb, cam_codes, top_k=top_k, exclude_same_cam=True)
    local_ids = np.asarray(local_ids)
    for lg in np.unique(local_ids):                         # must-link: same local -> affinity 1 (dist 0)
        idx = np.where(local_ids == lg)[0]
        if idx.size > 1:
            A[np.ix_(idx, idx)] = 1.0
    np.fill_diagonal(A, 1.0)
    D = 1.0 - A
    np.clip(D, 0.0, None, out=D)
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


def estimate_theta_by_violations(
    resolve_fn,
    theta_grid,
    cannot_link_pairs,
    *,
    budget: float = 0.01,
    rel_margin: float = 0.35,
):
    """GT-FREE op-point picker for the cross-video resolve.

    theta is not a universal constant - its scale depends on the embedding (DS1 osnet-xcam peaks ~0.04,
    MS02 SOLIDER-PCA64 ~0.78), so it must be calibrated per embedding. This does it with NO ground truth,
    using the one hard physical fact we always have: two tracklets co-visible in the same frame with
    distinct boxes are DEFINITELY different people (a ``cannot_link`` pair). A clustering that puts such a
    pair in one cluster commits a definite error. Lower theta merges more -> more violations; we pick the
    MOST-merging theta whose violation rate stays within tolerance (the TA1 asymmetry: a wrong merge
    costs far more than leaving an identity fragmented, so stay conservative).

    The tolerance is RELATIVE to the curve's own violation FLOOR (the min over the grid - the irreducible
    co-visibility merges this embedding always makes), not a fixed absolute number: ``thresh = max(budget,
    floor * (1 + rel_margin))``. The floor differs per embedding (DS1 ~0.012, others differ), so an absolute
    budget that's too tight for one dataset over-conservatively falls back to the no-merge end; the
    relative margin self-calibrates. On DS1 this lands on theta=0.04 (the IDF1/AssA peak).

    Args:
        resolve_fn: callable(theta) -> ResolveResult (e.g. ``lambda t: mustlink_resolve(emb, cc, local, t)``).
            Only clusters; does NOT score - so the sweep is cheap (no GT, no reid_hota).
        theta_grid: thetas to try.
        cannot_link_pairs: iterable of (i, j) item-index pairs known to be different entities.
        budget: absolute floor on the tolerance (a hard min so a near-zero violation floor doesn't pin theta).
        rel_margin: tolerance above the violation floor, as a fraction of the floor.

    Returns:
        (best_theta, curve) where curve = list of (theta, n_clusters, violation_rate), theta-ascending.
    """
    pairs = np.array(sorted({(int(a), int(b)) for a, b in cannot_link_pairs}), dtype=np.int64) \
        if cannot_link_pairs else np.zeros((0, 2), np.int64)
    curve = []
    for theta in sorted(theta_grid):              # ascending: fewer merges (more clusters) as theta rises
        res = resolve_fn(float(theta))
        if len(pairs):
            lab = res.labels
            vrate = float((lab[pairs[:, 0]] == lab[pairs[:, 1]]).mean())
        else:
            vrate = 0.0
        curve.append((float(theta), int(res.n_clusters), vrate))
    floor = min((c[2] for c in curve), default=0.0)
    thresh = max(budget, floor * (1.0 + rel_margin))
    ok = [c for c in curve if c[2] <= thresh]
    best = min(ok, key=lambda c: c[0])[0] if ok else max(curve, key=lambda c: c[0])[0]
    return best, curve
