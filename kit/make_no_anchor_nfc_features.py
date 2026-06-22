#!/usr/bin/env python
"""Apply no-anchor Neighbor Feature Centralization to tracklet features.

This is a lightweight adaptation of the Pose2ID/NFC idea for VLINCS tracklet
features.  It uses only feature-space nearest neighbors, never identity labels
or anchors, then writes a normal ``seqs``/``features`` npz that existing
no-anchor resolvers can consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _l2n(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _topk_neighbors(x: np.ndarray, top_k: int, chunk_size: int) -> np.ndarray:
    n = int(x.shape[0])
    k = max(0, min(int(top_k), max(n - 1, 0)))
    if k == 0:
        return np.zeros((n, 0), dtype=np.int64)
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        sim = x[start:stop] @ x.T
        rows = np.arange(stop - start)
        sim[rows, np.arange(start, stop)] = -np.inf
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        val = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-val, axis=1)
        out[start:stop] = np.take_along_axis(idx, order, axis=1)
    return out


def _centralize_once(
    x: np.ndarray,
    neighbors: np.ndarray,
    *,
    k1: int,
    k2: int,
    eta: float,
    include_self: bool,
) -> np.ndarray:
    n = int(x.shape[0])
    out = np.empty_like(x, dtype=np.float32)
    k1 = max(0, min(int(k1), neighbors.shape[1]))
    k2 = max(0, min(int(k2), neighbors.shape[1]))
    for i in range(n):
        pool: list[int] = []
        if include_self:
            pool.append(i)
        first = neighbors[i, :k1].tolist()
        pool.extend(int(v) for v in first)
        if k2 > 0:
            for nbr in first:
                pool.extend(int(v) for v in neighbors[int(nbr), :k2].tolist())
        if not pool:
            out[i] = x[i]
            continue
        unique = np.asarray(sorted(set(pool)), dtype=np.int64)
        center = x[unique].mean(axis=0)
        out[i] = (1.0 - float(eta)) * x[i] + float(eta) * center
    return _l2n(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="input npz with seqs/features")
    ap.add_argument("--output", required=True, help="output centralized npz")
    ap.add_argument("--k1", type=int, default=2, help="first-order neighbors")
    ap.add_argument("--k2", type=int, default=2, help="neighbors of first-order neighbors")
    ap.add_argument("--eta", type=float, default=0.5, help="centralization strength")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--include-self", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--topk-buffer", type=int, default=0, help="override cached neighbor width")
    args = ap.parse_args()

    src = Path(args.input)
    data = np.load(src, allow_pickle=True)
    if "features" not in data or "seqs" not in data:
        raise ValueError(f"{src} must contain seqs and features")
    features = _l2n(data["features"].astype(np.float32))
    max_k = max(int(args.k1), int(args.k2), int(args.topk_buffer))
    neighbors = _topk_neighbors(features, max_k, int(args.chunk_size))
    out = features
    for _ in range(max(int(args.iterations), 1)):
        out = _centralize_once(
            out,
            neighbors,
            k1=int(args.k1),
            k2=int(args.k2),
            eta=float(args.eta),
            include_self=bool(args.include_self),
        )

    payload = {key: data[key] for key in data.files if key != "features"}
    payload["features"] = out.astype(np.float32)
    payload["nfc_info"] = np.asarray(
        [
            json.dumps(
                {
                    "source": str(src),
                    "k1": int(args.k1),
                    "k2": int(args.k2),
                    "eta": float(args.eta),
                    "iterations": int(args.iterations),
                    "include_self": bool(args.include_self),
                    "uses_anchors": False,
                    "uses_gt": False,
                    "method": "no_anchor_neighbor_feature_centralization",
                },
                sort_keys=True,
            )
        ],
        dtype=object,
    )
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)
    print(
        json.dumps(
            {
                "input": str(src),
                "output": str(dst),
                "tracklets": int(out.shape[0]),
                "dim": int(out.shape[1]) if out.ndim == 2 else 0,
                "k1": int(args.k1),
                "k2": int(args.k2),
                "eta": float(args.eta),
                "iterations": int(args.iterations),
                "include_self": bool(args.include_self),
                "uses_anchors": False,
                "uses_gt": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
