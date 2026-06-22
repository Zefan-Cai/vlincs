#!/usr/bin/env python
"""Merge partial tracklet feature shards with an optional fallback NPZ.

Each input NPZ must contain `seqs` and `features`.  Later shards overwrite
earlier ones for the same seq.  When `--fallback` is provided, its full feature
matrix supplies every seq not present in a shard.  This is useful for
incremental no-anchor feature ablations, e.g. start from PersonViT sample=1 and
replace completed shards with PersonViT sample=3 as they finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    if "seqs" not in data or "features" not in data:
        raise ValueError(f"{path} must contain seqs and features")
    seqs = np.asarray(data["seqs"], dtype=np.int64)
    feats = np.asarray(data["features"], dtype=np.float32)
    if len(seqs) != len(feats):
        raise ValueError(f"{path} has {len(seqs)} seqs but {len(feats)} features")
    if feats.ndim != 2:
        raise ValueError(f"{path} features must be rank-2, got shape {feats.shape}")
    valid = None
    if "valid" in data:
        valid = np.asarray(data["valid"]).astype(bool)
        if len(valid) != len(seqs):
            raise ValueError(f"{path} valid has {len(valid)} rows but {len(seqs)} seqs")
    return seqs, feats, valid


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fallback", default=None, help="full NPZ used for missing seqs")
    ap.add_argument("--shard", action="append", default=[], help="partial NPZ; repeat, later wins")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = ap.parse_args()

    if not args.fallback and not args.shard:
        raise ValueError("provide --fallback and/or at least one --shard")

    features_by_seq: dict[int, np.ndarray] = {}
    valid_by_seq: dict[int, bool] = {}
    source_by_seq: dict[int, str] = {}
    dim: int | None = None
    fallback_count = 0
    saw_valid = False

    if args.fallback:
        seqs, feats, valid = _load(Path(args.fallback))
        saw_valid = saw_valid or valid is not None
        dim = int(feats.shape[1])
        for idx, (seq, feat) in enumerate(zip(seqs, feats)):
            features_by_seq[int(seq)] = np.asarray(feat, dtype=np.float32)
            if valid is not None:
                valid_by_seq[int(seq)] = bool(valid[idx])
            source_by_seq[int(seq)] = "fallback"
        fallback_count = int(len(seqs))

    shard_info = []
    for shard in args.shard:
        path = Path(shard)
        seqs, feats, valid = _load(path)
        saw_valid = saw_valid or valid is not None
        if dim is None:
            dim = int(feats.shape[1])
        if int(feats.shape[1]) != dim:
            raise ValueError(f"{path} dim {feats.shape[1]} != expected {dim}")
        for idx, (seq, feat) in enumerate(zip(seqs, feats)):
            features_by_seq[int(seq)] = np.asarray(feat, dtype=np.float32)
            if valid is not None:
                valid_by_seq[int(seq)] = bool(valid[idx])
            source_by_seq[int(seq)] = str(path)
        shard_item = {"path": str(path), "seqs": int(len(seqs)), "dim": int(feats.shape[1])}
        if valid is not None:
            shard_item["valid"] = int(valid.sum())
        shard_info.append(shard_item)

    out_seqs = np.asarray(sorted(features_by_seq), dtype=np.int64)
    out_feats = np.stack([features_by_seq[int(seq)] for seq in out_seqs]).astype(np.float32)
    out_feats = _l2n(out_feats)
    if args.dtype == "float16":
        out_feats = out_feats.astype(np.float16)

    replaced = sum(1 for src in source_by_seq.values() if src != "fallback")
    meta = {
        "kind": "merged_tracklet_feature_shards",
        "fallback": args.fallback,
        "fallback_count": fallback_count,
        "shards": shard_info,
        "n_tracklets": int(len(out_seqs)),
        "dim": int(out_feats.shape[1]) if out_feats.ndim == 2 else 0,
        "replaced_from_shards": int(replaced),
        "valid_tracklets": int(sum(valid_by_seq.get(int(seq), np.linalg.norm(features_by_seq[int(seq)]) > 1e-8) for seq in out_seqs)) if saw_valid else None,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seqs": out_seqs, "features": out_feats, "meta": json.dumps(meta, sort_keys=True)}
    if saw_valid:
        payload["valid"] = np.asarray(
            [valid_by_seq.get(int(seq), np.linalg.norm(features_by_seq[int(seq)]) > 1e-8) for seq in out_seqs],
            dtype=bool,
        )
    np.savez_compressed(out, **payload)
    print(json.dumps({"output": str(out), **meta}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
