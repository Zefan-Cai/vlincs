#!/usr/bin/env python
"""Build a no-anchor fused tracklet-feature NPZ.

Each input NPZ must contain `seqs` and `features`.  The script aligns by seq,
L2-normalizes each source independently, applies the requested weight, then
concatenates the weighted blocks.  No ground-truth labels are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_source(text: str) -> tuple[str, Path, float]:
    parts = text.split(":")
    if len(parts) == 2:
        name, path = parts
        weight = 1.0
    elif len(parts) == 3:
        name, path, weight_text = parts
        weight = float(weight_text)
    else:
        raise ValueError(f"source must be name:path[:weight], got {text!r}")
    if not name:
        raise ValueError(f"source name is empty in {text!r}")
    return name, Path(path), weight


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "seqs" not in data or "features" not in data:
        raise ValueError(f"{path} must contain seqs and features arrays")
    seqs = np.asarray(data["seqs"], dtype=np.int64)
    features = np.asarray(data["features"], dtype=np.float32)
    if len(seqs) != len(features):
        raise ValueError(f"{path} has {len(seqs)} seqs but {len(features)} features")
    return seqs, features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help="feature source as name:path[:weight]; repeat for each source",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()

    specs = [_parse_source(value) for value in args.source]
    base_seqs: np.ndarray | None = None
    blocks: list[np.ndarray] = []
    meta_sources: list[dict[str, object]] = []

    for name, path, weight in specs:
        seqs, features = _load_npz(path)
        if base_seqs is None:
            base_seqs = seqs
            order = np.arange(len(seqs))
        else:
            by_seq = {int(seq): idx for idx, seq in enumerate(seqs)}
            missing = [int(seq) for seq in base_seqs if int(seq) not in by_seq]
            if missing:
                raise ValueError(f"{path} is missing {len(missing)} seqs; first={missing[0]}")
            order = np.asarray([by_seq[int(seq)] for seq in base_seqs], dtype=np.int64)
        aligned = _l2n(features[order].astype(np.float32)) * float(weight)
        blocks.append(aligned)
        meta_sources.append(
            {
                "name": name,
                "path": str(path),
                "weight": float(weight),
                "dim": int(features.shape[1]),
            }
        )

    if base_seqs is None:
        raise RuntimeError("no sources loaded")

    fused = _l2n(np.concatenate(blocks, axis=1).astype(np.float32))
    if args.dtype == "float16":
        fused = fused.astype(np.float16)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "no_anchor_fused_features",
        "sources": meta_sources,
        "n_tracklets": int(len(base_seqs)),
        "dim": int(fused.shape[1]),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
    }
    np.savez_compressed(out, seqs=base_seqs, features=fused, meta=json.dumps(meta, sort_keys=True))
    print(json.dumps({"output": str(out), **meta}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
