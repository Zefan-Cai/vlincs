#!/usr/bin/env python
"""Export sample prototype tensors as pair-feature-view NPZ files.

``no_anchor_sample_parquet_sweep.py`` consumes pair feature views in the same
format as the full DS1 model: each file contains ``seqs`` and 2D ``features``.
The sample extractor can now save 3D prototype tensors, so this helper converts
each prototype slot into one pair-feature view without using GT labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from kit.no_anchor_sample_parquet_sweep import _build_records, _load_sample_parquets
except ModuleNotFoundError:
    from no_anchor_sample_parquet_sweep import _build_records, _load_sample_parquets


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)).astype(np.float32)


def _load_metadata(data: np.lib.npyio.NpzFile, path: Path) -> dict[str, object]:
    if "metadata" not in data.files:
        raise ValueError(f"{path} is missing metadata")
    return json.loads(str(data["metadata"].item()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklet-parquet", action="append", required=True)
    parser.add_argument("--feature-npz", required=True)
    parser.add_argument("--feature-key", default="features_osnet_prototypes")
    parser.add_argument("--valid-key", default="valid_osnet_prototypes")
    parser.add_argument("--name-prefix", default="osnet_proto")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--write-combined-view",
        action="store_true",
        help="also write one 3D prototype view for all-cross pair similarity features",
    )
    parser.add_argument("--combined-name", default="osnet_proto_all")
    args = parser.parse_args()

    parquet_paths = [Path(path) for path in args.tracklet_parquet]
    df = _load_sample_parquets(parquet_paths)
    records, _tracklet_table, _key_to_seq = _build_records(df, fps=float(args.fps))
    tracklet_keys = [record.tracklet_key for record in records]
    seqs = np.asarray([record.seq for record in records], dtype=np.int64)

    feature_path = Path(args.feature_npz)
    data = np.load(feature_path, allow_pickle=True)
    if args.feature_key not in data.files:
        raise ValueError(f"{feature_path} is missing {args.feature_key}")
    prototypes = np.asarray(data[args.feature_key], dtype=np.float32)
    if prototypes.ndim != 3:
        raise ValueError(f"{args.feature_key} must be 3D, got shape {prototypes.shape}")
    valid = np.ones(prototypes.shape[:2], dtype=bool)
    if args.valid_key and args.valid_key in data.files:
        valid = np.asarray(data[args.valid_key]).astype(bool)
        if valid.shape != prototypes.shape[:2]:
            raise ValueError(f"{args.valid_key} shape {valid.shape} does not match {prototypes.shape[:2]}")

    metadata = _load_metadata(data, feature_path)
    meta_records = metadata.get("records", [])
    index_by_key = {str(row["tracklet_key"]): int(row["index"]) for row in meta_records}
    missing = [key for key in tracklet_keys if key not in index_by_key]
    if missing:
        raise ValueError(f"{feature_path} is missing {len(missing)} tracklets; first missing={missing[0]}")
    indices = np.asarray([index_by_key[key] for key in tracklet_keys], dtype=np.int64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    pair_feature_args: list[str] = []
    if bool(args.write_combined_view):
        combined_name = str(args.combined_name)
        combined_features = prototypes[indices].astype(np.float32)
        combined_valid = valid[indices].astype(bool)
        combined_out = out_dir / f"{combined_name}.npz"
        np.savez_compressed(
            combined_out,
            seqs=seqs,
            features=combined_features,
            valid=combined_valid,
            metadata=json.dumps(
                {
                    "source_feature_npz": str(feature_path),
                    "source_feature_key": str(args.feature_key),
                    "source_valid_key": str(args.valid_key),
                    "name": combined_name,
                    "kind": "prototype_tensor",
                    "uses_anchors": False,
                    "uses_gt_for_training_or_anchors": False,
                    "uses_gt_for_evaluation_only": False,
                },
                sort_keys=True,
            ),
        )
        spec = f"{combined_name}:{combined_out}"
        pair_feature_args.extend(["--pair-feature-npz", spec])
        outputs.append(
            {
                "name": combined_name,
                "path": str(combined_out),
                "valid_tracklets": int((combined_valid.sum(axis=1) > 0).sum()),
                "valid_prototypes": int(combined_valid.sum()),
                "shape": list(combined_features.shape),
            }
        )
    slots = int(prototypes.shape[1])
    for slot in range(slots):
        name = f"{args.name_prefix}{slot}"
        features = _l2n(prototypes[indices, slot, :])
        slot_valid = valid[indices, slot].astype(bool)
        features[~slot_valid] = 0.0
        out = out_dir / f"{name}.npz"
        np.savez_compressed(
            out,
            seqs=seqs,
            features=features.astype(np.float32),
            valid=slot_valid,
            metadata=json.dumps(
                {
                    "source_feature_npz": str(feature_path),
                    "source_feature_key": str(args.feature_key),
                    "source_valid_key": str(args.valid_key),
                    "slot": int(slot),
                    "name": name,
                    "uses_anchors": False,
                    "uses_gt_for_training_or_anchors": False,
                    "uses_gt_for_evaluation_only": False,
                },
                sort_keys=True,
            ),
        )
        spec = f"{name}:{out}"
        pair_feature_args.extend(["--pair-feature-npz", spec])
        outputs.append(
            {
                "name": name,
                "path": str(out),
                "valid_tracklets": int(slot_valid.sum()),
                "dim": int(features.shape[1]),
            }
        )

    manifest = {
        "feature_npz": str(feature_path),
        "feature_key": str(args.feature_key),
        "valid_key": str(args.valid_key),
        "tracklet_parquet": [str(path) for path in parquet_paths],
        "outputs": outputs,
        "pair_feature_args": pair_feature_args,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    manifest_path = out_dir / f"{args.name_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"stage": "done", "manifest": str(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
