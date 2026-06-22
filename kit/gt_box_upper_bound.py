#!/usr/bin/env python
"""DS1 GT-box upper bounds for separating detection and identity errors."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from submit import _box_hash
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


_GT_ID_RE = re.compile(r"(\d+)$")


def _numeric_id(value: object, mapping: dict[str, int]) -> int:
    key = str(value)
    if key in mapping:
        return mapping[key]
    m = _GT_ID_RE.search(key)
    if m:
        candidate = int(m.group(1))
        if candidate > 0 and candidate not in mapping.values():
            mapping[key] = candidate
            return candidate
    mapping[key] = max(mapping.values(), default=0) + 1
    return mapping[key]


def build_comp(gt_by_video: dict[str, pd.DataFrame], mode: str) -> dict[str, pd.DataFrame]:
    id_map: dict[str, int] = {}
    comp: dict[str, pd.DataFrame] = {}
    local_offset = 50_000_000
    for video_index, (video, gt) in enumerate(sorted(gt_by_video.items())):
        out = gt[["frame", "id", "tracklet_id", "x1", "y1", "x2", "y2", "object_type"]].copy()
        if mode == "gt_identity":
            out["id"] = [_numeric_id(v, id_map) for v in out["id"]]
        elif mode == "local_tracklet":
            out["id"] = local_offset + video_index * 1_000_000 + out["tracklet_id"].astype(int)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        out["confidence"] = 1.0
        comp[video] = out[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
    return comp


def export_zip(comp: dict[str, pd.DataFrame], out_zip: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="vlincs_gtbox_submit_"))
    written = []
    for video, df in comp.items():
        out = df.copy()
        out["frame"] = out["frame"].astype("uint32")
        out["id"] = out["id"].astype("uint32")
        for col in ("x1", "y1", "x2", "y2"):
            out[col] = out[col].clip(lower=0).astype("uint32")
        out["object_type"] = 0
        out["object_type"] = out["object_type"].astype("uint8")
        out["confidence"] = out["confidence"].astype("float32")
        out["box_hash"] = [_box_hash(r.x1, r.y1, r.x2, r.y2) for r in out.itertuples()]
        for col in ("lat", "long", "alt"):
            out[col] = np.float64("nan")
        path = tmp / f"{video}.parquet"
        out[["frame", "id", "x1", "y1", "x2", "y2", "box_hash", "object_type", "confidence", "lat", "long", "alt"]].to_parquet(path)
        written.append(path)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in written:
            zf.write(path, path.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["gt_identity", "local_tracklet"], default="gt_identity")
    ap.add_argument("--submit", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    gt = load_ds1_gt_by_video()
    comp = build_comp(gt, args.mode)
    metrics = evaluate(gt, comp, dense=False, n_workers=1)
    result = {
        "dataset": "ds1",
        "mode": args.mode,
        "idf1": round(metrics.idf1, 6),
        "hota": round(metrics.hota, 6),
        "assa": round(metrics.assa, 6),
        "deta": round(metrics.deta, 6),
        "detre": round(metrics.detre, 6),
        "detpr": round(metrics.detpr, 6),
        "unmatched_fp": int(metrics.unmatched_fp),
        "per_video": {
            key: {metric: round(float(value), 6) for metric, value in vals.items()}
            for key, vals in sorted(metrics.per_video.items())
        },
        "uses_gt_boxes": True,
        "uses_gt_identity": args.mode == "gt_identity",
    }
    if args.submit:
        export_zip(comp, args.submit)
        result["submission"] = args.submit
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
