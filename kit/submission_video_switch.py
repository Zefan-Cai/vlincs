#!/usr/bin/env python
"""Switch whole videos between existing no-anchor submission zips.

This is a diagnostic/source-selection tool.  Each source zip must already be
produced by a no-anchor pipeline.  Ground truth is used only for scoring and,
when requested, for the oracle per-video policy diagnostic.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, KIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


def _parse_source(text: str) -> tuple[str, Path]:
    name, sep, path = str(text).partition(":")
    if not sep or not name or not path:
        raise ValueError(f"bad --source {text!r}; expected name:/path/to/submission.zip")
    return str(name), Path(path)


def _parse_policy(text: str, videos: list[str], default_source: str) -> dict[str, str]:
    policy = {video: default_source for video in videos}
    for part in str(text or "").split(","):
        if not part.strip():
            continue
        video, sep, source = part.rpartition(":")
        if not sep or not video or not source:
            raise ValueError(f"bad policy entry {part!r}; expected video:source")
        policy[str(video)] = str(source)
    return policy


def _load_zip(path: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".parquet"):
                continue
            with zf.open(name) as handle:
                df = pd.read_parquet(handle)
            video = Path(name).stem
            required = {"frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{path}:{name} missing columns {sorted(missing)}")
            out[video] = df[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
    if not out:
        raise ValueError(f"no parquet files found in {path}")
    return out


def _score(comp: dict[str, pd.DataFrame]) -> dict[str, object]:
    gt_by_video = load_ds1_gt_by_video()
    keys = sorted(set(gt_by_video).intersection(comp))
    metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    return {
        "idf1": round(float(metrics.idf1), 6),
        "hota": round(float(metrics.hota), 6),
        "assa": round(float(metrics.assa), 6),
        "deta": round(float(metrics.deta), 6),
        "detre": round(float(metrics.detre), 6),
        "detpr": round(float(metrics.detpr), 6),
        "unmatched_fp": int(metrics.unmatched_fp),
        "per_video": {
            key: {metric: round(float(value), 6) for metric, value in vals.items()}
            for key, vals in sorted(metrics.per_video.items())
        },
        "videos_scored": keys,
        "prediction_rows": int(sum(len(comp[key]) for key in keys)),
        "predicted_ids": int(sum(comp[key]["id"].nunique() for key in keys)),
    }


def _build_policy_comp(
    sources: dict[str, dict[str, pd.DataFrame]],
    policy: dict[str, str],
    *,
    id_offsets: dict[str, int],
) -> dict[str, pd.DataFrame]:
    comp: dict[str, pd.DataFrame] = {}
    for video, source in sorted(policy.items()):
        df = sources[source][video].copy()
        offset = int(id_offsets.get(source, 0))
        if offset:
            positive = df["id"].astype(np.int64) > 0
            df.loc[positive, "id"] = df.loc[positive, "id"].astype(np.int64) + offset
        comp[video] = df
    return comp


def _export_zip(comp: dict[str, pd.DataFrame], path: str) -> dict[str, object]:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="vlincs_switch_submit_"))
    written = []
    for video, df in sorted(comp.items()):
        out = df.copy()
        for col, dtype in [("frame", "uint32"), ("id", "uint32"), ("object_type", "uint8")]:
            out[col] = out[col].astype(dtype)
        for col in ("x1", "y1", "x2", "y2"):
            out[col] = out[col].clip(lower=0).astype("uint32")
        out["confidence"] = out["confidence"].astype("float32")
        if "box_hash" not in out.columns:
            from submit import _box_hash

            out["box_hash"] = [_box_hash(r.x1, r.y1, r.x2, r.y2) for r in out.itertuples()]
        for col in ("lat", "long", "alt"):
            if col not in out.columns:
                out[col] = np.float64("nan")
        p = tmp / f"{video}.parquet"
        out[
            [
                "frame",
                "id",
                "x1",
                "y1",
                "x2",
                "y2",
                "box_hash",
                "object_type",
                "confidence",
                "lat",
                "long",
                "alt",
            ]
        ].to_parquet(p)
        written.append(p)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in written:
            zf.write(p, arcname=p.name)
    return {"zip_out": str(out_path), "zip_files": int(len(written))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", required=True, help="name:/path/to/submission.zip")
    ap.add_argument("--base-source", required=True)
    ap.add_argument("--explicit-policy", default="")
    ap.add_argument("--oracle-per-video", action="store_true")
    ap.add_argument("--namespace-offsets", action="store_true", help="offset each source IDs to avoid cross-source numeric collisions")
    ap.add_argument("--zip-out", default="")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    source_paths = dict(_parse_source(spec) for spec in args.source)
    sources = {name: _load_zip(path) for name, path in source_paths.items()}
    if args.base_source not in sources:
        raise ValueError(f"base source {args.base_source!r} is not defined")
    videos = sorted(set.intersection(*(set(comp) for comp in sources.values())))
    if not videos:
        raise RuntimeError("no common videos across sources")
    id_offsets = {name: idx * 100_000_000 for idx, name in enumerate(sorted(sources))} if args.namespace_offsets else {}

    rows: list[dict[str, object]] = []
    base_policy = {video: args.base_source for video in videos}
    base_comp = _build_policy_comp(sources, base_policy, id_offsets=id_offsets)
    rows.append({"policy_name": "base", "policy": base_policy, **_score(base_comp)})

    if args.explicit_policy:
        policy = _parse_policy(args.explicit_policy, videos, args.base_source)
        unknown = sorted(set(policy.values()) - set(sources))
        if unknown:
            raise ValueError(f"unknown policy sources: {unknown}")
        comp = _build_policy_comp(sources, policy, id_offsets=id_offsets)
        rows.append({"policy_name": "explicit", "policy": policy, **_score(comp)})

    oracle_policy = None
    if args.oracle_per_video:
        per_video_choice: dict[str, str] = {}
        per_source_scores: dict[str, object] = {}
        for name, comp in sources.items():
            scored = _score(_build_policy_comp(sources, {video: name for video in videos}, id_offsets=id_offsets))
            per_source_scores[name] = scored
            for video, vals in scored["per_video"].items():
                if video not in per_video_choice or vals["idf1"] > per_source_scores[per_video_choice[video]]["per_video"][video]["idf1"]:
                    per_video_choice[video] = name
        oracle_policy = {video: per_video_choice[video] for video in videos}
        oracle_comp = _build_policy_comp(sources, oracle_policy, id_offsets=id_offsets)
        rows.append({"policy_name": "oracle_per_video", "policy": oracle_policy, "per_source_scores": per_source_scores, **_score(oracle_comp)})

    rows.sort(key=lambda row: float(row["idf1"]), reverse=True)
    if args.zip_out and rows:
        comp = _build_policy_comp(sources, rows[0]["policy"], id_offsets=id_offsets)
        rows[0].update(_export_zip(comp, args.zip_out))
    result = {
        "sources": {name: str(path) for name, path in source_paths.items()},
        "videos": videos,
        "id_offsets": id_offsets,
        "rows": rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "oracle_policy_uses_gt_for_selection": bool(args.oracle_per_video),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "done", "json": str(out), "best": rows[0]}, sort_keys=True))


if __name__ == "__main__":
    main()
