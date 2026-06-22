#!/usr/bin/env python
"""Video-level source switching for no-anchor sample assignment CSVs.

All sources must already be no-anchor artifacts.  Component namespaces are
aligned to a reference source using only tracklet-overlap majority, not ground
truth.  Ground truth in the sample parquet is used only after prediction for
diagnostic ranking and metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.sample_assignment_admission_grid import _pair_metrics
from kit.sample_assignment_state_policy_sweep import (
    _gt_mapping,
    _jsonable,
    _load_parquets,
    _sample_parquet_gt_score,
    _tracklet_table,
)


def _parse_source(text: str) -> tuple[str, Path]:
    name, sep, path = str(text).partition(":")
    if not sep or not name or not path:
        raise ValueError(f"bad --source {text!r}; expected name:/path/to/assignments.csv")
    return str(name), Path(path)


def _load_assignment(path: Path, table: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tracklet_key", "predicted_global_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    df = df[["tracklet_key", "predicted_global_id"]].copy()
    df["tracklet_key"] = df["tracklet_key"].astype(str)
    df["predicted_global_id"] = df["predicted_global_id"].astype(np.int64)
    df = df.drop_duplicates("tracklet_key", keep="first")
    meta = table[["tracklet_key", "video"]].copy()
    meta["tracklet_key"] = meta["tracklet_key"].astype(str)
    df = df.merge(meta, on="tracklet_key", how="inner")
    return df[["tracklet_key", "video", "predicted_global_id"]].copy()


def _align_sources(raw: dict[str, pd.DataFrame], reference: str) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if reference not in raw:
        raise ValueError(f"reference source {reference!r} not in sources")
    ref = raw[reference]
    ref_gid_by_key = ref.set_index("tracklet_key")["predicted_global_id"].astype(np.int64).to_dict()
    aligned: dict[str, pd.DataFrame] = {reference: ref.copy()}
    next_gid = int(max(df["predicted_global_id"].max() for df in raw.values() if len(df)) + 10_000_000)
    stats: dict[str, object] = {}

    for name, df in raw.items():
        raw_components = int(df["predicted_global_id"].nunique())
        if name == reference:
            stats[name] = {
                "rows": int(len(df)),
                "raw_components": raw_components,
                "mapped_components": raw_components,
                "new_components": 0,
                "overlap_rows": int(len(df)),
            }
            continue
        votes: dict[int, Counter[int]] = defaultdict(Counter)
        for row in df.itertuples(index=False):
            key = str(row.tracklet_key)
            if key in ref_gid_by_key:
                votes[int(row.predicted_global_id)][int(ref_gid_by_key[key])] += 1
        mapping: dict[int, int] = {}
        for gid, counter in votes.items():
            if counter:
                mapping[int(gid)] = int(counter.most_common(1)[0][0])
        new_components = 0
        for gid in sorted(int(gid) for gid in df["predicted_global_id"].unique()):
            if gid not in mapping:
                mapping[gid] = next_gid
                next_gid += 1
                new_components += 1
        out = df.copy()
        out["predicted_global_id"] = out["predicted_global_id"].map(mapping).astype(np.int64)
        aligned[name] = out
        stats[name] = {
            "rows": int(len(df)),
            "raw_components": raw_components,
            "mapped_components": int(raw_components - new_components),
            "new_components": int(new_components),
            "overlap_rows": int(df["tracklet_key"].isin(ref_gid_by_key).sum()),
        }
    return aligned, stats


def _build_policy_assignments(
    aligned: dict[str, pd.DataFrame],
    policy: dict[str, str],
    *,
    base_source: str,
    missing_mode: str,
) -> pd.DataFrame:
    frames = []
    for video, source in sorted(policy.items()):
        primary = aligned[source]
        primary_video = primary.loc[primary["video"].astype(str) == str(video)].copy()
        frames.append(primary_video)
        if missing_mode == "base_fallback" and source != base_source:
            base = aligned[base_source]
            base_video = base.loc[base["video"].astype(str) == str(video)].copy()
            have = set(primary_video["tracklet_key"].astype(str))
            fallback = base_video.loc[~base_video["tracklet_key"].astype(str).isin(have)].copy()
            frames.append(fallback)
    if not frames:
        return pd.DataFrame(columns=["tracklet_key", "video", "predicted_global_id"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["video", "tracklet_key"], keep="first")
    return out[["tracklet_key", "video", "predicted_global_id"]].copy()


def _pred_by_seq(table: pd.DataFrame, assignments: pd.DataFrame) -> dict[int, int]:
    meta = table[["seq", "tracklet_key"]].copy()
    meta["tracklet_key"] = meta["tracklet_key"].astype(str)
    merged = assignments.merge(meta, on="tracklet_key", how="inner")
    return {
        int(row.seq): int(row.predicted_global_id)
        for row in merged[["seq", "predicted_global_id"]].itertuples(index=False)
    }


def _score_policy(
    name: str,
    policy: dict[str, str],
    *,
    df: pd.DataFrame,
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    gt_by_seq: dict[int, int],
    weight_by_seq: dict[int, float],
) -> dict[str, object]:
    seqs = [int(row.seq) for row in table.itertuples(index=False)]
    pred = _pred_by_seq(table, assignments)
    pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
    full = _sample_parquet_gt_score(df, assignments)
    return {
        "policy_name": name,
        "policy": dict(sorted(policy.items())),
        "source_counts": dict(sorted(Counter(policy.values()).items())),
        "output_tracklets": int(assignments["tracklet_key"].nunique()),
        **pair,
        **full,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracklet-parquet", nargs="+", required=True)
    ap.add_argument("--source", action="append", required=True, help="name:/path/to/assignments.csv")
    ap.add_argument("--reference-source", required=True)
    ap.add_argument("--base-source", required=True)
    ap.add_argument("--missing-mode", choices=["drop", "base_fallback"], default="base_fallback")
    ap.add_argument("--eval-min-gt-fraction", type=float, default=0.5)
    ap.add_argument("--eval-min-rows", type=int, default=1)
    ap.add_argument("--json-top-n", type=int, default=200)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    source_paths = dict(_parse_source(spec) for spec in args.source)
    if len(source_paths) != len(args.source):
        raise ValueError("duplicate source names")
    if args.reference_source not in source_paths:
        raise ValueError("--reference-source must be one of --source")
    if args.base_source not in source_paths:
        raise ValueError("--base-source must be one of --source")

    df = _load_parquets(args.tracklet_parquet)
    table = _tracklet_table(df)
    videos = sorted(table["video"].astype(str).unique().tolist())
    raw = {name: _load_assignment(path, table) for name, path in source_paths.items()}
    aligned, align_stats = _align_sources(raw, str(args.reference_source))
    gt_by_seq, weight_by_seq, eval_info = _gt_mapping(
        table,
        min_gt_fraction=float(args.eval_min_gt_fraction),
        min_rows=int(args.eval_min_rows),
    )

    rows: list[dict[str, object]] = []
    base_policy = {video: str(args.base_source) for video in videos}
    base_assignments = _build_policy_assignments(
        aligned,
        base_policy,
        base_source=str(args.base_source),
        missing_mode=str(args.missing_mode),
    )
    rows.append(
        _score_policy(
            "base",
            base_policy,
            df=df,
            table=table,
            assignments=base_assignments,
            gt_by_seq=gt_by_seq,
            weight_by_seq=weight_by_seq,
        )
    )

    for source in sorted(source_paths):
        policy = {video: source for video in videos}
        assignments = _build_policy_assignments(
            aligned,
            policy,
            base_source=str(args.base_source),
            missing_mode=str(args.missing_mode),
        )
        rows.append(
            _score_policy(
                f"all:{source}",
                policy,
                df=df,
                table=table,
                assignments=assignments,
                gt_by_seq=gt_by_seq,
                weight_by_seq=weight_by_seq,
            )
        )

    for video in videos:
        for source in sorted(source_paths):
            if source == args.base_source:
                continue
            policy = dict(base_policy)
            policy[str(video)] = source
            assignments = _build_policy_assignments(
                aligned,
                policy,
                base_source=str(args.base_source),
                missing_mode=str(args.missing_mode),
            )
            rows.append(
                _score_policy(
                    f"switch:{video}:{source}",
                    policy,
                    df=df,
                    table=table,
                    assignments=assignments,
                    gt_by_seq=gt_by_seq,
                    weight_by_seq=weight_by_seq,
                )
            )

    rows.sort(
        key=lambda row: (
            float(row.get("sample_full_idf1", -1.0)),
            float(row.get("tracklet_pair_f1", -1.0)),
            int(row.get("output_tracklets", 0)),
        ),
        reverse=True,
    )
    result = {
        "tracklet_parquet": [str(path) for path in args.tracklet_parquet],
        "sources": {name: str(path) for name, path in source_paths.items()},
        "reference_source": str(args.reference_source),
        "base_source": str(args.base_source),
        "missing_mode": str(args.missing_mode),
        "videos": videos,
        "n_tracklets": int(len(table)),
        "eval_stats": eval_info,
        "align_stats": align_stats,
        "rows": rows[: int(args.json_top_n)],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "oracle_policy_uses_gt_for_selection": True,
        "sample_full_reference": "parquet_gt_same_detection_boxes",
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "done", "json": str(out), "best": _jsonable(rows[0] if rows else {})}, sort_keys=True))


if __name__ == "__main__":
    main()

