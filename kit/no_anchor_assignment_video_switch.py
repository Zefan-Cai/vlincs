#!/usr/bin/env python
"""Video-level switcher over existing no-anchor assignment CSVs.

All input sources must already be no-anchor assignment artifacts.  The script
aligns each source's component namespace to a reference source using only
tracklet-overlap majority, then evaluates source-per-video policies.  Ground
truth is loaded only after prediction for metrics and full scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _parse_source(text: str) -> tuple[str, Path]:
    name, sep, path = str(text).partition(":")
    if not sep or not name or not path:
        raise ValueError(f"bad --source {text!r}; expected name:/path/to/assignments.csv")
    return name, Path(path)


def _parse_policy(text: str, videos: list[str], default_source: str) -> dict[str, str]:
    policy = {video: default_source for video in videos}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        video, sep, source = part.rpartition(":")
        if not sep or not video or not source:
            raise ValueError(f"bad policy entry {part!r}; expected video:source")
        policy[video] = source
    return policy


def _load_assignment(path: Path) -> dict[int, tuple[str, int]]:
    df = pd.read_csv(path)
    required = {"seq", "video", "predicted_global_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    out: dict[int, tuple[str, int]] = {}
    for row in df[["seq", "video", "predicted_global_id"]].itertuples(index=False):
        out[int(row.seq)] = (str(row.video), int(row.predicted_global_id))
    return out


def _align_sources(
    raw: dict[str, dict[int, tuple[str, int]]],
    reference: str,
) -> tuple[dict[str, dict[int, tuple[str, int]]], dict[str, object]]:
    if reference not in raw:
        raise ValueError(f"reference source {reference!r} not in sources")
    ref = raw[reference]
    ref_gid_by_seq = {seq: gid for seq, (_video, gid) in ref.items()}
    aligned: dict[str, dict[int, tuple[str, int]]] = {reference: dict(ref)}
    next_gid = max((gid for rows in raw.values() for _video, gid in rows.values()), default=0) + 10_000_000
    stats: dict[str, object] = {}

    for name, rows in raw.items():
        if name == reference:
            stats[name] = {
                "rows": len(rows),
                "mapped_components": len(set(gid for _video, gid in rows.values())),
                "new_components": 0,
                "overlap_rows": len(rows),
            }
            continue
        votes: dict[int, Counter] = defaultdict(Counter)
        for seq, (_video, gid) in rows.items():
            if seq in ref_gid_by_seq:
                votes[int(gid)][int(ref_gid_by_seq[seq])] += 1
        mapping: dict[int, int] = {}
        mapped = 0
        for gid, counter in votes.items():
            if counter:
                mapping[int(gid)] = int(counter.most_common(1)[0][0])
                mapped += 1
        new_components = 0
        converted: dict[int, tuple[str, int]] = {}
        for seq, (video, gid) in rows.items():
            gid = int(gid)
            if gid not in mapping:
                mapping[gid] = next_gid
                next_gid += 1
                new_components += 1
            converted[int(seq)] = (video, int(mapping[gid]))
        aligned[name] = converted
        stats[name] = {
            "rows": len(rows),
            "mapped_components": int(mapped),
            "new_components": int(new_components),
            "overlap_rows": int(sum(1 for seq in rows if seq in ref_gid_by_seq)),
        }
    return aligned, stats


def _policy_pred(
    aligned: dict[str, dict[int, tuple[str, int]]],
    policy: dict[str, str],
    videos: list[str],
) -> dict[int, int]:
    allowed = set(videos)
    out: dict[int, int] = {}
    for video in videos:
        source = policy[video]
        rows = aligned[source]
        for seq, (row_video, gid) in rows.items():
            if row_video == video and row_video in allowed:
                out[int(seq)] = int(gid)
    return out


def _score_policy(
    name: str,
    policy: dict[str, str],
    *,
    aligned: dict[str, dict[int, tuple[str, int]]],
    videos: list[str],
    records,
    pred_by_video,
    gt_by_video,
    gt_by_seq,
    weight_by_seq,
) -> dict[str, object]:
    pred = _policy_pred(aligned, policy, videos)
    pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
    full = _score_full(pred_by_video, gt_by_video, pred)
    counts = Counter(policy.values())
    return {
        "policy_name": name,
        "policy": dict(sorted(policy.items())),
        "source_counts": dict(sorted(counts.items())),
        "output_tracklets": int(len(pred)),
        **pair,
        **{f"full_{key}": value for key, value in full.items() if key != "per_video"},
        "full_per_video": full["per_video"],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }


def _write_assignments(
    path: str,
    pred_by_seq: dict[int, int],
    aligned_rows: dict[str, dict[int, tuple[str, int]]],
    policy: dict[str, str],
    videos: list[str],
) -> dict[str, int]:
    by_seq_meta: dict[int, tuple[str, str]] = {}
    for video in videos:
        source = policy[video]
        for seq, (row_video, _gid) in aligned_rows[source].items():
            if row_video == video:
                by_seq_meta[int(seq)] = (row_video, source)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq", "video", "source", "predicted_global_id"])
        writer.writeheader()
        for seq, gid in sorted(pred_by_seq.items()):
            video, source = by_seq_meta.get(int(seq), ("", ""))
            writer.writerow({"seq": int(seq), "video": video, "source": source, "predicted_global_id": int(gid)})
    return {"assignment_rows": int(len(pred_by_seq))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--source", action="append", required=True, help="name:/path/to/assignments.csv")
    ap.add_argument("--reference-source", required=True)
    ap.add_argument("--base-source", required=True)
    ap.add_argument(
        "--explicit-policy",
        default="",
        help="comma list of video:source entries; unspecified videos use --base-source and only this policy is scored",
    )
    ap.add_argument("--max-greedy-iters", type=int, default=4)
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--assignments-out", default=None)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    sources = dict(_parse_source(spec) for spec in args.source)
    if len(sources) != len(args.source):
        raise ValueError("duplicate source names")
    raw = {name: _load_assignment(path) for name, path in sources.items()}
    aligned, align_stats = _align_sources(raw, str(args.reference_source))

    con = _connect(args.dbname)
    records, _emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": 0.5,
        "min_matches": 1,
        "min_purity": 0.0,
        "n_tracklets": len(records),
        "prediction_rows": int(sum(len(value) for value in pred_by_video.values())),
        "gt_rows": int(sum(len(value) for value in gt_by_video.values())),
    }
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached
    videos = sorted(pred_by_video)

    if args.explicit_policy:
        explicit = _parse_policy(str(args.explicit_policy), videos, str(args.base_source))
        unknown_sources = sorted(set(explicit.values()) - set(aligned))
        unknown_videos = sorted(set(explicit) - set(videos))
        if unknown_sources:
            raise ValueError(f"explicit policy references unknown sources: {unknown_sources}")
        if unknown_videos:
            raise ValueError(f"explicit policy references unknown videos: {unknown_videos}")
        row = _score_policy(
            "explicit_policy",
            explicit,
            aligned=aligned,
            videos=videos,
            records=records,
            pred_by_video=pred_by_video,
            gt_by_video=gt_by_video,
            gt_by_seq=gt_by_seq,
            weight_by_seq=weight_by_seq,
        )
        assignment_info = {}
        if args.assignments_out:
            pred = _policy_pred(aligned, row["policy"], videos)
            assignment_info = _write_assignments(args.assignments_out, pred, aligned, row["policy"], videos)
        result = {
            "dbname": args.dbname,
            "role": args.role,
            "sources": {name: str(path) for name, path in sorted(sources.items())},
            "reference_source": str(args.reference_source),
            "base_source": str(args.base_source),
            "alignment": align_stats,
            "eval_stats": eval_stats,
            "top": [row],
            "best": row,
            "assignment_info": assignment_info,
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": True,
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"stage": "done", "json": str(out), "best": row}, sort_keys=True), flush=True)
        return

    rows: list[dict[str, object]] = []
    for source in sorted(aligned):
        policy = {video: source for video in videos}
        row = _score_policy(
            f"all_{source}",
            policy,
            aligned=aligned,
            videos=videos,
            records=records,
            pred_by_video=pred_by_video,
            gt_by_video=gt_by_video,
            gt_by_seq=gt_by_seq,
            weight_by_seq=weight_by_seq,
        )
        rows.append(row)
        print(json.dumps({"stage": "source", "source": source, "full_idf1": row["full_idf1"], "pair_f1": row["tracklet_pair_f1"]}, sort_keys=True), flush=True)

    current = {video: str(args.base_source) for video in videos}
    best = _score_policy(
        "greedy_start",
        current,
        aligned=aligned,
        videos=videos,
        records=records,
        pred_by_video=pred_by_video,
        gt_by_video=gt_by_video,
        gt_by_seq=gt_by_seq,
        weight_by_seq=weight_by_seq,
    )
    rows.append(best)
    greedy_trace = [best]
    for iteration in range(1, int(args.max_greedy_iters) + 1):
        candidates = []
        for video in videos:
            for source in sorted(aligned):
                if source == current[video]:
                    continue
                trial_policy = dict(current)
                trial_policy[video] = source
                row = _score_policy(
                    f"greedy_iter{iteration}_{video}_to_{source}",
                    trial_policy,
                    aligned=aligned,
                    videos=videos,
                    records=records,
                    pred_by_video=pred_by_video,
                    gt_by_video=gt_by_video,
                    gt_by_seq=gt_by_seq,
                    weight_by_seq=weight_by_seq,
                )
                candidates.append(row)
        candidates.sort(
            key=lambda row: (
                float(row["full_idf1"]),
                float(row["tracklet_pair_f1"]),
                float(row["tracklet_pair_recall"]),
            ),
            reverse=True,
        )
        rows.extend(candidates[:20])
        if not candidates or float(candidates[0]["full_idf1"]) <= float(best["full_idf1"]) + 1.0e-12:
            break
        best = candidates[0]
        current = dict(best["policy"])
        greedy_trace.append(best)
        print(
            json.dumps(
                {
                    "stage": "greedy_accept",
                    "iteration": iteration,
                    "policy_name": best["policy_name"],
                    "full_idf1": best["full_idf1"],
                    "pair_f1": best["tracklet_pair_f1"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    source_rows = {row["policy_name"].replace("all_", ""): row for row in rows if str(row["policy_name"]).startswith("all_")}
    oracle_policy: dict[str, str] = {}
    for video in videos:
        best_source = max(
            source_rows,
            key=lambda source: float(source_rows[source]["full_per_video"].get(video, {}).get("idf1", -1.0)),
        )
        oracle_policy[video] = best_source
    oracle_row = _score_policy(
        "per_video_source_oracle",
        oracle_policy,
        aligned=aligned,
        videos=videos,
        records=records,
        pred_by_video=pred_by_video,
        gt_by_video=gt_by_video,
        gt_by_seq=gt_by_seq,
        weight_by_seq=weight_by_seq,
    )
    rows.append(oracle_row)

    rows.sort(
        key=lambda row: (
            float(row["full_idf1"]),
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
        ),
        reverse=True,
    )
    best_row = rows[0]
    assignment_info = {}
    if args.assignments_out:
        pred = _policy_pred(aligned, best_row["policy"], videos)
        assignment_info = _write_assignments(args.assignments_out, pred, aligned, best_row["policy"], videos)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "sources": {name: str(path) for name, path in sorted(sources.items())},
        "reference_source": str(args.reference_source),
        "base_source": str(args.base_source),
        "alignment": align_stats,
        "eval_stats": eval_stats,
        "greedy_trace": greedy_trace,
        "top": rows[:50],
        "best": best_row,
        "assignment_info": assignment_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "done", "json": str(out), "best": best_row}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
