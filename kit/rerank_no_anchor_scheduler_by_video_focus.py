#!/usr/bin/env python
"""Rerank no-anchor full-score scheduler rows by bottleneck-video coverage.

This is a research-budget allocator, not an identity resolver.  It does not
edit assignments or use anchors.  It uses prior full-score per-video metrics
only to decide which already-generated no-anchor candidates deserve the next
expensive DS1 full-score slots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.export_no_anchor_scheduler_manifest_assignments import _accepted_preview, _load_scheduler_selected
from kit.no_anchor_fullscore_scheduler import _as_float


def _load_seq_video(path: Path) -> dict[int, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"seq", "video"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return {int(float(row["seq"])): str(row["video"]) for row in reader}


def _pick_metric_row(path: Path, row_index: int) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        rows = [row for row in data["rows"] if isinstance(row, dict) and "per_video" in row]
        if not rows:
            raise ValueError(f"{path} has no rows[] with per_video metrics")
        if row_index < 0 or row_index >= len(rows):
            raise ValueError(f"--metric-row-index {row_index} outside 0..{len(rows) - 1}")
        return rows[row_index]
    if isinstance(data, dict) and "per_video" in data:
        return data
    raise ValueError(f"{path} does not contain per_video full-score metrics")


def _video_weights(metric_row: dict[str, Any], target_idf1: float) -> dict[str, float]:
    per_video = metric_row.get("per_video")
    if not isinstance(per_video, dict):
        raise ValueError("metric row is missing per_video")
    by_video_rows = metric_row.get("by_video_rows") if isinstance(metric_row.get("by_video_rows"), dict) else {}
    weights: dict[str, float] = {}
    for video, metrics in per_video.items():
        if not isinstance(metrics, dict):
            continue
        idf1 = _as_float(metrics.get("idf1"), 0.0) or 0.0
        row_info = by_video_rows.get(video, {}) if isinstance(by_video_rows, dict) else {}
        input_rows = _as_float(row_info.get("input_rows") if isinstance(row_info, dict) else None, 1.0) or 1.0
        # sqrt(input_rows) keeps MCAM04 visibly important without letting row
        # count swamp all association evidence.
        weights[str(video)] = max(float(target_idf1) - idf1, 0.0) * math.sqrt(max(input_rows, 1.0))
    return weights


def _preview_video_counts(preview: list[dict[str, Any]], seq_video: dict[int, str]) -> tuple[Counter[str], Counter[str]]:
    source = Counter()
    target = Counter()
    for item in preview:
        if not isinstance(item, dict):
            continue
        for seq in item.get("source_seqs") or []:
            video = seq_video.get(int(float(seq)))
            if video:
                source[video] += 1
        target_seqs = item.get("target_top_seqs")
        if not isinstance(target_seqs, list):
            target_seqs = item.get("target_indices")
        for seq in target_seqs or []:
            video = seq_video.get(int(float(seq)))
            if video:
                target[video] += 1
    return source, target


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _enrich_row(row: dict[str, Any], seq_video: dict[int, str], weights: dict[str, float]) -> dict[str, Any]:
    preview = _accepted_preview(row)
    source_counts, target_counts = _preview_video_counts(preview, seq_video)
    videos = set(source_counts) | set(target_counts)
    source_focus = sum(source_counts[video] * weights.get(video, 0.0) for video in videos)
    target_focus = sum(target_counts[video] * weights.get(video, 0.0) for video in videos)
    row_focus = source_focus + target_focus
    predicted = (
        _as_float(row.get("predicted_full_idf1"))
        or _as_float(row.get("learned_proxy_full_idf1"))
        or _as_float(row.get("full_side_effect_proxy"))
        or 0.0
    )
    enriched = dict(row)
    original_rank = enriched.pop("_selection_rank", enriched.get("scheduler_rank", ""))
    enriched.update(
        {
            "source_scheduler_selection_rank": original_rank,
            "video_focus_score": float(row_focus),
            "video_focus_source_score": float(source_focus),
            "video_focus_target_score": float(target_focus),
            "video_focus_source_counts": _counter_dict(source_counts),
            "video_focus_target_counts": _counter_dict(target_counts),
            "video_focus_weighted_videos": {
                video: round(float(weights.get(video, 0.0)), 6) for video in sorted(videos) if weights.get(video, 0.0) > 0
            },
            "video_focus_sort_score": float(row_focus + 4.0 * max(predicted - 0.65, 0.0)),
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_candidate_prioritization": True,
            "uses_gt_for_evaluation_only": True,
        }
    )
    return enriched


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        "video_focus_rank",
        "video_focus_sort_score",
        "video_focus_score",
        "source_scheduler_selection_rank",
        "predicted_full_idf1",
        "scheduler_score",
        "mode",
        "scheduler_family",
        "_source_file",
        "_source_rank",
        "video_focus_source_counts",
        "video_focus_target_counts",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {key: row.get(key, "") for key in keys}
            out["video_focus_rank"] = rank
            out["video_focus_source_counts"] = json.dumps(row.get("video_focus_source_counts", {}), sort_keys=True)
            out["video_focus_target_counts"] = json.dumps(row.get("video_focus_target_counts", {}), sort_keys=True)
            writer.writerow(out)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Video-Focus Scheduler",
        "",
        f"- input scheduler: `{result['input_scheduler_json']}`",
        f"- selected rows: `{len(result['selected'])}`",
        f"- target IDF1 floor: `{result['target_idf1']:.3f}`",
        f"- uses anchors: `{result['uses_anchors']}`",
        f"- uses GT for training/anchors: `{result['uses_gt_for_training_or_anchors']}`",
        f"- uses GT only for experiment-budget prioritization: `{result['uses_gt_for_candidate_prioritization']}`",
        "",
        "## Bottleneck Weights",
        "",
        "| video | focus weight |",
        "| --- | ---: |",
    ]
    for video, weight in sorted(result["video_focus_weights"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| `{video}` | `{weight:.6f}` |")
    lines.extend(
        [
            "",
            "## Selected Candidates",
            "",
            "| rank | focus | predicted full | original rank | family | source videos | target videos |",
            "| ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for rank, row in enumerate(result["selected"], start=1):
        lines.append(
            "| {rank} | `{focus:.3f}` | `{pred:.6f}` | `{orig}` | `{family}` | `{src}` | `{tgt}` |".format(
                rank=rank,
                focus=float(row.get("video_focus_score") or 0.0),
                pred=float(row.get("predicted_full_idf1") or row.get("learned_proxy_full_idf1") or 0.0),
                orig=row.get("source_scheduler_selection_rank", row.get("scheduler_rank", "")),
                family=row.get("scheduler_family", row.get("signature", "")),
                src=json.dumps(row.get("video_focus_source_counts", {}), sort_keys=True),
                tgt=json.dumps(row.get("video_focus_target_counts", {}), sort_keys=True),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rerank(args: argparse.Namespace) -> dict[str, Any]:
    scheduler_json = Path(args.scheduler_json)
    selected = _load_scheduler_selected(scheduler_json)
    seq_video = _load_seq_video(Path(args.seq_video_csv))
    metric_row = _pick_metric_row(Path(args.metric_json), int(args.metric_row_index))
    weights = _video_weights(metric_row, float(args.target_idf1))
    enriched = [_enrich_row(row, seq_video, weights) for row in selected]
    enriched.sort(
        key=lambda row: (
            float(row.get("video_focus_sort_score") or 0.0),
            float(row.get("predicted_full_idf1") or 0.0),
            float(row.get("scheduler_score") or 0.0),
        ),
        reverse=True,
    )
    kept = enriched[: int(args.top_n)]
    for rank, row in enumerate(kept, start=1):
        row["video_focus_rank"] = int(rank)
    result = {
        "input_scheduler_json": str(scheduler_json),
        "seq_video_csv": str(args.seq_video_csv),
        "metric_json": str(args.metric_json),
        "metric_row_index": int(args.metric_row_index),
        "target_idf1": float(args.target_idf1),
        "video_focus_weights": weights,
        "selected": kept,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_candidate_prioritization": True,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), kept)
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        seq_csv = root / "seqs.csv"
        seq_csv.write_text("seq,video\n1,A\n2,B\n3,B\n4,C\n", encoding="utf-8")
        metric_json = root / "metric.json"
        metric_json.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "per_video": {"A": {"idf1": 0.8}, "B": {"idf1": 0.5}, "C": {"idf1": 0.69}},
                            "by_video_rows": {"A": {"input_rows": 100}, "B": {"input_rows": 400}, "C": {"input_rows": 100}},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        scheduler = root / "scheduler.json"
        scheduler.write_text(
            json.dumps(
                {
                    "selected": [
                        {"mode": "x", "predicted_full_idf1": 0.66, "accepted_preview": [{"source_seqs": [1], "target_top_seqs": [1]}]},
                        {"mode": "y", "predicted_full_idf1": 0.65, "accepted_preview": [{"source_seqs": [2, 3], "target_top_seqs": [4]}]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = rerank(
            argparse.Namespace(
                scheduler_json=str(scheduler),
                seq_video_csv=str(seq_csv),
                metric_json=str(metric_json),
                metric_row_index=0,
                target_idf1=0.7,
                top_n=2,
                json=str(root / "out.json"),
                csv="",
                md="",
            )
        )
        assert out["selected"][0]["mode"] == "y"
        assert out["selected"][0]["video_focus_source_counts"]["B"] == 2
        print(json.dumps({"stage": "self_test", "status": "ok"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--scheduler-json", default="")
    ap.add_argument("--seq-video-csv", default="")
    ap.add_argument("--metric-json", default="")
    ap.add_argument("--metric-row-index", type=int, default=0)
    ap.add_argument("--target-idf1", type=float, default=0.70)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.scheduler_json or not args.seq_video_csv or not args.metric_json or not args.json:
        ap.error("--scheduler-json, --seq-video-csv, --metric-json, and --json are required")
    result = rerank(args)
    print(json.dumps({"json": args.json, "selected": len(result["selected"])}, sort_keys=True))


if __name__ == "__main__":
    main()
