#!/usr/bin/env python
"""Filter component-graph candidates with a no-GT rescue rule.

The rule targets a failure mode found by the eval-only opponent: high-value
component bridges can have moderate centroid similarity and low view-vote, so
they are buried by a conventional "highest visual consensus first" ranker.  The
filter itself reads only candidate metadata and no identity labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no rows[]")
    out = []
    for rank, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            out.append({"candidate_rank": rank, "_source_file": str(path), "_source_rank": rank, **row})
    return out


def _rescue_score(row: dict[str, Any]) -> float:
    vote = _as_float(row.get("target_view_vote"))
    low_vote_bonus = max(0.0, 1.0 - abs(vote - 0.30) / 0.20)
    moved = min(math.log1p(_as_float(row.get("moved_tracklets"))) / math.log(256.0), 1.0)
    return float(
        0.30 * _as_float(row.get("target_best_sim"))
        + 0.22 * _as_float(row.get("target_min_view_sim"))
        + 0.16 * _as_float(row.get("target_mean_sim"))
        + 0.14 * low_vote_bonus
        + 0.10 * moved
        + 0.08 * (1.0 - min(_as_float(row.get("same_video_overlap_ratio")) / 0.03, 1.0))
    )


def _keep(row: dict[str, Any], args: argparse.Namespace) -> bool:
    vote = _as_float(row.get("target_view_vote"))
    return (
        float(args.min_vote) <= vote <= float(args.max_vote)
        and _as_float(row.get("target_best_sim")) >= float(args.min_target_best_sim)
        and _as_float(row.get("target_mean_sim")) >= float(args.min_centroid_sim)
        and _as_float(row.get("target_min_view_sim")) >= float(args.min_min_view_sim)
        and _as_float(row.get("same_video_overlap_ratio")) <= float(args.max_same_video_overlap_ratio)
        and _as_float(row.get("moved_tracklets")) >= float(args.min_moved_tracklets)
    )


def filter_rows(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(Path(args.candidates_json))
    kept = []
    rejected = []
    for row in rows:
        new_row = dict(row)
        new_row["component_graph_rescue_score"] = _rescue_score(new_row)
        new_row["selection_rule"] = "low_vote_high_top_similarity_rescue"
        new_row["selection_rule_origin"] = str(args.rule_origin)
        new_row["uses_anchors"] = False
        new_row["uses_gt_for_training_or_anchors"] = False
        new_row["uses_gt_for_evaluation_only"] = False
        if _keep(new_row, args):
            kept.append(new_row)
        else:
            rejected.append(new_row)
    kept.sort(key=lambda row: (float(row["component_graph_rescue_score"]), float(row.get("accepted_pair_mass_proxy_sum", 0.0))), reverse=True)
    selected = kept[: int(args.top_n)]
    result = {
        "candidates_json": str(args.candidates_json),
        "raw_count": int(len(rows)),
        "kept_count": int(len(kept)),
        "selected": selected,
        "top_rejected": sorted(rejected, key=lambda row: float(row["component_graph_rescue_score"]), reverse=True)[:20],
        "rule": {
            "min_vote": float(args.min_vote),
            "max_vote": float(args.max_vote),
            "min_target_best_sim": float(args.min_target_best_sim),
            "min_centroid_sim": float(args.min_centroid_sim),
            "min_min_view_sim": float(args.min_min_view_sim),
            "max_same_video_overlap_ratio": float(args.max_same_video_overlap_ratio),
            "min_moved_tracklets": float(args.min_moved_tracklets),
            "origin": str(args.rule_origin),
        },
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(Path(args.csv), selected)
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_rank",
        "source_component_label",
        "target_component",
        "moved_tracklets",
        "target_size",
        "component_graph_rescue_score",
        "no_gt_component_graph_score",
        "target_mean_sim",
        "target_best_sim",
        "target_min_view_sim",
        "target_view_vote",
        "same_video_overlap_ratio",
        "accepted_pair_mass_proxy_sum",
        "selection_rule",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Component-Graph Rescue Rule",
        "",
        f"- raw rows: `{result['raw_count']}`",
        f"- kept rows: `{result['kept_count']}`",
        f"- selected rows: `{len(result['selected'])}`",
        f"- rule origin: `{result['rule']['origin']}`",
        "",
        "| rank | original rank | source | target | moved | rescue | graph | best | centroid | vote | overlap |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(result["selected"], start=1):
        lines.append(
            f"| {rank} | `{row['candidate_rank']}` | `{row['source_component_label']}` | `{row['target_component']}` | "
            f"`{row['moved_tracklets']}` | `{row['component_graph_rescue_score']:.6f}` | "
            f"`{row.get('no_gt_component_graph_score', 0.0):.6f}` | `{row.get('target_best_sim', 0.0):.6f}` | "
            f"`{row.get('target_mean_sim', 0.0):.6f}` | `{row.get('target_view_vote', 0.0):.3f}` | "
            f"`{row.get('same_video_overlap_ratio', 0.0):.6f}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-json", required=True)
    ap.add_argument("--min-vote", type=float, default=0.25)
    ap.add_argument("--max-vote", type=float, default=0.35)
    ap.add_argument("--min-target-best-sim", type=float, default=0.82)
    ap.add_argument("--min-centroid-sim", type=float, default=0.45)
    ap.add_argument("--min-min-view-sim", type=float, default=0.72)
    ap.add_argument("--max-same-video-overlap-ratio", type=float, default=0.020)
    ap.add_argument("--min-moved-tracklets", type=float, default=8.0)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--rule-origin", default="eval_only_opponent_inspired_no_gt_features")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    out = filter_rows(args)
    print(json.dumps({"raw": out["raw_count"], "kept": out["kept_count"], "selected": len(out["selected"])}, sort_keys=True))


if __name__ == "__main__":
    main()
