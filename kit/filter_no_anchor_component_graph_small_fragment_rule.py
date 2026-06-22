#!/usr/bin/env python
"""Select small-fragment no-anchor component-graph attachments."""

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
    return [{"candidate_rank": rank, "_source_file": str(path), "_source_rank": rank, **row} for rank, row in enumerate(rows, start=1) if isinstance(row, dict)]


def _small_fragment_score(row: dict[str, Any]) -> float:
    moved = max(_as_float(row.get("moved_tracklets")), 1.0)
    target_size = max(_as_float(row.get("target_size")), 1.0)
    small_bonus = 1.0 - min(max(moved - 1.0, 0.0) / 10.0, 1.0)
    size_ratio = min(target_size / moved / 32.0, 1.0)
    overlap_penalty = min(_as_float(row.get("same_video_overlap_ratio")) / 0.012, 1.0)
    return float(
        0.30 * _as_float(row.get("target_best_sim"))
        + 0.22 * _as_float(row.get("target_mean_sim"))
        + 0.18 * _as_float(row.get("target_min_view_sim"))
        + 0.12 * small_bonus
        + 0.10 * size_ratio
        + 0.08 * (1.0 - overlap_penalty)
    )


def _keep(row: dict[str, Any], args: argparse.Namespace) -> bool:
    moved = _as_float(row.get("moved_tracklets"))
    target_size = _as_float(row.get("target_size"))
    source_size = _as_float(row.get("source_size"))
    return (
        float(args.min_moved_tracklets) <= moved <= float(args.max_moved_tracklets)
        and target_size >= float(args.min_target_size)
        and target_size >= max(source_size, 1.0) * float(args.min_target_to_source_ratio)
        and _as_float(row.get("target_best_sim")) >= float(args.min_target_best_sim)
        and _as_float(row.get("target_mean_sim")) >= float(args.min_centroid_sim)
        and _as_float(row.get("target_min_view_sim")) >= float(args.min_min_view_sim)
        and _as_float(row.get("same_video_overlap_ratio")) <= float(args.max_same_video_overlap_ratio)
    )


def filter_rows(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(Path(args.candidates_json))
    kept = []
    rejected = []
    for row in rows:
        new_row = dict(row)
        new_row["component_graph_small_fragment_score"] = _small_fragment_score(new_row)
        new_row["selection_rule"] = "small_fragment_high_similarity_attachment"
        new_row["selection_rule_origin"] = str(args.rule_origin)
        new_row["uses_anchors"] = False
        new_row["uses_gt_for_training_or_anchors"] = False
        new_row["uses_gt_for_evaluation_only"] = False
        if _keep(new_row, args):
            kept.append(new_row)
        else:
            rejected.append(new_row)
    kept.sort(key=lambda row: (float(row["component_graph_small_fragment_score"]), float(row.get("target_best_sim", 0.0))), reverse=True)
    selected = kept[: int(args.top_n)]
    result = {
        "candidates_json": str(args.candidates_json),
        "raw_count": int(len(rows)),
        "kept_count": int(len(kept)),
        "selected": selected,
        "top_rejected": sorted(rejected, key=lambda row: float(row["component_graph_small_fragment_score"]), reverse=True)[:20],
        "rule": {
            "min_moved_tracklets": float(args.min_moved_tracklets),
            "max_moved_tracklets": float(args.max_moved_tracklets),
            "min_target_size": float(args.min_target_size),
            "min_target_to_source_ratio": float(args.min_target_to_source_ratio),
            "min_target_best_sim": float(args.min_target_best_sim),
            "min_centroid_sim": float(args.min_centroid_sim),
            "min_min_view_sim": float(args.min_min_view_sim),
            "max_same_video_overlap_ratio": float(args.max_same_video_overlap_ratio),
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
        "component_graph_small_fragment_score",
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
        "# Component-Graph Small-Fragment Rule",
        "",
        f"- raw rows: `{result['raw_count']}`",
        f"- kept rows: `{result['kept_count']}`",
        f"- selected rows: `{len(result['selected'])}`",
        f"- rule origin: `{result['rule']['origin']}`",
        "",
        "| rank | original rank | source | target | moved | target size | small score | graph | best | centroid | overlap |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(result["selected"], start=1):
        lines.append(
            f"| {rank} | `{row['candidate_rank']}` | `{row['source_component_label']}` | `{row['target_component']}` | "
            f"`{row['moved_tracklets']}` | `{row.get('target_size', 0)}` | `{row['component_graph_small_fragment_score']:.6f}` | "
            f"`{row.get('no_gt_component_graph_score', 0.0):.6f}` | `{row.get('target_best_sim', 0.0):.6f}` | "
            f"`{row.get('target_mean_sim', 0.0):.6f}` | `{row.get('same_video_overlap_ratio', 0.0):.6f}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-json", required=True)
    ap.add_argument("--min-moved-tracklets", type=float, default=2.0)
    ap.add_argument("--max-moved-tracklets", type=float, default=6.0)
    ap.add_argument("--min-target-size", type=float, default=40.0)
    ap.add_argument("--min-target-to-source-ratio", type=float, default=8.0)
    ap.add_argument("--min-target-best-sim", type=float, default=0.68)
    ap.add_argument("--min-centroid-sim", type=float, default=0.62)
    ap.add_argument("--min-min-view-sim", type=float, default=0.58)
    ap.add_argument("--max-same-video-overlap-ratio", type=float, default=0.011)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--rule-origin", default="current_k3_no_gt_small_fragment_attachment")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    out = filter_rows(args)
    print(json.dumps({"raw": out["raw_count"], "kept": out["kept_count"], "selected": len(out["selected"])}, sort_keys=True))


if __name__ == "__main__":
    main()
