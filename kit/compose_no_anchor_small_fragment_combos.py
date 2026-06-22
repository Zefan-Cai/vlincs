#!/usr/bin/env python
"""Compose no-anchor small-fragment edits into non-conflicting combo rows."""

from __future__ import annotations

import argparse
import csv
import itertools
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
    rows = data.get("selected") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path} is missing selected[] rows")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _source_seq_set(row: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for item in row.get("accepted_preview", []):
        if not isinstance(item, dict):
            continue
        for seq in item.get("source_seqs", []):
            out.add(int(float(seq)))
    return out


def _score_combo(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    small_scores = [_as_float(row.get("component_graph_small_fragment_score")) for row in rows]
    graph_scores = [_as_float(row.get("no_gt_component_graph_score")) for row in rows]
    best_sims = [_as_float(row.get("target_best_sim")) for row in rows]
    overlaps = [_as_float(row.get("same_video_overlap_ratio")) for row in rows]
    moved = sum(_as_float(row.get("moved_tracklets")) for row in rows)
    diversity_bonus = min(len(rows) / 4.0, 1.0) * 0.015
    impact_bonus = min(math.log1p(moved) / math.log(32.0), 1.0) * 0.020
    overlap_penalty = min(sum(overlaps) / max(len(overlaps), 1) / 0.012, 1.0) * 0.020
    return float(
        0.42 * min(small_scores)
        + 0.22 * (sum(small_scores) / len(small_scores))
        + 0.16 * min(graph_scores)
        + 0.14 * min(best_sims)
        + diversity_bonus
        + impact_bonus
        - overlap_penalty
    )


def _combo_row(rows: list[dict[str, Any]], rank: int, source_path: Path) -> dict[str, Any]:
    preview = []
    source_ranks = []
    source_components = []
    target_components = []
    for row in rows:
        source_ranks.append(int(row.get("candidate_rank") or row.get("_source_rank") or rank))
        source_components.append(int(row.get("source_component_label")))
        target_components.append(int(row.get("target_component")))
        for item in row.get("accepted_preview", []):
            if isinstance(item, dict):
                preview.append(dict(item))
    moved = sum(int(float(row.get("moved_tracklets", 0))) for row in rows)
    combo_score = _score_combo(rows)
    return {
        "_source_file": str(source_path),
        "_source_rank": rank,
        "mode": "small_fragment_combo",
        "accepted_preview": preview,
        "accepted_reassignments": int(len(preview)),
        "accepted_edges": int(len(preview)),
        "moved_tracklets": int(moved),
        "source_candidate_ranks": source_ranks,
        "source_components": source_components,
        "target_components": target_components,
        "component_graph_combo_score": combo_score,
        "component_graph_small_fragment_score_min": min(_as_float(row.get("component_graph_small_fragment_score")) for row in rows),
        "component_graph_small_fragment_score_mean": sum(_as_float(row.get("component_graph_small_fragment_score")) for row in rows) / len(rows),
        "no_gt_component_graph_score_min": min(_as_float(row.get("no_gt_component_graph_score")) for row in rows),
        "target_best_sim_min": min(_as_float(row.get("target_best_sim")) for row in rows),
        "same_video_overlap_ratio_mean": sum(_as_float(row.get("same_video_overlap_ratio")) for row in rows) / len(rows),
        "signature": repr(("small_fragment_combo", tuple(source_ranks), tuple(source_components), tuple(target_components))),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }


def compose(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(Path(args.small_fragment_json))
    rows = rows[: int(args.max_input_rows)]
    combos: list[dict[str, Any]] = []
    for size in range(int(args.min_combo_size), int(args.max_combo_size) + 1):
        for members in itertools.combinations(rows, size):
            used: set[int] = set()
            ok = True
            for row in members:
                seqs = _source_seq_set(row)
                if not seqs or used.intersection(seqs):
                    ok = False
                    break
                used.update(seqs)
            if not ok:
                continue
            combos.append(_combo_row(list(members), len(combos) + 1, Path(args.small_fragment_json)))
    combos.sort(key=lambda row: (float(row["component_graph_combo_score"]), int(row["moved_tracklets"])), reverse=True)
    selected = combos[: int(args.top_n)]
    for idx, row in enumerate(selected, start=1):
        row["_source_rank"] = idx
    result = {
        "small_fragment_json": str(args.small_fragment_json),
        "raw_rows": int(len(rows)),
        "combo_count": int(len(combos)),
        "selected": selected,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "rule": {
            "min_combo_size": int(args.min_combo_size),
            "max_combo_size": int(args.max_combo_size),
            "max_input_rows": int(args.max_input_rows),
            "top_n": int(args.top_n),
        },
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
        "source_candidate_ranks",
        "source_components",
        "target_components",
        "moved_tracklets",
        "accepted_edges",
        "component_graph_combo_score",
        "component_graph_small_fragment_score_min",
        "component_graph_small_fragment_score_mean",
        "no_gt_component_graph_score_min",
        "target_best_sim_min",
        "same_video_overlap_ratio_mean",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Small-Fragment Combo Candidates",
        "",
        f"- raw rows: `{result['raw_rows']}`",
        f"- combo rows: `{result['combo_count']}`",
        f"- selected rows: `{len(result['selected'])}`",
        "",
        "| rank | source ranks | source comps | target comps | moved | edges | combo score | min small | min graph | min best sim | overlap mean |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(result["selected"], start=1):
        lines.append(
            f"| {rank} | `{row['source_candidate_ranks']}` | `{row['source_components']}` | `{row['target_components']}` | "
            f"`{row['moved_tracklets']}` | `{row['accepted_edges']}` | `{row['component_graph_combo_score']:.6f}` | "
            f"`{row['component_graph_small_fragment_score_min']:.6f}` | `{row['no_gt_component_graph_score_min']:.6f}` | "
            f"`{row['target_best_sim_min']:.6f}` | `{row['same_video_overlap_ratio_mean']:.6f}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small-fragment-json", required=True)
    ap.add_argument("--min-combo-size", type=int, default=2)
    ap.add_argument("--max-combo-size", type=int, default=3)
    ap.add_argument("--max-input-rows", type=int, default=6)
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    out = compose(args)
    print(json.dumps({"combos": out["combo_count"], "selected": len(out["selected"])}, sort_keys=True))


if __name__ == "__main__":
    main()
