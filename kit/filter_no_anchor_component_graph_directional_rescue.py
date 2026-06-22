#!/usr/bin/env python
"""Direction-aware no-GT filter for component-graph rescue candidates.

The low-vote component-graph rescue can emit both directions for the same
component pair.  This filter keeps one direction per unordered pair using only
candidate metadata: prefer moving into a larger/stabler target with strong
visual support and bounded same-video overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any) -> int:
    return int(float(value))


def _rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("selected") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} is missing selected[] or rows[]")
    return [{**row, "_input_rank": idx} for idx, row in enumerate(rows, start=1) if isinstance(row, dict)]


def _direction_score(row: dict[str, Any]) -> float:
    source_size = max(_as_float(row.get("source_size")), 1.0)
    target_size = max(_as_float(row.get("target_size")), 1.0)
    size_bonus = min(math.log(target_size / source_size + 1.0), 0.75)
    target_best = _as_float(row.get("target_best_sim"))
    min_view = _as_float(row.get("target_min_view_sim"))
    target_quality = _as_float(row.get("target_quality"))
    source_quality = _as_float(row.get("source_quality"))
    vote = _as_float(row.get("target_view_vote"))
    overlap = _as_float(row.get("same_video_overlap_ratio"))
    mass = min(math.log1p(source_size * target_size) / math.log(100000.0), 1.0)
    return float(
        0.34 * target_best
        + 0.18 * min_view
        + 0.16 * size_bonus
        + 0.12 * max(target_quality - source_quality, -0.20)
        + 0.10 * mass
        + 0.08 * max(0.0, 1.0 - abs(vote - 0.30) / 0.25)
        - 0.45 * overlap
    )


def _passes(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        _as_float(row.get("target_size")) >= _as_float(row.get("source_size")) * float(args.min_target_source_size_ratio)
        and _as_float(row.get("target_best_sim")) >= float(args.min_target_best_sim)
        and _as_float(row.get("target_min_view_sim")) >= float(args.min_target_min_view_sim)
        and _as_float(row.get("same_video_overlap_ratio")) <= float(args.max_same_video_overlap_ratio)
    )


def filter_rows(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _rows(Path(args.candidates_json))
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in candidates:
        source = _as_int(row.get("source_component_label"))
        target = _as_int(row.get("target_component"))
        key = tuple(sorted((source, target)))
        new = dict(row)
        new["component_graph_direction_score"] = _direction_score(new)
        new["selection_rule"] = "directional_larger_target_high_sim_rescue"
        new["selection_rule_origin"] = "no_gt_directionality_filter"
        grouped.setdefault(key, []).append(new)

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda row: float(row["component_graph_direction_score"]), reverse=True)
        best = ordered[0]
        if _passes(best, args):
            selected.append(best)
            rejected.extend(ordered[1:])
        else:
            rejected.extend(ordered)

    selected.sort(
        key=lambda row: (
            float(row["component_graph_direction_score"]),
            float(row.get("accepted_pair_mass_proxy_sum", 0.0)),
        ),
        reverse=True,
    )
    selected = selected[: int(args.top_n)]
    result = {
        "candidates_json": str(args.candidates_json),
        "raw_count": len(candidates),
        "pair_groups": len(grouped),
        "selected": selected,
        "top_rejected": sorted(rejected, key=lambda row: float(row["component_graph_direction_score"]), reverse=True)[:20],
        "rule": {
            "min_target_source_size_ratio": float(args.min_target_source_size_ratio),
            "min_target_best_sim": float(args.min_target_best_sim),
            "min_target_min_view_sim": float(args.min_target_min_view_sim),
            "max_same_video_overlap_ratio": float(args.max_same_video_overlap_ratio),
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
        "_input_rank",
        "_source_rank",
        "source_component_label",
        "target_component",
        "moved_tracklets",
        "source_size",
        "target_size",
        "target_best_sim",
        "target_min_view_sim",
        "same_video_overlap_ratio",
        "component_graph_direction_score",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Component-Graph Directional Rescue",
        "",
        f"- source: `{result['candidates_json']}`",
        f"- raw rows: `{result['raw_count']}`",
        f"- pair groups: `{result['pair_groups']}`",
        f"- selected: `{len(result['selected'])}`",
        "",
        "| rank | input rank | source | target | moved | source size | target size | best | min-view | overlap | score |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(result["selected"][:30], start=1):
        lines.append(
            f"| {rank} | `{row.get('_input_rank')}` | `{row.get('source_component_label')}` | `{row.get('target_component')}` | "
            f"`{row.get('moved_tracklets')}` | `{row.get('source_size')}` | `{row.get('target_size')}` | "
            f"`{float(row.get('target_best_sim', 0.0)):.6f}` | `{float(row.get('target_min_view_sim', 0.0)):.6f}` | "
            f"`{float(row.get('same_video_overlap_ratio', 0.0)):.6f}` | `{float(row['component_graph_direction_score']):.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a no-GT production-side directionality rule.",
            "- It is intended to ablate broad bidirectional rescue by keeping one larger-target direction per unordered pair.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "candidates.json"
        src.write_text(
            json.dumps(
                {
                    "selected": [
                        {
                            "source_component_label": 1,
                            "target_component": 2,
                            "source_size": 10,
                            "target_size": 20,
                            "target_best_sim": 0.90,
                            "target_min_view_sim": 0.75,
                            "same_video_overlap_ratio": 0.01,
                        },
                        {
                            "source_component_label": 2,
                            "target_component": 1,
                            "source_size": 20,
                            "target_size": 10,
                            "target_best_sim": 0.90,
                            "target_min_view_sim": 0.75,
                            "same_video_overlap_ratio": 0.01,
                        },
                    ]
                }
            )
        )
        out = filter_rows(
            argparse.Namespace(
                candidates_json=str(src),
                min_target_source_size_ratio=1.0,
                min_target_best_sim=0.82,
                min_target_min_view_sim=0.70,
                max_same_video_overlap_ratio=0.02,
                top_n=10,
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
                md=str(root / "out.md"),
            )
        )
        assert len(out["selected"]) == 1, out
        assert out["selected"][0]["source_component_label"] == 1, out
        assert Path(root / "out.csv").read_text()
        assert "larger-target" in Path(root / "out.md").read_text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidates-json", default="local_runs/no_anchor_component_graph_low_vote_rescue_broad_20260620.json")
    ap.add_argument("--min-target-source-size-ratio", type=float, default=1.0)
    ap.add_argument("--min-target-best-sim", type=float, default=0.82)
    ap.add_argument("--min-target-min-view-sim", type=float, default=0.70)
    ap.add_argument("--max-same-video-overlap-ratio", type=float, default=0.02)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--json", default="local_runs/no_anchor_component_graph_directional_rescue_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_component_graph_directional_rescue_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_component_graph_directional_rescue_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    out = filter_rows(args)
    print(json.dumps({"raw": out["raw_count"], "pair_groups": out["pair_groups"], "selected": len(out["selected"])}, sort_keys=True))


if __name__ == "__main__":
    main()
