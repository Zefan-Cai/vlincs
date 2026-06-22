#!/usr/bin/env python
"""Filter component-graph candidates with a no-GT target-impurity proxy.

The previous component-graph sweep showed a counterintuitive failure mode:
high-vote/high-rank graph rows are often overconfident false merges, while a
few useful bridge rows are buried because the global graph score is modest.
This filter keeps only rows with strong local visual evidence but low global
graph confidence, then collapses opposite directions without reading GT.
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


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no rows[]")
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            out.append({"candidate_rank": rank, "_source_file": str(path), "_source_rank": rank, **row})
    return out


def _unordered_pair(row: dict[str, Any]) -> tuple[int, int]:
    return tuple(sorted((_as_int(row["source_component_label"]), _as_int(row["target_component"]))))


def _risk_proxy(row: dict[str, Any]) -> float:
    graph = _as_float(row.get("no_gt_component_graph_score"))
    vote = _as_float(row.get("target_view_vote"))
    overlap = min(_as_float(row.get("same_video_overlap_ratio")) / 0.02, 1.5)
    size = min(math.log1p(max(_as_float(row.get("source_size")), _as_float(row.get("target_size")))) / math.log(300), 1.0)
    # High graph consensus plus broad view vote is treated as overconfident
    # target-absorption risk.  The default filter reports this score but relies
    # mainly on the max-graph gate, because some true low-score bridges still
    # have broad view support.
    return float(0.45 * graph + 0.25 * vote + 0.20 * overlap + 0.10 * size)


def _opportunity_proxy(row: dict[str, Any]) -> float:
    best = _as_float(row.get("target_best_sim"))
    graph = _as_float(row.get("no_gt_component_graph_score"))
    min_view = _as_float(row.get("target_min_view_sim"))
    overlap = _as_float(row.get("same_video_overlap_ratio"))
    moved = min(math.log1p(_as_float(row.get("moved_tracklets"))) / math.log(256.0), 1.0)
    # Useful low-vote bridges look locally strong but globally under-ranked.
    return float(0.52 * best + 0.18 * min_view + 0.14 * max(best - graph, 0.0) + 0.10 * moved - 0.06 * min(overlap / 0.02, 1.0))


def _direction_score(row: dict[str, Any]) -> float:
    source_size = max(_as_float(row.get("source_size")), 1.0)
    target_size = max(_as_float(row.get("target_size")), 1.0)
    size_bonus = min(max(math.log((target_size / source_size) + 1.0), 0.0), 0.8)
    return float(
        0.38 * _as_float(row.get("target_best_sim"))
        + 0.22 * _as_float(row.get("target_min_view_sim"))
        + 0.15 * size_bonus
        + 0.15 * _opportunity_proxy(row)
        - 0.35 * _as_float(row.get("same_video_overlap_ratio"))
    )


def _passes(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        _as_float(row.get("target_best_sim")) >= float(args.min_target_best_sim)
        and _as_float(row.get("target_min_view_sim")) >= float(args.min_min_view_sim)
        and _as_float(row.get("no_gt_component_graph_score")) <= float(args.max_graph_score)
        and _as_float(row.get("same_video_overlap_ratio")) <= float(args.max_same_video_overlap_ratio)
        and _as_float(row.get("moved_tracklets")) >= float(args.min_moved_tracklets)
        and _risk_proxy(row) <= float(args.max_risk_proxy)
    )


def filter_rows(args: argparse.Namespace) -> dict[str, Any]:
    raw = _load_rows(Path(args.candidates_json))
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in raw:
        new = dict(row)
        new["target_impurity_risk_proxy"] = _risk_proxy(new)
        new["target_impurity_opportunity_proxy"] = _opportunity_proxy(new)
        new["target_impurity_direction_score"] = _direction_score(new)
        new["selection_rule"] = "low_graph_score_high_local_visual_target_impurity_proxy"
        new["uses_anchors"] = False
        new["uses_gt_for_training_or_anchors"] = False
        new["uses_gt_for_evaluation_only"] = False
        if _passes(new, args):
            kept.append(new)
        else:
            rejected.append(new)

    best_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    duplicate_directions: list[dict[str, Any]] = []
    for row in sorted(kept, key=lambda item: float(item["target_impurity_direction_score"]), reverse=True):
        key = _unordered_pair(row)
        if key not in best_by_pair:
            best_by_pair[key] = row
        else:
            duplicate_directions.append({**row, "selection_rule": "rejected_opposite_direction"})
    selected = sorted(best_by_pair.values(), key=lambda item: float(item["target_impurity_opportunity_proxy"]), reverse=True)[
        : int(args.top_n)
    ]
    result = {
        "candidates_json": str(args.candidates_json),
        "raw_count": len(raw),
        "kept_direction_count": len(kept),
        "selected_pair_count": len(selected),
        "selected": selected,
        "rejected_direction_duplicates": duplicate_directions,
        "top_rejected": sorted(rejected, key=lambda item: float(item["target_impurity_opportunity_proxy"]), reverse=True)[:20],
        "rule": vars(args),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        "source_size",
        "target_size",
        "no_gt_component_graph_score",
        "target_impurity_risk_proxy",
        "target_impurity_opportunity_proxy",
        "target_impurity_direction_score",
        "target_best_sim",
        "target_mean_sim",
        "target_min_view_sim",
        "target_view_vote",
        "same_video_overlap_ratio",
        "selection_rule",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Component-Graph Target-Impurity Proxy Filter",
        "",
        "Production filter uses no GT. Eval labels should be added only by a separate opponent audit.",
        "",
        f"- raw rows: `{result['raw_count']}`",
        f"- kept directions: `{result['kept_direction_count']}`",
        f"- selected unordered pairs: `{result['selected_pair_count']}`",
        f"- rejected opposite directions: `{len(result['rejected_direction_duplicates'])}`",
        "",
        "| rank | original | source | target | moved | graph | risk | opportunity | direction | best | min-view | vote | overlap |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(result["selected"], start=1):
        lines.append(
            f"| {rank} | `{row['candidate_rank']}` | `{row['source_component_label']}` | `{row['target_component']}` | "
            f"`{row.get('moved_tracklets')}` | `{_as_float(row.get('no_gt_component_graph_score')):.6f}` | "
            f"`{_as_float(row.get('target_impurity_risk_proxy')):.6f}` | "
            f"`{_as_float(row.get('target_impurity_opportunity_proxy')):.6f}` | "
            f"`{_as_float(row.get('target_impurity_direction_score')):.6f}` | "
            f"`{_as_float(row.get('target_best_sim')):.6f}` | `{_as_float(row.get('target_min_view_sim')):.6f}` | "
            f"`{_as_float(row.get('target_view_vote')):.3f}` | `{_as_float(row.get('same_video_overlap_ratio')):.6f}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "rows.json"
        src.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "source_component_label": 1,
                            "target_component": 2,
                            "source_size": 10,
                            "target_size": 20,
                            "moved_tracklets": 10,
                            "target_best_sim": 0.86,
                            "target_min_view_sim": 0.74,
                            "target_mean_sim": 0.50,
                            "target_view_vote": 0.3,
                            "same_video_overlap_ratio": 0.01,
                            "no_gt_component_graph_score": 0.64,
                        },
                        {
                            "source_component_label": 2,
                            "target_component": 1,
                            "source_size": 20,
                            "target_size": 10,
                            "moved_tracklets": 20,
                            "target_best_sim": 0.86,
                            "target_min_view_sim": 0.74,
                            "target_mean_sim": 0.50,
                            "target_view_vote": 0.3,
                            "same_video_overlap_ratio": 0.01,
                            "no_gt_component_graph_score": 0.64,
                        },
                        {
                            "source_component_label": 3,
                            "target_component": 4,
                            "source_size": 10,
                            "target_size": 100,
                            "moved_tracklets": 10,
                            "target_best_sim": 0.90,
                            "target_min_view_sim": 0.90,
                            "target_mean_sim": 0.70,
                            "target_view_vote": 1.0,
                            "same_video_overlap_ratio": 0.019,
                            "no_gt_component_graph_score": 0.78,
                        },
                    ]
                }
            )
        )
        out = filter_rows(
            argparse.Namespace(
                candidates_json=str(src),
                min_target_best_sim=0.78,
                min_min_view_sim=0.70,
                max_graph_score=0.665,
                max_same_video_overlap_ratio=0.02,
                min_moved_tracklets=8,
                max_risk_proxy=0.75,
                top_n=8,
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
                md=str(root / "out.md"),
            )
        )
        assert out["kept_direction_count"] == 2, out
        assert out["selected_pair_count"] == 1, out
        assert out["selected"][0]["target_component"] == 2, out
        assert Path(root / "out.csv").read_text()
        assert "no GT" in Path(root / "out.md").read_text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidates-json", default="local_runs/no_anchor_component_graph_high_mass_candidates_20260620.json")
    ap.add_argument("--min-target-best-sim", type=float, default=0.78)
    ap.add_argument("--min-min-view-sim", type=float, default=0.70)
    ap.add_argument("--max-graph-score", type=float, default=0.665)
    ap.add_argument("--max-same-video-overlap-ratio", type=float, default=0.020)
    ap.add_argument("--min-moved-tracklets", type=float, default=8)
    ap.add_argument("--max-risk-proxy", type=float, default=1.0)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--json", default="local_runs/no_anchor_component_graph_target_impurity_proxy_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_component_graph_target_impurity_proxy_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_component_graph_target_impurity_proxy_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    out = filter_rows(args)
    print(json.dumps({"raw": out["raw_count"], "kept": out["kept_direction_count"], "selected": out["selected_pair_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
