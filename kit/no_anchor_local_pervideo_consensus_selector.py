#!/usr/bin/env python
"""Consensus wrapper for the local no-GT per-video source selector.

Instead of trusting one threshold setting, this script runs a small grid of
no-GT selector settings and commits only source choices that are stable across
the grid.  This turns selector hyperparameters into a self-play opponent:
unstable per-video decisions are quarantined back to the base assignment.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.no_anchor_local_pervideo_source_selector import (
    _align_gid_maps,
    _load_assignment,
    _parse_source,
    _with_aligned_gid,
    _write_selected_assignment,
    select as base_select,
)


def _floats(text: str) -> list[float]:
    vals = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not vals:
        raise ValueError("empty float grid")
    return vals


def _strings(text: str) -> list[str]:
    vals = [item.strip() for item in str(text).split(",") if item.strip()]
    if not vals:
        raise ValueError("empty string grid")
    return vals


def _run_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    strategies = _strings(args.strategies)
    gains = _floats(args.min_score_gains)
    changed = _floats(args.max_changed_ratios)
    coverages = _floats(args.min_coverages)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for idx, (strategy, gain, change, coverage) in enumerate(
            itertools.product(strategies, gains, changed, coverages),
            start=1,
        ):
            run_args = argparse.Namespace(
                source=list(args.source),
                reference_source=args.reference_source,
                base_source=args.base_source,
                strategy=strategy,
                min_score_gain=float(gain),
                max_changed_ratio=float(change),
                min_coverage=float(coverage),
                assignments_out="",
                json=str(root / f"grid_{idx:03d}.json"),
                csv=None,
                md=None,
            )
            out = base_select(run_args)
            results.append(
                {
                    "grid_rank": idx,
                    "strategy": strategy,
                    "min_score_gain": float(gain),
                    "max_changed_ratio": float(change),
                    "min_coverage": float(coverage),
                    "policy": out["policy"],
                    "decisions": out["decisions"],
                }
            )
    return results


def _consensus_policy(grid: list[dict[str, Any]], *, base_source: str, min_vote_fraction: float) -> tuple[dict[str, str], dict[str, Any]]:
    videos = sorted({video for row in grid for video in row["policy"]})
    policy: dict[str, str] = {}
    details: dict[str, Any] = {}
    total = max(1, len(grid))
    for video in videos:
        votes = Counter(str(row["policy"].get(video, base_source)) for row in grid)
        top_source, top_votes = votes.most_common(1)[0]
        vote_fraction = float(top_votes / total)
        chosen = top_source if vote_fraction >= min_vote_fraction else base_source
        if chosen != top_source:
            reason = "quarantine_unstable_vote"
        elif chosen == base_source:
            reason = "base_consensus"
        else:
            reason = "committed_consensus"
        details[video] = {
            "chosen_source": chosen,
            "top_source": top_source,
            "top_votes": int(top_votes),
            "total_votes": int(total),
            "top_vote_fraction": vote_fraction,
            "vote_counts": dict(sorted(votes.items())),
            "decision_reason": reason,
        }
        policy[video] = chosen
    return policy, details


def _materialize(args: argparse.Namespace, policy: dict[str, str]) -> dict[str, Any]:
    if not args.assignments_out:
        return {}
    sources = dict(_parse_source(spec) for spec in args.source)
    raw = {name: _load_assignment(path) for name, path in sources.items()}
    maps, _stats = _align_gid_maps(raw, args.reference_source)
    aligned = {name: _with_aligned_gid(df, maps[name]) for name, df in raw.items()}
    return _write_selected_assignment(
        Path(args.assignments_out),
        raw=raw,
        aligned=aligned,
        base_source=args.base_source,
        policy=policy,
    )


def _write_csv(path: Path, details: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video",
        "chosen_source",
        "top_source",
        "top_votes",
        "total_votes",
        "top_vote_fraction",
        "decision_reason",
        "vote_counts_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for video, detail in sorted(details.items()):
            writer.writerow(
                {
                    "video": video,
                    **{key: detail.get(key) for key in fields if key not in {"video", "vote_counts_json"}},
                    "vote_counts_json": json.dumps(detail["vote_counts"], sort_keys=True),
                }
            )


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Per-Video Consensus Source Selector",
        "",
        "Production selector: no anchors, no GT, no full-score feedback.",
        "",
        f"- grid runs: `{result['grid_runs']}`",
        f"- min vote fraction: `{result['min_vote_fraction']}`",
        f"- assignment rows: `{result.get('assignment_info', {}).get('assignment_rows', 0)}`",
        f"- predicted IDs: `{result.get('assignment_info', {}).get('predicted_ids', 0)}`",
        "",
        "## Consensus Policy",
        "",
        "| video | chosen | top source | votes | fraction | reason |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for video, detail in sorted(result["consensus"].items()):
        lines.append(
            f"| `{video}` | `{detail['chosen_source']}` | `{detail['top_source']}` | "
            f"`{detail['top_votes']}/{detail['total_votes']}` | "
            f"`{detail['top_vote_fraction']:.3f}` | `{detail['decision_reason']}` |"
        )
    lines.extend(["", "## Grid Settings", ""])
    for row in result["grid"][:30]:
        lines.append(
            f"- rank `{row['grid_rank']}`: strategy `{row['strategy']}`, "
            f"gain `{row['min_score_gain']}`, changed `{row['max_changed_ratio']}`, "
            f"coverage `{row['min_coverage']}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    grid = _run_grid(args)
    policy, consensus = _consensus_policy(
        grid,
        base_source=args.base_source,
        min_vote_fraction=float(args.min_vote_fraction),
    )
    assignment_info = _materialize(args, policy)
    result = {
        "sources": dict(_parse_source(spec) for spec in args.source),
        "reference_source": args.reference_source,
        "base_source": args.base_source,
        "grid_runs": len(grid),
        "grid": grid,
        "min_vote_fraction": float(args.min_vote_fraction),
        "policy": policy,
        "consensus": consensus,
        "assignment_info": assignment_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "uses_full_score_feedback": False,
    }
    result["sources"] = {key: str(value) for key, value in result["sources"].items()}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), consensus)
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            {"seq": 1, "tracklet_key": "a", "video": "v1", "predicted_global_id": 10, "component_label": 10, "avg_conf": 0.9, "prediction_confidence": 0.7, "component_margin_prob": 0.1},
            {"seq": 2, "tracklet_key": "b", "video": "v1", "predicted_global_id": 11, "component_label": 11, "avg_conf": 0.8, "prediction_confidence": 0.7, "component_margin_prob": 0.1},
            {"seq": 3, "tracklet_key": "c", "video": "v2", "predicted_global_id": 12, "component_label": 12, "avg_conf": 0.7, "prediction_confidence": 0.7, "component_margin_prob": 0.1},
        ]
        base = root / "base.csv"
        cand = root / "cand.csv"
        pd.DataFrame(rows).to_csv(base, index=False)
        crows = [dict(row) for row in rows]
        crows[0]["predicted_global_id"] = 20
        crows[1]["predicted_global_id"] = 20
        crows[0]["component_margin_prob"] = 0.9
        crows[1]["component_margin_prob"] = 0.9
        pd.DataFrame(crows).to_csv(cand, index=False)
        out = run(
            argparse.Namespace(
                source=[f"base:{base}", f"cand:{cand}"],
                reference_source="base",
                base_source="base",
                strategies="conservative,balanced",
                min_score_gains="0.0,0.01",
                max_changed_ratios="1.0",
                min_coverages="0.9",
                min_vote_fraction=0.50,
                assignments_out=str(root / "assign.csv"),
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
                md=str(root / "out.md"),
            )
        )
        assert out["policy"]["v1"] == "cand", out
        assert out["policy"]["v2"] == "base", out
        assert Path(root / "assign.csv").is_file()
        assert "Consensus Policy" in Path(root / "out.md").read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", default=[], help="name:/path/to/assignments.csv")
    ap.add_argument("--reference-source", required=False)
    ap.add_argument("--base-source", required=False)
    ap.add_argument("--strategies", default="conservative,balanced")
    ap.add_argument("--min-score-gains", default="0.002,0.006,0.012,0.018")
    ap.add_argument("--max-changed-ratios", default="0.08,0.12,0.16,0.22")
    ap.add_argument("--min-coverages", default="0.96,0.98,0.995")
    ap.add_argument("--min-vote-fraction", type=float, default=0.70)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--json", default="local_runs/no_anchor_local_pervideo_consensus_selector_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_local_pervideo_consensus_selector_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_local_pervideo_consensus_selector_20260620.md")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    if not args.source:
        ap.error("--source is required")
    if not args.reference_source:
        ap.error("--reference-source is required")
    if not args.base_source:
        ap.error("--base-source is required")
    result = run(args)
    print(
        json.dumps(
            {
                "grid_runs": result["grid_runs"],
                "policy": result["policy"],
                "assignment_info": result["assignment_info"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
