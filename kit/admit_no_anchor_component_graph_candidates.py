#!/usr/bin/env python
"""Admission/quarantine layer for no-anchor component-graph candidates.

This is a Deli-style opponent/referee step: the component-graph worker proposes
bridges, but admission decides which rows are safe enough to spend canonical
full-score budget on.  It uses only candidate metadata and writes explicit
states instead of forcing every candidate into a single ranked list.
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
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("selected") or data.get("rows") or data.get("candidates") or []
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a row list")
    out = []
    for rank, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            out.append({"_input_rank": rank, **row})
    return out


def _unordered_pair(row: dict[str, Any]) -> tuple[int, int]:
    a = _as_int(row.get("source_component_label"))
    b = _as_int(row.get("target_component"))
    return tuple(sorted((a, b)))


def _direction_score(row: dict[str, Any]) -> float:
    source_size = max(_as_float(row.get("source_size")), 1.0)
    target_size = max(_as_float(row.get("target_size")), 1.0)
    target_best = _as_float(row.get("target_best_sim"))
    min_view = _as_float(row.get("target_min_view_sim"))
    target_quality = _as_float(row.get("target_quality"))
    source_quality = _as_float(row.get("source_quality"))
    overlap = _as_float(row.get("same_video_overlap_ratio"))
    size_ratio = target_size / source_size
    size_bonus = min(max(math.log(size_ratio + 1.0), 0.0), 0.80)
    return float(
        0.35 * target_best
        + 0.22 * min_view
        + 0.13 * size_bonus
        + 0.12 * max(target_quality - source_quality, -0.25)
        + 0.10 * min(math.log1p(source_size * target_size) / math.log(100000.0), 1.0)
        - 0.50 * overlap
    )


def _evidence_flags(row: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    flags: list[str] = []
    risks: list[str] = []
    target_best = _as_float(row.get("target_best_sim"))
    min_view = _as_float(row.get("target_min_view_sim"))
    mean_view = _as_float(row.get("target_mean_sim"))
    vote = _as_float(row.get("target_view_vote"))
    overlap = _as_float(row.get("same_video_overlap_ratio"))
    source_size = max(_as_float(row.get("source_size")), 1.0)
    target_size = max(_as_float(row.get("target_size")), 1.0)
    if target_best >= args.commit_best_sim:
        flags.append("strong_top_visual_match")
    elif target_best >= args.provisional_best_sim:
        flags.append("medium_top_visual_match")
    else:
        risks.append("weak_top_visual_match")
    if min_view >= args.commit_min_view_sim:
        flags.append("multi_view_support")
    elif min_view >= args.provisional_min_view_sim:
        flags.append("borderline_multi_view_support")
    else:
        risks.append("weak_multi_view_support")
    if overlap <= args.max_commit_overlap:
        flags.append("low_same_video_overlap")
    elif overlap <= args.max_provisional_overlap:
        flags.append("borderline_same_video_overlap")
    else:
        risks.append("same_video_overlap_risk")
    if target_size >= source_size * args.min_commit_target_source_ratio:
        flags.append("larger_or_equal_target")
    elif target_size >= source_size * args.min_provisional_target_source_ratio:
        flags.append("target_not_much_smaller")
    else:
        risks.append("small_target_absorption_risk")
    if mean_view < args.spiky_mean_sim and vote <= args.low_vote_max:
        risks.append("spiky_low_vote_evidence")
    return flags, risks


def _admit(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    flags, risks = _evidence_flags(row, args)
    target_best = _as_float(row.get("target_best_sim"))
    min_view = _as_float(row.get("target_min_view_sim"))
    overlap = _as_float(row.get("same_video_overlap_ratio"))
    source_size = max(_as_float(row.get("source_size")), 1.0)
    target_size = max(_as_float(row.get("target_size")), 1.0)
    direction = _direction_score(row)

    commit = (
        target_best >= args.commit_best_sim
        and min_view >= args.commit_min_view_sim
        and overlap <= args.max_commit_overlap
        and target_size >= source_size * args.min_commit_target_source_ratio
    )
    provisional = (
        target_best >= args.provisional_best_sim
        and min_view >= args.provisional_min_view_sim
        and overlap <= args.max_provisional_overlap
        and target_size >= source_size * args.min_provisional_target_source_ratio
    )
    status = "committed_probe" if commit else "provisional_probe" if provisional else "quarantine"
    if "same_video_overlap_risk" in risks or "weak_multi_view_support" in risks:
        status = "quarantine"
    return {
        **row,
        "admission_status": status,
        "admission_score": direction,
        "admission_flags": flags,
        "admission_risks": risks,
        "admission_reason": ";".join(flags + risks),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }


def admit_rows(args: argparse.Namespace) -> dict[str, Any]:
    raw = _load_rows(Path(args.candidates_json))
    annotated = [_admit(row, args) for row in raw]

    best_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    rejected_direction: list[dict[str, Any]] = []
    for row in sorted(annotated, key=lambda item: float(item["admission_score"]), reverse=True):
        key = _unordered_pair(row)
        if key not in best_by_pair:
            best_by_pair[key] = row
        else:
            rejected_direction.append({**row, "admission_status": "rejected_direction_duplicate"})

    admitted = sorted(best_by_pair.values(), key=lambda item: float(item["admission_score"]), reverse=True)
    selected = [row for row in admitted if row["admission_status"] == "committed_probe"][: int(args.top_n)]
    provisional = [row for row in admitted if row["admission_status"] == "provisional_probe"][: int(args.top_n)]
    quarantine = [row for row in admitted if row["admission_status"] == "quarantine"]
    result = {
        "candidates_json": str(args.candidates_json),
        "raw_count": len(raw),
        "pair_groups": len(best_by_pair),
        "selected": selected,
        "provisional": provisional,
        "quarantine": quarantine,
        "rejected_direction_duplicates": rejected_direction,
        "all_admitted": admitted,
        "rule": vars(args),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "note": "Admission status is for full-score budget triage; it is not a canonical e2e score.",
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(Path(args.csv), admitted)
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "_input_rank",
        "admission_status",
        "source_component_label",
        "target_component",
        "moved_tracklets",
        "source_size",
        "target_size",
        "target_best_sim",
        "target_mean_sim",
        "target_min_view_sim",
        "target_view_vote",
        "same_video_overlap_ratio",
        "admission_score",
        "admission_reason",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Component-Graph Admission",
        "",
        f"- source: `{result['candidates_json']}`",
        f"- raw rows: `{result['raw_count']}`",
        f"- unordered pair groups: `{result['pair_groups']}`",
        f"- committed probes: `{len(result['selected'])}`",
        f"- provisional probes: `{len(result['provisional'])}`",
        f"- quarantined groups: `{len(result['quarantine'])}`",
        f"- rejected direction duplicates: `{len(result['rejected_direction_duplicates'])}`",
        "",
        "| status | input | source | target | moved | best | mean | min-view | vote | overlap | score | reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["all_admitted"][:40]:
        lines.append(
            f"| `{row['admission_status']}` | `{row.get('_input_rank')}` | `{row.get('source_component_label')}` | "
            f"`{row.get('target_component')}` | `{row.get('moved_tracklets')}` | "
            f"`{_as_float(row.get('target_best_sim')):.6f}` | `{_as_float(row.get('target_mean_sim')):.6f}` | "
            f"`{_as_float(row.get('target_min_view_sim')):.6f}` | `{_as_float(row.get('target_view_vote')):.3f}` | "
            f"`{_as_float(row.get('same_video_overlap_ratio')):.6f}` | `{_as_float(row.get('admission_score')):.6f}` | "
            f"`{row.get('admission_reason', '')}` |"
        )
    lines.extend(["", "## Interpretation", "", "- `committed_probe` means safe to spend canonical full-score budget when the scorer is reachable."])
    lines.append("- `provisional_probe` keeps plausible but under-verified edges visible without forcing them into the next submission.")
    lines.append("- `quarantine` rows carry explicit no-GT counter-evidence and should not be scheduled until a new verifier addresses that risk.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "rows.json"
        src.write_text(json.dumps({"selected": [
            {"source_component_label": 1, "target_component": 2, "source_size": 4, "target_size": 8, "target_best_sim": 0.86, "target_mean_sim": 0.50, "target_min_view_sim": 0.75, "target_view_vote": 0.30, "same_video_overlap_ratio": 0.01},
            {"source_component_label": 2, "target_component": 1, "source_size": 8, "target_size": 4, "target_best_sim": 0.86, "target_mean_sim": 0.50, "target_min_view_sim": 0.75, "target_view_vote": 0.30, "same_video_overlap_ratio": 0.01},
            {"source_component_label": 3, "target_component": 4, "source_size": 8, "target_size": 8, "target_best_sim": 0.70, "target_mean_sim": 0.40, "target_min_view_sim": 0.40, "target_view_vote": 0.10, "same_video_overlap_ratio": 0.04},
        ]}))
        out = admit_rows(argparse.Namespace(
            candidates_json=str(src),
            commit_best_sim=0.84,
            commit_min_view_sim=0.70,
            provisional_best_sim=0.74,
            provisional_min_view_sim=0.68,
            max_commit_overlap=0.02,
            max_provisional_overlap=0.025,
            min_commit_target_source_ratio=1.0,
            min_provisional_target_source_ratio=0.75,
            spiky_mean_sim=0.55,
            low_vote_max=0.35,
            top_n=8,
            json=str(root / "out.json"),
            csv=str(root / "out.csv"),
            md=str(root / "out.md"),
        ))
        assert len(out["selected"]) == 1, out
        assert len(out["rejected_direction_duplicates"]) == 1, out
        assert out["quarantine"], out
        assert "committed probes" in Path(root / "out.md").read_text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidates-json", default="local_runs/no_anchor_component_graph_low_vote_rescue_broad_20260620.json")
    ap.add_argument("--commit-best-sim", type=float, default=0.84)
    ap.add_argument("--commit-min-view-sim", type=float, default=0.70)
    ap.add_argument("--provisional-best-sim", type=float, default=0.74)
    ap.add_argument("--provisional-min-view-sim", type=float, default=0.68)
    ap.add_argument("--max-commit-overlap", type=float, default=0.02)
    ap.add_argument("--max-provisional-overlap", type=float, default=0.025)
    ap.add_argument("--min-commit-target-source-ratio", type=float, default=1.0)
    ap.add_argument("--min-provisional-target-source-ratio", type=float, default=0.75)
    ap.add_argument("--spiky-mean-sim", type=float, default=0.55)
    ap.add_argument("--low-vote-max", type=float, default=0.35)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--json", default="local_runs/no_anchor_component_graph_admission_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_component_graph_admission_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_component_graph_admission_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    result = admit_rows(args)
    print(json.dumps({
        "raw": result["raw_count"],
        "pair_groups": result["pair_groups"],
        "committed_probe": len(result["selected"]),
        "provisional_probe": len(result["provisional"]),
        "quarantine": len(result["quarantine"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
