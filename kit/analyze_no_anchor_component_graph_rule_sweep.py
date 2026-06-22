#!/usr/bin/env python
"""Eval-only rule sweep for no-anchor component-graph bridge candidates.

The production rules under test read only candidate metadata.  This script
uses GT-derived coverage artifacts only after the fact to decide whether a
candidate rule family deserves full-score budget or should be rejected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any) -> int:
    return int(float(value))


def _load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path} is missing rows[]")
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            out.append({"candidate_rank": rank, **row})
    return out


def _load_audit_rows(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("top_rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} is missing top_rows[]")
    return {int(row["candidate_rank"]): row for row in rows if isinstance(row, dict)}


def _direction_score(row: dict[str, Any]) -> float:
    source_size = max(_as_float(row.get("source_size")), 1.0)
    target_size = max(_as_float(row.get("target_size")), 1.0)
    ratio = target_size / source_size
    low_vote_bonus = 1.0 if 0.25 <= _as_float(row.get("target_view_vote")) <= 0.35 else 0.0
    return float(
        0.28 * _as_float(row.get("target_best_sim"))
        + 0.20 * _as_float(row.get("target_min_view_sim"))
        + 0.15 * _as_float(row.get("target_mean_sim"))
        + 0.15 * min(math.log(ratio + 1.0), 0.80)
        + 0.08 * low_vote_bonus
        + 0.08 * min(math.log1p(source_size * target_size) / math.log(100000.0), 1.0)
        - 0.40 * _as_float(row.get("same_video_overlap_ratio"))
    )


def _merge_rows(candidates: list[dict[str, Any]], audit_by_rank: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for row in candidates:
        rank = int(row["candidate_rank"])
        audit = audit_by_rank.get(rank, {})
        new = {
            **row,
            "direction_score": _direction_score(row),
            "audit_positive_bridge_mass": _as_float(audit.get("audit_positive_bridge_mass")),
            "audit_gap_coverage": _as_float(audit.get("audit_gap_coverage")),
            "audit_positive_edges": _as_int(audit.get("audit_positive_edges") or 0),
            "uses_gt_for_evaluation_only": True,
        }
        merged.append(new)
    return merged


def _best_direction_per_pair(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        source = _as_int(row.get("source_component_label"))
        target = _as_int(row.get("target_component"))
        key = tuple(sorted((source, target)))
        if key not in best or float(row["direction_score"]) > float(best[key]["direction_score"]):
            best[key] = row
    return sorted(best.values(), key=lambda row: float(row["direction_score"]), reverse=True)


def _summarize(name: str, rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], *, top_n: int) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    selected = sorted(selected, key=lambda row: (float(row["direction_score"]), _as_float(row.get("accepted_pair_mass_proxy_sum"))), reverse=True)
    selected = selected[: int(top_n)]
    positives = [row for row in selected if float(row["audit_positive_bridge_mass"]) > 0]
    return {
        "name": name,
        "selected_count": len(selected),
        "positive_count": len(positives),
        "false_count": len(selected) - len(positives),
        "precision": float(len(positives) / max(len(selected), 1)),
        "positive_bridge_mass": float(sum(float(row["audit_positive_bridge_mass"]) for row in selected)),
        "top_gap_coverage": float(max([float(row["audit_gap_coverage"]) for row in selected] + [0.0])),
        "rows": [_short_row(row) for row in selected],
    }


def _short_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_rank": int(row["candidate_rank"]),
        "source_component_label": row.get("source_component_label"),
        "target_component": row.get("target_component"),
        "source_size": _as_int(row.get("source_size")),
        "target_size": _as_int(row.get("target_size")),
        "target_best_sim": _as_float(row.get("target_best_sim")),
        "target_mean_sim": _as_float(row.get("target_mean_sim")),
        "target_min_view_sim": _as_float(row.get("target_min_view_sim")),
        "target_view_vote": _as_float(row.get("target_view_vote")),
        "same_video_overlap_ratio": _as_float(row.get("same_video_overlap_ratio")),
        "direction_score": _as_float(row.get("direction_score")),
        "audit_positive_bridge_mass": _as_float(row.get("audit_positive_bridge_mass")),
        "audit_gap_coverage": _as_float(row.get("audit_gap_coverage")),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _load_candidate_rows(Path(args.candidates_json))
    audit_by_rank = _load_audit_rows(Path(args.audit_json))
    rows = _merge_rows(candidates, audit_by_rank)
    pair_rows = _best_direction_per_pair(rows)

    families = [
        _summarize(
            "current_committed_low_vote_directional",
            pair_rows,
            lambda r: (
                _as_float(r.get("target_best_sim")) >= 0.84
                and 0.66 <= _as_float(r.get("target_min_view_sim")) <= 0.75
                and _as_float(r.get("same_video_overlap_ratio")) <= 0.02
                and _as_float(r.get("target_size")) >= _as_float(r.get("source_size"))
            ),
            top_n=args.top_n,
        ),
        _summarize(
            "broad_high_vote_supplement",
            pair_rows,
            lambda r: (
                _as_float(r.get("target_best_sim")) >= 0.84
                and _as_float(r.get("target_min_view_sim")) >= 0.68
                and _as_float(r.get("target_mean_sim")) >= 0.46
                and _as_float(r.get("target_view_vote")) >= 0.90
                and _as_float(r.get("same_video_overlap_ratio")) <= 0.02
                and _as_float(r.get("target_size")) >= 0.75 * max(_as_float(r.get("source_size")), 1.0)
            ),
            top_n=args.top_n,
        ),
        _summarize(
            "clean_high_vote_singleton",
            pair_rows,
            lambda r: (
                _as_float(r.get("target_best_sim")) >= 0.90
                and _as_float(r.get("target_min_view_sim")) >= 0.86
                and _as_float(r.get("target_mean_sim")) >= 0.60
                and _as_float(r.get("same_video_overlap_ratio")) <= 0.015
                and _as_float(r.get("target_size")) >= _as_float(r.get("source_size"))
            ),
            top_n=args.top_n,
        ),
    ]
    result = {
        "candidates_json": str(args.candidates_json),
        "audit_json": str(args.audit_json),
        "candidate_rows": len(candidates),
        "unordered_pair_rows": len(pair_rows),
        "positive_pair_rows": sum(1 for row in pair_rows if float(row["audit_positive_bridge_mass"]) > 0),
        "positive_pair_bridge_mass": float(sum(float(row["audit_positive_bridge_mass"]) for row in pair_rows)),
        "families": families,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "conclusion": _conclusion(families),
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _conclusion(families: list[dict[str, Any]]) -> str:
    by_name = {row["name"]: row for row in families}
    broad = by_name.get("broad_high_vote_supplement", {})
    clean = by_name.get("clean_high_vote_singleton", {})
    if broad and float(broad.get("precision", 0.0)) < 0.5:
        return "Reject broad high-vote supplement: it pulls too many zero-mass edges. Keep current committed low-vote direction and treat clean high-vote singleton as redundant with existing portfolio edges."
    if clean and int(clean.get("positive_count", 0)) > 0:
        return "Only a tiny clean high-vote singleton survives; it is useful as evidence but not a broad production expansion."
    return "No reliable supplement found."


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Component-Graph Rule Sweep",
        "",
        f"- candidates: `{result['candidate_rows']}`",
        f"- unordered pair directions: `{result['unordered_pair_rows']}`",
        f"- positive pair directions: `{result['positive_pair_rows']}`",
        f"- positive pair bridge mass: `{result['positive_pair_bridge_mass']:.0f}`",
        f"- conclusion: {result['conclusion']}",
        "",
        "| family | selected | positive | false | precision | bridge mass | top coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in result["families"]:
        lines.append(
            f"| `{family['name']}` | `{family['selected_count']}` | `{family['positive_count']}` | "
            f"`{family['false_count']}` | `{family['precision']:.3f}` | "
            f"`{family['positive_bridge_mass']:.0f}` | `{family['top_gap_coverage']:.6f}` |"
        )
    for family in result["families"]:
        lines.extend(["", f"## {family['name']}", "", "| rank | source | target | mass | best | mean | min-view | vote | overlap |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in family["rows"]:
            lines.append(
                f"| `{row['candidate_rank']}` | `{row['source_component_label']}` | `{row['target_component']}` | "
                f"`{row['audit_positive_bridge_mass']:.0f}` | `{row['target_best_sim']:.3f}` | "
                f"`{row['target_mean_sim']:.3f}` | `{row['target_min_view_sim']:.3f}` | "
                f"`{row['target_view_vote']:.2f}` | `{row['same_video_overlap_ratio']:.4f}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        cand = root / "cand.json"
        aud = root / "aud.json"
        cand.write_text(json.dumps({"rows": [
            {"source_component_label": 1, "target_component": 2, "source_size": 4, "target_size": 8, "target_best_sim": 0.85, "target_mean_sim": 0.49, "target_min_view_sim": 0.72, "target_view_vote": 0.3, "same_video_overlap_ratio": 0.01, "accepted_pair_mass_proxy_sum": 32},
            {"source_component_label": 2, "target_component": 1, "source_size": 8, "target_size": 4, "target_best_sim": 0.85, "target_mean_sim": 0.49, "target_min_view_sim": 0.72, "target_view_vote": 0.3, "same_video_overlap_ratio": 0.01, "accepted_pair_mass_proxy_sum": 32},
        ]}))
        aud.write_text(json.dumps({"top_rows": [{"candidate_rank": 1, "audit_positive_bridge_mass": 10, "audit_gap_coverage": 0.1, "audit_positive_edges": 1}, {"candidate_rank": 2, "audit_positive_bridge_mass": 10, "audit_gap_coverage": 0.1, "audit_positive_edges": 1}]}))
        out = run(argparse.Namespace(candidates_json=str(cand), audit_json=str(aud), top_n=5, json=str(root / "out.json"), md=str(root / "out.md")))
        assert out["candidate_rows"] == 2
        assert out["unordered_pair_rows"] == 1
        assert Path(root / "out.md").read_text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidates-json", default="local_runs/no_anchor_component_graph_high_mass_candidates_20260620.json")
    ap.add_argument("--audit-json", default="local_runs/no_anchor_candidate_false_split_coverage_component_graph_high_mass_all_20260620.json")
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--json", default="local_runs/no_anchor_component_graph_rule_sweep_20260620.json")
    ap.add_argument("--md", default="reports/no_anchor_component_graph_rule_sweep_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    result = run(args)
    print(json.dumps({
        "candidate_rows": result["candidate_rows"],
        "unordered_pair_rows": result["unordered_pair_rows"],
        "conclusion": result["conclusion"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
