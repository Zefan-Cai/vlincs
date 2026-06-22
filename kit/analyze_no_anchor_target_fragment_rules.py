#!/usr/bin/env python
"""Audit no-anchor target-fragment rules on exported component edge tables.

This is eval-only analysis.  It reads candidate edge CSVs that were generated
without anchors, then uses the already-attached GT edge labels only to quantify
whether no-GT features such as small component size, score, votes, and rank can
identify false-split repair edges.  It does not train a production model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_rows(paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path_text in paths:
        path = Path(path_text)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row: dict[str, object] = dict(raw)
                source_size = _float(raw, "source_size")
                target_size = _float(raw, "target_size")
                small_side = max(min(source_size, target_size), 1.0)
                large_side = max(max(source_size, target_size), 1.0)
                row.update(
                    {
                        "source_csv": str(path),
                        "source_size": source_size,
                        "target_size": target_size,
                        "small_side": small_side,
                        "large_side": large_side,
                        "size_ratio": float(large_side / small_side),
                        "score": _float(raw, "score"),
                        "votes_top5": _float(raw, "votes_top5"),
                        "source_rank": _float(raw, "source_rank", 1.0e9),
                        "target_rank": _float(raw, "target_rank", 1.0e9),
                        "rank_max": max(_float(raw, "source_rank", 1.0e9), _float(raw, "target_rank", 1.0e9)),
                        "is_forbidden": _float(raw, "is_forbidden"),
                        "gt_edge_label": int(_float(raw, "gt_edge_label")),
                        "gt_top_false_split_target": _truthy(raw.get("gt_top_false_split_target", "")),
                        "gt_edge_same_mass": _float(raw, "gt_edge_same_mass"),
                        "gt_edge_all_mass": _float(raw, "gt_edge_all_mass"),
                    }
                )
                rows.append(row)
    return rows


def _candidate_passes(row: dict[str, object], rule: dict[str, object]) -> bool:
    if float(row["small_side"]) > int(rule["small_side_max"]):
        return False
    if float(row["size_ratio"]) < float(rule["min_size_ratio"]):
        return False
    if float(row["score"]) < float(rule["score_threshold"]):
        return False
    if float(row["votes_top5"]) < int(rule["min_votes_top5"]):
        return False
    if float(row["rank_max"]) > int(rule["rank_max"]):
        return False
    forbidden_mode = str(rule["forbidden_mode"])
    if forbidden_mode == "forbidden_only" and float(row["is_forbidden"]) <= 0:
        return False
    if forbidden_mode == "allowed_only" and float(row["is_forbidden"]) > 0:
        return False
    return True


def _score_rule(rows: list[dict[str, object]], rule: dict[str, object]) -> dict[str, object] | None:
    cand = [row for row in rows if _candidate_passes(row, rule)]
    if not cand:
        return None
    true_rows = [row for row in cand if int(row["gt_edge_label"]) == 1]
    false_rows = [row for row in cand if int(row["gt_edge_label"]) != 1]
    true_same_mass = sum(float(row["gt_edge_same_mass"]) for row in true_rows)
    false_proxy_mass = sum(
        max(float(row["gt_edge_all_mass"]) - float(row["gt_edge_same_mass"]), 0.0) for row in false_rows
    )
    all_proxy_mass = true_same_mass + false_proxy_mass
    out = dict(rule)
    out.update(
        {
            "candidate_edges": int(len(cand)),
            "true_edges": int(len(true_rows)),
            "false_edges": int(len(false_rows)),
            "true_top_false_split_edges": int(sum(bool(row["gt_top_false_split_target"]) for row in true_rows)),
            "edge_precision": round(float(len(true_rows) / max(len(cand), 1)), 6),
            "true_same_mass": round(float(true_same_mass), 3),
            "false_proxy_mass": round(float(false_proxy_mass), 3),
            "mass_precision_proxy": round(float(true_same_mass / max(all_proxy_mass, 1.0e-9)), 6),
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": True,
        }
    )
    return out


def _sweep(rows: list[dict[str, object]], args) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for small_side_max in _parse_ints(args.small_side_sizes):
        for min_size_ratio in _parse_floats(args.size_ratios):
            for score_threshold in _parse_floats(args.score_thresholds):
                for min_votes_top5 in _parse_ints(args.min_votes_top5):
                    for rank_max in _parse_ints(args.rank_maxes):
                        for forbidden_mode in [part for part in str(args.forbidden_modes).split(",") if part.strip()]:
                            scored = _score_rule(
                                rows,
                                {
                                    "small_side_max": int(small_side_max),
                                    "min_size_ratio": float(min_size_ratio),
                                    "score_threshold": float(score_threshold),
                                    "min_votes_top5": int(min_votes_top5),
                                    "rank_max": int(rank_max),
                                    "forbidden_mode": str(forbidden_mode),
                                },
                            )
                            if scored is not None:
                                out.append(scored)
    out.sort(
        key=lambda row: (
            float(row["edge_precision"]),
            float(row["true_same_mass"]),
            int(row["true_top_false_split_edges"]),
            -int(row["candidate_edges"]),
        ),
        reverse=True,
    )
    return out


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not path or not rows:
        return
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))})
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_md(path: str, result: dict[str, object], top: list[dict[str, object]]) -> None:
    if not path:
        return
    lines = [
        "# No-Anchor Target-Fragment Rule Audit",
        "",
        f"Input rows: `{result['input_rows']}`",
        f"True edges: `{result['true_edges']}`",
        f"True top false-split edges: `{result['true_top_false_split_edges']}`",
        "",
        "This is eval-only analysis. GT labels are used only after candidate generation",
        "to inspect whether no-GT fragment rules are promising.",
        "",
        "## Top Rules",
        "",
        "| precision | true / cand | true mass | fp proxy | small side | ratio | score | votes | rank | forbidden |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in top[:20]:
        lines.append(
            "| {edge_precision:.3f} | {true_edges}/{candidate_edges} | {true_same_mass:.0f} | "
            "{false_proxy_mass:.0f} | {small_side_max} | {min_size_ratio:g} | "
            "{score_threshold:g} | {min_votes_top5} | {rank_max} | {forbidden_mode} |".format(**row)
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def _self_test() -> None:
    rows = [
        {
            "small_side": 1.0,
            "size_ratio": 20.0,
            "score": 0.80,
            "votes_top5": 1.0,
            "rank_max": 999.0,
            "is_forbidden": 1.0,
            "gt_edge_label": 1,
            "gt_top_false_split_target": True,
            "gt_edge_same_mass": 100.0,
            "gt_edge_all_mass": 120.0,
        },
        {
            "small_side": 8.0,
            "size_ratio": 2.0,
            "score": 0.90,
            "votes_top5": 3.0,
            "rank_max": 999.0,
            "is_forbidden": 1.0,
            "gt_edge_label": 0,
            "gt_top_false_split_target": False,
            "gt_edge_same_mass": 0.0,
            "gt_edge_all_mass": 400.0,
        },
    ]
    scored = _score_rule(
        rows,
        {
            "small_side_max": 2,
            "min_size_ratio": 12.0,
            "score_threshold": 0.75,
            "min_votes_top5": 1,
            "rank_max": 999,
            "forbidden_mode": "forbidden_only",
        },
    )
    assert scored is not None
    assert scored["candidate_edges"] == 1
    assert scored["true_edges"] == 1
    assert scored["edge_precision"] == 1.0
    print(json.dumps({"stage": "self_test", "status": "ok", "rule": scored}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--edge-csv", action="append", default=[])
    ap.add_argument("--small-side-sizes", default="1,2,4,8,16")
    ap.add_argument("--size-ratios", default="1,4,8,12,16,24,32,48,64")
    ap.add_argument("--score-thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    ap.add_argument("--min-votes-top5", default="0,1,2,3")
    ap.add_argument("--rank-maxes", default="999,10,5,3,1")
    ap.add_argument("--forbidden-modes", default="all,forbidden_only,allowed_only")
    ap.add_argument("--top-n", type=int, default=80)
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.edge_csv:
        ap.error("--edge-csv is required unless --self-test is used")
    if not args.json:
        ap.error("--json is required unless --self-test is used")

    rows = _load_rows(args.edge_csv)
    swept = _sweep(rows, args)
    top = swept[: max(int(args.top_n), 1)]
    result = {
        "edge_csv": list(args.edge_csv),
        "input_rows": int(len(rows)),
        "true_edges": int(sum(int(row["gt_edge_label"]) == 1 for row in rows)),
        "true_top_false_split_edges": int(
            sum(int(row["gt_edge_label"]) == 1 and bool(row["gt_top_false_split_target"]) for row in rows)
        ),
        "rules_evaluated_with_candidates": int(len(swept)),
        "top": top,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_csv(args.csv, top)
    _write_md(args.md, result, top)
    print(json.dumps({"json": str(out), "best": top[0] if top else None}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
