#!/usr/bin/env python
"""Audit how much oracle false-split mass no-anchor edge rules can cover.

This is eval-only analysis. Candidate rules use no-GT edge-table features.
GT labels and oracle rows are used only after selection to estimate whether a
rule can plausibly close the full-score IDF1 gap.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _oracle_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    for section in ("top_full_rows", "full_rows", "rows", "top_pair_rows"):
        for row in data.get(section, []) or []:
            if not isinstance(row, dict) or "true_pair_mass" not in row:
                continue
            key = (str(row.get("name", "")), _float(row, "true_pair_mass"), _float(row, "full_idf1"))
            if key not in seen:
                rows.append(row)
                seen.add(key)
    return rows


def _pick_oracle_pair(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("oracle JSON contains no rows with true_pair_mass")

    base = None
    oracle = None
    for row in rows:
        name = str(row.get("name", ""))
        if name in {"base", "split_top_false_merge_components_0"}:
            if base is None or ("full_idf1" not in base and "full_idf1" in row):
                base = row
        if name in {"oracle_all_gt_majority", "all_gt_majority"}:
            if oracle is None or ("full_idf1" not in oracle and "full_idf1" in row):
                oracle = row
    full_rows = [row for row in rows if "full_idf1" in row]
    if base is None:
        candidates = full_rows or rows
        base = min(candidates, key=lambda row: (_float(row, "full_idf1", 1.0e9), _float(row, "true_pair_mass")))
    if oracle is None:
        candidates = full_rows or rows
        oracle = max(candidates, key=lambda row: (_float(row, "true_pair_mass"), _float(row, "full_idf1")))
    return base, oracle


def _load_oracle(path_text: str) -> dict[str, Any]:
    data = json.loads(Path(path_text).read_text())
    base, oracle = _pick_oracle_pair(_oracle_rows(data))
    base_true = _float(base, "true_pair_mass")
    oracle_true = _float(oracle, "true_pair_mass")
    base_idf1 = _float(base, "full_idf1")
    oracle_idf1 = _float(oracle, "full_idf1")
    missing_mass = max(oracle_true - base_true, 0.0)
    idf1_gap = max(oracle_idf1 - base_idf1, 0.0)
    return {
        "oracle_json": path_text,
        "base_name": str(base.get("name", "base")),
        "oracle_name": str(oracle.get("name", "oracle")),
        "base_true_pair_mass": base_true,
        "oracle_true_pair_mass": oracle_true,
        "missing_true_pair_mass": missing_mass,
        "base_full_idf1": base_idf1,
        "oracle_full_idf1": oracle_idf1,
        "oracle_full_idf1_gap": idf1_gap,
        "coverage_needed_for_70": max((0.70 - base_idf1) / idf1_gap, 0.0) if idf1_gap > 0 else None,
    }


def _edge_key(row: dict[str, Any]) -> tuple[str, str]:
    a = str(row.get("source", ""))
    b = str(row.get("target", ""))
    return tuple(sorted((a, b)))


def _load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_text in paths:
        path = Path(path_text)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for line_no, raw in enumerate(reader, start=2):
                source_size = _float(raw, "source_size")
                target_size = _float(raw, "target_size")
                small_side = max(min(source_size, target_size), 1.0)
                large_side = max(max(source_size, target_size), 1.0)
                row: dict[str, Any] = dict(raw)
                row.update(
                    {
                        "source_csv": str(path),
                        "line_no": line_no,
                        "edge_key": "|".join(_edge_key(row)),
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


def _candidate_passes(row: dict[str, Any], rule: dict[str, Any]) -> bool:
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
    mode = str(rule["forbidden_mode"])
    if mode == "forbidden_only" and float(row["is_forbidden"]) <= 0:
        return False
    if mode == "allowed_only" and float(row["is_forbidden"]) > 0:
        return False
    return True


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _edge_key(row)
        old = best.get(key)
        if old is None or (float(row["score"]), -float(row["rank_max"])) > (float(old["score"]), -float(old["rank_max"])):
            best[key] = row
    return list(best.values())


def _score_rule(rows: list[dict[str, Any]], rule: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any] | None:
    cand = _dedupe([row for row in rows if _candidate_passes(row, rule)])
    if not cand:
        return None
    true_rows = [row for row in cand if int(row["gt_edge_label"]) == 1]
    false_rows = [row for row in cand if int(row["gt_edge_label"]) != 1]
    true_same_mass = sum(float(row["gt_edge_same_mass"]) for row in true_rows)
    false_proxy_mass = sum(
        max(float(row["gt_edge_all_mass"]) - float(row["gt_edge_same_mass"]), 0.0) for row in false_rows
    )
    all_proxy_mass = true_same_mass + false_proxy_mass
    missing_mass = float(oracle["missing_true_pair_mass"])
    gap = float(oracle["oracle_full_idf1_gap"])
    coverage = true_same_mass / max(missing_mass, 1.0e-9)
    linear_gain = coverage * gap
    out = dict(rule)
    out.update(
        {
            "candidate_edges": len(cand),
            "true_edges": len(true_rows),
            "false_edges": len(false_rows),
            "true_top_false_split_edges": int(sum(bool(row["gt_top_false_split_target"]) for row in true_rows)),
            "edge_precision": round(len(true_rows) / max(len(cand), 1), 6),
            "true_same_mass": round(true_same_mass, 3),
            "false_proxy_mass": round(false_proxy_mass, 3),
            "mass_precision_proxy": round(true_same_mass / max(all_proxy_mass, 1.0e-9), 6),
            "missing_mass_coverage": round(coverage, 8),
            "estimated_full_idf1_gain_if_linear": round(linear_gain, 8),
            "estimated_full_idf1_if_linear": round(float(oracle["base_full_idf1"]) + linear_gain, 8),
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": True,
        }
    )
    return out


def _sweep(rows: list[dict[str, Any]], args: argparse.Namespace, oracle: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
                                oracle,
                            )
                            if scored is not None:
                                out.append(scored)
    return out


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
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


def _write_md(path: str, result: dict[str, Any]) -> None:
    if not path:
        return
    oracle = result["oracle_budget"]
    lines = [
        "# No-Anchor Edge Mass Budget Audit",
        "",
        f"Input edge rows: `{result['input_rows']}`",
        f"Unique undirected edges: `{result['unique_edges']}`",
        f"Base full IDF1: `{oracle['base_full_idf1']:.6f}`",
        f"Oracle full IDF1: `{oracle['oracle_full_idf1']:.6f}`",
        f"Missing true pair mass: `{oracle['missing_true_pair_mass']:.0f}`",
        f"Coverage needed for full IDF1 0.70 under a linear gap model: `{oracle['coverage_needed_for_70']:.3f}`",
        "",
        "This is eval-only analysis. The rules use no-GT edge features; GT mass is",
        "used only to estimate whether a rule family has enough coverage to matter.",
        "",
        "## Highest-Precision Rules",
        "",
        "| precision | coverage | est IDF1 | true / cand | true mass | fp proxy | small | ratio | score | votes | rank | forbidden |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["top_by_precision"][:20]:
        lines.append(
            "| {edge_precision:.3f} | {missing_mass_coverage:.5f} | {estimated_full_idf1_if_linear:.6f} | "
            "{true_edges}/{candidate_edges} | {true_same_mass:.0f} | {false_proxy_mass:.0f} | "
            "{small_side_max} | {min_size_ratio:g} | {score_threshold:g} | {min_votes_top5} | {rank_max} | {forbidden_mode} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Highest-Coverage Rules",
            "",
            "| coverage | precision | mass precision | est IDF1 | true / cand | true mass | fp proxy | small | ratio | score | votes | rank | forbidden |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["top_by_coverage"][:20]:
        lines.append(
            "| {missing_mass_coverage:.5f} | {edge_precision:.3f} | {mass_precision_proxy:.3f} | "
            "{estimated_full_idf1_if_linear:.6f} | {true_edges}/{candidate_edges} | {true_same_mass:.0f} | "
            "{false_proxy_mass:.0f} | {small_side_max} | {min_size_ratio:g} | {score_threshold:g} | "
            "{min_votes_top5} | {rank_max} | {forbidden_mode} |".format(**row)
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def _self_test() -> None:
    oracle = {
        "base_full_idf1": 0.65,
        "oracle_full_idf1": 0.75,
        "oracle_full_idf1_gap": 0.10,
        "missing_true_pair_mass": 1000.0,
    }
    rows = [
        {"source": "a", "target": "b", "small_side": 1.0, "size_ratio": 20.0, "score": 0.8, "votes_top5": 1.0, "rank_max": 2.0, "is_forbidden": 1.0, "gt_edge_label": 1, "gt_top_false_split_target": True, "gt_edge_same_mass": 100.0, "gt_edge_all_mass": 120.0},
        {"source": "b", "target": "a", "small_side": 1.0, "size_ratio": 20.0, "score": 0.7, "votes_top5": 1.0, "rank_max": 3.0, "is_forbidden": 1.0, "gt_edge_label": 1, "gt_top_false_split_target": True, "gt_edge_same_mass": 100.0, "gt_edge_all_mass": 120.0},
        {"source": "c", "target": "d", "small_side": 4.0, "size_ratio": 5.0, "score": 0.9, "votes_top5": 2.0, "rank_max": 1.0, "is_forbidden": 0.0, "gt_edge_label": 0, "gt_top_false_split_target": False, "gt_edge_same_mass": 0.0, "gt_edge_all_mass": 300.0},
    ]
    scored = _score_rule(
        rows,
        {
            "small_side_max": 2,
            "min_size_ratio": 12.0,
            "score_threshold": 0.75,
            "min_votes_top5": 1,
            "rank_max": 5,
            "forbidden_mode": "forbidden_only",
        },
        oracle,
    )
    assert scored is not None
    assert scored["candidate_edges"] == 1
    assert scored["true_edges"] == 1
    assert scored["missing_mass_coverage"] == 0.1
    assert scored["estimated_full_idf1_if_linear"] == 0.66
    print(json.dumps({"stage": "self_test", "status": "ok", "rule": scored}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--oracle-json", default="")
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
    if not args.oracle_json:
        ap.error("--oracle-json is required unless --self-test is used")
    if not args.edge_csv:
        ap.error("--edge-csv is required unless --self-test is used")
    if not args.json:
        ap.error("--json is required unless --self-test is used")

    oracle = _load_oracle(args.oracle_json)
    rows = _load_rows(args.edge_csv)
    swept = _sweep(rows, args, oracle)
    top_by_precision = sorted(
        swept,
        key=lambda row: (float(row["edge_precision"]), float(row["true_same_mass"]), -int(row["candidate_edges"])),
        reverse=True,
    )[: max(args.top_n, 1)]
    top_by_coverage = sorted(
        swept,
        key=lambda row: (float(row["missing_mass_coverage"]), float(row["mass_precision_proxy"]), float(row["edge_precision"])),
        reverse=True,
    )[: max(args.top_n, 1)]
    result = {
        "edge_csv": list(args.edge_csv),
        "oracle_budget": oracle,
        "input_rows": len(rows),
        "unique_edges": len({_edge_key(row) for row in rows}),
        "rules_evaluated_with_candidates": len(swept),
        "top_by_precision": top_by_precision,
        "top_by_coverage": top_by_coverage,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_csv(args.csv, top_by_precision)
    _write_md(args.md, result)
    print(json.dumps({"json": str(out), "best_precision": top_by_precision[0], "best_coverage": top_by_coverage[0]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
