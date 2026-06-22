#!/usr/bin/env python
"""Audit whether edge-table candidates cover high-mass oracle false splits.

This is evaluation-only: oracle and GT columns are used only to measure coverage
after no-anchor edge candidates have already been generated.  The production
lesson is whether an edge-table proposer has enough candidate recall to spend
full-score budget, or whether research should pivot to a different proposer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _read_oracle(path: str) -> tuple[dict[int, dict[str, Any]], float]:
    data = json.loads(Path(path).read_text())
    rows = data.get("false_split_rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} is missing false_split_rows")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        gt = _as_int(row.get("id"), -1)
        if gt < 0:
            continue
        out[gt] = {
            "gt_id": gt,
            "oracle_rank": _as_int(row.get("rank")),
            "oracle_false_split_mass": _as_float(row.get("value")),
            "oracle_parts": _as_int(row.get("parts")),
            "oracle_tracklets": _as_int(row.get("tracklets")),
            "oracle_coverage": _as_float(row.get("coverage")),
        }
    budget = data.get("budget") if isinstance(data.get("budget"), dict) else {}
    total = _as_float(budget.get("missing_true_pair_mass"))
    if total <= 0:
        total = sum(float(row["oracle_false_split_mass"]) for row in out.values())
    return out, float(total)


def _edge_key(row: dict[str, str], gt: int) -> tuple[int, int, int]:
    a = _as_int(row.get("source"))
    b = _as_int(row.get("target"))
    if a > b:
        a, b = b, a
    return int(gt), int(a), int(b)


def _collect_edges(edge_csvs: list[str]) -> tuple[list[dict[str, Any]], dict[tuple[int, int, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    union: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in edge_csvs:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for local_rank, row in enumerate(reader, start=1):
                gt = _as_int(row.get("gt_dominant_id"), -1)
                item = {
                    "source_file": path,
                    "source_file_name": Path(path).name,
                    "source_file_rank": int(local_rank),
                    "gt_id": int(gt),
                    "source": _as_int(row.get("source")),
                    "target": _as_int(row.get("target")),
                    "source_weight": _as_float(row.get("source_weight")),
                    "target_weight": _as_float(row.get("target_weight")),
                    "score": _as_float(row.get("score"), -1.0),
                    "centroid_score": _as_float(row.get("centroid_score"), -1.0),
                    "rank_margin": _as_float(row.get("rank_margin")),
                    "source_rank": _as_int(row.get("source_rank"), 1_000_000),
                    "target_rank": _as_int(row.get("target_rank"), 1_000_000),
                    "primary_rank_max": _as_int(row.get("primary_rank_max"), 1_000_000),
                    "votes_top5": _as_int(row.get("votes_top5")),
                    "is_forbidden": _as_bool(row.get("is_forbidden")),
                    "video_overlap": _as_bool(row.get("video_overlap")),
                    "camera_overlap": _as_bool(row.get("camera_overlap")),
                    "gt_edge_label": _as_int(row.get("gt_edge_label")),
                    "gt_top_false_split_target": _as_bool(row.get("gt_top_false_split_target")),
                    "gt_edge_same_mass": _as_float(row.get("gt_edge_same_mass")),
                    "gt_edge_all_mass": _as_float(row.get("gt_edge_all_mass")),
                    "gt_edge_same_frac": _as_float(row.get("gt_edge_same_frac")),
                    "gt_both_pure": _as_bool(row.get("gt_both_pure")),
                }
                rows.append(item)
                if item["gt_edge_label"] != 1 or item["gt_edge_same_mass"] <= 0 or gt < 0:
                    continue
                key = _edge_key(row, gt)
                prev = union.get(key)
                if prev is None or float(item["score"]) > float(prev["score"]):
                    union[key] = item
    return rows, union


def _summarize_table(path: str, rows: list[dict[str, Any]], oracle_by_gt: dict[int, dict[str, Any]], total_missing: float):
    table_rows = [row for row in rows if row["source_file"] == path]
    true_rows = [row for row in table_rows if row["gt_edge_label"] == 1 and row["gt_edge_same_mass"] > 0]
    true_mass = sum(float(row["gt_edge_same_mass"]) for row in true_rows)
    top_oracle_mass = 0.0
    for row in true_rows:
        if int(row["gt_id"]) in oracle_by_gt:
            top_oracle_mass += float(row["gt_edge_same_mass"])
    return {
        "edge_csv": path,
        "edge_csv_name": Path(path).name,
        "candidate_edges": int(len(table_rows)),
        "true_edges": int(len(true_rows)),
        "top_false_split_flag_edges": int(sum(1 for row in table_rows if row["gt_top_false_split_target"])),
        "true_edge_same_mass": round(float(true_mass), 3),
        "coverage_of_total_missing_mass": round(float(true_mass / total_missing), 8) if total_missing > 0 else 0.0,
        "coverage_of_oracle_top_rows_mass": round(float(top_oracle_mass / total_missing), 8) if total_missing > 0 else 0.0,
    }


def _per_gt_rows(
    oracle_by_gt: dict[int, dict[str, Any]],
    union_edges: dict[tuple[int, int, int], dict[str, Any]],
    total_missing: float,
) -> list[dict[str, Any]]:
    by_gt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (gt, _a, _b), edge in union_edges.items():
        by_gt[int(gt)].append(edge)
    rows = []
    for gt, oracle in oracle_by_gt.items():
        edges = by_gt.get(int(gt), [])
        edge_mass = sum(float(edge["gt_edge_same_mass"]) for edge in edges)
        best = max(edges, key=lambda edge: float(edge["score"]), default=None)
        rows.append(
            {
                **oracle,
                "union_true_edges": int(len(edges)),
                "union_true_edge_same_mass": round(float(edge_mass), 3),
                "edge_coverage_of_gt_missing_mass": round(
                    float(edge_mass / oracle["oracle_false_split_mass"]), 8
                )
                if oracle["oracle_false_split_mass"] > 0
                else 0.0,
                "edge_coverage_of_total_missing_mass": round(float(edge_mass / total_missing), 8)
                if total_missing > 0
                else 0.0,
                "best_edge_score": round(float(best["score"]), 6) if best else None,
                "best_edge_file": Path(str(best["source_file"])).name if best else "",
                "best_edge_source": int(best["source"]) if best else None,
                "best_edge_target": int(best["target"]) if best else None,
                "best_edge_source_weight": round(float(best["source_weight"]), 3) if best else None,
                "best_edge_target_weight": round(float(best["target_weight"]), 3) if best else None,
                "best_edge_same_mass": round(float(best["gt_edge_same_mass"]), 3) if best else None,
                "best_edge_source_rank": int(best["source_rank"]) if best else None,
                "best_edge_target_rank": int(best["target_rank"]) if best else None,
                "best_edge_is_forbidden": bool(best["is_forbidden"]) if best else None,
            }
        )
    rows.sort(key=lambda row: int(row["oracle_rank"]))
    return rows


def _prefix(rows: list[dict[str, Any]], total_missing: float, ns: list[int]) -> list[dict[str, Any]]:
    out = []
    for n in ns:
        mass = sum(float(row["union_true_edge_same_mass"]) for row in rows[:n])
        oracle_mass = sum(float(row["oracle_false_split_mass"]) for row in rows[:n])
        out.append(
            {
                "top_n": int(n),
                "oracle_false_split_mass": round(float(oracle_mass), 3),
                "edge_true_mass": round(float(mass), 3),
                "edge_coverage_of_prefix_oracle_mass": round(float(mass / oracle_mass), 8)
                if oracle_mass > 0
                else 0.0,
                "edge_coverage_of_total_missing_mass": round(float(mass / total_missing), 8)
                if total_missing > 0
                else 0.0,
            }
        )
    return out


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: str, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Edge Table Oracle Coverage",
        "",
        f"- missing true-pair mass: `{result['missing_true_pair_mass']}`",
        f"- union true edge mass: `{result['union_true_edge_same_mass']}`",
        f"- union coverage of missing mass: `{result['union_coverage_of_total_missing_mass']}`",
        "",
        "## Prefix Coverage",
        "",
        "| top oracle GTs | oracle mass | edge true mass | prefix coverage | total coverage |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["prefix_coverage"]:
        lines.append(
            f"| {row['top_n']} | `{row['oracle_false_split_mass']}` | `{row['edge_true_mass']}` | "
            f"`{row['edge_coverage_of_prefix_oracle_mass']}` | `{row['edge_coverage_of_total_missing_mass']}` |"
        )
    lines.extend(
        [
            "",
            "## Top Oracle False Splits",
            "",
            "| rank | GT | oracle mass | parts | edge mass | GT coverage | best score | best edge |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["per_gt_rows"][:20]:
        edge = f"{row['best_edge_file']}:{row['best_edge_source']}->{row['best_edge_target']}" if row["best_edge_file"] else ""
        lines.append(
            f"| {row['oracle_rank']} | {row['gt_id']} | `{row['oracle_false_split_mass']}` | "
            f"{row['oracle_parts']} | `{row['union_true_edge_same_mass']}` | "
            f"`{row['edge_coverage_of_gt_missing_mass']}` | `{row['best_edge_score']}` | `{edge}` |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        oracle = Path(tmp) / "oracle.json"
        edge = Path(tmp) / "edges.csv"
        oracle.write_text(
            json.dumps(
                {
                    "budget": {"missing_true_pair_mass": 1000},
                    "false_split_rows": [
                        {"id": 1, "rank": 1, "value": 600, "parts": 3, "tracklets": 10, "coverage": 0.6},
                        {"id": 2, "rank": 2, "value": 400, "parts": 2, "tracklets": 8, "coverage": 1.0},
                    ],
                }
            )
        )
        with edge.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source",
                    "target",
                    "score",
                    "gt_dominant_id",
                    "gt_edge_label",
                    "gt_edge_same_mass",
                    "source_weight",
                    "target_weight",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source": 3,
                    "target": 4,
                    "score": 0.9,
                    "gt_dominant_id": 1,
                    "gt_edge_label": 1,
                    "gt_edge_same_mass": 120,
                    "source_weight": 30,
                    "target_weight": 4,
                }
            )
        oracle_by_gt, total = _read_oracle(str(oracle))
        rows, union = _collect_edges([str(edge)])
        per_gt = _per_gt_rows(oracle_by_gt, union, total)
        assert per_gt[0]["edge_coverage_of_gt_missing_mass"] == 0.2, per_gt
        assert _prefix(per_gt, total, [1])[0]["edge_coverage_of_total_missing_mass"] == 0.12
    print(json.dumps({"stage": "self_test", "status": "ok"}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--oracle-json", default="local_runs/no_anchor_oracle_gap_concentration_20260620.json")
    ap.add_argument("--edge-csv", action="append", default=[])
    ap.add_argument("--prefix-ns", default="1,3,5,10,20,30")
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.edge_csv:
        raise SystemExit("--edge-csv is required unless --self-test is used")
    oracle_by_gt, total_missing = _read_oracle(args.oracle_json)
    rows, union = _collect_edges(args.edge_csv)
    per_gt = _per_gt_rows(oracle_by_gt, union, total_missing)
    prefix_ns = [_as_int(part) for part in str(args.prefix_ns).split(",") if str(part).strip()]
    true_mass = sum(float(edge["gt_edge_same_mass"]) for edge in union.values())
    table_summaries = [_summarize_table(path, rows, oracle_by_gt, total_missing) for path in args.edge_csv]
    result = {
        "oracle_json": str(args.oracle_json),
        "edge_csvs": list(args.edge_csv),
        "missing_true_pair_mass": round(float(total_missing), 3),
        "candidate_edges_total": int(len(rows)),
        "union_true_edges": int(len(union)),
        "union_true_edge_same_mass": round(float(true_mass), 3),
        "union_coverage_of_total_missing_mass": round(float(true_mass / total_missing), 8)
        if total_missing > 0
        else 0.0,
        "table_summaries": table_summaries,
        "prefix_coverage": _prefix(per_gt, total_missing, prefix_ns),
        "per_gt_rows": per_gt,
        "gt_ids_with_any_edge": int(sum(1 for row in per_gt if row["union_true_edges"] > 0)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, per_gt)
    if args.md:
        _write_md(args.md, result)
    print(
        json.dumps(
            {
                "candidate_edges_total": result["candidate_edges_total"],
                "union_true_edges": result["union_true_edges"],
                "union_coverage_of_total_missing_mass": result["union_coverage_of_total_missing_mass"],
                "gt_ids_with_any_edge": result["gt_ids_with_any_edge"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
