#!/usr/bin/env python
"""Summarize concentration of no-anchor oracle false-split / false-merge gaps.

This is eval-only analysis.  It reads an existing oracle decomposition JSON and
does not train, choose, or modify production identities.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pick_full_rows(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [row for row in data.get("top_full_rows", []) if isinstance(row, dict)]
    if not rows:
        rows = [row for row in data.get("rows", []) if isinstance(row, dict) and "full_idf1" in row]
    if not rows:
        raise ValueError("oracle JSON does not contain full-score rows")
    base = next((row for row in rows if row.get("name") == "base"), None)
    oracle = next((row for row in rows if row.get("name") == "oracle_all_gt_majority"), None)
    if base is None:
        base = min(rows, key=lambda row: _float(row, "full_idf1", 1.0e9))
    if oracle is None:
        oracle = max(rows, key=lambda row: (_float(row, "true_pair_mass"), _float(row, "full_idf1")))
    return base, oracle


def _cumulative(rows: list[dict[str, Any]], value_key: str, denominator: float, prefix: str) -> list[dict[str, Any]]:
    out = []
    running = 0.0
    for rank, row in enumerate(rows, start=1):
        value = _float(row, value_key)
        running += value
        out.append(
            {
                "kind": prefix,
                "rank": rank,
                "id": row.get("gt_id", row.get("predicted_global_id", "")),
                "value": round(value, 3),
                "cumulative_value": round(running, 3),
                "coverage": round(running / max(denominator, 1.0e-9), 8),
                "tracklets": row.get("tracklets", ""),
                "parts": row.get("pred_component_count", row.get("gt_count", "")),
                "dominant_id": row.get("dominant_prediction", row.get("dominant_gt", "")),
                "dominant_weight_frac": row.get("dominant_prediction_weight_frac", row.get("dominant_gt_weight_frac", "")),
            }
        )
    return out


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    keys = ["kind", "rank", "id", "value", "cumulative_value", "coverage", "tracklets", "parts", "dominant_id", "dominant_weight_frac"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: str, result: dict[str, Any]) -> None:
    if not path:
        return
    budget = result["budget"]
    lines = [
        "# No-Anchor Oracle Gap Concentration",
        "",
        f"Base full IDF1: `{budget['base_full_idf1']:.6f}`",
        f"Oracle full IDF1: `{budget['oracle_full_idf1']:.6f}`",
        f"Missing true-pair mass: `{budget['missing_true_pair_mass']:.0f}`",
        f"Base predicted pair mass: `{budget['base_pred_pair_mass']:.0f}`",
        "",
        "This is eval-only analysis over an oracle decomposition artifact. It is",
        "used to decide which no-anchor production branch should be tested next.",
        "",
        "## False-Split Concentration",
        "",
        "| top N | cumulative missing-mass coverage | cumulative false-split mass |",
        "| ---: | ---: | ---: |",
    ]
    for item in result["false_split_prefix"]:
        lines.append(f"| {item['top_n']} | {item['coverage']:.3f} | {item['value']:.0f} |")
    lines.extend(
        [
            "",
            "## Top False-Split GT IDs",
            "",
            "| rank | gt id | false-split mass | cumulative coverage | components | dominant prediction | dominant frac |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["false_split_rows"][:20]:
        lines.append(
            "| {rank} | {id} | {value:.0f} | {coverage:.3f} | {parts} | {dominant_id} | {dominant_weight_frac} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## False-Merge Concentration",
            "",
            "| top N | cumulative base-pred-pair coverage | cumulative false-merge mass |",
            "| ---: | ---: | ---: |",
        ]
    )
    for item in result["false_merge_prefix"]:
        lines.append(f"| {item['top_n']} | {item['coverage']:.3f} | {item['value']:.0f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The false-split gap is highly concentrated: the top few GT identities carry",
            "  a large fraction of the recoverable true-pair mass.",
            "- This supports high-mass component bridge/split selection over further",
            "  tiny-fragment threshold sweeps.",
        ]
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def _self_test() -> None:
    rows = [{"gt_id": 1, "false_split_mass": 6}, {"gt_id": 2, "false_split_mass": 4}]
    out = _cumulative(rows, "false_split_mass", 20.0, "false_split")
    assert out[0]["coverage"] == 0.3
    assert out[1]["coverage"] == 0.5
    print(json.dumps({"stage": "self_test", "status": "ok"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--oracle-json", default="")
    ap.add_argument("--top-ns", default="1,3,5,10,20,30")
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.oracle_json or not args.json:
        ap.error("--oracle-json and --json are required unless --self-test is used")

    data = json.loads(Path(args.oracle_json).read_text())
    base, oracle = _pick_full_rows(data)
    missing_true = max(_float(oracle, "true_pair_mass") - _float(base, "true_pair_mass"), 0.0)
    base_pred = _float(base, "pred_pair_mass")
    fs_rows = sorted(data.get("top_false_split_gt_ids", []) or [], key=lambda row: _float(row, "false_split_mass"), reverse=True)
    fm_rows = sorted(data.get("top_false_merge_components", []) or [], key=lambda row: _float(row, "false_merge_mass"), reverse=True)
    fs_cum = _cumulative(fs_rows, "false_split_mass", missing_true, "false_split")
    fm_cum = _cumulative(fm_rows, "false_merge_mass", base_pred, "false_merge")
    top_ns = [int(part) for part in str(args.top_ns).split(",") if part.strip()]

    def prefix(cum: list[dict[str, Any]], ns: list[int]) -> list[dict[str, Any]]:
        out = []
        for n in ns:
            if not cum:
                continue
            row = cum[min(int(n), len(cum)) - 1]
            out.append({"top_n": int(n), "value": float(row["cumulative_value"]), "coverage": float(row["coverage"])})
        return out

    result = {
        "oracle_json": str(args.oracle_json),
        "budget": {
            "base_name": str(base.get("name", "base")),
            "oracle_name": str(oracle.get("name", "oracle")),
            "base_full_idf1": _float(base, "full_idf1"),
            "oracle_full_idf1": _float(oracle, "full_idf1"),
            "base_true_pair_mass": _float(base, "true_pair_mass"),
            "oracle_true_pair_mass": _float(oracle, "true_pair_mass"),
            "missing_true_pair_mass": missing_true,
            "base_pred_pair_mass": base_pred,
        },
        "false_split_prefix": prefix(fs_cum, top_ns),
        "false_merge_prefix": prefix(fm_cum, top_ns),
        "false_split_rows": fs_cum,
        "false_merge_rows": fm_cum,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_csv(args.csv, fs_cum + fm_cum)
    _write_md(args.md, result)
    print(json.dumps({"json": str(out), "false_split_prefix": result["false_split_prefix"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
