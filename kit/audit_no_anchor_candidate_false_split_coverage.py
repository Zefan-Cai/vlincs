#!/usr/bin/env python
"""Audit no-anchor candidate rows against eval-only false-split mass.

This is an AutoResearch opponent, not a production selector.  It reads
candidate rows with accepted_preview edits and asks whether their proposed
source->target component moves touch large false-split GT identities from an
already completed evaluation artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(int(value))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _component_id(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if "+" in text:
        return None
    try:
        num = int(float(text))
    except ValueError:
        return None
    if num >= 70_000_000:
        num -= 70_000_000
    if num >= 90_000_000:
        num -= 90_000_000
    return int(num)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("selected", "rows", "candidate_rows", "top", "results", "top_full_rows", "full_rows"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def _preview(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("accepted_preview")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _edge_components(item: dict[str, Any]) -> tuple[int | None, int | None]:
    source = _component_id(item.get("source_component_label", item.get("source", item.get("source_rep"))))
    target = _component_id(item.get("target_component", item.get("target", item.get("target_rep"))))
    return source, target


def _load_budget(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    budget = data.get("budget") if isinstance(data, dict) else None
    if not isinstance(budget, dict):
        return {}
    return {key: float(value) for key, value in budget.items() if _as_float(value) is not None}


def _error_tables(path: Path) -> tuple[dict[int, dict[int, float]], dict[int, dict[str, Any]]]:
    data = json.loads(path.read_text())
    gt_to_comp: dict[int, dict[int, float]] = {}
    for row in data.get("top_false_split_gt_ids", []):
        if not isinstance(row, dict):
            continue
        gt = _component_id(row.get("gt_id"))
        if gt is None:
            continue
        comps: dict[int, float] = {}
        for item in row.get("pred_components", []):
            if not isinstance(item, dict):
                continue
            comp = _component_id(item.get("key"))
            weight = _as_float(item.get("value"), 0.0) or 0.0
            if comp is not None and weight > 0:
                comps[comp] = float(weight)
        if comps:
            gt_to_comp[gt] = comps
    false_merge: dict[int, dict[str, Any]] = {}
    for row in data.get("top_false_merge_components", []):
        if not isinstance(row, dict):
            continue
        comp = _component_id(row.get("predicted_global_id"))
        if comp is not None:
            false_merge[comp] = row
    return gt_to_comp, false_merge


def _edge_audit(
    source: int | None,
    target: int | None,
    gt_to_comp: dict[int, dict[int, float]],
    false_merge: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if source is None or target is None:
        return {
            "source_component": source,
            "target_component": target,
            "positive_bridge_mass": 0.0,
            "positive_gts": [],
            "target_false_merge_mass": 0.0,
            "target_dominant_gt_frac": None,
            "target_gt_count": None,
        }
    positives = []
    total = 0.0
    for gt, comps in gt_to_comp.items():
        if source in comps and target in comps:
            bridge = float(comps[source]) * float(comps[target])
            if bridge > 0:
                total += bridge
                positives.append(
                    {
                        "gt_id": gt,
                        "source_weight": float(comps[source]),
                        "target_weight": float(comps[target]),
                        "bridge_mass": float(bridge),
                    }
                )
    positives.sort(key=lambda item: float(item["bridge_mass"]), reverse=True)
    fm = false_merge.get(target, {})
    return {
        "source_component": int(source),
        "target_component": int(target),
        "positive_bridge_mass": float(total),
        "positive_gts": positives[:5],
        "target_false_merge_mass": _as_float(fm.get("false_merge_mass"), 0.0) or 0.0,
        "target_dominant_gt_frac": _as_float(fm.get("dominant_gt_weight_frac")),
        "target_gt_count": _as_float(fm.get("gt_count")),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(Path(args.candidates_json))
    gt_to_comp, false_merge = _error_tables(Path(args.error_analysis_json))
    budget = _load_budget(Path(args.oracle_gap_json) if args.oracle_gap_json else None)
    denom = float(budget.get("missing_true_pair_mass") or 1.0)
    out_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        preview = _preview(row)
        edge_rows = []
        for item in preview:
            source, target = _edge_components(item)
            edge = _edge_audit(source, target, gt_to_comp, false_merge)
            edge["source_size"] = _as_float(item.get("source_size"))
            edge["target_size"] = _as_float(item.get("target_size"))
            edge["target_mean_sim"] = _as_float(item.get("target_mean_sim"))
            edge["target_best_sim"] = _as_float(item.get("target_best_sim"))
            edge["target_view_vote"] = _as_float(item.get("target_view_vote"))
            edge_rows.append(edge)
        positive_mass = sum(float(edge["positive_bridge_mass"]) for edge in edge_rows)
        target_impure_edges = sum(1 for edge in edge_rows if (_as_float(edge.get("target_gt_count"), 0.0) or 0.0) > 1)
        positive_edges = sum(1 for edge in edge_rows if float(edge["positive_bridge_mass"]) > 0)
        out = {
            "candidate_rank": rank,
            "mode": row.get("mode"),
            "source_file": row.get("_source_file") or row.get("artifact") or args.candidates_json,
            "source_rank": row.get("_source_rank"),
            "moved_tracklets": _as_float(row.get("moved_tracklets")),
            "accepted_reassignments": _as_float(row.get("accepted_reassignments"), len(preview)),
            "predicted_full_idf1": _as_float(row.get("predicted_full_idf1"), _as_float(row.get("learned_proxy_full_idf1"))),
            "pair_f1": _as_float(row.get("pair_f1_norm"), _as_float(row.get("tracklet_pair_f1"), _as_float(row.get("pair_f1")))),
            "pair_precision": _as_float(row.get("pair_precision_norm"), _as_float(row.get("tracklet_pair_precision"), _as_float(row.get("pair_precision")))),
            "pair_recall": _as_float(row.get("pair_recall_norm"), _as_float(row.get("tracklet_pair_recall"), _as_float(row.get("pair_recall")))),
            "audit_positive_bridge_mass": float(positive_mass),
            "audit_gap_coverage": float(positive_mass / denom),
            "audit_positive_edges": int(positive_edges),
            "audit_zero_edges": int(max(len(edge_rows) - positive_edges, 0)),
            "audit_target_impure_edges": int(target_impure_edges),
            "audit_edges": edge_rows,
            "signature": row.get("signature"),
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": True,
        }
        out_rows.append(out)
    out_rows.sort(
        key=lambda item: (
            float(item["audit_positive_bridge_mass"]),
            float(item.get("predicted_full_idf1") or 0.0),
            float(item.get("pair_f1") or 0.0),
        ),
        reverse=True,
    )
    result = {
        "candidates_json": str(args.candidates_json),
        "error_analysis_json": str(args.error_analysis_json),
        "oracle_gap_json": str(args.oracle_gap_json or ""),
        "candidate_rows": int(len(rows)),
        "audited_rows": int(len(out_rows)),
        "missing_true_pair_mass": float(denom),
        "total_positive_bridge_mass": float(sum(float(row["audit_positive_bridge_mass"]) for row in out_rows)),
        "top_rows": out_rows[: int(args.top_n)],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), out_rows[: int(args.top_n)])
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_rank",
        "mode",
        "moved_tracklets",
        "accepted_reassignments",
        "predicted_full_idf1",
        "pair_f1",
        "audit_positive_bridge_mass",
        "audit_gap_coverage",
        "audit_positive_edges",
        "audit_zero_edges",
        "audit_target_impure_edges",
        "source_file",
        "source_rank",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Candidate False-Split Coverage Audit",
        "",
        "This is an eval-only opponent report. It must not be used as a production selector.",
        "",
        f"- candidates: `{result['candidate_rows']}`",
        f"- audited rows: `{result['audited_rows']}`",
        f"- missing true-pair mass denominator: `{result['missing_true_pair_mass']:.3f}`",
        f"- summed positive bridge mass in top rows: `{result['total_positive_bridge_mass']:.3f}`",
        "",
        "| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(result["top_rows"][:30], start=1):
        lines.append(
            f"| {idx} | `{row['candidate_rank']}` | `{row.get('moved_tracklets')}` | "
            f"`{row.get('accepted_reassignments')}` | `{row.get('predicted_full_idf1')}` | "
            f"`{row.get('pair_f1')}` | `{row['audit_gap_coverage']:.6f}` | "
            f"`{row['audit_positive_edges']}` | `{row['audit_target_impure_edges']}` | "
            f"`{row.get('mode')}` |"
        )
    lines.extend(["", "## Top Edge Evidence", ""])
    for row in result["top_rows"][:5]:
        lines.append(f"### Candidate rank {row['candidate_rank']}")
        for edge in row["audit_edges"][:8]:
            gts = edge.get("positive_gts") or []
            best = gts[0] if gts else {}
            lines.append(
                f"- `{edge.get('source_component')}` -> `{edge.get('target_component')}`: "
                f"mass `{edge.get('positive_bridge_mass'):.3f}`, "
                f"best_gt `{best.get('gt_id', '')}`, target_gt_count `{edge.get('target_gt_count')}`"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        err = root / "err.json"
        err.write_text(
            json.dumps(
                {
                    "top_false_split_gt_ids": [
                        {
                            "gt_id": 9,
                            "pred_components": [
                                {"key": "70000001", "value": 10},
                                {"key": "70000002", "value": 5},
                            ],
                        }
                    ],
                    "top_false_merge_components": [
                        {"predicted_global_id": 70000002, "gt_count": 2, "false_merge_mass": 7}
                    ],
                }
            )
        )
        cand = root / "cand.json"
        cand.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "mode": "x",
                            "accepted_preview": [
                                {
                                    "source_component_label": 1,
                                    "target_component": 2,
                                    "source_size": 3,
                                    "target_size": 4,
                                }
                            ],
                        }
                    ]
                }
            )
        )
        gap = root / "gap.json"
        gap.write_text(json.dumps({"budget": {"missing_true_pair_mass": 100.0}}))
        out = audit(
            argparse.Namespace(
                candidates_json=str(cand),
                error_analysis_json=str(err),
                oracle_gap_json=str(gap),
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
                md=str(root / "out.md"),
                top_n=10,
            )
        )
        assert out["top_rows"][0]["audit_positive_bridge_mass"] == 50.0
        assert abs(out["top_rows"][0]["audit_gap_coverage"] - 0.5) < 1e-9
        assert out["top_rows"][0]["audit_target_impure_edges"] == 1
        assert Path(root / "out.csv").read_text()
        assert "eval-only" in Path(root / "out.md").read_text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidates-json", default="local_runs/no_anchor_crossqueue_portfolio_relaxed_candidates_20260620.json")
    ap.add_argument("--error-analysis-json", default="local_runs/remote_h100_test_3_20260619/no_anchor_softcut_then_softoverlap_error_analysis_20260619.json")
    ap.add_argument("--oracle-gap-json", default="local_runs/no_anchor_oracle_gap_concentration_20260620.json")
    ap.add_argument("--json", default="local_runs/no_anchor_candidate_false_split_coverage_audit_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_candidate_false_split_coverage_audit_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_candidate_false_split_coverage_audit_20260620.md")
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    result = audit(args)
    print(
        json.dumps(
            {
                "candidate_rows": result["candidate_rows"],
                "audited_rows": result["audited_rows"],
                "top_coverage": result["top_rows"][0]["audit_gap_coverage"] if result["top_rows"] else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
