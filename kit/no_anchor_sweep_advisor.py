#!/usr/bin/env python
"""Suggest next no-anchor VLINCS experiments from a gate summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metric(row: dict[str, Any] | None, key: str) -> float:
    if not row:
        return -1.0
    value = row.get(f"{key}_normalized", row.get(key))
    if value is None or value == "":
        return -1.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return -1.0
    return out / 100.0 if out > 1.5 else out


def _case_command(run_script: str, case: str) -> str:
    return f"bash {run_script} {case}"


def _metric_snapshot(gate: dict[str, Any]) -> dict[str, float]:
    req = gate.get("requirements", {})
    best = gate.get("best_joint") or gate.get("best_global") or gate.get("best_e2e") or {}
    names = [
        str(req.get("global_metric", "tracklet_pair_f1")),
        str(req.get("precision_metric", "tracklet_pair_precision")),
        str(req.get("recall_metric", "tracklet_pair_recall")),
        str(req.get("e2e_metric", "full_idf1")),
    ]
    return {name: round(_metric(best, name), 6) for name in names}


def _thresholds(gate: dict[str, Any]) -> dict[str, float]:
    req = gate.get("requirements", {})
    return {
        str(req.get("global_metric", "tracklet_pair_f1")): float(req.get("global_threshold", 0.70)),
        str(req.get("precision_metric", "tracklet_pair_precision")): float(req.get("precision_threshold", 0.70)),
        str(req.get("recall_metric", "tracklet_pair_recall")): float(req.get("recall_threshold", 0.70)),
        str(req.get("e2e_metric", "full_idf1")): float(req.get("e2e_threshold", 0.70)),
    }


def _diagnose(gate: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    metrics = _metric_snapshot(gate)
    thresholds = _thresholds(gate)
    deficits = {name: thresholds[name] - metrics.get(name, -1.0) for name in thresholds}
    notes: list[str] = []
    cases: list[str] = []

    if gate.get("pass_joint"):
        return "passed", ["gate already passes all no-anchor requirements"], ["gate"]
    if int(gate.get("eligible_no_anchor_rows", 0)) <= 0:
        return "no_eligible_no_anchor_rows", ["no eligible no-anchor rows were found; rerun target or inspect metadata"], ["target"]
    if int(gate.get("rows_loaded", 0)) <= 0:
        return "no_rows_loaded", ["gate loaded no rows; check sweep output glob paths"], ["target"]

    missing_full = metrics.get("full_idf1", -1.0) < 0.0
    if missing_full:
        notes.append("no full_idf1 is present in the best row; run full scoring for more candidates")
        cases.extend(["consensus-attach-teacher-auto", "consensus-attach-allpairs-auto", "auto-admission"])

    precision_gap = deficits.get("tracklet_pair_precision", 0.0)
    recall_gap = deficits.get("tracklet_pair_recall", 0.0)
    f1_gap = deficits.get("tracklet_pair_f1", 0.0)
    e2e_gap = deficits.get("full_idf1", 0.0)

    if recall_gap > max(precision_gap, e2e_gap, f1_gap, 0.0):
        notes.append("recall is the largest gap; expand candidate/attach coverage while keeping cannot-link guards")
        cases.extend(["consensus-attach-teacher-auto", "consensus-attach-allpairs-auto", "consensus-guard-stream-auto"])
        return "recall_limited", notes, _dedupe(cases)
    if precision_gap > max(recall_gap, e2e_gap, f1_gap, 0.0):
        notes.append("precision is the largest gap; prefer guarded consensus and stricter component attach")
        cases.extend(["consensus-guard-auto", "consensus-attach-auto", "consensus-guard-stream-auto"])
        return "precision_limited", notes, _dedupe(cases)
    if e2e_gap > max(precision_gap, recall_gap, f1_gap, 0.0):
        notes.append("tracklet association may be acceptable but full IDF1 is low; focus on M3 admission and per-video detector-quality slices")
        cases.extend(["auto-admission", "admission", "consensus-attach-teacher-auto"])
        return "e2e_or_admission_limited", notes, _dedupe(cases)
    if f1_gap > 0:
        notes.append("pair F1 is still below target without a single dominant precision/recall gap; run mixed guard/attach ablations")
        cases.extend(["consensus-guard-auto", "consensus-attach-teacher-auto", "consensus-attach-allpairs-auto", "auto-admission"])
        return "balanced_pair_limited", notes, _dedupe(cases)

    notes.append("all fast metrics look near target; rerun gate after full scoring and inspect per-video rows")
    cases.extend(["gate", "auto-admission", "consensus-attach-teacher-auto"])
    return "needs_full_or_per_video_check", notes, _dedupe(cases)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _render_text(result: dict[str, Any]) -> str:
    lines = [
        f"bottleneck: {result['bottleneck']}",
        f"pass_joint: {result['pass_joint']}",
        "metrics:",
    ]
    for key, value in result["best_joint_metrics"].items():
        threshold = result["thresholds"].get(key)
        lines.append(f"  - {key}: {value:.6f} target={threshold:.6f}")
    lines.append("recommended_cases:")
    for case in result["recommended_cases"]:
        lines.append(f"  - {case}")
    lines.append("commands:")
    for command in result["commands"]:
        lines.append(f"  - {command}")
    lines.append("notes:")
    for note in result["notes"]:
        lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gate_json", help="JSON generated by kit/no_anchor_result_gate.py")
    ap.add_argument("--run-script", default="kit/run_no_anchor_ds1_pair_model_experiments.sh")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--text-out", default=None)
    args = ap.parse_args()

    gate_path = Path(args.gate_json)
    gate = json.loads(gate_path.read_text())
    bottleneck, notes, cases = _diagnose(gate)
    result = {
        "gate_json": str(gate_path),
        "pass_joint": bool(gate.get("pass_joint")),
        "bottleneck": bottleneck,
        "best_joint": gate.get("best_joint"),
        "best_global": gate.get("best_global"),
        "best_e2e": gate.get("best_e2e"),
        "best_joint_metrics": _metric_snapshot(gate),
        "thresholds": _thresholds(gate),
        "eligible_no_anchor_rows": int(gate.get("eligible_no_anchor_rows", 0)),
        "rows_loaded": int(gate.get("rows_loaded", 0)),
        "recommended_cases": cases,
        "commands": [_case_command(args.run_script, case) for case in cases],
        "notes": notes,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n")
    if args.text_out:
        Path(args.text_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.text_out).write_text(_render_text(result))
    print(text)


if __name__ == "__main__":
    main()
