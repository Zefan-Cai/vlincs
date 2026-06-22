#!/usr/bin/env python
"""Deli-style opponent check for the no-anchor VLINCS research loop.

This script does not train, select anchors, or inspect identity GT.  It reads
the already-written result gate and AutoResearch progress files, then decides
whether the current proposer should be accepted, rejected, or forced to pivot.
The intent is to make the "self-play opponent" explicit: pair metrics can
propose, but full DS1 IDF1/HOTA and no-anchor metadata get to veto.
"""

from __future__ import annotations

import argparse
import json
import tempfile
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


def _short_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    keys = [
        "source",
        "source_kind",
        "source_rank",
        "mode",
        "no_anchor_ok",
        "no_anchor_evidence",
        "tracklet_pair_f1",
        "tracklet_pair_precision",
        "tracklet_pair_recall",
        "full_idf1",
        "full_hota",
        "full_assa",
    ]
    out = {key: row.get(key) for key in keys if key in row}
    for metric in ("tracklet_pair_f1", "tracklet_pair_precision", "tracklet_pair_recall", "full_idf1"):
        out[f"{metric}_normalized"] = round(_metric(row, metric), 6)
    return out


def build_verdict(gate: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
    req = gate.get("requirements", {})
    pair_key = str(req.get("global_metric", "tracklet_pair_f1"))
    precision_key = str(req.get("precision_metric", "tracklet_pair_precision"))
    recall_key = str(req.get("recall_metric", "tracklet_pair_recall"))
    e2e_key = str(req.get("e2e_metric", "full_idf1"))
    pair_goal = float(progress.get("goal_global_id_pair_f1", req.get("global_threshold", 0.70)))
    precision_goal = float(progress.get("goal_global_id_pair_precision", req.get("precision_threshold", 0.70)))
    recall_goal = float(progress.get("goal_global_id_pair_recall", req.get("recall_threshold", 0.70)))
    e2e_goal = float(progress.get("goal_e2e_idf1", req.get("e2e_threshold", 0.70)))
    best_e2e_seen = float(progress.get("best_e2e_idf1", 0.0))
    stale_count = int(progress.get("stale_count", 0))

    best_global = gate.get("best_global") or {}
    best_e2e = gate.get("best_e2e") or {}
    best_joint = gate.get("best_joint") or {}
    best_global_pair = _metric(best_global, pair_key)
    best_global_precision = _metric(best_global, precision_key)
    best_global_recall = _metric(best_global, recall_key)
    best_global_e2e = _metric(best_global, e2e_key)
    best_e2e_value = _metric(best_e2e, e2e_key)
    best_joint_e2e = _metric(best_joint, e2e_key)

    pair_model_passes = (
        best_global_pair >= pair_goal
        and best_global_precision >= precision_goal
        and best_global_recall >= recall_goal
    )
    pair_overfit = pair_model_passes and best_global_e2e >= 0.0 and best_global_e2e < e2e_goal
    e2e_passes = bool(gate.get("pass_joint")) or best_e2e_value >= e2e_goal
    improves_standing_e2e = best_e2e_value > best_e2e_seen + 1.0e-9
    forced_structural_pivot = stale_count >= 2 and not e2e_passes
    escalate_external_blocker = stale_count >= 4 and not e2e_passes

    blockers: list[str] = []
    if int(gate.get("eligible_no_anchor_rows", 0)) <= 0:
        blockers.append("no eligible no-anchor rows")
    if pair_overfit:
        blockers.append("pair/global metric passes but full DS1 IDF1 is below target")
    if best_e2e_value >= 0.0 and not improves_standing_e2e:
        blockers.append("best newly gated full IDF1 does not beat standing promoted artifact")
    if forced_structural_pivot:
        blockers.append("stale_count requires a structural pivot, not another threshold-only sweep")

    if e2e_passes:
        verdict = "accept"
    elif forced_structural_pivot:
        verdict = "pivot"
    else:
        verdict = "reject"

    next_actions = []
    if verdict != "accept":
        next_direction = progress.get("next_direction")
        if next_direction:
            next_actions.append(str(next_direction))
        next_actions.append("run edge-rank target-fragment launcher when Pluto/SSH recovers")
        next_actions.append("spend full-score budget only on diverse unscored candidates")
        next_actions.append("reject pair-only winners unless full DS1 IDF1/HOTA improves")

    return {
        "verdict": verdict,
        "pass_joint": bool(gate.get("pass_joint")),
        "stale_count": stale_count,
        "forced_structural_pivot": bool(forced_structural_pivot),
        "escalate_external_blocker": bool(escalate_external_blocker),
        "pair_model_passes": bool(pair_model_passes),
        "pair_overfit_rejected_by_opponent": bool(pair_overfit),
        "improves_standing_e2e": bool(improves_standing_e2e),
        "standing_best_e2e_idf1": round(best_e2e_seen, 6),
        "best_gate_e2e_idf1": round(best_e2e_value, 6),
        "best_gate_joint_e2e_idf1": round(best_joint_e2e, 6),
        "requirements": {
            "pair_f1": pair_goal,
            "pair_precision": precision_goal,
            "pair_recall": recall_goal,
            "full_idf1": e2e_goal,
            "requires_no_anchor": True,
            "gt_allowed_for_evaluation_only": True,
        },
        "blockers": blockers,
        "next_actions": next_actions,
        "best_global": _short_row(best_global),
        "best_e2e": _short_row(best_e2e),
        "best_joint": _short_row(best_joint),
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Deli Opponent Verdict",
        "",
        f"- verdict: `{result['verdict']}`",
        f"- pass_joint: `{str(result['pass_joint']).lower()}`",
        f"- stale_count: `{result['stale_count']}`",
        f"- standing best full IDF1: `{result['standing_best_e2e_idf1']:.6f}`",
        f"- best gated full IDF1: `{result['best_gate_e2e_idf1']:.6f}`",
        f"- pair winner rejected by opponent: `{str(result['pair_overfit_rejected_by_opponent']).lower()}`",
        "",
        "## Blockers",
    ]
    for blocker in result["blockers"]:
        lines.append(f"- {blocker}")
    if not result["blockers"]:
        lines.append("- none")
    lines.extend(["", "## Next Actions"])
    for action in result["next_actions"]:
        lines.append(f"- {action}")
    if not result["next_actions"]:
        lines.append("- promote accepted candidate")
    return "\n".join(lines) + "\n"


def _self_test() -> None:
    gate = {
        "pass_joint": False,
        "eligible_no_anchor_rows": 2,
        "requirements": {
            "global_metric": "tracklet_pair_f1",
            "precision_metric": "tracklet_pair_precision",
            "recall_metric": "tracklet_pair_recall",
            "e2e_metric": "full_idf1",
        },
        "best_global": {
            "tracklet_pair_f1": 0.95,
            "tracklet_pair_precision": 0.91,
            "tracklet_pair_recall": 0.99,
            "full_idf1": 0.08,
            "no_anchor_ok": True,
        },
        "best_e2e": {"full_idf1": 0.654, "tracklet_pair_f1": 0.77, "no_anchor_ok": True},
        "best_joint": {"full_idf1": 0.654, "tracklet_pair_f1": 0.77, "no_anchor_ok": True},
    }
    progress = {
        "goal_global_id_pair_f1": 0.70,
        "goal_global_id_pair_precision": 0.70,
        "goal_global_id_pair_recall": 0.70,
        "goal_e2e_idf1": 0.70,
        "best_e2e_idf1": 0.65524,
        "stale_count": 17,
        "next_direction": "structural test",
    }
    result = build_verdict(gate, progress)
    assert result["verdict"] == "pivot", result
    assert result["pair_overfit_rejected_by_opponent"] is True, result
    assert result["forced_structural_pivot"] is True, result
    assert result["improves_standing_e2e"] is False, result
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "opponent.md"
        out.write_text(_render_markdown(result))
        assert "pair winner rejected" in out.read_text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-json", required=False)
    ap.add_argument("--progress-json", required=False)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "ok"}, sort_keys=True))
        return
    if not args.gate_json or not args.progress_json:
        raise SystemExit("--gate-json and --progress-json are required unless --self-test is set")

    gate = json.loads(Path(args.gate_json).read_text())
    progress = json.loads(Path(args.progress_json).read_text())
    result = build_verdict(gate, progress)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text(_render_markdown(result))
    print(text)


if __name__ == "__main__":
    main()
