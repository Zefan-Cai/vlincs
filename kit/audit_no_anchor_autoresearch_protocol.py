#!/usr/bin/env python
"""Audit the no-anchor AutoResearch protocol for selector leakage.

This is a protocol guard, not a scorer.  It checks that production-facing
candidate rows do not depend on anchors, oracle selectors, or GT-selected
fields, while allowing historical full-score labels to exist only as
post-hoc training labels for frozen proxy reviewers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.no_anchor_fullscore_scheduler import _oracle_or_gt_selection_reason


ROW_KEYS = (
    "selected",
    "rows",
    "top_rows",
    "top",
    "results",
    "full_rows",
    "top_full_rows",
    "candidate_rows",
)

GT_SELECTION_KEYS = {
    "uses_gt_for_filter_selection",
    "selection_uses_gt_metric",
    "selector_uses_gt",
    "uses_gt_for_training_or_anchors",
    "uses_anchors",
}

POSTHOC_LABEL_KEYS = {
    "known_full_idf1",
    "known_full_hota",
    "known_full_assa",
    "full_idf1",
    "full_hota",
    "full_assa",
    "idf1",
    "hota",
    "assa",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ROW_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    if not rows and any(
        key in data
        for key in (
            "pair_f1",
            "tracklet_pair_f1",
            "predicted_full_idf1",
            "policy",
            "assignment_info",
            "gt_available",
            "uses_gt_for_training_or_anchors",
            "uses_anchors",
        )
    ):
        rows.append(data)
    return rows


def _load_all_rows(paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for text in paths:
        matches = sorted(Path().glob(text)) if any(ch in text for ch in "*?[]") else [Path(text)]
        for path in matches:
            if not path.is_file():
                continue
            for rank, row in enumerate(_load_rows(path), start=1):
                out.append({"_audit_source_file": str(path), "_audit_source_rank": rank, **row})
    return out


def _row_findings(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    reason = _oracle_or_gt_selection_reason(row)
    if reason:
        hard.append(str(reason))
    for key in sorted(GT_SELECTION_KEYS):
        if _as_bool(row.get(key)):
            hard.append(key)
    for key in sorted(POSTHOC_LABEL_KEYS):
        if row.get(key) not in (None, ""):
            soft.append(key)
    if str(row.get("mode", "")).lower().startswith("oracle_"):
        hard.append("oracle_mode")
    source_text = " ".join(
        str(row.get(key, ""))
        for key in ("_audit_source_file", "_source_file", "artifact", "source")
    ).lower()
    if "oracle" in source_text or "with_gt" in source_text or "pervideo_filter_oracle" in source_text:
        hard.append("oracle_or_gt_path")
    return sorted(set(hard)), sorted(set(soft))


def _audit_models(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in paths:
        matches = sorted(Path().glob(text)) if any(ch in text for ch in "*?[]") else [Path(text)]
        for path in matches:
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            cols = [str(col) for col in data.get("columns", [])]
            forbidden_cols = sorted((GT_SELECTION_KEYS | POSTHOC_LABEL_KEYS).intersection(cols))
            rows.append(
                {
                    "path": str(path),
                    "columns": len(cols),
                    "forbidden_feature_columns": forbidden_cols,
                    "uses_gt_for_training_or_anchors": _as_bool(data.get("uses_gt_for_training_or_anchors")),
                    "uses_gt_for_posthoc_full_score_labels": _as_bool(data.get("uses_gt_for_posthoc_full_score_labels")),
                    "loocv_corr": _as_float(data.get("loocv_corr")),
                    "loocv_mae": _as_float(data.get("loocv_mae")),
                    "loocv_rmse": _as_float(data.get("loocv_rmse")),
                }
            )
    return rows


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_all_rows(args.candidate)
    row_reports = []
    hard_count = 0
    soft_count = 0
    for row in rows:
        hard, soft = _row_findings(row)
        if hard:
            hard_count += 1
        if soft:
            soft_count += 1
        if hard or soft:
            row_reports.append(
                {
                    "source_file": row.get("_audit_source_file"),
                    "source_rank": row.get("_audit_source_rank"),
                    "mode": row.get("mode"),
                    "scheduler_family": row.get("scheduler_family"),
                    "hard_blockers": hard,
                    "posthoc_label_fields_present": soft,
                    "predicted_full_idf1": _as_float(
                        row.get("predicted_full_idf1"), _as_float(row.get("learned_proxy_full_idf1"))
                    ),
                    "pair_f1": _as_float(row.get("pair_f1_norm"), _as_float(row.get("tracklet_pair_f1"))),
                }
            )

    model_reports = _audit_models(args.proxy_model_json)
    model_hard = [
        row
        for row in model_reports
        if row["uses_gt_for_training_or_anchors"] or row["forbidden_feature_columns"]
    ]
    pass_protocol = hard_count == 0 and not model_hard
    result = {
        "candidate_inputs": args.candidate,
        "proxy_model_inputs": args.proxy_model_json,
        "candidate_rows": len(rows),
        "candidate_rows_with_hard_blockers": hard_count,
        "candidate_rows_with_posthoc_labels": soft_count,
        "proxy_models": model_reports,
        "proxy_models_with_hard_blockers": model_hard,
        "row_findings": row_reports[: int(args.max_findings)],
        "pass_protocol": bool(pass_protocol),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "deli_distillation": {
            "state_files": "progress.json, directions_tried.json, findings.jsonl",
            "proposer": "no-anchor candidate generators",
            "reviewer": "proxy ensemble and result gate",
            "opponent": "false-split coverage audit and full DS1 scorer",
            "pivot_rule": "after stale iterations, change structural evidence source instead of retuning thresholds",
            "current_action": "only schedule candidates that pass leakage audit; mark eval-only opponents explicitly",
        },
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor AutoResearch Protocol Audit",
        "",
        "This report distills the Deli AutoResearch protocol into the VLINCS no-anchor global-ID loop.",
        "",
        "## Distillation",
        "",
        "- State is file-backed: `progress.json`, `directions_tried.json`, and append-only findings.",
        "- Proposers generate no-anchor candidate assignments.",
        "- Proxy reviewers rank scarce full-score slots, but are not completion evidence.",
        "- Opponents can veto: result gate, false-split coverage audit, and canonical DS1 scorer.",
        "- GT/oracle data is allowed only after prediction, as evaluator or historical proxy label.",
        "",
        "## Audit Summary",
        "",
        f"- candidate rows: `{result['candidate_rows']}`",
        f"- rows with hard blockers: `{result['candidate_rows_with_hard_blockers']}`",
        f"- rows carrying post-hoc labels: `{result['candidate_rows_with_posthoc_labels']}`",
        f"- proxy models: `{len(result['proxy_models'])}`",
        f"- proxy models with hard blockers: `{len(result['proxy_models_with_hard_blockers'])}`",
        f"- pass protocol: `{str(result['pass_protocol']).lower()}`",
        "",
        "## Proxy Models",
        "",
        "| model | columns | forbidden feature columns | posthoc labels | corr | mae |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for model in result["proxy_models"]:
        forbidden = ", ".join(model["forbidden_feature_columns"]) or "-"
        lines.append(
            f"| `{model['path']}` | `{model['columns']}` | `{forbidden}` | "
            f"`{str(model['uses_gt_for_posthoc_full_score_labels']).lower()}` | "
            f"`{model.get('loocv_corr')}` | `{model.get('loocv_mae')}` |"
        )
    lines.extend(["", "## Row Findings", ""])
    if not result["row_findings"]:
        lines.append("- no hard blockers or post-hoc row labels found")
    for row in result["row_findings"][:30]:
        hard = ", ".join(row["hard_blockers"]) or "-"
        soft = ", ".join(row["posthoc_label_fields_present"]) or "-"
        lines.append(
            f"- `{row['source_file']}` rank `{row['source_rank']}` mode `{row.get('mode')}`: "
            f"hard `{hard}`, posthoc `{soft}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        cand = root / "cand.json"
        cand.write_text(
            json.dumps(
                {
                    "selected": [
                        {"mode": "clean", "pair_f1": 0.8, "predicted_full_idf1": 0.66},
                        {"mode": "oracle_bad", "uses_gt_for_filter_selection": True, "full_idf1": 0.9},
                    ]
                }
            ),
            encoding="utf-8",
        )
        model = root / "model.json"
        model.write_text(
            json.dumps(
                {
                    "columns": ["pair_f1", "moved_tracklets"],
                    "coef": [1.0, 0.0],
                    "uses_gt_for_training_or_anchors": False,
                    "uses_gt_for_posthoc_full_score_labels": True,
                }
            ),
            encoding="utf-8",
        )
        out = audit(
            argparse.Namespace(
                candidate=[str(cand)],
                proxy_model_json=[str(model)],
                json=str(root / "out.json"),
                md=str(root / "out.md"),
                max_findings=20,
            )
        )
        assert out["candidate_rows"] == 2, out
        assert out["candidate_rows_with_hard_blockers"] == 1, out
        assert not out["pass_protocol"], out
        assert "Distillation" in Path(root / "out.md").read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", action="append", default=[])
    ap.add_argument("--proxy-model-json", action="append", default=[])
    ap.add_argument("--json", default="local_runs/no_anchor_autoresearch_protocol_audit_20260620.json")
    ap.add_argument("--md", default="reports/no_anchor_autoresearch_protocol_audit_20260620.md")
    ap.add_argument("--max-findings", type=int, default=80)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    if not args.candidate:
        ap.error("--candidate is required")
    result = audit(args)
    print(
        json.dumps(
            {
                "candidate_rows": result["candidate_rows"],
                "hard_blockers": result["candidate_rows_with_hard_blockers"],
                "posthoc_rows": result["candidate_rows_with_posthoc_labels"],
                "pass_protocol": result["pass_protocol"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
