#!/usr/bin/env python
"""Audit no-anchor scheduler candidates for replay and side-effect risk.

This is a production-side admission guard.  It does not use GT labels to decide
which candidate is good; optional full-score summaries are attached only as
historical labels for post-hoc calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.export_no_anchor_scheduler_manifest_assignments import (
    _accepted_preview,
    _intish,
    _load_assignment_rows,
    _load_scheduler_selected,
    _preview_component,
    _replay_preview,
    _source_seqs_for_preview_item,
)
from kit.no_anchor_fullscore_scheduler import _as_float


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_known_labels(paths: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for text in paths:
        path = Path(text)
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        manifest = str(data.get("source_manifest") or "")
        for row in data.get("rows", []):
            if not isinstance(row, dict) or "selection_rank" not in row:
                continue
            labels[(manifest, int(row["selection_rank"]))] = {
                "known_status": row.get("status", "scored" if row.get("full_idf1") is not None else "unknown"),
                "known_reason": row.get("reason", ""),
                "known_full_idf1": _as_float(row.get("full_idf1")),
                "known_full_hota": _as_float(row.get("full_hota")),
                "known_full_assa": _as_float(row.get("full_assa")),
            }
    return labels


def _group_rows(rows: list[dict[str, str]]) -> tuple[dict[int, dict[str, str]], dict[int, list[dict[str, str]]]]:
    by_seq: dict[int, dict[str, str]] = {}
    by_component: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        seq = _intish(row["seq"])
        comp = _intish(row["component_label"])
        by_seq[seq] = row
        by_component[comp].append(row)
    return by_seq, by_component


def _interval(row: dict[str, str]) -> tuple[int, int]:
    return _as_int(row.get("start_frame")), _as_int(row.get("end_frame"))


def _gap_or_overlap(a: dict[str, str], b: dict[str, str]) -> tuple[int, bool]:
    a0, a1 = _interval(a)
    b0, b1 = _interval(b)
    if a0 <= b1 and b0 <= a1:
        return 0, True
    return min(abs(a0 - b1), abs(b0 - a1)), False


def _preview_stats(
    preview: list[dict[str, Any]],
    base_rows: list[dict[str, str]],
    by_seq: dict[int, dict[str, str]],
    by_component: dict[int, list[dict[str, str]]],
) -> dict[str, Any]:
    source_rows: list[dict[str, str]] = []
    target_rows: list[dict[str, str]] = []
    source_components: set[int] = set()
    target_components: set[int] = set()
    missing_targets: set[int] = set()
    missing_sources: set[int] = set()

    for item in preview:
        target = _preview_component(item, "target_component", "target", "target_rep")
        if target is None:
            missing_targets.add(-1)
            continue
        target_components.add(int(target))
        target_part = by_component.get(int(target), [])
        if not target_part:
            missing_targets.add(int(target))
        target_rows.extend(target_part)
        try:
            source_seqs, source_component = _source_seqs_for_preview_item(item, base_rows)
        except Exception:
            source_seqs, source_component = [], _preview_component(item, "source_component_label", "source", "source_rep")
        if source_component is not None:
            source_components.add(int(source_component))
        if not source_seqs and source_component is not None:
            missing_sources.add(int(source_component))
        for seq in source_seqs:
            row = by_seq.get(_intish(seq))
            if row is None:
                missing_sources.add(_intish(seq))
            else:
                source_components.add(_intish(row["component_label"]))
                source_rows.append(row)

    overlap_pairs = 0
    same_video_pairs = 0
    min_gap: int | None = None
    for src in source_rows:
        for tgt in target_rows:
            if src.get("video") != tgt.get("video"):
                continue
            same_video_pairs += 1
            gap, overlap = _gap_or_overlap(src, tgt)
            min_gap = gap if min_gap is None else min(min_gap, gap)
            if overlap:
                overlap_pairs += 1

    return {
        "source_components": sorted(source_components),
        "target_components": sorted(target_components),
        "missing_source_refs": sorted(missing_sources),
        "missing_target_components": sorted(missing_targets),
        "source_tracklets": len({row["seq"] for row in source_rows}),
        "target_tracklets": len({row["seq"] for row in target_rows}),
        "source_video_count": len({row.get("video", "") for row in source_rows}),
        "target_video_count": len({row.get("video", "") for row in target_rows}),
        "same_video_source_target_pairs": same_video_pairs,
        "temporal_overlap_pairs": overlap_pairs,
        "min_same_video_gap": min_gap,
        "source_avg_conf": _mean([float(row.get("avg_conf", "nan")) for row in source_rows]),
        "target_avg_conf": _mean([float(row.get("avg_conf", "nan")) for row in target_rows]),
        "target_component_size_max": max((len(by_component[c]) for c in target_components), default=0),
    }


def _risk_and_recommendation(row: dict[str, Any]) -> tuple[float, str, list[str]]:
    reasons: list[str] = []
    if row["replay_status"] != "ok":
        return 99.0, "reject_replay_failure", [str(row["replay_error"])]
    risk = 0.0
    moved = int(row["moved_tracklets"])
    if moved > 20:
        risk += 2.0
        reasons.append("moved_tracklets>20")
    if int(row["accepted_preview_count"]) > 1:
        risk += 1.5
        reasons.append("multi_edge_preview")
    if int(row["temporal_overlap_pairs"]) > 0:
        risk += 3.0
        reasons.append("same_video_temporal_overlap")
    if int(row["target_video_count"]) >= 5:
        risk += 1.0
        reasons.append("large_multivideo_target")
    if int(row["target_component_size_max"]) >= 200:
        risk += 0.75
        reasons.append("large_target_component")
    if row.get("known_full_idf1") is not None and float(row["known_full_idf1"]) < 0.65524:
        risk += 1.0
        reasons.append("historical_fullscore_below_best")
    if row.get("known_status") not in (None, "", "scored", "unknown"):
        risk += 2.0
        reasons.append(str(row["known_status"]))
    if risk >= 4:
        return risk, "reject_or_quarantine", reasons
    if risk >= 2:
        return risk, "defer_needs_counterfactual", reasons
    return risk, "safe_to_fullscore", reasons or ["low_no_gt_side_effect_risk"]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    base_rows, _fields = _load_assignment_rows(Path(args.base_assignment_csv))
    by_seq, by_component = _group_rows(base_rows)
    labels = _load_known_labels(args.fullscore_summary)
    rows: list[dict[str, Any]] = []

    for manifest_text in args.scheduler_json:
        manifest_path = Path(manifest_text)
        for cand in _load_scheduler_selected(manifest_path):
            rank = int(cand["_selection_rank"])
            out: dict[str, Any] = {
                "manifest": str(manifest_path),
                "selection_rank": rank,
                "source_file": cand.get("_source_file", ""),
                "source_rank": cand.get("_source_rank", ""),
                "mode": cand.get("mode", ""),
                "scheduler_family": cand.get("scheduler_family", ""),
                "predicted_full_idf1": _as_float(cand.get("predicted_full_idf1")),
                "pair_f1": _as_float(cand.get("pair_f1"), _as_float(cand.get("tracklet_pair_f1"))),
            }
            out.update(labels.get((str(manifest_path), rank), {}))
            try:
                preview = _accepted_preview(cand)
                stats = _preview_stats(preview, base_rows, by_seq, by_component)
                _replayed, replay_info = _replay_preview(base_rows, preview, allow_missing_source_seqs=False)
                out.update(stats)
                out.update(
                    {
                        "accepted_preview_count": len(preview),
                        "moved_tracklets": replay_info["moved_tracklets"],
                        "replay_status": "ok",
                        "replay_error": "",
                    }
                )
            except Exception as exc:
                out.update(
                    {
                        "accepted_preview_count": 0,
                        "moved_tracklets": int(_as_float(cand.get("moved_tracklets"), 0) or 0),
                        "replay_status": "failed",
                        "replay_error": str(exc),
                        "source_components": [],
                        "target_components": [],
                        "missing_source_refs": [],
                        "missing_target_components": [],
                        "source_tracklets": 0,
                        "target_tracklets": 0,
                        "source_video_count": 0,
                        "target_video_count": 0,
                        "same_video_source_target_pairs": 0,
                        "temporal_overlap_pairs": 0,
                        "min_same_video_gap": None,
                        "source_avg_conf": None,
                        "target_avg_conf": None,
                        "target_component_size_max": 0,
                    }
                )
            risk, recommendation, reasons = _risk_and_recommendation(out)
            out["side_effect_risk"] = round(risk, 3)
            out["recommendation"] = recommendation
            out["risk_reasons"] = reasons
            rows.append(out)

    summary = {
        "date": args.date,
        "base_assignment_csv": args.base_assignment_csv,
        "scheduler_json": args.scheduler_json,
        "candidate_rows": len(rows),
        "recommendation_counts": dict(Counter(row["recommendation"] for row in rows)),
        "rows": sorted(rows, key=lambda row: (row["side_effect_risk"], -float(row.get("predicted_full_idf1") or 0))),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "posthoc_fullscore_labels_only": bool(args.fullscore_summary),
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), summary["rows"])
    if args.md:
        _write_md(Path(args.md), summary)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "manifest",
        "selection_rank",
        "mode",
        "scheduler_family",
        "predicted_full_idf1",
        "known_full_idf1",
        "pair_f1",
        "replay_status",
        "moved_tracklets",
        "accepted_preview_count",
        "target_components",
        "target_tracklets",
        "target_video_count",
        "temporal_overlap_pairs",
        "target_component_size_max",
        "side_effect_risk",
        "recommendation",
        "risk_reasons",
        "replay_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Scheduler Side-Effect Admission Audit",
        "",
        f"Date: {summary['date']}",
        "",
        "This audit is the VLINCS adaptation of the Deli AutoResearch pivot rule: after repeated score drops, change the structural guard instead of retuning thresholds.  It uses no anchors and no GT for production selection; optional full-score values are shown only as post-hoc labels.",
        "",
        "## Summary",
        "",
        f"- candidate rows: `{summary['candidate_rows']}`",
        f"- recommendation counts: `{summary['recommendation_counts']}`",
        f"- base assignment: `{summary['base_assignment_csv']}`",
        "",
        "## Candidate Rows",
        "",
        "| manifest | rank | moved | preview | proxy | known full | overlap | target videos | target max size | risk | recommendation | reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["rows"]:
        manifest = Path(str(row["manifest"])).name
        lines.append(
            f"| `{manifest}` | {row['selection_rank']} | {row['moved_tracklets']} | "
            f"{row['accepted_preview_count']} | {row.get('predicted_full_idf1')} | "
            f"{row.get('known_full_idf1')} | {row['temporal_overlap_pairs']} | "
            f"{row['target_video_count']} | {row['target_component_size_max']} | "
            f"{row['side_effect_risk']} | `{row['recommendation']}` | "
            f"`{'; '.join(row['risk_reasons'])}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use this as an admission/referee layer before spending canonical full-score budget.  Candidates with replay failure, namespace drift, multi-edge movement, temporal overlap, or historically negative full-score labels should be quarantined or reduced to smaller counterfactuals before remote scoring.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheduler-json", action="append", required=True)
    ap.add_argument("--base-assignment-csv", required=True)
    ap.add_argument("--fullscore-summary", action="append", default=[])
    ap.add_argument("--json", default="local_runs/no_anchor_scheduler_side_effect_audit_20260621.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_scheduler_side_effect_audit_20260621.csv")
    ap.add_argument("--md", default="reports/no_anchor_scheduler_side_effect_audit_20260621.md")
    ap.add_argument("--date", default="2026-06-21")
    args = ap.parse_args()
    result = audit(args)
    print(json.dumps({"candidate_rows": result["candidate_rows"], "recommendation_counts": result["recommendation_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
