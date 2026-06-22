#!/usr/bin/env python
"""Attach no-GT counter-target opponent audit features to candidate rows.

The counter-target verifier emits one edge row per ``accepted_preview`` item.
This utility copies the no-GT referee/opponent fields back into the original
candidate rows so the ordinary full-score scheduler can rank and penalize them
without knowing about the audit artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


EDGE_FIELDS = [
    "verdict",
    "weighted_target_score",
    "weighted_best_alt_score",
    "weighted_margin",
    "rank_vote",
    "margin_vote",
    "view_count",
    "view_margin_min",
    "view_margin_mean",
    "view_margin_max",
    "view_margin_std",
    "view_target_score_min",
    "view_target_score_mean",
    "view_target_score_max",
    "view_target_score_std",
    "view_target_rank_max",
    "view_weak_margin_count",
    "view_negative_margin_count",
    "view_non_rank1_count",
    "view_weak_margin_fraction",
    "view_negative_margin_fraction",
    "view_non_rank1_fraction",
    "visual_opponent_risk_score",
    "combined_opponent_risk_score",
    "source_video_count",
    "target_video_count",
    "shared_video_count",
    "source_target_video_jaccard",
    "source_dominant_video_fraction",
    "target_dominant_video_fraction",
]

TEMPORAL_FIELDS = [
    "max_same_video_overlap_frames",
    "same_video_pair_count",
    "total_pair_count",
    "local_pair_count",
    "min_same_video_gap_frames",
    "overlap_pair_count",
    "same_video_pair_fraction",
    "local_pair_fraction_total",
    "local_pair_fraction_same_video",
    "overlap_pair_fraction_total",
    "overlap_pair_fraction_same_video",
    "median_source_duration_frames",
    "max_overlap_source_duration_fraction",
    "temporal_opponent_risk_score",
]

SUMMARY_FIELDS = [
    "accepted_edges",
    "temporal_rejected_edges",
    "countertarget_rejected_edges",
    "accepted_fraction",
    "min_weighted_margin",
    "mean_weighted_margin",
    "mean_rank_vote",
    "min_view_margin_min",
    "mean_view_margin_min",
    "mean_view_weak_margin_fraction",
    "mean_view_non_rank1_fraction",
    "mean_visual_opponent_risk_score",
    "max_temporal_opponent_risk_score",
    "max_combined_opponent_risk_score",
    "max_same_video_overlap_frames",
    "mean_same_video_pair_fraction",
    "mean_local_pair_fraction_same_video",
    "mean_source_target_video_jaccard",
    "verdict",
]


def _candidate_rows(doc: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(doc, list):
        return "rows", [row for row in doc if isinstance(row, dict)]
    if isinstance(doc, dict):
        for key in ("selected", "top", "rows", "full_rows", "top_full_rows", "results"):
            value = doc.get(key)
            if isinstance(value, list):
                return key, [row for row in value if isinstance(row, dict)]
    return "rows", []


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _audit_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("candidate_pool") or ""),
        int(row.get("candidate_index", -1)),
        int(row.get("edge_index", -1)),
    )


def _summary_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("candidate_pool") or ""), int(row.get("candidate_index", -1)))


def _load_audit(path: Path) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not an audit object")
    edge_map = {
        _audit_key(row): row
        for row in data.get("edge_rows", [])
        if isinstance(row, dict)
    }
    summary_map = {
        _summary_key(row): row
        for row in data.get("candidate_summaries", [])
        if isinstance(row, dict)
    }
    return edge_map, summary_map


def _copy_edge_fields(item: dict[str, Any], edge: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["countertarget_verdict"] = edge.get("verdict")
    for key in EDGE_FIELDS:
        if key in edge:
            out[key] = edge[key]
    temporal = edge.get("temporal_opponent")
    if isinstance(temporal, dict):
        for key in TEMPORAL_FIELDS:
            if key in temporal:
                out[key] = temporal[key]
        out["temporal_opponent"] = temporal
    for key in ("source_video_counts", "target_video_counts"):
        if key in edge:
            out[key] = edge[key]
    return out


def _row_stats(preview: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    numeric_keys = [
        "combined_opponent_risk_score",
        "visual_opponent_risk_score",
        "temporal_opponent_risk_score",
        "max_same_video_overlap_frames",
        "view_weak_margin_fraction",
        "view_non_rank1_fraction",
        "weighted_margin",
        "rank_vote",
        "margin_vote",
        "same_video_pair_fraction",
        "local_pair_fraction_same_video",
        "overlap_pair_fraction_same_video",
        "source_target_video_jaccard",
    ]
    for key in numeric_keys:
        vals = [_as_float(item.get(key)) for item in preview]
        vals = [float(value) for value in vals if value is not None]
        if vals:
            out[f"opponent_mean_{key}"] = float(np.mean(vals))
            out[f"opponent_max_{key}"] = float(np.max(vals))
            out[f"opponent_min_{key}"] = float(np.min(vals))
    verdicts = [str(item.get("countertarget_verdict") or item.get("verdict") or "") for item in preview]
    out["opponent_edge_count"] = int(len(preview))
    out["opponent_accept_count"] = int(sum(verdict == "accept" for verdict in verdicts))
    out["opponent_temporal_reject_count"] = int(sum(verdict == "reject_temporal_overlap" for verdict in verdicts))
    out["opponent_countertarget_reject_count"] = int(sum(verdict == "reject_countertarget" for verdict in verdicts))
    out["opponent_all_accept"] = bool(preview and out["opponent_accept_count"] == len(preview))
    return out


def attach(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = Path(args.candidate_json)
    doc = json.loads(candidate_path.read_text())
    pool, rows = _candidate_rows(doc)
    edge_map, summary_map = _load_audit(Path(args.audit_json))
    out_rows: list[dict[str, Any]] = []
    attached_edges = 0
    missing_edges = 0
    for candidate_index, row in enumerate(rows):
        preview = row.get("accepted_preview")
        if not isinstance(preview, list):
            continue
        new_preview: list[dict[str, Any]] = []
        for edge_index, item in enumerate(preview):
            if not isinstance(item, dict):
                continue
            edge = edge_map.get((pool, candidate_index, edge_index))
            if edge is None:
                missing_edges += 1
                new_preview.append(dict(item))
                continue
            new_preview.append(_copy_edge_fields(item, edge))
            attached_edges += 1
        new_row = dict(row)
        new_row["accepted_preview"] = new_preview
        summary = summary_map.get((pool, candidate_index))
        if isinstance(summary, dict):
            for key in SUMMARY_FIELDS:
                if key in summary:
                    new_row[f"opponent_summary_{key}"] = summary[key]
        new_row.update(_row_stats(new_preview))
        new_row["opponent_audit_json"] = str(args.audit_json)
        out_rows.append(new_row)
    result = {
        "candidate_json": str(candidate_path),
        "candidate_pool": pool,
        "audit_json": str(args.audit_json),
        "input_rows": int(len(rows)),
        "rows": out_rows,
        "attached_edges": int(attached_edges),
        "missing_edges": int(missing_edges),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(Path(args.csv), out_rows)
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "source_component_label",
        "target_component",
        "moved_tracklets",
        "tracklet_pair_f1",
        "full_side_effect_proxy",
        "opponent_edge_count",
        "opponent_accept_count",
        "opponent_temporal_reject_count",
        "opponent_countertarget_reject_count",
        "opponent_max_combined_opponent_risk_score",
        "opponent_max_temporal_opponent_risk_score",
        "opponent_max_max_same_video_overlap_frames",
        "opponent_mean_view_weak_margin_fraction",
        "opponent_summary_verdict",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Candidate Opponent Feature Attachment",
        "",
        f"- candidate json: `{result['candidate_json']}`",
        f"- audit json: `{result['audit_json']}`",
        f"- input rows: `{result['input_rows']}`",
        f"- output rows: `{len(result['rows'])}`",
        f"- attached edges: `{result['attached_edges']}`",
        f"- missing edges: `{result['missing_edges']}`",
        "",
        "| rank | source | target | verdict | max risk | temporal reject | overlap | weak margin |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(result["rows"][:40], start=1):
        lines.append(
            f"| {rank} | `{row.get('source_component_label', '')}` | `{row.get('target_component', '')}` | "
            f"`{row.get('opponent_summary_verdict', '')}` | "
            f"`{float(row.get('opponent_max_combined_opponent_risk_score') or 0.0):.6f}` | "
            f"`{row.get('opponent_temporal_reject_count', 0)}` | "
            f"`{float(row.get('opponent_max_max_same_video_overlap_frames') or 0.0):.1f}` | "
            f"`{float(row.get('opponent_mean_view_weak_margin_fraction') or 0.0):.6f}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cand = root / "cand.json"
        audit = root / "audit.json"
        cand.write_text(json.dumps({"top": [{"accepted_preview": [{"source_component_label": 1, "target_component": 2}]}]}))
        audit.write_text(
            json.dumps(
                {
                    "edge_rows": [
                        {
                            "candidate_pool": "top",
                            "candidate_index": 0,
                            "edge_index": 0,
                            "verdict": "reject_temporal_overlap",
                            "combined_opponent_risk_score": 0.8,
                            "view_weak_margin_fraction": 0.5,
                            "temporal_opponent": {
                                "max_same_video_overlap_frames": 12,
                                "temporal_opponent_risk_score": 0.7,
                            },
                        }
                    ],
                    "candidate_summaries": [
                        {
                            "candidate_pool": "top",
                            "candidate_index": 0,
                            "accepted_edges": 0,
                            "temporal_rejected_edges": 1,
                            "countertarget_rejected_edges": 0,
                            "verdict": "reject_temporal_overlap",
                        }
                    ],
                }
            )
        )
        out = root / "out.json"
        result = attach(argparse.Namespace(candidate_json=str(cand), audit_json=str(audit), json=str(out), csv="", md=""))
        assert result["attached_edges"] == 1, result
        row = result["rows"][0]
        item = row["accepted_preview"][0]
        assert item["countertarget_verdict"] == "reject_temporal_overlap", item
        assert item["max_same_video_overlap_frames"] == 12, item
        assert row["opponent_temporal_reject_count"] == 1, row
    print(json.dumps({"stage": "self_test", "status": "ok"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidate-json", default="")
    ap.add_argument("--audit-json", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    missing = [name for name in ("candidate_json", "audit_json", "json") if not getattr(args, name)]
    if missing:
        ap.error("missing required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    result = attach(args)
    print(json.dumps({"json": args.json, "rows": len(result["rows"]), "attached_edges": result["attached_edges"], "missing_edges": result["missing_edges"]}, sort_keys=True))


if __name__ == "__main__":
    main()
