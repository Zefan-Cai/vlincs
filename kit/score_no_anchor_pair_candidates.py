#!/usr/bin/env python
"""Rank existing no-anchor pair-only proposal rows with a learned full proxy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.analyze_no_anchor_full_proxy_training import _as_float, _iter_rows, _row_features  # noqa: E402


def _score(features: dict[str, float], model: dict[str, Any]) -> float:
    score = float(model.get("intercept", 0.0))
    fills = model.get("fill_values", {})
    for idx, key in enumerate(model.get("columns", [])):
        if idx >= len(model.get("coef", [])):
            break
        value = features.get(key)
        if value is None:
            value = _as_float(fills.get(key) if isinstance(fills, dict) else None)
        if value is None:
            value = 0.0
        mean = float(model.get("mean", [0.0] * (idx + 1))[idx])
        scale = float(model.get("scale", [1.0] * (idx + 1))[idx])
        if abs(scale) < 1.0e-9:
            scale = 1.0
        score += float(model.get("coef", [0.0] * (idx + 1))[idx]) * ((float(value) - mean) / scale)
    return float(score)


def _delivery_value(row: dict[str, Any], key: str) -> float | None:
    value = _as_float(row.get(key))
    return value


def _delivery_summary(row: dict[str, Any]) -> dict[str, float | None]:
    vals = {
        "assigned_tracklets": _delivery_value(row, "assigned_tracklets"),
        "output_tracklets": _delivery_value(row, "output_tracklets"),
        "eval_tracklets": _delivery_value(row, "eval_tracklets"),
        "coverage_ratio": _delivery_value(row, "coverage_ratio"),
    }
    delivery_counts = [
        vals[key]
        for key in ("assigned_tracklets", "output_tracklets", "eval_tracklets")
        if vals[key] is not None
    ]
    vals["delivery_tracklets_min"] = min(delivery_counts) if delivery_counts else None
    return vals


def _passes_delivery_gate(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.disable_delivery_filter:
        return True
    vals = _delivery_summary(row)
    output = vals["output_tracklets"]
    eval_count = vals["eval_tracklets"]
    coverage = vals["coverage_ratio"]
    if output is not None and output < float(args.min_output_tracklets):
        return False
    if eval_count is not None and eval_count < float(args.min_eval_tracklets):
        return False
    if coverage is not None and coverage < float(args.min_coverage_ratio):
        return False
    return True


def _signature(row: dict[str, Any]) -> tuple[Any, ...]:
    preview = row.get("accepted_preview")
    if isinstance(preview, list) and preview:
        parts = []
        for item in preview:
            if not isinstance(item, dict):
                continue
            parts.append(
                (
                    tuple(int(seq) for seq in item.get("source_seqs", [])),
                    int(float(item.get("target_component", -1))),
                )
            )
        if parts:
            return ("accepted_preview", tuple(sorted(parts)))
    keys = (
        "mode",
        "tracklet_pair_f1",
        "tracklet_pair_precision",
        "tracklet_pair_recall",
        "candidate_search_prefix",
        "max_sources_per_target",
        "max_reassignments",
        "source_component_label",
        "target_component",
        "source_seed_sim",
        "source_expand_sim",
        "source_top_k",
        "source_min_group_size",
        "source_max_group_size",
        "min_target_size",
        "target_top_k",
        "min_target_best_sim",
        "min_target_mean_sim",
        "min_target_view_vote",
    )
    return tuple((key, row.get(key)) for key in keys if key in row)


def _collect(args: argparse.Namespace, model: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    stats = {
        "seen_rows": 0,
        "dropped_delivery_filter": 0,
        "dropped_pair_filter": 0,
        "dropped_no_features": 0,
        "dropped_duplicate": 0,
    }
    for path in sorted(Path(args.local_runs).rglob("*.json")):
        if "full_proxy" in path.name and "training_audit" in path.name:
            continue
        if "pair_candidates" in path.name:
            continue
        if "oracle" in path.name and not args.include_oracle_artifacts:
            continue
        for row in _iter_rows(path):
            if not isinstance(row, dict):
                continue
            stats["seen_rows"] += 1
            if row.get("uses_gt_for_analysis_only") is True and not args.include_oracle_artifacts:
                continue
            pair_f1 = _as_float(row.get("tracklet_pair_f1") or row.get("pair_f1"))
            if pair_f1 is None or pair_f1 < float(args.min_pair_f1):
                stats["dropped_pair_filter"] += 1
                continue
            if not _passes_delivery_gate(row, args):
                stats["dropped_delivery_filter"] += 1
                continue
            features = _row_features(row)
            if not features:
                stats["dropped_no_features"] += 1
                continue
            sig = _signature(row)
            if args.dedup and sig in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(sig)
            pred = _score(features, model)
            full = _as_float(row.get("full_idf1"))
            delivery = _delivery_summary(row)
            rows.append(
                {
                    "artifact": str(path),
                    "mode": row.get("mode") or row.get("name") or "",
                    "learned_proxy_full_idf1": pred,
                    "known_full_idf1": full,
                    "is_full_scored": full is not None,
                    "pair_f1": pair_f1,
                    "pair_precision": _as_float(row.get("tracklet_pair_precision")),
                    "pair_recall": _as_float(row.get("tracklet_pair_recall")),
                    "full_side_effect_proxy": _as_float(row.get("full_side_effect_proxy")),
                    "accepted_reassignments": _as_float(row.get("accepted_reassignments")),
                    "moved_tracklets": _as_float(row.get("moved_tracklets")),
                    "candidate_search_prefix": _as_float(row.get("candidate_search_prefix")),
                    "candidate_skip_first_edge_families": _as_float(row.get("candidate_skip_first_edge_families")),
                    "candidate_first_edge_family_rank": _as_float(row.get("candidate_first_edge_family_rank")),
                    "max_sources_per_target": _as_float(row.get("max_sources_per_target")),
                    "max_reassignments": _as_float(row.get("max_reassignments")),
                    "source_component_label": row.get("source_component_label"),
                    "target_component": row.get("target_component"),
                    "source_size": _as_float(row.get("source_size")),
                    "target_size": _as_float(row.get("target_size")),
                    "assigned_tracklets": delivery["assigned_tracklets"],
                    "output_tracklets": delivery["output_tracklets"],
                    "eval_tracklets": delivery["eval_tracklets"],
                    "coverage_ratio": delivery["coverage_ratio"],
                    "delivery_tracklets_min": delivery["delivery_tracklets_min"],
                    "signature": repr(sig),
                }
            )
    rows.sort(
        key=lambda item: (
            float(item["learned_proxy_full_idf1"]),
            float(item["pair_f1"]),
            float(item["pair_recall"] or 0.0),
        ),
        reverse=True,
    )
    return rows, stats


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "learned_proxy_full_idf1",
        "known_full_idf1",
        "is_full_scored",
        "pair_f1",
        "pair_precision",
        "pair_recall",
        "full_side_effect_proxy",
        "accepted_reassignments",
        "moved_tracklets",
        "candidate_search_prefix",
        "candidate_skip_first_edge_families",
        "candidate_first_edge_family_rank",
        "max_sources_per_target",
        "max_reassignments",
        "source_component_label",
        "target_component",
        "source_size",
        "target_size",
        "assigned_tracklets",
        "output_tracklets",
        "eval_tracklets",
        "coverage_ratio",
        "delivery_tracklets_min",
        "mode",
        "artifact",
        "signature",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **{key: row.get(key, "") for key in fields if key != "rank"}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-runs", default="local_runs")
    parser.add_argument("--model-json", default="local_runs/no_anchor_full_proxy_compact_ridge_model_20260620.json")
    parser.add_argument("--min-pair-f1", type=float, default=0.74)
    parser.add_argument("--include-full-scored", action="store_true")
    parser.add_argument("--include-oracle-artifacts", action="store_true")
    parser.add_argument("--min-output-tracklets", type=float, default=7000.0)
    parser.add_argument("--min-eval-tracklets", type=float, default=7000.0)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.70)
    parser.add_argument("--disable-delivery-filter", action="store_true")
    parser.add_argument("--dedup", action="store_true", default=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", default="")
    parser.add_argument("--md", default="")
    args = parser.parse_args()

    model = json.loads(Path(args.model_json).read_text())
    all_rows, collect_stats = _collect(args, model)
    rows = all_rows if args.include_full_scored else [row for row in all_rows if not row["is_full_scored"]]
    top = rows[: int(args.top_n)]
    result = {
        "model_json": str(args.model_json),
        "model_summary": {key: model.get(key) for key in ("model_type", "row_count", "alpha", "feature_mode", "min_full_idf1")},
        "min_pair_f1": float(args.min_pair_f1),
        "include_full_scored": bool(args.include_full_scored),
        "include_oracle_artifacts": bool(args.include_oracle_artifacts),
        "delivery_filter": {
            "enabled": not bool(args.disable_delivery_filter),
            "min_output_tracklets": float(args.min_output_tracklets),
            "min_eval_tracklets": float(args.min_eval_tracklets),
            "min_coverage_ratio": float(args.min_coverage_ratio),
            "only_applies_to_present_fields": True,
        },
        "collect_stats": collect_stats,
        "candidate_count": len(rows),
        "all_candidate_count": len(all_rows),
        "top": top,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True))
    _write_csv(args.csv, top)
    if args.md:
        lines = [
            "# Learned-Proxy Pair Candidate Ranking",
            "",
            f"- candidates: `{result['candidate_count']}`",
            f"- all candidates before full-score filter: `{result['all_candidate_count']}`",
            f"- min pair F1: `{result['min_pair_f1']}`",
            f"- include full-scored: `{result['include_full_scored']}`",
            f"- delivery filter: `{result['delivery_filter']}`",
            f"- dropped by delivery filter: `{collect_stats['dropped_delivery_filter']}`",
            "",
            "| rank | learned full IDF1 | pair F1 | known full | output | eval | coverage | mode | artifact |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for rank, row in enumerate(top[: int(args.top_n)], start=1):
            known = "" if row["known_full_idf1"] is None else f"{row['known_full_idf1']:.6f}"
            output = "" if row["output_tracklets"] is None else f"{row['output_tracklets']:.0f}"
            eval_count = "" if row["eval_tracklets"] is None else f"{row['eval_tracklets']:.0f}"
            coverage = "" if row["coverage_ratio"] is None else f"{row['coverage_ratio']:.6f}"
            lines.append(
                f"| `{rank}` | `{row['learned_proxy_full_idf1']:.6f}` | `{row['pair_f1']:.6f}` | "
                f"`{known}` | `{output}` | `{eval_count}` | `{coverage}` | `{row['mode']}` | `{row['artifact']}` |"
            )
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "dropped_delivery_filter": collect_stats["dropped_delivery_filter"],
                "top_pred": top[0]["learned_proxy_full_idf1"] if top else None,
                "top_pair_f1": top[0]["pair_f1"] if top else None,
                "top_artifact": top[0]["artifact"] if top else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
