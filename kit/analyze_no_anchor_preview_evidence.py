#!/usr/bin/env python
"""Audit accepted_preview evidence across no-anchor experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rows_from_json(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("top", "rows", "results", "top_pair_rows", "top_full_rows", "full_rows"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def _iter_artifact_rows(patterns: list[str]) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for pattern in patterns:
        for match in sorted(glob.glob(pattern, recursive=True)):
            path = Path(match)
            if not path.is_file() or path.suffix.lower() != ".json":
                continue
            for row in _rows_from_json(path):
                preview = row.get("accepted_preview")
                if isinstance(preview, list) and preview:
                    out.append((path, row))
    return out


def _preview_size(item: dict[str, Any], key: str, fallback_key: str = "") -> float | None:
    val = _as_float(item.get(key))
    if val is not None:
        return val
    if fallback_key:
        seqs = item.get(fallback_key)
        if isinstance(seqs, list):
            return float(len(seqs))
    return None


def _flatten_preview(path: Path, row: dict[str, Any], idx: int, item: dict[str, Any], current_best: float) -> dict[str, Any]:
    source_size = _preview_size(item, "source_size", "source_seqs")
    target_size = _preview_size(item, "target_size", "target_top_seqs")
    if source_size is None:
        source_size = _as_float(row.get("source_size"))
    if target_size is None:
        target_size = _as_float(row.get("target_size"))
    source_weight = _as_float(item.get("source_weight")) or max(float(source_size or 1.0), 1.0)
    target_weight = _as_float(item.get("target_weight")) or max(float(target_size or 1.0), 1.0)
    pair_mass_proxy = _as_float(item.get("pair_mass_proxy"))
    if pair_mass_proxy is None and source_size is not None and target_size is not None:
        pair_mass_proxy = float(max(source_size, 1.0) * max(target_size, 1.0))
    bridge_mass_proxy = _as_float(item.get("bridge_mass_proxy"))
    if bridge_mass_proxy is None:
        bridge_mass_proxy = float(np.sqrt(max(source_weight, 1.0) * max(target_weight, 1.0)))
    full_idf1 = _as_float(row.get("full_idf1"))
    known_full = _as_float(row.get("known_full_idf1"))
    full = full_idf1 if full_idf1 is not None else known_full
    return {
        "artifact": str(path),
        "mode": str(row.get("mode") or row.get("name") or ""),
        "preview_index": int(idx),
        "full_idf1": full,
        "above_current_best": (full is not None and full > float(current_best)),
        "pair_f1": _as_float(row.get("tracklet_pair_f1")) or _as_float(row.get("pair_f1")),
        "pair_precision": _as_float(row.get("tracklet_pair_precision")) or _as_float(row.get("pair_precision")),
        "pair_recall": _as_float(row.get("tracklet_pair_recall")) or _as_float(row.get("pair_recall")),
        "accepted_count": len(row.get("accepted_preview", [])) if isinstance(row.get("accepted_preview"), list) else None,
        "moved_tracklets": _as_float(row.get("moved_tracklets")),
        "source_component_label": item.get("source_component_label", item.get("source", item.get("source_rep", ""))),
        "target_component": item.get("target_component", item.get("target", item.get("target_rep", ""))),
        "source_size": source_size,
        "target_size": target_size,
        "size_ratio_small_over_large": (
            float(min(source_size, target_size) / max(source_size, target_size))
            if source_size and target_size and max(source_size, target_size) > 0
            else None
        ),
        "target_mean_sim": _as_float(item.get("target_mean_sim")),
        "target_best_sim": _as_float(item.get("target_best_sim")),
        "target_view_vote": _as_float(item.get("target_view_vote")),
        "target_min_view_sim": _as_float(item.get("target_min_view_sim")),
        "target_margin": _as_float(item.get("target_margin")),
        "target_score": _as_float(item.get("target_score")) or _as_float(item.get("score")),
        "source_quality": _as_float(item.get("source_quality")),
        "source_score": _as_float(item.get("source_score")),
        "source_margin_mean": _as_float(item.get("source_margin_mean")),
        "source_internal_sim": _as_float(item.get("source_internal_sim")),
        "source_cross_mean_sim": _as_float(item.get("source_cross_mean_sim")),
        "bridge_mass_proxy": bridge_mass_proxy,
        "pair_mass_proxy": pair_mass_proxy,
        "uses_anchors": bool(row.get("uses_anchors", False)),
        "uses_gt_for_training_or_anchors": bool(row.get("uses_gt_for_training_or_anchors", False)),
    }


def _summaries(flat: list[dict[str, Any]], current_best: float) -> dict[str, Any]:
    mode_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flat:
        mode_groups[str(row["mode"])].append(row)
    by_mode = []
    for mode, rows in sorted(mode_groups.items()):
        labelled = [row for row in rows if row["full_idf1"] is not None]
        vals = [float(row["full_idf1"]) for row in labelled]
        masses = [float(row["pair_mass_proxy"]) for row in rows if row["pair_mass_proxy"] is not None]
        by_mode.append(
            {
                "mode": mode,
                "preview_rows": len(rows),
                "labelled_preview_rows": len(labelled),
                "max_full_idf1": max(vals) if vals else None,
                "mean_full_idf1": float(np.mean(vals)) if vals else None,
                "rows_above_current_best": int(sum(val > float(current_best) for val in vals)),
                "max_pair_mass_proxy": max(masses) if masses else None,
                "mean_pair_mass_proxy": float(np.mean(masses)) if masses else None,
            }
        )
    labelled_rows = [row for row in flat if row["full_idf1"] is not None]
    def unique_edit(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        seen: set[tuple[Any, ...]] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            key = (
                row.get("mode"),
                row.get("source_component_label"),
                row.get("target_component"),
                row.get("source_size"),
                row.get("target_size"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= limit:
                break
        return out

    top_labelled = unique_edit(sorted(labelled_rows, key=lambda row: float(row["full_idf1"]), reverse=True), 20)
    top_mass_unlabelled = unique_edit(sorted(
        [row for row in flat if row["full_idf1"] is None],
        key=lambda row: float(row["pair_mass_proxy"] or 0.0),
        reverse=True,
    ), 50)
    return {
        "preview_rows": len(flat),
        "labelled_preview_rows": len(labelled_rows),
        "labelled_above_current_best": int(sum(float(row["full_idf1"]) > float(current_best) for row in labelled_rows)),
        "by_mode": by_mode,
        "top_labelled": top_labelled,
        "top_mass_unlabelled": top_mass_unlabelled,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_md(path: Path, summary: dict[str, Any], current_best: float) -> None:
    lines = [
        "# No-Anchor Accepted-Preview Evidence Audit",
        "",
        f"Current best full IDF1: `{current_best:.6f}`",
        "",
        f"Preview rows: `{summary['preview_rows']}`",
        f"Full-labelled preview rows: `{summary['labelled_preview_rows']}`",
        f"Labelled rows above current best: `{summary['labelled_above_current_best']}`",
        "",
        "## By Mode",
        "",
        "| mode | preview rows | labelled | max full | above best | max pair-mass proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["by_mode"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["mode"]),
                    _fmt(row["preview_rows"]),
                    _fmt(row["labelled_preview_rows"]),
                    _fmt(row["max_full_idf1"]),
                    _fmt(row["rows_above_current_best"]),
                    _fmt(row["max_pair_mass_proxy"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Top Full-Labelled Preview Rows", ""])
    lines.append("| full | pair F1 | mode | source -> target | source size | target size | artifact |")
    lines.append("| ---: | ---: | --- | --- | ---: | ---: | --- |")
    for row in summary["top_labelled"][:12]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["full_idf1"]),
                    _fmt(row["pair_f1"]),
                    str(row["mode"]),
                    f"`{row['source_component_label']} -> {row['target_component']}`",
                    _fmt(row["source_size"]),
                    _fmt(row["target_size"]),
                    f"`{Path(str(row['artifact'])).name}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Highest Unlabelled Pair-Mass Preview Rows", ""])
    lines.append("| pair-mass proxy | pair F1 | mode | source -> target | source size | target size | artifact |")
    lines.append("| ---: | ---: | --- | --- | ---: | ---: | --- |")
    for row in summary["top_mass_unlabelled"][:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["pair_mass_proxy"]),
                    _fmt(row["pair_f1"]),
                    str(row["mode"]),
                    f"`{row['source_component_label']} -> {row['target_component']}`",
                    _fmt(row["source_size"]),
                    _fmt(row["target_size"]),
                    f"`{Path(str(row['artifact'])).name}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Existing full-labelled preview edits do not prove a route beyond the current best if `above best` is zero.",
            "- High pair-mass unlabelled rows are candidates for future full-score slots only if their source artifacts are no-anchor and their accepted previews pass cannot-link/provenance checks.",
            "- This audit is an ablation/routing tool; it does not train or select with GT.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    pairs = _iter_artifact_rows(args.input_glob)
    flat: list[dict[str, Any]] = []
    for path, row in pairs:
        if row.get("uses_gt_for_training_or_anchors") is True or row.get("uses_anchors") is True:
            continue
        preview = row.get("accepted_preview")
        if not isinstance(preview, list):
            continue
        for idx, item in enumerate(preview, start=1):
            if isinstance(item, dict):
                flat.append(_flatten_preview(path, row, idx, item, float(args.current_best_full_idf1)))
    summary = _summaries(flat, float(args.current_best_full_idf1))
    out = {"summary": summary, "uses_anchors": False, "uses_gt_for_training_or_anchors": False, "uses_gt_for_evaluation_only": True}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), flat)
    if args.md:
        _write_md(Path(args.md), summary, float(args.current_best_full_idf1))
    return out


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        artifact = root / "case.json"
        artifact.write_text(
            json.dumps(
                {
                    "top": [
                        {
                            "mode": "conflict_subcluster_reassign",
                            "full_idf1": 0.7,
                            "tracklet_pair_f1": 0.8,
                            "accepted_preview": [
                                {"source_component_label": 1, "target_component": 2, "source_size": 3, "target_size": 10}
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = run(
            argparse.Namespace(
                input_glob=[str(root / "*.json")],
                current_best_full_idf1=0.65,
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
                md=str(root / "out.md"),
            )
        )
        assert out["summary"]["preview_rows"] == 1
        assert out["summary"]["labelled_above_current_best"] == 1
        assert Path(root / "out.csv").read_text()
        assert "conflict_subcluster_reassign" in Path(root / "out.md").read_text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-glob", action="append", default=["local_runs/**/*.json"])
    ap.add_argument("--current-best-full-idf1", type=float, default=0.655240)
    ap.add_argument("--json", default="local_runs/no_anchor_preview_evidence_audit_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_preview_evidence_audit_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_preview_evidence_audit_20260620.md")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    out = run(args)
    summary = out["summary"]
    compact = {
        "preview_rows": summary["preview_rows"],
        "labelled_preview_rows": summary["labelled_preview_rows"],
        "labelled_above_current_best": summary["labelled_above_current_best"],
        "modes": [
            {
                "mode": row["mode"],
                "preview_rows": row["preview_rows"],
                "labelled_preview_rows": row["labelled_preview_rows"],
                "max_full_idf1": row["max_full_idf1"],
                "rows_above_current_best": row["rows_above_current_best"],
            }
            for row in summary["by_mode"]
        ],
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
