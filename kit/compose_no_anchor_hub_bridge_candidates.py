#!/usr/bin/env python
"""Compose larger no-anchor hub-bridge candidates from accepted previews.

Existing conflict reassignment rows are often tiny source islands.  This script
is a structural proposer: it groups no-anchor accepted_preview edits that point
to the same target component, greedily packs non-overlapping source islands, and
emits composite rows that the full-score scheduler/exporter can execute.

No GT or anchors are used.  Full-score labels, when present in source artifacts,
are deliberately ignored for ranking and are not copied into output rows.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np


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


def _json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("top", "rows", "results", "top_pair_rows", "top_full_rows", "full_rows"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def _source_size(item: dict[str, Any]) -> int:
    val = _as_float(item.get("source_size"))
    if val is not None:
        return int(round(val))
    seqs = item.get("source_seqs")
    return len(seqs) if isinstance(seqs, list) else 0


def _target_size(item: dict[str, Any]) -> int:
    val = _as_float(item.get("target_size"))
    if val is not None:
        return int(round(val))
    seqs = item.get("target_top_seqs")
    return len(seqs) if isinstance(seqs, list) else 0


def _source_key(item: dict[str, Any]) -> tuple[int, ...]:
    seqs = item.get("source_seqs")
    if isinstance(seqs, list) and seqs:
        return tuple(sorted(int(float(seq)) for seq in seqs))
    label = item.get("source_component_label", item.get("source", item.get("source_rep", "")))
    return (hash(str(label)) % 1_000_000_007,)


def _item_score(item: dict[str, Any]) -> float:
    source = max(_source_size(item), 1)
    target = max(_target_size(item), 1)
    moved = min(math.log1p(source) / math.log(64.0), 1.0)
    hub = min(math.log1p(target) / math.log(256.0), 1.0)
    return float(
        0.22 * moved
        + 0.18 * hub
        + 0.20 * (_as_float(item.get("target_mean_sim"), 0.0) or 0.0)
        + 0.13 * (_as_float(item.get("target_best_sim"), 0.0) or 0.0)
        + 0.12 * (_as_float(item.get("target_view_vote"), 0.0) or 0.0)
        + 0.07 * min(_as_float(item.get("target_margin"), 0.0) or 0.0, 1.0)
        + 0.05 * (_as_float(item.get("source_quality"), 0.0) or 0.0)
        + 0.03 * (_as_float(item.get("source_score"), 0.0) or 0.0)
        - 0.05 * max(0.0, 0.60 - (_as_float(item.get("target_min_view_sim"), 0.60) or 0.60))
    )


def _iter_preview_items(patterns: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for pattern in patterns:
        for match in sorted(glob.glob(pattern, recursive=True)):
            path = Path(match)
            if not path.is_file() or path.suffix.lower() != ".json":
                continue
            for rank, row in enumerate(_json_rows(path), start=1):
                if row.get("uses_anchors") is True or row.get("uses_gt_for_training_or_anchors") is True:
                    continue
                mode = str(row.get("mode") or "")
                if args.mode_contains and args.mode_contains not in mode:
                    continue
                pair_f1 = _as_float(row.get("tracklet_pair_f1"), _as_float(row.get("pair_f1"), 0.0)) or 0.0
                pair_p = _as_float(row.get("tracklet_pair_precision"), _as_float(row.get("pair_precision"), 0.0)) or 0.0
                pair_r = _as_float(row.get("tracklet_pair_recall"), _as_float(row.get("pair_recall"), 0.0)) or 0.0
                if pair_f1 < float(args.min_source_pair_f1) or pair_p < float(args.min_source_pair_precision) or pair_r < float(args.min_source_pair_recall):
                    continue
                preview = row.get("accepted_preview")
                if not isinstance(preview, list):
                    continue
                for idx, raw in enumerate(preview, start=1):
                    if not isinstance(raw, dict):
                        continue
                    target = raw.get("target_component", raw.get("target", raw.get("target_rep", "")))
                    if target in ("", None):
                        continue
                    source_size = _source_size(raw)
                    target_size = _target_size(raw)
                    if source_size < int(args.min_source_size) or target_size < int(args.min_target_size):
                        continue
                    if (_as_float(raw.get("target_mean_sim"), 0.0) or 0.0) < float(args.min_target_mean_sim):
                        continue
                    if (_as_float(raw.get("target_view_vote"), 0.0) or 0.0) < float(args.min_target_view_vote):
                        continue
                    item = {
                        **raw,
                        "_source_artifact": str(path),
                        "_source_row_rank": rank,
                        "_preview_index": idx,
                        "_parent_mode": mode,
                        "_parent_pair_f1": float(pair_f1),
                        "_parent_pair_precision": float(pair_p),
                        "_parent_pair_recall": float(pair_r),
                        "_source_key": _source_key(raw),
                        "_item_score": _item_score(raw),
                        "source_size": int(source_size),
                        "target_size": int(target_size),
                    }
                    items.append(item)
    best: dict[tuple[str, tuple[int, ...], int, int], dict[str, Any]] = {}
    for item in items:
        key = (
            str(item.get("target_component", item.get("target", item.get("target_rep", "")))),
            tuple(item["_source_key"]),
            int(item["source_size"]),
            int(item["target_size"]),
        )
        old = best.get(key)
        if old is None or (
            float(item["_item_score"]),
            float(item["_parent_pair_f1"]),
            -int(item["_source_row_rank"]),
        ) > (
            float(old["_item_score"]),
            float(old["_parent_pair_f1"]),
            -int(old["_source_row_rank"]),
        ):
            best[key] = item
    return list(best.values())


def _mean(items: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [_as_float(item.get(key)) for item in items]
    vals = [val for val in vals if val is not None]
    return float(np.mean(vals)) if vals else float(default)


def _min(items: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [_as_float(item.get(key)) for item in items]
    vals = [val for val in vals if val is not None]
    return float(min(vals)) if vals else float(default)


def _compose_for_target(target: Any, candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = sorted(candidates, key=lambda item: (float(item["_item_score"]), int(item["source_size"])), reverse=True)
    out: list[dict[str, Any]] = []
    for start in range(min(len(candidates), int(args.max_start_items))):
        selected: list[dict[str, Any]] = []
        used_sources: set[tuple[int, ...]] = set()
        used_seqs: set[int] = set()
        for item in candidates[start:]:
            key = tuple(item["_source_key"])
            seqs = set(key)
            if key in used_sources or (len(seqs) > 1 and seqs & used_seqs):
                continue
            selected.append(item)
            used_sources.add(key)
            used_seqs.update(seqs)
            moved = sum(int(x["source_size"]) for x in selected)
            if len(selected) >= int(args.max_reassignments) or moved >= int(args.max_moved_tracklets):
                break
        moved = sum(int(x["source_size"]) for x in selected)
        if len(selected) < int(args.min_reassignments) or moved < int(args.min_moved_tracklets):
            continue
        preview = [
            {
                key: value
                for key, value in item.items()
                if not str(key).startswith("_")
            }
            for item in selected
        ]
        pair_mass = sum(max(int(item["source_size"]), 1) * max(int(item["target_size"]), 1) for item in selected)
        bridge_mass = sum(math.sqrt(max(int(item["source_size"]), 1) * max(int(item["target_size"]), 1)) for item in selected)
        target_size = max(int(item["target_size"]) for item in selected)
        source_labels = [str(item.get("source_component_label", item.get("source", item.get("source_rep", "")))) for item in selected]
        parent_f1 = _mean(selected, "_parent_pair_f1")
        parent_p = _mean(selected, "_parent_pair_precision")
        parent_r = _mean(selected, "_parent_pair_recall")
        confidence = float(
            0.22 * min(math.log1p(moved) / math.log(96.0), 1.0)
            + 0.22 * min(math.log1p(target_size) / math.log(256.0), 1.0)
            + 0.20 * _mean(selected, "target_mean_sim")
            + 0.12 * _mean(selected, "target_best_sim")
            + 0.12 * _mean(selected, "target_view_vote")
            + 0.07 * min(_mean(selected, "target_margin"), 1.0)
            + 0.05 * _mean(selected, "_item_score")
        )
        out.append(
            {
                "mode": "hub_component_bridge_composite",
                "source_component_label": "+".join(source_labels),
                "target_component": target,
                "accepted_preview": preview,
                "accepted_reassignments": int(len(selected)),
                "moved_tracklets": int(moved),
                "target_components_used": 1,
                "target_size": int(target_size),
                "tracklet_pair_f1": float(parent_f1),
                "tracklet_pair_precision": float(parent_p),
                "tracklet_pair_recall": float(parent_r),
                "full_side_effect_proxy": float(confidence),
                "accepted_edges": int(len(selected)),
                "accepted_score_mean": _mean(selected, "_item_score"),
                "accepted_view_mean_sim_mean": _mean(selected, "target_mean_sim"),
                "accepted_view_min_sim_mean": _mean(selected, "target_min_view_sim"),
                "accepted_mass_proxy_sum": float(bridge_mass),
                "accepted_pair_mass_proxy_sum": float(pair_mass),
                "accepted_min_weight_sum": float(sum(max(int(item["source_size"]), 1) for item in selected)),
                "accepted_max_weight_sum": float(sum(max(int(item["target_size"]), 1) for item in selected)),
                "accepted_size_product_sum": float(pair_mass),
                "preview_mean_target_mean_sim": _mean(selected, "target_mean_sim"),
                "preview_mean_target_best_sim": _mean(selected, "target_best_sim"),
                "preview_mean_target_view_vote": _mean(selected, "target_view_vote"),
                "preview_min_target_min_view_sim": _min(selected, "target_min_view_sim"),
                "preview_mean_source_quality": _mean(selected, "source_quality"),
                "preview_mean_source_score": _mean(selected, "source_score"),
                "preview_mean_source_size": _mean(selected, "source_size"),
                "source_artifacts": sorted({str(item["_source_artifact"]) for item in selected}),
                "signature": repr(("hub_component_bridge_composite", tuple((tuple(item.get("source_seqs", [])), target) for item in selected))),
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
                "uses_gt_for_evaluation_only": False,
            }
        )
    return out


def compose(args: argparse.Namespace) -> dict[str, Any]:
    items = _iter_preview_items(args.input_glob, args)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = str(item.get("target_component", item.get("target", item.get("target_rep", ""))))
        grouped.setdefault(key, []).append(item)
    rows: list[dict[str, Any]] = []
    for target, candidates in grouped.items():
        if len(candidates) < int(args.min_reassignments):
            continue
        rows.extend(_compose_for_target(target, candidates, args))
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["signature"])
        old = deduped.get(key)
        if old is None or (
            float(row["accepted_pair_mass_proxy_sum"]),
            float(row["full_side_effect_proxy"]),
            float(row["tracklet_pair_f1"]),
        ) > (
            float(old["accepted_pair_mass_proxy_sum"]),
            float(old["full_side_effect_proxy"]),
            float(old["tracklet_pair_f1"]),
        ):
            deduped[key] = row
    rows = list(deduped.values())
    rows.sort(
        key=lambda row: (
            float(row["accepted_pair_mass_proxy_sum"]),
            float(row["full_side_effect_proxy"]),
            float(row["tracklet_pair_f1"]),
        ),
        reverse=True,
    )
    rows = rows[: int(args.top_n)]
    out = {
        "input_glob": args.input_glob,
        "preview_items": int(len(items)),
        "target_components": int(len(grouped)),
        "rows": rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), rows)
    if args.md:
        _write_md(Path(args.md), out)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "source_component_label",
        "target_component",
        "accepted_reassignments",
        "moved_tracklets",
        "target_size",
        "tracklet_pair_f1",
        "tracklet_pair_precision",
        "tracklet_pair_recall",
        "full_side_effect_proxy",
        "accepted_pair_mass_proxy_sum",
        "accepted_mass_proxy_sum",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Hub-Bridge Composite Candidates",
        "",
        f"- preview items: `{out['preview_items']}`",
        f"- target components: `{out['target_components']}`",
        f"- composite rows: `{len(out['rows'])}`",
        "",
        "| rank | moved | edits | target | pair-mass proxy | pair F1 | score | sources |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(out["rows"][:30], start=1):
        lines.append(
            f"| {rank} | `{row['moved_tracklets']}` | `{row['accepted_reassignments']}` | "
            f"`{row['target_component']}` | `{row['accepted_pair_mass_proxy_sum']:.3f}` | "
            f"`{row['tracklet_pair_f1']:.6f}` | `{row['full_side_effect_proxy']:.6f}` | "
            f"`{str(row['source_component_label'])[:80]}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This proposer uses only existing no-anchor accepted-preview evidence.",
            "- It intentionally does not copy full-score labels from source artifacts.",
            "- Rows are executable by the scheduler/exporter path because each composite row keeps a full accepted_preview list.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        artifact = root / "a.json"
        artifact.write_text(
            json.dumps(
                {
                    "top": [
                        {
                            "mode": "conflict_subcluster_reassign_candidate_search",
                            "tracklet_pair_f1": 0.76,
                            "tracklet_pair_precision": 0.81,
                            "tracklet_pair_recall": 0.72,
                            "accepted_preview": [
                                {
                                    "source_component_label": 1,
                                    "target_component": 9,
                                    "source_seqs": [1, 2],
                                    "source_size": 2,
                                    "target_size": 100,
                                    "target_mean_sim": 0.82,
                                    "target_best_sim": 0.9,
                                    "target_view_vote": 0.75,
                                },
                                {
                                    "source_component_label": 2,
                                    "target_component": 9,
                                    "source_seqs": [3, 4, 5],
                                    "source_size": 3,
                                    "target_size": 100,
                                    "target_mean_sim": 0.83,
                                    "target_best_sim": 0.91,
                                    "target_view_vote": 0.75,
                                },
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = compose(
            argparse.Namespace(
                input_glob=[str(root / "*.json")],
                mode_contains="conflict",
                min_source_pair_f1=0.7,
                min_source_pair_precision=0.7,
                min_source_pair_recall=0.7,
                min_source_size=1,
                min_target_size=8,
                min_target_mean_sim=0.7,
                min_target_view_vote=0.5,
                min_reassignments=2,
                max_reassignments=4,
                min_moved_tracklets=4,
                max_moved_tracklets=32,
                max_start_items=4,
                top_n=10,
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
                md=str(root / "out.md"),
            )
        )
        assert out["rows"], out
        row = out["rows"][0]
        assert row["moved_tracklets"] == 5
        assert len(row["accepted_preview"]) == 2
        assert Path(root / "out.csv").read_text()
        assert "hub-bridge" in Path(root / "out.md").read_text().lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--input-glob", action="append", default=["local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign*_pair_20260620.json"])
    ap.add_argument("--mode-contains", default="conflict")
    ap.add_argument("--min-source-pair-f1", type=float, default=0.74)
    ap.add_argument("--min-source-pair-precision", type=float, default=0.78)
    ap.add_argument("--min-source-pair-recall", type=float, default=0.70)
    ap.add_argument("--min-source-size", type=int, default=4)
    ap.add_argument("--min-target-size", type=int, default=32)
    ap.add_argument("--min-target-mean-sim", type=float, default=0.74)
    ap.add_argument("--min-target-view-vote", type=float, default=0.50)
    ap.add_argument("--min-reassignments", type=int, default=2)
    ap.add_argument("--max-reassignments", type=int, default=8)
    ap.add_argument("--min-moved-tracklets", type=int, default=16)
    ap.add_argument("--max-moved-tracklets", type=int, default=96)
    ap.add_argument("--max-start-items", type=int, default=16)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--json", default="local_runs/no_anchor_hub_bridge_composite_candidates_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_hub_bridge_composite_candidates_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_hub_bridge_composite_candidates_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    out = compose(args)
    print(json.dumps({"preview_items": out["preview_items"], "target_components": out["target_components"], "rows": len(out["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
