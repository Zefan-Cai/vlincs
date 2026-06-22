#!/usr/bin/env python
"""Rerank no-anchor single-edge candidates for localized island attachment.

This is a narrow proposer for cases where aggregate source-target embedding
support is not the main signal. It uses only no-GT fields already present in
single-edge candidate rows and favors small, internally coherent source islands
that have one very strong match into a larger target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


def _preview_item(row: dict[str, Any]) -> dict[str, Any]:
    preview = row.get("accepted_preview")
    if isinstance(preview, list) and preview and isinstance(preview[0], dict):
        return preview[0]
    return row


def _component(item: dict[str, Any], row: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = item.get(key, row.get(key))
        if value not in (None, ""):
            return _as_int(value)
    return 0


def _score(row: dict[str, Any]) -> tuple[float, dict[str, float]]:
    item = _preview_item(row)
    target_best = _as_float(item.get("target_best_sim"), _as_float(item.get("fused_sim")))
    target_mean = _as_float(item.get("target_mean_sim"), _as_float(item.get("view_mean_sim")))
    target_view_vote = _as_float(item.get("target_view_vote"), _as_float(item.get("votes_top5")))
    if target_view_vote > 1.0:
        target_view_vote = min(target_view_vote / 5.0, 1.0)
    target_quality = _as_float(item.get("target_quality"), _as_float(row.get("target_quality")))
    target_score = _as_float(item.get("target_score"), _as_float(item.get("score")))
    source_quality = _as_float(item.get("source_quality"), _as_float(row.get("source_quality")))
    source_internal = _as_float(item.get("source_internal_sim"), _as_float(row.get("source_internal_sim")))
    source_cross_mean = _as_float(item.get("source_cross_mean_sim"), _as_float(row.get("source_cross_mean_sim"), 1.0))
    source_conflicts = _as_float(item.get("source_conflicts_to_rest"), _as_float(row.get("source_conflicts_to_rest"), 99.0))
    source_size = _as_float(item.get("source_size"), _as_float(row.get("moved_tracklets"), 1.0))
    target_size = _as_float(item.get("target_size"), _as_float(row.get("target_n_tracklets"), 1.0))
    pair_f1 = _as_float(row.get("tracklet_pair_f1"), _as_float(row.get("pair_f1"), _as_float(row.get("pair_f1_norm"))))

    localized_peak = max(target_best - target_mean, 0.0)
    low_cross = max(1.0 - source_cross_mean, 0.0)
    low_conflict = 1.0 / (1.0 + max(source_conflicts, 0.0))
    small_source = 1.0 / (1.0 + max(source_size - 1.0, 0.0) / 8.0)
    large_target = min(math.log1p(max(target_size, 1.0)) / math.log(256.0), 1.0)
    pair_bonus = max(pair_f1 - 0.765, 0.0)

    score = (
        0.30 * target_best
        + 0.18 * localized_peak
        + 0.14 * source_quality
        + 0.12 * source_internal
        + 0.12 * low_cross
        + 0.10 * low_conflict
        + 0.06 * target_quality
        + 0.05 * target_score
        + 0.04 * target_view_vote
        + 0.04 * small_source
        + 0.03 * large_target
        + 0.60 * pair_bonus
    )
    feats = {
        "localized_island_score": float(score),
        "target_best_sim": float(target_best),
        "target_mean_sim": float(target_mean),
        "localized_peak": float(localized_peak),
        "target_view_vote": float(target_view_vote),
        "target_quality": float(target_quality),
        "target_score": float(target_score),
        "source_quality": float(source_quality),
        "source_internal_sim": float(source_internal),
        "source_cross_mean_sim": float(source_cross_mean),
        "source_conflicts_to_rest": float(source_conflicts),
        "source_size": float(source_size),
        "target_size": float(target_size),
        "pair_f1": float(pair_f1),
    }
    return float(score), feats


def rerank(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(Path(args.candidates_json))
    out_rows: list[dict[str, Any]] = []
    rejected = 0
    for row in rows:
        if row.get("uses_anchors") is True or row.get("uses_gt_for_training_or_anchors") is True:
            rejected += 1
            continue
        score, feats = _score(row)
        if feats["pair_f1"] < float(args.min_pair_f1):
            rejected += 1
            continue
        if feats["target_best_sim"] < float(args.min_target_best_sim):
            rejected += 1
            continue
        if feats["source_cross_mean_sim"] > float(args.max_source_cross_mean_sim):
            rejected += 1
            continue
        if feats["source_conflicts_to_rest"] > float(args.max_source_conflicts_to_rest):
            rejected += 1
            continue
        if feats["source_internal_sim"] < float(args.min_source_internal_sim):
            rejected += 1
            continue
        if feats["source_quality"] < float(args.min_source_quality):
            rejected += 1
            continue
        if feats["target_score"] < float(args.min_target_score):
            rejected += 1
            continue
        if feats["source_size"] > float(args.max_source_size):
            rejected += 1
            continue
        if feats["target_size"] < float(args.min_target_size):
            rejected += 1
            continue
        item = _preview_item(row)
        new_row = dict(row)
        new_row.update(feats)
        new_row["localized_island_rank_score"] = float(score)
        new_row["source_component_label"] = _component(item, row, "source_component_label", "source", "source_rep")
        new_row["target_component"] = _component(item, row, "target_component", "target", "target_rep")
        new_row["uses_anchors"] = False
        new_row["uses_gt_for_training_or_anchors"] = False
        new_row["uses_gt_for_evaluation_only"] = False
        out_rows.append(new_row)
    out_rows.sort(
        key=lambda row: (
            float(row["localized_island_rank_score"]),
            float(row.get("tracklet_pair_f1", row.get("pair_f1", 0.0))),
            float(row.get("full_side_effect_proxy", 0.0)),
        ),
        reverse=True,
    )
    out_rows = out_rows[: int(args.top_n)]
    result = {
        "candidates_json": str(args.candidates_json),
        "input_rows": int(len(rows)),
        "rejected_rows": int(rejected),
        "rows": out_rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "rules": {
            "min_pair_f1": float(args.min_pair_f1),
            "min_target_best_sim": float(args.min_target_best_sim),
            "max_source_cross_mean_sim": float(args.max_source_cross_mean_sim),
            "max_source_conflicts_to_rest": float(args.max_source_conflicts_to_rest),
            "min_source_internal_sim": float(args.min_source_internal_sim),
            "min_source_quality": float(args.min_source_quality),
            "min_target_score": float(args.min_target_score),
            "max_source_size": float(args.max_source_size),
            "min_target_size": float(args.min_target_size),
        },
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), out_rows)
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_component_label",
        "target_component",
        "localized_island_rank_score",
        "target_best_sim",
        "target_mean_sim",
        "localized_peak",
        "source_cross_mean_sim",
        "source_conflicts_to_rest",
        "source_internal_sim",
        "source_quality",
        "target_score",
        "source_size",
        "target_size",
        "pair_f1",
        "full_side_effect_proxy",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Localized Island Single-Edge Rerank",
        "",
        f"- input rows: `{result['input_rows']}`",
        f"- emitted rows: `{len(result['rows'])}`",
        f"- rejected rows: `{result['rejected_rows']}`",
        "",
        "| rank | source | target | island score | target best | peak | cross mean | conflicts | pair F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(result["rows"][:40], start=1):
        lines.append(
            f"| {rank} | `{row['source_component_label']}` | `{row['target_component']}` | "
            f"`{float(row['localized_island_rank_score']):.6f}` | "
            f"`{float(row['target_best_sim']):.6f}` | `{float(row['localized_peak']):.6f}` | "
            f"`{float(row['source_cross_mean_sim']):.6f}` | `{float(row['source_conflicts_to_rest']):.0f}` | "
            f"`{float(row['pair_f1']):.6f}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "rows.json"
        src.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "tracklet_pair_f1": 0.77,
                            "full_side_effect_proxy": 0.66,
                            "accepted_preview": [
                                {
                                    "source_component_label": 9,
                                    "target_component": 7,
                                    "source_size": 8,
                                    "target_size": 216,
                                    "target_best_sim": 0.94,
                                    "target_mean_sim": 0.72,
                                    "target_view_vote": 0.66,
                                    "target_score": 0.79,
                                    "target_quality": 0.85,
                                    "source_quality": 0.88,
                                    "source_internal_sim": 0.91,
                                    "source_cross_mean_sim": 0.51,
                                    "source_conflicts_to_rest": 3,
                                }
                            ],
                        },
                        {
                            "tracklet_pair_f1": 0.77,
                            "accepted_preview": [
                                {
                                    "source_component_label": 40,
                                    "target_component": 21,
                                    "source_size": 8,
                                    "target_size": 227,
                                    "target_best_sim": 0.92,
                                    "target_mean_sim": 0.88,
                                    "target_score": 0.88,
                                    "source_quality": 0.71,
                                    "source_internal_sim": 0.84,
                                    "source_cross_mean_sim": 0.64,
                                    "source_conflicts_to_rest": 19,
                                }
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = rerank(
            argparse.Namespace(
                candidates_json=str(src),
                min_pair_f1=0.765,
                min_target_best_sim=0.90,
                max_source_cross_mean_sim=0.56,
                max_source_conflicts_to_rest=6,
                min_source_internal_sim=0.85,
                min_source_quality=0.75,
                min_target_score=0.75,
                max_source_size=16,
                min_target_size=32,
                top_n=10,
                json=str(root / "out.json"),
                csv="",
                md="",
            )
        )
        assert len(out["rows"]) == 1
        assert int(out["rows"][0]["source_component_label"]) == 9
        assert int(out["rows"][0]["target_component"]) == 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidates-json", default="")
    ap.add_argument("--min-pair-f1", type=float, default=0.765)
    ap.add_argument("--min-target-best-sim", type=float, default=0.90)
    ap.add_argument("--max-source-cross-mean-sim", type=float, default=0.56)
    ap.add_argument("--max-source-conflicts-to-rest", type=float, default=6.0)
    ap.add_argument("--min-source-internal-sim", type=float, default=0.85)
    ap.add_argument("--min-source-quality", type=float, default=0.75)
    ap.add_argument("--min-target-score", type=float, default=0.75)
    ap.add_argument("--max-source-size", type=float, default=16.0)
    ap.add_argument("--min-target-size", type=float, default=32.0)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--json", default="local_runs/no_anchor_single_edge_localized_island_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_single_edge_localized_island_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_single_edge_localized_island_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    if not args.candidates_json:
        ap.error("--candidates-json is required unless --self-test is used")
    out = rerank(args)
    print(json.dumps({"input_rows": out["input_rows"], "rows": len(out["rows"]), "rejected_rows": out["rejected_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
