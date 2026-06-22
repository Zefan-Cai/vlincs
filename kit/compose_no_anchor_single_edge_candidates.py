#!/usr/bin/env python
"""Explode no-anchor accepted_preview rows into single-edge candidates."""

from __future__ import annotations

import argparse
import csv
import glob
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


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
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


def _component(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return ""


def _source_size(item: dict[str, Any]) -> int:
    value = _as_float(item.get("source_size"))
    if value is not None:
        return int(round(value))
    seqs = item.get("source_seqs")
    return len(seqs) if isinstance(seqs, list) else 0


def _target_size(item: dict[str, Any]) -> int:
    value = _as_float(item.get("target_size"))
    if value is not None:
        return int(round(value))
    seqs = item.get("target_top_seqs")
    return len(seqs) if isinstance(seqs, list) else 0


def _item_score(item: dict[str, Any]) -> float:
    source = max(_source_size(item), 1)
    target = max(_target_size(item), 1)
    score = _as_float(item.get("score"), _as_float(item.get("target_score"), _as_float(item.get("edge_score"), 0.0))) or 0.0
    mean_sim = _as_float(item.get("target_mean_sim"), _as_float(item.get("view_mean_sim"), 0.0)) or 0.0
    best_sim = _as_float(item.get("target_best_sim"), _as_float(item.get("fused_sim"), 0.0)) or 0.0
    vote = _as_float(item.get("target_view_vote"), _as_float(item.get("votes_top5"), 0.0)) or 0.0
    if vote > 1.0:
        vote = min(vote / 5.0, 1.0)
    return float(
        0.30 * score
        + 0.18 * mean_sim
        + 0.15 * best_sim
        + 0.12 * vote
        + 0.14 * min(math.log1p(source) / math.log(256.0), 1.0)
        + 0.11 * min(math.log1p(target) / math.log(256.0), 1.0)
    )


def compose(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    raw_rows = 0
    for pattern in args.input:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for match in matches:
            path = Path(match)
            for row_rank, row in enumerate(_load_rows(path), start=1):
                raw_rows += 1
                if row.get("uses_anchors") is True or row.get("uses_gt_for_training_or_anchors") is True:
                    continue
                preview = row.get("accepted_preview")
                if not isinstance(preview, list):
                    continue
                parent_pair = _as_float(row.get("tracklet_pair_f1"), _as_float(row.get("pair_f1"), 0.0)) or 0.0
                parent_p = _as_float(row.get("tracklet_pair_precision"), _as_float(row.get("pair_precision"), parent_pair)) or 0.0
                parent_r = _as_float(row.get("tracklet_pair_recall"), _as_float(row.get("pair_recall"), parent_pair)) or 0.0
                if parent_pair < float(args.min_parent_pair_f1):
                    continue
                for edge_rank, item in enumerate(preview, start=1):
                    if not isinstance(item, dict):
                        continue
                    source = _component(item, "source_component_label", "source", "source_rep")
                    target = _component(item, "target_component", "target", "target_rep")
                    if source in (None, "") or target in (None, ""):
                        continue
                    source_size = _source_size(item)
                    target_size = _target_size(item)
                    if source_size < int(args.min_source_size) or target_size < int(args.min_target_size):
                        continue
                    edge_score = _item_score(item)
                    if edge_score < float(args.min_edge_score):
                        continue
                    moved = int(source_size) if source_size else len(item.get("source_seqs", []) or [])
                    row_out = {
                        "mode": str(args.mode),
                        "source_component_label": source,
                        "target_component": target,
                        "accepted_preview": [item],
                        "accepted_reassignments": 1,
                        "moved_tracklets": int(moved),
                        "target_components_used": 1,
                        "tracklet_pair_f1": float(parent_pair),
                        "tracklet_pair_precision": float(parent_p),
                        "tracklet_pair_recall": float(parent_r),
                        "full_side_effect_proxy": float(parent_pair - 0.115 + 0.012 * edge_score - 0.00005 * max(moved - 64, 0)),
                        "accepted_edges": 1,
                        "accepted_score_mean": float(edge_score),
                        "accepted_pair_mass_proxy_sum": float(max(source_size, 1) * max(target_size, 1)),
                        "accepted_mass_proxy_sum": float(math.sqrt(max(source_size, 1) * max(target_size, 1))),
                        "accepted_min_weight_sum": float(max(source_size, 1)),
                        "accepted_max_weight_sum": float(max(target_size, 1)),
                        "accepted_size_product_sum": float(max(source_size, 1) * max(target_size, 1)),
                        "preview_mean_score": float(edge_score),
                        "preview_mean_source_weight": float(max(source_size, 1)),
                        "preview_mean_target_weight": float(max(target_size, 1)),
                        "preview_max_pair_mass_proxy": float(max(source_size, 1) * max(target_size, 1)),
                        "parent_mode": row.get("mode"),
                        "parent_source_file": str(path),
                        "parent_row_rank": int(row_rank),
                        "parent_edge_rank": int(edge_rank),
                        "signature": repr(("single_edge", str(path), row_rank, edge_rank, str(source), str(target))),
                        "uses_anchors": False,
                        "uses_gt_for_training_or_anchors": False,
                        "uses_gt_for_evaluation_only": False,
                    }
                    rows.append(row_out)
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["source_component_label"]), str(row["target_component"]))
        old = deduped.get(key)
        if old is None or (
            float(row["accepted_score_mean"]),
            float(row["accepted_pair_mass_proxy_sum"]),
            float(row["tracklet_pair_f1"]),
        ) > (
            float(old["accepted_score_mean"]),
            float(old["accepted_pair_mass_proxy_sum"]),
            float(old["tracklet_pair_f1"]),
        ):
            deduped[key] = row
    rows = list(deduped.values())
    rows.sort(
        key=lambda row: (
            float(row["full_side_effect_proxy"]),
            float(row["accepted_score_mean"]),
            float(row["accepted_pair_mass_proxy_sum"]),
        ),
        reverse=True,
    )
    rows = rows[: int(args.top_n)]
    result = {
        "input": args.input,
        "raw_parent_rows": int(raw_rows),
        "rows": rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), rows)
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "source_component_label",
        "target_component",
        "moved_tracklets",
        "accepted_score_mean",
        "accepted_pair_mass_proxy_sum",
        "full_side_effect_proxy",
        "tracklet_pair_f1",
        "parent_mode",
        "parent_source_file",
        "parent_row_rank",
        "parent_edge_rank",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Single-Edge Candidates",
        "",
        f"- parent rows scanned: `{result['raw_parent_rows']}`",
        f"- emitted rows: `{len(result['rows'])}`",
        "",
        "| rank | source | target | moved | edge score | pair-mass proxy | proxy full | parent |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(result["rows"][:40], start=1):
        lines.append(
            f"| {rank} | `{row['source_component_label']}` | `{row['target_component']}` | "
            f"`{row['moved_tracklets']}` | `{row['accepted_score_mean']:.6f}` | "
            f"`{row['accepted_pair_mass_proxy_sum']:.3f}` | `{row['full_side_effect_proxy']:.6f}` | "
            f"`{row.get('parent_mode')}` |"
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
                            "mode": "edge_table_island_merge",
                            "tracklet_pair_f1": 0.77,
                            "tracklet_pair_precision": 0.82,
                            "tracklet_pair_recall": 0.73,
                            "accepted_preview": [
                                {"source": 1, "target": 2, "source_size": 5, "target_size": 10, "score": 0.8}
                            ],
                        }
                    ]
                }
            )
        )
        out = compose(
            argparse.Namespace(
                input=[str(src)],
                mode="single_edge",
                min_parent_pair_f1=0.7,
                min_source_size=1,
                min_target_size=1,
                min_edge_score=0.0,
                top_n=10,
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
                md=str(root / "out.md"),
            )
        )
        assert len(out["rows"]) == 1
        assert out["rows"][0]["source_component_label"] == 1
        assert out["rows"][0]["target_component"] == 2
        assert Path(root / "out.csv").read_text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--mode", default="single_edge_replay")
    ap.add_argument("--min-parent-pair-f1", type=float, default=0.70)
    ap.add_argument("--min-source-size", type=int, default=1)
    ap.add_argument("--min-target-size", type=int, default=1)
    ap.add_argument("--min-edge-score", type=float, default=0.0)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--json", default="local_runs/no_anchor_single_edge_candidates_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_single_edge_candidates_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_single_edge_candidates_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    if not args.input:
        ap.error("--input is required unless --self-test is used")
    out = compose(args)
    print(json.dumps({"raw_parent_rows": out["raw_parent_rows"], "rows": len(out["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
