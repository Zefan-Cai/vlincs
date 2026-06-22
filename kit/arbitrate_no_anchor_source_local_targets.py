#!/usr/bin/env python
"""Source-local target arbitration for no-anchor VLINCS bridge candidates.

This is a production-side selector: it flattens accepted_preview bridge
proposals from one or more scheduler/portfolio JSON files, then chooses at most
one target for each source component using only no-GT edge evidence.  The output
is scheduler-shaped so the existing assignment export and full-score runners can
consume it unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _as_int(value: Any, default: int | None = None) -> int | None:
    number = _as_float(value)
    if number is None:
        return default
    return int(number)


def _rows_from_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = []
        for key in ("selected", "rows", "top", "top_rows", "full_rows", "results"):
            value = data.get(key)
            if isinstance(value, list):
                rows.extend(value)
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def _preview(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("accepted_preview")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _component(item: dict[str, Any], row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = item.get(key, row.get(key))
        if value not in (None, ""):
            parsed = _as_int(value)
            if parsed is not None:
                return parsed
    return None


def _size(item: dict[str, Any], row: dict[str, Any], key: str, seq_key: str | None = None) -> float:
    value = _as_float(item.get(key, row.get(key)))
    if value is not None:
        return max(float(value), 1.0)
    if seq_key:
        seqs = item.get(seq_key)
        if isinstance(seqs, list) and seqs:
            return float(len(seqs))
    return 1.0


def _edge_feature(item: dict[str, Any], row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _as_float(item.get(key))
        if value is not None:
            return value
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _norm01(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return max(0.0, min(1.0, float(value)))


def _edge_score(edge: dict[str, Any]) -> float:
    # Every term below is available from proposal metadata.  No GT, anchors, or
    # full-score feedback are used here.
    target_score = _norm01(edge.get("target_score"), 0.0)
    best_sim = _norm01(edge.get("target_best_sim"), _norm01(edge.get("fused_sim"), 0.0))
    mean_sim = _norm01(edge.get("target_mean_sim"), _norm01(edge.get("fused_sim"), best_sim))
    min_view = _norm01(edge.get("target_min_view_sim"), _norm01(edge.get("target_view_vote"), 0.0))
    view_vote = _norm01(edge.get("target_view_vote"), 0.0)
    source_quality = _norm01(edge.get("source_quality"), 0.5)
    target_quality = _norm01(edge.get("target_quality"), 0.5)
    risk = _norm01(edge.get("target_impurity_risk_proxy"), 0.0)
    overlap = _norm01(edge.get("same_video_overlap_ratio"), 0.0)
    forbidden = _norm01(edge.get("target_forbidden_pairs"), _norm01(edge.get("is_forbidden"), 0.0))
    mass = math.log1p(max(float(edge.get("source_size", 1.0)), 1.0) * max(float(edge.get("target_size", 1.0)), 1.0))

    score = (
        0.28 * target_score
        + 0.18 * best_sim
        + 0.15 * mean_sim
        + 0.13 * view_vote
        + 0.10 * min_view
        + 0.08 * source_quality
        + 0.06 * target_quality
        + 0.02 * min(mass / 10.0, 1.0)
    )
    score -= 0.08 * risk
    score -= 0.10 * overlap
    score -= 0.06 * min(forbidden, 1.0)
    return float(score)


def _flatten(paths: list[Path]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for path in paths:
        for row_rank, row in enumerate(_rows_from_json(path), start=1):
            for preview_rank, item in enumerate(_preview(row), start=1):
                source = _component(item, row, "source_component_label", "source", "source_rep")
                target = _component(item, row, "target_component", "target", "target_rep")
                if source is None or target is None or source == target:
                    continue
                source_size = _size(item, row, "source_size", "source_seqs")
                target_size = _size(item, row, "target_size", "target_top_seqs")
                edge = {
                    "source_component_label": int(source),
                    "target_component": int(target),
                    "source_size": float(source_size),
                    "target_size": float(target_size),
                    "source_seqs": item.get("source_seqs") if isinstance(item.get("source_seqs"), list) else None,
                    "origin_file": str(path),
                    "origin_rank": int(row_rank),
                    "origin_preview_rank": int(preview_rank),
                    "origin_mode": str(row.get("mode") or row.get("policy_name") or "candidate"),
                    "referee_keep_reason": item.get("referee_keep_reason", row.get("referee_keep_reason")),
                    "target_score": _edge_feature(item, row, "target_score", "score", "edge_score", "no_gt_component_graph_score"),
                    "target_best_sim": _edge_feature(item, row, "target_best_sim", "fused_sim", "db_sim", "primary_sim"),
                    "target_mean_sim": _edge_feature(item, row, "target_mean_sim", "view_mean_sim", "fused_sim"),
                    "target_min_view_sim": _edge_feature(item, row, "target_min_view_sim", "view_min_sim"),
                    "target_view_vote": _edge_feature(item, row, "target_view_vote", "votes_top5"),
                    "source_quality": _edge_feature(item, row, "source_quality"),
                    "target_quality": _edge_feature(item, row, "target_quality"),
                    "same_video_overlap_ratio": _edge_feature(item, row, "same_video_overlap_ratio"),
                    "target_impurity_risk_proxy": _edge_feature(item, row, "target_impurity_risk_proxy"),
                    "target_forbidden_pairs": _edge_feature(item, row, "target_forbidden_pairs"),
                    "is_forbidden": _edge_feature(item, row, "is_forbidden"),
                    "raw_preview": item,
                }
                edge["arbitration_score"] = _edge_score(edge)
                edges.append(edge)
    return edges


def _dedup_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]] = {}
    for edge in edges:
        seqs = edge.get("source_seqs")
        seq_key = tuple(sorted(int(float(seq)) for seq in seqs)) if isinstance(seqs, list) else tuple()
        key = (int(edge["source_component_label"]), int(edge["target_component"]), seq_key)
        current = best.get(key)
        if current is None or float(edge["arbitration_score"]) > float(current["arbitration_score"]):
            best[key] = edge
    return list(best.values())


def _choose_by_source(edges: list[dict[str, Any]], *, min_score: float, min_margin: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        by_source[int(edge["source_component_label"])].append(edge)

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source, group in sorted(by_source.items()):
        group.sort(key=lambda edge: float(edge["arbitration_score"]), reverse=True)
        best = group[0]
        runner = group[1] if len(group) > 1 else None
        margin = float(best["arbitration_score"]) - (float(runner["arbitration_score"]) if runner else 0.0)
        best["source_local_margin"] = margin
        best["source_local_competing_targets"] = [
            {
                "target_component": int(edge["target_component"]),
                "arbitration_score": round(float(edge["arbitration_score"]), 6),
                "origin_file": edge["origin_file"],
                "origin_rank": edge["origin_rank"],
            }
            for edge in group[:5]
        ]
        if float(best["arbitration_score"]) >= min_score and margin >= min_margin:
            selected.append(best)
            for edge in group[1:]:
                edge["reject_reason"] = "source_local_lower_score"
                rejected.append(edge)
        else:
            for edge in group:
                edge["reject_reason"] = "source_local_low_score_or_margin"
                rejected.append(edge)
    return selected, rejected


def _drop_chains(edges: list[dict[str, Any]], *, max_sources_per_target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    target_counts: defaultdict[int, int] = defaultdict(int)
    for edge in sorted(edges, key=lambda item: float(item["arbitration_score"]), reverse=True):
        source = int(edge["source_component_label"])
        target = int(edge["target_component"])
        if source in used_sources:
            edge["reject_reason"] = "duplicate_source_after_sort"
            rejected.append(edge)
            continue
        if target in used_sources or source in used_targets:
            edge["reject_reason"] = "chained_component_edit"
            rejected.append(edge)
            continue
        if target_counts[target] >= max_sources_per_target:
            edge["reject_reason"] = "target_capacity"
            rejected.append(edge)
            continue
        kept.append(edge)
        used_sources.add(source)
        used_targets.add(target)
        target_counts[target] += 1
    return kept, rejected


def _edge_to_preview(edge: dict[str, Any]) -> dict[str, Any]:
    item = dict(edge["raw_preview"])
    item["source_component_label"] = int(edge["source_component_label"])
    item["target_component"] = int(edge["target_component"])
    item["source_size"] = int(round(float(edge["source_size"])))
    item["target_size"] = int(round(float(edge["target_size"])))
    item["arbitration_score"] = float(edge["arbitration_score"])
    item["source_local_margin"] = float(edge.get("source_local_margin", 0.0))
    item["source_local_competing_targets"] = edge.get("source_local_competing_targets", [])
    item["arbitration_origin_file"] = edge["origin_file"]
    item["arbitration_origin_rank"] = int(edge["origin_rank"])
    item["arbitration_origin_preview_rank"] = int(edge["origin_preview_rank"])
    return item


def _portfolio_row(name: str, edges: list[dict[str, Any]], current_best: float) -> dict[str, Any]:
    preview = [_edge_to_preview(edge) for edge in edges]
    moved = sum(int(round(float(edge["source_size"]))) for edge in edges)
    pair_mass = sum(float(edge["source_size"]) * float(edge["target_size"]) for edge in edges)
    score_mean = sum(float(edge["arbitration_score"]) for edge in edges) / max(len(edges), 1)
    predicted = current_best + min(0.012, 0.0015 * len(edges)) + min(0.002, 0.000002 * pair_mass)
    return {
        "mode": name,
        "source_component_label": "+".join(str(int(edge["source_component_label"])) for edge in edges),
        "target_component": "+".join(str(int(edge["target_component"])) for edge in edges),
        "accepted_preview": preview,
        "accepted_reassignments": int(len(edges)),
        "moved_tracklets": int(moved),
        "target_components_used": int(len({int(edge["target_component"]) for edge in edges})),
        "learned_proxy_full_idf1": float(predicted),
        "predicted_full_idf1": float(predicted),
        "full_side_effect_proxy": float(predicted),
        "accepted_edges": int(len(edges)),
        "accepted_score_mean": float(score_mean),
        "accepted_pair_mass_proxy_sum": float(pair_mass),
        "accepted_mass_proxy_sum": float(sum(math.sqrt(float(edge["source_size"]) * float(edge["target_size"])) for edge in edges)),
        "arbitration_score_mean": float(score_mean),
        "arbitration_score_min": float(min((float(edge["arbitration_score"]) for edge in edges), default=0.0)),
        "arbitration_source_margins_min": float(min((float(edge.get("source_local_margin", 0.0)) for edge in edges), default=0.0)),
        "origin_files": sorted({edge["origin_file"] for edge in edges}),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }


def arbitrate(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(path) for path in args.input_json]
    edges = _dedup_edges(_flatten(paths))
    by_source, rejected_source = _choose_by_source(edges, min_score=float(args.min_score), min_margin=float(args.min_source_margin))
    kept, rejected_chain = _drop_chains(by_source, max_sources_per_target=int(args.max_sources_per_target))
    kept.sort(key=lambda edge: float(edge["arbitration_score"]), reverse=True)

    selected: list[dict[str, Any]] = []
    if kept:
        selected.append(_portfolio_row("source_local_arbitrated_all", kept, float(args.current_best_full_idf1)))
    top_edges = kept[: int(args.top_edges)]
    if top_edges and len(top_edges) != len(kept):
        selected.append(_portfolio_row("source_local_arbitrated_top", top_edges, float(args.current_best_full_idf1)))
    high_edges = [edge for edge in kept if float(edge["arbitration_score"]) >= float(args.high_score)]
    if high_edges and {int(e["source_component_label"]) for e in high_edges} != {int(e["source_component_label"]) for e in top_edges}:
        selected.append(_portfolio_row("source_local_arbitrated_highconf", high_edges, float(args.current_best_full_idf1)))

    rejected = rejected_source + rejected_chain
    out = {
        "input_json": [str(path) for path in paths],
        "edge_pool": int(len(edges)),
        "kept_edges": int(len(kept)),
        "rejected_edges": int(len(rejected)),
        "selected": selected,
        "kept_edge_preview": [_edge_summary(edge) for edge in kept],
        "rejected_edge_preview": [_edge_summary(edge) for edge in sorted(rejected, key=lambda e: float(e.get("arbitration_score", 0.0)), reverse=True)[:50]],
        "selector": {
            "name": "source_local_target_arbitration",
            "min_score": float(args.min_score),
            "min_source_margin": float(args.min_source_margin),
            "high_score": float(args.high_score),
            "max_sources_per_target": int(args.max_sources_per_target),
            "uses_gt": False,
        },
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), selected)
    if args.md:
        _write_md(Path(args.md), out)
    return out


def _edge_summary(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_component_label": int(edge["source_component_label"]),
        "target_component": int(edge["target_component"]),
        "source_size": int(round(float(edge["source_size"]))),
        "target_size": int(round(float(edge["target_size"]))),
        "arbitration_score": round(float(edge["arbitration_score"]), 6),
        "source_local_margin": round(float(edge.get("source_local_margin", 0.0)), 6),
        "target_score": edge.get("target_score"),
        "target_best_sim": edge.get("target_best_sim"),
        "target_mean_sim": edge.get("target_mean_sim"),
        "target_min_view_sim": edge.get("target_min_view_sim"),
        "target_view_vote": edge.get("target_view_vote"),
        "risk_proxy": edge.get("target_impurity_risk_proxy"),
        "reject_reason": edge.get("reject_reason"),
        "origin_file": edge.get("origin_file"),
        "origin_rank": edge.get("origin_rank"),
        "origin_preview_rank": edge.get("origin_preview_rank"),
        "reason": edge.get("referee_keep_reason"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "source_component_label",
        "target_component",
        "accepted_reassignments",
        "moved_tracklets",
        "target_components_used",
        "learned_proxy_full_idf1",
        "arbitration_score_mean",
        "arbitration_score_min",
        "arbitration_source_margins_min",
        "accepted_pair_mass_proxy_sum",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Source-Local Target Arbitration",
        "",
        "Production selector: uses no GT, no anchors, and no full-score feedback.",
        "",
        f"- edge pool: `{out['edge_pool']}`",
        f"- kept edges: `{out['kept_edges']}`",
        f"- selected portfolios: `{len(out['selected'])}`",
        "",
        "## Selected Portfolios",
        "",
        "| rank | mode | sources | targets | moved | mean score | min margin |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(out["selected"], start=1):
        lines.append(
            "| "
            f"{rank} | `{row['mode']}` | `{row['source_component_label']}` | `{row['target_component']}` | "
            f"{row['moved_tracklets']} | {row['arbitration_score_mean']:.6f} | {row['arbitration_source_margins_min']:.6f} |"
        )
    lines.extend(["", "## Kept Edge Preview", ""])
    for edge in out["kept_edge_preview"][:20]:
        lines.append(
            "- "
            f"`{edge['source_component_label']}` -> `{edge['target_component']}` "
            f"score `{edge['arbitration_score']:.6f}`, margin `{edge['source_local_margin']:.6f}`, "
            f"origin `{Path(edge['origin_file']).name}` rank `{edge['origin_rank']}`"
        )
    lines.extend(["", "## Rejected Edge Preview", ""])
    for edge in out["rejected_edge_preview"][:20]:
        lines.append(
            "- "
            f"`{edge['source_component_label']}` -> `{edge['target_component']}` "
            f"score `{edge['arbitration_score']:.6f}`, reason `{edge.get('reject_reason')}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inp = root / "cands.json"
        inp.write_text(
            json.dumps(
                {
                    "selected": [
                        {
                            "mode": "unit",
                            "accepted_preview": [
                                {
                                    "source_component_label": 9,
                                    "target_component": 7,
                                    "source_size": 8,
                                    "target_size": 20,
                                    "target_score": 0.78,
                                    "target_best_sim": 0.94,
                                    "target_mean_sim": 0.72,
                                    "target_view_vote": 0.66,
                                    "source_quality": 0.88,
                                    "target_quality": 0.85,
                                },
                                {
                                    "source_component_label": 9,
                                    "target_component": 6,
                                    "source_size": 77,
                                    "target_size": 172,
                                    "target_score": 0.65,
                                    "target_best_sim": 0.80,
                                    "target_mean_sim": 0.41,
                                    "target_view_vote": 1.0,
                                    "target_impurity_risk_proxy": 0.79,
                                },
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = root / "out.json"
        args = argparse.Namespace(
            input_json=[str(inp)],
            json=str(out),
            csv=None,
            md=None,
            min_score=0.0,
            min_source_margin=0.0,
            high_score=0.70,
            top_edges=8,
            max_sources_per_target=1,
            current_best_full_idf1=0.65524,
        )
        result = arbitrate(args)
        preview = result["selected"][0]["accepted_preview"]
        assert preview[0]["source_component_label"] == 9
        assert preview[0]["target_component"] == 7, preview
        assert out.exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-json", action="append", default=[])
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--min-score", type=float, default=0.55)
    ap.add_argument("--min-source-margin", type=float, default=0.01)
    ap.add_argument("--high-score", type=float, default=0.70)
    ap.add_argument("--top-edges", type=int, default=8)
    ap.add_argument("--max-sources-per-target", type=int, default=1)
    ap.add_argument("--current-best-full-idf1", type=float, default=0.65524)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    if not args.input_json or not args.json:
        ap.error("--input-json and --json are required unless --self-test is set")
    result = arbitrate(args)
    print(json.dumps({"edge_pool": result["edge_pool"], "kept_edges": result["kept_edges"], "selected": len(result["selected"])}, sort_keys=True))


if __name__ == "__main__":
    main()
