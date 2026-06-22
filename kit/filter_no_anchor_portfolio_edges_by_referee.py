#!/usr/bin/env python
"""Filter portfolio accepted_preview edges with no-GT referee rules.

This script is a conservative challenger generator. It does not discover new
edges; it prunes weak edges from already proposed no-anchor portfolios while
preserving specialized edge families that have different evidence shapes.
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


def _preview(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("accepted_preview")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _component(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(_as_int(value))
    return ""


def _source_size(item: dict[str, Any]) -> float:
    if item.get("source_size") not in (None, ""):
        return _as_float(item.get("source_size"))
    seqs = item.get("source_seqs")
    return float(len(seqs)) if isinstance(seqs, list) else 0.0


def _target_size(item: dict[str, Any]) -> float:
    if item.get("target_size") not in (None, ""):
        return _as_float(item.get("target_size"))
    seqs = item.get("target_top_seqs")
    return float(len(seqs)) if isinstance(seqs, list) else 0.0


def _target_best(item: dict[str, Any]) -> float:
    return _as_float(item.get("target_best_sim"), _as_float(item.get("fused_sim")))


def _target_mean(item: dict[str, Any]) -> float:
    return _as_float(item.get("target_mean_sim"), _as_float(item.get("view_mean_sim")))


def _target_vote(item: dict[str, Any]) -> float:
    vote = _as_float(item.get("target_view_vote"), _as_float(item.get("votes_top5")))
    return min(vote / 5.0, 1.0) if vote > 1.0 else vote


def _countertarget_reject_reason(item: dict[str, Any], args: argparse.Namespace) -> str:
    if not bool(getattr(args, "require_countertarget_accept", False)):
        return ""
    verdict = str(item.get("countertarget_verdict", item.get("verdict", ""))).strip()
    if not verdict:
        return "missing_countertarget_verdict"
    if verdict != "accept":
        return f"countertarget_{verdict}"
    risk = _as_float(item.get("combined_opponent_risk_score"), 0.0)
    if risk > float(getattr(args, "max_countertarget_risk", 1.0)):
        return "countertarget_risk_too_high"
    return ""


def _keep_reason(item: dict[str, Any], args: argparse.Namespace) -> str:
    source_size = _source_size(item)
    target_size = _target_size(item)
    target_best = _target_best(item)
    target_mean = _target_mean(item)
    target_vote = _target_vote(item)
    target_min_view = _as_float(item.get("target_min_view_sim"), 0.0)
    target_score = _as_float(item.get("target_score"), _as_float(item.get("score")))
    source_quality = _as_float(item.get("source_quality"), 0.0)
    source_internal = _as_float(item.get("source_internal_sim"), 0.0)
    source_cross = _as_float(item.get("source_cross_mean_sim"), 1.0)
    source_conflicts = _as_float(item.get("source_conflicts_to_rest"), 99.0)
    localized_peak = max(target_best - target_mean, 0.0)

    if source_size >= args.min_large_source_size and target_size <= args.max_singleton_target_size:
        return "large_source_singleton_seed"
    if (
        target_best >= args.localized_min_target_best
        and localized_peak >= args.localized_min_peak
        and source_cross <= args.localized_max_source_cross
        and source_conflicts <= args.localized_max_conflicts
        and source_internal >= args.localized_min_source_internal
        and source_quality >= args.localized_min_source_quality
        and target_score >= args.localized_min_target_score
    ):
        return "localized_island"
    if (
        target_best >= args.stable_min_target_best
        and target_vote >= args.stable_min_vote
        and target_min_view >= args.stable_min_target_min_view
        and source_internal >= args.stable_min_source_internal
        and source_cross <= args.stable_max_source_cross
        and source_quality >= args.stable_min_source_quality
        and target_score >= args.stable_min_target_score
        and source_size <= args.stable_max_source_size
        and target_size >= args.stable_min_target_size
    ):
        return "stable_multiview_attach"
    return ""


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pair_mass(preview: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in preview:
        total += max(_source_size(item), 1.0) * max(_target_size(item), 1.0)
    return float(total)


def _rebuild_row(row: dict[str, Any], kept: list[dict[str, Any]], rejected: list[dict[str, Any]], rank: int, current_best: float) -> dict[str, Any]:
    sources = sorted({_component(item, "source_component_label", "source", "source_rep") for item in kept if _component(item, "source_component_label", "source", "source_rep")})
    targets = sorted({_component(item, "target_component", "target", "target_rep") for item in kept if _component(item, "target_component", "target", "target_rep")})
    moved = int(sum(_source_size(item) for item in kept))
    pair_mass = _pair_mass(kept)
    old_proxy = _as_float(row.get("learned_proxy_full_idf1"), _as_float(row.get("predicted_full_idf1"), _as_float(row.get("full_side_effect_proxy"), current_best)))
    # Pruning should lower delivery risk but also admits less evidence. Keep this
    # proxy conservative; canonical full-score is still required.
    kept_ratio = len(kept) / max(len(kept) + len(rejected), 1)
    predicted = max(current_best, old_proxy - 0.0007 * len(rejected) + 0.00025 * max(len(kept) - 2, 0) + 0.0004 * (1.0 - kept_ratio))
    new = dict(row)
    new.update(
        {
            "mode": "referee_pruned_portfolio",
            "source_component_label": "+".join(sources),
            "target_component": "+".join(targets),
            "accepted_preview": kept,
            "accepted_reassignments": int(len(kept)),
            "moved_tracklets": moved,
            "target_components_used": int(len(targets)),
            "learned_proxy_full_idf1": float(predicted),
            "predicted_full_idf1": float(predicted),
            "full_side_effect_proxy": float(predicted),
            "accepted_pair_mass_proxy_sum": float(pair_mass),
            "accepted_min_weight_sum": float(sum(max(_source_size(item), 1.0) for item in kept)),
            "accepted_max_weight_sum": float(sum(max(_target_size(item), 1.0) for item in kept)),
            "accepted_size_product_sum": float(pair_mass),
            "referee_kept_edges": int(len(kept)),
            "referee_rejected_edges": int(len(rejected)),
            "referee_kept_reasons": [str(item.get("referee_keep_reason", "")) for item in kept],
            "referee_rejected_edges_preview": rejected[:20],
            "source_rank": int(rank),
            "signature": repr(("referee_pruned_portfolio", row.get("signature"), tuple((item.get("source_component_label", item.get("source")), item.get("target_component", item.get("target")), item.get("referee_keep_reason")) for item in kept))),
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": False,
        }
    )
    return new


def filter_rows(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(Path(args.candidates_json))
    out_rows: list[dict[str, Any]] = []
    raw_edges = 0
    kept_edges = 0
    for rank, row in enumerate(rows, start=1):
        preview = _preview(row)
        if not preview:
            continue
        raw_edges += len(preview)
        kept: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for item in preview:
            countertarget_reason = _countertarget_reject_reason(item, args)
            if countertarget_reason:
                item2 = dict(item)
                item2["referee_reject_reason"] = countertarget_reason
                rejected.append(item2)
                continue
            reason = _keep_reason(item, args)
            item2 = dict(item)
            if reason:
                item2["referee_keep_reason"] = reason
                kept.append(item2)
            else:
                item2["referee_reject_reason"] = "no_rule_match"
                rejected.append(item2)
        kept_edges += len(kept)
        if len(kept) < int(args.min_kept_edges):
            continue
        moved = int(sum(_source_size(item) for item in kept))
        if moved < int(args.min_moved_tracklets):
            continue
        out_rows.append(_rebuild_row(row, kept, rejected, rank, float(args.current_best_full_idf1)))
    dedup: dict[str, dict[str, Any]] = {}
    for row in out_rows:
        old = dedup.get(str(row["signature"]))
        if old is None or (
            float(row.get("learned_proxy_full_idf1", 0.0)),
            -float(row.get("moved_tracklets", 0.0)),
        ) > (
            float(old.get("learned_proxy_full_idf1", 0.0)),
            -float(old.get("moved_tracklets", 0.0)),
        ):
            dedup[str(row["signature"])] = row
    out_rows = list(dedup.values())
    out_rows.sort(key=lambda row: (float(row["learned_proxy_full_idf1"]), -int(row["moved_tracklets"])), reverse=True)
    out_rows = out_rows[: int(args.top_n)]
    result = {
        "candidates_json": str(args.candidates_json),
        "input_rows": int(len(rows)),
        "raw_edges": int(raw_edges),
        "kept_edges": int(kept_edges),
        "require_countertarget_accept": bool(args.require_countertarget_accept),
        "max_countertarget_risk": float(args.max_countertarget_risk),
        "rows": out_rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
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
        "mode",
        "source_component_label",
        "target_component",
        "accepted_reassignments",
        "moved_tracklets",
        "target_components_used",
        "tracklet_pair_f1",
        "tracklet_pair_precision",
        "tracklet_pair_recall",
        "learned_proxy_full_idf1",
        "accepted_pair_mass_proxy_sum",
        "referee_kept_edges",
        "referee_rejected_edges",
        "referee_kept_reasons",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Referee-Pruned Portfolio Candidates",
        "",
        f"- source: `{result['candidates_json']}`",
        f"- input rows: `{result['input_rows']}`",
        f"- raw edges: `{result['raw_edges']}`",
        f"- kept edges: `{result['kept_edges']}`",
        f"- require countertarget accept: `{str(result['require_countertarget_accept']).lower()}`",
        f"- max countertarget risk: `{float(result['max_countertarget_risk']):.3f}`",
        f"- emitted rows: `{len(result['rows'])}`",
        "",
        "| rank | predicted full | moved | kept | rejected | targets | reasons |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(result["rows"][:30], start=1):
        reasons = ",".join(str(x) for x in row.get("referee_kept_reasons", []))
        lines.append(
            f"| {rank} | `{float(row['learned_proxy_full_idf1']):.6f}` | `{row['moved_tracklets']}` | "
            f"`{row['referee_kept_edges']}` | `{row['referee_rejected_edges']}` | "
            f"`{row['target_component']}` | `{reasons}` |"
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
                            "learned_proxy_full_idf1": 0.66,
                            "tracklet_pair_f1": 0.77,
                            "accepted_preview": [
                                {"source_component_label": 3, "target_component": 68, "source_size": 248, "target_size": 1},
                                {
                                    "source_component_label": 9,
                                    "target_component": 7,
                                    "source_size": 8,
                                    "target_size": 216,
                                    "target_best_sim": 0.94,
                                    "target_mean_sim": 0.72,
                                    "target_score": 0.79,
                                    "source_quality": 0.88,
                                    "source_internal_sim": 0.91,
                                    "source_cross_mean_sim": 0.51,
                                    "source_conflicts_to_rest": 3,
                                },
                                {
                                    "source_component_label": 0,
                                    "target_component": 6,
                                    "source_size": 8,
                                    "target_size": 172,
                                    "target_best_sim": 0.82,
                                    "target_mean_sim": 0.79,
                                    "target_view_vote": 0.75,
                                    "target_min_view_sim": 0.35,
                                    "target_score": 0.80,
                                    "source_quality": 0.91,
                                    "source_internal_sim": 0.84,
                                    "source_cross_mean_sim": 0.63,
                                },
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = filter_rows(
            argparse.Namespace(
                candidates_json=str(src),
                current_best_full_idf1=0.65524,
                min_large_source_size=64.0,
                max_singleton_target_size=4.0,
                localized_min_target_best=0.90,
                localized_min_peak=0.12,
                localized_max_source_cross=0.56,
                localized_max_conflicts=6.0,
                localized_min_source_internal=0.85,
                localized_min_source_quality=0.75,
                localized_min_target_score=0.75,
                stable_min_target_best=0.83,
                stable_min_vote=0.75,
                stable_min_target_min_view=0.45,
                stable_min_source_internal=0.88,
                stable_max_source_cross=0.65,
                stable_min_source_quality=0.75,
                stable_min_target_score=0.79,
                stable_max_source_size=16.0,
                stable_min_target_size=32.0,
                min_kept_edges=1,
                min_moved_tracklets=1,
                top_n=10,
                require_countertarget_accept=False,
                max_countertarget_risk=0.50,
                json=str(root / "out.json"),
                csv="",
                md="",
            )
        )
        assert len(out["rows"]) == 1
        assert out["rows"][0]["referee_kept_edges"] == 2
        assert out["rows"][0]["referee_rejected_edges"] == 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidates-json", default="")
    ap.add_argument("--current-best-full-idf1", type=float, default=0.65524)
    ap.add_argument("--min-large-source-size", type=float, default=64.0)
    ap.add_argument("--max-singleton-target-size", type=float, default=4.0)
    ap.add_argument("--localized-min-target-best", type=float, default=0.90)
    ap.add_argument("--localized-min-peak", type=float, default=0.12)
    ap.add_argument("--localized-max-source-cross", type=float, default=0.56)
    ap.add_argument("--localized-max-conflicts", type=float, default=6.0)
    ap.add_argument("--localized-min-source-internal", type=float, default=0.85)
    ap.add_argument("--localized-min-source-quality", type=float, default=0.75)
    ap.add_argument("--localized-min-target-score", type=float, default=0.75)
    ap.add_argument("--stable-min-target-best", type=float, default=0.83)
    ap.add_argument("--stable-min-vote", type=float, default=0.75)
    ap.add_argument("--stable-min-target-min-view", type=float, default=0.45)
    ap.add_argument("--stable-min-source-internal", type=float, default=0.88)
    ap.add_argument("--stable-max-source-cross", type=float, default=0.65)
    ap.add_argument("--stable-min-source-quality", type=float, default=0.75)
    ap.add_argument("--stable-min-target-score", type=float, default=0.79)
    ap.add_argument("--stable-max-source-size", type=float, default=16.0)
    ap.add_argument("--stable-min-target-size", type=float, default=32.0)
    ap.add_argument("--min-kept-edges", type=int, default=1)
    ap.add_argument("--min-moved-tracklets", type=int, default=1)
    ap.add_argument("--require-countertarget-accept", action="store_true")
    ap.add_argument("--max-countertarget-risk", type=float, default=0.50)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--json", default="local_runs/no_anchor_referee_pruned_portfolio_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_referee_pruned_portfolio_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_referee_pruned_portfolio_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    if not args.candidates_json:
        ap.error("--candidates-json is required unless --self-test is used")
    out = filter_rows(args)
    print(json.dumps({"input_rows": out["input_rows"], "rows": len(out["rows"]), "raw_edges": out["raw_edges"], "kept_edges": out["kept_edges"]}, sort_keys=True))


if __name__ == "__main__":
    main()
