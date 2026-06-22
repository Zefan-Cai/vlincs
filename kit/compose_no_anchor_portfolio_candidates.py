#!/usr/bin/env python
"""Compose portfolio-level no-anchor assignment repair candidates.

Single accepted-preview rows are often too small to move end-to-end IDF1.
This utility builds larger portfolio rows by combining already executable
no-anchor scheduler rows whose source tracklets do not overlap and whose
component edits are not chained.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.export_no_anchor_scheduler_manifest_assignments import _accepted_preview, _load_scheduler_selected


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


def _source_components(preview: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for item in preview:
        value = item.get("source_component_label", item.get("source", item.get("source_rep", "")))
        if value not in ("", None):
            out.add(str(value))
    return out


def _target_components(preview: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for item in preview:
        value = item.get("target_component", item.get("target", item.get("target_rep", "")))
        if value not in ("", None):
            out.add(str(value))
    return out


def _source_seqs(preview: list[dict[str, Any]]) -> set[int]:
    seqs: set[int] = set()
    for item in preview:
        raw = item.get("source_seqs")
        if isinstance(raw, list):
            seqs.update(int(float(seq)) for seq in raw)
    return seqs


def _moved(preview: list[dict[str, Any]]) -> int:
    total = 0
    for item in preview:
        size = _as_float(item.get("source_size"))
        if size is None:
            raw = item.get("source_seqs")
            size = len(raw) if isinstance(raw, list) else 0
        total += int(size)
    return total


def _target_size(item: dict[str, Any]) -> int:
    size = _as_float(item.get("target_size"))
    if size is not None:
        return int(size)
    raw = item.get("target_top_seqs")
    return len(raw) if isinstance(raw, list) else 1


def _pair_mass(preview: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in preview:
        source = _as_float(item.get("source_size"))
        if source is None:
            raw = item.get("source_seqs")
            source = float(len(raw)) if isinstance(raw, list) else 1.0
        total += max(float(source), 1.0) * max(float(_target_size(item)), 1.0)
    return float(total)


def _bridge_mass(preview: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in preview:
        source = _as_float(item.get("source_size"))
        if source is None:
            raw = item.get("source_seqs")
            source = float(len(raw)) if isinstance(raw, list) else 1.0
        total += math.sqrt(max(float(source), 1.0) * max(float(_target_size(item)), 1.0))
    return float(total)


def _mean(values: list[float | None], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else float(default)


def _preview_mean(preview: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    return _mean([_as_float(item.get(key)) for item in preview], default=default)


def _preview_min(preview: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [_as_float(item.get(key)) for item in preview]
    vals = [float(v) for v in vals if v is not None]
    return float(min(vals)) if vals else float(default)


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:+-]+", "_", str(text)).strip("_")


def _candidate_rows(scheduler_json: Path, ranks: str) -> list[dict[str, Any]]:
    rows = _load_scheduler_selected(scheduler_json)
    keep = None
    if ranks:
        keep = {int(part.strip()) for part in ranks.split(",") if part.strip()}
    out: list[dict[str, Any]] = []
    for row in rows:
        rank = int(row["_selection_rank"])
        if keep is not None and rank not in keep:
            continue
        preview = _accepted_preview(row)
        seqs = _source_seqs(preview)
        out.append(
            {
                "rank": rank,
                "row": row,
                "preview": preview,
                "source_components": _source_components(preview),
                "target_components": _target_components(preview),
                "source_seqs": seqs,
                "moved": _moved(preview),
                "pair_mass": _pair_mass(preview),
                "predicted_full": _as_float(row.get("predicted_full_idf1"), _as_float(row.get("learned_proxy_full_idf1"), 0.0)) or 0.0,
                "pair_f1": _as_float(row.get("pair_f1_norm"), _as_float(row.get("tracklet_pair_f1"), _as_float(row.get("pair_f1"), 0.0))) or 0.0,
                "pair_precision": _as_float(row.get("pair_precision_norm"), _as_float(row.get("tracklet_pair_precision"), _as_float(row.get("pair_precision"), 0.0))) or 0.0,
                "pair_recall": _as_float(row.get("pair_recall_norm"), _as_float(row.get("tracklet_pair_recall"), _as_float(row.get("pair_recall"), 0.0))) or 0.0,
            }
        )
    return out


def _compatible(items: tuple[dict[str, Any], ...]) -> bool:
    used_seqs: set[int] = set()
    all_sources: set[str] = set()
    all_targets: set[str] = set()
    for item in items:
        seqs = item["source_seqs"]
        if seqs and used_seqs & seqs:
            return False
        used_seqs.update(seqs)
        sources = set(item["source_components"])
        targets = set(item["target_components"])
        if sources & all_sources:
            return False
        if targets & all_sources or sources & all_targets:
            return False
        all_sources.update(sources)
        all_targets.update(targets)
    return True


def _make_portfolio(items: tuple[dict[str, Any], ...], current_best: float) -> dict[str, Any]:
    preview: list[dict[str, Any]] = []
    for item in items:
        preview.extend(item["preview"])
    moved = _moved(preview)
    pair_mass = _pair_mass(preview)
    bridge_mass = _bridge_mass(preview)
    predicted_vals = [float(item["predicted_full"]) for item in items]
    deltas = [max(value - float(current_best), 0.0) for value in predicted_vals]
    predicted = max(predicted_vals + [float(current_best)]) + 0.55 * sum(sorted(deltas, reverse=True)[1:])
    predicted += min(0.0015, 0.000015 * max(moved - 16, 0))
    predicted -= 0.0004 * max(len(items) - 3, 0)
    sources = sorted(set().union(*(item["source_components"] for item in items)))
    targets = sorted(set().union(*(item["target_components"] for item in items)))
    row = {
        "mode": "hub_bridge_portfolio",
        "source_component_label": "+".join(sources),
        "target_component": "+".join(targets),
        "accepted_preview": preview,
        "accepted_reassignments": int(len(preview)),
        "moved_tracklets": int(moved),
        "target_components_used": int(len(targets)),
        "tracklet_pair_f1": _mean([item["pair_f1"] for item in items]),
        "tracklet_pair_precision": _mean([item["pair_precision"] for item in items]),
        "tracklet_pair_recall": _mean([item["pair_recall"] for item in items]),
        "learned_proxy_full_idf1": float(predicted),
        "full_side_effect_proxy": float(predicted),
        "accepted_edges": int(len(preview)),
        "accepted_score_mean": _preview_mean(preview, "target_score", _preview_mean(preview, "score")),
        "accepted_view_mean_sim_mean": _preview_mean(preview, "target_mean_sim"),
        "accepted_view_min_sim_mean": _preview_mean(preview, "target_min_view_sim"),
        "accepted_mass_proxy_sum": float(bridge_mass),
        "accepted_pair_mass_proxy_sum": float(pair_mass),
        "accepted_min_weight_sum": float(sum(max(_as_float(item.get("source_size"), 1.0) or 1.0, 1.0) for item in preview)),
        "accepted_max_weight_sum": float(sum(max(float(_target_size(item)), 1.0) for item in preview)),
        "accepted_size_product_sum": float(pair_mass),
        "preview_mean_target_mean_sim": _preview_mean(preview, "target_mean_sim"),
        "preview_mean_target_best_sim": _preview_mean(preview, "target_best_sim"),
        "preview_mean_target_view_vote": _preview_mean(preview, "target_view_vote"),
        "preview_min_target_min_view_sim": _preview_min(preview, "target_min_view_sim"),
        "preview_mean_source_quality": _preview_mean(preview, "source_quality"),
        "preview_mean_source_score": _preview_mean(preview, "source_score"),
        "preview_mean_source_size": _preview_mean(preview, "source_size"),
        "portfolio_source_ranks": [int(item["rank"]) for item in items],
        "portfolio_source_modes": [str(item["row"].get("mode") or "") for item in items],
        "signature": repr(("hub_bridge_portfolio", tuple(int(item["rank"]) for item in items))),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    return row


def compose(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _candidate_rows(Path(args.scheduler_json), args.ranks)
    rows: list[dict[str, Any]] = []
    for size in range(int(args.min_items), int(args.max_items) + 1):
        for combo in itertools.combinations(candidates, size):
            if not _compatible(combo):
                continue
            row = _make_portfolio(combo, float(args.current_best_full_idf1))
            if int(row["moved_tracklets"]) < int(args.min_moved_tracklets):
                continue
            rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["learned_proxy_full_idf1"]),
            float(row["accepted_pair_mass_proxy_sum"]),
            int(row["moved_tracklets"]),
        ),
        reverse=True,
    )
    rows = rows[: int(args.top_n)]
    out = {
        "scheduler_json": str(args.scheduler_json),
        "candidate_rows": int(len(candidates)),
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
        "target_components_used",
        "tracklet_pair_f1",
        "tracklet_pair_precision",
        "tracklet_pair_recall",
        "learned_proxy_full_idf1",
        "accepted_pair_mass_proxy_sum",
        "portfolio_source_ranks",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Hub-Bridge Portfolio Candidates",
        "",
        f"- source scheduler: `{out['scheduler_json']}`",
        f"- candidate rows: `{out['candidate_rows']}`",
        f"- portfolio rows: `{len(out['rows'])}`",
        "",
        "| rank | predicted full | moved | edits | targets | source ranks | pair-mass proxy |",
        "| ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for rank, row in enumerate(out["rows"][:30], start=1):
        lines.append(
            f"| {rank} | `{row['learned_proxy_full_idf1']:.6f}` | `{row['moved_tracklets']}` | "
            f"`{row['accepted_reassignments']}` | `{_sanitize(row['target_component'])}` | "
            f"`{row['portfolio_source_ranks']}` | `{row['accepted_pair_mass_proxy_sum']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Portfolios are alternatives: each row is one assignment candidate.",
            "- Production selection is no-anchor; source full-score labels are not used.",
            "- Compatibility requires no source-seq overlap and no chained source/target component edits.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        sched = root / "sched.json"
        sched.write_text(
            json.dumps(
                {
                    "selected": [
                        {
                            "mode": "a",
                            "predicted_full_idf1": 0.66,
                            "tracklet_pair_f1": 0.8,
                            "tracklet_pair_precision": 0.82,
                            "tracklet_pair_recall": 0.78,
                            "accepted_preview": [{"source_component_label": 1, "target_component": 9, "source_seqs": [1, 2], "source_size": 2, "target_size": 100}],
                        },
                        {
                            "mode": "b",
                            "predicted_full_idf1": 0.661,
                            "tracklet_pair_f1": 0.79,
                            "tracklet_pair_precision": 0.81,
                            "tracklet_pair_recall": 0.77,
                            "accepted_preview": [{"source_component_label": 2, "target_component": 10, "source_seqs": [3, 4], "source_size": 2, "target_size": 120}],
                        },
                        {
                            "mode": "chain",
                            "predicted_full_idf1": 0.662,
                            "tracklet_pair_f1": 0.79,
                            "tracklet_pair_precision": 0.81,
                            "tracklet_pair_recall": 0.77,
                            "accepted_preview": [{"source_component_label": 9, "target_component": 11, "source_seqs": [5], "source_size": 1, "target_size": 50}],
                        },
                        {
                            "mode": "duplicate_source_without_seqs",
                            "predicted_full_idf1": 0.663,
                            "tracklet_pair_f1": 0.79,
                            "tracklet_pair_precision": 0.81,
                            "tracklet_pair_recall": 0.77,
                            "accepted_preview": [{"source_component_label": 1, "target_component": 12, "source_size": 2, "target_size": 50}],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = compose(
            argparse.Namespace(
                scheduler_json=str(sched),
                ranks="",
                current_best_full_idf1=0.65524,
                min_items=2,
                max_items=3,
                min_moved_tracklets=1,
                top_n=10,
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
                md=str(root / "out.md"),
            )
        )
        rank_sets = {tuple(row["portfolio_source_ranks"]) for row in out["rows"]}
        assert (1, 3) not in rank_sets, out
        assert (1, 4) not in rank_sets, out
        assert (1, 2) in rank_sets and (2, 3) in rank_sets, out
        assert Path(root / "out.csv").read_text()
        assert "portfolio" in Path(root / "out.md").read_text().lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--scheduler-json", default="local_runs/no_anchor_fullscore_scheduler_next_queue_20260620.json")
    ap.add_argument("--ranks", default="")
    ap.add_argument("--current-best-full-idf1", type=float, default=0.655240)
    ap.add_argument("--min-items", type=int, default=2)
    ap.add_argument("--max-items", type=int, default=4)
    ap.add_argument("--min-moved-tracklets", type=int, default=24)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--json", default="local_runs/no_anchor_hub_bridge_portfolio_candidates_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_hub_bridge_portfolio_candidates_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_hub_bridge_portfolio_candidates_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    out = compose(args)
    print(json.dumps({"candidate_rows": out["candidate_rows"], "rows": len(out["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
