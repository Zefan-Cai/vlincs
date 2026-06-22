#!/usr/bin/env python
"""Compose no-anchor portfolios directly from accepted-preview evidence.

Unlike the hub-only composer, this proposer can combine compatible edits across
different target components.  It is meant for scarce full-score exploration of
the low-video bottleneck while Pluto is flaky.  Candidate assignments still come
only from no-anchor accepted_preview evidence.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.compose_no_anchor_hub_bridge_candidates import _as_float, _iter_preview_items, _source_size, _target_size
from kit.rerank_no_anchor_scheduler_by_video_focus import _load_seq_video, _pick_metric_row, _video_weights


def _seqs(item: dict[str, Any], key: str) -> list[int]:
    raw = item.get(key)
    if not isinstance(raw, list):
        return []
    out = []
    for value in raw:
        try:
            out.append(int(float(value)))
        except (TypeError, ValueError):
            continue
    return out


def _source_component(item: dict[str, Any]) -> str:
    return str(item.get("source_component_label", item.get("source", item.get("source_rep", ""))))


def _target_component(item: dict[str, Any]) -> str:
    return str(item.get("target_component", item.get("target", item.get("target_rep", ""))))


def _mean(items: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [_as_float(item.get(key)) for item in items]
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float(default)


def _min(items: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [_as_float(item.get(key)) for item in items]
    vals = [float(v) for v in vals if v is not None]
    return float(min(vals)) if vals else float(default)


def _item_video_counts(item: dict[str, Any], seq_video: dict[int, str]) -> tuple[Counter[str], Counter[str]]:
    source = Counter(seq_video.get(seq, "") for seq in _seqs(item, "source_seqs"))
    target_seqs = _seqs(item, "target_top_seqs") or _seqs(item, "target_indices")
    target = Counter(seq_video.get(seq, "") for seq in target_seqs)
    source.pop("", None)
    target.pop("", None)
    return source, target


def _focus_score(source: Counter[str], target: Counter[str], weights: dict[str, float]) -> float:
    return float(sum((source[v] + target[v]) * weights.get(v, 0.0) for v in set(source) | set(target)))


def _item_quality(item: dict[str, Any]) -> float:
    return float(
        0.30 * (_as_float(item.get("target_mean_sim"), 0.0) or 0.0)
        + 0.20 * (_as_float(item.get("target_best_sim"), 0.0) or 0.0)
        + 0.16 * (_as_float(item.get("target_view_vote"), 0.0) or 0.0)
        + 0.12 * min(_as_float(item.get("target_margin"), 0.0) or 0.0, 1.0)
        + 0.12 * (_as_float(item.get("source_quality"), 0.0) or 0.0)
        + 0.10 * min(math.log1p(max(_source_size(item), 1)) / math.log(64.0), 1.0)
    )


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    for item in items:
        key = (_source_component(item), _target_component(item), tuple(_seqs(item, "source_seqs")))
        old = best.get(key)
        if old is None or (float(item["_focus_score"]), float(item["_quality"])) > (
            float(old["_focus_score"]),
            float(old["_quality"]),
        ):
            best[key] = item
    return list(best.values())


def _compatible_add(selected: list[dict[str, Any]], item: dict[str, Any], used_seqs: set[int]) -> bool:
    seqs = set(_seqs(item, "source_seqs"))
    if seqs & used_seqs:
        return False
    source = _source_component(item)
    target = _target_component(item)
    selected_sources = {_source_component(x) for x in selected}
    selected_targets = {_target_component(x) for x in selected}
    if target in selected_sources or source in selected_targets:
        return False
    if source in selected_sources and target in selected_targets:
        return False
    return True


def _row_from_items(items: list[dict[str, Any]], current_best: float) -> dict[str, Any]:
    moved = sum(max(_source_size(item), len(_seqs(item, "source_seqs"))) for item in items)
    pair_mass = sum(max(_source_size(item), 1) * max(_target_size(item), 1) for item in items)
    bridge_mass = sum(math.sqrt(max(_source_size(item), 1) * max(_target_size(item), 1)) for item in items)
    focus = sum(float(item["_focus_score"]) for item in items)
    quality = _mean(items, "_quality")
    parent_f1 = _mean(items, "_parent_pair_f1")
    parent_p = _mean(items, "_parent_pair_precision")
    parent_r = _mean(items, "_parent_pair_recall")
    predicted = (
        float(current_best)
        + min(0.009, 0.0012 * math.log1p(max(focus, 0.0)))
        + min(0.003, 0.000025 * max(moved - 24, 0))
        + 0.0015 * max(quality - 0.70, 0.0)
        - 0.00035 * max(len(items) - 6, 0)
    )
    source_counts = Counter()
    target_counts = Counter()
    for item in items:
        source_counts.update(item["_source_video_counts"])
        target_counts.update(item["_target_video_counts"])
    preview = [{key: value for key, value in item.items() if not str(key).startswith("_")} for item in items]
    return {
        "mode": "video_focus_portfolio",
        "source_component_label": "+".join(_source_component(item) for item in items),
        "target_component": "+".join(sorted({_target_component(item) for item in items})),
        "accepted_preview": preview,
        "accepted_reassignments": int(len(items)),
        "moved_tracklets": int(moved),
        "target_components_used": int(len({_target_component(item) for item in items})),
        "tracklet_pair_f1": float(parent_f1),
        "tracklet_pair_precision": float(parent_p),
        "tracklet_pair_recall": float(parent_r),
        "learned_proxy_full_idf1": float(predicted),
        "full_side_effect_proxy": float(predicted),
        "video_focus_score": float(focus),
        "video_focus_source_counts": dict(sorted(source_counts.items())),
        "video_focus_target_counts": dict(sorted(target_counts.items())),
        "accepted_edges": int(len(items)),
        "accepted_score_mean": float(quality),
        "accepted_view_mean_sim_mean": _mean(items, "target_mean_sim"),
        "accepted_view_min_sim_mean": _mean(items, "target_min_view_sim"),
        "accepted_mass_proxy_sum": float(bridge_mass),
        "accepted_pair_mass_proxy_sum": float(pair_mass),
        "accepted_min_weight_sum": float(sum(max(_source_size(item), 1) for item in items)),
        "accepted_max_weight_sum": float(sum(max(_target_size(item), 1) for item in items)),
        "accepted_size_product_sum": float(pair_mass),
        "preview_mean_target_mean_sim": _mean(items, "target_mean_sim"),
        "preview_mean_target_best_sim": _mean(items, "target_best_sim"),
        "preview_mean_target_view_vote": _mean(items, "target_view_vote"),
        "preview_min_target_min_view_sim": _min(items, "target_min_view_sim"),
        "preview_mean_source_quality": _mean(items, "source_quality"),
        "preview_mean_source_score": _mean(items, "source_score"),
        "preview_mean_source_size": _mean(items, "source_size"),
        "signature": repr(
            (
                "video_focus_portfolio",
                tuple(sorted((tuple(_seqs(item, "source_seqs")), _target_component(item)) for item in items)),
            )
        ),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_candidate_prioritization": True,
        "uses_gt_for_evaluation_only": True,
    }


def compose(args: argparse.Namespace) -> dict[str, Any]:
    seq_video = _load_seq_video(Path(args.seq_video_csv))
    metric_row = _pick_metric_row(Path(args.metric_json), int(args.metric_row_index))
    weights = _video_weights(metric_row, float(args.target_idf1))
    items = _iter_preview_items(args.input_glob, args)
    enriched = []
    for item in items:
        source_counts, target_counts = _item_video_counts(item, seq_video)
        focus = _focus_score(source_counts, target_counts, weights)
        if focus < float(args.min_item_focus):
            continue
        item = dict(item)
        item["_source_video_counts"] = source_counts
        item["_target_video_counts"] = target_counts
        item["_focus_score"] = focus
        item["_quality"] = _item_quality(item)
        enriched.append(item)
    enriched = _dedupe_items(enriched)
    enriched.sort(
        key=lambda item: (float(item["_focus_score"]), float(item["_quality"]), max(_source_size(item), 1)),
        reverse=True,
    )
    seeds = enriched[: int(args.max_seed_items)]
    rows: list[dict[str, Any]] = []
    for start in range(len(seeds)):
        selected: list[dict[str, Any]] = []
        used_seqs: set[int] = set()
        for item in itertools.chain(seeds[start:], seeds[:start]):
            if not _compatible_add(selected, item, used_seqs):
                continue
            selected.append(item)
            used_seqs.update(_seqs(item, "source_seqs"))
            if len(selected) >= int(args.max_items):
                break
        if len(selected) < int(args.min_items):
            continue
        row = _row_from_items(selected, float(args.current_best_full_idf1))
        if int(row["moved_tracklets"]) < int(args.min_moved_tracklets):
            continue
        if float(row["video_focus_score"]) < float(args.min_portfolio_focus):
            continue
        rows.append(row)
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["signature"])
        old = deduped.get(key)
        if old is None or (
            float(row["video_focus_score"]),
            float(row["accepted_pair_mass_proxy_sum"]),
            float(row["learned_proxy_full_idf1"]),
        ) > (
            float(old["video_focus_score"]),
            float(old["accepted_pair_mass_proxy_sum"]),
            float(old["learned_proxy_full_idf1"]),
        ):
            deduped[key] = row
    rows = list(deduped.values())
    rows.sort(
        key=lambda row: (
            float(row["video_focus_score"]),
            float(row["learned_proxy_full_idf1"]),
            float(row["accepted_pair_mass_proxy_sum"]),
        ),
        reverse=True,
    )
    rows = rows[: int(args.top_n)]
    out = {
        "input_glob": args.input_glob,
        "raw_items": int(len(items)),
        "focus_items": int(len(enriched)),
        "video_focus_weights": weights,
        "rows": rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_candidate_prioritization": True,
        "uses_gt_for_evaluation_only": True,
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
        "video_focus_score",
        "learned_proxy_full_idf1",
        "moved_tracklets",
        "accepted_reassignments",
        "target_components_used",
        "tracklet_pair_f1",
        "tracklet_pair_precision",
        "tracklet_pair_recall",
        "accepted_pair_mass_proxy_sum",
        "video_focus_source_counts",
        "video_focus_target_counts",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["video_focus_source_counts"] = json.dumps(row.get("video_focus_source_counts", {}), sort_keys=True)
            out["video_focus_target_counts"] = json.dumps(row.get("video_focus_target_counts", {}), sort_keys=True)
            writer.writerow(out)


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Video-Focus Portfolio Candidates",
        "",
        f"- raw preview items: `{out['raw_items']}`",
        f"- focus preview items: `{out['focus_items']}`",
        f"- portfolio rows: `{len(out['rows'])}`",
        "",
        "## Selected Rows",
        "",
        "| rank | focus | predicted full | moved | edits | targets | pair F1 | source videos |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(out["rows"][:30], start=1):
        src = json.dumps(row.get("video_focus_source_counts", {}), sort_keys=True)
        lines.append(
            f"| {rank} | `{row['video_focus_score']:.3f}` | `{row['learned_proxy_full_idf1']:.6f}` | "
            f"`{row['moved_tracklets']}` | `{row['accepted_reassignments']}` | `{row['target_components_used']}` | "
            f"`{row['tracklet_pair_f1']:.6f}` | `{src}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Candidate edits come from no-anchor accepted_preview evidence.",
            "- Prior per-video full-score metrics are used only to prioritize research budget.",
            "- These rows require canonical full-score verification before any e2e claim.",
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
                            "mode": "conflict",
                            "tracklet_pair_f1": 0.76,
                            "tracklet_pair_precision": 0.8,
                            "tracklet_pair_recall": 0.72,
                            "accepted_preview": [
                                {"source_component_label": 1, "target_component": 9, "source_seqs": [1, 2], "source_size": 2, "target_size": 50, "target_mean_sim": 0.8, "target_best_sim": 0.9, "target_view_vote": 0.75},
                                {"source_component_label": 2, "target_component": 10, "source_seqs": [3, 4], "source_size": 2, "target_size": 60, "target_mean_sim": 0.8, "target_best_sim": 0.9, "target_view_vote": 0.75},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        seq_csv = root / "seq.csv"
        seq_csv.write_text("seq,video\n1,V\n2,V\n3,V\n4,V\n", encoding="utf-8")
        metric = root / "metric.json"
        metric.write_text(json.dumps({"rows": [{"per_video": {"V": {"idf1": 0.5}}, "by_video_rows": {"V": {"input_rows": 100}}}]}), encoding="utf-8")
        args = argparse.Namespace(
            input_glob=[str(artifact)],
            mode_contains="conflict",
            min_source_pair_f1=0.7,
            min_source_pair_precision=0.7,
            min_source_pair_recall=0.7,
            min_source_size=1,
            min_target_size=8,
            min_target_mean_sim=0.7,
            min_target_view_vote=0.5,
            seq_video_csv=str(seq_csv),
            metric_json=str(metric),
            metric_row_index=0,
            target_idf1=0.7,
            current_best_full_idf1=0.65524,
            min_item_focus=0.0,
            min_portfolio_focus=0.1,
            min_items=2,
            max_items=3,
            min_moved_tracklets=4,
            max_seed_items=10,
            top_n=5,
            json=str(root / "out.json"),
            csv=str(root / "out.csv"),
            md=str(root / "out.md"),
        )
        out = compose(args)
        assert out["rows"], out
        assert out["rows"][0]["moved_tracklets"] == 4
        print(json.dumps({"stage": "self_test", "status": "ok"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--input-glob", action="append", default=["local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign*_20260620.json"])
    ap.add_argument("--mode-contains", default="conflict")
    ap.add_argument("--min-source-pair-f1", type=float, default=0.70)
    ap.add_argument("--min-source-pair-precision", type=float, default=0.70)
    ap.add_argument("--min-source-pair-recall", type=float, default=0.70)
    ap.add_argument("--min-source-size", type=int, default=2)
    ap.add_argument("--min-target-size", type=int, default=8)
    ap.add_argument("--min-target-mean-sim", type=float, default=0.68)
    ap.add_argument("--min-target-view-vote", type=float, default=0.50)
    ap.add_argument("--seq-video-csv", required=False, default="")
    ap.add_argument("--metric-json", required=False, default="")
    ap.add_argument("--metric-row-index", type=int, default=0)
    ap.add_argument("--target-idf1", type=float, default=0.70)
    ap.add_argument("--current-best-full-idf1", type=float, default=0.655240)
    ap.add_argument("--min-item-focus", type=float, default=0.05)
    ap.add_argument("--min-portfolio-focus", type=float, default=2.0)
    ap.add_argument("--min-items", type=int, default=2)
    ap.add_argument("--max-items", type=int, default=10)
    ap.add_argument("--min-moved-tracklets", type=int, default=24)
    ap.add_argument("--max-seed-items", type=int, default=80)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--json", default="local_runs/no_anchor_video_focus_portfolio_candidates_20260620.json")
    ap.add_argument("--csv", default="local_runs/no_anchor_video_focus_portfolio_candidates_20260620.csv")
    ap.add_argument("--md", default="reports/no_anchor_video_focus_portfolio_candidates_20260620.md")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.seq_video_csv or not args.metric_json:
        ap.error("--seq-video-csv and --metric-json are required unless --self-test is used")
    out = compose(args)
    print(json.dumps({"raw_items": out["raw_items"], "focus_items": out["focus_items"], "rows": len(out["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
