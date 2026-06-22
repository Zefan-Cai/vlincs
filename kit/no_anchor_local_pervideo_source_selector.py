#!/usr/bin/env python
"""Local no-GT per-video assignment source selector.

This script is the filesystem-only counterpart of the DB-backed video/source
switchers.  It reads complete no-anchor assignment CSVs, aligns every source's
predicted ID namespace to a reference source by tracklet-overlap majority, then
chooses a source per video using assignment metadata only.

It never uses anchors, GT labels, or full-score feedback for selection.  The
output assignment CSV keeps a single global ID namespace, so it avoids the
whole-video submission zip switching failure mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_source(text: str) -> tuple[str, Path]:
    name, sep, path = str(text).partition(":")
    if not sep or not name or not path:
        raise ValueError(f"bad --source {text!r}; expected name:/path/to/assignments.csv")
    return name, Path(path)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _load_assignment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"seq", "tracklet_key", "video", "predicted_global_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing assignment columns: {missing}")
    df = df.copy()
    df["seq"] = df["seq"].astype(int)
    df["tracklet_key"] = df["tracklet_key"].astype(str)
    df["video"] = df["video"].astype(str)
    df["predicted_global_id"] = df["predicted_global_id"].astype(int)
    if "component_label" not in df.columns:
        df["component_label"] = df["predicted_global_id"]
    df["component_label"] = pd.to_numeric(df["component_label"], errors="coerce").fillna(df["predicted_global_id"]).astype(int)
    return df.drop_duplicates("seq", keep="first")


def _align_gid_maps(raw: dict[str, pd.DataFrame], reference: str) -> tuple[dict[str, dict[int, int]], dict[str, Any]]:
    if reference not in raw:
        raise ValueError(f"reference source {reference!r} is not present")
    ref_gid = dict(zip(raw[reference]["seq"].astype(int), raw[reference]["predicted_global_id"].astype(int)))
    next_gid = max(int(df["predicted_global_id"].max()) for df in raw.values()) + 10_000_000
    maps: dict[str, dict[int, int]] = {}
    stats: dict[str, Any] = {}
    for name, df in raw.items():
        if name == reference:
            gids = sorted(int(gid) for gid in df["predicted_global_id"].unique())
            maps[name] = {gid: gid for gid in gids}
            stats[name] = {
                "rows": int(len(df)),
                "mapped_components": int(len(gids)),
                "new_components": 0,
                "overlap_rows": int(len(df)),
            }
            continue
        votes: dict[int, Counter[int]] = defaultdict(Counter)
        for row in df[["seq", "predicted_global_id"]].itertuples(index=False):
            seq = int(row.seq)
            if seq in ref_gid:
                votes[int(row.predicted_global_id)][int(ref_gid[seq])] += 1
        mapping: dict[int, int] = {}
        mapped = 0
        for gid, counter in votes.items():
            if counter:
                mapping[int(gid)] = int(counter.most_common(1)[0][0])
                mapped += 1
        new_components = 0
        for gid in sorted(int(gid) for gid in df["predicted_global_id"].unique()):
            if gid not in mapping:
                mapping[gid] = next_gid
                next_gid += 1
                new_components += 1
        maps[name] = mapping
        stats[name] = {
            "rows": int(len(df)),
            "mapped_components": int(mapped),
            "new_components": int(new_components),
            "overlap_rows": int(df["seq"].isin(ref_gid).sum()),
        }
    return maps, stats


def _with_aligned_gid(df: pd.DataFrame, mapping: dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out["aligned_predicted_global_id"] = out["predicted_global_id"].map(mapping).fillna(out["predicted_global_id"]).astype(int)
    return out


def _median(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or len(df) == 0:
        return float(default)
    value = pd.to_numeric(df[col], errors="coerce").median()
    return float(value) if pd.notna(value) else float(default)


def _mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or len(df) == 0:
        return float(default)
    value = pd.to_numeric(df[col], errors="coerce").mean()
    return float(value) if pd.notna(value) else float(default)


def _video_stats(df: pd.DataFrame, base_df: pd.DataFrame | None = None) -> dict[str, Any]:
    rows = int(len(df))
    counts = df.groupby("aligned_predicted_global_id")["seq"].nunique() if rows else pd.Series(dtype=float)
    comp_count = int(len(counts))
    largest = int(counts.max()) if len(counts) else 0
    singleton_rows = int(counts[counts == 1].sum()) if len(counts) else 0
    status_counts = dict(Counter(df.get("decision_status", pd.Series(dtype=str)).astype(str)))
    changed_ratio = 0.0
    if base_df is not None and rows:
        base_gid = dict(zip(base_df["seq"].astype(int), base_df["aligned_predicted_global_id"].astype(int)))
        changed = sum(1 for row in df[["seq", "aligned_predicted_global_id"]].itertuples(index=False) if base_gid.get(int(row.seq)) != int(row.aligned_predicted_global_id))
        changed_ratio = float(changed / max(rows, 1))
    return {
        "rows": rows,
        "components": comp_count,
        "largest_component": largest,
        "largest_component_ratio": float(largest / max(rows, 1)),
        "median_component_size": float(counts.median()) if len(counts) else 0.0,
        "singleton_row_ratio": float(singleton_rows / max(rows, 1)),
        "avg_conf_mean": _mean(df, "avg_conf", 0.0),
        "prediction_confidence_mean": _mean(df, "prediction_confidence", 0.0),
        "component_margin_prob_median": _median(df, "component_margin_prob", 0.0),
        "component_external_prob_max_mean": _mean(df, "component_external_prob_max", 0.0),
        "component_internal_prob_median": _median(df, "component_internal_prob_median", 0.0),
        "component_internal_score_median": _median(df, "component_internal_score_median", 0.0),
        "changed_ratio_from_base": changed_ratio,
        "manifest_reassign_rows": int(status_counts.get("manifest_reassign", 0)),
        "manifest_reassign_ratio": float(status_counts.get("manifest_reassign", 0) / max(rows, 1)),
        "status_counts": status_counts,
    }


def _score_stats(stats: dict[str, Any], *, base_rows: int) -> float:
    coverage = min(float(stats["rows"]) / max(float(base_rows), 1.0), 1.0)
    margin = max(float(stats["component_margin_prob_median"]), 0.0)
    internal = max(float(stats["component_internal_prob_median"]), 0.0)
    internal_score = max(float(stats["component_internal_score_median"]), 0.0)
    avg_conf = min(max(float(stats["avg_conf_mean"]), 0.0), 1.0)
    pred_conf = min(max(float(stats["prediction_confidence_mean"]), 0.0), 1.0)
    external = max(float(stats["component_external_prob_max_mean"]), 0.0)
    largest = min(float(stats["largest_component_ratio"]) / 0.25, 1.0)
    singleton = min(float(stats["singleton_row_ratio"]) / 0.25, 1.0)
    changed = min(float(stats["changed_ratio_from_base"]) / 0.25, 1.0)
    manifest = min(float(stats["manifest_reassign_ratio"]) / 0.20, 1.0)
    score = (
        0.20 * coverage
        + 0.17 * avg_conf
        + 0.15 * pred_conf
        + 0.14 * min(margin, 1.0)
        + 0.10 * min(internal, 1.0)
        + 0.07 * min(internal_score, 1.0)
        + 0.05 * manifest
    )
    score -= 0.13 * min(external, 1.0)
    score -= 0.10 * largest
    score -= 0.06 * singleton
    score -= 0.08 * changed
    return float(score)


def _select_policy(
    stats_by_source: dict[str, dict[str, dict[str, Any]]],
    *,
    videos: list[str],
    base_source: str,
    strategy: str,
    min_score_gain: float,
    max_changed_ratio: float,
    min_coverage: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    policy: dict[str, str] = {}
    decisions: dict[str, Any] = {}
    for video in videos:
        base_stats = stats_by_source[base_source][video]
        base_score = float(base_stats["selector_score"])
        base_rows = max(int(base_stats["rows"]), 1)
        candidates = []
        for source, per_video in stats_by_source.items():
            s = per_video[video]
            coverage = float(s["rows"]) / max(float(base_rows), 1.0)
            eligible = (
                source == base_source
                or (
                    coverage >= min_coverage
                    and float(s["changed_ratio_from_base"]) <= max_changed_ratio
                    and float(s["selector_score"]) >= base_score + min_score_gain
                )
            )
            if strategy == "balanced" and source != base_source:
                eligible = eligible and float(s["largest_component_ratio"]) <= max(float(base_stats["largest_component_ratio"]) * 1.25, 0.08)
            candidates.append(
                {
                    "source": source,
                    "selector_score": float(s["selector_score"]),
                    "score_gain_vs_base": float(s["selector_score"] - base_score),
                    "coverage": coverage,
                    "changed_ratio_from_base": float(s["changed_ratio_from_base"]),
                    "largest_component_ratio": float(s["largest_component_ratio"]),
                    "components": int(s["components"]),
                    "eligible": bool(eligible),
                }
            )
        candidates.sort(key=lambda row: (row["eligible"], row["selector_score"], -row["changed_ratio_from_base"]), reverse=True)
        choice = candidates[0]["source"] if candidates and candidates[0]["eligible"] else base_source
        policy[video] = str(choice)
        decisions[video] = {
            "chosen_source": str(choice),
            "base_score": base_score,
            "candidates": candidates,
        }
    return policy, decisions


def _write_selected_assignment(
    path: Path,
    *,
    raw: dict[str, pd.DataFrame],
    aligned: dict[str, pd.DataFrame],
    base_source: str,
    policy: dict[str, str],
) -> dict[str, Any]:
    base_cols = list(raw[base_source].columns)
    extra_cols = [col for col in base_cols if col in raw[base_source].columns]
    rows: list[dict[str, Any]] = []
    by_source_seq = {
        source: {int(row.seq): row for row in df.itertuples(index=False)}
        for source, df in aligned.items()
    }
    base = aligned[base_source]
    source_counts: Counter[str] = Counter()
    for base_row in base.sort_values("seq").itertuples(index=False):
        video = str(base_row.video)
        source = policy.get(video, base_source)
        selected = by_source_seq.get(source, {}).get(int(base_row.seq), base_row)
        row = {col: getattr(selected, col) if hasattr(selected, col) else getattr(base_row, col) for col in extra_cols}
        row["predicted_global_id"] = int(getattr(selected, "aligned_predicted_global_id"))
        row["component_label"] = int(row["predicted_global_id"])
        row["decision_status"] = "pervideo_source_select" if source != base_source else str(row.get("decision_status", "forced_component"))
        row["prediction_confidence"] = row.get("prediction_confidence", getattr(selected, "prediction_confidence", 0.7))
        rows.append(row)
        source_counts[source] += 1

    counts = Counter(int(row["predicted_global_id"]) for row in rows)
    for row in rows:
        if "component_size" in row:
            row["component_size"] = int(counts[int(row["predicted_global_id"])])
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(extra_cols)
    for col in ("decision_status", "prediction_confidence", "component_size"):
        if any(col in row for row in rows) and col not in fieldnames:
            fieldnames.append(col)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "assignment_rows": int(len(rows)),
        "source_row_counts": dict(sorted(source_counts.items())),
        "predicted_ids": int(len(counts)),
    }


def _write_csv(path: Path, decisions: dict[str, Any]) -> None:
    fields = [
        "video",
        "chosen_source",
        "base_score",
        "best_score",
        "best_score_gain_vs_base",
        "best_changed_ratio_from_base",
        "best_coverage",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for video, decision in sorted(decisions.items()):
            best = decision["candidates"][0] if decision["candidates"] else {}
            writer.writerow(
                {
                    "video": video,
                    "chosen_source": decision["chosen_source"],
                    "base_score": decision["base_score"],
                    "best_score": best.get("selector_score"),
                    "best_score_gain_vs_base": best.get("score_gain_vs_base"),
                    "best_changed_ratio_from_base": best.get("changed_ratio_from_base"),
                    "best_coverage": best.get("coverage"),
                }
            )


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Local Per-Video Source Selector",
        "",
        "Production selector: no anchors, no GT, no full-score feedback.",
        "",
        f"- strategy: `{out['selector']['strategy']}`",
        f"- base source: `{out['base_source']}`",
        f"- reference source: `{out['reference_source']}`",
        f"- assignment rows: `{out.get('assignment_info', {}).get('assignment_rows', 0)}`",
        f"- predicted IDs: `{out.get('assignment_info', {}).get('predicted_ids', 0)}`",
        "",
        "## Policy",
        "",
        "| video | source | base score | best score | gain | changed |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for video, decision in sorted(out["decisions"].items()):
        best = decision["candidates"][0] if decision["candidates"] else {}
        lines.append(
            "| "
            f"`{video}` | `{decision['chosen_source']}` | {decision['base_score']:.6f} | "
            f"{float(best.get('selector_score', 0.0)):.6f} | {float(best.get('score_gain_vs_base', 0.0)):.6f} | "
            f"{float(best.get('changed_ratio_from_base', 0.0)):.6f} |"
        )
    lines.extend(["", "## Source Counts", ""])
    for source, count in sorted(out.get("assignment_info", {}).get("source_row_counts", {}).items()):
        lines.append(f"- `{source}`: `{count}` rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select(args: argparse.Namespace) -> dict[str, Any]:
    sources = dict(_parse_source(spec) for spec in args.source)
    if len(sources) != len(args.source):
        raise ValueError("duplicate source names")
    if args.base_source not in sources:
        raise ValueError("--base-source must match one --source name")
    if args.reference_source not in sources:
        raise ValueError("--reference-source must match one --source name")

    raw = {name: _load_assignment(path) for name, path in sources.items()}
    maps, align_stats = _align_gid_maps(raw, args.reference_source)
    aligned = {name: _with_aligned_gid(df, maps[name]) for name, df in raw.items()}
    videos = sorted(raw[args.base_source]["video"].astype(str).unique())

    stats_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for source, df in aligned.items():
        per_video: dict[str, dict[str, Any]] = {}
        for video in videos:
            g = df.loc[df["video"].astype(str) == video].copy()
            base_g = aligned[args.base_source].loc[aligned[args.base_source]["video"].astype(str) == video].copy()
            stats = _video_stats(g, base_g)
            stats["selector_score"] = _score_stats(stats, base_rows=max(int(len(base_g)), 1))
            per_video[video] = stats
        stats_by_source[source] = per_video

    policy, decisions = _select_policy(
        stats_by_source,
        videos=videos,
        base_source=args.base_source,
        strategy=args.strategy,
        min_score_gain=float(args.min_score_gain),
        max_changed_ratio=float(args.max_changed_ratio),
        min_coverage=float(args.min_coverage),
    )
    assignment_info: dict[str, Any] = {}
    if args.assignments_out:
        assignment_info = _write_selected_assignment(
            Path(args.assignments_out),
            raw=raw,
            aligned=aligned,
            base_source=args.base_source,
            policy=policy,
        )
    out = {
        "sources": {name: str(path) for name, path in sorted(sources.items())},
        "reference_source": args.reference_source,
        "base_source": args.base_source,
        "selector": {
            "name": "local_pervideo_source_selector",
            "strategy": args.strategy,
            "min_score_gain": float(args.min_score_gain),
            "max_changed_ratio": float(args.max_changed_ratio),
            "min_coverage": float(args.min_coverage),
            "uses_gt": False,
            "uses_full_score_feedback": False,
        },
        "alignment": align_stats,
        "videos": videos,
        "policy": policy,
        "decisions": decisions,
        "source_stats_per_video": stats_by_source,
        "assignment_info": assignment_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), decisions)
    if args.md:
        _write_md(Path(args.md), out)
    return out


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            {"seq": 1, "tracklet_key": "a", "video": "v1", "predicted_global_id": 10, "component_label": 10, "avg_conf": 0.9, "prediction_confidence": 0.7, "component_margin_prob": 0.1},
            {"seq": 2, "tracklet_key": "b", "video": "v1", "predicted_global_id": 11, "component_label": 11, "avg_conf": 0.8, "prediction_confidence": 0.7, "component_margin_prob": 0.1},
            {"seq": 3, "tracklet_key": "c", "video": "v2", "predicted_global_id": 12, "component_label": 12, "avg_conf": 0.7, "prediction_confidence": 0.7, "component_margin_prob": 0.1},
        ]
        base = root / "base.csv"
        cand = root / "cand.csv"
        pd.DataFrame(rows).to_csv(base, index=False)
        crows = [dict(row) for row in rows]
        crows[0]["predicted_global_id"] = 20
        crows[1]["predicted_global_id"] = 20
        crows[0]["component_margin_prob"] = 0.9
        crows[1]["component_margin_prob"] = 0.9
        pd.DataFrame(crows).to_csv(cand, index=False)
        out_csv = root / "out.csv"
        args = argparse.Namespace(
            source=[f"base:{base}", f"cand:{cand}"],
            reference_source="base",
            base_source="base",
            strategy="conservative",
            min_score_gain=0.01,
            max_changed_ratio=1.0,
            min_coverage=0.9,
            assignments_out=str(out_csv),
            json=str(root / "out.json"),
            csv=None,
            md=None,
        )
        result = select(args)
        assert result["policy"]["v1"] == "cand", result
        assert result["policy"]["v2"] == "base", result
        written = pd.read_csv(out_csv)
        assert "tracklet_key" in written.columns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", default=[], help="name:/path/to/assignments.csv")
    ap.add_argument("--reference-source", default="")
    ap.add_argument("--base-source", default="")
    ap.add_argument("--strategy", choices=["conservative", "balanced"], default="conservative")
    ap.add_argument("--min-score-gain", type=float, default=0.012)
    ap.add_argument("--max-changed-ratio", type=float, default=0.12)
    ap.add_argument("--min-coverage", type=float, default=0.98)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    missing = []
    if not args.source:
        missing.append("--source")
    if not args.reference_source:
        missing.append("--reference-source")
    if not args.base_source:
        missing.append("--base-source")
    if not args.json:
        missing.append("--json")
    if missing:
        ap.error("missing required argument(s): " + ", ".join(missing))
    result = select(args)
    print(json.dumps({"videos": len(result["videos"]), "policy": result["policy"], "assignment_info": result["assignment_info"]}, sort_keys=True))


if __name__ == "__main__":
    main()
