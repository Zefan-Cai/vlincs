#!/usr/bin/env python
"""Apply no-anchor component-bridge candidates as counterfactual assignments.

This is a production-side generator: it uses only predicted assignment state and
candidate metadata.  It does not read GT or anchors.  The generated assignment
CSVs can then be reviewed by no-GT/proxy referees or sent to canonical scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
    elif isinstance(data, dict):
        for key in ("selected", "rows", "all_admitted", "top", "top_candidates"):
            value = data.get(key)
            if isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
    return rows


def _preview(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("accepted_preview")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _edge_score(item: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    flags: list[str] = []
    risks: list[str] = []
    best = _as_float(item.get("target_best_sim"), 0.0) or 0.0
    mean = _as_float(item.get("target_mean_sim"), _as_float(item.get("view_mean_sim"), 0.0)) or 0.0
    min_view = _as_float(item.get("target_min_view_sim"), _as_float(item.get("view_min_sim"), 0.0)) or 0.0
    vote = _as_float(item.get("target_view_vote"), 0.0) or 0.0
    margin = _as_float(item.get("target_margin"), 0.0) or 0.0
    overlap = _as_float(item.get("same_video_overlap_ratio"), 0.0) or 0.0
    source_size = _as_float(item.get("source_size"), 0.0) or 0.0
    target_size = _as_float(item.get("target_size"), 0.0) or 0.0
    source_quality = _as_float(item.get("source_quality"), 0.0) or 0.0
    target_quality = _as_float(item.get("target_quality"), 0.0) or 0.0

    score = 0.42 * best + 0.25 * min_view + 0.18 * mean + 0.08 * vote
    score += 0.04 * min(max(margin, 0.0), 0.25)
    score += 0.04 * min(target_quality, 1.0)
    score -= 1.5 * max(overlap - 0.015, 0.0)
    score -= 0.00045 * max(source_size - 140.0, 0.0)
    score -= 0.00025 * max(source_size + target_size - 340.0, 0.0)
    if best >= 0.80:
        flags.append("strong_top_match")
    elif best >= 0.74:
        flags.append("medium_top_match")
    else:
        risks.append("weak_top_match")
        score -= 0.04
    if min_view >= 0.70:
        flags.append("multi_view_floor")
    else:
        risks.append("weak_min_view")
        score -= 0.03
    if vote < 0.5 and best - mean > 0.25:
        risks.append("spiky_low_vote")
        score -= 0.035
    if overlap > 0.02:
        risks.append("same_video_overlap")
        score -= 0.04
    if source_size > 180:
        risks.append("large_source_move")
    if target_quality + 0.12 < source_quality:
        risks.append("target_quality_lower_than_source")
        score -= 0.025
    if target_size >= source_size:
        flags.append("larger_target")
    return float(score), flags, risks


def _row_score(row: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    previews = _preview(row)
    if not previews:
        return -1.0, [], ["missing_preview"]
    scores = []
    flags: list[str] = []
    risks: list[str] = []
    for item in previews:
        score, f, r = _edge_score(item)
        scores.append(score)
        flags.extend(f)
        risks.extend(r)
    mean_score = sum(scores) / max(len(scores), 1)
    mean_score -= 0.025 * max(len(previews) - 1, 0)
    return float(mean_score), sorted(set(flags)), sorted(set(risks))


def _component_to_gid(path: Path) -> dict[int, int]:
    df = pd.read_csv(path, usecols=["component_label", "predicted_global_id"])
    out: dict[int, int] = {}
    for comp, gid in zip(df["component_label"], df["predicted_global_id"]):
        try:
            out[int(float(comp))] = int(float(gid))
        except (TypeError, ValueError):
            continue
    return out


def _apply_preview(base: pd.DataFrame, preview: list[dict[str, Any]], comp_to_gid: dict[int, int], status: str) -> tuple[pd.DataFrame, int]:
    df = base.copy()
    changed = 0
    for item in preview:
        target = int(float(item.get("target_component")))
        target_gid = int(comp_to_gid.get(target, 90_000_000 + target))
        seqs = [int(float(seq)) for seq in item.get("source_seqs", [])]
        if not seqs:
            continue
        mask = df["seq"].astype(int).isin(seqs)
        changed += int((df.loc[mask, "predicted_global_id"].astype(int) != target_gid).sum())
        df.loc[mask, "predicted_global_id"] = target_gid
        if "component_label" in df.columns:
            df.loc[mask, "component_label"] = target_gid
        if "decision_status" in df.columns:
            df.loc[mask, "decision_status"] = status
    if "component_size" in df.columns:
        df["component_size"] = df.groupby("predicted_global_id")["seq"].transform("count")
    return df, int(changed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = pd.read_csv(args.base_assignment_csv)
    comp_to_gid = _component_to_gid(Path(args.component_map_csv or args.base_assignment_csv))
    raw_rows = _load_rows(Path(args.candidates_json))
    scored = []
    for idx, row in enumerate(raw_rows, start=1):
        score, flags, risks = _row_score(row)
        preview = _preview(row)
        if score < float(args.min_adversary_score):
            continue
        if not preview:
            continue
        row = dict(row)
        row["_input_rank"] = idx
        row["adversary_score"] = score
        row["adversary_flags"] = flags
        row["adversary_risks"] = risks
        scored.append(row)
    scored.sort(
        key=lambda row: (
            float(row["adversary_score"]),
            _as_float(row.get("accepted_pair_mass_proxy_sum"), 0.0) or 0.0,
        ),
        reverse=True,
    )
    scored = scored[: int(args.top_n)]

    out_dir = Path(args.assignment_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for rank, row in enumerate(scored, start=1):
        preview = _preview(row)
        df, changed = _apply_preview(base, preview, comp_to_gid, args.status)
        sources = "+".join(str(item.get("source_component_label", "")) for item in preview)
        targets = "+".join(str(item.get("target_component", "")) for item in preview)
        path = out_dir / f"rank{rank:02d}_adversary_bridge_{sources}_to_{targets}_assignments.csv"
        df.to_csv(path, index=False)
        outputs.append(
            {
                "rank": rank,
                "assignment_csv": str(path),
                "changed_from_base_count": int(changed),
                "source_component_label": sources,
                "target_component": targets,
                "adversary_score": float(row["adversary_score"]),
                "adversary_flags": row["adversary_flags"],
                "adversary_risks": row["adversary_risks"],
                "input_rank": row.get("_input_rank"),
                "moved_tracklets": row.get("moved_tracklets"),
                "accepted_preview": preview,
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
                "uses_gt_for_evaluation_only": False,
            }
        )
    result = {
        "base_assignment_csv": args.base_assignment_csv,
        "component_map_csv": args.component_map_csv or args.base_assignment_csv,
        "candidates_json": args.candidates_json,
        "raw_rows": len(raw_rows),
        "output_rows": len(outputs),
        "rows": outputs,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "note": "Counterfactual bridge assignments generated from no-GT adversarial metadata only.",
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "rank",
                "assignment_csv",
                "changed_from_base_count",
                "source_component_label",
                "target_component",
                "adversary_score",
                "adversary_flags",
                "adversary_risks",
                "input_rank",
                "moved_tracklets",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in outputs:
                writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})
    return result


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = pd.DataFrame(
            [
                {"seq": 1, "predicted_global_id": 90000001, "component_label": 1, "component_size": 2, "decision_status": "base"},
                {"seq": 2, "predicted_global_id": 90000001, "component_label": 1, "component_size": 2, "decision_status": "base"},
                {"seq": 3, "predicted_global_id": 90000002, "component_label": 2, "component_size": 1, "decision_status": "base"},
            ]
        )
        base_csv = root / "base.csv"
        base.to_csv(base_csv, index=False)
        candidates = root / "candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "accepted_preview": [
                                {
                                    "source_component_label": 1,
                                    "target_component": 2,
                                    "source_seqs": [1, 2],
                                    "source_size": 2,
                                    "target_size": 1,
                                    "target_best_sim": 0.9,
                                    "target_mean_sim": 0.8,
                                    "target_min_view_sim": 0.75,
                                    "target_view_vote": 1.0,
                                    "same_video_overlap_ratio": 0.0,
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = run(
            argparse.Namespace(
                base_assignment_csv=str(base_csv),
                component_map_csv=str(base_csv),
                candidates_json=str(candidates),
                assignment_out_dir=str(root / "out"),
                status="counterfactual",
                min_adversary_score=0.0,
                top_n=5,
                json=str(root / "out.json"),
                csv=str(root / "out.csv"),
            )
        )
        assert out["output_rows"] == 1, out
        df = pd.read_csv(out["rows"][0]["assignment_csv"])
        assert int(df.loc[df["seq"] == 1, "predicted_global_id"].iloc[0]) == 90000002


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-assignment-csv", default="")
    ap.add_argument("--component-map-csv", default="")
    ap.add_argument("--candidates-json", default="")
    ap.add_argument("--assignment-out-dir", default="")
    ap.add_argument("--status", default="adversary_bridge_counterfactual")
    ap.add_argument("--min-adversary-score", type=float, default=0.0)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    missing = [
        name
        for name in ("base_assignment_csv", "candidates_json", "assignment_out_dir", "json")
        if not getattr(args, name)
    ]
    if missing:
        ap.error(f"missing required argument(s): {', '.join('--' + key.replace('_', '-') for key in missing)}")
    result = run(args)
    print(json.dumps({"raw_rows": result["raw_rows"], "output_rows": result["output_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
