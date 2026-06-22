#!/usr/bin/env python
"""Build verified side-effect labels for no-anchor attach candidates.

The labels are derived only from completed canonical metric JSON files.  They
are scheduler training data, not assignment evidence: no anchors or GT labels
are written into production assignments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("selected")
    if not isinstance(rows, list):
        rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has neither selected[] nor rows[]")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _metric_row(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data.get("rows"), list) and data["rows"]:
        return dict(data["rows"][0])
    if isinstance(data.get("best"), dict):
        return dict(data["best"])
    if isinstance(data.get("primary"), dict):
        return dict(data["primary"])
    return dict(data)


def _preview_signature(row: dict[str, Any]) -> str:
    sig = row.get("signature")
    if isinstance(sig, str) and sig:
        return sig
    preview = row.get("accepted_preview")
    if not isinstance(preview, list):
        return ""
    parts = []
    for item in preview:
        if not isinstance(item, dict):
            continue
        seqs = item.get("source_seqs")
        target = item.get("target_component")
        if not isinstance(seqs, list) or not seqs or target in (None, ""):
            continue
        seq_text = "+".join(str(int(float(seq))) for seq in sorted(seqs, key=lambda value: int(float(value))))
        parts.append(f"{seq_text}->{int(float(target))}")
    return "|".join(sorted(parts))


def _parse_rank_metric(text: str) -> tuple[int, Path]:
    rank_text, sep, path_text = text.partition(":")
    if not sep:
        raise ValueError(f"bad --rank-metric {text!r}; expected rank:path")
    return int(rank_text), Path(path_text)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _video_deltas(metric: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    per_video = metric.get("per_video")
    base_per_video = base.get("per_video")
    if not isinstance(per_video, dict) or not isinstance(base_per_video, dict):
        return out
    for video, values in per_video.items():
        if not isinstance(values, dict):
            continue
        base_values = base_per_video.get(video)
        if not isinstance(base_values, dict):
            continue
        out[str(video)] = round(_as_float(values.get("idf1")) - _as_float(base_values.get("idf1")), 9)
    return out


def build_labels(
    *,
    candidate_json: Path,
    base_metric_json: Path,
    rank_metrics: list[tuple[int, Path]],
    run_name: str,
) -> list[dict[str, Any]]:
    candidates = _load_rows(candidate_json)
    base_metric = _metric_row(base_metric_json)
    out = []
    for rank, metric_path in rank_metrics:
        if rank < 1 or rank > len(candidates):
            raise ValueError(f"{candidate_json} has {len(candidates)} candidates, rank {rank} is invalid")
        candidate = dict(candidates[rank - 1])
        metric = _metric_row(metric_path)
        signature = _preview_signature(candidate)
        row = dict(candidate)
        row.update(
            {
                "run_name": run_name,
                "verified_rank": int(rank),
                "signature": signature,
                "metric_json": str(metric_path),
                "base_metric_json": str(base_metric_json),
                "base_idf1": round(_as_float(base_metric.get("idf1")), 9),
                "verified_idf1": round(_as_float(metric.get("idf1")), 9),
                "delta_idf1": round(_as_float(metric.get("idf1")) - _as_float(base_metric.get("idf1")), 9),
                "verified_hota": round(_as_float(metric.get("hota")), 9),
                "delta_hota": round(_as_float(metric.get("hota")) - _as_float(base_metric.get("hota")), 9),
                "verified_assa": round(_as_float(metric.get("assa")), 9),
                "delta_assa": round(_as_float(metric.get("assa")) - _as_float(base_metric.get("assa")), 9),
                "per_video_delta_idf1": _video_deltas(metric, base_metric),
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
                "uses_gt_for_evaluation_only": True,
            }
        )
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-json", required=True)
    ap.add_argument("--base-metric-json", required=True)
    ap.add_argument("--rank-metric", action="append", required=True, help="rank:metric_json")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    labels = build_labels(
        candidate_json=Path(args.candidate_json),
        base_metric_json=Path(args.base_metric_json),
        rank_metrics=[_parse_rank_metric(text) for text in args.rank_metric],
        run_name=args.run_name or Path(args.candidate_json).parent.name,
    )
    payload = {
        "candidate_json": args.candidate_json,
        "base_metric_json": args.base_metric_json,
        "labels": labels,
        "label_count": len(labels),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "note": "Metric-derived labels are for scheduler calibration only.",
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        fieldnames = [
            "run_name",
            "verified_rank",
            "signature",
            "delta_idf1",
            "delta_hota",
            "delta_assa",
            "verified_idf1",
            "source_component_label",
            "target_component",
            "source_count",
            "candidate_rank",
            "scheduler_score",
            "critic_score",
            "weak_video_source_fraction",
            "target_same_video_fraction",
            "source_same_video_fraction",
            "target_gap_min",
            "source_terminal_fraction",
        ]
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in labels:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(json.dumps({"json": str(out_path), "label_count": len(labels)}, sort_keys=True))


if __name__ == "__main__":
    main()

