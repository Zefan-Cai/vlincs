#!/usr/bin/env python
"""Export filtered assignment CSVs for explicit no-anchor admission policies.

This is a materializer, not a scorer.  It keeps predicted global IDs fixed and
filters delivered tracklets using only tracklet metadata such as video and
geometry.  GT is not loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from kit.no_anchor_resolve_sweep import _connect, _load_tracklets, _output_keep_seqs
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import _connect, _load_tracklets, _output_keep_seqs


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "policy"


def _parse_video_float_map(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.rpartition(":")
        if not sep or not key:
            raise ValueError(f"bad video map entry {part!r}; expected video:value")
        out[key] = float(value)
    return out


def _parse_policy(text: str) -> tuple[str, dict[str, float]]:
    name, sep, body = str(text).partition("=")
    if not sep or not name.strip():
        raise ValueError(f"bad --policy {text!r}; expected name=video:min_area,...")
    return name.strip(), _parse_video_float_map(body)


def _load_assignment(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "seq" not in fields:
            raise ValueError(f"{path} is missing seq")
        if "video" not in fields:
            raise ValueError(f"{path} is missing video")
        for row in reader:
            rows.append(dict(row))
    return rows, fields


def _admission_args(args: argparse.Namespace, video_min_area: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(
        output_min_dets=int(args.output_min_dets),
        output_min_conf=float(args.output_min_conf),
        output_min_area=float(args.output_min_area),
        output_min_quality=float(args.output_min_quality),
        output_min_area_by_video=",".join(f"{video}:{area}" for video, area in sorted(video_min_area.items())),
        output_drop_area_quantile=0.0,
        output_drop_area_quantile_by_video="",
        output_drop_quality_quantile=0.0,
        output_drop_quality_quantile_by_video="",
        output_auto_anomaly_admission=False,
        output_auto_anomaly_metric="quality",
        output_auto_anomaly_quantile=0.75,
        output_auto_anomaly_area_ratio=0.60,
        output_auto_anomaly_quality_mad=1.0,
        output_auto_anomaly_min_video_tracklets=20,
        output_auto_anomaly_max_videos=3,
    )


def _write_filtered(path: Path, rows: list[dict[str, str]], fields: list[str], keep: set[int]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped_by_video: Counter[str] = Counter()
    kept_by_video: Counter[str] = Counter()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            seq = int(float(row["seq"]))
            video = str(row.get("video") or "")
            if seq not in keep:
                dropped_by_video[video] += 1
                continue
            writer.writerow(row)
            kept += 1
            kept_by_video[video] += 1
    return {
        "assignment_csv": str(path),
        "assignment_rows": int(kept),
        "dropped_assignment_rows": int(len(rows) - kept),
        "kept_assignment_by_video": dict(sorted(kept_by_video.items())),
        "dropped_assignment_by_video": dict(sorted(dropped_by_video.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--policy", action="append", required=True, help="name=video:min_area,video2:min_area")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    args = ap.parse_args()

    assignment_rows, fields = _load_assignment(Path(args.assignment_csv))
    con = _connect(args.dbname)
    records, _emb = _load_tracklets(con, args.role)
    out_dir = Path(args.output_dir)
    outputs = []
    for policy_text in args.policy:
        name, video_min_area = _parse_policy(policy_text)
        keep, output_info = _output_keep_seqs(records, _admission_args(args, video_min_area))
        assignment_keep = {int(float(row["seq"])) for row in assignment_rows if int(float(row["seq"])) in keep}
        csv_path = out_dir / f"{_safe_name(name)}_assignments.csv"
        info = _write_filtered(csv_path, assignment_rows, fields, assignment_keep)
        outputs.append(
            {
                "policy_name": name,
                "video_min_area": {video: float(value) for video, value in sorted(video_min_area.items())},
                "output_info": output_info,
                **info,
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
                "uses_gt_for_evaluation_only": False,
            }
        )

    payload = {
        "assignment_csv": args.assignment_csv,
        "outputs": outputs,
        "output_dir": str(out_dir),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "note": "Explicit admission policies use only no-anchor tracklet metadata and preserve predicted_global_id values.",
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out), "outputs": len(outputs)}, sort_keys=True))


if __name__ == "__main__":
    main()
