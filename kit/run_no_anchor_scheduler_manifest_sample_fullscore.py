#!/usr/bin/env python
"""Run scheduler-manifest candidates through the parquet/sample full-score path.

This is the local counterpart of ``run_no_anchor_scheduler_manifest_fullscore.sh``.
It does not need PostgreSQL: selected scheduler rows are materialized as
tracklet-level assignment CSVs, then merged with detection-level tracklet
parquets.  If DS1 GT is mounted, it also runs the canonical scorer; otherwise
it can still export submission zips for later scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.evaluate_sample_assignments_full import (  # noqa: E402
    _build_comp,
    _export_zip,
    _load_assignments,
    _load_parquets,
    _merge_predictions,
    _metric_dict,
)
from kit.export_no_anchor_scheduler_manifest_assignments import export_manifest_assignments  # noqa: E402
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video  # noqa: E402


def _default_tracklet_parquets() -> list[str]:
    return [str(path) for path in sorted((REPO_ROOT / "kit/demo_data/ds1/tracklets").glob("*/tracklets.parquet"))]


def _status_counts(assignments: pd.DataFrame) -> dict[str, int]:
    series = assignments.get("resolution_status", assignments.get("decision_status", pd.Series(dtype=str)))
    return dict(Counter(series.astype(str)))


def _score_or_export(
    *,
    tracklets: pd.DataFrame,
    assignment_csv: str,
    fallback: str,
    json_out: Path,
    zip_out: Path | None,
    allow_no_gt_export: bool,
) -> dict[str, Any]:
    assignments = _load_assignments(assignment_csv)
    work, merge_info = _merge_predictions(tracklets, assignments, fallback)
    comp = _build_comp(work)
    gt_by_video = load_ds1_gt_by_video()
    keys = sorted(set(gt_by_video).intersection(comp))
    out: dict[str, Any]

    if keys:
        metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
        out = {
            **merge_info,
            **_metric_dict(metrics),
            "videos_scored": keys,
            "gt_available": True,
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": True,
            "assignment_status_counts": _status_counts(assignments),
        }
    else:
        if not allow_no_gt_export:
            raise RuntimeError("no overlap between predictions and local DS1 ground truth")
        out = {
            **merge_info,
            "videos_scored": [],
            "gt_available": False,
            "gt_message": "no overlap between predictions and local DS1 ground truth",
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": False,
            "assignment_status_counts": _status_counts(assignments),
        }

    if zip_out is not None:
        out.update(_export_zip(comp, str(zip_out)))
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run_manifest_sample_fullscore(
    *,
    scheduler_json: Path,
    base_assignment_csv: Path,
    run_dir: Path,
    selection_ranks: str,
    tracklet_parquets: list[str],
    fallback: str = "singleton",
    allow_no_gt_export: bool = False,
    write_zip: bool = True,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    assignment_dir = run_dir / "assignments"
    manifest_json = run_dir / "manifest_assignments.json"
    manifest = export_manifest_assignments(
        scheduler_json=scheduler_json,
        base_assignment_csv=base_assignment_csv,
        assignment_out_dir=assignment_dir,
        manifest_json=manifest_json,
        selection_ranks=selection_ranks,
    )
    tracklets = _load_parquets(tracklet_parquets)
    outputs = []
    results_jsonl = run_dir / "sample_full_results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as handle:
        for item in manifest["outputs"]:
            assignment_csv = str(item["output_csv"])
            stem = Path(assignment_csv).stem
            json_out = run_dir / f"{stem}_sample_full.json"
            zip_out = (run_dir / f"{stem}.zip") if write_zip else None
            result = _score_or_export(
                tracklets=tracklets,
                assignment_csv=assignment_csv,
                fallback=fallback,
                json_out=json_out,
                zip_out=zip_out,
                allow_no_gt_export=allow_no_gt_export,
            )
            row = {**item, "result_json": str(json_out), **result}
            outputs.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    scored = [row for row in outputs if row.get("gt_available")]
    summary = {
        "scheduler_json": str(scheduler_json),
        "base_assignment_csv": str(base_assignment_csv),
        "run_dir": str(run_dir),
        "manifest_json": str(manifest_json),
        "results_jsonl": str(results_jsonl),
        "selection_ranks": manifest.get("selection_ranks", []),
        "outputs": outputs,
        "gt_available": bool(scored),
        "best_idf1": max((float(row["idf1"]) for row in scored), default=None),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": bool(scored),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tracklets = root / "tracklets.parquet"
        pd.DataFrame(
            [
                {
                    "video_key": "video_a",
                    "tracklet_key": "video_a:0:0",
                    "frame_idx": 1,
                    "x1": 1,
                    "y1": 2,
                    "x2": 8,
                    "y2": 12,
                    "score": 0.9,
                    "coco_cls": 0,
                },
                {
                    "video_key": "video_a",
                    "tracklet_key": "video_a:1:0",
                    "frame_idx": 2,
                    "x1": 2,
                    "y1": 3,
                    "x2": 9,
                    "y2": 13,
                    "score": 0.8,
                    "coco_cls": 0,
                },
            ]
        ).to_parquet(tracklets)
        base = root / "base.csv"
        base.write_text(
            "\n".join(
                [
                    "seq,tracklet_key,component_label,component_size,predicted_global_id,prediction_confidence,decision_status",
                    "1,video_a:0:0,0,1,900,0.3,forced_singleton",
                    "2,video_a:1:0,1,1,901,0.3,forced_singleton",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        scheduler = root / "scheduler.json"
        scheduler.write_text(
            json.dumps(
                {
                    "selected": [
                        {
                            "mode": "unit",
                            "accepted_preview": [{"source_seqs": [2], "target_component": 0}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = run_manifest_sample_fullscore(
            scheduler_json=scheduler,
            base_assignment_csv=base,
            run_dir=root / "run",
            selection_ranks="1",
            tracklet_parquets=[str(tracklets)],
            allow_no_gt_export=True,
        )
        assert out["outputs"][0]["gt_available"] is False
        assert Path(out["outputs"][0]["zip_out"]).exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheduler-json", default="")
    ap.add_argument("--base-assignment-csv", default="")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--selection-ranks", "--ranks", default="")
    ap.add_argument("--tracklet-parquet", nargs="*", default=None)
    ap.add_argument("--fallback", choices=["drop", "singleton"], default="singleton")
    ap.add_argument("--allow-no-gt-export", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        print("self-test passed")
        return

    missing = [
        name
        for name in ("scheduler_json", "base_assignment_csv", "run_dir")
        if not getattr(args, name)
    ]
    if missing:
        ap.error(f"missing required argument(s): {', '.join('--' + key.replace('_', '-') for key in missing)}")

    tracklet_parquets = args.tracklet_parquet if args.tracklet_parquet else _default_tracklet_parquets()
    if not tracklet_parquets:
        ap.error("no --tracklet-parquet files provided and default kit/demo_data/ds1 tracklets are absent")

    out = run_manifest_sample_fullscore(
        scheduler_json=Path(args.scheduler_json),
        base_assignment_csv=Path(args.base_assignment_csv),
        run_dir=Path(args.run_dir),
        selection_ranks=str(args.selection_ranks),
        tracklet_parquets=[str(path) for path in tracklet_parquets],
        fallback=str(args.fallback),
        allow_no_gt_export=bool(args.allow_no_gt_export),
        write_zip=not bool(args.no_zip),
    )
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
