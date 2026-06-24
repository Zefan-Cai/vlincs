#!/usr/bin/env python3
"""Materialize selected no-anchor repair ranks from a proposer summary.

The input summary is produced from production-side evidence only.  This script
does not read GT, anchors, scores, or evaluator outputs.  It applies selected
ranked local repairs to a base assignment CSV and writes an audit record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_groups(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    groups = data.get("top_groups")
    if not isinstance(groups, list):
        raise SystemExit(f"{path} does not contain a top_groups list")
    out: dict[int, dict[str, Any]] = {}
    for group in groups:
        rank = int(group["rank"])
        if rank in out:
            raise SystemExit(f"duplicate rank {rank} in {path}")
        out[rank] = group
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-assignment-csv", required=True, type=Path)
    ap.add_argument("--summary-json", required=True, type=Path)
    ap.add_argument("--rank", action="append", required=True, type=int)
    ap.add_argument("--assignment-out", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--decision-status", default="no_anchor_ranked_repair")
    ap.add_argument(
        "--skip-already-target",
        action="store_true",
        help="Skip rows already carrying the selected target identity/component.",
    )
    ap.add_argument(
        "--recompute-component-size",
        action="store_true",
        help="Recompute component_size globally after all selected repairs.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.base_assignment_csv).reset_index(drop=True)
    required = {"seq", "tracklet_key", "predicted_global_id", "component_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"assignment missing columns: {missing}")

    groups = _load_groups(args.summary_json)
    by_seq = {int(seq): i for i, seq in enumerate(df["seq"].astype(int).tolist())}
    changes: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    edited: set[int] = set()

    for rank in args.rank:
        group = groups.get(int(rank))
        if group is None:
            raise SystemExit(f"rank {rank} missing from {args.summary_json}")
        target_gid = int(group["target_gid"])
        target_label = int(group["target_label"])
        source_gid = int(group["source_gid"])
        source_label = int(group["source_label"])
        moved = []
        moved_indices = []
        skipped = []
        for seq in [int(x) for x in group["seqs"]]:
            if seq not in by_seq:
                raise SystemExit(f"seq {seq} from rank {rank} missing in base assignment")
            row_i = by_seq[seq]
            before_gid = int(df.at[row_i, "predicted_global_id"])
            before_label = int(df.at[row_i, "component_label"])
            before_status = str(df.at[row_i, "decision_status"]) if "decision_status" in df.columns else ""
            if before_gid == target_gid and before_label == target_label:
                if args.skip_already_target:
                    skipped.append(seq)
                    continue
                raise SystemExit(f"seq {seq} already has target {target_gid}/{target_label}")
            if before_gid != source_gid or before_label != source_label:
                raise SystemExit(
                    f"seq {seq} has {before_gid}/{before_label}; expected source {source_gid}/{source_label}"
                )
            if seq in edited:
                raise SystemExit(f"seq {seq} selected by multiple ranks")
            df.at[row_i, "predicted_global_id"] = target_gid
            df.at[row_i, "component_label"] = target_label
            if "decision_status" in df.columns:
                df.at[row_i, "decision_status"] = args.decision_status
            edited.add(seq)
            moved.append(seq)
            moved_indices.append(row_i)
            changes.append(
                {
                    "rank": int(rank),
                    "seq": int(seq),
                    "tracklet_key": str(df.at[row_i, "tracklet_key"]),
                    "before_predicted_global_id": before_gid,
                    "after_predicted_global_id": target_gid,
                    "before_component_label": before_label,
                    "after_component_label": target_label,
                    "before_decision_status": before_status,
                    "after_decision_status": args.decision_status,
                }
            )
        if moved_indices and "component_size" in df.columns and not args.recompute_component_size:
            target_size = int((df["component_label"].astype(int) == target_label).sum())
            for row_i in moved_indices:
                df.at[row_i, "component_size"] = target_size
        selected.append(
            {
                "rank": int(rank),
                "source_gid": source_gid,
                "target_gid": target_gid,
                "source_label": source_label,
                "target_label": target_label,
                "original_seqs": [int(x) for x in group["seqs"]],
                "moved_seqs": moved,
                "skipped_already_target_seqs": skipped,
                "mean_score": group.get("mean_score"),
                "mean_centroid_margin": group.get("mean_centroid_margin"),
                "mean_neighbor_margin": group.get("mean_neighbor_margin"),
            }
        )

    if args.recompute_component_size and "component_size" in df.columns:
        sizes = df["component_label"].astype(int).map(df["component_label"].astype(int).value_counts())
        df["component_size"] = sizes.astype(int)

    args.assignment_out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.assignment_out, index=False)
    report = {
        "base_assignment_csv": str(args.base_assignment_csv),
        "summary_json": str(args.summary_json),
        "assignment_csv": str(args.assignment_out),
        "assignment_sha256": _sha256(args.assignment_out),
        "selected_ranks": [int(x) for x in args.rank],
        "selected": selected,
        "changes": changes,
        "moved_tracklets": len(changes),
        "uses_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "uses_gt_for_training_or_anchors": False,
    }
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "assignment_csv": str(args.assignment_out),
                "assignment_sha256": report["assignment_sha256"],
                "moved_tracklets": len(changes),
                "selected_ranks": report["selected_ranks"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
