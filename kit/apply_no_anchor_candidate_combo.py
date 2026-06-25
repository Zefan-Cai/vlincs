#!/usr/bin/env python3
"""Apply selected no-anchor proposer ranks as one assignment combo.

The proposer manifest is production-side evidence only.  This script materializes
chosen local repairs into a CSV for scoring or delivery validation; it does not
read GT, anchors, or evaluator outputs.
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


def _load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key in ("selected", "top_candidates", "rows"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict) and "rank" in item)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        rank = int(row["rank"])
        out.setdefault(rank, row)
    if not out:
        raise SystemExit(f"{path} has no ranked candidate rows")
    return out


def _parse_ranks(values: list[str]) -> list[int]:
    ranks: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                ranks.append(int(part))
    seen: set[int] = set()
    ordered = []
    for rank in ranks:
        if rank not in seen:
            ordered.append(rank)
            seen.add(rank)
    return ordered


def _intish(value: Any) -> int:
    return int(float(value))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-assignment-csv", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--rank", action="append", required=True, help="Rank to apply; may be repeated or comma-separated.")
    ap.add_argument("--assignment-out", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--decision-status", default="subpart_reassign_combo")
    ap.add_argument("--allow-source-mismatch", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.base_assignment_csv).reset_index(drop=True)
    required = {"seq", "predicted_global_id", "component_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"{args.base_assignment_csv} missing columns: {missing}")

    candidates = _load_manifest(args.manifest)
    ranks = _parse_ranks(list(args.rank))
    by_seq = {_intish(seq): i for i, seq in enumerate(df["seq"].tolist())}
    changed_seqs: set[int] = set()
    changes: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    for rank in ranks:
        cand = candidates.get(int(rank))
        if cand is None:
            raise SystemExit(f"rank {rank} missing from {args.manifest}")
        source_label = int(cand["source_component"])
        target_label = int(cand["target_component"])
        source_gid = int(cand["source_predicted_global_id"])
        target_gid = int(cand["target_predicted_global_id"])
        moved = []
        for seq in [_intish(x) for x in cand["source_seqs"]]:
            if seq in changed_seqs:
                raise SystemExit(f"seq {seq} appears in multiple selected ranks")
            if seq not in by_seq:
                raise SystemExit(f"seq {seq} from rank {rank} missing from base assignment")
            row_i = by_seq[seq]
            before_gid = _intish(df.at[row_i, "predicted_global_id"])
            before_label = _intish(df.at[row_i, "component_label"])
            before_status = str(df.at[row_i, "decision_status"]) if "decision_status" in df.columns else ""
            if (before_gid != source_gid or before_label != source_label) and not args.allow_source_mismatch:
                raise SystemExit(
                    f"rank {rank} seq {seq} has source {before_gid}/{before_label}; "
                    f"expected {source_gid}/{source_label}"
                )
            df.at[row_i, "predicted_global_id"] = target_gid
            df.at[row_i, "component_label"] = target_label
            if "decision_status" in df.columns:
                df.at[row_i, "decision_status"] = str(args.decision_status)
            if "prediction_confidence" in df.columns:
                df.at[row_i, "prediction_confidence"] = max(float(df.at[row_i, "prediction_confidence"]), 0.72)
            changed_seqs.add(seq)
            moved.append(seq)
            changes.append(
                {
                    "rank": int(rank),
                    "seq": int(seq),
                    "tracklet_key": str(df.at[row_i, "tracklet_key"]) if "tracklet_key" in df.columns else "",
                    "before_predicted_global_id": int(before_gid),
                    "after_predicted_global_id": int(target_gid),
                    "before_component_label": int(before_label),
                    "after_component_label": int(target_label),
                    "before_decision_status": before_status,
                    "after_decision_status": str(args.decision_status),
                }
            )
        selected.append(
            {
                "rank": int(rank),
                "source_component": source_label,
                "target_component": target_label,
                "source_predicted_global_id": source_gid,
                "target_predicted_global_id": target_gid,
                "moved_tracklets": int(len(moved)),
                "source_seqs": moved,
                "score": cand.get("score"),
                "target_sim": cand.get("target_sim"),
                "target_margin": cand.get("target_margin"),
                "source_rest_margin_mean": cand.get("source_rest_margin_mean"),
                "conflicts_to_rest": cand.get("conflicts_to_rest"),
            }
        )

    if "component_size" in df.columns:
        labels = df["component_label"].astype(int)
        sizes = labels.map(labels.value_counts()).astype(int)
        df["component_size"] = sizes

    args.assignment_out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.assignment_out, index=False)
    report = {
        "base_assignment_csv": str(args.base_assignment_csv),
        "base_assignment_sha256": _sha256(args.base_assignment_csv),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "assignment_csv": str(args.assignment_out),
        "assignment_sha256": _sha256(args.assignment_out),
        "selected_ranks": ranks,
        "selected": selected,
        "changes": changes,
        "moved_tracklets": int(len(changes)),
        "decision_status": str(args.decision_status),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "assignment_csv": report["assignment_csv"],
                "assignment_sha256": report["assignment_sha256"],
                "moved_tracklets": report["moved_tracklets"],
                "selected_ranks": ranks,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
