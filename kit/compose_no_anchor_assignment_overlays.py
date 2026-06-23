#!/usr/bin/env python3
"""Compose explicit no-anchor assignment overlays onto a base assignment CSV.

Each candidate assignment CSV must already be materialized from production-side
evidence. This utility compares each candidate against the same base assignment,
extracts only rows whose predicted identity/component changed, checks that no
two candidates edit the same seq differently, and writes the composed CSV.

It does not read anchors, GT labels, or evaluation files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


IDENTITY_COLS = ["predicted_global_id", "component_label", "decision_status"]


def _row_changes(base: pd.DataFrame, candidate: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    merged = base[["seq", *IDENTITY_COLS]].merge(
        candidate[["seq", *IDENTITY_COLS]],
        on="seq",
        suffixes=("_base", "_cand"),
        validate="one_to_one",
    )
    mask = False
    for col in IDENTITY_COLS:
        mask = mask | (merged[f"{col}_base"].astype(str) != merged[f"{col}_cand"].astype(str))
    changed = merged[mask]
    changes: list[dict[str, Any]] = []
    for seq in changed["seq"].astype(int).tolist():
        base_row = base[base.seq.eq(seq)].iloc[0].to_dict()
        cand_row = candidate[candidate.seq.eq(seq)].iloc[0].to_dict()
        changes.append(
            {
                "label": label,
                "seq": int(seq),
                "tracklet_key": str(cand_row.get("tracklet_key", "")),
                "before_predicted_global_id": int(base_row["predicted_global_id"]),
                "after_predicted_global_id": int(cand_row["predicted_global_id"]),
                "before_component_label": int(base_row["component_label"]),
                "after_component_label": int(cand_row["component_label"]),
                "before_decision_status": str(base_row["decision_status"]),
                "after_decision_status": str(cand_row["decision_status"]),
            }
        )
    return changes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-assignment-csv", required=True, type=Path)
    ap.add_argument("--candidate-assignment-csv", action="append", required=True, type=Path)
    ap.add_argument("--assignment-out", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args()

    base = pd.read_csv(args.base_assignment_csv)
    combo = base.copy()
    candidate_paths = [Path(p) for p in args.candidate_assignment_csv]
    if len(candidate_paths) < 2:
        raise SystemExit("provide at least two --candidate-assignment-csv values")

    overlay_by_seq: dict[int, dict[str, Any]] = {}
    changes: list[dict[str, Any]] = []
    for path in candidate_paths:
        candidate = pd.read_csv(path)
        if len(candidate) != len(base):
            raise SystemExit(f"{path} row count {len(candidate)} != base row count {len(base)}")
        if list(candidate.columns) != list(base.columns):
            raise SystemExit(f"{path} columns differ from base")
        label = path.stem
        for change in _row_changes(base, candidate, label):
            seq = int(change["seq"])
            cand_row = candidate[candidate.seq.eq(seq)].iloc[0].to_dict()
            if seq in overlay_by_seq:
                previous = overlay_by_seq[seq]
                same_identity = all(str(previous[col]) == str(cand_row[col]) for col in IDENTITY_COLS)
                if not same_identity:
                    raise SystemExit(
                        f"conflicting overlay for seq {seq}: {previous['predicted_global_id']}/"
                        f"{previous['component_label']} vs {cand_row['predicted_global_id']}/"
                        f"{cand_row['component_label']}"
                    )
            overlay_by_seq[seq] = cand_row
            changes.append(change)

    for seq, row in overlay_by_seq.items():
        combo.loc[combo.seq.eq(seq), base.columns] = [row[col] for col in base.columns]

    args.assignment_out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    combo.to_csv(args.assignment_out, index=False)
    report = {
        "base_assignment_csv": str(args.base_assignment_csv),
        "candidate_assignment_csvs": [str(p) for p in candidate_paths],
        "assignment_csv": str(args.assignment_out),
        "moved_tracklets": len(overlay_by_seq),
        "changes": changes,
        "uses_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "uses_gt_for_training_or_anchors": False,
        "note": "Composes explicit candidate assignment overlays only; no GT/anchor evidence is read.",
    }
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"assignment_csv": str(args.assignment_out), "moved_tracklets": len(overlay_by_seq)}, sort_keys=True))


if __name__ == "__main__":
    main()
