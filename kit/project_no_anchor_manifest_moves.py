#!/usr/bin/env python
"""Project no-anchor manifest moves onto a new assignment CSV.

Historical no-anchor candidates are often produced on an older base assignment.
This utility replays only their explicit moved tracklets onto a newer base, so
we can test whether a formerly useful local repair composes with the current
best namespace.  It does not use anchors or GT labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_ranks(text: str) -> set[int]:
    return {int(part) for part in str(text).split(",") if part.strip()}


def _load_selected(manifest_path: str, ranks: set[int]) -> list[dict[str, Any]]:
    data = json.loads(Path(manifest_path).read_text())
    selected = data.get("selected")
    if not isinstance(selected, list):
        raise ValueError(f"{manifest_path} has no selected list")
    if not ranks:
        return [item for item in selected if isinstance(item, dict)]
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(selected, start=1):
        if not isinstance(item, dict):
            continue
        item_rank = int(item.get("rank", idx) or idx)
        if idx in ranks or item_rank in ranks:
            out.append(item)
    if not out:
        raise ValueError(f"no selected manifest entries matched ranks={sorted(ranks)}")
    return out


def _moves_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("applied_moves", "moved_preview"):
        value = entry.get(key)
        if isinstance(value, list) and value:
            return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"selected entry has no applied_moves or moved_preview: {entry.get('assignment_csv')}")


def _status_from_name(manifest: str, entries: list[dict[str, Any]]) -> str:
    names = []
    for entry in entries:
        path = str(entry.get("assignment_csv", ""))
        names.append(Path(path).stem or f"rank{entry.get('rank', len(names) + 1)}")
    stem = Path(manifest).stem
    joined = "__".join(names)
    return f"projected_manifest_moves:{stem}:{joined}"[:240]


def _recompute_component_size(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sizes = out.groupby("predicted_global_id")["seq"].transform("nunique")
    out["component_size"] = sizes.astype(int)
    if "component_label" in out.columns:
        codes, _uniques = pd.factorize(out["predicted_global_id"].astype(int), sort=True)
        out["component_label"] = codes.astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-assignment-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--selected-ranks",
        default="1",
        help="comma-separated 1-based selected-list indices or selected rank values; empty means all",
    )
    parser.add_argument("--assignment-out", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.base_assignment_csv)
    if "seq" not in df.columns or "predicted_global_id" not in df.columns:
        raise ValueError("base assignment must contain seq and predicted_global_id")
    df["seq"] = df["seq"].astype(int)
    df["predicted_global_id"] = df["predicted_global_id"].astype(int)
    before = {int(row.seq): int(row.predicted_global_id) for row in df.itertuples(index=False)}

    entries = _load_selected(args.manifest, _parse_ranks(args.selected_ranks))
    seq_to_gid: dict[int, int] = {}
    raw_moves: list[dict[str, Any]] = []
    for entry in entries:
        for move in _moves_from_entry(entry):
            if "seq" not in move or "to_predicted_global_id" not in move:
                continue
            seq = int(move["seq"])
            target = int(move["to_predicted_global_id"])
            raw_moves.append(dict(move))
            seq_to_gid[seq] = target

    missing = sorted(seq for seq in seq_to_gid if seq not in before)
    effective = {
        seq: gid
        for seq, gid in sorted(seq_to_gid.items())
        if seq in before and int(before[seq]) != int(gid)
    }
    noops = {
        seq: gid
        for seq, gid in sorted(seq_to_gid.items())
        if seq in before and int(before[seq]) == int(gid)
    }

    out = df.copy()
    out["predicted_global_id"] = [
        int(effective.get(int(seq), int(cur)))
        for seq, cur in zip(out["seq"].tolist(), out["predicted_global_id"].tolist())
    ]
    if "decision_status" in out.columns:
        out["decision_status"] = _status_from_name(args.manifest, entries)
    out = _recompute_component_size(out)

    assignment_out = Path(args.assignment_out)
    assignment_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(assignment_out, index=False)

    changed_from_counts = Counter(before[seq] for seq in effective)
    changed_to_counts = Counter(effective.values())
    info = {
        "base_assignment_csv": str(args.base_assignment_csv),
        "manifest": str(args.manifest),
        "selected_ranks": sorted(_parse_ranks(args.selected_ranks)),
        "selected_entries": [
            {
                "rank": int(entry.get("rank", idx + 1) or idx + 1),
                "assignment_csv": entry.get("assignment_csv"),
                "moved_tracklets": int(entry.get("moved_tracklets", len(_moves_from_entry(entry)))),
                "source_component": entry.get("source_component"),
                "target_component": entry.get("target_component"),
                "score": entry.get("score"),
            }
            for idx, entry in enumerate(entries)
        ],
        "raw_move_rows": int(len(raw_moves)),
        "unique_move_seqs": int(len(seq_to_gid)),
        "effective_changed_tracklets": int(len(effective)),
        "noop_tracklets": int(len(noops)),
        "missing_tracklets": int(len(missing)),
        "missing_seq_preview": missing[:20],
        "changed_from_counts": {str(k): int(v) for k, v in sorted(changed_from_counts.items())},
        "changed_to_counts": {str(k): int(v) for k, v in sorted(changed_to_counts.items())},
        "effective_moves": [
            {"seq": int(seq), "from_predicted_global_id": int(before[seq]), "to_predicted_global_id": int(gid)}
            for seq, gid in sorted(effective.items())
        ],
        "assignment_out": str(assignment_out),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    json_out = Path(args.json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: info[k] for k in ("assignment_out", "effective_changed_tracklets", "noop_tracklets", "missing_tracklets")}, sort_keys=True))


if __name__ == "__main__":
    main()
