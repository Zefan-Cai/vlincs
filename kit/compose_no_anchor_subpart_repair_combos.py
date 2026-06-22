#!/usr/bin/env python
"""Compose no-anchor subpart repair candidates into assignment CSVs.

The input manifest is produced by propose_no_anchor_subpart_repair_candidates.py.
This utility only uses the production-side assignment CSV and candidate move
previews.  It does not load GT, anchors, or evaluation labels.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path
from typing import Any


def _safe_slug(text: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (slug or "combo")[:max_len]


def _load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    required = {"seq", "component_label", "predicted_global_id"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"{path} missing required columns {sorted(missing)}")
    return rows, fieldnames


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_combo_spec(text: str) -> list[int]:
    parts = [part.strip() for part in re.split(r"[+,]", str(text)) if part.strip()]
    if not parts:
        raise ValueError(f"empty combo spec {text!r}")
    return [int(part.replace("rank", "").replace("r", "")) for part in parts]


def _candidate_by_rank(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selected = manifest.get("selected") or manifest.get("top_candidates") or []
    out: dict[int, dict[str, Any]] = {}
    for pos, cand in enumerate(selected, start=1):
        rank = int(cand.get("rank", pos))
        out[rank] = cand
    return out


def _valid_combo(candidates: list[dict[str, Any]]) -> tuple[bool, str]:
    seq_to_target: dict[int, tuple[int, int]] = {}
    for cand in candidates:
        for move in cand.get("moved_preview", []):
            seq = int(move["seq"])
            target = (int(move["to_component"]), int(move["to_predicted_global_id"]))
            previous = seq_to_target.get(seq)
            if previous is not None and previous != target:
                return False, f"seq {seq} assigned to conflicting targets {previous} and {target}"
            seq_to_target[seq] = target
    return True, "ok"


def _apply_combo(rows: list[dict[str, str]], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    new_rows = [dict(row) for row in rows]
    row_by_seq = {int(float(row["seq"])): row for row in new_rows}
    applied: list[dict[str, Any]] = []
    for cand in candidates:
        for move in cand.get("moved_preview", []):
            seq = int(move["seq"])
            row = row_by_seq.get(seq)
            if row is None:
                raise ValueError(f"seq {seq} from candidate rank {cand.get('rank')} not found in assignment rows")
            before_component = int(float(row["component_label"]))
            before_gid = int(float(row["predicted_global_id"]))
            to_component = int(move["to_component"])
            to_gid = int(move["to_predicted_global_id"])
            row["component_label"] = str(to_component)
            row["predicted_global_id"] = str(to_gid)
            applied.append(
                {
                    "seq": seq,
                    "from_component": before_component,
                    "from_predicted_global_id": before_gid,
                    "to_component": to_component,
                    "to_predicted_global_id": to_gid,
                    "candidate_rank": int(cand.get("rank", -1)),
                }
            )
    return new_rows, applied


def _auto_combos(ranks: list[int], min_size: int, max_size: int, require_rank: int | None) -> list[list[int]]:
    combos: list[list[int]] = []
    for size in range(min_size, max_size + 1):
        for combo in itertools.combinations(ranks, size):
            if require_rank is not None and require_rank not in combo:
                continue
            combos.append(list(combo))
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--assignment-csv", type=Path, help="override manifest assignment_csv")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--combo", action="append", default=[], help="rank combo such as 1+2+4; repeatable")
    parser.add_argument("--auto", action="store_true", help="generate combinations from available ranks")
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--require-rank", type=int, default=None)
    parser.add_argument("--top-rank", type=int, default=None, help="keep only ranks <= this value for --auto")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    assignment_csv = args.assignment_csv or Path(manifest["assignment_csv"])
    rows, fieldnames = _load_rows(assignment_csv)
    by_rank = _candidate_by_rank(manifest)
    if not by_rank:
        raise SystemExit("manifest contains no selected/top candidates")

    specs: list[list[int]] = []
    for text in args.combo:
        specs.append(_parse_combo_spec(text))
    if args.auto:
        ranks = sorted(by_rank)
        if args.top_rank is not None:
            ranks = [rank for rank in ranks if rank <= int(args.top_rank)]
        specs.extend(_auto_combos(ranks, int(args.min_size), int(args.max_size), args.require_rank))
    if not specs:
        raise SystemExit("no combos requested; pass --combo or --auto")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_dir = args.output_dir / "assignments"
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()

    for combo_ranks in specs:
        combo_key = tuple(combo_ranks)
        if combo_key in seen:
            continue
        seen.add(combo_key)
        missing = [rank for rank in combo_ranks if rank not in by_rank]
        if missing:
            selected.append({"ranks": combo_ranks, "status": "skipped", "reason": f"missing ranks {missing}"})
            continue
        candidates = [by_rank[rank] for rank in combo_ranks]
        ok, reason = _valid_combo(candidates)
        if not ok:
            selected.append({"ranks": combo_ranks, "status": "skipped", "reason": reason})
            continue
        combo_rows, applied = _apply_combo(rows, candidates)
        rank_text = "_".join(f"r{rank:02d}" for rank in combo_ranks)
        move_count = len({move["seq"] for move in applied})
        stem = _safe_slug(f"subpart_combo_{rank_text}_{move_count}seq_assignments")
        out_csv = assignment_dir / f"{stem}.csv"
        _write_rows(out_csv, fieldnames, combo_rows)
        selected.append(
            {
                "ranks": combo_ranks,
                "status": "written",
                "assignment_csv": str(out_csv),
                "moved_tracklets": move_count,
                "applied_moves": applied,
                "candidate_scores": [float(cand.get("score", 0.0)) for cand in candidates],
                "candidate_target_margins": [float(cand.get("target_margin", 0.0)) for cand in candidates],
                "candidate_source_components": [int(cand.get("source_component")) for cand in candidates],
                "candidate_target_components": [int(cand.get("target_component")) for cand in candidates],
            }
        )

    out_manifest = {
        "manifest": str(args.manifest),
        "assignment_csv": str(assignment_csv),
        "output_dir": str(args.output_dir),
        "selected": selected,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    (args.output_dir / "subpart_combo_manifest.json").write_text(json.dumps(out_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
