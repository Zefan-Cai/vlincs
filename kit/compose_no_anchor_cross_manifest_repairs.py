#!/usr/bin/env python3
"""Compose selected no-anchor repair candidates from multiple manifests.

Each input candidate manifest is produced from production-side evidence. This
script only reads assignment CSVs and candidate move previews; it does not read
GT, anchors, evaluator labels, or score files.
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


def _load_manifest_candidate(path: Path, rank: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected") or data.get("top_candidates") or []
    for pos, candidate in enumerate(selected, start=1):
        candidate_rank = int(candidate.get("rank", pos))
        if candidate_rank == int(rank):
            out = dict(candidate)
            out["_manifest"] = str(path)
            out["_rank"] = int(rank)
            return out
    raise SystemExit(f"rank {rank} not found in {path}")


def _parse_candidate(text: str) -> tuple[Path, int]:
    if ":" not in text:
        raise SystemExit(f"candidate must be MANIFEST:RANK, got {text!r}")
    manifest, rank = text.rsplit(":", 1)
    return Path(manifest), int(rank.replace("rank", "").replace("r", ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-assignment-csv", required=True, type=Path)
    ap.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="candidate spec MANIFEST_JSON:RANK; repeatable",
    )
    ap.add_argument("--assignment-out", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--decision-status", default="cross_manifest_subpart_combo")
    args = ap.parse_args()

    df = pd.read_csv(args.base_assignment_csv).reset_index(drop=True)
    required = {"seq", "predicted_global_id", "component_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"assignment missing columns: {missing}")
    by_seq = {int(seq): i for i, seq in enumerate(df["seq"].astype(int).tolist())}

    candidates = [_load_manifest_candidate(*_parse_candidate(text)) for text in args.candidate]
    edited: set[int] = set()
    changes: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    for candidate in candidates:
        moved = []
        for move in candidate.get("moved_preview", []):
            seq = int(move["seq"])
            if seq not in by_seq:
                raise SystemExit(f"seq {seq} from {candidate['_manifest']} rank {candidate['_rank']} not found")
            if seq in edited:
                raise SystemExit(f"seq {seq} selected by multiple candidates")
            row_i = by_seq[seq]
            before_gid = int(df.at[row_i, "predicted_global_id"])
            before_component = int(df.at[row_i, "component_label"])
            target_gid = int(move["to_predicted_global_id"])
            target_component = int(move["to_component"])
            if before_gid == target_gid and before_component == target_component:
                continue
            if before_gid != int(move["from_predicted_global_id"]) or before_component != int(move["from_component"]):
                raise SystemExit(
                    f"seq {seq} has {before_gid}/{before_component}; "
                    f"expected {move['from_predicted_global_id']}/{move['from_component']}"
                )
            before_status = str(df.at[row_i, "decision_status"]) if "decision_status" in df.columns else ""
            df.at[row_i, "predicted_global_id"] = target_gid
            df.at[row_i, "component_label"] = target_component
            if "decision_status" in df.columns:
                df.at[row_i, "decision_status"] = args.decision_status
            edited.add(seq)
            moved.append(seq)
            changes.append(
                {
                    "manifest": candidate["_manifest"],
                    "rank": candidate["_rank"],
                    "seq": seq,
                    "before_predicted_global_id": before_gid,
                    "after_predicted_global_id": target_gid,
                    "before_component_label": before_component,
                    "after_component_label": target_component,
                    "before_decision_status": before_status,
                    "after_decision_status": args.decision_status,
                }
            )
        selected.append(
            {
                "manifest": candidate["_manifest"],
                "rank": candidate["_rank"],
                "moved_seqs": moved,
                "score": candidate.get("score"),
                "source_component": candidate.get("source_component"),
                "target_component": candidate.get("target_component"),
                "target_sim": candidate.get("target_sim"),
                "target_margin": candidate.get("target_margin"),
                "conflicts_to_rest": candidate.get("conflicts_to_rest"),
            }
        )

    if "component_size" in df.columns:
        component_counts = df["component_label"].astype(int).value_counts()
        df["component_size"] = df["component_label"].astype(int).map(component_counts).astype(int)

    args.assignment_out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.assignment_out, index=False)
    report = {
        "base_assignment_csv": str(args.base_assignment_csv),
        "assignment_csv": str(args.assignment_out),
        "assignment_sha256": _sha256(args.assignment_out),
        "decision_status": args.decision_status,
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
                "assignment_csv": report["assignment_csv"],
                "assignment_sha256": report["assignment_sha256"],
                "moved_tracklets": report["moved_tracklets"],
                "selected": [
                    {"manifest": item["manifest"], "rank": item["rank"], "moved_seqs": item["moved_seqs"]}
                    for item in selected
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
