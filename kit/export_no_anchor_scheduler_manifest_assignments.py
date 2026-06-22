#!/usr/bin/env python
"""Materialize selected no-anchor scheduler rows as assignment CSVs.

The full-score scheduler is deliberately offline: it ranks candidate rows
without calling the DS1 full scorer.  This utility performs the next step in a
reproducible way by replaying a selected row's accepted_preview edits on top of
a base assignment CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.no_anchor_fullscore_scheduler import _artifact_matching_row, _as_float


def _load_scheduler_selected(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is missing a selected[] scheduler manifest")
    source_rows = data.get("selected")
    if not isinstance(source_rows, list):
        source_rows = data.get("rows")
    if not isinstance(source_rows, list):
        raise ValueError(f"{path} is missing selected[] or rows[] scheduler rows")
    selected = []
    for idx, row in enumerate(source_rows, start=1):
        if isinstance(row, dict):
            selected.append({"_selection_rank": idx, **row})
    return selected


def _parse_ranks(text: str, max_rank: int) -> set[int]:
    if not text:
        return set(range(1, max_rank + 1))
    ranks = {int(part.strip()) for part in text.split(",") if part.strip()}
    bad = sorted(rank for rank in ranks if rank < 1 or rank > max_rank)
    if bad:
        raise ValueError(f"selection rank(s) outside 1..{max_rank}: {bad}")
    return ranks


def _accepted_preview(row: dict[str, Any]) -> list[dict[str, Any]]:
    preview = row.get("accepted_preview")
    if isinstance(preview, list) and preview:
        return [item for item in preview if isinstance(item, dict)]
    restored = _artifact_matching_row(row, require_full_context=False, include_source_file=True)
    preview = restored.get("accepted_preview") if isinstance(restored, dict) else None
    if isinstance(preview, list) and preview:
        return [item for item in preview if isinstance(item, dict)]
    raise ValueError(
        "selected row has no accepted_preview and provenance lookup failed: "
        f"source={row.get('_source_file')} rank={row.get('_source_rank')}"
    )


def _safe_slug(text: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (slug or "candidate")[:max_len]


def _load_assignment_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = {"seq", "predicted_global_id", "component_label"} - set(fieldnames)
        if missing:
            raise ValueError(f"{path} is missing assignment columns: {sorted(missing)}")
        return [dict(row) for row in reader], fieldnames


def _intish(value: Any) -> int:
    return int(float(value))


def _preview_component(item: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return _intish(value)
    return None


def _source_seqs_for_preview_item(
    item: dict[str, Any],
    base_rows: list[dict[str, str]],
) -> tuple[list[int], int | None]:
    raw_seqs = item.get("source_seqs")
    if isinstance(raw_seqs, list) and raw_seqs:
        return [_intish(seq) for seq in raw_seqs], None
    source_component = _preview_component(item, "source_component_label", "source", "source_rep")
    if source_component is None:
        raise ValueError(f"accepted_preview item has neither source_seqs nor source component: {item}")
    seqs = [
        _intish(row["seq"])
        for row in base_rows
        if _intish(row["component_label"]) == int(source_component)
    ]
    return seqs, int(source_component)


def _component_gid(rows: list[dict[str, str]], component: int) -> int:
    counts: Counter[int] = Counter()
    for row in rows:
        try:
            if _intish(row["component_label"]) == int(component):
                counts[_intish(row["predicted_global_id"])] += 1
        except (KeyError, TypeError, ValueError):
            continue
    if not counts:
        raise ValueError(f"target_component={component} does not exist in base assignment")
    return counts.most_common(1)[0][0]


def _resolve_target_component(
    item: dict[str, Any],
    rows: list[dict[str, str]],
    by_seq: dict[int, dict[str, str]],
    target: int,
) -> tuple[int, dict[str, Any] | None]:
    if any(_intish(row["component_label"]) == int(target) for row in rows):
        return int(target), None
    seqs = item.get("target_top_seqs")
    if not isinstance(seqs, list) or not seqs:
        return int(target), None
    votes: Counter[int] = Counter()
    seen_seqs: list[int] = []
    for seq in seqs:
        row = by_seq.get(_intish(seq))
        if row is None:
            continue
        seen_seqs.append(_intish(seq))
        votes[_intish(row["component_label"])] += 1
    if len(votes) != 1:
        return int(target), None
    repaired = votes.most_common(1)[0][0]
    return repaired, {
        "from_target_component": int(target),
        "to_target_component": int(repaired),
        "target_top_seqs": seen_seqs,
        "reason": "target_top_seqs_unique_component_vote",
    }


def _refresh_component_sizes(rows: list[dict[str, str]]) -> None:
    counts = Counter(_intish(row["component_label"]) for row in rows)
    for row in rows:
        label = _intish(row["component_label"])
        size = counts[label]
        if "component_size" in row:
            row["component_size"] = str(int(size))
        if "decision_status" in row and row.get("decision_status") not in {"manifest_reassign"}:
            row["decision_status"] = "forced_singleton" if size == 1 else "forced_component"
        if "prediction_confidence" in row and row.get("decision_status") != "manifest_reassign":
            confidence = 0.15 if size == 1 else min(0.85, 0.30 + 0.02 * min(size, 20))
            row["prediction_confidence"] = f"{confidence:.6f}"


def _replay_preview(
    base_rows: list[dict[str, str]],
    preview: list[dict[str, Any]],
    *,
    allow_missing_source_seqs: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = deepcopy(base_rows)
    by_seq = {_intish(row["seq"]): row for row in rows}
    original_component_counts = Counter(_intish(row["component_label"]) for row in base_rows)
    moved: list[dict[str, Any]] = []
    missing: list[int] = []
    skipped_empty_source_components: list[int] = []
    target_component_repairs: list[dict[str, Any]] = []

    for item in preview:
        target = _preview_component(item, "target_component", "target", "target_rep")
        if target is None:
            raise ValueError(f"accepted_preview item has no target component: {item}")
        target, repair = _resolve_target_component(item, rows, by_seq, int(target))
        if repair is not None:
            target_component_repairs.append(repair)
        target_gid = _component_gid(rows, target)
        source_seqs, source_component = _source_seqs_for_preview_item(item, rows)
        if not source_seqs and source_component is not None:
            if original_component_counts.get(int(source_component), 0) > 0:
                skipped_empty_source_components.append(int(source_component))
                continue
            missing.append(int(source_component))
            continue
        for seq in source_seqs:
            row = by_seq.get(seq)
            if row is None:
                missing.append(seq)
                continue
            before = {
                "seq": seq,
                "from_component": _intish(row["component_label"]),
                "from_predicted_global_id": _intish(row["predicted_global_id"]),
                "to_component": int(target),
                "to_predicted_global_id": int(target_gid),
            }
            row["component_label"] = str(int(target))
            row["predicted_global_id"] = str(int(target_gid))
            if "decision_status" in row:
                row["decision_status"] = "manifest_reassign"
            moved.append(before)

    if missing and not allow_missing_source_seqs:
        raise ValueError(f"accepted_preview references seqs absent from base assignment: {sorted(set(missing))[:20]}")
    _refresh_component_sizes(rows)
    return rows, {
        "moved_tracklets": int(len(moved)),
        "missing_source_seqs": sorted(set(missing)),
        "skipped_empty_source_components": sorted(set(skipped_empty_source_components)),
        "target_components": sorted({int(move["to_component"]) for move in moved}),
        "target_predicted_global_ids": sorted({int(move["to_predicted_global_id"]) for move in moved}),
        "target_component_repairs": target_component_repairs,
        "moved_preview": moved[:50],
    }


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_manifest_assignments(
    *,
    scheduler_json: Path,
    base_assignment_csv: Path,
    assignment_out_dir: Path,
    manifest_json: Path,
    selection_ranks: str,
    allow_missing_source_seqs: bool = False,
) -> dict[str, Any]:
    selected = _load_scheduler_selected(scheduler_json)
    keep = _parse_ranks(selection_ranks, len(selected))
    base_rows, fieldnames = _load_assignment_rows(base_assignment_csv)
    outputs = []

    for row in selected:
        rank = int(row["_selection_rank"])
        if rank not in keep:
            continue
        preview = _accepted_preview(row)
        output_rows, replay_info = _replay_preview(
            base_rows,
            preview,
            allow_missing_source_seqs=allow_missing_source_seqs,
        )
        mode = str(row.get("mode") or row.get("policy_name") or "candidate")
        source = _safe_slug(Path(str(row.get("_source_file") or "source")).stem)
        out_csv = assignment_out_dir / f"rank{rank:02d}_{_safe_slug(mode)}_{source}_assignments.csv"
        _write_csv(out_csv, output_rows, fieldnames)
        outputs.append(
            {
                "selection_rank": rank,
                "output_csv": str(out_csv),
                "source_file": row.get("_source_file"),
                "source_rank": row.get("_source_rank"),
                "mode": mode,
                "predicted_full_idf1": _as_float(row.get("predicted_full_idf1")),
                "pair_f1": _as_float(row.get("pair_f1"), _as_float(row.get("tracklet_pair_f1"))),
                "pair_precision": _as_float(row.get("pair_precision"), _as_float(row.get("tracklet_pair_precision"))),
                "pair_recall": _as_float(row.get("pair_recall"), _as_float(row.get("tracklet_pair_recall"))),
                "accepted_preview_count": int(len(preview)),
                **replay_info,
            }
        )

    out = {
        "scheduler_json": str(scheduler_json),
        "base_assignment_csv": str(base_assignment_csv),
        "assignment_out_dir": str(assignment_out_dir),
        "selection_ranks": sorted(keep),
        "outputs": outputs,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base.csv"
        fieldnames = ["seq", "component_label", "component_size", "predicted_global_id", "prediction_confidence", "decision_status"]
        rows = [
            {"seq": "1", "component_label": "0", "component_size": "2", "predicted_global_id": "900", "prediction_confidence": "0.34", "decision_status": "forced_component"},
            {"seq": "2", "component_label": "0", "component_size": "2", "predicted_global_id": "900", "prediction_confidence": "0.34", "decision_status": "forced_component"},
            {"seq": "3", "component_label": "1", "component_size": "2", "predicted_global_id": "901", "prediction_confidence": "0.34", "decision_status": "forced_component"},
            {"seq": "4", "component_label": "1", "component_size": "2", "predicted_global_id": "901", "prediction_confidence": "0.34", "decision_status": "forced_component"},
        ]
        _write_csv(base, rows, fieldnames)
        cand_csv = root / "candidate.csv"
        cand_csv.write_text("mode,tracklet_pair_f1\nconflict_subcluster_reassign,0.7\n", encoding="utf-8")
        cand_json = root / "candidate.json"
        cand_json.write_text(
            json.dumps(
                {
                    "top": [
                        {
                            "mode": "conflict_subcluster_reassign",
                            "tracklet_pair_f1": 0.7,
                            "accepted_preview": [{"source_seqs": [3], "target_component": 0}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        scheduler = root / "scheduler.json"
        scheduler.write_text(
            json.dumps(
                {
                    "selected": [
                        {
                            "_source_file": str(cand_csv),
                            "_source_rank": 1,
                            "mode": "conflict_subcluster_reassign",
                            "tracklet_pair_f1": 0.7,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        manifest = root / "manifest.json"
        out = export_manifest_assignments(
            scheduler_json=scheduler,
            base_assignment_csv=base,
            assignment_out_dir=root / "out",
            manifest_json=manifest,
            selection_ranks="1",
        )
        assert len(out["outputs"]) == 1
        out_csv = Path(out["outputs"][0]["output_csv"])
        loaded, _fields = _load_assignment_rows(out_csv)
        by_seq = {_intish(row["seq"]): row for row in loaded}
        assert by_seq[3]["predicted_global_id"] == "900"
        assert by_seq[3]["component_label"] == "0"
        assert by_seq[3]["decision_status"] == "manifest_reassign"
        assert by_seq[1]["component_size"] == "3"
        assert by_seq[4]["component_size"] == "1"
        assert json.loads(manifest.read_text())["outputs"][0]["moved_tracklets"] == 1

        component_scheduler = root / "component_scheduler.json"
        component_scheduler.write_text(
            json.dumps(
                {
                    "selected": [
                        {
                            "mode": "edge_table_island_merge",
                            "accepted_preview": [{"source": 1, "target": 0}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        component_manifest = root / "component_manifest.json"
        out = export_manifest_assignments(
            scheduler_json=component_scheduler,
            base_assignment_csv=base,
            assignment_out_dir=root / "component_out",
            manifest_json=component_manifest,
            selection_ranks="1",
        )
        loaded, _fields = _load_assignment_rows(Path(out["outputs"][0]["output_csv"]))
        by_seq = {_intish(row["seq"]): row for row in loaded}
        assert by_seq[3]["predicted_global_id"] == "900"
        assert by_seq[4]["predicted_global_id"] == "900"
        assert json.loads(component_manifest.read_text())["outputs"][0]["moved_tracklets"] == 2

        repaired_scheduler = root / "repaired_scheduler.json"
        repaired_scheduler.write_text(
            json.dumps(
                {
                    "selected": [
                        {
                            "mode": "repair_missing_target_component",
                            "accepted_preview": [
                                {
                                    "source_seqs": [3],
                                    "target_component": 99,
                                    "target_top_seqs": [1, 2],
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        repaired_manifest = root / "repaired_manifest.json"
        out = export_manifest_assignments(
            scheduler_json=repaired_scheduler,
            base_assignment_csv=base,
            assignment_out_dir=root / "repaired_out",
            manifest_json=repaired_manifest,
            selection_ranks="1",
        )
        loaded, _fields = _load_assignment_rows(Path(out["outputs"][0]["output_csv"]))
        by_seq = {_intish(row["seq"]): row for row in loaded}
        assert by_seq[3]["component_label"] == "0"
        assert json.loads(repaired_manifest.read_text())["outputs"][0]["target_component_repairs"][0]["to_target_component"] == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheduler-json", default="")
    ap.add_argument("--base-assignment-csv", default="")
    ap.add_argument("--assignment-out-dir", default="")
    ap.add_argument("--manifest-json", default="")
    ap.add_argument("--selection-ranks", default="")
    ap.add_argument("--allow-missing-source-seqs", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    missing = [
        name
        for name in ("scheduler_json", "base_assignment_csv", "assignment_out_dir", "manifest_json")
        if not getattr(args, name)
    ]
    if missing:
        ap.error(f"missing required argument(s) unless --self-test is used: {', '.join('--' + key.replace('_', '-') for key in missing)}")
    out = export_manifest_assignments(
        scheduler_json=Path(args.scheduler_json),
        base_assignment_csv=Path(args.base_assignment_csv),
        assignment_out_dir=Path(args.assignment_out_dir),
        manifest_json=Path(args.manifest_json),
        selection_ranks=str(args.selection_ranks),
        allow_missing_source_seqs=bool(args.allow_missing_source_seqs),
    )
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
