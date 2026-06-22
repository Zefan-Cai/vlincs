#!/usr/bin/env python
"""Gate and summarize no-anchor VLINCS global-ID sweep outputs.

The research loop produces many JSON/CSV sweep files. This helper makes the
success condition explicit: no anchors / no GT-for-training, tracklet global-ID
quality above a threshold, and end-to-end full IDF1 above a threshold.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import tempfile
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out / 100.0 if out > 1.5 else out


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _row_from_source(path: Path, source_kind: str, top_meta: dict[str, Any], row: dict[str, Any], rank: int) -> dict[str, Any]:
    merged = {**top_meta, **row}
    pair_metrics = merged.get("pair_metrics")
    if isinstance(pair_metrics, dict):
        for key in ("tracklet_pair_f1", "tracklet_pair_precision", "tracklet_pair_recall"):
            merged.setdefault(key, pair_metrics.get(key))
    full_metrics = merged.get("full") or merged.get("full_metrics")
    if isinstance(full_metrics, dict):
        for key in ("full_idf1", "full_hota", "full_assa"):
            merged.setdefault(key, full_metrics.get(key))
    resolve_info = merged.get("resolve_info")
    if isinstance(resolve_info, dict):
        merged.setdefault("components", resolve_info.get("components"))
        merged.setdefault("largest_component", resolve_info.get("largest_component"))
    if "assignment_components" in merged:
        merged["components"] = merged.get("assignment_components")
    if "largest_assignment_component" in merged:
        merged["largest_component"] = merged.get("largest_assignment_component")
    if "policy_name" in merged and "mode" not in merged:
        merged["mode"] = merged.get("policy_name")
    if "idf1" in merged and ("hota" in merged or "assa" in merged):
        merged.setdefault("full_idf1", merged.get("idf1"))
        merged.setdefault("full_hota", merged.get("hota"))
        merged.setdefault("full_assa", merged.get("assa"))
    uses_anchors = _as_bool(merged.get("uses_anchors"))
    uses_gt_train = _as_bool(merged.get("uses_gt_for_training_or_anchors"))
    uses_gt_analysis = _as_bool(merged.get("uses_gt_for_analysis_only"))
    uses_gt_filter = _as_bool(merged.get("uses_gt_for_filter_selection"))
    selection_uses_gt = _as_bool(merged.get("selection_uses_gt_metric"))
    path_says_no_anchor = "no_anchor" in path.name or "no_anchor" in str(path.parent.name)
    path_text = str(path).lower()
    mode_text = str(merged.get("mode") or merged.get("policy_name") or "").lower()
    path_or_mode_says_oracle = (
        "with_oracle" in path_text
        or "oracle_repair" in path_text
        or "pervideo_filter_oracle" in path_text
        or mode_text.startswith("oracle_")
    )
    if path_or_mode_says_oracle:
        no_anchor_ok = False
        no_anchor_evidence = "oracle_diagnostic"
    elif uses_gt_analysis is True or uses_gt_filter is True or selection_uses_gt is True:
        no_anchor_ok = False
        no_anchor_evidence = "gt_analysis_or_selection"
    elif uses_anchors is False and uses_gt_train is False:
        no_anchor_ok = True
        no_anchor_evidence = "metadata"
    elif uses_anchors is True or uses_gt_train is True:
        no_anchor_ok = False
        no_anchor_evidence = "metadata_violation"
    elif path_says_no_anchor:
        no_anchor_ok = True
        no_anchor_evidence = "path_name_only"
    else:
        no_anchor_ok = False
        no_anchor_evidence = "missing_metadata"
    return {
        "source": str(path),
        "source_kind": source_kind,
        "source_rank": int(rank),
        "no_anchor_ok": bool(no_anchor_ok),
        "no_anchor_evidence": no_anchor_evidence,
        "uses_anchors": uses_anchors,
        "uses_gt_for_training_or_anchors": uses_gt_train,
        "uses_gt_for_analysis_only": uses_gt_analysis,
        "uses_gt_for_filter_selection": uses_gt_filter,
        "selection_uses_gt_metric": selection_uses_gt,
        **merged,
    }


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        sections = [("json", data)]
        meta: dict[str, Any] = {}
    else:
        sections: list[tuple[str, list[Any]]] = []
        for key in ("top", "rows", "full_rows", "top_full_rows", "passing_rows", "top_rows"):
            value = data.get(key)
            if isinstance(value, list):
                sections.append((f"json:{key}", value))
        if not sections:
            sections = [("json", [data])]
        meta = {
            key: value
            for key, value in data.items()
            if key not in {"top", "rows", "full_rows", "top_full_rows", "passing_rows", "top_rows"}
            and not isinstance(value, (dict, list))
        }
    out = []
    for source_kind, rows in sections:
        for rank, row in enumerate(rows, start=1):
            if isinstance(row, dict):
                out.append(_row_from_source(path, source_kind, meta, row, rank))
    return out


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for rank, row in enumerate(csv.DictReader(f), start=1):
            rows.append(_row_from_source(path, "csv", {}, dict(row), rank))
    return rows


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.is_file() and str(path) not in seen:
                seen.add(str(path))
                paths.append(path)
    return sorted(paths)


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".json":
                rows.extend(_load_json(path))
            elif path.suffix.lower() == ".csv":
                rows.extend(_load_csv(path))
        except Exception as exc:  # pragma: no cover - diagnostic path
            rows.append(
                {
                    "source": str(path),
                    "source_kind": path.suffix.lower().lstrip("."),
                    "source_rank": 0,
                    "no_anchor_ok": False,
                    "no_anchor_evidence": "parse_error",
                    "parse_error": str(exc),
                }
            )
    return rows


def _metric(row: dict[str, Any], key: str) -> float:
    value = _as_float(row.get(key))
    return float(value) if value is not None else -1.0


def _compact_row(
    row: dict[str, Any] | None,
    global_metric: str,
    e2e_metric: str,
    precision_metric: str | None = None,
    recall_metric: str | None = None,
) -> dict[str, Any] | None:
    if row is None:
        return None
    keep = [
        "source",
        "source_kind",
        "source_rank",
        "no_anchor_ok",
        "no_anchor_evidence",
        "mode",
        "solver",
        "threshold",
        "blend",
        "tracklet_pair_f1",
        "tracklet_pair_precision",
        "tracklet_pair_recall",
        "full_idf1",
        "full_hota",
        "full_assa",
        "components",
        "largest_component",
        "assignment_status_counts",
        "pseudo_positive_pairs",
        "pseudo_negative_pairs",
        "attach_accepted",
        "attach_supported_groups",
    ]
    out = {key: row.get(key) for key in keep if key in row}
    out[f"{global_metric}_normalized"] = round(_metric(row, global_metric), 6)
    out[f"{e2e_metric}_normalized"] = round(_metric(row, e2e_metric), 6)
    if precision_metric:
        out[f"{precision_metric}_normalized"] = round(_metric(row, precision_metric), 6)
    if recall_metric:
        out[f"{recall_metric}_normalized"] = round(_metric(row, recall_metric), 6)
    return out


def _write_csv(
    path: str,
    rows: list[dict[str, Any]],
    global_metric: str,
    e2e_metric: str,
    precision_metric: str | None,
    recall_metric: str | None,
) -> None:
    compacted = [_compact_row(row, global_metric, e2e_metric, precision_metric, recall_metric) or {} for row in rows]
    keys = sorted({key for row in compacted for key in row})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(compacted)


def _self_test() -> None:
    payload = {
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_analysis_only": True,
        "rows": [
            {
                "policy_name": "density_oracle_lite",
                "idf1": 0.65524,
                "hota": 0.518652,
                "assa": 0.534359,
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
            }
        ],
        "top": [
            {
                "mode": "pair_only",
                "tracklet_pair_f1": 0.95,
                "tracklet_pair_precision": 0.92,
                "tracklet_pair_recall": 0.99,
                "full_idf1": 0.085,
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "no_anchor_rows_case.json"
        path.write_text(json.dumps(payload))
        rows = _load_rows([path])
    assert len(rows) == 2, rows
    by_kind = {row["source_kind"]: row for row in rows}
    assert round(_metric(by_kind["json:rows"], "full_idf1"), 6) == 0.65524, rows
    assert by_kind["json:rows"]["mode"] == "density_oracle_lite", rows
    assert by_kind["json:rows"]["no_anchor_ok"] is False, rows
    assert by_kind["json:rows"]["no_anchor_evidence"] == "gt_analysis_or_selection", rows
    assert round(_metric(by_kind["json:top"], "tracklet_pair_f1"), 6) == 0.95, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="JSON/CSV files or glob patterns to inspect")
    ap.add_argument("--global-metric", default="tracklet_pair_f1")
    ap.add_argument("--e2e-metric", default="full_idf1")
    ap.add_argument("--precision-metric", default="tracklet_pair_precision")
    ap.add_argument("--recall-metric", default="tracklet_pair_recall")
    ap.add_argument("--global-threshold", type=float, default=0.70)
    ap.add_argument("--e2e-threshold", type=float, default=0.70)
    ap.add_argument("--precision-threshold", type=float, default=0.0)
    ap.add_argument("--recall-threshold", type=float, default=0.0)
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--fail-under-threshold", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "ok"}, sort_keys=True))
        return
    if not args.paths:
        raise SystemExit("at least one JSON/CSV path or glob is required unless --self-test is set")

    paths = _expand_inputs(args.paths)
    rows = _load_rows(paths)
    eligible = [row for row in rows if row.get("no_anchor_ok")]
    best_global = max(eligible, key=lambda row: _metric(row, args.global_metric), default=None)
    best_e2e = max(eligible, key=lambda row: _metric(row, args.e2e_metric), default=None)
    passing_rows = [
        row
        for row in eligible
        if _metric(row, args.global_metric) >= float(args.global_threshold)
        and _metric(row, args.e2e_metric) >= float(args.e2e_threshold)
        and _metric(row, args.precision_metric) >= float(args.precision_threshold)
        and _metric(row, args.recall_metric) >= float(args.recall_threshold)
    ]
    joint_metric_names = [args.global_metric, args.e2e_metric]
    if float(args.precision_threshold) > 0:
        joint_metric_names.append(args.precision_metric)
    if float(args.recall_threshold) > 0:
        joint_metric_names.append(args.recall_metric)
    best_joint = max(
        eligible,
        key=lambda row: min(_metric(row, name) for name in joint_metric_names),
        default=None,
    )
    top_rows = sorted(
        eligible,
        key=lambda row: (
            min(_metric(row, name) for name in joint_metric_names),
            _metric(row, args.e2e_metric),
            _metric(row, args.global_metric),
        ),
        reverse=True,
    )[: max(int(args.top_n), 0)]

    result = {
        "input_files": [str(path) for path in paths],
        "rows_loaded": int(len(rows)),
        "eligible_no_anchor_rows": int(len(eligible)),
        "requirements": {
            "global_metric": args.global_metric,
            "global_threshold": float(args.global_threshold),
            "e2e_metric": args.e2e_metric,
            "e2e_threshold": float(args.e2e_threshold),
            "precision_metric": args.precision_metric,
            "precision_threshold": float(args.precision_threshold),
            "recall_metric": args.recall_metric,
            "recall_threshold": float(args.recall_threshold),
            "requires_no_anchor": True,
        },
        "pass_joint": bool(passing_rows),
        "passing_rows": [
            _compact_row(row, args.global_metric, args.e2e_metric, args.precision_metric, args.recall_metric)
            for row in passing_rows[: max(int(args.top_n), 0)]
        ],
        "best_global": _compact_row(best_global, args.global_metric, args.e2e_metric, args.precision_metric, args.recall_metric),
        "best_e2e": _compact_row(best_e2e, args.global_metric, args.e2e_metric, args.precision_metric, args.recall_metric),
        "best_joint": _compact_row(best_joint, args.global_metric, args.e2e_metric, args.precision_metric, args.recall_metric),
        "top_rows": [_compact_row(row, args.global_metric, args.e2e_metric, args.precision_metric, args.recall_metric) for row in top_rows],
        "disqualified_rows": int(len(rows) - len(eligible)),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n")
    if args.csv_out:
        _write_csv(args.csv_out, top_rows, args.global_metric, args.e2e_metric, args.precision_metric, args.recall_metric)
    print(text)
    if args.fail_under_threshold and not result["pass_joint"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
