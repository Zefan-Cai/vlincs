#!/usr/bin/env python
"""Eval-only repair decomposition for a no-anchor assignment.

This diagnostic starts from a delivered no-anchor assignment and asks how much
of the remaining end-to-end gap is due to false merges vs false splits.  It
constructs GT-guided repair variants only for analysis; none of these variants
is a deployable no-anchor model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _comb2(sum_w: float, sum_w2: float) -> float:
    return max((sum_w * sum_w - sum_w2) / 2.0, 0.0)


def _bucket_pair_mass(items: list[tuple[int, float]]) -> float:
    total = sum(weight for _key, weight in items)
    total2 = sum(weight * weight for _key, weight in items)
    return _comb2(total, total2)


def _load_assignment(path: str, pred_col: str) -> dict[int, int]:
    out: dict[int, int] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"seq", pred_col} - fields
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        for row in reader:
            out[int(float(row["seq"]))] = int(float(row[pred_col]))
    return out


def _component_tables(pred_by_seq, keep_seqs, gt_by_seq, weight_by_seq):
    by_pred: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    by_gt: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    for seq, pred in pred_by_seq.items():
        seq = int(seq)
        if seq not in keep_seqs or seq not in gt_by_seq:
            continue
        gid = int(gt_by_seq[seq])
        weight = float(weight_by_seq.get(seq, 1.0))
        by_pred[int(pred)].append((gid, weight, seq))
        by_gt[gid].append((int(pred), weight, seq))
    return by_pred, by_gt


def _rank_false_merges(by_pred, top_n: int) -> list[dict[str, object]]:
    rows = []
    for pred, triples in by_pred.items():
        total = sum(weight for _gt, weight, _seq in triples)
        total2 = sum(weight * weight for _gt, weight, _seq in triples)
        pred_pair = _comb2(total, total2)
        by_gt: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for gt, weight, seq in triples:
            by_gt[int(gt)].append((seq, weight))
        true_pair = sum(_bucket_pair_mass(values) for values in by_gt.values())
        dominant_gt, dominant_weight = max(
            ((gt, sum(weight for _seq, weight in values)) for gt, values in by_gt.items()),
            key=lambda item: item[1],
        )
        false_merge_mass = pred_pair - true_pair
        rows.append(
            {
                "predicted_global_id": int(pred),
                "false_merge_mass": round(float(false_merge_mass), 3),
                "pred_pair_mass": round(float(pred_pair), 3),
                "true_pair_mass": round(float(true_pair), 3),
                "tracklets": int(len(triples)),
                "gt_count": int(len(by_gt)),
                "dominant_gt": int(dominant_gt),
                "dominant_gt_weight_frac": round(float(dominant_weight / max(total, 1.0e-9)), 6),
            }
        )
    rows.sort(key=lambda row: (float(row["false_merge_mass"]), float(row["pred_pair_mass"])), reverse=True)
    return rows[: int(top_n)]


def _rank_false_splits(by_gt, top_n: int) -> list[dict[str, object]]:
    rows = []
    for gt, triples in by_gt.items():
        total = sum(weight for _pred, weight, _seq in triples)
        total2 = sum(weight * weight for _pred, weight, _seq in triples)
        gt_pair = _comb2(total, total2)
        by_pred: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for pred, weight, seq in triples:
            by_pred[int(pred)].append((seq, weight))
        true_pair = sum(_bucket_pair_mass(values) for values in by_pred.values())
        dominant_pred, dominant_weight = max(
            ((pred, sum(weight for _seq, weight in values)) for pred, values in by_pred.items()),
            key=lambda item: item[1],
        )
        false_split_mass = gt_pair - true_pair
        rows.append(
            {
                "gt_id": int(gt),
                "false_split_mass": round(float(false_split_mass), 3),
                "gt_pair_mass": round(float(gt_pair), 3),
                "true_pair_mass": round(float(true_pair), 3),
                "tracklets": int(len(triples)),
                "pred_component_count": int(len(by_pred)),
                "dominant_prediction": int(dominant_pred),
                "dominant_prediction_weight_frac": round(float(dominant_weight / max(total, 1.0e-9)), 6),
            }
        )
    rows.sort(key=lambda row: (float(row["false_split_mass"]), float(row["gt_pair_mass"])), reverse=True)
    return rows[: int(top_n)]


def _split_selected_components(pred_by_seq, selected_preds, gt_by_seq, *, offset: int) -> dict[int, int]:
    out = dict(pred_by_seq)
    selected = {int(value) for value in selected_preds}
    for seq, pred in list(out.items()):
        if int(pred) not in selected or int(seq) not in gt_by_seq:
            continue
        out[int(seq)] = int(offset) + int(gt_by_seq[int(seq)])
    return out


def _merge_selected_gt(pred_by_seq, selected_gt, gt_by_seq, *, offset: int) -> dict[int, int]:
    out = dict(pred_by_seq)
    selected = {int(value) for value in selected_gt}
    for seq in list(out):
        if int(seq) not in gt_by_seq:
            continue
        gid = int(gt_by_seq[int(seq)])
        if gid in selected:
            out[int(seq)] = int(offset) + gid
    return out


def _oracle_all(pred_by_seq, gt_by_seq, *, offset: int) -> dict[int, int]:
    out = dict(pred_by_seq)
    next_singleton = int(offset) + 10_000_000
    for seq in list(out):
        if int(seq) in gt_by_seq:
            out[int(seq)] = int(offset) + int(gt_by_seq[int(seq)])
        else:
            out[int(seq)] = next_singleton
            next_singleton += 1
    return out


def _score_variant(name, pred, records, pred_by_video, gt_by_video, gt_by_seq, weight_by_seq, *, full: bool) -> dict[str, object]:
    seqs = [int(record.seq) for record in records]
    pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
    row = {
        "name": name,
        **pair,
        "components": int(len(set(pred.values()))),
        "assigned_tracklets": int(len(pred)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_analysis_only": True,
    }
    if full:
        metrics = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in metrics.items() if key != "per_video"})
        row["full_per_video"] = metrics.get("per_video", {})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--output-drop-area-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-area-quantile-by-video", default="")
    ap.add_argument("--output-drop-quality-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-quality-quantile-by-video", default="")
    ap.add_argument("--output-auto-anomaly-admission", action="store_true")
    ap.add_argument("--output-auto-anomaly-metric", default="quality")
    ap.add_argument("--output-auto-anomaly-quantile", type=float, default=0.75)
    ap.add_argument("--output-auto-anomaly-area-ratio", type=float, default=0.60)
    ap.add_argument("--output-auto-anomaly-quality-mad", type=float, default=1.0)
    ap.add_argument("--output-auto-anomaly-min-video-tracklets", type=int, default=20)
    ap.add_argument("--output-auto-anomaly-max-videos", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--split-top-k", default="0,1,3,5,10,20")
    ap.add_argument("--merge-top-k", default="0,1,3,5,10,20")
    ap.add_argument("--full-top-n", type=int, default=12)
    ap.add_argument("--skip-full", action="store_true", help="skip full HOTA/IDF1 scoring and write pair-only diagnostics")
    ap.add_argument(
        "--skip-reference-full",
        action="store_true",
        help="when full scoring is enabled, skip base/oracle reference full scoring and only score selected combo rows",
    )
    ap.add_argument("--oracle-offset", type=int, default=88_000_000)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    input_pred = _load_assignment(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, _emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": 0.5,
        "min_matches": 1,
        "min_purity": 0.0,
        "n_tracklets": len(records),
        "prediction_rows": int(sum(len(value) for value in pred_by_video.values())),
        "gt_rows": int(sum(len(value) for value in gt_by_video.values())),
    }
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached
    keep_seqs, output_info = _output_keep_seqs(records, args)
    base_pred = {int(seq): int(pred) for seq, pred in input_pred.items() if int(seq) in keep_seqs}
    by_pred, by_gt = _component_tables(base_pred, keep_seqs, gt_by_seq, weight_by_seq)
    false_merges = _rank_false_merges(by_pred, int(args.top_n))
    false_splits = _rank_false_splits(by_gt, int(args.top_n))

    split_order = [int(row["predicted_global_id"]) for row in false_merges]
    merge_order = [int(row["gt_id"]) for row in false_splits]
    split_grid = _parse_ints(args.split_top_k)
    merge_grid = _parse_ints(args.merge_top_k)
    rows = []
    run_full = not bool(args.skip_full)
    run_reference_full = run_full and not bool(args.skip_reference_full)
    rows.append(
        _score_variant(
            "base",
            base_pred,
            records,
            pred_by_video,
            gt_by_video,
            gt_by_seq,
            weight_by_seq,
            full=run_reference_full,
        )
    )
    rows.append(
        _score_variant(
            "oracle_all_gt_majority",
            _oracle_all(base_pred, gt_by_seq, offset=int(args.oracle_offset)),
            records,
            pred_by_video,
            gt_by_video,
            gt_by_seq,
            weight_by_seq,
            full=run_reference_full,
        )
    )
    print(
        json.dumps({"stage": "scored_base_oracle", "full": run_reference_full, "rows": len(rows)}),
        file=sys.stderr,
        flush=True,
    )

    for split_k in split_grid:
        split_pred = _split_selected_components(
            base_pred,
            split_order[: int(split_k)],
            gt_by_seq,
            offset=int(args.oracle_offset),
        )
        rows.append(
            _score_variant(
                f"split_top_false_merge_components_{int(split_k)}",
                split_pred,
                records,
                pred_by_video,
                gt_by_video,
                gt_by_seq,
                weight_by_seq,
                full=False,
            )
        )
    for merge_k in merge_grid:
        merge_pred = _merge_selected_gt(
            base_pred,
            merge_order[: int(merge_k)],
            gt_by_seq,
            offset=int(args.oracle_offset),
        )
        rows.append(
            _score_variant(
                f"merge_top_false_split_gt_{int(merge_k)}",
                merge_pred,
                records,
                pred_by_video,
                gt_by_video,
                gt_by_seq,
                weight_by_seq,
                full=False,
            )
        )

    combo_rows = []
    for split_k in split_grid:
        for merge_k in merge_grid:
            pred = _split_selected_components(
                base_pred,
                split_order[: int(split_k)],
                gt_by_seq,
                offset=int(args.oracle_offset),
            )
            pred = _merge_selected_gt(
                pred,
                merge_order[: int(merge_k)],
                gt_by_seq,
                offset=int(args.oracle_offset),
            )
            row = _score_variant(
                f"split_top_{int(split_k)}_then_merge_top_{int(merge_k)}",
                pred,
                records,
                pred_by_video,
                gt_by_video,
                gt_by_seq,
                weight_by_seq,
                full=False,
            )
            row["oracle_split_top_k"] = int(split_k)
            row["oracle_merge_top_k"] = int(merge_k)
            combo_rows.append(row)
    combo_rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
            float(row["tracklet_pair_precision"]),
        ),
        reverse=True,
    )
    print(json.dumps({"stage": "scored_pair_grid", "combo_rows": len(combo_rows)}), file=sys.stderr, flush=True)
    full_budget = 0 if bool(args.skip_full) else max(int(args.full_top_n), 0)
    for full_idx, row in enumerate(combo_rows[:full_budget], start=1):
        split_k = int(row["oracle_split_top_k"])
        merge_k = int(row["oracle_merge_top_k"])
        pred = _split_selected_components(base_pred, split_order[:split_k], gt_by_seq, offset=int(args.oracle_offset))
        pred = _merge_selected_gt(pred, merge_order[:merge_k], gt_by_seq, offset=int(args.oracle_offset))
        metrics = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in metrics.items() if key != "per_video"})
        row["full_per_video"] = metrics.get("per_video", {})
        print(
            json.dumps(
                {
                    "stage": "scored_full_candidate",
                    "idx": full_idx,
                    "split_top_k": split_k,
                    "merge_top_k": merge_k,
                    "idf1": metrics.get("idf1"),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    rows.extend(combo_rows)

    result = {
        "assignment_csv": args.assignment_csv,
        "pred_col": args.pred_col,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "top_false_merge_components": false_merges,
        "top_false_split_gt_ids": false_splits,
        "rows": rows,
        "top_pair_rows": sorted(
            rows,
            key=lambda row: (
                float(row["tracklet_pair_f1"]),
                float(row["tracklet_pair_recall"]),
                float(row["tracklet_pair_precision"]),
            ),
            reverse=True,
        )[:30],
        "top_full_rows": sorted(
            [row for row in rows if "full_idf1" in row],
            key=lambda row: (
                float(row["full_idf1"]),
                float(row.get("full_hota", 0.0)),
                float(row["tracklet_pair_f1"]),
            ),
            reverse=True,
        )[:30],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_analysis_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "json": str(out),
                "base_idf1": rows[0].get("full_idf1"),
                "oracle_idf1": rows[1].get("full_idf1"),
                "best_full": result["top_full_rows"][0] if result["top_full_rows"] else None,
                "best_pair": result["top_pair_rows"][0] if result["top_pair_rows"] else None,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
