#!/usr/bin/env python
"""Merge current components from cross-assignment teacher consensus.

This no-anchor diagnostic treats existing no-anchor assignment CSVs as weak
teachers.  If several teachers independently map two current components to the
same predicted identity with high within-component dominance, the edge becomes
a merge candidate.  Ground truth is used only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from kit.no_anchor_louvain_sweep import _write_assignments
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _parse_floats, _parse_ints, _write_csv
    from no_anchor_louvain_sweep import _write_assignments
    from no_anchor_resolve_sweep import (
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


class UnionFind:
    def __init__(self, values: list[int]):
        self.parent = {int(value): int(value) for value in values}
        self.members = {int(value): {int(value)} for value in values}

    def find(self, x: int) -> int:
        x = int(x)
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def merge(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].update(self.members.pop(rb))
        return True


def _component_members_by_label(labels: np.ndarray, keep_indices: set[int]) -> tuple[list[int], dict[int, list[int]]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        out[int(labels[idx])].append(int(idx))
    label_ids = sorted(out, key=lambda label: min(out[label]))
    return label_ids, out


def _component_teacher_dominants(label_ids: list[int], members: dict[int, list[int]], records, teacher: dict[int, int], min_coverage: float):
    out: dict[int, tuple[int, float, float]] = {}
    for label in label_ids:
        indices = members[label]
        values = []
        for idx in indices:
            seq = int(records[idx].seq)
            if seq in teacher:
                values.append(int(teacher[seq]))
        coverage = float(len(values) / max(len(indices), 1))
        if coverage < float(min_coverage) or not values:
            continue
        counts = Counter(values)
        gid, n = counts.most_common(1)[0]
        dominance = float(n / max(len(values), 1))
        out[int(label)] = (int(gid), float(dominance), float(coverage))
    return out


def _component_centroids(emb: np.ndarray, label_ids: list[int], members: dict[int, list[int]]) -> dict[int, np.ndarray]:
    x = _l2n(emb.astype(np.float32))
    out = {}
    for label in label_ids:
        v = x[np.asarray(members[label], dtype=np.int64)].mean(axis=0)
        out[int(label)] = (v / (np.linalg.norm(v) + 1.0e-9)).astype(np.float32)
    return out


def _teacher_edges(label_ids, teacher_maps, *, dominance_min: float, min_valid_teachers: int):
    rows = []
    for pos, a in enumerate(label_ids):
        for b in label_ids[pos + 1 :]:
            valid = 0
            same = 0
            diff = 0
            doms = []
            covs = []
            same_teachers = []
            for teacher_idx, tmap in enumerate(teacher_maps):
                ta = tmap.get(int(a))
                tb = tmap.get(int(b))
                if ta is None or tb is None:
                    continue
                gid_a, dom_a, cov_a = ta
                gid_b, dom_b, cov_b = tb
                if dom_a < float(dominance_min) or dom_b < float(dominance_min):
                    continue
                valid += 1
                doms.extend([float(dom_a), float(dom_b)])
                covs.extend([float(cov_a), float(cov_b)])
                if int(gid_a) == int(gid_b):
                    same += 1
                    same_teachers.append(str(teacher_idx))
                else:
                    diff += 1
            if valid < int(min_valid_teachers):
                continue
            same_frac = float(same / max(valid, 1))
            rows.append(
                {
                    "source": int(a),
                    "target": int(b),
                    "teacher_valid": int(valid),
                    "teacher_same": int(same),
                    "teacher_diff": int(diff),
                    "teacher_same_frac": same_frac,
                    "teacher_mean_dominance": float(np.mean(doms)) if doms else 0.0,
                    "teacher_min_dominance": float(np.min(doms)) if doms else 0.0,
                    "teacher_mean_coverage": float(np.mean(covs)) if covs else 0.0,
                    "teacher_same_teachers": ",".join(same_teachers),
                }
            )
    return rows


def _annotate_visual(rows, centroids_by_view):
    names = sorted(centroids_by_view)
    for row in rows:
        a = int(row["source"])
        b = int(row["target"])
        sims = []
        for name in names:
            cents = centroids_by_view[name]
            sims.append(float(np.dot(cents[a], cents[b])))
        arr = np.asarray(sims, dtype=np.float32)
        row["view_mean_sim"] = float(arr.mean()) if arr.size else 0.0
        row["view_min_sim"] = float(arr.min()) if arr.size else 0.0
        row["view_max_sim"] = float(arr.max()) if arr.size else 0.0
        row["view_std_sim"] = float(arr.std()) if arr.size else 0.0
        row["teacher_visual_score"] = (
            0.50 * float(row["teacher_same_frac"])
            + 0.20 * float(row["teacher_min_dominance"])
            + 0.20 * float(row["view_mean_sim"])
            + 0.10 * float(row["view_min_sim"])
        )
    rows.sort(key=lambda item: float(item["teacher_visual_score"]), reverse=True)
    return rows


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int, min_overlap: int) -> bool:
    return min(a_end, b_end) - max(a_start, b_start) + 1 >= int(min_overlap)


def _component_conflicts(label_ids, members, records, *, min_overlap_frames: int) -> set[tuple[int, int]]:
    by_stream: dict[int, dict[tuple[str, str], list[tuple[int, int]]]] = {}
    for label in label_ids:
        table: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for idx in members[label]:
            rec = records[idx]
            table[(str(rec.video), str(rec.camera))].append((int(rec.start_frame), int(rec.end_frame)))
        by_stream[int(label)] = table
    out: set[tuple[int, int]] = set()
    for pos, a in enumerate(label_ids):
        ta = by_stream[int(a)]
        for b in label_ids[pos + 1 :]:
            tb = by_stream[int(b)]
            bad = False
            for stream in set(ta) & set(tb):
                for sa, ea in ta[stream]:
                    for sb, eb in tb[stream]:
                        if _interval_overlap(sa, ea, sb, eb, int(min_overlap_frames)):
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    break
            if bad:
                out.add((int(a), int(b)))
    return out


def _labels_from_component_uf(base_labels: np.ndarray, uf: UnionFind) -> np.ndarray:
    root_to_new: dict[int, int] = {}
    out = np.full(len(base_labels), -1, dtype=np.int64)
    next_label = 0
    for idx, old in enumerate(base_labels.tolist()):
        old = int(old)
        root = uf.find(old) if old in uf.parent else old
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        out[idx] = root_to_new[root]
    return out


def _merge_edges(base_labels, label_ids, members, edges, conflicts, *, threshold: float, min_same_votes: int, min_same_frac: float, max_diff_votes: int, min_view_mean: float, max_component_size: int):
    uf = UnionFind(list(set(int(x) for x in base_labels.tolist())))
    component_sizes = {int(label): len(members.get(int(label), [])) for label in label_ids}
    accepted = 0
    rejected_threshold = 0
    rejected_teacher = 0
    rejected_visual = 0
    rejected_conflict = 0
    rejected_size = 0
    stale = 0
    for row in sorted(edges, key=lambda item: float(item["teacher_visual_score"]), reverse=True):
        if float(row["teacher_visual_score"]) < float(threshold):
            rejected_threshold += 1
            continue
        if int(row["teacher_same"]) < int(min_same_votes) or float(row["teacher_same_frac"]) < float(min_same_frac) or int(row["teacher_diff"]) > int(max_diff_votes):
            rejected_teacher += 1
            continue
        if float(row["view_mean_sim"]) < float(min_view_mean):
            rejected_visual += 1
            continue
        a = int(row["source"])
        b = int(row["target"])
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            stale += 1
            continue
        size_a = sum(component_sizes.get(label, 1) for label in uf.members.get(ra, {ra}))
        size_b = sum(component_sizes.get(label, 1) for label in uf.members.get(rb, {rb}))
        if size_a + size_b > int(max_component_size):
            rejected_size += 1
            continue
        bad = False
        for ca in uf.members.get(ra, {ra}):
            for cb in uf.members.get(rb, {rb}):
                key = (int(ca), int(cb)) if int(ca) < int(cb) else (int(cb), int(ca))
                if key in conflicts:
                    bad = True
                    break
            if bad:
                break
        if bad:
            rejected_conflict += 1
            continue
        uf.merge(ra, rb)
        accepted += 1
    labels = _labels_from_component_uf(base_labels, uf)
    return labels, {
        "threshold": float(threshold),
        "min_same_votes": int(min_same_votes),
        "min_same_frac": float(min_same_frac),
        "max_diff_votes": int(max_diff_votes),
        "min_view_mean": float(min_view_mean),
        "max_component_size": int(max_component_size),
        "accepted_edges": int(accepted),
        "rejected_threshold": int(rejected_threshold),
        "rejected_teacher": int(rejected_teacher),
        "rejected_visual": int(rejected_visual),
        "rejected_conflict": int(rejected_conflict),
        "rejected_size": int(rejected_size),
        "rejected_stale": int(stale),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "uses_ground_truth": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--teacher-csv", action="append", default=[], required=True)
    ap.add_argument("--feature-npz", action="append", default=[])
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--teacher-min-coverage", type=float, default=0.60)
    ap.add_argument("--teacher-dominance-min", type=float, default=0.70)
    ap.add_argument("--min-valid-teachers", type=int, default=2)
    ap.add_argument("--thresholds", default="0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--min-same-votes-grid", default="1,2,3")
    ap.add_argument("--min-same-fracs", default="0.50,0.67,0.80,1.00")
    ap.add_argument("--max-diff-votes-grid", default="0,1")
    ap.add_argument("--min-view-means", default="-1.0,0.50,0.60,0.70")
    ap.add_argument("--max-component-sizes", default="500")
    ap.add_argument("--conflict-min-overlap-frames", type=int, default=1)
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=80_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
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
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    label_ids, members = _component_members_by_label(base_labels, keep_indices)
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    teacher_maps = []
    teacher_meta = []
    for path in args.teacher_csv:
        teacher = _load_assignment_labels(path, args.pred_col)
        cmap = _component_teacher_dominants(label_ids, members, records, teacher, float(args.teacher_min_coverage))
        teacher_maps.append(cmap)
        teacher_meta.append({"path": str(path), "component_labels": int(len(cmap))})
    edges = _teacher_edges(
        label_ids,
        teacher_maps,
        dominance_min=float(args.teacher_dominance_min),
        min_valid_teachers=int(args.min_valid_teachers),
    )
    centroids_by_view = {"db": _component_centroids(db_emb, label_ids, members)}
    for idx, path in enumerate(args.feature_npz):
        emb = _load_feature_npz(path, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0)
        centroids_by_view[f"view{idx}"] = _component_centroids(emb, label_ids, members)
    edges = _annotate_visual(edges, centroids_by_view)
    conflicts = _component_conflicts(label_ids, members, records, min_overlap_frames=int(args.conflict_min_overlap_frames))
    print(json.dumps({"stage": "teacher_edges", "edges": len(edges), "teachers": teacher_meta, "conflicts": len(conflicts)}, sort_keys=True), flush=True)

    rows = []
    for threshold in _parse_floats(args.thresholds):
        for min_same_votes in _parse_ints(args.min_same_votes_grid):
            for min_same_frac in _parse_floats(args.min_same_fracs):
                for max_diff_votes in _parse_ints(args.max_diff_votes_grid):
                    for min_view_mean in _parse_floats(args.min_view_means):
                        for max_component_size in _parse_ints(args.max_component_sizes):
                            labels, info = _merge_edges(
                                base_labels,
                                label_ids,
                                members,
                                edges,
                                conflicts,
                                threshold=float(threshold),
                                min_same_votes=int(min_same_votes),
                                min_same_frac=float(min_same_frac),
                                max_diff_votes=int(max_diff_votes),
                                min_view_mean=float(min_view_mean),
                                max_component_size=int(max_component_size),
                            )
                            pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                            pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                            rows.append(
                                {
                                    "mode": "teacher_consensus_merge",
                                    **info,
                                    **pair,
                                    "uses_anchors": False,
                                    "uses_gt_for_training_or_anchors": False,
                                    "uses_gt_for_evaluation_only": True,
                                }
                            )
    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
            float(row["tracklet_pair_precision"]),
        ),
        reverse=True,
    )
    labels_by_rank = {}
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        labels, _info = _merge_edges(
            base_labels,
            label_ids,
            members,
            edges,
            conflicts,
            threshold=float(row["threshold"]),
            min_same_votes=int(row["min_same_votes"]),
            min_same_frac=float(row["min_same_frac"]),
            max_diff_votes=int(row["max_diff_votes"]),
            min_view_mean=float(row["min_view_mean"]),
            max_component_size=int(row["max_component_size"]),
        )
        labels_by_rank[rank] = labels
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs))
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "full": full, "row": row}, sort_keys=True), flush=True)
    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            labels, _info = _merge_edges(
                base_labels,
                label_ids,
                members,
                edges,
                conflicts,
                threshold=float(row["threshold"]),
                min_same_votes=int(row["min_same_votes"]),
                min_same_frac=float(row["min_same_frac"]),
                max_diff_votes=int(row["max_diff_votes"]),
                min_view_mean=float(row["min_view_mean"]),
                max_component_size=int(row["max_component_size"]),
            )
        assignment_info = _write_assignments(args.assignments_out, records, labels, keep_seqs=keep_seqs, offset=int(args.assignment_offset))
        rows[0].update(assignment_info)
    result = {
        "dbname": args.dbname,
        "role": args.role,
        "assignment_csv": str(args.assignment_csv),
        "teacher_csvs": list(args.teacher_csv),
        "teacher_meta": teacher_meta,
        "feature_npz": list(args.feature_npz),
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "teacher_edge_count": int(len(edges)),
        "teacher_edge_preview": edges[:50],
        "component_conflicts": int(len(conflicts)),
        "assignment_info": assignment_info,
        "top": rows[: max(80, int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"base": base_pair, "teacher_edge_count": len(edges), "best": rows[0] if rows else None, "json": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
