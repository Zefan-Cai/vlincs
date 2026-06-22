#!/usr/bin/env python
"""No-anchor global-ID resolve sweeps over an existing VLINCS gallery DB.

This is an evaluation/research driver for the no-anchor setting. It reads the
tracklet evidence already stored by the pipeline (boxes, frames, metadata, and
role='resolve' embeddings), clusters tracklets without identity labels, and
uses DS1 GT only after the fact for metrics.

The fast metric is a detection-weighted tracklet-pair F1:

* precision: predicted same-ID tracklet pairs that share the same GT identity.
* recall: GT same-ID tracklet pairs recovered by the resolver.

Optional full scoring builds a submission-like dataframe from the proposed
cluster IDs and runs the canonical VLINCS HOTA/IDF1 scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
try:
    import psycopg
except ModuleNotFoundError:  # parquet-only local diagnostics do not need PostgreSQL.
    psycopg = None
from sklearn.cluster import AgglomerativeClustering

from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video
from vlincs_gallery.resolve import (
    _knn_sparse_affinity,
    build_knn_cosine_affinity,
    global_agglom_resolve,
    mustlink_resolve,
    two_tier_resolve,
)


_GT_ID_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class TrackletRecord:
    seq: int
    tracklet_key: str
    video: str
    camera: str
    start_frame: int
    end_frame: int
    start_abs_ms: int
    end_abs_ms: int
    n_dets: int
    avg_conf: float
    cx: float
    cy: float
    width: float
    height: float
    first_cx: float
    first_cy: float
    first_width: float
    first_height: float
    last_cx: float
    last_cy: float
    last_width: float
    last_height: float


@dataclass(frozen=True)
class ResolveConfig:
    mode: str
    theta: float = 0.02
    top_k: int = 15
    min_dets: int = 1
    exclude_same: str = "camera"
    cross_thr: float = 0.62
    intra_thr: float = 0.70
    max_gap: int = 450
    max_component_size: int = 120
    temporal_bonus: float = 0.04
    local_gap: int = 60
    local_thr: float = 0.78
    local_app_min: float = 0.50
    local_pos_scale: float = 1.5
    target_clusters: int = 0
    rerank_k1: int = 20
    rerank_k2: int = 6
    rerank_lambda: float = 0.3
    time_window_ms: int = 500


@dataclass
class GraphCache:
    nn_scores: np.ndarray
    nn_indices: np.ndarray
    forbidden: list[set[int]]
    edge_cache: dict[int, list[tuple[float, int, int]]]


_RERANK_DISTANCE_CACHE: dict[tuple[object, ...], tuple[np.ndarray, int]] = {}


def _connect(dbname: str):
    if psycopg is None:
        raise RuntimeError("psycopg is required for gallery DB mode; install psycopg or use a parquet sample runner")
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "55433"),
        user=os.environ.get("PGUSER", "gallery"),
        password=os.environ.get("PGPASSWORD", "gallery"),
        dbname=dbname,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _vec_array(value: object) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return np.fromstring(text, sep=",", dtype=np.float32)
    return np.asarray(value, np.float32)


def _load_tracklets(con, role: str) -> tuple[list[TrackletRecord], np.ndarray]:
    with con.cursor() as cur:
        cur.execute(
            """WITH seq_stats AS (
                   SELECT a.seq,
                          COUNT(*) AS cnt,
                          MIN(d.frame_idx) AS start_frame,
                          MAX(d.frame_idx) AS end_frame,
                          MIN(d.abs_ms) AS start_abs_ms,
                          MAX(d.abs_ms) AS end_abs_ms,
                          AVG((d.x1 + d.x2) / 2.0) AS cx,
                          AVG((d.y1 + d.y2) / 2.0) AS cy,
                          AVG(d.x2 - d.x1) AS width,
                          AVG(d.y2 - d.y1) AS height,
                          AVG(d.conf) AS avg_conf
                   FROM assignments a JOIN detections d ON d.det_id = a.det_id
                   GROUP BY a.seq
               )
               SELECT e.seq,
                      COALESCE(t.tracklet_key, e.entity_id::text) AS tracklet_key,
                      COALESCE(t.video, d.video) AS video,
                      COALESCE(t.camera, d.camera) AS camera,
                      COALESCE(t.start_frame, s.start_frame) AS start_frame,
                      COALESCE(t.end_frame, s.end_frame) AS end_frame,
                      COALESCE(t.n_dets, s.cnt) AS n_dets,
                      s.start_abs_ms, s.end_abs_ms,
                      s.avg_conf, s.cx, s.cy, s.width, s.height,
                      e.vec
               FROM embeddings e
               JOIN detections d ON d.det_id = e.entity_id
               JOIN seq_stats s ON s.seq = e.seq
               LEFT JOIN tracklets t ON t.seq = e.seq
               WHERE e.role = %s
               ORDER BY e.seq""",
            (role,),
        )
        rows = cur.fetchall()
    records: list[TrackletRecord] = []
    vectors = []
    for row in rows:
        records.append(
            TrackletRecord(
                seq=int(row[0]),
                tracklet_key=str(row[1]),
                video=str(row[2]),
                camera=str(row[3]),
                start_frame=int(row[4]),
                end_frame=int(row[5]),
                n_dets=int(row[6]),
                start_abs_ms=int(row[7] or 0),
                end_abs_ms=int(row[8] or row[7] or 0),
                avg_conf=float(row[9] or 0.0),
                cx=float(row[10] or 0.0),
                cy=float(row[11] or 0.0),
                width=float(row[12] or 0.0),
                height=float(row[13] or 0.0),
                first_cx=float(row[10] or 0.0),
                first_cy=float(row[11] or 0.0),
                first_width=float(row[12] or 0.0),
                first_height=float(row[13] or 0.0),
                last_cx=float(row[10] or 0.0),
                last_cy=float(row[11] or 0.0),
                last_width=float(row[12] or 0.0),
                last_height=float(row[13] or 0.0),
            )
        )
        vectors.append(_vec_array(row[14]))
    if not records:
        raise RuntimeError(f"no role={role!r} tracklet embeddings found")
    return records, _l2n(np.stack(vectors).astype(np.float32))


def _load_feature_npz(
    path: str,
    records: list[TrackletRecord],
    db_emb: np.ndarray,
    *,
    concat_db: bool,
    db_weight: float,
    feature_weight: float,
) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    seqs = [int(x) for x in data["seqs"].tolist()]
    features = data["features"].astype(np.float32)
    by_seq = {seq: features[i] for i, seq in enumerate(seqs)}
    missing = [record.seq for record in records if record.seq not in by_seq]
    if missing:
        raise ValueError(f"feature npz is missing {len(missing)} DB tracklets; first missing seq={missing[0]}")
    ext = _l2n(np.stack([by_seq[record.seq] for record in records]).astype(np.float32))
    if not concat_db:
        return _l2n((ext * float(feature_weight)).astype(np.float32))
    return _l2n(
        np.concatenate(
            [
                _l2n(db_emb.astype(np.float32)) * float(db_weight),
                ext * float(feature_weight),
            ],
            axis=1,
        ).astype(np.float32)
    )


def _load_predictions(con) -> dict[str, pd.DataFrame]:
    with con.cursor() as cur:
        cur.execute(
            """SELECT d.video, d.frame_idx AS frame, a.seq,
                      d.x1, d.y1, d.x2, d.y2, d.object_type, d.conf
               FROM detections d JOIN assignments a ON a.det_id = d.det_id
               ORDER BY d.video, d.frame_idx, a.seq"""
        )
        rows = cur.fetchall()
    df = pd.DataFrame(
        rows,
        columns=["video", "frame", "seq", "x1", "y1", "x2", "y2", "object_type", "confidence"],
    )
    return {
        str(video): g.drop(columns=["video"]).reset_index(drop=True)
        for video, g in df.groupby("video", sort=True)
    }


def _with_detection_endpoints(records: list[TrackletRecord], pred_by_video: dict[str, pd.DataFrame]) -> list[TrackletRecord]:
    endpoints: dict[int, dict[str, float | int]] = {}
    for pred in pred_by_video.values():
        if pred.empty:
            continue
        ordered = pred.sort_values(["seq", "frame"], kind="mergesort")
        for seq, group in ordered.groupby("seq", sort=False):
            first = group.iloc[0]
            last = group.iloc[-1]
            endpoints[int(seq)] = {
                "first_cx": float((first.x1 + first.x2) * 0.5),
                "first_cy": float((first.y1 + first.y2) * 0.5),
                "first_width": float(first.x2 - first.x1),
                "first_height": float(first.y2 - first.y1),
                "last_cx": float((last.x1 + last.x2) * 0.5),
                "last_cy": float((last.y1 + last.y2) * 0.5),
                "last_width": float(last.x2 - last.x1),
                "last_height": float(last.y2 - last.y1),
                "start_frame": int(group["frame"].min()),
                "end_frame": int(group["frame"].max()),
                "n_dets": int(len(group)),
            }
    return [replace(record, **endpoints.get(record.seq, {})) for record in records]


def _iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    px1, py1, px2, py2 = [pred_boxes[:, i][:, None] for i in range(4)]
    gx1, gy1, gx2, gy2 = [gt_boxes[:, i][None, :] for i in range(4)]
    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    p_area = np.maximum(px2 - px1, 0) * np.maximum(py2 - py1, 0)
    g_area = np.maximum(gx2 - gx1, 0) * np.maximum(gy2 - gy1, 0)
    return (inter / np.maximum(p_area + g_area - inter, 1e-9)).astype(np.float32)


def _gt_numeric_id(gt_id: object, mapping: dict[str, int]) -> int:
    key = str(gt_id)
    if key in mapping:
        return mapping[key]
    match = _GT_ID_RE.search(key)
    if match:
        candidate = int(match.group(1))
        if candidate > 0 and candidate not in mapping.values():
            mapping[key] = candidate
            return candidate
    mapping[key] = max(mapping.values(), default=0) + 1
    return mapping[key]


def _label_tracklets_for_eval(
    pred_by_video: dict[str, pd.DataFrame],
    gt_by_video: dict[str, pd.DataFrame],
    *,
    iou_thr: float,
    min_matches: int,
    min_purity: float,
) -> tuple[dict[int, int], dict[int, float], dict[str, object]]:
    seq_counts: dict[int, Counter] = defaultdict(Counter)
    seq_total = Counter()
    gt_id_map: dict[str, int] = {}
    matched_rows = 0
    total_rows = 0

    for video, pred in pred_by_video.items():
        gt = gt_by_video.get(video)
        if gt is None or pred.empty:
            continue
        pred_sorted = pred.sort_values("frame", kind="mergesort")
        gt_sorted = gt.sort_values("frame", kind="mergesort")
        p_frames = pred_sorted["frame"].to_numpy(np.int64)
        p_seqs = pred_sorted["seq"].to_numpy(np.int64)
        p_boxes_all = pred_sorted[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        g_frames = gt_sorted["frame"].to_numpy(np.int64)
        g_ids = gt_sorted["id"].to_numpy()
        g_boxes_all = gt_sorted[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        total_rows += int(len(p_frames))
        seq_values, seq_row_counts = np.unique(p_seqs, return_counts=True)
        for seq, count in zip(seq_values, seq_row_counts):
            seq_total[int(seq)] += int(count)

        pi = 0
        gi = 0
        n_pred = len(p_frames)
        n_gt = len(g_frames)
        while pi < n_pred:
            frame = int(p_frames[pi])
            pj = int(np.searchsorted(p_frames, frame, side="right"))
            gi = int(np.searchsorted(g_frames, frame, side="left", sorter=None)) if gi >= n_gt or g_frames[gi] < frame else gi
            if gi >= n_gt or int(g_frames[gi]) != frame:
                pi = pj
                continue
            gj = int(np.searchsorted(g_frames, frame, side="right"))
            pboxes = p_boxes_all[pi:pj]
            gboxes = g_boxes_all[gi:gj]
            ious = _iou_matrix(pboxes, gboxes)
            best = ious.argmax(axis=1) if ious.size else np.zeros((len(pboxes),), dtype=np.int64)
            best_iou = ious[np.arange(len(pboxes)), best] if ious.size else np.zeros((len(pboxes),), dtype=np.float32)
            gt_ids = g_ids[gi:gj]
            for seq, j, score in zip(p_seqs[pi:pj], best, best_iou):
                if float(score) >= iou_thr:
                    seq_counts[int(seq)][_gt_numeric_id(gt_ids[int(j)], gt_id_map)] += 1
                    matched_rows += 1
            pi = pj

    seq_to_gt: dict[int, int] = {}
    seq_weight: dict[int, float] = {}
    accepted = 0
    rejected = 0
    purities = []
    for seq, total in seq_total.items():
        counts = seq_counts.get(seq)
        if not counts:
            rejected += 1
            continue
        gid, count = counts.most_common(1)[0]
        purity = count / max(sum(counts.values()), 1)
        purities.append(purity)
        if count >= min_matches and purity >= min_purity:
            seq_to_gt[int(seq)] = int(gid)
            seq_weight[int(seq)] = float(total)
            accepted += 1
        else:
            rejected += 1

    stats = {
        "iou_thr": float(iou_thr),
        "min_matches": int(min_matches),
        "min_purity": float(min_purity),
        "eval_labeled_tracklets": int(accepted),
        "eval_rejected_tracklets": int(rejected),
        "matched_detection_rows": int(matched_rows),
        "total_detection_rows": int(total_rows),
        "matched_detection_fraction": round(matched_rows / max(total_rows, 1), 6),
        "mean_gt_purity_of_matched_tracklets": round(float(np.mean(purities)) if purities else 0.0, 6),
        "uses_gt_for_evaluation_only": True,
    }
    return seq_to_gt, seq_weight, stats


def _cache_eval_labels(
    path: str,
    seq_to_gt: dict[int, int],
    seq_weight: dict[int, float],
    stats: dict[str, object],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    seqs = np.array(sorted(seq_to_gt), dtype=np.int64)
    gids = np.array([seq_to_gt[int(seq)] for seq in seqs], dtype=np.int64)
    weights = np.array([seq_weight[int(seq)] for seq in seqs], dtype=np.float32)
    np.savez_compressed(out, seqs=seqs, gids=gids, weights=weights, stats=json.dumps(stats, sort_keys=True))


def _load_eval_label_cache(path: str, expected: dict[str, object]) -> tuple[dict[int, int], dict[int, float], dict[str, object]] | None:
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=False)
    stats = json.loads(str(data["stats"].item()))
    for key, value in expected.items():
        if stats.get(key) != value:
            return None
    seqs = data["seqs"].astype(np.int64)
    gids = data["gids"].astype(np.int64)
    weights = data["weights"].astype(np.float32)
    return (
        {int(seq): int(gid) for seq, gid in zip(seqs, gids)},
        {int(seq): float(weight) for seq, weight in zip(seqs, weights)},
        {**stats, "loaded_from_eval_cache": str(p)},
    )


def _comb2_weight(sum_w: float, sum_w2: float) -> float:
    return max((sum_w * sum_w - sum_w2) / 2.0, 0.0)


def _pair_metrics(
    seqs: Iterable[int],
    pred_by_seq: dict[int, int],
    gt_by_seq: dict[int, int],
    weight_by_seq: dict[int, float],
) -> dict[str, float]:
    pred_totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    gt_totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    cross_totals: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    n_eval = 0
    for seq in seqs:
        seq = int(seq)
        if seq not in gt_by_seq or seq not in pred_by_seq:
            continue
        w = float(weight_by_seq.get(seq, 1.0))
        p = int(pred_by_seq[seq])
        g = int(gt_by_seq[seq])
        pred_totals[p][0] += w
        pred_totals[p][1] += w * w
        gt_totals[g][0] += w
        gt_totals[g][1] += w * w
        cross_totals[(p, g)][0] += w
        cross_totals[(p, g)][1] += w * w
        n_eval += 1

    pred_pairs = sum(_comb2_weight(v[0], v[1]) for v in pred_totals.values())
    gt_pairs = sum(_comb2_weight(v[0], v[1]) for v in gt_totals.values())
    true_pairs = sum(_comb2_weight(v[0], v[1]) for v in cross_totals.values())
    precision = true_pairs / pred_pairs if pred_pairs > 0 else 0.0
    recall = true_pairs / gt_pairs if gt_pairs > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "tracklet_pair_precision": round(float(precision), 6),
        "tracklet_pair_recall": round(float(recall), 6),
        "tracklet_pair_f1": round(float(f1), 6),
        "eval_tracklets": int(n_eval),
        "pred_pair_mass": round(float(pred_pairs), 3),
        "gt_pair_mass": round(float(gt_pairs), 3),
        "true_pair_mass": round(float(true_pairs), 3),
    }


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.members = {i: {i} for i in range(n)}

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def can_merge(self, a: int, b: int, forbidden: list[set[int]], max_size: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        ma, mb = self.members[ra], self.members[rb]
        if len(ma) + len(mb) > max_size:
            return False
        small, large = (ma, mb) if len(ma) <= len(mb) else (mb, ma)
        for node in small:
            if forbidden[node] & large:
                return False
        return True

    def merge(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].update(self.members.pop(rb))

    def labels(self) -> np.ndarray:
        roots = [self.find(i) for i in range(len(self.parent))]
        mapping: dict[int, int] = {}
        out = np.zeros(len(roots), dtype=np.int64)
        for i, root in enumerate(roots):
            if root not in mapping:
                mapping[root] = len(mapping)
            out[i] = mapping[root]
        return out


def _same_stream_gap(a: TrackletRecord, b: TrackletRecord) -> int | None:
    if a.video != b.video or a.camera != b.camera:
        return None
    if a.start_frame <= b.end_frame and b.start_frame <= a.end_frame:
        return -1
    return max(0, b.start_frame - a.end_frame, a.start_frame - b.end_frame)


def _scale_similarity(a: TrackletRecord, b: TrackletRecord) -> float:
    h1 = max(float(a.last_height), 1.0)
    h2 = max(float(b.first_height), 1.0)
    w1 = max(float(a.last_width), 1.0)
    w2 = max(float(b.first_width), 1.0)
    return float(np.exp(-(abs(np.log(h1 / h2)) + 0.5 * abs(np.log(w1 / w2)))))


def _trajectory_link_score(a: TrackletRecord, b: TrackletRecord, visual: float, cfg: ResolveConfig) -> tuple[float, dict[str, float]]:
    gap = max(0, int(b.start_frame) - int(a.end_frame))
    duration = max(int(a.end_frame) - int(a.start_frame), 1)
    vx = (float(a.last_cx) - float(a.first_cx)) / duration
    vy = (float(a.last_cy) - float(a.first_cy)) / duration
    pred_x = float(a.last_cx) + vx * gap
    pred_y = float(a.last_cy) + vy * gap
    scale = max((float(a.last_height) + float(b.first_height)) * 0.5 * float(cfg.local_pos_scale), 1.0)
    dist = float(np.hypot(pred_x - float(b.first_cx), pred_y - float(b.first_cy)))
    pos = float(np.exp(-0.5 * (dist / scale) ** 2))
    gap_sim = float(np.exp(-gap / max(float(cfg.local_gap) * 0.5, 1.0)))
    size = _scale_similarity(a, b)
    score = 0.55 * max(float(visual), 0.0) + 0.25 * pos + 0.15 * gap_sim + 0.05 * size
    return float(score), {
        "visual": float(visual),
        "pos": pos,
        "gap_sim": gap_sim,
        "scale": size,
        "gap": float(gap),
        "pixel_dist": dist,
    }


def _local_component_labels(records: list[TrackletRecord], emb: np.ndarray, cfg: ResolveConfig) -> tuple[np.ndarray, dict[str, object]]:
    """No-anchor within-stream stitching by appearance + bbox motion continuity."""
    uf = _UnionFind(len(records))
    forbidden = _build_overlap_forbidden(records)
    stream_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        stream_to_indices[(record.video, record.camera)].append(i)

    candidates: list[tuple[float, int, int, dict[str, float]]] = []
    rejected_app = 0
    for indices in stream_to_indices.values():
        ordered = sorted(indices, key=lambda idx: (records[idx].start_frame, records[idx].end_frame, idx))
        for pos, i in enumerate(ordered):
            right_limit = int(records[i].end_frame) + int(cfg.local_gap)
            for j in ordered[pos + 1 :]:
                if int(records[j].start_frame) > right_limit:
                    break
                gap = _same_stream_gap(records[i], records[j])
                if gap is None or gap < 0 or gap > int(cfg.local_gap):
                    continue
                visual = float(emb[i] @ emb[j])
                if visual < float(cfg.local_app_min):
                    rejected_app += 1
                    continue
                score, parts = _trajectory_link_score(records[i], records[j], visual, cfg)
                if score >= float(cfg.local_thr):
                    candidates.append((score, i, j, parts))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    accepted = 0
    rejected_forbidden = 0
    rejected_size = 0
    score_sum = 0.0
    for score, i, j, _parts in candidates:
        before_a, before_b = uf.find(i), uf.find(j)
        if before_a == before_b:
            continue
        if len(uf.members[before_a]) + len(uf.members[before_b]) > int(cfg.max_component_size):
            rejected_size += 1
            continue
        if not uf.can_merge(i, j, forbidden, int(cfg.max_component_size)):
            rejected_forbidden += 1
            continue
        uf.merge(i, j)
        accepted += 1
        score_sum += float(score)

    labels = uf.labels()
    sizes = Counter(labels.tolist())
    return labels, {
        "local_candidate_edges": int(len(candidates)),
        "local_accepted_edges": int(accepted),
        "local_rejected_app": int(rejected_app),
        "local_rejected_forbidden": int(rejected_forbidden),
        "local_rejected_size": int(rejected_size),
        "local_components": int(len(sizes)),
        "local_largest_component": int(max(sizes.values(), default=0)),
        "local_mean_accepted_score": round(score_sum / max(accepted, 1), 6),
        "uses_ground_truth": False,
    }


def _build_overlap_forbidden(records: list[TrackletRecord]) -> list[set[int]]:
    forbidden = [set() for _ in records]
    stream_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        stream_to_indices[(record.video, record.camera)].append(i)
    for indices in stream_to_indices.values():
        ordered = sorted(indices, key=lambda idx: (records[idx].start_frame, records[idx].end_frame, idx))
        active: list[int] = []
        for j in ordered:
            active = [i for i in active if records[i].end_frame >= records[j].start_frame]
            for i in active:
                forbidden[i].add(j)
                forbidden[j].add(i)
            active.append(j)
    return forbidden


def _topk_edges(nn_scores: np.ndarray, nn_indices: np.ndarray, top_k: int) -> list[tuple[float, int, int]]:
    n = nn_indices.shape[0]
    k = min(max(int(top_k), 1), max(n - 1, 1))
    pairs: dict[tuple[int, int], float] = {}
    for i in range(n):
        kept = 0
        for score, j in zip(nn_scores[i], nn_indices[i]):
            j = int(j)
            if j == i:
                continue
            score = float(score)
            pair = (i, j) if i < j else (j, i)
            if score > pairs.get(pair, -1.0):
                pairs[pair] = score
            kept += 1
            if kept >= k:
                break
    return sorted(((score, i, j) for (i, j), score in pairs.items()), reverse=True)


def _graph_cache(records: list[TrackletRecord], emb: np.ndarray, max_top_k: int) -> GraphCache:
    n = len(records)
    k = min(max(int(max_top_k), 1), max(n - 1, 1))
    sim = (emb @ emb.T).astype(np.float32)
    np.fill_diagonal(sim, -np.inf)
    indices = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(sim, indices, axis=1)
    order = np.argsort(-scores, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    scores = np.take_along_axis(scores, order, axis=1).astype(np.float32)
    return GraphCache(
        nn_scores=scores,
        nn_indices=indices.astype(np.int64),
        forbidden=_build_overlap_forbidden(records),
        edge_cache={},
    )


def _graph_resolve(
    records: list[TrackletRecord],
    emb: np.ndarray,
    cfg: ResolveConfig,
    cache: GraphCache | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    cache = cache or _graph_cache(records, emb, cfg.top_k)
    if cfg.top_k not in cache.edge_cache:
        cache.edge_cache[cfg.top_k] = _topk_edges(cache.nn_scores, cache.nn_indices, cfg.top_k)
    edges = cache.edge_cache[cfg.top_k]
    forbidden = cache.forbidden
    uf = _UnionFind(len(records))
    accepted = 0
    considered = 0
    rejected_threshold = 0
    rejected_forbidden = 0
    rejected_size = 0
    min_possible_score = min(float(cfg.cross_thr), float(cfg.intra_thr) - float(cfg.temporal_bonus))
    for score, i, j in edges:
        if score < min_possible_score:
            break
        considered += 1
        a, b = records[i], records[j]
        gap = _same_stream_gap(a, b)
        is_same_stream = gap is not None
        edge_score = float(score)
        if is_same_stream and gap is not None and gap >= 0 and gap <= cfg.max_gap:
            edge_score = min(1.0, edge_score + float(cfg.temporal_bonus))
        threshold = cfg.intra_thr if is_same_stream else cfg.cross_thr
        if gap == -1 or (is_same_stream and gap is not None and gap > cfg.max_gap):
            rejected_threshold += 1
            continue
        if edge_score < threshold:
            rejected_threshold += 1
            continue
        before_a, before_b = uf.find(i), uf.find(j)
        if before_a == before_b:
            continue
        if len(uf.members[before_a]) + len(uf.members[before_b]) > cfg.max_component_size:
            rejected_size += 1
            continue
        if not uf.can_merge(i, j, forbidden, cfg.max_component_size):
            rejected_forbidden += 1
            continue
        uf.merge(i, j)
        accepted += 1
    labels = uf.labels()
    return labels, {
        "candidate_edges": int(len(edges)),
        "considered_edges": int(considered),
        "accepted_edges": int(accepted),
        "rejected_threshold": int(rejected_threshold),
        "rejected_forbidden": int(rejected_forbidden),
        "rejected_size": int(rejected_size),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "uses_ground_truth": False,
    }


def _group_codes(records: list[TrackletRecord], exclude_same: str, indices: list[int]) -> np.ndarray:
    values = []
    for i in indices:
        record = records[i]
        if exclude_same == "none":
            values.append(f"node:{i}")
        elif exclude_same == "video":
            values.append(record.video)
        elif exclude_same == "stream":
            values.append(f"{record.video}:{record.camera}")
        else:
            values.append(record.camera)
    mapping = {value: idx for idx, value in enumerate(sorted(set(values)))}
    return np.array([mapping[value] for value in values], dtype=np.int64)


def _agglom_resolve(records: list[TrackletRecord], emb: np.ndarray, cfg: ResolveConfig) -> tuple[np.ndarray, dict[str, object]]:
    keep = [i for i, record in enumerate(records) if record.n_dets >= cfg.min_dets]
    labels = np.full(len(records), -1, dtype=np.int64)
    if keep:
        group_codes = _group_codes(records, cfg.exclude_same, keep)
        cannot_link_pairs = None
        if cfg.mode == "agglom_cl":
            keep_pos = {idx: pos for pos, idx in enumerate(keep)}
            forbidden = _build_overlap_forbidden(records)
            pairs = set()
            for i in keep:
                for j in forbidden[i]:
                    if j in keep_pos:
                        a, b = keep_pos[i], keep_pos[j]
                        if a != b:
                            pairs.add((a, b) if a < b else (b, a))
            cannot_link_pairs = sorted(pairs)
        clustered = global_agglom_resolve(
            emb[np.array(keep)],
            group_codes,
            theta=cfg.theta,
            top_k=cfg.top_k,
            exclude_same_cam=(cfg.exclude_same != "none"),
            cannot_link_pairs=cannot_link_pairs,
        )
        for k, i in enumerate(keep):
            labels[i] = int(clustered.labels[k])
        next_label = int(clustered.n_clusters)
        n_edges = int(clustered.n_cand_edges)
    else:
        next_label = 0
        n_edges = 0
    for i in range(len(records)):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
    return labels, {
        "candidate_edges": n_edges,
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "cannot_link_pairs": int(len(cannot_link_pairs or ())),
        "uses_ground_truth": False,
    }


def _time_support_matrix(records: list[TrackletRecord], indices: list[int], window_ms: int) -> np.ndarray:
    n = len(indices)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    starts = np.asarray([records[i].start_abs_ms for i in indices], dtype=np.int64)
    ends = np.asarray([records[i].end_abs_ms for i in indices], dtype=np.int64)
    valid = (starts > 0) & (ends > 0)
    gap = np.maximum(0, np.maximum(starts[:, None] - ends[None, :], starts[None, :] - ends[:, None]))
    support = np.exp(-gap.astype(np.float32) / max(float(window_ms), 1.0)).astype(np.float32)
    support[gap > int(window_ms)] = 0.0
    support[~(valid[:, None] & valid[None, :])] = 0.0
    np.fill_diagonal(support, 0.0)
    return support


def _time_agglom_resolve(records: list[TrackletRecord], emb: np.ndarray, cfg: ResolveConfig) -> tuple[np.ndarray, dict[str, object]]:
    keep = [i for i, record in enumerate(records) if record.n_dets >= cfg.min_dets]
    labels = np.full(len(records), -1, dtype=np.int64)
    if keep:
        group_codes = _group_codes(records, cfg.exclude_same, keep)
        x = _l2n(emb[np.array(keep)].astype(np.float32))
        S = (x @ x.T).astype(np.float32)
        support = _time_support_matrix(records, keep, cfg.time_window_ms)
        if cfg.exclude_same != "none":
            same = group_codes[:, None] == group_codes[None, :]
            support[same] = 0.0
        S += float(cfg.temporal_bonus) * support
        if cfg.exclude_same != "none":
            S[group_codes[:, None] == group_codes[None, :]] = -2.0
        np.fill_diagonal(S, -2.0)
        A, n_edges = _knn_sparse_affinity(S, cfg.top_k)
        D = 1.0 - A
        np.clip(D, 0.0, None, out=D)
        clustered = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(1.0 - cfg.theta),
            metric="precomputed",
            linkage="average",
        ).fit_predict(D)
        _, clustered = np.unique(clustered, return_inverse=True)
        for k, i in enumerate(keep):
            labels[i] = int(clustered[k])
        next_label = int(clustered.max()) + 1 if clustered.size else 0
        positive_time_edges = int(np.count_nonzero(np.triu(support > 0, k=1)))
    else:
        next_label = 0
        n_edges = 0
        positive_time_edges = 0
    for i in range(len(records)):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
    return labels, {
        "candidate_edges": int(n_edges),
        "time_candidate_pairs": int(positive_time_edges),
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "uses_ground_truth": False,
    }


def _agglom_n_resolve(records: list[TrackletRecord], emb: np.ndarray, cfg: ResolveConfig) -> tuple[np.ndarray, dict[str, object]]:
    keep = [i for i, record in enumerate(records) if record.n_dets >= cfg.min_dets]
    labels = np.full(len(records), -1, dtype=np.int64)
    if len(keep) >= 2:
        group_codes = _group_codes(records, cfg.exclude_same, keep)
        target = int(cfg.target_clusters)
        if target <= 0:
            raise ValueError("agglom_n requires target_clusters > 0")
        target = min(max(1, target), len(keep))
        A, ix, _jx = build_knn_cosine_affinity(
            emb[np.array(keep)],
            group_codes,
            top_k=cfg.top_k,
            exclude_same_cam=(cfg.exclude_same != "none"),
        )
        D = 1.0 - A
        np.clip(D, 0.0, None, out=D)
        clustered = AgglomerativeClustering(
            n_clusters=target,
            distance_threshold=None,
            metric="precomputed",
            linkage="average",
        ).fit_predict(D)
        _, clustered = np.unique(clustered, return_inverse=True)
        for k, i in enumerate(keep):
            labels[i] = int(clustered[k])
        next_label = int(clustered.max()) + 1 if clustered.size else 0
        n_edges = int(len(ix))
        effective_target = int(target)
    elif keep:
        labels[keep[0]] = 0
        next_label = 1
        n_edges = 0
        effective_target = 1
    else:
        next_label = 0
        n_edges = 0
        effective_target = 0
    for i in range(len(records)):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
    return labels, {
        "candidate_edges": n_edges,
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "effective_target_clusters": int(effective_target),
        "uses_ground_truth": False,
    }


def _rerank_distance_matrix(
    emb: np.ndarray,
    group_codes: np.ndarray,
    *,
    exclude_same_group: bool,
    k1: int,
    k2: int,
    lambda_value: float,
) -> tuple[np.ndarray, int]:
    x = _l2n(np.asarray(emb, dtype=np.float32))
    n = int(x.shape[0])
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32), 0

    k1 = min(max(int(k1), 1), n - 1)
    k2 = min(max(int(k2), 1), n)
    lam = float(lambda_value)
    sim = np.clip(x @ x.T, -1.0, 1.0).astype(np.float32)
    same_group = None
    if exclude_same_group:
        group_codes = np.asarray(group_codes)
        same_group = group_codes[:, None] == group_codes[None, :]
        sim[same_group] = -1.0
    np.fill_diagonal(sim, 1.0)
    original_dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float32)
    rank = np.argsort(original_dist, axis=1).astype(np.int32)

    V = np.zeros((n, n), dtype=np.float32)
    half_k = max(1, int(np.around(k1 / 2)))
    accepted = 0
    for i in range(n):
        forward = rank[i, : k1 + 1]
        backward = rank[forward, : k1 + 1]
        reciprocal = forward[np.any(backward == i, axis=1)]
        reciprocal_exp = set(map(int, reciprocal.tolist()))
        for candidate in reciprocal.tolist():
            candidate_forward = rank[int(candidate), : half_k + 1]
            candidate_backward = rank[candidate_forward, : half_k + 1]
            candidate_recip = candidate_forward[np.any(candidate_backward == int(candidate), axis=1)]
            if len(candidate_recip) == 0:
                continue
            overlap = len(set(map(int, candidate_recip.tolist())) & reciprocal_exp)
            if overlap > (2.0 / 3.0) * len(candidate_recip):
                reciprocal_exp.update(map(int, candidate_recip.tolist()))
        inds = np.fromiter(sorted(reciprocal_exp), dtype=np.int32)
        weights = np.exp(-original_dist[i, inds]).astype(np.float32)
        denom = float(weights.sum())
        if denom > 0:
            V[i, inds] = weights / denom
            accepted += int(len(inds))

    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(n):
            V_qe[i, :] = V[rank[i, :k2], :].mean(axis=0)
        V = V_qe

    inv_index = [np.flatnonzero(V[:, i] > 0) for i in range(n)]
    jaccard = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        temp_min = np.zeros(n, dtype=np.float32)
        nonzero = np.flatnonzero(V[i] > 0)
        for ind in nonzero:
            related = inv_index[int(ind)]
            temp_min[related] += np.minimum(V[i, int(ind)], V[related, int(ind)])
        jaccard[i] = 1.0 - temp_min / np.maximum(2.0 - temp_min, 1e-12)
    final_dist = (1.0 - lam) * jaccard + lam * np.clip(original_dist, 0.0, 1.0)
    if same_group is not None:
        final_dist[same_group] = 1.0
    np.fill_diagonal(final_dist, 0.0)
    return final_dist.astype(np.float32), int(accepted)


def _rerank_agglom_resolve(records: list[TrackletRecord], emb: np.ndarray, cfg: ResolveConfig) -> tuple[np.ndarray, dict[str, object]]:
    keep = [i for i, record in enumerate(records) if record.n_dets >= cfg.min_dets]
    labels = np.full(len(records), -1, dtype=np.int64)
    if len(keep) >= 2:
        group_codes = _group_codes(records, cfg.exclude_same, keep)
        cache_key = (
            id(emb),
            tuple(keep),
            cfg.exclude_same,
            int(cfg.rerank_k1),
            int(cfg.rerank_k2),
            round(float(cfg.rerank_lambda), 6),
        )
        cached = _RERANK_DISTANCE_CACHE.get(cache_key)
        if cached is None:
            D, accepted = _rerank_distance_matrix(
                emb[np.array(keep)],
                group_codes,
                exclude_same_group=(cfg.exclude_same != "none"),
                k1=cfg.rerank_k1,
                k2=cfg.rerank_k2,
                lambda_value=cfg.rerank_lambda,
            )
            _RERANK_DISTANCE_CACHE[cache_key] = (D, accepted)
        else:
            D, accepted = cached
        clustered = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(1.0 - cfg.theta),
            metric="precomputed",
            linkage="average",
        ).fit_predict(D)
        _, clustered = np.unique(clustered, return_inverse=True)
        for k, i in enumerate(keep):
            labels[i] = int(clustered[k])
        next_label = int(clustered.max()) + 1 if clustered.size else 0
    elif keep:
        labels[keep[0]] = 0
        next_label = 1
        accepted = 0
    else:
        next_label = 0
        accepted = 0
    for i in range(len(records)):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
    return labels, {
        "candidate_edges": int(accepted),
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "uses_ground_truth": False,
    }


def _assign_singletons(labels: np.ndarray, start_label: int) -> tuple[np.ndarray, int]:
    next_label = int(start_label)
    out = labels.copy()
    for i in range(len(out)):
        if out[i] < 0:
            out[i] = next_label
            next_label += 1
    return out, next_label


def _local_mustlink_resolve(records: list[TrackletRecord], emb: np.ndarray, cfg: ResolveConfig) -> tuple[np.ndarray, dict[str, object]]:
    local_ids, local_info = _local_component_labels(records, emb, cfg)
    keep = [i for i, record in enumerate(records) if record.n_dets >= cfg.min_dets]
    labels = np.full(len(records), -1, dtype=np.int64)
    if keep:
        group_codes = _group_codes(records, cfg.exclude_same, keep)
        clustered = mustlink_resolve(
            emb[np.array(keep)],
            group_codes,
            local_ids[np.array(keep)],
            theta=cfg.theta,
            top_k=cfg.top_k,
        )
        for k, i in enumerate(keep):
            labels[i] = int(clustered.labels[k])
        next_label = int(clustered.n_clusters)
        n_edges = int(clustered.n_cand_edges)
    else:
        next_label = 0
        n_edges = 0
    labels, next_label = _assign_singletons(labels, next_label)
    info = {
        **local_info,
        "candidate_edges": n_edges,
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "uses_ground_truth": False,
    }
    return labels, info


def _group_value(record: TrackletRecord, exclude_same: str, fallback: int) -> str:
    if exclude_same == "none":
        return f"component:{fallback}"
    if exclude_same == "video":
        return record.video
    if exclude_same == "stream":
        return f"{record.video}:{record.camera}"
    return record.camera


def _local_bank_resolve(records: list[TrackletRecord], emb: np.ndarray, cfg: ResolveConfig) -> tuple[np.ndarray, dict[str, object]]:
    local_ids, local_info = _local_component_labels(records, emb, cfg)
    keep = [i for i, record in enumerate(records) if record.n_dets >= cfg.min_dets]
    labels = np.full(len(records), -1, dtype=np.int64)
    if keep:
        comp_to_indices: dict[int, list[int]] = defaultdict(list)
        for i in keep:
            comp_to_indices[int(local_ids[i])].append(i)
        comp_items = sorted(comp_to_indices.items(), key=lambda item: min(item[1]))
        group_values = [_group_value(records[indices[0]], cfg.exclude_same, comp_id) for comp_id, indices in comp_items]
        group_map = {value: idx for idx, value in enumerate(sorted(set(group_values)))}
        group_codes = np.array([group_map[value] for value in group_values], dtype=np.int64)
        banks = [emb[np.array(indices)] for _comp_id, indices in comp_items]
        clustered = two_tier_resolve(
            banks,
            group_codes,
            theta=cfg.theta,
            mode="bank",
            top_k=cfg.top_k,
            link="average",
        )
        for local_index, (_comp_id, indices) in enumerate(comp_items):
            for i in indices:
                labels[i] = int(clustered.labels[local_index])
        next_label = int(clustered.n_clusters)
        n_edges = int(clustered.n_cand_edges)
    else:
        next_label = 0
        n_edges = 0
    labels, next_label = _assign_singletons(labels, next_label)
    info = {
        **local_info,
        "candidate_edges": n_edges,
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "uses_ground_truth": False,
    }
    return labels, info


def _component_centroids(
    records: list[TrackletRecord],
    emb: np.ndarray,
    local_ids: np.ndarray,
    keep: list[int],
    exclude_same: str,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    comp_to_indices: dict[int, list[int]] = defaultdict(list)
    for i in keep:
        comp_to_indices[int(local_ids[i])].append(i)
    comp_items = sorted(comp_to_indices.items(), key=lambda item: min(item[1]))
    components = [indices for _comp_id, indices in comp_items]
    cents = []
    group_values = []
    for local_idx, indices in enumerate(components):
        v = emb[np.array(indices)].mean(axis=0)
        cents.append(v / (np.linalg.norm(v) + 1e-9))
        group_values.append(_group_value(records[indices[0]], exclude_same, local_idx))
    group_map = {value: idx for idx, value in enumerate(sorted(set(group_values)))}
    group_codes = np.array([group_map[value] for value in group_values], dtype=np.int64)
    return components, np.stack(cents).astype(np.float32), group_codes


def _cc_labels_from_edges(
    n_nodes: int,
    edges: list[tuple[float, int, int]],
    sizes: list[int],
    max_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    uf = _UnionFind(n_nodes)
    root_sizes = {i: int(sizes[i]) for i in range(n_nodes)}
    accepted = 0
    rejected_size = 0
    for _score, i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if root_sizes[ri] + root_sizes[rj] > int(max_size):
            rejected_size += 1
            continue
        uf.merge(i, j)
        root = uf.find(i)
        other = rj if root == ri else ri
        root_sizes[root] = root_sizes.get(ri, sizes[i]) + root_sizes.get(rj, sizes[j])
        root_sizes.pop(other, None)
        accepted += 1
    return uf.labels(), {"accepted_edges": int(accepted), "rejected_size": int(rejected_size)}


def _local_cc_resolve(records: list[TrackletRecord], emb: np.ndarray, cfg: ResolveConfig) -> tuple[np.ndarray, dict[str, object]]:
    local_ids, local_info = _local_component_labels(records, emb, cfg)
    keep = [i for i, record in enumerate(records) if record.n_dets >= cfg.min_dets]
    labels = np.full(len(records), -1, dtype=np.int64)
    if keep:
        components, cents, group_codes = _component_centroids(records, emb, local_ids, keep, cfg.exclude_same)
        n = len(components)
        if n > 1:
            sim = cents @ cents.T
            same = group_codes[:, None] == group_codes[None, :]
            sim[same] = -2.0
            np.fill_diagonal(sim, -2.0)
            k = min(max(int(cfg.top_k), 1), max(n - 1, 1))
            topidx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            pairs: dict[tuple[int, int], float] = {}
            for i in range(n):
                for j in topidx[i]:
                    j = int(j)
                    score = float(sim[i, j])
                    if score < float(cfg.cross_thr):
                        continue
                    pair = (i, j) if i < j else (j, i)
                    if score > pairs.get(pair, -1.0):
                        pairs[pair] = score
            edges = sorted(((score, i, j) for (i, j), score in pairs.items()), reverse=True)
            comp_labels, edge_info = _cc_labels_from_edges(
                n,
                edges,
                [len(indices) for indices in components],
                cfg.max_component_size,
            )
            n_edges = len(edges)
        else:
            comp_labels = np.zeros(n, dtype=np.int64)
            edge_info = {"accepted_edges": 0, "rejected_size": 0}
            n_edges = 0
        for comp_index, indices in enumerate(components):
            for i in indices:
                labels[i] = int(comp_labels[comp_index])
        next_label = int(comp_labels.max()) + 1 if len(comp_labels) else 0
    else:
        next_label = 0
        n_edges = 0
        edge_info = {"accepted_edges": 0, "rejected_size": 0}
    labels, next_label = _assign_singletons(labels, next_label)
    info = {
        **local_info,
        "candidate_edges": int(n_edges),
        "accepted_edges": int(edge_info["accepted_edges"]),
        "rejected_size": int(edge_info["rejected_size"]),
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "uses_ground_truth": False,
    }
    return labels, info


def _labels_to_seq_map(
    records: list[TrackletRecord],
    labels: np.ndarray,
    offset: int = 30_000_000,
    keep_seqs: set[int] | None = None,
) -> dict[int, int]:
    return {
        record.seq: int(offset + int(label))
        for record, label in zip(records, labels)
        if keep_seqs is None or int(record.seq) in keep_seqs
    }


def _build_comp(pred_by_video: dict[str, pd.DataFrame], pred_by_seq: dict[int, int]) -> dict[str, pd.DataFrame]:
    comp: dict[str, pd.DataFrame] = {}
    for video, pred in pred_by_video.items():
        out = pred.copy()
        out = out[out["seq"].map(lambda seq: int(seq) in pred_by_seq)].copy()
        if out.empty:
            comp[video] = out[["frame", "seq", "x1", "y1", "x2", "y2", "object_type", "confidence"]].drop(columns=["seq"])
            comp[video]["id"] = pd.Series(dtype=np.int64)
            comp[video] = comp[video][["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]]
            continue
        out["id"] = [int(pred_by_seq[int(seq)]) for seq in out["seq"]]
        comp[video] = out[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
    return comp


def _score_full(
    pred_by_video: dict[str, pd.DataFrame],
    gt_by_video: dict[str, pd.DataFrame],
    pred_by_seq: dict[int, int],
) -> dict[str, object]:
    comp = _build_comp(pred_by_video, pred_by_seq)
    keys = {key for key in gt_by_video if key in comp}
    metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    return {
        "idf1": round(float(metrics.idf1), 6),
        "hota": round(float(metrics.hota), 6),
        "assa": round(float(metrics.assa), 6),
        "deta": round(float(metrics.deta), 6),
        "detre": round(float(metrics.detre), 6),
        "detpr": round(float(metrics.detpr), 6),
        "unmatched_fp": int(metrics.unmatched_fp),
        "per_video": {
            key: {metric: round(float(value), 6) for metric, value in vals.items()}
            for key, vals in sorted(metrics.per_video.items())
        },
    }


def _parse_video_area_thresholds(text: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not text:
        return out
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition(":")
        if not sep:
            raise ValueError(f"bad --output-min-area-by-video entry {part!r}; expected video:min_area")
        out[key] = float(value)
    return out


def _tracklet_quality_score(record: TrackletRecord) -> float:
    area = max(float(record.width) * float(record.height), 1.0)
    aspect = max(float(record.height), 1.0) / max(float(record.width), 1.0)
    det_term = np.log1p(max(int(record.n_dets), 0)) / np.log1p(120.0)
    area_term = np.log1p(area) / np.log1p(50_000.0)
    conf_term = float(np.clip(record.avg_conf, 0.0, 1.0))
    aspect_penalty = min(abs(np.log(max(aspect, 1e-6) / 2.4)) / 1.25, 1.0)
    tiny_penalty = min(max(0.0, np.log(80.0 / max(float(record.height), 1.0))) / 2.0, 1.0)
    return float(0.30 * det_term + 0.35 * area_term + 0.25 * conf_term - 0.06 * aspect_penalty - 0.04 * tiny_penalty)


def _quantile_thresholds_by_video(
    records: list[TrackletRecord],
    values: dict[int, float],
    global_quantile: float,
    by_video_quantile: dict[str, float],
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    if global_quantile > 0.0:
        arr = np.asarray([values[int(record.seq)] for record in records], dtype=np.float32)
        thresholds["*"] = float(np.quantile(arr, min(max(float(global_quantile), 0.0), 1.0)))
    videos = sorted({record.video for record in records})
    for video in videos:
        q = by_video_quantile.get(video)
        if q is None:
            continue
        video_values = np.asarray([values[int(record.seq)] for record in records if record.video == video], dtype=np.float32)
        if video_values.size:
            thresholds[video] = float(np.quantile(video_values, min(max(float(q), 0.0), 1.0)))
    return thresholds


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _auto_anomaly_thresholds(
    records: list[TrackletRecord],
    area_by_seq: dict[int, float],
    quality_by_seq: dict[int, float],
    args,
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    if not bool(getattr(args, "output_auto_anomaly_admission", False)):
        return {}, {}, {"enabled": False}
    metric = str(getattr(args, "output_auto_anomaly_metric", "quality"))
    q = min(max(float(getattr(args, "output_auto_anomaly_quantile", 0.75)), 0.0), 1.0)
    area_ratio = float(getattr(args, "output_auto_anomaly_area_ratio", 0.60))
    quality_mad = float(getattr(args, "output_auto_anomaly_quality_mad", 1.0))
    min_tracklets = int(getattr(args, "output_auto_anomaly_min_video_tracklets", 20))
    max_videos = int(getattr(args, "output_auto_anomaly_max_videos", 3))

    by_video: dict[str, list[TrackletRecord]] = defaultdict(list)
    for record in records:
        by_video[record.video].append(record)

    stats: dict[str, dict[str, float | int]] = {}
    for video, video_records in sorted(by_video.items()):
        areas = np.asarray([area_by_seq[int(record.seq)] for record in video_records], dtype=np.float32)
        qualities = np.asarray([quality_by_seq[int(record.seq)] for record in video_records], dtype=np.float32)
        stats[video] = {
            "n": int(len(video_records)),
            "median_area": round(float(np.median(areas)) if areas.size else 0.0, 6),
            "median_quality": round(float(np.median(qualities)) if qualities.size else 0.0, 6),
            "median_conf": round(float(np.median([record.avg_conf for record in video_records])) if video_records else 0.0, 6),
            "median_dets": round(float(np.median([record.n_dets for record in video_records])) if video_records else 0.0, 6),
        }

    eligible = [video for video, value in stats.items() if int(value["n"]) >= min_tracklets]
    if not eligible:
        return {}, {}, {
            "enabled": True,
            "metric": metric,
            "quantile": q,
            "reason": "no_eligible_videos",
            "video_stats": stats,
        }

    area_medians = np.asarray([float(stats[video]["median_area"]) for video in eligible], dtype=np.float32)
    quality_medians = np.asarray([float(stats[video]["median_quality"]) for video in eligible], dtype=np.float32)
    global_area_median = float(np.median(area_medians))
    global_quality_median = float(np.median(quality_medians))
    quality_mad_value = max(_mad(quality_medians), 1.0e-6)

    flagged: list[tuple[float, str, list[str]]] = []
    for video in eligible:
        reasons: list[str] = []
        severity = 0.0
        median_area = float(stats[video]["median_area"])
        median_quality = float(stats[video]["median_quality"])
        if metric in {"area", "both"} and global_area_median > 0.0:
            ratio = median_area / global_area_median
            if ratio < area_ratio:
                reasons.append("low_area_median")
                severity = max(severity, (area_ratio - ratio) / max(area_ratio, 1.0e-6))
        if metric in {"quality", "both"}:
            z = (global_quality_median - median_quality) / quality_mad_value
            if z > quality_mad:
                reasons.append("low_quality_median")
                severity = max(severity, z / max(quality_mad, 1.0e-6))
        if reasons:
            flagged.append((severity, video, reasons))
    flagged.sort(reverse=True)
    if max_videos > 0:
        flagged = flagged[:max_videos]

    area_thresholds: dict[str, float] = {}
    quality_thresholds: dict[str, float] = {}
    flagged_info: dict[str, dict[str, object]] = {}
    for severity, video, reasons in flagged:
        video_records = by_video[video]
        if metric in {"area", "both"}:
            values = np.asarray([area_by_seq[int(record.seq)] for record in video_records], dtype=np.float32)
            area_thresholds[video] = float(np.quantile(values, q)) if values.size else 0.0
        if metric in {"quality", "both"}:
            values = np.asarray([quality_by_seq[int(record.seq)] for record in video_records], dtype=np.float32)
            quality_thresholds[video] = float(np.quantile(values, q)) if values.size else -1.0e9
        flagged_info[video] = {
            "severity": round(float(severity), 6),
            "reasons": reasons,
            **stats[video],
        }

    return area_thresholds, quality_thresholds, {
        "enabled": True,
        "metric": metric,
        "quantile": q,
        "area_ratio": area_ratio,
        "quality_mad": quality_mad,
        "min_video_tracklets": min_tracklets,
        "max_videos": max_videos,
        "global_area_median": round(global_area_median, 6),
        "global_quality_median": round(global_quality_median, 6),
        "quality_mad_value": round(quality_mad_value, 6),
        "flagged_videos": flagged_info,
        "video_stats": stats,
    }


def _output_keep_seqs(records: list[TrackletRecord], args) -> tuple[set[int], dict[str, object]]:
    min_dets = int(args.output_min_dets)
    min_conf = float(args.output_min_conf)
    min_area = float(args.output_min_area)
    video_min_area = _parse_video_area_thresholds(args.output_min_area_by_video)
    area_by_seq = {int(record.seq): float(record.width) * float(record.height) for record in records}
    quality_by_seq = {int(record.seq): _tracklet_quality_score(record) for record in records}
    area_quantiles = _parse_video_area_thresholds(getattr(args, "output_drop_area_quantile_by_video", ""))
    quality_quantiles = _parse_video_area_thresholds(getattr(args, "output_drop_quality_quantile_by_video", ""))
    area_thresholds = _quantile_thresholds_by_video(
        records,
        area_by_seq,
        float(getattr(args, "output_drop_area_quantile", 0.0)),
        area_quantiles,
    )
    quality_thresholds = _quantile_thresholds_by_video(
        records,
        quality_by_seq,
        float(getattr(args, "output_drop_quality_quantile", 0.0)),
        quality_quantiles,
    )
    auto_area_thresholds, auto_quality_thresholds, auto_info = _auto_anomaly_thresholds(
        records,
        area_by_seq,
        quality_by_seq,
        args,
    )
    for video, threshold in auto_area_thresholds.items():
        area_thresholds[video] = max(float(area_thresholds.get(video, 0.0)), float(threshold))
    for video, threshold in auto_quality_thresholds.items():
        quality_thresholds[video] = max(float(quality_thresholds.get(video, -1.0e9)), float(threshold))
    min_quality = float(getattr(args, "output_min_quality", -1.0e9))
    keep: set[int] = set()
    by_video_total = Counter(record.video for record in records)
    by_video_keep = Counter()
    drop_reasons = Counter()
    for record in records:
        area_thr = max(
            min_area,
            float(video_min_area.get(record.video, 0.0)),
            float(area_thresholds.get("*", 0.0)),
            float(area_thresholds.get(record.video, 0.0)),
        )
        quality_thr = max(
            min_quality,
            float(quality_thresholds.get("*", -1.0e9)),
            float(quality_thresholds.get(record.video, -1.0e9)),
        )
        area = float(area_by_seq[int(record.seq)])
        quality = float(quality_by_seq[int(record.seq)])
        if int(record.n_dets) < min_dets:
            drop_reasons["min_dets"] += 1
            continue
        if float(record.avg_conf) < min_conf:
            drop_reasons["min_conf"] += 1
            continue
        if area < area_thr:
            drop_reasons["min_area"] += 1
            continue
        if quality < quality_thr:
            drop_reasons["min_quality"] += 1
            continue
        keep.add(int(record.seq))
        by_video_keep[record.video] += 1
    info = {
        "output_min_dets": min_dets,
        "output_min_conf": min_conf,
        "output_min_area": min_area,
        "output_min_quality": min_quality,
        "output_min_area_by_video": video_min_area,
        "output_drop_area_quantile": float(getattr(args, "output_drop_area_quantile", 0.0)),
        "output_drop_area_quantile_by_video": area_quantiles,
        "output_drop_area_thresholds": area_thresholds,
        "output_drop_quality_quantile": float(getattr(args, "output_drop_quality_quantile", 0.0)),
        "output_drop_quality_quantile_by_video": quality_quantiles,
        "output_drop_quality_thresholds": quality_thresholds,
        "output_auto_anomaly": auto_info,
        "output_tracklets": int(len(keep)),
        "output_filtered_tracklets": int(len(records) - len(keep)),
        "output_drop_reasons": dict(drop_reasons),
        "output_by_video": {
            video: {
                "keep": int(by_video_keep.get(video, 0)),
                "total": int(total),
                "drop": int(total - by_video_keep.get(video, 0)),
            }
            for video, total in sorted(by_video_total.items())
        },
    }
    return keep, info


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _config_grid(args) -> list[ResolveConfig]:
    configs: list[ResolveConfig] = []
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    agglom_modes = [mode for mode in modes if mode in {"agglom", "agglom_cl", "time_agglom"}]
    for mode in agglom_modes:
        for theta in _parse_float_list(args.thetas):
            for top_k in _parse_int_list(args.top_ks):
                for min_dets in _parse_int_list(args.min_dets):
                    for exclude_same in [part.strip() for part in args.exclude_same.split(",") if part.strip()]:
                        for time_window_ms in (_parse_int_list(args.time_windows_ms) if mode == "time_agglom" else [0]):
                            configs.append(
                                ResolveConfig(
                                    mode=mode,
                                    theta=theta,
                                    top_k=top_k,
                                    min_dets=min_dets,
                                    exclude_same=exclude_same,
                                    temporal_bonus=float(args.temporal_bonus),
                                    time_window_ms=int(time_window_ms),
                                )
                            )
    if "agglom_n" in modes:
        for target_clusters in _parse_int_list(args.target_clusters):
            for top_k in _parse_int_list(args.top_ks):
                for min_dets in _parse_int_list(args.min_dets):
                    for exclude_same in [part.strip() for part in args.exclude_same.split(",") if part.strip()]:
                        configs.append(
                            ResolveConfig(
                                mode="agglom_n",
                                top_k=top_k,
                                min_dets=min_dets,
                                exclude_same=exclude_same,
                                target_clusters=target_clusters,
                            )
                        )
    if "rerank_agglom" in modes:
        for theta in _parse_float_list(args.thetas):
            for min_dets in _parse_int_list(args.min_dets):
                for exclude_same in [part.strip() for part in args.exclude_same.split(",") if part.strip()]:
                    for rerank_k1 in _parse_int_list(args.rerank_k1):
                        for rerank_k2 in _parse_int_list(args.rerank_k2):
                            for rerank_lambda in _parse_float_list(args.rerank_lambdas):
                                configs.append(
                                    ResolveConfig(
                                        mode="rerank_agglom",
                                        theta=theta,
                                        min_dets=min_dets,
                                        exclude_same=exclude_same,
                                        rerank_k1=rerank_k1,
                                        rerank_k2=rerank_k2,
                                        rerank_lambda=rerank_lambda,
                                    )
                                )
    if "graph" in modes:
        for cross_thr in _parse_float_list(args.cross_thrs):
            for intra_thr in _parse_float_list(args.intra_thrs):
                for top_k in _parse_int_list(args.top_ks):
                    for max_gap in _parse_int_list(args.max_gaps):
                        for max_component_size in _parse_int_list(args.max_component_sizes):
                            configs.append(
                                ResolveConfig(
                                    mode="graph",
                                    top_k=top_k,
                                    cross_thr=cross_thr,
                                    intra_thr=intra_thr,
                                    max_gap=max_gap,
                                    max_component_size=max_component_size,
                                    temporal_bonus=args.temporal_bonus,
                                )
                            )
    local_modes = [mode for mode in modes if mode in {"local_mustlink", "local_bank"}]
    for mode in local_modes:
        for theta in _parse_float_list(args.thetas):
            for top_k in _parse_int_list(args.top_ks):
                for min_dets in _parse_int_list(args.min_dets):
                    for exclude_same in [part.strip() for part in args.exclude_same.split(",") if part.strip()]:
                        for local_gap in _parse_int_list(args.local_gaps):
                            for local_thr in _parse_float_list(args.local_thrs):
                                for local_app_min in _parse_float_list(args.local_app_mins):
                                    for local_pos_scale in _parse_float_list(args.local_pos_scales):
                                        configs.append(
                                            ResolveConfig(
                                                mode=mode,
                                                theta=theta,
                                                top_k=top_k,
                                                min_dets=min_dets,
                                                exclude_same=exclude_same,
                                                local_gap=local_gap,
                                                local_thr=local_thr,
                                                local_app_min=local_app_min,
                                                local_pos_scale=local_pos_scale,
                                            )
                                        )
    if "local_cc" in modes:
        for cross_thr in _parse_float_list(args.cross_thrs):
            for top_k in _parse_int_list(args.top_ks):
                for min_dets in _parse_int_list(args.min_dets):
                    for exclude_same in [part.strip() for part in args.exclude_same.split(",") if part.strip()]:
                        for max_component_size in _parse_int_list(args.max_component_sizes):
                            for local_gap in _parse_int_list(args.local_gaps):
                                for local_thr in _parse_float_list(args.local_thrs):
                                    for local_app_min in _parse_float_list(args.local_app_mins):
                                        for local_pos_scale in _parse_float_list(args.local_pos_scales):
                                            configs.append(
                                                ResolveConfig(
                                                    mode="local_cc",
                                                    top_k=top_k,
                                                    min_dets=min_dets,
                                                    exclude_same=exclude_same,
                                                    cross_thr=cross_thr,
                                                    max_component_size=max_component_size,
                                                    local_gap=local_gap,
                                                    local_thr=local_thr,
                                                    local_app_min=local_app_min,
                                                    local_pos_scale=local_pos_scale,
                                                )
                                            )
    return configs


def _row_selection_score(row: dict[str, object], key: str) -> float:
    def val(name: str) -> float:
        try:
            return float(row.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0

    if key == "pair_pr_min":
        return min(val("tracklet_pair_precision"), val("tracklet_pair_recall"))
    if key == "pair_gate_margin":
        return min(val("tracklet_pair_f1"), val("tracklet_pair_precision"), val("tracklet_pair_recall"))
    if key == "pair_mean_f1_pr":
        return (val("tracklet_pair_f1") + val("tracklet_pair_precision") + val("tracklet_pair_recall")) / 3.0
    return val(key)


def _select_full_score_rows(rows: list[dict[str, object]], full_n: int, sort_key: str, selection_keys: str) -> list[dict[str, object]]:
    limit = max(int(full_n), 0)
    if limit <= 0 or not rows:
        return []
    keys = [part.strip() for part in str(selection_keys or "").split(",") if part.strip()]
    if not keys:
        keys = [str(sort_key)]
    if str(sort_key) not in keys:
        keys.insert(0, str(sort_key))

    order_by_key = [
        sorted(range(len(rows)), key=lambda idx, key=key: _row_selection_score(rows[idx], key), reverse=True)
        for key in keys
    ]
    selected: list[int] = []
    seen: set[int] = set()
    max_len = max((len(order) for order in order_by_key), default=0)
    for rank in range(max_len):
        for order in order_by_key:
            if rank >= len(order):
                continue
            idx = int(order[rank])
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)
            if len(selected) >= limit:
                return [rows[idx] for idx in selected]
    return [rows[idx] for idx in selected]


def _run_resolver(
    records: list[TrackletRecord],
    emb: np.ndarray,
    cfg: ResolveConfig,
    graph_cache: GraphCache | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if cfg.mode in {"agglom", "agglom_cl"}:
        return _agglom_resolve(records, emb, cfg)
    if cfg.mode == "time_agglom":
        return _time_agglom_resolve(records, emb, cfg)
    if cfg.mode == "agglom_n":
        return _agglom_n_resolve(records, emb, cfg)
    if cfg.mode == "rerank_agglom":
        return _rerank_agglom_resolve(records, emb, cfg)
    if cfg.mode == "graph":
        return _graph_resolve(records, emb, cfg, graph_cache)
    if cfg.mode == "local_mustlink":
        return _local_mustlink_resolve(records, emb, cfg)
    if cfg.mode == "local_bank":
        return _local_bank_resolve(records, emb, cfg)
    if cfg.mode == "local_cc":
        return _local_cc_resolve(records, emb, cfg)
    raise ValueError(f"unknown mode: {cfg.mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1", choices=["ds1"])
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve", choices=["resolve", "match"])
    ap.add_argument("--feature-npz", default=None, help="optional no-anchor external tracklet features keyed by seq")
    ap.add_argument("--concat-db-embedding", action="store_true", help="concatenate --feature-npz with DB role embedding")
    ap.add_argument("--db-weight", type=float, default=1.0, help="weight for DB embedding when concatenating")
    ap.add_argument("--feature-weight", type=float, default=1.0, help="weight for external feature embedding")
    ap.add_argument("--nfc-k1", type=int, default=0, help="run no-anchor mutual-neighbor feature centralization with this k1")
    ap.add_argument("--nfc-k2", type=int, default=2)
    ap.add_argument("--nfc-eta", type=float, default=1.0)
    ap.add_argument("--nfc-exclude-same", default="none", choices=["none", "camera", "stream", "video"])
    ap.add_argument("--modes", default="agglom,graph")
    ap.add_argument("--thetas", default="0.00,0.01,0.02,0.04,0.06,0.08,0.10,0.14,0.18")
    ap.add_argument("--top-ks", default="15,30,50")
    ap.add_argument("--min-dets", default="1,5,10,20")
    ap.add_argument("--exclude-same", default="camera,stream,video,none")
    ap.add_argument("--cross-thrs", default="0.50,0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--intra-thrs", default="0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--max-gaps", default="120,300,600,1200")
    ap.add_argument("--max-component-sizes", default="40,80,160")
    ap.add_argument("--temporal-bonus", type=float, default=0.04)
    ap.add_argument("--time-windows-ms", default="250,500,1000")
    ap.add_argument("--local-gaps", default="30,60,120")
    ap.add_argument("--local-thrs", default="0.72,0.78,0.84")
    ap.add_argument("--local-app-mins", default="0.45,0.50,0.55")
    ap.add_argument("--local-pos-scales", default="1.0,1.5,2.0")
    ap.add_argument("--target-clusters", default="36,48,64,80,100,128,160,220,320,480,640")
    ap.add_argument("--rerank-k1", default="20")
    ap.add_argument("--rerank-k2", default="6")
    ap.add_argument("--rerank-lambdas", default="0.3")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--eval-min-matches", type=int, default=1)
    ap.add_argument("--eval-min-purity", type=float, default=0.0)
    ap.add_argument("--eval-cache", default=None, help="optional NPZ cache for GT-matched eval tracklet labels")
    ap.add_argument("--output-min-dets", type=int, default=1, help="M3 admission: only submit tracklets with at least this many detections")
    ap.add_argument("--output-min-conf", type=float, default=0.0, help="M3 admission: only submit tracklets with avg detector confidence >= this")
    ap.add_argument("--output-min-area", type=float, default=0.0, help="M3 admission: only submit tracklets with avg bbox area >= this")
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9, help="M3 admission: only submit tracklets with no-GT quality score >= this")
    ap.add_argument("--output-drop-area-quantile", type=float, default=0.0, help="M3 admission: drop tracklets below this global bbox-area quantile")
    ap.add_argument("--output-drop-area-quantile-by-video", default="", help="comma list of video:quantile area drops")
    ap.add_argument("--output-drop-quality-quantile", type=float, default=0.0, help="M3 admission: drop tracklets below this global quality quantile")
    ap.add_argument("--output-drop-quality-quantile-by-video", default="", help="comma list of video:quantile quality drops")
    ap.add_argument("--output-auto-anomaly-admission", action="store_true", help="M3 admission: auto-detect low-quality videos from no-GT tracklet evidence")
    ap.add_argument("--output-auto-anomaly-metric", default="quality", choices=["area", "quality", "both"])
    ap.add_argument("--output-auto-anomaly-quantile", type=float, default=0.75, help="drop below this within-video quantile for auto-flagged videos")
    ap.add_argument("--output-auto-anomaly-area-ratio", type=float, default=0.60, help="flag videos whose median area is below this ratio of the global video median")
    ap.add_argument("--output-auto-anomaly-quality-mad", type=float, default=1.0, help="flag videos this many MADs below global median quality")
    ap.add_argument("--output-auto-anomaly-min-video-tracklets", type=int, default=20)
    ap.add_argument("--output-auto-anomaly-max-videos", type=int, default=3)
    ap.add_argument(
        "--output-min-area-by-video",
        default="",
        help="comma list of video:min_area overrides; combined with --output-min-area by max()",
    )
    ap.add_argument("--full-top-n", type=int, default=5, help="run full HOTA for top-N fast configs")
    ap.add_argument(
        "--full-selection-keys",
        default="",
        help="comma list of fast metrics used round-robin to choose configs for full scoring; supports pair_pr_min, pair_gate_margin, pair_mean_f1_pr",
    )
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--limit-configs", type=int, default=0)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, emb = _load_tracklets(con, args.role)
    print(json.dumps({"stage": "loaded_tracklets", "n_tracklets": len(records), "emb_dim": int(emb.shape[1])}), flush=True)
    if args.feature_npz:
        emb = _load_feature_npz(
            args.feature_npz,
            records,
            emb,
            concat_db=args.concat_db_embedding,
            db_weight=float(args.db_weight),
            feature_weight=float(args.feature_weight),
        )
        print(
            json.dumps(
                {
                    "stage": "loaded_feature_npz",
                    "feature_npz": args.feature_npz,
                    "concat_db_embedding": bool(args.concat_db_embedding),
                    "db_weight": float(args.db_weight),
                    "feature_weight": float(args.feature_weight),
                    "emb_dim": int(emb.shape[1]),
                }
            ),
            flush=True,
        )
    nfc_info = None
    if int(args.nfc_k1) > 0:
        from vlincs_gallery.feature_centralization import neighbor_feature_centralization

        indices = list(range(len(records)))
        group_codes = _group_codes(records, args.nfc_exclude_same, indices)
        emb, nfc_info = neighbor_feature_centralization(
            emb,
            k1=int(args.nfc_k1),
            k2=int(args.nfc_k2),
            eta=float(args.nfc_eta),
            group_codes=group_codes,
            exclude_same_group=args.nfc_exclude_same != "none",
        )
        print(json.dumps({"stage": "nfc", **asdict(nfc_info), "exclude_same": args.nfc_exclude_same}), flush=True)
    pred_by_video = _load_predictions(con)
    print(json.dumps({"stage": "loaded_predictions", "videos": len(pred_by_video), "rows": int(sum(len(v) for v in pred_by_video.values()))}), flush=True)
    records = _with_detection_endpoints(records, pred_by_video)
    print(json.dumps({"stage": "augmented_endpoints"}), flush=True)
    gt_by_video = load_ds1_gt_by_video()
    gt_by_video = {key: value for key, value in gt_by_video.items() if key in pred_by_video}
    print(json.dumps({"stage": "loaded_gt", "videos": len(gt_by_video), "rows": int(sum(len(v) for v in gt_by_video.values()))}), flush=True)
    eval_cache_expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": float(args.iou_thr),
        "min_matches": int(args.eval_min_matches),
        "min_purity": float(args.eval_min_purity),
        "n_tracklets": int(len(records)),
        "prediction_rows": int(sum(len(v) for v in pred_by_video.values())),
        "gt_rows": int(sum(len(v) for v in gt_by_video.values())),
    }
    cached_eval = _load_eval_label_cache(args.eval_cache, eval_cache_expected) if args.eval_cache else None
    if cached_eval is not None:
        gt_by_seq, weight_by_seq, eval_stats = cached_eval
    else:
        gt_by_seq, weight_by_seq, eval_stats = _label_tracklets_for_eval(
            pred_by_video,
            gt_by_video,
            iou_thr=args.iou_thr,
            min_matches=args.eval_min_matches,
            min_purity=args.eval_min_purity,
        )
        eval_stats.update(eval_cache_expected)
        if args.eval_cache:
            _cache_eval_labels(args.eval_cache, gt_by_seq, weight_by_seq, eval_stats)
    print(json.dumps({"stage": "labeled_tracklets_for_eval", **eval_stats}), flush=True)
    output_keep_seqs, output_info = _output_keep_seqs(records, args)
    print(json.dumps({"stage": "output_admission", **output_info}, sort_keys=True), flush=True)
    configs = _config_grid(args)
    if args.limit_configs and args.limit_configs > 0:
        configs = configs[: args.limit_configs]

    rows: list[dict[str, object]] = []
    seqs = [record.seq for record in records]
    graph_top_ks = [cfg.top_k for cfg in configs if cfg.mode == "graph"]
    graph_cache = _graph_cache(records, emb, max(graph_top_ks)) if graph_top_ks else None
    for idx, cfg in enumerate(configs, start=1):
        print(json.dumps({"stage": "start_config", "progress": idx, "total": len(configs), **asdict(cfg)}, sort_keys=True), flush=True)
        labels, info = _run_resolver(records, emb, cfg, graph_cache)
        pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=output_keep_seqs)
        metrics = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
        row = {
            "rank_input_order": idx,
            **asdict(cfg),
            **info,
            **{key: value for key, value in output_info.items() if not isinstance(value, dict)},
            **metrics,
        }
        rows.append(row)
        print(json.dumps({"progress": idx, "total": len(configs), "mode": cfg.mode, **metrics}, sort_keys=True), flush=True)

    rows.sort(key=lambda row: float(row.get(args.sort_key, 0.0)), reverse=True)
    full_n = max(int(args.full_top_n), 0)
    full_rows = _select_full_score_rows(rows, full_n, args.sort_key, args.full_selection_keys)
    for rank, row in enumerate(full_rows, start=1):
        cfg = ResolveConfig(
            mode=str(row["mode"]),
            theta=float(row["theta"]),
            top_k=int(row["top_k"]),
            min_dets=int(row["min_dets"]),
            exclude_same=str(row["exclude_same"]),
            cross_thr=float(row["cross_thr"]),
            intra_thr=float(row["intra_thr"]),
            max_gap=int(row["max_gap"]),
            max_component_size=int(row["max_component_size"]),
            temporal_bonus=float(row["temporal_bonus"]),
            local_gap=int(row["local_gap"]),
            local_thr=float(row["local_thr"]),
            local_app_min=float(row["local_app_min"]),
            local_pos_scale=float(row["local_pos_scale"]),
            target_clusters=int(row["target_clusters"]),
            rerank_k1=int(row["rerank_k1"]),
            rerank_k2=int(row["rerank_k2"]),
            rerank_lambda=float(row["rerank_lambda"]),
            time_window_ms=int(row["time_window_ms"]),
        )
        labels, _info = _run_resolver(records, emb, cfg, graph_cache)
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, keep_seqs=output_keep_seqs))
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = rank
        print(
            json.dumps(
                {
                    "full_rank": rank,
                    "config": asdict(cfg),
                    "full_selection_keys": args.full_selection_keys or args.sort_key,
                    "full": full,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = {
        "dataset": args.dataset,
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "nfc": ({**asdict(nfc_info), "exclude_same": args.nfc_exclude_same} if nfc_info is not None else None),
        "n_tracklets": len(records),
        "n_configs": len(configs),
        "eval_stats": eval_stats,
        "output_admission": output_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "sort_key": args.sort_key,
        "full_selection_keys": args.full_selection_keys or args.sort_key,
        "top": rows[: max(full_n, 20)],
    }

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        scalar_keys = sorted(
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list, tuple))
        )
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in scalar_keys})
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
