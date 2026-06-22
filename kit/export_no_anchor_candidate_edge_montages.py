#!/usr/bin/env python
"""Export no-anchor component-pair candidate montages for VLM verification.

The script does not use anchors or GT.  It starts from an assignment CSV,
builds component-level candidate edges with the same multi-view scorer used by
`no_anchor_assignment_multiview_merge_sweep.py`, and writes image montages plus
metadata for the top edges.  A downstream VLM can judge whether the two panels
depict the same identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2  # type: ignore
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from kit.extract_tracklet_foundation_features import _crop, _video_paths
    from kit.no_anchor_assignment_component_merge_sweep import _component_members, _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_assignment_multiview_merge_sweep import (
        _centroid_candidate_edges,
        _load_npz_aligned,
        _parse_view,
        _score_edges,
        _view_tables,
    )
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from extract_tracklet_foundation_features import _crop, _video_paths
    from no_anchor_assignment_component_merge_sweep import _component_members, _labels_from_assignment, _load_assignment_labels
    from no_anchor_assignment_multiview_merge_sweep import (
        _centroid_candidate_edges,
        _load_npz_aligned,
        _parse_view,
        _score_edges,
        _view_tables,
    )
    from no_anchor_resolve_sweep import (
        _connect,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _with_detection_endpoints,
    )


def _sample_detection_rows(con, seqs: list[int]) -> dict[int, list[tuple[str, int, float, float, float, float]]]:
    if not seqs:
        return {}
    out: dict[int, list[tuple[str, int, float, float, float, float]]] = {}
    with con.cursor() as cur:
        cur.execute(
            """WITH ranked AS (
                   SELECT a.seq, d.video, d.frame_idx, d.x1, d.y1, d.x2, d.y2,
                          ROW_NUMBER() OVER (PARTITION BY a.seq ORDER BY d.frame_idx) AS rn,
                          COUNT(*) OVER (PARTITION BY a.seq) AS cnt
                   FROM assignments a
                   JOIN detections d ON d.det_id = a.det_id
                   WHERE a.seq = ANY(%s)
               )
               SELECT seq, video, frame_idx, x1, y1, x2, y2
               FROM ranked
               WHERE rn = GREATEST(1, (cnt + 1) / 2)
               ORDER BY seq""",
            (seqs,),
        )
        for seq, video, frame_idx, x1, y1, x2, y2 in cur.fetchall():
            out.setdefault(int(seq), []).append((str(video), int(frame_idx), float(x1), float(y1), float(x2), float(y2)))
    return out


def _read_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_idx), 0))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    finally:
        cap.release()


def _representative_indices(records, members: list[int], count: int) -> list[int]:
    ranked = sorted(
        [int(idx) for idx in members],
        key=lambda idx: (float(records[idx].n_dets) * (0.25 + float(records[idx].avg_conf)), int(records[idx].n_dets)),
        reverse=True,
    )
    selected: list[int] = []
    seen_streams: set[tuple[str, str]] = set()
    for idx in ranked:
        stream = (str(records[idx].video), str(records[idx].camera))
        if stream in seen_streams and len(selected) < max(1, count // 2):
            continue
        selected.append(idx)
        seen_streams.add(stream)
        if len(selected) >= count:
            return selected
    for idx in ranked:
        if idx not in selected:
            selected.append(idx)
            if len(selected) >= count:
                break
    return selected


def _make_montage(
    edge_id: int,
    records,
    left_indices: list[int],
    right_indices: list[int],
    det_rows: dict[int, list[tuple[str, int, float, float, float, float]]],
    video_paths: dict[str, str],
    out_path: Path,
    *,
    crop_size: int,
    margin: float,
) -> dict[str, object]:
    font = ImageFont.load_default()

    def load_crop(idx: int, side: str) -> Image.Image:
        seq = int(records[idx].seq)
        rows = det_rows.get(seq, [])
        if not rows:
            image = Image.new("RGB", (crop_size, crop_size), (35, 35, 35))
        else:
            video, frame_idx, x1, y1, x2, y2 = rows[0]
            frame = _read_frame(video_paths.get(video, ""), frame_idx) if video in video_paths else None
            crop = _crop(frame, (x1, y1, x2, y2), margin) if frame is not None else None
            image = crop.convert("RGB") if crop is not None else Image.new("RGB", (crop_size, crop_size), (35, 35, 35))
        image.thumbnail((crop_size, crop_size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (crop_size, crop_size + 24), (245, 245, 245))
        x = (crop_size - image.width) // 2
        canvas.paste(image, (x, 0))
        draw = ImageDraw.Draw(canvas)
        label = f"{side} seq={seq} {records[idx].camera}"
        draw.text((4, crop_size + 6), label[:34], fill=(10, 10, 10), font=font)
        return canvas

    crops = [load_crop(idx, "A") for idx in left_indices] + [load_crop(idx, "B") for idx in right_indices]
    cols = max(len(left_indices), len(right_indices))
    w = cols * crop_size
    h = 2 * (crop_size + 24) + 56
    montage = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(montage)
    draw.text((8, 8), f"edge {edge_id}: top row A, bottom row B. Same global identity?", fill=(0, 0, 0), font=font)
    for pos, img in enumerate(crops[: len(left_indices)]):
        montage.paste(img, (pos * crop_size, 32))
    for pos, img in enumerate(crops[len(left_indices) :]):
        montage.paste(img, (pos * crop_size, 32 + crop_size + 24))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path)
    return {
        "montage_path": str(out_path),
        "left_seqs": [int(records[idx].seq) for idx in left_indices],
        "right_seqs": [int(records[idx].seq) for idx in right_indices],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--primary-feature-npz", required=True)
    ap.add_argument("--view", action="append", default=[])
    ap.add_argument("--video-root", action="append", default=[])
    ap.add_argument("--candidate-top-k", type=int, default=100)
    ap.add_argument("--rank-k", type=int, default=5)
    ap.add_argument("--sim-threshold", type=float, default=0.68)
    ap.add_argument("--score-mode", default="mean_min")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--crops-per-side", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=192)
    ap.add_argument("--crop-margin", type=float, default=0.08)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    keep_seqs, _output_info = _output_keep_seqs(
        records,
        argparse.Namespace(
            output_min_dets=1,
            output_min_conf=0.0,
            output_min_area=0.0,
            output_min_quality=-1.0e9,
            output_min_area_by_video="",
            output_drop_area_quantile=0.0,
            output_drop_area_quantile_by_video="",
            output_drop_quality_quantile=0.0,
            output_drop_quality_quantile_by_video="",
            output_auto_anomaly_admission=False,
            output_auto_anomaly_metric="quality",
            output_auto_anomaly_quantile=0.75,
            output_auto_anomaly_area_ratio=0.60,
            output_auto_anomaly_quality_mad=1.0,
            output_auto_anomaly_min_video_tracklets=20,
            output_auto_anomaly_max_videos=3,
        ),
    )
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
    base_labels, _raw_to_local = _labels_from_assignment(records, pred_input)
    reps, members = _component_members(base_labels, keep_indices)

    primary_emb = _load_feature_npz(args.primary_feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0)
    edges, edge_info = _centroid_candidate_edges(records, primary_emb, reps, members, int(args.candidate_top_k))
    view_embeddings = {"primary": primary_emb.astype(np.float32)}
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        if path.lower() == "db":
            view_embeddings[name] = db_emb.astype(np.float32) * float(weight)
        else:
            view_embeddings[name] = _load_npz_aligned(path, records, weight=float(weight))
    sims, ranks = _view_tables(view_embeddings, members)
    scored = _score_edges(edges, sims, ranks, score_mode=str(args.score_mode), rank_k=int(args.rank_k), sim_threshold=float(args.sim_threshold))
    scored = scored[: int(args.top_n)]

    chosen_indices = []
    for edge in scored:
        chosen_indices.extend(_representative_indices(records, members[int(edge["source"])], int(args.crops_per_side)))
        chosen_indices.extend(_representative_indices(records, members[int(edge["target"])], int(args.crops_per_side)))
    det_rows = _sample_detection_rows(con, sorted({int(records[idx].seq) for idx in chosen_indices}))
    video_paths = _video_paths(args.dataset, list(args.video_root))
    out_dir = Path(args.output_dir)
    rows = []
    for edge_id, edge in enumerate(scored):
        left = _representative_indices(records, members[int(edge["source"])], int(args.crops_per_side))
        right = _representative_indices(records, members[int(edge["target"])], int(args.crops_per_side))
        montage = _make_montage(
            edge_id,
            records,
            left,
            right,
            det_rows,
            video_paths,
            out_dir / f"edge_{edge_id:03d}.jpg",
            crop_size=int(args.crop_size),
            margin=float(args.crop_margin),
        )
        rows.append(
            {
                "edge_id": int(edge_id),
                **{key: value for key, value in edge.items() if isinstance(value, (int, float, str, bool))},
                **montage,
            }
        )
    result = {
        "assignment_csv": str(args.assignment_csv),
        "primary_feature_npz": str(args.primary_feature_npz),
        "views": sorted(view_embeddings),
        "edge_info": edge_info,
        "rows": rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"json": str(out), "montages": len(rows), "output_dir": str(out_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
