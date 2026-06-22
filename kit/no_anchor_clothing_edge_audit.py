#!/usr/bin/env python
"""Targeted no-anchor clothing/body verifier audit for component edges.

This is a lightweight referee probe. It trains the same weak-label verifier as
``no_anchor_clothing_positive_edge_verifier.py`` but scores only user-provided
component pairs from an existing assignment, avoiding the expensive full-graph
candidate generation and merge sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from kit.no_anchor_assignment_component_merge_sweep import (
        _component_members,
        _labels_from_assignment,
        _load_assignment_labels,
    )
    from kit.no_anchor_clothing_positive_edge_verifier import (
        _build_training,
        _fit_model,
        _load_view,
        _pair_features,
    )
    from kit.no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _with_detection_endpoints,
    )
    from kit.no_anchor_sample_positive_edge_verifier import _load_samples
except ModuleNotFoundError:
    from no_anchor_assignment_component_merge_sweep import (
        _component_members,
        _labels_from_assignment,
        _load_assignment_labels,
    )
    from no_anchor_clothing_positive_edge_verifier import (
        _build_training,
        _fit_model,
        _load_view,
        _pair_features,
    )
    from no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _with_detection_endpoints,
    )
    from no_anchor_sample_positive_edge_verifier import _load_samples


def _parse_edges(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            left, right = item.split(":", 1)
        elif "-" in item:
            left, right = item.split("-", 1)
        else:
            raise ValueError(f"edge must look like A:B or A-B, got {item!r}")
        out.append((int(left), int(right)))
    if not out:
        raise ValueError("at least one edge is required")
    return out


def _edge_probability_with_pairs(edge, records, members, samples, counts, mean, pose, color, model, args):
    left = np.asarray(members[int(edge["source"])], dtype=np.int64)
    right = np.asarray(members[int(edge["target"])], dtype=np.int64)
    if len(left) == 0 or len(right) == 0:
        return 0.0, {"sample_pair_count": 0, "sample_prob_top_mean": 0.0, "left_seqs": [], "right_seqs": []}
    osnet = mean[left] @ mean[right].T
    pose_sim = pose[left] @ pose[right].T
    color_sim = color[left] @ color[right].T
    blend = (
        float(args.edge_osnet_weight) * osnet
        + float(args.edge_posecolor_weight) * pose_sim
        + float(args.edge_colorhist_weight) * color_sim
    )
    flat = np.argsort(-blend.reshape(-1))[: max(int(args.edge_pair_topk), 1)]
    feature_rows = []
    selected = []
    for pos in flat.tolist():
        li_pos = int(pos // len(right))
        rj_pos = int(pos % len(right))
        li = int(left[li_pos])
        rj = int(right[rj_pos])
        if int(counts[li]) <= 0 or int(counts[rj]) <= 0:
            continue
        feature_rows.append(_pair_features(samples, counts, mean, pose, color, li, rj))
        selected.append(
            {
                "left_seq": int(records[li].seq),
                "right_seq": int(records[rj].seq),
                "left_index": int(li),
                "right_index": int(rj),
                "osnet": float(osnet[li_pos, rj_pos]),
                "posecolor": float(pose_sim[li_pos, rj_pos]),
                "colorhist": float(color_sim[li_pos, rj_pos]),
                "blend": float(blend[li_pos, rj_pos]),
            }
        )
    if not feature_rows:
        return 0.0, {"sample_pair_count": 0, "sample_prob_top_mean": 0.0, "left_seqs": [], "right_seqs": []}
    prob = model.predict_proba(np.asarray(feature_rows, dtype=np.float32))[:, 1].astype(np.float32)
    order = np.argsort(-prob)
    top_prob = np.sort(prob)[-min(5, len(prob)) :]
    support_rows = []
    left_seqs: list[int] = []
    right_seqs: list[int] = []
    seen_left: set[int] = set()
    seen_right: set[int] = set()
    for pos in order[: max(int(args.support_top_pairs), 1)].tolist():
        item = dict(selected[int(pos)])
        item["probability"] = float(prob[int(pos)])
        support_rows.append(item)
        if item["left_seq"] not in seen_left:
            left_seqs.append(int(item["left_seq"]))
            seen_left.add(int(item["left_seq"]))
        if item["right_seq"] not in seen_right:
            right_seqs.append(int(item["right_seq"]))
            seen_right.add(int(item["right_seq"]))
    sel_osnet = np.asarray([row["osnet"] for row in selected], dtype=np.float32)
    sel_pose = np.asarray([row["posecolor"] for row in selected], dtype=np.float32)
    sel_color = np.asarray([row["colorhist"] for row in selected], dtype=np.float32)
    return float(prob.max()), {
        "sample_pair_count": int(len(prob)),
        "sample_prob_top_mean": float(top_prob.mean()),
        "sample_prob_mean": float(prob.mean()),
        "selected_osnet_max": float(sel_osnet.max()),
        "selected_posecolor_max": float(sel_pose.max()),
        "selected_colorhist_max": float(sel_color.max()),
        "selected_osnet_mean": float(sel_osnet.mean()),
        "selected_posecolor_mean": float(sel_pose.mean()),
        "selected_colorhist_mean": float(sel_color.mean()),
        "left_seqs": left_seqs,
        "right_seqs": right_seqs,
        "support_pair_rows": support_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--sample-feature-npz", required=True)
    ap.add_argument("--posecolor-npz", required=True)
    ap.add_argument("--colorhist-npz", required=True)
    ap.add_argument("--edges", required=True, help="Comma list like 9:60,43:71")
    ap.add_argument("--positive-min-gap-frames", type=int, default=0)
    ap.add_argument("--positive-max-gap-frames", type=int, default=60)
    ap.add_argument("--positive-max-center-dist", type=float, default=1.25)
    ap.add_argument("--positive-min-scale-sim", type=float, default=0.50)
    ap.add_argument("--positive-min-sample-topmean", type=float, default=0.72)
    ap.add_argument("--positive-min-posecolor", type=float, default=0.68)
    ap.add_argument("--positive-min-colorhist", type=float, default=0.68)
    ap.add_argument("--edge-pair-topk", type=int, default=16)
    ap.add_argument("--edge-osnet-weight", type=float, default=0.55)
    ap.add_argument("--edge-posecolor-weight", type=float, default=0.30)
    ap.add_argument("--edge-colorhist-weight", type=float, default=0.15)
    ap.add_argument("--support-top-pairs", type=int, default=5)
    ap.add_argument("--max-positive-pairs", type=int, default=2000)
    ap.add_argument("--max-negative-pairs", type=int, default=2000)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf", "logreg"])
    ap.add_argument("--random-state", type=int, default=31)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    requested_edges = _parse_edges(args.edges)
    print(json.dumps({"stage": "load", "edges": requested_edges}), flush=True)
    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, _db_emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    samples, counts, mean_emb, sample_meta = _load_samples(args.sample_feature_npz, records)
    pose_emb, pose_meta = _load_view(args.posecolor_npz, records)
    color_emb, color_meta = _load_view(args.colorhist_npz, records)

    keep_seqs, output_info = _output_keep_seqs(records, args)
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    reps, members = _component_members(base_labels, keep_indices)
    forbidden = _build_overlap_forbidden(records)

    print(json.dumps({"stage": "train", "components": len(members), "assignment_components": len(raw_to_local)}), flush=True)
    X_train, y_train, train_info = _build_training(records, samples, counts, mean_emb, pose_emb, color_emb, forbidden, args)
    model, model_info = _fit_model(X_train, y_train, args)

    rows = []
    for source, target in requested_edges:
        if source < 0 or target < 0 or source >= len(members) or target >= len(members):
            rows.append({"source": source, "target": target, "error": "component_index_out_of_range"})
            continue
        edge = {"source": source, "target": target}
        prob, info = _edge_probability_with_pairs(edge, records, members, samples, counts, mean_emb, pose_emb, color_emb, model, args)
        rows.append(
            {
                "source": int(source),
                "target": int(target),
                "source_rep": int(reps[source]),
                "target_rep": int(reps[target]),
                "source_size": int(len(members[source])),
                "target_size": int(len(members[target])),
                "sample_probability": float(prob),
                **info,
            }
        )
        print(json.dumps({"stage": "edge", "source": source, "target": target, "sample_probability": prob}), flush=True)

    result = {
        "assignment_csv": args.assignment_csv,
        "sample_meta": sample_meta,
        "posecolor_meta": pose_meta,
        "colorhist_meta": color_meta,
        "output_admission": output_info,
        "train_info": train_info,
        "model_info": model_info,
        "edges": rows,
        "uses_anchors": False,
        "uses_ground_truth": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "done", "json": str(out), "edges": rows}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
