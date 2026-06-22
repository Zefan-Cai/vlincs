#!/usr/bin/env python
"""Train a no-anchor weak metric projection for tracklet features.

Weak positives come from same-tracklet crop pairs and short-gap same-stream
continuations.  Weak negatives come from same-stream temporal overlaps, which
are cannot-link pairs.  The script never reads anchors or identity labels; it
only writes a normal ``seqs``/``features`` npz for existing resolvers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, KIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from kit.no_anchor_continuation_positive_edge_verifier import (
        _center_dist,
        _sample_topmean,
        _scale_sim,
        _tracklet_gap,
    )
    from kit.no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _with_detection_endpoints,
    )
    from kit.no_anchor_sample_positive_edge_verifier import _load_samples
except ModuleNotFoundError:
    from no_anchor_continuation_positive_edge_verifier import (
        _center_dist,
        _sample_topmean,
        _scale_sim,
        _tracklet_gap,
    )
    from no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
        _connect,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _with_detection_endpoints,
    )
    from no_anchor_sample_positive_edge_verifier import _load_samples


def _l2n(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _continuation_pairs(records, samples, counts, args) -> tuple[list[tuple[int, int]], dict[str, object]]:
    by_stream: dict[tuple[str, str], list[int]] = {}
    for idx, rec in enumerate(records):
        if int(counts[idx]) <= 0:
            continue
        by_stream.setdefault((str(rec.video), str(rec.camera)), []).append(int(idx))
    rows: list[tuple[float, int, int]] = []
    considered = rejected_geom = rejected_sim = 0
    for indices in by_stream.values():
        ordered = sorted(indices, key=lambda idx: (int(records[idx].start_frame), int(records[idx].end_frame), idx))
        for pos, i in enumerate(ordered):
            a = records[i]
            for j in ordered[pos + 1 :]:
                b = records[j]
                gap = _tracklet_gap(a, b)
                if gap < int(args.positive_min_gap_frames):
                    continue
                if gap > int(args.positive_max_gap_frames):
                    break
                considered += 1
                if _center_dist(a, b) > float(args.positive_max_center_dist):
                    rejected_geom += 1
                    continue
                if _scale_sim(a, b) < float(args.positive_min_scale_sim):
                    rejected_geom += 1
                    continue
                sim = _sample_topmean(samples, counts, i, j)
                if sim < float(args.positive_min_sample_topmean):
                    rejected_sim += 1
                    continue
                rows.append((float(sim), int(i), int(j)))
    rows.sort(reverse=True)
    if int(args.max_continuation_positive_pairs) > 0:
        rows = rows[: int(args.max_continuation_positive_pairs)]
    return [(i, j) for _score, i, j in rows], {
        "continuation_positive_pairs": int(len(rows)),
        "continuation_positive_considered": int(considered),
        "continuation_rejected_geometry": int(rejected_geom),
        "continuation_rejected_similarity": int(rejected_sim),
    }


def _same_tracklet_sample_pairs(samples, counts, args) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(int(args.random_state))
    left = []
    right = []
    candidates = [idx for idx, n in enumerate(counts.tolist()) if int(n) >= 2]
    rng.shuffle(candidates)
    for idx in candidates[: int(args.max_same_tracklet_positive_pairs)]:
        n = int(counts[idx])
        a, b = rng.choice(n, size=2, replace=False).tolist()
        left.append(samples[idx, int(a)])
        right.append(samples[idx, int(b)])
    if not left:
        return np.zeros((0, samples.shape[-1]), dtype=np.float32), np.zeros((0, samples.shape[-1]), dtype=np.float32), {"same_tracklet_positive_pairs": 0}
    return np.stack(left).astype(np.float32), np.stack(right).astype(np.float32), {"same_tracklet_positive_pairs": int(len(left))}


def _cannot_link_pairs(records, counts, mean, args) -> tuple[list[tuple[int, int]], dict[str, object]]:
    forbidden = _build_overlap_forbidden(records)
    scored = []
    for i, nbrs in enumerate(forbidden):
        if int(counts[i]) <= 0:
            continue
        for j in nbrs:
            if i < j and int(counts[j]) > 0:
                scored.append((float(np.dot(mean[i], mean[j])), int(i), int(j)))
    scored.sort(reverse=True)
    total = len(scored)
    if int(args.max_negative_pairs) > 0:
        scored = scored[: int(args.max_negative_pairs)]
    return [(i, j) for _score, i, j in scored], {"cannot_link_negative_pairs": int(len(scored)), "cannot_link_negative_candidates": int(total)}


def _fit_projection(
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    labels: np.ndarray,
    *,
    output_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    logit_scale: float,
    random_state: int,
    device: str,
) -> tuple[np.ndarray, dict[str, object]]:
    torch.manual_seed(int(random_state))
    rng = np.random.default_rng(int(random_state))
    x1 = torch.from_numpy(pairs_a.astype(np.float32)).to(device)
    x2 = torch.from_numpy(pairs_b.astype(np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    layer = torch.nn.Linear(int(x1.shape[1]), int(output_dim), bias=False).to(device)
    torch.nn.init.orthogonal_(layer.weight)
    opt = torch.optim.AdamW(layer.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    n = int(len(labels))
    last_loss = 0.0
    for _epoch in range(max(int(epochs), 1)):
        order = rng.permutation(n)
        for start in range(0, n, int(batch_size)):
            idx = torch.from_numpy(order[start : start + int(batch_size)]).to(device)
            z1 = F.normalize(layer(x1.index_select(0, idx)), dim=1)
            z2 = F.normalize(layer(x2.index_select(0, idx)), dim=1)
            logits = float(logit_scale) * (z1 * z2).sum(dim=1)
            loss = F.binary_cross_entropy_with_logits(logits, y.index_select(0, idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())
    with torch.no_grad():
        z1 = F.normalize(layer(x1), dim=1)
        z2 = F.normalize(layer(x2), dim=1)
        scores = (z1 * z2).sum(dim=1).detach().cpu().numpy()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    margin = float(pos.mean() - neg.mean()) if len(pos) and len(neg) else 0.0
    return layer.weight.detach().cpu().numpy().astype(np.float32), {
        "train_pairs": int(n),
        "train_positive": int(np.sum(labels == 1)),
        "train_negative": int(np.sum(labels == 0)),
        "final_loss": round(float(last_loss), 6),
        "train_pos_cos_mean": round(float(pos.mean()) if len(pos) else 0.0, 6),
        "train_neg_cos_mean": round(float(neg.mean()) if len(neg) else 0.0, 6),
        "train_cos_margin": round(margin, 6),
        "projection_dim": int(output_dim),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--sample-feature-npz", required=True)
    ap.add_argument("--base-feature-npz", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--metadata-json", default="")
    ap.add_argument("--blend-weights", default="0.10", help="comma-separated weak weights; first value writes --output")
    ap.add_argument("--positive-min-gap-frames", type=int, default=0)
    ap.add_argument("--positive-max-gap-frames", type=int, default=90)
    ap.add_argument("--positive-max-center-dist", type=float, default=1.35)
    ap.add_argument("--positive-min-scale-sim", type=float, default=0.45)
    ap.add_argument("--positive-min-sample-topmean", type=float, default=0.70)
    ap.add_argument("--max-same-tracklet-positive-pairs", type=int, default=9000)
    ap.add_argument("--max-continuation-positive-pairs", type=int, default=9000)
    ap.add_argument("--max-negative-pairs", type=int, default=18000)
    ap.add_argument("--projection-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.0001)
    ap.add_argument("--logit-scale", type=float, default=16.0)
    ap.add_argument("--random-state", type=int, default=41)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(int(args.random_state))
    np.random.seed(int(args.random_state))
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    samples, counts, mean, sample_meta = _load_samples(args.sample_feature_npz, records)
    mean = _l2n(mean)

    st_a, st_b, st_info = _same_tracklet_sample_pairs(samples, counts, args)
    cont_pairs, cont_info = _continuation_pairs(records, samples, counts, args)
    neg_pairs, neg_info = _cannot_link_pairs(records, counts, mean, args)

    pos_a = [st_a] if len(st_a) else []
    pos_b = [st_b] if len(st_b) else []
    if cont_pairs:
        pos_a.append(np.stack([mean[i] for i, _j in cont_pairs]).astype(np.float32))
        pos_b.append(np.stack([mean[j] for _i, j in cont_pairs]).astype(np.float32))
    if not pos_a or not neg_pairs:
        raise RuntimeError(f"need weak positive and negative pairs, got pos_parts={len(pos_a)} neg={len(neg_pairs)}")
    neg_a = np.stack([mean[i] for i, _j in neg_pairs]).astype(np.float32)
    neg_b = np.stack([mean[j] for _i, j in neg_pairs]).astype(np.float32)
    pairs_a = np.concatenate([*pos_a, neg_a], axis=0)
    pairs_b = np.concatenate([*pos_b, neg_b], axis=0)
    labels = np.concatenate([np.ones(sum(len(x) for x in pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])

    weight, train_info = _fit_projection(
        pairs_a,
        pairs_b,
        labels,
        output_dim=int(args.projection_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        logit_scale=float(args.logit_scale),
        random_state=int(args.random_state),
        device=str(args.device),
    )
    weak = _l2n(mean @ weight.T)
    if args.base_feature_npz:
        base = _load_feature_npz(args.base_feature_npz, records, db_emb, concat_db=False, db_weight=1.0, feature_weight=1.0)
    else:
        base = _l2n(db_emb.astype(np.float32))
    weights = _parse_floats(args.blend_weights)
    if not weights:
        weights = [1.0]
    out_paths = []
    for pos, weak_weight in enumerate(weights):
        out_path = Path(args.output)
        if pos > 0:
            out_path = out_path.with_name(f"{out_path.stem}_w{str(weak_weight).replace('.', 'p')}{out_path.suffix}")
        blended = _l2n(np.concatenate([base, np.sqrt(max(float(weak_weight), 0.0)) * weak], axis=1))
        payload = {
            "seqs": np.asarray([int(record.seq) for record in records], dtype=np.int64),
            "features": blended.astype(np.float32),
            "weak_metric_info": np.asarray([json.dumps({"weak_weight": float(weak_weight), **train_info}, sort_keys=True)], dtype=object),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **payload)
        out_paths.append(str(out_path))

    metadata = {
        "outputs": out_paths,
        "sample_meta": sample_meta,
        "same_tracklet_info": st_info,
        "continuation_info": cont_info,
        "negative_info": neg_info,
        "train_info": train_info,
        "base_feature_npz": str(args.base_feature_npz),
        "sample_feature_npz": str(args.sample_feature_npz),
        "blend_weights": weights,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    meta_path = Path(args.metadata_json or str(Path(args.output).with_suffix(".json")))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
