#!/usr/bin/env python
"""Propose high-mass no-anchor component bridges from assignment + embeddings.

This is a structural pivot away from composing old accepted_preview rows.  It
builds a component graph directly from the current no-anchor assignment, using
only tracklet metadata and embedding similarity.  It emits executable
accepted_preview rows, but does not read GT labels or anchors.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any) -> int:
    return int(float(value))


def _l2n(values: np.ndarray) -> np.ndarray:
    return values / (np.linalg.norm(values, axis=-1, keepdims=True) + 1.0e-9)


def _load_assignment(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    by_component: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"seq", "tracklet_key", "component_label", "video", "start_frame", "end_frame", "n_dets", "avg_conf"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing assignment columns: {sorted(missing)}")
        for row in reader:
            item = dict(row)
            item["_seq"] = _as_int(row["seq"])
            item["_component"] = _as_int(row["component_label"])
            item["_start"] = _as_int(row["start_frame"])
            item["_end"] = _as_int(row["end_frame"])
            item["_n_dets"] = _as_int(row["n_dets"])
            item["_avg_conf"] = _as_float(row["avg_conf"])
            rows.append(item)
            by_component[item["_component"]].append(item)
    return rows, by_component


def _load_tracklet_embeddings(
    patterns: list[str],
    needed_keys: set[str],
    seq_to_key: dict[int, str],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    parts_by_source: list[tuple[int, dict[str, np.ndarray]]] = []
    source_meta: list[dict[str, Any]] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for match in matches:
            data = np.load(match, allow_pickle=True)
            if "track_ids" in data.files and "vectors" in data.files:
                ids = [str(track_id) for track_id in data["track_ids"]]
                vectors = data["vectors"].astype(np.float32)
                id_kind = "track_ids"
            elif "seqs" in data.files and "features" in data.files:
                ids = [seq_to_key.get(int(seq), str(seq)) for seq in data["seqs"]]
                vectors = data["features"].astype(np.float32)
                id_kind = "seqs"
            else:
                raise KeyError(f"{match} must contain track_ids/vectors or seqs/features; keys={data.files}")
            vectors = _l2n(vectors)
            local: dict[str, np.ndarray] = {}
            for idx, key in enumerate(ids):
                if key not in needed_keys:
                    continue
                local[key] = vectors[idx].copy()
            dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
            parts_by_source.append((dim, local))
            source_meta.append({"path": str(match), "id_kind": id_kind, "dim": dim, "matched_tracklets": int(len(local))})

    out: dict[str, np.ndarray] = {}
    for key in needed_keys:
        pieces = []
        present = False
        for dim, local in parts_by_source:
            vec = local.get(key)
            if vec is None:
                pieces.append(np.zeros((dim,), dtype=np.float32))
                continue
            present = True
            pieces.append(vec.astype(np.float32))
        if not present or not pieces:
            continue
        out[key] = _l2n(np.concatenate(pieces, axis=0)[None, :])[0].astype(np.float32)
    return out, source_meta


def _component_stats(
    component: int,
    rows: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    *,
    max_vectors: int,
) -> dict[str, Any]:
    vec_rows: list[tuple[float, dict[str, Any], np.ndarray]] = []
    total_dets = 0
    confs = []
    videos: dict[str, list[tuple[int, int]]] = defaultdict(list)
    cameras = set()
    for row in rows:
        total_dets += int(row["_n_dets"])
        confs.append(float(row["_avg_conf"]))
        videos[str(row["video"])].append((int(row["_start"]), int(row["_end"])))
        cameras.add(str(row.get("camera", "")))
        vec = embeddings.get(str(row["tracklet_key"]))
        if vec is None:
            continue
        weight = max(float(row["_n_dets"]) * max(float(row["_avg_conf"]), 0.05), 1.0)
        vec_rows.append((weight, row, vec))
    vec_rows.sort(key=lambda item: item[0], reverse=True)
    kept = vec_rows[: int(max_vectors)]
    if kept:
        weights = np.asarray([item[0] for item in kept], dtype=np.float32)
        mat = np.stack([item[2] for item in kept]).astype(np.float32)
        centroid = _l2n((mat * weights[:, None]).sum(axis=0, keepdims=True))[0]
    else:
        mat = np.zeros((0, 1), dtype=np.float32)
        weights = np.zeros((0,), dtype=np.float32)
        centroid = np.zeros((1,), dtype=np.float32)
    quality = 0.55 * (float(np.mean(confs)) if confs else 0.0) + 0.45 * min(math.log1p(total_dets) / math.log(50000.0), 1.0)
    return {
        "component": int(component),
        "rows": rows,
        "matrix": mat,
        "weights": weights,
        "centroid": centroid,
        "size": int(len(rows)),
        "vec_size": int(len(vec_rows)),
        "kept_vec_size": int(len(kept)),
        "n_dets": int(total_dets),
        "avg_conf": float(np.mean(confs)) if confs else 0.0,
        "quality": float(quality),
        "videos": videos,
        "video_count": int(len(videos)),
        "camera_count": int(len(cameras - {""})),
    }


def _topk_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    k = min(int(k), int(values.size))
    return float(np.sort(values.reshape(-1))[-k:].mean())


def _pair_embedding_features(src: dict[str, Any], tgt: dict[str, Any]) -> dict[str, float]:
    src_mat = src["matrix"]
    tgt_mat = tgt["matrix"]
    if src_mat.shape[0] == 0 or tgt_mat.shape[0] == 0 or src_mat.shape[1] != tgt_mat.shape[1]:
        return {
            "centroid_sim": 0.0,
            "src_to_tgt_top5": 0.0,
            "tgt_to_src_top5": 0.0,
            "pair_max_sim": 0.0,
            "pair_p99_sim": 0.0,
            "pair_gt70_count": 0.0,
        }
    sims = src_mat @ tgt_mat.T
    flat = sims.reshape(-1)
    centroid_sim = float(src["centroid"] @ tgt["centroid"])
    return {
        "centroid_sim": centroid_sim,
        "src_to_tgt_top5": _topk_mean(np.max(sims, axis=1), 5),
        "tgt_to_src_top5": _topk_mean(np.max(sims, axis=0), 5),
        "pair_max_sim": float(np.max(flat)),
        "pair_p99_sim": float(np.quantile(flat, 0.99)),
        "pair_gt70_count": float((flat >= 0.70).sum()),
    }


def _rank_maps(stats: dict[int, dict[str, Any]]) -> dict[int, dict[int, int]]:
    comps = sorted(stats)
    centroids = np.stack([stats[c]["centroid"] for c in comps]).astype(np.float32)
    sims = centroids @ centroids.T
    ranks: dict[int, dict[int, int]] = {}
    for i, comp in enumerate(comps):
        order = [int(comps[j]) for j in np.argsort(-sims[i]) if int(comps[j]) != int(comp)]
        ranks[int(comp)] = {other: rank for rank, other in enumerate(order, start=1)}
    return ranks


def _same_video_overlap(src: dict[str, Any], tgt: dict[str, Any], *, sample_limit: int = 96) -> dict[str, float]:
    overlap = 0
    possible = 0
    shared_videos = set(src["videos"]).intersection(tgt["videos"])
    for video in shared_videos:
        a = src["videos"][video][:sample_limit]
        b = tgt["videos"][video][:sample_limit]
        possible += len(a) * len(b)
        for s1, e1 in a:
            for s2, e2 in b:
                if max(s1, s2) <= min(e1, e2):
                    overlap += 1
    ratio = float(overlap / possible) if possible else 0.0
    return {"same_video_overlap_pairs": float(overlap), "same_video_overlap_ratio": ratio, "shared_video_count": float(len(shared_videos))}


def _source_seqs(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["_seq"]) for row in sorted(rows, key=lambda row: int(row["_seq"]))]


def _pair_score(
    src: dict[str, Any],
    tgt: dict[str, Any],
    feats: dict[str, float],
    ranks: dict[int, dict[int, int]],
    *,
    max_rank_for_bonus: int,
) -> float:
    src_rank = ranks[src["component"]].get(tgt["component"], 999)
    tgt_rank = ranks[tgt["component"]].get(src["component"], 999)
    mutual = 0.5 * max(0.0, 1.0 - (src_rank - 1) / max_rank_for_bonus) + 0.5 * max(0.0, 1.0 - (tgt_rank - 1) / max_rank_for_bonus)
    mass = min(math.log1p(src["size"]) / math.log(256.0), 1.0)
    target_mass = min(math.log1p(tgt["size"]) / math.log(256.0), 1.0)
    overlap_penalty = min(float(feats["same_video_overlap_ratio"]) * 1.5, 0.35)
    quality = 0.5 * float(src["quality"]) + 0.5 * float(tgt["quality"])
    return float(
        0.22 * feats["centroid_sim"]
        + 0.18 * feats["src_to_tgt_top5"]
        + 0.14 * feats["tgt_to_src_top5"]
        + 0.10 * feats["pair_p99_sim"]
        + 0.10 * mutual
        + 0.10 * mass
        + 0.06 * target_mass
        + 0.10 * quality
        - overlap_penalty
    )


def _row_from_pair(src: dict[str, Any], tgt: dict[str, Any], feats: dict[str, float], score: float) -> dict[str, Any]:
    source_seqs = _source_seqs(src["rows"])
    preview_item = {
        "source_component_label": int(src["component"]),
        "target_component": int(tgt["component"]),
        "source_seqs": source_seqs,
        "source_size": int(src["size"]),
        "target_size": int(tgt["size"]),
        "source_quality": float(src["quality"]),
        "target_quality": float(tgt["quality"]),
        "target_best_sim": float(feats["pair_max_sim"]),
        "target_mean_sim": float(feats["centroid_sim"]),
        "target_min_view_sim": float(min(feats["src_to_tgt_top5"], feats["tgt_to_src_top5"])),
        "target_view_vote": float(min(feats["pair_gt70_count"] / 10.0, 1.0)),
        "target_margin": float(max(0.0, feats["pair_p99_sim"] - 0.55)),
        "score": float(score),
        "view_mean_sim": float(feats["centroid_sim"]),
        "view_min_sim": float(min(feats["src_to_tgt_top5"], feats["tgt_to_src_top5"])),
        "pair_mass_proxy": float(max(src["size"], 1) * max(tgt["size"], 1)),
        "bridge_mass_proxy": float(math.sqrt(max(src["size"], 1) * max(tgt["size"], 1))),
        "same_video_overlap_ratio": float(feats["same_video_overlap_ratio"]),
    }
    pair_mass = float(max(src["size"], 1) * max(tgt["size"], 1))
    full_proxy = 0.648 + 0.018 * max(0.0, score - 0.62) + 0.004 * min(math.log1p(src["size"]) / math.log(256.0), 1.0) - 0.010 * feats["same_video_overlap_ratio"]
    return {
        "mode": "component_graph_high_mass_bridge",
        "source_component_label": int(src["component"]),
        "target_component": int(tgt["component"]),
        "accepted_preview": [preview_item],
        "accepted_reassignments": 1,
        "moved_tracklets": int(src["size"]),
        "target_components_used": 1,
        "source_size": int(src["size"]),
        "target_size": int(tgt["size"]),
        "source_quality": float(src["quality"]),
        "target_quality": float(tgt["quality"]),
        "target_best_sim": float(feats["pair_max_sim"]),
        "target_mean_sim": float(feats["centroid_sim"]),
        "target_min_view_sim": float(min(feats["src_to_tgt_top5"], feats["tgt_to_src_top5"])),
        "target_view_vote": float(min(feats["pair_gt70_count"] / 10.0, 1.0)),
        "target_margin": float(max(0.0, feats["pair_p99_sim"] - 0.55)),
        "accepted_edges": 1,
        "accepted_score_mean": float(score),
        "accepted_view_mean_sim_mean": float(feats["centroid_sim"]),
        "accepted_view_min_sim_mean": float(min(feats["src_to_tgt_top5"], feats["tgt_to_src_top5"])),
        "accepted_mass_proxy_sum": float(math.sqrt(pair_mass)),
        "accepted_pair_mass_proxy_sum": pair_mass,
        "accepted_min_weight_sum": float(min(src["size"], tgt["size"])),
        "accepted_max_weight_sum": float(max(src["size"], tgt["size"])),
        "accepted_size_product_sum": pair_mass,
        "full_side_effect_proxy": float(full_proxy),
        "no_gt_component_graph_score": float(score),
        "same_video_overlap_ratio": float(feats["same_video_overlap_ratio"]),
        "same_video_overlap_pairs": int(feats["same_video_overlap_pairs"]),
        "shared_video_count": int(feats["shared_video_count"]),
        "source_video_count": int(src["video_count"]),
        "target_video_count": int(tgt["video_count"]),
        "signature": repr(("component_graph_high_mass_bridge", int(src["component"]), int(tgt["component"]), int(src["size"]), int(tgt["size"]))),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }


def compose(args: argparse.Namespace) -> dict[str, Any]:
    assignment_rows, by_component = _load_assignment(Path(args.assignment_csv))
    needed = {str(row["tracklet_key"]) for rows in by_component.values() for row in rows}
    seq_to_key = {int(row["_seq"]): str(row["tracklet_key"]) for row in assignment_rows}
    embeddings, embedding_sources = _load_tracklet_embeddings(args.embeddings, needed, seq_to_key)
    stats = {
        comp: _component_stats(comp, rows, embeddings, max_vectors=int(args.max_vectors_per_component))
        for comp, rows in by_component.items()
    }
    stats = {comp: stat for comp, stat in stats.items() if stat["size"] >= int(args.min_component_size) and stat["kept_vec_size"] >= int(args.min_vec_tracklets)}
    ranks = _rank_maps(stats) if stats else {}
    rows: list[dict[str, Any]] = []
    for src_id, src in stats.items():
        if src["size"] < int(args.min_source_size):
            continue
        for tgt_id, tgt in stats.items():
            if src_id == tgt_id or tgt["size"] < int(args.min_target_size):
                continue
            base_feats = _pair_embedding_features(src, tgt)
            if base_feats["centroid_sim"] < float(args.min_centroid_sim):
                continue
            if max(base_feats["src_to_tgt_top5"], base_feats["tgt_to_src_top5"]) < float(args.min_topk_sim):
                continue
            overlap = _same_video_overlap(src, tgt)
            feats = {**base_feats, **overlap}
            if feats["same_video_overlap_ratio"] > float(args.max_same_video_overlap_ratio):
                continue
            src_rank = ranks[src_id].get(tgt_id, 999)
            tgt_rank = ranks[tgt_id].get(src_id, 999)
            if min(src_rank, tgt_rank) > int(args.max_one_sided_rank) or max(src_rank, tgt_rank) > int(args.max_mutual_rank):
                continue
            score = _pair_score(src, tgt, feats, ranks, max_rank_for_bonus=int(args.max_mutual_rank))
            if score < float(args.min_score):
                continue
            rows.append(_row_from_pair(src, tgt, feats, score))

    best_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["source_component_label"]), int(row["target_component"]))
        old = best_by_pair.get(key)
        if old is None or float(row["no_gt_component_graph_score"]) > float(old["no_gt_component_graph_score"]):
            best_by_pair[key] = row
    rows = sorted(best_by_pair.values(), key=lambda row: (float(row["no_gt_component_graph_score"]), float(row["accepted_pair_mass_proxy_sum"])), reverse=True)[: int(args.top_n)]

    result = {
        "assignment_csv": str(args.assignment_csv),
        "embedding_patterns": args.embeddings,
        "embedding_sources": embedding_sources,
        "embedding_total_dim": int(sum(item["dim"] for item in embedding_sources)),
        "component_count": int(len(by_component)),
        "eligible_component_count": int(len(stats)),
        "tracklet_embeddings_loaded": int(len(embeddings)),
        "rows": rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(Path(args.csv), rows)
    if args.md:
        _write_md(Path(args.md), result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "source_component_label",
        "target_component",
        "moved_tracklets",
        "source_size",
        "target_size",
        "no_gt_component_graph_score",
        "target_mean_sim",
        "target_best_sim",
        "target_min_view_sim",
        "target_view_vote",
        "same_video_overlap_ratio",
        "accepted_pair_mass_proxy_sum",
        "full_side_effect_proxy",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Component-Graph High-Mass Candidates",
        "",
        f"- assignment: `{result['assignment_csv']}`",
        f"- components: `{result['component_count']}`",
        f"- eligible components: `{result['eligible_component_count']}`",
        f"- tracklet embeddings loaded: `{result['tracklet_embeddings_loaded']}`",
        f"- emitted rows: `{len(result['rows'])}`",
        "",
        "| rank | source | target | moved | target size | graph score | centroid | top sim | overlap | pair mass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(result["rows"][:50], start=1):
        lines.append(
            f"| {rank} | `{row['source_component_label']}` | `{row['target_component']}` | "
            f"`{row['moved_tracklets']}` | `{row['target_size']}` | `{row['no_gt_component_graph_score']:.6f}` | "
            f"`{row['target_mean_sim']:.6f}` | `{row['target_best_sim']:.6f}` | "
            f"`{row['same_video_overlap_ratio']:.6f}` | `{row['accepted_pair_mass_proxy_sum']:.0f}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        assign = root / "assign.csv"
        assign.write_text(
            "seq,tracklet_key,video,camera,start_frame,end_frame,n_dets,avg_conf,predicted_global_id,component_label,component_size,prediction_confidence,decision_status\n"
            "1,v:1:0,v,MCAM00,0,10,10,0.9,900,1,2,0.7,forced_component\n"
            "2,v:2:0,v,MCAM00,20,30,8,0.8,900,1,2,0.7,forced_component\n"
            "3,v:3:0,v,MCAM00,40,50,9,0.9,901,2,2,0.7,forced_component\n"
            "4,v:4:0,v,MCAM00,60,70,7,0.8,901,2,2,0.7,forced_component\n"
        )
        emb = root / "embeddings.npz"
        np.savez(
            emb,
            track_ids=np.asarray(["v:1:0", "v:2:0", "v:3:0", "v:4:0"]),
            vectors=np.asarray([[1, 0], [0.98, 0.02], [0.97, 0.03], [0.96, 0.04]], dtype=np.float32),
            crop_ids=np.asarray(["a", "b", "c", "d"]),
            frame_idxs=np.asarray([1, 2, 3, 4]),
            video_ids=np.asarray(["v", "v", "v", "v"]),
        )
        out = compose(
            argparse.Namespace(
                assignment_csv=str(assign),
                embeddings=[str(emb)],
                max_vectors_per_component=10,
                min_component_size=1,
                min_source_size=1,
                min_target_size=1,
                min_vec_tracklets=1,
                min_centroid_sim=0.5,
                min_topk_sim=0.5,
                max_same_video_overlap_ratio=0.0,
                max_one_sided_rank=3,
                max_mutual_rank=3,
                min_score=0.1,
                top_n=10,
                json=str(root / "out.json"),
                csv="",
                md="",
            )
        )
        assert out["rows"], out
        assert out["rows"][0]["accepted_preview"][0]["source_seqs"], out["rows"][0]
    print("self-test passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignment-csv", default="")
    ap.add_argument("--embeddings", action="append", default=[])
    ap.add_argument("--max-vectors-per-component", type=int, default=96)
    ap.add_argument("--min-component-size", type=int, default=1)
    ap.add_argument("--min-source-size", type=int, default=8)
    ap.add_argument("--min-target-size", type=int, default=8)
    ap.add_argument("--min-vec-tracklets", type=int, default=1)
    ap.add_argument("--min-centroid-sim", type=float, default=0.40)
    ap.add_argument("--min-topk-sim", type=float, default=0.62)
    ap.add_argument("--max-same-video-overlap-ratio", type=float, default=0.015)
    ap.add_argument("--max-one-sided-rank", type=int, default=12)
    ap.add_argument("--max-mutual-rank", type=int, default=30)
    ap.add_argument("--min-score", type=float, default=0.58)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    missing = []
    if not args.assignment_csv:
        missing.append("--assignment-csv")
    if not args.embeddings:
        missing.append("--embeddings")
    if not args.json:
        missing.append("--json")
    if missing:
        ap.error(f"missing required argument(s): {', '.join(missing)}")
    out = compose(args)
    print(json.dumps({"rows": len(out["rows"]), "components": out["component_count"], "embeddings": out["tracklet_embeddings_loaded"]}, sort_keys=True))


if __name__ == "__main__":
    main()
