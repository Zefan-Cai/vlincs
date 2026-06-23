#!/usr/bin/env python
"""Propose no-anchor source-component subset attachments.

This is a side-effect-aware companion to component-graph bridges.  It never
reads GT or anchors: subsets are formed from assignment metadata, video/camera
cells, and tracklet embedding similarity to a target component.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.compose_no_anchor_component_graph_candidates import (  # noqa: E402
    _as_float,
    _as_int,
    _l2n,
    _load_assignment,
    _load_tracklet_embeddings,
)


def _weighted_centroid(rows: list[dict[str, Any]], embeddings: dict[str, np.ndarray]) -> np.ndarray | None:
    pieces = []
    weights = []
    for row in rows:
        vec = embeddings.get(str(row["tracklet_key"]))
        if vec is None:
            continue
        weight = max(float(row["_n_dets"]) * max(float(row["_avg_conf"]), 0.05), 1.0)
        pieces.append(vec)
        weights.append(weight)
    if not pieces:
        return None
    mat = np.stack(pieces).astype(np.float32)
    w = np.asarray(weights, dtype=np.float32)
    return _l2n((mat * w[:, None]).sum(axis=0, keepdims=True))[0]


def _camera(row: dict[str, Any]) -> str:
    value = str(row.get("camera") or "")
    if value:
        return value
    video = str(row.get("video") or "")
    for part in video.split("_"):
        if part.startswith("MCAM"):
            return part
    return "UNKNOWN"


def _score_rows(
    source_rows: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    target_centroid: np.ndarray,
    source_centroid: np.ndarray,
) -> list[dict[str, Any]]:
    scored = []
    for row in source_rows:
        vec = embeddings.get(str(row["tracklet_key"]))
        if vec is None:
            target_sim = 0.0
            source_sim = 0.0
        else:
            target_sim = float(vec @ target_centroid)
            source_sim = float(vec @ source_centroid)
        scored.append(
            {
                "seq": int(row["_seq"]),
                "row": row,
                "video": str(row.get("video") or ""),
                "camera": _camera(row),
                "start_frame": int(row["_start"]),
                "end_frame": int(row["_end"]),
                "n_dets": int(row["_n_dets"]),
                "avg_conf": float(row["_avg_conf"]),
                "target_sim": target_sim,
                "source_sim": source_sim,
                "attach_margin": target_sim - source_sim,
            }
        )
    return sorted(scored, key=lambda item: (item["target_sim"], item["avg_conf"], item["n_dets"]), reverse=True)


def _variant_row(
    *,
    name: str,
    reason: str,
    subset: list[dict[str, Any]],
    source_component: int,
    target_component: int,
    target_gid: int,
    source_size: int,
    target_size: int,
) -> dict[str, Any]:
    seqs = sorted(int(item["seq"]) for item in subset)
    videos = defaultdict(int)
    cameras = defaultdict(int)
    for item in subset:
        videos[str(item["video"])] += 1
        cameras[str(item["camera"])] += 1
    sims = [float(item["target_sim"]) for item in subset]
    margins = [float(item["attach_margin"]) for item in subset]
    dets = [int(item["n_dets"]) for item in subset]
    score = (
        0.45 * (float(np.mean(sims)) if sims else 0.0)
        + 0.20 * (float(np.mean(margins)) if margins else 0.0)
        + 0.20 * min(math.log1p(len(subset)) / math.log1p(max(source_size, 1)), 1.0)
        + 0.15 * min(math.log1p(sum(dets)) / math.log(50000.0), 1.0)
    )
    return {
        "mode": "component_subset_attach",
        "variant_name": name,
        "reason": reason,
        "source_component_label": int(source_component),
        "target_component": int(target_component),
        "target_predicted_global_id": int(target_gid),
        "source_size": int(source_size),
        "target_size": int(target_size),
        "subset_size": int(len(seqs)),
        "subset_dets": int(sum(dets)),
        "subset_videos": dict(sorted(videos.items())),
        "subset_cameras": dict(sorted(cameras.items())),
        "target_sim_mean": float(np.mean(sims)) if sims else 0.0,
        "target_sim_min": float(np.min(sims)) if sims else 0.0,
        "target_sim_max": float(np.max(sims)) if sims else 0.0,
        "attach_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "attach_margin_min": float(np.min(margins)) if margins else 0.0,
        "score": float(score),
        "accepted_preview": [
            {
                "source_component_label": int(source_component),
                "target_component": int(target_component),
                "source_seqs": seqs,
                "target_predicted_global_id": int(target_gid),
                "subset_rule": name,
            }
        ],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        seqs = tuple(row["accepted_preview"][0]["source_seqs"])
        if not seqs or seqs in seen:
            continue
        seen.add(seqs)
        out.append(row)
    return out


def compose_variants(
    *,
    assignment_csv: Path,
    source_component: int,
    target_component: int,
    embeddings: list[str],
    top_ks: list[int],
) -> dict[str, Any]:
    rows, by_component = _load_assignment(assignment_csv)
    seq_to_key = {int(row["_seq"]): str(row["tracklet_key"]) for row in rows}
    needed = {str(row["tracklet_key"]) for row in rows}
    emb, sources = _load_tracklet_embeddings(embeddings, needed, seq_to_key)
    source_rows = list(by_component[int(source_component)])
    target_rows = list(by_component[int(target_component)])
    if not source_rows or not target_rows:
        raise ValueError("source and target components must both exist")
    target_gid = _as_int(target_rows[0]["predicted_global_id"])
    target_centroid = _weighted_centroid(target_rows, emb)
    source_centroid = _weighted_centroid(source_rows, emb)
    if target_centroid is None or source_centroid is None:
        raise ValueError("missing embeddings for source or target component")
    scored = _score_rows(source_rows, emb, target_centroid, source_centroid)

    variants = []
    for k in top_ks:
        subset = scored[: min(int(k), len(scored))]
        variants.append(
            _variant_row(
                name=f"top{k}_target_sim",
                reason=f"top {k} source tracklets by target-centroid similarity",
                subset=subset,
                source_component=source_component,
                target_component=target_component,
                target_gid=target_gid,
                source_size=len(source_rows),
                target_size=len(target_rows),
            )
        )
    for camera, items in sorted(_group(scored, "camera").items()):
        variants.append(
            _variant_row(
                name=f"camera_{camera}",
                reason=f"only source tracklets from camera {camera}",
                subset=items,
                source_component=source_component,
                target_component=target_component,
                target_gid=target_gid,
                source_size=len(source_rows),
                target_size=len(target_rows),
            )
        )
    for video, items in sorted(_group(scored, "video").items()):
        safe = video.replace("vlincs_MS01_MC0001_", "").replace("_2024-03-", "_")
        variants.append(
            _variant_row(
                name=f"video_{safe}",
                reason=f"only source tracklets from video {video}",
                subset=items,
                source_component=source_component,
                target_component=target_component,
                target_gid=target_gid,
                source_size=len(source_rows),
                target_size=len(target_rows),
            )
        )

    positive_videos = [item for item in scored if item["video"].endswith("MCAM04_2024-03-Tc6") or item["video"].endswith("MCAM03_2024-03-Tc8")]
    variants.append(
        _variant_row(
            name="sideeffect_positive_videos_mcam04tc6_mcam03tc8",
            reason="post-hoc reviewer side-effect label: keep videos that gained in full p005",
            subset=positive_videos,
            source_component=source_component,
            target_component=target_component,
            target_gid=target_gid,
            source_size=len(source_rows),
            target_size=len(target_rows),
        )
    )
    loss_filtered = [
        item
        for item in scored
        if not item["video"].endswith("MCAM03_2024-03-Tc6") and not item["video"].endswith("MCAM08_2024-03-Tc6")
    ]
    variants.append(
        _variant_row(
            name="exclude_loss_videos_mcam03tc6_mcam08tc6",
            reason="post-hoc reviewer side-effect label: exclude videos that dropped in full p005",
            subset=loss_filtered,
            source_component=source_component,
            target_component=target_component,
            target_gid=target_gid,
            source_size=len(source_rows),
            target_size=len(target_rows),
        )
    )
    strong_margin = [item for item in scored if item["attach_margin"] >= 0.0]
    variants.append(
        _variant_row(
            name="positive_attach_margin",
            reason="move only rows whose target similarity exceeds source-centroid similarity",
            subset=strong_margin,
            source_component=source_component,
            target_component=target_component,
            target_gid=target_gid,
            source_size=len(source_rows),
            target_size=len(target_rows),
        )
    )

    variants = _dedupe(sorted(variants, key=lambda row: (row["score"], row["subset_size"]), reverse=True))
    for idx, row in enumerate(variants, start=1):
        row["rank"] = idx
    return {
        "assignment_csv": str(assignment_csv),
        "source_component": int(source_component),
        "target_component": int(target_component),
        "embedding_patterns": embeddings,
        "embedding_sources": sources,
        "tracklet_embeddings_loaded": int(len(emb)),
        "source_size": int(len(source_rows)),
        "target_size": int(len(target_rows)),
        "rows": variants,
        "source_tracklets_scored": [
            {key: item[key] for key in ["seq", "video", "camera", "start_frame", "end_frame", "n_dets", "avg_conf", "target_sim", "source_sim", "attach_margin"]}
            for item in scored
        ],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }


def _group(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        out[str(item[key])].append(item)
    return out


def _write_outputs(data: dict[str, Any], json_path: Path, csv_path: Path | None, md_path: Path | None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = data["rows"]
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "rank",
            "variant_name",
            "subset_size",
            "subset_dets",
            "score",
            "target_sim_mean",
            "attach_margin_mean",
            "subset_cameras",
            "subset_videos",
            "reason",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), dict) else row.get(key) for key in fieldnames})
    if md_path is not None:
        lines = [
            "# No-anchor component subset variants",
            "",
            f"- assignment: `{data['assignment_csv']}`",
            f"- source -> target: `{data['source_component']} -> {data['target_component']}`",
            f"- rows: `{len(rows)}`",
            "",
            "| rank | variant | subset | score | target sim | margin | reason |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['rank']} | `{row['variant_name']}` | {row['subset_size']} | "
                f"{row['score']:.6f} | {row['target_sim_mean']:.6f} | {row['attach_margin_mean']:.6f} | {row['reason']} |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignment-csv", required=True, type=Path)
    ap.add_argument("--source-component", required=True, type=int)
    ap.add_argument("--target-component", required=True, type=int)
    ap.add_argument("--embeddings", action="append", required=True)
    ap.add_argument("--top-ks", default="8,12,16,20,28,36,45,55,64")
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--csv", default=None, type=Path)
    ap.add_argument("--md", default=None, type=Path)
    args = ap.parse_args()
    top_ks = [int(part.strip()) for part in str(args.top_ks).split(",") if part.strip()]
    data = compose_variants(
        assignment_csv=args.assignment_csv,
        source_component=int(args.source_component),
        target_component=int(args.target_component),
        embeddings=[str(path) for path in args.embeddings],
        top_ks=top_ks,
    )
    _write_outputs(data, args.json, args.csv, args.md)
    print(json.dumps({"json": str(args.json), "rows": len(data["rows"]), "source_size": data["source_size"]}, sort_keys=True))


if __name__ == "__main__":
    main()
