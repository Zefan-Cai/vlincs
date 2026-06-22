#!/usr/bin/env python
"""Rerank no-anchor single-edge candidates with source-local embedding evidence.

The scheduler's global score is good at finding promising source components,
but duplicate-source candidates can disagree on which target fragment should
receive the source.  This script acts as a no-GT referee for that narrow case:
within each source component, pick the target with stronger crop-embedding
support and target quality, then preserve the original global score ordering.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np


def _as_int(value: Any) -> int:
    return int(float(value))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _l2n(vecs: np.ndarray) -> np.ndarray:
    return vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1.0e-9)


def _load_assignment(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    by_component: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"tracklet_key", "component_label", "n_dets", "avg_conf"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing assignment columns: {sorted(missing)}")
        for row in reader:
            item = dict(row)
            item["_component"] = _as_int(row["component_label"])
            item["_n_dets"] = _as_int(row["n_dets"])
            item["_avg_conf"] = _as_float(row["avg_conf"])
            rows.append(item)
            by_component[item["_component"]].append(item)
    return rows, by_component


def _candidate_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} is missing rows[]")
    return [row for row in rows if isinstance(row, dict)]


def _edge_item(row: dict[str, Any]) -> dict[str, Any]:
    preview = row.get("accepted_preview")
    if isinstance(preview, list) and preview and isinstance(preview[0], dict):
        return preview[0]
    return row


def _component_id(row: dict[str, Any], *keys: str) -> int:
    item = _edge_item(row)
    for key in keys:
        value = row.get(key, item.get(key))
        if value not in (None, ""):
            return _as_int(value)
    raise ValueError(f"candidate has no component key among {keys}: {row}")


def _needed_tracklets(by_component: dict[int, list[dict[str, Any]]], components: set[int]) -> set[str]:
    out: set[str] = set()
    for component in components:
        for row in by_component.get(component, []):
            out.add(str(row["tracklet_key"]))
    return out


def _load_tracklet_embeddings(patterns: list[str], needed_keys: set[str]) -> dict[str, np.ndarray]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for match in matches:
            data = np.load(match, allow_pickle=True)
            track_ids = data["track_ids"]
            vectors = _l2n(data["vectors"].astype(np.float32))
            for idx, track_id in enumerate(track_ids):
                key = str(track_id)
                if key not in needed_keys:
                    continue
                old = sums.get(key)
                sums[key] = vectors[idx].copy() if old is None else old + vectors[idx]
                counts[key] += 1
    return {key: _l2n(value).astype(np.float32) for key, value in sums.items() if counts[key] > 0}


def _component_stats(
    component: int,
    by_component: dict[int, list[dict[str, Any]]],
    tracklet_embeddings: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows = by_component.get(component, [])
    vecs = []
    weights = []
    total_dets = 0
    confs = []
    for row in rows:
        total_dets += int(row["_n_dets"])
        confs.append(float(row["_avg_conf"]))
        vec = tracklet_embeddings.get(str(row["tracklet_key"]))
        if vec is None:
            continue
        vecs.append(vec)
        weights.append(max(int(row["_n_dets"]), 1))
    if vecs:
        mat = np.stack(vecs).astype(np.float32)
        w = np.asarray(weights, dtype=np.float32)
        centroid = _l2n((mat * w[:, None]).sum(axis=0, keepdims=True))[0]
    else:
        mat = np.zeros((0, 1), dtype=np.float32)
        centroid = np.zeros((1,), dtype=np.float32)
    return {
        "rows": rows,
        "matrix": mat,
        "centroid": centroid,
        "n_tracklets": int(len(rows)),
        "n_vec_tracklets": int(len(vecs)),
        "n_dets": int(total_dets),
        "avg_conf": float(sum(confs) / len(confs)) if confs else 0.0,
    }


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else 0.0


def _edge_features(
    row: dict[str, Any],
    source_stats: dict[str, Any],
    target_stats: dict[str, Any],
) -> dict[str, float]:
    item = _edge_item(row)
    src = source_stats["matrix"]
    tgt_centroid = target_stats["centroid"]
    if src.size and tgt_centroid.size == src.shape[1]:
        sims = src @ tgt_centroid
        src_max = float(np.max(sims))
        src_p99 = _quantile(sims, 0.99)
        src_top5 = float(np.sort(sims)[-min(5, len(sims)) :].mean())
        src_gt70 = float((sims > 0.70).sum())
    else:
        src_max = src_p99 = src_top5 = src_gt70 = 0.0
    target_quality = 0.55 * target_stats["avg_conf"] + 0.45 * min(
        math.log1p(target_stats["n_dets"]) / math.log(32.0),
        1.0,
    )
    embedding_support = 0.45 * src_top5 + 0.35 * src_p99 + 0.20 * src_max
    scheduler_support = (
        0.50 * _as_float(item.get("fused_sim"))
        + 0.30 * _as_float(item.get("db_sim"))
        + 0.20 * _as_float(item.get("primary_sim"))
    )
    choice_score = 0.52 * embedding_support + 0.27 * target_quality + 0.21 * scheduler_support
    return {
        "choice_score": float(choice_score),
        "embedding_support": float(embedding_support),
        "target_quality": float(target_quality),
        "scheduler_support": float(scheduler_support),
        "src_to_target_max": float(src_max),
        "src_to_target_p99": float(src_p99),
        "src_to_target_top5": float(src_top5),
        "src_to_target_gt70_count": float(src_gt70),
        "target_n_dets": float(target_stats["n_dets"]),
        "target_avg_conf": float(target_stats["avg_conf"]),
        "target_n_tracklets": float(target_stats["n_tracklets"]),
        "target_n_vec_tracklets": float(target_stats["n_vec_tracklets"]),
    }


def rerank(args: argparse.Namespace) -> dict[str, Any]:
    rows = _candidate_rows(Path(args.candidates_json))
    _assign_rows, by_component = _load_assignment(Path(args.assignment_csv))
    components: set[int] = set()
    pairs: list[tuple[int, int, dict[str, Any]]] = []
    for row in rows:
        source = _component_id(row, "source_component_label", "source", "source_rep")
        target = _component_id(row, "target_component", "target", "target_rep")
        components.update([source, target])
        pairs.append((source, target, row))
    embeddings = _load_tracklet_embeddings(args.embeddings, _needed_tracklets(by_component, components))
    stats = {component: _component_stats(component, by_component, embeddings) for component in components}

    enriched = []
    for source, target, row in pairs:
        new_row = dict(row)
        new_row["source_component_label"] = source
        new_row["target_component"] = target
        new_row.update(_edge_features(new_row, stats[source], stats[target]))
        new_row["uses_anchors"] = False
        new_row["uses_gt_for_training_or_anchors"] = False
        new_row["uses_gt_for_evaluation_only"] = False
        enriched.append(new_row)

    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_source[int(row["source_component_label"])].append(row)
    selected = []
    suppressed = []
    for source, group in by_source.items():
        group.sort(key=lambda row: (float(row["choice_score"]), float(row.get("full_side_effect_proxy", 0.0))), reverse=True)
        selected.append(group[0])
        for row in group[1:]:
            suppressed.append({"suppressed_by_source": source, **row})
    selected.sort(
        key=lambda row: (
            float(row.get("full_side_effect_proxy", 0.0)),
            float(row["choice_score"]),
            float(row.get("accepted_score_mean", 0.0)),
        ),
        reverse=True,
    )
    top_rows = selected[: int(args.top_n)]
    out = {
        "candidates_json": str(args.candidates_json),
        "assignment_csv": str(args.assignment_csv),
        "embedding_patterns": args.embeddings,
        "rows": top_rows,
        "suppressed_rows": suppressed,
        "input_rows": int(len(rows)),
        "selected_rows": int(len(top_rows)),
        "tracklet_embeddings_loaded": int(len(embeddings)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), top_rows)
    if args.md:
        _write_md(Path(args.md), out)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_component_label",
        "target_component",
        "full_side_effect_proxy",
        "choice_score",
        "embedding_support",
        "target_quality",
        "scheduler_support",
        "src_to_target_max",
        "src_to_target_p99",
        "src_to_target_top5",
        "target_n_dets",
        "target_avg_conf",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Single-Edge Embedding Choice Rerank",
        "",
        f"- input rows: `{out['input_rows']}`",
        f"- selected rows: `{out['selected_rows']}`",
        f"- suppressed duplicate-source rows: `{len(out['suppressed_rows'])}`",
        f"- tracklet embeddings loaded: `{out['tracklet_embeddings_loaded']}`",
        "",
        "| rank | source | target | proxy | choice | emb | target q | sched | top5 | p99 | target dets |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(out["rows"], start=1):
        lines.append(
            f"| {rank} | `{row['source_component_label']}` | `{row['target_component']}` | "
            f"`{float(row.get('full_side_effect_proxy', 0.0)):.6f}` | `{float(row['choice_score']):.6f}` | "
            f"`{float(row['embedding_support']):.6f}` | `{float(row['target_quality']):.6f}` | "
            f"`{float(row['scheduler_support']):.6f}` | `{float(row['src_to_target_top5']):.6f}` | "
            f"`{float(row['src_to_target_p99']):.6f}` | `{int(row['target_n_dets'])}` |"
        )
    if out["suppressed_rows"]:
        lines.extend(["", "## Suppressed Duplicate-Source Rows", ""])
        lines.append("| source | suppressed target | kept target | suppressed choice |")
        lines.append("| ---: | ---: | ---: | ---: |")
        kept_by_source = {int(row["source_component_label"]): row for row in out["rows"]}
        for row in out["suppressed_rows"]:
            kept = kept_by_source.get(int(row["source_component_label"]))
            kept_target = kept.get("target_component") if kept else ""
            lines.append(
                f"| `{row['source_component_label']}` | `{row['target_component']}` | "
                f"`{kept_target}` | `{float(row['choice_score']):.6f}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        assignment = root / "assign.csv"
        assignment.write_text(
            "seq,tracklet_key,component_label,n_dets,avg_conf\n"
            "1,a,1,10,0.8\n2,b,1,10,0.8\n3,c,2,3,0.4\n4,d,3,12,0.9\n",
            encoding="utf-8",
        )
        emb = root / "embeddings.npz"
        np.savez(
            emb,
            track_ids=np.asarray(["a", "b", "c", "d"]),
            vectors=np.asarray([[1, 0], [0.9, 0.1], [0, 1], [1, 0.05]], dtype=np.float32),
        )
        cand = root / "cand.json"
        cand.write_text(
            json.dumps(
                {
                    "rows": [
                        {"source_component_label": 1, "target_component": 2, "full_side_effect_proxy": 0.8, "accepted_preview": [{"fused_sim": 0.8, "db_sim": 0.8, "primary_sim": 0.8}]},
                        {"source_component_label": 1, "target_component": 3, "full_side_effect_proxy": 0.79, "accepted_preview": [{"fused_sim": 0.7, "db_sim": 0.7, "primary_sim": 0.7}]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = rerank(
            argparse.Namespace(
                candidates_json=str(cand),
                assignment_csv=str(assignment),
                embeddings=[str(emb)],
                json=str(root / "out.json"),
                csv="",
                md="",
                top_n=10,
            )
        )
        assert len(out["rows"]) == 1
        assert int(out["rows"][0]["target_component"]) == 3
        assert len(out["suppressed_rows"]) == 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-json", required=True)
    parser.add_argument("--assignment-csv", required=True)
    parser.add_argument("--embeddings", action="append", required=True, help="NPZ path/glob; may be repeated")
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", default="")
    parser.add_argument("--md", default="")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    rerank(args)


if __name__ == "__main__":
    main()
