#!/usr/bin/env python
"""Clean no-anchor portfolio candidates with source-local target arbitration."""

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

from kit.rerank_no_anchor_single_edge_by_embedding_choice import (
    _as_float,
    _as_int,
    _component_stats,
    _edge_features,
    _load_assignment,
    _load_tracklet_embeddings,
    _needed_tracklets,
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    rows: list[dict[str, Any]] = []
    for key in ("rows", "selected", "top", "full_rows"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    if not rows:
        raise ValueError(f"{path} contains no candidate rows")
    return rows


def _preview(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("accepted_preview")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _component(item: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return _as_int(value)
    return None


def _source_size(item: dict[str, Any]) -> int:
    value = item.get("source_size")
    if value not in (None, ""):
        return max(_as_int(value), 1)
    seqs = item.get("source_seqs")
    return max(len(seqs), 1) if isinstance(seqs, list) else 1


def _target_size(item: dict[str, Any]) -> int:
    value = item.get("target_size")
    if value not in (None, ""):
        return max(_as_int(value), 1)
    seqs = item.get("target_top_seqs")
    return max(len(seqs), 1) if isinstance(seqs, list) else 1


def _item_row(item: dict[str, Any]) -> dict[str, Any]:
    source = _component(item, "source_component_label", "source", "source_rep")
    target = _component(item, "target_component", "target", "target_rep")
    if source is None or target is None:
        raise ValueError(f"accepted_preview item lacks source/target component: {item}")
    return {
        "source_component_label": source,
        "target_component": target,
        "accepted_preview": [item],
        "full_side_effect_proxy": _as_float(item.get("score"), _as_float(item.get("edge_score"), 0.0)),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _score_preview_items(
    rows: list[dict[str, Any]],
    by_component: dict[int, list[dict[str, Any]]],
    stats: dict[int, dict[str, Any]],
) -> dict[int, dict[str, float]]:
    features: dict[int, dict[str, float]] = {}
    for row in rows:
        for item in _preview(row):
            source = _component(item, "source_component_label", "source", "source_rep")
            target = _component(item, "target_component", "target", "target_rep")
            if source is None or target is None:
                continue
            features[id(item)] = _edge_features(_item_row(item), stats[source], stats[target])
    return features


def _clean_preview(preview: list[dict[str, Any]], features: dict[int, dict[str, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in preview:
        source = _component(item, "source_component_label", "source", "source_rep")
        if source is None:
            continue
        by_source[source].append(item)
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for source, items in by_source.items():
        targets = defaultdict(list)
        for item in items:
            target = _component(item, "target_component", "target", "target_rep")
            if target is not None:
                targets[target].append(item)
        if len(targets) <= 1:
            kept.extend(items)
            continue
        target_scores = []
        for target, target_items in targets.items():
            scores = [features.get(id(item), {}).get("choice_score", 0.0) for item in target_items]
            mass = sum(_source_size(item) for item in target_items)
            target_scores.append((float(_mean(scores)), int(mass), int(target)))
        target_scores.sort(reverse=True)
        keep_target = target_scores[0][2]
        for target, target_items in targets.items():
            if int(target) == int(keep_target):
                kept.extend(target_items)
            else:
                for item in target_items:
                    suppressed.append(
                        {
                            "source_component": int(source),
                            "target_component": int(target),
                            "choice_score": float(features.get(id(item), {}).get("choice_score", 0.0)),
                            "source_size": int(_source_size(item)),
                        }
                    )
    return kept, suppressed


def _refresh_row(row: dict[str, Any], preview: list[dict[str, Any]], suppressed: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(row)
    old_moved = _as_float(row.get("moved_tracklets"), 0.0)
    moved = sum(_source_size(item) for item in preview)
    pair_mass = sum(_source_size(item) * _target_size(item) for item in preview)
    bridge_mass = sum(math.sqrt(_source_size(item) * _target_size(item)) for item in preview)
    dropped = sum(int(item["source_size"]) for item in suppressed)
    base_proxy = _as_float(row.get("full_side_effect_proxy"), _as_float(row.get("learned_proxy_full_idf1"), 0.0))
    sources = [
        str(_component(item, "source_component_label", "source", "source_rep"))
        for item in preview
        if _component(item, "source_component_label", "source", "source_rep") is not None
    ]
    targets = sorted(
        {
            str(_component(item, "target_component", "target", "target_rep"))
            for item in preview
            if _component(item, "target_component", "target", "target_rep") is not None
        }
    )
    out["source_component_label"] = "+".join(sources)
    out["target_component"] = "+".join(targets)
    out["accepted_preview"] = preview
    out["accepted_reassignments"] = int(len(preview))
    out["accepted_edges"] = int(len(preview))
    out["moved_tracklets"] = int(moved)
    out["accepted_pair_mass_proxy_sum"] = float(pair_mass)
    out["accepted_mass_proxy_sum"] = float(bridge_mass)
    out["embedding_choice_suppressed_count"] = int(len(suppressed))
    out["embedding_choice_suppressed_moved"] = int(dropped)
    out["embedding_choice_suppressed_preview"] = suppressed[:12]
    out["full_side_effect_proxy"] = float(base_proxy - 0.00002 * max(old_moved - moved, 0.0))
    out["learned_proxy_full_idf1"] = float(out["full_side_effect_proxy"])
    out["uses_anchors"] = False
    out["uses_gt_for_training_or_anchors"] = False
    out.setdefault("uses_gt_for_evaluation_only", False)
    return out


def clean(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(Path(args.input_json))
    _assign_rows, by_component = _load_assignment(Path(args.assignment_csv))
    components: set[int] = set()
    for row in rows:
        for item in _preview(row):
            source = _component(item, "source_component_label", "source", "source_rep")
            target = _component(item, "target_component", "target", "target_rep")
            if source is not None:
                components.add(source)
            if target is not None:
                components.add(target)
    embeddings = _load_tracklet_embeddings(args.embeddings, _needed_tracklets(by_component, components))
    stats = {component: _component_stats(component, by_component, embeddings) for component in components}
    features = _score_preview_items(rows, by_component, stats)
    cleaned_rows = []
    suppressed_total = 0
    for row in rows:
        preview = _preview(row)
        cleaned_preview, suppressed = _clean_preview(preview, features)
        if len(cleaned_preview) < int(args.min_edges):
            continue
        refreshed = _refresh_row(row, cleaned_preview, suppressed)
        suppressed_total += len(suppressed)
        cleaned_rows.append(refreshed)
    cleaned_rows.sort(
        key=lambda row: (
            float(row.get("full_side_effect_proxy", 0.0)),
            float(row.get("accepted_pair_mass_proxy_sum", 0.0)),
            int(row.get("moved_tracklets", 0)),
        ),
        reverse=True,
    )
    cleaned_rows = cleaned_rows[: int(args.top_n)]
    out = {
        "input_json": str(args.input_json),
        "assignment_csv": str(args.assignment_csv),
        "input_rows": int(len(rows)),
        "rows": cleaned_rows,
        "selected_rows": int(len(cleaned_rows)),
        "suppressed_items": int(suppressed_total),
        "tracklet_embeddings_loaded": int(len(embeddings)),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), cleaned_rows)
    if args.md:
        _write_md(Path(args.md), out)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "source_component_label",
        "target_component",
        "moved_tracklets",
        "accepted_edges",
        "full_side_effect_proxy",
        "embedding_choice_suppressed_count",
        "embedding_choice_suppressed_moved",
        "signature",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, out: dict[str, Any]) -> None:
    lines = [
        "# No-Anchor Portfolio Embedding Choice Referee",
        "",
        f"- input rows: `{out['input_rows']}`",
        f"- selected rows: `{out['selected_rows']}`",
        f"- suppressed preview items: `{out['suppressed_items']}`",
        f"- tracklet embeddings loaded: `{out['tracklet_embeddings_loaded']}`",
        "",
        "| rank | mode | moved | edges | proxy | suppressed | suppressed moved |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(out["rows"], start=1):
        lines.append(
            f"| {rank} | `{row.get('mode')}` | `{int(row.get('moved_tracklets', 0))}` | "
            f"`{int(row.get('accepted_edges', 0))}` | `{float(row.get('full_side_effect_proxy', 0.0)):.6f}` | "
            f"`{int(row.get('embedding_choice_suppressed_count', 0))}` | "
            f"`{int(row.get('embedding_choice_suppressed_moved', 0))}` |"
        )
    examples = [row for row in out["rows"] if int(row.get("embedding_choice_suppressed_count", 0)) > 0]
    if examples:
        lines.extend(["", "## Suppression Examples", ""])
        for row in examples[:8]:
            lines.append(f"### {row.get('signature', '')}")
            for item in row.get("embedding_choice_suppressed_preview", []):
                lines.append(
                    f"- source `{item['source_component']}` target `{item['target_component']}` "
                    f"choice `{float(item['choice_score']):.6f}` moved `{item['source_size']}`"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
                        {
                            "mode": "portfolio",
                            "moved_tracklets": 20,
                            "full_side_effect_proxy": 0.8,
                            "accepted_preview": [
                                {"source": 1, "target": 2, "source_size": 10, "fused_sim": 0.8, "db_sim": 0.8, "primary_sim": 0.8},
                                {"source": 1, "target": 3, "source_size": 10, "fused_sim": 0.7, "db_sim": 0.7, "primary_sim": 0.7},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = clean(
            argparse.Namespace(
                input_json=str(cand),
                assignment_csv=str(assignment),
                embeddings=[str(emb)],
                json=str(root / "out.json"),
                csv="",
                md="",
                min_edges=1,
                top_n=10,
            )
        )
        assert len(out["rows"]) == 1
        assert out["rows"][0]["accepted_reassignments"] == 1
        assert int(out["rows"][0]["accepted_preview"][0]["target"]) == 3
        assert out["suppressed_items"] == 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--assignment-csv", required=True)
    parser.add_argument("--embeddings", action="append", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", default="")
    parser.add_argument("--md", default="")
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    clean(args)


if __name__ == "__main__":
    main()
