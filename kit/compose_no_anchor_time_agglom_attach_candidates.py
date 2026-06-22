#!/usr/bin/env python
"""Compose no-anchor current-best repairs from a time-agglom assignment.

The time-agglom resolver can be too broad as a full replacement.  This tool
uses it only as a local candidate generator: if a time-agglom component is
strongly dominated by one current-best component and contains a small fragment
from another current-best component, propose moving just that fragment.

No anchors or GT labels are used.  The output is compatible with
export_no_anchor_scheduler_manifest_assignments.py via selected[].accepted_preview.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WEAK_VIDEO_HINTS = ("MCAM04_2024-03-Tc6", "MCAM06_2024-03-Tc6", "MCAM03_2024-03-Tc8")


def _intish(value: Any) -> int:
    return int(float(value))


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _overlap_frames(a: dict[str, str], b: dict[str, str]) -> int:
    if str(a.get("video")) != str(b.get("video")):
        return 0
    start = max(_intish(a["start_frame"]), _intish(b["start_frame"]))
    end = min(_intish(a["end_frame"]), _intish(b["end_frame"]))
    return max(0, end - start + 1)


def _numeric(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _temporal_attach_features(
    *,
    source_seqs: list[int],
    target_seqs: list[int],
    source_component_seqs: list[int],
    target_component_seqs: list[int],
    base_by_seq: dict[int, dict[str, str]],
) -> dict[str, float | int]:
    """No-GT features for whether a local attach looks temporally harmless."""
    if not source_seqs:
        return {
            "source_avg_conf_mean": 0.0,
            "source_n_dets_mean": 0.0,
            "source_end_rank_frac_mean": 0.0,
            "source_start_rank_frac_mean": 0.0,
            "source_terminal_fraction": 0.0,
            "source_prev_gap_min": -1,
            "source_next_gap_min": -1,
            "target_prev_gap_min": -1,
            "target_next_gap_min": -1,
            "target_gap_min": -1,
            "target_same_video_fraction": 0.0,
            "source_same_video_fraction": 0.0,
            "source_video_entropy_norm": 0.0,
        }

    source_rows = [base_by_seq[seq] for seq in source_seqs if seq in base_by_seq]
    source_component_rows = [base_by_seq[seq] for seq in source_component_seqs if seq in base_by_seq]
    target_component_rows = [base_by_seq[seq] for seq in target_component_seqs if seq in base_by_seq]
    if not source_rows or not source_component_rows:
        return {}

    end_order = [int(row["seq"]) if "seq" in row else _intish(row.get("seq", -1)) for row in []]
    source_by_end = sorted(source_component_rows, key=lambda row: (_intish(row["end_frame"]), _intish(row["start_frame"])))
    source_by_start = sorted(source_component_rows, key=lambda row: (_intish(row["start_frame"]), _intish(row["end_frame"])))
    end_rank = { _intish(row["seq"]): idx + 1 for idx, row in enumerate(source_by_end) }
    start_rank = { _intish(row["seq"]): idx + 1 for idx, row in enumerate(source_by_start) }
    size = max(len(source_component_rows), 1)

    source_end_fracs = []
    source_start_fracs = []
    terminal_hits = 0
    source_prev_gaps = []
    source_next_gaps = []
    target_prev_gaps = []
    target_next_gaps = []
    target_gap_mins = []
    for row in source_rows:
        seq = _intish(row["seq"])
        start = _intish(row["start_frame"])
        end = _intish(row["end_frame"])
        source_end_fracs.append(float(end_rank.get(seq, 1) / size))
        source_start_fracs.append(float(start_rank.get(seq, 1) / size))

        other_source = [other for other in source_component_rows if _intish(other["seq"]) != seq]
        source_prev = [
            start - _intish(other["end_frame"])
            for other in other_source
            if _intish(other["end_frame"]) <= start
        ]
        source_next = [
            _intish(other["start_frame"]) - end
            for other in other_source
            if _intish(other["start_frame"]) >= end
        ]
        target_prev = [
            start - _intish(other["end_frame"])
            for other in target_component_rows
            if _intish(other["end_frame"]) <= start
        ]
        target_next = [
            _intish(other["start_frame"]) - end
            for other in target_component_rows
            if _intish(other["start_frame"]) >= end
        ]
        if source_prev:
            source_prev_gaps.append(min(source_prev))
        if source_next:
            source_next_gaps.append(min(source_next))
        if target_prev:
            target_prev_gaps.append(min(target_prev))
        if target_next:
            target_next_gaps.append(min(target_next))
        target_gap_candidates = []
        if target_prev:
            target_gap_candidates.append(min(target_prev))
        if target_next:
            target_gap_candidates.append(min(target_next))
        if target_gap_candidates:
            target_gap_mins.append(min(target_gap_candidates))
        else:
            target_gap_mins.append(999999)
        if not source_next or end_rank.get(seq, 0) == size:
            terminal_hits += 1

    source_videos = Counter(str(row.get("video", "")) for row in source_component_rows)
    source_video_entropy = 0.0
    for count in source_videos.values():
        p = count / max(len(source_component_rows), 1)
        if p > 0:
            source_video_entropy -= p * math.log(p)
    source_video_entropy_norm = source_video_entropy / math.log(max(len(source_videos), 2))
    source_video_set = {str(row.get("video", "")) for row in source_rows}
    target_same_video = sum(1 for row in target_component_rows if str(row.get("video", "")) in source_video_set)
    source_same_video = sum(1 for row in source_component_rows if str(row.get("video", "")) in source_video_set)

    def mean(values: list[float | int], default: float = 0.0) -> float:
        return float(sum(values) / len(values)) if values else default

    def min_or(values: list[int], default: int = -1) -> int:
        return int(min(values)) if values else int(default)

    return {
        "source_avg_conf_mean": round(mean([_numeric(row, "avg_conf") for row in source_rows]), 6),
        "source_n_dets_mean": round(mean([_numeric(row, "n_dets") for row in source_rows]), 3),
        "source_end_rank_frac_mean": round(mean(source_end_fracs), 6),
        "source_start_rank_frac_mean": round(mean(source_start_fracs), 6),
        "source_terminal_fraction": round(float(terminal_hits / max(len(source_rows), 1)), 6),
        "source_prev_gap_min": min_or(source_prev_gaps),
        "source_next_gap_min": min_or(source_next_gaps),
        "target_prev_gap_min": min_or(target_prev_gaps),
        "target_next_gap_min": min_or(target_next_gaps),
        "target_gap_min": min_or(target_gap_mins),
        "target_same_video_fraction": round(float(target_same_video / max(len(target_component_rows), 1)), 6),
        "source_same_video_fraction": round(float(source_same_video / max(len(source_component_rows), 1)), 6),
        "source_video_entropy_norm": round(float(source_video_entropy_norm), 6),
    }


def _critic_score(row: dict[str, Any]) -> float:
    target_gap = max(float(row.get("target_gap_min", -1)), 0.0)
    target_gap_term = min(target_gap / 1000.0, 1.0)
    source_conf = min(max(float(row.get("source_avg_conf_mean", 1.0)), 0.0), 1.0)
    source_low_conf = 1.0 - source_conf
    target_same_video = min(max(float(row.get("target_same_video_fraction", 0.0)), 0.0), 1.0)
    source_entropy = min(max(float(row.get("source_video_entropy_norm", 0.0)), 0.0), 1.0)
    source_terminal = min(max(float(row.get("source_terminal_fraction", 0.0)), 0.0), 1.0)
    safe_terminal = source_terminal * target_gap_term
    target_dominance = min(max(float(row.get("target_dominance", 0.0)), 0.0), 1.0)
    target_support = min(max(float(row.get("target_count_in_time", 0.0)) / 128.0, 0.0), 1.0)
    source_smallness = 1.0 - min(max(float(row.get("source_component_size", 0.0)) / 24.0, 0.0), 1.0)
    source_next_gap = float(row.get("source_next_gap_min", -1))
    source_insert_penalty = 0.20 if 0 <= source_next_gap <= 300 else 0.0
    target_gap_penalty = 0.12 if 0 <= target_gap <= 150 else 0.0
    score = (
        0.25 * target_dominance
        + 0.22 * target_gap_term
        + 0.20 * safe_terminal
        + 0.12 * target_same_video
        + 0.08 * source_low_conf
        + 0.05 * source_entropy * max(target_same_video, 0.25)
        + 0.05 * target_support
        + 0.03 * source_smallness
        - source_insert_penalty
        - target_gap_penalty
    )
    return round(float(score), 6)


def _component_index(rows: list[dict[str, str]]) -> tuple[dict[int, dict[str, str]], dict[int, list[int]], dict[int, list[int]]]:
    by_seq: dict[int, dict[str, str]] = {}
    seqs_by_component: dict[int, list[int]] = defaultdict(list)
    seqs_by_gid: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        seq = _intish(row["seq"])
        component = _intish(row["component_label"])
        gid = _intish(row["predicted_global_id"])
        by_seq[seq] = row
        seqs_by_component[component].append(seq)
        seqs_by_gid[gid].append(seq)
    return by_seq, seqs_by_component, seqs_by_gid


def _time_components(time_rows: list[dict[str, str]], base_by_seq: dict[int, dict[str, str]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for row in time_rows:
        seq = _intish(row["seq"])
        if seq not in base_by_seq:
            continue
        out[_intish(row["component_label"])].append(seq)
    return out


def _component_gid(rows: list[dict[str, str]], base_by_seq: dict[int, dict[str, str]]) -> int | None:
    votes = Counter(_intish(base_by_seq[seq]["predicted_global_id"]) for seq in rows if seq in base_by_seq)
    return votes.most_common(1)[0][0] if votes else None


def _video_hint_fraction(seqs: list[int], base_by_seq: dict[int, dict[str, str]]) -> float:
    if not seqs:
        return 0.0
    hits = 0
    for seq in seqs:
        video = str(base_by_seq[seq].get("video", ""))
        if any(hint in video for hint in WEAK_VIDEO_HINTS):
            hits += 1
    return _safe_div(hits, len(seqs))


def _candidate_from_pair(
    *,
    time_label: int,
    time_seqs: list[int],
    source_component: int,
    target_component: int,
    source_seqs_in_time: list[int],
    target_seqs_in_time: list[int],
    base_by_seq: dict[int, dict[str, str]],
    seqs_by_component: dict[int, list[int]],
    theta: float | None,
    tag: str,
) -> dict[str, Any]:
    source_component_seqs = seqs_by_component[source_component]
    target_component_seqs = seqs_by_component[target_component]
    source_gid = _component_gid(source_seqs_in_time, base_by_seq)
    target_gid = _component_gid(target_seqs_in_time, base_by_seq)

    same_video_overlap = 0
    same_camera_overlap = 0
    for s in source_seqs_in_time:
        sr = base_by_seq[s]
        for t in target_seqs_in_time:
            tr = base_by_seq[t]
            ov = _overlap_frames(sr, tr)
            same_video_overlap = max(same_video_overlap, ov)
            if ov and str(sr.get("camera")) == str(tr.get("camera")):
                same_camera_overlap = max(same_camera_overlap, ov)

    source_count = len(source_seqs_in_time)
    target_count = len(target_seqs_in_time)
    time_size = len(time_seqs)
    source_fraction = _safe_div(source_count, len(source_component_seqs))
    target_dominance = _safe_div(target_count, time_size)
    source_smallness = 1.0 - min(source_count / 16.0, 1.0)
    target_support = min(target_count / 64.0, 1.0)
    weak_fraction = _video_hint_fraction(source_seqs_in_time, base_by_seq)
    temporal_features = _temporal_attach_features(
        source_seqs=source_seqs_in_time,
        target_seqs=target_seqs_in_time,
        source_component_seqs=source_component_seqs,
        target_component_seqs=target_component_seqs,
        base_by_seq=base_by_seq,
    )
    overlap_penalty = min(same_camera_overlap / 20.0, 1.0) + 0.25 * min(same_video_overlap / 60.0, 1.0)
    partial_penalty = 0.12 if source_fraction < 0.25 and len(source_component_seqs) > 16 else 0.0
    scheduler_score = (
        0.45 * target_dominance
        + 0.20 * target_support
        + 0.15 * source_smallness
        + 0.10 * min(source_count / 6.0, 1.0)
        + 0.10 * weak_fraction
        - 0.35 * overlap_penalty
        - partial_penalty
    )
    out = {
        "mode": "time_agglom_local_attach",
        "family": f"time_agglom:{tag}:theta={theta}:time_component={time_label}:source={source_component}:target={target_component}",
        "theta": theta,
        "tag": tag,
        "time_component": int(time_label),
        "time_component_size": int(time_size),
        "source_component_label": int(source_component),
        "target_component": int(target_component),
        "source_predicted_global_id": int(source_gid) if source_gid is not None else None,
        "target_predicted_global_id": int(target_gid) if target_gid is not None else None,
        "source_count": int(source_count),
        "source_component_size": int(len(source_component_seqs)),
        "target_count_in_time": int(target_count),
        "target_component_size": int(len(target_component_seqs)),
        "target_dominance": round(float(target_dominance), 6),
        "source_fraction": round(float(source_fraction), 6),
        "weak_video_source_fraction": round(float(weak_fraction), 6),
        "same_video_overlap_frames_max": int(same_video_overlap),
        "same_camera_overlap_frames_max": int(same_camera_overlap),
        "scheduler_score": round(float(scheduler_score), 6),
        "accepted_preview": [
            {
                "source_seqs": [int(seq) for seq in sorted(source_seqs_in_time)],
                "target_component": int(target_component),
                "target_top_seqs": [int(seq) for seq in sorted(target_seqs_in_time)[:20]],
                "source_component_label": int(source_component),
                "source_component_size": int(len(source_component_seqs)),
                "source_count": int(source_count),
                "time_component": int(time_label),
                "target_dominance": round(float(target_dominance), 6),
            }
        ],
    }
    out.update(temporal_features)
    out["critic_score"] = _critic_score(out)
    return out


def generate_candidates(
    *,
    base_assignment_csv: Path,
    time_assignment_csvs: list[tuple[Path, float | None, str]],
    max_source_count: int,
    max_source_component_size: int,
    min_target_count: int,
    min_target_component_size: int,
    min_target_dominance: float,
    max_same_video_overlap: int,
    max_same_camera_overlap: int,
    rank_by: str = "scheduler",
) -> list[dict[str, Any]]:
    base_rows = _read_rows(base_assignment_csv)
    base_by_seq, seqs_by_component, _seqs_by_gid = _component_index(base_rows)
    candidates: list[dict[str, Any]] = []
    seen = set()

    for time_csv, theta, tag in time_assignment_csvs:
        time_rows = _read_rows(time_csv)
        for time_label, time_seqs in _time_components(time_rows, base_by_seq).items():
            if len(time_seqs) < min_target_count + 1:
                continue
            comp_counts = Counter(_intish(base_by_seq[seq]["component_label"]) for seq in time_seqs)
            if len(comp_counts) < 2:
                continue
            target_component, target_count = comp_counts.most_common(1)[0]
            target_seqs = [seq for seq in time_seqs if _intish(base_by_seq[seq]["component_label"]) == target_component]
            if target_count < min_target_count:
                continue
            if len(seqs_by_component[target_component]) < min_target_component_size:
                continue
            target_dominance = _safe_div(target_count, len(time_seqs))
            if target_dominance < min_target_dominance:
                continue
            for source_component, source_count in comp_counts.most_common()[1:]:
                if source_count <= 0 or source_count > max_source_count:
                    continue
                if len(seqs_by_component[source_component]) > max_source_component_size:
                    continue
                source_seqs = [
                    seq for seq in time_seqs if _intish(base_by_seq[seq]["component_label"]) == source_component
                ]
                row = _candidate_from_pair(
                    time_label=time_label,
                    time_seqs=time_seqs,
                    source_component=source_component,
                    target_component=target_component,
                    source_seqs_in_time=source_seqs,
                    target_seqs_in_time=target_seqs,
                    base_by_seq=base_by_seq,
                    seqs_by_component=seqs_by_component,
                    theta=theta,
                    tag=tag,
                )
                if row["same_video_overlap_frames_max"] > max_same_video_overlap:
                    continue
                if row["same_camera_overlap_frames_max"] > max_same_camera_overlap:
                    continue
                sig = (
                    tuple(row["accepted_preview"][0]["source_seqs"]),
                    int(row["target_component"]),
                )
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append(row)

    if rank_by == "critic":
        candidates.sort(
            key=lambda row: (
                float(row.get("critic_score", 0.0)),
                float(row["target_dominance"]),
                float(row.get("target_gap_min", -1)),
                float(row["scheduler_score"]),
            ),
            reverse=True,
        )
    elif rank_by == "hybrid":
        candidates.sort(
            key=lambda row: (
                0.5 * float(row.get("critic_score", 0.0)) + 0.5 * float(row["scheduler_score"]),
                float(row.get("critic_score", 0.0)),
                float(row["target_dominance"]),
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda row: (
                float(row["scheduler_score"]),
                float(row["target_dominance"]),
                int(row["source_count"]),
                float(row["weak_video_source_fraction"]),
            ),
            reverse=True,
        )
    for rank, row in enumerate(candidates, start=1):
        row["candidate_rank"] = int(rank)
    return candidates


def _compose_rows(candidates: list[dict[str, Any]], sizes: list[int], top_n: int, rank_by: str = "scheduler") -> list[dict[str, Any]]:
    singles = candidates[: max(top_n * 4, top_n)]
    rows: list[dict[str, Any]] = []
    used_sigs = set()
    for size in sizes:
        if size <= 1:
            for row in singles:
                rows.append(dict(row))
            continue
        for combo in itertools.combinations(singles, size):
            source_seqs: set[int] = set()
            source_components: set[int] = set()
            target_components: set[int] = set()
            ok = True
            for row in combo:
                preview = row["accepted_preview"][0]
                seqs = set(int(seq) for seq in preview["source_seqs"])
                if source_seqs.intersection(seqs):
                    ok = False
                    break
                source_seqs.update(seqs)
                source_components.add(int(row["source_component_label"]))
                target_components.add(int(row["target_component"]))
            if not ok or len(source_components) != size:
                continue
            sig = (tuple(sorted(source_seqs)), tuple(sorted(target_components)))
            if sig in used_sigs:
                continue
            used_sigs.add(sig)
            score = sum(float(row["scheduler_score"]) for row in combo) / math.sqrt(size)
            rows.append(
                {
                    "mode": "time_agglom_local_attach_combo",
                    "family": "+".join(str(row["family"]) for row in combo),
                    "candidate_rank": min(int(row["candidate_rank"]) for row in combo),
                    "combo_size": int(size),
                    "scheduler_score": round(float(score), 6),
                    "source_count": int(sum(int(row["source_count"]) for row in combo)),
                    "source_components": sorted(source_components),
                    "target_components": sorted(target_components),
                    "target_dominance_min": round(float(min(float(row["target_dominance"]) for row in combo)), 6),
                    "weak_video_source_fraction_mean": round(
                        float(sum(float(row["weak_video_source_fraction"]) for row in combo) / size),
                        6,
                    ),
                    "same_video_overlap_frames_max": int(max(int(row["same_video_overlap_frames_max"]) for row in combo)),
                    "same_camera_overlap_frames_max": int(max(int(row["same_camera_overlap_frames_max"]) for row in combo)),
                    "accepted_preview": [
                        preview for row in combo for preview in row["accepted_preview"]
                    ],
                }
            )
    if rank_by == "critic":
        rows.sort(
            key=lambda row: (
                float(row.get("critic_score", 0.0)),
                float(row.get("target_dominance", row.get("target_dominance_min", 0.0))),
                float(row.get("scheduler_score", 0.0)),
            ),
            reverse=True,
        )
    elif rank_by == "hybrid":
        rows.sort(
            key=lambda row: (
                0.5 * float(row.get("critic_score", 0.0)) + 0.5 * float(row.get("scheduler_score", 0.0)),
                float(row.get("critic_score", 0.0)),
                float(row.get("scheduler_score", 0.0)),
            ),
            reverse=True,
        )
    else:
        rows.sort(
            key=lambda row: (
                float(row["scheduler_score"]),
                int(row.get("combo_size", 1)),
                float(row.get("weak_video_source_fraction_mean", row.get("weak_video_source_fraction", 0.0))),
            ),
            reverse=True,
        )
    selected: list[dict[str, Any]] = []
    used_families = set()
    for row in rows:
        family = str(row["family"])
        if family in used_families:
            continue
        used_families.add(family)
        selected.append(row)
        if len(selected) >= top_n:
            break
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_rank",
        "mode",
        "scheduler_score",
        "critic_score",
        "combo_size",
        "source_count",
        "source_component_label",
        "target_component",
        "source_components",
        "target_components",
        "time_component",
        "time_component_size",
        "target_dominance",
        "target_dominance_min",
        "source_fraction",
        "weak_video_source_fraction",
        "weak_video_source_fraction_mean",
        "same_video_overlap_frames_max",
        "same_camera_overlap_frames_max",
        "source_avg_conf_mean",
        "source_n_dets_mean",
        "source_end_rank_frac_mean",
        "source_start_rank_frac_mean",
        "source_terminal_fraction",
        "source_prev_gap_min",
        "source_next_gap_min",
        "target_prev_gap_min",
        "target_next_gap_min",
        "target_gap_min",
        "target_same_video_fraction",
        "source_same_video_fraction",
        "source_video_entropy_norm",
        "family",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _parse_time_csvs(items: list[str]) -> list[tuple[Path, float | None, str]]:
    out = []
    for idx, text in enumerate(items, start=1):
        path_text, theta_text, tag = text, "", ""
        if "::" in text:
            parts = text.split("::")
            path_text = parts[0]
            theta_text = parts[1] if len(parts) > 1 else ""
            tag = parts[2] if len(parts) > 2 else ""
        theta = _floatish(theta_text, float("nan")) if theta_text else None
        out.append((Path(path_text), None if theta is not None and math.isnan(theta) else theta, tag or f"t{idx}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-assignment-csv", required=True)
    ap.add_argument("--time-assignment-csv", action="append", required=True, help="path[::theta[::tag]]")
    ap.add_argument("--max-source-count", type=int, default=8)
    ap.add_argument("--max-source-component-size", type=int, default=40)
    ap.add_argument("--min-target-count", type=int, default=24)
    ap.add_argument("--min-target-component-size", type=int, default=32)
    ap.add_argument("--min-target-dominance", type=float, default=0.74)
    ap.add_argument("--max-same-video-overlap", type=int, default=0)
    ap.add_argument("--max-same-camera-overlap", type=int, default=0)
    ap.add_argument("--compose-sizes", default="1,2,3")
    ap.add_argument("--selected-top-n", type=int, default=12)
    ap.add_argument("--rank-by", choices=["scheduler", "critic", "hybrid"], default="scheduler")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    candidates = generate_candidates(
        base_assignment_csv=Path(args.base_assignment_csv),
        time_assignment_csvs=_parse_time_csvs(args.time_assignment_csv),
        max_source_count=int(args.max_source_count),
        max_source_component_size=int(args.max_source_component_size),
        min_target_count=int(args.min_target_count),
        min_target_component_size=int(args.min_target_component_size),
        min_target_dominance=float(args.min_target_dominance),
        max_same_video_overlap=int(args.max_same_video_overlap),
        max_same_camera_overlap=int(args.max_same_camera_overlap),
        rank_by=str(args.rank_by),
    )
    sizes = [int(part) for part in str(args.compose_sizes).split(",") if part.strip()]
    selected = _compose_rows(candidates, sizes, int(args.selected_top_n), rank_by=str(args.rank_by))
    out = {
        "base_assignment_csv": str(args.base_assignment_csv),
        "time_assignment_csvs": [str(path) for path, _theta, _tag in _parse_time_csvs(args.time_assignment_csv)],
        "raw_candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "selected": selected,
        "rows": candidates,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), selected)
    print(json.dumps({"json": str(json_path), "raw_candidate_count": len(candidates), "selected_count": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
