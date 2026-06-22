#!/usr/bin/env python
"""Schedule the next no-anchor candidates for expensive DS1 full scoring.

This utility is intentionally offline: it does not use anchors and it does not
call the GT full scorer.  It reads already-generated no-anchor candidate tables,
applies delivery/metric guardrails learned from previous failures, then selects
a diverse top-k manifest for the next remote full-score budget.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from functools import lru_cache

import numpy as np


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(int(value))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    rows = []
    for key in ("top", "rows", "full_rows", "top_full_rows", "results", "top_rows"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    if not rows and any(key in data for key in ("tracklet_pair_f1", "pair_f1", "full_idf1")):
        rows.append(data)
    return rows


def _oracle_or_gt_selection_reason(row: dict[str, Any]) -> str | None:
    path_text = str(row.get("_source_file") or row.get("artifact") or "").lower()
    mode_text = str(row.get("mode") or row.get("policy_name") or "").lower()
    gt_flags = {
        "uses_gt_for_analysis_only",
        "uses_gt_for_filter_selection",
        "selection_uses_gt_metric",
    }
    for key in gt_flags:
        if _as_bool(row.get(key)) is True:
            return key
    if (
        "with_oracle" in path_text
        or "oracle_repair" in path_text
        or "pervideo_filter_oracle" in path_text
        or mode_text.startswith("oracle_")
    ):
        return "oracle_diagnostic"
    return None


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in paths:
        matches = sorted(Path().glob(text)) if any(ch in text for ch in "*?[]") else [Path(text)]
        for path in matches:
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                loaded = _load_json_rows(path)
            elif path.suffix.lower() == ".csv":
                loaded = _load_csv_rows(path)
            else:
                continue
            for rank, row in enumerate(loaded, start=1):
                rows.append({"_source_file": str(path), "_source_rank": rank, **row})
    return rows


@lru_cache(maxsize=512)
def _load_artifact_rows(path_text: str) -> tuple[dict[str, Any], ...]:
    path = Path(path_text)
    if not path.is_file():
        return tuple()
    try:
        if path.suffix.lower() == ".json":
            rows = _load_json_rows(path)
        elif path.suffix.lower() == ".csv":
            rows = _load_csv_rows(path)
        else:
            rows = []
    except Exception:
        return tuple()
    return tuple(dict(row) for row in rows if isinstance(row, dict))


def _metric_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _close_metric(a: float | None, b: float | None, tol: float = 5.0e-4) -> bool:
    if a is None or b is None:
        return True
    return abs(float(a) - float(b)) <= tol


def _candidate_artifact_paths(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    artifact = str(row.get("artifact") or "")
    if artifact:
        paths.append(artifact)
    source = str(row.get("_source_file") or "")
    if source:
        source_path = Path(source)
        if source_path.suffix.lower() == ".csv":
            sibling = str(source_path.with_suffix(".json"))
            if sibling not in paths:
                paths.append(sibling)
        if source not in paths:
            paths.append(source)
    return paths


def _artifact_known_row(row: dict[str, Any]) -> dict[str, Any] | None:
    return _artifact_matching_row(row, require_full_context=True, include_source_file=False)


def _artifact_matching_row(
    row: dict[str, Any], *, require_full_context: bool = False, include_source_file: bool = True
) -> dict[str, Any] | None:
    mode = str(row.get("mode") or "")
    pair = _metric_value(row, "pair_f1", "tracklet_pair_f1")
    prec = _metric_value(row, "pair_precision", "tracklet_pair_precision")
    rec = _metric_value(row, "pair_recall", "tracklet_pair_recall")
    fallback = None

    def has_required_context(cand: dict[str, Any]) -> bool:
        known_full = _metric_value(cand, "known_full_idf1", "full_idf1", "idf1")
        has_full_context = known_full is not None and (
            "full_hota" in cand or "hota" in cand or "full_assa" in cand or "assa" in cand
        )
        return (not require_full_context) or has_full_context

    def matches(cand: dict[str, Any]) -> bool:
        cand_mode = str(cand.get("mode") or "")
        if mode and cand_mode and mode != cand_mode:
            return False
        if not _close_metric(pair, _metric_value(cand, "pair_f1", "tracklet_pair_f1")):
            return False
        if not _close_metric(prec, _metric_value(cand, "pair_precision", "tracklet_pair_precision")):
            return False
        if not _close_metric(rec, _metric_value(cand, "pair_recall", "tracklet_pair_recall")):
            return False
        return True

    source_path = Path(str(row.get("_source_file") or ""))
    source_rank = _as_float(row.get("_source_rank"))
    artifacts = _candidate_artifact_paths(row) if include_source_file else [str(row.get("artifact") or "")]
    for artifact in artifacts:
        if not artifact:
            continue
        rows = _load_artifact_rows(artifact)
        if not rows:
            continue
        artifact_path = Path(artifact)
        direct_rank_checked = False
        if (
            source_rank is not None
            and source_path.suffix.lower() == ".csv"
            and artifact_path.suffix.lower() == ".json"
            and artifact_path.with_suffix(".csv") == source_path
        ):
            idx = int(source_rank) - 1
            if 0 <= idx < len(rows):
                direct_rank_checked = True
                cand = rows[idx]
                if has_required_context(cand):
                    if fallback is None:
                        fallback = cand
                    if matches(cand):
                        return cand
            if len(rows) > 1000:
                continue
        for cand in rows:
            if direct_rank_checked and cand is rows[int(source_rank) - 1]:
                continue
            if not has_required_context(cand):
                continue
            if fallback is None:
                fallback = cand
            if matches(cand):
                return cand
    return fallback


def _feature_value(row: dict[str, Any], key: str) -> float | None:
    if key.startswith("preview_mean_"):
        preview = row.get("accepted_preview")
        base = key.removeprefix("preview_mean_")
        if isinstance(preview, list):
            vals = [_as_float(item.get(base)) for item in preview if isinstance(item, dict)]
            vals = [val for val in vals if val is not None]
            return float(np.mean(vals)) if vals else None
    if key.startswith("preview_min_"):
        preview = row.get("accepted_preview")
        base = key.removeprefix("preview_min_")
        if isinstance(preview, list):
            vals = [_as_float(item.get(base)) for item in preview if isinstance(item, dict)]
            vals = [val for val in vals if val is not None]
            return float(min(vals)) if vals else None
    if key.startswith("preview_max_"):
        preview = row.get("accepted_preview")
        base = key.removeprefix("preview_max_")
        if isinstance(preview, list):
            vals = [_as_float(item.get(base)) for item in preview if isinstance(item, dict)]
            vals = [val for val in vals if val is not None]
            return float(max(vals)) if vals else None
    return _as_float(row.get(key))


def _model_score(row: dict[str, Any], model: dict[str, Any] | None) -> float | None:
    if not model:
        return None
    cols = model.get("columns")
    coefs = model.get("coef")
    means = model.get("mean")
    scales = model.get("scale")
    fills = model.get("fill_values", {})
    if not isinstance(cols, list) or not isinstance(coefs, list):
        return None
    score = _as_float(model.get("intercept"), 0.0) or 0.0
    for idx, key in enumerate(cols):
        if idx >= len(coefs) or not isinstance(key, str):
            break
        value = _feature_value(row, key)
        if value is None and isinstance(fills, dict):
            value = _as_float(fills.get(key), 0.0)
        if value is None:
            value = 0.0
        mean = _as_float(means[idx] if isinstance(means, list) and idx < len(means) else 0.0, 0.0) or 0.0
        scale = _as_float(scales[idx] if isinstance(scales, list) and idx < len(scales) else 1.0, 1.0) or 1.0
        if abs(scale) < 1.0e-9:
            scale = 1.0
        score += (_as_float(coefs[idx], 0.0) or 0.0) * ((float(value) - mean) / scale)
    lo = _as_float(model.get("target_min"))
    hi = _as_float(model.get("target_max"))
    if lo is not None and hi is not None and hi >= lo:
        margin = _as_float(model.get("target_clamp_margin"), 0.001)
        margin = 0.001 if margin is None or margin < 0 else float(margin)
        score = min(max(score, float(lo) - margin), float(hi) + margin)
    return float(score)


def _valid_idf_proxy(value: float | None) -> float | None:
    if value is None:
        return None
    if 0.0 < float(value) <= 1.0:
        return float(value)
    return None


def _try_literal_signature(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _component_pair(row: dict[str, Any]) -> tuple[Any, Any] | None:
    source = row.get("source_component_label")
    target = row.get("target_component")
    if source not in (None, "") or target not in (None, ""):
        return source, target
    source = row.get("source")
    target = row.get("target")
    if source not in (None, "") or target not in (None, ""):
        return source, target
    source = row.get("source_rep")
    target = row.get("target_rep")
    if source not in (None, "") or target not in (None, ""):
        return source, target
    preview = row.get("accepted_preview")
    if isinstance(preview, list) and preview:
        first = preview[0]
        if isinstance(first, dict):
            source = first.get("source_component_label")
            target = first.get("target_component")
            if source not in (None, "") or target not in (None, ""):
                return source, target
            source = first.get("source")
            target = first.get("target")
            if source not in (None, "") or target not in (None, ""):
                return source, target
            source = first.get("source_rep")
            target = first.get("target_rep")
            if source not in (None, "") or target not in (None, ""):
                return source, target
    return None


def _family_key(row: dict[str, Any]) -> str:
    mode = str(row.get("mode") or "")
    artifact_path = Path(str(row.get("artifact") or row.get("_source_file") or ""))
    artifact = artifact_path.stem or artifact_path.name
    action_signature = row.get("action_signature")
    if action_signature not in (None, ""):
        return f"{mode}:action:{str(action_signature)[:200]}"
    global_modes = {
        "louvain",
        "cannotlink_nms_singleton",
        "assignment_multiview_merge",
        "assignment_state_policy",
        "component_merge",
        "time_agglom",
    }
    if mode in global_modes:
        return f"{mode}:artifact:{artifact}"
    pair = _component_pair(row)
    if pair is None:
        artifact_row = _artifact_matching_row(row, require_full_context=False)
        if artifact_row is not None:
            pair = _component_pair(artifact_row)
    if pair is not None:
        source, target = pair
        return f"{mode}:component:{source}->{target}"
    signature = _try_literal_signature(row.get("signature"))
    if isinstance(signature, tuple) and signature:
        if signature[0] == "accepted_preview" and len(signature) > 1:
            preview = signature[1]
            if isinstance(preview, tuple) and preview:
                first = preview[0]
                if isinstance(first, tuple) and len(first) == 2:
                    seqs, target = first
                    seq_count = len(seqs) if isinstance(seqs, tuple) else 1
                    return f"{mode}:source_target:{seq_count}:{target}"
        return f"{mode}:signature:{str(signature)[:120]}"
    return f"{mode}:artifact:{artifact}"


def _label_full_value(row: dict[str, Any]) -> float | None:
    full_from_idf1 = _as_float(row.get("idf1")) if ("hota" in row or "assa" in row) else None
    return _as_float(row.get("known_full_idf1"), _as_float(row.get("full_idf1"), full_from_idf1))


def _load_fullscore_label_rows(paths: list[str]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for row in _load_rows(paths):
        known_full = _label_full_value(row)
        if known_full is None:
            continue
        family = _family_key(row)
        old = labels.get(family)
        item = {"known_full_idf1": float(known_full), "source": str(row.get("_source_file") or row.get("artifact") or "")}
        if old is None or float(known_full) < float(old["known_full_idf1"]):
            labels[family] = item
    return labels


def _side_effect_risk(row: dict[str, Any]) -> tuple[float, list[str]]:
    risk = 0.0
    reasons: list[str] = []
    moved = _as_float(row.get("moved_tracklets"), _as_float(row.get("moved_tracklets_norm"), 0.0)) or 0.0
    accepted = _as_float(row.get("accepted_reassignments"), _as_float(row.get("accepted_reassignments_norm"), 0.0)) or 0.0
    if moved > 20.0:
        risk += 2.0
        reasons.append("moved_tracklets>20")
    if moved >= 200.0:
        risk += 2.0
        reasons.append("massive_tracklet_relabel")
    elif moved >= 50.0:
        risk += 1.0
        reasons.append("broad_tracklet_relabel")
    if accepted > 1.0:
        risk += 1.5
        reasons.append("multi_edge_preview")
    if moved >= 50.0 and accepted >= 4.0:
        risk += 1.0
        reasons.append("broad_multi_edge_relabel")
    preview = row.get("accepted_preview")
    target_sizes: list[float] = []
    target_video_counts: list[float] = []
    endpoint_support_edges: list[float] = []
    endpoint_like = "endpoint" in str(row.get("mode") or "").lower()
    combined_opponent_risks: list[float] = []
    temporal_opponent_risks: list[float] = []
    visual_opponent_risks: list[float] = []
    same_video_overlaps: list[float] = []
    weak_margin_fractions: list[float] = []
    if isinstance(preview, list):
        for item in preview:
            if not isinstance(item, dict):
                continue
            size = _as_float(item.get("target_size"), _as_float(item.get("target_component_size_max")))
            if size is not None:
                target_sizes.append(float(size))
            if item.get("best_endpoint_score") not in (None, "") or item.get("source_seq") not in (None, "") and item.get("target_seq") not in (None, ""):
                endpoint_like = True
            support_edges = _as_float(item.get("support_edges"))
            if support_edges is not None:
                endpoint_support_edges.append(float(support_edges))
            video_count = _as_float(item.get("target_video_count"))
            if video_count is not None:
                target_video_counts.append(float(video_count))
            for key, bucket in (
                ("combined_opponent_risk_score", combined_opponent_risks),
                ("temporal_opponent_risk_score", temporal_opponent_risks),
                ("visual_opponent_risk_score", visual_opponent_risks),
                ("max_same_video_overlap_frames", same_video_overlaps),
                ("view_weak_margin_fraction", weak_margin_fractions),
            ):
                value = _as_float(item.get(key))
                if value is not None:
                    bucket.append(float(value))
    for key, bucket in (
        ("combined_opponent_risk_score", combined_opponent_risks),
        ("temporal_opponent_risk_score", temporal_opponent_risks),
        ("visual_opponent_risk_score", visual_opponent_risks),
        ("max_same_video_overlap_frames", same_video_overlaps),
        ("view_weak_margin_fraction", weak_margin_fractions),
    ):
        value = _as_float(row.get(key))
        if value is not None:
            bucket.append(float(value))
    target_size = max(target_sizes, default=_as_float(row.get("target_size"), 0.0) or 0.0)
    target_video_count = max(target_video_counts, default=_as_float(row.get("target_video_count"), 0.0) or 0.0)
    if target_video_count >= 5.0:
        risk += 1.0
        reasons.append("large_multivideo_target")
    if target_size >= 200.0:
        risk += 0.75
        reasons.append("large_target_component")
    elif target_size >= 150.0:
        risk += 0.5
        reasons.append("medium_large_target_component")
    if moved >= 50.0 and target_size >= 150.0:
        risk += 1.0
        reasons.append("broad_large_target_relabel")
    if endpoint_like:
        risk += 0.75
        reasons.append("endpoint_direct_action")
        if target_size >= 100.0:
            risk += 0.75
            reasons.append("endpoint_large_target_component")
        if endpoint_support_edges and min(endpoint_support_edges) <= 1.0:
            risk += 0.75
            reasons.append("endpoint_single_support_edge")
    combined_opponent_risk = max(combined_opponent_risks, default=0.0)
    temporal_opponent_risk = max(temporal_opponent_risks, default=0.0)
    visual_opponent_risk = max(visual_opponent_risks, default=0.0)
    max_same_video_overlap = max(same_video_overlaps, default=0.0)
    weak_margin_fraction = max(weak_margin_fractions, default=0.0)
    if max_same_video_overlap > 0.0:
        risk += 2.0
        reasons.append("temporal_same_video_overlap")
    if combined_opponent_risk >= 0.75:
        risk += 1.5
        reasons.append("high_combined_opponent_risk")
    elif combined_opponent_risk >= 0.50:
        risk += 0.75
        reasons.append("medium_combined_opponent_risk")
    if temporal_opponent_risk >= 0.50:
        risk += 0.75
        reasons.append("high_temporal_opponent_risk")
    if visual_opponent_risk >= 0.80:
        risk += 0.5
        reasons.append("high_visual_opponent_risk")
    if weak_margin_fraction >= 0.50:
        risk += 0.5
        reasons.append("weak_multiview_margin")
    return float(risk), reasons


def _normalize_row(
    row: dict[str, Any],
    model: dict[str, Any] | None,
    fullscore_labels: dict[str, dict[str, Any]] | None = None,
    side_effect_risk_weight: float = 0.0,
) -> dict[str, Any]:
    artifact_known = _artifact_known_row(row)
    pair_f1 = _as_float(row.get("pair_f1"), _as_float(row.get("tracklet_pair_f1"), 0.0)) or 0.0
    pair_precision = _as_float(row.get("pair_precision"), _as_float(row.get("tracklet_pair_precision"), 0.0)) or 0.0
    pair_recall = _as_float(row.get("pair_recall"), _as_float(row.get("tracklet_pair_recall"), 0.0)) or 0.0
    full_from_idf1 = _as_float(row.get("idf1")) if ("hota" in row or "assa" in row) else None
    known_full = _as_float(row.get("known_full_idf1"), _as_float(row.get("full_idf1"), full_from_idf1))
    if known_full is None and artifact_known is not None:
        artifact_full_from_idf1 = _as_float(artifact_known.get("idf1")) if (
            "hota" in artifact_known or "assa" in artifact_known
        ) else None
        known_full = _as_float(
            artifact_known.get("known_full_idf1"),
            _as_float(artifact_known.get("full_idf1"), artifact_full_from_idf1),
        )
    learned = _valid_idf_proxy(_as_float(row.get("learned_proxy_full_idf1"), _as_float(row.get("learned_full_proxy"))))
    if learned is None:
        learned = _valid_idf_proxy(_model_score(row, model))
    if learned is None:
        learned = _valid_idf_proxy(_as_float(row.get("full_side_effect_proxy"))) or pair_f1
    delivery = _as_float(row.get("delivery_tracklets_min"))
    if delivery is None and artifact_known is not None:
        delivery = _as_float(artifact_known.get("delivery_tracklets_min"))
    if delivery is None:
        vals = [
            _as_float(row.get("assigned_tracklets")),
            _as_float(row.get("output_tracklets")),
            _as_float(row.get("eval_tracklets")),
        ]
        if artifact_known is not None:
            vals.extend(
                [
                    _as_float(artifact_known.get("assigned_tracklets")),
                    _as_float(artifact_known.get("output_tracklets")),
                    _as_float(artifact_known.get("eval_tracklets")),
                ]
            )
        vals = [val for val in vals if val is not None]
        delivery = float(min(vals)) if vals else None
    moved = _as_float(row.get("moved_tracklets"), 0.0) or 0.0
    accepted = _as_float(row.get("accepted_reassignments"), 0.0) or 0.0
    source_acceptor = _as_float(row.get("source_acceptor_rank_score"), _as_float(row.get("source_acceptor_score"), 0.0)) or 0.0
    family = _family_key(row)
    label_source = ""
    if fullscore_labels and family in fullscore_labels:
        labelled = fullscore_labels[family]
        labelled_full = _as_float(labelled.get("known_full_idf1"))
        if labelled_full is not None and (known_full is None or labelled_full < float(known_full)):
            known_full = float(labelled_full)
            label_source = str(labelled.get("source") or "")
    risk = 0.0
    risk += max(0.0, 0.73 - pair_recall) * 0.20
    risk += max(0.0, 0.80 - pair_precision) * 0.10
    risk += max(0.0, moved - 32.0) * 0.00015
    risk += max(0.0, accepted - 4.0) * 0.0005
    if delivery is not None:
        risk += max(0.0, 7200.0 - delivery) * 0.00001
    side_effect_risk, side_effect_reasons = _side_effect_risk(row)
    risk += max(0.0, float(side_effect_risk_weight)) * side_effect_risk
    scheduler_score = float(learned) + 0.01 * max(source_acceptor - 0.5, 0.0) - risk
    return {
        **row,
        "pair_f1_norm": float(pair_f1),
        "pair_precision_norm": float(pair_precision),
        "pair_recall_norm": float(pair_recall),
        "known_full_idf1_norm": known_full,
        "predicted_full_idf1": float(learned),
        "delivery_tracklets_min_norm": delivery,
        "moved_tracklets_norm": float(moved),
        "accepted_reassignments_norm": float(accepted),
        "source_acceptor_rank_score_norm": float(source_acceptor),
        "scheduler_risk_penalty": float(risk),
        "side_effect_risk": float(side_effect_risk),
        "side_effect_reasons": side_effect_reasons,
        "scheduler_score": float(scheduler_score),
        "scheduler_family": family,
        "is_full_scored_norm": bool(_as_bool(row.get("is_full_scored")) or known_full is not None),
        "artifact_known_full_source": label_source or (str(row.get("artifact") or "") if artifact_known is not None else ""),
        "oracle_or_gt_selection_reason": _oracle_or_gt_selection_reason(row),
    }


def _eligible(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if _as_bool(row.get("uses_anchors")) is True or _as_bool(row.get("uses_gt_for_training_or_anchors")) is True:
        reasons.append("anchor_or_gt_training")
    if row.get("oracle_or_gt_selection_reason"):
        reasons.append(str(row["oracle_or_gt_selection_reason"]))
    if row["pair_f1_norm"] < float(args.min_pair_f1):
        reasons.append("low_pair_f1")
    if row["pair_precision_norm"] < float(args.min_pair_precision):
        reasons.append("low_pair_precision")
    if row["pair_recall_norm"] < float(args.min_pair_recall):
        reasons.append("low_pair_recall")
    delivery = row.get("delivery_tracklets_min_norm")
    if delivery is not None and float(delivery) < float(args.min_delivery_tracklets):
        reasons.append("low_delivery")
    if row["predicted_full_idf1"] < float(args.min_predicted_full):
        reasons.append("low_predicted_full")
    min_scheduler_score = getattr(args, "min_scheduler_score", None)
    if min_scheduler_score is not None and row["scheduler_score"] < float(min_scheduler_score):
        reasons.append("low_scheduler_score")
    if args.require_predicted_above_current and row["predicted_full_idf1"] <= float(args.current_best_full_idf1):
        reasons.append("predicted_not_above_current_best")
    if row["is_full_scored_norm"] and not args.include_full_scored:
        reasons.append("already_full_scored")
    known_full = row.get("known_full_idf1_norm")
    if known_full is not None and float(known_full) < float(args.current_best_full_idf1):
        reasons.append("known_below_current_best")
    return not reasons, reasons


def _select(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    mode_counts: dict[str, int] = {}
    max_per_mode = int(getattr(args, "max_per_mode", 0) or 0)
    for row in sorted(rows, key=lambda item: float(item["scheduler_score"]), reverse=True):
        if row["scheduler_family"] in used_families:
            continue
        mode = str(row.get("mode") or "")
        if max_per_mode > 0 and mode_counts.get(mode, 0) >= max_per_mode:
            continue
        selected.append(row)
        used_families.add(row["scheduler_family"])
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if len(selected) >= int(args.top_n):
            break
    return selected


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    keys = [
        "scheduler_rank",
        "scheduler_score",
        "predicted_full_idf1",
        "known_full_idf1_norm",
        "pair_f1_norm",
        "pair_precision_norm",
        "pair_recall_norm",
        "delivery_tracklets_min_norm",
        "moved_tracklets_norm",
        "accepted_reassignments_norm",
        "scheduler_risk_penalty",
        "side_effect_risk",
        "side_effect_reasons",
        "scheduler_family",
        "mode",
        "artifact",
        "_source_file",
        "_source_rank",
        "signature",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"scheduler_rank": rank, **row})


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        bad_artifact = Path(tmp) / "state_policy.json"
        pair_artifact = Path(tmp) / "pair_artifact.json"
        bad_artifact.write_text(
            json.dumps(
                {
                    "top": [
                        {
                            "mode": "assignment_state_policy",
                            "tracklet_pair_f1": 0.954691,
                            "tracklet_pair_precision": 0.918405,
                            "tracklet_pair_recall": 0.993962,
                            "full_idf1": 0.085412,
                            "full_hota": 0.154201,
                            "output_tracklets": 431,
                        }
                    ]
                }
            )
        )
        pair_artifact.write_text(
            json.dumps(
                {
                    "top": [
                        {
                            "mode": "conflict_subcluster_reassign_candidate_search",
                            "source_component_label": 21,
                            "target_component": 19,
                            "tracklet_pair_f1": 0.767329,
                            "tracklet_pair_precision": 0.814359,
                            "tracklet_pair_recall": 0.725434,
                        }
                    ]
                }
            )
        )
        pair_csv = Path(tmp) / "pair_artifact.csv"
        pair_csv.write_text(
            "mode,tracklet_pair_f1,tracklet_pair_precision,tracklet_pair_recall\n"
            "conflict_subcluster_reassign_candidate_search,0.767329,0.814359,0.725434\n"
        )
        family_row = {
            "mode": "conflict_subcluster_reassign_candidate_search",
            "artifact": str(pair_artifact),
            "pair_f1": 0.767329,
            "pair_precision": 0.814359,
            "pair_recall": 0.725434,
            "signature": "('accepted_preview', (((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 19),))",
        }
        assert _family_key(family_row) == "conflict_subcluster_reassign_candidate_search:component:21->19"
        csv_family_row = {
            "mode": "conflict_subcluster_reassign_candidate_search",
            "_source_file": str(pair_csv),
            "pair_f1": 0.767329,
            "pair_precision": 0.814359,
            "pair_recall": 0.725434,
        }
        assert _family_key(csv_family_row) == "conflict_subcluster_reassign_candidate_search:component:21->19"
        component_preview_row = {
            "mode": "assignment_component_merge",
            "accepted_preview": [{"source": 7, "target": 11, "source_rep": 70, "target_rep": 110}],
        }
        assert _family_key(component_preview_row) == "assignment_component_merge:component:7->11"
        action_row = {
            "mode": "source_plus_target_new",
            "action_signature": "source_plus_target_new:1549,1604|source_plus_target_new:1619,1622",
        }
        assert (
            _family_key(action_row)
            == "source_plus_target_new:action:source_plus_target_new:1549,1604|source_plus_target_new:1619,1622"
        )
        rows = [
            {"pair_f1": 0.77, "pair_precision": 0.82, "pair_recall": 0.73, "learned_proxy_full_idf1": 0.656, "signature": "('accepted_preview', (((1, 2), 9),))"},
            {"pair_f1": 0.78, "pair_precision": 0.82, "pair_recall": 0.74, "learned_proxy_full_idf1": 0.654, "signature": "('accepted_preview', (((1, 2), 9),))"},
            {"pair_f1": 0.76, "pair_precision": 0.81, "pair_recall": 0.72, "learned_proxy_full_idf1": 0.655, "signature": "('accepted_preview', (((3,), 8),))"},
            {"pair_f1": 0.99, "pair_precision": 0.99, "pair_recall": 0.99, "learned_proxy_full_idf1": 0.99, "mode": "oracle_all_gt_majority"},
            {
                "pair_f1": 0.95469,
                "pair_precision": 0.918405,
                "pair_recall": 0.993961,
                "learned_proxy_full_idf1": 0.7478,
                "mode": "assignment_state_policy",
                "artifact": str(bad_artifact),
            },
        ]
        ns = argparse.Namespace(
            min_pair_f1=0.7,
            min_pair_precision=0.7,
            min_pair_recall=0.7,
            min_delivery_tracklets=7000,
            min_predicted_full=0.0,
            current_best_full_idf1=0.65524,
            include_full_scored=False,
            require_predicted_above_current=True,
            top_n=2,
            max_per_mode=0,
        )
        norm = [_normalize_row(row, None) for row in rows]
        eligible = []
        for row in norm:
            ok, reasons = _eligible(row, ns)
            row["scheduler_exclusion_reasons"] = reasons
            if ok:
                eligible.append(row)
        selected = _select(eligible, ns)
        assert len(selected) == 1, selected
        assert selected[0]["scheduler_family"] == ":source_target:2:9", selected
        oracle = [row for row in norm if row.get("mode") == "oracle_all_gt_majority"][0]
        ok, reasons = _eligible(oracle, ns)
        assert not ok and "oracle_diagnostic" in reasons, reasons
        bad = [row for row in norm if row.get("mode") == "assignment_state_policy"][0]
        ok, reasons = _eligible(bad, ns)
        assert not ok and "known_below_current_best" in reasons and "low_delivery" in reasons, reasons
        label_file = Path(tmp) / "critic_labels.json"
        label_file.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "mode": "conflict_subcluster_reassign_candidate_search",
                            "source_component_label": 21,
                            "target_component": 19,
                            "full_idf1": 0.650001,
                            "full_hota": 0.51,
                        }
                    ]
                }
            )
        )
        labels = _load_fullscore_label_rows([str(label_file)])
        labelled = _normalize_row(family_row, None, labels)
        assert labelled["known_full_idf1_norm"] == 0.650001, labelled
        ok, reasons = _eligible(labelled, ns)
        assert not ok and "known_below_current_best" in reasons, reasons
        risky = _normalize_row(
            {
                "learned_proxy_full_idf1": 0.662,
                "tracklet_pair_f1": 0.77,
                "tracklet_pair_precision": 0.82,
                "tracklet_pair_recall": 0.73,
                "moved_tracklets": 40,
                "accepted_reassignments": 4,
                "accepted_preview": [{"target_size": 172}],
            },
            None,
            side_effect_risk_weight=0.002,
        )
        assert "moved_tracklets>20" in risky["side_effect_reasons"], risky
        assert risky["scheduler_score"] < risky["predicted_full_idf1"], risky
        zero_placeholder = _normalize_row(
            {
                "learned_full_proxy": 0.0,
                "full_side_effect_proxy": 0.661,
                "tracklet_pair_f1": 0.77,
                "tracklet_pair_precision": 0.82,
                "tracklet_pair_recall": 0.73,
            },
            None,
        )
        assert abs(zero_placeholder["predicted_full_idf1"] - 0.661) < 1.0e-9, zero_placeholder
        clamped_model = {
            "columns": ["pair_f1"],
            "coef": [10.0],
            "mean": [0.0],
            "scale": [1.0],
            "intercept": 0.0,
            "target_min": 0.654,
            "target_max": 0.656,
            "target_clamp_margin": 0.001,
        }
        clamped = _normalize_row(
            {
                "pair_f1": 0.77,
                "pair_precision": 0.82,
                "pair_recall": 0.73,
            },
            clamped_model,
        )
        assert abs(clamped["predicted_full_idf1"] - 0.657) < 1.0e-9, clamped
        opponent_risky = _normalize_row(
            {
                "learned_proxy_full_idf1": 0.662,
                "tracklet_pair_f1": 0.77,
                "tracklet_pair_precision": 0.82,
                "tracklet_pair_recall": 0.73,
                "accepted_preview": [
                    {
                        "combined_opponent_risk_score": 0.82,
                        "temporal_opponent_risk_score": 0.79,
                        "max_same_video_overlap_frames": 247,
                        "view_weak_margin_fraction": 0.5,
                    }
                ],
            },
            None,
            side_effect_risk_weight=0.002,
        )
        assert "temporal_same_video_overlap" in opponent_risky["side_effect_reasons"], opponent_risky
        assert "high_combined_opponent_risk" in opponent_risky["side_effect_reasons"], opponent_risky
        assert "weak_multiview_margin" in opponent_risky["side_effect_reasons"], opponent_risky
        endpoint_risky = _normalize_row(
            {
                "mode": "target_endpoint_singleton",
                "learned_proxy_full_idf1": 0.662,
                "tracklet_pair_f1": 0.770956,
                "tracklet_pair_precision": 0.817612,
                "tracklet_pair_recall": 0.729336,
                "accepted_preview": [
                    {
                        "best_endpoint_score": 0.854766,
                        "source_seq": 1604,
                        "target_seq": 1549,
                        "support_edges": 1,
                        "target_size": 149,
                    }
                ],
            },
            None,
            side_effect_risk_weight=0.002,
        )
        assert "endpoint_direct_action" in endpoint_risky["side_effect_reasons"], endpoint_risky
        assert "endpoint_large_target_component" in endpoint_risky["side_effect_reasons"], endpoint_risky
        assert "endpoint_single_support_edge" in endpoint_risky["side_effect_reasons"], endpoint_risky
        diverse_ns = argparse.Namespace(top_n=3, max_per_mode=1)
        diverse = _select(
            [
                {"scheduler_score": 3.0, "scheduler_family": "a", "mode": "m1"},
                {"scheduler_score": 2.0, "scheduler_family": "b", "mode": "m1"},
                {"scheduler_score": 1.0, "scheduler_family": "c", "mode": "m2"},
            ],
            diverse_ns,
        )
        assert [row["scheduler_family"] for row in diverse] == ["a", "c"], diverse
    print(json.dumps({"stage": "self_test", "status": "ok"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidate", action="append", default=[])
    ap.add_argument(
        "--fullscore-label",
        action="append",
        default=[],
        help="Optional post-hoc full-score label JSON/CSV. Rows are matched by scheduler family and used only to reject known bad families.",
    )
    ap.add_argument("--proxy-model-json", default="local_runs/no_anchor_full_proxy_delivery_ridge_model_20260620.json")
    ap.add_argument("--current-best-full-idf1", type=float, default=0.655240)
    ap.add_argument("--min-pair-f1", type=float, default=0.70)
    ap.add_argument("--min-pair-precision", type=float, default=0.70)
    ap.add_argument("--min-pair-recall", type=float, default=0.70)
    ap.add_argument("--min-delivery-tracklets", type=float, default=7000.0)
    ap.add_argument("--min-predicted-full", type=float, default=0.6530)
    ap.add_argument(
        "--min-scheduler-score",
        type=float,
        default=None,
        help="Optional risk-adjusted scheduler-score floor. Use when side-effect risk should gate, not only rank.",
    )
    ap.add_argument("--allow-predicted-below-current", action="store_true")
    ap.add_argument(
        "--side-effect-risk-weight",
        type=float,
        default=0.0,
        help="Subtract this weight times a no-GT side-effect risk from scheduler score.",
    )
    ap.add_argument("--include-full-scored", action="store_true")
    ap.add_argument(
        "--max-per-mode",
        type=int,
        default=0,
        help="Optional structural diversity cap per mode. 0 keeps the historical uncapped behavior.",
    )
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    args.require_predicted_above_current = not bool(args.allow_predicted_below_current)

    if args.self_test:
        _self_test()
        return
    if not args.candidate:
        raise SystemExit("--candidate is required unless --self-test is used")

    model = None
    if args.proxy_model_json and Path(args.proxy_model_json).is_file():
        model = json.loads(Path(args.proxy_model_json).read_text())
    raw = _load_rows(args.candidate)
    fullscore_labels = _load_fullscore_label_rows(args.fullscore_label) if args.fullscore_label else {}
    normalized = [
        _normalize_row(row, model, fullscore_labels, side_effect_risk_weight=float(args.side_effect_risk_weight))
        for row in raw
    ]
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in normalized:
        ok, reasons = _eligible(row, args)
        row["scheduler_exclusion_reasons"] = reasons
        if ok:
            eligible.append(row)
        else:
            rejected.append(row)
    selected = _select(eligible, args)
    result = {
        "input_candidates": args.candidate,
        "fullscore_labels": args.fullscore_label,
        "fullscore_label_families": len(fullscore_labels),
        "raw_count": len(raw),
        "eligible_count": len(eligible),
        "rejected_count": len(rejected),
        "current_best_full_idf1": float(args.current_best_full_idf1),
        "min_predicted_full": float(args.min_predicted_full),
        "min_scheduler_score": args.min_scheduler_score,
        "top_n": int(args.top_n),
        "side_effect_risk_weight": float(args.side_effect_risk_weight),
        "selected": selected,
        "top_rejected": sorted(rejected, key=lambda row: float(row["scheduler_score"]), reverse=True)[:20],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
        "note": "This is an offline no-anchor scheduler. It does not score full IDF1; selected rows must be submitted to DS1 full scoring.",
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, selected)
    if args.md:
        lines = [
            "# No-Anchor Full-Score Scheduler",
            "",
            f"- raw candidates: `{len(raw)}`",
            f"- eligible: `{len(eligible)}`",
            f"- rejected: `{len(rejected)}`",
            f"- current best full IDF1: `{args.current_best_full_idf1:.6f}`",
            f"- min predicted full: `{args.min_predicted_full:.6f}`",
            "",
            "## Selected Candidates",
            "",
            "| rank | scheduler | predicted full | pair F1 | P | R | family | artifact |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for rank, row in enumerate(selected, start=1):
            lines.append(
                f"| {rank} | `{row['scheduler_score']:.6f}` | `{row['predicted_full_idf1']:.6f}` | "
                f"`{row['pair_f1_norm']:.6f}` | `{row['pair_precision_norm']:.6f}` | `{row['pair_recall_norm']:.6f}` | "
                f"`{row['scheduler_family']}` | `{row.get('artifact') or row.get('_source_file')}` |"
            )
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n")
    print(json.dumps({"raw": len(raw), "eligible": len(eligible), "selected": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
