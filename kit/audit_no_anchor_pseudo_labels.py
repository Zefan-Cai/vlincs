#!/usr/bin/env python
"""Audit no-anchor pair pseudo-labels against held-out GT labels.

This is a diagnostic script.  It reproduces the pseudo positive/negative pair
construction used by ``no_anchor_global_id_model.py`` and then uses GT only for
post-hoc purity analysis.  It does not train a model and does not feed GT back
into any resolver.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_global_id_model import (
        PairModelConfig,
        _candidate_pairs,
        _cfg_theta_grid,
        _load_online_gids,
        _load_pair_feature_views,
        _pair_view_features,
        _sample_random_negatives,
    )
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _build_overlap_forbidden,
        _connect,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _parse_float_list,
        _time_agglom_resolve,
    )
except ModuleNotFoundError:
    from no_anchor_global_id_model import (
        PairModelConfig,
        _candidate_pairs,
        _cfg_theta_grid,
        _load_online_gids,
        _load_pair_feature_views,
        _pair_view_features,
        _sample_random_negatives,
    )
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _build_overlap_forbidden,
        _connect,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _parse_float_list,
        _time_agglom_resolve,
    )


def _score_bucket(score: float) -> str:
    score = float(score)
    if score < 0.42:
        return "lt_042"
    if score < 0.64:
        return "042_064"
    if score < 0.78:
        return "064_078"
    return "gte_078"


def _pair_truth(seq_i: int, seq_j: int, gt_by_seq: dict[int, int], weight_by_seq: dict[int, float]) -> tuple[bool | None, float]:
    if seq_i not in gt_by_seq or seq_j not in gt_by_seq:
        return None, 0.0
    wi = float(weight_by_seq.get(seq_i, 0.0))
    wj = float(weight_by_seq.get(seq_j, 0.0))
    if wi <= 0.0 or wj <= 0.0:
        return None, 0.0
    return bool(int(gt_by_seq[seq_i]) == int(gt_by_seq[seq_j])), wi * wj


def _summarize(rows: list[dict[str, object]], *, label_name: str) -> dict[str, object]:
    total = len(rows)
    evaluated = [row for row in rows if row["truth_same"] is not None]
    true_same = sum(1 for row in evaluated if bool(row["truth_same"]))
    true_diff = sum(1 for row in evaluated if row["truth_same"] is False)
    w_total = float(sum(float(row["truth_weight"]) for row in evaluated))
    w_same = float(sum(float(row["truth_weight"]) for row in evaluated if bool(row["truth_same"])))
    w_diff = float(sum(float(row["truth_weight"]) for row in evaluated if row["truth_same"] is False))
    source_counts = Counter(str(row["source"]) for row in rows)
    vote_counts = Counter(str(row["votes"]) for row in rows)
    bucket_counts = Counter(str(row["score_bucket"]) for row in rows)
    if label_name == "positive":
        purity = true_same / max(len(evaluated), 1)
        weighted_purity = w_same / max(w_total, 1.0e-9)
    else:
        purity = true_diff / max(len(evaluated), 1)
        weighted_purity = w_diff / max(w_total, 1.0e-9)
    return {
        "rows": int(total),
        "evaluated_rows": int(len(evaluated)),
        "true_same_rows": int(true_same),
        "true_diff_rows": int(true_diff),
        "truth_weight_total": round(w_total, 3),
        "truth_weight_same": round(w_same, 3),
        "truth_weight_diff": round(w_diff, 3),
        "purity": round(float(purity), 6),
        "weighted_purity": round(float(weighted_purity), 6),
        "source_counts": dict(sorted(source_counts.items())),
        "vote_counts": dict(sorted(vote_counts.items(), key=lambda item: int(item[0]))),
        "score_bucket_counts": dict(sorted(bucket_counts.items())),
    }


def _source_truth_table(rows: list[dict[str, object]], *, label_name: str) -> list[dict[str, object]]:
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    out = []
    for source, source_rows in sorted(by_source.items()):
        summary = _summarize(source_rows, label_name=label_name)
        summary["source"] = source
        out.append(summary)
    return out


def _config_from_args(args: argparse.Namespace) -> PairModelConfig:
    return PairModelConfig(
        train_top_k=int(args.train_top_k),
        infer_top_k=int(args.infer_top_k),
        min_dets=int(args.min_dets),
        max_component_size=int(args.max_component_size),
        min_merge_support=int(args.min_merge_support),
        min_merge_support_ratio=float(args.min_merge_support_ratio),
        exclude_same=str(args.exclude_same),
        pseudo_theta=float(args.pseudo_theta),
        pseudo_top_k=int(args.pseudo_top_k),
        pseudo_temporal_bonus=float(args.pseudo_temporal_bonus),
        pseudo_time_window_ms=int(args.pseudo_time_window_ms),
        pseudo_ensemble=bool(args.pseudo_ensemble),
        pseudo_ensemble_thetas=str(args.pseudo_ensemble_thetas),
        pseudo_ensemble_min_votes=int(args.pseudo_ensemble_min_votes),
        pseudo_ensemble_max_neg_votes=int(args.pseudo_ensemble_max_neg_votes),
        pseudo_consensus_pos_min_votes=int(args.pseudo_consensus_pos_min_votes),
        pseudo_consensus_pos_min_sim=float(args.pseudo_consensus_pos_min_sim),
        pseudo_pos_min_sim=float(args.pseudo_pos_min_sim),
        pseudo_strong_pos_sim=float(args.pseudo_strong_pos_sim),
        pseudo_neg_max_sim=float(args.pseudo_neg_max_sim),
        candidate_time_bonus=float(args.candidate_time_bonus),
        affinity_time_bonus=float(args.affinity_time_bonus),
        attach_threshold=float(args.attach_threshold),
        attach_margin=float(args.attach_margin),
        attach_model_weight=float(args.attach_model_weight),
        attach_max_source_size=int(args.attach_max_source_size),
        attach_min_target_size=int(args.attach_min_target_size),
        attach_top_k=int(args.attach_top_k),
        attach_min_edge_support=int(args.attach_min_edge_support),
        attach_score_agg=str(args.attach_score_agg),
        attach_top_mean_k=int(args.attach_top_mean_k),
        max_neg_per_pos=float(args.max_neg_per_pos),
        random_negatives=int(args.random_negatives),
        model_type=str(args.model_type),
        random_state=int(args.random_state),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--pair-feature-npz", action="append", default=[])
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--train-top-k", type=int, default=30)
    ap.add_argument("--infer-top-k", type=int, default=30)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--max-component-size", type=int, default=120)
    ap.add_argument("--min-merge-support", type=int, default=1)
    ap.add_argument("--min-merge-support-ratio", type=float, default=0.0)
    ap.add_argument("--exclude-same", default="camera", choices=["camera", "stream", "video", "none"])
    ap.add_argument("--pseudo-theta", type=float, default=0.018)
    ap.add_argument("--pseudo-top-k", type=int, default=15)
    ap.add_argument("--pseudo-temporal-bonus", type=float, default=0.005)
    ap.add_argument("--pseudo-time-window-ms", type=int, default=1000)
    ap.add_argument("--pseudo-ensemble", action="store_true")
    ap.add_argument("--pseudo-ensemble-thetas", default="")
    ap.add_argument("--pseudo-ensemble-min-votes", type=int, default=2)
    ap.add_argument("--pseudo-ensemble-max-neg-votes", type=int, default=0)
    ap.add_argument("--pseudo-consensus-pos-min-votes", type=int, default=0)
    ap.add_argument("--pseudo-consensus-pos-min-sim", type=float, default=0.70)
    ap.add_argument("--pseudo-pos-min-sim", type=float, default=0.64)
    ap.add_argument("--pseudo-strong-pos-sim", type=float, default=0.78)
    ap.add_argument("--pseudo-neg-max-sim", type=float, default=0.42)
    ap.add_argument("--candidate-time-bonus", type=float, default=0.0)
    ap.add_argument("--affinity-time-bonus", type=float, default=0.005)
    ap.add_argument("--attach-threshold", type=float, default=0.72)
    ap.add_argument("--attach-margin", type=float, default=0.06)
    ap.add_argument("--attach-model-weight", type=float, default=0.65)
    ap.add_argument("--attach-max-source-size", type=int, default=2)
    ap.add_argument("--attach-min-target-size", type=int, default=2)
    ap.add_argument("--attach-top-k", type=int, default=60)
    ap.add_argument("--attach-min-edge-support", type=int, default=1)
    ap.add_argument("--attach-score-agg", default="max", choices=["max", "top_mean", "hybrid"])
    ap.add_argument("--attach-top-mean-k", type=int, default=3)
    ap.add_argument("--max-neg-per-pos", type=float, default=4.0)
    ap.add_argument("--random-negatives", type=int, default=60000)
    ap.add_argument("--model-type", default="hgb", choices=["hgb", "rf"])
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--sample-csv", default="")
    ap.add_argument("--sample-top-n", type=int, default=80)
    ap.add_argument("--json", required=True)
    ap.add_argument("--random-state", type=int, default=17)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, emb = _load_tracklets(con, args.role)
    if args.feature_npz:
        emb = _load_feature_npz(
            args.feature_npz,
            records,
            emb,
            concat_db=bool(args.concat_db_embedding),
            db_weight=float(args.db_weight),
            feature_weight=float(args.feature_weight),
        )
    pair_feature_views, pair_feature_names, pair_feature_meta = _load_pair_feature_views(list(args.pair_feature_npz), records)
    pred_by_video = _load_predictions(con)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": 0.5,
        "min_matches": 1,
        "min_purity": 0.0,
        "n_tracklets": len(records),
        "prediction_rows": int(sum(len(value) for value in pred_by_video.values())),
        "gt_rows": int(sum(len(value) for value in gt_by_video.values())),
    }
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached
    cfg = _config_from_args(args)
    rng = np.random.default_rng(int(cfg.random_state))
    online_gids = _load_online_gids(con)
    ensemble_thetas = _cfg_theta_grid(cfg) if bool(cfg.pseudo_ensemble) else [float(cfg.pseudo_theta)]

    pseudo_label_sets = []
    pseudo_size_sets = []
    pseudo_infos = []
    for theta in ensemble_thetas:
        pseudo_cfg = ResolveConfig(
            mode="time_agglom",
            theta=float(theta),
            top_k=int(cfg.pseudo_top_k),
            min_dets=int(cfg.min_dets),
            exclude_same=str(cfg.exclude_same),
            temporal_bonus=float(cfg.pseudo_temporal_bonus),
            time_window_ms=int(cfg.pseudo_time_window_ms),
        )
        labels, info = _time_agglom_resolve(records, emb, pseudo_cfg)
        pseudo_label_sets.append(labels)
        pseudo_size_sets.append(Counter(labels.tolist()))
        pseudo_infos.append({"theta": float(theta), **info})

    online_sizes = Counter(online_gids.get(record.seq, -1) for record in records)
    forbidden = _build_overlap_forbidden(records)
    pairs = _candidate_pairs(
        records,
        emb,
        max(int(cfg.train_top_k), int(cfg.pseudo_top_k)),
        cfg.exclude_same,
        time_bonus=float(cfg.candidate_time_bonus),
        time_window_ms=int(cfg.pseudo_time_window_ms),
    )
    positives: dict[tuple[int, int], tuple[float, int, str]] = {}
    negatives: dict[tuple[int, int], tuple[float, int, str]] = {}

    for (i, j), score in pairs.items():
        votes = 0
        for labels, sizes in zip(pseudo_label_sets, pseudo_size_sets):
            if labels[i] == labels[j] and sizes[int(labels[i])] > 1:
                votes += 1
        min_votes = int(cfg.pseudo_ensemble_min_votes) if bool(cfg.pseudo_ensemble) else 1
        max_neg_votes = int(cfg.pseudo_ensemble_max_neg_votes) if bool(cfg.pseudo_ensemble) else 0
        same_pseudo = votes >= min_votes
        gid_i = online_gids.get(records[i].seq)
        gid_j = online_gids.get(records[j].seq)
        same_online = gid_i is not None and gid_i == gid_j and online_sizes.get(gid_i, 0) > 1
        consensus_positive = (
            int(cfg.pseudo_consensus_pos_min_votes) > 0
            and votes >= int(cfg.pseudo_consensus_pos_min_votes)
            and score >= float(cfg.pseudo_consensus_pos_min_sim)
        )
        if j in forbidden[i]:
            negatives[(i, j)] = (float(score), int(votes), "cannot_link")
            continue
        source = None
        if same_pseudo and same_online and score >= float(cfg.pseudo_pos_min_sim):
            source = "pseudo_online_agree"
        elif score >= float(cfg.pseudo_strong_pos_sim) and same_pseudo:
            source = "strong_visual_pseudo"
        elif consensus_positive:
            source = "consensus_votes"
        if source:
            positives[(i, j)] = (float(score), int(votes), source)
        elif votes <= max_neg_votes and (not same_online) and score <= float(cfg.pseudo_neg_max_sim):
            negatives[(i, j)] = (float(score), int(votes), "low_score_no_vote")

    random_negs = _sample_random_negatives(
        records,
        emb,
        set(pairs) | set(positives) | set(negatives),
        forbidden,
        n_samples=int(cfg.random_negatives),
        exclude_same=cfg.exclude_same,
        rng=rng,
    )
    for key, score in random_negs.items():
        negatives[key] = (float(score), -1, "random_negative")

    max_neg = int(max(len(positives) * float(cfg.max_neg_per_pos), 1))
    if len(negatives) > max_neg:
        items = list(negatives.items())
        order = rng.permutation(len(items))[:max_neg]
        negatives = {items[int(k)][0]: items[int(k)][1] for k in order}

    def build_rows(items: dict[tuple[int, int], tuple[float, int, str]], pseudo_label: int) -> list[dict[str, object]]:
        rows = []
        for (i, j), (score, votes, source) in items.items():
            seq_i = int(records[i].seq)
            seq_j = int(records[j].seq)
            truth_same, truth_weight = _pair_truth(seq_i, seq_j, gt_by_seq, weight_by_seq)
            row: dict[str, object] = {
                "pseudo_label": int(pseudo_label),
                "source": source,
                "votes": int(votes),
                "score": round(float(score), 6),
                "score_bucket": _score_bucket(float(score)),
                "seq_i": seq_i,
                "seq_j": seq_j,
                "tracklet_i": records[i].tracklet_key,
                "tracklet_j": records[j].tracklet_key,
                "video_i": records[i].video,
                "video_j": records[j].video,
                "camera_i": records[i].camera,
                "camera_j": records[j].camera,
                "truth_same": truth_same,
                "truth_weight": round(float(truth_weight), 3),
            }
            extra = _pair_view_features(pair_feature_views, i, j)
            for name, value in zip(pair_feature_names, extra):
                row[name] = round(float(value), 6)
            rows.append(row)
        return rows

    positive_rows = build_rows(positives, 1)
    negative_rows = build_rows(negatives, 0)
    result = {
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "feature_npz": args.feature_npz,
        "pair_feature_views": pair_feature_meta,
        "pair_feature_names": pair_feature_names,
        "pair_model_config": asdict(cfg),
        "pseudo_resolver": {
            "mode": "time_agglom",
            "ensemble": bool(cfg.pseudo_ensemble),
            "ensemble_thetas": [float(theta) for theta in ensemble_thetas],
            "base_runs": pseudo_infos,
        },
        "eval_stats": eval_stats,
        "candidate_pairs": int(len(pairs)),
        "random_negative_pairs_before_downsample": int(len(random_negs)),
        "positive_summary": _summarize(positive_rows, label_name="positive"),
        "negative_summary": _summarize(negative_rows, label_name="negative"),
        "positive_by_source": _source_truth_table(positive_rows, label_name="positive"),
        "negative_by_source": _source_truth_table(negative_rows, label_name="negative"),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.sample_csv:
        false_pos = [row for row in positive_rows if row["truth_same"] is False]
        false_neg = [row for row in negative_rows if row["truth_same"] is True]
        true_pos = [row for row in positive_rows if row["truth_same"] is True]
        true_neg = [row for row in negative_rows if row["truth_same"] is False]
        samples = []
        for tag, rows in [
            ("positive_false", sorted(false_pos, key=lambda row: float(row["score"]), reverse=True)),
            ("positive_true", sorted(true_pos, key=lambda row: float(row["score"]), reverse=True)),
            ("negative_false", sorted(false_neg, key=lambda row: float(row["truth_weight"]), reverse=True)),
            ("negative_true", sorted(true_neg, key=lambda row: float(row["truth_weight"]), reverse=True)),
        ]:
            for row in rows[: int(args.sample_top_n)]:
                item = dict(row)
                item["audit_bucket"] = tag
                samples.append(item)
        Path(args.sample_csv).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in samples for key in row})
        with open(args.sample_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

    print(json.dumps({"json": str(out), "positive": result["positive_summary"], "negative": result["negative_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
