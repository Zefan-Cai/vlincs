#!/usr/bin/env python
"""Audit generated positives with no-anchor tracklet-local CTF gates.

This is a lightweight reviewer/opponent stage. It checks whether each generated
image still resembles its source tracklet more than the nearest local counter
tracklet. It does not promote any identity edit and it does not read GT labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _resolve(path: str, base: Path) -> Path:
    item = Path(path)
    if item.is_absolute():
        return item
    return base / item


def _hist_feature(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((128, 256), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    parts = [rgb[:128], rgb[128:]]
    feats = []
    for part in parts:
        flat = part.reshape(-1, 3)
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(flat[:, channel], bins=24, range=(0.0, 1.0), density=False)
            hist = hist.astype(np.float32)
            hist /= float(hist.sum() + 1.0e-9)
            hist_parts.append(hist)
        stats = np.concatenate([flat.mean(axis=0), flat.std(axis=0), np.median(flat, axis=0)]).astype(np.float32)
        feats.append(np.concatenate([*hist_parts, stats]))
    feat = np.concatenate(feats).astype(np.float32)
    feat /= float(np.linalg.norm(feat) + 1.0e-9)
    return feat


def _transformers_image_feature_extractor(model_id: str, backend: str):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ModuleNotFoundError as exc:
        raise SystemExit(f"{backend} backend requires torch and transformers in the active environment") from exc

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    def pooled_tensor(output):
        if hasattr(output, "float"):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0, :]
        raise TypeError(f"{backend} model returned unsupported output type: {type(output).__name__}")

    def extract(path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                output = model.get_image_features(**inputs)
            else:
                model_output = model(**inputs)
                output = pooled_tensor(model_output)
            output = pooled_tensor(output)
        feat = output.float().numpy()[0].astype(np.float32)
        feat /= float(np.linalg.norm(feat) + 1.0e-9)
        return feat

    return extract


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1.0e-9))


def _source_features(prompt_manifest: dict[str, Any], repo_root: Path, feature_fn) -> tuple[dict[int, list[np.ndarray]], dict[str, np.ndarray]]:
    by_seq: dict[int, list[np.ndarray]] = {}
    by_path: dict[str, np.ndarray] = {}
    for row in prompt_manifest.get("crops", []):
        if not isinstance(row, dict):
            continue
        path = _resolve(str(row["crop_path"]), repo_root)
        if path.is_file():
            feat = feature_fn(path)
            by_seq.setdefault(int(row["seq"]), []).append(feat)
            by_path[str(path)] = feat
    return by_seq, by_path


def _centroids(by_seq: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    out = {}
    for seq, feats in by_seq.items():
        mean = np.stack(feats).mean(axis=0)
        mean /= float(np.linalg.norm(mean) + 1.0e-9)
        out[int(seq)] = mean
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-manifest", required=True, type=Path)
    ap.add_argument("--generated-manifest", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--feature-backend", choices=["hist", "dinov2", "siglip"], default="hist")
    ap.add_argument("--reference-mode", choices=["centroid", "source"], default="centroid")
    ap.add_argument("--dinov2-model-id", default="facebook/dinov2-small")
    ap.add_argument("--siglip-model-id", default="google/siglip-base-patch16-224")
    ap.add_argument("--min-same-sim", type=float, default=0.86)
    ap.add_argument("--min-margin", type=float, default=0.025)
    ap.add_argument("--repo-root", default="", help="root used to resolve relative source/generated paths; defaults to current directory")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    prompt = _load(args.prompt_manifest)
    generated = _load(args.generated_manifest)
    if not bool(prompt.get("no_anchor")) or bool(generated.get("uses_anchors")):
        raise SystemExit("expected no-anchor prompt and generated manifests")
    if args.feature_backend == "hist":
        feature_fn = _hist_feature
        model_id = ""
    elif args.feature_backend == "dinov2":
        model_id = str(args.dinov2_model_id)
        feature_fn = _transformers_image_feature_extractor(model_id, "dinov2")
    else:
        model_id = str(args.siglip_model_id)
        feature_fn = _transformers_image_feature_extractor(model_id, "siglip")
    source, source_by_path = _source_features(prompt, repo_root, feature_fn)
    centroids = _centroids(source)
    if len(centroids) < 2:
        raise SystemExit("need at least two local tracklet sequences for counter-tracklet CTF")

    rows = []
    pass_count = 0
    for item in generated.get("generated_images", []):
        if not isinstance(item, dict):
            continue
        seq = int(item["seq"])
        image_path = _resolve(str(item["image_path"]), repo_root)
        feat = feature_fn(image_path)
        source_paths = []
        if item.get("source_crop_path"):
            source_paths.append(_resolve(str(item["source_crop_path"]), repo_root))
        for raw in item.get("reference_crop_paths", []) if isinstance(item.get("reference_crop_paths"), list) else []:
            source_paths.append(_resolve(str(raw), repo_root))
        source_feats = [source_by_path[str(path)] for path in source_paths if str(path) in source_by_path]

        if args.reference_mode == "source":
            if not source_feats:
                raise SystemExit(f"reference-mode=source needs source_crop_path or reference_crop_paths for {image_path}")
            same_scores = [_cos(feat, src_feat) for src_feat in source_feats]
            same = float(max(same_scores))
            source_similarity = same
        else:
            same = _cos(feat, centroids[seq])
            source_similarity = None

        others = {other_seq: _cos(feat, centroid) for other_seq, centroid in centroids.items() if other_seq != seq}
        best_other_seq, best_other = max(others.items(), key=lambda kv: kv[1])
        margin = same - best_other
        ctf_pass = same >= float(args.min_same_sim) and margin >= float(args.min_margin)
        pass_count += int(ctf_pass)
        rows.append(
            {
                "seq": seq,
                "variant": int(item.get("variant", -1)),
                "backend": item.get("backend", ""),
                "image_path": str(image_path),
                "feature_backend": str(args.feature_backend),
                "reference_mode": str(args.reference_mode),
                "same_seq_similarity": round(float(same), 6),
                "source_similarity": None if source_similarity is None else round(float(source_similarity), 6),
                "best_other_seq": int(best_other_seq),
                "best_other_similarity": round(float(best_other), 6),
                "margin": round(float(margin), 6),
                "ctf_pass": bool(ctf_pass),
                "decision": "keep_for_feature_audit_only" if ctf_pass else "reject_identity_drift_or_low_margin",
            }
        )

    summary = {
        "task": "no_anchor_generated_positive_ctf_audit",
        "source_prompt_manifest": str(args.prompt_manifest),
        "generated_manifest": str(args.generated_manifest),
        "backend": generated.get("backend", ""),
        "feature_backend": str(args.feature_backend),
        "reference_mode": str(args.reference_mode),
        "model_id": model_id,
        "min_same_sim": float(args.min_same_sim),
        "min_margin": float(args.min_margin),
        "generated_images": int(len(rows)),
        "ctf_pass": int(pass_count),
        "ctf_reject": int(len(rows) - pass_count),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "promotion_status": "not_promoted_ctf_audit_only",
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["generated_images", "ctf_pass", "ctf_reject", "backend"]}, sort_keys=True))


if __name__ == "__main__":
    main()
