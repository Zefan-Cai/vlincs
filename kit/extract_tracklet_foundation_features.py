#!/usr/bin/env python
"""Extract no-anchor foundation-model crop features for gallery tracklets.

The script reads tracklet boxes from the existing gallery DB, samples a few
frames per tracklet, crops the person box from source videos, and averages a
pretrained image-model feature. It never reads identity labels or anchors.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg
import torch
from PIL import Image

from vlincs_gallery.paths import CARDDIRS


def _connect(dbname: str):
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "55433"),
        user=os.environ.get("PGUSER", "gallery"),
        password=os.environ.get("PGPASSWORD", "gallery"),
        dbname=dbname,
    )


def _video_paths(dataset: str, extra_roots: list[str] | None = None) -> dict[str, str]:
    out = {}
    for root in CARDDIRS.get(dataset, []):
        for path in Path(root).glob("*.mp4"):
            out[path.stem] = str(path)
    for root in extra_roots or []:
        root_path = Path(root)
        candidates = []
        if root_path.is_file() and root_path.suffix.lower() == ".mp4":
            candidates = [root_path]
        elif root_path.is_dir():
            candidates = list(root_path.glob("*.mp4")) + list(root_path.glob("videos/*.mp4"))
        for path in candidates:
            out[path.stem] = str(path)
    return out


def _sample_rows(
    con,
    samples: int,
    *,
    max_tracklets: int = 0,
    num_shards: int = 1,
    shard_index: int = 0,
    tracklet_order: str = "seq",
) -> list[tuple]:
    samples = max(int(samples), 1)
    num_shards = max(int(num_shards), 1)
    shard_index = min(max(int(shard_index), 0), num_shards - 1)
    max_tracklets = max(int(max_tracklets), 0)
    if samples == 1:
        pick = "rn = GREATEST(1, (cnt + 1) / 2)"
    else:
        picks = []
        for pos in range(samples):
            frac = float(pos) / float(max(samples - 1, 1))
            picks.append(f"rn = GREATEST(1, LEAST(cnt, 1 + ROUND((cnt - 1) * {frac:.10f})::int))")
        pick = " OR ".join(picks)
    if tracklet_order == "n_dets_desc":
        order_sql = "ORDER BY COALESCE(t.n_dets, 0) DESC, t.seq"
    elif tracklet_order == "seq":
        order_sql = "ORDER BY t.seq"
    else:
        raise ValueError(f"unknown tracklet_order={tracklet_order!r}")
    limit_sql = f"LIMIT {max_tracklets}" if max_tracklets > 0 else ""
    with con.cursor() as cur:
        cur.execute(
            f"""WITH selected AS (
                    SELECT t.seq
                    FROM tracklets t
                    WHERE (t.seq % {num_shards}) = {shard_index}
                    {order_sql}
                    {limit_sql}
                ),
                ranked AS (
                    SELECT a.seq, d.video, d.frame_idx, d.x1, d.y1, d.x2, d.y2,
                           ROW_NUMBER() OVER (PARTITION BY a.seq ORDER BY d.frame_idx) AS rn,
                           COUNT(*) OVER (PARTITION BY a.seq) AS cnt
                    FROM selected s
                    JOIN assignments a ON a.seq = s.seq
                    JOIN detections d ON d.det_id = a.det_id
                )
                SELECT seq, video, frame_idx, x1, y1, x2, y2
                FROM ranked
                WHERE {pick}
                ORDER BY video, frame_idx, seq"""
        )
        return cur.fetchall()


def _crop(frame: np.ndarray, box: tuple[float, float, float, float], margin: float) -> Image.Image | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= margin * bw
    x2 += margin * bw
    y1 -= margin * bh
    y2 += margin * bh
    xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    xi2, yi2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    rgb = frame[yi1:yi2, xi1:xi2, ::-1]
    if rgb.size == 0:
        return None
    return Image.fromarray(rgb)


def _load_model(
    model_name: str,
    device: str,
    *,
    trust_remote_code: bool,
    revision: str | None,
    processor_model: str | None = None,
):
    if model_name == "color-hist":
        return None, None
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPVisionModel

    load_kwargs = {"revision": revision} if revision else {}
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    processor_name = processor_model or model_name
    try:
        processor = AutoImageProcessor.from_pretrained(processor_name, **load_kwargs)
    except OSError:
        processor = AutoProcessor.from_pretrained(processor_name, **load_kwargs)
    if "clip" in model_name.lower() and not trust_remote_code:
        model = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    else:
        model = AutoModel.from_pretrained(model_name, use_safetensors=True, **load_kwargs).to(device).eval()
    return processor, model


def _color_hist_features(images: list[Image.Image]) -> np.ndarray:
    import cv2  # type: ignore

    feats = []
    for image in images:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        parts = [rgb[: max(1, rgb.shape[0] // 2)], rgb[max(1, rgb.shape[0] // 2) :]]
        frame_feats = []
        for part in parts:
            if part.size == 0:
                part = rgb
            hsv = cv2.cvtColor(part, cv2.COLOR_RGB2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).reshape(-1)
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).reshape(-1)
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).reshape(-1)
            hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
            hist /= float(hist.sum() + 1e-6)
            flat = part.reshape(-1, 3).astype(np.float32) / 255.0
            color = np.concatenate([flat.mean(axis=0), flat.std(axis=0), np.median(flat, axis=0)])
            frame_feats.append(np.concatenate([hist, color.astype(np.float32)]))
        feat = np.concatenate(frame_feats).astype(np.float32)
        feat /= float(np.linalg.norm(feat) + 1e-9)
        feats.append(feat)
    return np.stack(feats).astype(np.float32)


def _feature_tensor(out):
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "embeddings", "pooler_output"):
        value = getattr(out, attr, None)
        if value is not None:
            return value
    if isinstance(out, (tuple, list)) and len(out):
        return out[0]
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0]
    raise TypeError(f"cannot extract feature tensor from model output type {type(out).__name__}")


def _features(processor, model, images: list[Image.Image], device: str) -> np.ndarray:
    if model is None:
        return _color_hist_features(images)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            feat = _feature_tensor(model.get_image_features(**inputs))
            feat = torch.nn.functional.normalize(feat.float(), dim=1)
            return feat.detach().cpu().numpy().astype(np.float32)
        try:
            out = model(**inputs, return_dict=True)
        except TypeError:
            out = model(**inputs)
    feat = _feature_tensor(out)
    feat = torch.nn.functional.normalize(feat.float(), dim=1)
    return feat.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--model-revision", default=None, help="optional pinned Hugging Face revision")
    ap.add_argument("--processor-model", default=None, help="optional HF processor repo when the model repo has no preprocessor_config.json")
    ap.add_argument("--trust-remote-code", action="store_true", help="allow custom HF model code after inspection")
    ap.add_argument("--video-root", action="append", default=[], help="extra directory or mp4 path for source videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--crop-margin", type=float, default=0.08)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-tracklets", type=int, default=0, help="optional smoke-test limit")
    ap.add_argument("--num-shards", type=int, default=1, help="split tracklets into deterministic seq modulo shards")
    ap.add_argument("--shard-index", type=int, default=0, help="zero-based shard index for --num-shards")
    ap.add_argument(
        "--tracklet-order",
        default="seq",
        choices=["seq", "n_dets_desc"],
        help="order selected tracklets before --max-tracklets is applied",
    )
    ap.add_argument(
        "--sequential-read-max-gap",
        type=int,
        default=300,
        help="when rows are sorted by frame, read forward instead of seeking if the frame gap is at most this value",
    )
    args = ap.parse_args()

    con = _connect(args.dbname)
    rows = _sample_rows(
        con,
        args.samples,
        max_tracklets=int(args.max_tracklets),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        tracklet_order=str(args.tracklet_order),
    )
    video_paths = _video_paths(args.dataset, list(args.video_root))
    processor, model = _load_model(
        args.model,
        args.device,
        trust_remote_code=bool(args.trust_remote_code),
        revision=args.model_revision,
        processor_model=args.processor_model,
    )

    import cv2  # type: ignore

    caps = {}
    cap_frame: dict[str, int] = {}
    cap_cached: dict[str, np.ndarray] = {}
    seq_feats: dict[int, list[np.ndarray]] = defaultdict(list)
    pending_images: list[Image.Image] = []
    pending_seqs: list[int] = []
    missing_video = 0
    missing_crop = 0
    seen_crops = 0

    def flush() -> None:
        if not pending_images:
            return
        feats = _features(processor, model, pending_images, args.device)
        for seq, feat in zip(pending_seqs, feats):
            seq_feats[int(seq)].append(feat)
        pending_images.clear()
        pending_seqs.clear()

    def read_frame(path: str, frame_idx: int):
        cap = caps.get(path)
        if cap is None:
            cap = cv2.VideoCapture(path)
            caps[path] = cap
            cap_frame[path] = -1
        target = int(frame_idx)
        current = int(cap_frame.get(path, -1))
        if current == target and path in cap_cached:
            return True, cap_cached[path]
        if current >= 0 and target > current and target - current <= int(args.sequential_read_max_gap):
            ok = False
            frame = None
            for next_frame in range(current + 1, target + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap_frame[path] = next_frame - 1
                    return False, None
            cap_frame[path] = target
            cap_cached[path] = frame
            return True, frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok and frame is not None:
            cap_frame[path] = target
            cap_cached[path] = frame
        return ok, frame

    try:
        for seq, video, frame_idx, x1, y1, x2, y2 in rows:
            path = video_paths.get(str(video))
            if not path:
                missing_video += 1
                continue
            ok, frame = read_frame(path, int(frame_idx))
            if not ok or frame is None:
                missing_crop += 1
                continue
            image = _crop(frame, (x1, y1, x2, y2), args.crop_margin)
            if image is None:
                missing_crop += 1
                continue
            pending_images.append(image)
            pending_seqs.append(int(seq))
            seen_crops += 1
            if seen_crops % 500 == 0:
                print(json.dumps({"processed_crops": seen_crops, "seqs_with_features": len(seq_feats)}), flush=True)
            if len(pending_images) >= int(args.batch_size):
                flush()
        flush()
    finally:
        for cap in caps.values():
            cap.release()

    seqs = np.asarray(sorted(seq_feats), dtype=np.int64)
    features = []
    sample_counts = []
    for seq in seqs:
        arr = np.stack(seq_feats[int(seq)]).astype(np.float32)
        feat = arr.mean(axis=0)
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        features.append(feat.astype(np.float32))
        sample_counts.append(len(arr))
    features_np = np.stack(features).astype(np.float32) if len(features) else np.zeros((0, 0), np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        seqs=seqs,
        features=features_np,
        sample_counts=np.asarray(sample_counts, dtype=np.int16),
        model=np.asarray([args.model], dtype=object),
    )
    info = {
        "out": str(out),
        "model": args.model,
        "model_revision": args.model_revision,
        "processor_model": args.processor_model,
        "device": args.device,
        "tracklets_with_features": int(len(seqs)),
        "feature_dim": int(features_np.shape[1]) if features_np.ndim == 2 and len(features_np) else 0,
        "sample_rows": int(len(rows)),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "tracklet_order": str(args.tracklet_order),
        "sequential_read_max_gap": int(args.sequential_read_max_gap),
        "missing_video_rows": int(missing_video),
        "missing_crop_rows": int(missing_crop),
        "uses_anchors": False,
        "uses_gt": False,
    }
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
