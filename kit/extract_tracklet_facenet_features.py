#!/usr/bin/env python
"""Extract no-anchor FaceNet/MT-CNN face embeddings for VLINCS tracklets.

The extractor samples person boxes from each tracklet, searches for a face in
the upper person crop, and averages VGGFace2 FaceNet embeddings.  It reads no
identity labels or anchors.  Tracklets without a detected face are emitted with
a zero vector so downstream pair-feature alignment remains total.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from kit.extract_tracklet_foundation_features import _connect, _crop, _sample_rows, _video_paths
except ModuleNotFoundError:
    from extract_tracklet_foundation_features import _connect, _crop, _sample_rows, _video_paths


def _l2n(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x / (float(np.linalg.norm(x)) + 1.0e-9)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _all_tracklet_seqs(con, *, max_tracklets: int, num_shards: int, shard_index: int, tracklet_order: str) -> list[int]:
    max_tracklets = max(int(max_tracklets), 0)
    num_shards = max(int(num_shards), 1)
    shard_index = min(max(int(shard_index), 0), num_shards - 1)
    if tracklet_order == "n_dets_desc":
        order_sql = "ORDER BY COALESCE(n_dets, 0) DESC, seq"
    elif tracklet_order == "seq":
        order_sql = "ORDER BY seq"
    else:
        raise ValueError(f"unknown tracklet_order={tracklet_order!r}")
    limit_sql = f"LIMIT {max_tracklets}" if max_tracklets > 0 else ""
    with con.cursor() as cur:
        cur.execute(
            f"""SELECT seq
                FROM tracklets
                WHERE (seq % {num_shards}) = {shard_index}
                {order_sql}
                {limit_sql}"""
        )
        return [int(row[0]) for row in cur.fetchall()]


def _upper_region(image: Image.Image, frac: float) -> Image.Image:
    w, h = image.size
    cut = max(1, min(h, int(round(h * float(frac)))))
    return image.crop((0, 0, w, cut))


def _letterbox_square(image: Image.Image, size: int) -> Image.Image:
    size = max(int(size), 32)
    image = image.convert("RGB")
    w, h = image.size
    scale = min(float(size) / max(float(w), 1.0), float(size) / max(float(h), 1.0))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--video-root", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--crop-margin", type=float, default=0.08)
    ap.add_argument("--face-region", default="upper", choices=["upper", "full"])
    ap.add_argument("--upper-frac", type=float, default=0.58)
    ap.add_argument("--min-face-prob", type=float, default=0.90)
    ap.add_argument("--detect-square-size", type=int, default=320)
    ap.add_argument("--detect-batch-size", type=int, default=64)
    ap.add_argument("--embed-batch-size", type=int, default=128)
    ap.add_argument("--progress-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-tracklets", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--tracklet-order", default="seq", choices=["seq", "n_dets_desc"])
    ap.add_argument("--sequential-read-max-gap", type=int, default=300)
    args = ap.parse_args()

    from facenet_pytorch import InceptionResnetV1, MTCNN
    from facenet_pytorch.models.mtcnn import fixed_image_standardization
    from facenet_pytorch.models.utils.detect_face import extract_face

    con = _connect(args.dbname)
    all_seqs = _all_tracklet_seqs(
        con,
        max_tracklets=int(args.max_tracklets),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        tracklet_order=str(args.tracklet_order),
    )
    rows = _sample_rows(
        con,
        int(args.samples),
        max_tracklets=int(args.max_tracklets),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        tracklet_order=str(args.tracklet_order),
    )
    video_paths = _video_paths(args.dataset, list(args.video_root))
    device = str(args.device)
    mtcnn = MTCNN(image_size=160, margin=10, keep_all=False, post_process=True, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    import cv2  # type: ignore

    caps = {}
    cap_frame: dict[str, int] = {}
    cap_cached: dict[str, np.ndarray] = {}
    seq_feats: dict[int, list[np.ndarray]] = defaultdict(list)
    seq_probs: dict[int, list[float]] = defaultdict(list)
    pending_images: list[Image.Image] = []
    pending_detect_seqs: list[int] = []
    pending_faces: list[torch.Tensor] = []
    pending_seqs: list[int] = []
    missing_video = 0
    missing_crop = 0
    face_miss = 0
    face_low_prob = 0
    seen_crops = 0
    detected_faces = 0

    def flush() -> None:
        if not pending_faces:
            return
        batch = torch.stack(pending_faces).to(device)
        with torch.inference_mode():
            feats = torch.nn.functional.normalize(model(batch).float(), dim=1)
        arr = feats.detach().cpu().numpy().astype(np.float32)
        for seq, feat in zip(pending_seqs, arr):
            seq_feats[int(seq)].append(feat)
        pending_faces.clear()
        pending_seqs.clear()

    def flush_detect() -> None:
        nonlocal face_miss, face_low_prob, detected_faces
        if not pending_images:
            return
        batch_boxes, batch_probs = mtcnn.detect(pending_images)
        for image, seq, boxes, probs in zip(pending_images, pending_detect_seqs, batch_boxes, batch_probs):
            if boxes is None or probs is None or len(boxes) == 0:
                face_miss += 1
                continue
            best = int(np.argmax(np.asarray(probs, dtype=np.float32)))
            prob_f = float(probs[best])
            if prob_f < float(args.min_face_prob):
                face_low_prob += 1
                continue
            face = extract_face(
                image,
                boxes[best],
                image_size=int(mtcnn.image_size),
                margin=int(mtcnn.margin),
            )
            if bool(mtcnn.post_process):
                face = fixed_image_standardization(face)
            pending_faces.append(face.detach().cpu())
            pending_seqs.append(int(seq))
            seq_probs[int(seq)].append(prob_f)
            detected_faces += 1
            if len(pending_faces) >= int(args.embed_batch_size):
                flush()
        pending_images.clear()
        pending_detect_seqs.clear()

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
            image = _crop(frame, (x1, y1, x2, y2), float(args.crop_margin))
            if image is None:
                missing_crop += 1
                continue
            if args.face_region == "upper":
                image = _upper_region(image, float(args.upper_frac))
            seen_crops += 1
            pending_images.append(_letterbox_square(image, int(args.detect_square_size)))
            pending_detect_seqs.append(int(seq))
            if len(pending_images) >= int(args.detect_batch_size):
                flush_detect()
            progress_every = max(int(args.progress_every), 0)
            if progress_every and seen_crops and seen_crops % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "seen_crops": int(seen_crops),
                            "detected_faces": int(detected_faces),
                            "seqs_with_faces": int(len(seq_feats) + len(set(pending_seqs))),
                            "face_miss": int(face_miss),
                            "face_low_prob": int(face_low_prob),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        flush_detect()
        flush()
    finally:
        for cap in caps.values():
            cap.release()

    seqs = np.asarray(all_seqs, dtype=np.int64)
    features = np.zeros((len(seqs), 512), dtype=np.float32)
    valid = np.zeros((len(seqs),), dtype=bool)
    sample_counts = np.zeros((len(seqs),), dtype=np.int16)
    mean_probs = np.zeros((len(seqs),), dtype=np.float32)
    seq_to_pos = {int(seq): idx for idx, seq in enumerate(seqs.tolist())}
    for seq, feats in seq_feats.items():
        pos = seq_to_pos.get(int(seq))
        if pos is None or not feats:
            continue
        arr = np.stack(feats).astype(np.float32)
        feat = _l2n(arr.mean(axis=0).astype(np.float32))
        features[pos] = feat.astype(np.float32)
        valid[pos] = True
        sample_counts[pos] = int(len(feats))
        mean_probs[pos] = float(np.mean(seq_probs.get(int(seq), [0.0])))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": "facenet-pytorch InceptionResnetV1 pretrained=vggface2",
        "detector": "facenet-pytorch MTCNN",
        "samples": int(args.samples),
        "crop_margin": float(args.crop_margin),
        "face_region": str(args.face_region),
        "upper_frac": float(args.upper_frac),
        "min_face_prob": float(args.min_face_prob),
        "detect_batch_size": int(args.detect_batch_size),
        "detect_square_size": int(args.detect_square_size),
        "tracklets": int(len(seqs)),
        "tracklets_with_face": int(valid.sum()),
        "requested_rows": int(len(rows)),
        "seen_crops": int(seen_crops),
        "detected_faces": int(detected_faces),
        "missing_video": int(missing_video),
        "missing_crop": int(missing_crop),
        "face_miss": int(face_miss),
        "face_low_prob": int(face_low_prob),
        "uses_anchors": False,
        "uses_gt": False,
    }
    np.savez_compressed(
        out,
        seqs=seqs,
        features=features.astype(np.float32),
        valid=valid,
        sample_counts=sample_counts,
        mean_face_prob=mean_probs,
        meta=json.dumps(meta, sort_keys=True),
    )
    print(json.dumps({"out": str(out), "shape": list(features.shape), **meta}, sort_keys=True))


if __name__ == "__main__":
    main()
