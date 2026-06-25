#!/usr/bin/env python
"""Build no-anchor real-frame prompt assets for generated-positive probes.

The output is only an evidence-input package: real frames, bbox crops, a contact
sheet, and a prompt manifest for `generate_no_anchor_tracklet_local_positives.py`.
It does not use anchors or GT labels and it does not change any assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _find_video(raw_video_root: Path, video_key: str) -> Path:
    for suffix in (".mp4", ".mov", ".mkv", ".avi", ".m4v"):
        direct = raw_video_root / f"{video_key}{suffix}"
        if direct.is_file():
            return direct
    hits: list[Path] = []
    for suffix in (".mp4", ".mov", ".mkv", ".avi", ".m4v"):
        hits.extend(raw_video_root.rglob(f"{video_key}*{suffix}"))
    if not hits:
        raise FileNotFoundError(f"no raw video found for {video_key} under {raw_video_root}")
    return sorted(hits)[0]


def _load_assignments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "seq",
        "tracklet_key",
        "video",
        "component_label",
        "predicted_global_id",
        "decision_status",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df["seq"] = df["seq"].astype(int)
    return df


def _load_tracklet_rows(parquets: list[Path], tracklet_key: str, frames_per_seq: int) -> list[dict[str, Any]]:
    video_key = tracklet_key.split(":")[0]
    candidates = [p for p in parquets if p.parent.name == video_key] or parquets
    for path in candidates:
        df = pd.read_parquet(path)
        if "tracklet_key" not in df.columns:
            continue
        sub = df[df["tracklet_key"].astype(str) == tracklet_key].sort_values("frame_idx")
        if sub.empty:
            continue
        indexes = [round(i * (len(sub) - 1) / max(frames_per_seq - 1, 1)) for i in range(frames_per_seq)]
        rows: list[dict[str, Any]] = []
        for idx in indexes:
            item = sub.iloc[int(idx)].to_dict()
            rows.append(
                {
                    "video_key": str(item["video_key"]),
                    "tracklet_key": str(item["tracklet_key"]),
                    "frame_idx": int(item["frame_idx"]),
                    "bbox_xyxy": [
                        float(item["x1"]),
                        float(item["y1"]),
                        float(item["x2"]),
                        float(item["y2"]),
                    ],
                    "det_score": float(item.get("score", 0.0)),
                }
            )
        return rows
    raise KeyError(f"tracklet_key not found in tracklet parquets: {tracklet_key}")


def _extract_video_frames(
    video_path: Path,
    video_key: str,
    frame_indexes: list[int],
    frames_dir: Path,
    ffmpeg_bin: str,
) -> dict[int, Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    unique_frames = sorted(set(int(frame) for frame in frame_indexes))
    out = {frame: frames_dir / f"{_safe_stem(video_key)}_f{frame}.png" for frame in unique_frames}
    missing = [frame for frame, path in out.items() if not path.is_file()]
    if not missing:
        return out

    tmp_dir = frames_dir / f".extract_{_safe_stem(video_key)}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    expr = "+".join(f"eq(n\\,{frame})" for frame in missing)
    pattern = tmp_dir / "frame_%06d.png"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select='{expr}'",
        "-vsync",
        "0",
        "-start_number",
        "0",
        "-y",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    produced = sorted(tmp_dir.glob("frame_*.png"))
    if len(produced) != len(missing):
        raise RuntimeError(f"ffmpeg produced {len(produced)} frames for {len(missing)} requested frames from {video_key}")
    for frame, tmp_path in zip(missing, produced, strict=True):
        shutil.move(str(tmp_path), str(out[frame]))
    shutil.rmtree(tmp_dir)
    return out


def _crop_box(frame_path: Path, bbox: list[float], out_path: Path, margin: float) -> list[int]:
    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx1 = max(0, int(round(x1 - margin * bw)))
    cy1 = max(0, int(round(y1 - margin * bh)))
    cx2 = min(width, int(round(x2 + margin * bw)))
    cy2 = min(height, int(round(y2 + margin * bh)))
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(f"invalid crop for {frame_path}: {bbox}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((cx1, cy1, cx2, cy2)).save(out_path)
    return [cx1, cy1, cx2, cy2]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _contact_sheet(rows: list[dict[str, Any]], out_path: Path, repo_root: Path) -> None:
    thumb_w, thumb_h = 190, 260
    cols = min(3, max(1, len(rows)))
    rows_n = (len(rows) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w, rows_n * thumb_h), (239, 242, 246))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    small = _font(12)
    for i, row in enumerate(rows):
        x = (i % cols) * thumb_w
        y = (i // cols) * thumb_h
        crop = Image.open(repo_root / row["crop_path"]).convert("RGB")
        crop.thumbnail((thumb_w - 18, thumb_h - 64), Image.Resampling.LANCZOS)
        canvas.paste(crop, (x + (thumb_w - crop.width) // 2, y + 8))
        color = (24, 116, 74) if row["role"] == "source_positive_subject" else (161, 70, 42)
        draw.rectangle([x, y + thumb_h - 49, x + thumb_w, y + thumb_h], fill=color)
        draw.text((x + 8, y + thumb_h - 45), f"seq {row['seq']} f{row['frame_idx']}", fill=(255, 255, 255), font=font)
        draw.text((x + 8, y + thumb_h - 24), str(row["role"]).replace("_", " "), fill=(237, 242, 247), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignment-csv", required=True, type=Path)
    ap.add_argument("--tracklet-parquet", action="append", type=Path, default=[])
    ap.add_argument("--tracklet-root", type=Path, default=Path("kit/demo_data/ds1/tracklets"))
    ap.add_argument("--raw-video-root", required=True, type=Path)
    ap.add_argument("--source-seq", required=True, type=int)
    ap.add_argument("--counter-seq", action="append", type=int, required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--frames-per-seq", type=int, default=3)
    ap.add_argument("--crop-margin", type=float, default=0.35)
    ap.add_argument("--ffmpeg-bin", default="ffmpeg")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    crops_dir = output_dir / "crops"

    parquets = list(args.tracklet_parquet)
    if not parquets:
        parquets = sorted(args.tracklet_root.glob("*/tracklets.parquet"))
    if not parquets:
        raise SystemExit("no tracklet parquet files found")

    assignments = _load_assignments(args.assignment_csv)
    seqs = [int(args.source_seq), *[int(seq) for seq in args.counter_seq]]
    selected = assignments[assignments["seq"].isin(seqs)].copy()
    if len(selected) != len(set(seqs)):
        missing = sorted(set(seqs).difference(set(int(seq) for seq in selected["seq"])))
        raise SystemExit(f"missing seqs in assignment csv: {missing}")

    sampled: list[dict[str, Any]] = []
    for seq in seqs:
        meta = selected[selected["seq"] == seq].iloc[0].to_dict()
        role = "source_positive_subject" if seq == int(args.source_seq) else "counter_ctf_negative"
        for row in _load_tracklet_rows(parquets, str(meta["tracklet_key"]), int(args.frames_per_seq)):
            row.update(
                {
                    "seq": int(seq),
                    "role": role,
                    "component_label": int(meta["component_label"]),
                    "predicted_global_id": int(meta["predicted_global_id"]),
                    "decision_status": str(meta.get("decision_status", "")),
                }
            )
            sampled.append(row)

    by_video: dict[str, list[int]] = {}
    video_paths: dict[str, Path] = {}
    for row in sampled:
        video_key = str(row["video_key"])
        by_video.setdefault(video_key, []).append(int(row["frame_idx"]))
        if video_key not in video_paths:
            video_paths[video_key] = _find_video(args.raw_video_root, video_key)
    extracted: dict[tuple[str, int], Path] = {}
    for video_key, frame_indexes in by_video.items():
        frames = _extract_video_frames(video_paths[video_key], video_key, frame_indexes, frames_dir, str(args.ffmpeg_bin))
        for frame, path in frames.items():
            extracted[(video_key, int(frame))] = path

    crop_rows: list[dict[str, Any]] = []
    for row in sampled:
        seq = int(row["seq"])
        frame_idx = int(row["frame_idx"])
        frame_path = extracted[(str(row["video_key"]), frame_idx)]
        crop_path = crops_dir / f"seq{seq}_frame{frame_idx}_crop.png"
        crop_xyxy = _crop_box(frame_path, list(row["bbox_xyxy"]), crop_path, float(args.crop_margin))
        prompt_identity = (
            "source subject: preserve clothing colors, body proportions, silhouette, camera crop style; "
            "vary only pose, background, lighting, mild blur"
            if row["role"] == "source_positive_subject"
            else "counter only: local hard negative for CTF; never use as a positive identity match"
        )
        crop_rows.append(
            {
                "seq": seq,
                "frame_idx": frame_idx,
                "crop_path": _repo_rel(crop_path, repo_root),
                "raw_frame_path": _repo_rel(frame_path, repo_root),
                "bbox_xyxy": [round(float(v), 4) for v in row["bbox_xyxy"]],
                "crop_xyxy": crop_xyxy,
                "det_score": float(row["det_score"]),
                "video": str(row["video_key"]),
                "tracklet_key": str(row["tracklet_key"]),
                "component_label": int(row["component_label"]),
                "predicted_global_id": int(row["predicted_global_id"]),
                "decision_status": str(row["decision_status"]),
                "prompt_hint": f"tracklet-local generated-positive probe for seq{seq}; {prompt_identity}.",
                "role": str(row["role"]),
            }
        )

    contact_path = output_dir / "source_counter_contact_sheet.png"
    _contact_sheet(crop_rows, contact_path, repo_root)
    manifest = {
        "task": "identity_preserving_generated_positive_inputs",
        "source_seq": int(args.source_seq),
        "counter_seqs": [int(seq) for seq in args.counter_seq],
        "moved_seqs": [int(args.source_seq)],
        "source_component": int(selected[selected["seq"] == int(args.source_seq)].iloc[0]["component_label"]),
        "source_predicted_global_id": int(selected[selected["seq"] == int(args.source_seq)].iloc[0]["predicted_global_id"]),
        "no_anchor": True,
        "gt_used_for_training_or_anchors": False,
        "generation_policy": (
            "Diffusion-ReID-style no-anchor probe: generate same-tracklet positives from the source crops only; "
            "counter tracklets provide hard negatives for DINOv2/SigLIP CTF. Generated images require CTF and "
            "full method reproduction before any assignment edit can be promoted."
        ),
        "identity_pair_warning": (
            f"Only seq{int(args.source_seq)} is the intended generated-positive subject. "
            "Counter tracklets are included only for no-anchor CTF negatives."
        ),
        "assignment_csv": _repo_rel(args.assignment_csv, repo_root),
        "tracklet_parquets": [_repo_rel(path, repo_root) for path in parquets],
        "raw_video_root": str(args.raw_video_root.resolve()),
        "raw_video_sha256": {video_key: _sha256(path) for video_key, path in sorted(video_paths.items())},
        "contact_sheet": _repo_rel(contact_path, repo_root),
        "crops": crop_rows,
    }
    out_path = output_dir / "prompt_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prompt_manifest": _repo_rel(out_path, repo_root), "crops": len(crop_rows), "no_anchor": True}, sort_keys=True))


if __name__ == "__main__":
    main()
