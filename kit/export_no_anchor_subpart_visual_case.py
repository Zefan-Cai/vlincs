#!/usr/bin/env python3
"""Export a visual evidence case for a no-anchor subpart reassignment.

The case uses assignment CSVs, candidate manifests, and tracklet parquet boxes.
It does not use anchors. Raw videos are optional; when unavailable, it renders
zoomed coordinate panels around each bbox plus a full-frame locator so small
boxes remain reviewable.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CANVAS_W = 2560
CANVAS_H = 1440
GREEN = (31, 208, 126)
YELLOW = (255, 197, 71)
RED = (193, 76, 76)
MUTED = (96, 110, 128)


def _load_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(float(row["seq"])): dict(row) for row in csv.DictReader(handle)}


def _load_candidate(path: Path, rank: int) -> dict[str, Any]:
    data = json.loads(path.read_text())
    for row in data.get("selected", []) + data.get("top_candidates", []):
        if isinstance(row, dict) and int(row.get("rank", -1)) == int(rank):
            return row
    raise SystemExit(f"rank {rank} not found in {path}")


def _font(size: int = 22) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int = 2) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    consumed = 0
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
            consumed += 1
            continue
        if current:
            lines.append(current)
        current = word
        consumed += 1
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if consumed < len(words) and lines:
        while draw.textlength(lines[-1] + "...", font=font) > max_width and len(lines[-1]) > 4:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "..."
    return lines


def _crop_window(
    bbox: list[float],
    aspect: float,
    frame_w: int = CANVAS_W,
    frame_h: int = CANVAS_H,
) -> tuple[float, float, float, float]:
    bx1, by1, bx2, by2 = bbox
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    crop_h = max(155.0, bh * 2.35)
    crop_w = max(220.0, bw * 4.0)
    if crop_w / crop_h < aspect:
        crop_w = crop_h * aspect
    else:
        crop_h = crop_w / aspect
    crop_w = min(float(frame_w), crop_w)
    crop_h = min(float(frame_h), crop_h)
    x1 = max(0.0, min(cx - crop_w / 2.0, float(frame_w) - crop_w))
    y1 = max(0.0, min(cy - crop_h / 2.0, float(frame_h) - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def _map_point(px: float, py: float, crop: tuple[float, float, float, float], plot: tuple[int, int, int, int]) -> tuple[float, float]:
    cx1, cy1, cx2, cy2 = crop
    px1, py1, px2, py2 = plot
    return (
        px1 + (px - cx1) * (px2 - px1) / max(cx2 - cx1, 1.0),
        py1 + (py - cy1) * (py2 - py1) / max(cy2 - cy1, 1.0),
    )


def _draw_zoom_grid(draw: ImageDraw.ImageDraw, plot: tuple[int, int, int, int]) -> None:
    px1, py1, px2, py2 = plot
    for i in range(1, 4):
        xx = px1 + i * (px2 - px1) / 4.0
        draw.line([(xx, py1), (xx, py2)], fill=(43, 53, 68), width=1)
    for i in range(1, 3):
        yy = py1 + i * (py2 - py1) / 3.0
        draw.line([(px1, yy), (px2, yy)], fill=(43, 53, 68), width=1)


def _draw_locator(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    crop: tuple[float, float, float, float],
    bbox: list[float],
    frame_w: int = CANVAS_W,
    frame_h: int = CANVAS_H,
) -> None:
    _, py1, px2, _ = plot
    inset_w = 132
    inset_h = int(inset_w * frame_h / max(frame_w, 1))
    ix2 = px2 - 12
    iy1 = py1 + 12
    ix1 = ix2 - inset_w
    iy2 = iy1 + inset_h
    draw.rectangle([ix1 - 3, iy1 - 3, ix2 + 3, iy2 + 3], fill=(18, 24, 33), outline=(202, 210, 221))
    draw.rectangle([ix1, iy1, ix2, iy2], fill=(30, 38, 50), outline=(145, 157, 173))

    def loc_rect(rect: tuple[float, float, float, float] | list[float]) -> list[float]:
        x1, y1, x2, y2 = rect
        sx = inset_w / max(frame_w, 1)
        sy = inset_h / max(frame_h, 1)
        return [ix1 + x1 * sx, iy1 + y1 * sy, ix1 + x2 * sx, iy1 + y2 * sy]

    draw.rectangle(loc_rect(crop), outline=YELLOW, width=2)
    draw.rectangle(loc_rect(bbox), outline=GREEN, width=2)


def _draw_bbox_rect(draw: ImageDraw.ImageDraw, rect: list[float], color: tuple[int, int, int], width: int = 6) -> None:
    for offset in range(width):
        draw.rectangle(
            [rect[0] - offset, rect[1] - offset, rect[2] + offset, rect[3] + offset],
            outline=color,
        )


def _sample_boxes(tracklet_parquets: list[Path], tracklet_key: str, n: int) -> list[dict[str, Any]]:
    video = tracklet_key.split(":")[0]
    candidates = [p for p in tracklet_parquets if p.parent.name == video]
    if not candidates:
        candidates = tracklet_parquets
    rows = []
    for path in candidates:
        df = pd.read_parquet(path)
        if "tracklet_key" not in df.columns:
            continue
        sub = df[df["tracklet_key"] == tracklet_key].sort_values("frame_idx")
        if len(sub):
            for idx in [round(i * (len(sub) - 1) / max(n - 1, 1)) for i in range(n)]:
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
                        "score": float(item.get("score", 0.0)),
                    }
                )
            return rows
    return rows


def _safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _find_raw_frame(raw_frame_dir: Path | None, video_key: str, tracklet_key: str, frame_idx: int) -> Path | None:
    if raw_frame_dir is None or not raw_frame_dir.is_dir():
        return None
    patterns = [
        f"{video_key}*f{frame_idx}*.png",
        f"{video_key}*f{frame_idx}*.jpg",
        f"{video_key}*{frame_idx}*.png",
        f"{video_key}*{frame_idx}*.jpg",
        f"{_safe_stem(tracklet_key)}*f{frame_idx}*.png",
        f"{_safe_stem(tracklet_key)}*f{frame_idx}*.jpg",
    ]
    for pattern in patterns:
        hits = sorted(raw_frame_dir.rglob(pattern))
        if hits:
            return hits[0]
    return None


def _find_video_file(video_root: Path | None, video_key: str) -> Path | None:
    if video_root is None or not video_root.is_dir():
        return None
    for suffix in (".mp4", ".avi", ".mov", ".mkv", ".m4v"):
        direct = video_root / f"{video_key}{suffix}"
        if direct.is_file():
            return direct
    for suffix in (".mp4", ".avi", ".mov", ".mkv", ".m4v"):
        hits = sorted(video_root.rglob(f"{video_key}*{suffix}"))
        if hits:
            return hits[0]
    return None


def _extract_video_frame(
    video_root: Path | None,
    video_key: str,
    frame_idx: int,
    output_dir: Path,
    ffmpeg_bin: str,
) -> Path | None:
    video_path = _find_video_file(video_root, video_key)
    if video_path is None:
        return None
    out_dir = output_dir / "raw_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_stem(video_key)}_f{frame_idx}.png"
    if out_path.is_file():
        return out_path
    expr = f"select=eq(n\\,{int(frame_idx)})"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        expr,
        "-frames:v",
        "1",
        "-y",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out_path if out_path.is_file() else None


def _attach_raw_frames(
    boxes: list[dict[str, Any]],
    raw_frame_dir: Path | None,
    video_root: Path | None,
    output_dir: Path,
    ffmpeg_bin: str,
) -> None:
    for box in boxes:
        video_key = str(box.get("video_key", ""))
        tracklet_key = str(box.get("tracklet_key", ""))
        frame_idx = int(box.get("frame_idx", -1))
        raw_path = _find_raw_frame(raw_frame_dir, video_key, tracklet_key, frame_idx)
        source = "raw_frame_dir"
        if raw_path is None:
            raw_path = _extract_video_frame(video_root, video_key, frame_idx, output_dir, ffmpeg_bin)
            source = "video_root"
        if raw_path is not None:
            box["raw_frame_path"] = str(raw_path)
            box["raw_frame_source"] = source


def _draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    box: dict[str, Any],
    before: dict[str, str],
    after: dict[str, str],
    track_boxes: list[dict[str, Any]],
) -> None:
    font = _font(18)
    small = _font(15)
    tiny = _font(13)
    draw.rectangle([x, y, x + w, y + h], fill=(245, 247, 250), outline=(42, 54, 71), width=2)
    header_h = 62
    draw.rectangle([x, y, x + w, y + header_h], fill=(23, 31, 42))
    draw.text((x + 12, y + 10), title, fill=(255, 255, 255), font=font)
    draw.text((x + 12, y + 34), f"frame {box['frame_idx']}  det {box['score']:.3f}", fill=(196, 208, 220), font=small)

    plot_x = x + 16
    plot_y = y + header_h + 14
    plot_w = w - 32
    plot_h = h - header_h - 126
    plot = (plot_x, plot_y, plot_x + plot_w, plot_y + plot_h)
    bx1, by1, bx2, by2 = box["bbox_xyxy"]
    raw_image = None
    raw_bbox = [bx1, by1, bx2, by2]
    frame_w = CANVAS_W
    frame_h = CANVAS_H
    raw_frame_path = box.get("raw_frame_path")
    if raw_frame_path:
        try:
            raw_image = Image.open(str(raw_frame_path)).convert("RGB")
            frame_w, frame_h = raw_image.size
            if (frame_w, frame_h) != (CANVAS_W, CANVAS_H):
                sx = frame_w / CANVAS_W
                sy = frame_h / CANVAS_H
                raw_bbox = [bx1 * sx, by1 * sy, bx2 * sx, by2 * sy]
        except OSError:
            raw_image = None

    crop = _crop_window(raw_bbox, plot_w / max(plot_h, 1), frame_w=frame_w, frame_h=frame_h)
    if raw_image is not None:
        crop_i = tuple(int(round(v)) for v in crop)
        crop_img = raw_image.crop(crop_i).resize((plot_w, plot_h), Image.Resampling.BILINEAR)
        canvas.paste(crop_img, (plot_x, plot_y))
        draw.rectangle(plot, outline=(80, 90, 105))
    else:
        draw.rectangle(plot, fill=(28, 35, 45), outline=(80, 90, 105))
        draw.text(
            (plot_x + 10, plot_y + 30),
            "RAW FRAME UNAVAILABLE - coordinate-only fallback",
            fill=(255, 197, 71),
            font=small,
        )
        _draw_zoom_grid(draw, plot)
    _draw_locator(draw, plot, crop, raw_bbox, frame_w=frame_w, frame_h=frame_h)

    centers = []
    for sample in track_boxes:
        sx1, sy1, sx2, sy2 = sample["bbox_xyxy"]
        if (frame_w, frame_h) != (CANVAS_W, CANVAS_H):
            sx1, sx2 = sx1 * frame_w / CANVAS_W, sx2 * frame_w / CANVAS_W
            sy1, sy2 = sy1 * frame_h / CANVAS_H, sy2 * frame_h / CANVAS_H
        center = ((sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0)
        px, py = _map_point(center[0], center[1], crop, plot)
        if plot_x - 12 <= px <= plot_x + plot_w + 12 and plot_y - 12 <= py <= plot_y + plot_h + 12:
            centers.append((px, py, int(sample["frame_idx"])))
    if len(centers) > 1:
        draw.line([(px, py) for px, py, _ in centers], fill=YELLOW, width=3)
    for px, py, frame_idx in centers:
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=YELLOW, outline=(18, 24, 33), width=2)
        draw.text((px + 6, py - 6), str(frame_idx), fill=(255, 226, 145), font=tiny)

    rx1, ry1 = _map_point(raw_bbox[0], raw_bbox[1], crop, plot)
    rx2, ry2 = _map_point(raw_bbox[2], raw_bbox[3], crop, plot)
    rect = [rx1, ry1, rx2, ry2]
    _draw_bbox_rect(draw, rect, GREEN, width=6)
    draw.text((max(plot_x + 8, rect[0] + 6), max(plot_y + 8, rect[1] - 22)), "zoomed bbox", fill=GREEN, font=small)
    mode = "raw-frame crop; inset shows full-frame location" if raw_image is not None else "bbox-centered coordinate zoom; inset shows full-frame location"
    draw.text((plot_x + 10, plot_y + 10), mode, fill=(190, 202, 216), font=tiny)
    draw.text((plot_x + 10, plot_y + plot_h - 22), f"zoom crop xyxy {crop[0]:.0f},{crop[1]:.0f},{crop[2]:.0f},{crop[3]:.0f}", fill=(173, 184, 198), font=tiny)

    base_y = y + h - 104
    draw.text((x + 12, base_y), f"before: comp {before['component_label']} / gid {before['predicted_global_id']}", fill=RED, font=small)
    draw.text((x + 12, base_y + 22), f"after:  comp {after['component_label']} / gid {after['predicted_global_id']}", fill=(25, 135, 82), font=small)
    draw.text((x + 12, base_y + 44), f"{after['camera']}  frames {after['start_frame']}-{after['end_frame']}  n={after['n_dets']}", fill=(55, 65, 78), font=small)
    draw.text((x + 12, base_y + 66), f"bbox xyxy {bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}", fill=MUTED, font=tiny)


def _make_montage(out_path: Path, case: dict[str, Any]) -> None:
    panels = []
    for item in case["tracklets"]:
        for box in item["sampled_boxes"]:
            panels.append((item, box))
    cols = 3
    panel_w = 640
    panel_h = 470
    margin = 28
    header_h = 214
    rows = (len(panels) + cols - 1) // cols
    img = Image.new("RGB", (cols * panel_w + (cols + 1) * margin, header_h + rows * panel_h + (rows + 1) * margin), (238, 241, 245))
    draw = ImageDraw.Draw(img)
    title_font = _font(34)
    body_font = _font(20)
    draw.text((margin, 24), str(case["title"]), fill=(24, 31, 42), font=title_font)
    yy = 72
    max_text_w = img.width - margin * 2
    for text, color in (
        (case["summary"], (48, 58, 72)),
        (case["failure_reason"], (90, 66, 34)),
        (case["improvement"], (28, 100, 64)),
    ):
        for line in _wrap_text(draw, text, body_font, max_text_w, max_lines=2):
            draw.text((margin, yy), line, fill=color, font=body_font)
            yy += 28
        yy += 4
    for idx, (item, box) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = margin + col * (panel_w + margin)
        y = header_h + margin + row * (panel_h + margin)
        _draw_panel(img, draw, x, y, panel_w, panel_h, f"seq {item['seq']}", box, item["before"], item["after"], item["sampled_boxes"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _write_html(out_path: Path, case: dict[str, Any]) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{item['seq']}</td><td>{html.escape(item['after']['video'])}</td>"
        f"<td>{item['before']['predicted_global_id']} -> {item['after']['predicted_global_id']}</td>"
        f"<td>{item['before']['component_label']} -> {item['after']['component_label']}</td>"
        f"<td>{item['after']['start_frame']}-{item['after']['end_frame']}</td>"
        f"<td>{item['after']['n_dets']}</td>"
        "</tr>"
        for item in case["tracklets"]
    )
    failure_img = ""
    if case.get("failure_montage"):
        failure_img = f"<h2>Previous visual failure mode</h2><img src=\"{html.escape(case['failure_montage'])}\" alt=\"failure montage\">"
    out_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(case['title'])}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:32px;background:#f6f7f9;color:#17202c}}
main{{max-width:1120px;margin:auto}} img{{max-width:100%;border:1px solid #ccd3dc;background:white}}
.kpi{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}
.card{{background:white;border:1px solid #d7dde5;padding:14px}} table{{border-collapse:collapse;width:100%;background:white}}
td,th{{border:1px solid #d7dde5;padding:8px;text-align:left}} code{{background:#eef1f5;padding:2px 4px}}
</style></head><body><main>
<h1>{html.escape(case['title'])}</h1>
<p>{html.escape(case['summary'])}</p>
<div class="kpi">
<div class="card"><b>Before</b><br>{html.escape(case['before_state'])}</div>
<div class="card"><b>After</b><br>{html.escape(case['after_state'])}</div>
<div class="card"><b>Evidence</b><br>{html.escape(case['evidence'])}<br><b>Raw frame</b><br>{html.escape(case.get('raw_frame_status',''))}</div>
</div>
<h2>Failure and fix</h2>
<p><b>Old failure:</b> {html.escape(case['failure_reason'])}</p>
<p><b>New implementation:</b> {html.escape(case['improvement'])}</p>
<h2>BBox evidence</h2>
<img src="{html.escape(case['montage'])}" alt="rank15 bbox evidence">
<h2>Moved tracklets</h2>
<table><thead><tr><th>seq</th><th>video</th><th>gid</th><th>component</th><th>frames</th><th>dets</th></tr></thead><tbody>{rows}</tbody></table>
{failure_img}
</main></body></html>
""",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before-assignments", required=True, type=Path)
    ap.add_argument("--after-assignments", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--rank", required=True, type=int)
    ap.add_argument("--tracklet-parquet", nargs="+", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--raw-frame-dir", default=None, type=Path, help="Optional directory containing pre-extracted raw frame images.")
    ap.add_argument("--video-root", default=None, type=Path, help="Optional video root; if present, ffmpeg extracts exact frame images.")
    ap.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable used with --video-root.")
    ap.add_argument("--failure-montage", default="", type=Path)
    ap.add_argument("--title", default="No-anchor subpart reassignment visual case")
    ap.add_argument("--summary", default="")
    ap.add_argument("--failure-reason", default="")
    ap.add_argument("--improvement", default="")
    args = ap.parse_args()

    before = _load_rows(args.before_assignments)
    after = _load_rows(args.after_assignments)
    cand = _load_candidate(args.manifest, args.rank)
    seqs = [int(seq) for seq in cand.get("source_seqs", [])]
    tracklets = []
    for seq in seqs:
        b = before[seq]
        a = after[seq]
        boxes = _sample_boxes(args.tracklet_parquet, a["tracklet_key"], 3)
        _attach_raw_frames(boxes, args.raw_frame_dir, args.video_root, args.output_dir, args.ffmpeg_bin)
        tracklets.append({"seq": seq, "before": b, "after": a, "sampled_boxes": boxes})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_rel = ""
    if args.failure_montage and args.failure_montage.is_file():
        dst = args.output_dir / args.failure_montage.name
        shutil.copy2(args.failure_montage, dst)
        failure_rel = dst.name

    first_before = before[seqs[0]]
    first_after = after[seqs[0]]
    summary = args.summary or (
        f"{len(seqs)} tracklets move from component {first_before['component_label']} / gid {first_before['predicted_global_id']} "
        f"to component {first_after['component_label']} / gid {first_after['predicted_global_id']}."
    )
    failure_reason = args.failure_reason or (
        "The previous graph kept a small visual island inside a larger source component, splitting it away from a closer target component."
    )
    improvement = args.improvement or (
        "The subpart repair tests a conflict-supported subgroup and commits only compatible tracklets to the target component."
    )

    montage_name = f"rank{int(args.rank):02d}_bbox_evidence.png"
    case = {
        "title": args.title,
        "summary": summary,
        "before_state": f"component {first_before['component_label']}, gid {first_before['predicted_global_id']}, component_size {first_before.get('component_size', '')}",
        "after_state": f"component {first_after['component_label']}, gid {first_after['predicted_global_id']}, component_size {first_after.get('component_size', '')}, decision_status {first_after.get('decision_status', '')}",
        "evidence": f"target_sim={float(cand.get('target_sim', 0.0)):.4f}, target_margin={float(cand.get('target_margin', 0.0)):.4f}, source_rest_margin={float(cand.get('source_rest_margin_mean', 0.0)):.4f}",
        "failure_reason": failure_reason,
        "improvement": improvement,
        "candidate": cand,
        "tracklets": tracklets,
        "montage": montage_name,
        "failure_montage": failure_rel,
        "raw_frame_status": "available" if any(
            box.get("raw_frame_path")
            for item in tracklets
            for box in item.get("sampled_boxes", [])
        ) else "unavailable_coordinate_fallback",
        "raw_frame_dir": str(args.raw_frame_dir) if args.raw_frame_dir else "",
        "video_root": str(args.video_root) if args.video_root else "",
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    _make_montage(args.output_dir / case["montage"], case)
    _write_html(args.output_dir / "case.html", case)
    (args.output_dir / "case.json").write_text(json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "files": ["case.json", "case.html", case["montage"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
