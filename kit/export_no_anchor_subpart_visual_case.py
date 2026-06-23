#!/usr/bin/env python3
"""Export a visual evidence case for a no-anchor subpart reassignment.

The case uses assignment CSVs, candidate manifests, and tracklet parquet boxes.
It does not use anchors.  Raw videos are optional; when unavailable, it renders
the true bbox trajectory on a fixed video-coordinate canvas so the case still
shows the identity evidence that drove the graph edit.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CANVAS_W = 2560
CANVAS_H = 1440


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


def _draw_panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, box: dict[str, Any], before: dict[str, str], after: dict[str, str]) -> None:
    font = _font(18)
    small = _font(15)
    draw.rectangle([x, y, x + w, y + h], fill=(245, 247, 250), outline=(42, 54, 71), width=2)
    header_h = 58
    draw.rectangle([x, y, x + w, y + header_h], fill=(23, 31, 42))
    draw.text((x + 12, y + 10), title, fill=(255, 255, 255), font=font)
    draw.text((x + 12, y + 34), f"frame {box['frame_idx']}  det {box['score']:.3f}", fill=(196, 208, 220), font=small)

    plot_x = x + 16
    plot_y = y + header_h + 14
    plot_w = w - 32
    plot_h = h - header_h - 104
    draw.rectangle([plot_x, plot_y, plot_x + plot_w, plot_y + plot_h], fill=(28, 35, 45), outline=(80, 90, 105))
    sx = plot_w / CANVAS_W
    sy = plot_h / CANVAS_H
    bx1, by1, bx2, by2 = box["bbox_xyxy"]
    rect = [plot_x + bx1 * sx, plot_y + by1 * sy, plot_x + bx2 * sx, plot_y + by2 * sy]
    draw.rectangle(rect, outline=(44, 203, 122), width=4)
    draw.text((rect[0] + 4, max(plot_y + 4, rect[1] - 20)), "bbox", fill=(44, 203, 122), font=small)
    draw.text((plot_x + 8, plot_y + 8), "video coordinate canvas", fill=(143, 155, 170), font=small)

    base_y = y + h - 82
    draw.text((x + 12, base_y), f"before: comp {before['component_label']} / gid {before['predicted_global_id']}", fill=(174, 65, 65), font=small)
    draw.text((x + 12, base_y + 22), f"after:  comp {after['component_label']} / gid {after['predicted_global_id']}", fill=(28, 132, 81), font=small)
    draw.text((x + 12, base_y + 44), f"{after['camera']}  frames {after['start_frame']}-{after['end_frame']}  n={after['n_dets']}", fill=(55, 65, 78), font=small)


def _make_montage(out_path: Path, case: dict[str, Any]) -> None:
    panels = []
    for item in case["tracklets"]:
        for box in item["sampled_boxes"]:
            panels.append((item, box))
    cols = 3
    panel_w = 560
    panel_h = 390
    margin = 28
    header_h = 190
    rows = (len(panels) + cols - 1) // cols
    img = Image.new("RGB", (cols * panel_w + (cols + 1) * margin, header_h + rows * panel_h + (rows + 1) * margin), (238, 241, 245))
    draw = ImageDraw.Draw(img)
    title_font = _font(34)
    body_font = _font(20)
    draw.text((margin, 24), str(case["title"]), fill=(24, 31, 42), font=title_font)
    draw.text((margin, 72), case["summary"], fill=(48, 58, 72), font=body_font)
    draw.text((margin, 106), case["failure_reason"], fill=(90, 66, 34), font=body_font)
    draw.text((margin, 140), case["improvement"], fill=(28, 100, 64), font=body_font)
    for idx, (item, box) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = margin + col * (panel_w + margin)
        y = header_h + margin + row * (panel_h + margin)
        _draw_panel(draw, x, y, panel_w, panel_h, f"seq {item['seq']}", box, item["before"], item["after"])
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
<div class="card"><b>Evidence</b><br>{html.escape(case['evidence'])}</div>
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
        tracklets.append({"seq": seq, "before": b, "after": a, "sampled_boxes": boxes})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_rel = ""
    if args.failure_montage and args.failure_montage.exists():
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
