#!/usr/bin/env python
"""Generate no-anchor tracklet-local positive images from restored VLINCS crops.

The production path is `--backend openai`, which reads OPENAI_API_KEY from the
environment and calls gpt-image-2 without accepting a key on the command line.
The `local_aug` backend is a deterministic dry-run used to validate provenance,
CTF gates, uploads, and downstream bookkeeping when no safe API key is present.

Generated images are never identity evidence by themselves. They must pass a
separate CTF audit and then improve direct + delivery scoring before promotion.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _safe_rel(path: str, base: Path) -> Path:
    item = Path(path)
    if item.is_absolute():
        return item
    return base / item


def _best_crops(manifest: dict[str, Any], repo_root: Path, max_refs: int) -> dict[int, list[dict[str, Any]]]:
    by_seq: dict[int, list[dict[str, Any]]] = {}
    for row in manifest.get("crops", []):
        if not isinstance(row, dict):
            continue
        seq = int(row["seq"])
        path = _safe_rel(str(row["crop_path"]), repo_root)
        if not path.is_file():
            continue
        row = dict(row)
        row["resolved_crop_path"] = str(path)
        by_seq.setdefault(seq, []).append(row)
    for seq, rows in by_seq.items():
        rows.sort(key=lambda item: float(item.get("det_score") or 0.0), reverse=True)
        by_seq[seq] = rows[: max(int(max_refs), 1)]
    return by_seq


def _prompt_for_seq(seq: int, rows: list[dict[str, Any]], variants: int) -> str:
    hints = [str(row.get("prompt_hint", "")) for row in rows if row.get("prompt_hint")]
    hint_text = " ".join(hints)
    return (
        "Create identity-preserving positive examples for a person re-identification research audit. "
        "Use the reference crops only for the same tracklet, not for cross-tracklet identity linking. "
        "Preserve the visible clothing colors, body proportions, silhouette, occlusion cues, and camera crop style. "
        "Vary only pose, small viewpoint, illumination, background clutter, and mild motion blur. "
        "Do not add text, logos, watermarks, new people as the main subject, or change the outfit. "
        f"Tracklet seq={seq}. Generate {variants} candidate images. {hint_text}"
    )


def _augment_image(image: Image.Image, seed: int, variant: int) -> Image.Image:
    rng = random.Random((int(seed) + 1009) * 9176 + int(variant) * 37)
    img = image.convert("RGB")
    angle = rng.uniform(-4.5, 4.5)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(235, 238, 242))
    # Mild crop/scale jitter while preserving most identity evidence.
    w, h = img.size
    crop_frac = rng.uniform(0.00, 0.055)
    dx = int(w * crop_frac)
    dy = int(h * crop_frac)
    if w - 2 * dx > 16 and h - 2 * dy > 16:
        img = img.crop((dx, dy, w - dx, h - dy))
    scale = rng.uniform(0.92, 1.08)
    nw = max(32, int(img.width * scale))
    nh = max(32, int(img.height * scale))
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.86, 1.12))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.90, 1.14))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.92, 1.10))
    if rng.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.7)))
    canvas = Image.new("RGB", (max(180, img.width + 52), max(260, img.height + 62)), (231, 235, 240))
    ox = (canvas.width - img.width) // 2 + rng.randint(-8, 8)
    oy = (canvas.height - img.height) // 2 + rng.randint(-8, 8)
    canvas.paste(img, (ox, oy))
    return canvas


def _run_local_aug(seq: int, rows: list[dict[str, Any]], out_dir: Path, variants: int, seed: int) -> list[dict[str, Any]]:
    outputs = []
    for variant in range(int(variants)):
        src = rows[variant % len(rows)]
        src_path = Path(str(src["resolved_crop_path"]))
        image = Image.open(src_path)
        generated = _augment_image(image, seed=int(seed) + int(seq), variant=variant)
        out_path = out_dir / f"seq{seq}_local_aug_v{variant:02d}.png"
        generated.save(out_path)
        outputs.append(
            {
                "seq": int(seq),
                "variant": int(variant),
                "backend": "local_aug",
                "image_path": str(out_path),
                "source_crop_path": str(src_path),
                "source_frame_idx": int(src.get("frame_idx", -1)),
                "prompt": _prompt_for_seq(seq, rows, variants),
            }
        )
    return outputs


def _run_openai(seq: int, rows: list[dict[str, Any]], out_dir: Path, variants: int, model: str) -> list[dict[str, Any]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; refusing to use key literals or command-line secrets")
    try:
        from openai import OpenAI  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("openai package is missing; install it in the active environment") from exc

    client = OpenAI()
    outputs = []
    refs = [open(str(row["resolved_crop_path"]), "rb") for row in rows]
    try:
        for variant in range(int(variants)):
            prompt = _prompt_for_seq(seq, rows, 1) + f" Variant index {variant}."
            result = client.images.edit(model=model, image=refs, prompt=prompt)
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            out_path = out_dir / f"seq{seq}_{model}_v{variant:02d}.png"
            out_path.write_bytes(image_bytes)
            outputs.append(
                {
                    "seq": int(seq),
                    "variant": int(variant),
                    "backend": "openai",
                    "model": model,
                    "image_path": str(out_path),
                    "reference_crop_paths": [str(row["resolved_crop_path"]) for row in rows],
                    "prompt": prompt,
                }
            )
    finally:
        for handle in refs:
            handle.close()
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-manifest", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--backend", default="local_aug", choices=["local_aug", "openai"])
    ap.add_argument("--model", default="gpt-image-2")
    ap.add_argument("--variants-per-seq", type=int, default=2)
    ap.add_argument("--max-reference-crops", type=int, default=2)
    ap.add_argument(
        "--include-seq",
        action="append",
        type=int,
        default=[],
        help="Only generate for these seq ids; repeatable. Counter seqs can remain in the prompt manifest for CTF.",
    )
    ap.add_argument("--seed", type=int, default=62624)
    ap.add_argument("--repo-root", default="", help="root used to resolve relative crop paths; defaults to current directory")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    manifest = _load_json(args.prompt_manifest)
    if not bool(manifest.get("no_anchor")):
        raise SystemExit("prompt manifest must explicitly be no_anchor=true")
    by_seq = _best_crops(manifest, repo_root, int(args.max_reference_crops))
    if args.include_seq:
        keep = {int(seq) for seq in args.include_seq}
        by_seq = {seq: rows for seq, rows in by_seq.items() if seq in keep}
    if not by_seq:
        raise SystemExit("no readable source crops found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    for seq, rows in sorted(by_seq.items()):
        if args.backend == "openai":
            generated.extend(_run_openai(seq, rows, images_dir, int(args.variants_per_seq), str(args.model)))
        else:
            generated.extend(_run_local_aug(seq, rows, images_dir, int(args.variants_per_seq), int(args.seed)))

    out_manifest = {
        "task": "no_anchor_tracklet_local_generated_positives",
        "backend": str(args.backend),
        "model": str(args.model) if args.backend == "openai" else "",
        "source_prompt_manifest": str(args.prompt_manifest),
        "generation_policy": manifest.get("generation_policy", ""),
        "identity_pair_warning": manifest.get("identity_pair_warning", ""),
        "included_seqs": sorted(int(seq) for seq in by_seq),
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "generated_images": generated,
    }
    out_path = args.output_dir / "generated_manifest.json"
    out_path.write_text(json.dumps(out_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generated_manifest": str(out_path), "images": len(generated), "backend": args.backend}, sort_keys=True))


if __name__ == "__main__":
    main()
