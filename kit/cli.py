#!/usr/bin/env python
"""Kit CLI - the few commands a colleague runs around their own pipeline.

  list-videos --dataset ds1   the video stems + camera names to run YOUR pipeline on (+ GT availability)
  demo                        one-click REAL end-to-end run on MS02 or DS1; can import weak labels
  score       --dataset ds1   IDF1 (canonical reid_hota) from whatever is currently in the gallery DB
  generate-weak-labels --tracklets-root kit/demo_data/ds1/tracklets --out weak.csv
  import-weak-labels --dataset ds1 --csv weak.csv --source clip-vitb32
  weak-resolve --dataset ds1 --source clip-vitb32 --apply
  submit      --dataset ds1 --out out.zip    export a TA1 submission from the DB (the only file the kit writes)
  selftest    --dataset ms02  wiring check on RANDOM embeddings (proves the pipeline runs; not a real score)

The real work - pushing your tracklets/embeddings - is the Gallery API (see example.py), driven from
your own environment on your own cadence. This CLI is just the bookends.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from online import OnlineGallery, CARDDIRS, HAS_GT, DATA   # kit-local
from vlincs_gallery import WeakGraphConfig, WeakLabelGenerationConfig, generate_weak_labels

# Reference scores for context, NOT a hard target. The online ref is what this kit's reference config
# scores online (no prior knowledge); the supervised ceiling (ds1) is the batch funnel trained on DS1 GT.
ONLINE_REF = {
    "ms02": "AssA ~0.70 (lead AssA - sparse GT)",
    "ds1": "IDF1 ~0.53 online ref (supervised ceiling 0.69)",
    "ds2": "0.49 (leaderboard)",
}


def _videos(dataset: str):
    """Enumerate a dataset's videos from its card directories.

    Args:
        dataset: Dataset key (e.g. ``"ds1"``, ``"ms02"``); indexes into ``CARDDIRS``.

    Returns:
        list[tuple[str, str, str]]: ``(video_stem, camera, mp4_path)`` per ``.mp4``, sorted by path.
    """
    out = []
    for d in CARDDIRS.get(dataset, []):
        for mp4 in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            stem = Path(mp4).stem
            cam = next((p for p in stem.split("_") if p.startswith("MCAM")), "CAM")
            out.append((stem, cam, mp4))
    return out


def cmd_list_videos(a):
    """List a dataset's video stems, cameras, and GT availability.

    These are the inputs to run your own detector/tracker/embedder on before pushing results
    through the Gallery API (see ``example.py``).
    """
    vids = _videos(a.dataset)
    gt = "yes (local IDF1)" if HAS_GT.get(a.dataset) else "NONE (leaderboard only - no local IDF1)"
    print(f"[{a.dataset}] {len(vids)} videos  |  ground truth: {gt}  |  data root: {DATA}")
    print(f"{'video stem':52} {'camera':8} path")
    for stem, cam, mp4 in vids:
        print(f"{stem:52} {cam:8} {mp4}")
    print("\nRun YOUR detector+tracker+embedder on these, push via the Gallery API (see example.py).")


def cmd_score(a):
    """Print the canonical reid_hota score for whatever identities are currently in the gallery DB."""
    g = OnlineGallery(a.dataset, truncate=False)
    print(f"[score] {g.score()}   (this kit's online ref: {ONLINE_REF.get(a.dataset, '?')})")
    g.close()


def cmd_demo(a):
    """Run the one-click end-to-end demo: MS02 from the shipped bundle, or DS1 from pipelines/ds1.yaml."""
    # CLI flags (resolve_every / cannot_link) default to None so the dataset yaml decides; they
    # override only when explicitly passed.
    from demo import run_demo
    # resolve_every + cannot_link default to None so run_demo / the dataset yaml decide (DS1's yaml sets
    # cannot_link=false); the CLI flags below override only when explicitly passed.
    run_demo(
        a.dataset,
        resolve_every=a.resolve_every,
        submit=a.submit,
        cannot_link=a.cannot_link,
        weak_label_csv=a.weak_label_csv,
        weak_source=a.weak_source,
        auto_weak_labels=a.auto_weak_labels,
        weak_resolve=a.weak_resolve,
        weak_embedding_role=a.weak_embedding_role,
        weak_min_dets=a.weak_min_dets,
    )


def cmd_selftest(a):
    """Wiring check on RANDOM embeddings: proves the pipeline runs end-to-end.

    Not a real score - the embeddings are synthetic Gaussian clusters. Use ``demo`` for a real run.
    """
    import numpy as np
    g = OnlineGallery(a.dataset, truncate=True)
    vids = _videos(a.dataset)[:4] or [("demo_MCAM00", "MCAM00", ""), ("demo_MCAM01", "MCAM01", "")]
    rng = np.random.RandomState(0)
    centers = rng.randn(8, 256).astype("float32")
    n = 0
    for stem, cam, _ in vids:
        for pid in range(8):
            emb = centers[pid] + 0.1 * rng.randn(256).astype("float32")
            g.add_tracklet(stem, cam, [pid * 7], [[10, 10, 60, 160]], emb)
            n += 1
    g.resolve()
    print(f"[selftest] pushed {n} RANDOM tracklets across {len(vids)} videos -> {g.score()}")
    print("[selftest] (random embeddings - proves the pipeline runs end-to-end, not a real number; use `demo`)")
    g.close()


def cmd_submit(a):
    """Export a TA1 submission zip from the current gallery DB to ``a.out``."""
    g = OnlineGallery(a.dataset, truncate=False)
    g.export_submission(a.out)
    g.close()


def cmd_import_weak_labels(a):
    """Import an offline weak-label CSV keyed by tracklet_key into the current gallery DB."""
    g = OnlineGallery(a.dataset, truncate=False)
    info = g.import_weak_labels_csv(
        a.csv,
        source=a.source,
        key_col=a.key_col,
        token_col=a.token_col,
        confidence_col=a.confidence_col,
    )
    print(f"[import-weak-labels] {info}")
    g.close()


def cmd_generate_weak_labels(a):
    """Generate a no-GT weak-label CSV from tracklet boxes and optional source videos."""
    cfg = WeakLabelGenerationConfig(
        sample_frames=a.sample_frames,
        min_tracklet_dets=a.min_tracklet_dets,
        max_tracklets=a.max_tracklets,
        include_crop_colors=not a.no_crop_colors,
        crop_margin=a.crop_margin,
    )
    try:
        info = generate_weak_labels(a.tracklets_root, a.out, video_root=a.videos_root, cfg=cfg)
    except ImportError as e:
        raise SystemExit(f"[generate-weak-labels] {e}") from e
    print(f"[generate-weak-labels] {info}")


def cmd_weak_resolve(a):
    """Run no-GT weak-supervision global resolve over stored tracklets."""
    cfg = WeakGraphConfig(
        visual_top_k=a.visual_top_k,
        edge_threshold=a.edge_threshold,
        max_token_df=a.max_token_df,
        lp_iterations=a.lp_iterations,
        max_component_size=a.max_component_size,
    )
    g = OnlineGallery(a.dataset, truncate=False)
    info = g.resolve_weak_global(
        cfg,
        embedding_role=a.embedding_role,
        weak_source=a.source,
        min_dets=a.min_dets,
        apply=a.apply,
    )
    print(f"[weak-resolve] {info}")
    if a.submit:
        g.export_submission(a.submit)
        print(f"[weak-resolve] wrote submission {a.submit}")
    g.close()


def main():
    """Parse argv and dispatch to the selected subcommand."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list-videos", "score", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=list(CARDDIRS))
    pdemo = sub.add_parser("demo")                       # real one-click demo (ms02 shipped; ds1 = pipelines/ds1.yaml)
    pdemo.add_argument("--dataset", default="ms02", choices=["ms02", "ds1"],
                       help="ms02 = shipped bundle (offline); ds1 = pipelines/ds1.yaml (bundle or MLflow)")
    pdemo.add_argument("--resolve-every", type=int, default=None, help="periodic resolve every N tracklets (default ds1=500, ms02=100)")
    pdemo.add_argument("--submit", default=None, help="also write a TA1 submission zip here")
    pdemo.add_argument("--cannot-link", dest="cannot_link", action="store_true", default=None,
                       help="force vetoes ON (overrides the dataset yaml)")
    pdemo.add_argument("--no-cannot-link", dest="cannot_link", action="store_false",
                       help="force vetoes OFF / appearance-only (overrides the dataset yaml; DS1's yaml already sets this)")
    pdemo.add_argument("--weak-label-csv", default=None,
                       help="optional LM/VLM/CV weak-label CSV keyed by tracklet_key; imported after ingest")
    pdemo.add_argument("--weak-source", default="weak", help="provenance name for --weak-label-csv")
    pdemo.add_argument("--auto-weak-labels", action="store_true",
                       help="derive bbox/time weak tokens from loaded tracklets and store them during ingest")
    pdemo.add_argument("--weak-resolve", action="store_true",
                       help="run no-GT weak graph forced-output resolution before scoring")
    pdemo.add_argument("--weak-embedding-role", default="resolve", choices=["resolve", "match"],
                       help="stored embedding role used by --weak-resolve")
    pdemo.add_argument("--weak-min-dets", type=int, default=1,
                       help="minimum detections per tracklet considered by --weak-resolve")
    ps = sub.add_parser("submit")
    ps.add_argument("--dataset", required=True, choices=list(CARDDIRS))
    ps.add_argument("--out", default="submission.zip")
    pw = sub.add_parser("import-weak-labels")
    pw.add_argument("--dataset", required=True, choices=list(CARDDIRS))
    pw.add_argument("--csv", required=True)
    pw.add_argument("--source", default="weak")
    pw.add_argument("--key-col", default="tracklet_key")
    pw.add_argument("--token-col", default="weak_tokens")
    pw.add_argument("--confidence-col", default="confidence")
    pg = sub.add_parser("generate-weak-labels")
    pg.add_argument("--tracklets-root", required=True,
                    help="a tracklets.parquet file, one video tracklet dir, or a dataset tracklets root")
    pg.add_argument("--out", required=True, help="output CSV compatible with import-weak-labels")
    pg.add_argument("--videos-root", default=None,
                    help="optional source video root; if OpenCV is available, crop color tokens are added")
    pg.add_argument("--sample-frames", type=int, default=3,
                    help="number of frames sampled per tracklet for optional crop colors")
    pg.add_argument("--min-tracklet-dets", type=int, default=1,
                    help="skip tracklets shorter than this many detections")
    pg.add_argument("--max-tracklets", type=int, default=None,
                    help="optional cap for quick smoke runs")
    pg.add_argument("--no-crop-colors", action="store_true",
                    help="disable video crop color tokens; bbox/time tokens are still emitted")
    pg.add_argument("--crop-margin", type=float, default=0.05,
                    help="fractional bbox margin for crop color sampling")
    pr = sub.add_parser("weak-resolve")
    pr.add_argument("--dataset", required=True, choices=list(CARDDIRS))
    pr.add_argument("--source", default="weak")
    pr.add_argument("--embedding-role", default="resolve", choices=["resolve", "match"])
    pr.add_argument("--min-dets", type=int, default=1)
    pr.add_argument("--visual-top-k", type=int, default=30)
    pr.add_argument("--edge-threshold", type=float, default=0.64)
    pr.add_argument("--max-token-df", type=int, default=120)
    pr.add_argument("--lp-iterations", type=int, default=8)
    pr.add_argument("--max-component-size", type=int, default=80)
    pr.add_argument("--apply", dest="apply", action="store_true", default=True)
    pr.add_argument("--no-apply", dest="apply", action="store_false")
    pr.add_argument("--submit", default=None, help="also write a TA1 submission zip after applying")
    a = ap.parse_args()
    {"list-videos": cmd_list_videos, "score": cmd_score, "demo": cmd_demo,
     "selftest": cmd_selftest, "submit": cmd_submit,
     "generate-weak-labels": cmd_generate_weak_labels,
     "import-weak-labels": cmd_import_weak_labels, "weak-resolve": cmd_weak_resolve}[a.cmd](a)


if __name__ == "__main__":
    main()
