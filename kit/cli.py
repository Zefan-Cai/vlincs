#!/usr/bin/env python
"""Kit CLI — the few commands a colleague runs around their own pipeline.

  list-videos --dataset ds1   the video stems + camera names to run YOUR pipeline on (+ GT availability)
  demo                        one-click REAL end-to-end run on MS02 (shipped real tracklets+embeddings)
  score       --dataset ds1   IDF1 (canonical reid_hota) from whatever is currently in the gallery DB
  submit      --dataset ds1 --out out.zip    export a TA1 submission from the DB (the only file the kit writes)
  selftest    --dataset ms02  wiring check on RANDOM embeddings (proves the pipeline runs; not a real score)

The real work — pushing your tracklets/embeddings — is the Gallery API (see example.py), driven from
your own environment on your own cadence. This CLI is just the bookends.
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path

from online import OnlineGallery, CARDDIRS, HAS_GT, DATA   # kit-local

# Reference scores for context, NOT a hard target. The online ref is what this kit's reference config
# scores online (no prior knowledge); the supervised ceiling (ds1) is the batch funnel trained on DS1 GT.
ONLINE_REF = {
    "ms02": "AssA ~0.70 (lead AssA — sparse GT)",
    "ds1": "IDF1 ~0.53 online ref (supervised ceiling 0.69)",
    "ds2": "0.49 (leaderboard)",
}


def _videos(dataset: str):
    out = []
    for d in CARDDIRS.get(dataset, []):
        for mp4 in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            stem = Path(mp4).stem
            cam = next((p for p in stem.split("_") if p.startswith("MCAM")), "CAM")
            out.append((stem, cam, mp4))
    return out


def cmd_list_videos(a):
    vids = _videos(a.dataset)
    gt = "yes (local IDF1)" if HAS_GT.get(a.dataset) else "NONE (leaderboard only — no local IDF1)"
    print(f"[{a.dataset}] {len(vids)} videos  |  ground truth: {gt}  |  data root: {DATA}")
    print(f"{'video stem':52} {'camera':8} path")
    for stem, cam, mp4 in vids:
        print(f"{stem:52} {cam:8} {mp4}")
    print("\nRun YOUR detector+tracker+embedder on these, push via the Gallery API (see example.py).")


def cmd_score(a):
    g = OnlineGallery(a.dataset, truncate=False)
    print(f"[score] {g.score()}   (this kit's online ref: {ONLINE_REF.get(a.dataset, '?')})")
    g.close()


def cmd_demo(a):
    # the REAL one-click demo: ships real MS02 tracklets+embeddings, drives the live gallery, scores.
    from demo import run_demo
    run_demo("ms02", resolve_every=100, submit=a.submit, cannot_link=a.cannot_link)


def cmd_selftest(a):
    # wiring check only — RANDOM embeddings, not a real number (use `demo` for the real MS02 run).
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
    print("[selftest] (random embeddings — proves the pipeline runs end-to-end, not a real number; use `demo`)")
    g.close()


def cmd_submit(a):
    g = OnlineGallery(a.dataset, truncate=False)
    g.export_submission(a.out)
    g.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list-videos", "score", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=list(CARDDIRS))
    pdemo = sub.add_parser("demo")                       # real one-click MS02 demo — no --dataset needed
    pdemo.add_argument("--submit", default=None, help="also write a TA1 submission zip here")
    pdemo.add_argument("--no-cannot-link", dest="cannot_link", action="store_false",
                       help="disable the same_frame/simultaneity/travel vetoes (appearance-only; vetoes are ON by default)")
    ps = sub.add_parser("submit")
    ps.add_argument("--dataset", required=True, choices=list(CARDDIRS))
    ps.add_argument("--out", default="submission.zip")
    a = ap.parse_args()
    {"list-videos": cmd_list_videos, "score": cmd_score, "demo": cmd_demo,
     "selftest": cmd_selftest, "submit": cmd_submit}[a.cmd](a)


if __name__ == "__main__":
    main()
