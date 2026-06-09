#!/usr/bin/env python
"""One-click REAL demo of the online gallery — MS02, no weights, no pipeline to write.

Unlike `example.py` (a fill-in-the-blank template you complete with your own pipeline), this runs the WHOLE
thing end to end on real data. It loads a precomputed MS02 tracklet+embedding bundle (shipped in
`demo_data/ms02/` — it stands in for *a tracker's output*, since the kit ships no detector/embedder),
streams it through the live `Gallery` (DB + the in-process FAISS-equivalent index), resolves, and prints
the real IDF1/AssA. The DB is left populated, so the Gallery view (http://localhost:4200) shows real
decisions / identities / crops immediately.

    docker compose run --rm app demo                       # one-click (inside the kit container)
    # ...or from your own env with the kit on PYTHONPATH + the DB reachable:
    python demo.py

MS02 is used for speed (2 cameras, 380 tracklets). Its GT is sparse (~1.5 det/frame), so **lead with
AssA** — IDF1 is deflated by real-but-unannotated people. This is the honest as-delivered behavior; swap
in your own detector/tracker/embedder via `example.py` to beat it.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from online import OnlineGallery

_HERE = Path(__file__).resolve().parent
DEMO_DATA = os.environ.get("DEMO_DATA", str(_HERE / "demo_data" / "ms02"))


def _ms02_tracklets(data_dir: str):
    """Yield (video, camera, frames, boxes, pooled_embedding) per tracklet from the shipped bundle.

    This is exactly the shape your own `your_tracker()` (see example.py) would yield — one pooled
    embedding per within-camera track — so the demo drives the gallery through the same public API.
    """
    d = Path(data_dir)
    trk = pd.read_parquet(d / "tracklets.parquet")
    pooled = np.load(d / "pooled_emb.npy")
    seqs = np.load(d / "seqs.npy")
    emb_of = {int(s): pooled[i] for i, s in enumerate(seqs)}
    for seq, g in trk.groupby("seq"):
        video = g["det_id"].iloc[0].split("::")[0]      # the video stem is the det_id prefix
        camera = g["camera"].iloc[0]
        frames = g["frame"].astype(int).tolist()
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy().tolist()
        det_ids = g["det_id"].tolist()                  # the bundle's globally-unique ids — pass them so
        yield video, camera, frames, boxes, emb_of[int(seq)], det_ids   # same-frame dets across tracklets don't collide


def run_demo(dataset: str = "ms02", resolve_every: int = 100, submit: str | None = None,
             data_dir: str = DEMO_DATA, cannot_link: bool = True) -> dict:
    """Load the shipped bundle, stream it through the live OnlineGallery, resolve, score. Returns g.score().

    cannot_link (default ON) enforces the physical vetoes: same_frame (two spatially-distinct boxes in ONE
    (video,frame) can't share an id — the intra-camera case), plus the cross-camera simultaneity/travel
    vetoes (e.g. the MS02 cameras are 54 m apart, so a same-instant cross-camera match is impossible).
    Pass cannot_link=False to reproduce the old appearance-only config."""
    print(f"[demo] loading the shipped {dataset} bundle from {data_dir} (real SOLIDER tracklets — a tracker's output)")
    print(f"[demo] cannot_link={cannot_link} (cross-camera time/geo vetoes {'ON' if cannot_link else 'OFF — appearance only'})")
    g = OnlineGallery(dataset, truncate=True, cannot_link=cannot_link)   # empty DB; loads camera geo from the dataset
    n = 0
    for video, camera, frames, boxes, emb, det_ids in _ms02_tracklets(data_dir):
        g.add_tracklet(video, camera, frames, boxes, emb, det_ids=det_ids)   # match/expand/do-nothing -> gid, live
        n += 1
        if n % max(1, resolve_every) == 0:
            g.resolve()                                 # periodic consolidation
    g.resolve()                                         # final consolidation
    score = g.score()
    print(f"[demo] pushed {n} real MS02 tracklets across 2 cameras -> {score}")
    print("[demo] MS02 GT is sparse -> LEAD WITH AssA (IDF1 is deflated by real-but-unannotated people).")
    print("[demo] DB populated (it persists in the `db` service even though this one-shot container exits).")
    print("[demo] With the stack up (use `./demo.sh`, or `docker compose up -d`) explore it live:")
    print("[demo]   Gallery view http://localhost:4200  |  pgAdmin http://localhost:5050")
    if submit:
        g.export_submission(submit)
        print(f"[demo] wrote submission {submit}")
    g.close()
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ms02", choices=["ms02"], help="the demo ships MS02 (fast)")
    ap.add_argument("--resolve-every", type=int, default=100, help="periodic resolve every N tracklets")
    ap.add_argument("--submit", default=None, help="also write a TA1 submission zip here")
    ap.add_argument("--no-cannot-link", dest="cannot_link", action="store_false",
                    help="disable the same_frame/simultaneity/travel vetoes (appearance-only; vetoes are ON by default)")
    a = ap.parse_args()
    run_demo(a.dataset, a.resolve_every, a.submit, cannot_link=a.cannot_link)


if __name__ == "__main__":
    main()
