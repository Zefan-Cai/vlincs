"""Template: drive the online gallery from YOUR pipeline. Copy this, replace `your_tracker`.

The gallery is fed entirely through the Python API - no files. The recommended input is your TRACKER's
tracklets (one pooled embedding per within-camera track): the gallery does *cross-camera* identity, and
*within-camera* consolidation is your tracker's job. You push tracklets with their embeddings on whatever
cadence you like; you get a global id back; you resolve when you want; you score (ms02/ds1) and optionally
export a submission. The ONLY file produced is the submission.

    python example.py --dataset ds1        # runs YOUR pipeline below against the dataset's videos
"""
from __future__ import annotations
import argparse

from online import OnlineGallery, CARDDIRS   # kit-local


def your_tracker(dataset: str):
    """REPLACE THIS with your detector + tracker + embedder.

    Yield, per TRACKLET (a within-camera track your tracker has produced), in the order they complete:
        (video_stem, camera, [frame_idx, ...], [[x1, y1, x2, y2], ...], embedding)
    - video_stem / camera: from `cli.py list-videos --dataset <ds>` (camera must be the MCAMxx name)
    - frames / boxes: the per-detection frame indices and boxes of the tracklet (parallel lists)
    - embedding: your appearance vector for the tracklet - a pooled (D,) vector, OR a per-detection
      (n, D) array (it's mean-pooled for you). Any fixed dimension D.

    The stub below emits nothing real - it just documents the contract. Wire in your own loop:

        for video in list_videos(dataset):
            for tracklet in your_tracker_over(video):         # your detector + tracker
                frames = [d.frame for d in tracklet.dets]
                boxes  = [d.box   for d in tracklet.dets]
                emb    = your_embedder(tracklet.crops)        # pooled (D,) or per-det (n, D)
                yield video.stem, camera_of(video), frames, boxes, emb

    (No tracker? Per-DETECTION streaming is also supported - g.add_detection(video, camera, frame, box,
    emb) - but without a tracker's within-camera consolidation it over-fragments. Push tracklets if you can.)
    """
    raise NotImplementedError("Replace your_tracker() with your detector+tracker+embedder.")
    yield  # pragma: no cover


def main():
    """Run YOUR pipeline (``your_tracker``) over the dataset's videos.

    Pushes each tracklet to the gallery, resolves on a fixed cadence, scores from the live DB, and
    optionally exports a submission. Copy this file and replace ``your_tracker`` with your own.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CARDDIRS))
    ap.add_argument("--resolve-every", type=int, default=200, help="periodic resolve every N tracklets")
    ap.add_argument("--submit", default=None, help="write a TA1 submission zip here when done")
    a = ap.parse_args()

    g = OnlineGallery(a.dataset)                # connects to the EMPTY db; loads camera geo from the dataset
    pushed = 0
    for video, camera, frames, boxes, emb in your_tracker(a.dataset):
        g.add_tracklet(video, camera, frames, boxes, emb)   # match / expand / do-nothing -> gid, live
        pushed += 1
        if pushed % a.resolve_every == 0:
            g.resolve()                         # periodic resolve (consolidation) - your cadence
    g.resolve()                                 # final resolve
    print(g.score())                            # IDF1 from the live DB (ms02/ds1; ds2 -> None)
    if a.submit:
        g.export_submission(a.submit)           # the only file written
    g.close()


if __name__ == "__main__":
    main()
