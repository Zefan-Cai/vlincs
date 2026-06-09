"""Canonical TA1 scorer for the online-gallery PoC.

Wraps reid_hota with the ONE frozen config used everywhere in VLINCS TA1 scoring, plus the
MS02 sparse-GT firewall (lead with AssA; reference_contains_dense_annotations=False). Every
IDF1/HOTA number cited for the PoC must come through this module (or the canonical
create_submission.py CLI, which is bit-identical). See PROTOCOL.md section 8.

reid_hota note: the source tree at ~/git/reid_hota is 0.3.2; the working-venv wheel is 0.3.5.
The (HOTAReIDEvaluator, HOTAConfig, get_global_hota_data) API is stable across them — but pin
whichever you import in any provenance log.

MS02 metric trap (empirically confirmed, see _selftest): MS02 GT is sparse hand-curated
(~1.5 det/frame), so a hot detector's real-but-unannotated people would score as phantom FPs.
Under the canonical dense=False config the scorer routes pred-IDs that never match any GT id
into an UnmatchedFP bucket instead of penalising IDF1. So on MS02 we lead with AssA
(phantom-immune) and report IDF1(dense=False) + UnmatchedFP; DS1's dense GT is the fair test.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# reid_hota required columns for similarity_metric="iou"
_REQ_COLS = ["frame", "id", "x1", "y1", "x2", "y2", "object_type"]

# GT roots come from the central path config (vlincs_gallery.paths) so every module agrees and they
# resolve wherever the datastore is mounted (host /mnt/...; kit container /data). MS02 lives in the
# -selected tree, DS1 in the Box export.
from vlincs_gallery.paths import MS02_GT_DIR as _MS02_DIR, DS1_GT_DIRS as _DS1_DIRS
_CAM_RE = re.compile(r"(MCAM\d+)")

DatasetKind = Literal["ms02", "ds1"]


def canonical_config(dense: bool = False):
    """The one frozen scorer config. dense=False is the MS02 phantom-FP firewall.

    Never use dense=True on MS02. Never use per_frame/per_video id alignment.
    """
    from reid_hota import HOTAConfig

    return HOTAConfig(
        id_alignment_method="global",
        similarity_metric="iou",
        reference_contains_dense_annotations=dense,
    )


@dataclass
class Metrics:
    """Reduced (threshold-averaged) HOTA metrics + the raw per-threshold vectors."""

    idf1: float
    hota: float
    assa: float
    deta: float
    detre: float
    detpr: float
    unmatched_fp: int
    raw: dict

    def headline(self, dataset: DatasetKind) -> str:
        if dataset == "ms02":
            # sparse hand-curated GT -> association quality is the trustworthy signal
            return (
                f"AssA={self.assa:.4f}  IDF1(dense=False)={self.idf1:.4f}  "
                f"UnmatchedFP={self.unmatched_fp}  (MS02: lead with AssA)"
            )
        return (
            f"IDF1={self.idf1:.4f}  HOTA={self.hota:.4f}  AssA={self.assa:.4f}  "
            f"DetRe={self.detre:.4f}  DetPr={self.detpr:.4f}"
        )


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns reid_hota needs for IoU scoring (others are ignored/dropped)."""
    return df[[c for c in _REQ_COLS if c in df.columns]].copy()


def evaluate(
    ref_dfs: dict[str, pd.DataFrame],
    comp_dfs: dict[str, pd.DataFrame],
    *,
    dense: bool = False,
    n_workers: int = 8,
) -> Metrics:
    """Score predictions (comp) against ground truth (ref). Both dict[video_key -> df].

    video_key must match between ref and comp (e.g. "MCAM310"). Threshold-averaged metrics are
    returned along with the full per-IoU-threshold vectors in `.raw`.
    """
    from reid_hota import HOTAReIDEvaluator

    cfg = canonical_config(dense=dense)
    ev = HOTAReIDEvaluator(n_workers=n_workers, config=cfg)
    ref = {k: _coerce(v) for k, v in ref_dfs.items()}
    comp = {k: _coerce(v) for k, v in comp_dfs.items()}
    ev.evaluate(ref, comp)
    d = ev.get_global_hota_data()

    def mean(key: str) -> float:
        v = d[key]
        return float(np.mean(v)) if hasattr(v, "__len__") else float(v)

    return Metrics(
        idf1=mean("IDF1"),
        hota=mean("HOTA"),
        assa=mean("AssA"),
        deta=mean("DetA"),
        detre=mean("DetRe"),
        detpr=mean("DetPr"),
        unmatched_fp=int(d.get("UnmatchedFP", 0)),
        raw={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()},
    )


def _load_gt_dir(dir_path: str, pattern: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for f in sorted(glob.glob(str(Path(dir_path) / pattern))):
        mt = _CAM_RE.search(Path(f).name)
        if not mt:
            continue
        out[mt.group(1)] = pd.read_parquet(f)
    return out


def load_ms02_gt() -> dict[str, pd.DataFrame]:
    """MS02 GT (v.1.7.3), keyed by camera (MCAM310, MCAM318)."""
    return _load_gt_dir(_MS02_DIR, "*_v.1.7.3.parquet")


def load_ds1_gt(card: str = "Tc6") -> dict[str, pd.DataFrame]:
    """DS1 GT (v1.7.2) for a test card, keyed by camera. Tc6 has MCAM00/03/04/05/06/08."""
    return _load_gt_dir(_DS1_DIRS[card], "*_v1.7.2.parquet")


def load_ds1_gt_by_video(cards: tuple[str, ...] = ("Tc6", "Tc8")) -> dict[str, pd.DataFrame]:
    """DS1 GT keyed by VIDEO stem (= GT filename minus '_v1.7.2.parquet').

    The combined 10-video DS1 frame (the leaderboard frame) reuses camera NAMES across cards
    (MCAM00 appears in both Tc6 and Tc8). Keying by camera would conflate those two distinct
    video sessions into one stream with colliding frame indices. Key by video instead so the
    global ID alignment runs across all 10 videos honestly. Stems match the bundle `video` col.
    """
    out: dict[str, pd.DataFrame] = {}
    for card in cards:
        for f in sorted(glob.glob(str(Path(_DS1_DIRS[card]) / "*_v1.7.2.parquet"))):
            out[Path(f).name[: -len("_v1.7.2.parquet")]] = pd.read_parquet(f)
    return out


def restrict_to_gt_matched(comp_dfs: dict[str, pd.DataFrame],
                           ref_dfs: dict[str, pd.DataFrame],
                           iou_thr: float = 0.5) -> dict[str, pd.DataFrame]:
    """Keep only comp detections that IoU-match a GT box at the same frame.

    On sparse-GT datasets (MS02) a hot detector emits many real-but-unannotated people that fold into
    matched ids and confound IDF1/AssA (the residual leak the dense=False firewall can't fully stop).
    Restricting comp to GT-overlapping detections isolates pure ASSOCIATION quality on the annotated
    people. See PROTOCOL §8 / reference_ms02_eval_firewall. Use as a SECONDARY clean signal on MS02.
    """
    out: dict[str, pd.DataFrame] = {}
    for cam, c in comp_dfs.items():
        g = ref_dfs.get(cam)
        if g is None or len(c) == 0:
            out[cam] = c.iloc[0:0]
            continue
        gt_by_frame = {int(f): sub[["x1", "y1", "x2", "y2"]].to_numpy(float)
                       for f, sub in g.groupby("frame")}
        cf = c["frame"].to_numpy(int)
        cb = c[["x1", "y1", "x2", "y2"]].to_numpy(float)
        keep = np.zeros(len(c), bool)
        for i in range(len(c)):
            gb = gt_by_frame.get(int(cf[i]))
            if gb is None:
                continue
            ix1 = np.maximum(cb[i, 0], gb[:, 0]); iy1 = np.maximum(cb[i, 1], gb[:, 1])
            ix2 = np.minimum(cb[i, 2], gb[:, 2]); iy2 = np.minimum(cb[i, 3], gb[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            area_c = (cb[i, 2] - cb[i, 0]) * (cb[i, 3] - cb[i, 1])
            area_g = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            iou = inter / np.maximum(area_c + area_g - inter, 1e-9)
            if iou.size and iou.max() >= iou_thr:
                keep[i] = True
        out[cam] = c[keep]
    return out


def _selftest() -> None:
    """Reproduce the MS02 sparse-GT firewall through this harness.

    Perfect predictions + injected phantom (non-overlapping, uniquely-id'd) detections must
    leave IDF1=1.0 and AssA=1.0 under dense=False (phantoms -> UnmatchedFP), but tank under
    dense=True. This both validates the harness wiring and re-confirms the firewall on real data.
    """
    gt = load_ms02_gt()
    assert gt, "no MS02 GT found"
    cam, gdf = next(iter(gt.items()))
    print(f"[selftest] MS02 {cam}: {len(gdf)} GT rows, {gdf['id'].nunique()} ids")

    pred = gdf.copy()  # perfect prediction
    n_phantom = 6000
    fmax = int(gdf["frame"].max())
    phantom = pd.DataFrame(
        {
            "frame": np.random.randint(1, fmax + 1, n_phantom),
            "id": [f"PHANTOM_{i}" for i in range(n_phantom)],
            "x1": np.full(n_phantom, 5000.0),
            "y1": np.full(n_phantom, 5000.0),
            "x2": np.full(n_phantom, 5050.0),
            "y2": np.full(n_phantom, 5050.0),
            "object_type": "person",
        }
    )
    pred_phantom = pd.concat([pred, phantom], ignore_index=True)

    soft = evaluate({cam: gdf}, {cam: pred_phantom}, dense=False)
    hard = evaluate({cam: gdf}, {cam: pred_phantom}, dense=True)
    print(f"[selftest] dense=False : {soft.headline('ms02')}")
    print(
        f"[selftest] dense=True  : IDF1={hard.idf1:.4f}  AssA={hard.assa:.4f}  "
        f"UnmatchedFP={hard.unmatched_fp}"
    )
    ok = soft.idf1 > 0.99 and soft.assa > 0.99 and hard.idf1 < soft.idf1 - 0.05
    print(f"[selftest] FIREWALL {'CONFIRMED' if ok else 'NOT confirmed -- investigate'}")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        np.random.seed(0)
        _selftest()
    else:
        print("usage: python -m vlincs_gallery.eval.score --selftest")
