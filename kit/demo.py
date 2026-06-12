#!/usr/bin/env python
"""One-click REAL demo of the online gallery — streams a tracker's output through the live gallery + scores.

Unlike `example.py` (a fill-in-the-blank template you complete with your own pipeline), this runs the WHOLE
thing end to end on real data, drives the live `OnlineGallery` (DB + the in-process FAISS-equivalent index),
resolves, and prints the real IDF1/AssA. The DB is left populated so the Gallery view (http://localhost:4200)
shows real decisions / identities / crops immediately.

    docker compose run --rm app demo                       # MS02, one-click (inside the kit container)
    docker compose run --rm app demo --dataset ds1         # DS1 (on-network; see pipelines/ds1.yaml)
    python demo.py --dataset ds1                            # ...or from your env (kit on PYTHONPATH, DB reachable)

Datasets:
  ms02 — a shipped tracklet+embedding bundle (demo_data/ms02/), fast, offline. Sparse GT -> LEAD WITH AssA.
  ds1  — DS0001 (dense GT -> IDF1 trustworthy). The inputs come from pipelines/ds1.yaml, which selects EITHER
         a local `bundle:` OR MLflow `inputs:` (fetched, base SOLIDER).
         The yaml also carries the `gallery:` config (e.g. cannot_link) so a run is fully reproducible.

DS1 ingest is large and can take a while; progress is logged as `[demo +MM:SS] ...` with rate + ETA.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from online import OnlineGallery

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
DEMO_DATA = os.environ.get("DEMO_DATA", str(_HERE / "demo_data" / "ms02"))
DS1_PIPELINE = os.environ.get("DS1_PIPELINE", str(_HERE / "pipelines" / "ds1.yaml"))

# --- timestamped logger (elapsed since run start) so the slow DS1 ingest is observable ---
_T0 = [time.time()]


def _log(msg: str) -> None:
    el = time.time() - _T0[0]
    print(f"[demo +{int(el) // 60:02d}:{int(el) % 60:02d}] {msg}", flush=True)


def _resolve_path(p: str) -> Path:
    """Resolve a pipeline-yaml path: absolute as-is, else relative to the repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (_ROOT / pp)


# --------------------------------------------------------------------------------------------------------
# Sources — each returns a LIST of 8-tuples (video, camera, frames, boxes, pooled_emb, confs, object_type,
# det_ids), so run_demo knows the total up front and can show progress + ETA.
# --------------------------------------------------------------------------------------------------------
def _ms02_tracklets(data_dir: str):
    """The shipped MS02 bundle — one pooled embedding per within-camera track (exactly the shape your own
    `your_tracker()` (example.py) would yield)."""
    d = Path(data_dir)
    trk = pd.read_parquet(d / "tracklets.parquet")
    pooled = np.load(d / "pooled_emb.npy")
    seqs = np.load(d / "seqs.npy")
    emb_of = {int(s): pooled[i] for i, s in enumerate(seqs)}
    out = []
    for seq, g in trk.groupby("seq"):
        video = g["det_id"].iloc[0].split("::")[0]          # the video stem is the det_id prefix
        camera = g["camera"].iloc[0]
        frames = g["frame"].astype(int).tolist()
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy().tolist()
        det_ids = g["det_id"].tolist()                      # the bundle's globally-unique ids (pass them so
        out.append((video, camera, frames, boxes, emb_of[int(seq)], None, 0, det_ids))  # same-frame dets don't collide)
    return out


def _load_resolve_emb(local_path: str | None, mlflow_run: str | None):
    """Load the per-tracklet resolve embedding (osnet-xcam). LOCAL-first (fast, offline) then MLflow-fallback so
    a fresh clone without runs/decisions/ still reproduces: pulls artifact embeddings/osnet_xcam_emb_all.npy from
    the registered run."""
    if local_path:
        p = _resolve_path(local_path)
        if p.exists():
            _log(f"DS1: resolve emb from local {p}")
            return np.load(p)
        _log(f"DS1: resolve emb local {p} missing — falling back to MLflow")
    if mlflow_run:
        import mlflow
        _log(f"DS1: pulling resolve emb from MLflow run {mlflow_run[:12]} "
             f"({os.environ.get('MLFLOW_TRACKING_URI', '<MLFLOW_TRACKING_URI unset>')})")
        ap = mlflow.artifacts.download_artifacts(run_id=mlflow_run, artifact_path="embeddings/osnet_xcam_emb_all.npy")
        return np.load(ap)
    return None


def _ds1_from_bundle(pkl: str, resolve_emb_path: str | None = None, resolve_emb_mlflow: str | None = None):
    """A pickle of plain-kalman tracklets, each {video,cam,dids,frames,boxes,reds,t0,t1} where `reds` are the
    FT-SOLIDER PCA-64 per-detection vectors. Pool = mean(reds) L2-normed (== the matcher's `_pool`), 64-d
    already (NO re-reduce). Sorted by ingest order (t0, video, frame0).

    If `resolve_emb_path` is given, ALSO load a per-tracklet resolve embedding (osnet-xcam 512-d, index-aligned
    to the bundle's pre-sort order: emb[i] == tracklet i) and carry it through the sort so it stays aligned to
    the streamed order — the global-agglom resolve re-clusters on THESE (not the greedy SOLIDER match space)."""
    import pickle
    p = _resolve_path(pkl)
    _log(f"DS1: loading baseline bundle {p}  (plain-kalman tracklets + FT emb_red — the 0.531 reference)")
    with open(p, "rb") as f:
        tks = pickle.load(f)
    osn = _load_resolve_emb(resolve_emb_path, resolve_emb_mlflow)
    if osn is not None:
        if len(osn) != len(tks):
            raise SystemExit(f"[demo] resolve_emb has {len(osn)} rows but the bundle has {len(tks)} tracklets")
        _log(f"DS1: resolve embeddings {osn.shape} (osnet-xcam, for the global-agglom resolve)")
        for i, tk in enumerate(tks):
            tk["_remb"] = osn[i].astype(np.float32)
    tks = sorted(tks, key=lambda t: (t["t0"], t["video"], t["frames"][0]))    # the matcher ingest order
    out, remb = [], []
    for tk in tks:
        red = np.stack([np.asarray(r, np.float32) for r in tk["reds"]]).mean(0)
        red = (red / (np.linalg.norm(red) + 1e-9)).astype(np.float32)
        cam = tk.get("cam") or tk.get("camera")
        out.append((tk["video"], cam, [int(x) for x in tk["frames"]],
                    [np.asarray(b, float).tolist() for b in tk["boxes"]], red, None, 0, list(tk["dids"])))
        if osn is not None:
            remb.append(tk["_remb"])
    _log(f"DS1: {len(out)} tracklets loaded from the bundle")
    return out, (np.stack(remb) if remb else None)


def _ds1_from_mlflow(spec: dict):
    """Fetch the DS1 track + per-tracklet embed runs from MLflow (no recompute), join 1:1 on tracklet_key,
    and reduce the raw embeddings to the 64-d match space with the SDK reducer the yaml declares. On-network
    only: needs MLFLOW_TRACKING_URI + the DS1 extras (kit/requirements-ds1.txt)."""
    try:
        import mlflow
        from vlincs_sdk.harness.matching.clustering import reduce_embeddings
    except ImportError as e:
        raise SystemExit(
            f"[demo] DS1 (MLflow inputs) needs the on-network extras: `pip install -r requirements-ds1.txt` "
            f"(mlflow + vlincs-sdk[mlflow]) and a reachable MLFLOW_TRACKING_URI. ({e})")
    track_run = spec["inputs"]["track"]["mlflow_run"]
    embed_run = spec["inputs"]["embed"]["mlflow_run"]
    rc = spec.get("reduce", {})
    _log(f"DS1: fetching track={track_run[:12]} + embed={embed_run[:12]} from MLflow "
         f"({os.environ.get('MLFLOW_TRACKING_URI', '<MLFLOW_TRACKING_URI unset>')})")
    trk_root = Path(mlflow.artifacts.download_artifacts(run_id=track_run, artifact_path="tracklets"))
    emb_root = Path(mlflow.artifacts.download_artifacts(run_id=embed_run, artifact_path="embeddings"))
    # RESOLVE-space embed (osnet-xcam 512-d): a SECOND per-tracklet vector joined by the SAME tracklet_key,
    # stored in the gallery (role='resolve') and re-clustered by resolve_global(). Used as-is (no reduce).
    resolve_run = (spec.get("inputs", {}).get("resolve_embed") or {}).get("mlflow_run")
    resolve_vec_of: dict = {}
    if resolve_run:
        rroot = Path(mlflow.artifacts.download_artifacts(run_id=resolve_run, artifact_path="embeddings"))
        for vd in sorted(p for p in rroot.iterdir() if p.is_dir()):
            rz = np.load(vd / "embeddings.npz", allow_pickle=True)
            rvecs, rtids = rz["vectors"], rz["track_ids"]   # read each array ONCE — z[...] re-decodes the WHOLE
            for i, t in enumerate(rtids):                    # array per access; indexing it in-loop is quadratic
                resolve_vec_of[str(t)] = rvecs[i]
        _log(f"DS1: resolve embed {resolve_run[:12]} loaded ({len(resolve_vec_of)} tracklets, osnet-xcam 512-d)")
    _log("DS1: artifacts fetched; joining tracklets <-> per-tracklet embeddings per video")
    vdirs = sorted(p for p in emb_root.iterdir() if p.is_dir())
    rows = []
    for vi, vdir in enumerate(vdirs, 1):
        stem = vdir.name
        pq = trk_root / stem / "tracklets.parquet"
        if not pq.exists():
            continue
        z = np.load(vdir / "embeddings.npz", allow_pickle=True)
        vecs, tids = z["vectors"], z["track_ids"]           # read each npz array ONCE (z[...] re-decodes it)
        vec_of = {str(t): vecs[i] for i, t in enumerate(tids)}
        camera = stem.split("_")[3]
        df = pd.read_parquet(pq)
        for tkey, g in df.groupby("tracklet_key"):
            v = vec_of.get(str(tkey))
            if v is None:
                continue
            g = g.sort_values("frame_idx")
            frames = g["frame_idx"].astype(int).tolist()
            boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(float).tolist()
            confs = g["score"].astype(float).tolist()
            ltid, cls = int(g["local_track_id"].iloc[0]), int(g["coco_cls"].iloc[0])
            otype = 0 if cls == 0 else 3                    # coco 0=person -> 0; else -> vehicle(3)
            det_ids = [f"{stem}::{camera}:{int(fr)}:{ltid}:{cls}" for fr in frames]
            rows.append((stem, camera, frames, boxes, confs, otype, det_ids, v, resolve_vec_of.get(str(tkey))))
        _log(f"DS1: read {vi}/{len(vdirs)} videos ({stem.split('_')[3]}: {len(df)} dets) — {len(rows)} tracklets so far")
    if not rows:
        raise SystemExit("[demo] DS1: no tracklets joined — check the track/embed run ids in the YAML.")
    embs = np.stack([r[7] for r in rows]).astype(np.float32)
    method = (rc.get("method") or "none").lower()
    if method == "none":                                    # embed is already at the match dim (FT emb_red)
        _log(f"DS1: reduce=none — embeddings already {embs.shape[1]}-d; pushing as-is")
        reduced = embs
    else:
        _log(f"DS1: reducing {len(rows)} tracklet embeddings -> {int(rc.get('dim', 64))}-d ({method})")
        reduced, info = reduce_embeddings(
            embs, method=method, n_components=int(rc.get("dim", 64)),
            random_state=int(rc.get("random_state", 42)), metric=rc.get("metric", "euclidean"),
            umap_n_neighbors=int(rc.get("umap_n_neighbors") or 15), umap_min_dist=float(rc.get("umap_min_dist") or 0.0))
        _log(f"DS1: reduce -> {info}")
    resolve_arr = None
    if resolve_run:
        if any(r[8] is None for r in rows):
            n_missing = sum(r[8] is None for r in rows)
            raise SystemExit(f"[demo] resolve embed {resolve_run[:12]} is missing {n_missing}/{len(rows)} "
                             f"tracklet_keys — the resolve embed run must cover every joined tracklet")
        resolve_arr = np.stack([r[8] for r in rows]).astype(np.float32)
        _log(f"DS1: resolve embeddings aligned to stream order -> {resolve_arr.shape}")
    tracklets = [(r[0], r[1], r[2], r[3], red, r[4], r[5], r[6]) for r, red in zip(rows, reduced)]
    return tracklets, resolve_arr


def _ds1_source(pipeline_yaml: str = DS1_PIPELINE):
    """Return (tracklets, gallery_cfg, resolve_emb) for DS1: the source is a local `bundle:`
    or MLflow `inputs:`, whichever pipelines/ds1.yaml declares. `gallery_cfg` is the yaml's `gallery:` block
    (cannot_link, tau, resolve, ...). `resolve_emb` is the per-tracklet osnet-xcam matrix (aligned to the
    tracklet order) when `gallery.resolve_emb` is set for the global-agglom resolve, else None."""
    import yaml
    spec = yaml.safe_load(Path(pipeline_yaml).read_text())
    gcfg = spec.get("gallery", {}) or {}
    if spec.get("bundle"):
        trk, remb = _ds1_from_bundle(spec["bundle"], gcfg.get("resolve_emb"), gcfg.get("resolve_emb_mlflow"))
        return trk, gcfg, remb
    trk, remb = _ds1_from_mlflow(spec)
    return trk, gcfg, remb


# --------------------------------------------------------------------------------------------------------
_GALLERY_KEYS = ("tau", "merge_tau", "match_mode", "max_reps", "coherence_floor", "admit_tau",
                 "tracklet_coh_min", "max_speed", "sim_window_ms", "same_box_iou")


def run_demo(dataset: str = "ms02", resolve_every: int | None = None, submit: str | None = None,
             data_dir: str = DEMO_DATA, cannot_link: bool | None = None) -> dict:
    """Stream a tracker's output through the live OnlineGallery, resolve, score. Returns g.score().

    cannot_link enforces the physical vetoes (same_frame intra-camera, simultaneity/travel cross-camera).
    Precedence: explicit arg/CLI > the dataset yaml's `gallery.cannot_link` > default True. DS1 runs
    cannot_link=False (appearance-only — its cameras have overlapping FOVs)."""
    _T0[0] = time.time()
    gcfg: dict = {}
    resolve_emb = None
    if dataset == "ds1":
        tracklets, gcfg, resolve_emb = _ds1_source()
        src_desc = f"{len(tracklets)} DS1 tracklets"
    else:
        tracklets = _ms02_tracklets(data_dir)
        # MS02 config: tau=0.85 (the SOLIDER-PCA64 same-person cosine scale), cannot_link=False
        # (appearance-only; marginal on a within-camera set), resolve=auto (mean_cams/gid=1.0 -> the
        # global merge is SKIPPED; it would only collapse distinct people).
        gcfg = {"tau": 0.85, "cannot_link": False, "resolve": "auto"}
        src_desc = f"{len(tracklets)} MS02 tracklets (shipped bundle)"

    cl = cannot_link if cannot_link is not None else bool(gcfg.get("cannot_link", True))
    re_every = resolve_every if resolve_every is not None else (500 if dataset == "ds1" else 100)
    # resolve GATE: 'on' (always) | 'off' (never) | 'auto' (only when the gallery has cross-camera structure
    # to merge, i.e. mean_cams/gid >= xcam_gate). Global resolve() HELPS cross-camera fragmentation but HURTS a
    # within-camera set — MS02 sits at mean_cams/gid=1.0, so a merge can only collapse two distinct people. DS1
    # pins resolve:on in its yaml (its early resolves run below xcam_gate); MS02 (no yaml) defaults to 'auto'.
    _rv = gcfg.get("resolve", "auto")            # YAML parses `on`/`off` as booleans -> normalize them
    resolve_mode = {True: "on", False: "off"}.get(_rv, str(_rv).lower())
    xcam_gate = float(gcfg.get("xcam_gate", 1.2))
    okw = {k: gcfg[k] for k in _GALLERY_KEYS if k in gcfg}    # gallery knobs the yaml overrides (else defaults)
    _log(f"{dataset}: {src_desc} | cannot_link={cl} ({'vetoes ON' if cl else 'appearance-only'}) "
         f"| resolve={resolve_mode} | resolve_every={re_every}" + (f" | cfg={okw}" if okw else ""))

    g = OnlineGallery(dataset, truncate=True, cannot_link=cl, **okw)      # empty DB; loads camera geo + clock
    def _should_resolve():                                                # the cross-cam-gap gate
        if resolve_mode in ("on", "off"):
            return resolve_mode == "on"
        if resolve_mode == "global_agglom":
            return False                                                  # global re-cluster is a FINAL step, not periodic
        return g.m.camera_span_stats().get("mean_cameras_per_gid", 1.0) >= xcam_gate   # 'auto'
    total = len(tracklets)
    step = max(50, total // 25)                                           # ~25 progress lines over the run
    t_push = time.time()
    for n, (video, camera, frames, boxes, emb, confs, otype, det_ids) in enumerate(tracklets, 1):
        g.add_tracklet(video, camera, frames, boxes, emb, confs=confs, object_type=otype, det_ids=det_ids,
                       resolve_emb=(resolve_emb[n - 1] if resolve_emb is not None else None))  # stored role='resolve'
        if n % re_every == 0 and _should_resolve():
            g.resolve()                                                  # periodic consolidation (cross-cam gated)
        if n % step == 0 or n == total:
            rate = n / (time.time() - t_push + 1e-9)
            eta = (total - n) / rate if rate else 0
            _log(f"ingest {n}/{total} ({100 * n // total}%) | {rate:.1f} trk/s | ETA {int(eta)}s "
                 f"| live identities ~{g.m.next_gid - 1}")
    if resolve_mode == "global_agglom":
        if resolve_emb is None:
            raise SystemExit("[demo] resolve=global_agglom needs a resolve embed (inputs.resolve_embed in the yaml); "
                             "the gallery stores it per tracklet (role='resolve') and re-clusters on it")
        theta = float(gcfg.get("resolve_theta", 0.02))
        top_k = int(gcfg.get("resolve_top_k", 15)); min_dets = int(gcfg.get("resolve_min_dets", 20))
        _log(f"global-agglom resolve (PROTOCOL §13.3): re-cluster the gallery's STORED role='resolve' vecs from the "
             f"DB (theta={theta}, top_k={top_k}, min_dets={min_dets}) — recovers greedy over-split AND over-merge...")
        info = g.resolve_global(theta, top_k=top_k, min_dets=min_dets)
        _log(f"global-agglom: {info['n_clustered']}/{info['n_tracklets']} tracklets re-partitioned into "
             f"{info['clusters']} identities (+{info['n_singleton']} short-tracklet singletons); scoring (reid_hota)...")
    elif _should_resolve():
        _log("final resolve() (cross-camera structure present) + scoring (canonical reid_hota)...")
        g.resolve()
    else:
        _log(f"resolve SKIPPED (mode={resolve_mode}: within-camera, mean_cams/gid < {xcam_gate} -> a merge could "
             f"only collapse distinct people); scoring (canonical reid_hota)...")
    score = g.score()
    _log(f"DONE: {dataset} -> {score}")
    if dataset == "ms02":
        _log("MS02 GT is sparse -> LEAD WITH AssA (IDF1 is deflated by real-but-unannotated people).")
    else:
        _log("DS1 has dense GT -> IDF1 is the trustworthy number here.")
    _log("DB populated. With the stack up (./demo.sh, or docker compose up -d): "
         "Gallery view http://localhost:4200 | pgAdmin http://localhost:5050")
    if submit:
        g.export_submission(submit)
        _log(f"wrote submission {submit}")
    g.close()
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ms02", choices=["ms02", "ds1"],
                    help="ms02 = shipped bundle (fast, offline); ds1 = pipelines/ds1.yaml (bundle or MLflow)")
    ap.add_argument("--resolve-every", type=int, default=None, help="periodic resolve every N tracklets (default ds1=500, ms02=100)")
    ap.add_argument("--submit", default=None, help="also write a TA1 submission zip here")
    ap.add_argument("--cannot-link", dest="cannot_link", action="store_true", default=None,
                    help="force the same_frame/simultaneity/travel vetoes ON (overrides the yaml)")
    ap.add_argument("--no-cannot-link", dest="cannot_link", action="store_false",
                    help="force vetoes OFF / appearance-only (overrides the yaml)")
    a = ap.parse_args()
    run_demo(a.dataset, a.resolve_every, a.submit, cannot_link=a.cannot_link)


if __name__ == "__main__":
    main()
