# vlincs_gallery

![ingest · decide · store · resolve](assets/architecture.svg)

Online, revisable, **retrieval-based** identity assignment ("tracking-by-retrieval" / online MTMC) for
VLINCS TA1. Instead of the lossy batch funnel (`detect → track → pool → UMAP → HDBSCAN → GNN → split →
merge`, every stage an irreversible compression), each tracklet is matched **as it arrives** against a
live, queryable, revisable gallery of identities backed by **pgvector** (durable system-of-record + ANN)
and an in-process **FAISS-equivalent** exemplar index (the hot match space).

You bring a detector + tracker + embedder; the gallery does the *cross-camera* identity assignment, with a
visualization (`viz`) of how/when/why every identity is what it is.

- **Run it / drive it from your pipeline** → [`kit/README.md`](kit/README.md) (the one-click Docker demo + the `OnlineGallery` API).
- Deeper design notes (rationale, eval protocol, phased gates) live in `PROTOCOL.md` — an internal working doc, not committed to this repo.

## How it works

Your pipeline pushes tracklets (one pooled appearance embedding per within-camera track). For each, the
gallery makes exactly one **decision**, and periodically **re-resolves** the whole set:

```
your detector+tracker+embedder ─► tracklets+embeddings ─► [ match / expand / do-nothing ] ──┐
                                                                                            │ periodic
                                              global IDs + IDF1 + decision viz ◄── resolve() ┘  (merge)
```

**Per-tracklet decision** (one per ingest):

| decision | when |
|---|---|
| **match** | cosine to an existing identity ≥ `tau` (and the vetoes allow it) → join that identity |
| **expand** | best candidate `< tau` → spawn a new identity (seed its bank) |
| **do-nothing** (quarantine) | the tracklet's own self-coherence `< tracklet_coh_min` → give it a solo id, never admit it to a matchable bank (an ID-switch / mixed-people tracklet can't seed or poison an identity) |

**Bank admission** (the appearance "memory"): a matched tracklet's pooled vector is added to the identity's
**exemplar bank** only if it adds diversity — at most `admit_tau`-similar to existing exemplars, not farther
than `coherence_floor` from them, and the bank isn't at `max_reps`. The identity centroid is a
confidence-weighted EMA of its exemplars.

**Periodic resolve()** (re-resolution): consolidates over-split identities whose exemplar-centroid cosine
has reached `merge_tau`, subject to the cannot-link vetoes. Each merge is recorded with the cosine that
triggered it (the "why"), the ingest step (the "when"), and the pair (the "where").

**cannot-link vetoes** (`cannot_link=True` by default): block physically-impossible matches/merges —
`same_frame` (two spatially-distinct boxes in one `(video,frame)` can't be one person), `simultaneity`
(one person can't be in two non-overlapping cameras at the same instant), `travel` (a cross-camera jump
faster than `max_speed`). Camera geo/time come from the dataset's shipped extrinsics.

The gallery state is a **pure replayable fold** over an append-only event log (`decision_log` + `merges`),
so the viz can reconstruct the exact identity state as of any ingest step.

## Storage

- **Postgres + pgvector** (`pgvector/pgvector:pg16`) — the durable system-of-record AND the cannot-link
  query layer (haversine/time SQL). **One database per dataset** (`gallery_ms02` / `gallery_ds1` /
  `gallery_ds2`) so runs are isolated; `truncate=True` wipes one for a clean number.
  - `detections.embedding   vector(1024)` — raw embedder output, kept for audit/replay.
  - `detections.embedding_red vector(64)` — the reduced (warmup-PCA) **match space**; thresholds + ANN
    operate here. An **HNSW cosine index** (`vector_cosine_ops`) on `identity_reps.embedding_red` is the ANN.
- **In-process FAISS-equivalent** — exact inner-product cosine over the identity **exemplar bank**
  (`rep_mat`). This is the hot path the matcher scores against; pgvector mirrors it for the viz + durability.

The embedder is **yours** and **any dimension** — the index adapts on the first push. The gallery ships no
weights and no models; it matches on whatever vectors you push.

## Knobs (`OnlineGallery(...)` kwargs)

Defaults are the validated config. Tune `tau` first (to your embedding's cosine scale).

| knob | default | what it controls |
|---|---|---|
| `tau` | **0.60** | **match** threshold — cosine ≥ τ ⇒ match, else expand |
| `match_mode` | `"centroid"` | how a candidate is scored: `centroid` (cosine to the whole-bank mean; +0.06 IDF1 over `max` on DS1) / `max` (nearest exemplar) / `retrieval` (FAISS k-NN, needs `faiss-cpu`) |
| `merge_tau` | **0.35** | **resolve** threshold — identities whose exemplar centroids agree ≥ this are merged |
| `admit_tau` | **0.9** | bank redundancy cutoff — admit a new exemplar only if at most this similar to existing ones |
| `coherence_floor` | **0.4** | anti-accretion — reject a would-be exemplar farther than this from its bank (kills "matches-everything" attractors) |
| `tracklet_coh_min` | **0.0** | do-nothing/quarantine cutoff (off by default; only fires on per-detection input where self-coherence is computed) |
| `max_reps` | **16** | exemplar-bank cap per identity (merges may push a survivor over this; not re-capped) |
| `cannot_link` | **True** | enable the `same_frame` / `simultaneity` / `travel` vetoes (`False` = appearance-only, the old "best DS1 config") |
| `max_speed` | **3.0** m/s | travel veto — a cross-camera match implying ground speed above this is blocked |
| `sim_window_ms` | **200** | simultaneity slop — detections in two cameras within this window are "the same instant" |
| `same_box_iou` | **0.35** | same-frame veto — two boxes in one frame below this IoU are different people |
| `overlaps` | `None` | known overlapping-FOV camera pairs (suppress the simultaneity veto there) |
| `fps` | **30.0** | frames→ms for the absolute clock |
| `batch_commit` | **1** | DB commit cadence (tracklets) |
| `truncate` | **True** | wipe this dataset's DB before ingest (clean run) |

## Magic numbers & strings (documented constants)

- **Match space = `vector(64)`, raw = `vector(1024)`** (`online.py:_red_dim/_raw_dim`, `db/init.sql`). A
  pushed embedding is persisted into `embedding_red` iff its dim is 64, into `embedding` iff 1024; any other
  dim leaves both NULL (matcher + score still work; the embedding-projection viz is just empty).
- **Resolve cadence (demo)**: `resolve()` every **100** tracklets + a final resolve (`demo.py:resolve_every`).
  In a real stream, call it on your own cadence.
- **`det_id` format**: `"<video>::<camera>:<frame_idx>:<box_idx>"`, globally unique within a dataset DB.
  Auto-generated ids append `:t<tracklet_seq>` so two tracklets with a detection in the same `(video,frame)`
  never collide on the primary key.
- **dataset → DB name**: `ms02→gallery_ms02`, `ds1→gallery_ds1`, `ds2→gallery_ds2` (`vlincs_gallery/db.py`).
- **Scoring**: canonical `reid_hota` — **global ID alignment, IoU similarity, `dense=False`** (the takehome
  leaderboard config), keyed by **video** for DS1 (Tc6/Tc8 reuse camera names), by **camera** for MS02.
- **Ports** (`kit/docker-compose.yml`): db **55433**→5432, viz API **8077**, Angular UI **4200**, pgAdmin
  **5050**. DB creds `gallery`/`gallery`.
- **Data paths** — one place, [`vlincs_gallery/paths.py`](vlincs_gallery/paths.py). `DATA_ROOT` is the
  datastore mount (host `/mnt/datastore2_videolincs/data`, container `/data`); under it,
  `DATA = $DATA_ROOT/Box/VLINCS_Performer` (the canonical Box export — **the default data directory**) and
  `MS02_DATA = $DATA_ROOT/VLINCS_Performer-selected` (where the MS02 debug set still lives — it's bound to
  the `vlincs-baseline` repo, not in the Box export yet). `root_for_site('MS02')` → `-selected`, else Box.
- **`reid_hota`** is the public NIST scorer ([github.com/usnistgov/reid_hota](https://github.com/usnistgov/reid_hota),
  on PyPI) — the kit installs it straight from PyPI; no internal index, so a clean clone builds anywhere.
- **MS02 demo data** is two 5-minute (9000-frame @ 30fps) videos (MCAM310 + MCAM318); its GT is sparse
  (~1.5 det/frame), so on MS02 **lead with AssA**, not IDF1.

## Run it

```bash
cd kit
DATASET=ms02 docker compose up -d                 # Postgres(:55433) + viz(:8077) + UI(:4200) + pgAdmin(:5050)
docker compose run --rm app demo                  # one-click REAL run on shipped MS02 data → AssA ≈ 0.70
#   then explore the gallery at http://localhost:4200
```

Drive it from your own pipeline with the tiny `OnlineGallery` API (no intermediate files):

```python
from online import OnlineGallery                          # PYTHONPATH the kit, or run inside the app container
g = OnlineGallery("ds1")                                  # connects to the EMPTY per-dataset DB; loads camera geo
for video, camera, frames, boxes, embedding in your_tracker():
    gid = g.add_tracklet(video, camera, frames, boxes, embedding)   # match / expand / do-nothing → gid, live
    if your_cadence: g.resolve()                          # periodic consolidation, your cadence
print(g.score())                                          # IDF1/AssA from the live DB (ms02/ds1; ds2 → None)
g.export_submission("out.zip")                            # the ONLY file ever written (canonical TA1 zip)
```

Full usage, the Gallery-view walkthrough, and the deploy notes are in [`kit/README.md`](kit/README.md).

## Status

The online gallery is implemented end-to-end and **deployable** (clean clone → `docker compose build` →
`demo` → `viz`, verified). The MS02 shakeout works (demo: **AssA ≈ 0.70 / IDF1 ≈ 0.42**, sparse-GT artifact
on IDF1). **DS1 (dense GT) is the real test** — the honest streaming bar is the kit's online ref ≈ 0.53
IDF1; the batch funnel's GNN-on-DS1-GT supervised ceiling (0.69) is a different regime the online kit is
not trying to match.

## Modules / layout

| path | role |
|---|---|
| `vlincs_gallery/gallery.py` | **the one canonical matcher** (`IdentityGallery`) — match/expand/do-nothing + `consolidate()`. Pure, in-memory, numpy (+FAISS for `retrieval`). |
| `vlincs_gallery/paths.py` | **single source of truth for data locations** (DATA_ROOT / DATA / MS02_DATA / CARDDIRS). |
| `vlincs_gallery/db.py` · `db/init.sql` | pgvector system-of-record: schema, per-dataset DB, haversine veto fn. |
| `vlincs_gallery/clock.py` · `geo.py` | absolute clock + camera geo from shipped extrinsics (drives the time/geo vetoes). |
| `vlincs_gallery/eval/score.py` | canonical `reid_hota` scorer (`--selftest` confirms perfect input → IDF1 = 1.0). |
| `vlincs_gallery/viz/app.py` | FastAPI read API over the live DB (crops, decisions, identities, embedding projection). |
| `gallery-ui/` | the Angular **Gallery view** (decision feed, decision-order replay, embedding space, identity banks). |
| `kit/` | the deployable kit: `OnlineGallery` ingest service (`online.py`), CLI (`cli.py`), one-click `demo.py`, Docker stack. |

## Provenance

The stateful gallery is a *pure replayable function* of (sorted inputs, config, seed, code-SHA) with an
append-only decision event-log; output goes through the canonical `register_submission` + canary. One
`vlincs_sdk.research.start_run` per replay — never per-mutation logging, never a hand-rolled submission
parquet.

## Env

The **deployable kit** (`kit/requirements.txt`) is CPU-only and **public-PyPI-only** — no torch, no weights,
no internal index (`reid_hota` is the public NIST package). `vlincs_sdk` is **not** needed by the kit: it's
imported lazily only by two advanced gallery methods (`discriminability()` disc-ratio-keyed tau, and
`split_low_coherence()` revise) — the core match/expand/resolve/score path is pure numpy. The **full
dev/research** install is `pyproject.toml` (adds torch/ultralytics/sentence-transformers + `vlincs-sdk[harness]`
from the internal devpi index); the dedicated venv lives at `.venv` (the system Python is ABI-broken).
