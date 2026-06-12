# VLINCS online-gallery kit — *bring your tracker's tracklets + embeddings, we do the cross-camera identities*

You get a **dataset** and our **online cross-camera identity gallery**. You pick the detector, the
tracker, the embedder — whatever gets you there works. **The recommended input is your tracker's
tracklets** (one pooled embedding per within-camera track): the gallery's job is *cross-camera* identity,
and *within-camera* consolidation is your tracker's job — an appearance+motion tracker (e.g. BoT-SORT with
a ReID encoder) does it far better than per-detection cosine can. You stream tracklets in *as you ingest*;
the gallery decides **match / expand / do-nothing** against the live identity bank, does the **periodic
resolve (consolidation)**, assigns global IDs, and scores **IDF1** — with a visualization of how/when/why
every identity is what it is. (Per-*detection* ingestion is also supported for streaming use, but without a
tracker's within-camera consolidation it over-fragments — push tracklets if you have them.)

![ingest · decide · store · resolve](../assets/architecture.svg)

The gallery ships **no weights and no models** — it's pgvector (durable store + ANN) + an in-process
cosine index over *your* embeddings + the decision rules, so there's nothing to download. The DB ships
**empty**; you fill it by running your pipeline.

---

## What we give you

1. **A dataset** (videos + ground truth, on the datastore mount) — run *your* pipeline on these videos:

   | pick | videos | GT | local score? | this kit's online ref                          | supervised ceiling |
   |---|---|---|---|------------------------------------------------|---|
   | `ms02` | 2 | yes (sparse) | ✓ | **IDF1 ≈ 0.44** *(reference config, this kit)*                               | — |
   | `ds1`  | 10 (Tc6+Tc8) | yes (dense) | ✓ | **IDF1 ≈ 0.53** *(reference config, this kit)* | |
   | `ds2`  | 30 (Tc1–Tc8) | none | ✗ (viz only) | —                                              | |

   > **Two different regimes, worth keeping separate.** The **online ref** is what *this kit's* reference
   > config scores online (no per-dataset training, no prior knowledge of these identities) — a fair bar
   > for a streaming gallery. The **supervised ceiling** (DS1 0.69) is our *batch* funnel with a
   > GNN **trained on DS1 ground truth** — a different, supervised regime the online kit is not trying to
   > match. The **online ref** is the reference point to compare against with your
   > detector/tracker/embedder; treat the supervised ceiling as context, not a target.
   >
   > **DS1 is the real test** (dense GT → trustworthy IDF1). **MS02 GT is sparse** (~1.5 det/frame), so a
   > hot detector's real-but-unannotated people deflate IDF1 (the shipped demo, default vetoes on, scores
   > IDF1 ≈ 0.42 with AssA ≈ 0.70) — on MS02 **lead with AssA**, it's the phantom-immune signal.
   > **DS2 ships no GT** → ingest + viz only;
   > submit to the leaderboard for a number.

2. **The online gallery + ingest + viz + scorer** (this kit). CPU-only, no GPU, no weights. `docker
   compose up` and it's ready.

## What you give it — embeddings, pushed live (no files)

**No intermediate files.** Your pipeline drives the gallery through a tiny Python API; every call writes
straight to the DB + the live index and returns a global id. You push on **whatever cadence you like**
(per detection as frames stream, per tracklet when one completes, or batches) — state is consistent after
every call. The only file ever written is a submission.

```python
from online import OnlineGallery                        # (PYTHONPATH the kit, or run inside the app container)
g = OnlineGallery("ds1")                                # connects to the EMPTY db; loads camera geo from the dataset

for video, camera, frames, boxes, embedding in your_tracker():       # YOUR detector+tracker+embedder
    gid = g.add_tracklet(video, camera, frames, boxes, embedding)    # one pooled (or per-det) embedding -> gid, live
    # ...or, per-detection streaming (no tracker — note: over-fragments without within-camera consolidation):
    # gid = g.add_detection(video, camera, frame, box, embedding)    # match / expand / do-nothing -> gid
    if your_cadence: g.resolve()                        # periodic resolve (consolidation), your cadence

print(g.score())                  # {'idf1': ...} from the live DB (ms02/ds1; ds2 -> None, no GT)
g.export_submission("out.zip")    # the ONLY file ever written
```

- `embedding` is **your** appearance vector, **any dimension D** (the index adapts on the first push).
  **No global IDs from you** — the gallery assigns them.
- The physical vetoes are **on by default** (`cannot_link=True`): `same_frame` (two spatially-distinct
  boxes in one `(video,frame)` can't share an id — the intra-camera case), plus cross-camera
  `simultaneity`/`travel`. Timing + geo come from the dataset's shipped camera pose, which the kit reads
  off the datastore — DS1/DS2 from `*_camera_extrinsics_*.parquet`, MS02 from `MCAMxxx_camera_params.json`
  (camera position derived from `R,t`). Pass `cannot_link=False` (or `demo --no-cannot-link`) for the old
  appearance-only behavior.
- The viz reads the **live DB** — decisions/identities/crops update as your loop runs.

See `example.py` for the copy-me template (drop in your own pipeline), and `demo.py` — run via
`docker compose run --rm app demo` — for a **one-click real run** on shipped MS02 data: real
tracklets+embeddings streamed through the live gallery → real AssA, viz populated, no pipeline to write.

## Setup — DB + Gallery view (one `docker compose up`)

Everything runs from `kit/`. `docker compose up` starts the four long-running services (`db` / `viz` /
`ui` / `pgadmin`); `app` is the one-shot CLI (a `cli` compose profile, so `up` doesn't try to run it — you
invoke it with `docker compose run --rm app …`). The only thing you must point it at is the dataset mount
(videos + GT), via `DATA_ROOT`.

```bash
cd kit
export DATA_ROOT=/mnt/datastore2_videolincs/data      # where the VLINCS datastore is mounted (default if unset)
DATASET=ds1 docker compose up -d                      # brings up the four services below (app is run-on-demand)
```

| service | what | where |
|---|---|---|
| `db`  | empty Postgres + pgvector (the system-of-record + cannot-link query layer) | `localhost:55433` |
| `pgadmin` | pgAdmin 4 to inspect the DB visually — the `db` server is pre-registered (pwd `gallery`) | **http://localhost:5050** |
| `viz` | FastAPI read API over the live DB (crops, decisions, identities, state) | `localhost:8077` |
| `ui`  | the Angular **Gallery view**, proxied to `viz` | **http://localhost:4200** |
| `app` | the CLI (`demo` / `list-videos` / `score` / `submit` / `selftest`) — run on demand, not a daemon | — |

### Embedding-space view — *what the matcher/index actually sees*

The Gallery view's right column carries an **embedding-space panel**: a 2D projection (PCA, deterministic;
UMAP if installed) of the matcher's match space, coloured by global id. Toggle **bank exemplars** (one point
per live exemplar in the FAISS/hnsw bank — *this is the gallery the matcher scores against*) vs
**per-detection** (one point per detection, by assigned id — a denser cloud). Hover for the identity + a crop
thumbnail; click a point to focus that id. Backed by `GET /embedding_projection?mode=bank|det[&card=]` and
`GET /decision_geometry/{det_id}` (a single decision's query vector + the candidate exemplars it scored
against + τ). These read `detections.embedding_red` / `identity_reps.embedding_red`, which `OnlineGallery`
now persists as it ingests (the pooled 64-d match-space vector; a non-64-d embedder leaves them NULL and the
panel is simply empty — the matcher and score are unaffected).

### Setting up the DB

**There is no manual migration step.** The `db` service ships **empty**. The per-dataset database
(`gallery_ds1` / `gallery_ms02` / `gallery_ds2`) is created and the schema (`db/init.sql`) applied
**automatically** the first time an `OnlineGallery(<dataset>)` is constructed — i.e. the first `app demo`,
`app score`, or push from your own pipeline. So the DB "setup" is just:

```bash
DATASET=ds1 docker compose up -d                          # Postgres comes up empty
docker compose run --rm app demo                          # real one-click MS02 run (creates gallery_ms02 + schema en route)
```

- **One database per dataset** so runs are isolated; `OnlineGallery(dataset, truncate=True)` wipes that
  dataset's DB for a clean, uncontaminated number (the kit CLI's `demo` truncates; `score`/`submit` don't).
- **Your pipeline connects from your own env** to the same Postgres: `PGHOST=localhost PGPORT=55433`,
  user/password `gallery` / `gallery`, database `gallery_<dataset>` (the Gallery API picks the right db
  from the dataset name — you just pass `OnlineGallery("ds1")`).
- Persisted in the `pgdata` docker volume — it survives `docker compose down`; use `docker compose down -v`
  to drop it, or `truncate=True` to reset one dataset in place.
- Prefer to run the DB standalone (no viz/UI)? `db/docker-compose.yml` at the repo root is a bare
  pgvector instance on the same `:55433` that mounts `init.sql` directly.

### Setting up the Gallery view

The UI comes up with the same `docker compose up` — no separate build. Open **http://localhost:4200**.

- The view follows whichever dataset you set at compose time (`GALLERY_DATASET`, defaulted from
  `DATASET`). **To view a different dataset, restart with a new `DATASET`:**
  ```bash
  DATASET=ms02 docker compose up -d viz ui                 # re-point the viz + UI at gallery_ms02
  ```
- The UI needs the dataset mount for crops/frames — that's the same `DATA_ROOT` you exported above
  (mounted read-only into `viz`).
- It reads the **live DB**: leave it open while your pipeline pushes and scrub/reload to watch
  identities form. (First load builds the Angular dev server in the `node:20` container — give it a
  minute; watch `docker compose logs -f ui`.)

## Drive it from your pipeline + score

```bash
docker compose run --rm app demo                          # ONE-CLICK REAL run on shipped MS02 data (vetoes ON) -> real AssA + populated viz
docker compose run --rm app demo --no-cannot-link         # same, but appearance-only (vetoes OFF) for comparison
docker compose run --rm app list-videos --dataset ds1     # the video stems + camera names to run YOUR pipeline on

#   <in your env: connect to localhost:55433, drive the Gallery API over your pipeline (see example.py)>

docker compose run --rm app score  --dataset ds1          # IDF1 from the live DB  (this kit's online ref ≈ 0.53)
docker compose run --rm app submit --dataset ds1 --out out.zip    # TA1 submission (the only file)
docker compose run --rm app selftest --dataset ms02       # wiring check on RANDOM embeddings (not a real score)
```

> **Start with `./demo.sh`** (one-click) — it brings up the persistent stack (db + pgadmin + viz + ui)
> for MS02, streams shipped real tracklets+embeddings through the live gallery (no pipeline, no weights),
> prints a real result (AssA ≈ 0.70; IDF1 ≈ 0.42 — default vetoes on; MS02 GT is sparse, so lead with AssA), and **leaves
> the stack up** so the Gallery view (http://localhost:4200) and pgAdmin (:5050) show it immediately.
> `docker compose run --rm app demo` alone only *populates* the DB — the `app` container is one-shot and
> doesn't start viz/ui, so use `./demo.sh` (or `docker compose up -d` first) if you want the view.

### DS1 demo (on-network) — `./demo.sh ds1`

The MS02 demo ships its inputs. **DS1 doesn't** (too big), so the DS1 demo *fetches* a tracker's tracklets
+ per-tracklet embeddings from MLflow (no recompute, no GPU) per a small SDK pipeline recipe
(`pipelines/ds1.yaml`: the `track` + `embed` run ids and a `reduce` to the 64-d match space), then streams
them through the gallery exactly like MS02. So DS1 is **on-network only**:

```bash
# build the DS1-capable image (adds mlflow + vlincs-sdk[mlflow] from the internal index) — one-time
WITH_DS1=1 docker compose build app
export MLFLOW_TRACKING_URI=http://maxwell.novateur.com:9091   # passed through to the app container
./demo.sh ds1                                                 # fetch -> gallery -> real IDF1 (dense GT)
```

DS1 has dense GT, so **IDF1 is the trustworthy number** here (unlike MS02). Swap the tracker/embed by
editing the two run ids in `pipelines/ds1.yaml`. (Scoring + crops also need the datastore mount, same
`DATA_ROOT` as always.)

## Using the Gallery view

The view at **http://localhost:4200** is the *why/when/how* of every identity — it reads the live DB, so
it tracks your ingest loop. Top-to-bottom:

- **Config strip** (header) — the decision config this DB state was built with (`cannot_link`, `match_mode`,
  `tau`, `merge_tau`, …), persisted with the run as a `role='gallery'` row so you can always see whether the
  data in front of you is the vetoes-on or appearance-only run. `cannot_link` is highlighted green (on) / amber (off).
- **Timeline lens** (`Playback (t)` / `Decision order`) — the **default is Decision order**: the scrubber
  steps over **ingest order** (one step per tracklet decision), replaying the gallery exactly as it was built —
  the lens for *auditing decisions* (no "future" crops). `Playback (t)` is the wall-clock view (where boxes are
  on screen at time _t_). Everything below tracks whichever lens is active.
- **Card toggle** (DS1 only: `Tc6` / `Tc8` / all) — Tc6 and Tc8 reuse camera names and sit ~2.5h apart;
  the toggle un-conflates the two sessions in space *and* time. Pick one to scrub a single session at
  full resolution.
- **KPI strip** — live counts (identities, detections, tracklets, **cross-camera** identities) and the
  decision tallies (match / expand / admitted-to-bank).
- **Scrubber + transport** (⏮ / play / ⏭, speed) — in **Decision order** it steps over **ingest step N**
  (each step = one tracklet decision, in the order made); in **Playback** it maps over *actual detection
  time* (skipping the Tc6/Tc8 dead gap). Either way the whole view shows the gallery **as of the cursor**:
  which identities exist yet, how much of each has been committed.
- **Per-camera canvases** — boxes **colored by global id**. In Playback, every box near _t_ (same color
  across cameras = a cross-camera match). In Decision order, the boxes of the **tracklet decided at this
  step**, colored by the identity it had *as of this step*.
- **Decision feed** — the trail up to the cursor, newest first. Each **decision** row shows the type
  (match / expand / do-nothing), the chosen gid, `added` / `not-added` (bank admission), candidate ids +
  cosine scores, and any **veto** (`same_frame` / `simultaneity:CAM` / `travel:CAM` / `below_tau`).
  **Merge** rows (purple `⟳ merged`, Decision-order only) show a `resolve()` consolidation: `id A → id B`
  with the centroid cosine that triggered it (`≥ merge_tau`) and the step it fired — the *when/where/why*.
  A small legend distinguishes the two triggers (a decision per ingested tracklet; a merge per periodic
  `resolve()`). Click an identity and every feed row that **involves** it (chose / considered / merged it)
  is highlighted.
- **Embedding space** — a 2D PCA/UMAP projection of the **match space**, colored by id: toggle **bank
  exemplars** (the live FAISS/hnsw bank the matcher scores against) vs **per-detection**. **⤢ enlarge** opens
  a large zoomable view (scroll-zoom/pan); a **☀ light/dark** toggle helps read clusters (persisted). It
  tracks the cursor too — at step N it shows the bank *as of step N*.
- **Click an identity (gid)** → identity detail: its tracklets (admitted vs rejected), exemplar crops, the
  cameras it spans, and `X of Y tracklets added` — **cursor-aware**, so it matches the embedding bank at the
  current step (e.g. "2 of 3" at step 10, "10 of 35" at the end). This is how you audit a merge.
- **Click a tracklet (seq)** → tracklet detail (left column): its decision + why, and a subsampled **crop
  strip** with `+ load more` / `show all`. Every crop carries its `camera:frame:box` det-id caption.
- **Crops** are decoded on demand from the source video frames (`DATA_ROOT`), cached after first view.

**Reading the view while tuning:** lots of `expand` with low scores ⇒ `tau` is too high for your
embedding's cosine scale (sweep it down). One identity matching everything (an attractor) ⇒ lower
`coherence_floor`. Over-split identities the resolve should have merged ⇒ tune `merge_tau` and call
`g.resolve()` more often. See the strategy table below.

### The strategies you tune (`OnlineGallery(...)` kwargs / decision knobs)

| kwarg | strategy |
|---|---|
| `match_mode` | how a candidate identity is scored: **`centroid`** (default, validated — cosine to the id's whole-bank mean) / `max` (nearest single exemplar) / `retrieval` (FAISS k-NN vote, needs `faiss-cpu`). `centroid` was +0.06 IDF1 over `max` on DS1 |
| `tau` | **match** threshold — cosine ≥ τ to an existing identity ⇒ match, else **expand** (new id); default 0.60 for `centroid` |
| `tracklet_coh_min` | **do-nothing** — quarantine an internally-incoherent (mixed-person) tracklet: own id, never poisons a bank |
| `coherence_floor` | anti-accretion — reject a bank exemplar too far from the rest (stops "matches-everything" attractors) |
| `merge_tau` | **periodic resolve** — consolidate over-split identities whose exemplar centroids agree (used by `g.resolve()`) |
| `admit_tau` / `max_reps` | **bank admission** — add a matched tracklet as a new exemplar only if it's at most `admit_tau`-similar to the existing ones (redundant = skipped); cap the bank at `max_reps` exemplars per id |
| `cannot_link` | the physical vetoes — `same_frame` (intra-camera: two distinct boxes in one frame can't share an id) + cross-camera `simultaneity`/`travel` (default **on**; `False` = appearance-only, the old "best DS1 config") |
| `max_speed` / `sim_window_ms` / `same_box_iou` / `overlaps` | **veto thresholds** (active when `cannot_link` on) — travel-speed limit (m/s), simultaneity window (ms), same-frame IoU cutoff, and known overlapping-FOV camera pairs (suppress the simultaneity veto there) |

Defaults are the kit's current config (physically-correct vetoes **on** — see `cannot_link`).
**Tune `tau` to your embedding's cosine scale** (raw normalized features sit near ~0.9; sweep it).

## Trust the number

`gallery` scores with the **canonical** `reid_hota` (global ID alignment, IoU, dense=False — the takehome
leaderboard config), keyed by **video** for DS1 (Tc6/Tc8 reuse camera names). `gallery --selftest`
confirms perfect input → IDF1=1.0. Same metric, same frame as the leaderboard.
