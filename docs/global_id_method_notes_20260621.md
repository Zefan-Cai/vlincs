# Global ID Method Notes, 2026-06-21

This note records the parts of the VLINCS global-ID work that are easiest to misread: which method line is the verified full-DS1 best, how the pairwise identity model enters the end-to-end pipeline, and how weak positives and hard negatives are actually generated in the current no-anchor setting.

## Current Best Boundary

By verified full-DS1 metrics, the current best method line is not the newest OSNet sample branch. It is the no-anchor full-DS1 global-ID resolver:

- It does not use human or GT global IDs as anchors or training seeds.
- It builds a graph from tracklet features, temporal constraints, and same-stream overlap cannot-link constraints.
- It trains a pair scorer / resolver from weak positives and hard negatives.
- It emits `predicted_global_id` for each tracklet, then applies global consistency, admission, and postfiltering.

The verified pair-model metrics recorded on 2026-06-21 were:

| Pair F1 | Precision | Recall |
|---:|---:|---:|
| 0.775234 | 0.820504 | 0.734698 |

The end-to-end DS1 metrics from the same point were:

| IDF1 | HOTA | AssA |
|---:|---:|---:|
| 0.655911 | 0.519311 | 0.534922 |

The correct takeaway is not that the whole pipeline had already crossed 70. The pairwise global-ID evidence had crossed 70, while the end-to-end delivery pipeline had not. The README may contain a newer full-score snapshot, but this distinction still matters: pair metrics do not directly equal final IDF1.

## Original End-To-End Pipeline

The original repository path was an online gallery / tracking-by-retrieval pipeline:

1. A detector / tracker produces detections or tracklets.
2. Each tracklet and its embedding enter `OnlineGallery.add_tracklet(...)`.
3. The gallery maintains identities, tracklets, and embeddings in PostgreSQL + pgvector.
4. Online assignment gives temporary global IDs by centroid / retrieval similarity.
5. Periodically, or at the end, `resolve_global(theta=0.02, top_k=15, ...)` reclusters globally with the resolve embedding.
6. `export_submission(...)` exports a TA1 submission zip.
7. The canonical `reid_hota` scorer computes globally aligned IDF1 / AssA / DetRe.

Main entrypoints:

- `kit/run_pluto_ds1.sh`
- `kit/online.py`

The reproduced DS1 baseline was:

| IDF1 | AssA | DetRe | IDs |
|---:|---:|---:|---:|
| 0.5999 | 0.4751 | 0.5176 | 1331 |

The same result was reproduced on `h100-test-2`, `h100-test-3`, and `test-video-0`. Relative to that baseline, the 2026-06-21 no-anchor best moved IDF1 from `0.5999` to `0.655911`, an absolute gain of `+0.056011` and a relative gain of about `+9.3%`.

## Role Of The OSNet Sample Branch

The newer sample branch strengthens tracklet identity evidence. It had not yet replaced the full-DS1 best:

- It uses BoTSORT tracklets from the real pipeline.
- It extracts multi-frame crops for each tracklet.
- It builds no-anchor features from OSNet + color histograms.
- It resolves global IDs on the sample set.
- GT is used only for evaluation, not as anchor or training evidence.

Sample-level verified results at that point:

| Branch | Identity F1 | Pair F1 |
|---|---:|---:|
| weak crop+bbox | 0.191162 | 0.032079 |
| BoTSORT OSNet+color | 0.380024 | 0.253572 |

This showed clear evidence improvement, but it was not enough to move directly into full-DS1 production. The intended full-DS1 gate is to push sample identity evidence clearly beyond the `0.38` plateau, then:

1. Write the new tracklet feature as a resolver-readable feature block or a `role='resolve'` embedding.
2. Use the no-anchor pair scorer / graph resolver to generate `tracklet -> predicted_global_id`.
3. Materialize the result as an assignment CSV or DB assignments.
4. Run the existing submission exporter.
5. Verify IDF1 / HOTA / AssA with the canonical DS1 scorer.

## Why The Model Trains Pairwise

In the no-anchor setting there are no human-labeled global IDs, so the model cannot directly learn:

```text
tracklet -> M0012
```

The current implementation instead trains:

```text
(tracklet A, tracklet B) -> same person or different person
```

It first performs pairwise identity verification, then converts those pairwise scores into global IDs through graph resolution and assignment materialization.

## Actual Weak Positive Definition

At the method-design level, weak positives can come from local-track continuation, mutual nearest neighbors, cycle consistency, and multi-model agreement. In the current main implementation, the core signal is:

```text
pseudo resolver ensemble votes
+ current online assignment agreement
+ visual similarity threshold
+ cannot-link veto
```

The flow is below.

### 1. Tracklet Embeddings And Candidate Pairs

Each tracklet has a L2-normalized feature, such as OSNet / DINO / color / fused feature. The model first computes cosine similarity:

```text
score(i, j) = emb[i] dot emb[j]
```

In code this is equivalent to:

```python
sim = x @ x.T
```

The candidate pool is not all pairwise combinations. Each tracklet keeps only its top-k similar neighbors. The sample sweep defaults are approximately:

```text
train_top_k = 45
pseudo_top_k = 24
candidate top_k = max(train_top_k, pseudo_top_k) = 45
```

The default `exclude_same=camera` also removes same-camera candidates, biasing training toward cross-camera identity matching.

### 2. No-GT Pseudo Resolver Ensemble

The same features are passed through the no-GT `time_agglom_resolve` path. Roughly, it builds pseudo clusters from:

```text
visual similarity + temporal support + cannot-link constraints
```

The sample sweep defaults enable an ensemble:

```text
pseudo_ensemble = true
pseudo_theta = 0.018
pseudo_ensemble_thetas = [0.016, 0.018, 0.020, 0.022]
pseudo_top_k = 24
pseudo_temporal_bonus = 0.005
pseudo_time_window_ms = 1000
```

For a candidate pair `(i, j)`, each pseudo clustering version gives one vote when:

```text
label[i] == label[j]
and cluster_size(label[i]) > 1
```

The default is:

```text
pseudo_ensemble_min_votes = 2
```

Therefore `same_pseudo = true` means at least two pseudo resolvers placed `i` and `j` in the same non-singleton cluster.

### 3. Current Online Assignment Agreement

The code also uses `online_gids` from the current assignments:

```text
online_gid[i] == online_gid[j]
and size(online_gid[i]) > 1
```

This is not a GT anchor. It is an independent weak signal produced by the current pipeline.

### 4. Three Positive Sources

A pair `(i, j)` becomes positive when it satisfies one of the following sources and does not trigger cannot-link.

`pseudo_online_agree`:

```text
same_pseudo = true
same_online = true
score >= pseudo_pos_min_sim
```

The sample default is `pseudo_pos_min_sim = 0.62`.

`strong_visual_pseudo`:

```text
same_pseudo = true
score >= pseudo_strong_pos_sim
```

The sample default is `pseudo_strong_pos_sim = 0.76`.

`consensus_votes`:

```text
votes >= pseudo_consensus_pos_min_votes
score >= pseudo_consensus_pos_min_sim
```

The sample defaults are:

```text
pseudo_consensus_pos_min_votes = 3
pseudo_consensus_pos_min_sim = 0.66
```

### 5. Cannot-Link Veto

If `(i, j)` hits a forbidden / cannot-link rule, it cannot become positive and is directly treated as negative:

```python
if cannot:
    negatives[(i, j)] = score
    continue
```

Current code anchors:

- `kit/no_anchor_global_id_model.py:530` builds forbidden pairs.
- `kit/no_anchor_global_id_model.py:563` sends cannot-link pairs directly to negatives.
- `kit/no_anchor_global_id_model.py:566` through `kit/no_anchor_global_id_model.py:574` define the three positive sources.

Compact formula:

```text
positive(i,j) =
  not cannot_link(i,j)
  and candidate_topk(i,j)
  and (
      same_pseudo(i,j) and same_online(i,j) and sim(i,j) >= 0.62
      or
      same_pseudo(i,j) and sim(i,j) >= 0.76
      or
      votes(i,j) >= 3 and sim(i,j) >= 0.66
  )
```

## Role Of Mutual Nearest Neighbor And Cycle Consistency

Mutual nearest neighbor means A's top-1 neighbor is B and B's top-1 neighbor is A. It is more conservative than one-way similarity because it filters cases where a visually generic tracklet becomes a popular accidental match.

Cycle consistency means a set of similarity links forms a closed loop, for example:

```text
A -> B
B -> C
C -> A
```

If all three edges are strong and no cannot-link rule fires, that component is more credible than a single similarity edge.

These ideas are useful weak-positive design principles, but they are not the central positive rule in the current main implementation. The current core is still pseudo resolver ensemble votes + online assignment agreement + similarity thresholds.

## Actual Hard Negative Definition

A hard negative is a high-confidence pair that cannot be the same person. The key distinction is:

```text
label source: cannot-link / physical impossibility
hardness source: high CV similarity
```

For example, two tracklets that overlap in time inside the same stream cannot be the same person. Even if their OSNet/color features are similar, they should be labeled negative. This teaches the pair scorer that visual similarity is not enough when temporal evidence contradicts identity.

The current hard-negative logic has three layers.

### 1. Same-Stream Temporal Overlap Cannot-Link

The strongest rule is `_build_overlap_forbidden`:

```text
two tracklets overlap in frame/time inside the same video + camera
=> forbidden pair
=> negative label
```

Code anchors:

- `kit/no_anchor_resolve_sweep.py:625`
- `kit/no_anchor_global_id_model.py:530`
- `kit/no_anchor_global_id_model.py:563`

This is the cleanest hard-negative source because its label does not depend on a CV model.

### 2. Low-Sim / No-Vote / No-Online-Agreement Negatives

A candidate pair is also marked negative when:

```text
pseudo resolver votes <= pseudo_ensemble_max_neg_votes
not same_online
visual score <= pseudo_neg_max_sim
```

The default is `pseudo_neg_max_sim = 0.42`. The code anchor is `kit/no_anchor_global_id_model.py:577`. These are closer to background / easy negatives than true hard negatives.

### 3. Random Negatives

The training set is supplemented with random negative pairs, up to `60000` by default. The sampler:

- excludes pairs already used elsewhere;
- excludes forbidden pairs;
- excludes same-camera pairs by default through `exclude_same=camera`;
- caps negative count at `4x` the positive count.

Code anchors:

- `kit/no_anchor_global_id_model.py:581`
- `kit/no_anchor_sample_parquet_sweep.py:1087`

## Caveat

The current main pair scorer does not yet systematically hard-label cross-camera travel-time impossibilities as negatives. Those constraints are currently more present as pair features, resolver temporal support, and later opponent/referee logic. Cross-camera impossible transfer remains a natural next hard-negative source, but the strongest hard-coded constraint today is same-stream temporal overlap cannot-link.
