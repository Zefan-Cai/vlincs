# vlincs_gallery

Online, revisable, **retrieval-based** identity assignment ("tracking-by-retrieval" / online MTMC)
for VLINCS TA1 — a pivot away from the lossy batch funnel
(`detect → track → pool → UMAP → HDBSCAN → GNN → split → merge`) toward per-detection
match-or-expand against a live, queryable, revisable gallery of identities backed by FAISS +
pgvector.

The full design, rationale, eval protocol, and phased plan with gates live in
[`PROTOCOL.md`](PROTOCOL.md) — a good place to start.

## Status
Phase 0 (foundation). Eval harness (`vlincs_gallery/eval/score.py`) is the first runnable artifact;
run `python -m vlincs_gallery.eval.score --selftest` to confirm the MS02 sparse-GT firewall.

## Why
TA1 is a global-ID-aligned *assignment* metric, and the batch pipeline is a chain of irreversible
compressions (per-det oracle 0.7264 vs shipped 0.6944; tracker DetRe ≈ 0.61 is the wall). Online
assignment decouples detection recall from association — the only documented path past the DS1
wall — and is the natural fit for DS2's over-fragmentation.

## Provenance
Consumes provenanced detect/embed/geo/triage MLflow artifacts; the stateful gallery is a *pure
replayable function* of (sorted inputs, config, seed, code-SHA) with a decision event-log; output
goes through the canonical `register_submission` + canary. One `vlincs_sdk.research.start_run` per
replay — never per-mutation logging, never a hand-rolled submission parquet.

## Env
Dedicated venv at `.venv` (built from the internal devpi index). Base mirrors
`vlincs_fusion/.venv` + `faiss-cpu`, `psycopg[binary]`, `pgvector`, `geoalchemy2`, `sqlalchemy`,
`sentence-transformers`. The system Python is ABI-broken — always use the venv.
