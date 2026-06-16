"""Frozen policy configuration for the gallery PoC.

The generalizability gate (PROTOCOL §2) is that ONE PolicyConfig must work on MS02 and DS1 (and
thence DS2) with no per-dataset retuning. For the MVP shakeout `match_tau` is an explicit knob we
sweep to find the regime; the final policy derives it from the live disc-ratio (see policy notes in
PROTOCOL §4) so it self-adapts across datasets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    embedder: str = "ft"        # "ft" (SOLIDER DS1-FT) | "base" (SOLIDER MSMT17) - we compare both
    match_tau: float = 0.55     # cosine match threshold; MVP=fixed, final=disc-ratio-keyed
    admit_tau: float = 0.85     # diversity-gated exemplar admission (admit if max-cos to bank < this)
    max_reps: int = 16          # exemplar-bank cap (backstop; bank self-sizes via admit_tau)
    conf_floor: float = 0.10    # drop detections below this conf (free filter on the conf=0.1 superset)
    disc_k: int = 20            # neighborhood size for the discriminability ratio
    seed: int = 0
