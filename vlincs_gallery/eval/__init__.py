"""Canonical TA1 evaluation for the gallery PoC (reid_hota wrapper, MS02 firewall)."""

from .score import (
    Metrics,
    canonical_config,
    evaluate,
    load_ds1_gt,
    load_ms02_gt,
    restrict_to_gt_matched,
)

__all__ = [
    "Metrics",
    "canonical_config",
    "evaluate",
    "load_ds1_gt",
    "load_ms02_gt",
    "restrict_to_gt_matched",
]
