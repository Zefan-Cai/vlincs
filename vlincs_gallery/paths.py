"""Central data-path config — the single place that knows where the VLINCS datasets live on disk.

DATA_ROOT is the datastore mount (host: /mnt/datastore2_videolincs/data; kit container: /data). BOTH the
canonical 'Box' export AND the legacy '-selected' tree live under it, so one read-only mount of DATA_ROOT
exposes everything. Override DATA_ROOT to relocate the whole datastore.

  DATA       -> the canonical MITRE 'Box' export root — the DEFAULT data directory (DS0001/DS0002/MS01/...)
  MS02_DATA  -> the '-selected' tree, where the MS02 debug/demo set still lives. MS02 is bound to the
                vlincs-baseline repo (MITRE upstream) and is not in the Box export yet, so it stays pinned
                here until the baseline-sourcing adapter lands. ("a bad thing to have, but correct.")

Dependency-free (only `os`) so import-light modules (kit/online.py) can use it without pulling DB/numpy.
"""
from __future__ import annotations

import os

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/datastore2_videolincs/data").rstrip("/")

DATA = f"{DATA_ROOT}/Box/VLINCS_Performer"            # canonical Box export — the default data directory
MS02_DATA = f"{DATA_ROOT}/VLINCS_Performer-selected"  # MS02 (baseline-derived) still lives in the -selected tree

# Per-dataset card directories (each holds videos + GT + extrinsics). MS02 -> -selected; ds1/ds2 -> Box.
CARDDIRS = {
    "ms02": [f"{MS02_DATA}/MS02/MC0002/2018-03-Tc85"],
    "ds1": [f"{DATA}/MS01/MC0001/2024-03-Tc6", f"{DATA}/MS01/MC0001/2024-03-Tc8"],
    "ds2": [f"{DATA}/MS01/MC0001/2024-03-Tc{n}" for n in (1, 2, 3, 5, 7)]
    + [f"{DATA}/MS01/MC0001/2024-04-Tc4"],
}
HAS_GT = {"ms02": True, "ds1": True, "ds2": False}

# GT card dirs for the scorer (same trees: MS02 -> -selected, DS1 -> Box).
MS02_GT_DIR = f"{MS02_DATA}/MS02/MC0002/2018-03-Tc85"
DS1_GT_DIRS = {"Tc6": f"{DATA}/MS01/MC0001/2024-03-Tc6", "Tc8": f"{DATA}/MS01/MC0001/2024-03-Tc8"}


def root_for_site(site: str) -> str:
    """The dataset-tree root for a SITE token: 'MS02' -> the -selected tree; everything else -> Box."""
    return MS02_DATA if str(site).upper() == "MS02" else DATA
