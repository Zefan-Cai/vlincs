#!/usr/bin/env python3
"""Render gallery CI demo output into a SCORES.md table + headline SVG badges.

Stdlib only, fully internal (no external badge service). Reads each dataset's raw demo stdout (the file the
pipeline `tee`s), extracts the final `DONE: <dataset> -> {dict}` line, and writes:
  - SCORES.md           : full per-dataset table (IDF1 / AssA / DetRe / IDs) + commit + timestamp
  - badges/<ds>_<m>.svg : one flat badge per headline metric (always emitted; grey n/a when missing)

Usage: render_scores.py --sha <commit> [ms02_score.txt ds1_score.txt ...]
Missing/incomplete inputs degrade gracefully to "n/a" so the README never shows a broken image.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import re
from html import escape
from pathlib import Path

DONE_RE = re.compile(r"DONE:\s*([A-Za-z0-9_]+)\s*->\s*(\{.*\})")

# Datasets and the metrics shown in the table, in order.
DATASETS = [("ms02", "MS02"), ("ds1", "DS1")]
TABLE_METRICS = [("idf1", "IDF1"), ("assa", "AssA"), ("detre", "DetRe"), ("n_ids", "IDs")]
# Headline badges: (dataset_key, metric_key) — MS02 leads with AssA (sparse GT), DS1 with IDF1 (dense GT).
HEADLINE = [("ms02", "assa"), ("ms02", "idf1"), ("ds1", "idf1"), ("ds1", "assa")]


def parse_demo_output(path: str) -> dict | None:
    """Return the score dict from the LAST `DONE: <ds> -> {...}` line, or None if absent/unreadable."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    last = None
    for line in text.splitlines():
        m = DONE_RE.search(line)
        if m:
            last = m
    if last is None:
        return None
    try:
        return ast.literal_eval(last.group(2))
    except (ValueError, SyntaxError):
        return None


def metric_color(metric: str, v) -> str:
    if v is None:
        return "#9f9f9f"               # grey — no data
    if metric == "n_ids":
        return "#007ec6"               # blue — informational count
    if v >= 0.70:
        return "#4c1"                  # bright green
    if v >= 0.55:
        return "#97ca00"               # green
    if v >= 0.40:
        return "#dfb317"               # yellow
    if v >= 0.25:
        return "#fe7d37"               # orange
    return "#e05d44"                   # red


def fmt(metric: str, v) -> str:
    if v is None:
        return "n/a"
    return str(int(v)) if metric == "n_ids" else f"{v:.3f}"


def badge_svg(label: str, value: str, color: str) -> str:
    """A self-contained flat badge (shields-style), stdlib-only. Char width is approximated for DejaVu 11px."""
    cw, pad = 6.7, 9.0
    lw = int(len(label) * cw + 2 * pad)
    vw = int(len(value) * cw + 2 * pad)
    total = lw + vw
    lc, vc = lw / 2.0, lw + vw / 2.0
    lab, val = escape(label), escape(value)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" role="img" '
        f'aria-label="{lab}: {val}">\n'
        f'  <linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        f'<stop offset="1" stop-opacity=".1"/></linearGradient>\n'
        f'  <clipPath id="r"><rect width="{total}" height="20" rx="3" fill="#fff"/></clipPath>\n'
        f'  <g clip-path="url(#r)">\n'
        f'    <rect width="{lw}" height="20" fill="#555"/>\n'
        f'    <rect x="{lw}" width="{vw}" height="20" fill="{color}"/>\n'
        f'    <rect width="{total}" height="20" fill="url(#s)"/>\n'
        f'  </g>\n'
        f'  <g fill="#fff" text-anchor="middle" '
        f'font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">\n'
        f'    <text x="{lc:.0f}" y="15" fill="#010101" fill-opacity=".3">{lab}</text>\n'
        f'    <text x="{lc:.0f}" y="14">{lab}</text>\n'
        f'    <text x="{vc:.0f}" y="15" fill="#010101" fill-opacity=".3">{val}</text>\n'
        f'    <text x="{vc:.0f}" y="14">{val}</text>\n'
        f'  </g>\n'
        f'</svg>\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default="", help="commit sha to stamp (full or short)")
    ap.add_argument("inputs", nargs="*", help="demo stdout files (the *_score.txt the pipeline tee's)")
    args = ap.parse_args()

    # Collect scores by dataset from whichever input files parsed.
    scores: dict[str, dict] = {}
    for path in args.inputs:
        d = parse_demo_output(path)
        if d and isinstance(d, dict) and d.get("dataset"):
            scores[str(d["dataset"])] = d

    out = Path.cwd()
    badges = out / "badges"
    badges.mkdir(exist_ok=True)

    # Badges (always emit all headline badges so README references never 404).
    for ds_key, metric in HEADLINE:
        v = scores.get(ds_key, {}).get(metric)
        label = f"{dict(DATASETS).get(ds_key, ds_key.upper())} {dict(TABLE_METRICS)[metric]}"
        (badges / f"{ds_key}_{metric}.svg").write_text(
            badge_svg(label, fmt(metric, v), metric_color(metric, v))
        )

    # Table.
    short = (args.sha or "")[:12] or "(unknown)"
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    header = "| Dataset | " + " | ".join(lbl for _, lbl in TABLE_METRICS) + " |"
    sep = "|" + "---|" * (len(TABLE_METRICS) + 1)
    for ds_key, ds_lbl in DATASETS:
        d = scores.get(ds_key)
        cells = [fmt(m, (d or {}).get(m)) for m, _ in TABLE_METRICS]
        rows.append(f"| {ds_lbl} | " + " | ".join(cells) + " |")
    md = (
        "# Latest CI scores\n\n"
        "_Auto-generated by the self-hosted maxwell CI on each commit, via the canonical `reid_hota` "
        "scorer (global ID alignment, IoU — the takehome leaderboard metric). Do not edit by hand._\n\n"
        f"{header}\n{sep}\n" + "\n".join(rows) + "\n\n"
        f"Commit `{short}` · {now}\n\n"
        "> **MS02 GT is sparse** (~1.5 det/frame) → lead with **AssA**; IDF1 is deflated by real-but-"
        "unannotated people.\n"
        "> **DS1 GT is dense** → **IDF1** is the trustworthy number. Config: appearance-only "
        "(`--no-cannot-link`), the kit's reference config.\n"
    )
    (out / "SCORES.md").write_text(md)
    print(f"[render] wrote SCORES.md + {len(HEADLINE)} badges; datasets parsed: {sorted(scores)}")


if __name__ == "__main__":
    main()
