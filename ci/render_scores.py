#!/usr/bin/env python3
"""Render gallery CI demo output into per-dataset SCORES.md history tables + best-IDF1 SVG badges.

Maintains an append-only history at ci/scores_history.csv: one row per (dataset, commit) carrying who
checked it in, when, the commit hash, and the metrics. Each CI run parses the demo stdout, upserts the
current commit's row for each dataset that scored, then regenerates:
  - ci/scores_history.csv          : the durable history (source of truth)
  - SCORES.md                      : a "best so far" highlight + one history table per dataset
                                     (Checked in | Author | Commit | IDF1 | AssA | DetRe | IDs)
  - badges/<ds>_best_idf1.svg      : the BEST-ever IDF1 per dataset, value "<idf1> @ <commit>"
Stdlib only, fully internal (no external badge service).

Usage: render_scores.py --sha <commit> --author <name> --date <iso> [ms02_score.txt ds1_score.txt ...]
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from html import escape
from pathlib import Path

DONE_RE = re.compile(r"DONE:\s*([A-Za-z0-9_]+)\s*->\s*(\{.*\})")
DATASETS = [("ms02", "MS02"), ("ds1", "DS1")]           # key -> display label
HISTORY = Path("ci/scores_history.csv")
FIELDS = ["dataset", "commit", "commit_short", "author", "date", "idf1", "assa", "detre", "n_ids"]
METRICS = [("idf1", "IDF1"), ("assa", "AssA"), ("detre", "DetRe"), ("n_ids", "IDs")]
PV_METRICS = [("idf1", "IDF1"), ("assa", "AssA"), ("detre", "DetRe")]   # per-video columns (no per-video n_ids)
MAX_ROWS = 30                                           # newest N shown per table (full history kept in CSV)
README = Path("README.md")
README_START = "<!-- CI-SCORES:START -->"
README_END = "<!-- CI-SCORES:END -->"


def parse_demo_outputs(path: str) -> list[dict]:
    """Score dicts from every `DONE: <ds> -> {...}` line in a demo log."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        m = DONE_RE.search(line)
        if m:
            try:
                d = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def parse_demo_output(path: str) -> dict | None:
    """Score dict from the LAST `DONE: <ds> -> {...}` line; kept for callers/tests."""
    parsed = parse_demo_outputs(path)
    return parsed[-1] if parsed else None


def load_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    with HISTORY.open(newline="") as fh:
        return [row for row in csv.DictReader(fh)]


def save_history(rows: list[dict]) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def fnum(metric: str, v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return str(int(round(f))) if metric == "n_ids" else f"{f:.3f}"


def idf1_color(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "#9f9f9f"
    if f >= 0.70:
        return "#4c1"
    if f >= 0.55:
        return "#97ca00"
    if f >= 0.40:
        return "#dfb317"
    if f >= 0.25:
        return "#fe7d37"
    return "#e05d44"


def short_date(iso: str) -> str:
    return (iso or "")[:16].replace("T", " ")


def badge_svg(label: str, value: str, color: str) -> str:
    """Self-contained flat badge (shields-style); char width approximated for DejaVu 11px."""
    cw, pad = 6.7, 9.0
    lw = int(len(label) * cw + 2 * pad)
    vw = int(len(value) * cw + 2 * pad)
    total, lc, vc = lw + vw, lw / 2.0, lw + vw / 2.0
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


def best_row(rows: list[dict], ds_key: str) -> dict | None:
    cand = []
    for r in rows:
        if r.get("dataset") != ds_key:
            continue
        try:
            cand.append((float(r["idf1"]), r))
        except (TypeError, ValueError, KeyError):
            continue
    if not cand:
        return None
    # max IDF1; tie-break to the most recent date
    cand.sort(key=lambda t: (t[0], t[1].get("date", "")))
    return cand[-1][1]


def per_video_table(pv: dict) -> list[str]:
    """Markdown for one dataset's per-video sub-scores (worst IDF1 first — where we failed).

    pv: {video_key -> {idf1, assa, detre}}, all computed under the GLOBAL alignment.
    """
    out = ["| Video | " + " | ".join(l for _, l in PV_METRICS) + " |",
           "|" + "---|" * (1 + len(PV_METRICS))]

    def worst_first(item):
        try:
            return float(item[1].get("idf1"))
        except (TypeError, ValueError):
            return 1e9
    for vid, m in sorted(pv.items(), key=worst_first):
        out.append(f"| {vid} | " + " | ".join(fnum(k, m.get(k)) for k, _ in PV_METRICS) + " |")
    return out


def update_readme_block(body: str) -> bool:
    """Replace the text between the CI-SCORES sentinels in README.md. False if README/sentinels absent."""
    if not README.exists():
        return False
    text = README.read_text()
    if README_START not in text or README_END not in text:
        return False
    pre, rest = text.split(README_START, 1)
    _, post = rest.split(README_END, 1)
    README.write_text(f"{pre}{README_START}\n{body}\n{README_END}{post}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default="", help="full commit sha of the run")
    ap.add_argument("--author", default="(unknown)", help="commit author name")
    ap.add_argument("--date", default="", help="commit date (ISO 8601)")
    ap.add_argument("inputs", nargs="*", help="demo stdout files (the *_score.txt the pipeline tee's)")
    args = ap.parse_args()

    sha = (args.sha or "").strip()
    short = sha[:7] if sha else "(unknown)"

    rows = load_history()

    # Upsert this run's entries (one per dataset that produced a DONE line); replace any prior row for the
    # same (dataset, commit) so a re-run on the same commit refreshes rather than duplicates. Also capture
    # this run's per-video sub-scores (current commit only; not historized — they show where we fail NOW).
    pv_by_ds: dict[str, dict] = {}
    for path in args.inputs:
        for d in parse_demo_outputs(path):
            if not d.get("dataset"):
                continue
            ds = str(d["dataset"])
            rows = [r for r in rows if not (r.get("dataset") == ds and r.get("commit") == sha)]
            rows.append({
                "dataset": ds, "commit": sha, "commit_short": short,
                "author": args.author, "date": args.date,
                "idf1": d.get("idf1", ""), "assa": d.get("assa", ""),
                "detre": d.get("detre", ""), "n_ids": d.get("n_ids", ""),
            })
            if isinstance(d.get("per_video"), dict) and d["per_video"]:
                pv_by_ds[ds] = d["per_video"]

    save_history(rows)

    out = Path.cwd()
    badges = out / "badges"
    badges.mkdir(exist_ok=True)

    # Best-IDF1 badge per dataset, with the winning commit baked into the value.
    bests = {}
    for ds_key, ds_lbl in DATASETS:
        b = best_row(rows, ds_key)
        bests[ds_key] = b
        if b is None:
            svg = badge_svg(f"{ds_lbl} best IDF1", "n/a", "#9f9f9f")
        else:
            svg = badge_svg(f"{ds_lbl} best IDF1",
                            f"{float(b['idf1']):.3f} @ {b.get('commit_short', '')}",
                            idf1_color(b["idf1"]))
        (badges / f"{ds_key}_best_idf1.svg").write_text(svg)

    # SCORES.md: best-so-far highlight (prominent commit hash) + one history table per dataset.
    lines = [
        "# Gallery CI scores",
        "",
        "_Auto-generated by the self-hosted maxwell CI on each commit via the canonical `reid_hota` scorer "
        "(global ID alignment, IoU — the takehome leaderboard metric). Source of truth: "
        "`ci/scores_history.csv`. Do not edit by hand._",
        "",
        "## 🏆 Best so far (by IDF1)",
        "",
        "| Dataset | Best IDF1 | Commit | Author | Checked in |",
        "|---|---|---|---|---|",
    ]
    for ds_key, ds_lbl in DATASETS:
        b = bests[ds_key]
        if b is None:
            lines.append(f"| {ds_lbl} | n/a | — | — | — |")
        else:
            lines.append(
                f"| {ds_lbl} | **{fnum('idf1', b['idf1'])}** | `{b.get('commit_short', '')}` | "
                f"{b.get('author', '')} | {short_date(b.get('date', ''))} |"
            )
    lines += [
        "",
        "> Config pinned `--no-cannot-link` (appearance-only).",
        "",
    ]

    for ds_key, ds_lbl in DATASETS:
        hist = sorted((r for r in rows if r.get("dataset") == ds_key),
                      key=lambda r: r.get("date", ""), reverse=True)
        lines.append(f"## {ds_lbl} — history")
        lines.append("")
        lines.append("| Checked in | Author | Commit | " + " | ".join(l for _, l in METRICS) + " |")
        lines.append("|" + "---|" * (3 + len(METRICS)))
        for r in hist[:MAX_ROWS]:
            cells = " | ".join(fnum(m, r.get(m)) for m, _ in METRICS)
            lines.append(f"| {short_date(r.get('date', ''))} | {r.get('author', '')} | "
                         f"`{r.get('commit_short', '')}` | {cells} |")
        if not hist:
            lines.append("| — | — | — | n/a | n/a | n/a | n/a |")
        if len(hist) > MAX_ROWS:
            lines.append(f"\n_…showing the {MAX_ROWS} most recent of {len(hist)} runs "
                         f"(full history in `ci/scores_history.csv`)._")
        lines.append("")
        # Per-video sub-scores for THIS run (global-aligned; worst IDF1 first → where the score breaks down).
        lines.append(f"### {ds_lbl} — per-video (latest run @ `{short}`, global-aligned)")
        lines.append("")
        lines += per_video_table(pv_by_ds[ds_key]) if pv_by_ds.get(ds_key) \
            else ["_Per-video populated on the next CI run._"]
        lines.append("")

    (out / "SCORES.md").write_text("\n".join(lines) + "\n")

    # Mirror the latest-run per-video into the README's CI-managed block, so it's visible without opening
    # SCORES.md: a best-IDF1 one-liner + a per-video table per dataset.
    best_bits = [
        (f"**{ds_lbl}** {fnum('idf1', bests[ds_key]['idf1'])} @ `{bests[ds_key].get('commit_short', '—')}`"
         if bests[ds_key] else f"**{ds_lbl}** n/a")
        for ds_key, ds_lbl in DATASETS
    ]
    block = [f"**Best IDF1:** {' · '.join(best_bits)}", ""]
    if pv_by_ds:
        block.append(f"**Per-video — latest run @ `{short}`** (global-aligned, worst first — where we fail):")
        for ds_key, ds_lbl in DATASETS:
            if pv_by_ds.get(ds_key):
                block += ["", f"_{ds_lbl}_", ""] + per_video_table(pv_by_ds[ds_key])
        block += ["", "Full history → [`SCORES.md`](SCORES.md)."]
    else:
        block.append("_Per-video scores populate here on the next CI run — full history in "
                     "[`SCORES.md`](SCORES.md)._")
    wrote_readme = update_readme_block("\n".join(block))

    print(f"[render] history rows: {len(rows)} | per-video datasets: {sorted(pv_by_ds)} | "
          f"README block: {'updated' if wrote_readme else 'SKIPPED (no sentinels)'} | best IDF1: "
          + ", ".join(f"{lbl}={fnum('idf1', (bests[k] or {}).get('idf1'))}" for k, lbl in DATASETS))


if __name__ == "__main__":
    main()
