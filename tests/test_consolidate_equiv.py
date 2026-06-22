"""Differential test: fast_consolidate vs IdentityGallery.consolidate (reference).

Builds thousands of randomized IdentityGallery instances exercising every cannot_merge
gate (same-frame occ IoU, geo gate, cross-cam simultaneity / travel / overlaps,
merge_free_xcam toggle) plus adversarial exact-cosine ties, and asserts remap AND events
are bit-identical between the reference and fast_consolidate on the SAME object.

Run: PYTHONPATH=/home/abarnhill@novateur.com/git/VLINCS/vlincs_gallery \
     /home/abarnhill@novateur.com/git/VLINCS/vlincs_gallery/.venv/bin/python test_fast_consolidate.py
"""
import sys
import numpy as np

sys.path.insert(0, "/tmp/fc_impl2")
from vlincs_gallery.gallery import IdentityGallery


def new_gallery(emb_dim, *, same_box_iou=0.35, geo_max_m=0.0, merge_free_xcam=False,
                sim_window_ms=1000, max_speed=5.0, overlaps=None, dist=None):
    g = IdentityGallery(
        cam_xy={}, dist=dist or {}, overlaps=overlaps or set(), tau=0.5, max_speed=max_speed,
        sim_window_ms=sim_window_ms, admit_tau=0.5, max_reps=8, same_box_iou=same_box_iou,
        emb_dim=emb_dim,
    )
    g.geo_max_m = geo_max_m
    g.merge_free_xcam = merge_free_xcam
    return g


def rand_box(rng):
    x1 = rng.uniform(0, 100); y1 = rng.uniform(0, 100)
    return [x1, y1, x1 + rng.uniform(5, 40), y1 + rng.uniform(5, 40)]


def build_random_instance(rng, *, force_tie=False, larger=False):
    emb_dim = int(rng.choice([2, 3, 4, 8, 16, 64]))
    n_groups = int(rng.integers(2, 31)) if not larger else int(rng.integers(40, 120))
    same_box_iou = float(rng.choice([0.0, 0.2, 0.35, 0.5, 0.9]))
    geo_on = rng.random() < 0.35
    geo_max_m = float(rng.choice([1.0, 5.0, 50.0, 500.0])) if geo_on else 0.0
    merge_free_xcam = bool(rng.random() < 0.45)
    sim_window_ms = int(rng.choice([0, 100, 1000, 5000]))
    max_speed = float(rng.choice([1.0, 5.0, 50.0]))
    merge_tau = float(rng.choice([0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]))

    cams = [f"C{c}" for c in range(int(rng.integers(1, 5)))]

    overlaps = set()
    dist = {}
    for a in range(len(cams)):
        for b in range(a + 1, len(cams)):
            fs = frozenset((cams[a], cams[b]))
            if rng.random() < 0.5:
                overlaps.add(fs)
            if rng.random() < 0.6:
                dist[fs] = float(rng.uniform(1.0, 200.0))

    g = new_gallery(emb_dim, same_box_iou=same_box_iou, geo_max_m=geo_max_m,
                    merge_free_xcam=merge_free_xcam, sim_window_ms=sim_window_ms,
                    max_speed=max_speed, overlaps=overlaps, dist=dist)

    gids = sorted(rng.choice(np.arange(1, n_groups * 3 + 1), size=n_groups, replace=False).tolist())
    rep_rows = []
    rep_gids = []

    n_base = max(1, n_groups // 3)
    bases = rng.standard_normal((n_base, emb_dim)).astype(np.float32)

    for gi, gid in enumerate(gids):
        nreps = int(rng.integers(1, 6))
        if force_tie and rng.random() < 0.6:
            base = bases[gi % n_base]
            row = base.astype(np.float32)
            for _ in range(nreps):
                rep_rows.append(row.copy())
                rep_gids.append(gid)
        else:
            base = bases[gi % n_base]
            for _ in range(nreps):
                noise = rng.standard_normal(emb_dim).astype(np.float32) * float(rng.uniform(0.0, 0.6))
                rep_rows.append((base + noise).astype(np.float32))
                rep_gids.append(gid)

    g.rep_mat = np.asarray(rep_rows, dtype=np.float32).reshape(-1, emb_dim)
    g.rep_gid = np.asarray(rep_gids, dtype=np.int64)

    n_videos = int(rng.integers(1, 4))
    g.occ = {}
    shared_keys = [(f"V{rng.integers(0, n_videos)}", int(rng.integers(0, 50)))
                   for _ in range(int(rng.integers(0, 6)))]
    for gid in gids:
        occ = {}
        if rng.random() < 0.7:
            for _ in range(int(rng.integers(0, 4))):
                if shared_keys and rng.random() < 0.6:
                    k = tuple(shared_keys[int(rng.integers(0, len(shared_keys)))])
                else:
                    k = (f"V{rng.integers(0, n_videos)}", int(rng.integers(0, 50)))
                occ[k] = rand_box(rng)
        g.occ[gid] = occ

    g.crange = {}
    for gid in gids:
        cr = {}
        for c in cams:
            if rng.random() < 0.6:
                lo = float(rng.integers(0, 20000))
                hi = lo + float(rng.integers(0, 8000))
                cr[c] = [lo, hi]
        g.crange[gid] = cr

    g.world = {}
    base_lat, base_lon = 38.9209, -77.0
    for gid in gids:
        w = {}
        if geo_on or rng.random() < 0.3:
            for _ in range(int(rng.integers(0, 4))):
                s = int(rng.integers(0, 30))
                lst = []
                for _ in range(int(rng.integers(1, 4))):
                    lst.append((base_lat + rng.uniform(-0.01, 0.01),
                                base_lon + rng.uniform(-0.01, 0.01)))
                w[s] = lst
        g.world[gid] = w

    return g, merge_tau


def snapshot(g):
    return (
        g.rep_mat.copy(), g.rep_gid.copy(),
        {k: dict(v) for k, v in g.occ.items()},
        {k: {c: list(r) for c, r in v.items()} for k, v in g.crange.items()},
        {k: {s: list(lst) for s, lst in v.items()} for k, v in g.world.items()},
    )


def assert_unmutated(g, snap):
    rm, rg, occ, crange, world = snap
    assert np.array_equal(g.rep_mat, rm), "rep_mat mutated"
    assert np.array_equal(g.rep_gid, rg), "rep_gid mutated"
    assert g.occ == occ, "occ mutated"
    assert g.crange == crange, "crange mutated"
    assert set(g.world) == set(world), "world keys mutated"
    for k in world:
        assert set(g.world[k]) == set(world[k]), "world sec keys mutated"
        for s in world[k]:
            assert list(g.world[k][s]) == list(world[k][s]), "world sec list mutated"


def main():
    rng = np.random.default_rng(20260611)
    n_cases = 6000
    cases_run = 0
    divergences = []

    for idx in range(n_cases):
        force_tie = (idx % 3 == 0)
        larger = (idx % 500 == 0 and idx > 0)
        g, merge_tau = build_random_instance(rng, force_tie=force_tie, larger=larger)
        cases_run += 1

        snap = snapshot(g)
        ref_remap, ref_events = g._consolidate_ref(merge_tau, return_events=True)
        assert_unmutated(g, snap)

        snap2 = snapshot(g)
        fast_remap, fast_events = g.consolidate(merge_tau, return_events=True)
        try:
            assert_unmutated(g, snap2)
        except AssertionError as e:
            divergences.append(f"case {idx}: fast_consolidate MUTATED g: {e}")

        if ref_remap != fast_remap or ref_events != fast_events:
            divergences.append(
                f"case {idx} merge_tau={merge_tau} mfx={g.merge_free_xcam} "
                f"geo_max_m={g.geo_max_m} sbi={g.same_box_iou} "
                f"n_gids={len(set(g.rep_gid.tolist()))}\n"
                f"  remap_eq={ref_remap == fast_remap} events_eq={ref_events == fast_events}\n"
                f"  ref_events={ref_events}\n  fast_events={fast_events}\n"
                f"  ref_remap={ref_remap}\n  fast_remap={fast_remap}"
            )
            if len(divergences) >= 5:
                break

        r_only_ref = g._consolidate_ref(merge_tau, return_events=False)
        r_only_fast = g.consolidate(merge_tau, return_events=False)
        if r_only_ref != r_only_fast:
            divergences.append(f"case {idx}: return_events=False remap mismatch")
            if len(divergences) >= 5:
                break

    tie_cases = build_explicit_tie_cases()
    for name, g, merge_tau in tie_cases:
        cases_run += 1
        rr, re = g._consolidate_ref(merge_tau, return_events=True)
        fr, fe = g.consolidate(merge_tau, return_events=True)
        if rr != fr or re != fe:
            divergences.append(f"explicit {name}: remap_eq={rr==fr} events_eq={re==fe} "
                               f"ref_events={re} fast_events={fe} ref_remap={rr} fast_remap={fr}")

    g0 = new_gallery(8)
    cases_run += 1
    if g0._consolidate_ref(0.5) != g0.consolidate(0.5):
        divergences.append("edge 0-groups mismatch")
    g1 = new_gallery(8)
    g1.rep_mat = np.ones((3, 8), np.float32); g1.rep_gid = np.array([7, 7, 7], np.int64)
    g1.occ = {7: {}}; g1.crange = {7: {}}; g1.world = {7: {}}
    cases_run += 1
    if g1._consolidate_ref(0.5) != g1.consolidate(0.5):
        divergences.append("edge 1-group mismatch")

    all_passed = len(divergences) == 0
    print(f"cases_run={cases_run}")
    print(f"all_passed={all_passed}")
    if divergences:
        print("DIVERGENCES:")
        for d in divergences:
            print(d)
    else:
        print("ZERO DIVERGENCE")
    return all_passed


def build_explicit_tie_cases():
    cases = []

    # all-three-identical -> winner (20,30) per spec last-wins
    g = IdentityGallery({}, {}, set(), 0.5, 5.0, 1000, 0.5, 8, emb_dim=4)
    v = np.ones((1, 4), np.float32)
    g.rep_mat = np.vstack([v, v, v]).astype(np.float32)
    g.rep_gid = np.array([10, 20, 30], np.int64)
    g.occ = {10: {}, 20: {}, 30: {}}; g.crange = {10: {}, 20: {}, 30: {}}
    g.world = {10: {}, 20: {}, 30: {}}
    cases.append(("all3_identical", g, 0.5))

    # two tie clusters: gids 0,1 dir A ; gids 2,3 dir B -> winner (2,3) per spec
    g2 = IdentityGallery({}, {}, set(), 0.5, 5.0, 1000, 0.5, 8, emb_dim=4)
    a = np.array([1, 0, 0, 0], np.float32); b = np.array([0, 1, 0, 0], np.float32)
    g2.rep_mat = np.vstack([a, a, b, b]).astype(np.float32)
    g2.rep_gid = np.array([0, 1, 2, 3], np.int64)
    g2.occ = {k: {} for k in [0, 1, 2, 3]}
    g2.crange = {k: {} for k in [0, 1, 2, 3]}
    g2.world = {k: {} for k in [0, 1, 2, 3]}
    cases.append(("two_tie_clusters", g2, 0.99))

    return cases


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

def test_consolidate_bit_identical():
    # consolidate (fast) must be bit-identical to _consolidate_ref (original) - same remap AND events.
    assert main(), "fast consolidate diverged from _consolidate_ref"
