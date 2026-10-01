"""Unit tests for bm/data.py (THEORY §60): the exact composer must reproduce every true table (1.000) on every split,
and the true final tables must satisfy the four reachability-closure laws.  python bm/test_bm.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bm.data import (HELDOUT_FAMILIES, PATTERNS, TRAIN_FAMILIES, make_set, row_mask, run_exact,  # noqa: E402
                     true_table)


def laws(T, nP):
    """Extensivity, monotonicity, idempotence, union preservation on a thresholded table."""
    rm, _ = row_mask(nP)
    f = {}
    for r in range(16):
        if rm[r]:
            f[PATTERNS[r]] = tuple(int(T[r, j] > 0.5) if j < nP else 0 for j in range(4))
    idx = lambda s: next(r for r in range(16) if PATTERNS[r] == s)
    ok = True
    for a, fa in f.items():
        ok &= all(fa[j] >= a[j] for j in range(4))                                       # extensive
        ok &= f[fa] == fa                                                                 # idempotent
        for b, fb in f.items():
            if all(a[j] <= b[j] for j in range(4)):
                ok &= all(fa[j] <= fb[j] for j in range(4))                              # monotone
            u = tuple(max(a[j], b[j]) for j in range(4))
            ok &= f[u] == tuple(max(fa[j], fb[j]) for j in range(4))                     # union preserving
    return bool(ok)


def check(tasks, name):
    exact = 0; law_ok = 0; lab = 0
    for t in tasks:
        T = run_exact(t["edges"], t["pinned"], [frozenset(s) for s in t["nodes"]], t["kids"], t["ports"])
        TR = np.array(t["table"], np.float32)
        exact += bool((T[-1] == TR).all()); law_ok += laws(TR, len(t["pinned"]))
        lab += T[-1][t["row"], t["col"]] == t["label"]
    n = len(tasks)
    print(f"{name}: n {n} exact-composer table match {exact / n:.3f} label {lab / n:.3f} laws {law_ok / n:.3f}")
    assert exact == n and lab == n and law_ok == n, name


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sets = {"train": make_set(rng, 400, TRAIN_FAMILIES, 2, 8, "train"),
            "heldout_motifs": make_set(rng, 300, HELDOUT_FAMILIES, 2, 8, "train"),
            "heldout_orders": make_set(rng, 300, TRAIN_FAMILIES, 6, 8, "heldout"),
            "size_9_16": make_set(rng, 200, TRAIN_FAMILIES, 9, 16, "any")}
    for k, (tasks, rej) in sets.items():
        check(tasks, k)
        print("   rejections", {f: (rej["tried"][f], rej["rejected"].get(f, {})) for f in rej["tried"]})
        print("   label balance %.3f, |P| counts %s" % (np.mean([t["label"] for t in tasks]),
              {p: sum(len(t["pinned"]) == p for t in tasks) for p in (2, 3, 4)}))
    # schedule-independence of the exact composer: two schedules of the same region give the same table
    from bm.data import random_schedule, schedule_tables
    t = sets["heldout_orders"][0][0]
    for mode in ("train", "heldout"):
        s = random_schedule(rng, t["edges"], set(t["pinned"]), mode)
        if s:
            p, _ = schedule_tables(t["edges"], t["pinned"], *s)
            assert (run_exact(t["edges"], t["pinned"], s[0], s[1], p)[-1] == np.array(t["table"])).all()
    print("ALL TESTS PASSED")
