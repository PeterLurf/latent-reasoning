"""First-error hazard of the teacher-forced aux handle (187-3 rows, n 9-12) stratified by step structure. CPU, descriptive."""
import json
import sys

import numpy as np

F = json.load(open(sys.argv[1])); ns = F["n"]; feats = F["feats"]
idx = [i for i, n in enumerate(ns) if 9 <= n <= 12]
MODELS = sys.argv[2:]


def load(m):
    return json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_s187_deepaux_items_n9_12.json" % m))["items"]


def table(m):
    rows = load(m); assert len(rows) == len(idx)
    obs = []   # (t, prov, age_max, held, alive, hh, bdepth, r, err, clean_prefix, ops_clean)
    for k, i in enumerate(idx):
        hb = rows[k]["handle_by_t"]; f = feats[i]; n = ns[i]; assert rows[k]["n"] == n
        clean = True
        for t in range(1, n + 1):
            if str(t) not in hb:
                break
            e = 1 - hb[str(t)]; s = f[t - 1]
            obs.append((t, s["prov"], s["age_max"], s["held"], s["alive"], s["hh"], s["bdepth"], s["r"], e, clean, n))
            if e:
                clean = False
    return np.array(obs, dtype=float)


def rate(a, mask):
    return (a[mask, 8].mean(), int(mask.sum())) if mask.sum() else (float("nan"), 0)


for m in MODELS:
    a = table(m); c = a[:, 9] == 1; A = a[c]
    print("=== %s  prefix-clean steps %d, unconditional err %.3f" % (m, len(A), a[:, 8].mean()))
    # hazard by t
    print(" t      " + " ".join("%6d" % t for t in range(1, 13)))
    print(" haz    " + " ".join("%6.3f" % rate(A, A[:, 0] == t)[0] for t in range(1, 13)))
    print(" N      " + " ".join("%6d" % rate(A, A[:, 0] == t)[1] for t in range(1, 13)))
    # hazard by prov x t
    for p in (0, 1, 2):
        print(" prov%d  " % p + " ".join(("%6.3f" % rate(A, (A[:, 0] == t) & (A[:, 1] == p))[0]) if rate(A, (A[:, 0] == t) & (A[:, 1] == p))[1] >= 30 else "     ." for t in range(1, 13)))
    # hazard by hh x t
    for h in (1, 2, 3, 4, 5, 6):
        msk = (A[:, 5] == h) if h < 6 else (A[:, 5] >= 6)
        print(" hh%s   " % (str(h) if h < 6 else "6+") + " ".join(("%6.3f" % rate(A, (A[:, 0] == t) & msk)[0]) if rate(A, (A[:, 0] == t) & msk)[1] >= 30 else "     ." for t in range(1, 13)))
    for g in (0, 1, 2, 3, 4):
        msk = (A[:, 2] == g) if g < 4 else (A[:, 2] >= 4)
        print(" age%s  " % (str(g) if g < 4 else "4+") + " ".join(("%6.3f" % rate(A, (A[:, 0] == t) & msk)[0]) if rate(A, (A[:, 0] == t) & msk)[1] >= 30 else "     ." for t in range(1, 13)))
