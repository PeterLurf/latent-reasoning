"""held-load x t, and training-coverage of the (t, held, prov) cell; prefix-clean first-error hazard on 187-3 rows (n 9-12)."""
import json, sys
import numpy as np
F = json.load(open(sys.argv[1])); ns = F["n"]; feats = F["feats"]
idx = [i for i, n in enumerate(ns) if 9 <= n <= 12]
# training coverage proxy: test items n 1..8 (same generator law as training n ~ U{1..8})
cov = {}
for n, f in zip(ns, feats):
    if n <= 8:
        for s in f:
            k = (s["t"], s["held"]); cov[k] = cov.get(k, 0) + 1
tot_t = {}
for (t, h), c in cov.items():
    tot_t[t] = tot_t.get(t, 0) + c
def covfrac(t, h):
    return cov.get((t, h), 0) / max(1, tot_t.get(t, 1))
print("coverage P(held=h | t) in n<=8:")
for t in range(1, 9):
    print(" t%d " % t + " ".join("h%d:%.3f" % (h, covfrac(t, h)) for h in range(0, 5)))
for m in sys.argv[2:]:
    rows = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_s187_deepaux_items_n9_12.json" % m))["items"]
    obs = []
    for k, i in enumerate(idx):
        hb = rows[k]["handle_by_t"]; f = feats[i]; n = ns[i]; clean = True
        for t in range(1, n + 1):
            if str(t) not in hb: break
            e = 1 - hb[str(t)]; s = f[t - 1]
            if clean: obs.append((t, s["held"], s["prov"], s["hh"], e, cov.get((t, s["held"]), 0) > 0 if t <= 8 else False))
            if e: clean = False
    A = np.array(obs, dtype=float)
    print("=== %s" % m)
    for h in range(0, 6):
        line = []
        for t in range(1, 11):
            msk = (A[:, 0] == t) & (A[:, 1] == h)
            line.append(("%6.3f" % A[msk, 4].mean()) if msk.sum() >= 30 else "     .")
        print(" held%d " % h + " ".join(line))
    # within t<=8: seen vs unseen (t, held) cells
    for t in range(3, 9):
        ms = (A[:, 0] == t) & (A[:, 5] == 1); mu = (A[:, 0] == t) & (A[:, 5] == 0)
        print("  t%d seen-cell haz %.3f (N %d) | unseen-cell haz %s (N %d)" % (t, A[ms, 4].mean() if ms.sum() else np.nan, ms.sum(), ("%.3f" % A[mu, 4].mean()) if mu.sum() else "-", mu.sum()))
