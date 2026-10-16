import json, sys
import numpy as np
F = json.load(open(sys.argv[1])); ns = F["n"]; feats = F["feats"]
idx = [i for i, n in enumerate(ns) if 9 <= n <= 12]
for m in sys.argv[2:]:
    rows = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_s187_deepaux_items_n9_12.json" % m))["items"]
    obs = []
    for k, i in enumerate(idx):
        hb = rows[k]["handle_by_t"]; f = feats[i]; n = ns[i]
        for t in range(1, min(n, 8) + 1):
            if str(t) not in hb: break
            s = f[t - 1]; e = 1 - hb[str(t)]; obs.append((t, "+-*/".index(s["op"]), s["hh"], e))
            if e: break
    A = np.array(obs, float)
    out = []
    for o in range(4):
        # hazard by op, standardised over t-mix: mean over t of per-t hazard
        hz = [A[(A[:, 0] == t) & (A[:, 1] == o), 3].mean() for t in range(3, 9)]
        out.append("%s %.3f" % ("+-*/"[o], np.mean(hz)))
    print("%-18s t3-8 mean hazard by op: %s" % (m, "  ".join(out)))
