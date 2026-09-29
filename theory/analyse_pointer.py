"""Exploratory: pointer taxonomy from exp2 membership logits (top sweep head). Row k = thought h_{k-1}."""
import json, numpy as np
M = json.load(open("theory/results/exp2_membership.json"))
rows = {}
for f in ("results/p30.json", "results/p_l.json"): rows.update(json.load(open(f))["rows"])
def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool); p, n = x[y], x[~y]
    return float(((p[:, None] > n[None]) + 0.5 * (p[:, None] == n[None])).mean())
def ptr(L, thr=1.0):
    p = [int(np.argmax(r)) + 1 if r.max() >= thr else 0 for r in L]; a = [float(r.max()) for r in L]; return p, a
out = []
for name, o in M.items():
    L = np.array(o["L"]); K = o["K"]
    Lh = L[:, o["heads"][0], :]                     # top sweep head
    Lm = L.max(1)                                   # max over heads
    p, a = ptr(Lh)
    # rows index: row r (0-based) = thought h_r fed into step r+1
    pK1, pK = p[K - 1], p[K]                        # h_{K-1} (last fed back) vs h_K (dangling)
    aK1, aK = a[K - 1], a[K]
    reach = max(p[K:])
    first_bad = next((r for r in range(1, 12) if p[r] < p[r - 1] or p[r] == 0), None)
    typ = ("die" if aK < 1.0 or min(a[K:]) < 1.0 and reach <= pK1 else "revert" if pK < pK1 else "stall" if pK == pK1 and reach <= pK1 else "advance")
    r = rows[name]
    out.append(dict(name=name, K=K, extrap=r["extrap"], P7=r["P7"], d7=r["d7"], p=p, a=a, adv=pK - pK1, gain=aK / max(aK1, 1e-6),
                    reach=reach, reach_gain=reach - pK1, first_bad=first_bad, typ=typ,
                    peak=max(a[:K]), aK_rel=aK / max(a[:K])))
y = [o["extrap"] for o in out]
print("n", len(out), "extrap", sum(y))
for key in ("P7", "adv", "gain", "reach_gain", "aK_rel"):
    print("AUC %-10s %.3f" % (key, auc([o[key] for o in out], y)))
from collections import Counter
print("type x extrap:", Counter((o["typ"], o["extrap"]) for o in out))
print("first_bad step (1=h1..) among non-extrap:", sorted(Counter(o["first_bad"] - o["K"] if o["first_bad"] else 99 for o in out if not o["extrap"]).items()))
print("first_bad rel K among extrap:", sorted(Counter(o["first_bad"] - o["K"] if o["first_bad"] else 99 for o in out if o["extrap"]).items()))
for o in sorted(out, key=lambda o: (o["extrap"], o["P7"])):
    print("%-9s K%d %s P7 %.2f %-7s ptr %s | amp %s" % (o["name"], o["K"], "EXT" if o["extrap"] else "   ", o["P7"], o["typ"],
          " ".join(str(x) for x in o["p"]), " ".join("%.1f" % x for x in o["a"])))
json.dump(out, open("theory/results/pointer_taxonomy.json", "w"), indent=1)
