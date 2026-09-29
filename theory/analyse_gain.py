"""Exploratory: per-hop pointer gain g (in-distribution advances, depths 2..5) vs reach after the retrieval block."""
import json, numpy as np
M = json.load(open("theory/results/exp2_membership.json")); B = {r["n"]: r for r in json.load(open("theory/results/block_summary.json"))}
def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool); p, n = x[y], x[~y]
    return float(((p[:, None] > n[None]) + 0.5 * (p[:, None] == n[None])).mean())
def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
out = []
for n, o in M.items():
    K = o["K"]; L = np.array(o["L"])[:, o["heads"][0], :]
    p = [int(np.argmax(r)) + 1 for r in L]; a = [float(r.max()) for r in L]
    # amplitude at each depth's first arrival among the trained inputs h_0..h_{K-1}
    first = {}
    for k in range(K):
        if p[k] not in first: first[p[k]] = a[k]
    ds = sorted(d for d in first if 1 <= d <= 5)
    ratios = [first[d + 1] / first[d] for d in ds if d + 1 in first and first[d] > 0.5]
    g = float(np.exp(np.mean(np.log(np.clip(ratios, 1e-3, None))))) if ratios else np.nan
    out.append(dict(n=n, g=g, a1=first.get(1, np.nan), nr=len(ratios), ext=B[n]["ext"], retr=B[n]["retr"], base=B[n]["base"], blk=B[n]["blk"]))
ok = [o for o in out if not np.isnan(o["g"])]
print("models with a gain estimate:", len(ok), "/", len(out))
print("gain: extrapolators %.3f  non %.3f" % (np.median([o["g"] for o in ok if o["ext"]]), np.median([o["g"] for o in ok if not o["ext"]])))
print("AUC(g) for extrapolation: %.3f" % auc([o["g"] for o in ok], [o["ext"] for o in ok]))
print("Spearman(g, far8-10 after retrieval block): %.3f (all), %.3f (local-type only)" % (
    spear([o["g"] for o in ok], [o["blk"] for o in ok]), spear([o["g"] for o in ok if not o["retr"]], [o["blk"] for o in ok if not o["retr"]])))
yb = [o["blk"] >= 0.6 for o in ok]
print("AUC(g) for far8-10 >= 0.60 after block: %.3f (n pos %d)" % (auc([o["g"] for o in ok], yb), sum(yb)))
# combined in-budget rule: local (low retrieval) and g high
F = json.load(open("theory/results/share_features.json"))
for o in ok: o["retrK"] = F[o["n"]]["retr_K"]
score = [o["g"] - o["retrK"] for o in ok]
print("AUC(g - retr_K) for extrapolation: %.3f" % auc(score, [o["ext"] for o in ok]))
for o in sorted(ok, key=lambda o: o["g"]):
    print("%-9s %s %s g %.2f (n %d) a1 %.1f | far8-10 base %.2f blocked %.2f" % (o["n"], "EXT" if o["ext"] else "   ", "retr " if o["retr"] else "local", o["g"], o["nr"], o["a1"], o["base"], o["blk"]))
json.dump(out, open("theory/results/gain.json", "w"), indent=1, default=float)
