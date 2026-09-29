import json, numpy as np
R = json.load(open("theory/results/exp5_block.json")); F = json.load(open("theory/results/share_features.json"))
def m810(o, t): return float(np.mean(o[t]["far"][1:4]))     # distances 8..10
rows = []
for n, o in R.items():
    f = F[n]; retr = f["retr_max_tr"] >= 0.5 or f["retr_K"] >= 0.5
    rows.append(dict(n=n, ext=o["extrap"], retr=retr, typ=f["typ"], base=m810(o, "base"), blk=m810(o, "bL2@K"),
                     blk1=m810(o, "bL2@1"), blk12=m810(o, "bL12@K"), d7b=o["base"]["far"][0], d7k=o["bL2@K"]["far"][0]))
def summ(sel, lab):
    s = [r for r in rows if sel(r)]
    if not s: return
    b = np.array([r["base"] for r in s]); k = np.array([r["blk"] for r in s])
    print("%-38s n=%2d  far8-10 base %.3f -> blocked@K %.3f | reach>=0.60: base %2d -> blocked %2d | improved>=0.05: %d" % (
        lab, len(s), b.mean(), k.mean(), (b >= 0.6).sum(), (k >= 0.6).sum(), (k - b >= 0.05).sum()))
summ(lambda r: not r["ext"], "non-extrapolators (all)")
summ(lambda r: not r["ext"] and r["retr"], "  retrieval-type (retr share >= 0.5)")
summ(lambda r: not r["ext"] and not r["retr"], "  local-type")
summ(lambda r: r["ext"], "extrapolators (all)")
summ(lambda r: r["ext"] and r["retr"], "  retrieval-type")
summ(lambda r: r["ext"] and not r["retr"], "  local-type")
all_b = sum(r["base"] >= 0.6 for r in rows); all_k = sum(max(r["base"], r["blk"]) >= 0.6 for r in rows)
print("models with far8-10 >= 0.60: base %d/59; after blocking %d/59 (blocking alone %d/59)" % (all_b, all_k, sum(r["blk"] >= 0.6 for r in rows)))
print("d7 >= 0.60: base %d, blocked %d" % (sum(r["d7b"] >= 0.6 for r in rows), sum(r["d7k"] >= 0.6 for r in rows)))
print("\nnot rescued non-extrapolators:", [(r["n"], r["typ"], "retr" if r["retr"] else "local", round(r["blk"], 2)) for r in rows if not r["ext"] and r["blk"] < 0.58])
print("extrapolators hurt (blk < base - 0.03):", [(r["n"], round(r["base"], 2), round(r["blk"], 2)) for r in rows if r["ext"] and r["blk"] < r["base"] - 0.03])
json.dump(rows, open("theory/results/block_summary.json", "w"), indent=1)
