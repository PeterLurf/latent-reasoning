# read-only POST-HOC: value of halting = oracle per-n K' vs best single fixed K', from existing eval.json files; plus horizon-map fits
import json, os, math
R = "D:/Latent/boxes/runs_fsq"; OUT = {}
def pooled(a, k, ns, w=None):
    xs = [(a[k][str(n)], (w[str(n)] if w else 1.0)) for n in ns if str(n) in a[k] and a[k][str(n)] is not None]
    return sum(x * y for x, y in xs) / sum(y for _, y in xs)
nest = ["s184_reweight_s1", "s184_reweight_s2", "s185_rms_s1", "s185_rms_s2", "s187_nopass_s1", "s187_nopass_s2", "s190_rmsnopass_s1", "s190_rmsnopass_s2", "s182_window_s1", "s182_window_s2"]
for r in nest:
    a = json.load(open(os.path.join(R, r, "eval.json")))["s170"]["acc_by_n_kprime"]
    fk = [k for k in a if k.isdigit() and int(k) >= 4]
    for lab, ns in (("in n3-8", range(3, 9)), ("mix n3-12", range(3, 13)), ("deep n9-12", range(9, 13))):
        best = max(fk, key=lambda k: pooled(a, k, ns)); orc = sum(max(a[k][str(n)] for k in fk + ["n", "n+2"]) for n in ns) / len(ns)
        line = "%-20s %-10s bestfixed K'%-3s %.3f | K'n %.3f K'n+2 %.3f | oracle-per-n %.3f | K'32 %.3f | VOH(n+2 - best) %+.3f" % (
            r, lab, best, pooled(a, best, ns), pooled(a, "n", ns), pooled(a, "n+2", ns), orc, pooled(a, "32", ns), pooled(a, "n+2", ns) - pooled(a, best, ns))
        print(line); OUT.setdefault(r, {})[lab] = line
    # horizon map: error at K'16 relative to K'8, by trained post-completion horizon j0 = 8 - n (fixed K 8 only)
    if "window" not in r:
        hm = {8 - n: round(a["8"][str(n)] - a["16"][str(n)], 3) for n in range(4, 9)}
        print("   horizon map K'16 loss by j0:", hm)
# ProsQA s191 A
e = json.load(open(os.path.join(R, "s191_stepaux_relabel_s1", "eval.json"))); a = e["s168s2"]["acc_by_depth_kprime"]; w = e["n_by_depth"]
for lab, ds in (("train d3-10", range(3, 11)), ("all d3-14", range(3, 15)), ("deep d11-14", range(11, 15))):
    row = " ".join("K'%s %.3f" % (k, pooled(a, k, ds, w)) for k in ["8", "10", "12", "14", "16"])
    orc = sum(max(a[k][str(d)] for k in ["8", "10", "12", "14", "16"]) * w[str(d)] for d in ds) / sum(w[str(d)] for d in ds)
    print("s191 %-12s %s | oracle-per-depth %.3f" % (lab, row, orc))
print("s191 K'16 loss vs K'10 by j0 = 10 - d:", {10 - d: round(a["10"][str(d)] - a["16"][str(d)], 3) for d in range(3, 11)})
print("s191 K'14 loss vs K'10 by j0:", {10 - d: round(a["10"][str(d)] - a["14"][str(d)], 3) for d in range(3, 11)})
print("s191 K'12 loss vs K'10 by j0:", {10 - d: round(a["10"][str(d)] - a["12"][str(d)], 3) for d in range(3, 11)})
