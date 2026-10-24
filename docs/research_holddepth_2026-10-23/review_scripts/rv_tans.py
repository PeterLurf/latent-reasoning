"""REVIEW_HD: t_ans histograms by n (is there a pass-index horizon?) from hd_check_cpu.<run>.json; also 2x2 exact p-values for PAR enrichment.
    python -B review_scripts/rv_tans.py"""
import json, os, math
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(HERE)
def load(rn):
    for d in (D, HERE):
        p = os.path.join(d, "hd_check_cpu.%s.json" % rn)
        if os.path.exists(p): return json.load(open(p))["deep"]
def fisher2(a, b, c, d):   # two-sided exact p for [[a,b],[c,d]]
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def pr(x): return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = pr(a); lo = max(0, c1 - (n - r1)); hi = min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9))
out = {}
for rn in ["s213_d3t_s1", "s216_d3t_s2", "s224_rd_s1", "s224_rd_s2", "s224_rd_s3", "s224_fh_s1", "s224_fh_s2", "s224_rdft_s1", "s224_rdft_s2", "s224_ftctrl_s2", "s224_fhft_s2"]:
    o = load(rn)
    if o is None: continue
    out[rn] = {}
    for n in ("9", "10", "11"):
        c = Counter((r["t_ans"] if r["t_ans"] is not None else -1) for r in o[n] if r["ans"][int(n) + 2] == r["y"])
        out[rn][n] = {"right": sum(c.values()), "t_ans<=8": sum(v for k, v in c.items() if 0 <= k <= 8), "t_ans=9": c.get(9, 0), "t_ans=10": c.get(10, 0), "t_ans=11": c.get(11, 0), "t_ans>=12": sum(v for k, v in c.items() if k >= 12), "none": c.get(-1, 0)}
pairs = {}
for b, a in [("s213_d3t_s1", "s224_rd_s1"), ("s216_d3t_s2", "s224_rd_s2"), ("s213_d3t_s1", "s224_fh_s1"), ("s216_d3t_s2", "s224_fh_s2"), ("s213_d3t_s1", "s224_rdft_s1"), ("s216_d3t_s2", "s224_rdft_s2"), ("s216_d3t_s2", "s224_ftctrl_s2"), ("s216_d3t_s2", "s224_fhft_s2")]:
    ob, oa = load(b), load(a)
    if ob is None or oa is None: continue
    t = [0, 0, 0, 0]
    for n in ("9", "10", "11"):
        N = int(n)
        for rb, ra in zip(ob[n], oa[n]):
            if rb["ans"][N + 2] != rb["y"]: continue
            ah = rb["t_ans"] is not None and rb["t_ans"] <= N - 1; ok = ra["ans"][N + 2] == ra["y"]
            t[(0 if ah else 2) + (0 if ok else 1)] += 1
    pairs[a + "_vs_" + b] = {"base_ahead: arm right/wrong": t[:2], "base_right_not_ahead: arm right/wrong": t[2:], "P_keep_ahead": round(t[0] / sum(t[:2]), 3), "P_keep_notahead": round(t[2] / sum(t[2:]), 3), "fisher_p": round(fisher2(*t), 5)}
out["pairs"] = pairs
json.dump(out, open(os.path.join(HERE, "rv_tans.out.json"), "w", newline="\n"), indent=1)
for k, v in out.items():
    print(k, json.dumps(v))
