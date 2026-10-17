"""RT-2: a decode-free 'finish pass': t_settle = the first t >= 1 from which the coda answer stays equal to coda[n+2] through n+2
(answer-level analogue of t_fin); and t_right = first t with coda[t] == y and coda[u] == y for all u in t..n+2 (correct items only).
Deadline predicts t_settle / t_right pinned near ~9 across n >= 10; no deadline predicts ~n. Also the decode-vs-answer gap: t_settle - t_fin.
Also Audit (b)'s slope test on median t_fin and t_settle over n 9-14 (OLS slope, spread)."""
import json, sys
import numpy as np
OUT = sys.argv[1]; res = {}
def ols(xs, ys):
    xs = np.array(xs, float); ys = np.array(ys, float); return float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
for name, run in (("N1_carry", "D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"), ("N2_pa", "D:/Latent/boxes/runs_fsq/s211_vb_pa_s1")):
    rows = json.load(open(run + "/eval_vbhalt.json"))["rows"]; R = {}
    for n in range(4, 17):
        rs = [r for r in rows if r["n"] == n]; ts = []; tr = []; tfin = []; gap = []
        for r in rs:
            c = r["coda"]; e = n + 2
            s = next(t for t in range(1, e + 1) if all(c[u] == c[e] for u in range(t, e + 1))); ts.append(s)
            if c[e] == r["y"]:
                tr.append(s)
            tf = next((t for t in range(25) if r["alive"][t] == 0), None); tfin.append(99 if tf is None else tf)
            if tf is not None:
                gap.append(s - tf)
        R[str(n)] = {"median_t_settle": float(np.median(ts)), "median_t_right(correct items)": float(np.median(tr)) if tr else None, "n_correct": len(tr),
                     "median_t_fin": float(np.median(tfin)), "mean_settle_minus_tfin": round(float(np.mean(gap)), 2) if gap else None,
                     "p_settle_le_9": round(float(np.mean([s <= 9 for s in ts])), 3), "p_settle_ge_n": round(float(np.mean([s >= n for s in ts])), 3)}
    ns = list(range(9, 15))
    res[name] = {"by_n": R, "slope_tfin_9_14": ols(ns, [R[str(n)]["median_t_fin"] for n in ns]), "spread_tfin_9_14": R["14"]["median_t_fin"] - R["9"]["median_t_fin"],
                 "slope_settle_9_14": ols(ns, [R[str(n)]["median_t_settle"] for n in ns]), "spread_settle_9_14": R["14"]["median_t_settle"] - R["9"]["median_t_settle"]}
json.dump(res, open(OUT, "w", newline="\n"), indent=1)
for name, v in res.items():
    print("==", name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != "by_n"})
    for n, x in v["by_n"].items():
        print("  n", n, x)
