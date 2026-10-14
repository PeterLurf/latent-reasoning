"""N6: progress-contraction kappa(n) implied by the per-pass handle accuracy of eval_s187_deepaux_items_n9_12.json (read-only), margin model
p_t / p_{t-1} = Phi(a kappa^t) on t = 3..8 (the trained aux range), per model and n; then the slope of log kappa vs mean think length L."""
import json, numpy as np
from statistics import NormalDist
N = NormalDist(); R = "D:/Latent/boxes/runs_fsq/"; out = {}
for r in ["s185_rms_s1", "s185_rms_s2", "s185_rms_s3", "s182_window_s2", "s187_nopass_s1"]:
    e = json.load(open(R + r + "/eval_s187_deepaux_items_n9_12.json")); its = e["items"]; rows = []
    for n in (9, 10, 11, 12):
        sub = [i for i in its if i["n"] == n]; L = float(np.mean([i["think_tokens"] for i in sub]))
        acc = [np.mean([i["handle_by_t"][str(t)] for i in sub]) for t in range(1, 9)]
        ts = np.arange(3, 9); ratios = [min(acc[t - 1] / acc[t - 2], 0.9995) for t in ts]
        y = np.log([N.inv_cdf(p) for p in ratios]); A = np.c_[np.ones(len(ts)), ts]; c, *_ = np.linalg.lstsq(A, y, rcond=None)
        rows.append(dict(n=n, L=round(L, 1), kappa=round(float(np.exp(c[1])), 3), acc_t8=round(float(acc[7]), 3)))
    Ls = np.array([x["L"] for x in rows]); lk = np.log([x["kappa"] for x in rows]); sl = float(np.polyfit(Ls, lk, 1)[0])
    out[r] = dict(rows=rows, dlogkappa_dL=round(sl, 4)); print(r, out[r])
out["init_dlogkappa_dL"] = float(np.log(1 / (1 + np.exp(-5.0))))
print("init log alpha per token", out["init_dlogkappa_dL"])
json.dump(out, open("fit_size.json", "w", newline="\n"), indent=1)
