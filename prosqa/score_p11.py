"""Score PROTOCOL11 (docs/PROTOCOL11.md). Inputs prosqa/res_p11/<model>.json (bidir_p11.py).
Bidirectional answer at frontier size m: the option whose backward frontier alone meets the forward frontier, else
the forward answer. Gain = acc(bidir) - acc(forward).
P11-1: mean gain over the 6 step-1 models at d11-14 >= 0.10 and gain > 0 in >= 5/6 -> supported; mean <= 0 -> not
supported; else inconclusive. P11-2: median negative-hit share at d11-14 <= 0.30. P11-3: median gain at d3-6 >= -0.02.
    python prosqa/score_p11.py
"""
import json
import os

import numpy as np

R = "prosqa/res_p11"
STEP1 = [f"p8b1_{a}_s{s}" for a in ("ckpt", "ctl") for s in (1, 2, 3)]


def stats(rows, m, lo, hi):
    sel = [r for r in rows if lo <= r["depth"] <= hi]
    def bid(r):
        t, n = r[f"hit_t_m{m}"], r[f"hit_n_m{m}"]
        return 1.0 if (t and not n) else -1.0 if (n and not t) else r["fwd"]
    fwd = float(np.mean([r["fwd"] > 0 for r in sel])); bi = float(np.mean([bid(r) > 0 for r in sel]))
    return {"n": len(sel), "fwd": fwd, "bidir": bi, "gain": bi - fwd,
            "hit_t": float(np.mean([r[f"hit_t_m{m}"] for r in sel])), "hit_n": float(np.mean([r[f"hit_n_m{m}"] for r in sel]))}


def main():
    out = {}
    for f in sorted(os.listdir(R)):
        if f.endswith(".json"):
            rows = json.load(open(f"{R}/{f}"))["rows"]; name = f[:-5]
            out[name] = {f"m{m}": {"d11-14": stats(rows, m, 11, 14), "d7-10": stats(rows, m, 7, 10), "d3-6": stats(rows, m, 3, 6)}
                         for m in (10, 5, 15)}
    have = [n for n in STEP1 if n in out]
    res = {"models": out, "n_step1": len(have)}
    if have:
        g = [out[n]["m10"]["d11-14"]["gain"] for n in have]
        mg = float(np.mean(g))
        res["P11-1"] = {"gains": g, "mean": mg, "decision": ("supported" if mg >= 0.10 and sum(x > 0 for x in g) >= 5 and len(have) == 6
                                                            else "not supported" if mg <= 0 else "inconclusive")}
        hn = float(np.median([out[n]["m10"]["d11-14"]["hit_n"] for n in have]))
        res["P11-2"] = {"median_hit_n": hn, "met": hn <= 0.30}
        g3 = float(np.median([out[n]["m10"]["d3-6"]["gain"] for n in have]))
        res["P11-3"] = {"median_gain_d3-6": g3, "met": g3 >= -0.02}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/p11.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=1))
    for n, v in out.items():
        print(n, " | ".join(f"m{m[1:]} " + " ".join(f"{w}: f {v[m][w]['fwd']:.2f} b {v[m][w]['bidir']:.2f} hn {v[m][w]['hit_n']:.2f}"
                                                      for w in ("d3-6", "d11-14")) for m in ("m10", "m5")))


if __name__ == "__main__":
    main()
