"""PROOFS_TS check P5b (CPU, no model): independent re-fit of the POS-PROP / POS-ADD panels of TS_draft §3.5 (per-item logistic in rows, floor c,
least squares on a_0(9..12) of s170_cot_s1's bf16 curve; T_0 147, T_1 311, T_2 527), on a finer grid, then n50 and delta vs the latent 15.46 / 20.54.
Also the same panels with the pool-size row maxima (313 / 531). python -B pc_panels.py"""
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"; sys.dont_write_bytecode = True
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested170 as N  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "pc_panels.json")
LAT = {1: 15.460207612456747, 2: 20.538983050847456}


def n_cross(a, thr):
    ns = sorted(a)
    if a[ns[0]] < thr:
        return ns[0] - 0.5
    for n0, n1 in zip(ns, ns[1:]):
        if a[n0] >= thr > a[n1]:
            return n0 + (a[n0] - thr) / (a[n0] - a[n1])
    return ns[-1] + 0.5


def main():
    TE = N.test_set(1000, n_max=24); tot = {n: [] for n in range(1, 25)}
    for it in TE:
        tot[it["n"]].append(len(it["syms"]) + 2 + len(N.cot_ids(it["root"])))
    tot = {n: np.array(v, float) for n, v in tot.items()}
    a0 = {int(k): v for k, v in json.load(open("D:/Latent/boxes/runs_fsq/s170_cot_s1/eval.json"))["acc_by_n"].items()}
    c = float(np.mean([a0[n] for n in (13, 14, 15, 16)])); res = {"c": c}
    for mode in ("prop", "add"):
        g0 = np.linspace(1.0, 2.0, 1001) if mode == "prop" else np.linspace(0, 120, 1201)
        gw = np.linspace(0.005, 0.3, 296) if mode == "prop" else np.linspace(0.5, 40, 396)
        err = np.zeros((len(g0), len(gw)))
        for j, w_ in enumerate(gw):
            for n in (9, 10, 11, 12):
                x = tot[n] / 147 if mode == "prop" else tot[n] - 147
                p = c + (1 - c) / (1 + np.exp(-(g0[:, None] - x[None, :]) / w_)); err[:, j] += (p.mean(-1) - a0[n]) ** 2
        i, j = np.unravel_index(np.argmin(err), err.shape); x0, w = float(g0[i]), float(gw[j]); out = {"x0": x0, "w": w, "sse": float(err[i, j])}
        for tag, TR in (("draft_rows", {1: 311, 2: 527}), ("pool_rows", {1: 313, 2: 531})):
            for r in (1, 2):
                a = {}
                for n in range(1, 25):
                    x = tot[n] / TR[r] if mode == "prop" else tot[n] - TR[r]; a[n] = float((c + (1 - c) / (1 + np.exp(-(x0 - x) / w))).mean())
                B = 4 * r + 8; out["%s_r%d" % (tag, r)] = {"n50": n_cross(a, 0.5), "d": n_cross(a, 0.5) - LAT[r], "M": float(np.mean([a[n] for n in range(B + 1, B + 5)]))}
        res[mode] = out; print(mode, out, flush=True)
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
