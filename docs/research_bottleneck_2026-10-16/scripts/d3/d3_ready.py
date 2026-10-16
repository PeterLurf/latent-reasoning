"""D3 $0 CPU (1 thread): structural statistics of Experiments' fresh grid (B) items (seed 21600 + m, 2,000 per m; exact replica of the generator),
per (m, T) cell: mean #ready operators (Lemma 0: nodes with two leaf children), the leftmost handle's slot, the longest flat same-level chain,
bracket depth; then cell-level fits of N1's (CARRY) exact on these vs m and T (which structural quantity carries N1's graceful count decay)."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, sys.path[0])
from d3_split import generate, render, _leftmost_ready, n_ops, depth_max  # noqa: E402


def n_ready(nd):
    if nd.op is None:
        return 0
    if nd.l.op is None and nd.r.op is None:
        return 1
    return n_ready(nd.l) + n_ready(nd.r)


def handle_slot(nd):
    from d3_split import handle_end
    return handle_end(nd)[0]


def main():
    cells = defaultdict(list)
    for m in range(1, 17):
        rng = np.random.default_rng(21600 + m)
        for _ in range(2000):
            r, _t = generate(rng, m); s = render(r); T = len(s) + 2
            cells[(m, T)].append((n_ready(r), handle_slot(r), depth_max(s), s.count(" (")))
    obs = {}
    for c in ("s211_vb_carry_s1", "s211_vb_pa_s1"):
        d = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_steppos.json" % c))["fresh_m_T"]
        obs[c] = {tuple(int(z) for z in k.split("_")): (v["n_steps"], v["exact"]) for k, v in d.items()}
    rows = []
    for (m, T), v in sorted(cells.items()):
        a = np.array(v, dtype=float)
        n1 = obs["s211_vb_carry_s1"].get((m, T)); n2 = obs["s211_vb_pa_s1"].get((m, T))
        rows.append({"m": m, "T": T, "n": len(v), "ready": a[:, 0].mean(), "hslot": a[:, 1].mean(), "dmax": a[:, 2].mean(), "b": a[:, 3].mean(),
                     "n1": None if n1 is None else n1[1], "n1_n": None if n1 is None else n1[0], "n2": None if n2 is None else n2[1]})
    # weighted least squares of logit(N1 exact) on standardised (m, T) vs (ready, hslot) over cells with n >= 50 and m >= 10
    R = [r for r in rows if r["n"] >= 50 and r["m"] >= 10 and r["n1"] is not None]
    y = np.array([np.log((r["n1"] * r["n"] + 0.5) / ((1 - r["n1"]) * r["n"] + 0.5)) for r in R]); w = np.array([r["n"] for r in R], dtype=float)

    def fit(cols):
        X = np.column_stack([np.ones(len(R))] + [np.array([r[c] for r in R], dtype=float) for c in cols]); W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y); res = y - X @ beta; return {"beta": beta.round(3).tolist(), "wsse": float((w * res ** 2).sum())}
    fits = {"m": fit(["m"]), "T": fit(["T"]), "m+T": fit(["m", "T"]), "ready": fit(["ready"]), "m+ready": fit(["m", "ready"]), "hslot": fit(["hslot"]),
            "m+hslot": fit(["m", "hslot"]), "ready+hslot": fit(["ready", "hslot"])}
    out = {"rows": rows, "n1_logit_fits_m_ge_10": fits, "n_cells": len(R)}
    json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
    for r in rows:
        if r["m"] >= 9 and r["n"] >= 50:
            print(r["m"], r["T"], r["n"], "ready %.2f hslot %.1f dmax %.2f" % (r["ready"], r["hslot"], r["dmax"]), "N1", r["n1"] and round(r["n1"], 3), "N2", r["n2"] and round(r["n2"], 3))
    print(json.dumps(fits))


if __name__ == "__main__":
    main()
