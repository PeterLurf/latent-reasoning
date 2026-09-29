"""Toy simulation for Th3 (loop-gain reach): analog vs restoring pointer propagation fitted on depth <= D.

A pointer amplitude moves along a chain one hop per step. Each hop: a <- f(g * a + noise). Every node (reachable chain and
decoy chains) also receives fresh noise each step, which then propagates with the same map (decoy leakage).
Readout: node "reachable" if its amplitude when the front passes (max over steps) exceeds theta.
  analog : f(x) = x                     (graded update; parameters g, theta)
  digital: f(x) = sigmoid(beta*(x - tau)) (restoring update; parameters g, beta, tau, theta)
Parameters are fitted by grid search to minimise balanced error on depths 1..D only; reach = deepest depth with
balanced accuracy >= 0.9 on chains of length 40. Output: reach vs training depth D for both families.
    python theory/sim_gain.py
"""
import itertools
import json

import numpy as np

rng = np.random.default_rng(0)
L, N, SIG = 40, 4000, 0.08      # chain length, chains per class, per-hop noise sd


def run(f, a0=1.0):
    """Amplitude at each depth of a reachable chain (signal injected at depth 1) and of a decoy chain (noise only)."""
    pos = np.zeros((N, L)); neg = np.zeros((N, L))
    a = np.full(N, a0) + SIG * rng.standard_normal(N); b = SIG * rng.standard_normal(N)
    for d in range(L):
        pos[:, d], neg[:, d] = a, b
        a = f(a) + SIG * rng.standard_normal(N)
        b = f(b) + SIG * rng.standard_normal(N)
    return pos, neg


def bal(pos, neg, theta):
    return 0.5 * ((pos > theta).mean(0) + (neg <= theta).mean(0))


def fit(family, D):
    best = None
    if family == "analog":
        grid = [dict(g=g) for g in np.linspace(0.5, 1.2, 36)]
        mk = lambda p: (lambda x: p["g"] * x)
    else:
        grid = [dict(g=g, beta=be, tau=t) for g, be, t in itertools.product(np.linspace(0.6, 1.4, 9), (4, 8, 16), (0.2, 0.35, 0.5))]
        mk = lambda p: (lambda x: 1 / (1 + np.exp(-p["beta"] * (p["g"] * x - p["tau"]))))
    for p in grid:
        pos, neg = run(mk(p))
        for theta in np.linspace(0.05, 0.9, 35):
            acc = bal(pos, neg, theta)
            err = 1 - acc[:D].mean()
            if best is None or err < best[0] - 1e-9 or (abs(err - best[0]) < 1e-9 and p.get("g", 0) < best[1].get("g", 0)):
                best = (err, p, theta, acc)
    err, p, theta, acc = best
    ok = acc >= 0.9
    reach = int(np.argmin(ok)) if not ok.all() else L
    return {"D": D, "train_err": float(err), "params": p, "theta": float(theta), "reach": reach,
            "acc_at": {int(d + 1): float(acc[d]) for d in (D - 1, D, D + 2, D + 5, min(L - 1, D + 10))}}


if __name__ == "__main__":
    out = {}
    for fam in ("analog", "digital"):
        out[fam] = [fit(fam, D) for D in (3, 4, 5, 6, 8, 10)]
        for r in out[fam]:
            print(fam, "D=%2d train err %.3f reach %2d params %s theta %.2f" % (r["D"], r["train_err"], r["reach"],
                  {k: round(float(v), 3) for k, v in r["params"].items()}, r["theta"]))
    json.dump(out, open("theory/results/sim_gain.json", "w"), indent=1, default=float)
