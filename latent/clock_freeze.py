"""Exploratory follow-up to clock_swap.py: how far does a frozen clock let the search run?

Step embedding held at j (step_emb[min(k, j)]) for K' latent steps; balanced accuracy at distances 1..12
on long-chain graphs, with a graph-level bootstrap interval at distances 7, 9 and 11.

    python -m latent.clock_freeze runs/main_s3 --out results/clock_freeze.json --device cuda:1
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .competence import load
from .graphs import NPB
from .length_gen import long_chain_graphs
from .clock_swap import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n", type=int, default=768)
    a = ap.parse_args()
    res = {}
    for run_ in a.runs:
        m, c = load(run_, a.device)
        K = c["K"]
        rng = np.random.default_rng(707)
        g, br = long_chain_graphs(rng, a.n, [2, 4, 6])
        bi = np.arange(a.n)
        q = np.concatenate([g.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)
        R = {}
        for j in [None, 1, 2, 3, 4]:
            for Kp in (8, 12, 16, 20):
                idx = (lambda k: min(k, K)) if j is None else (lambda k, j=j: min(k, j))
                p = run(m, g, q, Kp, idx, a.device)[0].reshape(a.n, 4, NPB)
                pa, pb, pc, pd = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
                acc = {1 + d: 0.5 * (pa[:, d].mean() + 1 - pd[:, d].mean()) for d in range(NPB)}
                acc.update({NPB + 1 + d: 0.5 * (pb[:, d].mean() + 1 - pc[:, d].mean()) for d in range(NPB)})
                idx_b = np.random.default_rng(0).integers(0, a.n, (1000, a.n))
                ci = {}
                for d in (0, 2, 4):
                    ba = 0.5 * (pb[:, d][idx_b].mean(1) + 1 - pc[:, d][idx_b].mean(1))
                    ci[NPB + 1 + d] = [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))]
                key = f"{'normal' if j is None else f'freeze_{j}'}_K{Kp}"
                R[key] = {"acc": {int(k): float(v) for k, v in acc.items()}, "ci": ci}
                print(f"{run_} {key:14s} d1-12 " + " ".join(f"{acc[d]:.2f}" for d in range(1, 13)), flush=True)
        res[run_] = R
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
