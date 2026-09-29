"""Exploratory: is the search frontier carried by the thought vector? (rewind test)

Long-chain graphs as in latent/length_gen.py (nodes at distances 1..12). In a run of K' = 12 steps, the
thought fed into step j is replaced by the model's own thought from step j - r (same graph, same run);
the key/value cache of every earlier position is kept. If the thought is the search state, the rewound
run should reach as far as an unedited run of 12 - r steps; if the frontier lives in the cache, reach
should be unaffected. Beyond the trained budget the step-K embedding is reused.

    python -m latent.rewind runs/main_s2 runs/r3_pc_gated_gpt2 --out results/rewind.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .competence import load
from .graphs import NPB
from .length_gen import long_chain_graphs


@torch.no_grad()
def run_rewind(m, K, g, q, Kp, device, j=None, r=0):
    T = lambda a: torch.as_tensor(a, device=device)
    n = len(g)
    out = []
    for i in range(0, n, 256):
        sl = slice(i, min(n, i + 256))
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
        th = [h]
        for k in range(1, Kp + 1):
            if j is not None and k == j:
                h = th[k - 1 - r]                    # feed an older thought into step j
            h, cache, _ = m.latent_step(h, min(k, K), cache)
            th.append(h)
        out.append((m.query(cache, T(q[sl])) > 0).cpu().numpy())
    return np.concatenate(out)


def accuracy(pred, n):
    p = pred.reshape(n, 4, NPB)
    pa, pb, pc, pd = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    acc = {1 + d: 0.5 * (pa[:, d].mean() + 1 - pd[:, d].mean()) for d in range(NPB)}
    acc.update({NPB + 1 + d: 0.5 * (pb[:, d].mean() + 1 - pc[:, d].mean()) for d in range(NPB)})
    return {int(k): float(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=1024)
    a = ap.parse_args()
    res = {}
    for run in a.runs:
        m, c = load(run, a.device)
        K = c["K"]
        rng = np.random.default_rng(77)
        g, br = long_chain_graphs(rng, a.n, [2, 4, 6])
        bi = np.arange(a.n)
        q = np.concatenate([g.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)
        R = {"baseline": {}, "rewind": {}}
        for Kp in (4, 6, 8, 10, 12):
            R["baseline"][Kp] = accuracy(run_rewind(m, K, g, q, Kp, a.device), a.n)
        for j in (5, 7, 9):
            for r in (2, 4):
                if j - 1 - r < 0:
                    continue
                R["rewind"][f"j{j}_r{r}"] = accuracy(run_rewind(m, K, g, q, 12, a.device, j=j, r=r), a.n)
        res[run] = R
        fmt = lambda d: " ".join(f"{d[x]:.2f}" for x in range(3, 10))
        print(run, "distances 3..9")
        for Kp, d in R["baseline"].items():
            print(f"  baseline K'={Kp:2d}      {fmt(d)}")
        for key, d in R["rewind"].items():
            print(f"  K'=12 rewind {key:7s} {fmt(d)}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
