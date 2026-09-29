"""Exploratory follow-up to latent/rewind.py: grid over rewind step j and depth r, plus controls.

Conditions at K' total steps (thought fed into step j replaced; cache kept):
  rewind  : own thought from step j - r
  foreign : thought from step j - r of a *different* graph in the batch (rolled by one)
  zero    : thought set to zeros
  repeat  : own thought from step j - 1 fed again (r = 0 identity control is the baseline)
Reports balanced accuracy at distances 1..12 on long-chain graphs.

    python -m latent.rewind_grid runs/r3_pc_gated_gpt2 --out results/rewind_grid_s0.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .competence import load
from .length_gen import long_chain_graphs
from .rewind import accuracy


@torch.no_grad()
def run(m, K, g, q, Kp, device, j=None, r=0, mode="rewind"):
    T = lambda a: torch.as_tensor(a, device=device)
    n = len(g)
    out = []
    for i in range(0, n, 256):
        sl = slice(i, min(n, i + 256))
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
        th = [h]
        for k in range(1, Kp + 1):
            if j is not None and k == j:
                if mode == "rewind":
                    h = th[k - 1 - r]
                elif mode == "foreign":
                    h = torch.roll(th[k - 1 - r], 1, 0)
                elif mode == "zero":
                    h = torch.zeros_like(h)
            h, cache, _ = m.latent_step(h, min(k, K), cache)
            th.append(h)
        out.append((m.query(cache, T(q[sl])) > 0).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=78)
    a = ap.parse_args()
    m, c = load(a.run, a.device)
    K = c["K"]
    rng = np.random.default_rng(a.seed)
    g, br = long_chain_graphs(rng, a.n, [2, 4, 6])
    bi = np.arange(a.n)
    q = np.concatenate([g.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)
    res = {"run": a.run, "seed": a.seed, "baseline": {}, "grid": {}}
    for Kp in (6, 12, 16):
        res["baseline"][Kp] = accuracy(run(m, K, g, q, Kp, a.device), a.n)
    far = lambda d: np.mean([d[x] for x in (7, 8, 9)])
    print(a.run, "baseline far(7-9):", {k: round(far(v), 3) for k, v in res["baseline"].items()})
    for Kp in (12, 16):
        for j in range(3, 12):
            for r in (1, 2, 3, 4, 5, 6):
                if j - 1 - r < 0:
                    continue
                for mode in ("rewind", "foreign") if r in (2, 4) else ("rewind",):
                    acc = accuracy(run(m, K, g, q, Kp, a.device, j=j, r=r, mode=mode), a.n)
                    res["grid"][f"K{Kp}_j{j}_r{r}_{mode}"] = acc
            acc = accuracy(run(m, K, g, q, Kp, a.device, j=j, mode="zero"), a.n)
            res["grid"][f"K{Kp}_j{j}_zero"] = acc
        best = sorted(((far(v), k) for k, v in res["grid"].items() if k.startswith(f"K{Kp}_")), reverse=True)[:6]
        print(f"  K'={Kp} best far(7-9):", [(k, round(v, 3)) for v, k in best])
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
