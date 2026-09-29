"""Exploratory: can repeated rewinds keep a clock-locked search going? (follow-up to rewind_grid)

Schedules over K' steps (thought fed into each listed step replaced; cache always kept):
  fixed3 : the run's own step-3 thought at every listed step
  back3  : the run's own thought from three steps earlier at every listed step
Balanced accuracy at distances 1..12 on long-chain graphs.

    python -m latent.rewind_multi runs/r3_pc_gated_gpt2 --out results/rewind_multi_s0.json
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
def run(m, K, g, q, Kp, device, steps=(), mode="fixed3"):
    T = lambda a: torch.as_tensor(a, device=device)
    n, out = len(g), []
    for i in range(0, n, 256):
        sl = slice(i, min(n, i + 256))
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
        th = [h]
        for k in range(1, Kp + 1):
            if k in steps:
                h = th[3] if mode == "fixed3" else th[k - 4]
            h, cache, _ = m.latent_step(h, min(k, K), cache)
            th.append(h)
        out.append((m.query(cache, T(q[sl])) > 0).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    res = {}
    for run_ in a.runs:
        m, c = load(run_, a.device)
        rng = np.random.default_rng(79)
        g, br = long_chain_graphs(rng, a.n, [2, 4, 6])
        bi = np.arange(a.n)
        q = np.concatenate([g.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)
        R = {}
        sched = {"none": (), "once@7": (7,), "every3": (7, 10, 13, 16, 19), "every2": (7, 9, 11, 13, 15, 17, 19),
                 "every4": (7, 11, 15, 19)}
        for Kp in (12, 20):
            for name, st in sched.items():
                for mode in ("fixed3", "back3"):
                    if name == "none" and mode == "back3":
                        continue
                    key = f"K{Kp}_{name}_{mode}"
                    R[key] = accuracy(run(m, c["K"], g, q, Kp, a.device, tuple(s for s in st if s <= Kp), mode), a.n)
                    print(run_, f"{key:24s}", " ".join(f"{R[key][d]:.2f}" for d in range(3, 13)), flush=True)
        res[run_] = R
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
