"""Exploratory: does the step-keyed halt stop the sweep itself or only the read-out?

Long-chain graphs (chain A from the start, distances 1-6; chain B re-hung from A's leaf, distances 7-12).
Per latent step, head-averaged layer-1 attention on each edge of A and B (by distance), with normal step
embeddings and with the step embedding held at step 2. A sweep that keeps running moves its attention on
to B's edges one step at a time; a halted one does not.

    python -m latent.clock_trace runs/main_s3 runs/v2_deep_s0 --out results/clock_trace.json --device cuda:1
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
def trace(m, g, br, Kp, idx, device):
    T = lambda a: torch.as_tensor(a, device=device)
    N, bi = len(g), np.arange(len(g))
    edge = np.zeros((N, 2 * NPB), int)                 # edge into chain node at distance 1..12
    for c, name in enumerate(("A", "B")):
        for j in range(NPB):
            e = g.hub_pos[bi, br[name]] if j == 0 else np.argmax(g.dst == g.nodes[bi, br[name], j][:, None], 1)
            edge[:, c * NPB + j] = e
    for blk in m.blocks:
        blk.attn.record = []
    h, cache, _ = m.encode(m.embed_prefix(T(g.src), T(g.dst), T(g.start)))
    for k in range(1, Kp + 1):
        h, cache, _ = m.latent_step(h, idx(k), cache)
    rec = m.blocks[0].attn.record
    for blk in m.blocks:
        blk.attn.record = None
    out = []
    for a in rec:
        a = a.mean(1).cpu().numpy()
        out.append([float(a[bi, edge[:, d]].mean()) for d in range(2 * NPB)])
    return out                                           # [step][distance-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    res = {}
    for run_ in a.runs:
        m, c = load(run_, a.device)
        K = c["K"]
        rng = np.random.default_rng(808)
        g, br = long_chain_graphs(rng, a.n, [2, 4, 6])
        R = {}
        for name, idx in (("normal", lambda k: min(k, K)), ("freeze_2", lambda k: min(k, 2))):
            tr = trace(m, g, br, 14, idx, a.device)
            R[name] = tr
            print(f"{run_} {name}: layer-1 attention on the chain edge into distance d (argmax per step, value)")
            for k, row in enumerate(tr, 1):
                d = int(np.argmax(row)) + 1
                print(f"   step {k:2d}: peak at distance {d:2d} ({max(row):.3f})  | distances 7-12: " + " ".join(f"{v:.3f}" for v in row[NPB:]))
        res[run_] = R
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
