"""Exploratory: the long-chain length test in the upfront format (target given before the latent steps).

Same graphs as latent/length_gen.py. For each graph one target per distance: positive = node of the
re-hung reachable chain B at depth j (distance 7 + j) or of chain A (distance 1 + j); negative = the
matched node of the re-hung decoy chain C (or D). Balanced accuracy per distance; K' beyond the trained
budget reuses the last step embedding.

    python -m latent.length_gen_upfront runs/a runs/b --out results/length_gen_upfront.json
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
def answer(m, K, g, tgt, Kp, device):
    T = lambda a: torch.as_tensor(a, device=device)
    out = []
    for i in range(0, len(g), 256):
        sl = slice(i, min(len(g), i + 256))
        x = m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl]), T(tgt[sl]))
        h, cache, _ = m.encode(x)
        for k in range(1, Kp + 1):
            h, cache, _ = m.latent_step(h, min(k, K) if K > 0 else min(k, m.step_emb.num_embeddings - 1), cache)
        out.append((m.query(cache, None, B=len(tgt[sl]))[:, 0] > 0).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=768)
    a = ap.parse_args()
    res = {}
    for run in a.runs:
        m, c = load(run, a.device)
        rng = np.random.default_rng(99)
        g, br = long_chain_graphs(rng, a.n, [2, 4, 6])
        bi = np.arange(a.n)
        R = {}
        for Kp in (0, 4, 6, 8, 12, 16):
            acc = {}
            for j in range(NPB):
                for pos, neg, dist in (("A", "D", 1 + j), ("B", "C", NPB + 1 + j)):
                    tp = answer(m, c["K"], g, g.nodes[bi, br[pos], j], Kp, a.device).mean()
                    fp = answer(m, c["K"], g, g.nodes[bi, br[neg], j], Kp, a.device).mean()
                    acc[dist] = float(0.5 * (tp + 1 - fp))
            R[Kp] = acc
            print(run, f"K'={Kp:2d} " + " ".join(f"{acc[d]:.2f}" for d in range(1, 13)), flush=True)
        res[run] = R
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
