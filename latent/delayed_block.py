"""Exploratory causal test of the forward sweep (delayed format): block one branch's depth-x edges at step k.

W = 4 discovery graphs. For a reachable branch A whose deepest node is at within-branch depth >= 4, layer-1
attention of latent step k onto A's edges into depth-x nodes is blocked (logit bias -1e4). Outcome: change
in mean P(reachable) of A's nodes deeper than x (downstream), and of other reachable branches' nodes
(control). A forward sweep predicts a drop only when step k is the one that reads depth x.

    python -m latent.delayed_block runs/main_s1 --out results/delayed_block_s1.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .competence import load
from .graphs import E, NB, NPB, sample_graphs
from .interventions import all_queries
from .model import T_LAT
from .upfront_block import step_masked


@torch.no_grad()
def run(m, K, g, q, Kp, device, block_step=None, block_mask=None):
    T = lambda a: torch.as_tensor(a, device=device)
    N = len(g)
    h, cache, _ = m.encode(m.embed_prefix(T(g.src), T(g.dst), T(g.start)))
    te = m.type_emb.weight
    for k in range(1, Kp + 1):
        base = m.pause.expand_as(h) if m.mode == "pause" else h
        x = (base + te[T_LAT] + m.step_emb.weight[min(k, K)])[:, None]
        new = []
        for li, (blk, kv) in enumerate(zip(m.blocks, cache)):
            bias = None
            if k == block_step and li == 0:
                bias = torch.zeros(N, kv[0].shape[2] + 1, device=device)
                bias[:, :E] = T(block_mask).float() * -1e4
            x, kv = step_masked(blk, x, kv, bias)
            new.append(kv)
        h, cache = m._thought(x[:, 0]), new
    return torch.sigmoid(m.query(cache, T(q))).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    res = {}
    for run_ in a.runs:
        m, c = load(run_, a.device)
        rng = np.random.default_rng(555)
        g = sample_graphs(rng, 4 * a.n, widths=[4], split="discovery")
        deep = (g.bdepth.max(2) >= 4) & g.attached
        A = np.argmax(np.where(deep, rng.random(deep.shape), -1), 1)
        keep = np.nonzero(deep.any(1))[0][: a.n]
        g, A = g.subset(keep), A[keep]
        N, bi = len(g), np.arange(len(keep))
        q = all_queries(g)[:, : NB * NPB]                     # (N, NB*NPB): node of branch b, slot j at b*NPB + j
        qd = np.repeat(np.arange(NB)[None], N, 0)
        dA = g.bdepth[bi, A]                                  # (N, NPB) depths of A's nodes
        edge_of = np.zeros((N, NPB), int)                     # edge into A's slot-j node
        edge_of[:, 0] = g.hub_pos[bi, A]
        for j in range(1, NPB):
            edge_of[:, j] = np.argmax(g.dst == g.nodes[bi, A, j][:, None], 1)
        others = g.attached.copy(); others[bi, A] = False
        base = run(m, c["K"], g, q, 8, a.device).reshape(N, NB, NPB)
        R = {"n": N, "downstream": {}, "control": {}}
        print(f"{run_}: rows = blocked step k, cols = blocked depth x; ΔP(reachable) of A's nodes deeper than x | other reachable branches")
        for k in range(1, 9):
            dn, ct = [], []
            for x in range(0, 5):
                mask = np.zeros((N, E), bool)
                for j in range(NPB):
                    sel = dA[:, j] == x
                    mask[bi[sel], edge_of[sel, j]] = True
                p = run(m, c["K"], g, q, 8, a.device, k, mask).reshape(N, NB, NPB)
                d = (p - base)
                down = np.where(dA > x, d[bi, A], np.nan)
                dn.append(float(np.nanmean(down)))
                ct.append(float(np.nanmean(np.where(others[:, :, None], d, np.nan))))
            R["downstream"][k], R["control"][k] = dn, ct
            print(f"  k={k}: " + " ".join(f"{v:+.3f}" for v in dn) + "  | " + " ".join(f"{v:+.3f}" for v in ct), flush=True)
        res[run_] = R
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
