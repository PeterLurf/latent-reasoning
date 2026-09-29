"""Exploratory: forward search or backward chaining in the upfront format? (attention trace)

Upfront runs (target given before the latent steps) on discovery graphs whose target sits at distance 6
(within-branch depth 5) in a reachable branch, or in a decoy branch. For every latent step k and layer,
the head-averaged attention of the latent position onto edge tokens is split into
  path[x]    : the edge into the target's ancestor at path depth x (x = 1 hub edge ... x = 6 edge into target)
  offpath[x] : mean per-edge attention on other edges at the same depth in the *same* branch type
Forward search predicts attention moving outward from x = 1 with k; backward chaining predicts it
starting at x = 6 (the target's own incoming edge) and moving toward the start.

    python -m latent.upfront_trace runs/v3_rawfb_s0 --out results/upfront_trace_v3.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .competence import load
from .graphs import E, NB, NPB, sample_graphs


def pick(g, rng, attached):
    """Per graph: a branch of the requested type holding a depth-5 node (a chain), else -1."""
    ok = (g.bdepth[:, :, NPB - 1] == NPB - 1) & (g.attached == attached)
    r = np.where(ok, rng.random(ok.shape), -1)
    b = np.argmax(r, 1)
    return np.where(ok.any(1), b, -1)


@torch.no_grad()
def trace(m, K, g, b, Kp, device):
    N = len(g)
    bi = np.arange(N)
    tgt = g.nodes[bi, b, NPB - 1]
    # edge position of the edge into each node of the chosen chain (chains: node j has depth j)
    pos = np.zeros((N, NPB), int)
    pos[:, 0] = g.hub_pos[bi, b]
    for j in range(1, NPB):
        pos[:, j] = np.argmax(g.dst == g.nodes[bi, b, j][:, None], 1)
    # same-depth edges in other branches of the same type (off-path controls)
    same = g.attached == g.attached[bi, b][:, None]
    depth_of_edge = np.full((N, E), -1)
    type_ok = np.zeros((N, E), bool)
    for bb in range(NB):
        for j in range(NPB):
            e = g.hub_pos[:, bb] if j == 0 else np.argmax(g.dst == g.nodes[:, bb, j][:, None], 1)
            depth_of_edge[bi, e] = g.bdepth[:, bb, j]
            type_ok[bi, e] = same[:, bb] & (bb != b)
    T = lambda a: torch.as_tensor(a, device=device)
    for blk in m.blocks:
        blk.attn.record = []
    h, cache, _ = m.encode(m.embed_prefix(T(g.src), T(g.dst), T(g.start), T(tgt)))
    for k in range(1, Kp + 1):
        h, cache, _ = m.latent_step(h, min(k, K), cache)
    ans = (m.query(cache, None, B=N)[:, 0] > 0).cpu().numpy()
    out = {}
    for li, blk in enumerate(m.blocks):
        rec = blk.attn.record
        blk.attn.record = None
        for k, a in enumerate(rec):
            a = a.mean(1).cpu().numpy()[:, :E]                  # (N, E) head-averaged
            path = [float(a[bi, pos[:, j]].mean()) for j in range(NPB)]
            off = []
            for j in range(NPB):
                msk = type_ok & (depth_of_edge == j)
                off.append(float((a * msk).sum() / max(msk.sum(), 1)))
            out[f"L{li + 1}_k{k + 1}"] = {"path": path, "offpath": off}
    return out, float(ans.mean())


@torch.no_grad()
def trace_delayed(m, K, g, Kp, device):
    """Delayed format (no target): per-edge attention on reachable-branch edges vs decoy-branch edges, by depth."""
    N = len(g)
    bi = np.arange(N)
    dep = np.full((N, E), -1)
    att = np.zeros((N, E), bool)
    for bb in range(NB):
        for j in range(NPB):
            e = g.hub_pos[:, bb] if j == 0 else np.argmax(g.dst == g.nodes[:, bb, j][:, None], 1)
            dep[bi, e] = g.bdepth[:, bb, j]
            att[bi, e] = g.attached[:, bb]
    T = lambda a: torch.as_tensor(a, device=device)
    for blk in m.blocks:
        blk.attn.record = []
    h, cache, _ = m.encode(m.embed_prefix(T(g.src), T(g.dst), T(g.start)))
    for k in range(1, Kp + 1):
        h, cache, _ = m.latent_step(h, min(k, K), cache)
    out = {}
    for li, blk in enumerate(m.blocks):
        rec = blk.attn.record
        blk.attn.record = None
        for k, a in enumerate(rec):
            a = a.mean(1).cpu().numpy()[:, :E]
            per = lambda msk: float((a * msk).sum() / max(msk.sum(), 1))
            out[f"L{li + 1}_k{k + 1}"] = {"reachable": [per(att & (dep == j)) for j in range(NPB)],
                                          "decoy": [per(~att & (dep == j)) for j in range(NPB)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--delayed", action="store_true")
    a = ap.parse_args()
    res = {}
    for run in a.runs:
        m, c = load(run, a.device)
        rng = np.random.default_rng(123)
        g = sample_graphs(rng, 4 * a.n, widths=[a.W], split="discovery")
        if a.delayed:
            tr = trace_delayed(m, c["K"], g.subset(np.arange(a.n)), 8, a.device)
            res[run] = {"delayed": tr}
            print(f"{run} W={a.W} delayed: per-edge attention, reachable-branch edges by depth 0..5 | decoy")
            for k in range(1, 9):
                t = tr[f"L1_k{k}"]
                print(f"   L1 k{k}  " + " ".join(f"{v:.3f}" for v in t["reachable"]) + "  | " + " ".join(f"{v:.3f}" for v in t["decoy"]))
            continue
        R = {}
        for kind, att in (("reachable", True), ("decoy", False)):
            b = pick(g, rng, att)
            sub = g.subset(np.nonzero(b >= 0)[0][: a.n])
            bb = b[b >= 0][: a.n]
            tr, pr = trace(m, c["K"], sub, bb, 8, a.device)
            R[kind] = {"trace": tr, "p_reachable": pr, "n": len(sub)}
            print(f"{run} W={a.W} {kind} target at distance 6: P(answer reachable) = {pr:.2f} (n={len(sub)})")
            for li in range(1, len(m.blocks) + 1):
                for k in range(1, 9):
                    t = tr[f"L{li}_k{k}"]
                    print(f"   L{li} k{k}  path x1..6 " + " ".join(f"{v:.3f}" for v in t["path"]) +
                          "  | off-path " + " ".join(f"{v:.3f}" for v in t["offpath"]))
        res[run] = R
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
