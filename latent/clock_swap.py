"""Exploratory: is the sweep clocked by the step embedding or by the thought's content?

Only the step embedding fed to latent step k is changed (thought and cache untouched):
  normal      : step_emb[min(k, K)]
  freeze_j    : step_emb[min(k, j)]          (the clock stops at j)
  shift_s     : step_emb[clip(k + s, 1, K)]  (clock advanced or delayed by s)
Per condition: the layer-1 frontier schedule (depth whose edges get the most attention at each step,
W = 4 discovery graphs, addendum-F rule), depth-matched accuracy on ordinary graphs, and far accuracy
(distances 7-9) on long-chain graphs.

    python -m latent.clock_swap runs/a runs/b --out results/clock_swap.json --device cuda:1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .competence import load
from .graphs import E, NB, NPB, sample_graphs
from .interventions import all_queries
from .length_gen import long_chain_graphs
from .rewind import accuracy as chain_accuracy


def conditions(K):
    c = {"normal": lambda k: min(k, K)}
    for j in (1, 2, 3):
        c[f"freeze_{j}"] = lambda k, j=j: min(k, j)
    for s in (-2, -1, 1, 2):
        c[f"shift_{s:+d}"] = lambda k, s=s: int(np.clip(k + s, 1, K))
    return c


@torch.no_grad()
def run(m, g, q, Kp, idx, device, record=False):
    T = lambda a: torch.as_tensor(a, device=device)
    outs, recs = [], []
    for i in range(0, len(g), 256):
        sl = slice(i, min(len(g), i + 256))
        if record:
            for blk in m.blocks:
                blk.attn.record = []
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
        for k in range(1, Kp + 1):
            h, cache, _ = m.latent_step(h, idx(k), cache)
        outs.append((m.query(cache, T(q[sl])) > 0).cpu().numpy())
        if record:
            recs.append([a.mean(1).cpu().numpy()[:, :E] for a in m.blocks[0].attn.record])
            for blk in m.blocks:
                blk.attn.record = None
    return np.concatenate(outs), recs


def schedule(g, recs, Kp):
    """Addendum-F frontier depth per step from layer-1 attention (one record list per 256-graph chunk)."""
    N, bi = len(g), np.arange(len(g))
    dep, att = np.full((N, E), -1), np.zeros((N, E), bool)
    for bb in range(NB):
        for j in range(NPB):
            e = g.hub_pos[:, bb] if j == 0 else np.argmax(g.dst == g.nodes[:, bb, j][:, None], 1)
            dep[bi, e] = g.bdepth[:, bb, j]
            att[bi, e] = g.attached[:, bb]
    out = []
    for k in range(Kp):
        a = np.concatenate([r[k] for r in recs])
        per = lambda msk: float((a * msk).sum() / max(msk.sum(), 1))
        reach = [per(att & (dep == j)) for j in range(NPB)]
        decoy = float(np.mean([per(~att & (dep == j)) for j in range(NPB)]))
        j = int(np.argmax(reach))
        out.append(j if reach[j] >= 1.5 * decoy else None)
    return out


def matched(pred, g):
    p = pred.reshape(len(g), NB, NPB)
    att = np.broadcast_to(g.attached[:, :, None], p.shape)
    return {j + 1: 0.5 * (float(p[(g.bdepth == j) & att].mean()) + 1 - float(p[(g.bdepth == j) & ~att].mean())) for j in range(NPB)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    res = {}
    for run_ in a.runs:
        m, c = load(run_, a.device)
        K = c["K"]
        rng = np.random.default_rng(606)
        g4 = sample_graphs(rng, a.n, widths=[4], split="discovery")
        go = sample_graphs(rng, 2 * a.n, split="discovery")
        gl, br = long_chain_graphs(rng, a.n, [2, 4, 6])
        bi = np.arange(a.n)
        ql = np.concatenate([gl.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)
        q4, qo = all_queries(g4)[:, : NB * NPB], all_queries(go)[:, : NB * NPB]
        R = {}
        for name, idx in conditions(K).items():
            _, recs = run(m, g4, q4, K, idx, a.device, record=True)
            sched = schedule(g4, recs, K)
            acc = matched(run(m, go, qo, K, idx, a.device)[0], go)
            far12 = chain_accuracy(run(m, gl, ql, 12, idx, a.device)[0], a.n)
            R[name] = {"schedule": sched, "matched": acc, "chain_K12": far12}
            print(f"{run_} {name:9s} schedule {sched}  matched d3-6 " + " ".join(f"{acc[d]:.2f}" for d in (3, 4, 5, 6)) +
                  f" | far7-9 @12 {np.mean([far12[d] for d in (7, 8, 9)]):.3f}", flush=True)
        res[run_] = R
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
