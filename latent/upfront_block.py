"""Exploratory causal test of backward pointer-chasing (upfront format): block one edge at one step.

For reachable targets at distance 6 (W = 4, discovery graphs), layer-1 attention of latent step k onto one
edge token is blocked (logit bias -1e4). Blocking the path edge at depth x is compared with blocking a
same-depth edge of another reachable branch (control). Pointer-chasing predicts a large drop in
P(reachable) only where the trace says the edge is read (k, x on the walk's diagonal).

    python -m latent.upfront_block runs/v3_rawfb_s0 --out results/upfront_block_v3.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from .competence import load
from .graphs import NPB, sample_graphs
from .model import T_LAT
from .upfront_trace import pick


def step_masked(blk, x, kv, bias):
    """Block.step with an additive key bias on the attention logits (bias: (B, T_cache + 1) or None)."""
    at = blk.attn
    q, k, v = at._split(blk.ln1(x))
    K_ = torch.cat([kv[0], k], 2)
    V_ = torch.cat([kv[1], v], 2)
    s = q @ K_.transpose(-1, -2) / math.sqrt(at.dh)
    if bias is not None:
        s = s + bias[:, None, None, :s.shape[-1]]
    a = torch.softmax(s, -1)
    x = x + at._merge(a @ V_)
    return x + blk.mlp(blk.ln2(x)), (K_, V_)


@torch.no_grad()
def run(m, K, g, tgt, Kp, device, block_step=None, block_pos=None, layer=0):
    T = lambda a: torch.as_tensor(a, device=device)
    N = len(g)
    h, cache, _ = m.encode(m.embed_prefix(T(g.src), T(g.dst), T(g.start), T(tgt)))
    te = m.type_emb.weight
    for k in range(1, Kp + 1):
        base = m.pause.expand_as(h) if m.mode == "pause" else h
        x = (base + te[T_LAT] + m.step_emb.weight[min(k, K)])[:, None]
        new = []
        for li, (blk, kv) in enumerate(zip(m.blocks, cache)):
            bias = None
            if k == block_step and li == layer:
                bias = torch.zeros(N, kv[0].shape[2] + 1, device=device)
                bias[torch.arange(N), T(block_pos)] = -1e4
            x, kv = step_masked(blk, x, kv, bias)
            new.append(kv)
        h, cache = m._thought(x[:, 0]), new
    return torch.sigmoid(m.query(cache, None, B=N)[:, 0]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    m, c = load(a.run, a.device)
    rng = np.random.default_rng(321)
    g = sample_graphs(rng, 4 * a.n, widths=[4], split="discovery")
    b = pick(g, rng, True)
    keep = np.nonzero(b >= 0)[0][: a.n]
    g, b = g.subset(keep), b[keep]
    N, bi = len(g), np.arange(len(keep))
    tgt = g.nodes[bi, b, NPB - 1]
    edge_into = lambda br, j: g.hub_pos[bi, br] if j == 0 else np.argmax(g.dst == g.nodes[bi, br, j][:, None], 1)
    # control branch: another reachable branch (any shape); its edge into a node of the same depth if present
    other = np.argmax(np.where(g.attached & (np.arange(g.attached.shape[1])[None] != b[:, None]), rng.random(g.attached.shape), -1), 1)
    base = run(m, c["K"], g, tgt, 8, a.device)
    with torch.no_grad():                                   # the unmasked loop must equal the model's own steps
        T = lambda x: torch.as_tensor(x, device=a.device)
        h, cache, _ = m.encode(m.embed_prefix(T(g.src), T(g.dst), T(g.start), T(tgt)))
        for k in range(1, 9):
            h, cache, _ = m.latent_step(h, min(k, c["K"]), cache)
        ref = torch.sigmoid(m.query(cache, None, B=N)[:, 0]).cpu().numpy()
    assert np.abs(ref - base).max() < 1e-4, np.abs(ref - base).max()
    res = {"run": a.run, "n": N, "p_base": float(base.mean()), "path": {}, "control": {}}
    print(f"{a.run}: baseline P(reachable) {base.mean():.3f}")
    for k in range(1, 9):
        row_p, row_c = [], []
        for j in range(NPB):
            pp = run(m, c["K"], g, tgt, 8, a.device, k, edge_into(b, j))
            has = g.bdepth[bi, other].max(1) >= j
            cp = run(m, c["K"], g, tgt, 8, a.device, k, np.where(has, edge_into(other, min(j, NPB - 1)), edge_into(other, 0)))
            row_p.append(float(pp.mean() - base.mean())); row_c.append(float(cp.mean() - base.mean()))
        res["path"][k], res["control"][k] = row_p, row_c
        print(f"  step {k}: path x1..6 " + " ".join(f"{v:+.3f}" for v in row_p) + " | control " + " ".join(f"{v:+.3f}" for v in row_c), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
