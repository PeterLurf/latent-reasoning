"""PROTOCOL7 T1/T2: causal tests of the layer-1 frontier beyond the trained horizon.

Long-chain probe graphs (latent/length_gen.py): chain A at distance 1..6, chain B hung from A's leaf (7..12), look-alike
unreachable chains D (1..6) and C (7..12). Rollout to K' latent steps; only the *first* block's attention of the
*latent* position is edited, never the query/answer positions:

  * block-B / block-D (T1): logit -1e4 on the six B- (or D-) chain edge tokens at every latent step.
  * inject-B / inject-C (T2): at latent step 6 + j (j = 1..6), each head's attention on the chain's j-th edge (the edge
    into the node at distance 6 + j for B) is raised to that head's natural frontier mass (the same model's peak mean
    attention, over steps 1..6, on A's distance-6 edge, unintervened, same graphs); the rest of the row is rescaled so
    the row still sums to 1. A head already above the target is left unchanged. `--dose` scales the target mass.

Latent steps mirror LatentReasoner.latent_step (step embedding only if the model uses one). Outputs, per condition,
the share of queries answered "reachable" for every node of A, B, C, D (TPR on A/B, 1 - TNR on C/D).

    PYTHONPATH=. python -m latent.frontier_causal runs/a runs/b ... --out results/p7s_causal.json --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from .competence import load
from .graphs import E, NPB
from .length_gen import long_chain_graphs
from .model import T_LAT


def edge_positions(g, chain):
    """(N, NPB) token positions of a chain's edges; edge j leads into the chain's node j."""
    bi = np.arange(len(g))
    return np.stack([g.hub_pos[bi, chain] if j == 0 else np.argmax(g.dst == g.nodes[bi, chain, j][:, None], 1)
                     for j in range(NPB)], 1)


def _first_block_step(blk, x, kv, bias=None, inj=None, rec=None):
    """Block.step for the first block with an additive logit bias (N, T) and/or an attention-mass injection
    inj = (pos (N,) long, mass (H,) float): per head, raise attention on token pos to mass, rescale the rest."""
    at = blk.attn
    q, k, v = at._split(blk.ln1(x))
    K_ = torch.cat([kv[0], k], 2)
    V_ = torch.cat([kv[1], v], 2)
    s = q @ K_.transpose(-1, -2) / math.sqrt(at.dh)                   # (N, H, 1, T)
    if bias is not None:
        s = s + bias[:, None, None, :s.shape[-1]]
    a = torch.softmax(s, -1)
    if inj is not None:
        pos, mass = inj
        a0 = a[:, :, 0, :]                                             # (N, H, T)
        idx = pos[:, None, None].expand(-1, a0.shape[1], 1)
        cur = a0.gather(-1, idx).squeeze(-1)                           # (N, H)
        tgt = torch.maximum(cur, mass[None, :].expand_as(cur))
        scale = (1 - tgt) / (1 - cur).clamp_min(1e-9)
        a0 = a0 * scale[..., None]
        a0 = a0.scatter(-1, idx, tgt[..., None])
        a = a0[:, :, None, :]
    if rec is not None:
        rec.append(a[:, :, 0, :E].detach())                            # edge tokens only (row length grows)
    x = x + at._merge(a @ V_)
    return x + blk.mlp(blk.ln2(x)), (K_, V_)


@torch.no_grad()
def rollout(m, K, g, qnodes, Kp, device, bias_pos=None, inject=None, record=False, chunk=256):
    """bias_pos: (N, P) token positions blocked in the first block at every latent step, or None.
    inject: dict step -> (pos (N,), mass (H,) tensor), or None. Returns (answers (N, Q) bool, attention list)."""
    T = lambda a: torch.as_tensor(a, device=device)
    N = len(g)
    te = m.type_emb.weight
    use_step = getattr(m, "use_step", True)
    outs, recs = [], []
    for i in range(0, N, chunk):
        sl = slice(i, min(N, i + chunk))
        n = sl.stop - sl.start
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
        rec = [] if record else None
        for k in range(1, Kp + 1):
            kk = min(k, K) if K > 0 else min(k, m.step_emb.num_embeddings - 1)
            base = m.pause.expand_as(h) if m.mode == "pause" else h
            x = (base + te[T_LAT] + (m.step_emb.weight[kk] if use_step else 0))[:, None]
            new = []
            for li, (blk, kv) in enumerate(zip(m.blocks, cache)):
                if li == 0:
                    bias = None
                    if bias_pos is not None:
                        bias = torch.zeros(n, kv[0].shape[2] + 1, device=device)
                        bias.scatter_(1, T(bias_pos[sl]).long(), -1e4)
                    inj = None
                    if inject is not None and k in inject:
                        pos, mass = inject[k]
                        inj = (T(pos[sl]).long(), mass)
                    x, kv = _first_block_step(blk, x, kv, bias, inj, rec)
                else:
                    x, kv = blk.step(x, kv)
                new.append(kv)
            h, cache = m._thought(x[:, 0]), new
        outs.append((m.query(cache, T(qnodes[sl])) > 0).cpu().numpy())
        if record:
            recs.append(torch.stack(rec).cpu().numpy())                # (Kp, n, H, E)
    return np.concatenate(outs), (np.concatenate(recs, 1) if record else None)


def summarise(pred, n):
    p = pred.reshape(n, 4, NPB).mean(0)                                # share answered "reachable"
    return {c: p[i].round(4).tolist() for i, c in enumerate("ABCD")}


@torch.no_grad()
def measure(run, device, n=1024, Kp=12, seed=4242, doses=(0.5, 1.0, 2.0)):
    m, c = load(run, device)
    K = c["K"]
    g, br = long_chain_graphs(np.random.default_rng(seed), n, [2, 4, 6])
    bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[ch]] for ch in "ABCD"], 1)      # (n, 4 * NPB)
    pos = {ch: edge_positions(g, br[ch]) for ch in "ABCD"}
    out = {"K": K, "Kp": Kp, "seed": seed, "n": n}
    base, att = rollout(m, K, g, q, Kp, device, record=True)
    out["base"] = summarise(base, n)
    # natural frontier mass: per head, peak over steps 1..6 of the mean attention on A's distance-6 edge (unintervened)
    a6 = np.stack([att[k][bi, :, pos["A"][:, NPB - 1]] for k in range(min(6, Kp))])   # (steps, n, H)
    nat = a6.mean(1).max(0)                                            # (H,)
    out["natural_mass"] = nat.round(5).tolist()
    out["block_B"] = summarise(rollout(m, K, g, q, Kp, device, bias_pos=pos["B"])[0], n)
    out["block_D"] = summarise(rollout(m, K, g, q, Kp, device, bias_pos=pos["D"])[0], n)
    for ch in "BC":
        for dose in doses:
            mass = torch.as_tensor(np.minimum(nat * dose, 0.99), device=device, dtype=torch.float32)
            inj = {NPB + j: (pos[ch][:, j - 1], mass) for j in range(1, NPB + 1) if NPB + j <= Kp}
            out[f"inject_{ch}_x{dose:g}"] = summarise(rollout(m, K, g, q, Kp, device, inject=inj)[0], n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for r in a.runs:
        if r in res:
            continue
        res[r] = measure(r, a.device, seed=a.seed)
        b = res[r]
        print(f"{r}: B1 base {b['base']['B'][0]:.3f} blockB {b['block_B']['B'][0]:.3f} blockD {b['block_D']['B'][0]:.3f} "
              f"| injB {b['inject_B_x1']['B'][0]:.3f} C1 base {b['base']['C'][0]:.3f} injC {b['inject_C_x1']['C'][0]:.3f}",
              flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
