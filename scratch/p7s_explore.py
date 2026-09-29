"""PROTOCOL7 exploratory follow-up to T1/T2 (after both were scored "not supported"). Not confirmatory.

Conditions on the T1/T2 probe graphs (seed 4242) for a subset of T1 (extrapolators) and T2 (non-extrapolators) models:
  * block_A_L1: first-block latent attention to A's edges blocked (positive control: in-range nodes should fall).
  * block_B_L12: B's edges blocked in *both* blocks' latent attention.
  * block_B_query: the query tokens' attention (both blocks) to B's first edge (A6 -> B1) blocked, latent untouched
    (tests one-hop query-time reading of distance 7).
  * block_B_L1_query: first-block latent block of B plus the query block of B's first edge.
  * inject_B_m{0.1,0.3,0.5}: fixed per-head mass on B-edge j at step 6 + j (T2 with strong doses).
    PYTHONPATH=. python scratch/p7s_explore.py --device cuda:0
"""
import argparse
import json
import math
import sys

import numpy as np
import torch

from latent import frontier_causal as FC
from latent.competence import load
from latent.graphs import NPB
from latent.length_gen import long_chain_graphs
from latent.model import T_LAT, T_QUERY


def fork_masked(blk, x, kv, bias):
    """Block.fork with an additive bias (B, Tc) on the cache logits (self-logit untouched)."""
    at = blk.attn
    q, k, v = at._split(blk.ln1(x))
    s_c = q @ kv[0].transpose(-1, -2) / math.sqrt(at.dh)
    if bias is not None:
        s_c = s_c + bias[:, None, None, :s_c.shape[-1]]
    s_s = (q * k).sum(-1, keepdim=True) / math.sqrt(at.dh)
    a = torch.softmax(torch.cat([s_c, s_s], -1), -1)
    y = a[..., :-1] @ kv[1] + a[..., -1:] * v
    x = x + at._merge(y)
    return x + blk.mlp(blk.ln2(x))


@torch.no_grad()
def run(m, K, g, q, Kp, dev, lat_block=None, lat_layers=(0,), qry_block=None, inject=None, chunk=256):
    T = lambda a: torch.as_tensor(a, device=dev)
    te = m.type_emb.weight
    use_step = getattr(m, "use_step", True)
    outs = []
    for i in range(0, len(g), chunk):
        sl = slice(i, min(len(g), i + chunk)); n = sl.stop - sl.start
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
        for k in range(1, Kp + 1):
            kk = min(k, K)
            base = m.pause.expand_as(h) if m.mode == "pause" else h
            x = (base + te[T_LAT] + (m.step_emb.weight[kk] if use_step else 0))[:, None]
            new = []
            for li, (blk, kv) in enumerate(zip(m.blocks, cache)):
                bias = None
                if lat_block is not None and li in lat_layers:
                    bias = torch.zeros(n, kv[0].shape[2] + 1, device=dev)
                    bias.scatter_(1, T(lat_block[sl]).long(), -1e4)
                inj = None
                if li == 0 and inject is not None and k in inject:
                    inj = (T(inject[k][0][sl]).long(), inject[k][1])
                if bias is None and inj is None:
                    x, kv = blk.step(x, kv)
                else:
                    x, kv = FC._first_block_step(blk, x, kv, bias, inj) if li == 0 else _step_bias(blk, x, kv, bias)
                new.append(kv)
            h, cache = m._thought(x[:, 0]), new
        xq = m.node_emb(T(q[sl])) + te[T_QUERY]
        for blk, kv in zip(m.blocks, cache):
            qb = None
            if qry_block is not None:
                qb = torch.zeros(n, kv[0].shape[2], device=dev)
                qb.scatter_(1, T(qry_block[sl]).long(), -1e4)
            xq = fork_masked(blk, xq, kv, qb)
        outs.append((m.head(m.ln_f(xq))[..., 0] > 0).cpu().numpy())
    return np.concatenate(outs)


def _step_bias(blk, x, kv, bias):
    from latent.upfront_block import step_masked
    return step_masked(blk, x, kv, bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_each", type=int, default=6)
    a = ap.parse_args()
    S = json.load(open("results/p7s_t12.json"))
    runs = list(S["T1"])[: a.n_each] + list(S["T2"])[: a.n_each]
    res = {}
    for r in runs:
        m, c = load(r, a.device); K = c["K"]
        g, br = long_chain_graphs(np.random.default_rng(4242), 1024, [2, 4, 6])
        bi = np.arange(len(g))
        q = np.concatenate([g.nodes[bi, br[ch]] for ch in "ABCD"], 1)
        pos = {ch: FC.edge_positions(g, br[ch]) for ch in "ABCD"}
        H = m.blocks[0].attn.h
        conds = {"base": {}, "block_A_L1": {"lat_block": pos["A"]},
                 "block_B_L12": {"lat_block": pos["B"], "lat_layers": (0, 1)},
                 "block_B_query": {"qry_block": pos["B"][:, :1]},
                 "block_B_L1_query": {"lat_block": pos["B"], "qry_block": pos["B"][:, :1]}}
        for mass in (0.1, 0.3, 0.5):
            mt = torch.full((H,), mass, device=a.device)
            conds[f"inject_B_m{mass}"] = {"inject": {NPB + j: (pos["B"][:, j - 1], mt) for j in range(1, NPB + 1)}}
            conds[f"inject_C_m{mass}"] = {"inject": {NPB + j: (pos["C"][:, j - 1], mt) for j in range(1, NPB + 1)}}
        res[r] = {"group": "T1_extrap" if r in S["T1"] else "T2_non"}
        for name, kw in conds.items():
            p = run(m, K, g, q, 12, a.device, **kw).reshape(len(g), 4, NPB).mean(0)
            res[r][name] = {ch: p[i].round(3).tolist() for i, ch in enumerate("ABCD")}
        b = res[r]
        f = lambda v: " ".join(f"{x:.2f}" for x in v)
        print(f"{r} [{b['group']}]\n  base  A {f(b['base']['A'])} B {f(b['base']['B'])} C {f(b['base']['C'])}"
              f"\n  blkA1 A {f(b['block_A_L1']['A'])}\n  blkB12 B {f(b['block_B_L12']['B'])}   blkBq B {f(b['block_B_query']['B'])}"
              f"   blkB1+q B {f(b['block_B_L1_query']['B'])}"
              f"\n  injB .1 {f(b['inject_B_m0.1']['B'])} .3 {f(b['inject_B_m0.3']['B'])} .5 {f(b['inject_B_m0.5']['B'])}"
              f"\n  injC .3 C {f(b['inject_C_m0.3']['C'])}", flush=True)
        json.dump(res, open("results/p7s_explore.json", "w"), indent=1)


if __name__ == "__main__":
    main()
