"""PROTOCOL7 amendment 3 (T1b, T1c, T2b): corrected causal tests of the latent frontier beyond the trained horizon.

Written after T1/T2 were scored and explored (scratch/p7s_explore.py): distance 7 is answered by a one-hop read at
query time, so the latent frontier is tested on distances 8-12; natural injection scale is taken from the sharp part
of each model's own in-range sweep. Same probe graphs as T1/T2 (long_chain_graphs, fresh seed), K' = 12.

Conditions (latent positions only unless stated):
  base
  block_B_L12   B-chain edge tokens blocked (-1e4) in every block's latent attention, every step      (T1b)
  block_D_L12   the same for the unreachable in-range look-alike chain D (control)                    (T1b)
  block_A_L1    A-chain blocked in the first block (positive control: in-range deep nodes fall)
  query_B1      the query tokens' attention (every block) to B's first edge (A6 -> B1) blocked       (T1c)
  inject_B_nat  first block, step 6 + j: head h's attention on B-edge j raised to its natural mass m_h (T2b)
  inject_C_nat  the same on the C chain (control: readout trusts the frontier)                        (T2b)
Natural mass m_h = max over in-range depths 2..5 and steps 1..6 of head h's mean attention (over graphs) on A's edge
into that depth, unintervened, same graphs.

    PYTHONPATH=. python -m latent.frontier_causal2 runs/a ... --out results/p7s_causal2.json --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from . import frontier_causal as FC
from .competence import load
from .graphs import NPB
from .length_gen import long_chain_graphs
from .model import T_LAT, T_QUERY
from .upfront_block import step_masked


def _fork_masked(blk, x, kv, bias):
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
            base = m.pause.expand_as(h) if m.mode == "pause" else h
            x = (base + te[T_LAT] + (m.step_emb.weight[min(k, K)] if use_step else 0))[:, None]
            new = []
            for li, (blk, kv) in enumerate(zip(m.blocks, cache)):
                bias = None
                if lat_block is not None and li in lat_layers:
                    bias = torch.zeros(n, kv[0].shape[2] + 1, device=dev)
                    bias.scatter_(1, T(lat_block[sl]).long(), -1e4)
                inj = (T(inject[k][0][sl]).long(), inject[k][1]) if (li == 0 and inject and k in inject) else None
                if li == 0 and (bias is not None or inj is not None):
                    x, kv = FC._first_block_step(blk, x, kv, bias, inj)
                elif bias is not None:
                    x, kv = step_masked(blk, x, kv, bias)
                else:
                    x, kv = blk.step(x, kv)
                new.append(kv)
            h, cache = m._thought(x[:, 0]), new
        xq = m.node_emb(T(q[sl])) + te[T_QUERY]
        for blk, kv in zip(m.blocks, cache):
            qb = None
            if qry_block is not None:
                qb = torch.zeros(n, kv[0].shape[2], device=dev)
                qb.scatter_(1, T(qry_block[sl]).long(), -1e4)
            xq = _fork_masked(blk, xq, kv, qb)
        outs.append((m.head(m.ln_f(xq))[..., 0] > 0).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def measure(run_dir, device, n=1024, Kp=12, seed=4243):
    m, c = load(run_dir, device)
    K = c["K"]
    g, br = long_chain_graphs(np.random.default_rng(seed), n, [2, 4, 6])
    bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[ch]] for ch in "ABCD"], 1)
    pos = {ch: FC.edge_positions(g, br[ch]) for ch in "ABCD"}
    _, att = FC.rollout(m, K, g, q, Kp, device, record=True)            # (Kp, n, H, E)
    mh = np.stack([np.stack([att[k][bi, :, pos["A"][:, j]] for k in range(min(6, Kp))]).mean(1)
                   for j in range(1, 5)])                               # depths 2..5: (4, steps, H)
    nat = mh.max((0, 1))                                                # (H,)
    mt = torch.as_tensor(np.minimum(nat, 0.99), device=device, dtype=torch.float32)
    conds = {"base": {}, "block_B_L12": {"lat_block": pos["B"], "lat_layers": tuple(range(len(m.blocks)))},
             "block_D_L12": {"lat_block": pos["D"], "lat_layers": tuple(range(len(m.blocks)))},
             "block_A_L1": {"lat_block": pos["A"]}, "query_B1": {"qry_block": pos["B"][:, :1]},
             "inject_B_nat": {"inject": {NPB + j: (pos["B"][:, j - 1], mt) for j in range(1, NPB + 1) if NPB + j <= Kp}},
             "inject_C_nat": {"inject": {NPB + j: (pos["C"][:, j - 1], mt) for j in range(1, NPB + 1) if NPB + j <= Kp}}}
    out = {"K": K, "Kp": Kp, "seed": seed, "n": n, "natural_mass": nat.round(4).tolist()}
    for name, kw in conds.items():
        p = run(m, K, g, q, Kp, device, **kw).reshape(n, 4, NPB).mean(0)
        out[name] = {ch: p[i].round(4).tolist() for i, ch in enumerate("ABCD")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for r in a.runs:
        if r in res or not os.path.exists(f"{r}/model.pt"):
            continue
        res[r] = measure(r, a.device)
        b = res[r]
        f = lambda v: " ".join(f"{x:.2f}" for x in v)
        print(f"{r}: B base {f(b['base']['B'])} | blkB {f(b['block_B_L12']['B'])} | qB1 {b['query_B1']['B'][0]:.2f} "
              f"| injB {f(b['inject_B_nat']['B'])}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
