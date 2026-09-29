"""Exploratory exp3: which component writes the pointer into thought h_k?

For latent step k (input h_{k-1}, output h_k), the final residual x_k is a sum of components:
  emb (h_{k-1} + type + step), attn1 split by source {edges, START, earlier latent positions (by lag), self}, mlp1,
  attn2 split the same way, mlp2.
h_k = ln_f(x_k); the next step's sweep head reads q = W_q LN1(h_k + te_LAT). Both LayerNorms are linearised at the
actual input (scale frozen, mean removed), so each component's contribution to the next step's membership logit is
exact up to the frozen-scale linearisation (sum of parts = the actual logit minus a constant bias term).
Membership logit of the node at distance j = logit(edge out of it) - logit(matched decoy edge), sweep head.
    python theory/exp3_attrib.py runs/p30_s115 --steps 4 5 6 7
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout import load, probe  # noqa: E402
from latent.model import T_LAT  # noqa: E402


def ln_lin(ln, x):
    """Return a function mapping a component c (same shape as x) to its share of ln(x) (bias excluded)."""
    mu = x.mean(-1, keepdim=True)
    sd = (x.var(-1, keepdim=True, unbiased=False) + ln.eps).sqrt()
    return lambda c: (c - c.mean(-1, keepdim=True)) / sd * ln.weight


@torch.no_grad()
def attn_parts(attn, xin, kv_all, P, nlat):
    """Split the attention output of one latent position by key source. Returns dict name -> (B,d) and new kv."""
    q, k, v = attn._split(xin)
    Kall = torch.cat([kv_all[0], k], 2); Vall = torch.cat([kv_all[1], v], 2)
    s = q @ Kall.transpose(-1, -2) / math.sqrt(attn.dh)
    a = torch.softmax(s, -1)                                      # (B,H,1,T)
    def part(idx):
        y = (a[..., idx] @ Vall[:, :, idx])                       # (B,H,1,dh)
        B, H, T, dh = y.shape
        return y.transpose(1, 2).reshape(B, T, H * dh) @ attn.out.weight.T   # no bias
    parts = {"edges": part(list(range(P - 1))), "START": part([P - 1]), "self": part([P + nlat])}
    for lag in range(1, nlat + 1):
        parts[f"lat-{lag}"] = part([P + nlat - lag])              # latent position k-lag (input h_{k-lag-1})
    parts["bias"] = attn.out.bias.expand_as(parts["self"])
    return parts, (Kall, Vall)


@torch.no_grad()
def attribute(m, K, g, br, E, step, head):
    T = lambda a: torch.as_tensor(a)
    x0 = m.embed_prefix(T(g.src), T(g.dst), T(g.start))
    P = x0.shape[1]
    h, cache, _ = m.encode(x0)
    te = m.type_emb.weight
    for k in range(1, step + 1):
        emb = (h + te[T_LAT] + (m.step_emb.weight[min(k, K)] if getattr(m, "use_step", True) else 0))[:, None]
        comps = {"emb": emb}
        xx = emb
        new = []
        for li, (blk, kv) in enumerate(zip(m.blocks, cache)):
            parts, kv2 = attn_parts(blk.attn, blk.ln1(xx), kv, P, k - 1)
            a_out = sum(parts.values())
            if k == step:
                for n_, c in parts.items():
                    comps[f"a{li+1}.{n_}"] = c
            xx = xx + a_out
            mo = blk.mlp(blk.ln2(xx))
            if k == step:
                comps[f"mlp{li+1}"] = mo
            xx = xx + mo
            new.append(kv2)
        cache = new
        h = m._thought(xx[:, 0])
    # next step's sweep head query from h (= ln_f(xx)); linearise ln_f and ln1
    f1 = ln_lin(m.ln_f, xx[:, 0])
    xin = h[:, None] + te[T_LAT]
    f2 = ln_lin(m.blocks[0].ln1, xin[:, 0])
    blk = m.blocks[0].attn
    dh = blk.dh
    Wq = blk.qkv.weight[head * dh:(head + 1) * dh]                   # (dh, d)
    kv = cache[0][0][:, head, :P - 1]                                 # (B, E, dh) keys of edge tokens
    n = len(g); bi = np.arange(n)[:, None]
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    kr = kv[bi, reach]; kd = kv[bi, decoy]                            # (B, 11, dh)
    out = {}
    for name, c in comps.items():
        qc = f2(f1(c[:, 0])) @ Wq.T                    # (B, dh)
        out[name] = ((kr - kd) @ qc[:, :, None])[..., 0].mean(0).numpy() / math.sqrt(dh)
    total = sum(out.values())
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    import json
    Mj = json.load(open("theory/results/exp2_membership.json"))
    name = a.run.rstrip("/").split("/")[-1]
    head = a.head if a.head is not None else Mj[name]["heads"][0]
    m, c = load(a.run, "cpu")
    g, br, E = probe(a.n)
    np.set_printoptions(precision=1, suppress=True, linewidth=170)
    for s in a.steps:
        out, total = attribute(m, c["K"], g, br, E, s, head)
        print(f"== {name} head {head}: pointer in h_{s} (output of step {s}); cols = src distance 1..9")
        print("   %-12s %s" % ("TOTAL", total[:9]))
        for k_, v in sorted(out.items(), key=lambda kv: -np.abs(kv[1][:9]).max()):
            if np.abs(v[:9]).max() >= 0.4:
                print("   %-12s %s" % (k_, v[:9]))


if __name__ == "__main__":
    main()
