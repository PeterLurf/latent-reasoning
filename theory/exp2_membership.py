"""Exploratory exp2: how strongly does each thought 'contain' each chain node, as read by the frontier heads?

Layer-1 keys of edge tokens depend only on the edge's (src, dst) embedding, so the latent query's logit on the
edge u->v measures how much the thought fed into step k points at u (the src). Membership of the chain node at
distance j in thought h_{k-1}:
    M[k, j] = mean over graphs of  logit(A-edge out of A's distance-j node) - logit(matched D-edge)
for the head(s) that carry the sweep (largest mean attention excess on chain A over steps 1..K).
Beyond the horizon the chain continues on B (distance 7..12), matched by C.
Output: per model, M (Kp x 11) for distances 1..11 (edge out of node at distance j = edge into j+1), per head.
    python theory/exp2_membership.py runs/p30_s105 ... --out theory/results/exp2_membership.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout import load, probe, rollout  # noqa: E402


@torch.no_grad()
def measure(run, n=512, Kp=12):
    m, c = load(run, "cpu")
    K = c["K"]
    g, br, E = probe(n)
    T = lambda a: torch.as_tensor(a)
    out = rollout(m, K, T(g.src), T(g.dst), T(g.start), Kp)
    bi = np.arange(n)[:, None]
    # edge out of the node at distance j (j = 1..11): A edge j (into distance j+1) for j <= 5, then B edges
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1)          # (n, 11): out of distance 1..11
    decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    H = out["logit1"][0].shape[1]
    L = np.zeros((Kp, H, 11)); A = np.zeros((Kp, H, 11))
    for k in range(Kp):
        s = out["logit1"][k].numpy()                              # (n, H, T)
        a = out["attn1"][k].numpy()
        for h in range(H):
            L[k, h] = (s[bi, h, reach] - s[bi, h, decoy]).mean(0)
            A[k, h] = (a[bi, h, reach] - a[bi, h, decoy]).mean(0)
    sweep_heads = np.argsort(-A[:K].clip(0).sum((0, 2)))
    return {"K": K, "L": L.tolist(), "A": A.tolist(), "heads": sweep_heads.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="theory/results/exp2_membership.json")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    np.set_printoptions(precision=1, suppress=True, linewidth=160)
    for r in a.runs:
        name = r.rstrip("/").split("/")[-1]
        if name not in res:
            res[name] = measure(r)
            json.dump(res, open(a.out, "w"))
        o = res[name]
        L = np.array(o["L"]); h = o["heads"][0]
        print(f"== {name} K={o['K']} top sweep heads {o['heads'][:3]}  (logit excess, head {h}; rows = step, cols = src distance 1..11)")
        print(L[:, h, :])


if __name__ == "__main__":
    main()
