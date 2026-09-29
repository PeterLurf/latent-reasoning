"""Does the delayed sweep's per-edge attention follow the frontier edge count or the width W?

TRAINING_TODO Phase 3 "check first": the default graphs already vary branch shape, so at fixed W the number
of reachable edges at a given depth (the frontier) varies. For each model, the step that reads depth-j edges
is the step with the largest mean layer-1 attention on them (the clocked sweep). At that (step, depth), per
graph: a = mean layer-1 attention (heads averaged) per reachable depth-j edge, n = number of such edges.
a is taken as excess over the same graph's decoy depth-j edges (the noise floor). Per depth, over graphs:
    log a = c + b_W log W + b_n log n
Fixed budget split among frontier edges -> b_n ~ -1, b_W ~ 0.  Budget split among branches -> b_W ~ -1, b_n ~ 0.
Also reports the total excess on the frontier (a * n) binned by W and n. (A first version fit raw
attention with depths pooled; the ~0.008 floor and the weak deep cells pushed b_W to -1.2 on main_s1 against a
clear 1/n pattern in the (W, n) table, so the floor is subtracted and depths are fit separately.)

    python -m latent.frontier_amp runs/r3_pc_gated_gpt2 runs/main_s1 ... --out results/frontier_amp.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .competence import load
from .graphs import E, NB, NPB, sample_graphs


@torch.no_grad()
def per_graph_attention(m, K, g, Kp, device, chunk=256):
    """(Kp, N, E) layer-1 attention of the latent position onto each edge token, plus edge depth / reachable masks."""
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
    A = np.zeros((Kp, N, E), np.float32)
    blk = m.blocks[0]
    for i in range(0, N, chunk):
        s = slice(i, min(N, i + chunk))
        blk.attn.record = []
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[s]), T(g.dst[s]), T(g.start[s])))
        for k in range(1, Kp + 1):
            h, cache, _ = m.latent_step(h, min(k, K), cache)
        for k, a in enumerate(blk.attn.record):
            A[k, s] = a.mean(1).float().cpu().numpy()[:, :E]
        blk.attn.record = None
    return A, dep, att


def _fit(lw, ln, ly, rng):
    X = np.column_stack([np.ones_like(lw), lw, ln])
    beta = np.linalg.lstsq(X, ly, rcond=None)[0]
    boot = [np.linalg.lstsq(X[i], ly[i], rcond=None)[0][1:] for i in (rng.integers(0, len(ly), len(ly)) for _ in range(500))]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"b_W": float(beta[1]), "b_n": float(beta[2]), "b_W_ci": [float(lo[0]), float(hi[0])],
            "b_n_ci": [float(lo[1]), float(hi[1])], "n_obs": int(len(ly)), "corr_logW_logn": float(np.corrcoef(lw, ln)[0, 1])}


def analyse(A, dep, att, W, depths=range(1, 5)):
    """Per depth: a = per-edge attention on reachable depth-j edges minus the same graph's per-edge attention on
    decoy depth-j edges (the noise floor, ~0.008), at the step that reads depth j; fit on graphs with a > 0."""
    rng = np.random.default_rng(0)
    out = {}
    for j in depths:
        msk, dm = att & (dep == j), ~att & (dep == j)
        n = msk.sum(1)
        ok = n > 0
        per = (A * msk[None]).sum(2) / np.maximum(n, 1)[None]          # (Kp, N)
        k_star = int(np.argmax(per[:, ok].mean(1)))
        ex = per[k_star] - (A[k_star] * dm).sum(1) / np.maximum(dm.sum(1), 1)
        keep = ok & (ex > 0)
        o = {"step": k_star + 1, "frac_positive": float(keep.sum() / ok.sum()),
             "total_excess_by_W_n": {f"W{w}_n{b}": float((ex * n)[ok & (W == w) & (n == b)].mean())
                                     for w in (2, 4, 8) for b in (w, 2 * w) if (ok & (W == w) & (n == b)).sum() >= 20}}
        o.update(_fit(np.log(W[keep]), np.log(n[keep]), np.log(ex[keep]), rng))
        out[j] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    res = {}
    for run in a.runs:
        m, c = load(run, a.device)
        g = sample_graphs(np.random.default_rng(321), a.n, widths=range(1, 9), split="discovery")
        A, dep, att = per_graph_attention(m, c["K"], g, 8, a.device)
        r = analyse(A, dep, att, g.width)
        res[run] = r
        for j, o in r.items():
            print(f"{run} depth {j} @ step {o['step']}: b_W {o['b_W']:+.2f} [{o['b_W_ci'][0]:+.2f},{o['b_W_ci'][1]:+.2f}]  "
                  f"b_n {o['b_n']:+.2f} [{o['b_n_ci'][0]:+.2f},{o['b_n_ci'][1]:+.2f}]  (n={o['n_obs']}, "
                  f"{o['frac_positive']:.0%} above floor, corr {o['corr_logW_logn']:.2f})", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
