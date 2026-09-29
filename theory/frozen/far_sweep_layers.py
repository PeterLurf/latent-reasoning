"""PROTOCOL7 T9/T10: P7 per layer, and the layer chosen without outcome data (largest in-horizon excess, in_peak).
Same probe graphs and definitions as the frozen far_sweep_frozen_I.py (seed 2033), for every block instead of the first.
Written 2026-09-28 ~23:52, before any T9/T10 run existed.
    PYTHONPATH=. python scratch/far_sweep_layers.py runs/a ... --out results/far_sweep_layers_s2033.json --device cuda:0
"""
import argparse
import json
import os

import numpy as np
import torch

from latent.competence import load
from latent.graphs import E, NPB
from latent.length_gen import long_chain_graphs


@torch.no_grad()
def layer_attention(m, K, g, Kp, device, chunk=256):
    """(L, Kp, N, E): attention of the latent position onto each edge token in every block, heads averaged."""
    N = len(g)
    T = lambda a: torch.as_tensor(a, device=device)
    A = np.zeros((len(m.blocks), Kp, N, E), np.float32)
    for i in range(0, N, chunk):
        s = slice(i, min(N, i + chunk))
        for blk in m.blocks:
            blk.attn.record = []
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[s]), T(g.dst[s]), T(g.start[s])))
        for k in range(1, Kp + 1):
            h, cache, _ = m.latent_step(h, min(k, K), cache)
        for li, blk in enumerate(m.blocks):
            for k, a in enumerate(blk.attn.record):
                A[li, k, s] = a.mean(1).float().cpu().numpy()[:, :E]
            blk.attn.record = None
    return A


def measure(run, device, n=1024, Kp=12, seed=2033):
    m, c = load(run, device)
    K = c["K"]
    g, br = long_chain_graphs(np.random.default_rng(seed), n, [2, 4, 6])
    A = layer_attention(m, K, g, Kp, device)
    bi = np.arange(n)
    edge = lambda b, j: g.hub_pos[bi, b] if j == 0 else np.argmax(g.dst == g.nodes[bi, b, j][:, None], 1)
    Ed = {k: [edge(br[k], j) for j in range(NPB)] for k in "ABCD"}
    out = {"K": K, "layers": {}}
    for li in range(A.shape[0]):
        att = lambda k, ch, j: float(A[li, k, bi, Ed[ch][j]].mean())
        near = [[att(k, "A", j) - att(k, "D", j) for j in range(NPB)] for k in range(Kp)]
        far = [[att(k, "B", j) - att(k, "C", j) for j in range(NPB)] for k in range(Kp)]
        in_peak = float(np.max(np.array(near)[:K]))
        e7 = float(max(far[k][0] for k in range(Kp)))
        out["layers"][li + 1] = {"near": near, "far": far, "in_peak": in_peak, "e7": e7, "P7": e7 / in_peak,
                                 "P8": float(max(far[k][1] for k in range(Kp))) / in_peak}
    best = max(out["layers"], key=lambda li: out["layers"][li]["in_peak"])
    out["chosen_layer"] = best
    out.update({k: out["layers"][best][k] for k in ("P7", "P8", "in_peak", "e7")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for r in a.runs:
        if r in res:
            continue
        res[r] = measure(r, a.device)
        print(f"{r} layer {res[r]['chosen_layer']} in_peak {res[r]['in_peak']:.3f} P7 {res[r]['P7']:.2f}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
