"""Exploratory (PROTOCOL6 follow-up): when an open-sweep model fails to extrapolate, does its frontier stop at the
trained horizon (propagation failure) or keep advancing past it while the answer cannot use it (readout failure)?

Long-chain graphs of latent/length_gen.py (seed 99, widths 2/4/6): chain A attached (distance 1..6), chain B hung
from A's leaf (distance 7..12), matched unreachable chains D (1..6 lookalike) and C (7..12 lookalike).
Layer-1 attention of the latent position per step k = 1..12 onto each chain edge (per_graph_attention).
  * near[k][j] = mean attn on A's edge j - D's edge j   (in-horizon frontier, distance j+1)
  * far[k][j]  = mean attn on B's edge j - C's edge j   (beyond-horizon frontier, distance 7+j)
  * in_peak = max over steps and in-horizon edges of near  (how strongly the model marks its frontier at all)
  * e7      = max over steps of far[k][0]                   (the distance-7 edge, the one d7 queries depend on)
  * beyond  = max over steps 7..12 and far edges of far       (any beyond-horizon frontier activity)
  * P7 = e7 / in_peak; Pb = beyond / in_peak                 (~1: the sweep crosses the horizon; ~0: it stops)
  * far_peak_step: for each far edge j, the step at which its excess peaks
  (clock-free on purpose: many models do not advance one hop per step.)  K = 6 models only.
    PYTHONPATH=. python scratch/far_sweep.py runs/a runs/b ... --out results/far_sweep.json --device cuda:1
"""
import argparse
import json
import os

import numpy as np
import torch

from latent.competence import load
from latent.frontier_amp import per_graph_attention
from latent.graphs import NPB
from latent.length_gen import long_chain_graphs


@torch.no_grad()
def measure(run, device, n=1024, Kp=12, seed=99):
    m, c = load(run, device)
    K = c["K"]
    g, br = long_chain_graphs(np.random.default_rng(seed), n, [2, 4, 6])
    A, _, _ = per_graph_attention(m, K, g, Kp, device)            # (Kp, N, E)
    bi = np.arange(n)
    def edge(b, j):
        return g.hub_pos[bi, b] if j == 0 else np.argmax(g.dst == g.nodes[bi, b, j][:, None], 1)
    E = {k: [edge(br[k], j) for j in range(NPB)] for k in ("A", "B", "C", "D")}
    att = lambda k, ch, j: float(A[k, bi, E[ch][j]].mean())
    near = [[att(k, "A", j) - att(k, "D", j) for j in range(NPB)] for k in range(Kp)]
    far = [[att(k, "B", j) - att(k, "C", j) for j in range(NPB)] for k in range(Kp)]
    in_peak = float(np.max(np.array(near)[:K]))
    e7 = float(max(far[k][0] for k in range(Kp)))
    beyond = float(np.max(np.array(far)[K:]))
    lag = [int(np.argmax([far[k][j] for k in range(Kp)])) + 1 for j in range(NPB)]
    return {"K": K, "near": near, "far": far, "in_peak": in_peak, "e7": e7, "beyond": beyond,
            "P7": e7 / in_peak, "Pb": beyond / in_peak, "far_peak_step": lag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/far_sweep.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=99, help="long-chain graph seed (99 = the length_gen graphs)")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for r in a.runs:
        if r in res:
            continue
        res[r] = measure(r, a.device, seed=a.seed)
        json.dump(res, open(a.out, "w"), indent=1)
        o = res[r]
        print(r, "in_peak %.4f e7 %.4f beyond %.4f P7 %.2f Pb %.2f far_peak_step %s" % (
            o["in_peak"], o["e7"], o["beyond"], o["P7"], o["Pb"], o["far_peak_step"]), flush=True)


if __name__ == "__main__":
    main()
