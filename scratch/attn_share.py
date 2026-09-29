"""Exploratory (discovery split): attention mass from latent positions onto each branch's edges vs W."""
import sys

import numpy as np
import torch

from latent.competence import load
from latent.data import to_torch
from latent.graphs import E, NB, sample_graphs

run = sys.argv[1] if len(sys.argv) > 1 else "runs/r3_pc_gated_gpt2"
dev = "cuda:1"
m, c = load(run, dev)
K = c["K"]
rng = np.random.default_rng(11)
out = {}
for W in (1, 2, 4, 8):
    g = sample_graphs(rng, 400, widths=[W], split="discovery")
    # branch id of each edge position: hub edges and internal edges of branch b
    N = len(g)
    edge_branch = np.full((N, E), -1)
    for b in range(NB):
        edge_branch[np.arange(N), g.hub_pos[:, b]] = b
    # internal edges: dst is a non-root node of branch b
    node2b = {}
    dst = g.dst
    for b in range(NB):
        for j in range(1, g.nodes.shape[2]):
            mask = dst == g.nodes[:, b, j][:, None]
            edge_branch[mask] = b
    for blk in m.blocks:
        blk.attn.record = []
    src, dst_t, start = to_torch(g, dev)
    with torch.no_grad():
        m.run(src, dst_t, start, K)
    res = []
    for li, blk in enumerate(m.blocks):
        rec = blk.attn.record                       # K entries (latent steps), each (N, H, T)
        blk.attn.record = None
        for k, a in enumerate(rec):
            a = a.mean(1).cpu().numpy()[:, :E]       # head-averaged attention onto edge tokens
            att_mass, dec_mass = [], []
            for n in range(N):
                per = np.array([a[n][edge_branch[n] == b].sum() for b in range(NB)])
                att_mass.append(per[g.attached[n]].mean())
                dec_mass.append(per[~g.attached[n]].mean())
            res.append((li, k + 1, float(np.mean(att_mass)), float(np.mean(dec_mass))))
    out[W] = res
for li in range(len(m.blocks)):
    print(f"layer {li}: mean attention mass per reachable branch (per decoy branch) from LAT_k")
    for k in range(1, K + 1):
        row = []
        for W in (1, 2, 4, 8):
            t = [r for r in out[W] if r[0] == li and r[1] == k][0]
            row.append(f"W{W}: {t[2]:.3f} ({t[3]:.3f})")
        print(f"  LAT_{k}: " + "  ".join(row))
