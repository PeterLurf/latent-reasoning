import numpy as np
import torch

from latent.competence import load
from latent.graphs import move_branch, sample_graphs
from latent.interventions import run_thoughts

dev = "cuda:1"
m, c = load("scratch", dev)
rng = np.random.default_rng(0)
g = sample_graphs(rng, 300, widths=[4], split="discovery")
att = np.stack([np.nonzero(a)[0] for a in g.attached])[:, 0]
h = run_thoughts(m, g, 6, device=dev)
hh = run_thoughts(m, move_branch(g, att, False, "hub"), 6, device=dev)
hg = run_thoughts(m, move_branch(g, att, False, "graft", rng=rng), 6, device=dev)
for k in range(7):
    d1 = (hh[:, k] - h[:, k]).norm(dim=1).mean().item()
    d2 = (hg[:, k] - hh[:, k]).norm(dim=1).mean().item()
    print(f"k={k}  |don_hub - rec| = {d1:.4f}   |don_graft - don_hub| = {d2:.6f}   |rec| = {h[:, k].norm(dim=1).mean():.2f}")
