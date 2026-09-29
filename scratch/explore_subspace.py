"""Exploratory (discovery split only): where does the donor difference live?"""
import numpy as np
import torch

from latent.competence import load
from latent.graphs import move_branch, sample_graphs
from latent.interventions import make_basis, run_thoughts
from latent.study import _load_probes

run, dev = "runs/r3_pc_gated_gpt2", "cuda:1"
m, c = load(run, dev)
probes = _load_probes(f"{run}/discovery/probes_reach.pt", dev)
rng = np.random.default_rng(7)
g = sample_graphs(rng, 400, widths=[4], split="discovery")
att = np.stack([np.nonzero(a)[0] for a in g.attached])[:, 0]
h = run_thoughts(m, g, 6, device=dev)
hd = run_thoughts(m, move_branch(g, att, False), 6, device=dev)
gen = torch.Generator(device=dev).manual_seed(0)
for k in range(6):
    diff = hd[:, k] - h[:, k]
    Up = make_basis("probe", h[:, k], hd[:, k], g, att, k, probe=probes[k])
    Ur = make_basis("random", h[:, k], hd[:, k], g, att, k, gen=gen)
    fp = ((Up.transpose(1, 2) @ diff[:, :, None]).norm(dim=1)[:, 0] ** 2 / diff.norm(dim=1) ** 2).mean().item()
    fr = ((Ur.transpose(1, 2) @ diff[:, :, None]).norm(dim=1)[:, 0] ** 2 / diff.norm(dim=1) ** 2).mean().item()
    # PCA of diffs: how low-rank is the branch-removal difference across graphs?
    D = diff - diff.mean(0)
    s = torch.linalg.svdvals(D)
    ev = (s ** 2 / (s ** 2).sum()).cpu().numpy()
    # node-embedding subspace: span of the model's own id embeddings (node slot) for branch nodes
    E = m.id_emb.weight                                    # (V, d_id)
    print(f"k={k} |diff|={diff.norm(dim=1).mean():.2f} |h|={h[:, k].norm(dim=1).mean():.1f} "
          f"frac in probe-6 {fp:.4f}  random-6 {fr:.4f}  (6/256={6/256:.4f})  "
          f"PCA top1 {ev[0]:.2f} top5 {ev[:5].sum():.2f} top20 {ev[:20].sum():.2f}  |mean diff|/|diff| "
          f"{(diff.mean(0).norm() / diff.norm(dim=1).mean()).item():.2f}")
