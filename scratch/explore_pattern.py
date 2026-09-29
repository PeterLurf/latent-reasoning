"""Exploratory (discovery split only): probe filters vs Haufe patterns as branch subspaces."""
import numpy as np
import torch

from latent.competence import load
from latent.graphs import sample_graphs
from latent.interventions import bootstrap_ci, effect_matrix, summarize
from latent.probes import collect, probe_directions
from latent.study import _load_probes

run, dev = "runs/r3_pc_gated_gpt2", "cuda:1"
m, c = load(run, dev)
K = c["K"]
probes = _load_probes(f"{run}/discovery/probes_reach.pt", dev)
bases = torch.load(f"{run}/discovery/pca_bases.pt")
rng = np.random.default_rng(3003)
g_cov = sample_graphs(rng, 3000, split="discovery")
H = collect(m, g_cov, K, dev)
g4 = sample_graphs(rng, 300, widths=[4], split="discovery")
for k in range(4):
    Hk = H[:, k]
    C = torch.cov(Hk.T)
    pat = C @ probe_directions(probes[k])                     # (d, V)
    P = dict(probes[k]); P["pattern"] = pat
    rows = {}
    for kind, kw in [("full", {}), ("pattern", {"probe": P}), ("probe", {"probe": probes[k]}),
                     ("pca", {"rank": 6, "pca": bases})]:
        s = summarize(effect_matrix(m, g4, K, k, kind=kind, device=dev, **kw))
        rows[kind] = (bootstrap_ci(s["targeted"])[0], bootstrap_ci(s["collateral_attached"])[0],
                      bootstrap_ci(s["collateral_signed"])[0])
    print(f"k={k} " + " | ".join(f"{n}: tgt {a:+.3f} col {b:.3f} signed {c_:+.3f}" for n, (a, b, c_) in rows.items()))
