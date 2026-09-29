"""Existing models on fork-mode graphs (no training): at fixed W, per-edge excess layer-1 attention on depth-j
frontier edges in fork graphs (frontier 2W) divided by chain graphs (frontier W), at the step that reads depth j
(from results/frontier_amp.json). Budget split per frontier edge -> 0.5; per branch -> 1.0.
Fork-mode graphs are in the default generator's support but rare there.   -> results/frontier_amp_fork.json"""
import json
import numpy as np
from latent.competence import load
from latent.graphs import sample_graphs, set_shapes
from latent.frontier_amp import per_graph_attention

R = json.load(open("results/frontier_amp.json")); out = {}
for run in ["runs/r3_pc_gated_gpt2", "runs/main_s1", "runs/main_s2", "runs/main_s3"]:
    m, c = load(run, "cpu"); set_shapes("fork", 0.5)               # load() resets shapes to the run's (default)
    g = sample_graphs(np.random.default_rng(99), 8000, widths=[2, 4, 8], split=None); set_shapes("default")
    A, dep, att = per_graph_attention(m, c["K"], g, 8, "cpu"); W = g.width
    kids = np.stack([(g.parent[:, 0] == j).sum(1) for j in range(6)], 1)
    fork = kids.max(1) == 2; chain = ~fork
    out[run] = {}
    for j in (1, 2, 3):
        k = R[run][str(j)]["step"] - 1
        msk, dm = att & (dep == j), ~att & (dep == j); n = msk.sum(1)
        ex = (A[k] * msk).sum(1) / np.maximum(n, 1) - (A[k] * dm).sum(1) / np.maximum(dm.sum(1), 1)
        assert (n[chain] == W[chain]).all()
        fj = fork & (n == 2 * W)
        out[run][j] = {int(w): [float(ex[fj & (W == w)].mean() / ex[chain & (W == w)].mean()), int((fj & (W == w)).sum())]
                       for w in (2, 4, 8) if (fj & (W == w)).sum() >= 30}
    print(run, {j: {w: round(r[0], 2) for w, r in v.items()} for j, v in out[run].items()})
json.dump(out, open("results/frontier_amp_fork.json", "w"), indent=1)
