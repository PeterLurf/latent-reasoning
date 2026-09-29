"""Seed 0, single rewind at step 7 (own step-3 thought), K' = 20: far-distance accuracy by width, TPR/TNR."""
import sys

import numpy as np
import torch

from latent.competence import load
from latent.graphs import NPB
from latent.length_gen import long_chain_graphs
from latent.rewind_multi import run

run_ = sys.argv[1] if len(sys.argv) > 1 else "runs/r3_pc_gated_gpt2"
dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
m, c = load(run_, dev)
rng = np.random.default_rng(81)
n = 1536
g, br = long_chain_graphs(rng, n, [2, 4, 6])
bi = np.arange(n)
q = np.concatenate([g.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)
for name, st in (("baseline", ()), ("rewind@7", (7,))):
    p = run(m, c["K"], g, q, 20, dev, st, "fixed3").reshape(n, 4, NPB)
    pb, pc = p[:, 1], p[:, 2]
    print(name)
    for W in (2, 4, 6):
        s = g.width == W
        tpr = pb[s].mean(0); tnr = 1 - pc[s].mean(0)
        print(f"  W={W}: TPR d7..12 " + " ".join(f"{v:.2f}" for v in tpr) + " | TNR " + " ".join(f"{v:.2f}" for v in tnr))
    # per-graph: are far positives all-or-nothing?
    allpos = pb.all(1).mean(); nonepos = (~pb.astype(bool)).all(1).mean()
    print(f"  graphs with all 6 far positives called reachable: {allpos:.2f}; none: {nonepos:.2f}")
