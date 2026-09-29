"""Check GraphedStep with several captured K against an eager reference (fp32, small batch).

    python scratch/test_krandom.py <device>
"""
import copy
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from latent.data import make_batch
from latent.graphs import E, sample_graphs
from latent.graphstep import GraphedStep
from latent.model import LatentReasoner

dev = sys.argv[1]
B, Q = 64, 16
torch.manual_seed(0)
base = LatentReasoner(256, 2, 8, k_max=8, init="gpt2").to(dev)


class Eager(GraphedStep):
    def _capture(self):
        pass

    def step(self, batch, K=None):
        for k, v in self.inp.items():
            v.copy_(batch[k])
        self.opt.zero_grad(set_to_none=False)
        self._fwd_bwd(K)


def make(Ks, cls):
    m = copy.deepcopy(base)
    lr = torch.tensor(1e-3, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98), capturable=True)
    if cls is Eager:                       # materialise optimizer state like the captured version does
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
        with torch.no_grad():
            for p, p0 in zip(m.parameters(), base.parameters()):
                p.copy_(p0)
            for st in opt.state.values():
                for v in st.values():
                    v.zero_()
    g = cls(m, opt, Ks, B, E, Q, 1.0, dev, None)
    g.w_aux.fill_(1.0); g.w_children.fill_(1.0)
    return m, g


rng = np.random.default_rng(0)
batches = [make_batch(rng, sample_graphs(rng, B, split="train", max_bdepth=None), Q, dev) for _ in range(12)]
seq = [8, 4, 6, 5, 8, 7, 4, 4, 6, 8, 5, 7]

runs = {"eager": make(list(range(4, 9)), Eager), "multi": make(list(range(4, 9)), GraphedStep)}
worst = 0.0
for i, (b, K) in enumerate(zip(batches, seq)):
    ls = {}
    for name, (m, g) in runs.items():
        g.step(b, K)
        ls[name] = g.loss.item()
    d = max((p - q).abs().max().item() for p, q in zip(runs["eager"][0].parameters(), runs["multi"][0].parameters()))
    worst = max(worst, d)
    print(f"step {i} K={K} loss eager {ls['eager']:.6f} multi {ls['multi']:.6f} max|dparam| {d:.2e}")

# single-K path: graphed K=6 vs eager K=6
runs = {"eager": make([6], Eager), "single": make([6], GraphedStep)}
for i, b in enumerate(batches[:6]):
    ls = {n: (g.step(b, 6), g.loss.item())[1] for n, (m, g) in runs.items()}
    d = max((p - q).abs().max().item() for p, q in zip(runs["eager"][0].parameters(), runs["single"][0].parameters()))
    worst = max(worst, d)
    print(f"single step {i} loss eager {ls['eager']:.6f} graphed {ls['single']:.6f} max|dparam| {d:.2e}")
print("WORST", worst, "OK" if worst < 1e-4 else "MISMATCH")
