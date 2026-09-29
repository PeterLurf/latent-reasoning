import numpy as np
import torch

from latent.competence import load
from latent.flex import forward
from latent.graphs import sample_graphs

m, c = load("runs/main_s1", "cuda:1")
g = sample_graphs(np.random.default_rng(0), 64, split="discovery")
T = lambda a: torch.as_tensor(a, device="cuda:1")
tgt = g.nodes[np.arange(64), np.argmax(g.attached, 1), 2]
with torch.no_grad():
    a = m.run(T(g.src), T(g.dst), T(g.start), 6, target=T(tgt))["logit"]
    b = forward(m, T(g.src), T(g.dst), T(g.start), 6, None, target=T(tgt))["logit"]
print("max |diff| upfront flex vs run:", float((a - b).abs().max()))
