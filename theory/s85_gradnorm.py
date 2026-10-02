"""THEORY §85 (iii), descriptive: gradient norm of the exact-marginal loss at the default free-table init (0.1 randn),
std vs ge3 batches (lengths drawn from each curriculum's first stage and from its full mix), A5, M 120, 5 seeds.
    python theory/s85_gradnorm.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c1_group import comp_table  # noqa: E402
from c1_rl import Cell  # noqa: E402


def gnorm(m, COMP, e, L, rng, bs=256, tau=4.0):
    A = COMP.shape[1]; M = m.M; Ct = torch.as_tensor(COMP)
    ops = torch.as_tensor(rng.integers(0, A, (bs, L))); s = torch.full((bs,), e, dtype=torch.long)
    for t in range(L):
        s = Ct[s, ops[:, t]]
    T = F.softmax(m.table() / tau, -1); alpha = torch.zeros(bs, M); alpha[:, 0] = 1.0
    for t in range(L):
        alpha = torch.einsum("bi,bij->bj", alpha, T[:, ops[:, t]].permute(1, 0, 2))
    loss = -torch.log((alpha * F.softmax(m.D, -1)[:, s].T).sum(1).clamp_min(1e-12)).mean()
    m.zero_grad(); loss.backward()
    return float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters())))


def main():
    torch.set_num_threads(3); COMP, e, A = comp_table("A5"); res = {}
    for L in (1, 2, 3, 4, 6, 8):
        vals = []
        for seed in range(5):
            torch.manual_seed(2121 + seed); m = Cell("tab", 120, A)
            vals.append(gnorm(m, COMP, e, L, np.random.default_rng(2121 + seed)))
        res[L] = {"median": float(np.median(vals)), "all": vals}
        print("A5 L", L, "grad norm median %.3e" % res[L]["median"], flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "s85_gradnorm.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
