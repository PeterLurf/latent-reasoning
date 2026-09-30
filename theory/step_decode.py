"""THEORY §39 (T12 manipulation check): how well can a fresh probe read the step index from a trained model's thoughts?

For each run: roll out on training-distribution graphs (latent.graphs.sample_graphs, split "train", seed 1, 2048
graphs) at the run's K; collect thoughts h_1..h_K (delayed chain, rollout harness); train a fresh 2-layer probe
(d -> 128 -> K, GELU, Adam lr 3e-3, 400 steps, batch 1024) to predict k; report accuracy on held-out graphs (seed 2,
1024 graphs). Chance = 1/K. Measurement only; no labels about the problem are used.
    python theory/step_decode.py theory/runs/pd_at_s900 ... --out theory/results/step_decode.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "2")))
from eval_arch import load_arch  # noqa: E402
from latent.graphs import sample_graphs  # noqa: E402
from rollout import rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


@torch.no_grad()
def thoughts(m, K, seed, n):
    g = sample_graphs(np.random.default_rng(seed), n, split="train", max_bdepth=None)
    o = rollout(m, K, T(g.src), T(g.dst), T(g.start), K)
    h = o["thoughts"][:, 1:K + 1]                                   # (n, K, d): h_1..h_K
    y = torch.arange(K).expand(n, K)
    return h.reshape(-1, h.shape[-1]).float(), y.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="theory/results/step_decode.json")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for path in a.runs:
        name = os.path.basename(path.rstrip("/\\"))
        if name in res or not os.path.exists(os.path.join(path, "model.pt")):
            continue
        m, c = load_arch(path, "cpu")
        K = c["K"]
        xtr, ytr = thoughts(m, K, 1, 2048)
        xte, yte = thoughts(m, K, 2, 1024)
        torch.manual_seed(0)
        probe = nn.Sequential(nn.Linear(xtr.shape[1], 128), nn.GELU(), nn.Linear(128, K))
        opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
        for _ in range(400):
            i = torch.randint(0, len(xtr), (1024,))
            loss = F.cross_entropy(probe(xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = float((probe(xte).argmax(1) == yte).float().mean())
        res[name] = {"K": K, "step_acc": acc, "chance": 1 / K}
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        print(name, "K", K, "step decode acc %.3f (chance %.3f)" % (acc, 1 / K), flush=True)


if __name__ == "__main__":
    main()
