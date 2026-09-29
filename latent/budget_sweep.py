"""Exploratory: depth-matched competence as a function of the inference-time latent budget K'.

Covers K' = 0..K (step/hop alignment: which within-branch depth resolves at which latent step) and
K' > K (test-time extrapolation). Beyond the training budget the step embedding is either the
untrained row k (``raw``, only up to k_max) or the last trained row K reused (``repeat``).
Discovery-split graphs only; this is not a confirmatory analysis.

    python -m latent.budget_sweep runs/a runs/b ... --out results/budget_sweep.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .competence import load
from .graphs import NB, NPB, sample_graphs
from .interventions import all_queries


@torch.no_grad()
def logits(m, g, sl, Kp, K, variant, qn, device):
    T = lambda a: torch.as_tensor(a, device=device)
    h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
    for k in range(1, Kp + 1):
        kk = k if (variant == "raw" or k <= K) else K
        h, cache, _ = m.latent_step(h, kk, cache)
    return m.query(cache, T(qn[sl]))


@torch.no_grad()
def sweep(run, device, n=2048, seed=4242, kmax_repeat=10):
    m, c = load(run, device)
    K, kmax = c["K"], m.step_emb.num_embeddings - 1
    rng = np.random.default_rng(seed)
    g = sample_graphs(rng, n, split="discovery")
    q = all_queries(g)[:, : NB * NPB]
    att = np.broadcast_to(g.attached[:, :, None], (n, NB, NPB))
    plan = [("train", k) for k in range(K + 1)] + [("raw", k) for k in range(K + 1, kmax + 1)] + \
           [("repeat", k) for k in range(K + 1, kmax_repeat + 1)]
    out = {"K": K, "mode": c["mode"], "rows": []}
    for variant, Kp in plan:
        pred = np.concatenate([(logits(m, g, slice(i, min(n, i + 256)), Kp, K, variant, q, device) > 0).cpu().numpy()
                               for i in range(0, n, 256)]).reshape(n, NB, NPB)
        row = {"variant": variant, "K'": Kp, "by_depth": {}}
        for j in range(NPB):
            mj = g.bdepth == j
            pos, neg = pred[mj & att], pred[mj & ~att]
            tpr, tnr = float(pos.mean()), float(1 - neg.mean())
            row["by_depth"][j + 1] = {"bal_acc": 0.5 * (tpr + tnr), "tpr": tpr, "tnr": tnr}
        allpos, allneg = pred[att], pred[~att]
        row["bal_acc"] = 0.5 * (float(allpos.mean()) + float(1 - allneg.mean()))
        row["by_width_depth"] = {}
        for W in np.unique(g.width):
            mw = np.broadcast_to((g.width == W)[:, None, None], pred.shape)
            row["by_width_depth"][int(W)] = {
                j + 1: 0.5 * (float(pred[mw & (g.bdepth == j) & att].mean()) + 1 - float(pred[mw & (g.bdepth == j) & ~att].mean()))
                for j in range(NPB)}
        out["rows"].append(row)
        print(f"{run} {variant:6s} K'={Kp:2d} all {row['bal_acc']:.3f} " +
              " ".join(f"d{d}:{v['bal_acc']:.2f}" for d, v in row["by_depth"].items()), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    json.dump({r: sweep(r, a.device) for r in a.runs}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
