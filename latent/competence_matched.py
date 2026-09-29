"""Depth-matched competence: at each within-branch depth j, balanced accuracy of
reachable-branch nodes (distance j+1) vs shape-matched decoy-branch nodes at the same depth j.
Unlike per-distance recall, this cannot be inflated by a bias toward answering 'reachable'.

    python -m latent.competence_matched runs/a runs/b ... --out results/competence_matched.json
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
def matched(run, device, n=2048, seed=777):
    m, c = load(run, device)
    K = c["K"]
    rng = np.random.default_rng(seed)
    g = sample_graphs(rng, n, split="confirm")
    q = all_queries(g)[:, : NB * NPB]
    T = lambda a: torch.as_tensor(a, device=device)
    pred = []
    for i in range(0, n, 256):
        sl = slice(i, min(n, i + 256))
        pred.append((m.run(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl]), K, qnodes=T(q[sl]))["logit"] > 0).cpu().numpy())
    pred = np.concatenate(pred).reshape(n, NB, NPB)
    att = np.broadcast_to(g.attached[:, :, None], pred.shape)
    out = {}
    for j in range(NPB):
        mj = g.bdepth == j
        pos, neg = pred[mj & att], pred[mj & ~att]
        tpr, tnr = float(pos.mean()), float(1 - neg.mean())
        out[j + 1] = {"bal_acc": 0.5 * (tpr + tnr), "recall_reachable": tpr, "specificity_decoy": tnr,
                      "n_pos": int(pos.size), "n_neg": int(neg.size)}
    print(run, " ".join(f"d{d}:{v['bal_acc']:.2f}" for d, v in out.items()), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:1")
    a = p.parse_args()
    json.dump({r: matched(r, a.device) for r in a.runs}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
