"""Exploratory mechanism measures (discovery split only), per run -> results/extra.json.

1. Energy of the single-branch donor difference captured by each subspace at k*
   (probe filters, Haufe patterns, PCA-6/20, random-6), vs the 6/256 random baseline.
2. Layer-wise attention budget from latent positions onto a reachable vs decoy branch's edges.

    python -m latent.mechanism runs/r3_pc_gated_gpt2 --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .competence import load
from .data import to_torch
from .graphs import E, NB, move_branch, sample_graphs
from .interventions import make_basis, run_thoughts
from .study import _load_probes


@torch.no_grad()
def energy_fractions(model, run, K, k, device, n=400, W=4):
    probes = _load_probes(os.path.join(run, "discovery", "probes_reach.pt"), device)
    pats = torch.load(os.path.join(run, "discovery", "patterns.pt"))
    pca = torch.load(os.path.join(run, "discovery", "pca_bases.pt"))
    P = dict(probes[k]); P["pattern"] = pats[k].to(device)
    rng = np.random.default_rng(9090)
    g = sample_graphs(rng, n, widths=[W], split="discovery")
    att = np.stack([np.nonzero(a)[0] for a in g.attached])[:, 0]
    h = run_thoughts(model, g, K, device=device)[:, k]
    hd = run_thoughts(model, move_branch(g, att, False), K, device=device)[:, k]
    diff = hd - h
    gen = torch.Generator(device=device).manual_seed(0)
    out = {}
    for name, kind, kw in [("probe filters (6)", "probe", {"probe": probes[k]}),
                           ("branch patterns (6)", "pattern", {"probe": P}),
                           ("PCA (6)", "pca", {"probe": {"pca": pca}, "rank": 6}),
                           ("PCA (20)", "pca", {"probe": {"pca": pca}, "rank": 20}),
                           ("random (6)", "random", {"rank": 6})]:
        U = make_basis(kind, h, hd, g, att, k, gen=gen, **kw)
        frac = ((U.transpose(1, 2) @ diff[:, :, None]).norm(dim=1)[:, 0] ** 2 / diff.norm(dim=1) ** 2)
        out[name] = float(frac.mean())
    return out


@torch.no_grad()
def attention_budget(model, K, device, n=400):
    rng = np.random.default_rng(11)
    res = {}
    for W in (1, 2, 4, 8):
        g = sample_graphs(rng, n, widths=[W], split="discovery")
        N = len(g)
        eb = np.full((N, E), -1)
        for b in range(NB):
            eb[np.arange(N), g.hub_pos[:, b]] = b
            for j in range(1, g.nodes.shape[2]):
                eb[g.dst == g.nodes[:, b, j][:, None]] = b
        for blk in model.blocks:
            blk.attn.record = []
        src, dst, start = to_torch(g, device)
        model.run(src, dst, start, K)
        for li, blk in enumerate(model.blocks):
            rec = blk.attn.record
            blk.attn.record = None
            for k, a in enumerate(rec):
                a = a.mean(1).cpu().numpy()[:, :E]
                per = np.stack([(a * (eb == b)).sum(1) for b in range(NB)], 1)      # (N, NB)
                att_m = np.nanmean(np.where(g.attached, per, np.nan), 1)
                dec_m = np.nanmean(np.where(~g.attached, per, np.nan), 1)
                res.setdefault(f"layer{li + 1}", {}).setdefault(f"LAT{k + 1}", {})[W] = {
                    "reachable": float(att_m.mean()), "decoy": float(dec_m.mean()),
                    "excess": float((att_m - dec_m).mean())}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default="results/extra.json")
    a = p.parse_args()
    path = a.out
    X = json.load(open(path)) if os.path.exists(path) else {}
    for run in a.runs:
        model, c = load(run, a.device)
        d = json.load(open(os.path.join(run, "discovery", "discovery.json")))
        k = d["k_star"]
        X[run] = {"k_star": k, "energy_W4": energy_fractions(model, run, c["K"], k, a.device),
                  "attention": attention_budget(model, c["K"], a.device)}
        print(run, X[run]["energy_W4"])
    os.makedirs("results", exist_ok=True)
    json.dump(X, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
