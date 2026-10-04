"""THEORY note 101-1 premise ($0): k-means (M = 240) on frozen §97 HYB-4 last-slot output states u over 20k fresh chunks;
purity (share of points whose cluster's majority true state equals their own) and NMI against the true S5 state after
the chunk. u = mu_E + sd_E * (h - mu_h) / sd_h (per coordinate, stats from the collected h; §101's g). CPU only.
Gate (fixed in 101-1, before this ran): SELF-SNAP runs only if the median purity over s1-3 >= 0.90.
    python theory/s101_premise.py --threads 3
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import gen97 as G  # noqa: E402
from boxes import s5  # noqa: E402


@torch.no_grad()
def last_slot_h(m, start_els, ops):
    """Replays Gen97.chunk without forks; returns the last slot's output hidden (B, D)."""
    g = m.g
    x = torch.cat([m.start_emb(start_els), g.embed(m.optok[ops[:, 0]])], 1); past = None
    for j in range(ops.shape[1]):
        h, past = g(x, past); s_in = h[:, -1]
        hs, past = g(s_in[:, None], past); hslot = hs[:, 0]
        if j + 1 < ops.shape[1]:
            x = g.embed(m.optok[ops[:, j + 1]])
    return hslot


def kmeans(X, M, iters, rng, restarts=3):
    best = None
    for r in range(restarts):
        idx = [int(rng.integers(len(X)))]; d2 = ((X - X[idx[0]]) ** 2).sum(1)
        for _ in range(M - 1):                                              # k-means++
            p = (d2 / d2.sum()).numpy(); i = int(rng.choice(len(X), p=p)); idx.append(i)
            d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(1))
        C = X[idx].clone()
        for _ in range(iters):
            a = torch.cdist(X, C).argmin(1)
            for c in range(M):
                sel = a == c
                C[c] = X[sel].mean(0) if sel.any() else X[int(rng.integers(len(X)))]
        a = torch.cdist(X, C).argmin(1); inertia = float(((X - C[a]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, a.numpy())
    return best[1]


def purity_nmi(a, y):
    N = len(y); pur = sum(np.bincount(y[a == c]).max() for c in np.unique(a)) / N
    def H(v):
        p = np.bincount(v) / len(v); p = p[p > 0]; return float(-(p * np.log(p)).sum())
    joint = {}
    for ai, yi in zip(a, y):
        joint[(ai, yi)] = joint.get((ai, yi), 0) + 1
    pa, py = np.bincount(a) / N, np.bincount(y) / N
    mi = sum(c / N * math.log((c / N) / (pa[ai] * py[yi])) for (ai, yi), c in joint.items())
    return float(pur), float(mi / math.sqrt(H(a) * H(y)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=3); ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    res = {}
    for s in (1, 2, 3):
        m = G.Gen97(s5.load_gpt2("cpu"), "hyb"); m.load_state_dict(torch.load(os.path.join(ROOT, "boxes", "runs_s5", "g97_hyb4_s%d" % s, "model.pt"), map_location="cpu")); m.eval()
        rng = np.random.default_rng(10100 + s); S = rng.integers(0, 120, a.n); O = rng.integers(0, 120, (a.n, 4))
        Y = G.true_chain(S, O)[:, -1]; H = []
        for i in range(0, a.n, 250):
            H.append(last_slot_h(m, torch.tensor(S[i:i + 250]), torch.tensor(O[i:i + 250])).float())
        Hm = torch.cat(H); wte = m.g.w("wte.weight").detach().float()
        U = wte.mean(0) + wte.std(0) * (Hm - Hm.mean(0)) / Hm.std(0).clamp_min(1e-6)
        asg = kmeans(U, 240, 30, np.random.default_rng(7 + s))
        pur, nmi = purity_nmi(asg, Y); res[s] = {"purity": pur, "nmi": nmi, "n": a.n, "clusters_used": int(len(np.unique(asg)))}
        print("s%d purity %.4f nmi %.4f clusters %d" % (s, pur, nmi, res[s]["clusters_used"]), flush=True)
    med = float(np.median([v["purity"] for v in res.values()]))
    out = {"runs": res, "median_purity": med, "gate_selfsnap_runs": med >= 0.90}
    print("median purity %.4f -> SELF-SNAP runs: %s" % (med, out["gate_selfsnap_runs"]))
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "s101_premise.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
