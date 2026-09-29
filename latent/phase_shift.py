"""Exploratory: is seed 0's halting phase a linear, graph-independent direction in the thought?

mu_k = mean thought after step k over a calibration batch (unedited runs, separate graphs). At step j the
fed thought h is shifted: h + alpha * (mu_s - mu_{j-1}); a norm-matched random direction is the control.
Evaluated on long-chain graphs (distances 1..12) and on ordinary discovery graphs (depth-matched, 1..6).

    python -m latent.phase_shift runs/r3_pc_gated_gpt2 --out results/phase_shift_s0.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .competence import load
from .graphs import NB, NPB, sample_graphs
from .interventions import all_queries
from .length_gen import long_chain_graphs
from .rewind import accuracy


@torch.no_grad()
def thoughts(m, K, g, Kp, device):
    T = lambda a: torch.as_tensor(a, device=device)
    hs = []
    for i in range(0, len(g), 256):
        sl = slice(i, min(len(g), i + 256))
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
        th = [h]
        for k in range(1, Kp + 1):
            h, cache, _ = m.latent_step(h, min(k, K), cache)
            th.append(h)
        hs.append(torch.stack(th, 1))
    return torch.cat(hs)                         # (N, Kp+1, d)


@torch.no_grad()
def run(m, K, g, q, Kp, device, j=None, delta=None):
    T = lambda a: torch.as_tensor(a, device=device)
    out = []
    for i in range(0, len(g), 256):
        sl = slice(i, min(len(g), i + 256))
        h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
        for k in range(1, Kp + 1):
            if j is not None and k == j:
                h = h + delta
            h, cache, _ = m.latent_step(h, min(k, K), cache)
        out.append((m.query(cache, T(q[sl])) > 0).cpu().numpy())
    return np.concatenate(out)


def matched_acc(pred, g):
    n = len(g)
    p = pred.reshape(n, NB, NPB)
    att = np.broadcast_to(g.attached[:, :, None], p.shape)
    return {j + 1: 0.5 * (float(p[(g.bdepth == j) & att].mean()) + 1 - float(p[(g.bdepth == j) & ~att].mean())) for j in range(NPB)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=1024)
    a = ap.parse_args()
    m, c = load(a.run, a.device)
    K, dev = c["K"], a.device
    rng = np.random.default_rng(90)
    gc, _ = long_chain_graphs(rng, 512, [2, 4, 6])
    mu = thoughts(m, K, gc, 12, dev).mean(0)                     # (13, d) calibration means
    g, br = long_chain_graphs(rng, a.n, [2, 4, 6])
    bi = np.arange(a.n)
    q = np.concatenate([g.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)
    go = sample_graphs(rng, a.n, split="discovery")
    qo = all_queries(go)[:, : NB * NPB]
    gen = torch.Generator(device=dev).manual_seed(0)
    far = lambda d: float(np.mean([d[x] for x in (7, 8, 9)]))
    res = {"run": a.run, "mu_norms": mu.norm(dim=1).tolist(), "rows": []}
    base = accuracy(run(m, K, g, q, 12, dev), a.n)
    base_o = matched_acc(run(m, K, go, qo, 12, dev), go)
    res["baseline"] = {"chain": base, "ordinary": base_o}
    print(f"baseline far {far(base):.3f} | ordinary d4-6 " + " ".join(f"{base_o[d]:.2f}" for d in (4, 5, 6)), flush=True)
    for j in (6, 7, 8):
        for s in (1, 2, 3):
            for alpha in (0.5, 1.0, 1.5):
                dv = alpha * (mu[s] - mu[j - 1])
                rnd = torch.randn(dv.shape, generator=gen, device=dev)
                rnd = rnd / rnd.norm() * dv.norm()
                for kind, delta in (("phase", dv), ("random", rnd)):
                    if kind == "random" and alpha != 1.0:
                        continue
                    ch = accuracy(run(m, K, g, q, 12, dev, j, delta), a.n)
                    od = matched_acc(run(m, K, go, qo, 12, dev, j, delta), go)
                    row = {"j": j, "s": s, "alpha": alpha, "kind": kind, "norm": float(dv.norm()), "chain": ch, "ordinary": od}
                    res["rows"].append(row)
                    print(f"j={j} s={s} a={alpha:.1f} {kind:6s} |d|={dv.norm():.2f}  far {far(ch):.3f}  near3-6 "
                          f"{np.mean([ch[x] for x in (3, 4, 5, 6)]):.3f} | ordinary d4-6 " + " ".join(f"{od[d]:.2f}" for d in (4, 5, 6)), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
