"""Exploratory: can the latent search follow chains longer than any seen in training?

Graphs: every branch a 6-node chain (p_chain = 1). Two edits, each moving one hub edge's source
(edge count, node count and token positions unchanged):
  * decoy chain B is re-hung from s' onto the leaf of an attached chain A  -> B nodes at distance 7..12
  * decoy chain C is re-hung from s' onto the leaf of a decoy chain D      -> matched unreachable 12-chain
Balanced accuracy at distance 7 + j pairs B's node at depth j with C's node at depth j. Labels are
checked against BFS. Budgets beyond the trained K reuse the last trained step embedding.

    python -m latent.length_gen runs/a runs/b ... --out results/length_gen.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import torch

from .competence import load
from .graphs import NB, NPB, SHAPES, UNREACH, bfs_dist, sample_graphs, set_shapes


def long_chain_graphs(rng, n, widths):
    # The probe is defined on chain branches. A fork-trained run (PROTOCOL7 T10, PROTOCOL9 fork family) has set the
    # global graph shape to "fork", which would override p_chain, so the probe is built under the default shape and
    # the run's setting restored (no change for default-shape runs; fix 2026-09-29, PROTOCOL7 amendment 9).
    saved = dict(SHAPES)
    set_shapes("default")
    try:
        g = sample_graphs(rng, n, widths=widths, split=None, p_chain=1.0)
    finally:
        set_shapes(saved["mode"], saved["p_fork"])
    B = len(g)
    bi = np.arange(B)
    r = rng.random((B, NB))
    A = np.argmax(np.where(g.attached, r, -1), 1)
    free = np.argsort(np.where(~g.attached, r, 2), 1)          # unattached first, random order
    Bb, C, D = free[:, 0], free[:, 1], free[:, 2]
    assert (~g.attached[bi, Bb]).all() and (~g.attached[bi, D]).all()
    src = g.src.copy()
    src[bi, g.hub_pos[bi, Bb]] = g.nodes[bi, A, NPB - 1]
    src[bi, g.hub_pos[bi, C]] = g.nodes[bi, D, NPB - 1]
    g2 = replace(g, src=src)
    d = bfs_dist(g2.src, g2.dst, g2.start)
    for j in range(NPB):
        assert (d[bi, g.nodes[bi, Bb, j]] == NPB + 1 + j).all()
        assert (d[bi, g.nodes[bi, C, j]] == UNREACH).all()
        assert (d[bi, g.nodes[bi, A, j]] == 1 + j).all()
    return g2, {"A": A, "B": Bb, "C": C, "D": D}


@torch.no_grad()
def evaluate(run, device, n=1024, seed=99, budgets=(0, 2, 4, 6, 8, 10, 12, 16), widths=(2, 4, 6)):
    m, c = load(run, device)
    K = c["K"]
    rng = np.random.default_rng(seed)
    g, br = long_chain_graphs(rng, n, list(widths))
    bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)   # (n, 4*NPB)
    T = lambda a: torch.as_tensor(a, device=device)
    out = {"K": K, "mode": c["mode"], "rows": []}
    for Kp in budgets:
        preds = []
        for i in range(0, n, 256):
            sl = slice(i, min(n, i + 256))
            h, cache, _ = m.encode(m.embed_prefix(T(g.src[sl]), T(g.dst[sl]), T(g.start[sl])))
            for k in range(1, Kp + 1):
                h, cache, _ = m.latent_step(h, min(k, K) if K > 0 else min(k, m.step_emb.num_embeddings - 1), cache)
            preds.append((m.query(cache, T(q[sl])) > 0).cpu().numpy())
        p = np.concatenate(preds).reshape(n, 4, NPB)
        pa, pb, pc, pd = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        brng = np.random.default_rng(0)
        idx = brng.integers(0, n, (1000, n))                    # graph-level bootstrap
        far_ci = {}
        for j in range(NPB):
            ba = 0.5 * (pb[:, j][idx].mean(1) + 1 - pc[:, j][idx].mean(1))
            far_ci[NPB + 1 + j] = [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))]
        row = {"K'": Kp, "far_ci": far_ci,
               "near": {1 + j: 0.5 * (pa[:, j].mean() + 1 - pd[:, j].mean()) for j in range(NPB)},
               "far": {NPB + 1 + j: 0.5 * (pb[:, j].mean() + 1 - pc[:, j].mean()) for j in range(NPB)},
               "far_tpr": {NPB + 1 + j: float(pb[:, j].mean()) for j in range(NPB)},
               "far_tnr": {NPB + 1 + j: float(1 - pc[:, j].mean()) for j in range(NPB)}}
        row = json.loads(json.dumps(row, default=float))
        out["rows"].append(row)
        print(f"{run} K'={Kp:2d} near " + " ".join(f"{v:.2f}" for v in row["near"].values()) +
              " | far " + " ".join(f"{v:.2f}" for v in row["far"].values()), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    json.dump({r: evaluate(r, a.device) for r in a.runs}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
