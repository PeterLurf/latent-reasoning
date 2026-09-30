"""Deep long-chain probe for iterated re-anchoring distillation (THEORY §37): a reachable chain of 6·R hops and a
look-alike unreachable chain of the same length, built like latent/length_gen.long_chain_graphs (unchanged; this file
only generalises it from R = 2 to R = 3 or 4 branches).

Reachable: attached branch A, then free branches B1..B{R-1}, each hung from the previous branch's leaf.
Decoy: free branch D, then C1..C{R-1} hung the same way (unreachable). Needs 2R - 1 free branches besides A, so R = 4
works for widths <= 3 (NB = 10) and R = 3 for widths <= 5.
    g, seqs = deep_chain_graphs(rng, n, widths=[2, 3], R=4)   # seqs["reach"], seqs["decoy"]: (n, 6R) node ids
"""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.graphs import NB, NPB, SHAPES, UNREACH, bfs_dist, sample_graphs, set_shapes  # noqa: E402


def deep_chain_graphs(rng, n, widths, R):
    saved = dict(SHAPES)
    set_shapes("default")
    try:
        g = sample_graphs(rng, n, widths=widths, split=None, p_chain=1.0)
    finally:
        set_shapes(saved["mode"], saved["p_fork"])
    bi = np.arange(n)
    r = rng.random((n, NB))
    A = np.argmax(np.where(g.attached, r, -1), 1)
    free = np.argsort(np.where(~g.attached, r, 2), 1)
    need = 2 * R - 1
    assert (~g.attached[bi[:, None], free[:, :need]]).all(), "not enough free branches for R at these widths"
    reach_br = [A] + [free[:, i] for i in range(R - 1)]
    decoy_br = [free[:, R - 1 + i] for i in range(R)]
    src = g.src.copy()
    for chain in (reach_br, decoy_br):
        for prev, nxt in zip(chain[:-1], chain[1:]):
            src[bi, g.hub_pos[bi, nxt]] = g.nodes[bi, prev, NPB - 1]
    g2 = replace(g, src=src)
    reach = np.concatenate([g.nodes[bi, b] for b in reach_br], 1)
    decoy = np.concatenate([g.nodes[bi, b] for b in decoy_br], 1)
    d = bfs_dist(g2.src, g2.dst, g2.start)
    for j in range(NPB * R):
        assert (d[bi, reach[:, j]] == 1 + j).all()
        assert (d[bi, decoy[:, j]] == UNREACH).all()
    return g2, {"reach": reach, "decoy": decoy, "R": R}


if __name__ == "__main__":
    g, s = deep_chain_graphs(np.random.default_rng(0), 64, [2, 3], 4)
    print("ok", s["reach"].shape, s["decoy"].shape)
