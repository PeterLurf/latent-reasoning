"""THEORY §24 PG5: trajectory of the monotone set (A6) on the long-chain probe — measurement only.

For each step k = 1..12 the set m_k (model._m after latent step k) is read at the chain's nodes: distance d = 1..6 on
chain A, 7..12 on its continuation B, against the matched look-alike nodes (D, then C). excess[k][d] = mean over
graphs of m_k(chain node at d) - m_k(decoy node at d). Set depth at step k = largest d such that excess[k][d'] >= 0.5
for every d' <= d (0 if none).
    python theory/set_traj.py theory/runs/pg_g_s900 ...
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from eval_arch import load_arch  # noqa: E402
from latent.graphs import NPB  # noqa: E402
from rollout import probe  # noqa: E402

OUT = "theory/results/set_traj.json"


@torch.no_grad()
def traj(m, g, br, Kp=12):
    T = lambda a: torch.as_tensor(a)
    n = len(g); bi = np.arange(n)
    ids = lambda ch: np.stack([g.nodes[bi, br[ch], j] for j in range(NPB)], 1)      # (n, NPB) node ids along a chain
    # chain node j sits at distance j + 1 (edge 0 of a chain enters its node 0 from the hub; exp2's convention)
    reach = np.concatenate([ids("A"), ids("B")], 1)                                 # distances 1..12 (A then B)
    decoy = np.concatenate([ids("D"), ids("C")], 1)
    h, cache, _ = m.encode(m.embed_prefix(T(g.src), T(g.dst), T(g.start)))
    ex = []
    for k in range(1, Kp + 1):
        h, cache, _ = m.latent_step(h, k, cache)
        s = m._m.float().numpy()
        ex.append((s[bi[:, None], reach] - s[bi[:, None], decoy]).mean(0).tolist())
    depth = []
    for row in ex:
        d = 0
        while d < len(row) and row[d] >= 0.5:
            d += 1
        depth.append(d)
    return ex, depth


def main(paths):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    g, br, _ = probe(512)
    for path in paths:
        path = path.rstrip("/\\"); name = os.path.basename(path)
        if name in res or not os.path.exists(os.path.join(path, "model.pt")):
            continue
        m, c = load_arch(path, "cpu")
        if not hasattr(m, "s_gate"):
            continue
        ex, depth = traj(m, g, br)
        res[name] = {"K": c["K"], "gate": float(m.s_gate), "excess": ex, "set_depth": depth}
        json.dump(res, open(OUT, "w", newline="\n"), indent=1)
        print(name, "gate %.3f" % float(m.s_gate), "set depth by step", " ".join(map(str, depth)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
