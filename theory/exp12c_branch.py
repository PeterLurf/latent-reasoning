"""Exploratory (after §26 and its controls): re-anchoring with branching over the whole decoded frontier.

Episode 2 is run from each of the top-M most-attended edges' targets at episode 1's step K (sweep head), M = 1, 3, 6,
max-combined with episode 1 (M = 1 and 3 reproduce exp12's re2 / re2top3). Long-chain probe(512), 59 P30 ∪ L models.
    python theory/exp12c_branch.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from exp12_reanchor import bal_by_dist  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402

OUT = "theory/results/exp12c_branch.json"


@torch.no_grad()
def measure(m, K, g, br, head, Ms=(1, 3, 6)):
    T = lambda a: torch.as_tensor(a)
    n = len(g); bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1)
    src, dst = T(g.src), T(g.dst)
    o1 = rollout(m, K, src, dst, T(g.start), K, qnodes=T(q))
    a = o1["attn1"][K - 1][:, head, :o1["P"] - 1]
    order = torch.argsort(a, 1, descending=True).numpy()
    q2 = [rollout(m, K, src, dst, T(g.dst[bi, order[:, r]]), K, qnodes=T(q))["qlogit"].numpy() for r in range(max(Ms))]
    out = {}
    for M in Ms:
        out[f"top{M}"] = bal_by_dist(np.maximum(o1["qlogit"].numpy(), np.max(q2[:M], 0)), n)
    return out


if __name__ == "__main__":
    R = json.load(open("theory/results/exp12_reanchor.json"))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    g, br, _ = probe(512)
    for name in sorted(R, key=lambda r: (R[r]["extrap"], r)):
        if name in res:
            continue
        m, c = load("runs/" + name, "cpu")
        o = measure(m, c["K"], g, br, R[name]["head"]); o["extrap"] = R[name]["extrap"]
        res[name] = o
        json.dump(res, open(OUT, "w", newline="\n"), indent=1)
        print(name, " | ".join("%s %s" % (k, " ".join("%.2f" % x for x in o[k][6:10])) for k in ("top1", "top3", "top6")), flush=True)
