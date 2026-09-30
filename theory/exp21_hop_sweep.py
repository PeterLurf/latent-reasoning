"""THEORY §46: where does a written anchor help most? Toy counterpart of PROTOCOL8b's hop sweep. CPU, measurement only.

For each of the 59 P30 ∪ L models: episode 1 (K steps from START) + one restart episode (K steps) from an ORACLE anchor
= the reachable chain's node at distance k (k = 1..10), answers max-combined (§26). Long-chain probe(512). For every
reachable chain node at distance d (7..12) the reachable/look-alike balanced accuracy is recorded, indexed by hops
remaining r = d - k (r <= 0: already at or past the anchor).
    python theory/exp21_hop_sweep.py --out theory/results/exp21_hop_sweep.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "2")))
from exp12_reanchor import bal_by_dist  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


@torch.no_grad()
def run(m, K, g, br):
    n = len(g); bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1)
    Q = T(q)
    reach = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)
    base = rollout(m, K, T(g.src), T(g.dst), T(g.start), K, qnodes=Q)["qlogit"].numpy()
    out = {"base": bal_by_dist(base, n)}
    for k in range(1, 11):
        e = rollout(m, K, T(g.src), T(g.dst), T(reach[:, k - 1]), K, qnodes=Q)["qlogit"].numpy()
        out[f"k{k}"] = bal_by_dist(np.maximum(base, e), n)          # index j -> distance j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp21_hop_sweep.json")
    a = ap.parse_args()
    R = json.load(open("theory/results/exp12_reanchor.json"))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, _ = probe(512)
    for name in sorted(R):
        if name in res:
            continue
        m, c = load("runs/" + name, "cpu")
        res[name] = run(m, c["K"], g, br)
        res[name]["K"] = c["K"]
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
