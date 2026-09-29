"""THEORY §26 controls (requested by the experiments session): is the re-anchoring gain specific to the model's own
frontier decode, or does any fresh episode help?

Two-episode re-anchoring (re2) on the 59 P30 ∪ L models, long-chain probe(512), with four anchors for episode 2:
  model      : dst of the edge token with the largest sweep-head attention at episode 1's step K (as exp12)
  rand_node  : a uniformly random node present in the graph (not START)
  rand_depth : a uniformly random node reachable from START at the same BFS distance as the model's anchor
               (if the model's anchor is not reachable: a uniformly random reachable node)
  oracle     : the chain node (A then B) at the model anchor's BFS distance (if unreachable: at distance 4)
Balanced accuracy by distance as exp12 (near 1..6: A vs D; far 7..12: B vs C). RNG seeded per model (seed 0).
    python theory/exp12b_controls.py --out theory/results/exp12b_controls.json
"""
import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from exp12_reanchor import bal_by_dist  # noqa: E402
from latent.graphs import NPB  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402


def bfs(src, dst, s):
    adj = {}
    for a, b in zip(src, dst):
        adj.setdefault(int(a), []).append(int(b))
    d = {int(s): 0}; q = deque([int(s)])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


@torch.no_grad()
def measure(m, K, g, br, head, seed=0):
    rng = np.random.default_rng(seed)
    T = lambda a: torch.as_tensor(a)
    n = len(g); bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1)
    chain = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)       # distance j+1 at column j
    src, dst = T(g.src), T(g.dst)
    o1 = rollout(m, K, src, dst, T(g.start), K, qnodes=T(q))
    q1 = o1["qlogit"].numpy()
    a = o1["attn1"][K - 1][:, head, :o1["P"] - 1]
    model_anc = g.dst[bi, a.argmax(1).numpy()]
    D = [bfs(g.src[i], g.dst[i], g.start[i]) for i in range(n)]
    anchors = {"model": model_anc, "rand_node": np.zeros(n, int), "rand_depth": np.zeros(n, int), "oracle": np.zeros(n, int)}
    for i in range(n):
        present = np.setdiff1d(np.unique(np.concatenate([g.src[i], g.dst[i]])), [g.start[i]])
        anchors["rand_node"][i] = rng.choice(present)
        dm = D[i].get(int(model_anc[i]))
        reach = [v for v, dv in D[i].items() if dv > 0]
        same = [v for v, dv in D[i].items() if dm is not None and dv == dm]
        anchors["rand_depth"][i] = rng.choice(same if same else reach)
        dd = dm if dm is not None and 1 <= dm <= 2 * NPB else 4
        anchors["oracle"][i] = chain[i, dd - 1]
    out = {}
    for k, anc in anchors.items():
        q2 = rollout(m, K, src, dst, T(anc), K, qnodes=T(q))["qlogit"].numpy()
        out[k] = bal_by_dist(np.maximum(q1, q2), n)
    out["model_anchor_reachable"] = float(np.mean([D[i].get(int(model_anc[i])) is not None for i in range(n)]))
    out["model_anchor_on_chain"] = float(np.mean([(chain[i] == model_anc[i]).any() for i in range(n)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp12b_controls.json")
    a = ap.parse_args()
    R = json.load(open("theory/results/exp12_reanchor.json"))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, _ = probe(512)
    for name in sorted(R, key=lambda r: (R[r]["extrap"], r)):
        if name in res:
            continue
        m, c = load("runs/" + name, "cpu")
        o = measure(m, c["K"], g, br, R[name]["head"])
        o.update(extrap=R[name]["extrap"], K=c["K"])
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda v: " ".join("%.2f" % x for x in v[6:10])
        print("%-10s %s | model %s | rand_node %s | rand_depth %s | oracle %s" % (
            name, "EXT" if o["extrap"] else "   ", f(o["model"]), f(o["rand_node"]), f(o["rand_depth"]), f(o["oracle"])), flush=True)


if __name__ == "__main__":
    main()
