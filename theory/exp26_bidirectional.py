"""THEORY §53 (frontier F2): bidirectional latent reasoning, zero-shot on trained toy models. CPU, inference only.

For a target t, the model is run twice, one episode of K steps each:
  * forward : the graph as given, START = root; F = graph nodes the model answers reachable
  * backward: the same graph with every edge reversed (src <-> dst), START = t; B = graph nodes the model answers
              reachable, i.e. nodes from which t can be reached (ancestors of t)
Bidirectional answer: "t reachable" iff F ∩ B is non-empty (both sets are written node-id sets).
Probe: rollout.probe(512) long-chain graphs (chain A then B: distance 1..12; look-alikes D then C, unreachable).
Targets: every chain node and its look-alike. Balanced accuracy by distance 1..12; also, per distance, the
false-positive rate on look-alikes (Literature's precision check) and forward-only / §26 references.
    python theory/exp26_bidirectional.py --out theory/results/exp26_bidirectional.json [--runs ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from rollout import load, probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


@torch.no_grad()
def reach_set(m, K, src, dst, start, nodes):
    """(n, N) bool: model's answer 'reachable from start' for each candidate node (one episode of K steps)."""
    o = rollout(m, K, T(src), T(dst), T(start), K, qnodes=T(nodes))
    return o["qlogit"].numpy() > 0


@torch.no_grad()
def run(m, K, g, br):
    n = len(g); bi = np.arange(n)
    nodes = np.unique(np.concatenate([g.src, g.dst], 1), axis=None)
    # per-graph candidate node list: all node ids appearing in that graph's edges (padded with the root)
    cand = [np.unique(np.r_[g.src[i], g.dst[i]]) for i in range(n)]
    W = max(len(c) for c in cand)
    C = np.stack([np.r_[c, np.full(W - len(c), c[0])] for c in cand])
    F = reach_set(m, K, g.src, g.dst, g.start, C) | (C == g.start[:, None])
    reach = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)       # distance j+1
    decoy = np.concatenate([g.nodes[bi, br["D"]], g.nodes[bi, br["C"]]], 1)
    out = {"bidir_reach": [], "bidir_decoy_fp": [], "fwd_reach": [], "fwd_decoy_fp": []}
    for j in range(12):
        res = {}
        for name, tg in (("reach", reach[:, j]), ("decoy", decoy[:, j])):
            B = reach_set(m, K, g.dst, g.src, tg, C) | (C == tg[:, None])           # reversed edges, START = target
            res[name] = (F & B).any(1)
            res[name + "_fwd"] = F[bi, np.argmax(C == tg[:, None], 1)] & (C == tg[:, None]).any(1)
        out["bidir_reach"].append(float(res["reach"].mean())); out["bidir_decoy_fp"].append(float(res["decoy"].mean()))
        out["fwd_reach"].append(float(res["reach_fwd"].mean())); out["fwd_decoy_fp"].append(float(res["decoy_fwd"].mean()))
    out["bidir_bal"] = [0.5 * (a + 1 - b) for a, b in zip(out["bidir_reach"], out["bidir_decoy_fp"])]
    out["fwd_bal"] = [0.5 * (a + 1 - b) for a, b in zip(out["fwd_reach"], out["fwd_decoy_fp"])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp26_bidirectional.json")
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    R = json.load(open("theory/results/exp12_reanchor.json"))
    names = a.runs or sorted(R)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, _ = probe(512)
    for name in names:
        if name in res:
            continue
        path = name if os.path.isdir(name) else "runs/" + name
        m, c = load(path, "cpu")
        o = run(m, c["K"], g, br)
        o["K"] = c["K"]; o["extrap"] = R.get(os.path.basename(name), {}).get("extrap")
        o["re2top3"] = R.get(os.path.basename(name), {}).get("re2top3")
        res[os.path.basename(name.rstrip("/\\"))] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda v, lo, hi: np.mean(v[lo:hi])
        print("%-10s bidir d1-6 %.3f d7-12 %.3f | fwd d7-12 %.3f | decoy FP d7-12 %.3f" % (
            os.path.basename(name), f(o["bidir_bal"], 0, 6), f(o["bidir_bal"], 6, 12), f(o["fwd_bal"], 6, 12),
            f(o["bidir_decoy_fp"], 6, 12)), flush=True)


if __name__ == "__main__":
    main()
