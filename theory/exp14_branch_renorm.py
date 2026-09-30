"""THEORY §30: F3a renormalisation (RN1–RN2) and B1 branch selection (amendment B1-a) on the 59 P30 ∪ L models.

F3a: rollout to K' = 12 with exp13's renorm feed (past K, each thought's deviation from the batch mean rescaled to its
     example's mean in-range norm), without and with the layer-2 block from K; bal-acc by distance (exp12 convention).
B1:  per step k = 1..K, per layer-1 head (and head-mean): among the frontier edges (edges whose source is at the BFS depth of
     the source of the sweep head's most-attended edge at that step), (i) the number with attention >= 0.5 x the frontier
     maximum, (ii) exp(entropy) of the attention renormalised over frontier edges; averaged over graphs, split by W.
    python theory/exp14_branch_renorm.py --out theory/results/exp14_branch_renorm.json
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
from exp13_decode_restart import bfs_depth, renorm_feed  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402


@torch.no_grad()
def branch_stats(m, K, g, head, D):
    T = lambda a: torch.as_tensor(a)
    o = rollout(m, K, T(g.src), T(g.dst), T(g.start), K)
    n = len(g); P = o["P"]
    widths = sorted(set(g.width.tolist()))
    out = {}
    for k in range(1, K + 1):
        A = o["attn1"][k - 1][:, :, :P - 1].numpy()                    # (n, H, E) attention on edge tokens
        H = A.shape[1]
        top = A[:, head].argmax(1)
        rows = {w: [[] for _ in range(H + 1)] for w in widths}
        for i in range(n):
            dp = D[i].get(int(g.src[i, top[i]]))
            if dp is None:
                continue
            fr = np.array([D[i].get(int(s)) == dp for s in g.src[i]])
            if fr.sum() < 1:
                continue
            for hh in range(H + 1):
                a = A[i, hh, fr] if hh < H else A[i][:, fr].mean(0)   # A[i][:, fr]: (H, n_fr); A[i, :, fr] would be (n_fr, H)
                if a.max() <= 0:
                    continue
                cnt = float((a >= 0.5 * a.max()).sum())
                q = a / a.sum(); eff = float(np.exp(-(q * np.log(q + 1e-12)).sum()))
                rows[int(g.width[i])][hh].append((cnt, eff))
        out[k] = {w: [[float(np.mean([r[0] for r in rows[w][hh]])) if rows[w][hh] else None,
                       float(np.mean([r[1] for r in rows[w][hh]])) if rows[w][hh] else None] for hh in range(H + 1)]
                  for w in widths}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp14_branch_renorm.json")
    ap.add_argument("--branch_only", action="store_true", help="recompute the branch statistics of every stored model")
    a = ap.parse_args()
    R = json.load(open("theory/results/exp12_reanchor.json"))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, _ = probe(512)
    D = [bfs_depth(g.src[i], g.dst[i], g.start[i]) for i in range(len(g))]
    T = lambda x: torch.as_tensor(x)
    bi = np.arange(len(g))
    q = T(np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1))
    if a.branch_only:
        for name in sorted(res):
            m, c = load("runs/" + name, "cpu")
            res[name]["branch"] = branch_stats(m, c["K"], g, res[name]["head"], D)
            json.dump(res, open(a.out, "w", newline="
"), indent=1)
            print(name, "branch recomputed", flush=True)
        return
    for name in sorted(R, key=lambda r: (R[r]["extrap"], r)):
        if name in res:
            continue
        m, c = load("runs/" + name, "cpu")
        K, head = c["K"], R[name]["head"]
        o = {"K": K, "head": head, "extrap": R[name]["extrap"]}
        with torch.no_grad():
            for tag, kw in (("renorm", {}), ("renorm_block", {"block_from": K, "block_layers": (1,)})):
                r = rollout(m, K, T(g.src), T(g.dst), T(g.start), 12, qnodes=q, feed=renorm_feed(K), **kw)
                o[tag] = bal_by_dist(r["qlogit"].numpy(), len(g))
        o["branch"] = branch_stats(m, K, g, head, D)
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        bK = o["branch"][K]
        print("%-10s %s | renorm %s | +blk %s | step-K sweep head eff W2/4/6: %s | head-mean eff: %s" % (
            name, "EXT" if o["extrap"] else "   ", " ".join("%.2f" % x for x in o["renorm"][6:10]),
            " ".join("%.2f" % x for x in o["renorm_block"][6:10]),
            " ".join("%.2f" % bK[w][head][1] for w in sorted(bK)), " ".join("%.2f" % bK[w][-1][1] for w in sorted(bK))), flush=True)


if __name__ == "__main__":
    main()
