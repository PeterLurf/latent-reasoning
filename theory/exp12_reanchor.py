"""THEORY §26 (CTM-inspired): re-anchored latent episodes at inference — training-free.

Continuous Thought Machines extend to longer maze paths by *re-applying* a bounded-horizon policy from the current
state (Darlow et al. 2505.05522). The latent analogue: run the trained K-step latent loop, decode where its frontier is,
and start a fresh K-step episode from that node, so the step count never leaves the trained range (no clock past K,
C2) and the pointer restarts from a crisp node embedding (amplitude restored, C3).

Per model, long-chain probe(512) (chain A -> B, 12 hops; look-alike chains D, C), K = trained budget:
  single  : one episode, K' = 12 (standard extrapolation; as exp5 'base')
  reE     : E episodes of K steps (E = 2, 3). Episode 1 starts at START; episode e+1 starts at the anchor of episode e =
            dst of the edge token with the largest sweep-head attention at the episode's last step K (model-internal:
            the edge the last step reads). Query logits are combined by max over episodes.
Balanced accuracy per distance d: near d = 1..6 = 0.5(TPR_A + TNR_D), far d = 7..12 = 0.5(TPR_B + TNR_C).
Also records the anchor's true chain distance (diagnostic; decoys = -1).
    python theory/exp12_reanchor.py [--runs NAME ...] --out theory/results/exp12_reanchor.json
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
from analyse_pilotA import ptr_traj  # noqa: E402
from latent.graphs import NPB  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402


def bal_by_dist(qmax, n):
    p = (qmax > 0).reshape(n, 4, NPB)
    A, B, C, D = (p[:, i].mean(0) for i in range(4))
    return [float(0.5 * (A[j] + 1 - D[j])) for j in range(NPB)] + [float(0.5 * (B[j] + 1 - C[j])) for j in range(NPB)]


@torch.no_grad()
def measure(m, K, g, br, E_edges, head, episodes=(2, 3)):
    T = lambda a: torch.as_tensor(a)
    n = len(g); bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1)
    chain = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)          # distance j+1 at column j
    src, dst = T(g.src), T(g.dst)
    single = rollout(m, K, src, dst, T(g.start), 12, qnodes=T(q))
    out = {"single": bal_by_dist(single["qlogit"].numpy(), n)}
    start = T(g.start); qmax = None; anchors = []
    for e in range(1, max(episodes) + 1):
        o = rollout(m, K, src, dst, start, K, qnodes=T(q))
        ql = o["qlogit"].numpy()
        qmax = ql if qmax is None else np.maximum(qmax, ql)
        if e in episodes:
            out[f"re{e}"] = bal_by_dist(qmax, n)
        a = o["attn1"][K - 1][:, head, :]                        # (n, T_K): step K's sweep-head attention
        P = o["P"]
        a = a[:, :P - 1]                                         # edge tokens only (START is position P-1)
        top = a.argmax(1).numpy()
        anc = g.dst[bi, top]
        dist = np.array([int(np.where(chain[i] == anc[i])[0][0]) + 1 if (chain[i] == anc[i]).any() else -1 for i in range(n)])
        anchors.append({"mean_dist": float(dist[dist > 0].mean()) if (dist > 0).any() else None,
                        "off_chain": float((dist < 0).mean())})
        start = T(anc)
    out["anchors"] = anchors
    # variant re2top3: episode 2 from each of the 3 most-attended edges' targets at episode 1's last step (branching)
    o1 = rollout(m, K, src, dst, T(g.start), K, qnodes=T(q))
    qmax = o1["qlogit"].numpy()
    a = o1["attn1"][K - 1][:, head, :o1["P"] - 1]
    order = torch.argsort(a, 1, descending=True).numpy()
    for r in range(3):
        anc = g.dst[bi, order[:, r]]
        qmax = np.maximum(qmax, rollout(m, K, src, dst, T(anc), K, qnodes=T(q))["qlogit"].numpy())
    out["re2top3"] = bal_by_dist(qmax, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--out", default="theory/results/exp12_reanchor.json")
    a = ap.parse_args()
    T = {r["name"]: r for r in json.load(open("theory/results/pointer_taxonomy.json"))}
    names = a.runs or sorted(T, key=lambda r: (T[r]["extrap"], r))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(512)
    for name in names:
        if name in res:
            continue
        path = name if os.path.isdir(name) else ("runs/" + name)
        m, c = load(path, "cpu")
        K = c["K"]
        _, _, head = ptr_traj(m, K, g, br, E, {})
        o = measure(m, K, g, br, E, head)
        o.update(K=K, head=head, extrap=T[name]["extrap"] if name in T else None)
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda v: " ".join("%.2f" % x for x in v[6:10])
        print("%-10s %s d7-10 single %s | re2 %s | re3 %s | anchors %s" % (
            name, "EXT" if o["extrap"] else "   ", f(o["single"]), f(o["re2"]), f(o["re3"]),
            [(x["mean_dist"] and round(x["mean_dist"], 1), round(x["off_chain"], 2)) for x in o["anchors"]]),
            "| re2top3", f(o["re2top3"]), flush=True)


if __name__ == "__main__":
    main()
