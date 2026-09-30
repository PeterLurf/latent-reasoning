"""Feasibility for iterated re-anchoring distillation (THEORY §37; exploratory, CPU): can a Pilot L student (one loop of
K' = 12, distilled from re-anchoring) be re-anchored itself to answer at distances 13-24?

Deep probe: theory/deep_chain.py, R = 4 (24-hop reachable chain vs look-alike decoy), widths 2-3, 512 graphs, seed 7.
Per model (sweep head from analyse_pilotA.ptr_traj on rollout.probe(512)):
  * single24 : one episode, K' = 24
  * single12 : one episode, K' = 12
  * re12top3 : episode 1 of 12 steps; episode 2 (12 steps) from each of the 3 edges most attended by the sweep head at
               step 12 (their targets as START); logits max-combined (§26's re2top3 with 12-step episodes)
Balanced accuracy by distance j = 1..24: 0.5 (TPR on reach[j] + TNR on decoy[j]).
    python theory/exp18_deep_reanchor.py --out theory/results/exp18_deep_reanchor.json [--runs ...] [--r5]
--r5 (THEORY §38, descriptive) adds r5_single32: one episode at K' = 32 on a 30-hop probe (R = 5, width 1, 512 graphs,
seed 8), balanced accuracy by distance 1..30.
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
from analyse_pilotA import ptr_traj  # noqa: E402
from deep_chain import deep_chain_graphs  # noqa: E402
from eval_arch import load_arch  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


def bal(q, L):
    p = q > 0
    return [float(0.5 * (p[:, j].mean() + 1 - p[:, L + j].mean())) for j in range(L)]


@torch.no_grad()
def run(m, K, head, g, s, ep=12):
    L = s["reach"].shape[1]
    Q = T(np.concatenate([s["reach"], s["decoy"]], 1))
    src, dst = T(g.src), T(g.dst)
    n = len(g); bi = np.arange(n)
    out = {}
    o24 = rollout(m, K, src, dst, T(g.start), 24, qnodes=Q)
    out["single24"] = bal(o24["qlogit"].numpy(), L)
    o1 = rollout(m, K, src, dst, T(g.start), ep, qnodes=Q)
    out["single12"] = bal(o1["qlogit"].numpy(), L)
    best = o1["qlogit"]
    a = o1["attn1"][ep - 1][:, head, :o1["P"] - 1]
    order = torch.argsort(a, 1, descending=True).numpy()
    for r in range(3):
        anc = g.dst[bi, order[:, r]]
        best = torch.maximum(best, rollout(m, K, src, dst, T(anc), ep, qnodes=Q)["qlogit"])
    out["re12top3"] = bal(best.numpy(), L)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp18_deep_reanchor.json")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--r5", action="store_true")
    a = ap.parse_args()
    H = [102, 109, 111, 112, 113, 115, 118, 119, 125, 126, 130, 131, 132, 135, 140, 145, 146, 147, 157, 158]
    names = a.runs or [f"theory/runs/pl_t8_s{s}" for s in H]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, s = deep_chain_graphs(np.random.default_rng(7), 512, [2, 3], 4)
    pg, pb, pE = probe(512)
    if a.r5:
        g5, s5 = deep_chain_graphs(np.random.default_rng(8), 512, [1], 5)
    for path in names:
        name = os.path.basename(path.rstrip("/\\"))
        if name in res:
            continue
        m, c = (load_arch if path.startswith("theory/") else load)(path, "cpu")
        with torch.no_grad():
            _, _, head = ptr_traj(m, c["K"], pg, pb, pE, {})
        o = run(m, c["K"], head, g, s)
        o["head"] = head
        if a.r5:
            with torch.no_grad():
                Q5 = T(np.concatenate([s5["reach"], s5["decoy"]], 1))
                o["r5_single32"] = bal(rollout(m, c["K"], T(g5.src), T(g5.dst), T(g5.start), 32, qnodes=Q5)["qlogit"].numpy(), 30)
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda k: " ".join("%.2f" % x for x in o[k][12:18])
        print("%-14s | d13-18 single24 %s | single12 %s | re12top3 %s" % (name, f("single24"), f("single12"), f("re12top3")), flush=True)


if __name__ == "__main__":
    main()
