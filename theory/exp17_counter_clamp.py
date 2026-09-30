"""THEORY §36: counter clamp — hold the iteration count carried in the thoughts inside its trained range at inference.

§35 (P10e) located the fall-back's key in an iteration count carried in the thought, separable from node identity. The
batch mean of the thoughts at step i over many graphs, m_i, carries no node identity (identities average out), so it is
the candidate for that count. Clamp: from thought index i0 on, feed h_i - m_i + m_t(i) instead of h_i, where the target
index t(i) stays in range:
  * freeze_c : t(i) = c for every i >= c + 1 (c = 2, 3, 4)
  * cycle    : t(i) = 2 + ((i - 2) mod 3), i.e. 2, 3, 4, 2, 3, 4, ... for i >= 2 (reproduces an in-range stretch)
  * rand_c   : control for freeze_c, h_i + r_i with r_i a fixed random direction of norm |m_c - m_i| (per step)
m_i is the batch mean over the probe's 512 graphs of the thoughts actually produced in this rollout (so it can be read at
any step, also past K). Outcomes on the long-chain probe (512, seed 4247) at K' = 12: balanced accuracy by distance
(exp12's bal_by_dist, index j = distance j + 1), 7-hop success = bal-acc(7) >= 0.60 (§26's criterion), d8, and the
pointer (ptr_traj's aggregate rule, fixed head) with its peak depth (exp16's rule).
    python theory/exp17_counter_clamp.py --out theory/results/exp17_counter_clamp.json [--runs ...]
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
from eval_arch import load_arch  # noqa: E402
from exp12_reanchor import bal_by_dist  # noqa: E402
from exp16_stop_key import agg_ptr, peaks  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


def clamp_feed(target, rand=False, seed=0):
    """target(i) -> in-range index or None (no clamp). hist holds the produced thoughts h_0..h_{k-1}."""
    gen = torch.Generator().manual_seed(seed)
    dirs = {}

    def feed(k, h, hist):
        i = k - 1                                           # h is h_i, fed into step k
        t = target(i)
        if t is None or t == i:
            return h
        mi, mt = h.mean(0, keepdim=True), hist[t].mean(0, keepdim=True)
        if not rand:
            return h - mi + mt
        if i not in dirs:
            r = torch.randn(1, h.shape[1], generator=gen)
            dirs[i] = r / r.norm()
        return h + dirs[i] * (mt - mi).norm()
    return feed


VARIANTS = {
    "freeze2": (lambda i: 2 if i >= 3 else None, False),
    "freeze3": (lambda i: 3 if i >= 4 else None, False),
    "freeze4": (lambda i: 4 if i >= 5 else None, False),
    "cycle": (lambda i: 2 + (i - 2) % 3 if i >= 2 else None, False),
    "rand3": (lambda i: 3 if i >= 4 else None, True),
}


@torch.no_grad()
def run(m, K, head, g, br, E):
    n = len(g); bi = np.arange(n)
    q = T(np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1))
    src, dst, st = T(g.src), T(g.dst), T(g.start)
    out = {}
    for name, (tgt, rnd) in [("base", (lambda i: None, False))] + list(VARIANTS.items()):
        o = rollout(m, K, src, dst, st, 12, qnodes=q, feed=clamp_feed(tgt, rnd))
        b = bal_by_dist(o["qlogit"].numpy(), n)
        p = agg_ptr(o, E, head)
        (pd, ps), = peaks(np.array([p]))
        out[name] = {"bal": b, "d7": b[6], "d8": b[7], "ok7": bool(b[6] >= 0.60), "ptr": p, "peak_depth": int(pd), "peak_step": int(ps),
                     "delayed_in_range": float(np.mean(b[:K]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp17_counter_clamp.json")
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    R = json.load(open("theory/results/exp12_reanchor.json"))
    M = json.load(open("theory/results/pilotDE_meas.json"))
    if a.runs:
        names = a.runs
    else:
        names = ["runs/" + r for r in sorted(R, key=lambda r: (R[r]["extrap"], r))]
        names += ["theory/runs/" + r for r in sorted(M) if (r.startswith("pd_at_") or r.startswith("pe8_at_")) and "ptr" in M[r]]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(512)
    for path in names:
        name = os.path.basename(path.rstrip("/\\"))
        if name in res:
            continue
        m, c = (load_arch if path.startswith("theory/") else load)(path, "cpu")
        if name in R:
            head = R[name]["head"]
        elif name in M:
            head = M[name]["head"]
        else:
            from analyse_pilotA import ptr_traj
            _, _, head = ptr_traj(m, c["K"], g, br, E, {})
        o = run(m, c["K"], head, g, br, E)
        o.update(K=c["K"], head=head, extrap=R[name]["extrap"] if name in R else None,
                 group="p30L" if name in R else ("anytime" if name in M else "smoke"),
                 re2top3_d7=R[name]["re2top3"][6] if name in R else None)
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda k: "%.2f/%.2f %d" % (o[k]["d7"], o[k]["d8"], o[k]["peak_depth"])
        print("%-14s %-7s K %d | d7/d8 peak: base %s | fr2 %s | fr3 %s | fr4 %s | cyc %s | rand3 %s | re2top3 d7 %s" % (
            name, o["group"], c["K"], f("base"), f("freeze2"), f("freeze3"), f("freeze4"), f("cycle"), f("rand3"),
            "%.2f" % o["re2top3_d7"] if o["re2top3_d7"] is not None else "-"), flush=True)


if __name__ == "__main__":
    main()
