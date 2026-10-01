"""THEORY §56 (external unifying hypothesis, "entry-state normal form"): are latent states with the same residual problem
but different ages interchangeable? An interchange intervention (IIA) on trained toy models. CPU, inference only.

Probe: §37's 24-hop chain (deep_chain_graphs, seed 7, widths 2-3, R = 4, 512 graphs); c_j = the chain node at distance j
from the root. Transport point f (frontier c_f). Every condition continues for the same 12 steps from the same frontier,
on the same graph, and is scored on the same residual queries: the chain nodes at distances f+1..f+12 and their
look-alikes (24 per graph).
  CT   (canonical transport, the reference): a fresh episode from the written node START = c_{f-1}, 13 steps. Its step 1
       reads c_{f-1} -> c_f, so its own thought h_1 holds frontier c_f at age 1.
  RTa  (raw latent transplant, chain route of age a): the same fresh episode, but the thought fed into step 2 is h_a of
       an episode from START = c_{f-a} (a steps along the chain: frontier c_f, age a). a in {2, 3, ~f/2, f-1}. Chain
       routes start on the chain, so their frontier is the chain alone: the residual problem matches exactly.
  RTroot (secondary): the donor is h_f of the root episode (age f). Its route also visited the root's other branches.
  AR   (age-only reset, the control): the oldest chain donor with its age moved to 1 along the per-step mean,
       h - mu_{f-1} + mu_1 (mu_k = mean root-episode thought at step k on a separate probe, seed 8). AR_root likewise.
  RND  (content control): the oldest chain donor taken from another graph (graph i+1): same age, wrong content.
  T0   (descriptive): the root episode itself, continued to f + 12 steps.
IIA_a = agreement of RTa's answers with CT's over the residual queries (the high-level model "the future depends only on
the residual problem" predicts IIA = 1). Interchange gap G_a = (1 - IIA_a) / (1 - IIA_RND): an age swap's effect as a share
of a content swap's. Age effect dG = G_{f-1} - G_2. A model is informative only if 1 - IIA_RND >= 0.05 (otherwise the
transplanted thought is ignored). AR's share of the gap closed: (IIA_AR - IIA_{f-1}) / (1 - IIA_{f-1}).
    python theory/exp28_interchange.py --out theory/results/exp28_interchange.json [--runs ...] [--f 5 11]
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
from deep_chain import deep_chain_graphs  # noqa: E402
from eval_arch import load_arch  # noqa: E402
from rollout import load, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)
H = [102, 109, 111, 112, 113, 115, 118, 119, 125, 126, 130, 131, 132, 135, 140, 145, 146, 147, 157, 158]
FAMILIES = {"p30": "runs/p30_s{}", "pl_gt": "theory/runs/pl_gt_s{}", "pl_t8": "theory/runs/pl_t8_s{}",
            "pclr1": "theory/runs/pclr1_s{}"}


def ages(f):
    """Chain-route donor ages for transport point f: 2, 3, about f/2, and f-1 (START = c_1)."""
    return sorted({2, 3, (f + 1) // 2, f - 1})


def bal(z, n):
    """z: (B, 2n) bool answers, reach then decoy -> bal-acc per residual distance."""
    return [float(0.5 * (z[:, j].mean() + 1 - z[:, n + j].mean())) for j in range(n)]


@torch.no_grad()
def run(m, K, g, s, mu, f, R=12):
    src, dst = T(g.src), T(g.dst)
    Q = T(np.concatenate([s["reach"][:, f:f + R], s["decoy"][:, f:f + R]], 1)).long()
    root = rollout(m, K, src, dst, T(g.start), f + R, qnodes=Q)
    st = T(s["reach"][:, f - 2]).long()                                       # recipient START = c_{f-1}

    def rec(donor):
        feed = None if donor is None else (lambda k, h, th: donor if k == 2 else h)
        return rollout(m, K, src, dst, st, R + 1, feed=feed, qnodes=Q)["qlogit"].numpy() > 0

    donor = {a_: rollout(m, K, src, dst, T(s["reach"][:, f - a_ - 1]).long(), a_)["thoughts"][:, a_] for a_ in ages(f)}
    old, hroot = donor[f - 1], root["thoughts"][:, f]
    z = {"CT": rec(None), "RTroot": rec(hroot), "AR": rec(old - mu[f - 1] + mu[1]), "AR_root": rec(hroot - mu[f] + mu[1]),
         "RND": rec(torch.roll(old, 1, 0)), "T0": root["qlogit"].numpy() > 0}
    for a_ in ages(f):
        z[f"RT{a_}"] = rec(donor[a_])
    out = {k: bal(v, R) for k, v in z.items()}
    agree = lambda a, b: float((z[a] == z[b]).mean())
    out["IIA_RND"] = agree("RND", "CT")
    gap = lambda iia: (1 - iia) / max(1 - out["IIA_RND"], 1e-9)
    out["IIA_by_age"] = {a_: agree(f"RT{a_}", "CT") for a_ in ages(f)}
    out["G_by_age"] = {a_: gap(v) for a_, v in out["IIA_by_age"].items()}
    out["dG"] = out["G_by_age"][f - 1] - out["G_by_age"][2]
    out["IIA_root"], out["G_root"] = agree("RTroot", "CT"), gap(agree("RTroot", "CT"))
    out["IIA_AR"], out["IIA_AR_root"], out["IIA_T0"] = agree("AR", "CT"), agree("AR_root", "CT"), agree("T0", "CT")
    old_iia = out["IIA_by_age"][f - 1]
    out["AR_closed"] = (out["IIA_AR"] - old_iia) / max(1 - old_iia, 1e-9)
    out["AR_root_closed"] = (out["IIA_AR_root"] - out["IIA_root"]) / max(1 - out["IIA_root"], 1e-9)
    out["informative"] = (1 - out["IIA_RND"]) >= 0.05
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp28_interchange.json")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--f", type=int, nargs="*", default=[5, 11])
    a = ap.parse_args()
    names = a.runs or [p.format(s) for p in FAMILIES.values() for s in H]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, s = deep_chain_graphs(np.random.default_rng(7), 512, [2, 3], 4)
    g8, _ = deep_chain_graphs(np.random.default_rng(8), 512, [2, 3], 4)
    for path in names:
        name = os.path.basename(path.rstrip("/\\"))
        if name in res:
            continue
        m, c = (load_arch if path.startswith("theory/") else load)(path, "cpu"); m.eval()
        with torch.no_grad():
            th = rollout(m, c["K"], T(g8.src), T(g8.dst), T(g8.start), max(a.f) + 1)["thoughts"]
        mu = th.mean(0)                                                       # mu[k] = mean root-episode thought at step k
        res[name] = {"K": c["K"]}
        for f in a.f:
            o = run(m, c["K"], g, s, mu, f)
            res[name][f"f{f}"] = o
            print("%-14s f %2d  G by age %s | dG %+.2f G_root %.2f | AR closed %+.2f (root %+.2f) | IIA_RND %.3f | "
                  "resid bal CT %.3f RT%d %.3f root %.3f T0 %.3f%s" % (
                      name, f, " ".join("%d:%.2f" % kv for kv in o["G_by_age"].items()), o["dG"], o["G_root"],
                      o["AR_closed"], o["AR_root_closed"], o["IIA_RND"], np.mean(o["CT"]), f - 1, np.mean(o[f"RT{f - 1}"]),
                      np.mean(o["RTroot"]), np.mean(o["T0"]), "" if o["informative"] else "  (uninformative)"), flush=True)
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
