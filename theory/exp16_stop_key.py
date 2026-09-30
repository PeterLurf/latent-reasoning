"""THEORY §35 P10e: is the anytime runs' fall-back keyed to the hop count or to the step count? (CPU, no training)

Anytime runs advance one hop per step, so hop count = step index and the two keys cannot be told apart by watching them.
Two interventions de-align them on the long-chain probe (512, seed 4247), per graph, at thought h_2:
  * skip  : h_2 + 2 (Enc(reach[p+1]) - Enc(reach[p-1]))   -- the pointer jumps two hops ahead (lin encoder of exp15, x2)
  * delay : the thought fed into step 4 is h_2 again instead of h_3 (the pointer re-reads one edge: one hop behind)
Pointer = ptr_traj's rule with the model's fixed sweep head: at step k, the mean over the 512 graphs of the head's
layer-1 attention logit on the 11 chain edges (into A1..A5, B0..B5) minus the look-alike decoy edges; p_k = argmax + 1
(the depth of the edge's source) if the maximum is >= 1, else 0. (Per-graph argmaxes were tried on the smoke model and
are noise-dominated.) The skip edit uses the base pointer at h_2 (read at step 3), p, for every graph: reach[p-1] ->
reach[p+1]. On each trajectory:
  * peak depth : the running maximum of p before the first step with p < running maximum - 1 (the fall-back);
                 peak step: the step at which it is reached
Hop key predicts: skip and delay leave the median peak depth unchanged (|change| <= 1) and move the peak step (skip
earlier, delay later). Step key predicts: the peak step unchanged and the peak depth moved (skip +2, delay -1).
    python theory/exp16_stop_key.py --out theory/results/exp16_stop_key.json
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
from exp15_c7_edit import fit_encoders, seqs  # noqa: E402
from rollout import probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


def peaks(P):
    """P: (n, S) pointer per step (step 1..S) -> (peak depth, peak step) per graph."""
    out = []
    for row in P:
        best, bstep = 0, 0
        for s_, v in enumerate(row, 1):
            if best > 0 and v < best - 1:
                break
            if v > best:
                best, bstep = v, s_
        out.append((best, bstep))
    return np.array(out)


def agg_ptr(o, E, head):
    """Aggregate pointer per step (ptr_traj's rule with a fixed head)."""
    n = E["A"].shape[0]; bi = np.arange(n)[:, None]
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    out = []
    for L in o["logit1"]:
        s_ = np.nan_to_num(L[:, head].numpy())
        d = (s_[bi, reach] - s_[bi, decoy]).mean(0)
        out.append(int(d.argmax()) + 1 if d.max() >= 1 else 0)
    return out


@torch.no_grad()
def run(m, K, head, g, br, E, Kp=12):
    reach, _ = seqs(g, br)
    src, dst, st = T(g.src), T(g.dst), T(g.start)
    base = rollout(m, K, src, dst, st, Kp)
    Pb = agg_ptr(base, E, head)
    p = Pb[2]                                           # step 3 reads h_2
    res = {"p_h2": p, "base": {"ptr": Pb}}
    if not 1 <= p <= 8:
        res["eligible"] = False
        return res
    res["eligible"] = True
    enc = fit_encoders(m, K, head)
    d = 2 * (enc["lin"](reach[:, p + 1]) - enc["lin"](reach[:, p - 1]))
    skip = rollout(m, K, src, dst, st, Kp, feed=lambda k, h, hist: h + d if k == 3 else h)
    delay = rollout(m, K, src, dst, st, Kp, feed=lambda k, h, hist: hist[2] if k == 4 else h)
    res["skip"] = {"ptr": agg_ptr(skip, E, head)}
    res["delay"] = {"ptr": agg_ptr(delay, E, head)}
    for k in ("base", "skip", "delay"):
        (dep, stp), = peaks(np.array([res[k]["ptr"]]))
        res[k].update(peak_depth=int(dep), peak_step=int(stp))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp16_stop_key.json")
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    M = json.load(open("theory/results/pilotDE_meas.json"))
    names = a.runs or sorted(r for r in M if (r.startswith("pe8_at_") or r.startswith("pd_at_")) and "ptr" in M[r])
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(512)
    for r in names:
        if r in res:
            continue
        m, c = load_arch(f"theory/runs/{r}", "cpu")
        if r in M:
            head, p = M[r]["head"], M[r]["ptr"]
        else:
            from analyse_pilotA import ptr_traj
            gg, bb, EE = probe(512)
            p, _, head = ptr_traj(m, c["K"], gg, bb, EE, {})
        o = run(m, c["K"], head, g, br, E)
        o.update(K=c["K"], ptr=p)
        res[r] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        if not o["eligible"]:
            print("%-14s K %d not eligible (p at h_2 = %d)" % (r, c["K"], o["p_h2"]), flush=True)
            continue
        f = lambda k: "%d@%d %s" % (o[k]["peak_depth"], o[k]["peak_step"], "".join(map(str, o[k]["ptr"])))
        print("%-14s K %d | peak depth@step, ptr: base %s | skip %s | delay %s" % (r, c["K"], f("base"), f("skip"), f("delay")), flush=True)


if __name__ == "__main__":
    main()
