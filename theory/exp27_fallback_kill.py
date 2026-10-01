"""THEORY §55 (external direction 3's kill test): is the anytime runs' fall-back mutation reward-neutral in training?
CPU, inference only, no training.

The fall-back (§35, P10e) is a state change keyed to an iteration count: the pointer advances one hop per step up to its
peak step t* and then drops. The kill replaces every update after the peak by the identity: from step t*+2 on, the
thought fed into each step is h_{t*} again (the state stays where it was at the peak), instead of h_{k-1}. A positive
control removes one real advance instead: from step t*+1 on, the input is h_{t*-1}.
  t*   : the model's base peak step on the long-chain probe, from theory/results/exp16_stop_key.json (P10e; aggregate
         pointer, fixed sweep head, 12-step rollout)
  data : in-distribution graphs (sample_graphs split "discovery", the training graph family; 2048 graphs, seed 2027),
         make_batch's 16 balanced reachability queries per graph, answers read at the trained K (the training readout)
Reported per model: terminal BCE and bal-acc for base / kill / pos (positive control), and the kill applies only if
t*+2 <= K (otherwise the fall-back never runs inside the trained budget and is "outside training": reported as such).
Descriptive, long-chain probe (512, seed 4247) at K' = 12: bal-acc by distance 1..12 for base / kill.
    python theory/exp27_fallback_kill.py --out theory/results/exp27_fallback_kill.json [--runs ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from eval_arch import load_arch  # noqa: E402
from latent.data import make_batch  # noqa: E402
from latent.graphs import sample_graphs  # noqa: E402
from rollout import probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


def hold(src_step, from_step):
    """feed: from step `from_step` on, the input thought is thoughts[src_step] (identity update after src_step)."""
    return lambda k, h, th: th[src_step] if k >= from_step else h


def terminal(m, K, b, feed):
    o = rollout(m, K, b["src"], b["dst"], b["start"], K, feed=feed, qnodes=b["q"])
    z = o["qlogit"].float(); y = b["qy"]
    bce = float(F.binary_cross_entropy_with_logits(z, y))
    p = z > 0; yy = y > 0.5
    return bce, float(0.5 * (p[yy].float().mean() + (~p[~yy]).float().mean()))


def probe_profile(m, K, g, br, feed, Kp=12):
    n = len(g); bi = np.arange(n)
    reach = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)
    decoy = np.concatenate([g.nodes[bi, br["D"]], g.nodes[bi, br["C"]]], 1)
    Q = T(np.concatenate([reach, decoy], 1)).long()
    z = rollout(m, K, T(g.src), T(g.dst), T(g.start), Kp, feed=feed, qnodes=Q)["qlogit"].numpy() > 0
    return [float(0.5 * (z[:, j].mean() + 1 - z[:, 12 + j].mean())) for j in range(12)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp27_fallback_kill.json")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--peaks", default="theory/results/exp16_stop_key.json")
    ap.add_argument("--tstar", type=int, help="smoke only: t* for models absent from --peaks")
    a = ap.parse_args()
    P = json.load(open(a.peaks))
    names = a.runs or sorted(P)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rng = np.random.default_rng(2027)
    g = sample_graphs(rng, 2048, split="discovery")
    b = make_batch(rng, g, 16, "cpu")
    pg, pb, _ = probe(512)
    for name in names:
        if name in res:
            continue
        path = name if os.path.isdir(name) else "theory/runs/" + name
        m, c = load_arch(path, "cpu"); m.eval()
        K = c["K"]
        ts = P[os.path.basename(name)]["base"]["peak_step"] if os.path.basename(name) in P else a.tstar
        o = {"K": K, "t_star": ts}
        o["base"] = terminal(m, K, b, None)
        if ts is not None and ts + 2 <= K:
            o["inside"] = True
            o["kill"] = terminal(m, K, b, hold(ts, ts + 2))
            o["pos"] = terminal(m, K, b, hold(ts - 1, ts + 1)) if ts >= 2 else None
        else:
            o["inside"] = False
        o["probe_base"] = probe_profile(m, K, pg, pb, None)
        if ts is not None:
            o["probe_kill"] = probe_profile(m, K, pg, pb, hold(ts, ts + 2))
        res[os.path.basename(name)] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        msg = "%-14s K %d t* %s base bce %.4f bal %.3f" % (os.path.basename(name), K, ts, *o["base"])
        if o["inside"]:
            msg += " | kill bce %.4f bal %.3f" % tuple(o["kill"])
            if o["pos"]:
                msg += " | pos bce %.4f bal %.3f" % tuple(o["pos"])
        else:
            msg += " | fall-back outside the trained budget"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
