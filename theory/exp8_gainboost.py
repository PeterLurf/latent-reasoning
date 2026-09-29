"""Exploratory exp8: training-free extrapolation repair at inference — retrieval block + gain boost past K.

For steps k >= K+1 (inputs h_K, h_{K+1}, ...): layer-2 attention cannot see earlier latent positions (as exp5 bL2@K)
and the input thought is amplified around the per-step batch mean: h <- c_k + alpha * (h - c_k).
Answers at the trained budget are unchanged by construction. Measures far (distance 7..12, 0.5*(TPR_B + TNR_C)) and
false-positive rate on the unreachable C chain at K' = 12, for alpha in {1.0, 1.3, 1.6, 2.0}.
    python theory/exp8_gainboost.py --out theory/results/exp8_gainboost.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_window import members  # noqa: E402
from rollout import load, probe, query_eval  # noqa: E402


def booster(K, alpha):
    def feed(k, h, hist):
        if k >= K + 1 and alpha != 1.0:
            c = h.mean(0, keepdim=True)
            return c + alpha * (h - c)
        return h
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp8_gainboost.json")
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    rows = members()
    names = sorted(rows, key=lambda r: (rows[r]["extrap"], rows[r]["P7"]))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(a.n)
    for name in names:
        if name in res:
            continue
        m, c = load("runs/" + name, "cpu")
        K = c["K"]
        o = {"K": K, "extrap": rows[name]["extrap"]}
        for alpha in (1.0, 1.3, 1.6, 2.0):
            q, _ = query_eval(m, K, g, br, 12, "cpu", block_from=K, block_layers=(1,), feed=booster(K, alpha))
            o[str(alpha)] = {"far": [float(0.5 * (q["B"][j] + 1 - q["C"][j])) for j in range(6)],
                             "fprC": [float(x) for x in q["C"]], "tprB": [float(x) for x in q["B"]],
                             "near": [float(0.5 * (q["A"][j] + 1 - q["D"][j])) for j in range(6)]}
        res[name] = o
        json.dump(res, open(a.out, "w"))
        print("%-9s %s " % (name, "EXT" if o["extrap"] else "   ") + " | ".join(
            "a%.1f far8-10 %.2f fpC %.2f" % (al, np.mean(o[str(al)]["far"][1:4]), np.mean(o[str(al)]["fprC"][1:4])) for al in (1.0, 1.3, 1.6, 2.0)), flush=True)


if __name__ == "__main__":
    main()
