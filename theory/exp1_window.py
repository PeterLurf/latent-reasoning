"""Exploratory exp1: is the post-K failure caused by the growing latent context (implicit step count)?

For each competent P30 / L model: front trajectory (layer-1 excess attention, graphs averaged) and beyond-horizon
query accuracy at K' = 12 under
  base   : normal rollout
  winK   : from step K+1 on, the latent step sees only the last K-1 latent positions (count held at step K's)
  win2   : from step K+1 on, only the last 2 latent positions
Queries always read the full cache. far = 0.5 * (TPR_B + TNR_C) at distances 7..12.
    python theory/exp1_window.py --out theory/results/exp1_window.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout import front_profile, load, probe, query_eval  # noqa: E402


def members():
    rows = {}
    for f in ("results/p30.json", "results/p_l.json"):
        rows.update(json.load(open(f))["rows"])
    return rows


def summarise(prof):
    m = prof.mean(1)                                   # (Kp, 12)
    front = [int(np.argmax(r)) + 1 if r.max() >= 0.01 else 0 for r in m]
    amp = [float(r.max()) for r in m]
    return front, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp1_window.json")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    rows = members()
    names = a.runs or sorted(rows, key=lambda r: (rows[r]["extrap"], rows[r]["P7"]))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(a.n)
    for name in names:
        if name in res:
            continue
        m, c = load("runs/" + name, "cpu")
        K = c["K"]
        o = {"K": K, **{k: rows[name][k] for k in ("P7", "extrap", "d7", "EA")}}
        for tag, kw in (("base", {}), ("winK", {"window": K - 1, "window_from": K + 1}),
                        ("win2", {"window": 2, "window_from": K + 1})):
            q, out = query_eval(m, K, g, br, 12, "cpu", **kw)
            front, amp = summarise(front_profile(out["attn1"], g, br, E))
            o[tag] = {"front": front, "amp": amp,
                      "near": [float(0.5 * (q["A"][j] + 1 - q["D"][j])) for j in range(6)],
                      "far": [float(0.5 * (q["B"][j] + 1 - q["C"][j])) for j in range(6)],
                      "tprB": [float(x) for x in q["B"]], "fprC": [float(x) for x in q["C"]]}
        res[name] = o
        json.dump(res, open(a.out, "w"), indent=1)
        f = lambda t: " ".join("%2d" % x for x in o[t]["front"])
        print("%-9s %s P7 %.2f | base %s far7-9 %s | winK %s far7-9 %s" % (
            name, "EXT" if o["extrap"] else "   ", o["P7"], f("base"),
            " ".join("%.2f" % x for x in o["base"]["far"][:3]), f("winK"),
            " ".join("%.2f" % x for x in o["winK"]["far"][:3])), flush=True)


if __name__ == "__main__":
    main()
