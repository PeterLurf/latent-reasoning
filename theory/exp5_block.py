"""Exploratory exp5 (causal): does blocking retrieval from earlier latent positions rescue extrapolation?

Conditions (rollout to K' = 12, queries read the full cache):
  base      : normal
  bL2@K     : from step K on, layer-2 attention of the latent position cannot see earlier latent positions
              (answers at K' = K are unchanged: layer-2 keys/values of position K come from its layer-1 output)
  bL2@1     : the same from step 1 (changes the trained dynamics too)
  bL12@K    : both layers from step K on
far = 0.5 * (TPR_B + TNR_C) at distances 7..12; near at 1..6 (A vs D) at K' = 12.
    python theory/exp5_block.py --out theory/results/exp5_block.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_window import members, summarise  # noqa: E402
from rollout import front_profile, load, probe, query_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp5_block.json")
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
        for tag, kw in (("base", {}), ("bL2@K", {"block_from": K, "block_layers": (1,)}),
                        ("bL2@1", {"block_from": 1, "block_layers": (1,)}),
                        ("bL12@K", {"block_from": K, "block_layers": (0, 1)})):
            q, out = query_eval(m, K, g, br, 12, "cpu", **kw)
            front, amp = summarise(front_profile(out["attn1"], g, br, E))
            o[tag] = {"front": front, "amp": amp,
                      "near": [float(0.5 * (q["A"][j] + 1 - q["D"][j])) for j in range(6)],
                      "far": [float(0.5 * (q["B"][j] + 1 - q["C"][j])) for j in range(6)]}
        res[name] = o
        json.dump(res, open(a.out, "w"), indent=1)
        f = lambda t: " ".join("%2d" % x for x in o[t]["front"][K - 1:])
        s = lambda t: " ".join("%.2f" % x for x in o[t]["far"][:4])
        print("%-9s %s P7 %.2f | base %s [%s] | bL2@K %s [%s] | bL12@K [%s] | bL2@1 near6 %.2f far [%s]" % (
            name, "EXT" if o["extrap"] else "   ", o["P7"], f("base"), s("base"), f("bL2@K"), s("bL2@K"), s("bL12@K"),
            o["bL2@1"]["near"][5], s("bL2@1")), flush=True)


if __name__ == "__main__":
    main()
