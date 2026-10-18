"""PROOFS_LANG item 1b: the double-neutralised p8 ("NN") level panel, with MY panel (anchor_check.heur, text-parsed).
Construction = the repo's own: theory/prosqa_na.neutralise (the §59 construction) and prosqa/make_long.extend, imported
read-only. NN = neutralise(base) -> extend(m) -> neutralise(extended). Base = prosqa_<split> items with <= 4 steps.
Levels: k = 0 (anchors = the base target / negative), k = 1..m (c_k vs d_k), and the options.
    python -B nn_check.py <out.json> --seed S [--split test] [--m 0,2,4,6,8,10,12]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, "D:/Latent/theory"); sys.path.insert(0, "D:/Latent/prosqa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prosqa_na as NA  # noqa: E402
import make_long as ML  # noqa: E402
from anchor_check import parse, heur  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test"); ap.add_argument("--m", default="0,2,4,6,8,10,12")
    ap.add_argument("--mode", default="nn", choices=["nn", "p8na"])
    a = ap.parse_args()
    tr = json.load(open("D:/Latent/prosqa/data/prosqa_train.json"))
    ML.NAMES = sorted({s for x in tr for s in x["idx_to_symbol"] if s[:1].islower()})
    base = [x for x in json.load(open("D:/Latent/prosqa/data/prosqa_%s.json" % a.split)) if len(x["steps"]) <= 4]
    rng = np.random.default_rng(a.seed); ms = [int(v) for v in a.m.split(",")]
    rows = collections.defaultdict(lambda: collections.defaultdict(list)); drop = collections.Counter(); n = collections.Counter()
    for x in base:
        for m in ms:
            if a.mode == "nn":
                xb, _ = NA.neutralise(x, rng)
                if xb is None:
                    drop["base"] += 1; continue
            else:
                xb = x
            L0 = len(xb["idx_to_symbol"])
            y = ML.extend(xb, m, rng)
            if y is None:
                drop["names"] += 1; continue
            z, _ = NA.neutralise(y, rng)
            if z is None:
                drop["opt"] += 1; continue
            edges, words, _q = parse(z)
            assert sorted(edges) == sorted(tuple(e) for e in z["edges"])
            D = len(z["steps"]); n[D] += 1
            lev = [(xb["target"], xb["neg_target"])] + [(L0 + k, L0 + m + k) for k in range(m)]
            assert m == 0 or lev[-1] == (z["target"], z["neg_target"])
            for k, (u, v) in enumerate(lev):
                for h, val in heur(z, edges, words, u, v).items():
                    if h == "last_first":
                        continue
                    rows[D]["L%d_%s" % (k, h)].append(val)
            for h, val in heur(z, edges, words, z["target"], z["neg_target"]).items():
                if h != "last_first":
                    rows[D]["opt_" + h].append(val)
    out = {"mode": a.mode, "seed": a.seed, "split": a.split, "n": dict(sorted(n.items())), "drop": dict(drop), "bands": {}}
    for D in sorted(rows):
        band = {}; worst = (0.0, None)
        for k, v in rows[D].items():
            mu = sum(v) / len(v); df = max(mu, 1 - mu); band[k] = round(df, 4)
            if df > worst[0]:
                worst = (df, k)
        band["n"] = n[D]; band["max"] = round(worst[0], 4); band["max_which"] = worst[1]
        out["bands"][D] = band
        print(D, "n", n[D], "max %.3f %s" % worst, "L0_pindeg %.3f L0_prox %.3f opt_prox %.3f" % (band["L0_pindeg"], band["L0_proximity"], band["opt_proximity"]))
    json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
