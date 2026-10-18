"""LANG-VB draft, $0 CPU prototype: a balanced ProsQA-deep regeneration "NN" (double neutralisation) and the LEVEL-PANEL.
NN item = neutralise(base) [§59 at the base: the original target / negative = the future chain anchors]
          -> make_long.extend(m) [p8 chains of m fresh names off both anchors]
          -> neutralise(extended) [§59 at the options c_m / d_m].
LEVEL-PANEL: the six §59 heuristics (proximity, parents, mentions, first, last, parents' in-degree; direction-free
max(a, 1 - a)) at EVERY chain level k = 0..m (c_k vs d_k; k = 0 = the anchors), plus the option level; reported per
depth band. The binding gate (Audit, 10-18): a band with any heuristic at any level >= 0.6 is not used for readings.
Reads repo files read-only (imports theory/prosqa_na.py, prosqa/make_long.py with -B); writes JSON to argv[1].
    python -B nn_regen.py <out.json> [--split test|train] [--m 0,1,...] [--seed 0] [--n_base N]
"""
import argparse
import collections
import json
import sys

import numpy as np

sys.path.insert(0, "D:/Latent/theory"); sys.path.insert(0, "D:/Latent/prosqa")
import prosqa_na as NA  # noqa: E402
import make_long as ML  # noqa: E402
sys.path.insert(0, "C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad/research7/scripts")
from heur_panel import panel_for  # noqa: E402


def chain_levels(x, m, L0):
    """chain node indices: c_1..c_m = L0 .. L0+m-1, d_1..d_m = L0+m .. L0+2m-1 (make_long appends c then d)."""
    c = list(range(L0, L0 + m)); d = list(range(L0 + m, L0 + 2 * m)); return c, d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--split", default="test")
    ap.add_argument("--m", default="0,2,4,6,8,10,12"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_base", type=int, default=0)
    ap.add_argument("--mode", default="nn", choices=["nn", "p8na"], help="p8na = extend then neutralise once (the §59 / p8_na order)")
    a = ap.parse_args()
    tr = json.load(open("D:/Latent/prosqa/data/prosqa_train.json"))
    ML.NAMES = sorted({s for x in tr for s in x["idx_to_symbol"] if s[:1].islower()})
    base = [x for x in json.load(open("D:/Latent/prosqa/data/prosqa_%s.json" % a.split)) if len(x["steps"]) <= 4]
    if a.n_base:
        base = base[:a.n_base]
    rng = np.random.default_rng(a.seed); ms = [int(v) for v in a.m.split(",")]
    rows = collections.defaultdict(lambda: collections.defaultdict(list)); cnt = collections.Counter(); drop = collections.Counter()
    for x in base:
        for m in ms:
            if a.mode == "nn":
                xb, _ = NA.neutralise(x, rng)
                if xb is None:
                    drop[(m, "base")] += 1; continue
            else:
                xb = x
            L0 = len(xb["idx_to_symbol"])
            y = ML.extend(xb, m, rng)
            if y is None:
                drop[(m, "names")] += 1; continue
            bt, bn = xb["target"], xb["neg_target"]
            z, _ = NA.neutralise(y, rng)
            if z is None:
                drop[(m, "opt")] += 1; continue
            depth = len(z["steps"]); cnt[depth] += 1
            c, d = chain_levels(z, m, L0)
            levels = [(bt, bn)] + list(zip(c, d))
            worst = {}
            for k, (u, v) in enumerate(levels):
                r = panel_for(z, u, v)
                for h, val in r.items():
                    rows[depth]["L%d_%s" % (k, h)].append(val)
            r = panel_for(z, z["target"], z["neg_target"])
            for h, val in r.items():
                rows[depth]["opt_" + h].append(val)
            rows[depth]["T_words"].append(len(z["question"].split()))
    out = {"mode": a.mode, "split": a.split, "seed": a.seed, "ms": ms, "n_by_depth": dict(sorted(cnt.items())), "dropped": {"%s_%s" % k: v for k, v in drop.items()}, "bands": {}}
    for depth in sorted(rows):
        R = rows[depth]; band = {"n": cnt[depth], "T_words_mean": float(np.mean(R["T_words"]))}
        worst_k, worst_v = None, 0.0
        for k_, v in R.items():
            if k_ == "T_words":
                continue
            a_ = float(np.mean(v)); df = max(a_, 1 - a_); band[k_] = round(df, 4)
            if df > worst_v:
                worst_v, worst_k = df, k_
        band["max_heuristic"] = round(worst_v, 4); band["max_which"] = worst_k; band["gate_pass"] = worst_v < 0.6
        out["bands"][depth] = band
        print(depth, "n", cnt[depth], "max", round(worst_v, 3), worst_k, "pass" if worst_v < 0.6 else "FAIL", "words", round(band["T_words_mean"]))
    json.dump(out, open(a.out, "w", newline="\n"), indent=1)
    print("dropped", dict(drop))


if __name__ == "__main__":
    main()
