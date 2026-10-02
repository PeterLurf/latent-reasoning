"""Score THEORY §72 (5918e15) from c1/runs/p72_<arm>_s<seed>/eval.json, arms pf / rf / pfstd / pt, seeds 2001-2005.
Committed before any §72 eval.json was read (the runs had been drained but not opened). TOY is Theory's CPU arm.
Rule (fixed in §72): an arm succeeds at >= 4/5 certified (product-automaton certificate), fails at <= 1/5, and 2-3/5 is
inconclusive. Reported: certified count, hard accuracy at N 8-128, code distinctness (mean pairwise cosine of head
inputs across codes, normed and raw) at init and end, and for PT the per-layer between/within-code variance ratio,
averaged over certified vs failed seeds.
    python c1/score_72.py
"""
import json
import os

import numpy as np

ARMS = ("pf", "rf", "pfstd", "pt"); SEEDS = range(2001, 2006)


def main():
    import sys
    pre = sys.argv[1] if len(sys.argv) > 1 else "p72"                 # §72b: python c1/score_72.py p72b (same rules)
    res = {}
    for a in ARMS:
        runs = {}
        for s in SEEDS:
            f = f"c1/runs/{pre}_{a}_s{s}/eval.json"
            if os.path.exists(f):
                runs[s] = json.load(open(f))
        k = sum(bool(e.get("certificate", {}).get("certified")) for e in runs.values())
        verdict = "succeeds" if k >= 4 else ("fails" if k <= 1 else "inconclusive")
        r = {"n": len(runs), "certified": k, "verdict": verdict if len(runs) == 5 else f"incomplete ({verdict} so far)",
             "per_seed": {s: {"certified": e.get("certificate", {}).get("certified"), "acc": e["acc"],
                              "distinct_init": e.get("distinctness", {}).get("init"), "distinct_end": e.get("distinctness", {}).get("end")}
                          for s, e in runs.items()}}
        if a == "pt":
            split = {}
            for grp, sel in (("certified", True), ("failed", False)):
                L = [e["layer_code_variance"] for e in runs.values()
                     if bool(e.get("certificate", {}).get("certified")) == sel and "layer_code_variance" in e]
                if L:
                    split[grp] = [float(np.mean([l[i]["ratio"] for l in L])) for i in range(len(L[0]))]
            r["layer_ratio_by_outcome"] = split
        res[a] = r
    json.dump(res, open(f"theory/results/score_s{pre[1:]}.json", "w", newline="\n"), indent=1)   # p72 -> s72, p72b -> s72b
    for a, r in res.items():
        print(f"{a:6s} certified {r['certified']}/{r['n']} -> {r['verdict']}")
        for s, x in r["per_seed"].items():
            di, de = x["distinct_init"] or {}, x["distinct_end"] or {}
            print(f"   s{s} cert {x['certified']} acc " + " ".join(f"{x['acc'][N]:.3f}" for N in ("8", "16", "32", "64", "128"))
                  + f" | cos normed init {di.get('mean_pairwise_cos_normed', float('nan')):.3f} end {de.get('mean_pairwise_cos_normed', float('nan')):.3f}"
                  + f" raw init {di.get('mean_pairwise_cos_raw', float('nan')):.3f} end {de.get('mean_pairwise_cos_raw', float('nan')):.3f}")
        if "layer_ratio_by_outcome" in r:
            for g, v in r["layer_ratio_by_outcome"].items():
                print(f"   PT {g}: between/within ratio by layer " + " ".join(f"{x:.2f}" for x in v))


if __name__ == "__main__":
    main()
