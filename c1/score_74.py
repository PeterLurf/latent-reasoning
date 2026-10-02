"""Score THEORY §74 from c1/runs/p74_<arm>_s<seed>/eval.json (arms pfstd, rfstd, dense, pool; seeds 2021-2025).
Committed before any §74 eval.json was read. Medians over seeds.
P1 (prerequisite): PF-std certified on the training phrasing in >= 4/5 seeds.
P2: for at least one of F2, F3, F4, PF-std's median paraphrase agreement (frozen sigma) >= 0.90; F3 counts only if PF-std's
    median negative-control (" copy i j") agreement <= 0.50. RF-std is reported alongside as the floor.
P3: for a family that meets P2, PF-std's family certificate holds in >= 3/5 seeds.
Descriptive: F1 (reversed), pos/neg controls, sigma-floor agreement, family hard accuracy, feature diagnostic, " one" vs
" 1", the other arms. A verdict with fewer than 5 PF-std seeds is marked "incomplete".
    python c1/score_74.py
"""
import json
import os

import numpy as np

ARMS = ("pfstd", "rfstd", "dense", "pool"); SEEDS = range(2021, 2026); FAMS = ("F1", "F2", "F3", "F4", "pos", "neg")


def main():
    R = {a: {s: json.load(open(f"c1/runs/p74_{a}_s{s}/eval.json")) for s in SEEDS if os.path.exists(f"c1/runs/p74_{a}_s{s}/eval.json")}
         for a in ARMS}
    med = lambda a, f, col="frozen_sigma": float(np.median([e["families"][f][col]["agreement"] for e in R[a].values()])) if R[a] else None
    pf = R["pfstd"]; v = {}
    k1 = sum(e["train"]["certificate"]["certified"] for e in pf.values())
    v["P1"] = {"certified": k1, "of": len(pf), "met": k1 >= 4}
    neg = med("pfstd", "neg")
    cand = {f: med("pfstd", f) for f in ("F2", "F3", "F4")}
    ok = [f for f, x in cand.items() if x is not None and x >= 0.90 and (f != "F3" or (neg is not None and neg <= 0.50))]
    v["P2"] = {"median_agreement": cand, "neg_control": neg, "families_meeting": ok, "met": bool(ok)}
    v["P3"] = {f: sum(e["families"][f]["frozen_sigma"]["certificate"]["certified"] for e in pf.values()) for f in ok}
    v["P3"]["met"] = any(x >= 3 for f, x in v["P3"].items() if f != "met") if ok else False
    if len(pf) < 5:
        for k in ("P1", "P2", "P3"):
            v[k]["status"] = "incomplete"
    table = {a: {f: {"frozen": med(a, f), "floor": med(a, f, "sigma_floor")} for f in FAMS} for a in ARMS}
    certs = {a: sum(e["train"]["certificate"]["certified"] for e in R[a].values()) for a in ARMS}
    res = {"n_runs": {a: len(R[a]) for a in ARMS}, "verdicts": v, "median_agreement": table, "train_certified": certs,
           "one_vs_1": {s: e.get("one_vs_1") for s, e in pf.items()},
           "feature_diag_end_pfstd": {s: e["feature_diag"]["end"] for s, e in pf.items()}}
    json.dump(res, open("theory/results/score_s74.json", "w", newline="\n"), indent=1)
    print("runs", res["n_runs"], "train certified", certs)
    for a in ARMS:
        print(f"{a:6s}", {f: (round(x["frozen"], 3) if x["frozen"] is not None else None) for f, x in table[a].items()})
    for k in ("P1", "P2", "P3"):
        print(k, v[k])


if __name__ == "__main__":
    main()
