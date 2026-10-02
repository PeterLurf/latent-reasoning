"""Score THEORY §73 (8da9ea5) from c1/runs/p73_<arm>_s<seed>/eval.json; seeds 2011-2020 (pfctr 2011-2015).
Committed before any §73 eval.json was read.
L1: PF-std certified (product-automaton certificate) in >= 9/10 seeds.
L2: on the PF-std seeds, every T0/T1/T2 null's Wilson 95 % upper bound <= 0.1867 at deep depths (N >= 16; n_null =
    10,000), AND every RESET seed's accuracy upper bound <= 0.1867 at deep depths (n_test = 10,000).
L3: in every certified PF-std seed: donor accuracy >= 0.99; irrelevant change (visited-code substitutes) <= 0.01 or
    vacuous (reported apart); same-semantic agreement >= 0.99.
Descriptive: counter check; PT-std and PF-ctr certified counts; C1-5 vs the existing fpcmp2 runs (median PF-std minus
median fpcmp2 accuracy at N >= 16); sigma distributions; dead-unit check; distinctness.
    python c1/score_73.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from c1.core import wilson_upper  # noqa: E402

THR = 0.1867; DEEP = ("16", "32", "64", "128"); DEPTHS = ("8",) + DEEP
SEEDS = {"pfstd": range(2011, 2021), "ptstd": range(2011, 2021), "reset": range(2011, 2021), "pfctr": range(2011, 2016)}


def load(arm):
    out = {}
    for s in SEEDS[arm]:
        f = f"c1/runs/p73_{arm}_s{s}/eval.json"
        if os.path.exists(f):
            out[s] = json.load(open(f))
    return out


def main():
    R = {a: load(a) for a in SEEDS}; n = {a: len(v) for a, v in R.items()}; v = {}
    cert = lambda e: bool(e.get("certificate", {}).get("certified"))
    pf = R["pfstd"]; k = sum(cert(e) for e in pf.values())
    v["L1"] = {"certified": k, "of": len(pf), "met": k >= 9, "per_seed": {s: cert(e) for s, e in pf.items()}}
    worst = 0.0
    for e in pf.values():
        for N in DEEP:
            for key in ("T0", "T1", "T2"):
                worst = max(worst, e["nulls"][N][key]["upper95"])
    rub = {s: {N: wilson_upper(round(e["acc"][N] * e["n_test"]), e["n_test"]) for N in DEEP} for s, e in R["reset"].items()}
    rok = bool(rub) and all(u <= THR for d in rub.values() for u in d.values())
    v["L2"] = {"nulls_worst_upper95": worst, "nulls_ok": worst <= THR, "reset_upper95": rub, "reset_ok": rok,
               "reset_n_test": sorted({e["n_test"] for e in R["reset"].values()}), "met": worst <= THR and rok}
    mech = {s: e.get("mechanism", {}) for s, e in pf.items() if cert(e)}
    irr = {s: ("vacuous" if m.get("irrelevant_change_rate_reachable") is None else
               ("met" if m["irrelevant_change_rate_reachable"] <= 0.01 else "not met")) for s, m in mech.items()}
    ok3 = bool(mech) and all(m.get("donor_acc", 0) >= 0.99 and m.get("same_semantic_agreement", 0) >= 0.99 and irr[s] != "not met"
                             for s, m in mech.items())
    v["L3"] = {"met": ok3, "irrelevant_part": irr, "per_seed": {s: {k2: m.get(k2) for k2 in ("donor_acc", "irrelevant_change_rate_reachable",
               "irrelevant_n_reachable", "same_semantic_agreement")} for s, m in mech.items()}}
    for key, arms in (("L1", ["pfstd"]), ("L2", ["pfstd", "reset"]), ("L3", ["pfstd"])):
        v[key]["n_runs"] = {a: n[a] for a in arms}
        if any(n[a] < len(SEEDS[a]) for a in arms):
            v[key]["status"] = "incomplete"
    fp = []
    for s in range(1, 6):
        f = f"c1/runs/p67b_fpcmp2_s{s}/eval.json"
        if os.path.exists(f):
            fp.append(json.load(open(f))["acc"])
    d = {"counter_check": {s: e.get("counter_check") for s, e in pf.items() if cert(e)},
         "ptstd_certified": f"{sum(cert(e) for e in R['ptstd'].values())}/{n['ptstd']}",
         "pfctr_certified": f"{sum(cert(e) for e in R['pfctr'].values())}/{n['pfctr']}",
         "C1_5_vs_fpcmp2": {N: float(np.median([e["acc"][N] for e in pf.values()]) - np.median([a[N] for a in fp])) for N in DEEP} if pf and fp else None,
         "sigma": {s: e.get("table_stats") for s, e in pf.items()}, "relu_units": {s: e.get("relu_units") for s, e in pf.items()}}
    res = {"n_runs": n, "verdicts": v, "descriptive": d,
           "acc": {a: {s: e["acc"] for s, e in R[a].items()} for a in R}}
    json.dump(res, open("theory/results/score_s73.json", "w", newline="\n"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("runs", n)
    for a in R:
        for s, e in sorted(R[a].items()):
            print(f"{a:6s} s{s} cert {cert(e)} acc " + " ".join(f"{e['acc'][N]:.3f}" for N in DEPTHS))
    for k2 in ("L1", "L2", "L3"):
        print(k2, {x: y for x, y in v[k2].items() if x not in ("reset_upper95", "per_seed")})
    print("descriptive", {x: y for x, y in d.items() if x in ("ptstd_certified", "pfctr_certified", "C1_5_vs_fpcmp2")})


if __name__ == "__main__":
    main()
