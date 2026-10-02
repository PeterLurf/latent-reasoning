"""Score THEORY §67b (registered 10dd84c; 67b-1, 67b-2) from c1/runs/p67b_<arm>_s<seed>/eval.json, seeds 1-5.
Written before any panel result was read (the seed-99 pilot was read; it is not a panel run).
Arms: d1b (p67b_d1b), tab, reset, fpex, fpcmp. Depths N 8, 16, 32, 64, 128; "deep" = N >= 16.
C1-1: D1b hard-execution accuracy >= 0.99 at every depth 8-128 in >= 4/5 seeds.
C1-2: the product-automaton certificate is error-free in every seed that passes C1-1.
C1-3: on the D1b seeds that pass C1-1, every null's Wilson 95 % upper bound <= 0.1867 at every deep depth; and RESET's
      accuracy upper bound (Wilson, n_test) <= 0.1867 at every deep depth in every RESET seed.
C1-4: on the D1b seeds that pass C1-1: donor accuracy >= 0.99, irrelevant change rate <= 0.01 among REACHABLE codes (or
      vacuous: no second reachable code with the same decoded state), same-semantic agreement >= 0.99; met if every such
      seed meets all three. (Clarified before any panel result was read: with M = 24, unreachable codes are outside the
      certified machine; the unrestricted rate is descriptive. Audit's condition: substitutes are codes visited in the donors'
      real hard rollouts after >= 1 op; on pilot seed 99 those are 6 codes, one per state, so the test is vacuous there. Panel hosts ran the old eval, so the reachable-restricted
      mechanism is recomputed locally by c1/remech.py on the fetched D1b models.)
C1-5: median over seeds, D1b minus FP >= 0.20 at every N >= 16, for FP_ex AND FP_cmp (both matchings).
Descriptive: TAB per seed (certificate, accuracy); counter check (H(code|state), MI with length mod 2/3/4) on certified
runs; audit fields; training seconds. A verdict whose arm has < 5 seeds is marked "incomplete".
    python c1/score_67b.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from c1.core import wilson_upper  # noqa: E402

ARMS = {"d1b": "p67b_d1b", "tab": "p67b_tab", "reset": "p67b_reset", "fpex": "p67b_fpex", "fpcmp": "p67b_fpcmp"}
DEPTHS = ("8", "16", "32", "64", "128"); DEEP = ("16", "32", "64", "128"); THR = 0.1867


def load():
    out = {}
    for a, pre in ARMS.items():
        out[a] = {}
        for s in range(1, 6):
            f = f"c1/runs/{pre}_s{s}/eval.json"
            if os.path.exists(f):
                out[a][s] = json.load(open(f))
    return out


def main():
    R = load(); n = {a: len(v) for a, v in R.items()}; v = {}
    d1 = R["d1b"]
    passed = [s for s, e in d1.items() if all(e["acc"][N] >= 0.99 for N in DEPTHS)]
    v["C1-1"] = {"passing_seeds": passed, "k": len(passed), "met": len(passed) >= 4,
                 "per_seed_min_acc": {s: min(e["acc"].values()) for s, e in d1.items()}}
    v["C1-2"] = {"certified": {s: d1[s]["certificate"]["certified"] for s in passed},
                 "met": bool(passed) and all(d1[s]["certificate"]["certified"] for s in passed)}
    nulls_ok, worst = True, 0.0
    for s in passed:
        for N in DEEP:
            for k, x in d1[s]["nulls"][N].items():
                worst = max(worst, x["upper95"]); nulls_ok &= x["upper95"] <= THR
    reset_ub = {s: {N: wilson_upper(round(e["acc"][N] * e["n_test"]), e["n_test"]) for N in DEEP} for s, e in R["reset"].items()}
    reset_ok = bool(reset_ub) and all(u <= THR for d in reset_ub.values() for u in d.values())
    v["C1-3"] = {"nulls_worst_upper95": worst, "nulls_ok": nulls_ok, "reset_upper95": reset_ub, "reset_ok": reset_ok,
                 "met": bool(passed) and nulls_ok and reset_ok}
    mech = {s: d1[s]["mechanism"] for s in passed}
    ok4 = bool(passed) and all(m["donor_acc"] >= 0.99 and (m.get("irrelevant_change_rate_reachable") is None or m["irrelevant_change_rate_reachable"] <= 0.01)
                               and m["same_semantic_agreement"] >= 0.99 for m in mech.values())
    v["C1-4"] = {"mechanism": mech, "met": ok4}
    c5 = {}
    for fp in ("fpex", "fpcmp"):
        c5[fp] = {N: float(np.median([e["acc"][N] for e in d1.values()]) - np.median([e["acc"][N] for e in R[fp].values()]))
                  if d1 and R[fp] else None for N in DEEP}
    v["C1-5"] = {"median_gap": c5, "met": all(x is not None and x >= 0.20 for d in c5.values() for x in d.values())}
    for k, arms in (("C1-1", ["d1b"]), ("C1-2", ["d1b"]), ("C1-3", ["d1b", "reset"]), ("C1-4", ["d1b"]), ("C1-5", ["d1b", "fpex", "fpcmp"])):
        v[k]["n_runs"] = {a: n[a] for a in arms}
        if any(n[a] < 5 for a in arms):
            v[k]["status"] = "incomplete"
    desc = {a: {s: {"acc": e["acc"], "certified": e.get("certificate", {}).get("certified"),
                    "counter_check": e.get("counter_check"), "train_sec": e.get("train", {}).get("sec")}
                for s, e in R[a].items()} for a in R}
    res = {"n_runs": n, "verdicts": v, "per_run": desc}
    os.makedirs("theory/results", exist_ok=True)
    json.dump(res, open("theory/results/score_s67b.json", "w", newline="\n"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("runs", n)
    for a in R:
        for s, e in sorted(R[a].items()):
            print(f"{a:6s} s{s} cert {e.get('certificate', {}).get('certified')} acc " + " ".join(f"{e['acc'][N]:.3f}" for N in DEPTHS))
    for k in ("C1-1", "C1-2", "C1-3", "C1-4", "C1-5"):
        print(k, {x: y for x, y in v[k].items() if x not in ("mechanism", "reset_upper95", "per_seed_min_acc")})


if __name__ == "__main__":
    main()
