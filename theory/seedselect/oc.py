"""Operating characteristics (rule (a)) of SEEDSELECT's rev-2 two-part pass rule at R1's actual strata (3070: 5 S / 10 F;
4060: 7 S / 6 F; 92 pairs). Synthetic scores only (equal-variance binormal), using seedselect.py's own perm_test and
paired_beat_null. Writes theory/seedselect/oc.json.
    python theory/seedselect/oc.py
"""
import json
import math
import os
import sys

import numpy as np

try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seedselect as S  # noqa: E402

STRATA = [(5, 10), (7, 6)]


def Phi_inv(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def one(rng, auc_f, auc_n, rho, n_perm):
    df = math.sqrt(2) * Phi_inv(auc_f); dn = math.sqrt(2) * Phi_inv(auc_n)
    st2, st3 = [], []
    for ns, nf in STRATA:
        y = np.array([1.0] * ns + [0.0] * nf); n = ns + nf
        e1 = rng.normal(0, 1, n); e2 = rho * e1 + math.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
        xf = df * y + e1; xn = dn * y + e2
        st2.append((xf, y)); st3.append((xf, xn, y))
    r = S.perm_test(st2, n_perm, rng, True)
    feat_part = r["p"] < 0.05 and r["C"] >= S.SPEC["thresholds"]["useful_C"]
    rev_part = r["p"] < 0.05 and r["C"] <= 1 - S.SPEC["thresholds"]["useful_C"]
    b = S.paired_beat_null(st3, n_perm, rng)
    beat = b["p_one_sided"] < S.SPEC["beat_null"]["alpha"] and b["dC"] >= S.SPEC["beat_null"]["min_delta_C"]
    return feat_part, rev_part, beat


def main():
    rng = np.random.default_rng(20261014); n_sim, n_perm = 400, 500
    scen = {
        "A no effect (feature 0.5, null 0.5, rho 0.5)": (0.5, 0.5, 0.5),
        "B confound only (feature 0.8 = the null's signal: null 0.8, rho 0.7)": (0.8, 0.8, 0.7),
        "C real effect, uninformative null (feature 0.8, null 0.5, rho 0.3)": (0.8, 0.5, 0.3),
        "D real effect beyond a partly informative null (feature 0.8, null 0.65, rho 0.5)": (0.8, 0.65, 0.5),
        "E real effect 0.9, null 0.65, rho 0.5": (0.9, 0.65, 0.5),
        "F marginal effect 0.7, null 0.5, rho 0.3": (0.7, 0.5, 0.3),
    }
    out = {"strata": STRATA, "pairs": sum(a * b for a, b in STRATA), "n_sim": n_sim, "n_perm": n_perm, "rule": {
        "feature_part": "two-sided permutation p < 0.05 and C >= %.2f" % S.SPEC["thresholds"]["useful_C"],
        "beat_part": "paired rank-swap p < %.2f (one-sided) and dC >= %.2f vs the primary null" % (S.SPEC["beat_null"]["alpha"], S.SPEC["beat_null"]["min_delta_C"])},
        "scenarios": {}}
    for name, (af, an, rho) in scen.items():
        fp = rv = bt = both = 0
        for _ in range(n_sim):
            f, r_, b = one(rng, af, an, rho, n_perm); fp += f; rv += r_; bt += b; both += (f and b)
        se = lambda k: math.sqrt(max(k, 1) * (n_sim - k) / n_sim) / n_sim  # noqa: E731
        out["scenarios"][name] = {"P_feature_part_rev1": fp / n_sim, "P_reversed_rev1": rv / n_sim, "P_beats_null": bt / n_sim,
                                  "P_USEFUL_rev2": both / n_sim, "mc_se_USEFUL": round(se(both), 4)}
        print(name, out["scenarios"][name], flush=True)
    json.dump(out, open(os.path.join(HERE, "oc.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
