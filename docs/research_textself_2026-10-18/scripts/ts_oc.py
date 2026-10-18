"""TEXT-SELF draft check 5 (CPU): OC arithmetic for the per-cell readings and the matched comparison (one lineage per arm, and POOLED over two).
Every prior is [ASSUMED]; the panel means are DERIVED in ts_panels.json (s1 bf16 base curve). Lineage noise: Gaussian on M / M2 (sd sig_M) and on
dn50 (sd s at round 1, s * sqrt(2) at round 2, rounds correlated rho 0.5). python -B ts_oc.py"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ts_oc.json")
PAN = json.load(open(os.path.join(HERE, "ts_panels.json")))
P1 = PAN["bases"]["s170_cot_s1"]["panels"]
PRI = {"N-SHIFT": 0.25, "POS-ADD": 0.20, "N-PROP": 0.15, "POS-PROP": 0.20, "BETTER": 0.10, "WORSE": 0.10}
# panel means: M (r1, n 13-16), M2 (r2, n 17-20), dn50 r1 / r2
MEAN = {k: {"M": P1[k]["1"]["M"], "M2": P1[k]["2"]["M"], "d1": P1[k]["1"]["dn50"], "d2": P1[k]["2"]["dn50"]} for k in ("N-SHIFT", "POS-ADD", "N-PROP", "POS-PROP")}
MEAN["BETTER"] = {"M": 0.90, "M2": 0.95, "d1": 1.5, "d2": 3.0}
MEAN["WORSE"] = {"M": 0.30, "M2": 0.30, "d1": MEAN["N-SHIFT"]["d1"] - 1.0, "d2": MEAN["N-SHIFT"]["d2"] - 2.0}
# per-cell structural probabilities [ASSUMED]
P_T1_UNCENS = 0.95          # not dead / diverged / contaminated / args (engineering + FT-DIVERGED-T 0.03)
P_ABSORB = 0.90             # T-ABSORB (base learned n <= 8 by step ~600; REROOT labels keep ~1, pi ~1)
P_LAB_OK = 0.99             # LABEL-AUDIT-T and the gates
P_T2_UNCENS = 0.95
P_GATE2 = 0.85             # GATE-T2 / GATE-T1p pass with T1 (the base fallback fails W2: base REROOT keep 0.63 at n 16 < 0.70 [MEASURED ts_label_check_s1])
P_ABSORB2 = 0.85            # T-ABSORB-2 (rows to 527, 2/3 dose)
P_RETAIN = 0.95             # T-RETAIN | T-ABSORB-2


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def p_ge(mean, bar, sd):
    return 1 - phi((bar - mean) / sd)


def main():
    res = {"priors": PRI, "panel_means": MEAN, "structural": {"P_T1_UNCENS": P_T1_UNCENS, "P_ABSORB": P_ABSORB, "P_LAB_OK": P_LAB_OK,
                                                             "P_T2_UNCENS": P_T2_UNCENS, "P_GATE2": P_GATE2, "P_ABSORB2": P_ABSORB2, "P_RETAIN": P_RETAIN}}
    cell = {}
    for sd in (0.06, 0.09, 0.12):
        row = {}
        for k in PRI:
            row[k] = {"T-DBS": p_ge(MEAN[k]["M"], 0.80, sd), "T-DBS-2": p_ge(MEAN[k]["M2"], 0.80, sd)}
        head = {r: sum(PRI[k] * row[k][r] for k in PRI) for r in ("T-DBS", "T-DBS-2")}
        cell["sdM_%.2f" % sd] = {"by_panel": row, "prior_weighted": head}
    res["cell"] = cell
    p_t1_read = P_T1_UNCENS * P_LAB_OK
    p_round2 = p_t1_read * P_ABSORB                        # round-2 condition
    p_t2_read = p_round2 * P_GATE2 * P_T2_UNCENS * P_LAB_OK
    res["headline_cell"] = {"T-SELF (T1)": p_t1_read * P_ABSORB, "round-2 condition": p_round2, "T2 readable": p_t2_read,
                            "T-SELF-2": p_t2_read * P_ABSORB2 * P_RETAIN,
                            "T-DBS (sd 0.09)": p_t1_read * P_ABSORB * cell["sdM_0.09"]["prior_weighted"]["T-DBS"],
                            "T-DBS-2 & T-SELF-2 (sd 0.09)": p_t2_read * P_ABSORB2 * P_RETAIN * cell["sdM_0.09"]["prior_weighted"]["T-DBS-2"],
                            "T-ABSORB | no FT effect": 0.0, "T-DBS | no FT effect (base M 0.078)": 0.0}
    # comparison OC
    rng = np.random.default_rng(23194); Z = 400_000

    def classify(d1, d2):
        par = (np.abs(d1) <= 0.5) & (np.abs(d2) <= 0.5)
        ta = ~par & (d2 > 0.5) & (d1 >= -0.5); la = ~par & (d2 < -0.5) & (d1 <= 0.5)
        return {"PARITY": par, "TEXT-AHEAD": ta, "LATENT-AHEAD": la, "CROSSED": ~(par | ta | la)}
    comp = {}
    for pooled in (False, True):
        for s in (0.2, 0.35, 0.5):
            s_eff = s / math.sqrt(2) if pooled else s
            z1 = rng.standard_normal(Z); z2 = 0.5 * z1 + math.sqrt(0.75) * rng.standard_normal(Z)
            tab = {}
            cases = [(k, MEAN[k]["d1"], MEAN[k]["d2"]) for k in PRI] + [("NO-DIFF (d 0, 0)", 0.0, 0.0), ("d +1 / +1", 1.0, 1.0), ("d -1 / -1", -1.0, -1.0),
                                                                         ("d -0.5 / -1", -0.5, -1.0), ("d +0.5 / +0.5", 0.5, 0.5)]
            for name, m1, m2 in cases:
                c = classify(m1 + s_eff * z1, m2 + s_eff * math.sqrt(2) * z2)
                tab[name] = {k: float(v.mean()) for k, v in c.items()}
            head = {lab: sum(PRI[k] * tab[k][lab] for k in PRI) for lab in ("PARITY", "TEXT-AHEAD", "LATENT-AHEAD", "CROSSED")}
            # unconditional: the comparison needs both rounds readable; ROUND-1-ONLY otherwise (its own band on d1)
            comp["%s_s%.2f" % ("POOLED" if pooled else "ONE", s)] = {"by_case": tab, "given_both_rounds": head,
                                                                      "unconditional_both_rounds": {k: v * p_t2_read for k, v in head.items()}}
    res["comparison"] = comp
    res["p_both_rounds_readable"] = p_t2_read
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)
    print(json.dumps(res["headline_cell"], indent=1))
    for k, v in cell.items():
        print(k, {kk: round(vv, 3) for kk, vv in v["prior_weighted"].items()}, {p: {r: round(x, 3) for r, x in d.items()} for p, d in v["by_panel"].items()})
    for k, v in comp.items():
        print(k, "given both:", {kk: round(vv, 3) for kk, vv in v["given_both_rounds"].items()}, "uncond:", {kk: round(vv, 3) for kk, vv in v["unconditional_both_rounds"].items()})
        for c, d in v["by_case"].items():
            if c.startswith("NO-DIFF") or c.startswith("d "):
                print("    ", c, {kk: round(vv, 3) for kk, vv in d.items()})


if __name__ == "__main__":
    main()
