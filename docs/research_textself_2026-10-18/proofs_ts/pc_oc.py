"""PROOFS_TS check P5 (CPU, no model): base / latent n50s from the committed eval files, the N-* panel deltas re-derived, the bars, the noise
model vs the measured between-seed n50 spreads, and the comparison OC by exact 1-D quadrature of the bivariate normal (no Monte Carlo).
Reads (read only): s170_cot_s1 / s2 eval.json; s213_d3t_s1, s215_ftself_s1, s216_d3t_s2, s216_r1_s2 eval_vb.json; s216_r2_s1 eval_deep.json;
TS_draft's ts_panels.json (POS-* deltas only). python -B pc_oc.py"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "pc_oc.json"); R = "D:/Latent/boxes/runs_fsq"
TS = "D:/Latent/docs/research_textself_2026-10-18/scripts"


def n_cross(a, thr, lo=None, hi=None):
    lo = lo or min(a); hi = hi or max(a); ns = list(range(lo, hi + 1))
    if a[ns[0]] < thr:
        return lo - 0.5
    for n0, n1 in zip(ns, ns[1:]):
        if a[n0] >= thr > a[n1]:
            return n0 + (a[n0] - thr) / (a[n0] - a[n1])
    return hi + 0.5


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def rect(m1, s1, m2, s2, rho, a, b, c, d, M=4000):
    """P(a <= X1 <= b, c <= X2 <= d) for a bivariate normal, by Simpson quadrature over X1."""
    lo = max(a, m1 - 9 * s1); hi = min(b, m1 + 9 * s1)
    if hi <= lo:
        return 0.0
    h = (hi - lo) / M; sc = s2 * math.sqrt(1 - rho ** 2); tot = 0.0
    for i in range(M + 1):
        x = lo + i * h; w = 1 if i in (0, M) else (4 if i % 2 else 2)
        mc = m2 + rho * s2 / s1 * (x - m1); pdf = math.exp(-0.5 * ((x - m1) / s1) ** 2) / (s1 * math.sqrt(2 * math.pi))
        up = 1.0 if d == math.inf else Phi((d - mc) / sc); dn = 0.0 if c == -math.inf else Phi((c - mc) / sc)
        tot += w * pdf * (up - dn)
    return tot * h / 3


def classify(m1, m2, s, rho=0.5, k2=math.sqrt(2)):
    s1, s2 = s, s * k2; inf = math.inf
    par = rect(m1, s1, m2, s2, rho, -0.5, 0.5, -0.5, 0.5)
    ta = rect(m1, s1, m2, s2, rho, -0.5, inf, 0.5000000001, inf)
    la = rect(m1, s1, m2, s2, rho, -inf, 0.5, -inf, -0.5000000001)
    return {"PARITY": par, "TEXT-AHEAD": ta, "LATENT-AHEAD": la, "CROSSED": 1 - par - ta - la}


def main():
    res = {}
    base = {}
    for run in ("s170_cot_s1", "s170_cot_s2"):
        a = {int(k): v for k, v in json.load(open(os.path.join(R, run, "eval.json")))["acc_by_n"].items()}
        base[run] = {"n50": n_cross(a, 0.5), "n90": n_cross(a, 0.9), "a9_12": [a[n] for n in (9, 10, 11, 12)], "mean9_12": sum(a[n] for n in (9, 10, 11, 12)) / 4,
                     "floor13_16": sum(a[n] for n in (13, 14, 15, 16)) / 4, "eval_dtype": json.load(open(os.path.join(R, run, "eval.json"))).get("eval_dtype"),
                     "gpu": json.load(open(os.path.join(R, run, "eval.json"))).get("gpu"), "a": a}
    lat = {}
    for run, f in (("s213_d3t_s1", "eval_vb.json"), ("s215_ftself_s1", "eval_vb.json"), ("s216_d3t_s2", "eval_vb.json"), ("s216_r1_s2", "eval_vb.json"), ("s216_r2_s1", "eval_deep.json")):
        a = {int(k): v for k, v in json.load(open(os.path.join(R, run, f)))["acc_n2"].items()}; lat[run] = {"n50": n_cross(a, 0.5), "n90": n_cross(a, 0.9), "a": a}
    res["text_base"] = {k: {kk: vv for kk, vv in v.items() if kk != "a"} for k, v in base.items()}; res["latent"] = {k: {kk: vv for kk, vv in v.items() if kk != "a"} for k, v in lat.items()}
    L = {0: lat["s213_d3t_s1"]["n50"], 1: lat["s215_ftself_s1"]["n50"], 2: lat["s216_r2_s1"]["n50"]}; res["latent_s1_n50"] = L
    res["ratio_n50_support"] = {"latent": [L[0] / 8, L[1] / 12, L[2] / 16], "text_s1_base": base["s170_cot_s1"]["n50"] / 8}
    # N-SHIFT / N-PROP deltas (independent re-derivation; a_0 beyond 16 := floor c)
    a0 = dict(base["s170_cot_s1"]["a"]); c = base["s170_cot_s1"]["floor13_16"]
    for n in range(17, 25):
        a0[n] = c

    def interp(x):
        if x <= 1:
            return 1.0
        if x >= 16:
            return c
        lo = int(math.floor(x)); f = x - lo; return (1 - f) * a0[lo] + f * a0[lo + 1]
    B = {1: 12, 2: 16}; pan = {}
    for r in (1, 2):
        sh = {n: (a0[n - 4 * r] if n - 4 * r >= 1 else 1.0) for n in range(1, 25)}
        pr = {n: (1.0 if n <= B[r] else interp(8 + (n - B[r]) * 8 / B[r])) for n in range(1, 25)}
        pan["N-SHIFT_%d" % r] = {"n50": n_cross(sh, 0.5, 1, 24), "d": n_cross(sh, 0.5, 1, 24) - L[r], "M": sum(sh[n] for n in range(B[r] + 1, B[r] + 5)) / 4}
        pan["N-PROP_%d" % r] = {"n50": n_cross(pr, 0.5, 1, 24), "d": n_cross(pr, 0.5, 1, 24) - L[r], "M": sum(pr[n] for n in range(B[r] + 1, B[r] + 5)) / 4}
    res["panels_N"] = pan
    PJ = json.load(open(os.path.join(TS, "ts_panels.json")))["bases"]["s170_cot_s1"]["panels"]
    D = {"N-SHIFT": (pan["N-SHIFT_1"]["d"], pan["N-SHIFT_2"]["d"]), "N-PROP": (pan["N-PROP_1"]["d"], pan["N-PROP_2"]["d"]),
         "POS-PROP": (PJ["POS-PROP"]["1"]["dn50"], PJ["POS-PROP"]["2"]["dn50"]), "POS-ADD": (PJ["POS-ADD"]["1"]["dn50"], PJ["POS-ADD"]["2"]["dn50"])}
    D["BETTER"] = (1.5, 3.0); D["WORSE"] = (D["N-SHIFT"][0] - 1.0, D["N-SHIFT"][1] - 2.0)
    PRI = {"N-SHIFT": 0.25, "POS-ADD": 0.20, "N-PROP": 0.15, "POS-PROP": 0.20, "BETTER": 0.10, "WORSE": 0.10}
    res["panel_deltas"] = D
    # bars (ceil to 0.005)
    ce = lambda x: math.ceil(round(x / 0.005, 9)) * 0.005  # noqa: E731
    m0 = base["s170_cot_s1"]["mean9_12"]; res["text_bars_bf16"] = {"MG_T": ce(m0 + 0.15), "SB_T": ce(m0 - 0.08)}
    al = lat["s213_d3t_s1"]["a"]; a1 = lat["s215_ftself_s1"]["a"]; ml = sum(al[n] for n in (9, 10, 11, 12)) / 4
    m2s = sum(a1[n] for n in (13, 14, 15, 16)) / 4

    def a1i(x):
        lo = int(math.floor(x)); f = x - lo; return (1 - f) * a1[lo] + f * a1[lo + 1]
    m2p = sum(a1i(12 + 0.75 * d) for d in (1, 2, 3, 4)) / 4
    res["latent_bars_check"] = {"MG": ce(ml + 0.15), "SB": ce(ml - 0.08), "MG2": ce((m2s + m2p) / 2), "SB2": ce((ml + m2s) / 2), "want": [0.63, 0.40, 0.785, 0.60]}
    # noise: the measured between-lineage n50 differences (each = one lineage minus another of the same arm, i.e. a draw of Delta under no difference)
    diffs = {"latent r0 (s2 - s1)": lat["s216_d3t_s2"]["n50"] - lat["s213_d3t_s1"]["n50"], "latent r1 (s2 - s1)": lat["s216_r1_s2"]["n50"] - lat["s215_ftself_s1"]["n50"],
             "text r0 bf16 (s2 - s1)": base["s170_cot_s2"]["n50"] - base["s170_cot_s1"]["n50"]}
    s_hat = math.sqrt(sum(v ** 2 for v in diffs.values()) / len(diffs))
    res["noise"] = {"diffs": diffs, "s_hat_rms_3df": s_hat, "s_95ci_chi2_3df": [s_hat * math.sqrt(3 / 9.348), s_hat * math.sqrt(3 / 0.2158)],
                    "note": "a between-lineage difference of one arm has the variance of Delta (one text minus one latent lineage) when both arms' lineage sd are equal"}
    # comparison OC
    oc = {}
    for s in (s_hat, 0.2, 0.35, 0.5):
        for pooled in (False, True):
            se = s / math.sqrt(2) if pooled else s; key = "%s_s%.3f" % ("POOLED" if pooled else "ONE", s); tab = {}
            for name, (m1, m2) in list(D.items()) + [("NO-DIFF", (0.0, 0.0)), ("-1/-1", (-1.0, -1.0)), ("+1/+1", (1.0, 1.0)), ("+0.5/+0.5", (0.5, 0.5))]:
                tab[name] = classify(m1, m2, se)
            head = {lab: sum(PRI[k] * tab[k][lab] for k in PRI) for lab in ("PARITY", "TEXT-AHEAD", "LATENT-AHEAD", "CROSSED")}
            oc[key] = {"by_case": tab, "headline_given_both": head}
            print(key, {k: round(v, 3) for k, v in head.items()}, "NO-DIFF", {k: round(v, 3) for k, v in tab["NO-DIFF"].items()}, "-1/-1", {k: round(v, 3) for k, v in tab["-1/-1"].items()}, flush=True)
    # sensitivity: random-walk rounds (rho = 1/sqrt 2) and no variance growth (k2 = 1, rho 0.5)
    sens = {}
    for lab, rho, k2 in (("rw_rho0.707", 1 / math.sqrt(2), math.sqrt(2)), ("no_growth_rho0.5", 0.5, 1.0)):
        for s in (s_hat, 0.35):
            sens["%s_s%.3f" % (lab, s)] = {"NO-DIFF": classify(0, 0, s, rho, k2), "-1/-1": classify(-1, -1, s, rho, k2)}
    res["oc"] = oc; res["oc_sensitivity"] = sens
    # structural products
    P = {"T1_UNCENS": 0.95, "ABSORB": 0.90, "LAB": 0.99, "GATE2": 0.85, "T2_UNCENS": 0.95, "ABSORB2": 0.85, "RETAIN": 0.95}
    t1 = P["T1_UNCENS"] * P["LAB"]; r2c = t1 * P["ABSORB"]; both = r2c * P["GATE2"] * P["T2_UNCENS"] * P["LAB"]
    res["structural"] = {"T-SELF": r2c, "round2_condition": r2c, "both_readable": both, "T-SELF-2": both * P["ABSORB2"] * P["RETAIN"],
                         "round1_only_not_absorb": P["T1_UNCENS"] * P["LAB"] * (1 - P["ABSORB"]), "label_stall": r2c * (1 - P["GATE2"]),
                         "censored": 1 - both - P["T1_UNCENS"] * P["LAB"] * (1 - P["ABSORB"]) - r2c * (1 - P["GATE2"])}
    print(json.dumps({k: v for k, v in res.items() if k not in ("oc", "oc_sensitivity")}, indent=1, default=str))
    for k, v in sens.items():
        print(k, {c_: {kk: round(vv, 3) for kk, vv in d.items()} for c_, d in v.items()})
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
