"""TEXT-SELF draft check 4 (CPU, no model): hypothesis panels for the text lineage and the matched-comparison OC.
Inputs [MEASURED]: s170_cot_s1 / s2 eval.json acc_by_n (n 1-16; RTX 3070, bf16 autocast - the registered base curve is the fp32 re-eval, which
replaces these numbers before any text cell is read); latent s1 n50 per round 10.31 / 15.46 / 20.54 (THEORY §216 s1 round-2 result; SD2_v1 §3.4).
Per-item structure: test_set(1000, n_max 24) totals tot = P + C (generator statistics; ts_tokens.py).
Panels (each a CONJECTURE; numbers DERIVED here), round r = 1 (support B 12, trained rows T_1 = 311) and r = 2 (B 16, T_2 = 527); base B 8, T_0 = 147:
  N-SHIFT : a_r(n) = a_0(n - 4r)                      (the latent H_shift analogue: constant margin in operations)
  N-PROP  : a_r(n) = a_0(8 + (n - B_r) * 8 / B_r)      (margin proportional to support, in operations; linear interpolation in n)
  POS-PROP: P(correct | item) = c + (1 - c) * sig((x0 - tot / T_r) / w), fitted on the base at n 9-12 with x = tot / T_0
  POS-ADD : the same with x = tot - T_r (a constant number of positions past the trained rows)
  where c = the base's mean a_0 over n 13-16 (chance-like floor). a_0 beyond n 16 := c.
Readings per panel: M_1 = mean a_1(13..16) (T-DBS bar 0.80), M_2 = mean a_2(17..20), n50 / n90 / WIDTH per round, dn50_r = n50_text - n50_latent.
OC: dn50 noise ~ Normal(0, s) per round, rounds correlated rho; PARITY iff |d1| <= 0.5 and |d2| <= 0.5; TEXT-AHEAD iff not PARITY and d2 > 0.5 and
d1 >= -0.5; LATENT-AHEAD iff not PARITY and d2 < -0.5 and d1 <= 0.5; else CROSSED. Priors over panels [ASSUMED] in PRIORS.
python -B ts_panels.py"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1"); sys.dont_write_bytecode = True
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested170 as N  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ts_panels.json")
LAT = {0: 10.31, 1: 15.46, 2: 20.54}
B = {0: 8, 1: 12, 2: 16}; TR = {0: 147, 1: 311, 2: 527}                    # trained max rows (ts_tokens: plain n<=8 max 147; T1 max 311; T2 max 527)
PRIORS = {"N-SHIFT": 0.25, "POS-ADD": 0.20, "N-PROP": 0.15, "POS-PROP": 0.20, "BETTER": 0.10, "WORSE": 0.10}


def n_cross(a, thr, lo=1, hi=24):
    """first downward crossing of thr by a(n), n lo..hi, linear interpolation; '>hi' / '<lo' sentinels as floats."""
    ns = list(range(lo, hi + 1))
    if a[ns[0]] < thr:
        return float(lo) - 0.5
    for n0, n1 in zip(ns, ns[1:]):
        if a[n0] >= thr > a[n1]:
            return n0 + (a[n0] - thr) / (a[n0] - a[n1])
    return float(hi) + 0.5


def interp(a0, x, c):
    if x <= 1:
        return 1.0
    if x >= 16:
        return c
    lo = int(np.floor(x)); f = x - lo
    return (1 - f) * a0[lo] + f * a0[lo + 1]


def fit_pos(a0, tot, c, mode):
    """grid-fit (x0, w) of the per-item logistic on the base at n 9..12 (least squares on a(n))."""
    best = None
    xs = {n: (np.array(tot[n]) / TR[0] if mode == "prop" else np.array(tot[n]) - TR[0]) for n in tot}
    grid0 = np.linspace(1.0, 2.0, 201) if mode == "prop" else np.linspace(0, 120, 241)
    gridw = np.linspace(0.01, 0.3, 59) if mode == "prop" else np.linspace(1, 40, 79)
    for x0 in grid0:
        for w in gridw:
            err = 0.0
            for n in (9, 10, 11, 12):
                p = c + (1 - c) / (1 + np.exp(-(x0 - xs[n]) / w)); err += (p.mean() - a0[n]) ** 2
            if best is None or err < best[0]:
                best = (err, float(x0), float(w))
    return best


def curve_pos(tot, c, x0, w, Tr, mode):
    out = {}
    for n in range(1, 25):
        x = np.array(tot[n]) / Tr if mode == "prop" else np.array(tot[n]) - Tr
        out[n] = float((c + (1 - c) / (1 + np.exp(-(x0 - x) / w))).mean())
    return out


def main():
    TE = N.test_set(1000, n_max=24); tot = {n: [] for n in range(1, 25)}
    for it in TE:
        st = N.cot_steps(it["root"]); C = sum(len(s) for s in st[1:]) + len(st) - 1   # e_1..e_n tokens + (n - 1) ' =' + EOT
        tot[it["n"]].append(len(it["syms"]) + 2 + C)
    res = {"latent_n50": LAT, "priors": PRIORS, "bases": {}}
    for run in ("s170_cot_s1", "s170_cot_s2"):
        e = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval.json" % run)); a0 = {int(k): v for k, v in e["acc_by_n"].items()}
        c = float(np.mean([a0[n] for n in (13, 14, 15, 16)]))
        for n in range(17, 25):
            a0[n] = c
        pan = {}
        pan["N-SHIFT"] = {r: {n: (a0[n - 4 * r] if n - 4 * r >= 1 else 1.0) for n in range(1, 25)} for r in (1, 2)}
        pan["N-PROP"] = {r: {n: (1.0 if n <= B[r] else interp(a0, 8 + (n - B[r]) * 8 / B[r], c)) for n in range(1, 25)} for r in (1, 2)}
        fp = fit_pos(a0, tot, c, "prop"); fa = fit_pos(a0, tot, c, "add")
        pan["POS-PROP"] = {r: curve_pos(tot, c, fp[1], fp[2], TR[r], "prop") for r in (1, 2)}
        pan["POS-ADD"] = {r: curve_pos(tot, c, fa[1], fa[2], TR[r], "add") for r in (1, 2)}
        base = {"a0": a0, "floor_c": c, "n50": n_cross(a0, 0.5), "n90": n_cross(a0, 0.9), "fit_pos_prop": fp, "fit_pos_add": fa,
                "fit_check": {"prop": {n: curve_pos(tot, c, fp[1], fp[2], TR[0], "prop")[n] for n in (9, 10, 11, 12)},
                              "add": {n: curve_pos(tot, c, fa[1], fa[2], TR[0], "add")[n] for n in (9, 10, 11, 12)}}}
        rows = {}
        for k, pr in pan.items():
            rr = {}
            for r in (1, 2):
                a = pr[r]; n50 = n_cross(a, 0.5); n90 = n_cross(a, 0.9)
                rr[r] = {"M": float(np.mean([a[n] for n in range(B[r] + 1, B[r] + 5)])), "n50": n50, "n90": n90, "width": n50 - n90,
                         "n50_over_B": n50 / B[r], "dn50": n50 - LAT[r], "absorb_min": float(min(a[n] for n in range(B[r] - 3, B[r] + 1))),
                         "a": {n: round(a[n], 4) for n in range(B[r] - 3, 25)}}
            rows[k] = rr
        base["panels"] = rows; res["bases"][run] = base
        print(run, "base n50 %.3f n90 %.3f c %.3f" % (base["n50"], base["n90"], c), "pos-prop fit", fp, "pos-add fit", fa, flush=True)
        for k, rr in rows.items():
            print("  %-8s r1 M %.3f n50 %.2f d %.2f | r2 M %.3f n50 %.2f d %.2f" % (k, rr[1]["M"], rr[1]["n50"], rr[1]["dn50"], rr[2]["M"], rr[2]["n50"], rr[2]["dn50"]), flush=True)
    # OC over panels (s1 base), with BETTER / WORSE as pre-stated offsets [ASSUMED]: BETTER = latent + 1.5 per round, WORSE = N-SHIFT - 1 per round
    p1 = res["bases"]["s170_cot_s1"]["panels"]
    mean_d = {k: (p1[k][1]["dn50"], p1[k][2]["dn50"]) for k in ("N-SHIFT", "POS-ADD", "N-PROP", "POS-PROP")}
    mean_d["BETTER"] = (1.5, 3.0); mean_d["WORSE"] = (p1["N-SHIFT"][1]["dn50"] - 1.0, p1["N-SHIFT"][2]["dn50"] - 2.0)
    rng = np.random.default_rng(23193); oc = {}

    def classify(d1, d2):
        par = (np.abs(d1) <= 0.5) & (np.abs(d2) <= 0.5)
        ta = ~par & (d2 > 0.5) & (d1 >= -0.5); la = ~par & (d2 < -0.5) & (d1 <= 0.5)
        return par, ta, la, ~(par | ta | la)
    for s in (0.2, 0.35, 0.5):
        for rho in (0.5,):
            z1 = rng.standard_normal(400_000); z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(400_000)
            tab = {}
            for k, (m1, m2) in list(mean_d.items()) + [("TRUE-PARITY", (0.0, 0.0)), ("DELTA+1", (1.0, 1.0)), ("DELTA-1", (-1.0, -1.0)), ("DELTA+-0.5", (0.5, 0.5))]:
                par, ta, la, cr = classify(m1 + s * z1, m2 + s * np.sqrt(2) * z2)          # round-2 noise compounds: sd x sqrt(2) [ASSUMED]
                tab[k] = {"PARITY": float(par.mean()), "TEXT-AHEAD": float(ta.mean()), "LATENT-AHEAD": float(la.mean()), "CROSSED": float(cr.mean()), "mean_d": (m1, m2)}
            head = {lab: sum(PRIORS[k] * tab[k][lab] for k in PRIORS) for lab in ("PARITY", "TEXT-AHEAD", "LATENT-AHEAD", "CROSSED")}
            oc["s%.2f_rho%.1f" % (s, rho)] = {"by_panel": tab, "headline_given_T_SELF_both_rounds": head}
            print("OC s %.2f" % s, {k: round(v, 3) for k, v in head.items()}, flush=True)
            for k in tab:
                print("   %-12s" % k, {kk: round(vv, 3) for kk, vv in tab[k].items() if kk != "mean_d"}, flush=True)
    res["oc"] = oc
    # round-3 context check on FRESH draws (rng 23192; never the sealed n 25-30 test items): share with P + C <= 1025
    rng3 = np.random.default_rng(23192); r3 = {}
    for n in range(21, 31):
        ok = 0; mx = 0
        for _ in range(2000):
            r_, _t = N.generate(rng3, n); st = N.cot_steps(r_); C = sum(len(s) for s in st[1:]) + len(st) - 1; T_ = len(N.render(r_)) + 2 + C
            ok += T_ <= 1025; mx = max(mx, T_)
        r3[n] = {"feasible_share": ok / 2000, "max_tot": mx}
    res["fresh_ctx_n21_30"] = r3; print("ctx", r3, flush=True)
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
