"""REVIEW_OW independent OC (CPU, 1 thread): re-derives OW_draft §8 by (i) a normal approximation (no simulation) and (ii) a separate simulation
(own seed, paired R2 arms with item-level correlation), and adds scenarios the draft omits: a near-bar null, a serial carrier that recomputes
back-references from the visible question (outputs cf_b, not cf(a), where they differ; share from rv_task.json), and that carrier under a
cf(a)-or-cf_b rule. Writes rv_oc.json next to this file.   python -B docs/research_ordered_words_2026-10-23/review_scripts/rv_oc.py"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, math  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); REPS = 40000; DEFF = 1.3; NB = 3000; NP = 2000   # NP: B∩{d>=4} >= 2 depths x 1,000 (cap)
BAR = dict(g1=0.30, a1=0.60, g2=0.30, g3=0.30)
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
CFB_DIFF = 0.60          # share of kept pairs (d 4-8) with cf_b != cf(a): rv_task.json 0.54-0.69


def normal_p(s, nb=NB, npairs=NP):
    ne, nep = nb / DEFF, npairs / DEFF
    se = lambda p, n: math.sqrt(p * (1 - p) / n)  # noqa: E731
    g = lambda d, sd: 1 - Phi((0 - d) / sd) if sd > 0 else float(d >= 0)  # noqa: E731  P(X >= 0), X ~ N(d, sd)
    p1 = g(s["lat"] - s["dir"] - BAR["g1"], math.hypot(se(s["lat"], ne), se(s["dir"], ne))) * g(s["lat"] - BAR["a1"], se(s["lat"], ne))  # indep. approx (lower bound)
    p2 = g(s["lat"] - s["k0"] - BAR["g2"], math.hypot(se(s["lat"], ne), se(s["k0"], ne))) * g(s["lat"] - s["k1"] - BAR["g2"], math.hypot(se(s["lat"], ne), se(s["k1"], ne)))
    p3 = g(s["cf"] - s["fl"] - BAR["g3"], math.hypot(se(s["cf"], nep), se(s["fl"], nep)))
    return {"R1_indep_approx": p1, "R2_indep_approx": p2, "R3": p3, "z_R1_gap": (s["lat"] - s["dir"] - BAR["g1"]) / math.hypot(se(s["lat"], ne), se(s["dir"], ne))}


def sim(rng, s, nb=NB, npairs=NP):
    ne, nep = int(round(nb / DEFF)), int(round(npairs / DEFF))
    lat = rng.binomial(ne, s["lat"], REPS) / ne; dr = rng.binomial(ne, s["dir"], REPS) / ne
    # paired K' arms: item-level coupling (an item wrong at K is wrong at K' w.p. high) -> draw K' = lat - (gap drawn as a paired difference)
    def paired(pk):
        p_drop = max(s["lat"] - pk, 0.0)                     # share right at K, wrong at K' (minimal-discordance coupling)
        p_gain = max(pk - s["lat"], 0.0)
        disc = rng.multinomial(ne, [p_drop, p_gain, 1 - p_drop - p_gain], REPS)
        return (disc[:, 0] - disc[:, 1]) / ne              # lat_hat - k_hat
    g0, g1 = paired(s["k0"]), paired(s["k1"])
    cf = rng.binomial(nep, s["cf"], REPS) / nep; fl = rng.binomial(nep, s["fl"], REPS) / nep
    r1 = (lat - dr >= BAR["g1"]) & (lat >= BAR["a1"]); r2 = (g0 >= BAR["g2"]) & (g1 >= BAR["g2"]); r3 = cf - fl >= BAR["g3"]
    return {"R1": float(r1.mean()), "R2": float(r2.mean()), "R3": float(r3.mean()), "C1": float((r1 & r2 & r3).mean())}


SCEN = {
    "no_effect (draft)": dict(lat=0.30, dir=0.30, k0=0.30, k1=0.30, cf=0.10, fl=0.10),
    "null_near_bar (gap 0.20, cf+0.20)": dict(lat=0.55, dir=0.35, k0=0.35, k1=0.35, cf=0.30, fl=0.10),
    "at_bars (draft)": dict(lat=0.60, dir=0.30, k0=0.30, k1=0.30, cf=0.40, fl=0.10),
    "effect_marginal (draft)": dict(lat=0.64, dir=0.30, k0=0.30, k1=0.32, cf=0.43, fl=0.10),
    "effect_strong (draft)": dict(lat=0.80, dir=0.30, k0=0.25, k1=0.30, cf=0.55, fl=0.10),
    # serial carrier whose post-t steps re-read back-referenced S_j (j < t) from the visible question: it emits cf_b where cf_b != cf(a)
    "serial_cfb_type, cf(a) scored": dict(lat=0.75, dir=0.30, k0=0.25, k1=0.30, cf=0.55 * (1 - CFB_DIFF) + 0.10 * CFB_DIFF, fl=0.10),
    "serial_cfb_type, cf(a)-or-cf_b scored": dict(lat=0.75, dir=0.30, k0=0.25, k1=0.30, cf=0.55, fl=0.10 + 0.10 * CFB_DIFF),
    # floor collapse: mismatched-donor swaps fall back to own (excluded) -> floor ~0; follow small
    "floor_collapse_weak_carrier": dict(lat=0.70, dir=0.30, k0=0.30, k1=0.30, cf=0.25, fl=0.01),
}


def main():
    rng = np.random.default_rng(777); out = {"reps": REPS, "deff": DEFF, "n_band": NB, "n_pairs": NP, "bars": BAR, "cfb_diff_share": CFB_DIFF, "scen": {}}
    for nm, s in SCEN.items():
        out["scen"][nm] = {"truth": s, "normal": normal_p(s), "sim": sim(rng, s), "sim_npairs_1000": sim(rng, s, npairs=1000)}
    # band-selection bias: DIRECT acc at the selected depths is read from the same test items that select B (winner's curse at the 0.50 edge)
    true_acc = np.array([0.55, 0.50, 0.45, 0.35, 0.25]); n = 1000; sel_bias = []
    for _ in range(20000):
        hat = rng.binomial(n, true_acc) / n; m = hat <= 0.50
        if m.any():
            sel_bias.append(hat[m].mean() - true_acc[m].mean())
    out["band_selection_bias_mean"] = float(np.mean(sel_bias))
    with open(os.path.join(HERE, "rv_oc.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    for nm, r in out["scen"].items():
        print("%-42s normal R1 %.3f R2 %.3f R3 %.3f (zR1 %+.1f) | sim R1 %.3f R2 %.3f R3 %.3f C1 %.3f | N1000 C1 %.3f" % (
            nm, r["normal"]["R1_indep_approx"], r["normal"]["R2_indep_approx"], r["normal"]["R3"], r["normal"]["z_R1_gap"],
            r["sim"]["R1"], r["sim"]["R2"], r["sim"]["R3"], r["sim"]["C1"], r["sim_npairs_1000"]["C1"]))
    print("band selection bias", out["band_selection_bias_mean"])


if __name__ == "__main__":
    main()
