"""REDTEAM_TS check R2 ($0, CPU arithmetic): re-shade the TEXT-SELF comparison OC.
(1) measured lineage noise from the committed seed pairs (n50: latent base 10.31 / 10.51, text base 10.44 / 10.60, latent R1 15.46 / 15.33 (R1-s2
    censored, its n50 descriptive, 216-9)); (2) the four-way rule at s = sigma(Delta) in {0.17, 0.20, 0.35} with rounds correlated 0.5 and round 2 at
    s*sqrt(2) (the draft's model), plus a constant protocol bias b in {0, +-0.15, +-0.30}; (3) POOLED with the latent s2 lineage dead (2 text vs 1
    latent lineage); (4) band +-0.75; (5) base-corrected Delta^c_r = Delta_r - Delta_0; (6) shaded headline with P(GATE-T2 | T-ABSORB) 0.93 (rt_reach).
Panels from the draft's ts_panels.json (bf16 base curve). Writes rt_oc.json."""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); SCR = os.path.join(os.path.dirname(HERE), "scripts")
PAN = json.load(open(os.path.join(SCR, "ts_panels.json")))["bases"]["s170_cot_s1"]["panels"]
PRI = {"N-SHIFT": 0.25, "POS-ADD": 0.20, "N-PROP": 0.15, "POS-PROP": 0.20, "BETTER": 0.10, "WORSE": 0.10}
MEAN = {k: (PAN[k]["1"]["dn50"], PAN[k]["2"]["dn50"]) for k in ("N-SHIFT", "POS-ADD", "N-PROP", "POS-PROP")}
MEAN["BETTER"] = (1.5, 3.0); MEAN["WORSE"] = (MEAN["N-SHIFT"][0] - 1.0, MEAN["N-SHIFT"][1] - 2.0)
D0 = 10.4376 - 10.31                                                   # round-0 offset on the bf16 base curve (fp32 replaces)
rng = np.random.default_rng(23996); Z = 400_000
z1 = rng.standard_normal(Z); z2 = 0.5 * z1 + math.sqrt(0.75) * rng.standard_normal(Z); z0 = rng.standard_normal(Z)
out = {}
# (1) measured noise
pairs = {"latent base": (10.31, 10.51), "text base": (10.4376, 10.60), "latent R1 (s2 censored)": (15.46, 15.33)}
d = np.array([abs(a - b) for a, b in pairs.values()]); sig_arm = float(np.sqrt(np.mean(d ** 2) / 2))
out["noise"] = {"pairs": pairs, "abs_diffs": d.tolist(), "sigma_arm_rms": sig_arm, "sigma_Delta_one_lineage_each": sig_arm * math.sqrt(2),
                "chi2_3dof_95_upper_factor": math.sqrt(3 / 0.2158), "note": "3 seed pairs; sigma poorly determined (95% upper ~3.7x)"}


def classify(d1, d2, band=0.5):
    par = (np.abs(d1) <= band) & (np.abs(d2) <= band)
    ta = ~par & (d2 > band) & (d1 >= -band); la = ~par & (d2 < -band) & (d1 <= band)
    return {"PARITY": float(par.mean()), "TEXT-AHEAD": float(ta.mean()), "LATENT-AHEAD": float(la.mean()), "CROSSED": float((~(par | ta | la)).mean())}


def run(m1, m2, s, band=0.5, b=0.0):
    return classify(m1 + b + s * z1, m2 + b + s * math.sqrt(2) * z2, band)


cases = dict(MEAN); cases["NO-DIFF"] = (0.0, 0.0); cases["-1/-1"] = (-1.0, -1.0); cases["+1/+1"] = (1.0, 1.0)
tab = {}
for s in (0.17, 0.20, 0.35):
    for band in (0.5, 0.75):
        for b in (0.0, 0.15, 0.30, -0.15, -0.30):
            key = "s%.2f_band%.2f_bias%+.2f" % (s, band, b); r = {k: run(m[0], m[1], s, band, b) for k, m in cases.items()}
            head = {lab: sum(PRI[k] * r[k][lab] for k in PRI) for lab in ("PARITY", "TEXT-AHEAD", "LATENT-AHEAD", "CROSSED")}
            tab[key] = {"by_case": r, "prior_weighted_given_both": head}
out["one_lineage"] = tab
# (3) POOLED with the latent s2 lineage dead: Var = s^2 (1/4 + 1/2) if the two arms share s^2 equally
pool = {}
for s in (0.20, 0.35):
    se = s * math.sqrt(0.75)
    pool["s%.2f" % s] = {"s_eff": se, "NO-DIFF": run(0, 0, se), "-1/-1": run(-1, -1, se), "draft_claimed_s_eff (s/sqrt2)": s / math.sqrt(2)}
out["pooled_2text_1latent"] = pool
# (5) base-corrected Delta^c = Delta_r - Delta_0; Delta_0 noise s at round 0, corr(Delta_0, Delta_r) rho0 in {0, 0.5}
corr = {}
for s in (0.20, 0.35):
    for rho0 in (0.0, 0.5):
        e0 = s * z0
        r = {}
        for k, (m1, m2) in list(MEAN.items()) + [("NO-DIFF (proportional parity incl. offset scaling 12/8, 16/8)", (D0 * 1.5, D0 * 2.0))]:
            n1 = m1 + s * z1; n2 = m2 + s * math.sqrt(2) * z2
            # correlated round-0 error: Delta_0 = D0 + rho0 * (s z1) + sqrt(1 - rho0^2) * s * z0
            d0 = D0 + rho0 * s * z1 + math.sqrt(1 - rho0 ** 2) * e0
            r[k] = classify(n1 - d0, n2 - d0)
        corr["s%.2f_rho0_%.1f" % (s, rho0)] = {"by_case": r, "prior_weighted": {lab: sum(PRI[k] * r[k][lab] for k in PRI) for lab in ("PARITY", "TEXT-AHEAD", "LATENT-AHEAD", "CROSSED")}}
out["base_corrected"] = corr
# (6) shaded structural headline
P_T1 = 0.95 * 0.99; P_ABS = 0.90; P_G2_draft = 0.85; P_G2_rt = 0.93; P_T2 = 0.95 * 0.99
both_draft = P_T1 * P_ABS * P_G2_draft * P_T2; both_rt = P_T1 * P_ABS * P_G2_rt * P_T2
out["structural"] = {"p_both_rounds_draft": both_draft, "p_both_rounds_rt": both_rt, "p_stall_draft": P_T1 * P_ABS * (1 - P_G2_draft), "p_stall_rt": P_T1 * P_ABS * (1 - P_G2_rt),
                     "p_round1_only_not_absorbed": P_T1 * (1 - P_ABS)}
json.dump(out, open(os.path.join(HERE, "rt_oc.json"), "w", newline="\n"), indent=1)
print("noise", json.dumps(out["noise"]))
for key in ("s0.17_band0.50_bias+0.00", "s0.20_band0.50_bias+0.00", "s0.35_band0.50_bias+0.00", "s0.20_band0.50_bias+0.15", "s0.20_band0.50_bias+0.30",
            "s0.20_band0.50_bias-0.30", "s0.20_band0.75_bias+0.00", "s0.35_band0.75_bias+0.00"):
    t = tab[key]; print(key, "| NO-DIFF", {k: round(v, 3) for k, v in t["by_case"]["NO-DIFF"].items()}, "| -1/-1 LA", round(t["by_case"]["-1/-1"]["LATENT-AHEAD"], 3),
                        "PAR", round(t["by_case"]["-1/-1"]["PARITY"], 3), "| N-SHIFT LA", round(t["by_case"]["N-SHIFT"]["LATENT-AHEAD"], 3),
                        "| POS-PROP PAR", round(t["by_case"]["POS-PROP"]["PARITY"], 3), "N-PROP PAR", round(t["by_case"]["N-PROP"]["PARITY"], 3),
                        "| head", {k: round(v, 3) for k, v in t["prior_weighted_given_both"].items()})
print("pooled", json.dumps(pool))
for k, v in corr.items():
    print(k, {c: round(x["PARITY"], 3) for c, x in v["by_case"].items()}, "LA N-SHIFT", round(v["by_case"]["N-SHIFT"]["LATENT-AHEAD"], 3), "head", {a: round(b_, 3) for a, b_ in v["prior_weighted"].items()})
print("structural", json.dumps(out["structural"]))
