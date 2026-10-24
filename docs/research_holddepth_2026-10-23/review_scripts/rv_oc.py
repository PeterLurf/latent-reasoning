"""REVIEW_HD: independent OC for the DEPTH bar (per-seed Delta >= -0.03), closed form (normal), CPU.
Model: per-seed observed Delta_s = mu + e_s, e_s ~ N(0, sigma) (sigma = per-pair noise: eval binomial + scratch-training seed/trajectory noise + effect heterogeneity).
    python -B docs/research_holddepth_2026-10-23/review_scripts/rv_oc.py"""
import json, os, math
from math import comb
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))   # noqa: E731
BAR = -0.03; out = {}
# sigma evidence
pairs_ft = [0.005, -0.004, -0.030, 0.007, -0.033]
out["rms_5_pairs"] = math.sqrt(sum(x * x for x in pairs_ft) / 5)
out["base_s1_s2_diff"] = 0.666 - 0.597; out["sigma_run_from_bases"] = (0.666 - 0.597) / math.sqrt(2)
rd = [0.521 - 0.597, 0.512 - 0.666]; out["rd_delta_mean"] = sum(rd) / 2
out["eval_binomial_se_diff"] = math.sqrt(2) * math.sqrt(sum(p * (1 - p) / 1000 for p in (0.9, 0.6, 0.25))) / 3
for sig in (0.02, 0.03, 0.05, 0.07):
    row = {}
    for name, mu in (("full (mu 0)", 0.0), ("half (mu -0.057)", -0.057), ("none (mu -0.115, RD s1/s2 mean)", -0.115), ("none, s1-like (mu -0.076)", -0.076)):
        p = Phi((mu - BAR) / sig)
        both2 = p * p
        two_of_three = 3 * p * p * (1 - p) + p ** 3
        mean2 = Phi((mu - BAR) / (sig / math.sqrt(2)))
        guard = Phi((mu + 0.08) / sig)       # each seed >= -0.08
        mean2_guard = mean2 * guard * guard   # approx (positively correlated -> slight under-estimate)
        mean3 = Phi((mu - BAR) / (sig / math.sqrt(3)))
        row[name] = {"p_seed": round(p, 3), "both_of_2": round(both2, 3), "2_of_3": round(two_of_three, 3), "mean_of_2>=bar": round(mean2, 3),
                     "mean_of_2 & each>=-0.08": round(mean2_guard, 3), "mean_of_3>=bar": round(mean3, 3)}
    out["sigma=%.2f" % sig] = row
# P(HD-LEVER | full effect) with HOLD prob h and NOLOSS 0.97
out["P_lever_full"] = {"sig %.2f, P(HOLD both) %.2f" % (s, h): round(Phi(-BAR / s) ** 2 * h * 0.97, 3) for s in (0.03, 0.05, 0.07) for h in (0.85, 0.5)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rv_oc.out.json"), "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
