"""PROOFS_D4 check 7: OC arithmetic (D4 section 5) and cost / memory arithmetic (D4 section 3 (v)).
OC: P(DG) = P(E) P(DG|E) + (1 - P(E)) P(DG|not E); D4-C2 = DG and LF and VB-C1 and K24 (independence vs dependence);
    DEADLINE-GONE false-positive by binomial sampling when the true h(7-9) >= 0.30 and >= 300 events; 0 of 18 -> Clopper-Pearson / rule of three;
    the pre-stated OC shifts; the 48k cell.
Cost: s/step scaling of augmented batches (linear in Kb vs quadratic), 12k FT and 48k costs at $0.18-0.26/h; memory split 2 + 0.58 Kb + c sum(t+1)."""
import json, sys, math
from math import comb, lgamma, log, exp
def binom_cdf(k, N, p):
    return sum(exp(lgamma(N + 1) - lgamma(i + 1) - lgamma(N - i + 1) + i * log(p) + (N - i) * log(1 - p)) for i in range(0, k + 1))
out = {}
# ---- OC
PE = 0.45; dgE = (0.35, 0.45); dgN = 0.05
out["P_DG"] = {"with_leak": [PE * x + (1 - PE) * dgN for x in dgE], "no_leak": [PE * x for x in dgE]}
LF = 0.85
for vb, k24 in ((1.0, 1.0), (0.95, 0.95), (0.9, 0.9)):
    out.setdefault("P_C2_independent", []).append({"P_VB_C1": vb, "P_K24": k24, "P_C2": [p * LF * vb * k24 for p in out["P_DG"]["with_leak"]]})
# dependence bounds (Frechet) for P(DG and LF) given marginals
out["P_DG_and_LF_frechet"] = [[max(0.0, p + LF - 1), min(p, LF)] for p in out["P_DG"]["with_leak"]]
# DEADLINE-GONE: P(h_hat <= 0.10 | true h, N events)
out["DGone_false_pos_binomial"] = {"h=%.2f,N=%d" % (h, N): float(binom_cdf(math.floor(0.10 * N), N, h)) for h in (0.30, 0.20, 0.15, 0.12) for N in (300, 1000)}
# 0 of 18 VB-family models reached DG: upper confidence bounds
out["zero_of_18"] = {"clopper_pearson_95_upper": float(1 - 0.05 ** (1 / 18)), "rule_of_three": 3 / 18}
# shifts
out["P_DG_shifts"] = {k: [pe * x + (1 - pe) * dgN for x in dgE] for k, pe in {"AGE-CLOCK 0.6": 0.6, "PROGRESS-ONLY 0.2": 0.2, "notEC_notORD 0.3": 0.3}.items()}
# 48k
out["P_DG_48k_uncond"] = 0.7 * 0.2
# P(FT reads DEADLINE-GONE): P(E)*0.8 + P(notE)*FP; FP depends on whether the base is EC
for fp in (0.0, 0.05, 0.25):
    out.setdefault("P_FT_reads_DGone", []).append({"FP": fp, "P": PE * 0.8 + (1 - PE) * fp})
# ---- cost / memory
s_plain = 0.407; Kb_plain = 11.80; KbA = 18.59; KbB = 23.76
def sstep(Kb, kind):
    return s_plain * (Kb / Kb_plain if kind == "lin" else (Kb / Kb_plain) ** 2)
cost = {}
for name, Kb, draft_aug in (("A", KbA, 0.70), ("B", KbB, 0.9)):
    for kind in ("lin", "quad"):
        aug = sstep(Kb, kind); mean = 0.5 * s_plain + 0.5 * aug; h12 = 12000 * mean / 3600 + 0.4
        cost["%s_%s" % (name, kind)] = {"aug_s_step": aug, "mean_s_step": mean, "12k_hours": h12, "12k_usd": [h12 * 0.18, h12 * 0.26]}
    mean_d = 0.5 * s_plain + 0.5 * draft_aug; h12 = 12000 * mean_d / 3600 + 0.4
    cost["%s_draft_numbers" % name] = {"mean_s_step": mean_d, "12k_hours": h12, "12k_usd": [h12 * 0.18, h12 * 0.26]}
cost["CTRL"] = {"12k_hours": 12000 * s_plain / 3600 + 0.4, "12k_usd": [(12000 * s_plain / 3600 + 0.4) * r for r in (0.18, 0.26)]}
cost["A_48k_draft"] = [(48000 * 0.50 / 3600 + 0.5) * r for r in (0.18, 0.26)]; cost["B_48k_draft"] = [(48000 * 0.65 / 3600 + 0.5) * r for r in (0.18, 0.26)]
out["cost"] = cost
c = (11.28 - 2 - 0.58 * 12) / sum(t + 1 for t in range(1, 13))
def mem(passes):
    return 2 + 0.58 * len(passes) + c * sum(t + 1 for t in passes)
out["memory"] = {"c_per_unit": c, "full_Kb20": mem(range(1, 21)), "full_Kb24": mem(range(1, 25)),
                 "chunked12_Kb20_peak (per-chunk backward)": max(mem(range(1, 13)), mem(range(13, 21))),
                 "chunked12_Kb24_peak (per-chunk backward)": max(mem(range(1, 13)), mem(range(13, 25))),
                 "chunked12, single backward at the end": "no saving (= full)"}
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
