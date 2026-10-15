"""JOINT-COT-L1 calcs (CPU, light; reads D:\\Latent read-only, writes calc_jointcot.json next to this script).

1 CROWD: kappa = A's / C's median pre-clip gradient norm by phase (log.jsonl), and the Adam step shares s_T = g/sqrt(g^2 + k^2), s_S = k/sqrt(g^2 + k^2).
2 RAW-TEACHER: for a raw-init joint cell (24k, cosine), the C step t* whose integrated lr equals s_T x the joint run's integrated lr, and
  C's Aug-valid accuracy there (linear interpolation of C's valid_hist): the teacher's predicted cot_valid at 24k.
3 OC: paired SE (A vs W items, eval_items.json); P(read >= T | true effect d) with seed SD sigma_s per run (two independent runs).
4 COST: per-step and per-cell $; T2 / T1 token ratio from Aug train.
5 LORA: per-step entrywise weight change, LoRA (r 128, alpha 32, lr 3e-3, PEFT-default A init) vs full FT (Adam, lr 1e-4).
"""
import json
import math
import os
import sys

os.environ["BOXES_NB"] = "5"
ROOT = r"D:\Latent"; sys.path.insert(0, ROOT); OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "boxes", "runs_fsq")


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def med(v):
    v = sorted(v); n = len(v); return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def logrecs(d):
    return [json.loads(l_) for l_ in open(os.path.join(RUNS, d, "log.jsonl"))]


def main():
    res = {}
    A, C = logrecs("gsm_lat_aug_s1"), logrecs("gsm_txt_aug_s1")
    ph = [(1, 1000), (1000, 2000), (2000, 6000), (6000, 12000), (12000, 24001)]; kap = {}
    for lo, hi in ph:
        ga = med([r["gn_preclip"] for r in A if "gn_preclip" in r and lo <= r["step"] < hi]); gc = med([r["gn_preclip"] for r in C if "gn_preclip" in r and lo <= r["step"] < hi])
        kap["%d-%d" % (lo, hi - 1)] = {"A": ga, "C": gc, "kappa": ga / gc}
    k_all = med([v["kappa"] for v in kap.values()]); res["CROWD"] = {"by_phase": kap, "kappa_median": k_all, "shares": {}}
    for k_ in (4.0, k_all, 8.0):
        res["CROWD"]["shares"]["kappa_%.2f" % k_] = {str(g): {"s_T": g / math.hypot(g, k_), "s_S": k_ / math.hypot(g, k_)} for g in (0.5, 1, 2, 3, 5)}
    # 2 RAW-TEACHER
    vh = [(0, 0.0)] + [(r["step"], r["valid_acc"]) for r in C if "valid_acc" in r]; N = 24000

    def I(t):
        return t / 2 + N / (2 * math.pi) * math.sin(math.pi * t / N)

    def tstar(s):
        lo, hi = 0.0, float(N)
        for _ in range(60):
            mid = (lo + hi) / 2
            if I(mid) < s * I(N):
                lo = mid
            else:
                hi = mid
        return lo

    def vC(t):
        for (s0, v0), (s1, v1) in zip(vh, vh[1:]):
            if s0 <= t <= s1:
                return v0 + (v1 - v0) * (t - s0) / (s1 - s0)
        return vh[-1][1]
    rt = {}
    for k_ in (4.0, k_all, 8.0):
        for g in (1, 2, 3, 5):
            s = g / math.hypot(g, k_); t = tstar(s); rt["kappa_%.2f_gamma_%d" % (k_, g)] = {"s_T": s, "t_star": t, "cot_valid_pred": vC(t)}
    res["RAW_TEACHER"] = {"def": "C's valid at the step whose integrated cosine lr equals s_T x the 24k joint run's (gradient-flow time; warm-up ignored)", "rows": rt}
    # 3 OC
    def items(d, f="eval_items.json"):
        return [bool(x["ok"]) for x in json.load(open(os.path.join(RUNS, d, f)))]
    a, w = items("gsm_lat_aug_s1"), items("gsm_lat_warm_s1"); n = len(a); dd = [float(y) - float(x) for x, y in zip(a, w)]
    mu = sum(dd) / n; se = math.sqrt(sum((v - mu) ** 2 for v in dd) / (n - 1) / n)
    T = 0.028; TD = T * math.sqrt(1319 / 879); oc = {"paired_SE_AW": se, "discordant": int(sum(1 for v in dd if v != 0)), "n": n, "T": T, "T_D": TD, "rows": {}}
    for ss in (0.0, 0.01, 0.02):
        for se_ in (se, 0.0135):
            sig = math.sqrt(se_ ** 2 + 2 * ss ** 2)
            oc["rows"]["seedSD_%.2f_SE_%.4f" % (ss, se_)] = {str(eff): 1 - Phi((T - eff) / sig) for eff in (0.0, 0.02, 0.03, 0.04, 0.06)}
    sd_ = se * math.sqrt(1319 / 879)                                        # deep stratum (m >= 3, n 879), the same paired discordance rate
    for ss in (0.0, 0.01, 0.02):
        sig = math.sqrt(sd_ ** 2 + 2 * ss ** 2); oc["rows"]["DEEP_seedSD_%.2f" % ss] = {str(eff): 1 - Phi((TD - eff) / sig) for eff in (0.0, 0.03, 0.05)}
    res["OC"] = oc
    # 4 COST
    from boxes.prosqa168s2 import ENC                                        # noqa: E402
    import boxes.gsm8k_data as G                                            # noqa: E402
    Rtr, _ = G.load_aug("train"); import numpy as np                        # noqa: E401,E402
    idx = np.random.default_rng(7).permutation(len(Rtr))[:4000]; l1 = l2 = 0; n2 = 0
    for i in idx:
        r = Rtr[i]; pre = len(ENC.encode(r["q"] + "\n")); L = [len(ENC.encode("<<%s>>\n" % e)) for e in r["eqs"]]; a1 = len(ENC.encode("### " + r["ans"])) + 1
        l1 += pre + sum(L) + a1
        if len(L) >= 2:
            l2 += pre + sum(L[:-1]) + a1; n2 += 1
    ratio = (l2 / max(1, len(idx))) / (l1 / len(idx)); m2share = n2 / len(idx)
    sA = 43678 / 24000; sW = 13230 / 12000; sC = 6116 / 24000; sCp = 5072 / 12000
    teach = sC * (1 + ratio)                                                # T1 forward+backward (C's) + T2 (m >= 2 items, ratio x T1 tokens per item)
    cost = {"s_per_step": {"A_realised": sA, "W_realised": sW, "C_realised": sC, "Cplus_realised": sCp, "teacher_T1_plus_T2": teach, "T2_over_T1_tokens": ratio,
                           "m_ge2_share_train": m2share}, "rate_per_h": [0.21, 0.25, 0.28], "cells": {}}
    for nm, steps, base in (("warm12k_A_basis", 12000, sA), ("warm12k_W_basis", 12000, sW), ("raw24k_A_basis", 24000, sA)):
        for tag, extra in (("J0_or_J1", teach), ("no_teacher", 0.0)):
            h = steps * (base + extra) / 3600 + 0.1                             # + eval about 0.1 h (A 192 s, W 136 s)
            cost["cells"]["%s_%s" % (nm, tag)] = {"hours": h, "usd_central": h * 0.25 + 0.10, "usd_low": h * 0.21 + 0.10, "usd_high": h * 1.35 * 0.28 + 0.10}
    res["COST"] = cost
    # 5 LORA
    s = 32 / 128; lr = 3e-3; fan = 768; bound = math.sqrt(2 / (1 + 5)) * math.sqrt(3 / fan); sdA = bound / math.sqrt(3)
    res["LORA"] = {"scale": s, "A_std": sdA, "dW_per_step_random_sign": s * lr * math.sqrt(128) * sdA, "dW_per_step_coherent": s * lr * 128 * bound / 2,
                   "full_ft_adam_per_step": 1e-4, "def": "Adam moves each B entry by ~lr per step; dW = s B A summed over r with random (sqrt r) or aligned (r) signs; "
                   "A ~ kaiming_uniform(a = sqrt 5) (PEFT default, assumed for CODI's code: unverified); E|A| = bound / 2"}
    json.dump(res, open(os.path.join(OUT, "calc_jointcot.json"), "w", newline="\n"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
