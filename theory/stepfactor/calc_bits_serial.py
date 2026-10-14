"""STEP-TARGET-FACTOR calcs (CPU, no torch). (1) plug-in entropies (bits) of each factor of the step target on a 20k Aug-train sample (recomputable
items): first-operand literal token (what the string target asks the memory to identify, the aux decoder seeing no question), its pointer (slot among the
item's candidates), operator signature, value first token; (2) the serial-model K' ratio K'1 / K'10 on the GSM8K test m-mix (n_by_m from §200 A's
eval.json) under a strict per-pass serial memory with per-step fidelity rho, a direct path a_D, and an answer decoder that adds s in {0, 1, 2} last
steps itself; (3) cost. Output calc_bits_serial.json."""
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "D:/Latent")
import boxes.gsm8k_data as G  # noqa: E402
import tiktoken  # noqa: E402

ENC = tiktoken.get_encoding("gpt2"); OUT = os.path.dirname(os.path.abspath(__file__))


def H(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values())


def bits():
    R, _ = G.load_aug("train"); R = [R[i] for i in np.random.default_rng(7).permutation(len(R))[:20000]]
    c_lit, c_slot, c_sig, c_val, c_valtok, c_ncand, c_lit_q, c_lit_r, c_slot_q, c_slot_r = (Counter() for _ in range(10)); sig_all = Counter(); n_eq = 0
    for r in R:
        tpl, _ = G.parse_recomputable(r)
        if tpl is None:
            continue
        for t, eqt in enumerate(tpl["eqt"]):
            n_eq += 1; lhs = eqt["lhs"]; sig = "".join(ch for ch in lhs if ch in "+-*/%()"); c_sig[sig] += 1
            c_val[eqt["rtok"]] += 1; c_valtok[ENC.encode(eqt["rtok"])[0]] += 1
            if eqt["ops"]:
                s, e_, tok, kind, ref = eqt["ops"][0]; ft = ENC.encode(tok)[0]; c_lit[ft] += 1
                nq = len(tpl["qvals"]); ncand = nq + t + 1                  # question values + previous results + one constant class
                c_ncand[ncand] += 1
                slot = ("q%d" % ref) if kind == "q" else ("r-%d" % (t - ref)) if kind == "r" else "k"; c_slot[slot] += 1
                if kind == "q":
                    c_lit_q[ft] += 1; c_slot_q[slot] += 1
                if kind == "r":
                    c_lit_r[ft] += 1; c_slot_r[slot] += 1
    top = c_sig.most_common(16); cover16 = sum(v for _, v in top) / n_eq
    return {"n_equations": n_eq, "H_first_operand_token_bits": H(c_lit), "H_first_operand_token_q_bits": H(c_lit_q), "H_first_operand_token_r_bits": H(c_lit_r),
            "H_pointer_slot_bits": H(c_slot), "H_pointer_slot_q_bits": H(c_slot_q), "H_pointer_slot_r_bits": H(c_slot_r),
            "mean_log2_candidates": float(sum(v * math.log2(k) for k, v in c_ncand.items()) / sum(c_ncand.values())),
            "H_operator_signature_bits": H(c_sig), "n_signatures": len(c_sig), "top16_signatures": top, "top16_coverage": cover16,
            "H_value_string_bits": H(c_val), "H_value_first_token_bits": H(c_valtok), "distinct_first_operand_tokens": len(c_lit), "distinct_values": len(c_val),
            "note": "plug-in entropies on ~30k equations of recomputable items: lower bounds for large supports (values, literals); the pointer slot is r-k = k-th previous result"}


def serial(n_by_m):
    ms = {int(k): v for k, v in n_by_m.items()}; N = sum(ms.values()); out = {}
    for s in (0, 1, 2):
        for aD in (0.0, 0.022, 0.083):
            for rho in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                def acc(Kp):
                    a = 0.0
                    for m, n in ms.items():
                        need = max(0, max(m, 1) - s)                        # passes the memory must run before the decoder can finish (m = 0 as 1)
                        p = rho ** max(m, 1) if Kp >= need else 0.0
                        a += n * (p + (1 - p) * aD)
                    return a / N
                k0, k1, k10 = acc(0), acc(1), acc(10)
                out["s%d_aD%.3f_rho%.2f" % (s, aD, rho)] = {"K0": round(k0, 4), "K1": round(k1, 4), "K10": round(k10, 4), "K1_over_K10": round(k1 / k10, 3),
                                                             "THINK_NECESSARY": bool(k0 <= 0.5 * k10 and k1 <= 0.5 * k10)}
    # the same on items with m >= 3 only (THINK-NECESSARY-DEEP)
    deep = {}
    ms3 = {m: n for m, n in ms.items() if m >= 3}; N3 = sum(ms3.values())
    for s in (0, 1, 2):
        for aD in (0.0, 0.022, 0.083):
            for rho in (0.6, 0.8, 0.9):
                def acc3(Kp):
                    a = 0.0
                    for m, n in ms3.items():
                        p = rho ** m if Kp >= max(0, m - s) else 0.0; a += n * (p + (1 - p) * aD)
                    return a / N3
                deep["s%d_aD%.3f_rho%.2f" % (s, aD, rho)] = {"K1_over_K10": round(acc3(1) / acc3(10), 3), "K10": round(acc3(10), 4)}
    return {"n_by_m": n_by_m, "pooled": out, "deep_m_ge_3": deep,
            "model": "acc_m(K') = rho^m if K' >= m - s else 0, plus a direct path a_D on the remainder; s = steps the answer decoder (question in context) finishes itself"}


def cost():
    sps = 1.482; sps_A = 43678 / 24000; rate = 0.255; host = 0.10; ev = 0.05
    def c(steps, s_, hi=1.0):
        h = steps * s_ / 3600 * hi + ev; return round(h, 2), round(h * rate + host, 2)
    return {"anchor_s_per_step": sps, "A_realised_s_per_step": round(sps_A, 3), "rate_usd_h": rate,
            "cell_12k": {"central": c(12000, sps), "high_x1.35": c(12000, sps, 1.35), "at_A_realised": c(12000, sps_A)},
            "cell_24k": {"central": c(24000, sps), "high_x1.35": c(24000, sps, 1.35), "at_A_realised": c(24000, sps_A)}}


if __name__ == "__main__":
    nbm = json.load(open("D:/Latent/boxes/runs_fsq/gsm_lat_aug_s1/eval.json"))["n_by_m"]
    res = {"bits": bits(), "serial": serial(nbm), "cost": cost()}
    json.dump(res, open(os.path.join(OUT, "calc_bits_serial.json"), "w", newline="\n"), indent=1)
    print(json.dumps(res["bits"], indent=0)[:2500]); print(json.dumps(res["cost"]))
    for k in ("s0_aD0.000_rho0.80", "s1_aD0.000_rho0.80", "s1_aD0.022_rho0.80", "s1_aD0.083_rho0.80", "s0_aD0.083_rho0.80", "s1_aD0.083_rho0.90", "s2_aD0.022_rho0.80", "s1_aD0.022_rho0.60"):
        print(k, res["serial"]["pooled"][k], res["serial"]["deep_m_ge_3"].get(k))
