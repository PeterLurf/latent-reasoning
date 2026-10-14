"""GSM-TEMPLATE draft, measurement 6: the cost model (block-token units calibrated on the realised §191 A run and the nested RMS run) and the OC block
(seedstats.py read-only, plus a normal model of seed + eval noise).   python -I oc_cost.py"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "genlat", "tools"))
import seedstats as SS  # noqa: E402

PRICE = 0.227; HOST = 0.10                                                  # $/h (the §191 host class), $ per host (G3)


def units_latent(L, K, m, aux_len, ans_extra, cached=True, ckpt=False, mem_layer=6):
    """forward block-token units per item: lower blocks once (cached) or per pass, upper blocks per pass, aux decoders (12 blocks, t <= m), answer decoder."""
    up = 12 - mem_layer
    f = (mem_layer * L + K * up * L) if cached else K * 12 * L
    f += m * 12 * aux_len + 12 * (L + ans_extra)
    train = 3 * f + (K * up * L if ckpt else 0)                              # fwd + bwd ~ 3x fwd; checkpointing recomputes each pass's forward once
    return f, train


def main():
    out = {}
    # anchor P: §191 A (ProsQA, realised): K 10, cached, ckpt_passes, chunk_delta 64, micro 4, bs 64; mean train think L 439.96, mean depth 6.55, aux 4 + 9.75 + 1
    fP, tP = units_latent(439.96, 10, 6.55, 14.75, 12, cached=True, ckpt=True)
    sP = 83823 / 12000                                                      # realised s/step (train_sec / steps)
    # anchor N: nested RMS s1 (realised 42,491 s / 48k): K 8, uncached, no ckpt, micro 1; L 26.4 (readslots.L_mean), mean n 4.5, aux ~ 4 + 12 + 1 (assumed)
    fN, tN = units_latent(26.4, 8, 4.5, 17, 3, cached=False, ckpt=False)
    sN = 42491 / 48000
    us = sP / (64 * tP)                                                     # s per training unit (compute-bound anchor: bs 32 -> 3.50 s, bs 64 -> 6.96 s)
    ovh = sN - 64 * tN * us                                                 # per-step overhead not explained by units at nested scale (launch-bound)
    out["calibration"] = {"P_units_train_per_item": round(tP), "P_s_per_step": round(sP, 3), "s_per_unit": us, "N_units_train_per_item": round(tN),
                          "N_s_per_step": round(sN, 3), "N_pred_from_units": round(64 * tN * us, 3), "overhead_s_per_step": round(ovh, 3)}
    # GSM8K-Aug latent cell: L 43.34 (Aug train mean, incl. EOT), mean m 2.598, aux tokens 7.82 + 4 prompt + 1 EOT, answer " ####" + ~3 + EOS
    cells = {}
    for name, K, ckpt in (("LAT K10 cached", 10, False), ("LAT K10 cached+ckpt", 10, True), ("LAT WINDOW mean-batch-max K~10", 10, False), ("LAT K8 cached", 8, False)):
        f, t = units_latent(43.34, K, 2.598, 12.82, 5, cached=True, ckpt=ckpt)
        lo = 64 * t * us; mid = lo + ovh; hi = mid * 1.5
        cells[name] = {"units_train_per_item": round(t), "s_step_compute_only": round(lo, 3), "s_step_central": round(mid, 3), "s_step_high": round(hi, 3)}
    # TEXT-COT: GPT-2 on question + eq-format CoT + answer, mean 66.92 tokens; overhead for a plain 12-block model ~ 1/3 of the loop's (assumed)
    tT = 3 * 12 * 66.92; loT = 64 * tT * us; cells["TXT"] = {"units_train_per_item": round(tT), "s_step_compute_only": round(loT, 3),
                                                             "s_step_central": round(loT + ovh / 3, 3), "s_step_high": round((loT + ovh / 3) * 1.5, 3)}
    # check against the measured ProsQA TEXT-COT (seq 518.9): predicted vs realised 1.348 s/step
    tTP = 3 * 12 * 518.93; cells["_check_TXT_prosqa"] = {"pred_s_step": round(64 * tTP * us + ovh / 3, 3), "realised": round(16180 / 12000, 3)}
    out["s_per_step"] = cells
    # per-cell $ (projection; replaced by realised $ once each cell lands, 198-2a)
    def dollars(s_step, steps, eval_h):
        return round((s_step * steps / 3600 + eval_h) * PRICE + HOST, 2)
    plan = {}
    for steps in (12000, 24000, 48000):
        c = cells["LAT K10 cached"]; t = cells["TXT"]
        plan[steps] = {"LAT_h_central": round(c["s_step_central"] * steps / 3600, 2), "LAT_$": [dollars(c["s_step_central"], steps, 0.4), dollars(c["s_step_high"], steps, 0.6)],
                       "TXT_h_central": round(t["s_step_central"] * steps / 3600, 2), "TXT_$": [dollars(t["s_step_central"], steps, 0.3), dollars(t["s_step_high"], steps, 0.5)]}
    out["per_cell_dollars_central_high"] = plan
    out["anchor_realised_dollars_s191A"] = round(83823 / 3600 * PRICE, 2)
    # item-level SE on 1,319
    n = 1319
    out["item_se"] = {p: round(math.sqrt(p * (1 - p) / n) * 100, 2) for p in (0.42, 0.45, 0.50)}
    out["paired_diff_2se_pts"] = {d: round(2 * math.sqrt(d / n) * 100, 2) for d in (0.15, 0.20, 0.25, 0.30, 0.35)}
    # seed-level OC for "above PCCoT" (per seed: acc >= 49.48 + 2.8), all-of-N, stop at first failure
    Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
    bar = 49.48 + 2.8; se_eval = math.sqrt(0.5 * 0.5 / n) * 100
    oc = {}
    for sd in (0.5, 1.5):
        rows = {}
        for mu in (49.48, 51.0, 52.28, 53.5, 55.0):
            p = Phi((mu - bar) / math.sqrt(sd ** 2 + se_eval ** 2))
            rows[mu] = {"p_seed_above": round(p, 4), "P_claim_N2": round(p ** 2, 4), "P_claim_N3": round(p ** 3, 4),
                        "E_seeds_N3_stop_first_fail": round((1 - p ** 3) / (1 - p) if p < 1 else 3, 3)}
        oc["seed_sd_%.1f" % sd] = rows
    out["oc_above_pccot"] = oc
    # seed-rate framing via seedstats (exact): 'above PCCoT' as Bernoulli(p) per seed
    out["seedstats_allofN"] = {N: {"size_at_p0.2": round(0.2 ** N, 4), "size_at_p0.5": round(0.5 ** N, 4), "power_at_p0.9": round(0.9 ** N, 4)} for N in (2, 3)}
    out["seedstats_exact_ci_2of2"] = [round(x, 3) for x in SS.exact_ci(2, 2)]
    out["seedstats_exact_ci_3of3"] = [round(x, 3) for x in SS.exact_ci(3, 3)]
    json.dump(out, open(os.path.join(HERE, "oc_cost.json"), "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
