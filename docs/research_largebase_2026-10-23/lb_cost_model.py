"""LB draft check B ($0, CPU, pure Python): memory and cost model for INJECT / DIRECT / TEXT on candidate bases, calibrated on §228 / §229's
measured GPT-2 small numbers. An F2 measurement on the 3090 replaces every projection here before any admission (house rule).

Measured inputs (field names quoted; files under boxes/runs_fsq/ and scratch/gate_s228/):
  M1 s228_plat2_s1_mem/mem_check.json  peak_gib_train 18.758  (micro 1 = 64 items per micro, d_max 2, K 8, value target; 3090)
  M2 scratch/gate_s228/f2.json lat_d2  peak_gib_train  6.232  (micro 4 = 16 items per micro, d_max 2; 4060)
  M3 scratch/gate_s228/f2.json lat_d4  peak_gib_train  5.229  (micro 8 = 8 items per micro, d_max 4; 4060)        <- validation only
  M4 s229_plat2c_s1_mem/mem_check.json peak_gib_train 20.192  (micro 1, d_max 2, cumulative target; 3090)          <- validation only
  T1 s228_plat2_s1/train_summary.json  train_sec 5740 / last_step 8000 (inject, micro 1, 3090)
  T2 s228_pdir2c_s1/train_summary.json train_sec  617 / last_step 8000 (direct, micro 1, 3090)
  T3 s229_plat2c_s1/train_summary.json train_sec 7398 / last_step 8000 (inject, cumulative target, 3090)          <- validation only
Model (per item, bytes saved for backward; one calibration factor c fitted on M1 - M2, then checked on M3 / M4):
  trunk  : token-layers x s_layer, s_layer = 2 (12 D + 2 (D + 2 D_kv) + n_I I)  (n_I = 2 GELU MLP, 3 SwiGLU)
           token-layers = L_m T (lower, cached once) + (L - L_m) T K (upper, K passes) + d L T_aux (aux, d passes) + L T_dec (decoder)
  read   : 16 D per token read  (K T + d T_aux + T_dec reads)
  write  : per pass T (20 D + 3 h_del 64^2 4)  (fp32 LN / k / v + three saved delta-state tensors per token)
  logits : 10 V per logit position (aux target positions + 1 decoder position)
Time per step = FLOPs / R_eff (R_eff from T2, DIRECT) + n_micro x c_loop x K x T_pad (c_loop from T1's residual; the launch-bound delta loop).
Usage: python -B lb_cost_model.py [--tok lb_tok_check.json] [--out lb_cost.json]
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GiB = 2.0 ** 30
PRICE_3090 = 0.22            # $/h (Vast RTX 3090, the house rate)
GUARD = 1.3                  # F2 x 1.3 admission rule
K = 8                        # §184 / §228 think_k
BS = 64
BUDGET_GIB = 21.0            # inject_ow --mem_budget default (3090)
CTX_GIB = 0.17               # CUDA context + workspace: M2's fixed part minus GPT-2 small's 16 B / param (fitted below, reported)
CHUNK_SPEEDUP = 8.1          # Note 168-11: CHUNK_DELTA ran the delta loop at 8.1x its speed on the same 3090 (applied to the loop term only; memory unchanged)
SFX = 4                      # " The answer is:" (GPT-2: 4 tokens; Qwen / Llama: 4 + 1 separate space token, added per base below)

# base specs (configs downloaded 2026-10-24 from the HF hub; GPT-2 sizes from openai-community/gpt2-*/config.json)
BASES = {
    "gpt2":         dict(L=12, D=768,  H=12, KV=12, I=3072, V=50257,  n_I=2, emb=50257 * 768 + 1024 * 768,    tok="gpt2", sfx_extra=0),
    "gpt2_medium":  dict(L=24, D=1024, H=16, KV=16, I=4096, V=50257,  n_I=2, emb=50257 * 1024 + 1024 * 1024,  tok="gpt2", sfx_extra=0),
    "gpt2_large":   dict(L=36, D=1280, H=20, KV=20, I=5120, V=50257,  n_I=2, emb=50257 * 1280 + 1024 * 1280,  tok="gpt2", sfx_extra=0),
    "qwen25_05b":   dict(L=24, D=896,  H=14, KV=2,  I=4864, V=151936, n_I=3, emb=151936 * 896,               tok="qwen25_05b", sfx_extra=1),
    "llama32_1b":   dict(L=16, D=2048, H=32, KV=8,  I=8192, V=128256, n_I=3, emb=128256 * 2048,              tok="llama32_1b", sfx_extra=1),
}


def layer_params(b):
    D, I, hd = b["D"], b["I"], b["D"] // b["H"]; Dkv = b["KV"] * hd
    attn = D * (D + 2 * Dkv) + (D + 2 * Dkv) + D * D + (D if b["n_I"] == 2 else 0)
    mlp = (2 if b["n_I"] == 2 else 3) * D * I + ((I + D) if b["n_I"] == 2 else 0)
    norms = (4 if b["n_I"] == 2 else 2) * D
    return attn + mlp + norms


def n_params(b):
    return b["L"] * layer_params(b) + b["emb"] + 2 * b["D"]


def mem_params(b):
    D = b["D"]; return D * 3 * D + 3 * D + D * D + D + D * 2 * (D // 128) + 2 * D + 4 * D + 2


def s_layer(b):
    D, I, hd = b["D"], b["I"], b["D"] // b["H"]; Dkv = b["KV"] * hd
    return 2 * (12 * D + 2 * (D + 2 * Dkv) + b["n_I"] * I)


def lengths(tok, base, d, kind):
    """kind 'max' (memory: padded to the longest of 256) or 'pad' (time: per-step batch max ~ the max of 256, slight over-estimate)."""
    r = tok["bases"][base["tok"]]["len"][str(d)]
    T = r["think_max"]; Ta_v = r["value_tgt_max"]; Ta_c = r["cum_tgt_max"]; Tc = r["text_cot_max"]
    return T, Ta_v, Ta_c, Tc


def inject_item(b, T, Ta, d, ckpt=False):
    """-> (bytes analytic per item, token-layer count, logit positions)."""
    L, D = b["L"], b["D"]; Lm = L // 2; hdel = D // 128; Td = T + SFX + b["sfx_extra"]; Tap = Ta + 4 + b["sfx_extra"]   # aux prompt [BOS, Step, (sp), t, :]
    up = (L - Lm) * T * (1 if ckpt else K)
    tl = Lm * T + up + d * L * Tap + L * Td
    reads = (1 if ckpt else K) * T + d * Tap + Td
    write = (1 if ckpt else K) * T * (20 * D + 3 * hdel * 64 * 64 * 4)
    states = (K * (2 * hdel) * 64 * 64 * 4) if ckpt else 0
    lpos = d * Ta + 1
    byts = tl * s_layer(b) + reads * 16 * D + write + lpos * 10 * b["V"] + states
    return byts, tl, lpos


def direct_item(b, T):
    Td = T + SFX + b["sfx_extra"]; return b["L"] * Td * s_layer(b) + 10 * b["V"], b["L"] * Td, 1


def text_item(b, Tc, Tans):
    return b["L"] * Tc * s_layer(b) + Tans * 10 * b["V"], b["L"] * Tc, Tans


def fixed_gib(b, latent):
    return (16 * n_params(b) + (16 * mem_params(b) if latent else 0)) / GiB + CTX_GIB


def pick_micro(fixed, per_item, budget=BUDGET_GIB):
    for micro in (1, 2, 4, 8, 16, 32, 64):
        if fixed + (BS // micro) * per_item <= budget:
            return micro, fixed + (BS // micro) * per_item
    return None, None


def flops_inject(b, T, Ta, d, ckpt=False):
    L = b["L"]; Lm = L // 2; Td = T + SFX + b["sfx_extra"]; Tap = Ta + 4 + b["sfx_extra"]; P = layer_params(b); D = b["D"]
    tl = Lm * T + (L - Lm) * T * K + d * L * Tap + L * Td
    f = 6 * P * tl + 6 * D * b["V"] * (d * Ta + 1) + 6 * 4 * D * D * (K * T * 2 + d * Tap + Td)    # trunk + logits + memory projections
    if ckpt:
        f += 2 * P * (L - Lm) * T * K + 2 * 4 * D * D * K * T                                     # recompute of each pass in backward
    return BS * f


def flops_direct(b, T):
    Td = T + SFX + b["sfx_extra"]; return BS * (6 * layer_params(b) * b["L"] * Td + 6 * b["D"] * b["V"])


def flops_text(b, Tc, Tans):
    return BS * (6 * layer_params(b) * b["L"] * Tc + 6 * b["D"] * b["V"] * Tans)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--tok", default=os.path.join(HERE, "lb_tok_check.json")); ap.add_argument("--out", default=os.path.join(HERE, "lb_cost.json"))
    a = ap.parse_args(argv); tok = json.load(open(a.tok)); g = BASES["gpt2"]; res = {"inputs": {}, "calibration": {}, "bases": {}}

    # ---- memory calibration (M1, M2) and checks (M3, M4)
    M1, M2, M3, M4 = 18.758, 6.232, 5.229, 20.192
    per_item_meas = (M1 - M2) / (64 - 16); fixed_meas = M2 - 16 * per_item_meas
    T2, Tv2, Tc2, _ = lengths(tok, g, 2, "max"); T4, Tv4, _, _ = lengths(tok, g, 4, "max")
    an2 = inject_item(g, T2, Tv2, 2)[0] / GiB; c = per_item_meas / an2
    pred_M3 = fixed_meas + 8 * c * inject_item(g, T4, Tv4, 4)[0] / GiB; pred_M4 = fixed_meas + 64 * c * inject_item(g, T2, Tc2, 2)[0] / GiB
    global CTX_GIB
    CTX_GIB = fixed_meas - (16 * n_params(g) + 16 * mem_params(g)) / GiB
    res["calibration"]["memory"] = {"per_item_gib_measured_d2": round(per_item_meas, 4), "fixed_gib_measured": round(fixed_meas, 3), "analytic_per_item_gib_d2": round(an2, 4),
                                    "c": round(c, 3), "ctx_gib": round(CTX_GIB, 3), "gpt2_params": n_params(g),
                                    "check_M3_d4_micro8": {"measured": M3, "predicted": round(pred_M3, 3)}, "check_M4_cum_micro1": {"measured": M4, "predicted": round(pred_M4, 3)}}

    # ---- time calibration (T1 inject, T2 direct) and check (T3)
    s_inj, s_dir, s_cum = 5740 / 8000, 617 / 8000, 7398 / 8000
    def avg_over(dmax, fn):
        return sum(fn(d) for d in range(1, dmax + 1)) / dmax
    fd = avg_over(2, lambda d: flops_direct(g, lengths(tok, g, d, "pad")[0])); R_eff = fd / s_dir
    fi = avg_over(2, lambda d: flops_inject(g, lengths(tok, g, d, "pad")[0], lengths(tok, g, d, "pad")[1], d))
    loopT = avg_over(2, lambda d: K * lengths(tok, g, d, "pad")[0]); c_loop = (s_inj - fi / R_eff) / loopT
    fc = avg_over(2, lambda d: flops_inject(g, lengths(tok, g, d, "pad")[0], lengths(tok, g, d, "pad")[2], d))
    pred_cum = fc / R_eff + c_loop * loopT
    res["calibration"]["time"] = {"R_eff_tflops_3090": round(R_eff / 1e12, 2), "c_loop_ms_per_iter": round(c_loop * 1e3, 3), "inject_compute_share": round(fi / R_eff / s_inj, 3),
                                  "check_T3_cum": {"measured_s_per_step": round(s_cum, 4), "predicted": round(pred_cum, 4)}}

    # ---- per base
    for name, b in BASES.items():
        out = {"n_params": n_params(b), "layer_params": layer_params(b), "mem_params": mem_params(b), "s_layer_bytes": s_layer(b),
               "fullft_states_gib": round(16 * n_params(b) / GiB, 2), "fits_8gb_fullft_states": 16 * n_params(b) / GiB + CTX_GIB < 7.5, "cells": {}}
        for dmax, steps, tag in ((2, 8000, "pilot_d2_8k"), (4, 16000, "pilot_d4_16k"), (8, 48000, "stage1_d8_48k"), (6, 36000, "stage1_d6_36k"), (5, 30000, "stage1_d5_30k")):
            Tm, Tvm, _, _ = lengths(tok, b, dmax, "max")
            row = {}
            for ckpt, chunk in ((False, False), (True, False), (False, True), (True, True)):
                key = "inject" + ("_ckpt" if ckpt else "") + ("_chunk" if chunk else "")
                fx = fixed_gib(b, True); pi = c * inject_item(b, Tm, Tvm, dmax, ckpt)[0] / GiB; micro, peak = pick_micro(fx, pi)
                if micro is None:
                    row[key] = {"fixed_gib": round(fx, 2), "per_item_gib": round(pi, 3), "fits_3090": False}; continue
                comp = avg_over(dmax, lambda d: flops_inject(b, lengths(tok, b, d, "pad")[0], lengths(tok, b, d, "pad")[1], d, ckpt)) / R_eff
                loop = micro * c_loop * avg_over(dmax, lambda d: K * lengths(tok, b, d, "pad")[0]) * (1.5 if ckpt else 1.0) / (CHUNK_SPEEDUP if chunk else 1.0)
                sps = comp + loop; hours = sps * steps / 3600 * 1.15                                        # +15 %: eval parts + setup (§228: $0.58 for 1.77 h train)
                row[key] = {"fixed_gib": round(fx, 2), "per_item_gib": round(pi, 3), "micro": micro, "peak_gib": round(peak, 2), "fits_3090": True,
                                                            "s_per_step": round(sps, 3), "compute_s": round(comp, 3), "loop_s": round(loop, 3), "hours": round(hours, 2),
                                                            "usd": round(hours * PRICE_3090, 2), "usd_guard": round(hours * PRICE_3090 * GUARD, 2), "usd_2x_host": round(2 * hours * PRICE_3090, 2)}
            fx = fixed_gib(b, False); pi = c * direct_item(b, Tm)[0] / GiB; micro, peak = pick_micro(fx, pi)
            if micro is not None:
                sps = avg_over(dmax, lambda d: flops_direct(b, lengths(tok, b, d, "pad")[0])) / R_eff; hours = sps * steps / 3600 * 1.15
                row["direct"] = {"micro": micro, "peak_gib": round(peak, 2), "s_per_step": round(sps, 3), "hours": round(hours, 2), "usd": round(hours * PRICE_3090, 2), "usd_guard": round(hours * PRICE_3090 * GUARD, 2)}
            else:
                row["direct"] = {"fits_3090": False, "fixed_gib": round(fx, 2)}
            out["cells"][tag] = row
        # TEXT (§227 recipe repeated: CurGate, stop 18k); stage mix ~ d U{1..5} as in §227 (stages 1-5 by 16k)
        Tcm = lengths(tok, b, 6, "max")[3]; fx = fixed_gib(b, False); pi = c * text_item(b, Tcm, Tcm // 2)[0] / GiB; micro, peak = pick_micro(fx, pi)
        sps = avg_over(5, lambda d: flops_text(b, tok["bases"][b["tok"]]["len"][str(d)]["text_cot_mean"] * 1.15, tok["bases"][b["tok"]]["len"][str(d)]["text_cot_mean"] * 0.5)) / R_eff
        dec = 18 * 200 * 8 * 0.02 / 3600                                                                  # 18 decode evals x 200 items x <= 8 d; ~20 ms / item-batch (generous)
        hours = sps * 18000 / 3600 * 1.15 + dec
        out["cells"]["text_s227_18k"] = {"micro": micro, "peak_gib": None if peak is None else round(peak, 2), "s_per_step": round(sps, 3), "hours": round(hours, 2),
                                         "usd": round(hours * PRICE_3090, 2), "usd_guard": round(hours * PRICE_3090 * GUARD, 2)}
        res["bases"][name] = out
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res["calibration"], indent=1))
    for name, o in res["bases"].items():
        print("== %s  params %.1fM  full-FT states %.2f GiB (fits 8 GB: %s)" % (name, o["n_params"] / 1e6, o["fullft_states_gib"], o["fits_8gb_fullft_states"]))
        for tag, row in o["cells"].items():
            print("   %-15s %s" % (tag, json.dumps(row)))


if __name__ == "__main__":
    main()
