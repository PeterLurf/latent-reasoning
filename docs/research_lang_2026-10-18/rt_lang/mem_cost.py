"""REDTEAM_LANG $0: memory / speed / labelling-time cross-check of the LANG draft's §9 (its cost_model.py counts attention entries and
token-layers only). Assumptions [ASSUMED, stated]:
  - training under bf16 autocast (vbank.py line 630: amp = bf16 on capability >= 8), SDPA everywhere (deltamem._block is_causal -> flash /
    mem-efficient, no T^2 save in the prelude); the bank attention has a float mask that depends on beta / gamma, so it is saved:
    ~4 B per (query, key, head) entry (mask + probs, bf16) as an UPPER figure, 2 B as a LOWER figure;
  - saved activations per token-layer of a GPT-2 block under autocast ~ 38 * d bytes (2 fp32 LN inputs + ~15 d bf16 tensors);
  - FLOPs per token-layer fwd ~ 24 d^2 + 4 T_k d; fwd+bwd = 3x; 3090 sustained 20 TFLOP/s (bf16, fp32 accumulate), local 3070 / 4060
    fp32 eval ~ 8 TFLOP/s; GPU $0.21 / h (3090 fleet rate, §210-1 / 193-cost).
  - nested D3-T reference measured 0.42 s / step (draft §9) -> compute share check.
Token-layer counts and |S|, T, K follow the draft's cost_model.py rows; the token count per word is MEASURED below with tiktoken on
sym_proto draws (names are the prototype's NONCE-style words).
    python -B mem_cost.py
"""
import json
import random
import sys

import tiktoken

sys.path.insert(0, "D:/Latent/docs/research_lang_2026-10-18/scripts")
import sym_proto as SP  # noqa: E402

d = 768; TFL = 20e12; TFL_LOCAL = 8e12; USD_H = 0.21
enc = tiktoken.get_encoding("gpt2"); SP.NAMES = SP.name_pool(random.Random(12345)); rng = random.Random(4242)
tok = {}
for D in (8, 12, 16):
    T = []
    for _ in range(300):
        x, ok, lv, adj = SP.make(rng, D); T.append(len(enc.encode(x["question"])) + 1)
    T.sort(); tok[D] = {"mean": round(sum(T) / len(T)), "p95": T[int(0.95 * len(T))], "max": T[-1]}
print("GPT-2 tokens of SYM prototype text (no cap):", tok)


def cell(name, Tq, S, K, Ldec, nwin, bs=64):
    pre_tl = 6 * Tq; core_tl = 4 * S * K; coda_tl = 2 * (S + Ldec) * nwin; dec_tl = 10 * Ldec
    tl = pre_tl + core_tl + coda_tl + dec_tl
    keys_core = sum(Tq + (t - 1) * S + S for t in range(1, K + 1))          # keys per core query summed over passes
    flops = 3 * (tl * 24 * d * d + 4 * d * (pre_tl * Tq / 2 / 6 * 6 / 6 + 4 * S * keys_core))
    act = tl * 38 * d                                                       # bytes saved per item
    att_lo, att_hi = [12 * 4 * S * keys_core * b for b in (2, 4)]
    gb_lo, gb_hi = (act + att_lo) * bs / 1e9, (act + att_hi) * bs / 1e9
    s_step = flops * bs / TFL
    return {"cell": name, "token_layers": tl, "GFLOP_per_item": round(flops / 1e9, 1), "act_GB_bs64": round(act * bs / 1e9, 1),
            "bank_att_GB_bs64": [round(att_lo * bs / 1e9, 1), round(att_hi * bs / 1e9, 1)], "total_GB_bs64": [round(gb_lo, 1), round(gb_hi, 1)],
            "compute_s_per_step": round(s_step, 2)}


rows = [cell("nested_D3T_ref", 34, 34, 12, 1, 5), cell("A1_SYM_D8_mean", tok[8]["mean"], 50, 12, 8, 3), cell("A1_SYM_D8_worst", tok[8]["max"], 75, 12, 8, 5),
        cell("A2_self_D12_mean", min(tok[12]["mean"], 1000), 64, 16, 8, 3), cell("A2_self_D12_worst(T1000,|S|80,K16)", 1000, 80, 16, 8, 5),
        cell("B1_GSM_mean", 77, 12, 16, 10, 8), cell("B1_GSM_worst", 194, 12, 16, 10, 16)]
ref = rows[0]["compute_s_per_step"]; scale = 0.42 / ref
for r in rows:
    r["s_per_step_scaled_to_D3T"] = round(max(0.42, r["compute_s_per_step"] * scale), 2)
    print(r)
proj = {"A1_24k_mean_$": round(24000 * rows[1]["s_per_step_scaled_to_D3T"] / 3600 * USD_H, 2),
        "A1_24k_forced_worst_$": round(24000 * rows[2]["s_per_step_scaled_to_D3T"] / 3600 * USD_H, 2),
        "A2_12k_mix_$": round(12000 * (0.5 * rows[3]["s_per_step_scaled_to_D3T"] + 0.5 * rows[1]["s_per_step_scaled_to_D3T"]) / 3600 * USD_H, 2),
        "A2_12k_forced_worst_$": round(12000 * rows[4]["s_per_step_scaled_to_D3T"] / 3600 * USD_H, 2)}
print("projections (no micro-batch overhead):", proj)
# labelling: L-RR, about s steps of a 2-pass rollout on a re-rendered input (prelude over T + 2 core passes + no coda), fp32 local eval
lab = {}
for yld in (0.2, 0.3, 0.5):
    kept_needed = 0.5 * 12000 * 64 * 1.10                                   # A2 self demand + 10% margin
    drawn = kept_needed / yld; steps_mean = 8                               # early rejection: mean L-RR steps per drawn item [ASSUMED]
    fl = drawn * steps_mean * (6 * 650 * 24 * d * d + 4 * 64 * 2 * 24 * d * d)
    lab[yld] = {"kept_needed": int(kept_needed), "drawn": int(drawn), "local_GPU_h_fp32": round(fl / TFL_LOCAL / 3600, 1)}
print("self-pool labelling (local, fp32):", lab)
json.dump({"tokens": tok, "cells": rows, "proj": proj, "labelling": lab}, open("mem_cost.json", "w", newline="\n"), indent=1)
