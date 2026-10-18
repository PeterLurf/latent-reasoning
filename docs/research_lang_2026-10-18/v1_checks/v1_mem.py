"""LANG v1 $0: training memory (T^2 K bank model), micro-batch choice, s/step, $ and admission / guard proposals per cell, from the
v1 generator's measured shapes. The memory / FLOP model is RT's rt_lang/mem_cost.py cell() (copied, unchanged in form) [ASSUMED there:
bf16 autocast, SDPA prelude, saved bank mask + probs 2-4 B per entry, 38 d bytes per token-layer, 20 TFLOP/s 3090 sustained, calibrated to
D3-T's measured 0.42 s/step]. New here: shapes are BATCH MAXIMA of 64-item batches drawn from v1_sym.make (padding), micro-batching
(the smallest of {1, 2, 4, 8} with peak <= 21 GiB incl. ~2.5 GiB of weights + AdamW state), a +4 % overhead per extra micro-batch
[ASSUMED], and a price band $0.17-0.26 / h ($0.21 central).
    python -B v1_mem.py   -> v1_mem.json
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
import v1_sym as V  # noqa: E402

V.LEAD = " "
P = {"dlo": 0, "dhi": 0, "klo": "half", "rmax": 4, "cap": (lambda D: None if D < 9 else 2 * D + 6), "jmax": 2, "leaves": "last",
     "pextra": 0.1, "toplev": 0, "mlo": 1, "mhi": 1, "rr": 4, "rr_small": 5}
d = 768; TFL = 20e12; FIXED_GB = 2.5; BUDGET_GB = 21.0; USD = (0.17, 0.21, 0.26)


def batch_shapes(Ds, nb=120, bs=64, seed=0):
    rng = random.Random(seed); Tm = []; Sm = []; Tmean = []; Smean = []
    for _ in range(nb):
        its = [V.make(rng, rng.choice(Ds), P) for _ in range(bs)]
        T = [x["T"] for x in its]; S = [x["n_concepts"] for x in its]
        Tm.append(max(T)); Sm.append(max(S)); Tmean.append(sum(T) / bs); Smean.append(sum(S) / bs)
    Tm.sort(); Sm.sort()
    return {"T_pad_mean": round(sum(Tm) / nb), "T_pad_max": Tm[-1], "S_pad_mean": round(sum(Sm) / nb), "S_pad_max": Sm[-1],
            "T_item_mean": round(sum(Tmean) / nb), "S_item_mean": round(sum(Smean) / nb)}


def cell(Tq, S, K, Ldec, nwin, bs=64, direct=False):
    if direct:                                                      # plain GPT-2, 12 blocks over the question + decoder string
        tl = 12 * (Tq + Ldec); flops = 3 * tl * (24 * d * d + 2 * (Tq + Ldec) * d); act = tl * 38 * d; att = (0, 0)
    else:
        pre_tl = 6 * Tq; core_tl = 4 * S * K; coda_tl = 2 * (S + Ldec) * nwin; dec_tl = 10 * Ldec
        tl = pre_tl + core_tl + coda_tl + dec_tl
        keys_core = sum(Tq + (t - 1) * S + S for t in range(1, K + 1))
        flops = 3 * (tl * 24 * d * d + 4 * d * (pre_tl * Tq / 2 + 4 * S * keys_core))
        act = tl * 38 * d; att = tuple(12 * 4 * S * keys_core * b for b in (2, 4))
    gb = [(act + a) * bs / 2 ** 30 for a in att]
    s_step = max(0.42, flops * bs / TFL * SCALE) if SCALE else flops * bs / TFL
    return gb, s_step


def micro_for(gb_hi):
    for m in (1, 2, 4, 8):
        if gb_hi / m + FIXED_GB <= BUDGET_GB:
            return m
    return None


SCALE = None
REF = cell(34, 34, 12, 1, 5)[1]; SCALE = 0.42 / REF                 # calibrate on D3-T's measured 0.42 s/step (nested T 34, |S| 34, K 12)
out = {"shapes": {}, "cells": {}, "ref_compute_s": round(REF, 3), "scale": round(SCALE, 3)}
sh = {"A1_final": batch_shapes(list(range(3, 9))), "A2_self": batch_shapes(list(range(9, 13)), seed=1),
      "eval_D16": batch_shapes([16], nb=40, seed=2), "eval_D8": batch_shapes([8], nb=40, seed=3)}
out["shapes"] = sh
rows = {}
# A1: 24k steps; final-stage shapes (earlier curriculum stages are smaller: a conservative projection); K 12 per batch; Ldec 5; R1 window mean 3
for name, shp, K, nwin, steps in (("A1", sh["A1_final"], 12, 3, 24000), ("A2_plain_steps", sh["A1_final"], 12, 3, 6008),
                                   ("A2_self_steps", sh["A2_self"], 16, 3, 5992), ("A2CTRL", sh["A1_final"], 12, 3, 12000)):
    gb_m, s_m = cell(shp["T_pad_mean"], shp["S_pad_mean"], K, 5, nwin)
    gb_w, s_w = cell(shp["T_pad_max"], shp["S_pad_max"], K, 5, 5)
    mic = micro_for(gb_w[1]); ov = 1 + 0.04 * (mic - 1)
    rows[name] = {"steps": steps, "GB_bs64_mean_shape": [round(g, 1) for g in gb_m], "GB_bs64_worst_shape": [round(g, 1) for g in gb_w],
                  "micro": mic, "s_step_mean": round(s_m * ov, 2), "s_step_forced_worst": round(s_w * ov, 2),
                  "hours_mean": round(steps * s_m * ov / 3600, 2), "hours_forced": round(steps * s_w * ov / 3600, 2)}
# DIRECT-SYM: plain GPT-2, K = 1, the A1 stream (final-stage law), 24k x 64
gb_m, s_m = cell(sh["A1_final"]["T_pad_mean"], 0, 1, 5, 1, direct=True); gb_w, s_w = cell(sh["A1_final"]["T_pad_max"], 0, 1, 5, 1, direct=True)
mic = micro_for(gb_w[1]); ov = 1 + 0.04 * (mic - 1)
rows["DIRECT_SYM"] = {"steps": 24000, "GB_bs64_mean_shape": [round(gb_m[0], 1)], "GB_bs64_worst_shape": [round(gb_w[0], 1)], "micro": mic,
                      "s_step_mean": round(s_m * ov, 2), "s_step_forced_worst": round(s_w * ov, 2),
                      "hours_mean": round(24000 * s_m * ov / 3600, 2), "hours_forced": round(24000 * s_w * ov / 3600, 2)}
# A-TXT (optional): TEXT-COT on SYM, the question + a path CoT of ~ 8 tokens per hop (D <= 8): 12k x 64
cot = 8 * 8
gb_m, s_m = cell(sh["A1_final"]["T_pad_mean"] + cot, 0, 1, 5, 1, direct=True); gb_w, s_w = cell(sh["A1_final"]["T_pad_max"] + cot, 0, 1, 5, 1, direct=True)
mic = micro_for(gb_w[0]); ov = 1 + 0.04 * (mic - 1)
rows["A_TXT"] = {"steps": 12000, "micro": mic, "s_step_mean": round(s_m * ov, 2), "s_step_forced_worst": round(s_w * ov, 2),
                 "hours_mean": round(12000 * s_m * ov / 3600, 2), "hours_forced": round(12000 * s_w * ov / 3600, 2)}
# B1: GSM8K; launch-bound floor scales with max K per batch (16 vs D3-T's 12) -> 0.42 * 16/12 = 0.56 s, up to RT's 0.70 [ASSUMED]
gb_m, s_m = cell(77, 12, 16, 10, 8); gb_w, s_w = cell(194, 12, 16, 10, 16)
rows["B1"] = {"steps": 120000, "GB_bs64_mean_shape": [round(g, 1) for g in gb_m], "GB_bs64_worst_shape": [round(g, 1) for g in gb_w], "micro": micro_for(gb_w[1]),
              "s_step_mean": [0.56, 0.70], "s_step_forced_worst": 0.75, "hours_mean": [round(120000 * s / 3600, 1) for s in (0.56, 0.70)],
              "hours_forced": round(120000 * 0.75 / 3600, 1)}
for k, r in rows.items():
    hm = r["hours_mean"] if isinstance(r["hours_mean"], list) else [r["hours_mean"], r["hours_mean"]]
    r["usd_mean_range"] = [round(hm[0] * USD[0], 2), round(hm[1] * USD[2], 2)]; r["usd_mean_central"] = round(sum(hm) / 2 * USD[1], 2)
    r["usd_forced_at_0.26"] = round(r["hours_forced"] * USD[2], 2)
a2 = {"hours_mean": round(rows["A2_plain_steps"]["hours_mean"] + rows["A2_self_steps"]["hours_mean"], 2),
      "hours_forced": round(rows["A2_plain_steps"]["hours_forced"] + rows["A2_self_steps"]["hours_forced"], 2)}
a2["usd_mean_range"] = [round(a2["hours_mean"] * USD[0], 2), round(a2["hours_mean"] * USD[2], 2)]; a2["usd_forced_at_0.26"] = round(a2["hours_forced"] * USD[2], 2)
rows["A2"] = a2
out["cells"] = rows
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "v1_mem.json"), "w", newline="\n"), indent=1)
