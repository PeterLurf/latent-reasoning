"""TEXT-SELF draft check 3 (CPU, no model): compute / time projections from structure-only lengths.
(a) eval: nested170.cot_answers' batching (group by len(syms), cot_eval_bs 32, steps per batch between max C and max C + 50) over test_set(1000, n_max)
    for n 1-16 (calibrated against s170_cot_s1's measured eval_sec 1411 s, RTX 3070, bf16 autocast, cot_eval_bs 32) and n 1-24 (projection; KV
    context grows to ~1,085, so a per-step latency model is used: a + b * mean context per step).
(b) labelling with TF-GREEDY (REROOT): token-positions per item = sum_k (P_k + |pre_k|) + (1 + |suf_k|), from struct lengths, on fresh draws
    (own stream 23191, not a pool) for the T1 window (n 9-12 uniform) and T2 mix (2/3 n 13-16, 1/3 n 9-12); FULL protocol for comparison.
(c) training: per-step time model t = c0 + c1 * (batch-64 max T), calibrated at the measured 0.27 s/step steady rate of s170_cot_s1/s2 (3090, bs 64,
    plain n 1-8, batch max T mean 134), with an attention correction (1 + T / (6 D)) relative to T 134.
Every model form here is [ASSUMED]; inputs [MEASURED] are named. python -B ts_cost.py"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2"); sys.dont_write_bytecode = True
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested170 as N  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "ts_cost.json")
GFLOP_POS = 2 * 124.4e6 / 1e9          # forward FLOPs per token position, GPT-2 small incl. the tied LM head (~2 x params) [ASSUMED]


def chain_struct(root):
    """per step k: (L_{k-1}, |pre_k|, |suf_k|, L_k) from shape only."""
    nd = N._copy(root); out = []
    while nd.op is not None:
        Lp = len(N.render(nd)); path = N._leftmost_ready(nd); tgt = nd
        for p in path:
            tgt = getattr(tgt, p)
        tgt.op = tgt.l = tgt.r = None; tgt.v = 0; syms, pos = N._render_pos(nd, tgt); out.append((Lp, pos, len(syms) - pos - 1, len(syms)))
    return out


def C_of(st):
    return sum(s[3] for s in st) + len(st)                                 # e_k tokens + ' =' / EOT terminators


def eval_steps(TE, ns, bs=32, extra=0):
    tot = 0; ctx = 0.0; per_n = {}
    for n in ns:
        its = [it for it in TE if it["n"] == n]; by = {}
        for it in its:
            by.setdefault(len(it["syms"]), []).append(it)
        sn = 0; cn = 0.0
        for L, grp in by.items():
            for s in range(0, len(grp), bs):
                ch = grp[s:s + bs]; Cs = [C_of(chain_struct(it["root"])) for it in ch]; P = L + 2
                steps = min(max(Cs) + extra, 1025 - P); sn += steps; cn += steps * (P + steps / 2)   # mean KV context over the batch's steps
        per_n[str(n)] = {"steps": sn, "mean_ctx": cn / max(1, sn)}; tot += sn; ctx += cn
    return tot, ctx, per_n


def main():
    TE = N.test_set(1000, n_max=24); res = {}
    s16_lo, c16_lo, _ = eval_steps(TE, range(1, 17), extra=0); s16_hi, c16_hi, _ = eval_steps(TE, range(1, 17), extra=50)
    res["eval_calib"] = {"measured_eval_sec": 1411, "steps_n1_16_lo": s16_lo, "steps_n1_16_hi": s16_hi, "sec_per_step_hi": 1411 / s16_lo, "sec_per_step_lo": 1411 / s16_hi,
                         "mean_ctx_lo": c16_lo / s16_lo, "src": "s170_cot_s1/eval.json eval_sec (3070, bf16, cot_eval_bs 32)"}
    s24_lo, c24_lo, pn_lo = eval_steps(TE, range(1, 25), extra=0); s24_hi, c24_hi, pn_hi = eval_steps(TE, range(1, 25), extra=50)
    # latency model: per-step a + b * ctx; take b from KV bytes: B 32 x 12 layers x 2 x 768 x 4 B (fp32) per context token, read + cat-copy x 3, at ~400 GB/s
    b = 32 * 12 * 2 * 768 * 4 * 3 / 400e9; a_lo = 1411 / s16_hi - b * (c16_hi / s16_hi); a_hi = 1411 / s16_lo - b * (c16_lo / s16_lo)
    res["eval_n1_24"] = {"steps_lo": s24_lo, "steps_hi": s24_hi, "per_n_lo": pn_lo, "b_sec_per_ctx_tok": b, "a_lo": a_lo, "a_hi": a_hi,
                         "sec_lo": a_lo * s24_lo + b * c24_lo, "sec_hi": a_hi * s24_hi + b * c24_hi,
                         "note": "a fitted on n 1-16 (bf16 eval); fp32 doubles the KV bytes: b x 2 used in sec_fp32"}
    res["eval_n1_24"]["sec_fp32_lo"] = a_lo * s24_lo + 2 * b * c24_lo; res["eval_n1_24"]["sec_fp32_hi"] = a_hi * s24_hi + 2 * b * c24_hi
    print("eval", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["eval_calib"].items()}), flush=True)
    print("eval24", {k: round(v, 1) for k, v in res["eval_n1_24"].items() if isinstance(v, float)}, flush=True)
    # (b) labelling
    rng = np.random.default_rng(23191); lab = {}
    for name, ns, w in (("T1", [9, 10, 11, 12], [0.25] * 4), ("T2", list(range(9, 17)), [1 / 12] * 4 + [1 / 6] * 4), ("gate_n13_16", [13, 14, 15, 16], [0.25] * 4)):
        rr, ff, cc = [], [], []
        for _ in range(4000):
            n = int(rng.choice(ns, p=w)); r, _ = N.generate(rng, n); st = chain_struct(r); P0 = len(N.render(r)) + 2
            rr.append(sum((Lp + 2 + pre) + (1 + suf) for (Lp, pre, suf, L) in st)); ff.append(P0 + C_of(st)); cc.append(C_of(st))
        lab[name] = {"reroot_pos_mean": float(np.mean(rr)), "full_pos_mean": float(np.mean(ff)), "C_mean": float(np.mean(cc)),
                     "reroot_calls_per_item": "2 per step", "gflop_reroot_per_item": float(np.mean(rr)) * GFLOP_POS}
    res["label"] = lab; print("label", lab, flush=True)
    items = 384_000 + 8_000                                                 # pool + gate items per round (keep ~ 1 assumed; see TS_draft §2.1)
    for eff in (10e12, 17e12):                                              # achieved fp32 FLOP/s (tf32 off): 3070 ~10, 3090 ~17 [ASSUMED]
        for name in ("T1", "T2"):
            res.setdefault("label_hours", {})["%s_%.0fT" % (name, eff / 1e12)] = items * lab[name]["gflop_reroot_per_item"] * 1e9 / eff / 3600
    print("label_hours", res["label_hours"], flush=True)
    # (c) training
    D = 768; base_T = 134.23; base_s = 0.27
    def t_of(T):
        return base_s * (T / base_T) * (1 + T / (6 * D)) / (1 + base_T / (6 * D))
    sh = json.load(open(os.path.join(HERE, "ts_tokens.json")))["train_shapes"]
    tr = {}
    for name, key in (("T1_self", "T1_n9_12"), ("T2_self", "T2_mix")):
        tr[name] = t_of(sh[key]["batch64_max_mean"])
    tr["plain"] = base_s; tr["T1_hours"] = (5992 * tr["T1_self"] + 6008 * tr["plain"]) / 3600; tr["T2_hours"] = (5992 * tr["T2_self"] + 6008 * tr["plain"]) / 3600
    res["train"] = tr; print("train", {k: round(v, 3) for k, v in tr.items()}, flush=True)
    # memory model: GiB = fixed 1.9 + k * T / micro, k from s170 mem_check (12.99 GiB incl. AdamW at T ~147, micro 1) [ASSUMED linear]
    k = (12.99 - 1.9) / 147; mem = {}
    for name, T in (("T1_deep", 311), ("T2_deep", 527)):
        for m_ in (1, 2, 4):
            mem["%s_micro%d" % (name, m_)] = 1.9 + k * T / m_
    res["mem"] = mem; print("mem", {k_: round(v, 1) for k_, v in mem.items()}, flush=True)
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
