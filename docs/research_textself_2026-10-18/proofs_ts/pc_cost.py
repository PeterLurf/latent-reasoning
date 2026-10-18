"""PROOFS_TS check P6 (CPU, no model, 1 thread, Idle): cost items the draft's ts_cost.py does not cover, with ts_cost's own calibration.
(a) TF-GREEDY REROOT token-positions per item, unpadded (sum_k L_{k-1} + L_k + 2) and padded as label_reroot batches them (per step, active items
    sorted by length, label batch 128, both chunks right-padded to the batch max), T1 window and T2 mix; fresh SHAPE-ONLY draws (rng 23189).
(b) eval_textself's free-generation parts, in sequential decode steps under cot_answers' batching (prompt-length groups, bs 32, steps between max C
    and max C + 50): MEMO-GAP-T = 2,000 consumed pool items + 2,000 gate items per window n (T1: n 9-12; T2: n 9-16); converted to hours with
    ts_cost.json's own latency model (a_lo..a_hi + 2 b x context, fp32; 3070 class).
(c) the per-lineage totals with (b) added. python -B pc_cost.py"""
import ctypes
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"; sys.dont_write_bytecode = True
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass
sys.path.insert(0, "D:/Latent/docs/research_textself_2026-10-18/scripts"); sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import ts_cost as TC  # noqa: E402  (read-only import; main() not called)

N = TC.N; HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "pc_cost.json")
CAL = json.load(open("D:/Latent/docs/research_textself_2026-10-18/scripts/ts_cost.json"))["eval_n1_24"]


def draw(rng, ns, w, k):
    out = []
    for _ in range(k):
        n = int(rng.choice(ns, p=w)); out.append(N.generate(rng, n)[0])
    return out


def tf_positions(roots, bs=128):
    sts = [TC.chain_struct(r) for r in roots]; unp = sum(sum(Lp + L + 2 for (Lp, pre, suf, L) in st) for st in sts)
    pad = 0; nmax = max(len(st) for st in sts)
    for k in range(nmax):
        act = sorted([st[k] for st in sts if k < len(st)], key=lambda s: s[0])
        for b0 in range(0, len(act), bs):
            ch = act[b0:b0 + bs]; c1 = max(Lp + 2 + pre for (Lp, pre, suf, L) in ch); c2 = max(1 + suf for (Lp, pre, suf, L) in ch); pad += len(ch) * (c1 + c2)
    return unp / len(roots), pad / len(roots)


def gen_steps(roots, bs=32, extra=0):
    by = {}
    for r in roots:
        by.setdefault(len(N.render(r)), []).append(r)
    steps = 0; ctx = 0.0
    for L, grp in by.items():
        for s in range(0, len(grp), bs):
            ch = grp[s:s + bs]; Cs = [TC.C_of(TC.chain_struct(r)) for r in ch]; P = L + 2; st = min(max(Cs) + extra, 1025 - P); steps += st; ctx += st * (P + st / 2)
    return steps, ctx


def hours(steps_lo, ctx_lo, steps_hi, ctx_hi):
    b2 = 2 * CAL["b_sec_per_ctx_tok"]; return ((CAL["a_lo"] * steps_lo + b2 * ctx_lo) / 3600, (CAL["a_hi"] * steps_hi + b2 * ctx_hi) / 3600)


def main():
    rng = np.random.default_rng(23189); res = {}
    T1 = draw(rng, [9, 10, 11, 12], [0.25] * 4, 4000); T2 = draw(rng, list(range(9, 17)), [1 / 12] * 4 + [1 / 6] * 4, 4000)
    res["tf_positions_per_item"] = {"T1": tf_positions(T1), "T2": tf_positions(T2), "note": "(unpadded, padded at label batch 128)"}
    print(res["tf_positions_per_item"], flush=True)
    memo = {}
    for name, ns in (("T1", [9, 10, 11, 12]), ("T2", list(range(9, 17)))):
        roots = []
        for n in ns:
            roots += [N.generate(rng, n)[0] for _ in range(4000)]            # 2,000 pool + 2,000 gate items per window n
        lo, clo = gen_steps(roots, extra=0); hi, chi = gen_steps(roots, extra=50); h = hours(lo, clo, hi, chi)
        memo[name] = {"items": len(roots), "steps_lo": lo, "steps_hi": hi, "hours_3070_fp32": h, "hours_3090_x0.7": (0.7 * h[0], 0.7 * h[1])}
        print(name, memo[name], flush=True)
    res["memo_gap"] = memo
    eh = (CAL["sec_fp32_lo"] / 3600, CAL["sec_fp32_hi"] / 3600); res["eval_text_hours_3070"] = eh
    res["eval_stage_hours_3070"] = {k: (eh[0] + v["hours_3070_fp32"][0], eh[1] + v["hours_3070_fp32"][1]) for k, v in memo.items()}
    res["eval_stage_hours_3090_x0.7"] = {k: (0.7 * a, 0.7 * b) for k, (a, b) in res["eval_stage_hours_3070"].items()}
    # draft's Plan A table with the eval_textself free generation added (draft rows: T1 / T2 eval stage 1.5-2.7 each)
    draft_lo, draft_hi = 10.2, 19.1
    add_lo = (res["eval_stage_hours_3090_x0.7"]["T1"][0] - 1.5) + (res["eval_stage_hours_3090_x0.7"]["T2"][0] - 1.5)
    add_hi = (res["eval_stage_hours_3070"]["T1"][1] - 2.7) + (res["eval_stage_hours_3070"]["T2"][1] - 2.7)
    res["per_lineage_hours_revised"] = (draft_lo + max(0.0, add_lo), draft_hi + max(0.0, add_hi))
    res["per_lineage_usd_revised"] = (0.17 * res["per_lineage_hours_revised"][0], 0.26 * res["per_lineage_hours_revised"][1])
    print(json.dumps({k: v for k, v in res.items() if k not in ("memo_gap",)}, indent=1), flush=True)
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
