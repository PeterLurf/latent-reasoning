"""REDTEAM_TS check R1 ($0, CPU, generator statistics only; no model is run).
(a) Rebuild the draft's precondition items (ts_label_check.py: default_rng(23180), one stream, 200 per n for n 9..20 in order, redrawn on a hit
    against test_set(1000, 24) + SERIAL) and join them with the draft's per-item REROOT kept flags (ts_label_check_s1.json n 9-16,
    ts_label_check_s1_deep.json n 17-20). Keep vs the input length L0 = len(render(root)) and vs the step-1 output length.
(b) Training-distribution copy-length maxima: base (train_items law, n 1-8) and the T1 pool law (uniform n 9-12) on large fresh draws (own rngs).
(c) Project T1's REROOT keep at n 13-16 from the base's keep-vs-L0 curve under two reach laws (additive offset / ratio to the trained max).
(d) Sequential decode-step counts (cot_answers batching: equal prompt length, bs 32, max C + 50 per batch) for eval_text (n 1-24, 1000 per n)
    and for MEMO-GAP-T (2 x 2000 items per window n: T1 n 9-12, T2 n 9-16), to check the draft's eval-cost table.
Writes rt_reach.json beside this file (newline "\n")."""
import ctypes
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); sys.dont_write_bytecode = True
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
except Exception:  # noqa: BLE001
    pass
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested170 as N  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); SCR = os.path.join(os.path.dirname(HERE), "scripts"); t0 = time.time()


def nops(syms):
    return sum(1 for s in syms if s.strip() in N.PREC)


def chain_lens(root):
    st = N.cot_steps(root)                      # shape lengths only are used (gold values never enter the statistics)
    return [len(s) for s in st]                 # st[0] = input, st[k] = after k reductions


out = {"clock_start": time.strftime("%Y-%m-%d %H:%M:%S")}
# ---------- (a) precondition items ----------
excl = {tuple(it["syms"]) for it in N.test_set(1000, n_max=24)}; r_sp = np.random.default_rng(17300)
for _ in range(2200):
    excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
rng = np.random.default_rng(23180); items = {}
for n in range(9, 21):
    items[n] = []
    while len(items[n]) < 200:
        r_, _ = N.generate(rng, n)
        if tuple(N.render(r_)) in excl:
            continue
        items[n].append(r_)
s1 = json.load(open(os.path.join(SCR, "ts_label_check_s1.json"))); dp = json.load(open(os.path.join(SCR, "ts_label_check_s1_deep.json")))
rows = []
for n in range(9, 21):
    src = s1 if n <= 16 else dp; recs = src["items"]["%d_reroot" % n]
    for i, rec in enumerate(recs):
        L = chain_lens(items[n][i]); rows.append({"n": n, "L0": L[0], "L1": L[1], "kept": rec[0], "fail": rec[1]})
# sanity: the per-n keep recomputed from the joined rows equals the file's by_n keep
chk = {}
for n in range(9, 21):
    rr = [r for r in rows if r["n"] == n]; src = s1 if n <= 16 else dp
    chk[n] = (round(np.mean([r["kept"] for r in rr]), 4), src["by_n"][str(n)]["reroot"]["keep"])
out["join_check_keep"] = chk
# keep by L0 bin (all n pooled) and the step-1 failure share by L0
byL = {}
for r in rows:
    byL.setdefault(r["L0"], []).append(r)
out["keep_by_L0"] = {str(L): {"items": len(v), "keep": round(float(np.mean([x["kept"] for x in v])), 3),
                              "fail_step1": round(float(np.mean([x["fail"] == 1 for x in v])), 3)} for L, v in sorted(byL.items())}
out["mean_L0_by_n"] = {str(n): round(float(np.mean([r["L0"] for r in rows if r["n"] == n])), 2) for n in range(9, 21)}
# keep within n, split by L0 above / below the n's median (is it L0 or n that matters?)
split = {}
for n in range(9, 21):
    rr = [r for r in rows if r["n"] == n]; med = float(np.median([r["L0"] for r in rr]))
    lo = [r["kept"] for r in rr if r["L0"] <= med]; hi = [r["kept"] for r in rr if r["L0"] > med]
    split[str(n)] = {"median_L0": med, "keep_L0_le_med": round(float(np.mean(lo)), 3), "keep_L0_gt_med": round(float(np.mean(hi)), 3) if hi else None}
out["keep_split_by_L0_within_n"] = split
# logistic fit P(kept) ~ sigmoid(a + b * L0) (pooled n 9-20), Newton iterations
X = np.array([[1.0, r["L0"]] for r in rows]); y = np.array([r["kept"] for r in rows], float); w = np.zeros(2)
for _ in range(50):
    p = 1 / (1 + np.exp(-X @ w)); H = X.T @ (X * (p * (1 - p))[:, None]); g = X.T @ (y - p); w = w + np.linalg.solve(H + 1e-9 * np.eye(2), g)
out["logit_keep_vs_L0"] = {"a": float(w[0]), "b": float(w[1]), "L0_at_keep_0.70": float((np.log(0.7 / 0.3) - w[0]) / w[1]),
                           "L0_at_keep_0.90": float((np.log(0.9 / 0.1) - w[0]) / w[1]), "L0_at_keep_0.50": float(-w[0] / w[1])}
# ---------- (b) training-distribution copy-length maxima ----------
rb = np.random.default_rng(23999); Lb = []
for _ in range(200000):
    n_ = int(rb.integers(1, 9)); r_, _ = N.generate(rb, n_); Lb.append(len(N.render(r_)))
Lb = np.array(Lb)
rt = np.random.default_rng(23998); Lt = {}
for n in range(9, 17):
    Lt[n] = np.array([len(N.render(N.generate(rt, n)[0])) for _ in range(25000)])
test = N.test_set(1000, n_max=24)
out["L0_train"] = {"base_n1_8_200k": {"max": int(Lb.max()), "p99_9": float(np.percentile(Lb, 99.9)), "p99": float(np.percentile(Lb, 99)),
                                      "n8_test_max": int(max(len(it["syms"]) for it in test if it["n"] == 8))},
                   "per_n_25k": {str(n): {"mean": round(float(v.mean()), 2), "p99": float(np.percentile(v, 99)), "max": int(v.max())} for n, v in Lt.items()}}
# expected max over the real consumed counts: base saw 48k x 64 items (~384k at n 8); T1 pool ~96k per n at n 9-12
# ---------- (c) projection of T1's REROOT keep at n 13-16 ----------
a_, b_ = w
f = lambda L: 1 / (1 + np.exp(-(a_ + b_ * L)))  # noqa: E731
Lmax_base = 31.0; Lmax_t1 = float(max(Lt[12].max(), 43))
proj = {}
for n in range(13, 17):
    Ls = np.array([len(it["syms"]) for it in test if it["n"] == n], float)    # test items as the shape sample (shape only)
    proj[str(n)] = {"base_fit": round(float(f(Ls).mean()), 3),
                    "T1_additive (L - (Lmax_t1 - Lmax_base))": round(float(f(Ls - (Lmax_t1 - Lmax_base)).mean()), 3),
                    "T1_ratio (L * Lmax_base / Lmax_t1)": round(float(f(Ls * Lmax_base / Lmax_t1).mean()), 3),
                    "T1_half_transfer (L - 0.5 (Lmax_t1 - Lmax_base))": round(float(f(Ls - 0.5 * (Lmax_t1 - Lmax_base)).mean()), 3)}
out["T1_keep_projection"] = {"Lmax_base": Lmax_base, "Lmax_t1": Lmax_t1, "by_n": proj,
                             "note": "pooled logistic in L0 on the base's 2,000 REROOT items (n 9-20); a reach law is ASSUMED; T1 must also ABSORB first"}
# ---------- (d) decode-step counts ----------


def steps_for(roots_by_n, bs=32):
    tot = 0; batches = 0
    for n, roots in roots_by_n.items():
        by_len = {}
        for r_ in roots:
            s_ = N.render(r_); by_len.setdefault(len(s_), []).append(len(N.cot_ids(r_)))
        for L, cs in by_len.items():
            for s0 in range(0, len(cs), bs):
                tot += max(cs[s0:s0 + bs]) + 50; batches += 1
    return tot, batches


ev = {}
for n in range(1, 25):
    ev[n] = [it["root"] for it in test if it["n"] == n]
st_ev, b_ev = steps_for(ev)
st_ev16, b_ev16 = steps_for({n: ev[n] for n in range(1, 17)})
rm = np.random.default_rng(23997)
mg1 = {n: [N.generate(rm, n)[0] for _ in range(4000)] for n in range(9, 13)}
mg2 = {n: (mg1[n] if n <= 12 else [N.generate(rm, n)[0] for _ in range(4000)]) for n in range(9, 17)}
st_m1, b_m1 = steps_for(mg1); st_m2, b_m2 = steps_for(mg2)
out["decode_steps"] = {"eval_text_n1_24": st_ev, "eval_text_batches": b_ev, "eval_n1_16 (the measured 1,411 s bf16 3070 eval)": st_ev16,
                       "memo_T1_n9_12_4000_per_n": st_m1, "memo_T2_n9_16_4000_per_n": st_m2,
                       "memo_over_eval_T1": round(st_m1 / st_ev, 3), "memo_over_eval_T2": round(st_m2 / st_ev, 3),
                       "note": "upper bound per batch (max C + 50; a batch stops early only if all rows hit EOT); same bs 32 grouping as cot_answers"}
out["sec"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(HERE, "rt_reach.json"), "w", newline="\n"), indent=1)
print(json.dumps({k: out[k] for k in ("join_check_keep", "logit_keep_vs_L0", "L0_train", "T1_keep_projection", "decode_steps", "sec")}, indent=1))
