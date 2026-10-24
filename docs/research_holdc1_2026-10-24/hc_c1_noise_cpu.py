"""HOLD-C1 (HC_draft.md) $0 check N: the seed / fine-tune / card noise of every C1 statistic, from COMMITTED eval_vb.json files only
(no model load, no torch, no GPU; 2 threads, idle priority). Writes hc_c1_noise_cpu.out.json next to this script (LF).
    python -B docs/research_holdc1_2026-10-24/hc_c1_noise_cpu.py
Statistics per model (definitions copied from theory/score_s211.py, the registered VB-C1 scorer):
  ACC (acc_n2 pooled n 3..8), DEEP_min (min acc_n2 n 3..8), PROBE(t) pooled n 3..8, PROBE_GAIN = ACC - max(PROBE(0), PROBE(1)),
  onset: coverage min over n 3..8, slope_h (OLS of median onset on height, heights with >= 30 items), gap_8_3; vswap follow_i / own_i /
  follow_ii / own_ii / own_iv; stall recovery; carry_likeness (descriptive).
Pairs:
  FT  : same-lineage parent -> fine-tuned child, all RTX 3070 fp32 bs 32 (the plain FT null s213_d3t_s1 -> s215_ftctrl_s1, plus the three
        self-item FTs, which are larger perturbations and so an upper bound for a matched 12k FT);
  SEED: s213_d3t_s1 vs s216_d3t_s2 (3070);
  CARD: the same model on the 3070 vs the 3090 (s224_base_s*_re; acc / probe / stall only), which includes probe-training nondeterminism.
Binomial SE per statistic from the item counts; slope_h SE by a 300-draw item bootstrap within n (rng 23024).
Margin rule (HC_draft section 4.3): delta_S = max(floor_S, 2 x max_FT |d_S|, 3 x sqrt(2) x SE_S), harm direction only.
OC (HC_draft section 6): per statistic, P(no false fail | true d = 0) with sd_null = max(max_FT |d_S| / 1.5, sqrt(2) x SE_S) [ASSUMED: the
largest of four FT pairs ~ 1.5 sd], vs both references (FT-CTRL: sqrt(2) sd; BASE: 1 sd), per seed and for 2 seeds; and P(pass | harm h_S)."""
import json
import math
import os
import statistics
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if hasattr(psutil, "IDLE_PRIORITY_CLASS") else 19)
except Exception:                                                          # noqa: BLE001
    pass
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(ROOT, "boxes", "runs_fsq")
T_SET = (0, 1, 2, 3, 4, 6, 8, 12); NS = range(3, 9); INF = float("inf")
MODELS = ["s213_d3t_s1", "s215_ftctrl_s1", "s215_ftself_s1", "s216_r2_s1", "s216_d3t_s2", "s216_r1_s2", "s224_base_s1_re", "s224_base_s2_re"]
FT_PAIRS = [("s213_d3t_s1", "s215_ftctrl_s1", "plain FT (matched null)"), ("s213_d3t_s1", "s215_ftself_s1", "self-item FT"),
            ("s215_ftself_s1", "s216_r2_s1", "self-item FT r2"), ("s216_d3t_s2", "s216_r1_s2", "self-item FT s2")]
SEED_PAIRS = [("s213_d3t_s1", "s216_d3t_s2")]
CARD_PAIRS = [("s213_d3t_s1", "s224_base_s1_re"), ("s216_d3t_s2", "s224_base_s2_re")]
# decisive C1-KEPT statistics: (name, harm sign (-1 = a drop harms), floor, meaningful harm h_S for the OC)
DECISIVE = [("PROBE_GAIN", -1, 0.03, 0.10), ("slope_h", -1, 0.15, 0.40), ("follow_ii", -1, 0.02, 0.15), ("own_ii", +1, 0.02, 0.15),
            ("stall", -1, 0.02, 0.10), ("DEEP_min", -1, 0.02, 0.05), ("PROBE_2", +1, 0.03, 0.15)]
DESCRIPTIVE = ["follow_i", "own_i", "own_iv", "carry_likeness", "cov_min", "gap_8_3", "PROBE_3", "PROBE_4", "PROBE_6"]


def ols(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys); den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else None


def onsets(ev):
    pr, meta = ev["probe"], ev["test_meta"]; out = []
    for i, n in enumerate(meta["n"]):
        o = INF
        for t in T_SET:
            ts = [u for u in T_SET if t <= u <= n + 4]
            if ts and all(pr[str(u)]["items"][i] for u in ts):
                o = t; break
        out.append(o)
    return out


def onset_stats(ns, hs, on, idx=None):
    idx = range(len(ns)) if idx is None else idx
    byh, byn = {}, {}
    for i in idx:
        if 3 <= ns[i] <= 8:
            byn.setdefault(ns[i], []).append(on[i])
            if on[i] < INF:
                byh.setdefault(hs[i], []).append(on[i])
    hk = sorted(h for h in byh if len(byh[h]) >= 30)
    slope = ols(hk, [statistics.median(byh[h]) for h in hk]) if len(hk) >= 2 else None
    cov = min(sum(1 for o in byn[n] if o < INF) / len(byn[n]) for n in NS)
    med = {n: statistics.median(byn[n]) for n in NS}
    gap = med[8] - med[3] if med[8] < INF and med[3] < INF else None
    return slope, cov, gap


def stats(run):
    ev = json.load(open(os.path.join(R, run, "eval_vb.json"))); a2 = ev["acc_n2"]; o = {"card": ev.get("card"), "eval_bs": ev.get("eval_bs")}
    o["ACC"] = sum(a2[str(n)] for n in NS) / 6; o["DEEP_min"] = min(a2[str(n)] for n in NS)
    if "probe" in ev:
        p = {t: sum(ev["probe"][str(t)]["acc_by_n"][str(n)] for n in NS) / 6 for t in T_SET}
        o.update({"PROBE_%d" % t: p[t] for t in T_SET}); o["PROBE_GAIN"] = o["ACC"] - max(p[0], p[1])
        if "test_meta" in ev and "items" in ev["probe"]["0"]:
            ns, hs = ev["test_meta"]["n"], ev["test_meta"]["height"]; on = onsets(ev)
            o["slope_h"], o["cov_min"], o["gap_8_3"] = onset_stats(ns, hs, on)
            rng = np.random.default_rng(23024); byn = {n: [i for i, m in enumerate(ns) if m == n] for n in NS}; bs = []
            for _ in range(300):
                idx = [int(j) for n in NS for j in rng.choice(byn[n], size=len(byn[n]), replace=True)]
                s_, _, _ = onset_stats(ns, hs, on, idx); bs.append(s_)
            bs = [b for b in bs if b is not None]; o["slope_h_boot_sd"] = float(np.std(bs, ddof=1)) if len(bs) > 2 else None
    vs = ev.get("vswap")
    if vs:
        o.update({k: vs.get(k) for k in ("follow_i", "own_i", "follow_ii", "own_ii", "own_iv")}); o["n_pairs"] = vs.get("n_pairs")
    st = ev.get("stall")
    if st:
        o["stall"] = st.get("recovery")
    cl = ev.get("carry_likeness")
    o["carry_likeness"] = cl if isinstance(cl, (int, float)) else None
    return o


def se(name, o):
    """binomial SE of one model's statistic (a proportion of N items), or the bootstrap SD for slope_h."""
    if name == "slope_h":
        return o.get("slope_h_boot_sd")
    if name in ("follow_ii", "own_ii", "follow_i", "own_i", "own_iv"):
        N_, p = o.get("n_pairs") or 4184, o.get(name)
    elif name == "stall":
        N_, p = 5000, min(0.999, o.get("stall") or 1.0)
    elif name == "DEEP_min":
        N_, p = 1000, o.get(name)
    elif name.startswith("PROBE"):
        N_, p = 6000, (o.get("PROBE_1") if name == "PROBE_GAIN" else o.get(name))
    else:
        return None
    p = min(max(p, 0.5 / N_), 1 - 0.5 / N_)                                 # a floor so saturated statistics do not get SE 0
    return math.sqrt(p * (1 - p) / N_)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    S = {r: stats(r) for r in MODELS}; keys = [d[0] for d in DECISIVE] + DESCRIPTIVE

    def diff(a, b, k):
        return (S[b].get(k) - S[a].get(k)) if (S[a].get(k) is not None and S[b].get(k) is not None) else None
    pairs = {"FT": [(a, b, lab, {k: diff(a, b, k) for k in keys}) for a, b, lab in FT_PAIRS],
             "SEED": [(a, b, "seed", {k: diff(a, b, k) for k in keys}) for a, b in SEED_PAIRS],
             "CARD": [(a, b, "card 3070->3090", {k: diff(a, b, k) for k in keys}) for a, b in CARD_PAIRS]}
    margins = {}; oc = {}
    for name, sign, floor, harm in DECISIVE:
        ft = [abs(d[3][name]) for d in pairs["FT"] if d[3][name] is not None]; mft = max(ft) if ft else 0.0
        ses = [se(name, S[r]) for r in MODELS if S[r].get(name) is not None and se(name, S[r]) is not None]; s1 = max(ses) if ses else 0.0
        delta = max(floor, 2 * mft, 3 * math.sqrt(2) * s1)
        sd0 = max(mft / 1.5, math.sqrt(2) * s1)                               # [ASSUMED] null sd of one FT-vs-parent difference
        p_ctrl = Phi(delta / (math.sqrt(2) * sd0)) if sd0 > 0 else 1.0       # HC vs FT-CTRL: two independent FTs of one base
        p_base = Phi(delta / sd0) if sd0 > 0 else 1.0                        # HC vs BASE: one FT
        q_ctrl = Phi((delta - harm) / (math.sqrt(2) * sd0)) if sd0 > 0 else float(harm <= delta)
        q_base = Phi((delta - harm) / sd0) if sd0 > 0 else float(harm <= delta)
        margins[name] = {"harm_sign": sign, "floor": floor, "max_abs_FT_diff": mft, "max_single_SE": s1, "delta": delta, "sd_null_assumed": sd0}
        oc[name] = {"P_keep_per_seed_no_effect": p_ctrl * p_base, "P_keep_2seeds_no_effect": (p_ctrl * p_base) ** 2,
                    "harm_h": harm, "P_pass_per_seed_given_harm": min(q_ctrl, q_base), "P_pass_2seeds_given_harm": min(q_ctrl, q_base) ** 2}
    joint0 = 1.0
    for v in oc.values():
        joint0 *= v["P_keep_2seeds_no_effect"]
    out = {"models": S, "pairs": {k: [{"a": a, "b": b, "what": lab, "d": d} for a, b, lab, d in v] for k, v in pairs.items()},
           "margins": margins, "oc": oc, "P_C1KEPT_all_stats_2seeds_no_effect_indep": joint0,
           "def": "see the module docstring; every number comes from committed eval_vb.json files; decisive = %s" % [d[0] for d in DECISIVE]}
    fo = os.path.join(HERE, "hc_c1_noise_cpu.out.json")
    with open(fo, "w", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for name in margins:
        m_, o_ = margins[name], oc[name]
        print("%-11s maxFT %.4f  SE %.4f  delta %.3f | keep|0 per seed %.3f, 2 seeds %.3f | pass|harm %.2f -> %.3f" % (
            name, m_["max_abs_FT_diff"], m_["max_single_SE"], m_["delta"], o_["P_keep_per_seed_no_effect"], o_["P_keep_2seeds_no_effect"],
            o_["harm_h"], o_["P_pass_2seeds_given_harm"]))
    print("joint P(C1-KEPT, all decisive stats, 2 seeds | no effect) = %.3f" % joint0)
    for r in MODELS:
        print(r, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in S[r].items() if k in keys + ["n_pairs"]})
    for k, v in pairs.items():
        for a, b, lab, d in v:
            print(k, a, "->", b, {kk: round(vv, 4) for kk, vv in d.items() if vv is not None})
    print("wrote", fo)


if __name__ == "__main__":
    sys.exit(main())
