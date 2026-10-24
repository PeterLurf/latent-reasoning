"""HOLD-DEPTH $0 check A (HD_draft.md §1): reads ONLY committed eval_vb.json / eval_khorizon.json / log.jsonl / args.json of the §224 cells,
their D3-T bases and the Stage-1 FT cells. No model is loaded, no GPU, no torch. Writes hd_check_json.out.json next to this script (LF).
Run: python -B docs/research_holddepth_2026-10-23/hd_check_json.py
Readings (field names as in the source files; all descriptive):
  depth        mean acc_n2 n 9..11 (score_s224_s2 'depth'); acc_n2 n 9..16
  late / settle  acc_k[K'][n] for n 9..11 at K' = n - 1, n, n + 1, n + 2, 16, 24: does a deeper read add answers (late arrival) or remove them (collapse)?
  r_t          r_acc_by_t[t][n] (R-decoder slot accuracy of V_t vs E_t, real slots) for n 9..11, t 0..n + 2: where the state goes wrong
  stepcap_t    stepcap.by_t_a pooled over a, per t (value-slot R acc at pass t on items whose V_{t-1} decodes E_{t-1} exactly; pooled over n >= t)
  bank         bank_off / bank_ablate acc_n2 n 9..11
  probe        probe[t].acc_by_n n 9..11 at t 8 / 12 (frozen-model readout trained on n 3..8)
  log          CurGate advance steps (cur_stage changes), basin entry (first 1k bin of answer_ce < 0.5, 10 records of 100 steps), final 1k-bin
               answer_ce / rec_ce / far_ce, fh_r_far (last 10 records = r_log), far_t_mean (mean), gn_preclip (median, steps > 8,000)"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); RUNS = os.path.join(ROOT, "boxes", "runs_fsq")
# eval dir -> training-log dir (base re-evals are eval-only; their logs are the original D3-T runs)
CELLS = {"base_s1": ("s224_base_s1_re", "s213_d3t_s1"), "base_s2": ("s224_base_s2_re", "s216_d3t_s2"),
         "rd_s1": ("s224_rd_s1", "s224_rd_s1"), "rd_s2": ("s224_rd_s2", "s224_rd_s2"), "rd_s3": ("s224_rd_s3", "s224_rd_s3"),
         "fh_s1": ("s224_fh_s1", "s224_fh_s1"), "fh_s2": ("s224_fh_s2", "s224_fh_s2"),
         "ftctrl_s1": ("s224_ftctrl_s1_re", None), "ftctrl_s2": ("s224_ftctrl_s2", "s224_ftctrl_s2"),
         "rdft_s1": ("s224_rdft_s1", "s224_rdft_s1"), "rdft_s2": ("s224_rdft_s2", "s224_rdft_s2"),
         "fhft_s1": ("s224_fhft_s1", "s224_fhft_s1"), "fhft_s2": ("s224_fhft_s2", "s224_fhft_s2"),
         "d3t_s1_orig": ("s213_d3t_s1", None), "d3t_s2_orig": ("s216_d3t_s2", None)}
DN = (9, 10, 11)


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rows(p):
    return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else None


def bins(lg, key, w=10):
    v = [(r["step"], r.get(key)) for r in lg if r.get(key) is not None]
    return [(v[i + w - 1][0], float(np.mean([x for _, x in v[i:i + w]]))) for i in range(0, len(v) - w + 1, w)]


def logread(lg):
    adv, prev = [], None
    for r in lg:
        c = r.get("cur_stage")
        if c is not None and prev is not None and c != prev:
            adv.append(r["step"])
        if c is not None:
            prev = c
    a = bins(lg, "answer_ce"); o = {"adv": adv, "entry_ans_lt_0p5": next((s for s, v in a if v < 0.5), None), "last_step": lg[-1]["step"]}
    for k in ("answer_ce", "rec_ce", "far_ce"):
        b = bins(lg, k); o["final_" + k] = b[-1][1] if b else None
    o["far_ce_bins_4k"] = [(s, v) for s, v in bins(lg, "far_ce", 40)] or None
    o["far_entry_lt_0p5"] = next((s for s, v in bins(lg, "far_ce") if v < 0.5), None)
    rr = [r["fh_r_far"] for r in lg if r.get("fh_r_far") is not None]; o["r_log"] = float(np.mean(rr[-10:])) if rr else None
    ft = [r["far_t_mean"] for r in lg if r.get("far_t_mean") is not None]; o["far_t_mean"] = float(np.mean(ft)) if ft else None
    g = [r["gn_preclip"] for r in lg if r.get("gn_preclip") is not None and r["step"] > 8000]; o["gn_preclip_med_after8k"] = float(np.median(g)) if g else None
    c = [r["clip_frac_100"] for r in lg if r.get("clip_frac_100") is not None and r["step"] > 8000]; o["clip_frac_mean_after8k"] = float(np.mean(c)) if c else None
    a5 = [(r["step"], r["answer_ce"]) for r in lg if r.get("answer_ce") is not None]   # entry, finer: first 5-record (500-step) mean < 0.5
    o["entry5_ans_lt_0p5"] = next((a5[i + 4][0] for i in range(len(a5) - 4) if np.mean([v for _, v in a5[i:i + 5]]) < 0.5), None)
    o["repair_dose"] = float(sum((1 - r["fh_gate_share"]) * 100 for r in lg if r.get("fh_gate_share") is not None)) if any("fh_gate_share" in r for r in lg) else None
    o["far_ce_int_0_12k"] = float(sum(r["far_ce"] * 100 for r in lg if r.get("far_ce") is not None and r["step"] <= 12000)) if any("far_ce" in r for r in lg) else None
    g0 = [r["gn_preclip"] for r in lg if r.get("gn_preclip") is not None and r["step"] <= 1000]; o["gn_preclip_mean_0_1k"] = float(np.mean(g0)) if g0 else None
    a23 = [r["answer_ce"] for r in lg if r.get("answer_ce") is not None and 2000 < r["step"] <= 3000]; o["answer_ce_2k_3k"] = float(np.mean(a23)) if a23 else None
    gs = [(r["step"], r["fh_gate_share"]) for r in lg if r.get("fh_gate_share") is not None]
    gb = [[v for s, v in gs if i * 1000 < s <= (i + 1) * 1000] for i in range(14)]
    o["gate_share_1k_bins_0_14k"] = [round(float(np.mean(b)), 3) if b else None for b in gb] if gs else None
    return o


def evread(vb):
    a, ak, ra = vb["acc_n2"], vb["acc_k"], vb["r_acc_by_t"]
    o = {"depth": float(np.mean([a[str(n)] for n in DN])), "acc_n2_9_16": {n: a[str(n)] for n in range(9, 17)},
         "acc18_min": min(a[str(n)] for n in range(1, 9)), "early": float(np.mean([ak[str(n - 1)][str(n)] for n in range(2, 9)]))}
    o["acc_k_deep"] = {n: {lab: ak[str(k)][str(n)] for lab, k in (("n-1", n - 1), ("n", n), ("n+1", n + 1), ("n+2", n + 2), ("16", 16), ("24", 24))} for n in DN}
    o["late_gain_24_minus_n2"] = float(np.mean([ak["24"][str(n)] - ak[str(n + 2)][str(n)] for n in DN]))
    o["n_minus_n2"] = float(np.mean([ak[str(n)][str(n)] - ak[str(n + 2)][str(n)] for n in DN]))
    o["r_t"] = {n: {t: round(ra[str(t)][str(n)], 4) for t in range(0, n + 3) if str(n) in ra.get(str(t), {})} for n in DN}
    o["r_t_first_below_0p99"] = {n: next((t for t in range(1, n + 3) if ra[str(t)][str(n)] < 0.99), None) for n in DN}
    o["r_t_at_n"] = float(np.mean([ra[str(n)][str(n)] for n in DN]))
    bt = {}
    for k, v in vb["stepcap"]["by_t_a"].items():
        t = int(k.split(",")[0]); s = bt.setdefault(t, [0.0, 0]); s[0] += v["acc"] * v["n"]; s[1] += v["n"]
    o["stepcap_t"] = {t: (round(s[0] / s[1], 4), s[1]) for t, s in sorted(bt.items())}
    o["bank_off_deep"] = float(np.mean([vb["bank_off"]["acc_n2"][str(n)] for n in DN]))
    o["bank_ablate_deep"] = float(np.mean([vb["bank_ablate"]["acc_n2"][str(n)] for n in DN]))
    o["stall"] = vb["stall"]["recovery"]
    if "probe" in vb:
        o["probe_deep"] = {t: float(np.mean([vb["probe"][t]["acc_by_n"][str(n)] for n in DN])) for t in ("8", "12") if t in vb["probe"]}
    o["card"] = vb.get("card")
    return o


out = {"def": __doc__, "cells": {}}
for name, (ed, ld) in CELLS.items():
    c = {"eval_dir": ed, "log_dir": ld}; vb = jl(os.path.join(RUNS, ed, "eval_vb.json"))
    if vb is not None:
        c.update(evread(vb))
    kh = jl(os.path.join(RUNS, ed, "eval_khorizon.json"))
    if kh is not None:
        c["H128"] = float(np.mean([kh["acc"]["intact"]["128"][str(n)] for n in range(1, 9)])) if "128" in kh["acc"]["intact"] else None
    if ld:
        lg = rows(os.path.join(RUNS, ld, "log.jsonl"))
        if lg:
            c["log"] = logread(lg)
    out["cells"][name] = c
C = out["cells"]
# same-seed deltas (Audit (3)); s3 has no base: both bases shown, flagged
dd = {}
for arm in ("rd", "fh"):
    for s in ("s1", "s2", "s3"):
        k = "%s_%s" % (arm, s)
        if k in C and "depth" in C[k]:
            if s == "s3":
                dd[k] = {"vs_base_s1": C[k]["depth"] - C["base_s1"]["depth"], "vs_base_s2": C[k]["depth"] - C["base_s2"]["depth"], "same_seed_base": None}
            else:
                dd[k] = {"same_seed": C[k]["depth"] - C["base_" + s]["depth"]}
for arm, ctrl in (("rdft", "ftctrl"), ("fhft", "ftctrl")):
    for s in ("s1", "s2"):
        dd["%s_%s" % (arm, s)] = {"vs_ftctrl": C["%s_%s" % (arm, s)]["depth"] - C["%s_%s" % (ctrl, s)]["depth"],
                                  "vs_d3t_base_re": C["%s_%s" % (arm, s)]["depth"] - C["base_" + s]["depth"]}
out["depth_deltas"] = dd
json.dump(out, open(os.path.join(HERE, "hd_check_json.out.json"), "w", newline="\n"), indent=1)
for k, c in C.items():
    lg = c.get("log", {})
    print("%-12s depth %.3f H128 %s n-n2 %+.3f late %+.3f r@n %.3f bo %.3f ba %.3f early %.3f | adv %s entry %s far_entry %s r_log %s ans %.2e far %s" % (
        k, c.get("depth", float("nan")), None if c.get("H128") is None else round(c["H128"], 3), c.get("n_minus_n2", float("nan")), c.get("late_gain_24_minus_n2", float("nan")),
        c.get("r_t_at_n", float("nan")), c.get("bank_off_deep", float("nan")), c.get("bank_ablate_deep", float("nan")), c.get("early", float("nan")),
        lg.get("adv"), lg.get("entry_ans_lt_0p5"), lg.get("far_entry_lt_0p5"), None if lg.get("r_log") is None else "%.2e" % lg["r_log"],
        lg.get("final_answer_ce") or float("nan"), None if lg.get("final_far_ce") is None else "%.2e" % lg["final_far_ce"]))
print(json.dumps(dd, indent=0))
