"""Aggregate emp_<run>.json -> EMP_gates.json (compact numbers) + EMP_gates_tables.md (tables; the prose report EMP_gates.md is written by hand)."""
import os
import json
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
CONV = ["s185_rms_s1", "s185_rms_s2", "s190_rmsnopass_s1", "s184_reweight_s2"]
FAIL = ["s185_rms_s3", "s185_den_s1", "s182_hold_s1", "s188_rwwin_s1"]
R = {}
for r in CONV + FAIL:
    fn = os.path.join(OUT, "emp_%s.json" % r)
    if os.path.exists(fn):
        R[r] = json.load(open(fn))
runs = [r for r in CONV + FAIL if r in R]
grp = {r: ("conv" if r in CONV else "FAIL") for r in runs}


def f(x, d=3):
    return "-" if (x is None or x != x) else ("%.*f" % (d, x))


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(head, rows):
    return "\n".join([row(head), row(["---"] * len(head))] + [row(r_) for r_ in rows])


def gmean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


md = []; comp = {"runs": runs, "groups": {"converged": [r for r in CONV if r in R], "failed": [r for r in FAIL if r in R]}, "models": {}}
P8 = [str(t) for t in range(1, 9)]; P12 = [str(t) for t in range(1, 13)]

# ---- (a) gates: pass-mean over 1..8 and per pass
md.append("## (a) Delta gates over real think tokens (items n 4..8; mean over passes 1..8; per-pass in JSON)")
rows = []
for r in runs:
    P = R[r]["passes"]; G = [P[t]["gates"] for t in P8]
    v = {k: float(np.mean([g_["beta"][k] for g_ in G])) for k in ("mean", "q05", "q50", "q95")}
    fr = {k: float(np.mean([g_[k] for g_ in G])) for k in ("beta_frac_lt0.2", "beta_frac_0.8_1.2", "beta_frac_gt1.8", "beta_frac_gt1.0")}
    am = float(np.mean([g_["alpha"]["mean"] for g_ in G])); a05 = float(np.mean([g_["alpha"]["q05"] for g_ in G]))
    comp["models"].setdefault(r, {})["gates_pass_mean"] = {"beta": v, **fr, "alpha_mean": am, "alpha_q05": a05}
    rows.append([r, grp[r], f(v["mean"]), f(v["q05"]), f(v["q50"]), f(v["q95"]), f(fr["beta_frac_lt0.2"]), f(fr["beta_frac_0.8_1.2"]), f(fr["beta_frac_gt1.8"]),
                 f(fr["beta_frac_gt1.0"]), f(am, 4), f(a05, 4)])
md.append(table(["run", "grp", "beta mean", "q05", "q50", "q95", "frac<0.2", "frac 0.8-1.2", "frac>1.8", "frac>1", "alpha mean", "alpha q05"], rows))
md.append("\nbeta mean by pass:")
md.append(table(["run", "grp"] + P12, [[r, grp[r]] + [f(R[r]["passes"][t]["gates"]["beta"]["mean"]) for t in P12] for r in runs]))
md.append("\nper-head beta mean (pass-mean 1..8) / per-head alpha mean:")
rows = []
for r in runs:
    ph = [[R[r]["passes"][t]["gates"]["per_head"][h] for t in P8] for h in range(6)]
    rows.append([r, grp[r]] + ["%.2f / %.4f" % (np.mean([x["beta_mean"] for x in ph[h]]), np.mean([x["alpha_mean"] for x in ph[h]])) for h in range(6)])
md.append(table(["run", "grp"] + ["h%d" % h for h in range(6)], rows))
md.append("\ngate parameters: |W_alpha| / |W_beta| row norms (per head), alpha bias mean, beta bias range:")
md.append(table(["run", "grp", "|W_a| mean", "|W_b| mean", "b_alpha mean", "b_beta min..max", "g_add", "g_del"],
                [[r, grp[r], f(np.mean(R[r]["gate_w_norm_alpha"])), f(np.mean(R[r]["gate_w_norm_beta"])), f(np.mean(R[r]["gate_bias_alpha"])),
                  "%.2f..%.2f" % (min(R[r]["gate_bias_beta"]), max(R[r]["gate_bias_beta"])), f(R[r]["g_add"]), f(R[r]["g_del"])] for r in runs]))

# ---- (b) contraction
md.append("\n## (b) Open-loop one-pass delta map P = prod_t alpha_t (I - beta_t k_t k_t^T) (64x64, per item and head; items n 4..8)")
md.append("mean log singular value (= log|det P| / 64, analytic) by pass:")
md.append(table(["run", "grp"] + P12, [[r, grp[r]] + [f(R[r]["passes"][t]["contraction"]["mean_log_sv_det"]["mean"]) for t in P12] for r in runs]))
for key, lab in (("top_sv", "top singular value"), ("n_sv_lt0.5", "number of singular values < 0.5 (of 64)"), ("n_sv_gt0.9", "number of singular values > 0.9"),
                 ("frob2_over_64", "||P||_F^2 / 64 (mean squared retention of a random direction)"), ("sum_log_alpha", "sum_t log alpha_t (isotropic decay per pass)")):
    md.append("\n%s by pass:" % lab)
    md.append(table(["run", "grp"] + P12, [[r, grp[r]] + [f(R[r]["passes"][t]["contraction"][key]["mean"], 2 if key.startswith("n_sv") else 3) for t in P12] for r in runs]))
md.append("\nretention of the ACTUAL previous delta state ||S_{t-1} P_t|| / ||S_{t-1}|| by pass (t >= 2):")
md.append(table(["run", "grp"] + P12[1:], [[r, grp[r]] + [f(R[r]["passes"][t]["contraction"]["state_retention"]["mean"]) for t in P12[1:]] for r in runs]))
md.append("\ncarried share ||S_{t-1} P_t|| / ||S_t|| by pass:")
md.append(table(["run", "grp"] + P12[1:], [[r, grp[r]] + [f(R[r]["passes"][t]["contraction"]["carried_share"]["mean"]) for t in P12[1:]] for r in runs]))
for r in runs:
    c = {t: R[r]["passes"][t]["contraction"] for t in P12}
    comp["models"][r]["contraction_by_pass"] = {t: {k: c[t][k]["mean"] for k in ("mean_log_sv_det", "mean_log_sv_svd", "top_sv", "n_sv_lt0.5", "n_sv_gt0.9", "frob2_over_64", "sum_log_alpha")}
                                                for t in P12}
    for t in P12[1:]:
        comp["models"][r]["contraction_by_pass"][t].update({"state_retention": c[t]["state_retention"]["mean"], "carried_share": c[t]["carried_share"]["mean"],
                                                             "cos_carried_new": c[t]["cos_carried_new"]["mean"]})
    comp["models"][r]["replay_rel_err_max"] = max(c[t]["replay_rel_err_max"] for t in P12)
    comp["models"][r]["contraction_pass_mean_1_8"] = {k: float(np.mean([c[t][k]["mean"] for t in P8])) for k in ("mean_log_sv_det", "top_sv", "n_sv_lt0.5", "frob2_over_64")}
    comp["models"][r]["contraction_pass_mean_1_8"]["state_retention_2_8"] = float(np.mean([c[t]["state_retention"]["mean"] for t in P8[1:]]))
md.append("\npass-mean (1..8) summary, group means in the last rows:")
rows = []; summ = {}
for r in runs:
    s = comp["models"][r]["contraction_pass_mean_1_8"]; summ[r] = s
    rows.append([r, grp[r], f(s["mean_log_sv_det"]), f(s["top_sv"]), f(s["n_sv_lt0.5"], 2), f(s["frob2_over_64"]), f(s["state_retention_2_8"])])
for gname, gl in (("CONVERGED mean", CONV), ("FAILED mean", FAIL)):
    gl = [r for r in gl if r in summ]
    rows.append([gname, ""] + [f(gmean([summ[r][k] for r in gl]), 2 if k == "n_sv_lt0.5" else 3) for k in ("mean_log_sv_det", "top_sv", "n_sv_lt0.5", "frob2_over_64", "state_retention_2_8")])
md.append(table(["run", "grp", "mean log sv", "top sv", "#sv<0.5", "|P|_F^2/64", "state retention"], rows))

# ---- by n (Audit request)
md.append("\n## By n (Audit request): alpha, alpha_eff^L = exp(mean_items sum_t log alpha_t), P spectrum; mean over passes 1..8 (n 10 / 12 from the extra set)")


def byn(r, n, key, passes=P8):
    src = R[r]["passes"] if n <= 8 else R[r].get("passes_ext", {})
    vals = [src[t]["by_n"][str(n)][key] for t in passes if t in src and str(n) in src[t]["by_n"] and src[t]["by_n"][str(n)][key] is not None]
    return float(np.mean(vals)) if vals else None


NS = [4, 6, 8, 10, 12]
for key, lab, d in (("L_mean", "real think tokens L", 1), ("alpha_mean", "mean alpha (heads averaged)", 4), ("alpha_effL", "alpha_eff^L (heads averaged in log)", 3),
                    ("alpha_effL_min_head", "alpha_eff^L of the most-decaying head", 3), ("mean_log_sv_det", "mean log singular value of P", 3), ("top_sv", "top singular value of P", 3),
                    ("n_sv_lt05", "# singular values < 0.5", 2), ("beta_mean", "beta mean", 3), ("beta_frac_0.8_1.2", "beta frac in [0.8, 1.2]", 3),
                    ("state_retention", "retention of the actual previous state (passes 2..8)", 3)):
    md.append("\n%s:" % lab)
    rows = []
    for r in runs:
        rows.append([r, grp[r]] + [f(byn(r, n, key), d) for n in NS])
    for gname, gl in (("CONVERGED mean", CONV), ("FAILED mean", FAIL)):
        gl = [r for r in gl if r in R]
        rows.append([gname, ""] + [f(gmean([byn(r, n, key) for r in gl]), d) for n in NS])
    md.append(table(["run", "grp"] + ["n=%d" % n for n in NS], rows))
for r in runs:
    comp["models"][r]["by_n_pass_mean_1_8"] = {str(n): {k: byn(r, n, k) for k in ("L_mean", "alpha_mean", "alpha_effL", "alpha_effL_min_head", "mean_log_sv_det", "top_sv",
                                                                                  "n_sv_lt05", "n_sv_gt09", "beta_mean", "beta_frac_lt0.2", "beta_frac_0.8_1.2", "beta_frac_gt1.8",
                                                                                  "state_retention")} for n in NS}
    comp["models"][r]["by_n_alpha_effL_per_head"] = {str(n): [float(np.exp(np.mean([np.log((R[r]["passes"] if n <= 8 else R[r]["passes_ext"])[t]["by_n"][str(n)]["alpha_effL_per_head"][h]) for t in P8])))
                                                              for h in range(6)] for n in NS if (n <= 8 or "passes_ext" in R[r])}
md.append("\nalpha_eff^L per head at n = 8 (pass-mean 1..8, geometric):")
md.append(table(["run", "grp"] + ["h%d" % h for h in range(6)], [[r, grp[r]] + [f(v) for v in comp["models"][r]["by_n_alpha_effL_per_head"]["8"]] for r in runs]))

# ---- (c) channel ablation
md.append("\n## (c) Aux exact match of e_{t+1} at pass t (items n 4..8 with n >= t; counts 160,160,160,160,128,96,64,32)")
CONDS = [("none|aux_none", "clean"), ("add|all", "ADD ablated, all passes + aux read"), ("delta|all", "DELTA ablated, all passes + aux read"),
         ("add|write_only", "ADD ablated in the think-pass reads only (aux read clean)"), ("delta|write_only", "DELTA ablated in the think-pass reads only"),
         ("none|aux_add", "ADD ablated in the aux read only (clean trajectory)"), ("none|aux_delta", "DELTA ablated in the aux read only")]
for key, lab in CONDS:
    md.append("\n%s:" % lab)
    md.append(table(["run", "grp"] + P8, [[r, grp[r]] + [f(R[r]["aux"][key][t]["acc"]) for t in P8] for r in runs]))
for r in runs:
    comp["models"][r]["aux_acc"] = {key: {t: R[r]["aux"][key][t]["acc"] for t in P8} for key, _ in CONDS}
    comp["models"][r]["aux_n"] = {t: R[r]["aux"]["none|aux_none"][t]["n"] for t in P8}
    comp["models"][r]["aux_role_clean"] = {t: {"handle": R[r]["aux"]["none|aux_none"][t].get("handle_acc"), "copy": R[r]["aux"]["none|aux_none"][t].get("copy_acc")} for t in P8}
md.append("\nclean teacher-forced role accuracy, handle (the newly reduced value) / copy tokens:")
md.append(table(["run", "grp"] + P8, [[r, grp[r]] + ["%s/%s" % (f(R[r]["aux"]["none|aux_none"][t]["handle_acc"], 2), f(R[r]["aux"]["none|aux_none"][t]["copy_acc"])) for t in P8] for r in runs]))

md.append("\nanswer accuracy (n 4..8, 160 items) at K' = 8 (clean also K' 4 / 12):")
rows = []
for r in runs:
    A = R[r]["answer"]
    rows.append([r, grp[r], f(A["none|dec_none"]["4"]["acc"]), f(A["none|dec_none"]["8"]["acc"]), f(A["none|dec_none"]["12"]["acc"]), f(A["add|all"]["8"]["acc"]),
                 f(A["delta|all"]["8"]["acc"]), f(A["add|write_only"]["8"]["acc"]), f(A["delta|write_only"]["8"]["acc"]), f(A["none|dec_add"]["8"]["acc"]), f(A["none|dec_delta"]["8"]["acc"])])
    comp["models"][r]["answer_acc"] = {k: {t: v["acc"] for t, v in d.items()} for k, d in A.items()}
    comp["models"][r]["answer_acc_by_n"] = {k: {t: v["acc_by_n"] for t, v in d.items()} for k, d in A.items()}
md.append(table(["run", "grp", "clean K'4", "clean K'8", "clean K'12", "add all", "delta all", "add write-only", "delta write-only", "add dec-only", "delta dec-only"], rows))

# ---- (d) increments
md.append("\n## (d) Relative state increments ||S_t - S_{t-1}|| / ||S_{t-1}|| (items n 4..8)")
for blk in ("add", "delta"):
    md.append("\n%s block:" % blk)
    md.append(table(["run", "grp"] + P12[1:], [[r, grp[r]] + [f(R[r]["passes"][t]["rel_inc_" + blk]["mean"]) for t in P12[1:]] for r in runs]))
    for r in runs:
        comp["models"][r]["rel_inc_" + blk] = {t: R[r]["passes"][t]["rel_inc_" + blk]["mean"] for t in P12[1:]}
        comp["models"][r]["norm_" + blk] = {t: R[r]["passes"][t]["norm_" + blk]["mean"] for t in P12}
md.append("\ndelta-block cos(S_{t-1}, S_t) by pass:")
md.append(table(["run", "grp"] + P12[1:], [[r, grp[r]] + [f(R[r]["passes"][t]["cos_prev_delta"]["mean"]) for t in P12[1:]] for r in runs]))
md.append("\nnorms ||S_add|| / ||S_delta|| at pass 1, 4, 8, 12:")
md.append(table(["run", "grp"] + ["t=%s" % t for t in ("1", "4", "8", "12")],
                [[r, grp[r]] + ["%.1f / %.2f" % (R[r]["passes"][t]["norm_add"]["mean"], R[r]["passes"][t]["norm_delta"]["mean"]) for t in ("1", "4", "8", "12")] for r in runs]))
for r in runs:
    comp["models"][r]["validate_greedy"] = R[r].get("validate_greedy"); comp["models"][r]["timing"] = R[r]["timing"]; comp["models"][r]["args_subset"] = R[r]["args_subset"]
    comp["models"][r]["gate_params"] = {k: R[r][k] for k in ("gate_bias_alpha", "gate_bias_beta", "gate_w_norm_alpha", "gate_w_norm_beta", "g_add", "g_del", "fade", "gamma")}
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


json.dump(_clean(comp), open(os.path.join(OUT, "EMP_gates.json"), "w", newline="\n"), indent=1, allow_nan=False)
open(os.path.join(OUT, "EMP_gates_tables.md"), "w", newline="\n").write("\n".join(md) + "\n")
print("runs", runs)
