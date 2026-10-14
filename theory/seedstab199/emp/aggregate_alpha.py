"""emp_alpha_pass_<run>.json -> EMP_alpha_pass.json keyed [model][n] (+ markdown tables EMP_alpha_pass_tables.md)."""
import os
import json
import glob
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = ["s182_window_s1", "s182_window_s2", "s184_reweight_s2", "s185_den_s1", "s185_rms_s1", "s185_rms_s2", "s185_rms_s3", "s187_nopass_s1", "s187_nopass_s2",
        "s188_rwwin_s1", "s190_rmsnopass_s1", "s190_rmsnopass_s2", "s182_hold_s1"]
tab = {}; meta = {}
for r in RUNS:
    fn = os.path.join(OUT, "emp_alpha_pass_%s.json" % r)
    if not os.path.exists(fn):
        continue
    d = json.load(open(fn)); tab[r] = {}
    for n, v in d["by_n"].items():
        tab[r][n] = {k: v[k] for k in ("alpha_pass", "mean_log_sv_pass", "alpha_pass_min_head", "alpha_pass_per_head", "mean_log_sv_pass_per_head", "alpha_token_mean",
                                        "alpha_token_geomean", "L_mean", "top_sv_pass", "n_sv_lt05_pass", "mean_log_sv_svd_pass", "alpha_pass_per_t", "mean_log_sv_per_t", "n_items")}
    f2 = os.path.join(OUT, "emp_alpha_pass_v2_%s.json" % r)              # v2: same items / seed, adds open-loop state retention, |P|_F^2/64, median log sv
    if os.path.exists(f2):
        d2 = json.load(open(f2))
        for n, v in d2["by_n"].items():
            assert abs(v["alpha_pass"] - tab[r][n]["alpha_pass"]) < 1e-6 and abs(v["mean_log_sv_pass"] - tab[r][n]["mean_log_sv_pass"]) < 1e-6, ("v2 != v1", r, n)
            tab[r][n].update({k: v[k] for k in ("state_retention_pass", "state_retention_per_t", "frob2_pass", "median_log_sv_pass")})
    pi = None
    try:
        e = json.load(open(os.path.join(r"D:\Latent\boxes\runs_fsq", r, "eval.json")))
        pi = e.get("s182", {}).get("path_indep")
    except Exception:  # noqa: BLE001
        pass
    meta[r] = {"replay_rel_err_max": d["replay_rel_err_max"], "fade": d["fade"], "gamma": d["gamma"], "timing_s": d["timing_s"], "items": d["items"],
               "path_indep_from_eval_json": ({k: pi.get(k) for k in ("delta_cos_final", "answer_agree", "acc_from_zero", "acc_from_donor")} if pi else None)}
out = {"defs": {"alpha_pass": "exp(mean over items of that n, the 6 delta heads and passes t = 1..n of sum over real think tokens of log alpha_t)",
                "mean_log_sv_pass": "mean over items, heads and passes t = 1..n of log|det P_t| / 64, P_t = prod_tokens alpha (I - beta k k^T) (analytic, = mean log singular value)",
                "items": "n = 9..12, 64 per n, numpy default_rng(19615), boxes.nested170.generate; clean trajectory, K = 12, CPU fp32"},
       "table": tab, "meta": meta}
json.dump(out, open(os.path.join(OUT, "EMP_alpha_pass.json"), "w", newline="\n"), indent=1)
NS = ["9", "10", "11", "12"]; md = []
for key, lab in (("alpha_pass", "alpha_pass (geometric-mean per-pass delta decay over passes 1..n)"), ("mean_log_sv_pass", "mean log singular value of P per pass, passes 1..n"),
                 ("alpha_pass_min_head", "alpha_pass of the most-decaying head"), ("alpha_token_mean", "token-mean alpha"), ("L_mean", "real think tokens L"),
                 ("top_sv_pass", "top singular value of P"), ("n_sv_lt05_pass", "# singular values < 0.5"),
                 ("state_retention_pass", "open-loop retention of the actual previous delta state ||S_{t-1} P_t|| / ||S_{t-1}||, passes 2..n"),
                 ("frob2_pass", "|P|_F^2 / 64, passes 1..n"), ("median_log_sv_pass", "median log singular value of P, passes 1..n")):
    md.append("\n%s:\n" % lab)
    md.append("| run | delta_cos_final | " + " | ".join("n=%s" % n for n in NS) + " |"); md.append("|---" * (len(NS) + 2) + "|")
    for r in tab:
        pi = meta[r]["path_indep_from_eval_json"]; dc = ("%.3f" % pi["delta_cos_final"]) if pi and pi.get("delta_cos_final") is not None else "-"
        md.append("| %s | %s | " % (r, dc) + " | ".join(("%.3f" % tab[r][n][key]) if (n in tab[r] and key in tab[r][n]) else "-" for n in NS) + " |")
open(os.path.join(OUT, "EMP_alpha_pass_tables.md"), "w", newline="\n").write("\n".join(md) + "\n")
print("models", list(tab))
