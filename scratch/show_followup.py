import json
import sys

split = sys.argv[1] if len(sys.argv) > 1 else "discovery"
runs = {"s0": "runs/r3_pc_gated_gpt2", "s1": "runs/main_s1", "s2": "runs/main_s2", "p0": "runs/pause_s0",
        "p1": "runs/pause_s1", "p2": "runs/pause_s2"}
f3 = lambda v: f"{v[0]:+.3f}" if isinstance(v, list) else f"{v:+.3f}"
for name, run in runs.items():
    try:
        S = json.load(open(f"{run}/followup/{split}/summary.json"))
    except FileNotFoundError:
        continue
    print(f"\n######## {name} ({S['mode']}, k={S['k']})")
    for W, R in S["results"].items():
        print(f"--- W={W}")
        for d in ("dose_remove", "dose_insert"):
            if d not in R:
                continue
            x = R[d]
            print(f" {d}: clean gap {f3(x['clean_gap'])} competent {x['competent_frac'][0]:.2f} diff_norm {x['diff_norm']:.2f}")
            for kind, kd in x["kinds"].items():
                row = " ".join(f"{lam}:{e['targeted'][0]:.3f}/{e['collateral_mean'][0]:.3f}" for lam, e in kd["doses"].items())
                e1 = kd["doses"]["1.0"]
                print(f"   {kind:8s} norm {kd['proj_frac_norm']:.2f} | tgt/col by dose {row} | rec {e1['recovery_ratio_of_means']:.3f} "
                      f"signed {e1['collateral_signed'][0]:+.3f} worst {e1['collateral_worst'][0]:.3f} competent {e1['targeted_competent'][0]:.3f}")
        if "swap" in R:
            print(" swap (full, dose 1): A_drop / B_rise / col_mean / col_signed / clean A_drop, B_rise, col")
            for c, cd in R["swap"]["conds"].items():
                e = cd["kinds"]["full"]["1.0"]; cl = cd["clean"]
                print(f"   {c:12s} w_don={cd['donor_width']} {e['A_drop'][0]:+.3f} {e['B_rise'][0]:+.3f} {e['collateral_mean'][0]:.3f} {e['collateral_signed'][0]:+.3f}"
                      f" | clean {cl['A_drop'][0]:+.3f} {cl['B_rise'][0]:+.3f} {cl['collateral_mean'][0]:.3f}")
        for j in (2, 3):
            if f"cont_j{j}" in R:
                c = R[f"cont_j{j}"]
                s = c["sets"]
                print(f" cont j={j} n={c['n']}: " + " ".join(f"{st}: nat {s[st]['natural_Nt'][0]:+.3f} full {s[st]['full_Dn'][0]:+.3f} patt {s[st]['pattern_Dn'][0]:+.3f} wrong {s[st]['full_Wr'][0]:+.3f}" for st in ("C_rec", "C_don", "B_shared")),
                      f"| follow full {f3(c['follow_index_full_Dn'])} patt {f3(c['follow_index_pattern_Dn'])} | Nt comp {c['competence']['Nt_C_rec_reachable'][0]:.2f}")
        if "cache" in R:
            c = R["cache"]
            print(f" cache (k={c['k']}) clean {f3(c['clean_gap'])}: " + " ".join(f"{k}:{v['targeted'][0]:.3f}/{v['collateral_mean'][0]:.3f}" for k, v in c["conds"].items()))
            print("   margins: " + " | ".join(f"{k}:" + ",".join(f"{m:.2f}" for m in v["margin_by_step"]) for k, v in c["conds"].items() if k in ("base", "T", "TL", "TG", "clamp", "L", "G", "don")))
            if "interaction_TxL_targeted" in c:
                print(f"   TxL interaction prob {c['interaction_TxL_targeted']:+.3f} logit {c['interaction_TxL_targeted_logit']:+.3f}; TxG prob {c['interaction_TxG_targeted']:+.3f}")
        if "attn" in R:
            print(" attn: " + " | ".join(f"b={b}: self {e['att_boost_self'][0]:+.3f} others {e['att_boost_other_attached'][0]:+.3f} decB->att {e['dec_boost_attached'][0]:+.3f} decB->dec {e['dec_boost_self_decoy'][0]:+.3f} all->att {e['all_boost_attached'][0]:+.3f}" for b, e in R["attn"]["doses"].items()))
        if "gain" in R:
            g = R["gain"]
            print(f" gain: plain tgt {f3(g['plain']['targeted'])} col {g['plain']['collateral_mean'][0]:.3f} acc {g['plain']['accuracy'][0]:.3f} | restored tgt {f3(g['restored']['targeted'])} col {g['restored']['collateral_mean'][0]:.3f} acc {g['restored']['accuracy'][0]:.3f} clean {f3(g['restored']['clean_gap'])}")
        if "jac_rows" in R:
            import numpy as np
            rows = R["jac_rows"]
            a = lambda k: np.array([r_[k] for r_ in rows], float)
            ok = ~np.isnan(a("obs_col_logit"))
            print(f" jac: gain {a('gain').mean():.3f} resid_ratio {a('resid_ratio').mean():.3f} | corr(pred,obs) target {np.corrcoef(a('pred_target_logit'), a('obs_target_logit'))[0, 1]:.2f}"
                  f" col {np.corrcoef(a('pred_col_logit')[ok], a('obs_col_logit')[ok])[0, 1]:.2f} | mean pred/obs target {a('pred_target_logit').mean():+.3f}/{a('obs_target_logit').mean():+.3f}")
