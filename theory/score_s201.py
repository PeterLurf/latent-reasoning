"""THEORY §201 scoring (OVERTHINK-EXIT; rules fixed in §201 before any output; Theory is the only scorer). Accuracies are fractions.
T1 <run>/eval_s201_freeze.json = {"arms": {"CLEAN"|"FA"|"FD"|"FB": {"acc": {"16"|"24"|"32": {"pool_3_8": x, "pool_9_10": y}}}},
                                  "gates": {"fb_identity": bool, "clean_vs_s173_maxdiff": float}}
   At K'24 pool_3_8: D = FB - CLEAN; D < 0.03 -> NOTHING-TO-RECOVER (no-harm: |FA - CLEAN|, |FD - CLEAN| <= 0.01); else R_FA, R_FD ->
   ADD-COUPLED (R_FA >= 0.7, R_FD >= 0.5) / ADD-DIRECT (R_FA >= 0.7, R_FD < 0.5) / DELTA-INTRINSIC (R_FD >= 0.7, R_FA <= 0.3) / JOINT (both <= 0.3) / mixed.
   Falsifier: >= 2 of {rms s1, rms s2, §190 s1} with R_FA <= 0.3.
T2 <run>/eval_s201_halt.json = {"primary": {"acc_by_n": {n: acc}, "event_halt_mean_by_n": {n: t}, "censored_by_n": {n: share},
                                            "halt_t_hist_by_n": {n: {t: count}}},
                                "best_fixed_k_train": K, "acc_by_n_kprime": {K': {n: acc}}}
   NO-OVERTHINK: pooled 3..8 >= pooled(best fixed K') - 0.01; SIZE-FREE at n 9 and 10 (acc >= acc(K' n+2) - 0.03, event mean in [n-1, n+3], censored <= 0.10);
   PREMATURE: >= 0.05 below K' n+2 at any n in 3..10. T2-c: PREMATURE or censored share at n 10 > 0.10. Beta-cue falsifier (both RMS seeds):
   >= 50% of n = 10 halts at t in {8, 9} AND the rule's n = 10 accuracy <= acc(K'12 at n 10) - 0.05.
    python theory/score_s201.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
NESTED = ["s185_rms_s1", "s185_rms_s2", "s190_rmsnopass_s1", "s187_nopass_s1", "s187_nopass_s2", "s184_reweight_s1", "s184_reweight_s2"]
CONTROL = "s182_window_s2"
EPS = 1e-9


def jl(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def t1(d):
    j = jl(d, "eval_s201_freeze.json")
    if j is None:
        return None
    a = {k: j["arms"][k]["acc"]["24"]["pool_3_8"] for k in ("CLEAN", "FA", "FD", "FB")}; D = a["FB"] - a["CLEAN"]
    out = {"acc_k24": a, "D": D, "gates": j.get("gates")}
    if D < 0.03 - EPS:
        out["reading"] = "NOTHING-TO-RECOVER" + ("" if abs(a["FA"] - a["CLEAN"]) <= 0.01 + EPS and abs(a["FD"] - a["CLEAN"]) <= 0.01 + EPS else " (no-harm FAILED)")
        return out
    rfa, rfd = (a["FA"] - a["CLEAN"]) / D, (a["FD"] - a["CLEAN"]) / D; out.update(R_FA=rfa, R_FD=rfd)
    out["reading"] = ("ADD-COUPLED" if rfa >= 0.7 - EPS and rfd >= 0.5 - EPS else "ADD-DIRECT" if rfa >= 0.7 - EPS else
                      "DELTA-INTRINSIC" if rfd >= 0.7 - EPS and rfa <= 0.3 + EPS else "JOINT" if rfa <= 0.3 + EPS and rfd <= 0.3 + EPS else "mixed")
    return out


def pool(accn):
    return sum(accn[str(n)] for n in range(3, 9)) / 6


def t2(d):
    j = jl(d, "eval_s201_halt.json")
    if j is None:
        return None
    p = j["primary"]; kp = j["acc_by_n_kprime"]; bk = str(j["best_fixed_k_train"])
    no_ot = pool(p["acc_by_n"]) >= pool(kp[bk]) - 0.01 - EPS
    sf = {}
    for n in (9, 10):
        ref = kp[str(n + 2)][str(n)]
        sf[n] = (p["acc_by_n"][str(n)] >= ref - 0.03 - EPS and n - 1 - EPS <= p["event_halt_mean_by_n"][str(n)] <= n + 3 + EPS and p["censored_by_n"][str(n)] <= 0.10 + EPS)
    prem = any(p["acc_by_n"][str(n)] <= kp[str(n + 2)][str(n)] - 0.05 + EPS for n in range(3, 11))
    h = p["halt_t_hist_by_n"].get("10", {}); tot = sum(h.values()); early = sum(v for t, v in h.items() if int(t) in (8, 9))
    beta_fal = tot > 0 and early / tot >= 0.5 - EPS and p["acc_by_n"]["10"] <= kp["12"]["10"] - 0.05 + EPS
    return {"NO-OVERTHINK": no_ot, "SIZE-FREE": sf, "PREMATURE": prem, "beta_cue_falsifier_condition": beta_fal, "share_n10_halts_t8_9": early / tot if tot else None, "censored_n10": p["censored_by_n"]["10"]}


def main():
    T1 = {d: t1(d) for d in NESTED + [CONTROL]}; T2 = {d: t2(d) for d in NESTED}; out = {"T1": T1, "T2": T2}; lines = []
    if all(T1[d] for d in NESTED):
        key = ["s185_rms_s1", "s185_rms_s2", "s190_rmsnopass_s1"]
        fal = sum(1 for d in key if T1[d].get("R_FA") is not None and T1[d]["R_FA"] <= 0.3 + EPS) >= 2
        a = all(T1[d]["reading"] == "ADD-COUPLED" for d in key)
        b = all(T1[d]["reading"] in ("ADD-DIRECT", "ADD-COUPLED") and T1[d].get("R_FD", 1) < T1[d].get("R_FA", 0) for d in ("s184_reweight_s1", "s184_reweight_s2"))
        out["T1_summary"] = {"additive_account_FALSIFIED": fal, "T1-a": a, "T1-b": b}
        lines.append("T1: %s; T1-a %s, T1-b %s :: %s" % ("additive account FALSIFIED" if fal else "additive account not falsified", a, b,
                     ", ".join("%s %s" % (d, T1[d]["reading"]) for d in NESTED)))
    else:
        lines.append("T1 pending (%s)" % [d for d in NESTED if not T1[d]])
    if all(T2[d] for d in NESTED):
        ta = all(T2[d]["NO-OVERTHINK"] for d in NESTED)
        tb = T2["s185_rms_s1"]["SIZE-FREE"][9] and T2["s185_rms_s2"]["SIZE-FREE"][9] and T2["s185_rms_s2"]["SIZE-FREE"][10]
        tc = all(T2[d]["PREMATURE"] or T2[d]["censored_n10"] > 0.10 + EPS for d in ("s184_reweight_s1", "s184_reweight_s2"))
        bf = T2["s185_rms_s1"]["beta_cue_falsifier_condition"] and T2["s185_rms_s2"]["beta_cue_falsifier_condition"]
        out["T2_summary"] = {"T2-a": ta, "T2-b": tb, "T2-c": tc, "beta_cue_FALSIFIED": bf}
        lines.append("T2: T2-a %s, T2-b %s, T2-c %s, beta cue %s" % (ta, tb, tc, "FALSIFIED" if bf else "not falsified"))
    else:
        lines.append("T2 pending (%s)" % [d for d in NESTED if not T2[d]])
    out["reading"] = " | ".join(lines); print("READING:", out["reading"])
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s201.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
