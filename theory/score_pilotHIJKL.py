"""Score THEORY §27 (Pilot H: PH1–PH4), §28 (Pilot I: PI1–PI3), §29 (Pilot J: PJ1–PJ3), §31 (Pilot K: PK1–PK4) and
§32 (Pilot L: PL1–PL4), each exactly as registered.

Inputs: <run>/eval.json (theory/eval_arch.py); pointer trajectories p[0..11] from theory/results/le_pointer.json
(theory/le_pointer.py; Pilot D controls/anytime from theory/results/pilotDE_meas.json, same ptr_traj on probe(512));
initial P30 models' delayed bal-acc from results/competence_p30_s*.json (latent.competence, the function eval_arch
calls; budget K, first element). Pair rule and eval reader from theory/score_pilotFG.py.
    python theory/score_pilotHIJKL.py H L        # any of H I J K L
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pilotFG import ev, pair_rule  # noqa: E402

SEEDS = range(900, 908)
H_MODELS = [102, 109, 111, 112, 113, 115, 118, 119, 125, 126, 130, 131, 132, 135, 140, 145, 146, 147, 157, 158]
RETRIEVAL = {111, 112, 113, 118, 125, 130, 131, 132, 140, 146, 147, 157}
LOCAL = set(H_MODELS) - RETRIEVAL
PTR = json.load(open("theory/results/le_pointer.json")) if os.path.exists("theory/results/le_pointer.json") else {}
DEM = json.load(open("theory/results/pilotDE_meas.json")) if os.path.exists("theory/results/pilotDE_meas.json") else {}
med = lambda v: float(np.median(v)) if v else None
share = lambda v: float(np.mean(v)) if v else None


def ptr(r):
    if r in PTR:
        return PTR[r]["ptr"], PTR[r]["K"]
    if r in DEM and "ptr" in DEM[r]:
        return DEM[r]["ptr"], DEM[r]["K"]
    return None, None


def d7(r):
    e = ev(r)
    if e is None:
        return None
    return json.load(open(f"theory/runs/{r}/eval.json"))["d7_K12"]


def arm_summary(runs):
    E = {r: ev(r) for r in runs}
    got = [r for r in runs if E[r] is not None]
    comp = [r for r in got if E[r]["gate"]]
    return {"registered": len(runs), "evaluated": len(got), "competent": len(comp),
            "extrapolate_competent": sum(E[r]["extrap"] for r in comp),
            "extrapolate_all": sum(E[r]["extrap"] for r in got),
            "median_d8": med([E[r]["d8"] for r in comp]), "median_d7": med([d7(r) for r in comp]),
            "missing": [r for r in runs if E[r] is None]}


def pilot_H():
    arms = {"FT-C": [f"ph_c_s{s}" for s in H_MODELS], "FT-AT": [f"ph_at_s{s}" for s in H_MODELS],
            "FT-JAT": [f"ph_jat_s{s}" for s in sorted(LOCAL)]}
    out = {"arms": {a: arm_summary(r) for a, r in arms.items()}}
    E = {r: ev(r) for rs in arms.values() for r in rs}
    sh = lambda a: (out["arms"][a]["extrapolate_competent"] / out["arms"][a]["competent"]) if out["arms"][a]["competent"] else None
    sAT, sC = sh("FT-AT"), sh("FT-C")
    both = [s for s in H_MODELS if E[f"ph_at_s{s}"] and E[f"ph_c_s{s}"] and E[f"ph_at_s{s}"]["gate"] and E[f"ph_c_s{s}"]["gate"]]
    dd = [(d7(f"ph_at_s{s}"), d7(f"ph_c_s{s}")) for s in both]
    w = sum(x > y for x, y in dd)
    met = sAT is not None and sC is not None and sAT >= 0.40 and sC <= 0.15 and dd and w / len(dd) >= 0.75
    out["PH1"] = {"share_AT": sAT, "share_C": sC, "d7_pairs_both_competent": len(dd), "d7_AT_gt_C": w,
                  "met": bool(met),
                  "critical_period_reading": (sAT is not None and sC is not None and sAT <= sC + 0.10)}
    pk = lambda a: [ptr(r)[0][ptr(r)[1] - 1] for r in arms[a] if E[r] and E[r]["gate"] and ptr(r)[0]]
    pa, pc = pk("FT-AT"), pk("FT-C")
    out["PH2"] = {"median_pK1_AT": med(pa), "n_AT": len(pa), "median_pK1_C": med(pc), "n_C": len(pc),
                  "met": (med(pa) is not None and med(pc) is not None and med(pa) >= 4 and med(pc) <= 3)}
    def sub(group):
        c = [s for s in group if E[f"ph_at_s{s}"] and E[f"ph_at_s{s}"]["gate"]]
        return (share([E[f"ph_at_s{s}"]["extrap"] for s in c]), len(c))
    (sr, nr), (sl, nl) = sub(RETRIEVAL), sub(LOCAL)
    out["PH3"] = {"AT_share_retrieval": sr, "n_retrieval": nr, "AT_share_local": sl, "n_local": nl,
                  "met": (sr is not None and sl is not None and sr - sl >= 0.25)}
    pj = [(d7(f"ph_jat_s{s}"), d7(f"ph_at_s{s}")) for s in sorted(LOCAL)
          if E[f"ph_jat_s{s}"] and E[f"ph_at_s{s}"] and E[f"ph_jat_s{s}"]["gate"] and E[f"ph_at_s{s}"]["gate"]]
    cnt = [(x, y) for x, y in pj if x != y]
    wj = sum(x > y for x, y in cnt)
    out["PH4"] = {"counted": len(cnt), "JAT_gt_AT": wj,
                  "verdict": "unresolved (< 5 counted)" if len(cnt) < 5 else ("supported" if wj >= 6 else "not supported")}
    out["guard"] = {a: {"gate_rate": (v["competent"] / v["evaluated"]) if v["evaluated"] else None,
                        "flagged": (v["competent"] / v["evaluated"] < 0.75) if v["evaluated"] else None}
                    for a, v in out["arms"].items()}
    return out


def pilot_L():
    arms = {k: [f"pl_{k}_s{s}" for s in H_MODELS] for k in ("t8", "self", "gt")}
    out = {"arms": {a: arm_summary(r) for a, r in arms.items()}}
    E = {r: ev(r) for rs in arms.values() for r in rs}
    out["arms"]["FT-C (Pilot H)"] = arm_summary([f"ph_c_s{s}" for s in H_MODELS])
    pairs = [(E[f"pl_t8_s{s}"]["d8"], E[f"pl_self_s{s}"]["d8"]) for s in H_MODELS
             if E[f"pl_t8_s{s}"] and E[f"pl_self_s{s}"] and E[f"pl_t8_s{s}"]["gate"] and E[f"pl_self_s{s}"]["gate"]]
    out["PL1"] = pair_rule(pairs, lambda x, y: x > y)
    t8 = sum(bool(E[f"pl_t8_s{s}"] and E[f"pl_t8_s{s}"]["extrap"]) for s in H_MODELS) / len(H_MODELS)
    ftc = sum(bool(ev(f"ph_c_s{s}") and ev(f"ph_c_s{s}")["extrap"]) for s in H_MODELS) / len(H_MODELS)
    out["PL2"] = {"T8_extrap_share_of_20": t8, "FTC_extrap_share_of_20": ftc, "met": t8 >= 0.50 and t8 - ftc >= 0.30}
    mg, mt = out["arms"]["gt"]["median_d8"], out["arms"]["t8"]["median_d8"]
    out["PL3"] = {"median_d8_GT": mg, "median_d8_T8": mt, "GT_ge_T8": (mg >= mt) if mg is not None and mt is not None else None}
    drops = []
    for s in H_MODELS:
        e = E[f"pl_t8_s{s}"]
        f = f"results/competence_p30_s{s}.json"
        if e is None or not os.path.exists(f):
            continue
        c0 = json.load(open(f))
        c0 = c0[0] if isinstance(c0, list) else c0
        K = c0["config"]["K"]
        drops.append(c0["budgets"][str(K)]["delayed"]["bal_acc"][0] - e["delayed"])
    out["PL4"] = {"n": len(drops), "median_drop_delayed": med(drops), "met": (med(drops) <= 0.02) if drops else None}
    return out


def pd_arm(prefix):
    return {s: ev(f"{prefix}_s{s}") for s in SEEDS}


def pilot_K():
    C, AT, RT = pd_arm("pd_c"), pd_arm("pd_at"), pd_arm("pk_rt")
    out = {"arms": {"RT": arm_summary([f"pk_rt_s{s}" for s in SEEDS]), "C": arm_summary([f"pd_c_s{s}" for s in SEEDS]),
                    "AT": arm_summary([f"pd_at_s{s}" for s in SEEDS])}}
    pr = lambda X, Y: [(X[s]["d8"], Y[s]["d8"]) for s in SEEDS if X[s] and Y[s] and X[s]["gate"] and Y[s]["gate"]]
    out["PK1"] = pair_rule(pr(RT, C), lambda x, y: x > y)
    out["AT_vs_C_same_seeds"] = pair_rule(pr(AT, C), lambda x, y: x > y)
    a = out["arms"]
    shr = lambda v: v["extrapolate_competent"] / v["competent"] if v["competent"] else None
    md = (a["RT"]["median_d8"] - a["AT"]["median_d8"]) if a["RT"]["median_d8"] is not None and a["AT"]["median_d8"] is not None else None
    out["PK2"] = {"median_d8_RT_minus_AT": md, "extrap_share_RT": shr(a["RT"]), "extrap_share_AT": shr(a["AT"]),
                  "met": (md is not None and abs(md) <= 0.05 and shr(a["RT"]) is not None and shr(a["AT"]) is not None
                          and shr(a["RT"]) >= shr(a["AT"]) - 0.15)}
    pk = [ptr(f"pk_rt_s{s}")[0][5] for s in SEEDS if RT[s] and RT[s]["gate"] and ptr(f"pk_rt_s{s}")[0]]
    out["PK3"] = {"n": len(pk), "median_pK1": med(pk), "met": (med(pk) >= 4) if pk else None}
    out["PK4"] = {"competent_RT": a["RT"]["competent"], "competent_C": a["C"]["competent"],
                  "met": a["RT"]["competent"] >= a["C"]["competent"] - 2}
    return out


def advancing(p, K):
    return max(p[K:12]) >= p[K - 1] + 2 and all(p[i] >= p[i - 1] for i in range(K, 12))


def pilot_I():
    C = pd_arm("pd_c")
    P = {a: pd_arm(f"pi_{a}") for a in ("per2", "per3")}
    out = {"arms": {a: arm_summary([f"pi_{a}_s{s}" for s in SEEDS]) for a in P}}
    out["arms"]["C"] = arm_summary([f"pd_c_s{s}" for s in SEEDS])
    comp_per = [(f"pi_{a}_s{s}", s) for a in P for s in SEEDS if P[a][s] and P[a][s]["gate"]]
    adv = [advancing(*ptr(r)) for r, _ in comp_per if ptr(r)[0]]
    seeds = sorted({s for _, s in comp_per})
    adv_c = [advancing(*ptr(f"pd_c_s{s}")) for s in seeds if C[s] and C[s]["gate"] and ptr(f"pd_c_s{s}")[0]]
    out["PI1"] = {"n_PER": len(adv), "share_PER": share(adv), "n_C_same_seeds": len(adv_c), "share_C": share(adv_c),
                  "met": (share(adv) is not None and share(adv_c) is not None and share(adv) >= 0.60
                          and share(adv) - share(adv_c) >= 0.30)}
    pairs = [(P[a][s]["far8_10_K16"], C[s]["far8_10_K16"]) for a in P for s in SEEDS
             if P[a][s] and C[s] and P[a][s]["gate"] and C[s]["gate"]]
    out["PI2"] = pair_rule(pairs, lambda x, y: x > y)
    kept = []
    for r, _ in comp_per:
        p, K = ptr(r)
        if p:
            v_in, v_post = p[K - 1] / K, (p[11] - p[K - 1]) / (11 - (K - 1))
            kept.append(abs(v_post - v_in) <= 0.25)
    out["PI3"] = {"n": len(kept), "share_kept": share(kept), "met": (share(kept) >= 0.60) if kept else None}
    return out


def pilot_J():
    C, AT = pd_arm("pd_c"), pd_arm("pd_at")
    J = {"ch3": pd_arm("pj_ch3"), "ch3at": pd_arm("pj_ch3at")}
    out = {"arms": {a: arm_summary([f"pj_{a}_s{s}" for s in SEEDS]) for a in J}}
    out["arms"]["C"] = arm_summary([f"pd_c_s{s}" for s in SEEDS]); out["arms"]["AT"] = arm_summary([f"pd_at_s{s}" for s in SEEDS])
    cj = [(f"pj_{a}_s{s}", s) for a in J for s in SEEDS if J[a][s] and J[a][s]["gate"]]
    pj = [max(ptr(r)[0]) >= 8 for r, _ in cj if ptr(r)[0]]
    seeds = sorted({s for _, s in cj})
    base = [max(ptr(f"{pf}_s{s}")[0]) >= 8 for pf, X in (("pd_c", C), ("pd_at", AT)) for s in seeds
            if X[s] and X[s]["gate"] and ptr(f"{pf}_s{s}")[0]]
    out["PJ1"] = {"n_J": len(pj), "share_J": share(pj), "n_base_same_seeds": len(base), "share_base": share(base),
                  "met": (share(pj) is not None and share(base) is not None and share(pj) >= 0.50 and share(pj) > share(base))}
    pairs = [(J["ch3"][s]["d8"], C[s]["d8"]) for s in SEEDS if J["ch3"][s] and C[s] and J["ch3"][s]["gate"] and C[s]["gate"]] + \
            [(J["ch3at"][s]["d8"], AT[s]["d8"]) for s in SEEDS if J["ch3at"][s] and AT[s] and J["ch3at"][s]["gate"] and AT[s]["gate"]]
    out["PJ2"] = pair_rule(pairs, lambda x, y: x > y)
    a = out["arms"]
    out["PJ3"] = {"CH3_vs_C": [a["ch3"]["competent"], a["C"]["competent"]], "CH3AT_vs_AT": [a["ch3at"]["competent"], a["AT"]["competent"]],
                  "met": abs(a["ch3"]["competent"] - a["C"]["competent"]) <= 2 and abs(a["ch3at"]["competent"] - a["AT"]["competent"]) <= 2}
    return out


def main(which):
    fns = {"H": pilot_H, "I": pilot_I, "J": pilot_J, "K": pilot_K, "L": pilot_L}
    for w in which:
        o = fns[w]()
        json.dump(o, open(f"theory/results/score_pilot{w}.json", "w", newline="\n"), indent=1)
        print(f"==== Pilot {w}"); print(json.dumps(o, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["H", "I", "J", "K", "L"])
