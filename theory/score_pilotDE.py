"""Score THEORY §17 (Pilot D: PD1–PD4), §20 (Pilot E: PE1–PE4, amendments D-1, E-1, E-2) and §25 (LE1–LE4).

Measurements (cached in theory/results/pilotDE_meas.json; CPU, THEORY_THREADS threads):
  * eval.json of every run (gate, extrap_K12, d7, far8–10, length_gen, matched, final_stage)
  * competent runs: pointer trajectory (analyse_pilotA.ptr_traj, probe(512), K' = 12), query_eval base and with the
    layer-2 latent->latent block from step K (probe(512), K' = 12)  [E-2]
  * transfer-curve index z for Pilot D runs (theory/transfer_pairs.py, run separately; read if present)  [LE4]
Queued runs without eval.json whose stage-1 guard was exhausted count as run and not competent (§15 rule).
    python theory/score_pilotDE.py            # measure what is missing, then score
    python theory/score_pilotDE.py --no-measure
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))

MEAS, OUT = "theory/results/pilotDE_meas.json", "theory/results/score_pilotDE.json"
PAIRS = ([("D", s, f"pd_at_s{s}", f"pd_c_s{s}") for s in range(900, 912)] +
         [("E8", s, f"pe8_at_s{s}", f"pe8_c_s{s}") for s in range(1000, 1008)] +
         [("E3", s, f"pe3_at_s{s}", f"pe3_c_s{s}") for s in range(1010, 1018)])


def competent(arm, ev):
    if ev is None:
        return False
    if arm == "E3":                         # §20: final stage full and matched 4-hop bal-acc >= 0.60 (R rule, cap 3)
        return ev["final_stage"] == "full" and ev["matched"].get("4", 0) >= 0.60
    return bool(ev["gate"])


def bal(q, key_pos, key_neg, j):
    return float(0.5 * (q[key_pos][j] + 1 - q[key_neg][j]))


def measure(names):
    from analyse_pilotA import ptr_traj
    from eval_arch import load_arch
    from rollout import probe, query_eval
    res = json.load(open(MEAS)) if os.path.exists(MEAS) else {}
    g, br, E = probe(512)
    for arm, r in names:
        path = f"theory/runs/{r}"
        if not os.path.exists(f"{path}/eval.json"):
            continue
        ev = json.load(open(f"{path}/eval.json"))
        row = res.get(r, {})
        row["eval"] = {k: ev[k] for k in ("final_stage", "gate", "extrap_K12", "d7_K12", "far8_10_K12", "matched", "delayed")}
        lg = next(x for x in ev["length_gen"]["rows"] if x["K'"] == 12)
        row["d8_lg"] = lg["far"]["8"]
        if competent(arm, ev) and "ptr" not in row:
            m, c = load_arch(path, "cpu")
            K = c["K"]
            with torch.no_grad():
                p, a, head = ptr_traj(m, K, g, br, E, {})
                qb, _ = query_eval(m, K, g, br, 12, "cpu")
                qk, _ = query_eval(m, K, g, br, 12, "cpu", block_from=K, block_layers=(1,))
            row.update(K=K, ptr=p, amp=a, head=head,
                       d8_probe=bal(qb, "B", "C", 1), d8_probe_block=bal(qk, "B", "C", 1),
                       d6_probe=bal(qb, "A", "D", 5), d6_probe_block=bal(qk, "A", "D", 5))
            print(r, "K", K, "ptr", " ".join(map(str, p)), "d8 %.3f->%.3f d6 %.3f->%.3f" % (
                row["d8_probe"], row["d8_probe_block"], row["d6_probe"], row["d6_probe_block"]), flush=True)
        res[r] = row
        json.dump(res, open(MEAS, "w", newline="\n"), indent=1)
    return res


def fisher_one_sided(a, b, c, d):
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] (rows AT, C; columns extrap, not)."""
    from math import comb
    n1, n2, k = a + b, c + d, a + c
    tot = comb(n1 + n2, k)
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot


def sign_p(wins, n):
    from math import comb
    return sum(comb(n, x) for x in range(wins, n + 1)) / 2 ** n if n else None


def score(M):
    queued = set()
    for f in glob.glob("theory/queues/pilotDE_h*.txt") + glob.glob("theory/queues/pilotD_h*.txt"):
        queued |= {l.split()[0] for l in open(f) if l.strip() and not l.startswith("#")}
    T = json.load(open("theory/results/transfer_pilots.json")) if os.path.exists("theory/results/transfer_pilots.json") else {}

    def info(arm, r):
        row = M.get(r)
        if row is None:
            return {"ran": r in queued, "comp": False, "eval": None}
        ev = row["eval"]
        comp = competent(arm, {**ev, "matched": ev["matched"]})
        beyond = row["d8_lg"] if arm in ("D", "E8") else row.get("d6_probe")
        return {"ran": True, "comp": comp, "eval": ev, "beyond": beyond, "row": row}

    out = {"pairs": []}
    for arm, s, at, c in PAIRS:
        A, C = info(arm, at), info(arm, c)
        out["pairs"].append({"arm": arm, "seed": s, "at": at, "c": c, "at_comp": A["comp"], "c_comp": C["comp"],
                             "at_beyond": A.get("beyond"), "c_beyond": C.get("beyond"),
                             "at_extrap": A["eval"]["extrap_K12"] if A["eval"] else None,
                             "c_extrap": C["eval"]["extrap_K12"] if C["eval"] else None})
    P = out["pairs"]

    # ---- Pilot D (§17 with D-1)
    D = [x for x in P if x["arm"] == "D"]
    def arm_stats(key_comp, key_ext, runs):
        comp = [x for x in runs if x[key_comp]]
        return len(runs), len(comp), sum(bool(x[key_ext]) for x in comp)
    nA, cA, eA = arm_stats("at_comp", "at_extrap", D); nC, cC, eC = arm_stats("c_comp", "c_extrap", D)
    sA = eA / cA if cA else None; sC = eC / cC if cC else None
    if cA < 6 or cC < 6:
        pd1 = "unresolved (< 6 competent in an arm)"
    elif sA - sC >= 0.20 and sA >= 0.60:
        pd1 = "supported"
    elif sA <= sC:
        pd1 = "not supported"
    else:
        pd1 = "inconclusive"
    out["PD1"] = {"AT": [nA, cA, eA], "C": [nC, cC, eC], "share_AT": sA, "share_C": sC, "verdict": pd1,
                  "fisher_p_one_sided": fisher_one_sided(eA, cA - eA, eC, cC - eC) if cA and cC else None}
    far = lambda runs, k: [M[x[k]]["eval"]["far8_10_K12"] for x in runs if x[k + "_comp"]]
    fA, fC = far(D, "at"), far(D, "c")
    out["PD2"] = {"mean_far_AT": float(np.mean(fA)) if fA else None, "mean_far_C": float(np.mean(fC)) if fC else None}
    out["PD2"]["met"] = (out["PD2"]["mean_far_AT"] - out["PD2"]["mean_far_C"] >= 0.05) if fA and fC else None

    def every_step(p, K):
        rep = 0
        for i in range(2, K + 1):
            if p[i] == p[i - 1] + 1:
                continue
            if p[i] == p[i - 1] and rep == 0:
                rep += 1; continue
            return False
        return True
    adv = lambda runs, k: [every_step(M[x[k]]["ptr"], M[x[k]]["K"]) for x in runs if x[k + "_comp"]]
    aA, aC = adv(D, "at"), adv(D, "c")
    out["PD3"] = {"share_AT": float(np.mean(aA)) if aA else None, "share_C": float(np.mean(aC)) if aC else None}
    out["PD3"]["met"] = (out["PD3"]["share_AT"] >= 0.60 and out["PD3"]["share_C"] <= 0.30) if aA and aC else None
    out["PD4"] = {"gate_rate_AT": cA / nA if nA else None, "gate_rate_C": cC / nC if nC else None}
    out["PD4"]["met"] = (cA / nA >= cC / nC - 0.25) if nA and nC else None

    # ---- Pilot E (§20 with E-1, E-2)
    ev_pairs = [x for x in P if x["at_comp"] and x["c_comp"] and x["at_beyond"] is not None and x["c_beyond"] is not None]
    untied = [x for x in ev_pairs if x["at_beyond"] != x["c_beyond"]]
    wins = sum(x["at_beyond"] > x["c_beyond"] for x in untied)
    E83_at = [x for x in P if x["arm"] in ("E8", "E3") and x["at_comp"]]
    cross_at = [x["at_beyond"] >= 0.60 for x in E83_at]
    share_cross = float(np.mean(cross_at)) if cross_at else None
    if len(untied) < 8:
        pe1 = "unresolved (< 8 evaluable pairs)"
    elif wins / len(untied) >= 0.70 and sign_p(wins, len(untied)) < 0.05 and share_cross is not None and share_cross >= 0.50:
        pe1 = "SE supported"
    elif share_cross is not None and share_cross <= 0.20:
        pe1 = "DW supported"
    else:
        pe1 = "inconclusive"
    out["PE1"] = {"evaluable_pairs": len(untied), "ties": len(ev_pairs) - len(untied), "anytime_wins": wins,
                  "sign_p_one_sided": sign_p(wins, len(untied)), "E8E3_anytime_competent": len(cross_at),
                  "E8E3_anytime_cross_share": share_cross, "verdict": pe1,
                  "by_arm": {a: {"evaluable": sum(1 for x in untied if x["arm"] == a),
                                 "wins": sum(1 for x in untied if x["arm"] == a and x["at_beyond"] > x["c_beyond"])}
                             for a in ("D", "E8", "E3")}}
    def cross_share(a, k):
        v = [x[k + "_beyond"] >= 0.60 for x in P if x["arm"] == a and x[k + "_comp"] and x[k + "_beyond"] is not None]
        return (float(np.mean(v)) if v else None, len(v))
    out["PE2"] = {a: {"AT": cross_share(a, "at"), "C": cross_share(a, "c")} for a in ("D", "E8", "E3")}
    def passes(x, k):
        row = M[x[k]]; X = 6 if x["arm"] == "E3" else 8
        return max(row["ptr"]) >= X
    pa = [passes(x, "at") for x in P if x["at_comp"]]; pc = [passes(x, "c") for x in P if x["c_comp"]]
    out["PE3"] = {"AT_share": float(np.mean(pa)) if pa else None, "C_share": float(np.mean(pc)) if pc else None,
                  "n": [len(pa), len(pc)]}
    out["PE3"]["met"] = (out["PE3"]["AT_share"] >= 0.60 and out["PE3"]["C_share"] <= 0.30) if pa and pc else None
    def blk(x, k):
        row = M[x[k]]
        return (row["d6_probe_block"] - row["d6_probe"]) if x["arm"] == "E3" else (row["d8_probe_block"] - row["d8_probe"])
    ctl_fail = [blk(x, "c") for x in P if x["c_comp"] and x["c_beyond"] is not None and x["c_beyond"] < 0.60]
    at_all = [blk(x, "at") for x in P if x["at_comp"]]
    out["PE4"] = {"controls_not_crossing": len(ctl_fail),
                  "share_raised_ge_0.05": float(np.mean([d >= 0.05 for d in ctl_fail])) if ctl_fail else None,
                  "anytime_mean_abs_change": float(np.mean(np.abs(at_all))) if at_all else None}
    out["PE4"]["met"] = ((out["PE4"]["share_raised_ge_0.05"] or 0) >= 0.50 and (out["PE4"]["anytime_mean_abs_change"] or 1) <= 0.05) \
        if ctl_fail and at_all else None

    # ---- §25 LE1–LE4
    C = json.load(open("theory/results/score_pilotC.json"))["runs"] if os.path.exists("theory/results/score_pilotC.json") else {}
    pK = lambda runs: [M[r]["ptr"][M[r]["K"] - 1] for r in runs]
    med = lambda v: float(np.median(v)) if v else None
    ctrl = {"K5": [v["pK1"] for r, v in C.items() if v["arm"] == "K5" and v.get("gate") and "pK1" in v],
            "K7": [v["pK1"] for r, v in C.items() if v["arm"] == "K7" and v.get("gate") and "pK1" in v],
            "K6(D C)": pK([x["c"] for x in P if x["arm"] == "D" and x["c_comp"]]),
            "K8(E8 C)": pK([x["c"] for x in P if x["arm"] == "E8" and x["c_comp"]])}
    meds = {k: med(v) for k, v in ctrl.items()}
    have = [v for v in meds.values() if v is not None]
    if len(have) < 4:
        le1 = "unresolved (an arm has no competent runs)"
    elif max(have) - min(have) <= 1:
        le1 = "supported"
    elif (meds["K8(E8 C)"] - meds["K5"] >= 2) or (meds["K7"] - meds["K5"] >= 2):
        le1 = "not supported"
    else:
        le1 = "inconclusive"
    out["LE1"] = {"medians": meds, "values": ctrl, "verdict": le1}
    le2 = {}
    for a in ("D", "E8"):
        pr = [x for x in P if x["arm"] == a and x["at_comp"] and x["c_comp"]]
        at_vals = pK([x["at"] for x in P if x["arm"] == a and x["at_comp"]])
        K = 6 if a == "D" else 8
        hi = sum(M[x["at"]]["ptr"][K - 1] > M[x["c"]]["ptr"][K - 1] for x in pr)
        le2[a] = {"median_pK1_AT": med(at_vals), "K-2": K - 2, "pairs": len(pr), "AT_higher": hi}
    tot_pairs = sum(v["pairs"] for v in le2.values()); tot_hi = sum(v["AT_higher"] for v in le2.values())
    ok_med = all(v["median_pK1_AT"] is not None and v["median_pK1_AT"] >= v["K-2"] for v in le2.values())
    out["LE2"] = {"by_arm": le2, "verdict": "unresolved (< 5 pairs)" if tot_pairs < 5 else
                  ("supported" if ok_med and tot_hi / tot_pairs >= 0.75 else "not supported")}
    e3c = pK([x["c"] for x in P if x["arm"] == "E3" and x["c_comp"]])
    out["LE3"] = {"median_E3_C": med(e3c), "median_D_C": meds["K6(D C)"],
                  "verdict": None if not e3c or meds["K6(D C)"] is None else
                  ("supported" if med(e3c) <= 2 and med(e3c) < meds["K6(D C)"] else "not supported")}
    zp = [(T[x["at"]]["z"], T[x["c"]]["z"]) for x in D if x["at_comp"] and x["c_comp"] and x["at"] in T and x["c"] in T
          and T[x["at"]]["z"] is not None and T[x["c"]]["z"] is not None]
    zu = [(a, b) for a, b in zp if a != b]; zl = sum(a < b for a, b in zu)
    out["LE4"] = {"pairs": len(zu), "AT_lower_z": zl,
                  "verdict": "unresolved (< 5 pairs)" if len(zu) < 5 else
                  ("supported" if zl / len(zu) >= 0.75 else "not supported" if zl / len(zu) <= 0.50 else "inconclusive")}
    json.dump(out, open(OUT, "w", newline="\n"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    for x in P:
        f = lambda v: "  -  " if v is None else "%.3f" % v
        print("%-3s s%-5d AT %-5s %s | C %-5s %s" % (x["arm"], x["seed"], x["at_comp"], f(x["at_beyond"]), x["c_comp"], f(x["c_beyond"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-measure", action="store_true")
    a = ap.parse_args()
    names = [(arm, r) for arm, _, at, c in PAIRS for r in (at, c)]
    M = json.load(open(MEAS)) if a.no_measure and os.path.exists(MEAS) else measure(names)
    score(M)
