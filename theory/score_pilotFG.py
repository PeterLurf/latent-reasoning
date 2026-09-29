"""Score THEORY §23 (Pilot F: PF1–PF5) and §24 (Pilot G: PG1–PG4 + manipulation check).

Pairs by seed 900–907 with Pilot D's pd_c_s* (C) and pd_at_s* (AT). Beyond-depth accuracy d8 = length_gen far["8"] at
K' = 12 (eval.json). Competence = the §6 gate. Pair rule (§23): a pair counts if both runs are competent and values differ;
supported if the predicted direction holds in >= 75 % of counted pairs, not supported if <= 50 %, else inconclusive,
unresolved with < 5 counted pairs. Restoring index z from theory/results/transfer_pilots.json (theory/transfer_pairs.py).
Set ablation (PG manipulation check) from theory/results/setmem_ablation.json (theory/ablate_set.py).
    python theory/score_pilotFG.py
"""
import json
import os
from math import comb

import numpy as np

SEEDS = range(900, 908)
OUT = "theory/results/score_pilotFG.json"


def ev(r):
    f = f"theory/runs/{r}/eval.json"
    if not os.path.exists(f):
        return None
    e = json.load(open(f))
    lg = next(x for x in e["length_gen"]["rows"] if x["K'"] == 12)
    lg16 = next((x for x in e["length_gen"]["rows"] if x["K'"] == 16), None)
    return {"gate": bool(e["gate"]), "extrap": bool(e["extrap_K12"]), "d8": lg["far"]["8"], "far": e["far8_10_K12"],
            "far8_10_K16": (sum(lg16["far"][str(d)] for d in (8, 9, 10)) / 3) if lg16 else None,
            "delayed": e["delayed"]}


def pair_rule(pairs, better):
    """pairs: list of (x, y) values for (predicted-higher arm, other arm) with both runs competent."""
    cnt = [(x, y) for x, y in pairs if x is not None and y is not None and x != y]
    w = sum(better(x, y) for x, y in cnt)
    if len(cnt) < 5:
        v = "unresolved (< 5 counted pairs)"
    elif w / len(cnt) >= 0.75:
        v = "supported"
    elif w / len(cnt) <= 0.50:
        v = "not supported"
    else:
        v = "inconclusive"
    p = sum(comb(len(cnt), k) for k in range(w, len(cnt) + 1)) / 2 ** len(cnt) if cnt else None
    return {"counted": len(cnt), "in_direction": w, "sign_p_one_sided": p, "verdict": v}


def main():
    T = json.load(open("theory/results/transfer_pilots.json")) if os.path.exists("theory/results/transfer_pilots.json") else {}
    A = json.load(open("theory/results/setmem_ablation.json")) if os.path.exists("theory/results/setmem_ablation.json") else {}
    D = json.load(open("theory/results/score_pilotDE.json")) if os.path.exists("theory/results/score_pilotDE.json") else {}
    arms = {"C": "pd_c_s{}", "AT": "pd_at_s{}", "J": "pf_j_s{}", "JAT": "pf_jat_s{}", "ATW": "pf_atw_s{}",
            "ATL": "pf_atl_s{}", "G": "pg_g_s{}", "GAT": "pg_gat_s{}"}
    R = {a: {s: ev(f.format(s)) for s in SEEDS} for a, f in arms.items()}
    out = {"arms": {}}
    for a in arms:
        runs = [v for v in R[a].values() if v is not None]
        comp = [v for v in runs if v["gate"]]
        out["arms"][a] = {"runs": len(runs), "competent": len(comp), "extrapolate": sum(v["extrap"] for v in comp),
                          "median_d8": float(np.median([v["d8"] for v in comp])) if comp else None,
                          "mean_far8_10": float(np.mean([v["far"] for v in comp])) if comp else None}

    def pairs(x, y, key="d8"):
        return [(R[x][s][key], R[y][s][key]) for s in SEEDS
                if R[x][s] and R[y][s] and R[x][s]["gate"] and R[y][s]["gate"]]

    def zpairs(x, y):
        v = []
        for s in SEEDS:
            rx, ry = arms[x].format(s), arms[y].format(s)
            if R[x][s] and R[y][s] and R[x][s]["gate"] and R[y][s]["gate"] and rx in T and ry in T:
                v.append((T[rx]["z"], T[ry]["z"]))
        return v
    out["PF1"] = pair_rule(zpairs("J", "C"), lambda x, y: x < y)
    out["PF2"] = pair_rule(pairs("J", "C"), lambda x, y: x > y)
    out["PF3"] = pair_rule(pairs("JAT", "AT"), lambda x, y: x > y)
    jat = out["arms"]["JAT"]
    out["PF3"]["JAT_extrap_share"] = jat["extrapolate"] / jat["competent"] if jat["competent"] else None
    pd1 = D.get("PD1", {}).get("verdict")
    out["PF4"] = pair_rule(pairs("ATW", "ATL"), lambda x, y: x > y)
    out["PF4"]["PD1"] = pd1
    if pd1 == "not supported":
        out["PF4"]["verdict"] = "moot (PD1 not supported)"
    mw, ma = out["arms"]["ATW"]["median_d8"], out["arms"]["AT"]["median_d8"]
    out["PF4"]["ATW_minus_AT_median_d8"] = (mw - ma) if mw is not None and ma is not None else None
    ca = out["arms"]
    out["PF5"] = {"J_vs_C": [ca["J"]["competent"], ca["C"]["competent"]], "JAT_vs_AT": [ca["JAT"]["competent"], ca["AT"]["competent"]],
                  "met": abs(ca["J"]["competent"] - ca["C"]["competent"]) <= 2 and abs(ca["JAT"]["competent"] - ca["AT"]["competent"]) <= 2}
    # Pilot G
    used = []
    for a in ("G", "GAT"):
        for s in SEEDS:
            r = arms[a].format(s)
            if R[a][s] and R[a][s]["gate"] and r in A:
                ab = A[r]
                used.append(bool(R[a][s]["d8"] - ab["d8_ablated"] >= 0.05 or R[a][s]["delayed"] - ab["delayed_ablated"] >= 0.05))
    out["PG_manipulation"] = {"measured": len(used), "used": sum(used),
                              "set_used": (sum(used) >= len(used) / 2) if used else None}
    tag = lambda v: v if out["PG_manipulation"]["set_used"] is not False else {**v, "verdict": "set unused (no verdict on A6)"}
    out["PG1"] = tag(pair_rule(pairs("G", "C"), lambda x, y: x > y))
    out["PG2"] = tag(pair_rule(pairs("GAT", "AT"), lambda x, y: x > y))
    oc = [R[a][s]["far8_10_K16"] - R[a][s]["far"] for a in ("G", "GAT") for s in SEEDS
          if R[a][s] and R[a][s]["gate"] and R[a][s]["far8_10_K16"] is not None]
    out["PG3"] = {"n": len(oc), "share_ge_-0.01": float(np.mean([d >= -0.01 for d in oc])) if oc else None}
    out["PG4"] = {"G_vs_C": [ca["G"]["competent"], ca["C"]["competent"]], "GAT_vs_AT": [ca["GAT"]["competent"], ca["AT"]["competent"]],
                  "met": abs(ca["G"]["competent"] - ca["C"]["competent"]) <= 2 and abs(ca["GAT"]["competent"] - ca["AT"]["competent"]) <= 2}
    json.dump(out, open(OUT, "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
