"""Score docs/PROTOCOL4.md (addenda A-D): step embedding and the stop, budget cells, variant replications.

    python -m latent.protocol4_confirm

Reads results/<experiment>_<run>.json written by scratch/eval_battery.sh (run key "runs/<run>") plus the
earlier seed-0 files, and writes results/protocol4_confirm.json. Definitions are those of addendum D.
"""
from __future__ import annotations

import json
import os

from .protocol3_confirm import frontier_E, jload

NOSTEP = ["nostep_s1", "nostep_s2", "nostep_s3", "nostep_s4", "nostep_s5"]                     # P19
K8 = ["budget_k8_s1", "budget_k8_s2"]                                                           # P20
NOSTEP_K8 = ["nostep_k8_s0", "nostep_k8_s1", "nostep_k8_s2", "nostep_k8_s3", "nostep_k8_s4"]   # P21
RANDK = ["randk_s0", "randk_s1", "randk_s2"]                                                    # P22
K4 = ["budget_k4_s3"]                                                                           # addendum I/J
K10 = ["budget_k10_s1", "budget_k10_s2", "budget_k10_s3"]                                       # addendum G
ALL = NOSTEP + K8 + NOSTEP_K8 + RANDK + K4 + K10 + ["v1_wide_s2", "v2_deep_s1", "v3_rawfb_s1"]


def final_stage(r):
    f = f"runs/{r}/log.jsonl"
    if not os.path.exists(f):
        return None
    adv = [json.loads(l) for l in open(f) if "curriculum_advance_to" in l]
    return "full" if adv and adv[-1]["curriculum_advance_to"] is None else (adv[-1]["curriculum_advance_to"] if adv else 1)


def attempts(r):
    return 1 + len([d for d in os.listdir("runs") if d.startswith(r + "_stage1fail")]) if os.path.isdir("runs") else None


def block(r):
    run = f"runs/{r}"
    c = jload(f"results/competence_{r}.json")
    m = jload(f"results/competence_matched_{r}.json")
    L = (jload(f"results/length_gen_{r}.json") or {}).get(run)
    o = {"trained": os.path.exists(f"{run}/model.pt"), "final_stage": final_stage(r), "attempts": attempts(r),
         "evaluated": c is not None and m is not None}
    if c is not None:
        c = c[0]
        top = c["budgets"][str(max(int(k) for k in c["budgets"]))]
        o["delayed"] = top["delayed"]["bal_acc"][0]
    if m is not None:
        o["matched5"] = m[run]["5"]["bal_acc"]
    if o["evaluated"]:
        # addendum D: a run whose curriculum ends below full depth at 30k is a gate failure, whatever its accuracy
        o["gate_accuracy"] = bool(o["delayed"] >= 0.85 and o["matched5"] >= 0.60)
        o["gate"] = bool(o["gate_accuracy"] and o["final_stage"] == "full")
    o["E"] = frontier_E(r, run)
    if L:
        for k in (12, 16):
            row = next((x for x in L["rows"] if x["K'"] == k), None)
            if row:
                ci = row.get("far_ci", {}).get("7")
                o[f"d7_K{k}"] = row["far"]["7"]
                o[f"d7_K{k}_ci"] = ci
                o[f"extrap_K{k}"] = bool(row["far"]["7"] >= 0.60 and ci is not None and ci[0] > 0.55)
    CF = (jload(f"results/clock_freeze_{r}.json") or {}).get(run)
    if CF and "freeze_2_K12" in CF:
        f = CF["freeze_2_K12"]
        o["P17_freeze"] = {"d7": f["acc"]["7"], "ci": f["ci"]["7"], "pass": f["acc"]["7"] >= 0.60 and f["ci"]["7"][0] > 0.55}
    return o


def main():
    M = {r: block(r) for r in ALL}
    V = {}
    ev = lambda rs: [r for r in rs if M[r]["evaluated"]]
    # P19: each competent no-step K = 6 run extrapolates at K' = 12
    rows = {r: {"gate": M[r]["gate"], "extrapolates": M[r].get("extrap_K12"), "d7": M[r].get("d7_K12"),
                "ci": M[r].get("d7_K12_ci")} for r in ev(NOSTEP)}
    comp = [r for r in rows if rows[r]["gate"]]
    V["P19"] = {"per_run": rows, "competent": comp, "pending": [r for r in NOSTEP[:3] if r not in rows],
                "replicated": bool(len(comp) >= 1 and all(rows[r]["extrapolates"] for r in comp)) if comp else None}
    # P20: each competent K = 8 run has E >= 4 and d7(K'=16) < 0.60; secondary P17 frozen clock
    rows = {}
    for r in ev(K8):
        m = M[r]
        o = {"gate": m["gate"], "E": m["E"], "d7_K16": m.get("d7_K16")}
        if m["gate"] and m["E"] is not None and m.get("d7_K16") is not None:
            o["pass"] = bool(m["E"] >= 4 and m["d7_K16"] < 0.60)
            if not m.get("extrap_K12") and "P17_freeze" in m:
                o["P17"] = m["P17_freeze"]
        rows[r] = o
    V["P20"] = {"per_run": rows, "pending": [r for r in K8 if r not in rows]}
    # P21: scorable = gate and E >= 4; halt account: extrapolates at K' = 16
    rows = {r: {"gate": M[r]["gate"], "E": M[r]["E"], "scorable": bool(M[r]["gate"] and (M[r]["E"] or 0) >= 4),
                "extrapolates_K16": M[r].get("extrap_K16"), "d7_K16": M[r].get("d7_K16"), "ci": M[r].get("d7_K16_ci")}
            for r in ev(NOSTEP_K8)}
    sc = [r for r in rows if rows[r]["scorable"]]
    yes = sum(bool(rows[r]["extrapolates_K16"]) for r in sc)
    no = len(sc) - yes
    reading = ("unresolved" if (yes >= 2 and no >= 2) or len(sc) < 2 else
               "halt account" if yes >= 2 else "finish-early account" if no >= 2 else "unresolved")
    V["P21"] = {"per_run": rows, "scorable": sc, "n_extrapolate": yes, "n_not": no, "reading": reading,
                "contingency_triggered": len([r for r in NOSTEP_K8[:3] if r in rows]) == 3 and len(sc) < 2}
    # P22: randk runs that pass the gate extrapolate at K' = 16
    rows = {r: {"gate": M[r]["gate"], "E": M[r]["E"], "extrapolates_K16": M[r].get("extrap_K16"),
                "d7_K16": M[r].get("d7_K16")} for r in ev(RANDK)}
    comp = [r for r in rows if rows[r]["gate"]]
    V["P22"] = {"per_run": rows, "competent": comp,
                "pass": bool(all(rows[r]["extrapolates_K16"] for r in comp)) if comp else None}
    # addendum I/J (K = 4) and G (K = 10), as restated in addendum C
    for r in ev(K4):
        m = M[r]
        V[f"IJ_{r}"] = {"gate": m["gate"], "extrapolates_K12": m.get("extrap_K12"), "d7_K12": m.get("d7_K12"),
                        "note": "query-time share clause needs the Part II cache battery"}
    for r in ev(K10):
        m = M[r]
        o = {"gate": m["gate"], "E": m["E"], "d7_K16": m.get("d7_K16")}
        if m["gate"] and m["E"] is not None and m.get("d7_K16") is not None:
            o["pass"] = bool(m["E"] >= 4 and m["d7_K16"] < 0.60)
        V[f"G_{r}"] = o
    for r in ("v1_wide_s2", "v2_deep_s1", "v3_rawfb_s1"):
        if M[r]["evaluated"]:
            V[f"variant_{r}"] = {"gate": M[r]["gate"], "note": "P11/H, P12, P13 need the Part II battery (variant_pipeline)"}
    json.dump({"models": M, "verdicts": V}, open("results/protocol4_confirm.json", "w"), indent=1, default=float)
    print(json.dumps(V, indent=1, default=float))


if __name__ == "__main__":
    main()
