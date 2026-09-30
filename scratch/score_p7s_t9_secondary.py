"""Score the registered secondary analyses of PROTOCOL7 T9/T10: the amendment-3 causal tests (positive control, T1b,
T1c, T2b, T13) and T4, with the layer mapping of amendments 10 and 11. Rules are copied unchanged from
scratch/score_p7s_amend3.py and scratch/p7s_t4.py; written 2026-09-30 before any of these values was read.
Populations: competent T9/T10 models above the P7 floor (results/p7s_t9.json), T9 pooled and T10 separately.
Inputs: results/p7s_causal2_t9.json (literal first block), results/p7s_causal2_t9_chosen.json (chosen layer, the 7
models whose in_peak layer is not 1), results/far_sweep_layers_s2033.json, results/p6_endact.json.
Amendment 10: positive control and T2b use the chosen-layer measurement (primary), with the literal one reported;
T1b / T1c are identical in both. Amendment 11: T4 compares the layer-1 P7 with the first-block instruments.
    PYTHONPATH=. python scratch/score_p7s_t9_secondary.py
"""
import importlib.util
import json

import numpy as np

from latent import protocol5_confirm as P5

spec = importlib.util.spec_from_file_location("t4", "scratch/p7s_t4.py"); T4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T4)
FLOOR = 0.03
S = json.load(open("results/p7s_t9.json"))
CL = json.load(open("results/p7s_causal2_t9.json"))
CC = json.load(open("results/p7s_causal2_t9_chosen.json"))
F = json.load(open("results/far_sweep_layers_s2033.json"))
E = json.load(open("results/p6_endact.json"))
auc = T4.auc


def d8(r):
    L = json.load(open(f"results/length_gen_{r}.json"))[f"runs/{r}"]
    row = next(x for x in L["rows"] if x["K'"] == 12)
    v, ci = row["far"]["8"], row["far_ci"]["8"]
    return v, bool(v >= 0.60 and ci[0] > 0.55)


def causal_rows(runs, chosen):
    rows = {}
    for r in runs:
        c = (CC.get(f"runs/{r}") if chosen else None) or CL[f"runs/{r}"]
        cl = CL[f"runs/{r}"]
        mB = lambda d, k, j0, j1: float(np.mean(d[k]["B"][j0:j1]))
        mC = lambda d, k, j0, j1: float(np.mean(d[k]["C"][j0:j1]))
        f = F[f"runs/{r}"]; dv, ex8 = d8(r)
        rows[r] = {"extrap": S["rows"][r]["extrap"], "extrap8": ex8, "d8": dv, "P7": f["P7"], "P8": f["P8"],
                   "in_peak": f["in_peak"], "layer": f["chosen_layer"],
                   "room": mB(cl, "base", 1, 4) - mC(cl, "base", 1, 4) >= 0.10,
                   "t1b_drop": mB(cl, "base", 1, 4) - mB(cl, "block_B_L12", 1, 4),
                   "t1b_ctrl": abs(mB(cl, "base", 1, 4) - mB(cl, "block_D_L12", 1, 4)),
                   "TPR_B1": cl["base"]["B"][0], "t1c_drop": cl["base"]["B"][0] - cl["query_B1"]["B"][0],
                   "posctrl_drop": float(np.mean(c["base"]["A"][3:6]) - np.mean(c["block_A_L1"]["A"][3:6])),
                   "t2b_gain": mB(c, "inject_B_nat", 1, 6) - mB(c, "base", 1, 6),
                   "t2b_fp_gain": mC(c, "inject_C_nat", 1, 6) - mC(c, "base", 1, 6)}
    return rows


def amend3(rows, prim):
    out = {}
    k = sum(rows[r]["posctrl_drop"] >= 0.20 for r in prim)
    out["positive_control"] = {"counts": f"{k}/{len(prim)}", "ok": bool(prim) and k / len(prim) >= 0.80}
    el = [r for r in prim if rows[r]["extrap"] and rows[r]["room"]]
    k = sum(rows[r]["t1b_drop"] >= 0.15 and rows[r]["t1b_ctrl"] <= 0.05 for r in el)
    out["T1b"] = {"counts": f"{k}/{len(el)}", "decision": "unresolved" if len(el) < 3 else
                  "supported" if k / len(el) >= 0.70 else "not supported" if k / len(el) <= 0.40 else "mixed"}
    el = [r for r in prim if rows[r]["TPR_B1"] >= 0.35]
    k = sum(rows[r]["t1c_drop"] >= 0.30 for r in el)
    out["T1c"] = {"counts": f"{k}/{len(el)}", "decision": "unresolved" if len(el) < 3 else
                  "supported" if k / len(el) >= 0.80 else "not supported" if k / len(el) <= 0.50 else "mixed"}
    el = [r for r in prim if not rows[r]["extrap"] and rows[r]["P7"] < 0.10]
    k = sum(rows[r]["t2b_gain"] >= 0.20 and rows[r]["t2b_fp_gain"] >= 0.10 for r in el)
    out["T2b"] = {"counts": f"{k}/{len(el)}", "decision": "unresolved" if len(el) < 3 else
                  "supported" if k / len(el) >= 0.60 else "not supported" if k / len(el) <= 0.20 else "mixed"}
    y8 = np.array([rows[r]["extrap8"] for r in prim], bool)
    if y8.sum() >= 3 and (~y8).sum() >= 3:
        a8 = auc([rows[r]["P8"] for r in prim], y8)
        out["T13"] = {"auc_P8": a8, "n_extrap8": int(y8.sum()), "n": len(prim),
                      "decision": "supported" if a8 >= 0.80 else "not supported" if a8 < 0.65 else "inconclusive"}
    else:
        out["T13"] = {"decision": "unresolved", "n_extrap8": int(y8.sum()), "n": len(prim)}
    return out


def t4(runs):
    rows = {}
    for r in runs:
        p = P5.pilot_run(r)
        adv = [json.loads(l) for l in open(f"runs/{r}/log.jsonl") if "curriculum_advance_to" in l]
        e = E[f"runs/{r}"]
        rows[r] = {"extrap": bool(p["extrap_K12"]), "P7": F[f"runs/{r}"]["layers"]["1"]["P7"],   # layer-1 P7 (am. 11)
                   "EA": e["end_activity"], "delayed": p["delayed"], "matched5": p["matched5"],
                   "matched6": p["matched6"],
                   "full_step": next(x["step"] for x in adv if x["curriculum_advance_to"] is None),
                   **T4.instruments(e["trace4"], e["K"])}
    y = np.array([rows[r]["extrap"] for r in runs])
    if y.sum() < 3 or (~y).sum() < 3:
        return {"T4": "unresolved", "n": len(runs), "n_extrap": int(y.sum())}
    idp = ["EA", "delayed", "matched5", "matched6", "full_step", "ct_d5", "ct_d6", "cK_d5", "cK_d6", "reach"]
    X = {k: np.array([rows[r][k] for r in runs], float) for k in idp + ["P7"]}
    best = lambda idx: max(max(auc(X[k][idx], y[idx]), 1 - auc(X[k][idx], y[idx])) for k in idp)
    single = {k: auc(X[k], y) for k in idp + ["P7"]}
    diff0 = auc(X["P7"], y) - best(np.arange(len(runs)))
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(runs), len(runs))
        if y[i].all() or (~y[i]).all():
            continue
        bs.append(auc(X["P7"][i], y[i]) - best(i))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    best_id = max(max(v, 1 - v) for k, v in single.items() if k != "P7")
    dec = ("supported" if diff0 >= 0.10 and lo > 0 else
           "not supported" if best_id >= 0.90 and abs(single["P7"] - best_id) <= 0.05 else "inconclusive")
    return {"n": len(runs), "n_extrap": int(y.sum()), "single_auc": single, "best_id_auc": best_id, "diff": diff0,
            "diff_ci": [float(lo), float(hi)], "T4": dec}


def main():
    comp = S["competent"]
    prim = [r for r in comp if S["rows"][r]["in_peak"] >= FLOOR]
    out = {}
    for name, sel in (("T9", [r for r in prim if not r.startswith("t10")]), ("T10", [r for r in prim if r.startswith("t10")])):
        out[name] = {"n": len(sel), "amend3_primary_chosen_layer": amend3(causal_rows(sel, True), sel),
                     "amend3_literal_first_block": amend3(causal_rows(sel, False), sel), "T4_layer1": t4(sel)}
    json.dump(out, open("results/p7s_t9_secondary.json", "w"), indent=1)
    for name, v in out.items():
        print(name, "n", v["n"])
        print("  amend3 (primary):", {k: (x.get("decision", x) if isinstance(x, dict) else x) for k, x in v["amend3_primary_chosen_layer"].items()})
        print("  amend3 (literal):", {k: (x.get("decision", x) if isinstance(x, dict) else x) for k, x in v["amend3_literal_first_block"].items()})
        t = v["T4_layer1"]; print("  T4:", t.get("T4"), {k: t[k] for k in ("diff", "diff_ci", "best_id_auc") if k in t})


if __name__ == "__main__":
    main()
