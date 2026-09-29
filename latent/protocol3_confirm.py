"""Score docs/PROTOCOL3.md predictions (architecture ablations, held-out seeds, test-time search).

    python -m latent.protocol3_confirm
"""
from __future__ import annotations

import json
import os

import numpy as np

VAR = {"v1": "runs/v1_wide_s0", "v2": "runs/v2_deep_s0", "v3": "runs/v3_rawfb_s0"}
REF = {"s0": "runs/r3_pc_gated_gpt2", "s1": "runs/main_s1", "s2": "runs/main_s2"}
HELD = {"s3": "runs/main_s3", "s4": "runs/dev_s4", "s5": "runs/main_s5"}
EXTRA = {"v1b": "runs/v1_wide_s1", "k4": "runs/budget_k4_s0", "k10": "runs/budget_k10_s0", "k4b": "runs/budget_k4_s1", "k4c": "runs/budget_k4_s2", "k8": "runs/budget_k8_s0", "nostep": "runs/nostep_s0"}   # addenda G-K
ALL = {**REF, **HELD, **VAR, **EXTRA}
NEW_RECURRENT = ("s3", "s4", "v1", "v2")          # P15 is scored on these (addendum E)


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load(run, tag):
    return jload(os.path.join(run, "followup", tag, "summary.json"))


def comp(tag):
    c = jload(f"results/competence_{tag}.json")
    if c is None:
        return None
    c = c[0]
    top = c["budgets"][str(max(int(k) for k in c["budgets"]))]
    return {"delayed": top["delayed"]["bal_acc"][0], "upfront": top["upfront"]["bal_acc"][0]}


def matched(tag, run):
    m = jload(f"results/competence_matched_{tag}.json")
    if m is None:                                   # reference seeds: pooled Part II file
        for f in ("results/competence_matched_partial.json", "results/competence_matched.json"):
            m = jload(f)
            if m and run in m:
                return m[run]
        return None
    return list(m.values())[0]


def free_beta(run):
    S = load(run, "fresh_gainfit")
    if S is None:
        return None
    x, y = [], []
    for W, R in S["results"].items():
        d = np.array(R["gainfit"]["diff_norm"])
        x += [np.log(int(W))] * len(d); y += list(np.log(d))
    return float(-np.polyfit(x, y, 1)[0])


def shares(run):
    S = load(run, "fresh")
    if S is None:
        return None
    out = {}
    for W, R in S["results"].items():
        c = R["cache"]
        cl = c["clean_gap"][0]
        out[W] = {k: c["conds"][k]["targeted"][0] / cl for k in ("T", "TL", "Gq") if k in c["conds"]}
    return out


def attn_ratio(tag, run):
    X = jload(f"results/extra_{tag}.json") or jload("results/extra.json") or {}
    if run not in X:
        return None
    ex = {W: float(np.mean([X[run]["attention"]["layer1"][L][W]["excess"] for L in ("LAT1", "LAT2", "LAT3")]))
          for W in ("1", "2", "4", "8")}
    return {"excess": ex, "W8_over_W2": ex["8"] / ex["2"], "energy_W4": X[run].get("energy_W4")}


def search(tag, run):
    """P15 ingredients: depth-6 gain K' 6 -> 8 (budget sweep) and distance-7 accuracy at K' = 12 (length test)."""
    B = jload(f"results/budget_sweep_{tag}.json") or {}
    for f in ("results/budget_sweep_wd.json", "results/budget_sweep.json"):
        if run not in B:
            B = jload(f) or {}
    L = jload(f"results/length_gen_{tag}.json") or {}
    if run not in L:
        L = jload("results/length_gen.json") or {}
    o = {}
    if run in B:
        K = B[run]["K"]
        rows = {(r["variant"], r["K'"]): r for r in B[run]["rows"]}
        row = lambda k: rows.get(("train", k) if k <= K else ("repeat", k))     # trained row, else step-K reused
        if row(6) is not None and row(8) is not None:
            o["d6_K6"] = row(6)["by_depth"]["6"]["bal_acc"]
            o["d6_K8"] = row(8)["by_depth"]["6"]["bal_acc"]
            o["gain_6_to_8"] = o["d6_K8"] - o["d6_K6"]
    if run in L:
        r12 = [r for r in L[run]["rows"] if r["K'"] == 12][0]
        o["d7_K12"] = r12["far"]["7"]
        o["d7_K12_ci"] = r12.get("far_ci", {}).get("7")
        o["extrapolates"] = bool(o["d7_K12"] >= 0.60 and o["d7_K12_ci"] is not None and o["d7_K12_ci"][0] > 0.55)
    return o


def frontier_E(tag, run):
    """Addendum F: largest layer-1 frontier depth reached at any latent step k <= 6 (delayed format)."""
    D = jload(f"results/delayed_trace_{tag}.json") or {}
    if run not in D:
        D = jload("results/delayed_trace.json") or {}
    if run not in D:
        return None
    tr, depths = D[run]["delayed"], []
    for k in range(1, 7):
        t = tr[f"L1_k{k}"]
        j = int(np.argmax(t["reachable"]))
        if t["reachable"][j] >= 1.5 * float(np.mean(t["decoy"])):
            depths.append(j)
    return max(depths) if depths else None


def model_block(tag, run):
    o = {"run": run, "competence": comp(tag), "matched": matched(tag, run), "beta_fresh": free_beta(run),
         "route_shares": shares(run), "attention": attn_ratio(tag, run), "search": search(tag, run),
         "frontier_E": frontier_E(tag, run)}
    S = load(run, "fresh")
    if S is not None:
        R = S["results"]
        full = lambda W: R[W]["dose_remove"]["kinds"]["full"]["doses"]["1.0"]
        o["targeted_full"] = {W: full(W)["targeted"][0] for W in R}
        o["collateral_full"] = {W: full(W)["collateral_mean"][0] for W in R}
        if all("attn" in R[W] for W in R):
            o["attn_b1"] = {W: {k: e[0] for k, e in R[W]["attn"]["doses"]["1.0"].items()} |
                            {"others_hi": R[W]["attn"]["doses"]["1.0"]["att_boost_other_attached"][2]} for W in R}
    Sd = load(run, "fresh_cachedepth")
    if Sd is not None:
        o["route_depth"] = {W: R_["cache"]["by_depth"] for W, R_ in Sd["results"].items()}
    return o


def main():
    res = {"models": {t: model_block(t, r) for t, r in ALL.items()}, "verdicts": {}}
    M, V = res["models"], res["verdicts"]
    gate = lambda t: M[t]["competence"] is not None and M[t]["competence"]["delayed"] >= 0.85
    v1 = M["v1"]
    if "targeted_full" in v1 and v1["attention"]:
        r_t = v1["targeted_full"]["8"] / v1["targeted_full"]["2"]
        r_a = v1["attention"]["W8_over_W2"]
        V["P11"] = {"competent": gate("v1"), "targeted_W8_over_W2": r_t, "attention_W8_over_W2": r_a,
                    "pass": gate("v1") and r_t <= 0.5 and r_a <= 0.5}
    v2 = M["v2"]
    if v2.get("route_shares") and v2.get("route_depth"):
        ref_min = min(M[s]["route_shares"]["4"]["Gq"] for s in REF if M[s]["route_shares"])
        gq3 = v2["route_depth"]["4"]["3"]["shares"]["Gq"]
        V["P12"] = {"competent": gate("v2"), "Gq_W4": v2["route_shares"]["4"]["Gq"], "ref_min": ref_min, "Gq_d3_W4": gq3,
                    "pass": gate("v2") and v2["route_shares"]["4"]["Gq"] > ref_min and gq3 > 0.5}
    v3 = M["v3"]
    if v3.get("beta_fresh") is not None and "attn_b1" in v3:
        b = v3["beta_fresh"]
        V["P13"] = {"competent": gate("v3"), "beta": b, "outside_range": not (0.62 <= b <= 0.87), "toward_1": b > 0.87,
                    "competition_persists": v3["attn_b1"]["4"]["others_hi"] < 0,
                    "pass": gate("v3") and not (0.62 <= b <= 0.87) and v3["attn_b1"]["4"]["others_hi"] < 0}
    # P15 (addendum E): in-range test-time gain predicts extrapolation, on newly evaluated recurrent models
    rows = {}
    for t in NEW_RECURRENT:
        s = M[t]["search"]
        if "gain_6_to_8" in s and "extrapolates" in s and gate(t):
            rows[t] = {"gain_6_to_8": s["gain_6_to_8"], "predicts": s["gain_6_to_8"] >= 0.05,
                       "extrapolates": s["extrapolates"], "agree": (s["gain_6_to_8"] >= 0.05) == s["extrapolates"]}
    if rows:
        V["P15"] = {"per_model": rows, "n_scored": len(rows), "pending": [t for t in NEW_RECURRENT if t not in rows and M[t]["competence"] is None],
                    "failed_gate": [t for t in NEW_RECURRENT if M[t]["competence"] is not None and not gate(t)],
                    "pass": all(r["agree"] for r in rows.values())}
    # P16 (addendum F): frontier exhausted within the budget (E >= 4) => no extrapolation
    rows16 = {t: {"E": M[t]["frontier_E"], "extrapolates": M[t]["search"].get("extrapolates")}
              for t in NEW_RECURRENT if gate(t) and M[t]["frontier_E"] is not None and "extrapolates" in M[t]["search"]}
    if rows16:
        V["P16"] = {"per_model": rows16, "pending": [t for t in NEW_RECURRENT if t not in rows16 and M[t]["competence"] is None],
                    "failed_gate": [t for t in NEW_RECURRENT if M[t]["competence"] is not None and not gate(t)],
                    "pass": all(not r["extrapolates"] for r in rows16.values() if r["E"] >= 4)}
    # addendum H: second wide-thought seed
    v1b = M["v1b"]
    if v1b["competence"] is not None and not gate("v1b"):
        V["H_v1b"] = {"competent": False, "targeted_W8_over_W2": None, "reading": None}
    elif "targeted_full" in v1b:
        r = v1b["targeted_full"]["8"] / v1b["targeted_full"]["2"]
        V["H_v1b"] = {"competent": gate("v1b"), "targeted_W8_over_W2": r,
                      "reading": None if not gate("v1b") else "architecture" if r > 0.5 else "run" if 0.18 <= r <= 0.37 else "unresolved"}
    # addendum G: training-budget runs (exploratory; competence gate at their own K)
    for t in ("k4", "k10", "k4b"):
        m = M[t]
        if m["competence"] is None:
            continue
        o = {"competent": gate(t), "frontier_E": m["frontier_E"], "search": m["search"],
             "Gq_W4": (m["route_shares"] or {}).get("4", {}).get("Gq")}
        if t in ("k4", "k4b") and gate(t) and m["search"] and o["Gq_W4"] is not None:
            ref = [M[s]["route_shares"]["4"]["Gq"] for s in REF if M[s]["route_shares"]]
            o["query_above_ref"] = o["Gq_W4"] > max(ref)
            o["neither"] = not m["search"].get("extrapolates") and not o["query_above_ref"]
        V[f"G_{t}"] = o
    # addenda J/K: stricter gate (overall >= 0.85 and depth-matched 5-hop >= 0.60)
    gateJ = lambda t: gate(t) and M[t]["matched"] is not None and M[t]["matched"]["5"]["bal_acc"] >= 0.60

    def len_row(t, k):
        L = jload(f"results/length_gen_{t}.json") or {}
        r = L.get(ALL[t])
        return next((x for x in r["rows"] if x["K'"] == k), None) if r else None

    if M["k4c"]["competence"] is not None:
        m = M["k4c"]
        o = {"gate": gateJ("k4c"), "search": m["search"], "Gq_W4": (m["route_shares"] or {}).get("4", {}).get("Gq")}
        if o["gate"] and o["Gq_W4"] is not None:
            ref = max(M[s_]["route_shares"]["4"]["Gq"] for s_ in REF if M[s_]["route_shares"])
            o["pass"] = bool(m["search"].get("extrapolates") or o["Gq_W4"] > ref)
        V["J_k4c"] = o
    if M["k8"]["competence"] is not None:
        m = M["k8"]
        r16 = len_row("k8", 16)
        o = {"gate": gateJ("k8"), "E": m["frontier_E"], "d7_K16": r16["far"]["7"] if r16 else None}
        if o["gate"] and o["E"] is not None and o["d7_K16"] is not None:
            o["pass"] = bool(o["E"] >= 4 and o["d7_K16"] < 0.60)
        CFk = jload("results/clock_freeze_k8.json") or {}
        if ALL["k8"] in CFk and o["gate"] and not m["search"].get("extrapolates"):
            f = CFk[ALL["k8"]]["freeze_2_K12"]
            o["P17"] = {"d7": f["acc"]["7"], "ci": f["ci"]["7"], "pass": f["acc"]["7"] >= 0.60 and f["ci"]["7"][0] > 0.55}
        V["J_k8"] = o
    if M["nostep"]["competence"] is not None:
        m = M["nostep"]
        o = {"gate": gateJ("nostep"), "search": m["search"], "E": m["frontier_E"]}
        if o["gate"] and "extrapolates" in m["search"]:
            o["pass"] = bool(m["search"]["extrapolates"])
        V["P18"] = o
    # held-out seeds (addenda C/D): the Part II scorer's verdicts for s3 / s4
    fc = jload("results/followup_confirm.json")
    if fc:
        for s in HELD:
            vs = {p: v["per_model"][s] for p, v in fc["verdicts"].items() if s in v["per_model"]}
            if vs:
                expected = ("P1", "P2", "P3", "P5", "P7", "P9", "P14")
                V[f"heldout_{s}"] = {"competent": gate(s), "verdicts": vs,
                                     "expected_all_hold": all(vs.get(p) for p in expected if p in vs),
                                     "expected_scored": [p for p in expected if p in vs],
                                     "competition_W4": (fc.get("P14_competition_W4_heldout") or {}).get(s)}
    json.dump(res, open("results/protocol3_confirm.json", "w"), indent=1, default=float)
    print(json.dumps(V, indent=1, default=float))


if __name__ == "__main__":
    main()
