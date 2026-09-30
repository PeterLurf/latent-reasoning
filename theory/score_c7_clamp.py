"""Score THEORY §34 (T-C7c: E1–E6) from theory/results/exp15_c7_edit.json and §36 (counter clamp: CC1–CC3) from
theory/results/exp17_counter_clamp.json, exactly as registered.
    python theory/score_c7_clamp.py 34 36
"""
import json
import sys

import numpy as np

med = lambda v: float(np.median(v)) if len(v) else None


def score34():
    R = json.load(open("theory/results/exp15_c7_edit.json"))
    rows = [o["i2"] for o in R.values() if "i2" in o]
    g = lambda cond, k: [r[cond][k] for r in rows]
    out = {"models": len(R), "with_i2": len(rows)}
    e, rc = "lin_edit_x2", "lin_rand_x2"
    out["E1"] = {"median_M0": med(g(e, "M0_read")), "median_M1": med(g(e, "M1_follow")),
                 "rand_share_models_M0_M1_le_0.05": float(np.mean([(r[rc]["M0_read"] <= 0.05 and r[rc]["M1_follow"] <= 0.05) for r in rows]))}
    out["E1"]["met"] = out["E1"]["median_M0"] >= 0.50 and out["E1"]["median_M1"] >= 0.40 and out["E1"]["rand_share_models_M0_M1_le_0.05"] >= 0.90
    out["E2"] = {"median_ins": med(g(e, "ins")), "median_del": med(g(e, "del")),
                 "rand_median_abs_ins": med(np.abs(g(rc, "ins"))), "rand_median_abs_del": med(np.abs(g(rc, "del")))}
    out["E2"]["met"] = (out["E2"]["median_ins"] >= 0.40 and out["E2"]["median_del"] <= -0.30
                        and out["E2"]["rand_median_abs_ins"] <= 0.10 and out["E2"]["rand_median_abs_del"] <= 0.10)
    out["E3"] = {"median_collat": med(g(e, "M3_collat")), "rand_median_collat": med(g(rc, "M3_collat"))}
    out["E3"]["met"] = out["E3"]["median_collat"] <= 0.05 and out["E3"]["median_collat"] - out["E3"]["rand_median_collat"] <= 0.03
    iia = g(e, "IIA")
    out["E4"] = {"median_IIA": med(iia), "share_models_IIA_ge_0.30": float(np.mean([x >= 0.30 for x in iia])), "met": med(iia) >= 0.30}
    out["E5"] = {"median_agree": med(g("lin_rtrip", "agree_all")), "median_read_v": med(g("lin_rtrip", "read_v"))}
    out["E5"]["met"] = out["E5"]["median_agree"] >= 0.95 and out["E5"]["median_read_v"] >= 0.85
    out["E6"] = {"emb_median_M0": med(g("emb_edit", "M0_read")), "met": med(g("emb_edit", "M0_read")) <= 0.05}
    out["secondary"] = {c: {k: med(g(c, k)) for k in ("M0_read", "M1_follow", "ins", "del", "M3_collat", "IIA")}
                        for c in ("lin_edit", "lin_edit_x4", "lin_rand", "lin_rand_x4", "emb_rand")}
    rows3 = [o["i3"] for o in R.values() if "i3" in o]
    out["secondary"]["i3_lin_edit_x2"] = {k: med([r[e][k] for r in rows3]) for k in ("M0_read", "M1_follow", "ins", "del", "IIA")}
    ex = [o["i2"][e]["IIA"] for o in R.values() if o.get("extrap") and "i2" in o]
    nx = [o["i2"][e]["IIA"] for o in R.values() if o.get("extrap") is False and "i2" in o]
    out["moderator_extrap"] = {"median_IIA_extrap": med(ex), "n": len(ex), "median_IIA_nonextrap": med(nx), "n_non": len(nx)}
    try:
        from scipy.stats import mannwhitneyu
        out["moderator_extrap"]["mannwhitney_p"] = float(mannwhitneyu(ex, nx).pvalue)
    except Exception as err:                                  # scipy optional
        out["moderator_extrap"]["mannwhitney_p"] = f"unavailable ({err.__class__.__name__})"
    return out


def score36():
    R = json.load(open("theory/results/exp17_counter_clamp.json"))
    A = {r: o for r, o in R.items() if o["group"] == "anytime"}
    B = {r: o for r, o in R.items() if o["group"] == "p30L"}
    Bn = {r: o for r, o in B.items() if o["extrap"] is False}
    Bx = {r: o for r, o in B.items() if o["extrap"] is True}
    d = lambda G, v, k: [G[r][v][k] - G[r]["base"][k] for r in G]
    out = {"n": {"A": len(A), "B": len(B), "B_nonextrap": len(Bn), "B_extrap": len(Bx)}}
    out["CC1"] = {"median_d8_change_freeze3": med(d(A, "freeze3", "d8")), "median_peak_change_freeze3": med(d(A, "freeze3", "peak_depth")),
                  "median_d8_change_rand3": med(d(A, "rand3", "d8"))}
    c = out["CC1"]
    c["met"] = bool(A) and c["median_d8_change_freeze3"] >= 0.05 and c["median_peak_change_freeze3"] >= 2 and abs(c["median_d8_change_rand3"]) <= 0.02
    ok = lambda G, v: sum(G[r][v]["ok7"] for r in G)
    out["CC2"] = {"freeze3_ok7": ok(Bn, "freeze3"), "rand3_ok7": ok(Bn, "rand3"), "base_ok7": ok(Bn, "base"), "of": len(Bn),
                  "re2top3_ok7": sum((Bn[r]["re2top3_d7"] or 0) >= 0.60 for r in Bn)}
    out["CC2"]["met"] = out["CC2"]["freeze3_ok7"] >= 8 and out["CC2"]["rand3_ok7"] <= 3
    inr = lambda G: [G[r]["base"]["delayed_in_range"] - G[r]["freeze3"]["delayed_in_range"] for r in G]
    out["CC3"] = {"median_in_range_drop_A": med(inr(A)), "median_in_range_drop_B": med(inr(B))}
    out["CC3"]["met"] = (out["CC3"]["median_in_range_drop_A"] or 0) <= 0.02 and (out["CC3"]["median_in_range_drop_B"] or 0) <= 0.02
    desc = {}
    for gname, G in (("A", A), ("B_nonextrap", Bn), ("B_extrap", Bx)):
        desc[gname] = {v: {"median_d7": med([G[r][v]["d7"] for r in G]), "median_d8": med([G[r][v]["d8"] for r in G]),
                           "ok7": ok(G, v), "median_peak": med([G[r][v]["peak_depth"] for r in G])}
                       for v in ("base", "freeze2", "freeze3", "freeze4", "cycle", "rand3")}
    out["descriptive"] = desc
    return out


if __name__ == "__main__":
    for w in sys.argv[1:] or ["34", "36"]:
        o = score34() if w == "34" else score36()
        json.dump(o, open(f"theory/results/score_s{w}.json", "w", newline="\n"), indent=1)
        print(f"==== §{w}"); print(json.dumps(o, indent=1))
