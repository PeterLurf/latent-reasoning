"""PROTOCOL4 addendum C variant seeds, scored with PROTOCOL3's frozen readings and the addendum-J gate
(delayed >= 0.85 and matched 5-hop >= 0.60). Writes results/protocol4_variants.json."""
import json
from latent.protocol3_confirm import model_block, REF, jload

M = {t: model_block(t, f"runs/{t}") for t in ("v1_wide_s2", "v2_deep_s1", "v3_rawfb_s1")}
R = {s: model_block(s, r) for s, r in REF.items()}
gate = lambda m: bool(m["competence"] and m["competence"]["delayed"] >= 0.85 and m["matched"] and m["matched"]["5"]["bal_acc"] >= 0.60)
V = {}
m = M["v1_wide_s2"]
r = m["targeted_full"]["8"] / m["targeted_full"]["2"] if "targeted_full" in m else None
V["H_v1_wide_s2"] = {"gate": gate(m), "delayed": m["competence"]["delayed"], "matched5": m["matched"]["5"]["bal_acc"],
                     "targeted_W8_over_W2": r,
                     "reading": None if not gate(m) else "architecture" if r > 0.5 else "run" if 0.18 <= r <= 0.37 else "unresolved"}
m = M["v2_deep_s1"]
ref_min = min(R[s]["route_shares"]["4"]["Gq"] for s in R if R[s]["route_shares"])
gq, gq3 = m["route_shares"]["4"]["Gq"], m["route_depth"]["4"]["3"]["shares"]["Gq"]
o = {"gate": gate(m), "delayed": m["competence"]["delayed"], "matched5": m["matched"]["5"]["bal_acc"], "Gq_W4": gq, "ref_min": ref_min,
     "Gq_d3_W4": gq3, "pass": gate(m) and gq > ref_min and gq3 > 0.5, "extrapolates_K12": m["search"].get("extrapolates"),
     "d7_K12": m["search"].get("d7_K12")}
CF = (jload("results/clock_freeze_v2_deep_s1.json") or {}).get("runs/v2_deep_s1")
if CF and not o["extrapolates_K12"]:
    f = CF["freeze_2_K12"]
    o["P17"] = {"d7": f["acc"]["7"], "ci": f["ci"]["7"], "pass": f["acc"]["7"] >= 0.60 and f["ci"]["7"][0] > 0.55}
V["P12_v2_deep_s1"] = o
m = M["v3_rawfb_s1"]
b = m["beta_fresh"]
comp_ = m["attn_b1"]["4"]["others_hi"] < 0
V["P13_v3_rawfb_s1"] = {"gate": gate(m), "delayed": m["competence"]["delayed"], "matched5": m["matched"]["5"]["bal_acc"], "beta": b,
                        "toward_1": b > 0.87, "outside_range": not (0.62 <= b <= 0.87), "competition_persists": comp_,
                        "pass": gate(m) and b > 0.87 and comp_}
json.dump({"verdicts": V}, open("results/protocol4_variants.json", "w"), indent=1, default=float)
print(json.dumps(V, indent=1, default=float))
