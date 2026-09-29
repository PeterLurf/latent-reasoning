"""Score PROTOCOL7 T1 (frontier necessity) and T2 (frontier rescue) from results/p7s_causal.json.
Written 2026-09-28 ~23:39, before any test-population model was measured.   python scratch/score_p7s_t12.py
"""
import json

IN, OUT = "results/p7s_causal.json", "results/p7s_t12.json"


def pops():
    t1, t2 = [], []
    for f in ("results/p30.json", "results/p_l.json"):
        d = json.load(open(f)); rows = d["rows"]
        for r in d["primary"]:
            (t1 if rows[r]["extrap"] else t2 if rows[r]["P7"] < 0.10 else []).append(f"runs/{r}")
    return t1, t2


def d7(c):
    return 0.5 * (c["B"][0] + 1 - c["C"][0])


R = json.load(open(IN))
t1, t2 = pops()
out = {"T1": {}, "T2": {}}
n1 = 0
for r in t1:
    if r not in R:
        continue
    x = R[r]; b, kb, kd = x["base"], x["block_B"], x["block_D"]
    drop = b["B"][0] - kb["B"][0]; ctrl = abs(b["B"][0] - kd["B"][0])
    near = abs(sum(b["A"]) / 6 - sum(kb["A"]) / 6)
    ok = drop >= 0.20 and ctrl <= 0.05 and near <= 0.05
    n1 += ok
    out["T1"][r] = {"TPR_B1_base": b["B"][0], "drop_blockB": drop, "change_blockD": ctrl, "near_change": near, "necessity": ok}
m1 = len(out["T1"])
out["T1_counts"] = f"{n1}/{m1} of {len(t1)}"
out["T1_decision"] = ("incomplete" if m1 < len(t1) else "supported" if n1 / m1 >= 0.80 else
                      "not supported" if n1 / m1 <= 0.50 else "mixed")
n2 = nc = 0
for r in t2:
    if r not in R:
        continue
    x = R[r]; b, ib, ic = x["base"], x["inject_B_x1"], x["inject_C_x1"]
    resc = d7(ib) >= 0.60 and d7(ib) - d7(b) >= 0.10
    trust = (1 - ic["C"][0]) <= (1 - b["C"][0]) - 0.10
    n2 += resc; nc += trust
    out["T2"][r] = {"d7_base": d7(b), "d7_injB": d7(ib), "d7_injB_x0.5": d7(x["inject_B_x0.5"]), "d7_injB_x2": d7(x["inject_B_x2"]),
                    "TPR_B_injB": ib["B"], "TNR_C1_base": 1 - b["C"][0], "TNR_C1_injC": 1 - ic["C"][0],
                    "near_change_injB": abs(sum(b["A"]) / 6 - sum(ib["A"]) / 6), "rescued": resc, "readout_trusts": trust}
m2 = len(out["T2"])
out["T2_counts"] = f"{n2}/{m2} of {len(t2)}"
out["T2_decision"] = ("incomplete" if m2 < len(t2) else "supported" if n2 / m2 >= 0.60 else
                      "not supported" if n2 / m2 <= 0.20 else "mixed")
out["T2_secondary_trust"] = f"{nc}/{m2}"
json.dump(out, open(OUT, "w"), indent=1)
for k in ("T1_counts", "T1_decision", "T2_counts", "T2_decision", "T2_secondary_trust"):
    print(k, out[k])
