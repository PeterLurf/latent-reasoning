"""Score PROTOCOL6 addendum P from prosqa/res_p/<tag>_{long,ctl}.json (written 2026-09-28 ~21:25, before any output).
    python prosqa/score_p.py
"""
import json
import os

TAGS = ["g_s0", "g_s1", "g_s2", "j_s0", "j_s1", "j_s2"]
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res_p")


def cls(a6, c6):
    if a6 <= 0.60:
        return "wall" if c6 >= 0.80 else "length failure"
    return "generalises" if a6 >= 0.80 else "intermediate"


out, rows = {}, {}
for t in TAGS:
    f, g = f"{D}/{t}_long.json", f"{D}/{t}_ctl.json"
    if not (os.path.exists(f) and os.path.exists(g)):
        continue
    L, C = json.load(open(f))["budgets"], json.load(open(g))["budgets"]
    a = {k: L[k]["by_ext"] for k in L}; c = {k: C[k]["by_ext"] for k in C}
    a6, c6 = a["6"]["6"], c["6"]["6"]
    rows[t] = {"a_m_K6": a["6"], "c_m_K6": c["6"], "a6_K12": a["12"]["6"], "a6_K1": a["1"]["6"], "a6_K0": a["0"]["6"],
               "gain12": a["12"]["6"] - a6, "class": cls(a6, c6),
               "by_path_len": {k: L[k]["by_path_len"] for k in L}}
n = len(rows)
cnt = lambda c: sum(r["class"] == c for r in rows.values())
out["rows"], out["n_models"], out["complete"] = rows, n, n == len(TAGS)
out["P1_counts"] = {c: cnt(c) for c in ("wall", "generalises", "length failure", "intermediate")}
out["P1"] = "supported" if cnt("wall") >= 4 else "not supported" if cnt("generalises") >= 4 else "mixed"
g = sum(r["gain12"] >= 0.05 for r in rows.values())
out["P2_counts"] = f"{g}/{n}"
out["P2"] = "supported" if g >= 4 else "not supported" if g <= 1 else "inconclusive"
json.dump(out, open(f"{D}/p_score.json", "w"), indent=1)
for t, r in rows.items():
    print(t, "a_m(K'6)", {k: round(v, 3) for k, v in r["a_m_K6"].items()}, "c_m", {k: round(v, 3) for k, v in r["c_m_K6"].items()},
          "a6 K'0/1/12", round(r["a6_K0"], 3), round(r["a6_K1"], 3), round(r["a6_K12"], 3), r["class"])
print({k: v for k, v in out.items() if k != "rows"})
