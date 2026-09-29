"""Score PROTOCOL6 addendum Q (GPT-2 trained on 3-4 hop ProsQA). Written 2026-09-28 ~22:20, before any Q model was trained.
    python prosqa/score_q.py      (reads prosqa/res_q_s{0,1}/kprime_test_e50.json and p_{long,ctl}.json)
"""
import json
import os

D = os.path.dirname(os.path.abspath(__file__))


def acc(b, lens):
    x = b["by_path_len"]; n = b["n_by_path_len"]
    return sum(x[str(l)] * n[str(l)] for l in lens) / sum(n[str(l)] for l in lens)


rows = {}
for s in (0, 1):
    f = f"{D}/res_q_s{s}/kprime_test_e50.json"
    if not os.path.exists(f):
        continue
    k = json.load(open(f))
    a34, a56 = acc(k["6"], (3, 4)), acc(k["6"], (5, 6))
    r = {"acc_K6": k["6"]["acc"], "a34_K6": a34, "a56_K6": a56, "gap": a34 - a56, "a56_K12": acc(k["12"], (5, 6)),
         "Q0": a34 >= 0.85, "K'": {b: k[b]["acc"] for b in k if b.isdigit()}}
    r["gain12"] = r["a56_K12"] - a56
    r["wall"] = "wall" if r["gap"] >= 0.15 else "no wall" if r["gap"] <= 0.05 else "between"
    L = f"{D}/res_q_s{s}/p_long.json"
    if os.path.exists(L):
        P = json.load(open(L))["budgets"]["6"]
        m, pl, ok = P["ext"], P["path_len"], [v > 0 for v in P["margins"]]
        ext56 = [o for o, e, l in zip(ok, m, pl) if e == 2 and l in (5, 6)]
        r["long_by_ext_K6"] = P["by_ext"]; r["long_by_path_len_K6"] = P["by_path_len"]
        r["long_ext2_path56_K6"] = sum(ext56) / len(ext56)
        C = f"{D}/res_q_s{s}/p_ctl.json"
        if os.path.exists(C):
            r["ctl_by_ext_K6"] = json.load(open(C))["budgets"]["6"]["by_ext"]
    rows[s] = r
out = {"rows": rows, "complete": len(rows) == 2}
comp = [r for r in rows.values() if r["Q0"]]
if len(comp) == 2:
    w = [r["wall"] for r in comp]
    out["Q1"] = "supported" if all(x == "wall" for x in w) else "not supported" if all(x == "no wall" for x in w) else "mixed"
    g = [r["gain12"] for r in comp]
    out["Q2"] = "supported" if all(x >= 0.05 for x in g) else "not supported" if all(x <= 0.02 for x in g) else "inconclusive"
else:
    out["Q1"] = out["Q2"] = f"not scored ({len(comp)} competent of {len(rows)})"
json.dump(out, open(f"{D}/res_q_score.json", "w"), indent=1)
for s, r in rows.items():
    print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)})
print({k: v for k, v in out.items() if k != "rows"})
