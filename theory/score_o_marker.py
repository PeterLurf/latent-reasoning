"""Score THEORY §21 (PO1-PO3 + exploratory scaffold comparison). Written 2026-09-29 ~13:25, before any O label was read
(only the structure of results/p_o.json was inspected). Labels: rows[o_sNNN] long_competent / long_success (success =
competent and extrapolates, the experiments session's O scoring); dev6 runs from dev6_pairs when present.
    python theory/score_o_marker.py
"""
import json

import numpy as np

M = json.load(open("theory/results/exp11_o_marker.json"))
P = json.load(open("theory/results/exp11b_o_earlyp7.json"))
O = json.load(open("results/p_o.json"))


def label(r):
    if r.startswith("o_s"):
        row = O["rows"].get(r)
        return (row["long_competent"], row["long_success"]) if row else (None, None)
    s = r.split("_s")[-1]
    d = O.get("dev6_pairs", {}).get(s)
    if isinstance(d, dict) and isinstance(d.get("long"), dict):
        L = d["long"]
        return L.get("competent"), L.get("success", L.get("extrap"))
    return None, None


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    p, n = x[y], x[~y]
    if not len(p) or not len(n):
        return None
    return float(((p[:, None] > n[None]) + 0.5 * (p[:, None] == n[None])).mean())


rows = []
for r, v in M.items():
    if not (r.startswith("o_s") or r.startswith("o_dev6")) or v is None:
        continue
    comp, succ = label(r)
    if not comp:
        continue
    e = P.get(r)
    rows.append({"run": r, "share": v["share"], "extrap": bool(succ), "P7_early": None if e is None else e["P7_early"]})
y = np.array([x["extrap"] for x in rows])
share = np.array([x["share"] for x in rows])
out = {"n_competent": len(rows), "n_extrap": int(y.sum())}
hi = share >= 0.25
out["PO1"] = {"n_share_ge_0.25": int(hi.sum()), "extrap_among_them": int(y[hi].sum()),
              "extrap_below": int(y[~hi].sum()), "n_below": int((~hi).sum())}
frac = y[hi].mean() if hi.sum() else None
out["PO1"]["verdict"] = ("unresolved" if hi.sum() < 5 else "supported" if frac <= 0.15 else
                         "not supported" if frac > 0.35 else "inconclusive")
out["PO2"] = {"auc_neg_share": auc(-share, y)}
out["PO2"]["verdict"] = None if out["PO2"]["auc_neg_share"] is None else ("met" if out["PO2"]["auc_neg_share"] >= 0.70 else "not met")
ok = [i for i, x in enumerate(rows) if x["P7_early"] is not None]
p7 = np.array([rows[i]["P7_early"] for i in ok]); yy = y[ok]; sh = share[ok]
a1, a2 = auc(-sh, yy), auc(p7, yy)
rng = np.random.default_rng(0); diffs = []
for _ in range(5000):
    i = rng.integers(0, len(ok), len(ok))
    u, v = auc(-sh[i], yy[i]), auc(p7[i], yy[i])
    if u is not None and v is not None:
        diffs.append(u - v)
out["PO3"] = {"n": len(ok), "auc_stage3_marker": a1, "auc_early_P7": a2,
              "diff": None if a1 is None or a2 is None else a1 - a2,
              "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else None}
# exploratory: long scaffold vs same-seed P30 run, stage-3 retrieval share
pairs = []
for r in M:
    if r.startswith("o_s") and M[r] is not None:
        b = r.replace("o_", "p30_", 1)
        if M.get(b) is not None:
            pairs.append((r, M[r]["share"], M[b]["share"]))
d = np.array([o - b for _, o, b in pairs])
out["exploratory_scaffold"] = {"n_pairs": len(pairs), "mean_share_long": float(np.mean([o for _, o, _ in pairs])) if pairs else None,
                               "mean_share_base": float(np.mean([b for _, _, b in pairs])) if pairs else None,
                               "long_lower": int((d < 0).sum()), "long_higher": int((d > 0).sum())}
out["rows"] = rows
json.dump(out, open("theory/results/score_o_marker.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
for x in sorted(rows, key=lambda x: x["share"]):
    print("  %-11s share %.3f  early P7 %s  extrapolates %s" % (x["run"], x["share"],
          "%.3f" % x["P7_early"] if x["P7_early"] is not None else "  - ", x["extrap"]))
