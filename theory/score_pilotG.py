"""Score THEORY §24 Pilot G as registered, including amendments PG5 and G-1 (GL/GLAT arms).

For each arm pair (G vs C, GAT vs AT) and (GL vs C, GLAT vs AT):
  manipulation check : the set counts as used in a competent run if ablating it (gate -> 0; theory/ablate_set.py ->
                       theory/results/setmem_ablation.json) lowers d8 or delayed bal-acc by >= 0.05; if used in fewer
                       than half the competent runs of the family, PG1-PG3 read "set unused (no verdict on A6)"
  PG1/PG2            : §23 pair rule on d8 (K' = 12)
  PG3                : (far8-10 at K' 16) - (at K' 12) >= -0.01 in >= 75 % of competent set runs, and higher, paired by
                       seed, than in C/AT (pair rule)
  PG4                : competent counts within 2 of 8 of C/AT
  PG5                : in competent runs whose set is used, set depth at step K (theory/results/set_traj.json) >= K - 2
                       in >= 75 % (>= 4 such runs, else unresolved)
    python theory/score_pilotG.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pilotFG import ev, pair_rule  # noqa: E402

SEEDS = range(900, 908)


def main():
    A = json.load(open("theory/results/setmem_ablation.json"))
    S = json.load(open("theory/results/set_traj.json"))
    E = {p: {s: ev(f"{p}_s{s}") for s in SEEDS} for p in ("pd_c", "pd_at", "pg_g", "pg_gat", "pg_gl", "pg_glat")}
    comp = lambda p, s: E[p][s] is not None and E[p][s]["gate"]
    out = {}
    for fam, (x, xat) in {"G": ("pg_g", "pg_gat"), "GL": ("pg_gl", "pg_glat")}.items():
        o = {}
        used = {}
        for p in (x, xat):
            for s in SEEDS:
                r = f"{p}_s{s}"
                if comp(p, s) and r in A:
                    used[r] = bool(E[p][s]["d8"] - A[r]["d8_ablated"] >= 0.05 or E[p][s]["delayed"] - A[r]["delayed_ablated"] >= 0.05)
        n_comp = sum(comp(p, s) for p in (x, xat) for s in SEEDS)
        o["manipulation"] = {"competent": n_comp, "measured": len(used), "used": sum(used.values()),
                             "set_used": (sum(used.values()) >= n_comp / 2) if len(used) == n_comp and n_comp else None,
                             "gates": {r: A[r]["gate"] for r in used}}
        tag = lambda v: v if o["manipulation"]["set_used"] is not False else {**v, "verdict": "set unused (no verdict on A6)"}
        pr = lambda a, b, key="d8": [(E[a][s][key], E[b][s][key]) for s in SEEDS if comp(a, s) and comp(b, s)]
        o["PG1"] = tag(pair_rule(pr(x, "pd_c"), lambda u, v: u > v))
        o["PG2"] = tag(pair_rule(pr(xat, "pd_at"), lambda u, v: u > v))
        dd = lambda p, s: E[p][s]["far8_10_K16"] - E[p][s]["far"]
        runs = [(p, s) for p in (x, xat) for s in SEEDS if comp(p, s)]
        share = float(np.mean([dd(p, s) >= -0.01 for p, s in runs])) if runs else None
        paired = [(dd(p, s), dd(q, s)) for p, q in ((x, "pd_c"), (xat, "pd_at")) for s in SEEDS if comp(p, s) and comp(q, s)]
        pp = pair_rule(paired, lambda u, v: u > v)
        o["PG3"] = tag({"share_ge_-0.01": share, "n": len(runs), "paired_vs_C_AT": pp,
                        "met": bool(share is not None and share >= 0.75 and pp["verdict"] == "supported")})
        cnt = lambda p: sum(comp(p, s) for s in SEEDS)
        o["PG4"] = {"x_vs_C": [cnt(x), cnt("pd_c")], "xAT_vs_AT": [cnt(xat), cnt("pd_at")],
                    "met": abs(cnt(x) - cnt("pd_c")) <= 2 and abs(cnt(xat) - cnt("pd_at")) <= 2}
        deep = [S[r]["set_depth"][S[r]["K"] - 1] >= S[r]["K"] - 2 for r, u in used.items() if u and r in S]
        o["PG5"] = {"n_used_runs": len(deep), "share_depth_ge_K-2": float(np.mean(deep)) if deep else None,
                    "verdict": "unresolved (< 4 runs with the set used)" if len(deep) < 4 else
                    ("supported" if np.mean(deep) >= 0.75 else "not supported")}
        o["arms"] = {p: {"competent": cnt(p), "extrapolate": sum(E[p][s]["extrap"] for s in SEEDS if comp(p, s)),
                         "median_d8": float(np.median([E[p][s]["d8"] for s in SEEDS if comp(p, s)])) if cnt(p) else None}
                     for p in (x, xat)}
        out[fam] = o
    json.dump(out, open("theory/results/score_pilotG.json", "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
