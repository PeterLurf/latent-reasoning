"""Score PROTOCOL6 addendum R (cap-3 dose-response R1/R2, pooled cap-4 R3). Adapted from score_m.py, written 2026-09-28
~22:20 before any addendum-R run started.  PYTHONPATH=. python scratch/score_r.py --device cuda:0

Original header: Score PROTOCOL6 addendum M (depth cap 4: M1 share of competent runs whose frontier stops at distance 5, P6 < 0.10;
M2 AUC of P6 for extrapolation by one level, matched 6-hop bal-acc >= 0.60). Adapted from score_l.py.
Written 2026-09-28 ~20:05 after all 17 runs finished; while checking the file format, the matched-competence JSON of
m_s300 was printed (its 5/6-hop bal-acc were seen); no P6, no other run's outcome was read before this file was written.
    PYTHONPATH=. python scratch/score_m.py --device cuda:0
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from latent import protocol5_confirm as P5

TEST3 = [f"r3_s{s}" for s in range(400, 412)]
TEST4 = [f"m_s{s}" for s in range(300, 329)]          # addendum M (300-316) pooled with addendum R (317-328)
L2_BASIS = (12, 13)                                   # addendum L: non-extrapolators with P7 < 0.10
FLOOR = 0.03
FAR, OUT = "results/far_sweep_r_s2033.json", "results/p_r.json"




def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def horizon(f, cap):
    """cap 4: in5 / e6 / P6 (addendum M); cap 3: in4 / e5 / P5 (addendum R)."""
    near = np.array(f["near"])
    inn = float(near[:6, :cap + 1].max())
    e = float(near[:, cap + 1].max())
    return inn, e, e / inn


def rate(rows, runs, key, thr=0.10):
    x = [rows[r][key] for r in runs]
    return sum(v < thr for v in x), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    fin = [r for r in TEST3 + TEST4 if os.path.exists(f"results/competence_matched_{r}.json")]
    P = {r: P5.pilot_run(r) for r in fin}
    M = {r: json.load(open(f"results/competence_matched_{r}.json"))[f"runs/{r}"] for r in fin}
    cap = {r: 3 if r.startswith("r3_") else 4 for r in fin}
    gate = {r: (P[r].get("final_stage") == "full" and M[r]["4"]["bal_acc"] >= 0.60) if cap[r] == 3 else bool(P[r].get("gate"))
            for r in fin}
    comp = [r for r in fin if gate[r]]
    env = dict(os.environ, PYTHONPATH=".")
    if comp:
        subprocess.run([sys.executable, "scratch/far_sweep_frozen_I.py", *[f"runs/{r}" for r in comp], "--seed", "2033",
                        "--out", FAR, "--device", a.device], check=True, env=env)
    F = json.load(open(FAR)) if comp else {}
    rows = {}
    for r in comp:
        inn, e, ratio = horizon(F[f"runs/{r}"], cap[r])
        nxt = str(cap[r] + 2)                                  # one level beyond the deepest trained distance
        rows[r] = {"cap": cap[r], "in": inn, "e": e, "P": ratio, "matched_next": M[r][nxt]["bal_acc"],
                   "extrap1": bool(M[r][nxt]["bal_acc"] >= 0.60), "d7_K12": P[r].get("d7_K12"), "P7": F[f"runs/{r}"]["P7"]}
    prim = [r for r in comp if rows[r]["in"] >= FLOOR]
    c3 = [r for r in prim if cap[r] == 3]; c4 = [r for r in prim if cap[r] == 4]
    out = {"finished": fin, "competent": comp, "primary": prim, "below_floor": [r for r in comp if r not in prim],
           "not_competent": [r for r in fin if r not in comp], "rows": rows,
           "complete": all(os.path.exists(f"results/competence_matched_{r}.json") for r in TEST3 + TEST4)}
    k3, n3 = rate(rows, c3, "P"); k4, n4 = rate(rows, c4, "P")
    out["R1_counts"] = f"{k3}/{n3}"
    out["R1"] = "unresolved" if n3 < 5 else "supported" if k3 / n3 >= 0.30 else "not supported" if k3 / n3 <= 0.10 else "inconclusive"
    out["M1_pooled_counts"] = f"{k4}/{n4}"
    shares = {"cap3": k3 / n3 if n3 else None, "cap4": k4 / n4 if n4 else None, "depth6_L2": L2_BASIS[0] / L2_BASIS[1]}
    out["R2_shares"] = shares
    if None in shares.values():
        out["R2"] = "unresolved"
    else:
        out["R2"] = "supported" if min(shares.values()) >= 0.30 else "not supported" if min(shares.values()) <= 0.10 else "inconclusive"
    for tag, runs in (("R3", c4), ("R3_cap3_exploratory", c3)):
        y = np.array([rows[r]["extrap1"] for r in runs], bool); x = np.array([rows[r]["P"] for r in runs])
        if y.sum() >= 3 and (~y).sum() >= 3:
            au = auc(x, y)
            out[tag] = {"auc": au, "n_extrap": int(y.sum()), "n": len(runs),
                        "reading": "supported" if au >= 0.80 else "not supported" if au < 0.65 else "inconclusive"}
        else:
            out[tag] = {"reading": "unresolved", "n_extrap": int(y.sum()), "n": len(runs), "auc_descriptive": auc(x, y)}
    json.dump(out, open(OUT, "w"), indent=1)
    for r in comp:
        v = rows[r]
        print(f"{r:8s} cap {v['cap']} in {v['in']:.3f} P {v['P']:.2f} next {v['matched_next']:.3f} extrap1 {int(v['extrap1'])}"
              f"{'  (below floor)' if r not in prim else ''}")
    print({k: v for k, v in out.items() if k not in ("rows", "finished", "competent", "primary")})


if __name__ == "__main__":
    main()
