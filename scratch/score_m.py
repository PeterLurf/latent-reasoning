"""Score PROTOCOL6 addendum M (depth cap 4: M1 share of competent runs whose frontier stops at distance 5, P6 < 0.10;
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

TEST = [f"m_s{s}" for s in range(300, 317)]
FLOOR = 0.03
FAR, OUT = "results/far_sweep_m_s2033.json", "results/p_m.json"


def card(r):
    s = int(r.split("_s")[1])
    return "3070" if s < 308 else "4060" if s < 314 else "2070"


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def p6(f):
    near = np.array(f["near"])                      # (steps 1..12, distances 1..6)
    in5 = float(near[:6, :5].max())
    e6 = float(near[:, 5].max())
    return in5, e6, e6 / in5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    fin = [r for r in TEST if os.path.exists(f"results/competence_matched_{r}.json")]
    P = {r: P5.pilot_run(r) for r in fin}
    comp = [r for r in fin if P[r].get("gate")]
    env = dict(os.environ, PYTHONPATH=".")
    if comp:
        subprocess.run([sys.executable, "scratch/far_sweep_frozen_I.py", *[f"runs/{r}" for r in comp], "--seed", "2033",
                        "--out", FAR, "--device", a.device], check=True, env=env)
    F = json.load(open(FAR)) if comp else {}
    rows = {}
    for r in comp:
        f = F[f"runs/{r}"]
        in5, e6, P6 = p6(f)
        rows[r] = {"in5": in5, "e6": e6, "P6": P6, "matched5": P[r]["matched5"], "matched6": P[r]["matched6"],
                   "extrap6": bool(P[r]["matched6"] >= 0.60), "d7_K12": P[r].get("d7_K12"), "P7": f["P7"],
                   "far_peak_step": f["far_peak_step"], "card": card(r)}
    main_ = [r for r in comp if rows[r]["in5"] >= FLOOR]
    below = [r for r in comp if rows[r]["in5"] < FLOOR]
    y = np.array([rows[r]["extrap6"] for r in main_], bool)
    x = np.array([rows[r]["P6"] for r in main_])
    out = {"finished": fin, "competent": comp, "primary": main_, "below_floor": below,
           "not_competent": {r: {k: P[r].get(k) for k in ("final_stage", "delayed", "matched5")} for r in fin if r not in comp},
           "rows": rows, "n_extrap6": int(y.sum()), "n_non": int((~y).sum()), "complete": len(fin) == len(TEST)}
    stop = int((x < 0.10).sum())
    out["M1_counts"] = f"{stop}/{len(main_)}"
    out["M1"] = ("unresolved" if len(main_) < 5 else "supported" if stop / len(main_) >= 0.30 else
                 "not supported" if stop / len(main_) <= 0.10 else "inconclusive")
    if y.sum() >= 3 and (~y).sum() >= 3:
        au = auc(x, y)
        out.update(auc_P6=au, M2="supported" if au >= 0.80 else "not supported" if au < 0.65 else "inconclusive")
    else:
        out["M2"] = "unresolved"
        out["auc_P6_descriptive"] = auc(x, y)
    json.dump(out, open(OUT, "w"), indent=1)
    for r in comp:
        v = rows[r]
        print(f"{r:8s} {v['card']} in5 {v['in5']:.3f} e6 {v['e6']:.3f} P6 {v['P6']:.2f} m5 {v['matched5']:.3f} "
              f"m6 {v['matched6']:.3f} extrap6 {int(v['extrap6'])} P7 {v['P7']:.2f} "
              f"d7 {v['d7_K12'] if v['d7_K12'] is None else round(v['d7_K12'], 3)}{'  (below floor)' if r in below else ''}")
    print({k: v for k, v in out.items() if k not in ("rows", "finished", "competent", "primary")})


if __name__ == "__main__":
    main()
