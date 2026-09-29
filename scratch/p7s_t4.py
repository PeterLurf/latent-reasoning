"""PROTOCOL7 T4 (+ amendment 5): prospective head-to-head of P7 against in-distribution predictors on addendum-O runs.
Written 2026-09-29 ~00:00, before any O run was measured. Needs results/far_sweep_o_s2033.json (made by
scratch/score_p7s_amend3.py); measures end activity + the in-distribution trace (latent.protocol6 measure) if missing.
In-distribution instruments from the W = 4 in-distribution trace (excess = reachable - decoy, steps x depths 1..6):
  completion time at depth d (d = 5, 6): first step whose depth-d excess reaches half its maximum over steps;
  completion at K at depth d: depth-d excess at step K / its maximum over steps; sweep reach: deepest depth whose peak
  excess is >= 25 % of the largest in-range excess; EA; delayed, matched 5, matched 6; full-depth arrival step.
Each predictor is also scored with its sign flipped (the better direction kept), so no instrument loses by sign.
Decision: AUC(P7) - max AUC(in-distribution) with model-bootstrap 95 % CI (max recomputed per resample).
    PYTHONPATH=. python scratch/p7s_t4.py --device cuda:0
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

from latent import protocol5_confirm as P5

FAR, EAF = "results/far_sweep_o_s2033.json", "results/p6_endact.json"


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def instruments(tr, K):
    ex = np.array(tr["reachable"]) - np.array(tr["decoy"])            # (steps, 6)
    out = {}
    for d in (5, 6):
        col = ex[:, d - 1]
        mx = col.max()
        out[f"ct_d{d}"] = float(np.argmax(col >= 0.5 * mx) + 1) if mx > 0 else float(len(col))
        out[f"cK_d{d}"] = float(col[K - 1] / mx) if mx > 0 else 0.0
    pk = ex[:K].max(0)
    out["reach"] = float(max([d + 1 for d in range(6) if pk[d] >= 0.25 * pk.max()], default=0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    S = json.load(open("results/p7s_amend3.json"))
    runs = S["primary"]
    E = json.load(open(EAF)) if os.path.exists(EAF) else {}
    need = [f"runs/{r}" for r in runs if f"runs/{r}" not in E]
    if need:
        subprocess.run([sys.executable, "-m", "latent.protocol6", "measure", *need, "--device", a.device], check=True,
                       env=dict(os.environ, PYTHONPATH="."))
        E = json.load(open(EAF))
    F = json.load(open(FAR))
    rows = {}
    for r in runs:
        p = P5.pilot_run(r)
        adv = [json.loads(l) for l in open(f"runs/{r}/log.jsonl") if "curriculum_advance_to" in l]
        e = E[f"runs/{r}"]
        rows[r] = {"extrap": bool(p["extrap_K12"]), "P7": F[f"runs/{r}"]["P7"], "EA": e["end_activity"],
                   "delayed": p["delayed"], "matched5": p["matched5"], "matched6": p["matched6"],
                   "full_step": next(x["step"] for x in adv if x["curriculum_advance_to"] is None),
                   **instruments(e["trace4"], e["K"])}
    y = np.array([rows[r]["extrap"] for r in runs])
    idp = ["EA", "delayed", "matched5", "matched6", "full_step", "ct_d5", "ct_d6", "cK_d5", "cK_d6", "reach"]
    X = {k: np.array([rows[r][k] for r in runs], float) for k in idp + ["P7"]}
    best = lambda idx: max(max(auc(X[k][idx], y[idx]), 1 - auc(X[k][idx], y[idx])) for k in idp)
    single = {k: auc(X[k], y) for k in idp + ["P7"]}
    full = np.arange(len(runs))
    diff0 = auc(X["P7"], y) - best(full)
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(runs), len(runs))
        if y[i].all() or (~y[i]).all():
            continue
        bs.append(auc(X["P7"][i], y[i]) - best(i))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    best_id = max(max(v, 1 - v) for k, v in single.items() if k != "P7")
    dec = ("supported" if diff0 >= 0.10 and lo > 0 else
           "not supported" if best_id >= 0.90 and abs(single["P7"] - best_id) <= 0.05 else "inconclusive")
    out = {"n": len(runs), "n_extrap": int(y.sum()), "single_auc": single, "best_id_auc": best_id,
           "diff": diff0, "diff_ci": [float(lo), float(hi)], "T4": dec}
    json.dump({"rows": rows, "summary": out}, open("results/p7s_t4.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
