"""Score PROTOCOL7 amendment 3 (T1b, T1c, T2b, positive control, T13) on addendum-O runs.
Written 2026-09-28 ~23:47, before any O run was measured with frontier_causal2 or far_sweep.
    PYTHONPATH=. python scratch/score_p7s_amend3.py --device cuda:0   (measures missing runs, then scores)
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

from latent import protocol5_confirm as P5

CAUSAL, FAR, OUT = "results/p7s_causal2.json", "results/far_sweep_o_s2033.json", "results/p7s_amend3.json"
FLOOR = 0.03


def o_runs():
    return sorted(os.path.basename(p) for p in glob.glob("runs/o_*") + glob.glob("runs/ob_*")
                  if os.path.isdir(p) and "_stage1fail" not in p)


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def d8(r):
    L = json.load(open(f"results/length_gen_{r}.json"))[f"runs/{r}"]
    row = next(x for x in L["rows"] if x["K'"] == 12)
    v, ci = row["far"]["8"], row["far_ci"]["8"]
    return v, bool(v >= 0.60 and ci[0] > 0.55)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    fin = [r for r in o_runs() if os.path.exists(f"results/competence_matched_{r}.json")
           and os.path.exists(f"results/length_gen_{r}.json") and os.path.exists(f"runs/{r}/model.pt")]
    P = {r: P5.pilot_run(r) for r in fin}
    comp = [r for r in fin if P[r].get("gate") and "extrap_K12" in P[r]]
    env = dict(os.environ, PYTHONPATH=".")
    if comp:
        subprocess.run([sys.executable, "-m", "latent.frontier_causal2", *[f"runs/{r}" for r in comp], "--out", CAUSAL,
                        "--device", a.device], check=True, env=env)
        subprocess.run([sys.executable, "scratch/far_sweep_frozen_I.py", *[f"runs/{r}" for r in comp], "--seed", "2033",
                        "--out", FAR, "--device", a.device], check=True, env=env)
    C, F = json.load(open(CAUSAL)), json.load(open(FAR))
    rows = {}
    for r in comp:
        c, f = C[f"runs/{r}"], F[f"runs/{r}"]
        mB = lambda k, j0, j1: float(np.mean(c[k]["B"][j0:j1]))
        mC = lambda k, j0, j1: float(np.mean(c[k]["C"][j0:j1]))
        dv, ex8 = d8(r)
        rows[r] = {"extrap": bool(P[r]["extrap_K12"]), "extrap8": ex8, "d8": dv, "P7": f["P7"], "in_peak": f["in_peak"],
                   "P8": max(k[1] for k in f["far"]) / f["in_peak"],
                   "room": mB("base", 1, 4) - mC("base", 1, 4) >= 0.10,
                   "t1b_drop": mB("base", 1, 4) - mB("block_B_L12", 1, 4),
                   "t1b_ctrl": abs(mB("base", 1, 4) - mB("block_D_L12", 1, 4)),
                   "posctrl_drop": float(np.mean(c["base"]["A"][3:6]) - np.mean(c["block_A_L1"]["A"][3:6])),
                   "TPR_B1": c["base"]["B"][0], "t1c_drop": c["base"]["B"][0] - c["query_B1"]["B"][0],
                   "t2b_gain": mB("inject_B_nat", 1, 6) - mB("base", 1, 6),
                   "t2b_fp_gain": mC("inject_C_nat", 1, 6) - mC("base", 1, 6)}
    prim = [r for r in comp if rows[r]["in_peak"] >= FLOOR]
    out = {"finished": fin, "competent": comp, "primary": prim, "rows": rows}
    frac = lambda k, n: (k / n) if n else None
    pc = [r for r in prim]
    k = sum(rows[r]["posctrl_drop"] >= 0.20 for r in pc)
    out["positive_control"] = {"counts": f"{k}/{len(pc)}", "ok": bool(pc) and k / len(pc) >= 0.80}
    el = [r for r in prim if rows[r]["extrap"] and rows[r]["room"]]
    k = sum(rows[r]["t1b_drop"] >= 0.15 and rows[r]["t1b_ctrl"] <= 0.05 for r in el)
    out["T1b"] = {"counts": f"{k}/{len(el)}", "decision": "unresolved" if len(el) < 3 else
                  "supported" if k / len(el) >= 0.70 else "not supported" if k / len(el) <= 0.40 else "mixed"}
    el = [r for r in prim if rows[r]["TPR_B1"] >= 0.35]
    k = sum(rows[r]["t1c_drop"] >= 0.30 for r in el)
    out["T1c"] = {"counts": f"{k}/{len(el)}", "decision": "unresolved" if len(el) < 3 else
                  "supported" if k / len(el) >= 0.80 else "not supported" if k / len(el) <= 0.50 else "mixed"}
    el = [r for r in prim if not rows[r]["extrap"] and rows[r]["P7"] < 0.10]
    k = sum(rows[r]["t2b_gain"] >= 0.20 and rows[r]["t2b_fp_gain"] >= 0.10 for r in el)
    out["T2b"] = {"counts": f"{k}/{len(el)}", "decision": "unresolved" if len(el) < 3 else
                  "supported" if k / len(el) >= 0.60 else "not supported" if k / len(el) <= 0.20 else "mixed"}
    y8 = np.array([rows[r]["extrap8"] for r in prim], bool)
    if y8.sum() >= 3 and (~y8).sum() >= 3:
        a8 = auc([rows[r]["P8"] for r in prim], y8)
        out["T13"] = {"auc_P8": a8, "auc_P7_for_d8": auc([rows[r]["P7"] for r in prim], y8), "n_extrap8": int(y8.sum()),
                      "n": len(prim), "decision": "supported" if a8 >= 0.80 else "not supported" if a8 < 0.65 else "inconclusive"}
    else:
        out["T13"] = {"decision": "unresolved", "n_extrap8": int(y8.sum()), "n": len(prim)}
    json.dump(out, open(OUT, "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("rows", "finished", "competent", "primary")})


if __name__ == "__main__":
    main()
