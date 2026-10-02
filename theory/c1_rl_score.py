"""THEORY §69 Stage-1 scoring (rules fixed in §69 and amendment 69-1): per (cell, method), certified scored seeds out of
10 at the matched budget; a pass needs >= 9/10 on a cell where marg (at its chosen tau0) certifies <= 6/10 on the same
seeds. Also reports the unmatched 8,000-update runs (full_*) and median process-CPU seconds (descriptive).
    python theory/c1_rl_score.py [--dir theory/results/c1_rl] [--out theory/results/score_s69_stage1.json]
"""
import argparse
import json
import os

import numpy as np

METHODS = ["marg", "hem", "grpo", "rmer", "hyb", "bon"]
CELLS = ["tab", "mlp"]


def load(d, name):
    f = os.path.join(d, name + ".json")
    return next(iter(json.load(open(f)).values())) if os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="theory/results/c1_rl"); ap.add_argument("--out", default="theory/results/score_s69_stage1.json")
    a = ap.parse_args()
    U, ch = json.load(open(os.path.join(a.dir, "calib.json"))), json.load(open(os.path.join(a.dir, "choice.json")))
    res, passers = {}, []
    for c in CELLS:
        marg = [load(a.dir, "scored_%s_marg_s%d" % (c, s)) for s in range(200, 210)]
        n_marg = sum(bool(r and r["certified"]) for r in marg)
        res[c] = {"marg_certified": n_marg, "discriminating": n_marg <= 6}
        for m in METHODS:
            rs = [load(a.dir, "scored_%s_%s_s%d" % (c, m, s)) for s in range(200, 210)]
            fl = [load(a.dir, "full_%s_%s_s%d" % (c, m, s)) for s in range(200, 210)]
            k = sum(bool(r and r["certified"]) for r in rs)
            row = {"tau0": ch[c][m]["tau0"], "updates": U[c][m], "certified": k, "missing": sum(r is None for r in rs),
                   "acc128_mean": round(float(np.mean([r["acc"]["128"] for r in rs if r])), 3),
                   "proc_sec_median": round(float(np.median([r["proc_sec"] for r in rs if r])), 1),
                   "pass": m != "marg" and k >= 9 and n_marg <= 6}
            if any(fl):
                row["full8000_certified"] = sum(bool(r and r["certified"]) for r in fl)
            res[c][m] = row
            if row["pass"]:
                passers.append(m)
            print(c, m, "tau0", row["tau0"], "U", row["updates"], "certified %d/10" % k, "acc128 %.3f" % row["acc128_mean"],
                  "proc %.1fs" % row["proc_sec_median"], ("full8000 %d/10" % row["full8000_certified"]) if "full8000_certified" in row else "",
                  "PASS" if row["pass"] else "")
    res["stage1_passers"] = sorted(set(passers))
    print("Stage-1 passers:", res["stage1_passers"])
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
