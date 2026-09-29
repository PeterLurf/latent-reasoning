"""Score PROTOCOL6 addendum L (K = 8: L1 AUC of P7, L2 share of non-extrapolators with P7 < 0.10). Adapted from score_p30.py
(written 2026-09-28 ~12:10, before any addendum-L outcome was read).

Original header of score_p30.py: Score PROTOCOL6 addendum K (P30). Test set, floor and rules as written in the addendum; adapted from score_p29.py.
    PYTHONPATH=. python scratch/score_p30.py --device cuda:1
Written 2026-09-27 ~14:57-15:05, before any P30 outcome was read (p30_s100 and p30_s112 had finished, results unopened);
includes the ~14:56 amendment (p30_s124..s159 on three Vast.ai cards).
Measures P7 (frozen far_sweep, seed 2033) and EA for finished competent test runs, then writes results/p30.json.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from latent import protocol5_confirm as P5

TEST = [f"k8_s{s}" for s in range(200, 223)]


def card(r):
    s = int(r.split("_s")[1])
    return "4060" if s < 206 or s in (220, 221) else "3070" if s < 212 or s in (216, 217, 218, 219) else "2070"


FLOOR = 0.03
FAR, EAF, OUT = "results/far_sweep_l_s2033.json", "results/p6_endact.json", "results/p_l.json"


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    fin = [r for r in TEST if os.path.exists(f"results/competence_matched_{r}.json")
           and os.path.exists(f"results/length_gen_{r}.json")]
    P = {r: P5.pilot_run(r) for r in fin}
    comp = [r for r in fin if P[r].get("gate") and "extrap_K12" in P[r]]
    env = dict(os.environ, PYTHONPATH=".")
    E = json.load(open(EAF)) if os.path.exists(EAF) else {}
    if comp:
        subprocess.run([sys.executable, "scratch/far_sweep_frozen_I.py", *[f"runs/{r}" for r in comp], "--seed", "2033",
                        "--out", FAR, "--device", a.device], check=True, env=env)
        need_ea = [r for r in comp if f"runs/{r}" not in E]
        if need_ea:
            subprocess.run([sys.executable, "-m", "latent.protocol6", "measure", *[f"runs/{r}" for r in need_ea],
                            "--device", a.device], check=True, env=env)
    F, E = json.load(open(FAR)) if os.path.exists(FAR) else {}, json.load(open(EAF))
    rows = {r: {"P7": F[f"runs/{r}"]["P7"], "in_peak": F[f"runs/{r}"]["in_peak"],
                "EA": E[f"runs/{r}"]["end_activity"], "extrap": bool(P[r]["extrap_K12"]), "d7": P[r]["d7_K12"],
                "far_peak_step": F[f"runs/{r}"]["far_peak_step"], "recipe": r.split("_")[0], "card": card(r)} for r in comp}
    main_ = [r for r in comp if rows[r]["in_peak"] >= FLOOR]
    below = [r for r in comp if rows[r]["in_peak"] < FLOOR]
    y = np.array([rows[r]["extrap"] for r in main_], bool)
    p7 = np.array([rows[r]["P7"] for r in main_]); ea = np.array([rows[r]["EA"] for r in main_])
    out = {"finished": fin, "competent": comp, "primary": main_, "below_floor": below,
           "not_competent": {r: P[r].get("final_stage") for r in fin if r not in comp},
           "rows": rows, "n_extrap": int(y.sum()), "n_non": int((~y).sum()),
           "complete": len(fin) == len(TEST)}
    if y.sum() >= 2 and (~y).sum() >= 2:
        au, ae = auc(p7, y), auc(ea, y)
        out.update(auc_P7=au, auc_EA=ae, L1="supported" if au >= 0.80 else "not supported" if au < 0.65 else "inconclusive")
    else:
        out["L1"] = "unresolved"
    non = [r for r in main_ if not rows[r]["extrap"]]
    stop = sum(rows[r]["P7"] < 0.10 for r in non)
    out["L2_counts"] = f"{stop}/{len(non)}"
    out["L2"] = ("unresolved" if len(non) < 3 else "supported" if stop / len(non) >= 0.8 else
                 "not supported" if stop / len(non) <= 0.5 else "inconclusive")
    # secondary (descriptive)
    out["below_floor_extrap"] = f"{sum(rows[r]['extrap'] for r in below)}/{len(below)}"
    if len(main_):
        out["acc_P7_ge_0.2"] = float(((p7 >= 0.2) == y).mean())
    out["per_recipe_auc_P7"] = {}
    for rec in ("k8",):
        rr = [r for r in main_ if rows[r]["recipe"] == rec]
        out["per_recipe_auc_P7"][rec] = auc([rows[r]["P7"] for r in rr], [rows[r]["extrap"] for r in rr]) if rr else None
    out["per_card_auc_P7"] = {}
    for cd in ("4060", "3070", "2070"):
        rr = [r for r in main_ if rows[r]["card"] == cd]
        out["per_card_auc_P7"][cd] = auc([rows[r]["P7"] for r in rr], [rows[r]["extrap"] for r in rr]) if rr else None
    json.dump(out, open(OUT, "w"), indent=1)
    for r in comp:
        x = rows[r]
        print(f"{r:12s} in_peak {x['in_peak']:.3f} P7 {x['P7']:.2f} EA {x['EA']:.2f} d7 {x['d7']:.3f} "
              f"extrap {int(x['extrap'])}{'  (below floor)' if r in below else ''}")
    print({k: v for k, v in out.items() if k not in ("rows", "finished", "competent", "primary")})


if __name__ == "__main__":
    main()
