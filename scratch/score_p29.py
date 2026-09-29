"""Score PROTOCOL6 addendum I (P29). Test set, P7 measurement and rules as written in the addendum.
    PYTHONPATH=. python scratch/score_p29.py --device cuda:1
Measures P7 (frozen far_sweep, seed 2033) and EA for finished competent test runs, then writes results/p29.json.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from latent import protocol5_confirm as P5

TEST = (["stall_bf16_s73", "stall_fp32_s73"] + [f"dev6_s{s}" for s in (48, 49, 50)]
        + [f"p29_s{s}" for s in range(80, 96)])
FAR, EAF, OUT = "results/far_sweep_s2033.json", "results/p6_endact.json", "results/p29.json"


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
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
    if comp:
        subprocess.run([sys.executable, "scratch/far_sweep_frozen_I.py", *[f"runs/{r}" for r in comp], "--seed", "2033",
                        "--out", FAR, "--device", a.device], check=True, env=env)
        subprocess.run([sys.executable, "-m", "latent.protocol6", "measure", *[f"runs/{r}" for r in comp],
                        "--device", a.device], check=True, env=env)
    F, E = json.load(open(FAR)) if os.path.exists(FAR) else {}, json.load(open(EAF))
    rows = {r: {"P7": F[f"runs/{r}"]["P7"], "EA": E[f"runs/{r}"]["end_activity"], "extrap": bool(P[r]["extrap_K12"]),
                "d7": P[r]["d7_K12"], "far_peak_step": F[f"runs/{r}"]["far_peak_step"]} for r in comp}
    y = np.array([rows[r]["extrap"] for r in comp], bool)
    p7 = np.array([rows[r]["P7"] for r in comp]); ea = np.array([rows[r]["EA"] for r in comp])
    out = {"finished": fin, "competent": comp, "not_competent": {r: P[r].get("final_stage") for r in fin if r not in comp},
           "rows": rows, "n_extrap": int(y.sum()), "n_non": int((~y).sum())}
    if y.sum() >= 2 and (~y).sum() >= 2:
        acc = float(((p7 >= 0.2) == y).mean())
        au = auc(p7, y)
        out.update(acc=acc, auc_P7=au, auc_EA=auc(ea, y),
                   P29="supported" if acc >= 0.8 and au >= 0.85 else "not supported" if au < 0.70 or acc < 0.65
                   else "inconclusive")
    else:
        out["P29"] = "unresolved"
    hi_non = [r for r in comp if rows[r]["EA"] >= P5.END_T and not rows[r]["extrap"]]
    out["high_EA_non_extrap"] = {"n": len(hi_non), "P7<0.1": sum(rows[r]["P7"] < 0.1 for r in hi_non),
                                 "P7>=0.2": sum(rows[r]["P7"] >= 0.2 for r in hi_non)}
    json.dump(out, open(OUT, "w"), indent=1)
    for r in comp:
        x = rows[r]
        print(f"{r:16s} P7 {x['P7']:.2f} EA {x['EA']:.2f} d7 {x['d7']:.3f} extrap {int(x['extrap'])}")
    print({k: v for k, v in out.items() if k not in ("rows", "finished", "competent")})


if __name__ == "__main__":
    main()
