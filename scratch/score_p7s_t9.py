"""Score PROTOCOL7 T9 (architecture generality: L3, L4, W512) and T10 (fork graphs). Written 2026-09-29 ~00:50, before
any T9/T10 run started.

Per run: competence and extrapolation by PROTOCOL5/6 rules (latent.protocol5_confirm.pilot_run); P7 from
scratch/far_sweep_layers.py (seed 2033), layer chosen per model as the one with the largest in-horizon excess
(in_peak), floor in_peak >= 0.03.
Per family (needs >= 3 extrapolating and >= 3 non-extrapolating competent runs above the floor): AUC(P7) >= 0.80
supported, < 0.65 not, else inconclusive; H2: share of non-extrapolators with P7 < 0.10 >= 60 % supported.
Pooled T9 (L3 + L4 + W512): AUC >= 0.85 supported. A family with < 3 of either outcome: "unresolved"; with no competent
run: "not learned by this recipe".
    PYTHONPATH=. python scratch/score_p7s_t9.py --device cuda:0
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

from latent import protocol5_confirm as P5

FAR = "results/far_sweep_layers_s2033.json"
FAMS = {"t9l3": "T9", "t9l4": "T9", "t9w512": "T9", "t10fork": "T10"}
FLOOR = 0.03


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def verdict(x, y, lo=0.65, hi=0.80):
    y = np.asarray(y, bool)
    if y.sum() < 3 or (~y).sum() < 3:
        return {"decision": "unresolved", "auc": auc(x, y), "n_extrap": int(y.sum()), "n": len(y)}
    a = auc(x, y)
    return {"decision": "supported" if a >= hi else "not supported" if a < lo else "inconclusive", "auc": a,
            "n_extrap": int(y.sum()), "n": len(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    runs = sorted(os.path.basename(p) for p in glob.glob("runs/t9*_s*") + glob.glob("runs/t10*_s*")
                  if "_stage1fail" not in p and os.path.exists(f"results/competence_matched_{os.path.basename(p)}.json")
                  and os.path.exists(f"results/length_gen_{os.path.basename(p)}.json"))
    P = {r: P5.pilot_run(r) for r in runs}
    comp = [r for r in runs if P[r].get("gate") and "extrap_K12" in P[r]]
    if comp:
        subprocess.run([sys.executable, "scratch/far_sweep_layers.py", *[f"runs/{r}" for r in comp], "--out", FAR,
                        "--device", a.device], check=True, env=dict(os.environ, PYTHONPATH="."))
    F = json.load(open(FAR)) if os.path.exists(FAR) else {}
    rows = {r: {"family": r.split("_s")[0], "extrap": bool(P[r]["extrap_K12"]), "d7": P[r]["d7_K12"],
                "P7": F[f"runs/{r}"]["P7"], "in_peak": F[f"runs/{r}"]["in_peak"], "layer": F[f"runs/{r}"]["chosen_layer"],
                "P7_layer1": F[f"runs/{r}"]["layers"]["1"]["P7"] if "1" in F[f"runs/{r}"]["layers"] else F[f"runs/{r}"]["layers"][1]["P7"]}
            for r in comp}
    out = {"finished": runs, "competent": comp, "rows": rows, "families": {}}
    for fam in FAMS:
        fin = [r for r in runs if r.startswith(fam + "_")]
        prim = [r for r in comp if r.startswith(fam + "_") and rows[r]["in_peak"] >= FLOOR]
        if not fin:
            continue
        if not prim:
            out["families"][fam] = {"finished": len(fin), "competent": 0, "decision": "not learned by this recipe"}
            continue
        x = [rows[r]["P7"] for r in prim]; y = [rows[r]["extrap"] for r in prim]
        non = [r for r in prim if not rows[r]["extrap"]]
        k = sum(rows[r]["P7"] < 0.10 for r in non)
        out["families"][fam] = {"finished": len(fin), "competent": len([r for r in comp if r.startswith(fam + "_")]),
                                "primary": len(prim), **verdict(x, y),
                                "H2": {"counts": f"{k}/{len(non)}", "decision": "unresolved" if len(non) < 3 else
                                       "supported" if k / len(non) >= 0.60 else "not supported"},
                                "layers_chosen": {str(l): sum(rows[r]["layer"] in (l, str(l)) for r in prim) for l in (1, 2, 3, 4)},
                                "auc_layer1_P7": auc([rows[r]["P7_layer1"] for r in prim], y)}
    t9 = [r for r in comp if FAMS[r.split("_s")[0]] == "T9" and rows[r]["in_peak"] >= FLOOR]
    out["T9_pooled"] = verdict([rows[r]["P7"] for r in t9], [rows[r]["extrap"] for r in t9], lo=0.65, hi=0.85)
    json.dump(out, open("results/p7s_t9.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k in ("families", "T9_pooled")})


if __name__ == "__main__":
    main()
