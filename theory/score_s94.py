"""THEORY §94 scoring (rules fixed in §94 and note 94-1; written before Theory or the experiments session read any §94 arm
value; Theory is the only scorer).
Runs: boxes/runs/f94_{stddec3x,stddecinv3x}_s6..10, f94_fadestd3x_s6..8 (eval.json as §89: modes none / zero / mean).
W1 (budget):    STD-DEC-3x "where" accuracy at N 5-8 (pure latent, mode none) >= 0.90 in >= 4/5 seeds.
W2 (direction): STD-DEC-INV-3x "where" >= 0.90 in >= 4/5 seeds.
C1 for "where" (per seed): claimed only in seeds with "where" >= 0.90 whose zero- AND mean-removal "where" accuracy is
    <= the NC "where" floor + 0.05; the floor = median over §66's nc_s1-3 of accuracy on "where" items at N 5-8
    (bench/results/boxes__boxes_nc_s*.json; question type from the item id "boxes:N:<type>:i").
Descriptive: "all" per type with removal; N 9-16; FADE-STD-3x (no decoder) per type; aux decode accuracy.
    python theory/score_s94.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"STD-DEC-3x": "stddec3x", "STD-DEC-INV-3x": "stddecinv3x", "FADE-STD-3x": "fadestd3x"}


def nc_floor(qtype):
    vals = []
    for s in (1, 2, 3):
        d = json.load(open(os.path.join(ROOT, "bench", "results", "boxes__boxes_nc_s%d.json" % s)))
        it = [x for x in d["items"] if 5 <= x["depth"] <= 8 and x["id"].split(":")[2] == qtype]
        vals.append(float(np.mean([x["correct"] for x in it])))
    return float(np.median(vals)), vals


def main():
    fw, fw_seeds = nc_floor("where"); fa, fa_seeds = nc_floor("all")
    res = {"nc_where_floor": fw, "nc_where_seeds": fw_seeds, "nc_all_floor": fa, "nc_all_seeds": fa_seeds, "arms": {}}
    for name, tag in ARMS.items():
        runs = {}
        for f in sorted(glob.glob(os.path.join(ROOT, "boxes", "runs", "f94_%s_s*" % tag, "eval.json"))):
            s = int(os.path.basename(os.path.dirname(f)).rsplit("_s", 1)[1]); runs[s] = json.load(open(f))["modes"]
        g = lambda m, k: {s: r[m][k] for s, r in runs.items()}                               # noqa: E731
        A = {"n": len(runs), "where": g("none", "acc_N5_8_where"), "all": g("none", "acc_N5_8_all"),
             "where_zero": g("zero", "acc_N5_8_where"), "where_mean": g("mean", "acc_N5_8_where"),
             "all_zero": g("zero", "acc_N5_8_all"), "all_mean": g("mean", "acc_N5_8_all"),
             "N9_16": g("none", "acc_N9_16"), "aux_decode": {s: r["none"].get("aux_decode_acc_alpha1") for s, r in runs.items()}}
        A["where_ge_0.90"] = [s for s, v in A["where"].items() if v >= 0.90]
        A["C1_where_seeds"] = [s for s in A["where_ge_0.90"] if A["where_zero"][s] <= fw + 0.05 and A["where_mean"][s] <= fw + 0.05]
        res["arms"][name] = A
        print("%-15s n %d | where %s | all %s | where zero/mean %s / %s | C1-where seeds %s" % (
            name, A["n"], {s: round(v, 3) for s, v in A["where"].items()}, {s: round(v, 3) for s, v in A["all"].items()},
            {s: round(v, 3) for s, v in A["where_zero"].items()}, {s: round(v, 3) for s, v in A["where_mean"].items()}, A["C1_where_seeds"]))
    sd, si = res["arms"]["STD-DEC-3x"], res["arms"]["STD-DEC-INV-3x"]
    res["W1"] = sd["n"] == 5 and len(sd["where_ge_0.90"]) >= 4
    res["W2"] = si["n"] == 5 and len(si["where_ge_0.90"]) >= 4
    print("NC where floor %.3f %s | W1 %s | W2 %s" % (fw, [round(x, 3) for x in fw_seeds], res["W1"], res["W2"]))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s94.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
