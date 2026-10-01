"""Score PROTOCOL10 (docs/PROTOCOL10.md). Inputs prosqa/res_p10/p8b1_<arm>_s<seed>.json (segchain_p10.py, K' = 10, S = 4,
answer depths 11-14) and, descriptive, p8_coconut.json and the amendment 1/2 files (<model>_k<K'>_s8[_d3-6].json).
G(model) = acc(S = 4) - acc(S = 1), pooled over all problems.
P10-1: mean G(ckpt) >= 0.10 and every ckpt G > 0 -> supported; mean G(ckpt) <= 0 -> not supported; else inconclusive.
P10-2: mean G(ctl) <= 0.02 -> as predicted; > 0.02 -> chaining gain without checkpoint training.
P10-3: mean G(ckpt) - mean G(ctl) >= 0.05 and complete separation -> met.
    python prosqa/score_p10.py
"""
import glob
import json
import os

import numpy as np

R = "prosqa/res_p10"


def summ(f):
    d = json.load(open(f)); rows = d["rows"]; S = d["S"]
    acc = [float(np.mean([r["margins"][s] > 0 for r in rows])) for s in range(S)]
    by_depth = {dd: [float(np.mean([r["margins"][s] > 0 for r in rows if r["depth"] == dd])) for s in range(S)]
                for dd in sorted({r["depth"] for r in rows})}
    w = [r["writes"][0] for r in rows if r["writes"]]
    ed = [r["edit"] for r in rows if "edit" in r]
    return {"n": len(rows), "S": S, "acc_by_S": acc, "by_depth": by_depth,
            "halt_rate": float(np.mean([r["halted"] is not None for r in rows])),
            "first_write_on_path": float(np.mean([x["on_path"] for x in w])) if w else None,
            "first_write_unreachable": float(np.mean([x["hops"] is None for x in w])) if w else None,
            "first_write_hops_median": float(np.median([x["hops"] for x in w if x["hops"] is not None])) if w else None,
            "edit_n": len(ed),
            "edit_follow": (float(np.mean([e["margin_edited"] < 0 for e in ed]) - np.mean([e["margin_unedited"] < 0 for e in ed]))
                            if ed else None)}


def main():
    per = {}
    for arm in ("ckpt", "ctl"):
        for s in (1, 2, 3):
            f = f"{R}/p8b1_{arm}_s{s}.json"
            if os.path.exists(f):
                per[f"{arm}_s{s}"] = summ(f)
    G = lambda arm: [v["acc_by_S"][-1] - v["acc_by_S"][0] for k, v in per.items() if k.startswith(arm + "_")]
    gc, gt = G("ckpt"), G("ctl")
    out = {"per_model": per, "G_ckpt": gc, "G_ctl": gt}
    if gc:
        m = float(np.mean(gc))
        out["P10-1"] = ("supported" if m >= 0.10 and min(gc) > 0 and len(gc) == 3 else "not supported" if m <= 0
                        else "inconclusive")
    if gt:
        out["P10-2"] = "as predicted (<= 0.02)" if np.mean(gt) <= 0.02 else "chaining gain without checkpoint training"
    if len(gc) == 3 and len(gt) == 3:
        out["P10-3"] = {"diff": float(np.mean(gc) - np.mean(gt)), "separation": min(gc) > max(gt),
                        "met": bool(np.mean(gc) - np.mean(gt) >= 0.05 and min(gc) > max(gt))}
    desc = {}
    for f in sorted(glob.glob(f"{R}/*.json")):
        n = os.path.basename(f)[:-5]
        if n.startswith("p8b1_") and n.count("_") == 2:
            continue
        desc[n] = summ(f)
    out["descriptive"] = desc
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/p10.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("G_ckpt", "G_ctl", "P10-1", "P10-2", "P10-3") if k in out}, indent=1))
    for n, v in list(per.items()) + list(desc.items()):
        print(f"{n:28s} n {v['n']:3d} acc_by_S " + " ".join(f"{a:.3f}" for a in v["acc_by_S"]) +
              f" | halt {v['halt_rate']:.2f} write on-path {v['first_write_on_path']} unreach {v['first_write_unreachable']}"
              f" | edit n {v['edit_n']} follow {v['edit_follow']}")


if __name__ == "__main__":
    main()
