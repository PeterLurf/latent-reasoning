"""Score THEORY §78 (75ec955) from gsmlen/runs/<arm>_s<seed>/pred_<split>.json (arms tc, ws, wspad; seeds 1-3).
Committed before any §78 prediction existed.
* Accuracy per test bin (id n 1-2, mid 3-4, long n >= 5; n0 and human n >= 6 reported separately), per seed and the
  median over seeds; item-bootstrap 95 % CIs (10,000 resamples, seed 0) of the seed-pooled accuracy.
* L1 (descriptive): TC long vs TC id (difference and ratio, medians over seeds).
* L2: WS-pad - TC on long, paired by item and pooled over seeds (per item: mean over seeds of correct_wspad -
  correct_tc); met iff the pooled difference >= +0.05 and its bootstrap 95 % CI lies above 0. The median over seeds
  of the per-seed difference is reported alongside.
* L3: median over seeds of WS-pad id accuracy >= median TC id accuracy - 0.03.
* L4: WS-pad - WS on long (paired, pooled, CI); accuracy by final values-list length for WS and WS-pad.
* C3: long-bin inference FLOPs per problem ~ 2 * 124.4M * tokens_processed (KV cache within a call; WS re-encodes its
  state each call; the attention term is ignored and stated). C3 win iff WS-pad's long accuracy > TC's and WS-pad's
  median long FLOPs <= TC's.
* Secondary: by DAG depth; the aug_long split (if predicted); D&W references are benched separately.
    python gsmlen/score78.py
"""
import glob
import json
import os

import numpy as np

ARMS = ("tc", "ws", "wspad"); SEEDS = (1, 2, 3); P = 124.4e6


def load(split):
    R = {}
    for a in ARMS:
        for s in SEEDS:
            f = f"gsmlen/runs/{a}_s{s}/pred_{split}.json"
            if os.path.exists(f):
                R[(a, s)] = json.load(open(f))["rows"]
    return R


def boot(v, n=10_000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    if len(v) == 0:
        return [None, None]
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def item_mean(R, arm, sel):
    rows = [R[(arm, s)] for s in SEEDS if (arm, s) in R]
    if not rows:
        return None, []
    idx = [i for i, r in enumerate(rows[0]) if sel(r)]
    return np.array([np.mean([rr[i]["correct"] for rr in rows]) for i in idx]), rows


def main():
    R = load("test"); out = {"runs": sorted(f"{a}_s{s}" for a, s in R)}
    bins = {"id": lambda r: r["bin"] == "id", "mid": lambda r: r["bin"] == "mid", "long": lambda r: r["bin"] == "long",
            "n0": lambda r: r["bin"] == "n0", "n_ge6": lambda r: r["n"] >= 6}
    acc = {}
    for a in ARMS:
        acc[a] = {}
        for b, sel in bins.items():
            per = {s: float(np.mean([r["correct"] for r in R[(a, s)] if sel(r)])) for s in SEEDS if (a, s) in R}
            m, _ = item_mean(R, a, sel)
            acc[a][b] = {"per_seed": per, "median": float(np.median(list(per.values()))) if per else None,
                         "pooled": float(m.mean()) if m is not None else None, "ci95": boot(m) if m is not None else None,
                         "n_items": int(len(m)) if m is not None else 0}
    out["accuracy"] = acc

    def paired(a1, a2, b):
        m1, _ = item_mean(R, a1, bins[b]); m2, _ = item_mean(R, a2, bins[b])
        if m1 is None or m2 is None:
            return None
        d = m1 - m2
        per = [acc[a1][b]["per_seed"][s] - acc[a2][b]["per_seed"][s] for s in SEEDS
               if s in acc[a1][b]["per_seed"] and s in acc[a2][b]["per_seed"]]
        return {"pooled_diff": float(d.mean()), "ci95": boot(d), "median_seed_diff": float(np.median(per)) if per else None}

    tc = acc["tc"]
    if tc["long"]["median"] is not None:
        out["L1"] = {"tc_id": tc["id"]["median"], "tc_long": tc["long"]["median"],
                     "diff": tc["long"]["median"] - tc["id"]["median"],
                     "ratio": tc["long"]["median"] / tc["id"]["median"] if tc["id"]["median"] else None}
    l2 = paired("wspad", "tc", "long")
    if l2:
        l2["met"] = l2["pooled_diff"] >= 0.05 and l2["ci95"][0] > 0; out["L2"] = l2
    if acc["wspad"]["id"]["median"] is not None and tc["id"]["median"] is not None:
        out["L3"] = {"wspad_id": acc["wspad"]["id"]["median"], "tc_id": tc["id"]["median"],
                     "met": acc["wspad"]["id"]["median"] >= tc["id"]["median"] - 0.03}
    l4 = paired("wspad", "ws", "long")
    if l4:
        out["L4"] = l4
    out["by_values_len"] = {}
    for a in ("ws", "wspad"):
        rows = [r for s in SEEDS if (a, s) in R for r in R[(a, s)]]
        out["by_values_len"][a] = {str(L): [float(np.mean([r["correct"] for r in rows if r["values_len"] == L])),
                                            sum(r["values_len"] == L for r in rows)]
                                   for L in sorted({r["values_len"] for r in rows})}
    out["by_depth"] = {a: {str(d): float(np.mean([r["correct"] for s in SEEDS if (a, s) in R for r in R[(a, s)] if r["depth"] == d]))
                           for d in sorted({r["depth"] for s in SEEDS if (a, s) in R for r in R[(a, s)]})} for a in ARMS}
    fl = {}
    for a in ARMS:
        rows = [r for s in SEEDS if (a, s) in R for r in R[(a, s)] if r["bin"] == "long"]
        if rows:
            fl[a] = {"median_flops": float(np.median([2 * P * r["tokens_processed"] for r in rows])),
                     "median_passes": float(np.median([r["passes"] for r in rows])),
                     "median_written": float(np.median([r["written_tokens"] for r in rows]))}
    out["compute_long"] = fl
    if "tc" in fl and "wspad" in fl and acc["wspad"]["long"]["pooled"] is not None:
        out["C3"] = {"win": acc["wspad"]["long"]["pooled"] > tc["long"]["pooled"] and fl["wspad"]["median_flops"] <= fl["tc"]["median_flops"],
                     "note": "FLOPs ~ 2 * params * tokens processed; attention term ignored"}
    A = load("aug_long")
    if A:
        out["aug_long"] = {a: float(np.median([np.mean([r["correct"] for r in A[(a, s)]]) for s in SEEDS if (a, s) in A]))
                           for a in ARMS if any((a, s) in A for s in SEEDS)}
    json.dump(out, open("theory/results/score_s78.json", "w", newline="\n"), indent=1)
    for a in ARMS:
        print(a, {b: (round(x["median"], 3) if x["median"] is not None else None) for b, x in acc[a].items()})
    for k in ("L1", "L2", "L3", "L4", "C3", "aug_long"):
        if k in out:
            print(k, out[k])


if __name__ == "__main__":
    main()
