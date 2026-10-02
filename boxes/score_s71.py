"""Score THEORY §71 (registered 36f2352; amendment 71-1) with one scorer on the same 2,000 test sequences (the first 2,000
of S5=512.csv). Written before any §71 panel result existed (only the seed-99 pilot and smokes were run).
Run directories boxes/runs_s5/<arm>_s<seed>/preds_test.npy ((2000, 512) element ids; -1 = invalid / not predicted):
  wc0k1, wc0k2, wc0k4 (seeds 1-5; eval.json has the certificate (k 1, 2) or the chunk check (k 4)), tagger (1-3),
  p3wc0, p3wc4 (1-3; final position only), dp, dpvar (1-3), gru (1-3, descriptive).
Metrics: (i) DeltaProduct's cumulative sequence accuracy at t = 128, 256, 512; (ii) final-state accuracy at 512.
S1: WC0_1 median cumulative at 512 >= 0.99, AND the k = 1 certificate holds in >= 4/5 seeds.
S2: tagger median cumulative at 256 <= 0.50.
S3: WC0_1 vs DeltaProduct (dp), paired on the same 2,000 sequences at t = 512: per sequence, the seed-mean of "all
    positions <= 512 correct" for each arm; difference WC0_1 - dp; 95 % bootstrap CI over sequences (10,000 resamples,
    seed 0). "exceeds" if the CI lower bound > 0; else "matches" if |mean difference| <= 0.01; else "below".
S4: P3 WC0_4 median final-state accuracy at 512 >= 0.95.
C1: median over seeds of final-state(p3wc4_s) - final-state(p3wc0_s) at 512 >= 0.05 (seed-number pairing; item-paired
    seed-mean difference with a bootstrap CI reported as descriptive).
A verdict whose arms have fewer seeds than registered is marked "incomplete".
    python boxes/score_s71.py
"""
import csv
import json
import os

import numpy as np

R = "boxes/runs_s5"
SEEDS = {"wc0k1": 5, "wc0k2": 5, "wc0k4": 5, "tagger": 3, "p3wc0": 3, "p3wc4": 3, "dp": 3, "dpvar": 3, "gru": 3}


def labels(n=2000, path="bench/data/s5/S5=512.csv"):
    Y = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            Y.append([int(v) for v in r["target"].split()])
    return np.array(Y)


def boot(d, n=10_000, seed=0):
    rng = np.random.default_rng(seed); b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    Y = labels(); runs = {}
    for arm, n in SEEDS.items():
        runs[arm] = {}
        for s in range(1, n + 1):
            d = f"{R}/{arm}_s{s}"
            if os.path.exists(f"{d}/preds_test.npy"):
                P = np.load(f"{d}/preds_test.npy")
                ev = json.load(open(f"{d}/eval.json")) if os.path.exists(f"{d}/eval.json") else {}
                runs[arm][s] = {"P": P, "eval": ev}
    per = {}
    for arm, d in runs.items():
        for s, r in d.items():
            P = r["P"]; ok = np.cumprod(P == Y, axis=1)
            per[f"{arm}_s{s}"] = {"cumulative": {str(t): float(ok[:, t - 1].mean()) for t in (128, 256, 512)},
                                  "final_512": float((P[:, -1] == Y[:, -1]).mean()),
                                  "certificate": r["eval"].get("certificate"), "chunk_check": r["eval"].get("chunk_check")}
    med = lambda arm, key, t=None: (float(np.median([per[f"{arm}_s{s}"][key][t] if t else per[f"{arm}_s{s}"][key] for s in runs[arm]]))
                                    if runs[arm] else None)
    v = {}
    certs = [per[f"wc0k1_s{s}"]["certificate"] for s in runs["wc0k1"]]
    ncert = sum(1 for c in certs if c and c.get("certified"))
    m1 = med("wc0k1", "cumulative", "512")
    v["S1"] = {"wc0k1_median_cum512": m1, "certified_seeds": ncert, "met": m1 is not None and m1 >= 0.99 and ncert >= 4}
    m2 = med("tagger", "cumulative", "256")
    v["S2"] = {"tagger_median_cum256": m2, "met": m2 is not None and m2 <= 0.50}
    if runs["wc0k1"] and runs["dp"]:
        a = np.mean([np.cumprod(r["P"] == Y, 1)[:, 511] for r in runs["wc0k1"].values()], 0)
        b = np.mean([np.cumprod(r["P"] == Y, 1)[:, 511] for r in runs["dp"].values()], 0)
        d = a - b; ci = boot(d); md = float(d.mean())
        word = "exceeds" if ci[0] > 0 else ("matches" if abs(md) <= 0.01 else "below")
        v["S3"] = {"mean_diff_wc0k1_minus_dp": md, "ci95": ci, "word": word,
                   "dp_median_cum512": med("dp", "cumulative", "512"), "dpvar_median_cum512": med("dpvar", "cumulative", "512")}
    m4 = med("p3wc0", "final_512")
    v["S4"] = {"p3wc0_median_final512": m4, "met": m4 is not None and m4 >= 0.95}
    common = sorted(set(runs["p3wc4"]) & set(runs["p3wc0"]))
    diffs = [per[f"p3wc4_s{s}"]["final_512"] - per[f"p3wc0_s{s}"]["final_512"] for s in common]
    c1 = {"per_seed": diffs, "median": float(np.median(diffs)) if diffs else None}
    if common:
        a = np.mean([(runs["p3wc4"][s]["P"][:, -1] == Y[:, -1]) for s in common], 0)
        b = np.mean([(runs["p3wc0"][s]["P"][:, -1] == Y[:, -1]) for s in common], 0)
        c1["item_paired"] = {"mean": float((a - b).mean()), "ci95": boot(a - b)}
    c1["met"] = c1["median"] is not None and c1["median"] >= 0.05
    v["C1"] = c1
    for key, arms in (("S1", ["wc0k1"]), ("S2", ["tagger"]), ("S3", ["wc0k1", "dp"]), ("S4", ["p3wc0"]), ("C1", ["p3wc0", "p3wc4"])):
        if key in v:
            v[key]["n_runs"] = {a: len(runs[a]) for a in arms}
            if any(len(runs[a]) < SEEDS[a] for a in arms):
                v[key]["status"] = "incomplete"
    k4 = [per[f"wc0k4_s{s}"]["chunk_check"] for s in runs["wc0k4"] if per[f"wc0k4_s{s}"]["chunk_check"]]
    v["k4_chunk_check"] = k4
    res = {"n_runs": {a: len(d) for a, d in runs.items()}, "per_run": per, "verdicts": v,
           "arm_medians": {a: {"cum": {t: med(a, "cumulative", t) for t in ("128", "256", "512")}, "final_512": med(a, "final_512")}
                           for a in runs if runs[a]}}
    os.makedirs("theory/results", exist_ok=True)
    json.dump(res, open("theory/results/score_s71.json", "w", newline="\n"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("runs", res["n_runs"])
    for a, m in res["arm_medians"].items():
        print(f"{a:7s}", m)
    for k, x in v.items():
        print(k, x)


if __name__ == "__main__":
    main()
