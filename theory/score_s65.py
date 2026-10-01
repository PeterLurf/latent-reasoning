"""Score THEORY §65 (restart-distillation on S3; registered 2026-10-01 14:55) by Experiments, from
theory/runs/st/s3<rd|gt|self>_s<k>/eval_st.json and valid.json, k in the 9 gated parents {1-6, 8-10}.
Written before any §65 eval_st.json was read.
Single-loop accuracy = eval_st "single" (K' = max(8, N)); band means over N.
D1: median RD single at N 9-16 >= 0.85, AND RD >= SELF + 0.50 (same parent) in 9/9 pairs.
D2: median RD single at N 9-16 >= median GT single at N 9-16 - 0.05.
D3: median RD valid share at N 12 (valid.json) >= 0.95.
D4: median RD single at N 1-8 >= 0.97.
Descriptive: N 17-24 single, K' = 24 at N 9-16 and 17-24, restart, edit IIA, SELF and GT profiles; the teacher's per-N
error = 1 - the parent's restart accuracy at N 9-16 (parent s3_s<k> eval_st.json), with its jumps at N = 9 and 16.
A verdict whose arm has fewer than 9 runs is marked "incomplete".
    python theory/score_s65.py
"""
import json
import os

import numpy as np

PARENTS = [1, 2, 3, 4, 5, 6, 8, 9, 10]
ARMS = ("rd", "gt", "self")


def band(d, lo, hi):
    x = [d[str(n)] for n in range(lo, hi + 1) if str(n) in d]
    return float(np.mean(x)) if x else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main():
    runs = {a: {} for a in ARMS}
    for a in ARMS:
        for k in PARENTS:
            d = f"theory/runs/st/s3{a}_s{k}"
            if not os.path.exists(f"{d}/eval_st.json"):
                continue
            e = json.load(open(f"{d}/eval_st.json"))
            v = json.load(open(f"{d}/valid.json")) if os.path.exists(f"{d}/valid.json") else {}
            runs[a][k] = {"in_range": band(e["single"], 1, 8), "s9_16": band(e["single"], 9, 16),
                          "s17_24": band(e["single"], 17, 24), "K24_9_16": band(e["single_K24"], 9, 16),
                          "K24_17_24": band(e["single_K24"], 17, 24), "restart_9_16": band(e["restart"], 9, 16),
                          "restart_17_24": band(e["restart"], 17, 24), "edit_IIA": float(e["edit_IIA"]),
                          "valid": {str(n): v.get(str(n)) for n in (12, 16, 20)}}
    teach = {}
    for k in PARENTS:
        e = json.load(open(f"theory/runs/st/s3_s{k}/eval_st.json"))
        err = {n: 1 - e["restart"][str(n)] for n in range(8, 18) if str(n) in e["restart"]}
        teach[k] = {"err_by_N": err, "mean_err_9_16": float(np.mean([err[n] for n in range(9, 17) if n in err]))}
    rd, gt, sf = runs["rd"], runs["gt"], runs["self"]
    n = {a: len(runs[a]) for a in ARMS}
    v = {}
    pairs = sorted(set(rd) & set(sf))
    marg = [rd[k]["s9_16"] - sf[k]["s9_16"] for k in pairs]
    v["D1"] = {"rd_median_9_16": med([r["s9_16"] for r in rd.values()]), "pairs": len(pairs),
               "pairs_rd>=self+0.5": int(sum(m >= 0.50 for m in marg)), "margins": marg}
    v["D1"]["met"] = (v["D1"]["rd_median_9_16"] is not None and v["D1"]["rd_median_9_16"] >= 0.85
                      and v["D1"]["pairs_rd>=self+0.5"] == 9)
    v["D2"] = {"rd_median_9_16": v["D1"]["rd_median_9_16"], "gt_median_9_16": med([r["s9_16"] for r in gt.values()])}
    v["D2"]["met"] = None not in v["D2"].values() and v["D2"]["rd_median_9_16"] >= v["D2"]["gt_median_9_16"] - 0.05
    v["D3"] = {"rd_median_valid_N12": med([r["valid"]["12"] for r in rd.values()])}
    v["D3"]["met"] = v["D3"]["rd_median_valid_N12"] is not None and v["D3"]["rd_median_valid_N12"] >= 0.95
    v["D4"] = {"rd_median_in_range": med([r["in_range"] for r in rd.values()])}
    v["D4"]["met"] = v["D4"]["rd_median_in_range"] is not None and v["D4"]["rd_median_in_range"] >= 0.97
    for key, arms in (("D1", ("rd", "self")), ("D2", ("rd", "gt")), ("D3", ("rd",)), ("D4", ("rd",))):
        v[key]["n_runs"] = {a: n[a] for a in arms}
        if any(n[a] < 9 for a in arms):
            v[key]["status"] = "incomplete"
    prof = {a: {m: med([r[m] for r in runs[a].values()]) for m in ("in_range", "s9_16", "s17_24", "K24_9_16", "K24_17_24",
                                                                    "restart_9_16", "restart_17_24", "edit_IIA")}
            for a in ARMS if runs[a]}
    for a in prof:
        prof[a]["valid_median"] = {N: med([r["valid"][N] for r in runs[a].values()]) for N in ("12", "16", "20")}
    res = {"n_runs": n, "verdicts": v, "profiles": prof, "teacher_error": teach, "runs": runs}
    os.makedirs("theory/results", exist_ok=True)
    json.dump(res, open("theory/results/score_s65.json", "w", newline="\n"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("runs", n)
    for a, p in prof.items():
        print(a, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in p.items()})
    for k in ("D1", "D2", "D3", "D4"):
        print(k, v[k])
    print("teacher mean err 9-16", {k: round(t["mean_err_9_16"], 3) for k, t in teach.items()})


if __name__ == "__main__":
    main()
