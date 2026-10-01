"""Score THEORY §61b (F1-S replication, registered da85e97) from prosqa/runs_ail/<run>/eval.json, host.txt and
bench/results/prosqa_deep_na__f1s_<run>_P=16.json. Written 2026-10-01 before any §61b eval.json was read.
Success per run (all three, §61's eval, hardened, P = 16): accuracy >= 0.99 in every band d1-2, d3-6, d7-10, d11-14;
mean label precision and recall >= 0.99; edit IIA >= 0.90.
R1: >= 2/3 of f1s_r3a/b/c succeed, counted over reruns on the original class (RTX 3090, driver 570.x); reruns on any
    other class are reported, not counted.
R2: 2-5 of f1s_s4..s13 succeed (< 2: §61's 1/3 optimistic; > 5: more reliable than §61 suggested).
Failure mode (descriptive; operationalised here, the registration names the modes only):
    "flooding"    : not a success and mean label precision at P = 16 < 0.90 (extra entities labelled, as §61's s1, s2);
                    flag exact_at_P_eq_d = P = d accuracy >= 0.95 in every evaluated band of d1-6 (as s1);
    "no learning" : otherwise, P = d accuracy at d1-6 < 0.60;
    "partial"     : anything else.
Bands without P = d rows (the P = d eval may not cover d7-14) are reported as None and skipped.
DISCLOSURE (Audit): the failure-mode thresholds were set after bench prosqa_deep_na results for f1s_s6-s11 were
committed (4823373, all at chance 0.49-0.51), though before any §61b eval.json was read. The failure-mode classification is
therefore descriptive, defined after bench outcomes were visible; R1/R2 and the success profile (da85e97) are unaffected.
    python prosqa/score_f1sb.py
"""
import json
import os

import numpy as np

BANDS = ((1, 2), (3, 6), (7, 10), (11, 14))
RERUNS = ["f1s_r3a", "f1s_r3b", "f1s_r3c"]
FRESH = [f"f1s_s{s}" for s in range(4, 14)]


def score(run):
    d = f"prosqa/runs_ail/{run}"
    R = json.load(open(f"{d}/eval.json"))["rows"]
    def m(k, lo, hi):
        x = [float(r[k]) for r in R if lo <= r["depth"] <= hi and r.get(k) is not None]
        return float(np.mean(x)) if x else None
    v = {"acc": {f"d{lo}_{hi}": m("acc", lo, hi) for lo, hi in BANDS},
         "acc_P_eq_d": {f"d{lo}_{hi}": m("acc_at_P_eq_d", lo, hi) for lo, hi in BANDS},
         "label_prec": float(np.mean([r["label_prec"] for r in R])),
         "label_recall": float(np.mean([r["label_recall"] for r in R])),
         "edit_IIA": float(np.mean([r["edit_neg_on"] and r["edit_tgt_same"] for r in R if "edit_neg_on" in r]))}
    v["success"] = (all(x is not None and x >= 0.99 for x in v["acc"].values()) and v["label_prec"] >= 0.99 and v["label_recall"] >= 0.99
                    and v["edit_IIA"] >= 0.90)
    pd6 = [x for x in (v["acc_P_eq_d"]["d1_2"], v["acc_P_eq_d"]["d3_6"]) if x is not None]
    v["exact_at_P_eq_d"] = bool(pd6) and min(pd6) >= 0.95
    if v["success"]:
        v["mode"] = "success"
    elif v["label_prec"] < 0.90:
        v["mode"] = "flooding"
    elif not pd6 or np.mean(pd6) < 0.60:
        v["mode"] = "no learning"
    else:
        v["mode"] = "partial"
    host = open(f"{d}/host.txt").read().strip() if os.path.exists(f"{d}/host.txt") else ""
    v["host"] = host.replace("\n", " | ")
    first = host.split("\n")[0].split(",")
    v["orig_class"] = "3090" in first[0] and len(first) > 1 and first[1].strip().startswith("570.")
    b = f"bench/results/prosqa_deep_na__f1s_{run}_P=16.json"
    if os.path.exists(b):
        bj = json.load(open(b)); v["bench_deep_na"] = bj["accuracy"]
    return v


def main():
    res = {"runs": {r: score(r) for r in RERUNS + FRESH}}
    same = [r for r in RERUNS if res["runs"][r]["orig_class"]]
    k1 = sum(res["runs"][r]["success"] for r in same)
    res["R1"] = {"reruns_on_original_class": same, "successes": k1,
                 "met": (k1 >= 2) if len(same) == 3 else None,
                 "note": None if len(same) == 3 else "fewer than 3 reruns on the original class: count k of n reported"}
    k2 = sum(res["runs"][r]["success"] for r in FRESH)
    res["R2"] = {"successes": k2, "of": len(FRESH), "met": 2 <= k2 <= 5,
                 "reading": "optimistic draw" if k2 < 2 else "more reliable than §61" if k2 > 5 else "about 0.3"}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/f1sb.json", "w", newline="\n"), indent=1)
    for r, v in res["runs"].items():
        f = lambda x: "  -  " if x is None else f"{x:.3f}"
        print(f"{r:9s} {v['mode']:12s}{'*' if v['exact_at_P_eq_d'] else ' '} acc " + " ".join(f(x) for x in v["acc"].values())
              + " | P=d " + " ".join(f(x) for x in v["acc_P_eq_d"].values())
              + f" | prec {v['label_prec']:.3f} rec {v['label_recall']:.3f} IIA {v['edit_IIA']:.3f}"
              + f" | bench {v.get('bench_deep_na')} | {v['host'][:60]}")
    print("R1", res["R1"]); print("R2", res["R2"])


if __name__ == "__main__":
    main()
