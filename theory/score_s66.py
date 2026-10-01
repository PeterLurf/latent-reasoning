"""Score THEORY §66 (boxes; registered 2026-10-01 15:02, amendment 66-1) by Experiments, from the bench results
bench/results/boxes__boxes_<run>.json (per-item rows: correct, flops, seq_passes, written_tokens, validity) and, for WC,
boxes/runs/<run>/eval_edit.json. Written before any panel result was read (35/50 drained, none opened).
Arms (5 seeds each): nc, tc, tc2, cl, wc2, wc4, wc8, wc0_2, wc0_4, wc0_8 (run names <arm>_s<seed>).
Bands: in range N 1-8, extrapolation 9-16 and 17-32. Accuracy over both question types unless stated; per type reported.
Gate (note 64-2 form): NC median accuracy at N 5-8, full-arrangement ("all") questions, <= 0.50; else results descriptive.
P1: CL median accuracy at N 17-32 <= 0.5 x CL median in-range accuracy (N 1-8).
P2: WC (k = 4) median accuracy at N 17-32 >= 0.95.
P3: N 17-32, WC (k = 4) vs TC paired by item: per item, the seed-mean correctness of WC minus that of TC; mean difference
    with a 95 % bootstrap CI over items (10,000 resamples, seed 0). Met: mean >= 0 and CI lower bound > -0.02 and WC's
    median per-item FLOPs <= TC's. Partial: FLOPs condition holds and mean difference >= -0.05. Else not met.
    (The registration's "WC accuracy >= TC accuracy" is read as mean paired difference >= 0.)
P4: WC (k = 4) median edit IIA (eval_edit.json) >= 0.90.
P5: k = 8, N 17-32: per seed s, accuracy(wc8_s) - accuracy(wc0_8_s) on the same items; median over seeds >= 0.05.
Descriptive: WC0 (k = 4) and WC (k = 2) under P3's rule; k = 2 / 4 WC vs WC0; accuracy vs FLOPs / sequential passes /
written tokens per arm and band; validity (valid final arrangement, chunk-valid share); baselines; CL K' curve from
bench/results/boxes__boxes_cl_s<seed>_kp<half|double>.json if present (PILOT.md K' set: ceil(N/2), N, 2N).
Audit's additions (47e6804 review, before any result was read): every verdict carries n_runs of its arm(s) and is marked
"incomplete" if any has < 5 seeds; P2, P3 and P5 are also given per question type (descriptive); P5 also as the
item-paired seed-mean difference wc8 - wc0_8 with a bootstrap CI (descriptive; seed numbers pair independent trainings).
    python theory/score_s66.py
"""
import glob
import json
import os
import re

import numpy as np

ARMS = ["nc", "tc", "tc2", "cl", "wc2", "wc4", "wc8", "wc0_2", "wc0_4", "wc0_8"]
SEEDS = range(1, 6)
BANDS = {"1-8": (1, 8), "9-16": (9, 16), "17-32": (17, 32)}
COMP = ("flops", "seq_passes", "written_tokens")


def load(tag):
    f = f"bench/results/boxes__boxes_{tag}.json"
    if not os.path.exists(f):
        return None
    rows = json.load(open(f))["items"]
    for r in rows:
        _, n, qt, _ = r["id"].split(":"); r["N"] = int(n); r["qt"] = qt
    return {r["id"]: r for r in rows}


def acc(rows, lo, hi, qt=None):
    x = [r["correct"] for r in rows.values() if lo <= r["N"] <= hi and (qt is None or r["qt"] == qt)]
    return float(np.mean(x)) if x else None


def run_summary(rows):
    out = {}
    for b, (lo, hi) in BANDS.items():
        sel = [r for r in rows.values() if lo <= r["N"] <= hi]
        out[b] = {"acc": acc(rows, lo, hi), "acc_all": acc(rows, lo, hi, "all"), "acc_where": acc(rows, lo, hi, "where")}
        for c in COMP:
            out[b][c] = float(np.mean([r[c] for r in sel]))
        for v in ("valid_final", "chunk_valid_share"):
            x = [r[v] for r in sel if v in r]
            if x:
                out[b][v] = float(np.mean(x))
    out["acc_N5_8_all"] = acc(rows, 5, 8, "all")
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def paired(a_runs, b_runs, lo, hi, qt=None):
    """Per-item seed-mean correctness difference a - b over items both have; mean, bootstrap 95 % CI, n items."""
    ids = sorted(set.intersection(*[set(k for k, r in R.items() if lo <= r["N"] <= hi and (qt is None or r["qt"] == qt))
                                    for R in a_runs + b_runs]))
    da = np.array([np.mean([R[i]["correct"] for R in a_runs]) for i in ids])
    db = np.array([np.mean([R[i]["correct"] for R in b_runs]) for i in ids])
    d = da - db; rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (10_000, len(d)))].mean(1)
    fa = float(np.median([np.mean([R[i]["flops"] for R in a_runs]) for i in ids]))
    fb = float(np.median([np.mean([R[i]["flops"] for R in b_runs]) for i in ids]))
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_items": len(ids), "median_flops_a": fa, "median_flops_b": fb, "flops_ok": fa <= fb}


def p3_rule(p):
    if p["flops_ok"] and p["mean_diff"] >= 0 and p["ci95"][0] > -0.02:
        return "met"
    if p["flops_ok"] and p["mean_diff"] >= -0.05:
        return "partial"
    return "not met"


def main():
    runs = {a: {s: load(f"{a}_s{s}") for s in SEEDS} for a in ARMS}
    runs = {a: {s: R for s, R in d.items() if R is not None} for a, d in runs.items()}
    res = {"n_runs": {a: len(d) for a, d in runs.items()}, "per_run": {}, "arm": {}}
    for a, d in runs.items():
        for s, R in d.items():
            res["per_run"][f"{a}_s{s}"] = run_summary(R)
            e = f"boxes/runs/{a}_s{s}/eval_edit.json"
            if os.path.exists(e):
                res["per_run"][f"{a}_s{s}"]["edit"] = json.load(open(e))
        summ = [res["per_run"][f"{a}_s{s}"] for s in d]
        res["arm"][a] = {b: {k: med([x[b].get(k) for x in summ]) for k in summ[0][b]} for b in BANDS} if summ else {}
        if summ and "edit" in summ[0]:
            res["arm"][a]["edit_IIA_median"] = med([x["edit"]["IIA"] for x in summ])
    any_run = next(iter(next(d for d in runs.values() if d).values()))
    from bench.tasks import TASKS
    bl = {}
    items = {f"boxes:{it['depth']}:{it['raw']['qtype']}:{it['id'].split(':')[-1]}": it for it in TASKS["boxes"].load()[0]}
    for name, fn in TASKS["boxes"].baselines.items():
        bl[name] = {b: float(np.mean([fn(items[i]["raw"]) for i in any_run if lo <= any_run[i]["N"] <= hi]))
                    for b, (lo, hi) in BANDS.items()}
    res["baselines"] = bl

    v = {}
    nc = [res["per_run"][f"nc_s{s}"]["acc_N5_8_all"] for s in runs["nc"]]
    v["gate"] = {"nc_median_acc_N5_8_all": med(nc), "passed": med(nc) is not None and med(nc) <= 0.50}
    cl = [res["per_run"][f"cl_s{s}"] for s in runs["cl"]]
    if cl:
        hi_, in_ = med([x["17-32"]["acc"] for x in cl]), med([x["1-8"]["acc"] for x in cl])
        v["P1"] = {"cl_median_17_32": hi_, "cl_median_in_range": in_, "met": hi_ <= 0.5 * in_}
    w4 = [res["per_run"][f"wc4_s{s}"] for s in runs["wc4"]]
    if w4:
        v["P2"] = {"wc4_median_17_32": med([x["17-32"]["acc"] for x in w4])}; v["P2"]["met"] = v["P2"]["wc4_median_17_32"] >= 0.95
        v["P4"] = {"wc4_median_edit_IIA": med([x["edit"]["IIA"] for x in w4 if "edit" in x])}
        v["P4"]["met"] = v["P4"]["wc4_median_edit_IIA"] is not None and v["P4"]["wc4_median_edit_IIA"] >= 0.90
    if runs["wc4"] and runs["tc"]:
        p = paired(list(runs["wc4"].values()), list(runs["tc"].values()), 17, 32); p["verdict"] = p3_rule(p); v["P3"] = p
    desc = {}
    for a in ("wc0_4", "wc2", "wc0_2", "tc2"):
        if runs[a] and runs["tc"]:
            p = paired(list(runs[a].values()), list(runs["tc"].values()), 17, 32); p["rule"] = p3_rule(p); desc[f"{a}_vs_tc_P3rule"] = p
    pd = {}
    for k in (2, 4, 8):
        common = sorted(set(runs[f"wc{k}"]) & set(runs[f"wc0_{k}"]))
        diffs = [acc(runs[f"wc{k}"][s], 17, 32) - acc(runs[f"wc0_{k}"][s], 17, 32) for s in common]
        pd[k] = {"seeds": common, "per_seed": diffs, "median": med(diffs)}
    v["P5"] = {**pd[8], "met": pd[8]["median"] is not None and pd[8]["median"] >= 0.05}
    desc["wc_minus_wc0_17_32"] = {str(k): pd[k] for k in (2, 4)}
    kp = {}
    for s in SEEDS:
        for mode in ("half", "double"):
            R = load(f"cl_s{s}_kp{mode}")
            if R is not None:
                kp[f"cl_s{s}_{mode}"] = run_summary(R)
    if kp:
        desc["cl_Kprime_curve"] = {mode: {b: {k: med([kp[f"cl_s{s}_{mode}"][b][k] for s in SEEDS if f"cl_s{s}_{mode}" in kp])
                                              for k in ("acc", "flops", "seq_passes")} for b in BANDS}
                                   for mode in ("half", "double")}
        desc["cl_Kprime_curve"]["N"] = {b: {k: res["arm"]["cl"][b][k] for k in ("acc", "flops", "seq_passes")} for b in BANDS}
    v["descriptive"] = desc
    def stamp(key, arms):
        if key in v:
            v[key]["n_runs"] = {a: len(runs[a]) for a in arms}
            if any(len(runs[a]) < 5 for a in arms):
                v[key]["status"] = "incomplete"
    for key, arms in (("gate", ["nc"]), ("P1", ["cl"]), ("P2", ["wc4"]), ("P3", ["wc4", "tc"]), ("P4", ["wc4"]),
                      ("P5", ["wc8", "wc0_8"])):
        stamp(key, arms)
    by_qt = {}
    for qt in ("all", "where"):
        d = {}
        if runs["wc4"]:
            d["P2_wc4_median_17_32"] = med([acc(R, 17, 32, qt) for R in runs["wc4"].values()])
        if runs["wc4"] and runs["tc"]:
            p = paired(list(runs["wc4"].values()), list(runs["tc"].values()), 17, 32, qt); p["rule"] = p3_rule(p); d["P3"] = p
        common = sorted(set(runs["wc8"]) & set(runs["wc0_8"]))
        d["P5_median_per_seed"] = med([acc(runs["wc8"][s], 17, 32, qt) - acc(runs["wc0_8"][s], 17, 32, qt) for s in common])
        by_qt[qt] = d
    desc["per_question_type"] = by_qt
    if runs["wc8"] and runs["wc0_8"]:
        desc["P5_item_paired"] = paired(list(runs["wc8"].values()), list(runs["wc0_8"].values()), 17, 32)
    v["evidence_status"] = "reasoning evidence" if v["gate"]["passed"] else "descriptive only (gate not passed)"
    res["verdicts"] = v
    os.makedirs("theory/results", exist_ok=True)
    json.dump(res, open("theory/results/score_s66.json", "w", newline="\n"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("runs per arm", res["n_runs"])
    print(f"{'arm':7s} " + " ".join(f"{b:>7s}" for b in BANDS) + "   GFLOPs(17-32) seq(17-32) valid(17-32)")
    for a in ARMS:
        if res["arm"].get(a):
            r = res["arm"][a]
            print(f"{a:7s} " + " ".join(f"{r[b]['acc']:7.3f}" for b in BANDS)
                  + f"   {r['17-32']['flops'] / 1e9:8.1f} {r['17-32']['seq_passes']:8.1f} {r['17-32'].get('valid_final', float('nan')):8.3f}")
    for k in ("gate", "P1", "P2", "P3", "P4", "P5", "evidence_status"):
        print(k, v.get(k))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
