"""THEORY §100 scoring (rules fixed in §100; written before any §100 implementation or run; Theory is the only scorer).
Runs: boxes/runs_mult/m100_{gen2,mask2}_{4x4,5x5}_s{1,2,3}/eval.json (§98's keys, plus per_chunk_acc).
Comparators from §98 on the same items: TC-M (m98_tc_*), NC-M (m98_nc_*), ICoT-SI (icotsi_*; fallback 1.000 / 0.996 is
not used: our eval 0.998 / 0.996 is on disk).
Per task, medians over seeds 1-3:
M1-2      GEN2 >= 0.95.
M-SOTA-2  GEN2 >= ICoT-SI bar.
M-C3-2    GEN2 >= TC-M - 0.01 AND GEN2 median FLOPs <= TC-M median FLOPs (as metered; restricted reported).
M-C1-2    MASK2 >= 0.95 AND, in MASK2 seeds >= 0.95, zero- and mean-removal answer accuracy <= NC-M median + 0.05.
Reading flag (Literature (1)): GEN2 seeds whose zero- or mean-removal answer accuracy stays >= 0.95 -> "text checkpoints
every 2 ops", not latent carriage.
    python theory/score_s100.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "boxes", "runs_mult")
ROW = 2 * 768 * 50257


def load(prefix):
    out = {}
    for s in (1, 2, 3):
        f = os.path.join(RUNS, "%s_s%d" % (prefix, s), "eval.json")
        if os.path.exists(f):
            out[s] = json.load(open(f))
    return out


def med(rs, fn):
    v = [fn(r) for r in rs.values()]
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def main():
    res = {}
    for task in ("4x4", "5x5"):
        g, mk = load("m100_gen2_" + task), load("m100_mask2_" + task)
        tc, nc = load("m98_tc_" + task), load("m98_nc_" + task)
        f = os.path.join(RUNS, "icotsi_" + task, "eval.json")
        bar = json.load(open(f))["acc_test"] if os.path.exists(f) else None
        acc = lambda r: r["acc_test"]                                                          # noqa: E731
        ncm = med(nc, acc); floor = ncm + 0.05 if ncm is not None else None
        T = {"n": {"gen2": len(g), "mask2": len(mk), "tc": len(tc), "nc": len(nc)}, "icotsi_bar": bar, "nc_floor": floor,
             "acc": {"gen2": {s: r["acc_test"] for s, r in g.items()}, "mask2": {s: r["acc_test"] for s, r in mk.items()}}}
        gm, tm = med(g, acc), med(tc, acc)
        T["gen2_median"], T["tc_median"] = gm, tm
        T["M1_2"] = gm is not None and gm >= 0.95
        T["M_SOTA_2"] = gm is not None and bar is not None and gm >= bar
        gf, tf = med(g, lambda r: r["compute"]["flops"]), med(tc, lambda r: r["compute"]["flops"])
        gr = med(g, lambda r: r["compute"]["flops"] - r["compute"].get("lm_head_rows", 0) * ROW)
        tr = med(tc, lambda r: r["compute"]["flops"] - r["compute"].get("lm_head_rows", 0) * ROW)
        T["flops_ratio"] = gf / tf if gf and tf else None
        T["flops_ratio_restricted"] = gr / tr if gr and tr else None
        T["seq_passes"] = {"gen2": med(g, lambda r: r["compute"].get("sequential_passes")), "tc": med(tc, lambda r: r["compute"].get("sequential_passes"))}
        T["M_C3_2"] = gm is not None and tm is not None and gm >= tm - 0.01 and T["flops_ratio"] is not None and T["flops_ratio"] <= 1.0
        mm = med(mk, acc); T["mask2_median"] = mm
        good = [s for s, r in mk.items() if r["acc_test"] >= 0.95]
        T["M_C1_2"] = (floor is not None and mm is not None and mm >= 0.95 and bool(good) and
                       all(mk[s]["removal"]["zero"]["answer_acc"] <= floor and mk[s]["removal"]["mean"]["answer_acc"] <= floor for s in good))
        T["removal"] = {a: {s: {m: r["removal"][m]["answer_acc"] for m in ("zero", "mean")} for s, r in rs.items()} for a, rs in (("gen2", g), ("mask2", mk))}
        T["text_checkpoint_seeds"] = [s for s, r in g.items() if max(r["removal"]["zero"]["answer_acc"], r["removal"]["mean"]["answer_acc"]) >= 0.95]
        T["per_chunk_acc"] = {a: {s: r.get("per_chunk_acc") for s, r in rs.items()} for a, rs in (("gen2", g), ("mask2", mk))}
        T["validity"] = {s: (r.get("checkpoint_validity") or {}).get("rate") for s, r in g.items()}
        T["edit_iia"] = {s: (r.get("edit_iia") or {}).get("iia") for s, r in g.items()}
        res[task] = T
        print(task, {k: T[k] for k in ("n", "icotsi_bar", "gen2_median", "M1_2", "M_SOTA_2", "tc_median", "flops_ratio",
                                       "flops_ratio_restricted", "M_C3_2", "mask2_median", "M_C1_2", "text_checkpoint_seeds")})
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s100.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
