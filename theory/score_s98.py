"""THEORY §98 scoring (rules fixed in §98; written before any §98 implementation or run; Theory is the only scorer).
Runs: boxes/runs_mult/m98_{genm,mask,coco,tc}_{4x4,5x5}_s{1,2,3}/eval.json; ICoT-SI: boxes/runs_mult/icotsi_{4x4,5x5}/eval.json.
Per task (seeds 1-3, medians):
M1      GEN-M median acc_test >= 0.95.
M-SOTA  GEN-M median acc_test >= the ICoT-SI bar = our eval of the released checkpoint (fallback: repo log 1.000 / 0.996).
M-C1    GEN-M-MASK median acc_test >= 0.95 AND, in those seeds, zero- and mean-removal answer accuracy <= No-CoT published
        (0.29 at 4x4, 0.01 at 5x5) + 0.05.
M-C3    GEN-M median acc_test >= TC-M median - 0.01 AND GEN-M median FLOPs <= TC-M median FLOPs (as metered); the
        restricted-vocabulary ratio (lm_head_rows removed on both sides) is reported.
Descriptive: COCO-M under M1's bar with removal; GEN-M - COCO-M paired by seed; per-slot decode; validity / edit IIA
(5x5); sequential passes; C2 4x8 (range-limited).
    python theory/score_s98.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "boxes", "runs_mult")
NC = {"4x4": 0.29, "5x5": 0.01}
ICOT_FALLBACK = {"4x4": 1.000, "5x5": 0.996}
ROW = 2 * 768 * 50257


def load(arm, task):
    out = {}
    for s in (1, 2, 3):
        f = os.path.join(RUNS, "m98_%s_%s_s%d" % (arm, task, s), "eval.json")
        if os.path.exists(f):
            out[s] = json.load(open(f))
    return out


def med(runs, fn):
    v = [fn(r) for r in runs.values()]
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def restricted(r):
    c = r["compute"]
    return c["flops"] - c.get("lm_head_rows", 0) * ROW


def main():
    res = {}
    for task in ("4x4", "5x5"):
        A = {a: load(a, task) for a in ("genm", "mask", "coco", "tc")}
        f = os.path.join(RUNS, "icotsi_%s" % task, "eval.json")
        icot = json.load(open(f)) if os.path.exists(f) else None
        bar = icot["acc_test"] if icot else ICOT_FALLBACK[task]
        g = A["genm"]; T = {"n": {a: len(r) for a, r in A.items()}, "icotsi_bar": bar, "icotsi_evaluated": icot is not None}
        acc = lambda r: r["acc_test"]                                                          # noqa: E731
        T["acc"] = {a: {s: r["acc_test"] for s, r in rs.items()} for a, rs in A.items()}
        T["genm_median"] = med(g, acc)
        T["M1"] = T["genm_median"] is not None and T["genm_median"] >= 0.95
        T["M_SOTA"] = T["genm_median"] is not None and T["genm_median"] >= bar
        mk = A["mask"]; ok_seeds = [s for s, r in mk.items() if r["acc_test"] >= 0.95 and
                                    r["removal"]["zero"]["answer_acc"] <= NC[task] + 0.05 and r["removal"]["mean"]["answer_acc"] <= NC[task] + 0.05]
        T["mask_median"] = med(mk, acc)
        T["M_C1"] = T["mask_median"] is not None and T["mask_median"] >= 0.95 and \
            all(mk[s]["removal"]["zero"]["answer_acc"] <= NC[task] + 0.05 and mk[s]["removal"]["mean"]["answer_acc"] <= NC[task] + 0.05
                for s in mk if mk[s]["acc_test"] >= 0.95) and len(ok_seeds) >= 1
        T["mask_seeds_passing_removal"] = ok_seeds
        tc = A["tc"]; T["tc_median"] = med(tc, acc)
        gf, tf = med(g, lambda r: r["compute"]["flops"]), med(tc, lambda r: r["compute"]["flops"])
        T["flops_ratio_genm_tc"] = gf / tf if gf and tf else None
        gr, tr = med(g, restricted), med(tc, restricted)
        T["flops_ratio_restricted"] = gr / tr if gr and tr else None
        T["M_C3"] = (T["genm_median"] is not None and T["tc_median"] is not None and T["genm_median"] >= T["tc_median"] - 0.01
                     and T["flops_ratio_genm_tc"] is not None and T["flops_ratio_genm_tc"] <= 1.0)
        T["removal"] = {a: {s: {m: r.get("removal", {}).get(m, {}).get("answer_acc") for m in ("zero", "mean")} for s, r in A[a].items()}
                        for a in ("genm", "mask", "coco")}
        T["coco_median"] = med(A["coco"], acc)
        T["genm_minus_coco_paired"] = [g[s]["acc_test"] - A["coco"][s]["acc_test"] for s in g if s in A["coco"]]
        T["per_slot_decode"] = {a: {s: r.get("per_slot_decode_acc") for s, r in A[a].items()} for a in ("genm", "mask")}
        T["seq_passes"] = {a: med(A[a], lambda r: r["compute"].get("sequential_passes")) for a in A}
        if icot:
            T["icotsi"] = {"acc": icot["acc_test"], "flops": icot.get("compute", {}).get("flops"),
                           "seq_passes": icot.get("compute", {}).get("sequential_passes")}
        if task == "5x5":
            T["validity"] = {s: (r.get("checkpoint_validity") or {}).get("rate") for s, r in g.items()}
            T["edit_iia"] = {s: (r.get("edit_iia") or {}).get("iia") for s, r in g.items()}
        else:
            T["c2_4x8"] = {s: (r.get("c2_4x8") or {}).get("acc") for s, r in g.items()}
        res[task] = T
        print(task, {k: T[k] for k in ("n", "icotsi_bar", "genm_median", "M1", "M_SOTA", "mask_median", "M_C1", "tc_median",
                                       "flops_ratio_genm_tc", "flops_ratio_restricted", "M_C3", "coco_median")})
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s98.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
