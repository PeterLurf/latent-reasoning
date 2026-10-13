"""THEORY §198 scoring (CHEAP-THINK family; rules fixed in §198; written before any §198 implementation or run; Theory is the only scorer).
One all-cached family on nested, seed 1, RTX 3090, the RMS + per-pass step-aux recipe (s185_rms_s1's args):
  (a) s198_injc_s1   = + --cache_lower                       (cached INJECT, memory at block 6: the family comparator)
  (b) s198_memhi_s1  = + --cache_lower --mem_layer 9          (MEM-HIGH: each pass recomputes blocks 9..11 only)
  (c) s198_rsaux_s1  = --arm readslots (M slots over the once-encoded input KV) + RMS + step aux
Cost (block-token forwards per item, K passes, L think-input tokens, 12 blocks) relative to the CURRENT uncached INJECT (K+1)*L*12; the cost clause is
evaluated at the natural-task target L = 300 (the nested-L ratio is reported beside; (b)'s ratio is L-independent):
  (a) L*12 + K*L*6; (b) L*12 + K*L*3; (c) L*12 + K*M*12.  L = eval readslots.L_mean, else mean(3n + 2, n 3..8).
For (b) and (c) vs (a): CHEAPER-EQUAL = DEEP and THINK-NECESSARY and DEPTH >= (a)'s - 0.05 and cost ratio <= 0.5 (opens s2);
CHEAPER-WORSE = DEEP, DEPTH below; UNCLEAR-ENTRY = not DEEP and entry(0.5) never (seed 2 = a new decision); FAILS = entered, not DEEP.
(a) vs s185_rms_s1 (uncached, same recipe) is descriptive: the cached-family check. Meter FLOPs reported if present.
    python theory/score_s198.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basin_entry as BE  # noqa: E402
import score_s185 as S185  # noqa: E402
import score_s187 as S187  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
EPS = 1e-9
RMS = dict(S185.ARMS["rms"], seed=1)
CELLS = {"a": ("s198_injc_s1", dict(RMS, cache_lower=True)),
         "b": ("s198_memhi_s1", dict(RMS, cache_lower=True, mem_layer=9)),
         "c": ("s198_rsaux_s1", dict(RMS, arm="readslots"))}


def cost(key, d):
    e = json.load(open(os.path.join(R, d, "eval.json"))); a = json.load(open(os.path.join(R, d, "args.json")))
    K, M = int(a.get("think_k", 8)), int(a.get("n_slots", 8)); rs = e.get("readslots") or {}
    L = float(rs.get("L_mean") or sum(3 * n + 2 for n in range(3, 9)) / 6)
    f = lambda L_: ({"a": L_ * 12 + K * L_ * 6, "b": L_ * 12 + K * L_ * 3, "c": L_ * 12 + K * M * 12}[key] / ((K + 1) * L_ * 12))   # noqa: E731
    return {"L_mean": L, "K": K, "M": M, "ratio_at_nested_L": f(L), "ratio_vs_current": f(300.0), "meter_flops": (e.get("meter") or rs).get("meter_flops")}   # cost clause at L = 300 (natural-task target)


def main():
    res = {k: S187.read(d, w) for k, (d, w) in CELLS.items()}; out = {"cells": res}
    for k, (d, _) in CELLS.items():
        if res[k].get("status") == "done":
            res[k]["cost"] = cost(k, d); res[k]["entry"] = BE.summary(d)
    lines = []
    a = res["a"]
    if a.get("status") == "done":
        u = S187.read("s185_rms_s1", RMS)
        lines.append("(a) cached INJECT: DEEP %s, DEPTH %.3f (uncached s185_rms_s1 %.3f), entry %s/%s, cost ratio %.3f" % (
            a["deep"], a["depth_9_12_n2"], u.get("depth_9_12_n2", float("nan")), a["entry"]["entry_0.5"], a["entry"]["entry_0.1"], a["cost"]["ratio_vs_current"]))
    for k in ("b", "c"):
        c = res[k]
        if c.get("status") != "done" or a.get("status") != "done":
            lines.append("(%s) pending (cell %s; comparator (a) %s)" % (k, c.get("status"), a.get("status"))); continue
        if not c["deep"]:
            lab = "UNCLEAR-ENTRY" if c["entry"]["entry_0.5"] is None else "FAILS"
        elif c["think_necessary"] and c["depth_9_12_n2"] >= a["depth_9_12_n2"] - 0.05 - EPS and c["cost"]["ratio_vs_current"] <= 0.5 + EPS:
            lab = "CHEAPER-EQUAL"
        else:
            lab = "CHEAPER-WORSE"
        c["label"] = lab
        lines.append("(%s) %s: DEEP %s, THINK-NECESSARY %s, DEPTH %.3f vs (a) %.3f; cost ratio vs current %.3f at L 300 (%.3f at nested L; Meter %s); entry %s/%s" % (
            k, lab, c["deep"], c["think_necessary"], c["depth_9_12_n2"], a["depth_9_12_n2"], c["cost"]["ratio_vs_current"], c["cost"]["ratio_at_nested_L"], c["cost"]["meter_flops"],
            c["entry"]["entry_0.5"], c["entry"]["entry_0.1"]))
    out["reading"] = " | ".join(lines); print("READING:", out["reading"])
    errs = S187.ERRORS
    if errs:
        print("ERRORS:", errs)
    out["errors"] = [str(x) for x in errs]
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s198.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
