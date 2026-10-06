"""THEORY §131 scoring (rules fixed in §131; written before any held-out run is read; Theory is the only scorer).
A label-free second restart point: FLAG4 = loss(4k) >= 0.9 * loss(2.6k) AND loss(4k) >= 0.25 * loss(100) (nearest logged
steps; the floor as 123-3's). The windowed form (means over 3.8-4k vs 2.4-2.6k, Literature 2d097e1c) is reported beside it,
descriptively: in-sample it lowered recall from 9/13 to 7/13 (slowly rising runs average under the threshold). Target: the run does not learn (final long_acc < 0.99).
IN_SAMPLE (fixed at registration): the 30 fsq120 S5 runs on disk with >= 4k logged steps and an eval.json when the rule was
derived (Audit / claude-13). HELD-OUT: every other fsq120 run (§127, §128 any family, s130chk) with >= 4k steps, an eval.json,
and no 2k plateau stop; §129 continuations (schedule clock restarted) are reported separately, not scored.
PASS: (i) no held-out run that ends exact (long_acc >= 0.99) is flagged, with >= 5 exact held-out runs; and (ii) >= 2/3 of the
held-out runs that escape the 2k flag and end long_acc < 0.99 are flagged, with >= 3 such runs. Otherwise "incomplete" with
counts. Precision / recall with two-sided 95 % Clopper-Pearson intervals are reported; the ~3.2-basin misses are a named class.
    python theory/score_s131.py
"""
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
IN_SAMPLE = {
    "s120_s5_final_s1", "s120_s5_final_s2", "s120_s5_trace_s1", "s120_s5_trace_s2", "s123_s5_dither000_s1", "s123_s5_dither000_s2",
    "s123_s5_dither010_s1", "s123_s5_dither010_s2", "s123_s5_dither025_s1", "s123_s5_dither025_s2", "s123_s5_dither050_s1",
    "s123_s5_dither050_s1_3090", "s123_s5_dither050_s2", "s123_s5_dither050_s2_3090", "s123_s5_fsqdither_s1", "s123_s5_fsqdither_s2",
    "s125_s5_std_s2", "s126_s5_anneal_s2", "s126_s5_anneal_s2_fp32cast", "s126_s5_rawh_s1", "s126_s5_rawh_s2", "s126_s5_rawh_s3",
    "s126_s5_rawh_s4", "s127_s5_lr10_s2", "s127_s5_skip_s2", "s127_s5_skip_s2_rerun", "s127_s5_skip_s4", "s127_s5_std2_s1",
    "s127_s5_std2_s2", "s127_s5_var_s2_rerun"}
SAME_TRAJECTORY = {"s127_s5_skip_s2", "s128_s5_s3", "s128_s5_s1"}   # 131-8: s128_s5_s1 is record-for-record s120_s5_final_s1 (in-sample) to 14k; counted once, in-sample   # s128_s5_s3 is record-for-record s127_s5_ctrl_s3 (Audit, 08:20); each counted once   # 131-1 (b): identical to s127_s5_skip_s2_rerun up to 9k (ratio 1.057 both); counted once, as the rerun
HELD = re.compile(r"^(s127_|s128_|s130chk_|s131x_)")   # 131-3: s131x_* added before those runs exist
SEEN_AT_REG = set(["s127_ff_skip_s1", "s127_ff_skip_s2", "s128_cyc_s1", "s128_cyc_s2", "s128_cyc_s3", "s128_mod_s1", "s128_mod_s2", "s128_mod_s3", "s128_mod_s4", "s128_rdfa_s1", "s128_rdfa_s2", "s128_rdfa_s3", "s128_rdfa_s4"])   # held-out runs on disk when the scorer first ran (all exact, none flagged); the rest are "prospective"


def loss_map(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None, False
    L, stopped = {}, False
    for line in open(p):
        x = json.loads(line)
        if "loss" in x and "step" in x and x.get("phase", 1) == 1:
            L[x["step"]] = x["loss"]
        if str(x.get("stopped", "")).startswith("plateau"):
            stopped = True
    return L, stopped


def ratios(L):
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    w = lambda lo, hi: sum(v for s, v in L.items() if lo <= s <= hi) / max(1, sum(1 for s in L if lo <= s <= hi))  # noqa: E731
    f2 = bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))
    f4 = bool(g(4000) >= 0.9 * g(2600) and g(4000) >= 0.25 * g(100))      # the registered rule: single nearest points
    a4, a26 = w(3800, 4000), w(2400, 2600)                                # windowed form (Literature 2d097e1c), descriptive only
    return f2, f4, g(4000) / g(2600), g(4000), bool(a4 >= 0.9 * a26 and a4 >= 0.25 * g(100))


def cp(x, n, alpha=0.05):
    from math import comb
    if n == 0:
        return (None, None)
    def tail(p, ge):
        ks = range(x, n + 1) if ge else range(0, x + 1)
        return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in ks)
    def bis(ge, target):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (tail(mid, ge) < target) == ge:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return (0.0 if x == 0 else bis(True, alpha / 2), 1.0 if x == n else bis(False, alpha / 2))


def collect(names):
    out = {}
    for d in names:
        L, stopped = loss_map(d)
        ev = os.path.join(R, d, "eval.json")
        if L is None or stopped or not L or max(L) < 4000 or not os.path.exists(ev):
            continue
        f2, f4, r, l4, f4w = ratios(L)
        e = json.load(open(ev)); a = e.get("long_acc")
        fam = d.split("_")[1] if d.startswith(("s127_", "s128_", "s131x_")) else ("s5" if "_s5_" in d else "?")
        out[d] = {"family": fam, "flag2k": f2, "flag4k": f4, "flag4k_windowed": f4w, "ratio": round(r, 3), "loss_4k": round(l4, 3), "long_acc": a, "exact": (a or 0) >= 0.99,
                  "final_loss": e.get("final_loss"), "gpu": e.get("train_gpu", e.get("gpu"))}
    return out


def summary(runs, label):
    ex = [d for d, v in runs.items() if v["exact"]]; fail = [d for d, v in runs.items() if not v["exact"] and not v["flag2k"]]
    fp = [d for d in ex if runs[d]["flag4k"]]; tp = [d for d in fail if runs[d]["flag4k"]]
    miss = [d for d in fail if not runs[d]["flag4k"]]
    prec = len(tp) / (len(tp) + len(fp)) if tp or fp else None; rec = len(tp) / len(fail) if fail else None
    print("%s: exact %d (flagged %d) | escaped-and-failed %d (flagged %d) | precision %s %s | recall %s %s" % (
        label, len(ex), len(fp), len(fail), len(tp), prec, cp(len(tp), len(tp) + len(fp)), rec, cp(len(tp), len(fail))))
    print("  misses:", {d: (runs[d]["ratio"], runs[d]["final_loss"]) for d in miss}, "| false positives:", {d: runs[d]["ratio"] for d in fp})
    ex_s5 = [d for d in ex if runs[d]["family"] == "s5"]; ex_nf2 = [d for d in ex if not runs[d]["flag2k"]]
    print("  (i) restricted: S5 learners flagged %d / %d (131-1 (a)); learners not flagged at 2k flagged %d / %d (131-1 (d))" % (
        sum(runs[d]["flag4k"] for d in ex_s5), len(ex_s5), sum(runs[d]["flag4k"] for d in ex_nf2), len(ex_nf2)))
    print("  windowed form (descriptive): exact flagged %d / %d, escaped-and-failed flagged %d / %d" % (
        sum(runs[d]["flag4k_windowed"] for d in ex), len(ex), sum(runs[d]["flag4k_windowed"] for d in fail), len(fail)))
    return {"exact_s5": ex_s5, "fp_s5": [d for d in ex_s5 if runs[d]["flag4k"]], "exact_not_flag2k": ex_nf2,
            "fp_not_flag2k": [d for d in ex_nf2 if runs[d]["flag4k"]], "exact": ex, "false_pos": fp, "escaped_failed": fail, "true_pos": tp, "misses": miss, "precision": prec, "recall": rec,
            "precision_ci95": cp(len(tp), len(tp) + len(fp)), "recall_ci95": cp(len(tp), len(fail))}


def main():
    names = sorted(d for d in os.listdir(R) if os.path.isdir(os.path.join(R, d))) if os.path.isdir(R) else []
    ins = collect([d for d in names if d in IN_SAMPLE and d not in SAME_TRAJECTORY])
    held = collect([d for d in names if HELD.match(d) and d not in IN_SAMPLE and d not in SAME_TRAJECTORY and ".void" not in d])   # 131-1 (c)
    s_in = summary(ins, "IN-SAMPLE (descriptive)"); s_h = summary(held, "HELD-OUT")
    def verdict_of(s):
        ok_n = len(s["exact"]) >= 5 and len(s["escaped_failed"]) >= 3
        return (bool(not s["false_pos"] and len(s["true_pos"]) >= (2 / 3) * len(s["escaped_failed"])) if ok_n else
                "incomplete (%d exact, need >= 5; %d escaped-and-failed, need >= 3)" % (len(s["exact"]), len(s["escaped_failed"])))
    verdict = verdict_of(s_h)
    pro = {d: v for d, v in held.items() if d not in SEEN_AT_REG}; s_p = summary(pro, "PROSPECTIVE (held-out minus seen-at-registration)")
    print("PASS (all held-out)", verdict, "| PASS (prospective)", verdict_of(s_p))
    json.dump({"in_sample": ins, "held_out": held, "summary_in": s_in, "summary_held": s_h, "PASS": verdict,
               "summary_prospective": s_p, "PASS_prospective": verdict_of(s_p)},
              open(os.path.join(ROOT, "theory", "results", "score_s131.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
