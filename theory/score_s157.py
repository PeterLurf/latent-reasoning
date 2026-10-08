"""THEORY §157 scoring (rules fixed in §157; written before any §157 implementation or run; Theory is the only scorer).
READ-PANEL: §152 A's recipe exactly (multiplier-major, b-first, the teacher S_j reader, 48k, the gate), one read change per arm (multrow --read_mode):
  periodic   scalar channels per additive head (STE-rounded writes) + a fixed [s, cos, sin](2 pi s / P), P in {10, 100, 1000} featuriser (fp32), zero-init map
  multiread  R = 4 learned queries per head -> 4 read-token positions at every pass start (RMT-style), the per-position read kept; zero-init output map
  ccq        a curvature-conditioned query (running key covariance per head; the read step only)
Condition: run only if s152_a96k_4x4_s1 (b165) reads test_acc < 0.95. Gate per arm: s1 >= 0.95 opens s2-s4 under §152's PASS rule (>= 3/4 >= 0.99 + SERIAL).
Readings: one arm alone >= 0.95 names the block (geometry / bandwidth / interference); several: the read, cheapest named; none: not the read (§155 remains).
Dirs boxes/runs_fsq/s157_{arm}_4x4_s{seed}.
    python theory/score_s157.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s152 as S152  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = {"periodic": "geometry", "multiread": "bandwidth", "ccq": "interference"}
BASE = dict(S152.BASE, reader=True, seg_order="b_first", label_source="teacher_generated")
SKIPPED, ERRORS = [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def cell(arm, seed):
    d = "s157_%s_4x4_s%d" % (arm, seed); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return None
    if int(a.get("steps", 0) or 0) != 48000 or a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        SKIPPED.append(d); return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in dict(BASE, read_mode=arm).items() if a.get(k) != v}
    if a.get("removal") or int(a.get("fixed_L") or 0) or a.get("stop_plateau"):
        bad["other"] = (a.get("removal"), a.get("fixed_L"), a.get("stop_plateau"))
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    s = r.get("s157") or {}
    return {"status": "done", "test_acc": r.get("test_acc"), "acc_by_length": r.get("acc_by_length"), "serial": S152.serial(r.get("donor")),
            "digit_acc_tf": s.get("digit_acc_teacher_forced"), "read_exposure": s.get("read_exposure"), "write_round_err": s.get("write_round_err"),
            "offcol_weight": s.get("offcol_weight"), "c7_scalar_edit": s.get("c7_scalar_edit"), "compute": r.get("compute"), "gpu": r.get("train_gpu")}


def main():
    res = {}
    g = load("s152_a96k_4x4_s1")
    cond = "pending (b165 not read)" if g is None else ("RUN (b165 test_acc %s < 0.95)" % g.get("test_acc") if (g.get("test_acc") or 0) < 0.95
                                                       else "NOT RUN (b165 test_acc %s >= 0.95: budget alone fixed column 6)" % g.get("test_acc"))
    res["condition"] = cond; print("CONDITION:", cond)
    gate = {}
    for arm in ARMS:
        runs = {s: cell(arm, s) for s in (1, 2, 3, 4)}
        s1 = runs[1]
        gate[arm] = None if not s1 or s1.get("status") != "done" else (s1["test_acc"] or 0) >= 0.95
        v = S152.verdict(arm, runs) if gate[arm] else ("gate closed (s1 %s < 0.95)" % s1["test_acc"] if gate[arm] is False else "pending")
        res[arm] = {"runs": runs, "gate_open": gate[arm], "verdict": v}
        print(arm, "| gate", gate[arm], "|", v, "|", s1 and {k: s1.get(k) for k in ("status", "test_acc", "acc_by_length", "digit_acc_tf", "read_exposure", "error")})
    passed = [a for a in ARMS if gate[a]]
    if any(gate[a] is None for a in ARMS):
        rd = "pending"
    elif len(passed) == 1:
        rd = "%s alone >= 0.95: read %s was the block" % (passed[0].upper(), ARMS[passed[0]])
    elif passed:                                                             # Audit: cheapest passing arm by Meter FLOPs (vs TC-M)
        fl = {a: ((res[a]["runs"][1] or {}).get("compute") or {}).get("flops") for a in passed}
        ch = min(passed, key=lambda a: fl[a] if fl[a] else float("inf"))
        rd = "several pass (%s): the read was the block; cheapest by FLOPs: %s (%s)" % (passed, ch, {a: (fl[a] and round(fl[a] / S152.TCM["flops"], 2)) for a in passed})
    else:
        rd = "none >= 0.95: not the read; §155's recurrence is the remaining route"
    res["reading"] = rd; print("READING:", rd, "| comparator §152 A s1 at 48k: 0.331 (column 6 0.338); budget control b165:", g and g.get("test_acc"))
    if (res.get("multiread") or {}).get("gate_open"):
        print("CONDITIONAL NEXT ARM: MULTI-READ-COPY (R = 4 positions with copies of one query's read): bandwidth vs positions")
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s157.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
