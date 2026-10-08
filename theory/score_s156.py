"""THEORY §156 scoring (rules fixed in §156; written before any §156 implementation or run; Theory is the only scorer).
SEP-SLOT: a forced separator "|" (GPT-2 id 91) before each answer digit (LSB first), no loss on separators, digits predicted at the separator positions.
  (a) §148's PLAIN direct-answer 4x4 (loop148 --loops 1, 24k, the same recipe / GPU class): sep (interleaved) and pause (the same 8 "|" before the answer), seeds 1 / 2;
      comparator §148 PLAIN s1 (0.002, RTX 3090), or s156_plain_4x4_s1 if the arms ran on another GPU class.
  (b) §152 A's reader decode with separators: asep s1 (48k); comparator §152 A s1 (0.331).
(a) An arm reads >= 0.5 only if both seeds' test exact >= 0.5, < 0.5 only if both are below; mixed = seed-dependent (no verdict).
    SEP >= 0.5 & PAUSE < 0.5: a per-digit slot makes in-pass carry resolution learnable; both >= 0.5: positions, not placement; both < 0.5 with SEP - PAUSE >= 0.2 in
    both seed pairs: placement helps, not sufficient at this budget (descriptive); SEP < 0.5 otherwise: not position count at this budget.
(b) A-SEP s1 >= 0.95: slots close §152's 4-row resolution (s2-s4 + B-EARLY then follow under §152's rules as a new arm).
    python theory/score_s156.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
PLAIN = ("s148_plain_4x4_s1", 0.002)
EXP_A = {"family": "mult_direct", "task": "4x4", "steps": 24000, "sep_token": "|"}
EXP_B = {"family": "mult_row", "task": "4x4", "carrier": "deltamem", "reader": True, "seg_order": "b_first", "steps": 48000, "sep_token": "|", "answer_sep": "interleave"}
SKIPPED, ERRORS = [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def cell(d, exp):
    r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return None
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu") or float(a.get("sched_scale", 1.0)) != 1.0:
        SKIPPED.append(d); return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    s = r.get("s156") or {}
    return {"status": "done", "test_acc": r.get("test_acc"), "per_column": (r.get("acc_by_column") or {}).get("per_column"), "gpu": r.get("train_gpu", r.get("gpu")),
            "compute": r.get("compute"), "acc_by_length": r.get("acc_by_length"), "ring_probe": s.get("ring_probe"), "slot_patch": s.get("slot_patch"),
            "seq_len": s.get("seq_len")}


def arm_read(cells):
    accs = [c["test_acc"] for c in cells if c and c.get("status") == "done"]
    if len(accs) < 2:
        return "pending (%d / 2 seeds)" % len(accs), accs
    if all(x >= 0.5 for x in accs):
        return ">= 0.5", accs
    if all(x < 0.5 for x in accs):
        return "< 0.5", accs
    return "seed-dependent (no verdict)", accs


def main():
    res = {}
    sep = [cell("s156_sep_4x4_s%d" % s, dict(EXP_A, answer_sep="interleave")) for s in (1, 2)]
    pau = [cell("s156_pause_4x4_s%d" % s, dict(EXP_A, answer_sep="pause")) for s in (1, 2)]
    asep = cell("s156_asep_4x4_s1", EXP_B)
    gpus = {c["gpu"] for c in sep + pau if c and c.get("status") == "done"}
    pg = (load(PLAIN[0]) or {}).get("train_gpu")
    comp = PLAIN
    if gpus and gpus != {pg}:                                                 # Audit (3): one GPU class per experiment
        rp = cell("s156_plain_4x4_s1", dict(EXP_A, answer_sep=None, sep_token=None)) if load("s156_plain_4x4_s1") else None
        comp = ("s156_plain_4x4_s1", rp and rp.get("test_acc"))
        res["gpu_note"] = "arms ran on %s, not %s: PLAIN rerun on that class is the comparator" % (sorted(gpus), pg)
    rs, a_s = arm_read(sep); rp_, a_p = arm_read(pau)
    res.update({"sep": {"cells": sep, "read": rs}, "pause": {"cells": pau, "read": rp_}, "asep": asep, "comparator": comp})
    for nm, cs in (("SEP", sep), ("PAUSE", pau)):
        for i, c in enumerate(cs):
            print(nm, "s%d" % (i + 1), c and {k: c.get(k) for k in ("status", "test_acc", "per_column", "gpu", "seq_len", "error")})
    print("comparator", comp, "| A-SEP s1", asep and {k: asep.get(k) for k in ("status", "test_acc", "acc_by_length", "per_column", "error")})
    lens = {(c.get("seq_len") or None) for c in sep + pau if c and c.get("status") == "done"}
    if len(lens) > 1:
        ERRORS.append(("seq_len", "PAUSE / SEP total sequence lengths differ: %s (Audit (4))" % sorted(map(str, lens))))
    rd = []
    if rs == ">= 0.5" and rp_ == "< 0.5":
        rd.append("a per-digit slot, not more positions, makes in-pass carry resolution learnable in a pretrained LM (the middle-column wall was partly a missing-slot artefact)")
    elif rs == ">= 0.5" and rp_ == ">= 0.5":
        rd.append("extra positions, not their placement")
    elif rs == "< 0.5" and rp_ == "< 0.5" and len(a_s) == 2 and len(a_p) == 2 and all(sep[i]["test_acc"] - pau[i]["test_acc"] >= 0.2 for i in (0, 1)):
        rd.append("placement helps, not sufficient at this budget (descriptive; no follow-up implied)")
    elif rs == "< 0.5":
        rd.append("at this budget the wall is not about position count")
    elif rs.startswith("seed"):
        rd.append("SEP seed-dependent (no verdict)")
    if asep and asep.get("status") == "done":
        rd.append(("A-SEP s1 %.3f >= 0.95: slots close §152's 4-row resolution (s2-s4 + B-EARLY follow under §152's rules)" if (asep["test_acc"] or 0) >= 0.95
                   else "A-SEP s1 %.3f < 0.95: slots do not close §152's 4-row resolution at 48k") % asep["test_acc"])
    res["reading"] = "; ".join(rd) if rd else "pending"; print("READING:", res["reading"])
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s156.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
