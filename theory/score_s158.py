"""THEORY §158 scoring (rules fixed in §158; written before any §158 implementation or run; Theory is the only scorer).
FREE-PAIRS: §155 R's recipe exactly (DeltaMem-R, top writes, gamma 0 to 2k -> 1 at 6k, §147's digit-pair 4x4, 24k, restart slots k, k+100, k+200, k+300) plus
--free_pairs 0.5: half the training chains draw every pair's digits independently (same (i, j) schedule and flags), label = the digits of
sum_t a^(t) b^(t) 10^(i_t + j_t - 2) (LSB first, 8 digits). Test = the consistent bench 4x4 test. Dirs boxes/runs_fsq/s158_free_4x4_s{seed}.
PASS: >= 3/4 slots with test_acc >= 0.99 and SERIAL (row-level swap after pairs 4 / 8 / 12: donor >= 0.90, recipient - chance < 0.10).
Readings: PASS = R's swap gap was data-induced (an operand-consistency shortcut); exact but not SERIAL = not data-induced (the state code, 155-4); test < 0.95 = budget.
Descriptive: free-pairs held-out accuracy, 155-4 what_state, RECURRENT USE in 155-3's form, ext_4x8.
    python theory/score_s158.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s155 as S155  # noqa: E402  (slot machinery pattern, plateau / misfire rule, SERIAL, propagation helpers)

R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
EXP = {"family": "mult_pair", "task": "4x4", "carrier": "deltamem", "mem_add_heads": 6, "mem_delta_heads": 6, "mem_group_gain": True, "mem_write": "top",
       "write_mix_start": 2000, "write_mix_end": 6000, "free_pairs": 0.5}
SKIPPED, ERRORS = [], []
load = S155.load


def slot(k):
    att = []
    for o in OFFS:
        d = "s158_free_4x4_s%d" % (k + o)
        if S155.smoke(d):
            SKIPPED.append(d); break
        pl = S155.plateau(d); r = load(d)
        if pl is None and r is None:
            break
        if pl is None:
            return {"attempts": att, "status": "running (phase 1)", "accepted": None}
        att.append({"seed": k + o, "plateau": pl})
        if pl is False:
            if r is None:
                return {"attempts": att, "status": "running", "accepted": None}
            a_ = load(d, "args.json") or {}
            bad = {kk: a_.get(kk) for kk, v in EXP.items() if a_.get(kk) != v}
            if a_.get("add_reset") or a_.get("add_rotary"):
                bad["other"] = (a_.get("add_reset"), a_.get("add_rotary"))
            if bad:
                ERRORS.append((d, bad)); return {"attempts": att, "status": "error", "accepted": None}
            s = dict(r.get("s155") or {}); s.update(load(d, "eval_s155_posthoc.json") or {}); s8 = dict(r.get("s158") or {})
            ph8 = ((load(d, "eval_s158_posthoc.json") or {}).get("s158") or {})   # the free-item propagation (post-hoc on the host)
            if ph8.get("by_item_type"):
                bit = dict(s8.get("by_item_type") or {})
                for kk, vv in ph8["by_item_type"].items():
                    bit[kk] = dict(bit.get(kk) or {}, **vv)
                s8["by_item_type"] = bit
            ws = load(d, "eval_what_state.json")
            ps, rs, nv = S155.propagation(s)
            a = {"seed": k + o, "test_acc": r.get("test_acc"), "serial": S155.S154.serial(r.get("donor")), "per_column": (r.get("acc_by_column") or {}).get("per_column"),
                 "free_pairs_test_acc": s8.get("free_pairs_test_acc"), "ext_4x8": (r.get("ext_4x8") or {}).get("test_acc"),
                 "donor_row": {kk: (v.get("row_level") or {}).get("donor_consistent") for kk, v in (r.get("donor") or {}).items()},
                 "propagation_share": ps, "random_edit_share": rs, "null_value": nv,
                 "recurrent_use_155_3": bool(ps is not None and rs is not None and nv is not None and ps - nv >= 0.5 and rs - nv < 0.1),
                 "what_state": (ws or {}).get("what_state"), "acc_vs_raw": s8.get("acc_vs_raw"), "by_item_type": s8.get("by_item_type"),
                 "gpu": r.get("train_gpu", r.get("gpu"))}
            av = a["acc_vs_raw"] or {}                                       # note 158-1 (b): {"acc_exact": x, "raw_digit_acc": y} pooled over pairs 4 / 8 / 12
            a["accumulator_dominates"] = bool(av.get("acc_exact") is not None and av.get("raw_digit_acc") is not None and av["acc_exact"] >= 0.9
                                              and av["acc_exact"] >= av["raw_digit_acc"])
            a["c1_credit"] = bool(a["recurrent_use_155_3"] and a["accumulator_dominates"])                         # note 158-1: (a) and (b)
            a["exact"] = (a["test_acc"] or 0) >= 0.99
            a["pass"] = bool(a["exact"] and a["serial"])
            return {"attempts": att, "status": "accepted", "accepted": a}
    if len(att) == len(OFFS) and all(x["plateau"] for x in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def main():
    sl = {k: slot(k) for k in (1, 2, 3, 4)}
    acc = [v["accepted"] for v in sl.values() if v["accepted"]]
    decided = [v for v in sl.values() if v["status"] == "exhausted" or v["accepted"]]
    n = sum(1 for a in acc if a["pass"])
    v = "PASS" if n >= 3 else ("FAIL" if len(decided) - n >= 2 else "incomplete (%d / 4 slots decided)" % len(decided)) if decided else "not run"
    res = {"slots": sl, "verdict": v}
    for k, x in sl.items():
        a = x["accepted"]
        print("slot", k, x["status"], a and {kk: a.get(kk) for kk in ("seed", "test_acc", "serial", "donor_row", "free_pairs_test_acc", "ext_4x8", "propagation_share",
                                                                    "null_value", "recurrent_use_155_3")})
    s1 = sl[1]["accepted"]
    nc1 = sum(1 for a in acc if a.get("c1_credit"))
    if v == "PASS":                                                          # note 158-1: SERIAL here is a sanity check, not C1 evidence
        rd = "PASS: the swap gap was data-induced (the operand-consistency shortcut)"
        rd += ("; C1 credit (carry-edit propagation + accumulator dominates) in %d / %d accepted runs%s" %
               (nc1, len(acc), " -> C1 on this harness" if nc1 >= 3 else " -> no C1 credit (the IIT signature if SERIAL passes without it)"))
    elif s1 and s1["exact"] and not s1["serial"]:
        rd = "exact but not SERIAL: the swap gap is not data-induced -> the state code, per 155-4's result"
    elif s1 and (s1["test_acc"] or 0) < 0.95:
        rd = "test_acc < 0.95: free pairs are the harder task (no operand reuse): budget"
    else:
        rd = "pending"
    res["reading"] = rd; print("VERDICT", v, "| READING:", rd)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s158.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
