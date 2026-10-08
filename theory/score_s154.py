"""THEORY §154 scoring (rules fixed in §154; written before any §154 implementation or run; Theory is the only scorer).
SHIFT-MEM: §147's digit-pair 4x4 DeltaMem recipe exactly (24k, restart slots k, k+100, k+200, k+300 with 147-1's misfire rule), the additive store addressed two ways:
  shift   --add_reset column: the harness zeroes the additive heads' state at every new-column segment (flag " ;") and at the flush (scaffold: hand-wired reset)
  rotary  --add_rotary column --rotary_read_offsets 0,0,0,-1,-1,-1: RoPE-rotated additive features by the pair's column index (3 heads read offset 0, 3 read -1)
Dirs boxes/runs_fsq/s154_{arm}_4x4_s{seed}.
PASS(arm) (§147's rule): >= 3/4 slots with test_acc >= 0.99 at 4x4 + SERIAL (row-level donor after pairs 4 / 8 / 12).
LENGTH-GENERAL(arm) (C2): >= 3/4 accepted runs with ext_4x8 test_acc >= 0.9 (all 12 digits; §147 0.0).
Proof: slot 1 per arm; the arm stops if slot 1 has 4x4 test_acc < 0.95 AND ext_4x8 < 0.5. IMPROVES (descriptive): 4x4 columns 3-7 all >= 0.95.
Descriptive: per-column 4x4 / 4x8, ext_4x16, colsum R2 by segment, the carry probe by column (4x4 / 4x8), alpha by head, the C7 edit test (shift; delta in -2, -1, +1, +2).
    python theory/score_s154.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
ARMS = {"shift": {"add_reset": "column"}, "rotary": {"add_rotary": "column", "rotary_read_offsets": "0,0,0,-1,-1,-1"}}
S147 = {"test_acc": 0.743, "ext_4x8": 0.0}                                  # §147 s1 (24k), the comparator
SKIPPED, ERRORS, MISFIRES = [], [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def smoke(d):
    a = load(d, "args.json")
    if not a:
        return False
    bad = (int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu")
           or bool(a.get("mem_check")) or bool(a.get("train_n")))
    if bad:
        SKIPPED.append(d)
    return bad


def args_ok(d, arm):
    a = load(d, "args.json") or {}
    exp = dict({"family": "mult_pair", "task": "4x4", "carrier": "deltamem", "mem_add_heads": 6, "mem_delta_heads": 6, "mem_group_gain": True}, **ARMS[arm])
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k) != v}
    other = "rotary" if arm == "shift" else "shift"
    for k in ARMS[other]:
        if a.get(k) and k not in ARMS[arm]:
            bad[k] = a.get(k)                                                # one addressing change per arm
    if a.get("copy_frac"):
        bad["copy_frac"] = a.get("copy_frac")
    if bad:
        ERRORS.append((d, bad))
    return not bad


def plateau(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    L = {x["step"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x and "step" in x}
    if not L or max(L) < 2000:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    fl = bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))
    if fl and g(2000) < 1.0:                                                 # note 147-1: a misfire on near-0 digit CE, logged, not a stop
        MISFIRES.append(d); return False
    return fl


def serial(dn):
    """Row level (multpair's row_level per position): donor >= 0.90, recipient - chance < 0.10 at every position."""
    if not dn:
        return False
    out = []
    for v in dn.values():
        x = v.get("row_level", v)
        out.append(x["donor_consistent"] >= 0.90 and x["recipient_consistent"] - x.get("chance_overlap", 0.0) < 0.10)
    return all(out)


def slot(arm, k):
    att = []
    for o in OFFS:
        d = "s154_%s_4x4_s%d" % (arm, k + o)
        if smoke(d):
            break
        pl = plateau(d); r = load(d)
        if pl is None and r is None:
            break
        if pl is None:
            return {"attempts": att, "status": "running (phase 1)", "accepted": None}
        att.append({"seed": k + o, "plateau": pl})
        if pl is False:
            if r is None:
                return {"attempts": att, "status": "running", "accepted": None}
            if not args_ok(d, arm):
                return {"attempts": att, "status": "error", "accepted": None}
            s = dict(r.get("s154") or {}); s.update(load(d, "eval_s154_posthoc.json") or {}); s7 = r.get("s147") or {}   # notes 154-1 / 154-2: post-hoc on host
            a = {"seed": k + o, "test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "per_column": (r.get("acc_by_column") or {}).get("per_column"),
                 "ext_4x8": (r.get("ext_4x8") or {}).get("test_acc"), "ext_4x8_per_column": ((r.get("ext_4x8") or {}).get("acc_by_column") or {}).get("per_column"),
                 "ext_4x16": (r.get("ext_4x16") or {}).get("test_acc"), "colsum_r2_by_segment": s.get("colsum_r2_by_segment"),
                 "carry_probe_by_column": s.get("carry_probe_by_column"), "c7_edit": s.get("c7_edit"), "alpha_by_head": s7.get("alpha_by_head"),
                 "c7_state_edit": s.get("c7_state_edit"), "acc_by_column_carry": s.get("acc_by_column_carry"), "alpha_by_flag": s.get("alpha_by_flag"),
                 "propagation": s.get("propagation"),                                                        # §155's null comparator (scored there)
                 "head_ablation": s7.get("head_ablation"), "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact"] = (a["test_acc"] or 0) >= 0.99
            a["pass"] = bool(a["exact"] and a["serial"])                   # §147: no contraction term (integrator by design)
            a["length_general"] = (a["ext_4x8"] or 0) >= 0.9
            a["reading"] = "IN-PASS / shortcut, not a latent carry (exact without SERIAL)" if (a["exact"] and not a["serial"]) else None
            return {"attempts": att, "status": "accepted", "accepted": a}
    if len(att) == len(OFFS) and all(x["plateau"] for x in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def verdicts(arm):
    sl = {k: slot(arm, k) for k in (1, 2, 3, 4)}
    s1 = sl[1]["accepted"]
    if s1 is not None and (s1["test_acc"] or 0) < 0.95 and (s1["ext_4x8"] or 0) < 0.5:
        return sl, "stopped at proof (slot 1: 4x4 %s < 0.95 and ext_4x8 %s < 0.5)" % (s1["test_acc"], s1["ext_4x8"]), "stopped at proof"
    if not any(v["attempts"] or v["status"] != "open" for v in sl.values()):
        return sl, "not run", "not run"
    acc = [v["accepted"] for v in sl.values() if v["accepted"]]
    decided = [v for v in sl.values() if v["status"] == "exhausted" or v["accepted"]]
    n_p = sum(1 for a in acc if a["pass"]); n_l = sum(1 for a in acc if a["length_general"])
    v_pass = "PASS" if n_p >= 3 else ("FAIL" if len(decided) - n_p >= 2 else "incomplete (%d / 4 slots decided)" % len(decided))
    v_len = "LENGTH-GENERAL" if n_l >= 3 else ("NOT LENGTH-GENERAL" if len(decided) - n_l >= 2 else "incomplete (%d / 4 slots decided)" % len(decided))
    return sl, v_pass, v_len


def improves(a):
    pc = (a or {}).get("per_column")
    return None if not pc or len(pc) < 7 else all(x >= 0.95 for x in pc[2:7])


def main():
    res = {"comparator_s147": S147}
    for arm in ARMS:
        sl, vp, vl = verdicts(arm); s1 = sl[1]["accepted"]
        res[arm] = {"slots": sl, "verdict": vp, "length": vl, "improves": improves(s1)}
        print(arm, "|", vp, "|", vl, {k: (s["status"], s["accepted"] and (s["accepted"]["test_acc"], s["accepted"]["serial"], s["accepted"]["ext_4x8"]))
                                      for k, s in sl.items()})
        if s1:
            print("   proof per column 4x4", s1["per_column"], "| 4x8", s1["ext_4x8_per_column"], "| ext_4x16", s1["ext_4x16"], "| IMPROVES:", improves(s1))
            print("   carry probe by column", s1["carry_probe_by_column"], "| C7 read edit (descriptive)", s1["c7_edit"])
            print("   C7 STATE edit (the test, note 154-2)", s1["c7_state_edit"], "| acc by carry-in", s1["acc_by_column_carry"], "| alpha by flag", s1["alpha_by_flag"])
    rd = []
    sh, ro = res["shift"], res["rotary"]
    if sh["length"] == "LENGTH-GENERAL":
        rd.append("SHIFT LENGTH-GENERAL: a memory carrier generalises in length when its active state is local (the Neural GPU principle; a task-structured reset, "
                  "stage 2 tests a learned one)")
    if ro["length"] == "LENGTH-GENERAL":
        rd.append("ROTARY LENGTH-GENERAL: relative keys suffice, with no reset")
    if sh["verdict"] == "PASS":
        rd.append("SHIFT PASS at 4x4 where §147 dropped (0.743): cross-column interference in the additive store was §147's ceiling")
    if sh["verdict"] == "PASS" and sh["length"] == "NOT LENGTH-GENERAL":
        rd.append("SHIFT passes 4x4 but not ext_4x8: the reset is not sufficient for addressing; the carry hand-off is still count-based (carry probe by column)")
    if sh["length"] == "NOT LENGTH-GENERAL" and ro["length"] == "NOT LENGTH-GENERAL":
        rd.append("neither generalises: the per-column computation or the emission does not transfer (the 4x8 per-column profile and the carry probe locate it)")
    res["reading"] = "; ".join(rd) if rd else "pending"; print("READING:", res["reading"])
    if MISFIRES:
        print("2k-flag misfires (not stops, note 147-1):", MISFIRES); res["misfires"] = MISFIRES
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s154.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
