"""THEORY §155 scoring (rules fixed in §155; written before any §155 implementation or run; Theory is the only scorer).
DELTA-R: §147's digit-pair 4x4 DeltaMem with WRITE-AFTER-READ: k, v, alpha, beta from u = (1 - gamma) LN_6(h_6) + gamma LN_w(h_top) (gamma 0 to 2k, linear to 1
at 6k), h_top = the residual after block 11; reads at block-6 entry from the pass-start state. 24k, restart slots k, k+100, k+200, k+300 (147-1's misfire rule).
Arms (dirs boxes/runs_fsq/s155_{arm}_4x4_s{seed}): r (no reset), shiftr (+ §154's per-column reset of the additive store); fallback arms r9 / shiftr9 (writes
from block 9's output), run only if the arm's top-write slot 1 shows non-finite steps, a write-gradient blow-up (> 100x its 1k-step median) or exhausts its slot.
PASS(arm) (§147's rule): >= 3/4 slots test_acc >= 0.99 + SERIAL (row level, pairs 4 / 8 / 12). LENGTH-GENERAL: >= 3/4 accepted runs with ext_4x8 >= 0.9.
RECURRENT USE (shiftr): >= 3/4 accepted runs with propagation_share >= 0.8 (a rank-1 state edit of the delta heads moving column k+1's carry-in probe by delta at its
first pair; column k+2's digit follows the edited chain, on rows where the edit changes column k+2's carry) AND the equal-norm random-edit null < 0.3.
Null comparator: §154 SHIFT scored the same way (expected = its unedited-next rate).
    python theory/score_s155.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s154 as S154  # noqa: E402  (§154's slot machinery pattern; SHIFT is the no-recurrence null)

R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
ARMS = {"r": {"mem_write": "top"}, "shiftr": {"mem_write": "top", "add_reset": "column"},
        "r9": {"mem_write": "block9"}, "shiftr9": {"mem_write": "block9", "add_reset": "column"},
        "rg0": {"mem_write": "top", "write_mix_start": 1000000000, "write_mix_end": 1000000000}}   # note 155-1 (3): gamma held at 0 (block-6 writes, pass-start reads)
MIGRATE = {"write_mix_start": 2000, "write_mix_end": 6000}                 # gamma 0 until 2k, linear to 1 at 6k
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
    exp = dict({"family": "mult_pair", "task": "4x4", "carrier": "deltamem", "mem_add_heads": 6, "mem_delta_heads": 6, "mem_group_gain": True}, **MIGRATE)
    exp.update(ARMS[arm])                                                    # the arm's own keys override (rg0 holds gamma at 0)
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k) != v}
    if "add_reset" not in ARMS[arm] and a.get("add_reset"):
        bad["add_reset"] = a.get("add_reset")
    if a.get("add_rotary") or a.get("copy_frac"):
        bad["other"] = (a.get("add_rotary"), a.get("copy_frac"))
    if bad:
        ERRORS.append((d, bad))
    return not bad


def plateau(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    L = {x["step"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x and "step" in x and x["loss"] is not None}
    if not L or max(L) < 2000:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    fl = bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))
    if fl and g(2000) < 1.0:                                                 # note 147-1: a misfire on near-0 digit CE, logged, not a stop
        MISFIRES.append(d); return False
    return fl


def propagation(s):
    """(propagation_share, random-edit share, null value) pooled over k and delta, from s155.propagation / s154.propagation."""
    pr = (s or {}).get("propagation") or {}
    pooled = pr.get("pooled") or {}
    return pooled.get("propagation_share"), pooled.get("random_edit_share"), pooled.get("null_value")


def coupling(s):
    """note 155-2: the share of rows where a non-target column changes under the carry edit (pooled)."""
    return (((s or {}).get("propagation") or {}).get("pooled") or {}).get("coupling_share")


def slot(arm, k):
    att = []
    for o in OFFS:
        d = "s155_%s_4x4_s%d" % (arm, k + o)
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
            s = dict(r.get("s155") or {}); s.update(load(d, "eval_s155_posthoc.json") or {}); s4 = dict(r.get("s154") or {}); s7 = r.get("s147") or {}
            ps, rs, nv = propagation(s)
            a = {"seed": k + o, "test_acc": r.get("test_acc"), "serial": S154.serial(r.get("donor")), "per_column": (r.get("acc_by_column") or {}).get("per_column"),
                 "ext_4x8": (r.get("ext_4x8") or {}).get("test_acc"), "ext_4x16": (r.get("ext_4x16") or {}).get("test_acc"),
                 "propagation_share": ps, "random_edit_share": rs, "null_value": nv, "coupling_share": coupling(s), "recurrence": s.get("recurrence"),
                 "mem_norm_by_pass": s.get("mem_norm_by_pass"), "write_norm_ratio": s.get("write_norm_ratio"), "n_nonfinite": r.get("n_nonfinite"),
                 "c7_state_edit": s4.get("c7_state_edit"), "carry_probe_by_column": s4.get("carry_probe_by_column"), "acc_by_column_carry": s4.get("acc_by_column_carry"),
                 "alpha_by_head": s7.get("alpha_by_head"), "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact"] = (a["test_acc"] or 0) >= 0.99
            a["pass"] = bool(a["exact"] and a["serial"])
            a["length_general"] = (a["ext_4x8"] or 0) >= 0.9
            a["recurrent_use"] = bool(ps is not None and rs is not None and ps >= 0.8 and rs < 0.3)
            return {"attempts": att, "status": "accepted", "accepted": a}
    if len(att) == len(OFFS) and all(x["plateau"] for x in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def verdicts(arm):
    sl = {k: slot(arm, k) for k in (1, 2, 3, 4)}
    if not any(v["attempts"] or v["status"] != "open" for v in sl.values()):
        return sl, {"pass": "not run", "length": "not run", "recurrent_use": "not run"}
    acc = [v["accepted"] for v in sl.values() if v["accepted"]]
    decided = [v for v in sl.values() if v["status"] == "exhausted" or v["accepted"]]
    out = {}
    for key, lab in (("pass", ("PASS", "FAIL")), ("length_general", ("LENGTH-GENERAL", "NOT LENGTH-GENERAL")), ("recurrent_use", ("RECURRENT USE", "NO RECURRENT USE"))):
        n = sum(1 for a in acc if a[key])
        out[key] = lab[0] if n >= 3 else (lab[1] if len(decided) - n >= 2 else "incomplete (%d / 4 slots decided)" % len(decided))
    return sl, out


def main():
    res = {}
    for arm in ARMS:
        sl, v = verdicts(arm); res[arm] = {"slots": sl, "verdicts": v}
        if v["pass"] == "not run":
            continue
        print(arm, v, {k: (s["status"], s["accepted"] and (s["accepted"]["test_acc"], s["accepted"]["serial"], s["accepted"]["ext_4x8"],
                                                         s["accepted"]["propagation_share"], s["accepted"]["random_edit_share"])) for k, s in sl.items()})
    # the no-recurrence null: §154 SHIFT's propagation, scored the same way (expected = its unedited-next rate)
    nul = []
    for k in (1, 2, 3, 4):
        x = S154.slot("shift", k)["accepted"]
        if x:
            nul.append(propagation({"propagation": x.get("propagation")}))                                   # score_s154 merges eval_s154_posthoc.json
    res["null_s154_shift"] = nul; print("NULL (§154 SHIFT: propagation, random, expected):", nul)
    rd = []
    sv = res["shiftr"]["verdicts"]
    if sv["pass"] == "PASS" and sv["length_general"] == "LENGTH-GENERAL" and sv["recurrent_use"] == "RECURRENT USE":
        rd.append("SHIFT-R: a pretrained LM's memory carries computed intermediates step by step, as a local recurrent process that generalises in length "
                  "(C1 in the north-star sense, with C2 and C7 edits)")
    cps = [x["coupling_share"] for x in (v_["accepted"] for v_ in res["shiftr"]["slots"].values()) if x and x.get("coupling_share") is not None]
    if sv["recurrent_use"] == "RECURRENT USE" and cps and sorted(cps)[len(cps) // 2] >= 0.2:            # note 155-2
        rd.append("SHIFT-R propagates, but NON-SPECIFICALLY (median coupling_share %.2f >= 0.2; note 155-2)" % sorted(cps)[len(cps) // 2])
    elif sv["recurrent_use"] == "RECURRENT USE":
        rd.append("SHIFT-R RECURRENT USE: state edits propagate through recomputed carries (pass %s, length %s)" % (sv["pass"], sv["length_general"]))
    elif sv["recurrent_use"] == "NO RECURRENT USE":
        rd.append("SHIFT-R: no propagation (a failed linear edit reads 'not linearly editable', not 'not carried'; Literature (c))")
    r1 = res["r"]["slots"][1]["accepted"]; g1 = res["rg0"]["slots"][1]["accepted"]               # note 155-1 (3): R vs §147 read only through R-gamma0
    if r1 and g1:
        rd.append("R %s vs R-gamma0 %s: %s" % (r1["test_acc"], g1["test_acc"], "the write source helps" if (r1["test_acc"] or 0) > (g1["test_acc"] or 0) else
                                                "no gain from the write source"))
        rd.append("R-gamma0 %s vs §147 0.743: %s" % (g1["test_acc"], "the read lag costs" if (g1["test_acc"] or 0) < 0.743 else "no read-lag cost"))
    res["reading"] = "; ".join(rd) if rd else "pending"; print("READING:", res["reading"])
    if MISFIRES:
        print("2k-flag misfires (not stops):", MISFIRES); res["misfires"] = MISFIRES
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s155.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
