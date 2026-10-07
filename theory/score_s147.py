"""THEORY §147 scoring (rules fixed in §147; written before any §147 implementation or run; Theory is the only scorer).
DELTA-STATE (the research program's B1) on §140's digit-pair 4x4 harness: segments [EOT, a_i, b_j, column flag] (+ the flush), NO carried code token;
a memory branch after block l_m (6 additive heads S += phi(k) v^T, no decay / erase; 6 gated-delta heads, beta in (0, 2)) whose state is the ONLY
cross-segment channel; p_k emitted by the LM head at the column's last pair; restart slots k, k+100, k+200, k+300 (123-3's 2k flag); 24k; RTX 3090.
Dirs boxes/runs_fsq/s147_delta_4x4_s{seed}.
PASS: >= 3/4 slots whose accepted run has test_acc >= 0.99 and SERIAL (the donor's whole memory state swapped in after pair 4 / 8 / 12; later digits
follow the recipient's pairs with the donor's accumulator; row level). Contraction is NOT part of PASS (an exact integrator has eigenvalue 1 on its
additive subspace by design; stated in §147); ext_4x8 is the stability read. DROP at proof: slot 1 accepted with test_acc < 0.95. FAIL: >= 2 decided
slots not passing. IMPROVES (descriptive): columns 3-7 all >= 0.9 on the proof. Note 147-1: a 2k flag with loss(2k) < 1.0 is a misfire (logged,
not a restart); s147.colsum_readout_r2_by_column (the column sum read from the additive state, per column index) locates interference.
    python theory/score_s147.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
ARMS = {"delta": None}
S140_PROOF = [1.0, 1.0, 0.315, 0.132, 0.104, 0.24, 0.50, 0.936]              # §140 result: s140_pair_4x4_s1 per column (comparator)
SKIPPED, ERRORS, MISFIRES = [], [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def smoke(d):
    a = load(d, "args.json")
    if not a:
        return False
    bad = (int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu")
           or bool(a.get("mem_check")))
    if bad:
        SKIPPED.append(d)
    return bad


def args_ok(d, arm):
    a = load(d, "args.json") or {}
    exp = {"family": "mult_pair", "task": "4x4", "carrier": "deltamem", "mem_add_heads": 6, "mem_delta_heads": 6, "mem_group_gain": True}   # note 147-2
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k) != v}
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
    if fl and g(2000) < 1.0:                                                 # note 147-1 (137-1 (3)): a misfire on near-0 digit CE: logged, not a stop
        MISFIRES.append(d); return False
    return fl


def rho_k3(cf, eps="0.1"):
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k"], e["dist_med_k"]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(min(3, len(rk), len(dk))) if dk[k] > 1e-2 and rk[k] > 0]
    if not use:
        return rk[0] if rk else None
    return math.exp(sum(math.log(x) for x in use) / len(use))


def serial(dn):
    """Row level where present (multpair writes digit-level + row_level per position): donor >= 0.90, recipient - chance < 0.10."""
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
        d = "s147_%s_4x4_s%d" % (arm, k + o)
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
            s7 = r.get("s147") or {}
            a = {"seed": k + o, "test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "per_column": (r.get("acc_by_column") or {}).get("per_column"),
                 "ext_4x8": r.get("ext_4x8"), "acc_by_magnitude": s7.get("acc_by_magnitude"), "head_ablation": s7.get("head_ablation"),
                 "colsum_readout_r2": s7.get("colsum_readout_r2"), "colsum_readout_r2_by_column": s7.get("colsum_readout_r2_by_column"),
                 "read_norm_ratio_by_handoff": s7.get("read_norm_ratio_by_handoff"),   # note 147-2 (Audit): {"step0": [...], "end": [...]} "rho_by_head_type": s7.get("rho_by_head_type"),
                 "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact"] = (a["test_acc"] or 0) >= 0.99
            a["pass"] = bool(a["exact"] and a["serial"])                   # §147: no contraction term (integrator by design)
            a["reading"] = "IN-PASS / shortcut, not a latent carry (exact without SERIAL)" if (a["exact"] and not a["serial"]) else None
            return {"attempts": att, "status": "accepted", "accepted": a}
    if len(att) == len(OFFS) and all(x["plateau"] for x in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def arm_verdict(arm):
    sl = {k: slot(arm, k) for k in (1, 2, 3, 4)}
    s1 = sl[1]["accepted"]
    if s1 is not None and (s1["test_acc"] or 0) < 0.95:
        return sl, "dropped at proof (slot 1 test_acc < 0.95)"
    if not any(v["attempts"] or v["status"] != "open" for v in sl.values()):
        return sl, "not run"
    passes = [v["accepted"]["pass"] for v in sl.values() if v["accepted"]]
    decided = [v for v in sl.values() if v["status"] == "exhausted" or (v["accepted"] and v["accepted"]["pass"] is not None)]
    n = sum(1 for p in passes if p)
    if n >= 3:
        return sl, "PASS"
    if len(decided) - n >= 2:
        return sl, "FAIL"
    return sl, "incomplete (%d / 4 slots decided)" % len(decided)


def improves(a):
    pc = (a or {}).get("per_column")
    return None if not pc or len(pc) < 7 else all(x >= 0.9 for x in pc[2:7])


def main():
    res = {}
    sl, v = arm_verdict("delta"); res["delta"] = {"slots": sl, "verdict": v}
    print("delta", v, {k: (s["status"], s["accepted"] and (s["accepted"]["test_acc"], s["accepted"]["serial"])) for k, s in sl.items()})
    s1 = sl[1]["accepted"]
    if s1:
        print("   proof per column", s1["per_column"], "| §140 1 x FSQ-8", S140_PROOF, "| IMPROVES (cols 3-7 >= 0.9):", improves(s1))
        print("   ext_4x8", s1["ext_4x8"], "| by magnitude", s1["acc_by_magnitude"], "| head ablation", s1["head_ablation"], "| colsum readout R2", s1["colsum_readout_r2"])
        print("   read-norm ratio delta / additive by hand-off (step 0 / end):", s1["read_norm_ratio_by_handoff"])
    i4 = improves(s1); res["improves"] = i4
    if v == "PASS":
        rd = "a linear-recurrent memory as the sole cross-step channel makes a pretrained LM multiply exactly (C1 by design + SERIAL; C4 as reworded)"
    elif v.startswith("dropped") or v.startswith("FAIL"):
        rd = ("the additive substrate fixes the accumulator but the readout's carry resolution falls short at this budget" if i4
              else "a linear memory does not fix in-pass digit products / readout at this budget")
    else:
        rd = "pending"
    res["reading"] = rd; print("READING:", rd)
    if MISFIRES:
        print("2k-flag misfires (loss(2k) < 1.0; not stops, note 147-1):", MISFIRES); res["misfires"] = MISFIRES
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s147.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
