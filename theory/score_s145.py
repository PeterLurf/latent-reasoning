"""THEORY §145 scoring (rules fixed in §145; written before any §145 implementation or run; Theory is the only scorer).
MULTI-SLOT carrier on §140's digit-pair 4x4 multiplication (multpair: one segment per pair in output-column order plus a flush; column flag;
streamed product digits; column curriculum; restart slots k, k+100, k+200, k+300 via 123-3's 2k flag; 24k; RTX 3090). The segment ends with K write
positions; each one's last hidden state -> its own W_down_k -> NOISE-64 (continuous, dither U(+-0.25) after 2k) -> its own W_up_k = slot_k of the
next segment; the decoder reads [EOT, slot_1..slot_K].
Arms (dirs boxes/runs_fsq/s145_{arm}_4x4_s{seed}): slot4 (K = 4 write positions, 4 carried slots); carry1 (the control, Literature d8ccceb5 (1): the same
4 write positions and compute, only the slot read at the LAST write position carried (note 145-1: --carry_from last); runs ONLY if slot4's
proof IMPROVES). SERIAL is read at the row level (all later digits).
M-PASS(arm): >= 3/4 slots whose accepted run has test_acc >= 0.99 (the 1,000 bench items), SERIAL (all K slots of a donor injected after pair 4 / 8 / 12;
later digits follow the recipient's pairs with the donor's accumulator, exact) and contractive (133-1's first-3 form on the concatenated K x 64 code,
rho <= 0.28). DROP at proof: slot 1 accepted with test_acc < 0.95. FAIL: >= 2 decided slots not passing.
IMPROVES (descriptive; the user's falsifiable claim): on slot4's proof, per-column accuracy at columns 3-7 all >= 0.9 (§140's 1 x FSQ-8: 0.10-0.50).
    python theory/score_s145.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
ARMS = {"slot4": (4, 4), "carry1": (4, 1)}                                  # (K write positions / slots, carried slots)
S140_PROOF = [1.0, 1.0, 0.315, 0.132, 0.104, 0.24, 0.50, 0.936]              # §140 result: s140_pair_4x4_s1 per column (comparator)
SKIPPED, ERRORS = [], []


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
    exp = {"family": "mult_pair", "task": "4x4", "slots": ARMS[arm][0], "carry_slots": ARMS[arm][1], "d_q": 64, "no_round": True, "dither": 0.25}
    if arm == "carry1":
        exp["carry_from"] = "last"                                          # note 145-1: the slot read at w_4 (causal attention)
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
    return bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))


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
        d = "s145_%s_4x4_s%d" % (arm, k + o)
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
            a = {"seed": k + o, "test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "rho": rho_k3(load(d, "eval_contraction_fp32.json")),
                 "per_column": (r.get("acc_by_column") or {}).get("per_column"), "ext_4x8": r.get("ext_4x8"), "probe": r.get("s145_probe"),
                 "acc_by_magnitude": (r.get("s145") or {}).get("acc_by_magnitude"), "slot_cosine": (r.get("s145") or {}).get("slot_cosine"),
                 "slot_swap": (r.get("s145") or {}).get("slot_swap"), "slot_ablation": (r.get("s145") or {}).get("slot_ablation"),
                 "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact"] = (a["test_acc"] or 0) >= 0.99
            a["pass"] = None if a["rho"] is None else bool(a["exact"] and a["serial"] and a["rho"] <= 0.28)
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
    for arm in ARMS:
        sl, v = arm_verdict(arm); res[arm] = {"slots": sl, "verdict": v}
        print(arm, v, {k: (s["status"], s["accepted"] and (s["accepted"]["test_acc"], s["accepted"]["serial"], s["accepted"]["rho"])) for k, s in sl.items()})
        s1 = sl[1]["accepted"]
        if s1:
            print("   proof per column", s1["per_column"], "| §140 1 x FSQ-8", S140_PROOF, "| IMPROVES (cols 3-7 >= 0.9):", improves(s1))
            print("   probe", s1["probe"], "| by magnitude", s1["acc_by_magnitude"], "| slot cosine", s1["slot_cosine"])
    i4 = improves(res["slot4"]["slots"][1]["accepted"]); i1 = improves(res["carry1"]["slots"][1]["accepted"])
    res["improves"] = {"slot4": i4, "carry1": i1}
    if res["carry1"]["verdict"] != "not run" and i4 is not True:
        ERRORS.append("carry1 ran although slot4's proof does not IMPROVE (§145 order)")
    v4 = res["slot4"]["verdict"]
    if v4 == "PASS":
        rd = "a pretrained LM multiplies exactly through a multi-slot latent carrier (design-forced C1; streamed digits, C4 as reworded)"
    elif v4.startswith("dropped") or v4.startswith("FAIL"):
        rd = "spreading the carrier over K positions does not fix the accumulator at this budget"
    else:
        rd = "pending"
    if i4 is True and i1 is False:
        rd += "; IMPROVES with 4 carried slots but not with 1 (same writes): the multi-vector state represents the larger accumulators"
    elif i4 is True and i1 is True:
        rd += "; both IMPROVE: NOISE-64 plus the extra write compute was enough, the carried slots are not shown to matter"
    res["reading"] = rd; print("READING:", rd)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s145.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
