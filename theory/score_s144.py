"""THEORY §144 scoring (rules fixed in §144; written before any §144 implementation or run; Theory is the only scorer).
RUNNING-SUM diagnostic on the task-agnostic fsq120 carrier: family sum4 (state s in 0..9999, start = the 4-digit text of a uniform random s0,
items x in 0..99 as two digit tokens LSB first, s <- (s + x) mod 10^4, FINAL answer = s as 4 digit tokens LSB first); §133's harness (§128's S5
recipe, restart slots k, k+100, k+200, k+300 via 123-3's 2k flag, 24k, RTX 3090, checkpoints).
Arms (dirs boxes/runs_fsq/s144_{arm}_s{seed}): f8 = FSQ-8 (§140's carrier); n64 = NOISE-64 (--d_q 64 --no_round --dither 0.25; §142's carrier).
PASS(arm): >= 3/4 slots whose accepted run is exact at 512 (long_acc >= 0.99), SERIAL (donor >= 0.90, recipient - chance < 0.10 at every injected
position) and contractive (rho <= 0.28: 127-12's clause for f8, 133-1's first-3 form for n64). DROP at proof: slot 1 accepted with long_acc < 0.95.
FAIL: >= 2 decided slots not passing. Readings (fixed in §144): either PASS -> number carriage is not the barrier; both FAIL (high digits lost) -> it is;
f8 FAIL + n64 PASS -> the 8-d rounded code is the bottleneck. Descriptive: per-digit accuracy by length, digit / value probes from code and h, digit-wise vs
clock R2 (s144.code_form), start-decode accuracy (s144.start_acc), ext_2048. Main runs: args sum_mod 10000. The conditional magnitude ladder (Literature
d8ccceb5 (3)): an arm dropped at proof at 10^4 reruns slot 1 at sum_mod 100 / 1000 (dirs s144_{arm}_m{2,3}_s1), read descriptively (long_acc at 512).
    python theory/score_s144.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
ARMS = {"f8": {"d_q": 8, "levels": 8}, "n64": {"d_q": 64, "no_round": True, "dither": 0.25}}
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
    exp = dict(ARMS[arm], family="sum4", p_hard_start=2000, p_hard_end=4000, sum_mod=10000)
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k) != v}
    for k in ("copy_frac", "soft_beta", "hinge", "grow"):
        if a.get(k):
            bad[k] = a.get(k)
    if arm == "f8" and (a.get("no_round") or a.get("dither")):
        bad["no_round/dither"] = (a.get("no_round"), a.get("dither"))
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


def rho_clause(cf, eps="0.1", floor=1e-3):
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k"], e["dist_med_k"]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(len(rk)) if k < len(dk) and dk[k] > floor and rk[k] > 0]
    if len(use) >= 2:
        return math.exp(sum(math.log(x) for x in use) / len(use))
    return rk[0] if rk and any(x <= floor for x in dk) else None


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
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def slot(arm, k):
    att = []
    for o in OFFS:
        d = "s144_%s_s%d" % (arm, k + o)
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
            cf = load(d, "eval_contraction_fp32.json")
            a = {"seed": k + o, "long_acc": r.get("long_acc"), "ext_2048": (r.get("ext_2048") or {}).get("cumulative"), "serial": serial(r.get("donor")),
                 "rho": (rho_k3 if arm == "n64" else rho_clause)(cf), "codes_per_state": r.get("codes_per_state"),
                 "digit_acc_by_length": (r.get("s144") or {}).get("digit_acc_by_length"), "probe_r2": (r.get("s144") or {}).get("probe_r2"),
                 "code_form": (r.get("s144") or {}).get("code_form"), "start_acc": (r.get("s144") or {}).get("start_acc"),
                 "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact"] = (a["long_acc"] or 0) >= 0.99
            a["pass"] = None if a["rho"] is None else bool(a["exact"] and a["serial"] and a["rho"] <= 0.28)
            return {"attempts": att, "status": "accepted", "accepted": a}
    if len(att) == len(OFFS) and all(x["plateau"] for x in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def arm_verdict(arm):
    sl = {k: slot(arm, k) for k in (1, 2, 3, 4)}
    s1 = sl[1]["accepted"]
    if s1 is not None and (s1["long_acc"] or 0) < 0.95:
        return sl, "dropped at proof (slot 1 < 0.95 at 512)"
    if sl[1]["status"] == "exhausted":                                       # note 144-1
        return sl, "dropped at proof (slot 1 exhausted: no entry within 2k in 4 attempts)"
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


def main():
    res = {}
    for arm in ARMS:
        sl, v = arm_verdict(arm); res[arm] = {"slots": sl, "verdict": v}
        print(arm, v, {k: (s["status"], s["accepted"] and (s["accepted"]["long_acc"], s["accepted"]["serial"], s["accepted"]["rho"])) for k, s in sl.items()})
        for k, s in sl.items():
            if s["accepted"]:
                print("   slot %d digit_acc_by_length" % k, s["accepted"]["digit_acc_by_length"], "| probe", s["accepted"]["probe_r2"])
    vf, vn = res["f8"]["verdict"], res["n64"]["verdict"]
    fail = lambda x: x.startswith("FAIL") or x.startswith("dropped")  # noqa: E731
    if vf == "PASS" or vn == "PASS":
        rd = ("number carriage is not the barrier: the carrier carries and updates a 4-digit number (multiplication's failures point to in-pass "
              "products / column bookkeeping, or to the missing per-step supervision §142 adds)")
        if fail(vf) and vn == "PASS":
            rd += "; f8 fails where n64 passes: the 8-d rounded code is the bottleneck for numbers"
    elif fail(vf) and fail(vn):
        rd = "number carriage itself is the barrier (read the per-digit profile): every single-vector design inherits it; §145 is the response"
    else:
        rd = "pending"
    res["reading"] = rd; print("READING:", rd)
    lad = {}
    for arm in ARMS:                                                         # the conditional magnitude ladder (descriptive)
        for m, mod in (("m2", 100), ("m3", 1000)):
            d = "s144_%s_%s_s1" % (arm, m); r = load(d); a = load(d, "args.json") or {}
            if r is not None:
                if a.get("sum_mod") != mod or a.get("family") != "sum4":
                    ERRORS.append((d, "ladder args sum_mod %s / family %s" % (a.get("sum_mod"), a.get("family"))))
                if not res[arm]["verdict"].startswith("dropped"):
                    ERRORS.append((d, "ladder run although the arm did not drop at 10^4 (§144 order)"))
                lad["%s|%d" % (arm, mod)] = r.get("long_acc")
    if lad:
        res["ladder"] = lad; print("LADDER (long_acc at 512 by modulus):", lad)
    nf = {}
    for arm in ARMS:                                                         # note 144-1 (ii): NOFLAG, mod 10^4, no plateau stop (descriptive)
        d = "s144_%s_noflag_s1" % arm; r = load(d); a = load(d, "args.json") or {}
        if r is not None:
            if a.get("stop_plateau") or a.get("sum_mod") != 10000 or a.get("family") != "sum4":
                ERRORS.append((d, "noflag args stop_plateau %s sum_mod %s" % (a.get("stop_plateau"), a.get("sum_mod"))))
            la = r.get("long_acc")
            nf[arm] = {"long_acc": la, "ext_2048": (r.get("ext_2048") or {}).get("cumulative"), "serial": serial(r.get("donor")),
                       "reading": None if la is None else ("carries a 4-digit running sum when given time to enter" if la >= 0.99 else "does not within 24k")}
    if nf:
        res["noflag"] = nf; print("NOFLAG (note 144-1):", nf)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s144.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
