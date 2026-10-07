"""THEORY §140 scoring (rules fixed in §140; written before any §140 implementation or run, and before §137 4x4 reads; Theory is the only scorer).
DIGPAIR fallback, run ONLY if §137's 4x4 reads DROP at proof or FAIL. Multiplication on the task-agnostic fsq120 carrier with digit-pair ops:
one segment per pair (i, j) in output-column order (i + j - 1 = k, ascending i), then one flush segment; the segment sees ONLY a_i, b_j and
the tokens (i, j) (no other digits, so the carried code must hold the column accumulator: SERIAL is design-forced here, unlike §137);
the product digit p_k is decoded from the code at column k's last pair (streamed final-answer digits, the C4 qualifier; no accumulator
labels). FSQ-8 (d 8, 8 levels), §128's restart-on-plateau slots (seeds k, k+100, k+200, k+300; 123-3's 2k flag), 24k steps, RTX 3090.
Dirs: boxes/runs_fsq/s140_pair_{4x4,5x5}_s{seed}; eval.json: test_acc (exact match of all 2n digits on the 1,000 published bench items),
donor (inject a donor code after pair 4 / 8 / 12: later digits follow the recipient's pairs with the donor's accumulator), acc_by_column,
ext_4x8 (descriptive); eval_contraction_fp32.json (rho). Args: family "mult_pair", task "4x4" / "5x5", d_q 8, levels 8, p_hard 2000 / 4000.
M-PASS(size): >= 3/4 slots whose accepted run has test_acc >= 0.99, SERIAL and rho <= 0.28 (127-12's clause). FAIL: >= 2 decided slots not
passing. DROP at proof: slot 1 accepted with test_acc < 0.95 stops that size. 5x5 only after 4x4 PASS. Descriptive: M-SOTA vs ICoT-SI
0.998 / 0.996, GEN-M 0.163 / 0.005, TC-M 1.0; accumulator probe; acc_by_column; 4x8 (32 pairs; j indices above 4 unseen).
    python theory/score_s140.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
SIZES = ("4x4", "5x5")
BARS = {"4x4": {"icot_si": 0.998, "gen_m": 0.163, "tc_m": 1.0}, "5x5": {"icot_si": 0.996, "gen_m": 0.005, "tc_m": 1.0}}
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


def args_ok(d, size):
    a = load(d, "args.json") or {}
    exp = {"family": "mult_pair", "task": size, "d_q": 8, "levels": 8, "p_hard_start": 2000, "p_hard_end": 4000}
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k, v if k in ("d_q", "levels") else None) != v}
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


def misfire(d):
    """137-1 (3): the 123-3 flag 'misfires' on digit CE if it fires while loss(2k) is already below 1.0 nat per digit."""
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    L = {x["step"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x and "step" in x}
    if not L or max(L) < 2000:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    fl = g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100)
    return bool(fl and g(2000) < 1.0)


def rho_clause(cf, eps="0.1", floor=1e-3):
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k"], e["dist_med_k"]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(len(rk)) if k < len(dk) and dk[k] > floor and rk[k] > 0]
    if len(use) >= 2:
        return math.exp(sum(math.log(x) for x in use) / len(use))
    return rk[0] if rk and any(x <= floor for x in dk) else None


def serial(dn):
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def slot(size, k):
    att = []
    for o in OFFS:
        d = "s140_pair_%s_s%d" % (size, k + o)
        if smoke(d):
            break
        pl = plateau(d); r = load(d)
        if pl is None and r is None:
            break
        if pl is None:
            return {"attempts": att, "status": "running (phase 1)", "accepted": None}
        att.append({"seed": k + o, "plateau": pl, "misfire": misfire(d) if pl else None})
        if pl is False:
            if r is None:
                return {"attempts": att, "status": "running", "accepted": None}
            if not args_ok(d, size):
                return {"attempts": att, "status": "error", "accepted": None}
            a = {"seed": k + o, "test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "rho": rho_clause(load(d, "eval_contraction_fp32.json")),
                 "acc_by_column": r.get("acc_by_column"), "ext_4x8": r.get("ext_4x8"), "probe": r.get("s140_probe"),
                 "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact"] = (a["test_acc"] or 0) >= 0.99
            a["pass"] = None if a["rho"] is None else bool(a["exact"] and a["serial"] and a["rho"] <= 0.28)
            a["reading"] = "IN-PASS (ICoT-like), not a latent carry" if (a["exact"] and not a["serial"]) else None   # 137-1 (2)
            r2 = (r.get("s140_probe") or {})
            a["carry_r2"] = {k: r2.get(k) for k in ("r2_acc_code", "r2_acc_h", "r2_colsum_context")}       # 137-1 (2): code_k and pre-W_down h_k
            a["flag2k_misfire"] = misfire(d)                                                                     # 137-1 (3)
            return {"attempts": att, "status": "accepted", "accepted": a}
    if len(att) == len(OFFS) and all(x["plateau"] for x in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def size_verdict(size):
    sl = {k: slot(size, k) for k in (1, 2, 3, 4)}
    s1 = sl[1]["accepted"]
    if s1 is not None and (s1["test_acc"] or 0) < 0.95:
        return sl, "dropped at proof (slot 1 test_acc < 0.95)"
    if not any(v["attempts"] or v["status"] != "open" for v in sl.values()):
        return sl, "not run"
    passes = [v["accepted"]["pass"] for v in sl.values() if v["accepted"]]
    decided = [v for v in sl.values() if v["status"] == "exhausted" or (v["accepted"] and v["accepted"]["pass"] is not None)]
    n_pass = sum(1 for p in passes if p)
    if n_pass >= 3:
        return sl, "PASS"
    if len(decided) - n_pass >= 2:
        return sl, "FAIL"
    return sl, "incomplete (%d / 4 slots decided)" % len(decided)


def s137_4x4_dropped_or_failed():
    """§140 runs only if §137's 4x4 reads DROP at proof or FAIL."""
    p = os.path.join(ROOT, "theory", "results", "score_s137.json")
    if not os.path.exists(p):
        return None
    v = (json.load(open(p)).get("4x4") or {}).get("verdict", "")
    return v.startswith("dropped") or v.startswith("FAIL")


def main():
    res = {}
    gate = s137_4x4_dropped_or_failed(); print("§137 4x4 DROP / FAIL (§140's precondition):", gate); res["s137_gate"] = gate
    for size in SIZES:
        sl, v = size_verdict(size)
        accs = sorted(x["accepted"]["test_acc"] for x in sl.values() if x["accepted"] and x["accepted"]["test_acc"] is not None)
        med = accs[len(accs) // 2] if accs else None
        res[size] = {"slots": sl, "verdict": v, "median_test_acc": med, "bars": BARS[size],
                     "m_sota": None if med is None else bool(med >= BARS[size]["icot_si"])}
        print(size, v, {k: (s["status"], s["accepted"] and (s["accepted"]["test_acc"], s["accepted"]["serial"], s["accepted"]["rho"])) for k, s in sl.items()},
              "| median", med, "| ICoT-SI", BARS[size]["icot_si"], "| GEN-M", BARS[size]["gen_m"])
    if res["5x5"]["verdict"] != "not run" and res["4x4"]["verdict"] != "PASS":
        ERRORS.append("5x5 runs exist but 4x4 is not PASS (§140 order)")
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s140.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
