"""THEORY §137 scoring (rules fixed in §137; written before any §137 implementation or run; Theory is the only scorer).
Multiplication on the task-agnostic fsq120 carrier, COLUMN-major: op k = output column k (k = 1 .. 2n); the segment sees a's and b's
digits (least-significant first) and the column index k; the carried code holds the carry; the product digit p_k is decoded from code_k
(streamed final-answer digits: the disclosed C4 qualifier, no carry labels). FSQ-8 (d 8, 8 levels), §128's restart-on-plateau slots
(seeds k, k+100, k+200, k+300; 123-3's 2k flag), 24k steps, RTX 3090.
Dirs: boxes/runs_fsq/s137_col_{4x4,5x5}_s{seed}; eval.json: test_acc (exact match of all 2n digits on the 1,000 published bench items),
donor (as fsq120: per position donor_consistent / recipient_consistent / chance_overlap), acc_by_column, ext_4x8 (descriptive);
eval_contraction_fp32.json (rho). Args: family "mult_col", task "4x4" / "5x5", d_q 8, levels 8, p_hard 2000 / 4000, no copy_frac.
M-PASS(size): >= 3/4 slots whose accepted run has test_acc >= 0.99, SERIAL (donor >= 0.90, recipient - chance < 0.10) and rho <= 0.28
(127-12's clause, as score_s134). FAIL: >= 2 decided slots not passing. DROP at proof: slot 1 accepted with test_acc < 0.95 -> that size stops.
5x5 runs only after 4x4 PASS. Descriptive: M-SOTA (median test_acc vs ICoT-SI 0.998 / 0.996 on the same items; GEN-M 0.163 / 0.005; TC-M
1.0), the shortcut probe (carry R2 vs context-only column-sum R2), acc_by_column, 4x8 (4x4 models on generated 4x8 items).
    python theory/score_s137.py
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
    exp = {"family": "mult_col", "task": size, "d_q": 8, "levels": 8, "p_hard_start": 2000, "p_hard_end": 4000}
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
        d = "s137_col_%s_s%d" % (size, k + o)
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
            if not args_ok(d, size):
                return {"attempts": att, "status": "error", "accepted": None}
            a = {"seed": k + o, "test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "rho": rho_clause(load(d, "eval_contraction_fp32.json")),
                 "acc_by_column": r.get("acc_by_column"), "ext_4x8": r.get("ext_4x8"), "probe": r.get("s137_probe"),
                 "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact"] = (a["test_acc"] or 0) >= 0.99
            a["pass"] = None if a["rho"] is None else bool(a["exact"] and a["serial"] and a["rho"] <= 0.28)
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


def main():
    res = {}
    for size in SIZES:
        sl, v = size_verdict(size)
        accs = sorted(x["accepted"]["test_acc"] for x in sl.values() if x["accepted"] and x["accepted"]["test_acc"] is not None)
        med = accs[len(accs) // 2] if accs else None
        res[size] = {"slots": sl, "verdict": v, "median_test_acc": med, "bars": BARS[size],
                     "m_sota": None if med is None else bool(med >= BARS[size]["icot_si"])}
        print(size, v, {k: (s["status"], s["accepted"] and (s["accepted"]["test_acc"], s["accepted"]["serial"], s["accepted"]["rho"])) for k, s in sl.items()},
              "| median", med, "| ICoT-SI", BARS[size]["icot_si"], "| GEN-M", BARS[size]["gen_m"])
    if res["5x5"]["verdict"] != "not run" and res["4x4"]["verdict"] != "PASS":
        ERRORS.append("5x5 runs exist but 4x4 is not PASS (§137 order)")
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s137.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
