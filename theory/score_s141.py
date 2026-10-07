"""THEORY §141 scoring (rules fixed in §141; written before any §141 implementation or run; Theory is the only scorer).
§137's column-major 4x4 (fsq120 --family mult_col --task 4x4) with LATENT REPEATS and / or a WIDER carrier, after §137 4x4 DROP (test_acc 0.0).
Cells (dirs boxes/runs_fsq/s141_{cell}_4x4_s{seed}; restart slots k, k+100, k+200, k+300 as §137):
  r4f8    r = 4 segments per column, each [EOT, code, a, b, column k, repeat index], chained through the code; digit decoded after the 4th; FSQ-8
  r4s16   the same with SOFT-16 (d 16, soft_beta 0.1)
  r1s16   §137 unchanged but SOFT-16; runs ONLY if 141's nonlinear code probe on s137 shows the carry missing from the code (gate file below)
M-PASS(cell): >= 3/4 slots whose accepted run has test_acc >= 0.99, SERIAL (inject a donor code after column 2 / 4 / 6, at the column's last repeat)
and contractive (rho <= 0.28: 127-12's clause for FSQ-8, 133-1's first-3 form for SOFT-16). FAIL: >= 2 decided slots not passing. DROP at proof:
slot 1 < 0.95 stops the cell. Descriptive: acc_by_column, the per-repeat column-sum probe and code change per repeat (s141_probe), 4x8.
    python theory/score_s141.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
CELLS = {"r4f8": {"repeats": 4, "repeat_index": True, "d_q": 8}, "r4s16": {"repeats": 4, "repeat_index": True, "d_q": 16, "soft_beta": 0.1},
         "r1s16": {"repeats": 1, "d_q": 16, "soft_beta": 0.1}}
SIZES = tuple(CELLS)
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
    exp = dict({"family": "mult_col", "task": "4x4", "levels": 8, "p_hard_start": 2000, "p_hard_end": 4000}, **CELLS[size])
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


def slot(size, k):
    att = []
    for o in OFFS:
        d = "s141_%s_4x4_s%d" % (size, k + o)
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
            cf = load(d, "eval_contraction_fp32.json")
            a = {"seed": k + o, "test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "rho": (rho_k3(cf) if CELLS[size].get("soft_beta") else rho_clause(cf)),
                 "acc_by_column": r.get("acc_by_column"), "ext_4x8": r.get("ext_4x8"), "probe": r.get("s141_probe"),
                 "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact"] = (a["test_acc"] or 0) >= 0.99
            a["pass"] = None if a["rho"] is None else bool(a["exact"] and a["serial"] and a["rho"] <= 0.28)
            a["reading"] = "IN-PASS (ICoT-like), not a latent carry" if (a["exact"] and not a["serial"]) else None   # 137-1 (2)
            r2 = (r.get("s141_probe") or {})
            a["carry_r2"] = {k: r2.get(k) for k in ("r2_carry_code", "r2_carry_h", "r2_colsum_context")}       # 137-1 (2): code_k and pre-W_down h_k
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


def main():
    res = {}
    for size in SIZES:
        sl, v = size_verdict(size)
        accs = sorted(x["accepted"]["test_acc"] for x in sl.values() if x["accepted"] and x["accepted"]["test_acc"] is not None)
        med = accs[len(accs) // 2] if accs else None
        res[size] = {"slots": sl, "verdict": v, "median_test_acc": med}
        print(size, v, {k: (s["status"], s["accepted"] and (s["accepted"]["test_acc"], s["accepted"]["serial"], s["accepted"]["rho"])) for k, s in sl.items()},
              "| median", med)
    g = os.path.join(R, "s137_col_4x4_s1", "eval_probe_percol.json")              # 141: r1s16 runs only if the nonlinear code probe shows the carry missing
    if os.path.exists(g):
        gp = json.load(open(g)); res["r1s16_gate"] = gp.get("carry_code_nonlinear_acc"); print("r1s16 gate (nonlinear code->carry acc < 0.9 opens it):", res["r1s16_gate"])
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s141.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
