"""THEORY §185 scoring (rules fixed in §185; written before any §185 implementation or run; Theory is the only scorer).
ADD-NORM: §182 NOFADE exactly + the additive read normalised: DEN = by phi(q)^T z (z = the running key sum; Katharopoulos 2006.16236); RMS = per-head
RMSNorm of the read (TransNormer 2210.10340's NormAttention). S_add's writes unchanged. Rules per arm.
Comparator = s182_nofade_s1. Pooled = n 3..8. STABLE = pooled K'16 >= pooled K'8 - 0.02; NO-COST = pooled K'8 >= NOFADE's K'8 - 0.03.
FIX = STABLE and NO-COST while NOFADE is not STABLE; COUNTER-LOSS = STABLE but K'8 < NOFADE - 0.03; NO-FIX = not STABLE; NOT ATTRIBUTABLE = NOFADE itself STABLE
(then read NO-COST only). A FIX opens s2.
    python theory/score_s185.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s170 as S170  # noqa: E402
import score_s182 as S182  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
NS, CH, EPS = 3, 1.0 / 11, 1e-9
ARMS = {"den": dict(S182.CELLS["nofade"], add_norm="den"), "rms": dict(S182.CELLS["nofade"], add_norm="rms")}
COMP = "s182_nofade_s1"
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def read(d, want):
    a = load(d, "args.json") or {}; r = load(d)
    if r is None:
        return {"status": "running" if a else "open"}
    bad = {k: a.get(k) for k, v in want.items() if str(a.get(k)) != str(v)}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    x = r.get("s170") or {}
    p8, p16, p0, p1 = (S170.pooled(S170.kseries(x, k), NS) for k in (8, 16, 0, 1))
    if None in (p8, p16, p0, p1):
        return {"status": "incomplete", "missing": [k for k, v in zip((8, 16, 0, 1), (p8, p16, p0, p1)) if v is None]}
    rem = (p0 - CH) <= 0.5 * (p8 - CH) + EPS and (p1 - CH) <= 0.5 * (p8 - CH) + EPS
    return {"status": "done", "pooled_k8": p8, "pooled_k16": p16, "pooled_k32": S170.pooled(S170.kseries(x, 32), NS), "stable": bool(p16 >= p8 - 0.02 - EPS),
            "deep": all((S170.acc(S170.kseries(x, 8), n) or 0) >= 0.90 - EPS for n in range(NS, 9)), "think_necessary": None if p8 - CH < 0.1 else bool(rem),
            "gpu": r.get("train_gpu")}


def verdict(arm, comp):
    runs = {s: read("s185_%s_s%d" % (arm, s), dict(ARMS[arm], seed=s)) for s in (1, 2)}; r1 = runs[1]
    if r1.get("status") != "done" or comp.get("status") != "done":
        rd = "pending (%s s1 %s; NOFADE %s)" % (arm.upper(), r1.get("status"), comp.get("status"))
    else:
        nocost = r1["pooled_k8"] >= comp["pooled_k8"] - 0.03 - EPS
        if comp["stable"]:
            rd = "NOT ATTRIBUTABLE (NOFADE itself STABLE: %.3f -> %.3f); NO-COST %s" % (comp["pooled_k8"], comp["pooled_k16"], nocost)
        elif r1["stable"] and nocost:
            rd = "FIX"
        elif r1["stable"]:
            rd = "COUNTER-LOSS"
        else:
            rd = "NO-FIX"
        rd += ": %s K'8 %.3f -> K'16 %.3f (K'32 %s) vs NOFADE %.3f -> %.3f; DEEP %s, THINK-NECESSARY %s" % (
            arm.upper(), r1["pooled_k8"], r1["pooled_k16"], None if r1["pooled_k32"] is None else round(r1["pooled_k32"], 3), comp["pooled_k8"], comp["pooled_k16"], r1["deep"], r1["think_necessary"])
        if rd.startswith("FIX"):
            rd += "; s2 opens" if runs[2].get("status") != "done" else "; s2 STABLE %s" % runs[2].get("stable")
    return {"runs": runs, "reading": rd}


def main():
    comp = read(COMP, S182.CELLS["nofade"]); res = {"comparator": comp}
    for arm in ARMS:
        res[arm] = verdict(arm, comp); print("READING %s:" % arm.upper(), res[arm]["reading"])
    d1, r1 = res["den"]["runs"][1], res["rms"]["runs"][1]
    if d1.get("status") == "done" and r1.get("status") == "done" and comp.get("status") == "done" and not comp.get("stable"):
        ds, rs = d1["stable"], r1["stable"]
        res["pair"] = ("both STABLE: either works" if ds and rs else "DEN STABLE, RMS not: a mean-read fixes it" if ds else
                       "RMS STABLE, DEN not (check DEN's grad_add p99 vs RMS's for TransNormer's warning)" if rs else "neither STABLE: C6 rests on HOLD / NORM-CAP")
        print("PAIR:", res["pair"])
    if ERRORS:
        print("ERRORS:", ERRORS)
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s185.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
