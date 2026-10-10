"""THEORY §184 scoring (rules fixed in §184; written before any §184 implementation or run; Theory is the only scorer).
AUX-REWEIGHT: §182 NOFADE exactly + each (item, pass t <= n) aux term weighted by w_t = Ebar / E_t (E_t = (1/8) sum_{n=t..8} 1/n, §175's expected aux weight at
pass t), so every pass position gets equal expected aux weight at an unchanged total. Comparator = s182_nofade_s1 (read from its eval.json).
Pooled = n 3..8 at K' = 8. DEEP / THINK-NECESSARY as §182. Lever (D = REWEIGHT - NOFADE): HELPS = Dpooled >= +0.05 AND D late handle (mean t 6-8) >= +0.10;
HURTS = Dpooled <= -0.05; NO EFFECT = |Dpooled| < 0.03 AND |D late handle| < 0.05; else mixed. A HELPS + DEEP opens s2.
    python theory/score_s184.py
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
BASE = dict(S182.CELLS["nofade"], aux_reweight=True)
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
    x = r.get("s170") or {}; k8 = S170.kseries(x, 8)
    p8, p0, p1 = S170.pooled(k8, NS), S170.pooled(S170.kseries(x, 0), NS), S170.pooled(S170.kseries(x, 1), NS)
    role = (r.get("s175") or {}).get("aux_acc_by_role") or {}
    handle = {int(t): (v or {}).get("handle") for t, v in role.items()}
    late = [handle.get(t) for t in (6, 7, 8)]
    clip = (r.get("s184") or {}).get("clip_rate")
    out = {"status": "done", "pooled_k8": p8, "handle": handle, "late_handle": None if None in late else sum(late) / 3, "gpu": r.get("train_gpu"), "clip_rate": clip}
    if p8 is None or p0 is None or p1 is None:
        return dict(out, status="incomplete")
    rem = (p0 - CH) <= 0.5 * (p8 - CH) + EPS and (p1 - CH) <= 0.5 * (p8 - CH) + EPS
    out.update({"deep": all((S170.acc(k8, n) or 0) >= 0.90 - EPS for n in range(NS, 9)), "think_necessary": None if p8 - CH < 0.1 else bool(rem)})
    return out


def main():
    comp = read(COMP, S182.CELLS["nofade"]); runs = {s: read("s184_reweight_s%d" % s, dict(BASE, seed=s)) for s in (1, 2)}
    res = {"comparator": comp, "runs": runs}; r1 = runs[1]
    if r1.get("status") != "done" or comp.get("status") != "done" or r1.get("late_handle") is None or comp.get("late_handle") is None:
        rd = "pending (REWEIGHT s1 %s; NOFADE %s)" % (r1.get("status"), comp.get("status"))
    else:
        dp = r1["pooled_k8"] - comp["pooled_k8"]; dl = r1["late_handle"] - comp["late_handle"]
        if dp >= 0.05 - EPS and dl >= 0.10 - EPS:
            lever = "HELPS"
        elif dp <= -0.05 + EPS:
            cr, cc = r1.get("clip_rate"), comp.get("clip_rate")
            lever = "HURTS (clip-confounded)" if cr is not None and ((cc is not None and cr >= 2 * cc) or (cc is None and cr >= 0.5)) else "HURTS"
        elif abs(dp) < 0.03 and abs(dl) < 0.05:
            lever = "NO EFFECT (supervision count is not what limits late passes)"
        else:
            lever = "mixed"
        res.update({"d_pooled": dp, "d_late_handle": dl, "lever": lever})
        rd = "%s: Dpooled %+.3f (%.3f vs %.3f), D late handle %+.3f; REWEIGHT DEEP %s, THINK-NECESSARY %s" % (
            lever, dp, r1["pooled_k8"], comp["pooled_k8"], dl, r1["deep"], r1["think_necessary"])
        if lever == "HELPS" and r1["deep"]:
            rd += "; s2 opens" if runs[2].get("status") != "done" else "; s2 DEEP %s" % runs[2].get("deep")
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s184.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
