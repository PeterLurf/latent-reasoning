"""THEORY §187 scoring (rules fixed in §187; written before any §187 implementation or run; Theory is the only scorer).
AUX-NOPASS: §184 exactly with the aux prompt [EOT, " Step", " :"] (no pass token). Comparator = s184_reweight_s1.
DEEP = >= 0.90 at every n 3..8 at K' 8; THINK-NECESSARY = removal (K' 0 / 1). DEPTH = mean n 9-12 at per-item K' = n + 2 (s170.acc_by_n_kprime["n+2"]).
TOKEN-WAS-THE-LIMIT = DEEP and DEPTH >= comparator + 0.10 and >= 0.535 (TEXT-COT's two-seed mean); NOT-THE-LIMIT = DEEP and |DEPTH - comparator| < 0.05;
COSTLY = not DEEP; else mixed. TOKEN-WAS-THE-LIMIT opens s2.
    python theory/score_s187.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s170 as S170  # noqa: E402
import score_s182 as S182  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
NS, CH, EPS, TEXTCOT_9_12 = 3, 1.0 / 11, 1e-9, 0.535
COMP_ARGS = dict(S182.CELLS["nofade"], aux_reweight=True)
BASE = dict(COMP_ARGS, aux_nopass=True)
COMP = "s184_reweight_s1"
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
    x = r.get("s170") or {}; k8, kn2 = S170.kseries(x, 8), S170.kseries(x, "n+2")
    p8, p0, p1 = S170.pooled(k8, NS), S170.pooled(S170.kseries(x, 0), NS), S170.pooled(S170.kseries(x, 1), NS)
    depth = S170.mean([S170.acc(kn2, n) for n in range(9, 13)]) if kn2 else None
    if None in (p8, p0, p1) or depth is None:
        return {"status": "incomplete", "missing": {"K'8": p8 is None, "K'0/1": p0 is None or p1 is None, "n+2": depth is None}}
    rem = (p0 - CH) <= 0.5 * (p8 - CH) + EPS and (p1 - CH) <= 0.5 * (p8 - CH) + EPS
    return {"status": "done", "pooled_k8": p8, "depth_9_12_n2": depth, "by_n_n2": {n: S170.acc(kn2, n) for n in range(3, 13)},
            "deep": all((S170.acc(k8, n) or 0) >= 0.90 - EPS for n in range(NS, 9)), "think_necessary": None if p8 - CH < 0.1 else bool(rem), "gpu": r.get("train_gpu")}


def main():
    comp = read(COMP, COMP_ARGS); runs = {s: read("s187_nopass_s%d" % s, dict(BASE, seed=s)) for s in (1, 2)}
    res = {"comparator": comp, "runs": runs}; r1 = runs[1]
    if r1.get("status") != "done" or comp.get("status") != "done":
        rd = "pending (NOPASS s1 %s; §184 %s)" % (r1.get("status"), comp.get("status"))
    else:
        dd = r1["depth_9_12_n2"] - comp["depth_9_12_n2"]
        if not r1["deep"]:
            rd = "COSTLY"
        elif dd >= 0.10 - EPS and r1["depth_9_12_n2"] >= TEXTCOT_9_12 - EPS:
            rd = "TOKEN-WAS-THE-LIMIT"
        elif abs(dd) < 0.05:
            rd = "NOT-THE-LIMIT"
        else:
            rd = "mixed"
        rd += ": DEPTH (n 9-12 at K'=n+2) %.3f vs §184 %.3f (%+.3f; TEXT-COT %.3f); K'8 pooled %.3f; DEEP %s, THINK-NECESSARY %s" % (
            r1["depth_9_12_n2"], comp["depth_9_12_n2"], dd, TEXTCOT_9_12, r1["pooled_k8"], r1["deep"], r1["think_necessary"])
        if rd.startswith("TOKEN-WAS"):
            rd += "; s2 opens" if runs[2].get("status") != "done" else "; s2 depth %s" % runs[2].get("depth_9_12_n2")
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s187.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
