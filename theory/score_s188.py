"""THEORY §188 scoring (rules fixed in §188; written before any §188 implementation or run; Theory is the only scorer).
REWEIGHT x WINDOW: s182_window_s1's args + --aux_reweight. Comparators s184_reweight_s1 (DEEP) and s182_window_s1 (STABLE), from their eval.json.
DEEP = >= 0.90 at every n 3..8 at K' 8; THINK-NECESSARY = 170-4 removal (K' 0 / 1); STABLE = pooled (n 3..8) K'16 >= K'8 - 0.02 (§182's letter).
JOINT = DEEP and THINK-NECESSARY and STABLE (opens s2; the claim needs 2 / 2); DEEP-ONLY = DEEP, not STABLE; STABLE-ONLY = STABLE, not DEEP; else NEITHER.
Note 188-1: DWELL = acc K'=n+2 - K'=n per n 3..8 (descriptive; > 0.05 = needs slack passes). LONG-STABLE = pooled K'24 >= K'8 - 0.05; K'32 reported. C6 CLAIM (Audit) = JOINT and LONG-STABLE in both seeds. Descriptive: DEPTH (n 9-12 at K' = n + 2), clip rate.
    python theory/score_s188.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s170 as S170  # noqa: E402
import score_s182 as S182  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
NS, CH, EPS, LONG_TOL = 3, 1.0 / 11, 1e-9, 0.05
COMPS = {"s184": ("s184_reweight_s1", dict(S182.CELLS["nofade"], aux_reweight=True)), "window": ("s182_window_s1", S182.CELLS["window"])}
BASE = dict(S182.CELLS["window"], aux_reweight=True)
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
    p = {k: S170.pooled(S170.kseries(x, k), NS) for k in (0, 1, 8, 16, 24, 32)}
    kn2 = S170.kseries(x, "n+2"); depth = S170.mean([S170.acc(kn2, n) for n in range(9, 13)]) if kn2 else None
    if None in (p[0], p[1], p[8], p[16], p[24]):
        return {"status": "incomplete", "missing": [k for k, v in p.items() if v is None and k != 32]}
    rem = (p[0] - CH) <= 0.5 * (p[8] - CH) + EPS and (p[1] - CH) <= 0.5 * (p[8] - CH) + EPS
    return {"status": "done", "pooled": p, "min_k8_3_8": min(S170.acc(k8, n) or 0 for n in range(NS, 9)), "deep": all((S170.acc(k8, n) or 0) >= 0.90 - EPS for n in range(NS, 9)),
            "think_necessary": None if p[8] - CH < 0.1 else bool(rem), "stable": bool(p[16] >= p[8] - 0.02 - EPS), "long_stable": bool(p[24] >= p[8] - LONG_TOL - EPS),
            "depth_9_12_n2": depth, "clip_rate": (r.get("s184") or {}).get("clip_rate"), "gpu": r.get("train_gpu"),
            "dwell_by_n": {n: (None if None in (S170.acc(kn2, n), S170.acc(S170.kseries(x, "n"), n)) else S170.acc(kn2, n) - S170.acc(S170.kseries(x, "n"), n)) for n in range(NS, 9)} if kn2 else None}   # note 188-1


def label(r):
    if r["deep"] and r["think_necessary"] and r["stable"]:
        return "JOINT"
    if r["deep"] and not r["stable"]:
        return "DEEP-ONLY"
    if r["stable"] and not r["deep"]:
        return "STABLE-ONLY"
    return "NEITHER"


def main():
    comps = {k: read(d, w) for k, (d, w) in COMPS.items()}; runs = {s: read("s188_rwwin_s%d" % s, dict(BASE, seed=s)) for s in (1, 2)}
    res = {"comparators": comps, "runs": runs}; r1 = runs[1]
    if r1.get("status") != "done":
        rd = "pending (s188 s1 %s)" % r1.get("status")
    else:
        rd = label(r1); p = r1["pooled"]
        rd += ": K'8 %.3f -> K'16 %.3f (STABLE %s), K'24 %.3f (LONG-STABLE %s), K'32 %s; min n 3-8 at K'8 %.3f (DEEP %s); THINK-NECESSARY %s; DEPTH n 9-12 %s" % (
            p[8], p[16], r1["stable"], p[24], r1["long_stable"], None if p[32] is None else round(p[32], 3), r1["min_k8_3_8"], r1["deep"], r1["think_necessary"],
            None if r1["depth_9_12_n2"] is None else round(r1["depth_9_12_n2"], 3))
        for k, c in comps.items():
            if c.get("status") == "done":
                rd += "; vs %s K'8 %.3f K'16 %.3f K'24 %.3f DEPTH %.3f" % (k, c["pooled"][8], c["pooled"][16], c["pooled"][24], c["depth_9_12_n2"])
        if rd.startswith("JOINT"):
            r2 = runs[2]
            rd += "; s2 opens" if r2.get("status") != "done" else "; s2 %s, LONG-STABLE %s" % (label(r2), r2["long_stable"])
            res["c6_claim"] = None if r2.get("status") != "done" else bool(r1["long_stable"] and label(r2) == "JOINT" and r2["long_stable"])
            rd += "; C6 CLAIM (JOINT + LONG-STABLE, 2 / 2) %s" % ("pending s2" if res["c6_claim"] is None else res["c6_claim"])
    try:                                                                    # basin-entry flag (pinned before output; theory/basin_entry.py; descriptive)
        import basin_entry as BE
        if os.path.exists(os.path.join(R, "s188_rwwin_s1", "log.jsonl")):
            res["entry"] = {c: BE.confounded("s188_rwwin_s1", c) for c in ('s184_reweight_s1', 's182_window_s1', 's182_window_s2')}
            rd += "; entry(0.5/0.1) cell %s vs " % ((res["entry"]['s184_reweight_s1']["cell"]["entry_0.5"], res["entry"]['s184_reweight_s1']["cell"]["entry_0.1"]),) + ", ".join(
                "%s %s/%s%s" % (c, v["comp"]["entry_0.5"], v["comp"]["entry_0.1"], " CONFOUNDED-BY-ENTRY" if v["confounded_by_entry"] else "") for c, v in res["entry"].items())
    except Exception as ex:                                                 # noqa: BLE001  (descriptive; never blocks the reading)
        rd += "; entry flag unavailable (%s)" % ex
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s188.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
