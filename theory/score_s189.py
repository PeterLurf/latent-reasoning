"""THEORY §189 scoring (readouts fixed in §189; written before any seed-2 RMS run; Theory is the only scorer).
RMS-REP: §185 RMS exactly with seed 2 (s185_rms_s2). Comparator = §184 s2 (s184_reweight_s2, same seed). DEEP = >= 0.90 at every n 3..8 at K' 8;
THINK-NECESSARY = 170-4 removal (K' 0 / 1); DEPTH = mean n 9-12 at K' = n + 2.
SUBSTITUTE = DEEP and THINK-NECESSARY and DEPTH >= §184 s2's - 0.05; DEEP-ONLY = DEEP and THINK-NECESSARY, DEPTH below that; NOT REPLICATED = not DEEP.
Descriptive: two-seed means vs §184's, K' curve, §182 STABLE.
    python theory/score_s189.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s185 as S185  # noqa: E402  (read(): DEEP / THINK-NECESSARY / STABLE / pooled K')
import score_s187 as S187  # noqa: E402  (read(): DEPTH at K' = n + 2)

TOL = 0.05


def main():
    want = dict(S185.ARMS["rms"]); out = {}
    for s in (1, 2):
        r = S185.read("s185_rms_s%d" % s, dict(want, seed=s)); d = S187.read("s185_rms_s%d" % s, dict(want, seed=s))
        out["rms_s%d" % s] = dict(r, depth_9_12_n2=d.get("depth_9_12_n2")) if r.get("status") == "done" else r
    for s in (1, 2):
        c = S187.read("s184_reweight_s%d" % s, dict(S187.COMP_ARGS, seed=s))
        out["s184_s%d" % s] = c
    r2, c2 = out["rms_s2"], out["s184_s2"]
    if r2.get("status") != "done" or c2.get("status") != "done":
        rd = "pending (RMS s2 %s; §184 s2 %s)" % (r2.get("status"), c2.get("status"))
    else:
        if not r2["deep"]:
            rd = "NOT REPLICATED"
        elif r2["think_necessary"] and r2["depth_9_12_n2"] >= c2["depth_9_12_n2"] - TOL - S185.EPS:
            rd = "SUBSTITUTE"
        elif r2["think_necessary"]:
            rd = "DEEP-ONLY"
        else:
            rd = "DEEP but not THINK-NECESSARY (no reading)"
        r1, c1 = out["rms_s1"], out["s184_s1"]
        rd += ": s2 K'8 %.3f, DEEP %s, THINK-NECESSARY %s, DEPTH %.3f vs §184 s2 %.3f (bar %.3f); K'16 %.3f (STABLE %s)" % (
            r2["pooled_k8"], r2["deep"], r2["think_necessary"], r2["depth_9_12_n2"], c2["depth_9_12_n2"], c2["depth_9_12_n2"] - TOL, r2["pooled_k16"], r2["stable"])
        if r1.get("status") == "done" and c1.get("status") == "done":
            rd += "; two-seed DEPTH RMS %.3f vs §184 %.3f" % ((r1["depth_9_12_n2"] + r2["depth_9_12_n2"]) / 2, (c1["depth_9_12_n2"] + c2["depth_9_12_n2"]) / 2)
    out["reading"] = rd; print("READING:", rd)
    errs = S185.ERRORS + S187.ERRORS
    if errs:
        print("ERRORS:", errs)
    out["errors"] = [str(e) for e in errs]
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s189.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
