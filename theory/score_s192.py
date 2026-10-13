"""THEORY §192 scoring (RMS-SEED3; readouts fixed in §192; written before any seed-3 run; Theory is the only scorer).
s185_rms_s3 = s185_rms_s1's args with seed 3. DEEP / THINK-NECESSARY / DEPTH as §189 (score_s185.read + score_s187.read).
CONSISTENT = DEEP and DEPTH >= 0.537 (RMS s1 0.587 - 0.05); LOW-SEED = DEEP and DEPTH < 0.537; NOT-DEEP = not DEEP.
Descriptive: STABLE, basin entry (theory/basin_entry.py) vs s1 / s2.
    python theory/score_s192.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s185 as S185  # noqa: E402
import score_s187 as S187  # noqa: E402

BAR, EPS = 0.537, 1e-9


def main():
    want = dict(S185.ARMS["rms"], seed=3); d = "s185_rms_s3"
    r = S185.read(d, want); dd = S187.read(d, want); res = {"s3": r}
    if r.get("status") != "done" or dd.get("status") != "done":
        rd = "pending (s3 %s)" % r.get("status")
    else:
        dep = dd["depth_9_12_n2"]; res["s3"]["depth_9_12_n2"] = dep
        lab = "NOT-DEEP" if not r["deep"] else "CONSISTENT" if dep >= BAR - EPS else "LOW-SEED"
        rd = "%s: s3 K'8 %.3f, DEEP %s, THINK-NECESSARY %s, DEPTH %.3f (bar %.3f; s1 0.587, s2 0.662); K'16 %.3f (STABLE %s)" % (
            lab, r["pooled_k8"], r["deep"], r["think_necessary"], dep, BAR, r["pooled_k16"], r["stable"])
        try:
            import basin_entry as BE
            e = {s: BE.summary("s185_rms_s%d" % s) for s in (1, 2, 3)}; res["entry"] = e
            rd += "; entry(0.5/0.1) s1 %s/%s, s2 %s/%s, s3 %s/%s" % tuple(x for s in (1, 2, 3) for x in (e[s]["entry_0.5"], e[s]["entry_0.1"]))
        except Exception as ex:  # noqa: BLE001
            rd += "; entry unavailable (%s)" % ex
    res["reading"] = rd; print("READING:", rd)
    errs = S185.ERRORS + S187.ERRORS
    if errs:
        print("ERRORS:", errs)
    res["errors"] = [str(e) for e in errs]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s192.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
