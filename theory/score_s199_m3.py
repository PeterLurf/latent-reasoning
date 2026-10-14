"""THEORY §199 note 199-M3 scoring (LAG-SPLIT; rules fixed before any output; Theory is the only scorer).
<run>/eval_s199_lag.json = {"by_t": {"5": {"n", "correct", "lag1", "lag2plus", "other"}, ...}, ...}.
LAGFRAC = (lag1 + lag2plus) / (wrong), pooled over t 5..8. PREDICTED LAGFRAC < 0.5 in each failed model; FALSIFIED if >= 0.5 in >= 2 of 3;
INSUFFICIENT if a model has < 30 wrong outputs. Converged models are descriptive.
    python theory/score_s199_m3.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
FAILED = ["s185_rms_s3", "s185_den_s1", "s188_rwwin_s1"]
DESC = ["s185_rms_s1", "s185_rms_s2"]


def lagfrac(d):
    p = os.path.join(R, d, "eval_s199_lag.json")
    if not os.path.exists(p):
        return None
    bt = json.load(open(p))["by_t"]; tot = {k: sum(bt[t][k] for t in bt if 5 <= int(t) <= 8) for k in ("n", "correct", "lag1", "lag2plus", "other")}
    wrong = tot["lag1"] + tot["lag2plus"] + tot["other"]
    assert wrong == tot["n"] - tot["correct"], (d, tot)
    return {"wrong": wrong, "lagfrac": (tot["lag1"] + tot["lag2plus"]) / wrong if wrong else None, "lag1_share": tot["lag1"] / wrong if wrong else None, **tot}


def main():
    res = {d: lagfrac(d) for d in FAILED + DESC}
    f = [res[d] for d in FAILED]
    if any(r is None for r in f):
        lab = "pending (%s)" % [d for d in FAILED if res[d] is None]
    elif any(r["wrong"] < 30 for r in f):
        lab = "INSUFFICIENT (%s)" % [d for d in FAILED if res[d]["wrong"] < 30]
    else:
        hi = sum(r["lagfrac"] >= 0.5 for r in f)
        lab = "FALSIFIED (%d of 3 failed models LAGFRAC >= 0.5)" % hi if hi >= 2 else ("PREDICTED (all < 0.5)" if hi == 0 else "MIXED (1 of 3 >= 0.5)")
    rd = "M3 %s :: %s" % (lab, " | ".join("%s %s" % (d, "pending" if r is None else "LAGFRAC %s (lag1 %s) over %d wrong / %d" % (
        None if r["lagfrac"] is None else round(r["lagfrac"], 3), None if r["lag1_share"] is None else round(r["lag1_share"], 3), r["wrong"], r["n"])) for d, r in res.items()))
    print("READING:", rd)
    json.dump({"reading": rd, "models": res}, open(os.path.join(ROOT, "theory", "results", "score_s199_m3.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
