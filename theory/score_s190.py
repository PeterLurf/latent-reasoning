"""THEORY §190 scoring (RMS x NOPASS; rules fixed in §190; written before any §190 run; Theory is the only scorer).
Cell s190_rmsnopass_s<seed> = s185_rms_s1's args + --aux_nopass; comparator s185_rms_s<seed> (same seed).
LATE = mean aux handle over t in {7, 8} at n 10-12, nesting <= 2 (per-item rows eval_s187_deepaux_items_n9_12.json; per-n mean, then mean over n x t).
INDIST = mean of s175.aux_acc_by_role[t].handle at t 7, 8. DEEP / THINK-NECESSARY / DEPTH as score_s187.read.
COSTLY-AUX (first) = INDIST <= comp INDIST - 0.05; COSTLY = not DEEP; TOKEN-FIX = DEEP and LATE >= comp + 0.15 and DEPTH >= comp + 0.10 (opens s2;
the claim needs 2 / 2, s2 vs s185_rms_s2); TOKEN-NOT-THE-LIMIT = DEEP and |LATE - comp| < 0.05; else mixed. EDGE (t 9-10) descriptive.
    python theory/score_s190.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s185 as S185  # noqa: E402
import score_s187 as S187  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
EPS = 1e-9
RMS = dict(S185.ARMS["rms"]); CELL = dict(RMS, aux_nopass=True)


def rows_stat(d, ts):
    p = os.path.join(R, d, "eval_s187_deepaux_items_n9_12.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))["items"]; vals = []
    for n in (10, 11, 12):
        it = [r for r in rows if r["n"] == n and r["nest_depth"] <= 2]
        for t in ts:
            c = [r["handle_by_t"][str(t)] for r in it if str(t) in r["handle_by_t"]]
            if c:
                vals.append(sum(c) / len(c))
    return sum(vals) / len(vals) if vals else None


def model(d, want):
    r = S187.read(d, want)
    if r.get("status") != "done":
        return r
    e = json.load(open(os.path.join(R, d, "eval.json"))); ab = (e.get("s175") or {}).get("aux_acc_by_role") or {}
    indist = None if not all(str(t) in ab for t in (7, 8)) else (ab["7"]["handle"] + ab["8"]["handle"]) / 2
    return dict(r, indist_t7_8=indist, late=rows_stat(d, (7, 8)), edge=rows_stat(d, (9, 10)))


def label(c, k):
    if None in (c.get("late"), k.get("late"), c.get("indist_t7_8"), k.get("indist_t7_8")):
        return "incomplete (rows or s175 missing)"
    if c["indist_t7_8"] <= k["indist_t7_8"] - 0.05 + EPS:
        return "COSTLY-AUX"
    if not c["deep"]:
        return "COSTLY"
    if c["late"] >= k["late"] + 0.15 - EPS and c["depth_9_12_n2"] >= k["depth_9_12_n2"] + 0.10 - EPS:
        return "TOKEN-FIX"
    if abs(c["late"] - k["late"]) < 0.05 - EPS:
        return "TOKEN-NOT-THE-LIMIT"
    return "mixed"


def main():
    res = {}
    for s in (1, 2):
        c, k = model("s190_rmsnopass_s%d" % s, dict(CELL, seed=s)), model("s185_rms_s%d" % s, dict(RMS, seed=s))
        res["s%d" % s] = {"cell": c, "comp": k}
        if c.get("status") == "done" and k.get("status") == "done":
            res["s%d" % s]["label"] = label(c, k)
    a = res["s1"]
    if "label" not in a:
        rd = "pending (s1 cell %s; comp %s)" % (a["cell"].get("status"), a["comp"].get("status"))
    else:
        c, k = a["cell"], a["comp"]
        rd = "%s (s1): LATE %s vs %s, DEPTH %.3f vs %.3f, in-dist t7-8 %s vs %s, DEEP %s; EDGE %s vs %s" % (
            a["label"], c["late"] and round(c["late"], 3), k["late"] and round(k["late"], 3), c["depth_9_12_n2"], k["depth_9_12_n2"],
            c["indist_t7_8"] and round(c["indist_t7_8"], 3), k["indist_t7_8"] and round(k["indist_t7_8"], 3), c["deep"],
            c["edge"] and round(c["edge"], 3), k["edge"] and round(k["edge"], 3))
        if a["label"] == "TOKEN-FIX":
            rd += "; s2 opens" if "label" not in res["s2"] else "; s2 %s -> claim %s" % (res["s2"]["label"], res["s2"]["label"] == "TOKEN-FIX")
    res["reading"] = rd; print("READING:", rd)
    errs = S185.ERRORS + S187.ERRORS
    if errs:
        print("ERRORS:", errs)
    res["errors"] = [str(e) for e in errs]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s190.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
