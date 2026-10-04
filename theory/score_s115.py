"""THEORY §115 scoring (rules fixed in §115; written before any §115 run; Theory is the only scorer).
Inputs: <run>/eval_s115.json for §108 MK-4 models (boxes/runs_s5/s108_mk4_s*) and §105 models (boxes/runs_s5/g105_mask4_s*).
NOSNAP (per Markov model): nosnap.cumulative["512"] >= 0.99.
TRIGGER (per model): trigger.first_error_detection >= 0.95 AND trigger.cumulative["512"] >= fixed_k4.cumulative["512"]
  AND trigger.snaps_per_problem.mean < fixed_k4.snaps_per_problem.mean.
Descriptive: per-hand-off by position and drift (no-snap); false-snap and miss rates; 2,048 where available.
    python theory/score_s115.py
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    res = {"models": {}}
    for f in sorted(glob.glob(os.path.join(ROOT, "boxes", "runs_s5", "*", "eval_s115.json"))):
        name = os.path.basename(os.path.dirname(f)); r = json.load(open(f)); M = {}
        ns = r.get("nosnap")
        if ns:
            M["nosnap_cum"] = ns["cumulative"]; M["NOSNAP"] = ns["cumulative"]["512"] >= 0.99
            M["drift"] = ns.get("drift")
        tr, fk = r.get("trigger"), r.get("fixed_k4")
        if tr and fk:
            M["trigger"] = {k: tr.get(k) for k in ("first_error_detection", "false_snap_rate", "miss_rate", "cumulative", "snaps_per_problem")}
            M["fixed_k4"] = fk
            M["TRIGGER"] = (tr["first_error_detection"] is not None and tr["first_error_detection"] >= 0.95
                            and tr["cumulative"]["512"] >= fk["cumulative"]["512"]
                            and tr["snaps_per_problem"]["mean"] < fk["snaps_per_problem"]["mean"])
        res["models"][name] = M
        print("%-16s NOSNAP %s %s | TRIGGER %s fed %s snaps %s vs %s | cum %s vs %s" % (
            name, M.get("NOSNAP"), (M.get("nosnap_cum") or {}).get("512"), M.get("TRIGGER"),
            (M.get("trigger") or {}).get("first_error_detection"), ((M.get("trigger") or {}).get("snaps_per_problem") or {}).get("mean"),
            ((M.get("fixed_k4") or {}).get("snaps_per_problem") or {}).get("mean"),
            ((M.get("trigger") or {}).get("cumulative") or {}).get("512"), ((M.get("fixed_k4") or {}).get("cumulative") or {}).get("512")))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s115.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
