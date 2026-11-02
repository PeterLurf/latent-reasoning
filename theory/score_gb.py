"""THEORY §260 GENERIC-BUN pilot gate (Note 260-1; Theory is the only scorer; committed before output).
Pilot pilot_gb_s1: --init_model bun_s1, bun_s1's args + --cue generic (' What is the next value? The answer is:' at every pass), --steps 8000 --stop_at 2000.
Host evals at the 2k ckpt (generic cue throughout): eval_acc (TEST2 d 1-6), eval_ws (TEST2 d 1-4 per pass).
Step-0 reference (pilot_gb_s1_step0, the bun_s1 init under the generic cue): same files, descriptive.
Rule at 2k: GO if eval_acc by_d d 2, 3, 4 acc >= 0.50 each AND eval_ws by_d['3'] passes 2, 3 >= 0.50; PARK if eval_acc by_d['3'].acc < 0.20; else NOTE.
    python -B theory/score_gb.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_gb.json")
RUN, REF0 = "pilot_gb_s1", "pilot_gb_s1_step0"


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def read(runs, run):
    a = jl(os.path.join(runs, run, "eval_acc.json")); w = jl(os.path.join(runs, run, "eval_ws.json"))
    if a is None or w is None:
        return None
    wb = w.get("by_d", w)
    return {"acc": {d: a["by_d"][d]["acc"] for d in ("1", "2", "3", "4", "5", "6") if d in a["by_d"]}, "ws3": {t: wb["3"][t] for t in ("1", "2", "3")}}


def score(runs=R):
    m = read(runs, RUN); r0 = read(runs, REF0)
    if m is None:
        return {"verdict": "pending"}
    go = all(m["acc"][d] >= 0.50 for d in ("2", "3", "4")) and m["ws3"]["2"] >= 0.50 and m["ws3"]["3"] >= 0.50
    v = "GO" if go else "PARK" if m["acc"]["3"] < 0.20 else "NOTE"
    return {"verdict": v, "pilot_2k": m, "step0": r0}


def fixtures():
    res = []

    def world(acc, ws):
        d = tempfile.mkdtemp()
        for run in (RUN, REF0):
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"by_d": {x: {"acc": acc} for x in ("1", "2", "3", "4", "5", "6")}}, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {"3": {"1": 1.0, "2": ws, "3": ws}}}, open(os.path.join(p, "eval_ws.json"), "w"))
        return d
    for name, args, want in (("go", (0.8, 0.8), "GO"), ("park", (0.1, 0.1), "PARK"), ("note", (0.4, 0.6), "NOTE"), ("ws low", (0.9, 0.3), "NOTE")):
        o = score(world(*args)); res.append((name, o["verdict"] == want, o["verdict"]))
    o = score(tempfile.mkdtemp()); res.append(("pending", o["verdict"] == "pending", o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1)
