"""THEORY §261 TARGET-CUE pilot gate (Theory is the only scorer; committed before output).
Pilot pilot_tc_s1: --init_model bun_s1, bun_s1's args + --cue target (the same text every pass, naming only the asked entity S_d:
  ' To find how many <u> <S_d> has, what is the next value? The answer is:'), --steps 8000 --stop_at 2000. Host evals at 2k under the target cue:
  eval_acc (TEST2 d 1-6), eval_ws, eval_insert_ho (H-EXTEND, H-LIVE, H-CHAINSRC-LIVE, ...), gates_pass.json 'target_only' (cue byte-identical across passes; contains S_d's name and no other entity name).
Step-0 reference pilot_tc_s1_step0 (bun_s1 under the target cue): same files, descriptive.
Rule at 2k: GO if TEST2 d 2-4 >= 0.90 each, WS d 3 passes 2-3 >= 0.50, H-EXTEND acc8 >= 0.60 and H-LIVE acc8 >= 0.60, and target_only true;
  PARK if TEST2 d 3 < 0.20, or (H-EXTEND < 0.30 and H-LIVE < 0.50) (no better than the generic cue, Note 260-5); else NOTE.
    python -B theory/score_tc.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_tc.json")
RUN, REF0 = "pilot_tc_s1", "pilot_tc_s1_step0"


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def read(runs, run):
    a = jl(os.path.join(runs, run, "eval_acc.json")); w = jl(os.path.join(runs, run, "eval_ws.json")); h = jl(os.path.join(runs, run, "eval_insert_ho.json"))
    g = jl(os.path.join(runs, run, "gates_pass.json"))
    if a is None or w is None or h is None:
        return None
    wb = w.get("by_d", w)
    return {"acc": {d: a["by_d"][d]["acc"] for d in ("1", "2", "3", "4", "5", "6") if d in a["by_d"]}, "ws3": {t: wb["3"][t] for t in ("1", "2", "3")},
            "fam": {F: v["acc8"] for F, v in h["fam"].items()}, "target_only": (g or {}).get("target_only")}


def score(runs=R):
    m = read(runs, RUN); r0 = read(runs, REF0)
    if m is None:
        return {"verdict": "pending"}
    f = m["fam"]
    go = (all(m["acc"][d] >= 0.90 for d in ("2", "3", "4")) and m["ws3"]["2"] >= 0.50 and m["ws3"]["3"] >= 0.50 and f["H-EXTEND"] >= 0.60 and f["H-LIVE"] >= 0.60
          and m["target_only"] is True)
    park = m["acc"]["3"] < 0.20 or (f["H-EXTEND"] < 0.30 and f["H-LIVE"] < 0.50)
    return {"verdict": "GO" if go else "PARK" if park else "NOTE", "pilot_2k": m, "step0": r0}


def fixtures():
    res = []

    def world(acc=0.98, ws=0.95, ext=0.8, live=0.8, to=True):
        d = tempfile.mkdtemp()
        for run in (RUN, REF0):
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"by_d": {x: {"acc": acc} for x in ("1", "2", "3", "4", "5", "6")}}, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {"3": {"1": 1.0, "2": ws, "3": ws}}}, open(os.path.join(p, "eval_ws.json"), "w"))
            json.dump({"fam": {"H-EXTEND": {"acc8": ext}, "H-LIVE": {"acc8": live}}}, open(os.path.join(p, "eval_insert_ho.json"), "w"))
            json.dump({"target_only": to}, open(os.path.join(p, "gates_pass.json"), "w"))
        return d
    for name, kw, want in (("go", {}, "GO"), ("like generic -> park", {"ext": 0.1, "live": 0.4}, "PARK"), ("chain lost -> park", {"acc": 0.1}, "PARK"),
                           ("partial -> note", {"ext": 0.5, "live": 0.7}, "NOTE"), ("assert missing -> not GO", {"to": False}, "NOTE")):
        o = score(world(**kw)); res.append((name, o["verdict"] == want, o["verdict"]))
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
