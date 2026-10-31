"""THEORY §252 PERM-LOOKUP pilot gate (NS_v1 §3; Note 252-1; Theory is the only scorer). Committed before the pilot output is read (Audit condition 1).
Reads log.jsonl of the two local pilot cells (seed 1, 2k steps): pilot_np_s1 (--perm_p 0.5) and pilot_gp_s1 (--perm_p 0.5 --cue generic).
Field: ws_readout_perm["2"]["1"] (pass-1 ':' readout acc vs v_1 on PERMUTED d 2 training items) in the step-2000 record (1k record descriptive).
Rule per arm: GO if >= 0.50; PARK if <= 0.20; otherwise MID (Theory: NP goes, GP parks).
    python -B theory/score_ns_pilot.py [--fixtures] [--runs DIR]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_ns_pilot.json")
CELLS = {"NP": "pilot_np_s1", "GP": "pilot_gp_s1"}


def read(runs, run):
    rec = {}
    try:
        for ln in open(os.path.join(runs, run, "log.jsonl")):
            x = json.loads(ln)
            if x.get("step") in (1000, 2000) and "ws_readout_perm" in x:
                rec[x["step"]] = x["ws_readout_perm"]["2"]["1"]
    except OSError:
        return None
    return rec


def score(runs=R):
    o = {"section": 252, "stage": "pilot", "arms": {}}
    for arm, run in CELLS.items():
        rec = read(runs, run)
        if not rec or 2000 not in rec:
            o["arms"][arm] = {"label": "pending", "rec": rec}; continue
        v = rec[2000]; lab = "GO" if v >= 0.50 else "PARK" if v <= 0.20 else "MID"
        o["arms"][arm] = {"label": lab, "ws_perm_d2_p1": rec}
    labs = {a: o["arms"][a]["label"] for a in CELLS}
    if "pending" in labs.values():
        o["verdict"] = "pending %s" % labs; return o
    run_np = labs["NP"] in ("GO", "MID"); run_gp = labs["GP"] == "GO"
    o["verdict"] = "NP %s (%s) | GP %s (%s)" % (labs["NP"], "run" if run_np else "park", labs["GP"], "run" if run_gp else "park")
    return o


def fixtures():
    res = []

    def world(np_v, gp_v, drop=None):
        d = tempfile.mkdtemp()
        for run, v in ((CELLS["NP"], np_v), (CELLS["GP"], gp_v)):
            os.makedirs(os.path.join(d, run))
            with open(os.path.join(d, run, "log.jsonl"), "w") as fh:
                for st in (500, 1000, 1500, 2000):
                    if drop == run and st == 2000:
                        continue
                    r = {"step": st}
                    if st in (1000, 2000):
                        r["ws_readout_perm"] = {"2": {"1": v}}
                    fh.write(json.dumps(r) + "\n")
        return d
    o = score(world(0.8, 0.6)); res.append(("both GO", o["verdict"] == "NP GO (run) | GP GO (run)", o["verdict"]))
    o = score(world(0.3, 0.3)); res.append(("both MID -> NP run, GP park", o["verdict"] == "NP MID (run) | GP MID (park)", o["verdict"]))
    o = score(world(0.1, 0.15)); res.append(("both PARK", o["verdict"] == "NP PARK (park) | GP PARK (park)", o["verdict"]))
    o = score(world(0.5, 0.2)); res.append(("boundaries 0.50 GO / 0.20 PARK", o["verdict"] == "NP GO (run) | GP PARK (park)", o["verdict"]))
    o = score(world(0.9, 0.9, drop=CELLS["GP"])); res.append(("missing 2k record -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); ap.add_argument("--runs", default=R); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(a.runs); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1)
