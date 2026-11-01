"""THEORY Note 254-1 / 254-2 curriculum pilot gate (Theory is the only scorer; committed before the pilot output is read).
Cell pilot_cur_s1: --init_model p2pc_s1, rcb_plaw4_s1's args + --perm_p 0.5, --steps 16000 --stop_at 2000 (Audit 254-2 (1)).
Fields (raw logged values, the step-1000 / step-2000 records): ws_readout_chain['2']['2'], ws_readout_perm['2']['2'].
Rule at step 2000: GO if chain >= 0.90 and perm >= 0.50; PARK if chain < 0.50; else NOTE (Theory decides with a Note).
    python -B theory/score_cur_pilot.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq"); RUN = "pilot_cur_s1"
OUT = os.path.join(_HR, "theory", "results", "score_cur_pilot.json")


def score(runs=R):
    rec = {}
    try:
        for ln in open(os.path.join(runs, RUN, "log.jsonl")):
            x = json.loads(ln)
            if x.get("step") in (1000, 2000) and "ws_readout_chain" in x and "ws_readout_perm" in x:
                rec[x["step"]] = {"chain": x["ws_readout_chain"]["2"]["2"], "perm": x["ws_readout_perm"]["2"]["2"]}
    except OSError:
        pass
    if 2000 not in rec:
        return {"verdict": "pending", "rec": rec}
    c, p = rec[2000]["chain"], rec[2000]["perm"]
    return {"verdict": "GO" if (c >= 0.90 and p >= 0.50) else "PARK" if c < 0.50 else "NOTE", "rec": rec}


def fixtures():
    res = []

    def world(c, p, drop=False):
        d = tempfile.mkdtemp(); q = os.path.join(d, RUN); os.makedirs(q)
        with open(os.path.join(q, "log.jsonl"), "w") as fh:
            for st in (1000, 2000):
                if drop and st == 2000:
                    continue
                fh.write(json.dumps({"step": st, "ws_readout_chain": {"2": {"2": c}}, "ws_readout_perm": {"2": {"2": p}}}) + "\n")
        return d
    for name, args, want in (("go", (0.95, 0.6), "GO"), ("park", (0.4, 0.9), "PARK"), ("note: perm low", (0.95, 0.3), "NOTE"),
                             ("note: chain mid", (0.7, 0.7), "NOTE"), ("boundary", (0.90, 0.50), "GO"), ("pending", (0.9, 0.9, True), "pending")):
        o = score(world(*args)); res.append((name, o["verdict"] == want, o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1)
