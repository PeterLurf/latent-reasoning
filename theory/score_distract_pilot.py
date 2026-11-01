"""THEORY Note 255-5 DISTRACT pilot gate ($0 local; committed before output; Theory is the only scorer).
Pilot pilot_dis_s1: --init_model rcb_plaw4_s1, rcb_plaw4_s1's args + --distract_p (0-2 irrelevant op sentences per training item at random op-block positions; chain order kept),
--steps 16000 --stop_at 2000. The INSERT eval (Note 255-3 driver, same items / conditions) on its 1k and 2k checkpoints -> scratch/s255_distract/insert_{1k,2k}.json (insert.json format).
Rule on 2k (model key pilot_dis_s1): GO if clean acc8 >= 0.95 and before_k2 hop['1'] >= 0.60 and inside_k2 acc8 >= 0.50;
PARK if clean acc8 < 0.80 or before_k2 hop['1'] < 0.25; else NOTE. (Step 0 = rcb_plaw4_s1: clean 1.000, before_k2 hop1 0.118, inside_k2 acc8 0.113.)
    python -B theory/score_distract_pilot.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_HR, "scratch", "s255_distract"); KEY = "pilot_dis_s1"
OUT = os.path.join(_HR, "theory", "results", "score_distract_pilot.json")


def read(path):
    try:
        return json.load(open(path))[KEY]
    except (OSError, ValueError, KeyError):
        return None


def summ(m):
    return {"clean_acc8": m["clean"]["acc8"], "before_k2_hop1": m["before_k2"]["hop"]["1"], "inside_k2_acc8": m["inside_k2"]["acc8"],
            "after_k2_acc8": m["after_k2"]["acc8"]}


def score(d=D):
    m1, m2 = read(os.path.join(d, "insert_1k.json")), read(os.path.join(d, "insert_2k.json"))
    if m2 is None:
        return {"verdict": "pending"}
    s2 = summ(m2)
    v = ("GO" if (s2["clean_acc8"] >= 0.95 and s2["before_k2_hop1"] >= 0.60 and s2["inside_k2_acc8"] >= 0.50) else
         "PARK" if (s2["clean_acc8"] < 0.80 or s2["before_k2_hop1"] < 0.25) else "NOTE")
    return {"verdict": v, "2k": s2, "1k": summ(m1) if m1 else None}


def fixtures():
    res = []

    def world(c, b, i):
        d = tempfile.mkdtemp()
        m = {"clean": {"acc8": c, "hop": {"1": c}}, "before_k2": {"acc8": b, "hop": {"1": b}}, "inside_k2": {"acc8": i, "hop": {"1": i}}, "after_k2": {"acc8": 0.5, "hop": {"1": 0.5}}}
        for t in ("1k", "2k"):
            json.dump({KEY: m}, open(os.path.join(d, "insert_%s.json" % t), "w"))
        return d
    for name, args, want in (("go", (0.98, 0.7, 0.6), "GO"), ("chain broken", (0.7, 0.9, 0.9), "PARK"), ("no recovery", (0.99, 0.15, 0.1), "PARK"),
                             ("partial", (0.97, 0.4, 0.3), "NOTE"), ("boundary", (0.95, 0.60, 0.50), "GO")):
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
