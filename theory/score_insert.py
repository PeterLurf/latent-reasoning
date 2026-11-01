"""THEORY Note 255-3 INSERT read ($0, eval-only; committed before output; Theory is the only scorer).
Input scratch/s255_insert/insert.json: {model: {cond: {"hop": {"1": p, "2": p, "3": p}, "acc8": p, "n": int}}}, cond in clean, before_k{1,2}, inside_k{1,2}, after_k{1,2}.
hop[k] = share of d 4 items whose answer at K' = k equals v_{k+1} (the decoder adds one step to S_k); acc8 = d 4 acc at K' 8.
Per model, drop(cond) = clean.hop['1'] - cond.hop['1'] (first hop, the d 5-6 sign in Note 255-2), at k 2 (strongest insertion):
  COUNT-DILUTION if drop(after_k2) >= 0.20 (extra op sentences that do not shift the chain's slots still hurt);
  POSITION-SHIFT if drop(before_k2) >= 0.20 and drop(after_k2) <= 0.05 (the pointer counts op-sentence slots);
  CHAIN-SPECIFIC if drop(before_k2), drop(inside_k2), drop(after_k2) all <= 0.05;
  else MIXED.
    python -B theory/score_insert.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(_HR, "scratch", "s255_insert", "insert.json")
OUT = os.path.join(_HR, "theory", "results", "score_insert.json")


def label(m):
    d = lambda c: m["clean"]["hop"]["1"] - m[c]["hop"]["1"]  # noqa: E731
    db, di, da = d("before_k2"), d("inside_k2"), d("after_k2")
    lab = ("COUNT-DILUTION" if da >= 0.20 else "POSITION-SHIFT" if (db >= 0.20 and da <= 0.05) else
           "CHAIN-SPECIFIC" if max(db, di, da) <= 0.05 else "MIXED")
    return {"label": lab, "drop_k2": {"before": db, "inside": di, "after": da},
            "drop_k1": {c: d(c + "_k1") for c in ("before", "inside", "after")}}


def score(path=IN):
    try:
        J = json.load(open(path))
    except (OSError, ValueError):
        return {"verdict": "pending"}
    o = {m: label(v) for m, v in J.items() if isinstance(v, dict) and "clean" in v}
    o["verdict"] = " | ".join("%s %s" % (m, o[m]["label"]) for m in sorted(k for k in o if k != "verdict")); return o


def fixtures():
    res = []

    def world(b, i, a):
        base = {"hop": {"1": 0.95, "2": 0.95, "3": 0.95}, "acc8": 1.0, "n": 1000}
        m = {"clean": base}
        for c, v in (("before", b), ("inside", i), ("after", a)):
            for k in ("k1", "k2"):
                m["%s_%s" % (c, k)] = {"hop": {"1": 0.95 - v, "2": 0.9, "3": 0.9}, "acc8": 0.9, "n": 1000}
        p = os.path.join(tempfile.mkdtemp(), "insert.json"); json.dump({"rcb_plaw4_s1": m}, open(p, "w")); return p
    for name, args, want in (("dilution", (0.3, 0.3, 0.3), "COUNT-DILUTION"), ("position", (0.4, 0.2, 0.02), "POSITION-SHIFT"),
                             ("chain", (0.02, 0.03, 0.01), "CHAIN-SPECIFIC"), ("mixed", (0.1, 0.1, 0.1), "MIXED")):
        o = score(world(*args)); res.append((name, o["verdict"] == "rcb_plaw4_s1 " + want, o["verdict"]))
    o = score("nonexistent.json"); res.append(("pending", o["verdict"] == "pending", o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1)
