"""THEORY §241 LEN-STEP scorer (docs/research_lenstep_2026-10-28/LS_v1.md; $0 diagnostic; Theory is the only scorer).
Per model (lh_fh_s1 / lh_fh_s2), eval_lenstep.json cells A (d 4) / B (d 4 padded to C's length) / C (d 6) / D (d 6 short, optional):
LEN4 = A - B paired by prog_sig (Note 241-1); STEP = A - D (or STEP' = B - C if D unavailable); LEN6 = D - C paired. Labels: LENGTH (LEN4 >= 0.10, STEP < 0.10) / STEPS (STEP >= 0.10, LEN4 < 0.05) /
BOTH (both >= 0.10) / NEITHER. Length-match gate: |tok_mean B - C| <= 5 and |tok_p90 B - C| <= 10, else INVALID. Overall: the shared label, else MIXED.
    python -B theory/score_ls.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_ls.json")
MODELS = ("lh_fh_s1", "lh_fh_s2")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def one(e):
    if e is None:
        return {"label": "pending"}
    c = e["cells"]
    if any(k not in c for k in "ABC"):
        return {"label": "pending"}
    if abs(c["B"]["tok_mean"] - c["C"]["tok_mean"]) > 5 or abs(c["B"]["tok_p90"] - c["C"]["tok_p90"]) > 10:
        return {"label": "INVALID", "why": "B / C length mismatch"}
    a = {k: c[k]["acc"] for k in c if c[k] and c[k].get("acc") is not None}

    def paired(x, y):   # Note 241-1 (3): same program (prog_sig) rendered short (x) vs long (y); mean of ok_x - ok_y over matched items
        ix, iy = (c[x].get("items") or {}), (c[y].get("items") or {})
        mx = dict(zip(ix.get("prog_sig", []), ix.get("ok", []))); my = dict(zip(iy.get("prog_sig", []), iy.get("ok", [])))
        common = [s for s in mx if s in my]
        return (sum(mx[s] - my[s] for s in common) / len(common), len(common)) if common else (None, 0)
    len4, n4 = paired("A", "B")
    if len4 is None:
        return {"label": "INVALID", "why": "no paired A / B items (241-1 (3))"}
    if "D" in a:
        step, how = a["A"] - a["D"], "A-D"
    else:
        step, how = a["B"] - a["C"], "B-C"
    len6, n6 = paired("D", "C") if "D" in a else (None, 0)
    lab = "BOTH" if (len4 >= 0.10 and step >= 0.10) else "LENGTH" if (len4 >= 0.10 and step < 0.10) else "STEPS" if (step >= 0.10 and len4 < 0.05) else "NEITHER"
    return {"label": lab, "LEN4_paired": len4, "n_pairs_AB": n4, "STEP": step, "step_from": how, "LEN6_paired": len6, "n_pairs_DC": n6, "acc": a,
            "ws": {k: c[k].get("ws") for k in c}}


def score(runs=R):
    o = {"section": 241, "registered": "LS_v1.md", "models": {m: one(jl(os.path.join(runs, m, "eval_lenstep.json"))) for m in MODELS}}
    labs = [v["label"] for v in o["models"].values()]
    if "pending" in labs:
        o["verdict"] = "pending"; return o
    o["verdict"] = "LEN-STEP %s | %s" % (labs[0] if len(set(labs)) == 1 else "MIXED", " / ".join("%s: %s" % (m[-2:], v["label"]) for m, v in o["models"].items()))
    return o


def fixtures():
    res = []

    def world(cells_by_model):
        d = tempfile.mkdtemp()
        for m, cells in cells_by_model.items():
            os.makedirs(os.path.join(d, m)); json.dump({"cells": cells}, open(os.path.join(d, m, "eval_lenstep.json"), "w"))
        return d
    def items(acc, n=1000):
        k = int(round(acc * n)); return {"prog_sig": ["p%d" % i for i in range(n)], "ok": [1] * k + [0] * (n - k)}
    def cells(A, B, C, D=None, tb=100, tc=100):
        c = {"A": {"acc": A, "tok_mean": 80, "tok_p90": 90, "items": items(A)}, "B": {"acc": B, "tok_mean": tb, "tok_p90": tb + 10, "items": items(B)},
             "C": {"acc": C, "tok_mean": tc, "tok_p90": tc + 10, "items": items(C)}}
        if D is not None:
            c["D"] = {"acc": D, "tok_mean": 85, "tok_p90": 92, "items": items(D)}
        return c
    o = score(world({m: cells(1.0, 0.6, 0.4, 0.95) for m in MODELS})); res.append(("length hurts, steps don't -> LENGTH", o["verdict"].startswith("LEN-STEP LENGTH"), o["verdict"]))
    o = score(world({m: cells(1.0, 0.98, 0.4, 0.5) for m in MODELS})); res.append(("steps hurt -> STEPS", o["verdict"].startswith("LEN-STEP STEPS"), o["verdict"]))
    o = score(world({m: cells(1.0, 0.7, 0.3, 0.6) for m in MODELS})); res.append(("both -> BOTH", o["verdict"].startswith("LEN-STEP BOTH"), o["verdict"]))
    o = score(world({"lh_fh_s1": cells(1.0, 0.6, 0.4, 0.95), "lh_fh_s2": cells(1.0, 0.98, 0.4, 0.5)})); res.append(("disagree -> MIXED", o["verdict"].startswith("LEN-STEP MIXED"), o["verdict"]))
    o = score(world({m: cells(1.0, 0.98, 0.5) for m in MODELS})); res.append(("no D: STEP' = B - C -> STEPS", o["verdict"].startswith("LEN-STEP STEPS") and "B-C" in json.dumps(o), o["verdict"]))
    o = score(world({m: cells(1.0, 0.6, 0.4, 0.95, tb=80) for m in MODELS})); res.append(("B not length-matched -> INVALID", o["verdict"].startswith("LEN-STEP INVALID"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
