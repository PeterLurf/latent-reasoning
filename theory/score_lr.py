"""THEORY §239 LONG-RUN scorer (docs/research_longrun_2026-10-28/LR_v1.md; Theory is the only scorer). Eval-only C6 stability.
Binding (df_plaw4_s1 / df_plaw4_s2, eval_long.json on TEST2-DF d 2-4): HOLD-LONG acc(K') >= acc(8) - 0.02 for K' in {16, 32, 64}, every d;
  SEED-VAR |acc_s1 - acc_s2| <= 0.05 at every (d, K'). Labels per seed: PASS / DRIFT / COLLAPSE (any acc < 0.5 or non-finite s_norm); overall
  PASS (both pass HOLD-LONG and SEED-VAR) / SEED-SPLIT (HOLD-LONG both, SEED-VAR fails) / FAIL. Descriptive: s_norm growth, s_cos, rcb_* table, d 5.
    python -B theory/score_lr.py [--fixtures]
"""
import argparse, json, math, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_lr.json")
LATS, DESC = ("df_plaw4_s1", "df_plaw4_s2"), ("rcb_plaw4_s1", "rcb_plaw4_s2")
DS, KS = ("2", "3", "4"), ("16", "32", "64")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def seed(e):
    if e is None:
        return {"label": "pending", "why": "missing eval_long"}
    b = e["by_d"]
    if any(d not in b or any(k not in b[d]["acc"] for k in ("8",) + KS) for d in DS):
        return {"label": "pending", "why": "missing d / K'"}
    nonfin = any(not math.isfinite(v) for d in DS for v in b[d].get("s_norm", {}).values())
    drops = {d: {k: b[d]["acc"][k] - b[d]["acc"]["8"] for k in KS} for d in DS}
    mn = min(b[d]["acc"][k] for d in DS for k in ("8",) + KS)
    ok = all(v >= -0.02 for r in drops.values() for v in r.values())
    lab = "COLLAPSE" if (nonfin or mn < 0.5) else ("PASS" if ok else "DRIFT")
    growth = {d: (b[d]["s_norm"]["64"] / b[d]["s_norm"]["8"]) if b[d].get("s_norm") else None for d in DS}
    return {"label": lab, "drops": drops, "min_acc": mn, "s_norm_growth": growth, "s_cos64": {d: (b[d].get("s_cos") or {}).get("64") for d in DS},
            "acc": {d: b[d]["acc"] for d in b}}


def score(runs=R):
    ev = {r: jl(os.path.join(runs, r, "eval_long.json")) for r in LATS + DESC}
    o = {"section": 239, "registered": "LR_v1.md", "seeds": {r: seed(ev[r]) for r in LATS}, "descriptive": {r: seed(ev[r]) for r in DESC}}
    labs = [o["seeds"][r]["label"] for r in LATS]
    if "pending" in labs:
        o["verdict"] = "pending (%s)" % [o["seeds"][r].get("why") for r in LATS]; return o
    a = [ev[r]["by_d"] for r in LATS]
    var = max(abs(a[0][d]["acc"][k] - a[1][d]["acc"][k]) for d in DS for k in ("8",) + KS); o["seed_var_max"] = var
    if labs == ["PASS", "PASS"]:
        o["verdict"] = "LONG-RUN PASS" if var <= 0.05 else "LONG-RUN SEED-SPLIT"
    else:
        o["verdict"] = "LONG-RUN FAIL"
    o["verdict"] += " | s1: %s / s2: %s | seed var max %.3f" % (labs[0], labs[1], var); return o


def fixtures():
    res = []

    def world(acc_fn=lambda r, d, k: 1.0, norm_fn=lambda r, d, k: 10.0):
        dd = tempfile.mkdtemp()
        for r in LATS + DESC:
            os.makedirs(os.path.join(dd, r))
            by = {d: {"acc": {k: acc_fn(r, d, k) for k in ("8",) + KS}, "s_norm": {k: norm_fn(r, d, k) for k in ("8",) + KS}, "s_cos": {k: 0.99 for k in ("8",) + KS}} for d in DS + ("5",)}
            json.dump({"by_d": by}, open(os.path.join(dd, r, "eval_long.json"), "w"))
        return dd
    o = score(world()); res.append(("stable -> PASS", o["verdict"].startswith("LONG-RUN PASS"), o["verdict"]))
    o = score(world(lambda r, d, k: 1.0 - (0.04 if (k == "64" and r == "df_plaw4_s1") else 0))); res.append(("s1 drops 0.04 at 64 -> FAIL (s1 DRIFT)", "s1: DRIFT" in o["verdict"] and o["verdict"].startswith("LONG-RUN FAIL"), o["verdict"]))
    o = score(world(lambda r, d, k: 0.3 if k == "64" else 1.0)); res.append(("collapse at 64 -> COLLAPSE", o["verdict"].count("COLLAPSE") == 2, o["verdict"]))
    o = score(world(norm_fn=lambda r, d, k: float("inf") if k == "64" else 10.0)); res.append(("non-finite norm -> COLLAPSE", "COLLAPSE" in o["verdict"], o["verdict"]))
    o = score(world(lambda r, d, k: 0.90 if r == "df_plaw4_s2" else 1.0)); res.append(("both stable, 0.10 apart -> SEED-SPLIT", o["verdict"].startswith("LONG-RUN SEED-SPLIT"), o["verdict"]))
    o = score(world(lambda r, d, k: 1.0 - (0.019 if k == "32" else 0))); res.append(("drop 0.019 within tolerance -> PASS", o["verdict"].startswith("LONG-RUN PASS"), o["verdict"]))
    d = world(); os.remove(os.path.join(d, "df_plaw4_s2", "eval_long.json")); o = score(d); res.append(("missing -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
