"""THEORY §233 STEP-TOLD scorer, P-2 pilot (registered text docs/research_steptold_2026-10-24/ST_v1.md at da924dca + Notes 233-1 / 233-2;
field paths per Experiments' inject_st_eval layout (523b61f1); Theory is the only scorer).
Runs (boxes/runs_fsq): LAT-ST st_plst2_s1, CUED-DIRECT st_pcd2_s1 (same RTX 3090), plain DIRECT s228_pdir2c_s1 (CUE-ONLY bar only).
Gates (ST_v1 sec. 3), all at d = 2 unless stated:
  LP0 (each arm; REVIEW_LB #12): no non-finite gn row; no 100-step mean gn_preclip after window 10 > 2 x max(windows 3..10).
  LP1: LAT d = 1 train acc (last 1k bin) >= 0.90.   LP2: LAT aux value acc at pass 1 on d = 2 items (log aux_vacc_by_d_pass["2"]["1"], last 1k bin) >= 0.80.
  LP3 (binding): LAT TEST2 acc >= 0.70 and >= CD + 0.30.
  LP5 (binding): eval_lp5 intact[K'=d] - mean_at_1[K'=d] >= 0.30.   MEAN-STATE: eval_mean_s pooled <= CD + 0.10.
  CUE-ONLY guard: eval_cue_only answer_acc <= max(majority, plain DIRECT d = 2) + 0.05, else VOID.
  CUE-WRONG label: intact[K'=d] - all[K'=d] >= 0.20 -> C1-TOLD (the cue is used), else plain C1 (descriptive label on a pass).
  STILL-RISING flag: LAT d = 2 train acc OLS slope over the last 4 1k bins > +0.02 (one extension to 16k allowed by ST_v1 sec. 3).
P-2 PASS = LP0 (both) & LP1 & LP2 & LP3 & LP5 & MEAN-STATE & not VOID.
    python -B theory/score_st.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_st.json")
LAT, CD, DIR = "st_plst2_s1", "st_pcd2_s1", "s228_pdir2c_s1"
D = "2"


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def rows(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except OSError:
        return None


def lp0(lg):
    g = [r.get("gn_preclip_mean100") for r in lg if "gn_preclip_mean100" in r]
    if any(x is None or not np.isfinite(x) for x in g):
        return False, "non-finite"
    if len(g) < 11:
        return False, "too few windows"
    cap = 2 * max(g[2:10]); bad = [x for x in g[10:] if x > cap]
    return not bad, "cap %.3g, %d over" % (cap, len(bad))


def bins(lg):
    return [r for r in lg if isinstance(r.get("by_d"), dict)]


def score(runs=R):
    o = {"section": 233, "stage": "P-2", "registered": "ST_v1.md @ da924dca + 233-1 / 233-2"}
    f = {}
    for k, run, files in (("lat", LAT, ("eval_acc", "eval_lp5", "eval_mean_s", "eval_cue_only", "eval_cue_wrong")), ("cd", CD, ("eval_acc",)), ("dir", DIR, ("eval_acc",))):
        for x in files:
            f[(k, x)] = jl(os.path.join(runs, run, x + ".json"))
    lgl, lgc = rows(os.path.join(runs, LAT, "log.jsonl")), rows(os.path.join(runs, CD, "log.jsonl"))
    miss = [("%s/%s" % k) for k, v in f.items() if v is None] + [n for n, v in (("lat log", lgl), ("cd log", lgc)) if v is None]
    if miss:
        o["verdict"] = "pending (missing %s)" % miss; return o
    l0a, l0c = lp0(lgl), lp0(lgc); b = bins(lgl)
    lp1 = bool(b) and b[-1]["by_d"]["1"]["acc"] >= 0.90
    av = (b[-1].get("aux_vacc_by_d_pass") or {}).get(D, {}) if b else {}
    lp2v = av.get("1"); lp2 = lp2v is not None and lp2v >= 0.80
    aL = f[("lat", "eval_acc")]["by_d"][D]["acc"]; aC = f[("cd", "eval_acc")]["by_d"][D]["acc"]; aD = f[("dir", "eval_acc")]["by_d"][D]["acc"]
    lp3 = aL >= 0.70 and aL >= aC + 0.30
    l5 = f[("lat", "eval_lp5")]["by_d"][D]; kb = str(l5.get("k_binding", D)); lp5v = l5["intact"][kb] - l5["mean_at_1"][kb]; lp5 = lp5v >= 0.30
    ms = f[("lat", "eval_mean_s")]["by_d"][D]["pooled"]; msok = ms <= aC + 0.10
    co = f[("lat", "eval_cue_only")]["by_d"][D]; guard = co["answer_acc"] <= max(co["majority"], aD) + 0.05
    cw = f[("lat", "eval_cue_wrong")]["by_d"][D]; cwd = l5["intact"][kb] - cw["all"][kb]
    tail = [r["by_d"][D]["acc"] for r in b if D in r["by_d"] and r["by_d"][D].get("acc") is not None][-4:]
    rising = len(tail) == 4 and float(np.polyfit(range(4), tail, 1)[0]) > 0.02
    o.update({"LP0": {"lat": l0a, "cd": l0c}, "LP1": lp1, "LP2": lp2, "LP2_value": lp2v, "LP3": lp3, "acc_d2": {"lat": aL, "cd": aC, "dir": aD},
              "LP5": lp5, "LP5_drop": lp5v, "MEAN_STATE": msok, "mean_state_acc": ms, "CUE_ONLY_ok": guard, "cue_only_acc": co["answer_acc"],
              "CUE_WRONG_drop": cwd, "STILL_RISING": rising})
    if not guard:
        o["verdict"] = "VOID (CUE-ONLY guard: cue alone answers d = 2 at %.3f)" % co["answer_acc"]; return o
    ok = l0a[0] and l0c[0] and lp1 and lp2 and lp3 and lp5 and msok
    lab = ("C1-TOLD" if cwd >= 0.20 else "plain C1 (cue unused)") if ok else None
    fails = [k for k, v in (("LP0-lat", l0a[0]), ("LP0-cd", l0c[0]), ("LP1", lp1), ("LP2", lp2), ("LP3", lp3), ("LP5", lp5), ("MEAN-STATE", msok)) if not v]
    o["verdict"] = ("P-2 PASS (%s)" % lab) if ok else ("P-2 FAIL (%s)%s" % (",".join(fails), " | STILL-RISING" if rising else ""))
    return o


# ------------------------------------------------------------------ fixtures
def _world(**kw):
    d = tempfile.mkdtemp(); g = dict(lat=0.9, cd=0.1, dir=0.1, lp2=0.9, intact=0.9, mean1=0.2, ms=0.12, co=0.1, maj=0.1, cw=0.5, gn_spike=False, d1=0.99, rise=False)
    g.update(kw)
    for run, acc in ((LAT, g["lat"]), (CD, g["cd"]), (DIR, g["dir"])):
        p = os.path.join(d, run); os.makedirs(p)
        json.dump({"by_d": {"1": {"acc": 1.0}, "2": {"acc": acc}}}, open(os.path.join(p, "eval_acc.json"), "w"))
        with open(os.path.join(p, "log.jsonl"), "w") as fh:
            for i in range(1, 121):
                gn = 40.0 if (g["gn_spike"] and run == LAT and i == 100) else 1.0 + 0.1 * (i % 3)
                fh.write(json.dumps({"step": i * 100, "gn_preclip_mean100": gn}) + "\n")
                if i % 10 == 0:
                    b = i // 10; a2 = (0.5 + 0.05 * b) if g["rise"] else g["lat"]
                    fh.write(json.dumps({"bin": b, "by_d": {"1": {"acc": g["d1"]}, "2": {"acc": a2}}, "aux_vacc_by_d_pass": {"2": {"1": g["lp2"], "2": 0.5}}}) + "\n")
    p = os.path.join(d, LAT)
    json.dump({"by_d": {"2": {"intact": {"2": g["intact"], "8": g["intact"]}, "mean_at_1": {"2": g["mean1"], "8": g["mean1"]}, "k_binding": 2}}}, open(os.path.join(p, "eval_lp5.json"), "w"))
    json.dump({"by_d": {"2": {"pooled": g["ms"], "per_d": g["ms"]}}}, open(os.path.join(p, "eval_mean_s.json"), "w"))
    json.dump({"by_d": {"2": {"answer_acc": g["co"], "majority": g["maj"]}}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
    json.dump({"by_d": {"2": {"all": {"2": g["intact"] - g["cw"], "8": 0}}}}, open(os.path.join(p, "eval_cue_wrong.json"), "w"))
    return d


def fixtures():
    res = []
    o = score(_world()); res.append(("serial cue-using stub -> P-2 PASS (C1-TOLD)", o["verdict"] == "P-2 PASS (C1-TOLD)", o["verdict"]))
    o = score(_world(cw=0.05)); res.append(("cue unused -> PASS (plain C1)", o["verdict"] == "P-2 PASS (plain C1 (cue unused))", o["verdict"]))
    o = score(_world(lat=0.3)); res.append(("LAT d2 0.3 -> FAIL LP3", "LP3" in o["verdict"] and "FAIL" in o["verdict"], o["verdict"]))
    o = score(_world(mean1=0.8)); res.append(("recompute (mean pass-1 state harmless) -> FAIL LP5", "LP5" in o["verdict"], o["verdict"]))
    o = score(_world(ms=0.5)); res.append(("mean-state high -> FAIL MEAN-STATE", "MEAN-STATE" in o["verdict"], o["verdict"]))
    o = score(_world(co=0.5)); res.append(("cue alone answers -> VOID", o["verdict"].startswith("VOID"), o["verdict"]))
    o = score(_world(lp2=0.5)); res.append(("pass-1 aux 0.5 -> FAIL LP2", "LP2" in o["verdict"], o["verdict"]))
    o = score(_world(gn_spike=True)); res.append(("gn spike 40 vs cap ~2.4 -> FAIL LP0-lat", "LP0-lat" in o["verdict"], o["verdict"]))
    o = score(_world(lat=0.6, rise=True)); res.append(("rising and short -> FAIL + STILL-RISING", "STILL-RISING" in o["verdict"], o["verdict"]))
    d = _world(); os.remove(os.path.join(d, LAT, "eval_lp5.json")); o = score(d); res.append(("missing -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
