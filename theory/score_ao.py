"""THEORY §244 ANSWER-ONLY scorer (docs/research_answeronly_2026-10-29/AO_v1.md; Theory is the only scorer).
AO runs ao_s1 / ao_s2 (P-2 recipe, --aux_fade 0,0) vs AUX comparators aw_plaw2_s1 / aux_s2 (descriptive SURVIVAL); CD-AN aw_pcd2_s1; DIRECT s228_pdir2c_s1.
Per AO seed, binding: LP1 (last-bin d 1 train acc >= 0.90), LP3 (d 2 >= 0.70 and >= CD + 0.30), LP5 (mean pass-1 drop >= 0.30), MEAN-STATE (<= CD + 0.10),
STATE-USE (swap_k follow_U >= 0.50, >= floor + 0.30, n_pairs >= 1000), CUE-ONLY guard (VOID), STABLE (n_nonfinite <= 10; no unrecovered loss rise).
Labels (first match): VOID / UNSTABLE / STILL-RISING / NOT-FORMED / ACC-ONLY / AUX-FREE. Overall AUX-FREE (both) / SPLIT / FAIL.
    python -B theory/score_ao.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_st as ST  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_ao.json")
SEEDS = {"s1": ("ao_s1", "aw_plaw2_s1"), "s2": ("ao_s2", "aux_s2")}
CD, DIR, D = "aw_pcd2_s1", "s228_pdir2c_s1", "2"
FINAL = False         # §245 AO-REP: the 18k reading is final (no STILL-RISING extension)


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def stable(runs, run, base=None):
    """base: for an extension dir, the 12k run whose log precedes it (Note 244-4: the combined log; the 500 steps after the resume's LR re-warm are skipped)."""
    ts = jl(os.path.join(runs, run, "train_summary.json")) or {}
    lg = (ST.dedupe(ST.rows(os.path.join(runs, base, "log.jsonl"))) or []) if base else []
    lg = ST.dedupe(lg + (ST.rows(os.path.join(runs, run, "log.jsonl")) or [])) or []
    L = [(r["step"], r["loss"]) for r in lg if "step" in r and "loss" in r]
    resume = None
    if base:
        bsteps = [r["step"] for r in (ST.dedupe(ST.rows(os.path.join(runs, base, "log.jsonl"))) or []) if "step" in r and "loss" in r]
        resume = max(bsteps) if bsteps else None
    if resume is not None:
        L = [(s, x) for s, x in L if not (resume < s <= resume + 500)]
    if not L:
        return None, "no loss rows"
    nnf = ts.get("n_nonfinite", 0) or 0
    bad = []
    for i, (s, x) in enumerate(L):
        if s <= 1000 or i < 10:
            continue
        m = float(np.median([y for _, y in L[i - 10:i]])); hi, lo = max(3 * m, 0.05), max(m, 0.05)
        if x > hi:
            nxt = [y for _, y in L[i + 1:i + 6]]
            if len(nxt) < 5:
                if L[-1][1] > hi:
                    bad.append((s, x))
            elif not any(y <= lo for y in nxt):
                bad.append((s, x))
    return (nnf <= 10 and not bad), {"n_nonfinite": nnf, "unrecovered": bad[:3]}


def seed(runs, ao, aux, base=None):
    """base: set when `ao` is a STILL-RISING extension dir (Note 244-4); the extension's evals are final, bins / loss from base + extension logs."""
    f = {(r, x): jl(os.path.join(runs, r, x + ".json")) for r in (ao,) for x in ("eval_acc", "eval_lp5", "eval_mean_s", "eval_cue_only", "eval_cue_wrong", "eval_swap_k")}
    f[(CD, "eval_acc")] = jl(os.path.join(runs, CD, "eval_acc.json")); f[(DIR, "eval_acc")] = jl(os.path.join(runs, DIR, "eval_acc.json"))
    lg = ST.rows(os.path.join(runs, ao, "log.jsonl"))
    if base and lg is not None:
        lg = (ST.rows(os.path.join(runs, base, "log.jsonl")) or []) + lg
    lg = ST.dedupe(lg)
    miss = ["%s/%s" % k for k, v in f.items() if v is None] + (["log"] if lg is None else [])
    if miss:
        return {"label": "pending", "why": miss}
    b = ST.bins(lg)
    lp1 = bool(b) and b[-1]["by_d"]["1"]["acc"] >= 0.90
    aL, aC, aD = f[(ao, "eval_acc")]["by_d"][D]["acc"], f[(CD, "eval_acc")]["by_d"][D]["acc"], f[(DIR, "eval_acc")]["by_d"][D]["acc"]
    lp3 = aL >= 0.70 and aL >= aC + 0.30
    l5 = f[(ao, "eval_lp5")]["by_d"][D]; kb = str(l5.get("k_binding", D)); lp5v = l5["intact"][kb] - l5["mean_at_1"][kb]; lp5 = lp5v >= 0.30
    ms = f[(ao, "eval_mean_s")]["by_d"][D]["pooled"]; msok = ms <= aC + 0.10
    co = f[(ao, "eval_cue_only")]["by_d"][D]; guard = co["answer_acc"] <= max(co["majority"], aD) + 0.05
    sk = f[(ao, "eval_swap_k")]; su = sk.get("n_pairs", 0) >= 1000 and sk["follow_U"] >= 0.50 and sk["follow_U"] >= sk["floor"] + 0.30
    cw = f[(ao, "eval_cue_wrong")]["by_d"][D]; cwd = l5["intact"][kb] - cw["all"][kb]
    tail = [r["by_d"][D]["acc"] for r in b if D in r["by_d"] and r["by_d"][D].get("acc") is not None][-4:]
    rising = len(tail) == 4 and float(np.polyfit(range(4), tail, 1)[0]) > 0.02
    if base:
        rising = False                                                            # Note 244-4: the extension reading is final
    sok, smsg = stable(runs, ao, base)
    o = {"LP1": lp1, "LP3": lp3, "acc_d2": {"ao": aL, "cd": aC, "dir": aD}, "LP5": lp5, "LP5_drop": lp5v, "MEAN_STATE": msok, "mean_state_acc": ms,
         "STATE_USE": su, "swap_k": {k: sk.get(k) for k in ("n_pairs", "follow_U", "floor")}, "CUE_ONLY_ok": guard, "CUE_WRONG_drop": cwd, "STABLE": (sok, smsg), "still_rising": rising}
    ref = {x: jl(os.path.join(runs, aux, x + ".json")) for x in ("eval_acc", "eval_lp5", "eval_swap_k")}
    if all(v is not None for v in ref.values()):
        r5 = ref["eval_lp5"]["by_d"][D]; rk = str(r5.get("k_binding", D))
        o["survival"] = {"acc": aL / max(ref["eval_acc"]["by_d"][D]["acc"], 1e-9), "lp5": lp5v / max(r5["intact"][rk] - r5["mean_at_1"][rk], 1e-9),
                         "swap": sk["follow_U"] / max(ref["eval_swap_k"]["follow_U"], 1e-9)}
    if not guard:
        o["label"] = "VOID"
    elif sok is False:
        o["label"] = "UNSTABLE"
    elif not lp3 and rising and not FINAL:
        o["label"] = "STILL-RISING"
    elif not lp3 or not lp1:
        o["label"] = "NOT-FORMED"
    elif not (lp5 and msok and su):
        o["label"] = "ACC-ONLY (%s)" % ",".join(k for k, v in (("LP5", lp5), ("MEAN-STATE", msok), ("STATE-USE", su)) if not v)
    else:
        o["label"] = "AUX-FREE (%s)" % ("C1-TOLD" if cwd >= 0.20 else "plain")
    return o


def score(runs=R):
    seeds = {}
    for k, (ao, aux) in SEEDS.items():           # Note 244-4: a STILL-RISING seed's final reading is its <ao>_ext dir (18k), logs combined
        ext = ao + "_ext"
        seeds[k] = seed(runs, ext, aux, base=ao) if os.path.isdir(os.path.join(runs, ext)) else seed(runs, ao, aux)
        if os.path.isdir(os.path.join(runs, ext)):
            seeds[k]["final_from"] = ext; seeds[k]["survival_caveat"] = "18k extension (LR re-warm) vs the 12k comparator; not step-matched"
    o = {"section": 244, "registered": "AO_v1.md", "seeds": seeds}
    labs = [o["seeds"][k]["label"] for k in SEEDS]
    if "pending" in labs or "STILL-RISING" in labs:
        o["verdict"] = "pending (%s)" % " / ".join("%s: %s" % (k, v["label"]) for k, v in o["seeds"].items()); return o
    n = sum(x.startswith("AUX-FREE") for x in labs)
    o["verdict"] = "%s | s1: %s / s2: %s" % ("AO AUX-FREE" if n == 2 else "AO SPLIT" if n == 1 else "AO FAIL", labs[0], labs[1]); return o


def fixtures():
    res = []

    def world(ao=0.9, intact=0.9, mean1=0.2, ms=0.12, co=0.1, follow=0.8, d1=0.99, rise=False, spike=None, nnf=0, per=None):
        d = tempfile.mkdtemp()
        for run in (CD, DIR):
            os.makedirs(os.path.join(d, run)); json.dump({"by_d": {D: {"acc": 0.1}}}, open(os.path.join(d, run, "eval_acc.json"), "w"))
        for sd, (aorun, aux) in SEEDS.items():
            g = dict(ao=ao, intact=intact, mean1=mean1, ms=ms, co=co, follow=follow, d1=d1, rise=rise, spike=spike, nnf=nnf); g.update((per or {}).get(sd, {}))
            for run, acc_ in ((aorun, g["ao"]), (aux, 1.0)):
                p = os.path.join(d, run); os.makedirs(p)
                json.dump({"by_d": {D: {"acc": acc_}}}, open(os.path.join(p, "eval_acc.json"), "w"))
                json.dump({"by_d": {D: {"intact": {D: g["intact"]}, "mean_at_1": {D: g["mean1"]}, "k_binding": 2}}}, open(os.path.join(p, "eval_lp5.json"), "w"))
                json.dump({"n_pairs": 1000, "follow_U": g["follow"], "floor": 0.2}, open(os.path.join(p, "eval_swap_k.json"), "w"))
            p = os.path.join(d, aorun)
            json.dump({"by_d": {D: {"pooled": g["ms"]}}}, open(os.path.join(p, "eval_mean_s.json"), "w"))
            json.dump({"by_d": {D: {"answer_acc": g["co"], "majority": 0.1}}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
            json.dump({"by_d": {D: {"all": {D: 0.4}}}}, open(os.path.join(p, "eval_cue_wrong.json"), "w"))
            json.dump({"n_nonfinite": g["nnf"]}, open(os.path.join(p, "train_summary.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 121):
                    x = 0.01 if i > 30 else 2.0
                    if g["spike"] and i == g["spike"]:
                        x = 5.0
                    if g["spike"] and g["spike"] < i <= g["spike"] + 10:
                        x = 3.0
                    fh.write(json.dumps({"step": i * 100, "loss": x, "gn_preclip_mean100": 1.0}) + "\n")
                for k in range(12):
                    a2 = (0.2 + 0.05 * k) if g["rise"] else g["ao"]
                    fh.write(json.dumps({"bin": k, "by_d": {"1": {"acc": g["d1"]}, D: {"acc": a2}}}) + "\n")
        return d
    o = score(world()); res.append(("chain forms -> AUX-FREE", o["verdict"].startswith("AO AUX-FREE"), o["verdict"]))
    o = score(world(ao=0.3)); res.append(("no learning -> NOT-FORMED", o["verdict"].count("NOT-FORMED") == 2, o["verdict"]))
    o = score(world(mean1=0.85)); res.append(("answers w/o pass-1 use -> ACC-ONLY", o["verdict"].count("ACC-ONLY") == 2, o["verdict"]))
    o = score(world(co=0.5)); res.append(("cue alone -> VOID", o["verdict"].count("VOID") == 2, o["verdict"]))
    o = score(world(spike=60)); res.append(("unrecovered loss rise -> UNSTABLE", o["verdict"].count("UNSTABLE") == 2, o["verdict"]))
    o = score(world(nnf=50)); res.append(("many non-finite -> UNSTABLE", o["verdict"].count("UNSTABLE") == 2, o["verdict"]))
    o = score(world(ao=0.5, rise=True)); res.append(("still rising -> pending (extension)", o["verdict"].startswith("pending") and "STILL-RISING" in o["verdict"], o["verdict"]))
    o = score(world(per={"s2": {"ao": 0.3}})); res.append(("one seed -> SPLIT", o["verdict"].startswith("AO SPLIT"), o["verdict"]))
    d = world(ao=0.5, rise=True); import shutil
    shutil.copytree(os.path.join(d, "ao_s2"), os.path.join(d, "ao_s2_ext"))
    json.dump({"by_d": {D: {"acc": 0.85}}}, open(os.path.join(d, "ao_s2_ext", "eval_acc.json"), "w"))
    o = score(d); res.append(("s2 extension final at 18k (no second STILL-RISING)", o["seeds"]["s2"].get("final_from") == "ao_s2_ext" and o["seeds"]["s2"]["label"] != "STILL-RISING", o["seeds"]["s2"]["label"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--rep", action="store_true", help="§245 AO-REP: ao_s3 / ao_s4 at a straight 18k, final (no STILL-RISING extension); comparator aw_plaw2_s1 (descriptive)")
    a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    out = OUT
    if a.rep:
        SEEDS = {"s3": ("ao_s3", "aw_plaw2_s1"), "s4": ("ao_s4", "aw_plaw2_s1")}; FINAL = True; out = OUT.replace(".json", "_rep.json")
    o = score(); o["section"] = 245 if a.rep else 244
    if a.rep and "seeds" in o:
        o["formed"] = sum(v.get("label", "").startswith("AUX-FREE") for v in o["seeds"].values())
        o["verdict"] = "AO-REP %d/2 formed | %s" % (o["formed"], " / ".join("%s: %s" % (k, v.get("label")) for k, v in o["seeds"].items())) if not o["verdict"].startswith("pending") else o["verdict"]
    print(o["verdict"])
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(o, open(out, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
