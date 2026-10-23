"""THEORY §224 FIXED-HOLD scorer, Stage 2 (FH_v1.md at 232e84ab §4 + Notes 224-1..5; written before any Stage-2 output; Theory is the only scorer).

Cells (boxes/runs_fsq): FH s224_fh_s{1,2[,3]}, RD s224_rd_s{1,2[,3]}; plain controls = 3090 re-evals of D3-T s1 / s2 in s224_base_s1_re /
s224_base_s2_re (eval files only). Per cell: eval_khorizon.json (acc.intact[K'][n], K' 24..128), eval_vb.json (acc_n2, acc_k, stall.recovery,
probe["4"].acc_by_n), card; trained cells log.jsonl (step, cur_stage, fh_r_far) + args.json (steps); K10 drift file
docs/research_fixedhold_2026-10-21/fh_check_cpu.<run>.json (bases: s213_d3t_s1 / s216_d3t_s2). One 3090 class for all eval files (else VOID).
H(K') = mean n 1..8 of acc.intact[K'][n]. Per (arm A, seed s), vs the same-seed base (s3: the mean of the s1 / s2 bases, disclosed):
  HOLD: H >= 0.95 at every K' in {24, 32, 48, 64, 96, 128}.
  NOLOSS: acc_n2 n 1..8 >= 0.99; stall >= base - 0.02; PROBE(4) mean n 7..8 <= base + 0.05; early-finish mean_{n 2..8} acc_k[n-1][n] <= base + 0.05;
          C1 floor acc_k['0'][n] n 1..16 and acc_k['1'][n] n 2..16 each <= base + 0.05.
  DEPTH: mean acc_n2 n 9..11 >= base - 0.10 (WARN if the seed mean of (A - base) < -0.05).
  r_eval = 2 * mean step_ang_v over t 24..64, all items, n {1,4,8} (224-5); r_log = mean of the last 10 logged fh_r_far.
Verdicts (each "both seeds" criterion = every run seed; with s3 run after a split, >= 2 of 3 seeds, labelled "(2/3)"):
  FH-RECIPE: FH HOLD & NOLOSS & DEPTH.  RD SUFFICES: RD HOLD & NOLOSS & DEPTH.
  UNEXPOSED: r_log(FH) > r_log(RD) / 3 in some seed -> no PENALTY / UNDERDOSED reading.
  PENALTY NEEDED: FH-RECIPE & H_FH(128) - H_RD(128) >= 0.15 & r_eval(FH) <= r_eval(RD) / 3 (not UNEXPOSED).
  PARTIAL: H_FH(128) - H_base(128) >= 0.30 in every seed without FH HOLD.
  UNDERDOSED: not UNEXPOSED & FH fails HOLD (some seed) & r_eval(FH) >= r_eval(RD) / 1.5 & FH NOLOSS.
  R-MECH (supporting): r_eval(FH) <= r_eval(base) / 7.3. CONFOUNDED flag: |CurGate advance step - base's| > 2,000 at any advance.
  SPLIT: s1 and s2 differ on FH-RECIPE, RD SUFFICES or PENALTY NEEDED -> s3 required (pending until it lands).
Descriptive: drift_ang_v at t 64 (mean) and its slope over t 24..64 per model (224-5).
    python -B theory/score_s224_s2.py [--final] [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
K10 = os.path.join(_HR, "docs", "research_fixedhold_2026-10-21")
OUT = os.path.join(_HR, "theory", "results", "score_s224_s2.json")
KS = [24, 32, 48, 64, 96, 128]
BASE_RUN = {"s1": "s213_d3t_s1", "s2": "s216_d3t_s2"}
STEPS2 = 48000


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


def drift(k10dir, run):
    d = jl(os.path.join(k10dir, "fh_check_cpu.%s.json" % run))
    if d is None:
        return None
    st, dr = [], {t: [] for t in range(24, 65)}
    for n in ("1", "4", "8"):
        for t in range(24, 65):
            st += [v for v in d[n]["step_ang_v"][t] if v is not None]; dr[t] += [v for v in d[n]["drift_ang_v"][t] if v is not None]
    ts = [t for t in dr if dr[t]]; m = [float(np.mean(dr[t])) for t in ts]
    return {"r_eval": 2 * float(np.mean(st)), "drift64": m[-1] if m else None, "drift_slope": float(np.polyfit(ts, m, 1)[0]) if len(ts) > 1 else None}


def advances(lg):
    out, prev = [], None
    for r in lg or []:
        c = r.get("cur_stage")
        if c is not None and prev is not None and c != prev:
            out.append(r["step"])
        if c is not None:
            prev = c
    return out


def cell(runs, k10dir, name, trained, drift_run=None):
    d = os.path.join(runs, name); kh, vb = jl(os.path.join(d, "eval_khorizon.json")), jl(os.path.join(d, "eval_vb.json"))
    o = {"name": name, "missing": [f for f, v in (("eval_khorizon.json", kh), ("eval_vb.json", vb)) if v is None], "cards": []}
    if trained:
        lg, ar = rows(os.path.join(d, "log.jsonl")), jl(os.path.join(d, "args.json"))
        if lg is None or ar is None:
            o["missing"] += [f for f, v in (("log.jsonl", lg), ("args.json", ar)) if v is None]
        else:
            o["dead"] = max((r.get("step", 0) for r in lg), default=0) < int(ar.get("steps", STEPS2))
            rr = [r["fh_r_far"] for r in lg if r.get("fh_r_far") is not None]; o["r_log"] = float(np.mean(rr[-10:])) if rr else None
            o["adv"] = advances(lg)
    dk = drift(k10dir, drift_run or name)
    if dk is None:
        o["missing"].append("K10 fh_check_cpu")
    else:
        o.update(dk)
    if kh is not None:
        ks = [k for k in KS if str(k) in kh["acc"]["intact"]]
        o["H"] = {k: float(np.mean([kh["acc"]["intact"][str(k)][str(n)] for n in range(1, 9)])) for k in ks}; o["cards"].append(kh.get("card"))
        if 128 not in o["H"]:
            o["missing"].append("K'128")
    if vb is not None:
        a = vb["acc_n2"]; ak = vb["acc_k"]
        o.update({"acc18": min(a[str(n)] for n in range(1, 9)), "depth": float(np.mean([a[str(n)] for n in (9, 10, 11)])), "stall": vb["stall"]["recovery"],
                  "probe4": float(np.mean([vb["probe"]["4"]["acc_by_n"][str(n)] for n in (7, 8)])),
                  "early": float(np.mean([ak[str(n - 1)][str(n)] for n in range(2, 9)])),
                  "k0": {n: ak["0"][str(n)] for n in range(1, 17)}, "k1": {n: ak["1"][str(n)] for n in range(2, 17)}})
        o["cards"].append(vb.get("card"))
    return o


def base_mean(b1, b2):
    o = {"name": "mean(base s1, s2)", "missing": b1["missing"] + b2["missing"], "cards": []}
    for k in ("depth", "stall", "probe4", "early", "r_eval"):
        o[k] = (b1[k] + b2[k]) / 2
    o["k0"] = {n: (b1["k0"][n] + b2["k0"][n]) / 2 for n in b1["k0"]}; o["k1"] = {n: (b1["k1"][n] + b2["k1"][n]) / 2 for n in b1["k1"]}
    o["H"] = {k: (b1["H"][k] + b2["H"][k]) / 2 for k in b1["H"]}; o["adv"] = None
    return o


def judge(A, B):
    hold = all(A["H"].get(k, 0) >= 0.95 for k in KS)
    c1 = all(A["k0"][n] <= B["k0"][n] + 0.05 for n in A["k0"]) and all(A["k1"][n] <= B["k1"][n] + 0.05 for n in A["k1"])
    nol = A["acc18"] >= 0.99 and A["stall"] >= B["stall"] - 0.02 and A["probe4"] <= B["probe4"] + 0.05 and A["early"] <= B["early"] + 0.05 and c1
    dep = A["depth"] >= B["depth"] - 0.10
    conf = bool(A.get("adv") and B.get("adv") and (len(A["adv"]) != len(B["adv"]) or any(abs(x - y) > 2000 for x, y in zip(A["adv"], B["adv"]))))
    return {"HOLD": hold, "NOLOSS": nol, "C1_floor": c1, "DEPTH": dep, "depth_delta": A["depth"] - B["depth"], "OK": hold and nol and dep,
            "hold_gain": A["H"][128] - B["H"][128], "R_MECH": A["r_eval"] <= B["r_eval"] / 7.3, "CONFOUNDED": conf}


def score(runs=R, k10dir=K10, final=False):
    out = {"section": 224, "stage": 2, "registered": "FH_v1.md @ 232e84ab + 224-1..5", "seeds": {}}
    base = {s: cell(runs, k10dir, "s224_base_%s_re" % s, False, drift_run=BASE_RUN[s]) for s in ("s1", "s2")}
    for s in base:                                                          # CurGate advances of the plain base = the original D3-T run's log
        base[s]["adv"] = advances(rows(os.path.join(runs, BASE_RUN[s], "log.jsonl")))
    seeds = ["s1", "s2"] + (["s3"] if os.path.isdir(os.path.join(runs, "s224_fh_s3")) or os.path.isdir(os.path.join(runs, "s224_rd_s3")) else [])
    cards = [x for b in base.values() for x in b["cards"]]; per = {}
    for s in seeds:
        fh, rd = cell(runs, k10dir, "s224_fh_%s" % s, True), cell(runs, k10dir, "s224_rd_%s" % s, True)
        B = base[s] if s in base else base_mean(base["s1"], base["s2"])
        cards += fh["cards"] + rd["cards"]
        miss = [(c["name"], m) for c in (fh, rd, B) for m in c["missing"]]
        if miss:
            per[s] = {"status": ("FAIL (rule b: missing %s)" if final else "pending (missing %s)") % miss[:6]}; continue
        dead = [c["name"] for c in (fh, rd) if c.get("dead")]
        jf, jr = (judge(fh, B) if not fh.get("dead") else None), (judge(rd, B) if not rd.get("dead") else None)
        unexp = not (fh.get("r_log") is not None and rd.get("r_log") and fh["r_log"] <= rd["r_log"] / 3)
        r = {"status": "SCORED" + (" (dead: %s)" % dead if dead else ""), "fh": jf, "rd": jr, "UNEXPOSED": unexp,
             "fh_minus_rd_128": (fh["H"][128] - rd["H"][128]) if jf and jr else None,
             "r_eval": {"fh": fh["r_eval"], "rd": rd["r_eval"], "base": B["r_eval"]}, "r_log": {"fh": fh.get("r_log"), "rd": rd.get("r_log")},
             "drift": {k: {"drift64": c.get("drift64"), "slope": c.get("drift_slope")} for k, c in (("fh", fh), ("rd", rd))}, "H": {"fh": fh["H"], "rd": rd["H"], "base": B["H"]}}
        r["FH_RECIPE"] = bool(jf and jf["OK"]); r["RD_SUFFICES"] = bool(jr and jr["OK"])
        r["PENALTY"] = bool(r["FH_RECIPE"] and jr and not unexp and r["fh_minus_rd_128"] >= 0.15 and fh["r_eval"] <= rd["r_eval"] / 3)
        r["PARTIAL_seed"] = bool(jf and jf["hold_gain"] >= 0.30 and not jf["HOLD"])
        r["UNDERDOSED_seed"] = bool(jf and not unexp and not jf["HOLD"] and jr and fh["r_eval"] >= rd["r_eval"] / 1.5 and jf["NOLOSS"])
        per[s] = r
    out["seeds"] = per; cls = sorted({str(c) for c in cards if c})
    out["card_classes"] = cls
    if cls and (len(cls) > 1 or "3090" not in cls[0]):
        out["verdict"] = "VOID (card %s)" % cls; return out
    if any(per[s]["status"].startswith("pending") for s in per):
        out["verdict"] = "pending"; return out
    sc = [s for s in per if per[s]["status"].startswith("SCORED")]
    keys = ("FH_RECIPE", "RD_SUFFICES", "PENALTY")
    if any(per[s]["UNEXPOSED"] for s in sc):
        keys = ("FH_RECIPE", "RD_SUFFICES")                                 # an unexposed penalty is not read, so it cannot split
    if "s3" not in per and all(s in sc for s in ("s1", "s2")) and any(per["s1"][k] != per["s2"][k] for k in keys):
        out["verdict"] = "SPLIT (s3 required)"; return out
    need = len(per) if len(per) == 2 else 2; tag = "" if len(per) == 2 else " (2/3)"
    cnt = lambda k: sum(1 for s in sc if per[s].get(k))  # noqa: E731
    fails = [s for s in per if s not in sc]
    lab = []
    if cnt("FH_RECIPE") >= need and not (len(per) == 2 and fails):
        lab.append("FH-RECIPE" + tag)
    if cnt("RD_SUFFICES") >= need and not (len(per) == 2 and fails):
        lab.append("RD SUFFICES" + tag)
    if any(per[s]["UNEXPOSED"] for s in sc):
        lab.append("UNEXPOSED")
    else:
        if cnt("PENALTY") >= need:
            lab.append("PENALTY NEEDED" + tag)
        if any(per[s]["UNDERDOSED_seed"] for s in sc) and "FH-RECIPE" not in " ".join(lab):
            lab.append("UNDERDOSED")
    if cnt("PARTIAL_seed") >= need and "FH-RECIPE" not in " ".join(lab):
        lab.append("PARTIAL" + tag)
    dd = [per[s]["fh"]["depth_delta"] for s in sc if per[s].get("fh")]
    if dd and float(np.mean(dd)) < -0.05:
        lab.append("DEPTH-WARN")
    out["verdict"] = " | ".join(lab) if lab else "NONE (no recipe / partial label)"
    return out


# ------------------------------------------------------------------ fixtures
def _cell(d, k10, name, H128, trained=True, acc18=1.0, depth=0.6, stall=1.0, probe4=0.12, early=0.2, k0=0.2, k1=0.3, card="NVIDIA GeForce RTX 3090",
          r_log=1e-3, step=5e-4, last=48000, adv=(6000, 9000), drift_run=None):
    p = os.path.join(d, name); os.makedirs(p, exist_ok=True)
    acc = {"intact": {str(k): {str(n): (1.0 if k <= 64 else H128) for n in range(1, 9)} for k in KS}}
    json.dump({"acc": acc, "card": card}, open(os.path.join(p, "eval_khorizon.json"), "w"))
    ak = {str(K): {str(n): 0.5 for n in range(1, 17)} for K in range(0, 17)}
    for n in range(1, 17):
        ak["0"][str(n)] = k0; ak["1"][str(n)] = k1
    for n in range(2, 9):
        ak[str(n - 1)][str(n)] = early
    json.dump({"acc_n2": {str(n): (acc18 if n <= 8 else depth) for n in range(1, 17)}, "acc_k": ak, "stall": {"recovery": stall},
               "probe": {"4": {"acc_by_n": {str(n): probe4 for n in range(1, 17)}}}, "card": card}, open(os.path.join(p, "eval_vb.json"), "w"))
    if trained:
        json.dump({"steps": 48000}, open(os.path.join(p, "args.json"), "w"))
        with open(os.path.join(p, "log.jsonl"), "w") as fh:
            for s in range(100, last + 1, 100):
                fh.write(json.dumps({"step": s, "cur_stage": sum(s >= a for a in adv), "fh_r_far": r_log}) + "\n")
    blk = {n: {"step_ang_v": [[None] * 4] + [[step / 2] * 4 for _ in range(64)], "drift_ang_v": [[None] * 4] + [[step * t] * 4 for t in range(1, 65)]} for n in ("1", "4", "8")}
    json.dump(blk, open(os.path.join(k10, "fh_check_cpu.%s.json" % (drift_run or name)), "w"))


def _world(over=None):
    over = over or {}; d, k10 = tempfile.mkdtemp(), tempfile.mkdtemp()
    for s in ("s1", "s2"):
        kw = dict(over.get(("base", s), {})); h = kw.pop("H128", 0.15)
        _cell(d, k10, "s224_base_%s_re" % s, h, trained=False, step=5e-4, drift_run=BASE_RUN[s], **kw)
        os.makedirs(os.path.join(d, BASE_RUN[s]), exist_ok=True)
        with open(os.path.join(d, BASE_RUN[s], "log.jsonl"), "w") as fh:
            for st in range(100, 48001, 100):
                fh.write(json.dumps({"step": st, "cur_stage": sum(st >= a for a in (6000, 9000))}) + "\n")
    for s in over.get("seeds", ("s1", "s2")):
        for arm, dflt in (("fh", dict(H128=0.99, r_log=1e-4, step=5e-5)), ("rd", dict(H128=0.7, r_log=1e-3, step=3e-4))):
            kw = dict(dflt); kw.update(over.get((arm, s), {})); h = kw.pop("H128")
            _cell(d, k10, "s224_%s_%s" % (arm, s), h, **kw)
    return d, k10


def fixtures():
    res = []
    def run(o_):
        d, k = _world(o_); return score(d, k)
    o = run({}); res.append(("FH holds, RD not, r 1/6 -> FH-RECIPE + PENALTY NEEDED", o["verdict"] == "FH-RECIPE | PENALTY NEEDED", o["verdict"]))
    o = run({("rd", "s1"): {"H128": 0.97}, ("rd", "s2"): {"H128": 0.97}}); res.append(("RD holds too -> FH-RECIPE + RD SUFFICES, no PENALTY", o["verdict"] == "FH-RECIPE | RD SUFFICES", o["verdict"]))
    o = run({("fh", "s1"): {"r_log": 5e-4}}); res.append(("FH log r only 2x below RD -> UNEXPOSED", "UNEXPOSED" in o["verdict"] and "PENALTY" not in o["verdict"], o["verdict"]))
    o = run({("fh", "s1"): {"step": 2e-4}, ("fh", "s2"): {"step": 2e-4}}); res.append(("FH holds but eval r 0.67x RD -> FH-RECIPE without PENALTY", o["verdict"] == "FH-RECIPE", o["verdict"]))
    o = run({("fh", "s1"): {"H128": 0.6, "step": 2.5e-4}, ("fh", "s2"): {"H128": 0.6, "step": 2.5e-4}}); res.append(("FH fails hold, r >= RD/1.5 -> PARTIAL + UNDERDOSED", "UNDERDOSED" in o["verdict"] and "PARTIAL" in o["verdict"], o["verdict"]))
    o = run({("fh", "s2"): {"H128": 0.8}}); res.append(("s2 FH fails hold -> SPLIT (s3 required)", o["verdict"].startswith("SPLIT"), o["verdict"]))
    o = run({("fh", "s1"): {"depth": 0.45}, ("fh", "s2"): {"depth": 0.45}}); res.append(("FH depth -0.15 -> no FH-RECIPE, DEPTH-WARN", "FH-RECIPE" not in o["verdict"] and "DEPTH-WARN" in o["verdict"], o["verdict"]))
    o = run({("fh", "s1"): {"stall": 0.95}, ("fh", "s2"): {"stall": 0.95}}); res.append(("FH stall -0.05 -> no FH-RECIPE", "FH-RECIPE" not in o["verdict"], o["verdict"]))
    o = run({("fh", "s1"): {"k0": 0.3}, ("fh", "s2"): {"k0": 0.3}}); res.append(("C1 floor broken (K'0 +0.10) -> no FH-RECIPE", "FH-RECIPE" not in o["verdict"], o["verdict"]))
    o = run({("fh", "s1"): {"early": 0.3}, ("fh", "s2"): {"early": 0.3}}); res.append(("early-finish +0.10 -> no FH-RECIPE", "FH-RECIPE" not in o["verdict"], o["verdict"]))
    o = run({("fh", "s1"): {"adv": (6000, 12000)}}); res.append(("advance 3k later -> CONFOUNDED flag", o["seeds"]["s1"]["fh"]["CONFOUNDED"] is True, o["seeds"]["s1"]["fh"]["CONFOUNDED"]))
    o = run({("fh", "s1"): {"card": "NVIDIA GeForce RTX 4090"}}); res.append(("other card -> VOID", o["verdict"].startswith("VOID"), o["verdict"]))
    d, k = _world(); os.remove(os.path.join(d, "s224_rd_s2", "eval_vb.json")); o = score(d, k); res.append(("missing -> pending", o["verdict"] == "pending", o["verdict"]))
    o = run({"seeds": ("s1", "s2", "s3"), ("fh", "s2"): {"H128": 0.8}}); res.append(("s3 run, FH holds 2/3 -> FH-RECIPE (2/3)", "FH-RECIPE (2/3)" in o["verdict"], o["verdict"]))
    o = run({("fh", "s1"): {"last": 30000}}); res.append(("dead FH s1 -> no FH-RECIPE", "FH-RECIPE" not in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(final=a.final); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
