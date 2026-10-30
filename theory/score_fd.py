"""THEORY §250 FRAME-DIV Stage-1 scorer (docs/research_framediv_2026-10-30/FD_v1.md + Note 250-1; Theory is the only scorer).
Arms: fd_div_s1 / fd_div_s2 (train_frames 0-5 + 8-11) vs fd_ctrl_s1 / fd_ctrl_s2 (0-5), 8k FT of lh_fh_s1 / s2. eval_frames.json per arm (frames 6-15, d 1-4: acc, ws).
FAILING frames per model come from Stage 0 (lh_fh_s*/eval_frames.json: held-out 12-15 with d 2 acc < 0.80).
Per seed, binding:
  GEN: on >= 2 FAILING frames, DIV d 2-4 acc each >= 0.80 and >= CTRL + 0.20, and WS >= 0.50 at every pass 2..d (d 3-4);
  NOLOSS: TEST2-DF acc d 1-4 >= 0.98 (eval_acc) + the §238 gates + HOLD-LONG3 (eval_long_k128, K' 64 / 128 >= K' 8 - 0.02, d 1-4) + retention (drop > 0.05 named) [injectable];
  LP0: DIV gn_plain windows with step > 1000 <= max(2 x max(CTRL windows step > 1000), 10).
Labels (first match): LP0-FAIL / NOLOSS-FAIL (named) / GEN-PASS / MEMO (TRAIN-NEW frames 8-11 DIV d 2-4 >= 0.90 but GEN fails) / NO-GAIN.
Overall: PASS (both GEN-PASS) / SPLIT / FAIL. Descriptive: frames 6 / 7 both arms; the per-frame table.
    python -B theory/score_fd.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory")); sys.path.insert(0, _HR)
import score_st as ST  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_fd.json")
SEEDS = {"s1": ("fd_div_s1", "fd_ctrl_s1", "lh_fh_s1"), "s2": ("fd_div_s2", "fd_ctrl_s2", "lh_fh_s2")}
HELD, TRAIN_NEW, DS = ("12", "13", "14", "15"), ("8", "9", "10", "11"), ("2", "3", "4")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def frames(e):
    return (e or {}).get("frames") or {k: v for k, v in (e or {}).items() if k.isdigit()}


def lp0(runs, div, ctrl):
    def win(run):
        lg = ST.dedupe(ST.rows(os.path.join(runs, run, "log.jsonl"))) or []
        return [r.get("gn_plain_mean100", r.get("gn_preclip_mean100")) for r in lg if r.get("step", 0) > 1000 and ("gn_plain_mean100" in r or "gn_preclip_mean100" in r)]
    c, d = win(ctrl), win(div)
    if not c or not d:
        return None, "missing windows"
    cap = max(2 * max(c), 10.0); bad = [x for x in d if x is None or not np.isfinite(x) or x > cap]
    return not bad, "cap %.3g, %d over" % (cap, len(bad))


def default_noloss(runs, div, init):
    """TEST2 >= 0.98, §238 gates, HOLD-LONG3, retention (reusing score_lh); returns list of failed names, or None if pending."""
    fails = []
    a = jl(os.path.join(runs, div, "eval_acc.json")); L = jl(os.path.join(runs, div, "eval_long_k128.json"))
    if a is None or L is None:
        return None
    fails += ["TEST2 d%s" % d for d in ("1", "2", "3", "4") if a["by_d"][d]["acc"] < 0.98]
    b = L["by_d"]; fails += ["HOLD d%s" % d for d in ("1", "2", "3", "4") if any(b[d]["acc"][k] < b[d]["acc"]["8"] - 0.02 for k in ("64", "128"))]
    try:
        import score_lh as LH
        dn = LH.default_df_noloss(div, init)
        if dn.get("label") == "pending":
            return None
        fails += [k for k, v in (dn.get("gates") or {}).items() if k != "LP0" and not v]
        ret, named = LH.retention(runs, div, init, {"lh_fh_s1": "rcb_plaw4_s1", "lh_fh_s2": "rcb_plaw4_s2"}[init], LH.default_df_noloss)
        if ret is None:
            return None
        fails += named
    except Exception as ex:  # noqa: BLE001
        return None if "missing" in str(ex).lower() else fails + ["noloss-error: %s" % ex]
    return fails


def seed(runs, sd, noloss_fn):
    div, ctrl, init = SEEDS[sd]
    F0, FD, FC = frames(jl(os.path.join(runs, init, "eval_frames.json"))), frames(jl(os.path.join(runs, div, "eval_frames.json"))), frames(jl(os.path.join(runs, ctrl, "eval_frames.json")))
    if not (F0 and FD and FC):
        return {"label": "pending", "why": "missing eval_frames"}
    failing = [f for f in HELD if F0[f]["2"]["acc"] < 0.80]
    gen_frames = []
    for f in failing:
        accok = all(FD[f][d]["acc"] >= 0.80 and FD[f][d]["acc"] >= FC[f][d]["acc"] + 0.20 for d in DS)
        chain = all(FD[f][d]["ws"][str(t)] >= 0.50 for d in ("3", "4") for t in range(2, int(d) + 1))
        if accok and chain:
            gen_frames.append(f)
    gen = len(gen_frames) >= 2 and len(failing) >= 2
    memo = all(FD[f][d]["acc"] >= 0.90 for f in TRAIN_NEW for d in DS)
    lok, lmsg = lp0(runs, div, ctrl); nl = noloss_fn(runs, div, init)
    if lok is None or nl is None:
        return {"label": "pending", "why": [lmsg, "noloss pending" if nl is None else ""]}
    lab = ("LP0-FAIL" if not lok else "NOLOSS-FAIL (%s)" % ",".join(nl) if nl else "GEN-PASS" if gen else "MEMO" if memo else "NO-GAIN")
    tab = {f: {"div": {d: FD[f][d]["acc"] for d in DS}, "ctrl": {d: FC[f][d]["acc"] for d in DS}, "step0": {d: F0[f][d]["acc"] for d in DS} if f in F0 else None}
           for f in ("6", "7") + TRAIN_NEW + HELD if f in FD and f in FC}
    return {"label": lab, "failing": failing, "gen_frames": gen_frames, "memo": memo, "lp0": (lok, lmsg), "noloss": nl, "table": tab}


def score(runs=R, noloss_fn=None):
    o = {"section": 250, "registered": "FD_v1.md", "seeds": {sd: seed(runs, sd, noloss_fn or default_noloss) for sd in SEEDS}}
    labs = [o["seeds"][k]["label"] for k in SEEDS]
    if "pending" in labs:
        o["verdict"] = "pending (%s)" % {k: v.get("why") for k, v in o["seeds"].items() if v["label"] == "pending"}; return o
    n = labs.count("GEN-PASS")
    o["verdict"] = "%s | s1: %s / s2: %s" % ("FD PASS" if n == 2 else "FD SPLIT" if n == 1 else "FD FAIL", labs[0], labs[1]); return o


def fixtures():
    res = []

    def world(div_held=0.9, div_train=0.95, ctrl_held=0.3, ws=0.9, spike=False, step0_held=0.3):
        d = tempfile.mkdtemp()
        for sd, (div, ctrl, init) in SEEDS.items():
            for run, h, t, w in ((div, div_held, div_train, ws), (ctrl, ctrl_held, 0.6, 0.5), (init, step0_held, 0.6, 0.5)):
                os.makedirs(os.path.join(d, run), exist_ok=True)
                fr = {str(f): {dd: {"acc": (h if str(f) in HELD else t if str(f) in TRAIN_NEW else 0.5), "ws": {str(x): w for x in range(1, 9)}} for dd in ("1", "2", "3", "4")} for f in range(6, 16)}
                json.dump({"frameset_shas": {}, **fr}, open(os.path.join(d, run, "eval_frames.json"), "w"))
                if run in (div, ctrl):
                    with open(os.path.join(d, run, "log.jsonl"), "w") as fh:
                        for i in range(1, 81):
                            fh.write(json.dumps({"step": i * 100, "gn_plain_mean100": 30.0 if (spike and run == div and i == 40) else 1.0}) + "\n")
        return d
    ok = lambda runs, div, init: []  # noqa: E731
    o = score(world(), ok); res.append(("held-out learned -> FD PASS", o["verdict"].startswith("FD PASS"), o["verdict"]))
    o = score(world(div_held=0.4), ok); res.append(("trained-new only -> MEMO", o["verdict"].count("MEMO") == 2, o["verdict"]))
    o = score(world(div_held=0.4, div_train=0.6), ok); res.append(("nothing -> NO-GAIN", o["verdict"].count("NO-GAIN") == 2, o["verdict"]))
    o = score(world(ctrl_held=0.75), ok); res.append(("ctrl close (< +0.20) -> not GEN", "GEN-PASS" not in o["verdict"], o["verdict"]))
    o = score(world(ws=0.3), ok); res.append(("no chain -> not GEN", "GEN-PASS" not in o["verdict"], o["verdict"]))
    o = score(world(spike=True), ok); res.append(("gn spike -> LP0-FAIL", o["verdict"].count("LP0-FAIL") == 2, o["verdict"]))
    o = score(world(), lambda runs, div, init: ["TEST2 d4"]); res.append(("noloss fails -> NOLOSS-FAIL", o["verdict"].count("NOLOSS-FAIL") == 2, o["verdict"]))
    o = score(world(step0_held=0.9), ok); res.append(("no FAILING frames -> not GEN", "GEN-PASS" not in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
