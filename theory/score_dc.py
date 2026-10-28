"""THEORY §243 DEPTH-CURRIC scorer (docs/research_depthcurric_2026-10-28/DC_v1.md; Theory is the only scorer). C2: does a wider depth range extrapolate?
Arms: dc_s1 / dc_s2 (d <= 6) vs dc_ctrl_s1 / dc_ctrl_s2 (d <= 4), think_k 10, from lh_fh_s1 / s2. eval_long_dc.json (K' 8/10/16/64/128, d 1..12) and
eval_ws_long.json (WS per pass, d 1..10) per arm; ops joined from TEST2-DF regen by sig.
Per seed: UNINFORMATIVE (DC d 5-6 at K' 8 < 0.90); H-ARCH (|DC(6+k) - CTRL(4+k)| <= 0.10, k 1..3, at K' 16); EXTRAP (DC(8) - CTRL(6) >= 0.20 and DC(9) >= 0.24 at K' 16);
STEP-READ (regular-op WS, DC passes 7-8 of d 8 minus CTRL passes 5-6 of d 6, >= 0.15); C2-RISE (DC d 12: K'16 - K'10 >= 0.10 and K'16 >= 0.25);
NOLOSS (HOLD-LONG3 d 1-6 at K' 64/128; §238 gates; retention, via injectable checks); LP0 (skip steps <= 1000; cap from Note 243-x, LP0_CAP);
NORM GUARD per d <= 4 (DC / CTRL > 1.25 -> GAMED). Labels first match: GAMED / LP0-FAIL / NOLOSS-FAIL / UNINFORMATIVE / H-ARCH / PASS / EXTRAP / IN-RANGE-ONLY.
    python -B theory/score_dc.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory")); sys.path.insert(0, _HR)
import score_st as ST  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_dc.json")
SEEDS = {"s1": ("dc_s1", "dc_ctrl_s1", "lh_fh_s1"), "s2": ("dc_s2", "dc_ctrl_s2", "lh_fh_s2")}
LP0_CAP = {}          # filled from the pilot Note before launch (DC_v1 sec. 2); empty -> pending


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def acc(t, d, k):
    return t[str(d)]["acc"][str(k)]


def is_backref(it, t):
    return it["ops"][t - 1] in ("add_s", "mul_s") and it["src_j"][t - 1] is not None and it["src_j"][t - 1] <= t - 3


def reg_ws(ws, d, passes, regen):
    """mean WS over the given passes on regular (non-back-reference) steps at depth d."""
    it = ws["items"][str(d)]; V = np.array(it["v"]); okk = []
    for t in passes:
        pr = np.array(it["pred_by_pass"][str(t)])
        for i, sg in enumerate(it["sig"]):
            if sg in regen and not is_backref(regen[sg], t):
                okk.append(pr[i] == V[i, t - 1])
    return float(np.mean(okk)) if okk else None


def lp0(runs, run, cap):
    lg = ST.dedupe(ST.rows(os.path.join(runs, run, "log.jsonl"))) or []
    g = [(r["step"], r.get("gn_plain_mean100", r.get("gn_preclip_mean100"))) for r in lg if "step" in r and ("gn_plain_mean100" in r or "gn_preclip_mean100" in r)]
    bad = [(s, x) for s, x in g if s > 1000 and (x is None or not np.isfinite(x) or x > cap)]
    return (not bad) if g else None, bad[:3]


def norm_guard_d(runs, dc, ctrl):
    def bins(run):
        lg = ST.dedupe(ST.rows(os.path.join(runs, run, "log.jsonl"))) or []
        return {r["bin"]: (r.get("hold") or {}).get("norm_guard_by_d") for r in lg if "bin" in r and isinstance((r.get("hold") or {}).get("norm_guard_by_d"), dict)}
    a, b = bins(dc), bins(ctrl); common = sorted(set(a) & set(b))
    if not common:
        return None, "no paired per-d bins"
    rat = [a[i][d][k] / b[i][d][k] for i in common for d in ("1", "2", "3", "4") if d in a[i] and d in b[i] for k in ("s8", "sd2")]
    return all(x <= 1.25 for x in rat), {"max_ratio": max(rat) if rat else None}


def seed(runs, sd, regen, noloss_fn):
    dc, ctrl, init = SEEDS[sd]
    L = {r: (jl(os.path.join(runs, r, "eval_long_dc.json")) or {}).get("by_d") for r in (dc, ctrl, init)}
    W = {r: jl(os.path.join(runs, r, "eval_ws_long.json")) for r in (dc, ctrl)}
    if any(v is None for v in list(L.values()) + list(W.values())):
        return {"label": "pending", "why": "missing eval_long_dc / eval_ws_long"}
    if sd not in LP0_CAP:
        return {"label": "pending", "why": "LP0 cap not yet fixed (pilot Note)"}
    o = {}
    gok, gmsg = norm_guard_d(runs, dc, ctrl); lok, lmsg = lp0(runs, dc, LP0_CAP[sd])
    nl = noloss_fn(runs, dc, init, L[dc])
    if gok is None or lok is None or nl is None:
        return {"label": "pending", "why": [gmsg, lmsg, "noloss"]}
    D, C = L[dc], L[ctrl]
    inform = min(acc(D, 5, 8), acc(D, 6, 8)) >= 0.90
    harch = all(abs(acc(D, 6 + k, 16) - acc(C, 4 + k, 16)) <= 0.10 for k in (1, 2, 3))
    extrap = acc(D, 8, 16) - acc(C, 6, 16) >= 0.20 and acc(D, 9, 16) >= 0.24
    sr_dc, sr_c = reg_ws(W[dc], 8, (7, 8), regen), reg_ws(W[ctrl], 6, (5, 6), regen)
    sread = sr_dc is not None and sr_c is not None and sr_dc - sr_c >= 0.15
    rise = acc(D, 12, 16) - acc(D, 12, 10) >= 0.10 and acc(D, 12, 16) >= 0.25
    o.update({"norm_guard": (gok, gmsg), "lp0": (lok, lmsg), "noloss": nl, "informative": inform, "H_ARCH": harch, "EXTRAP": extrap, "STEP_READ": sread,
              "step_read": (sr_dc, sr_c), "C2_RISE": rise, "f_dc": {k: acc(D, 6 + k, 16) for k in (1, 2, 3, 4)}, "f_ctrl": {k: acc(C, 4 + k, 16) for k in (1, 2, 3, 4)}})
    o["label"] = ("GAMED" if not gok else "LP0-FAIL" if not lok else "NOLOSS-FAIL (%s)" % ",".join(nl) if nl else "UNINFORMATIVE" if not inform
                  else "H-ARCH" if harch else "PASS" if (extrap and sread and rise) else "EXTRAP" if (extrap and sread) else "IN-RANGE-ONLY")
    return o


def default_noloss(runs, dc, init, D):
    """HOLD-LONG3 at d 1-6 here; the §238 gates and retention via score_lh's machinery. Returns a list of failed names ([] = pass), or None if pending."""
    fails = [("HOLD d%s" % d) for d in "123456" if any(D[d]["acc"][k] < D[d]["acc"]["8"] - 0.02 for k in ("64", "128"))]
    try:
        import score_lh as LH
        dn = LH.default_df_noloss(dc, init)
        if dn.get("label") == "pending":
            return None
        fails += [k for k, v in (dn.get("gates") or {}).items() if k != "LP0" and not v]
        ret, named = LH.retention(runs, dc, init, {"lh_fh_s1": "rcb_plaw4_s1", "lh_fh_s2": "rcb_plaw4_s2"}[init], LH.default_df_noloss)
        if ret is None:
            return None
        fails += named
    except Exception as ex:  # noqa: BLE001
        return None if "missing" in str(ex).lower() else fails + ["noloss-error: %s" % ex]
    return fails


def score(runs=R, regen=None, noloss_fn=None):
    if regen is None:
        import score_df as DF
        sets = DF.regen(); regen = sets[0] if sets else None
    o = {"section": 243, "registered": "DC_v1.md", "seeds": {sd: seed(runs, sd, regen or {}, noloss_fn or default_noloss) for sd in SEEDS}}
    labs = [o["seeds"][k]["label"] for k in SEEDS]
    if "pending" in labs:
        o["verdict"] = "pending (%s)" % {k: v.get("why") for k, v in o["seeds"].items() if v["label"] == "pending"}; return o
    v = ("DC PASS" if labs == ["PASS", "PASS"] else "DC EXTRAP" if all(x in ("PASS", "EXTRAP") for x in labs) else
         "DC H-ARCH" if labs == ["H-ARCH", "H-ARCH"] else "DC SPLIT" if any(x in ("PASS", "EXTRAP") for x in labs) else "DC FAIL")
    o["verdict"] = "%s | s1: %s / s2: %s" % (v, labs[0], labs[1]); return o


def fixtures():
    res = []; rng = np.random.default_rng(0)
    global LP0_CAP

    def world(dc_curve, ctrl_curve, inform=True, gn_spike=False, gamed=False, ws_dc=0.9, ws_c=0.6):
        d = tempfile.mkdtemp(); regen = {}
        for sd, (dc, ctrl, init) in SEEDS.items():
            for run, curve in ((dc, dc_curve), (ctrl, ctrl_curve), (init, ctrl_curve)):
                os.makedirs(os.path.join(d, run), exist_ok=True)
                by = {}
                for dd in range(1, 13):
                    a = curve(dd)
                    if run == dc and dd in (5, 6) and not inform:
                        a = 0.6
                    by[str(dd)] = {"acc": {"8": a, "10": a, "16": a + (0.15 if (run == dc and dd == 12) else 0.0), "64": a, "128": a}}
                json.dump({"by_d": by}, open(os.path.join(d, run, "eval_long_dc.json"), "w"))
                items = {}
                for dd in range(1, 11):
                    n = 200; V = rng.integers(0, 11, size=(n, 10)); sig = ["%s_%d_%d" % (run, dd, i) for i in range(n)]
                    pv = {}
                    for t in range(1, 11):
                        p = ws_dc if run == dc else ws_c
                        pr = V[:, t - 1].copy(); bad = rng.random(n) > p; pr[bad] = (pr[bad] + 1) % 11; pv[str(t)] = pr.tolist()
                    items[str(dd)] = {"sig": sig, "pred_by_pass": pv, "v": V.tolist()}
                    for sg in sig:
                        regen[sg] = {"ops": ["add_c"] * dd, "src_j": [None] * dd}
                json.dump({"items": items}, open(os.path.join(d, run, "eval_ws_long.json"), "w"))
                with open(os.path.join(d, run, "log.jsonl"), "w") as f:
                    for i in range(1, 81):
                        f.write(json.dumps({"step": i * 100, "gn_plain_mean100": 9.0 if (gn_spike and run == dc and i == 30) else 1.0}) + "\n")
                    for i in range(9):
                        ng = {str(x): {"s8": 10.0 * (1.5 if (gamed and run == dc) else 1.0), "sd2": 8.0} for x in range(1, 7)}
                        f.write(json.dumps({"bin": i, "by_d": {}, "hold": {"norm_guard_by_d": ng}}) + "\n")
        return d, regen
    LP0_CAP = {"s1": 5.0, "s2": 5.0}
    nl_ok = lambda runs, dc, init, D: []  # noqa: E731
    chance = 0.09
    ctrl = lambda d: 1.0 if d <= 4 else 0.86 if d == 5 else 0.42 if d == 6 else chance  # noqa: E731
    shifted = lambda d: 1.0 if d <= 6 else 0.86 if d == 7 else 0.42 if d == 8 else chance  # noqa: E731
    narrow = lambda d: 1.0 if d <= 6 else 0.9 if d == 7 else 0.7 if d == 8 else 0.4 if d == 9 else 0.2  # noqa: E731
    dd, rg = world(shifted, ctrl); o = score(dd, rg, nl_ok); res.append(("pure +2 shift -> H-ARCH", o["verdict"].startswith("DC H-ARCH"), o["verdict"]))
    dd, rg = world(narrow, ctrl); o = score(dd, rg, nl_ok); res.append(("extrapolates + d12 rise -> PASS", o["verdict"].startswith("DC PASS"), o["verdict"]))
    dd, rg = world(narrow, ctrl, ws_dc=0.6, ws_c=0.6); o = score(dd, rg, nl_ok); res.append(("acc up but step-read not -> IN-RANGE-ONLY", o["verdict"].count("IN-RANGE-ONLY") == 2, o["verdict"]))
    dd, rg = world(narrow, ctrl, inform=False); o = score(dd, rg, nl_ok); res.append(("d 5-6 not learned -> UNINFORMATIVE", o["verdict"].count("UNINFORMATIVE") == 2, o["verdict"]))
    dd, rg = world(narrow, ctrl, gn_spike=True); o = score(dd, rg, nl_ok); res.append(("gn spike after 1k -> LP0-FAIL", o["verdict"].count("LP0-FAIL") == 2, o["verdict"]))
    dd, rg = world(narrow, ctrl, gamed=True); o = score(dd, rg, nl_ok); res.append(("per-d norm inflated -> GAMED", o["verdict"].count("GAMED") == 2, o["verdict"]))
    dd, rg = world(narrow, ctrl); o = score(dd, rg, lambda runs, dc, init, D: ["HOLD d6"]); res.append(("hold fails -> NOLOSS-FAIL", o["verdict"].count("NOLOSS-FAIL") == 2, o["verdict"]))
    LP0_CAP = {}; dd, rg = world(narrow, ctrl); o = score(dd, rg, nl_ok); res.append(("no LP0 cap yet -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
