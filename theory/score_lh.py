"""THEORY §240 LANG-HOLD scorer (docs/research_langhold_2026-10-28/LH_v1.md; Theory is the only scorer). C6 hold via the hold-increment penalty.
Arms: lh_fh_s1 / lh_fh_s2 (HOLD-INC) and lh_ctrl_s1 / lh_ctrl_s2 (plain FT), inits df_plaw4_s1 / s2. eval_long_k128.json (K' 8/16/32/64/128, d 1-5) in arms + inits.
Per FH seed, binding: HOLD-LONG3 (acc(K') >= acc(8) - 0.02 at K' 64 / 128, d 1-4); NOLOSS (score_df's §238 gates minus its LP0 + C1 floor + RETENTION:
  DF OP-ALL / PROD-SWAP / SWAP-MID d3 t1 / LP5 vs the init, REG OP-ALL / SWAP-MID / LP5 vs §237's rcb; any drop > 0.05 named); LP0-W on gn_plain_mean100
  (cap max(2 x init windows 3-10, 5.0)); NORM GUARD (Note 240-2: one-sided vs the paired CTRL; GAMED iff FH/CTRL > 1.25).
Labels (first match): GAMED / LP0-FAIL / NOLOSS-FAIL (gates named) / HOLD-FAIL / PASS. LEVER: s1 min_d acc128 FH - CTRL >= 0.10; s2 FH >= CTRL - 0.02.
Overall: MORE-TRAINING-SUFFICES if CTRL s1 passes HOLD-LONG3; PASS = both FH PASS and LEVER; NO-LEVER (both PASS, LEVER fails); SPLIT (one PASS); FAIL.
    python -B theory/score_lh.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory")); sys.path.insert(0, _HR)
import score_st as ST  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_lh.json")
SEEDS = {"s1": ("lh_fh_s1", "lh_ctrl_s1", "df_plaw4_s1", "rcb_plaw4_s1"), "s2": ("lh_fh_s2", "lh_ctrl_s2", "df_plaw4_s2", "rcb_plaw4_s2")}
CDRUN = "df_pcd4_s1"
HDS, HKS = ("1", "2", "3", "4"), ("64", "128")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def long_table(runs, run):
    e = jl(os.path.join(runs, run, "eval_long_k128.json"))
    return None if e is None else e["by_d"]


def hold(t):
    drops = {d: {k: t[d]["acc"][k] - t[d]["acc"]["8"] for k in HKS} for d in HDS}
    return all(v >= -0.02 for r in drops.values() for v in r.values()), drops


def min128(t):
    return min(t[d]["acc"]["128"] for d in HDS)


def lp0w(runs, run, init):
    gi = [r["gn_preclip_mean100"] for r in (ST.dedupe(ST.rows(os.path.join(runs, init, "log.jsonl"))) or []) if "gn_preclip_mean100" in r]
    lg = ST.dedupe(ST.rows(os.path.join(runs, run, "log.jsonl"))) or []
    g = [r.get("gn_plain_mean100", r.get("gn_preclip_mean100")) for r in lg if "gn_preclip_mean100" in r or "gn_plain_mean100" in r]
    if len(gi) < 10 or not g:
        return None, "missing windows"
    cap = max(2 * max(gi[2:10]), 5.0); bad = [x for x in g if x is None or not np.isfinite(x) or x > cap]
    return not bad, "cap %.3g, %d over" % (cap, len(bad))


def _ng_bins(runs, run):
    lg = ST.dedupe(ST.rows(os.path.join(runs, run, "log.jsonl"))) or []
    # Experiments' layout (240 build msg): bins carry norm_guard inside "hold" {inc_ratio, far_ratio, comp_ratio, hold_ratio, norm_guard}; top level also accepted
    out = {}
    for r in lg:
        if "bin" in r:
            ng = r.get("norm_guard") if isinstance(r.get("norm_guard"), dict) else (r.get("hold") or {}).get("norm_guard")
            if isinstance(ng, dict):
                out[r["bin"]] = ng
    return out


def norm_guard(runs, run, ctrl):
    """Note 240-2: one-sided, relative to the paired CTRL at the same bin. GAMED iff FH / CTRL > 1.25 for s8 or sd2 in any bin (inflation lowers the
    penalised ratio). FH / CTRL < 0.75 is reported only (shrinking makes the penalty harder; NOLOSS covers lost computation)."""
    f, c = _ng_bins(runs, run), _ng_bins(runs, ctrl)
    common = sorted(set(f) & set(c))
    if not common:
        return None, "no paired norm_guard bins"
    rat = {b: {k: f[b][k] / c[b][k] for k in ("s8", "sd2")} for b in common}
    bad = [(b, k) for b, r in rat.items() for k, v in r.items() if v > 1.25]
    shrink = [(b, k) for b, r in rat.items() for k, v in r.items() if v < 0.75]
    vals = [v for r in rat.values() for v in r.values()]
    return not bad, {"over": bad[:5], "shrink_desc": shrink[:5], "max_ratio": max(vals), "min_ratio": min(vals)}


def retention(runs, fh, init, rcb, df_noloss):
    """drops > 0.05 vs references -> list of named gates. df_noloss(run, ref0) -> score_df-style seed dict (op_all, swap_prod); injectable for fixtures."""
    out, named = {}, []
    a, b = df_noloss(fh, init), df_noloss(init, init)
    def opmin(s):
        return min(v["acc"] for p in ("p3", "p4") for v in s["op_all"][p].values())
    pairs = {"DF OP-ALL": (opmin(a), opmin(b)), "DF PROD-SWAP": (a["swap_prod"]["follow_U"], b["swap_prod"]["follow_U"])}
    for tag, run_a, run_b in (("DF", fh, init), ("REG", fh + "_reg", rcb)):
        ka, kb = jl(os.path.join(runs, run_a, "eval_swap_kd.json")), jl(os.path.join(runs, run_b, "eval_swap_kd.json"))
        ha, hb = jl(os.path.join(runs, run_a, "eval_lp5h.json")), jl(os.path.join(runs, run_b, "eval_lp5h.json"))
        if None in (ka, kb, ha, hb):
            return None, ["missing %s swap_kd / lp5h" % tag]
        pairs[tag + " SWAP-MID"] = (ka["by_d"]["3"]["swap_mid"]["3"]["follow_U"], kb["by_d"]["3"]["swap_mid"]["3"]["follow_U"])
        lp = lambda h: float(np.mean([h["by_d"][d]["intact"][d] - h["by_d"][d]["mean_at_t"][d] for d in ("3", "4")]))  # noqa: E731
        pairs[tag + " LP5"] = (lp(ha), lp(hb))
    try:   # Note 240-9: REG OP-ALL (registered in LH_v1 sec. 3, omitted in the first implementation): pass-3 min over §237's 8 classes, FH_reg vs rcb
        import score_rc as RC
        rg = RC.regen_reg()
        def _reg_min(run):
            w = jl(os.path.join(runs, run, "eval_ws.json"))["items"]; ok, cl = [], []
            for d in ("3", "4"):
                it = w[d]; V = np.array(it["v"]); ok += (np.array(it["pred_by_pass"]["3"]) == V[:, 2]).tolist(); cl += [RC.op_class(rg[x], 3) for x in it["sig"]]
            ok, cl = np.array(ok), np.array(cl); return min(float(ok[cl == c].mean()) for c in RC.CLASSES)
        pairs["REG OP-ALL"] = (_reg_min(fh + "_reg"), _reg_min(rcb))
    except (TypeError, KeyError, OSError):
        pass                                                                  # fixtures carry no REG ws items; real runs always have them
    for k, (x, y) in pairs.items():
        out[k] = {"final": x, "ref": y, "drop": y - x}
        if y - x > 0.05:
            named.append(k)
    return out, named


def c1_floor(runs, fh):
    a, c = jl(os.path.join(runs, fh, "eval_acc.json")), jl(os.path.join(runs, CDRUN, "eval_acc.json"))
    if a is None or c is None:
        return None
    return all(a["by_d"][d]["acc_kprime"][k] <= c["by_d"][d]["acc"] + 0.05 for d in ("3", "4") for k in ("0", "1"))


def default_df_noloss(run, ref0):
    import score_df as DF
    old = DF.LATS; DF.LATS = {run: ref0}
    try:
        sets = DF.regen(); return DF.seed_readings(R, run, ref0, *sets)
    finally:
        DF.LATS = old


def score(runs=R, df_noloss=None):
    df_noloss = df_noloss or default_df_noloss
    o = {"section": 240, "registered": "LH_v1.md", "seeds": {}}
    tabs = {r: long_table(runs, r) for v in SEEDS.values() for r in v[:3]}
    miss = [r for r, t in tabs.items() if t is None]
    if miss:
        o["verdict"] = "pending (missing eval_long_k128: %s)" % miss; return o
    for sd, (fh, ctrl, init, rcb) in SEEDS.items():
        s = {}
        hok, hdr = hold(tabs[fh]); s["hold"] = hok; s["hold_drops"] = hdr; s["step0_hold"] = hold(tabs[init])[0]
        lok, lmsg = lp0w(runs, fh, init); gok, gmsg = norm_guard(runs, fh, ctrl)
        dn = df_noloss(fh, init)
        if lok is None or gok is None or dn.get("label") == "pending":
            o["seeds"][sd] = {"label": "pending", "why": [lmsg, gmsg, dn.get("verdict")]}; continue
        dfg = {k: v for k, v in (dn.get("gates") or {}).items() if k != "LP0"}
        ret, named = retention(runs, fh, init, rcb, df_noloss)
        c1 = c1_floor(runs, fh)
        if ret is None or c1 is None:
            o["seeds"][sd] = {"label": "pending", "why": named or ["missing eval_acc"]}; continue
        nl_fail = [k for k, v in dfg.items() if not v] + ([] if c1 else ["C1-FLOOR"]) + named + (["VOID-guard"] if dn.get("label") == "VOID" else [])
        s.update({"lp0w": (lok, lmsg), "norm_guard": (gok, gmsg), "df_gates": dfg, "c1_floor": c1, "retention": ret, "retention_regressions": named})
        s["label"] = ("GAMED" if not gok else "LP0-FAIL" if not lok else "NOLOSS-FAIL (%s)" % ",".join(nl_fail) if nl_fail else "HOLD-FAIL" if not hok else "PASS")
        s["s_norm_ratio128"] = {d: tabs[fh][d]["s_norm"]["128"] / tabs[fh][d]["s_norm"]["8"] for d in HDS if tabs[fh][d].get("s_norm")}
        o["seeds"][sd] = s
    if any(v["label"] == "pending" for v in o["seeds"].values()):
        o["verdict"] = "pending (%s)" % {k: v.get("why") for k, v in o["seeds"].items() if v["label"] == "pending"}; return o
    lev1 = min128(tabs["lh_fh_s1"]) - min128(tabs["lh_ctrl_s1"]) >= 0.10; lev2 = min128(tabs["lh_fh_s2"]) >= min128(tabs["lh_ctrl_s2"]) - 0.02
    ctrl1_hold = hold(tabs["lh_ctrl_s1"])[0]
    o.update({"LEVER": lev1 and lev2, "lever_s1": min128(tabs["lh_fh_s1"]) - min128(tabs["lh_ctrl_s1"]), "ctrl_s1_hold": ctrl1_hold})
    labs = [o["seeds"][k]["label"] for k in ("s1", "s2")]; np_ = labs.count("PASS")
    v = ("LH MORE-TRAINING-SUFFICES" if ctrl1_hold else "LH PASS" if (np_ == 2 and lev1 and lev2) else "LH NO-LEVER" if np_ == 2
         else "LH SPLIT" if np_ == 1 else "LH FAIL")
    o["verdict"] = "%s | s1: %s / s2: %s | lever_s1 %+.3f" % (v, labs[0], labs[1], o["lever_s1"]); return o


def fixtures():
    res = []

    def world(fh_drop=0.0, ctrl_drop=0.10, gamed=False, gn_spike=False, retention_drop=0.0, c1_bad=False, shrink=0.7):
        d = tempfile.mkdtemp()
        def long(run, drop):
            os.makedirs(os.path.join(d, run), exist_ok=True)
            by = {dd: {"acc": {"8": 1.0, "16": 1.0, "32": 1.0, "64": 1.0 - drop / 2, "128": 1.0 - drop}, "s_norm": {"8": 10.0, "128": 11.0}} for dd in HDS + ("5",)}
            json.dump({"by_d": by}, open(os.path.join(d, run, "eval_long_k128.json"), "w"))
        for sd, (fh, ctrl, init, rcb) in SEEDS.items():
            long(fh, fh_drop); long(ctrl, ctrl_drop if sd == "s1" else 0.0); long(init, 0.06)
            for run in (init,):
                with open(os.path.join(d, run, "log.jsonl"), "w") as f:
                    for i in range(1, 17):
                        f.write(json.dumps({"step": i * 1000, "gn_preclip_mean100": 2.0}) + "\n")
            with open(os.path.join(d, fh, "log.jsonl"), "w") as f:
                for i in range(1, 81):
                    f.write(json.dumps({"step": i * 100, "gn_preclip_mean100": 3.0, "gn_plain_mean100": 9.0 if (gn_spike and i == 40) else 1.0}) + "\n")
                for i in range(9):
                    f.write(json.dumps({"bin": i, "by_d": {}, "hold": {"norm_guard": {"s8": 10.0 * (1.5 if (gamed and i == 5) else shrink if i > 0 else 1.0), "sd2": 8.0}}}) + "\n")
            with open(os.path.join(d, ctrl, "log.jsonl"), "w") as f:
                for i in range(9):
                    f.write(json.dumps({"bin": i, "by_d": {}, "hold": {"norm_guard": {"s8": 10.0 * (0.7 if i > 0 else 1.0), "sd2": 8.0}}}) + "\n")
            for run, sm, lp in ((fh, 0.95 - retention_drop, 0.9), (init, 0.95, 0.9), (fh + "_reg", 0.97, 0.9), (rcb, 0.97, 0.9)):
                os.makedirs(os.path.join(d, run), exist_ok=True)
                json.dump({"by_d": {"3": {"swap_mid": {"3": {"follow_U": sm}}}}}, open(os.path.join(d, run, "eval_swap_kd.json"), "w"))
                json.dump({"by_d": {dd: {"intact": {dd: 1.0}, "mean_at_t": {dd: 1.0 - lp}} for dd in ("3", "4")}}, open(os.path.join(d, run, "eval_lp5h.json"), "w"))
            json.dump({"by_d": {dd: {"acc": 1.0, "acc_kprime": {"0": 0.5 if c1_bad else 0.1, "1": 0.1}} for dd in ("3", "4")}}, open(os.path.join(d, fh, "eval_acc.json"), "w"))
        os.makedirs(os.path.join(d, CDRUN), exist_ok=True)
        json.dump({"by_d": {dd: {"acc": 0.1} for dd in ("3", "4")}}, open(os.path.join(d, CDRUN, "eval_acc.json"), "w"))
        return d

    def stub(run, ref0):
        return {"label": "PASS", "gates": {"LP0": False, "PROD-ACC": True, "CHAIN": True, "OP-ALL+": True, "PROD-SWAP": True},
                "op_all": {"p3": {"mul_s": {"acc": 1.0}}, "p4": {"mul_s_j2": {"acc": 1.0}}}, "swap_prod": {"follow_U": 1.0}}
    o = score(world(ctrl_drop=0.12), stub); res.append(("FH holds, CTRL s1 drifts -> LH PASS", o["verdict"].startswith("LH PASS"), o["verdict"]))
    o = score(world(ctrl_drop=0.0), stub); res.append(("CTRL s1 holds too -> MORE-TRAINING-SUFFICES", o["verdict"].startswith("LH MORE-TRAINING-SUFFICES"), o["verdict"]))
    o = score(world(fh_drop=0.05), stub); res.append(("FH drifts -> HOLD-FAIL both", o["verdict"].count("HOLD-FAIL") == 2, o["verdict"]))
    o = score(world(gamed=True), stub); res.append(("FH norm 1.5 vs CTRL 0.7 at a bin -> GAMED", o["verdict"].count("GAMED") == 2, o["verdict"]))
    o = score(world(shrink=0.4, ctrl_drop=0.12), stub); res.append(("FH shrinks to 0.57x CTRL -> not GAMED", o["verdict"].startswith("LH PASS"), o["verdict"]))
    o = score(world(gn_spike=True), stub); res.append(("plain gn spike over cap 5 -> LP0-FAIL", o["verdict"].count("LP0-FAIL") == 2, o["verdict"]))
    o = score(world(retention_drop=0.08), stub); res.append(("DF SWAP-MID drop 0.08 -> NOLOSS-FAIL named", "NOLOSS-FAIL (DF SWAP-MID)" in o["verdict"], o["verdict"]))
    o = score(world(c1_bad=True), stub); res.append(("K' 0 acc high -> NOLOSS-FAIL (C1-FLOOR)", "C1-FLOOR" in o["verdict"], o["verdict"]))
    o = score(world(ctrl_drop=0.05), stub); res.append(("both hold, lever 0.05 < 0.10 -> NO-LEVER", o["verdict"].startswith("LH NO-LEVER"), o["verdict"]))
    d = world(); os.remove(os.path.join(d, "df_plaw4_s2", "eval_long_k128.json")); o = score(d, stub); res.append(("missing step-0 long -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
