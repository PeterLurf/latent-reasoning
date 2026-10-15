"""THEORY §204 scoring (STEP-TARGET-FACTOR; registered text docs/THEORY_STEPFACTOR_DRAFT_2026-10-14.md at 3ec408c6 + notes 204-3 / 204-5 / 204-6;
Theory is the only scorer). The branch is read from theory/results/score_s202.json (W_LIFT): WARM-HELPS -> F-P gsm_lat_factor_warm_s1 vs W
gsm_lat_warm_s1, F-NR gsm_lat_factor_nr_warm_s1 vs F-P; otherwise F-P gsm_lat_factor_s1 vs A gsm_lat_aug_s1. FINAL checkpoint.
Inputs per cell: <run>/eval.json (acc, acc_by_kprime, acc_by_m_kprime, n_by_m, aux_suite). Comparator: <comp>/eval_s204ref.json (its eval-only
re-run with the §204 fields, same items / rng); its registered fields must equal its eval.json (checked: acc, acc_by_kprime).
Eval-only add-ons (read if present): <run>/eval_s204_probe.json {"probe_leak": {"plan": {t: ratio}}}, <run>/eval_s204_clock.json {"pair_acc": {"3v4": x}},
aux_suite[src]["plan_ce"] {t: nats} (note 204-6), aux_suite["lotus_sep"] (note 204-5's added field).
T = 0.028, T_D = 0.034, deep = m >= 3. plan-head accuracy (PLAN-LEAK, note 204-6) = mean of the present heads' accuracies (op_acc, ptr1_acc, ptr2_acc).
    python theory/score_s204.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
T, TD, EPS = 0.028, 0.034, 1e-9
PLAN = ("op_acc", "ptr1_acc", "ptr2_acc")


def jl(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def three(d, t=T):
    return "LIFTS" if d >= t - EPS else ("HURTS" if d <= -t + EPS else "MATCHES")


def deep(e, kp):
    a, n = e["acc_by_m_kprime"][str(kp)], {int(m): v for m, v in e["n_by_m"].items()}
    ms = [m for m in n if m >= 3 and str(m) in a]
    return sum(a[str(m)] * n[m] for m in ms) / sum(n[m] for m in ms)


def pooled_t(S, src, field, ts, grounding=None):
    """count-weighted accuracy over passes ts (optionally one grounding stratum) from aux_suite[set][src]."""
    s = S.get(src)
    if not s or field not in s:
        return None
    v = s[field + "_by_grounding"][grounding] if grounding else s[field]
    nn = s["n_" + field + "_by_grounding"][grounding] if grounding and ("n_" + field + "_by_grounding") in s else s.get("n_" + field, {})
    ts = [str(t) for t in ts if str(t) in v]
    if not ts:
        return None
    w = [nn.get(t, 1) for t in ts]
    return sum(v[t] * w_ for t, w_ in zip(ts, w)) / sum(w) if sum(w) else None


def plan_acc(S, src, t):
    s = S.get(src) or {}
    xs = [s[f][str(t)] for f in PLAN if f in s and str(t) in s[f]]
    return sum(xs) / len(xs) if xs else None


def core(e):
    k = e["acc_by_kprime"]; out = {"acc": e["acc"], "K0": k["0"], "K1": k["1"], "K10": k["10"]}
    out["loop_formed"] = k["10"] - k["0"] >= T - EPS
    out["think_necessary"] = k["0"] <= 0.5 * k["10"] + EPS and k["1"] <= 0.5 * k["10"] + EPS
    out["shortcut_ratio"] = k["1"] / k["10"] if k["10"] > 0 else None
    d0, d1, d10 = deep(e, 0), deep(e, 1), deep(e, 10)
    out.update(deep_K0=d0, deep_K1=d1, deep_K10=d10, deep_ratio=d1 / d10 if d10 > 0 else None,
               think_necessary_deep=d0 <= 0.5 * d10 + EPS and d1 <= 0.5 * d10 + EPS, serial_gain_deep=d10 - d1)
    return out


def factor_reads(e):
    S = e["aux_suite"]["test_extras"]; o = {}
    o["value_exact_by_t"] = S["Mt"].get("value_exact"); o["value_exact_3"] = (S["Mt"].get("value_exact") or {}).get("3")
    o["value_R_t23_Mt"] = pooled_t(S, "Mt", "value_exact", (2, 3), "R"); o["value_R_t23_M0"] = pooled_t(S, "M0", "value_exact", (2, 3), "R")
    o["value_R_t23_M1"] = pooled_t(S, "M1", "value_exact", (2, 3), "R")
    o["value_R_all"] = pooled_t(S, "Mt", "value_exact", range(1, 9), "R"); o["value_Q_all"] = pooled_t(S, "Mt", "value_exact", range(1, 9), "Q")
    o["plan_acc_1"] = {f: S["Mt"][f]["1"] for f in PLAN if f in S["Mt"] and "1" in S["Mt"][f]}
    o["plan_leak"] = {t: (plan_acc(S, "M1", t) / plan_acc(S, "Mt", t)) if plan_acc(S, "M1", t) and plan_acc(S, "Mt", t) else None for t in (2, 3)}
    o["plan_ce_t3p"] = pooled_t(S, "Mt", "plan_ce", range(3, 9)) if "plan_ce" in S["Mt"] else None
    return o


def main():
    s202 = json.load(open(os.path.join(ROOT, "theory", "results", "score_s202.json"))) if os.path.exists(os.path.join(ROOT, "theory", "results", "score_s202.json")) else {}
    warm = s202.get("W_LIFT") == "WARM-HELPS"
    fp, comp, fnr = ("gsm_lat_factor_warm_s1", "gsm_lat_warm_s1", "gsm_lat_factor_nr_warm_s1") if warm else ("gsm_lat_factor_s1", "gsm_lat_aug_s1", None)
    out = {"branch": "W base (WARM-HELPS)" if warm else ("A base" if s202.get("W_LIFT") else "pending (§202 not scored)"), "cells": {"F-P": fp, "comparator": comp, "F-NR": fnr}}
    if not s202.get("W_LIFT"):
        print("READING: pending (§202 not scored)"); json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s204.json"), "w", newline=chr(10)), indent=1); return
    E, Cref, Cev = jl(fp, "eval.json"), jl(comp, "eval_s204ref.json"), jl(comp, "eval.json")
    if E is None or Cref is None:
        out["reading"] = "pending (F-P or comparator eval-only not landed)"; print("READING:", out["reading"])
        json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s204.json"), "w", newline=chr(10)), indent=1); return
    out["comparator_registered_fields_equal"] = bool(Cev and Cref["acc"] == Cev["acc"] and Cref["acc_by_kprime"] == Cev["acc_by_kprime"])
    f, c = core(E), core(Cref); fr = factor_reads(E); Sc = Cref["aux_suite"]["test_extras"]; pr = {}; L = []
    out.update(F=f, C=c, F_reads=fr)
    d = f["acc"] - c["acc"]; out["ACC"] = three(d); out["acc_delta"] = d
    L.append("ACC %s (F-P %.3f vs comp %.3f, %+.3f)" % (out["ACC"], f["acc"], c["acc"], d))
    L.append("SHORTCUT-RATIO F %.3f vs comp %.3f; deep ratio F %.3f vs comp %.3f; THINK-NECESSARY %s / DEEP %s; SERIAL-GAIN-DEEP %+.3f (%s)" % (
        f["shortcut_ratio"], c["shortcut_ratio"], f["deep_ratio"], c["deep_ratio"], f["think_necessary"], f["think_necessary_deep"],
        f["serial_gain_deep"] - c["serial_gain_deep"], "MORE-SERIAL" if f["serial_gain_deep"] - c["serial_gain_deep"] >= TD - EPS else "not more serial"))
    # P-F1: pointer 1 / operator at pass 1 vs the comparator's string proxies (note 204-3: string equality; collision-free beside)
    cp = Sc["Mt"]; p1, op1 = fr["plan_acc_1"].get("ptr1_acc"), fr["plan_acc_1"].get("op_acc")
    px_p, px_o = (cp.get("ptr1_in_string") or {}).get("1"), (cp.get("op_in_string") or {}).get("1")
    if None not in (p1, op1, px_p, px_o):
        pr["P-F1"] = "FALSIFIED" if p1 < px_p + 0.02 - EPS else ("holds" if p1 >= px_p + 0.05 - EPS and op1 >= px_o - EPS else "between")
    pl = fr["plan_leak"]
    if None not in pl.values():
        pr["P-F2"] = "FALSIFIED" if min(pl.values()) < 0.5 - EPS else ("holds (PROBE-LEAK pending)" if min(pl.values()) >= 0.8 - EPS else "between")
    vt, v0, v1 = fr["value_R_t23_Mt"], fr["value_R_t23_M0"], fr["value_R_t23_M1"]
    if None not in (vt, v0):
        if v1 is not None and vt > 0 and v1 >= 0.7 * vt - EPS:
            pr["P-F3"] = "FALSIFIED"
        else:
            pr["P-F3"] = "holds" if vt >= v0 + 0.10 - EPS and (v1 is None or v1 <= v0 + 0.05 + EPS) else "not met"
    if fr["value_exact_3"] is not None:
        pr["P-F4a (Audit)"] = "holds" if fr["value_exact_3"] >= 0.5 - EPS else "FALSIFIED (predicted by the derivation)"
    sr = f["shortcut_ratio"] - c["shortcut_ratio"]
    pr["P-F5"] = "FALSIFIED" if sr >= -0.03 - EPS else ("holds" if sr <= -0.10 + EPS else "between")
    pr["P-F5a (Audit)"] = "holds" if f["think_necessary"] else "FALSIFIED"
    pr["P-F6"] = "FALSIFIED" if f["deep_ratio"] >= c["deep_ratio"] - 0.05 - EPS else ("holds" if f["think_necessary_deep"] else "deep ratio lower but DEEP not passed")
    pr["P-F7"] = "FALSIFIED" if out["ACC"] == "HURTS" else "holds"
    if None not in (fr["value_R_all"], fr["value_Q_all"]):
        g = fr["value_Q_all"] - fr["value_R_all"]; pr["P-F9"] = "holds" if g >= 0.10 - EPS else ("FALSIFIED" if g <= 0.02 + EPS else "between")
    # note 204-5 LOTUS-SEP (descriptive)
    vs = pr.get("P-F3") == "holds"; dr = f["deep_ratio"] - c["deep_ratio"]; ls = E["aux_suite"].get("lotus_sep")
    if vs and dr <= -0.20 + EPS:
        lot = "(ii)-SUFFICIENT"
    elif vs and dr > -0.10 + EPS:
        lot = "DECODER-SHORTCUT"
    elif vt is not None and v0 is not None and vt <= v0 + 0.05 + EPS and dr > -0.10 + EPS:
        if ls is None:
            lot = "pending (lotus_sep field)"
        else:
            vd, pd_ = ls["value_le2"] - ls["value_ge4"], ls["plan_le2"] - ls["plan_ge4"]
            lot = "WIDTH-CANDIDATE" if vd >= 0.10 - EPS and vd - pd_ >= 0.05 - EPS else "CARRY-FAILS"
    else:
        lot = "INCONCLUSIVE"
    out["LOTUS_SEP"] = lot
    # F-NR vs F-P (W base only)
    if fnr:
        N = jl(fnr, "eval.json")
        if N:
            n = core(N); nr = factor_reads(N); dn = n["acc"] - f["acc"]; ddr = n["deep_ratio"] - f["deep_ratio"]
            pb = (jl(fnr, "eval_s204_probe.json") or {}).get("probe_leak", {}).get("plan", {}).get("3"); ck = (jl(fnr, "eval_s204_clock.json") or {}).get("pair_acc", {}).get("3v4")
            ce_up = None if None in (nr["plan_ce_t3p"], fr["plan_ce_t3p"]) else nr["plan_ce_t3p"] - fr["plan_ce_t3p"] >= 0.05 - EPS
            if dn <= -T + EPS and ce_up:
                fo = "HURTS"
            elif ddr <= -0.08 + EPS and pb is not None and pb <= 0.7 + EPS and dn > -T + EPS:
                fo = "MORE-SERIAL"
            elif pb is None or ck is None:
                fo = "pending (PROBE-LEAK / CLOCK)"
            else:
                fo = "NO-CHANGE + CLOCK-REBUILT" if ck >= 0.90 - EPS else "NO-CHANGE"
            out["F-NR"] = {"acc_delta": dn, "ACC": three(dn), "deep_ratio_delta": ddr, "probe_leak3": pb, "clock_3v4": ck, "plan_ce_up": ce_up, "outcome": fo}
            pr["P-F8"] = "holds" if (abs(dn) <= T + EPS and ddr <= -0.08 + EPS and (pb is None or pb <= 0.7 + EPS) and (ck is None or ck < 0.90 - EPS)) else "not met"
            L.append("F-NR vs F-P: %s (ACC %+.3f, deep ratio %+.3f)" % (fo, dn, ddr))
    out["predictions"] = pr; out["reading"] = " | ".join(L) + " | LOTUS-SEP %s" % lot
    print("READING:", out["reading"]); print("PREDICTIONS:", pr)
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s204.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
