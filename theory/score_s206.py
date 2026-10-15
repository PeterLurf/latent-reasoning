"""THEORY §206 scoring (JOINT-COT-L1; registered text docs/THEORY_JOINTCOT_DRAFT_2026-10-14.md at a005c7c7 + notes 206-1..; Theory is the only scorer).
Cells (boxes/runs_fsq): J0 gsm_lat_jcot_s1, J1 gsm_lat_jcotl1_s1; comparators W gsm_lat_warm_s1 (eval.json; deep fields from eval_s206ref.json, W re-run on
J's eval card / bs), A gsm_lat_aug_s1, C+ gsm_txt_aug_cont_s1. FINAL checkpoint. T = 0.028, T_D = 0.034 (deep = m >= 3).
Readings (draft §6.2): COUPLE = J1 - J0, PROTECT = J0 - W ("joint-CE effect: protection + extra text exposure"), TOTAL = J1 - W, J1 - A, R1-J = J1 - C+;
COT-RETAIN (min cot_valid over the 2k records >= 0.30; a missing record = failure); LOOP-FORMED; THINK-NECESSARY / -DEEP; DEEP-MULTIPASS (deep K'10 - K'1
>= T_D); K0-GUARD (K'0(J1) <= 0.139); KD-ROUTE (kd_route.share_mem >= 0.10); SHORTCUT (COUPLE-HELPS with K'0(J1) - K'0(J0) >= T). Predictions P-J0a..P-J1j.
Censoring (rule b): a cell whose log ends before 12000 -> every reading of it is a failure.
    python theory/score_s206.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
T, TD, EPS, FULL = 0.028, 0.034, 1e-9, 12000
CELLS = {"J0": "gsm_lat_jcot_s1", "J1": "gsm_lat_jcotl1_s1", "W": "gsm_lat_warm_s1", "A": "gsm_lat_aug_s1", "Cp": "gsm_txt_aug_cont_s1"}


def jl(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def log(d):
    out = {}
    f = os.path.join(R, d, "log.jsonl")
    if os.path.exists(f):
        for line in open(f):
            r = json.loads(line)
            if "step" in r:
                out.setdefault(int(r["step"]), {}).update(r)
    return out


def three(d):
    return "HELPS" if d >= T - EPS else ("HURTS" if d <= -T + EPS else "NEUTRAL")


def deep(e, kp):
    a, n = e["acc_by_m_kprime"][str(kp)], {int(m): v for m, v in e["n_by_m"].items()}
    ms = [m for m in n if m >= 3 and str(m) in a]
    return sum(a[str(m)] * n[m] for m in ms) / sum(n[m] for m in ms)


def cell(key):
    d = CELLS[key]; e = jl(d, "eval.json"); L = log(d)
    if e is None:
        return None
    last = max(L) if L else 0; o = {"dir": d, "last_step": last, "dead": last < FULL}
    k = e["acc_by_kprime"]; o.update(acc=e["acc"], K0=k["0"], K1=k["1"], K10=k["10"])
    o["loop_formed"] = k["10"] - k["0"] >= T - EPS
    o["think_necessary"] = k["0"] <= 0.5 * k["10"] + EPS and k["1"] <= 0.5 * k["10"] + EPS
    d0, d1, d10 = deep(e, 0), deep(e, 1), deep(e, 10)
    o.update(deep_K0=d0, deep_K1=d1, deep_K10=d10, think_necessary_deep=d0 <= 0.5 * d10 + EPS and d1 <= 0.5 * d10 + EPS, deep_multipass=d10 - d1)
    vr = {s: r["cot_valid_acc"] for s, r in L.items() if "cot_valid_acc" in r and r["cot_valid_acc"] is not None}
    need = list(range(2000, FULL + 1, 2000)); vals = [vr.get(s) for s in need]
    o["cot_valid_by_2k"] = dict(zip(need, vals)); o["cot_retain_min_event_only"] = min([v for v in vals if v is not None], default=None)
    o["cot_retain"] = (not o["dead"]) and None not in vals and min(vals) >= 0.30 - EPS
    kr = e.get("kd_route") or {}; o["kd_share_mem"], o["kd_share_multipass"] = kr.get("share_mem"), kr.get("share_multipass")
    on = [s for s in sorted(L) if (L[s].get("kdr_share_mem") or 0) >= 0.10 - EPS]; o["kd_route_onset"] = on[0] if on else "> 12k"
    o["kd_gen"] = (e.get("kd_gen") or {}).get("ratio")
    t2 = ((e.get("t2_acc") or {}).get("valid") or {}); o["t2_valid_top1"], o["t2_valid_acc"] = t2.get("top1"), t2.get("acc")
    f = e.get("fit") or {}; o["recall_share"] = (f["fit_fix"] - f["fit_rs"]) if f.get("fit_rs") is not None else None
    return o


def main():
    J0, J1 = cell("J0"), cell("J1"); W, A, Cp = jl(CELLS["W"], "eval.json"), jl(CELLS["A"], "eval.json"), jl(CELLS["Cp"], "eval.json")
    Wref = jl(CELLS["W"], "eval_s206ref.json"); out = {"J0": J0, "J1": J1}; pr = {}; L = []
    if J0 is None and J1 is None:
        print("READING: pending (no J cell landed)"); json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s206.json"), "w", newline=chr(10)), indent=1); return
    if J0 and J1:
        if J0["dead"] or J1["dead"]:
            out["COUPLE"] = "FAILURE (a cell died before 12k; rule b)"
        else:
            dc = J1["acc"] - J0["acc"]; out["COUPLE"] = three(dc); out["couple_delta"] = dc
            out["SHORTCUT"] = out["COUPLE"] == "HELPS" and J1["K0"] - J0["K0"] >= T - EPS
            pr["P-J1b"] = "holds" if out["COUPLE"] == "HELPS" and not out["SHORTCUT"] else ("SHORTCUT (not a success)" if out["SHORTCUT"] else "FALSIFIED")
        L.append("COUPLE %s (J1 - J0 %s)%s" % (out["COUPLE"], "%+.3f" % out["couple_delta"] if "couple_delta" in out else "n/a", " SHORTCUT" if out.get("SHORTCUT") else ""))
    if J0 and W:
        dp = J0["acc"] - W["acc"]; out["PROTECT"] = "FAILURE (J0 died)" if J0["dead"] else three(dp); out["protect_delta"] = dp
        pr["P-J0b"] = "FALSIFIED" if out["PROTECT"] == "HELPS" else ("holds" if out["PROTECT"] == "NEUTRAL" else "HURTS (not predicted)")
        L.append("PROTECT (joint-CE effect: protection + extra text exposure) %s (%+.3f)" % (out["PROTECT"], dp))
    for key, c in (("J0", J0), ("J1", J1)):
        if c:
            pr["P-%sa" % key] = "holds" if c["cot_retain"] else "FALSIFIED"
    if J1:
        tw = J1["acc"] - W["acc"] if W else None; ta = J1["acc"] - A["acc"] if A else None; rc = J1["acc"] - Cp["acc"] if Cp else None
        out.update(TOTAL=None if tw is None else ("FAILURE (J1 died)" if J1["dead"] else three(tw)), total_delta=tw, J1_minus_A=ta,
                   R1_J=None if rc is None else ("above" if rc >= T - EPS else ("below" if rc <= -T + EPS else "matches")))
        pr["P-J1c"] = "holds" if (not J1["dead"] and tw is not None and tw >= T - EPS and 0.15 - EPS <= J1["acc"] <= 0.25 + EPS and ta is not None and ta >= T - EPS) else "FALSIFIED"
        pr["P-J1d"] = "holds" if out["R1_J"] == "below" else "FALSIFIED"
        dm = J1["deep_multipass"] >= TD - EPS
        pr["P-J1e"] = ("holds" if J1["loop_formed"] and dm else "FALSIFIED") + " (DEEP-MULTIPASS %+.3f; pooled THINK-NECESSARY %s, not a falsifier)" % (J1["deep_multipass"], J1["think_necessary"])
        wk0 = W["acc_by_kprime"]["0"] if W else 0.111; out["K0_GUARD"] = J1["K0"] <= wk0 + T + EPS
        pr["P-J1f"] = "holds" if out["K0_GUARD"] else "FALSIFIED (direct-path strengthening, G1)"
        sm, smp = J1["kd_share_mem"], J1["kd_share_multipass"]
        pr["P-J1g"] = None if sm is None else ("holds" if sm >= 0.10 - EPS and (smp or 0) >= 0.02 - EPS else ("FALSIFIED" if sm < 0.10 - EPS else "mem holds, multipass < 0.02"))
        rs = J1["recall_share"]; pr["P-J1h"] = None if rs is None else ("holds" if rs >= 0.25 - EPS else ("FALSIFIED" if rs <= 0.10 + EPS else "between"))
        kg = J1["kd_gen"]; pr["P-J1i"] = None if kg is None else ("holds" if kg <= 1.3 + EPS else ("FALSIFIED" if kg > 1.5 + EPS else "between"))
        t2 = J1["t2_valid_top1"]; pr["P-J1j"] = None if t2 is None else ("holds" if t2 >= 0.15 - EPS else ("FALSIFIED" if t2 < 0.05 - EPS else "between"))
        L.append("TOTAL %s (J1 - W %s); J1 - A %s; R1-J %s; K0-GUARD %s (K'0 %.3f); DEEP-MULTIPASS %+.3f; KD-ROUTE share_mem %s (onset %s)" % (
            out["TOTAL"], None if tw is None else "%+.3f" % tw, None if ta is None else "%+.3f" % ta, out["R1_J"], out["K0_GUARD"], J1["K0"], J1["deep_multipass"], sm, J1["kd_route_onset"]))
    if Wref is not None:
        out["W_ref_deep_multipass"] = deep(Wref, 10) - deep(Wref, 1)
    for key, c in (("J0", J0), ("J1", J1)):                                 # rule b: every reading of a cell that died before 12k is a failure
        if c and c["dead"]:
            for k_ in list(pr):
                if k_.startswith("P-" + key):
                    pr[k_] = "FAILURE (%s died at %d; rule b)" % (key, c["last_step"])
            if key == "J1":
                pr["P-J1b"] = "FAILURE (J1 died at %d; rule b)" % c["last_step"]
    out["predictions"] = pr; out["reading"] = " | ".join(L)
    print("READING:", out["reading"]); print("PREDICTIONS:", pr)
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s206.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
