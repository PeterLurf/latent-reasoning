"""THEORY §168 step 2 scoring (note 168-7: STEP-AUX-ProsQA; rules = §168's registered step-2 rules; written before any step-2 implementation or run).
STEP-AUX: §175's recipe on p8 NA (K = 10 INJECT passes over the question; blind per-pass aux emits steps[t]); answer = §168's candidate accuracy (chance 0.5).
Window = max(D*, D*_curr, 3) .. 10, D* from §168's NA DIRECT runs (score_s168.seed_boundary), D*_curr from CURR-DIRECT (168-4; same rule).
DEEP-LATENT(seed) = acc >= 0.90 at every window depth at K' = 10 AND THINK-NECESSARY (168-1 above chance, removal only per 170-4: pooled over the window,
(acc(K') - 0.5) <= 0.5 x (acc(10) - 0.5) for K' = 0 and 1). DEEP-LATENT(arm) = 2 / 2 seeds (s2 opens only if s1 is DEEP).
DEPTH-GENERAL (C2) = mean acc at depths 11-14 at K' = 16 (the pre-set test budget) >= 0.80. BEYOND TEXT = TEXT-COT's 11-14 mean < 0.80 while DEPTH-GENERAL.
Dirs boxes/runs_fsq/s168s2_{stepaux,currdirect,textcot}_s{seed}.
    python theory/score_s168s2.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s168 as S168  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
CH = 0.5
EPS = 1e-9
FADES = ("9000,12000", "12000,12000")                                      # §175's fade scaled to 12k, or none (note 168-7: chosen by §182 NOFADE)
CELLS = {"stepaux": {"data": "na", "arm": "stepaux", "think_k": 10, "aux_step": 1.0, "steps": 12000},
         "currdirect": {"data": "na", "arm": "currdirect", "steps": 12000},
         "textcot": {"data": "na", "arm": "textcot", "steps": 12000},
         "stepauxrw": {"data": "na", "arm": "stepaux", "think_k": 10, "aux_step": 1.0, "steps": 12000, "aux_reweight": True}}   # note 168-12
RW_HIST = {3: 815, 4: 1628, 5: 1711, 6: 1777, 7: 1716, 8: 1675, 9: 1756, 10: 907}   # p8_na_train len(steps) histogram (note 168-12)
_P = {L: c / sum(RW_HIST.values()) for L, c in RW_HIST.items()}
_E = [sum(_P[L] / L for L in _P if L >= t) for t in range(1, 11)]
RW_W = [round(sum(_E) / 10 / e, 3) for e in _E]
assert RW_W == [0.583, 0.583, 0.583, 0.671, 0.869, 1.156, 1.619, 2.42, 4.193, 13.214], RW_W   # pinned (note 168-12)
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def header(cell, s):
    d = "s168s2_%s_s%d" % (cell, s); a = load(d, "args.json") or {}; r = load(d)
    if r is None:
        return d, None, {"status": "running" if a else "open"}
    bad = {k: a.get(k) for k, v in dict(CELLS[cell], seed=s).items() if str(a.get(k)) != str(v)}
    if cell in ("stepaux", "stepauxrw") and str(a.get("aux_fade")) not in FADES:
        bad["aux_fade"] = a.get("aux_fade")
    if cell == "stepaux" and a.get("aux_reweight"):                         # note 168-12: the unweighted arm must not carry the reweight
        bad["aux_reweight"] = a.get("aux_reweight")
    if bad:
        ERRORS.append((d, bad)); return d, None, {"status": "error", "error": bad}
    return d, r, {"status": "done", "gpu": r.get("train_gpu"), "aux_fade": a.get("aux_fade")}


def dstar_of(ab):
    fail = [L for L in S168.DEPTHS if str(L) in ab and float(ab[str(L)]) < 0.90 - EPS]
    return fail[0] if fail else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def acc(ab, L):
    v = (ab or {}).get(str(L))
    return float(v) if v is not None else None


def arm_read(cell, lo, dcur, tc):
    runs = {}
    for s in (1, 2):
        d, r, h = header(cell, s)
        if r is None:
            runs[s] = h; continue
        x = r.get("s168s2") or {}; kp = x.get("acc_by_depth_kprime") or {}
        k10, k0, k1, k16 = kp.get("10"), kp.get("0"), kp.get("1"), kp.get("16")
        if not (k10 and k0 and k1 and k16):
            runs[s] = dict(h, status="incomplete", missing=[k for k in ("0", "1", "10", "16") if not kp.get(k)]); continue
        hop = x.get("aux_exact_by_hop") or {}; late = [hop.get(str(t)) for t in (8, 9, 10)]
        base = dict(h, pooled_3_10=mean([acc(k10, L) for L in range(3, 11)]), late_aux=None if None in late else sum(late) / 3,
                    clip_rate=(r.get("s184") or x.get("clip") or {}).get("clip_rate"))
        dg = mean([acc(k16, L) for L in range(11, 15)])
        if lo > 10:                                                         # 168-4: the curriculum single pass already reaches depth 10
            runs[s] = dict(base, no_window="D*_curr %s > 10: no beyond-single-pass window in 3..10" % dcur, mean_11_14_k16=dg,
                           depth_general=bool(dg is not None and dg >= 0.80 - EPS)); continue
        win = range(lo, 11)
        p10, p0, p1 = mean([acc(k10, L) for L in win]), mean([acc(k0, L) for L in win]), mean([acc(k1, L) for L in win])
        tn = None if p10 - CH < 0.1 else bool((p0 - CH) <= 0.5 * (p10 - CH) + EPS and (p1 - CH) <= 0.5 * (p10 - CH) + EPS)   # 170-5's guard
        deep = all((acc(k10, L) or 0) >= 0.90 - EPS for L in win)
        runs[s] = dict(base, pooled_window_k10=p10, pooled_window_k0=p0, pooled_window_k1=p1, think_necessary=tn, deep_at_window=deep, deep_latent=bool(deep and tn),
                       mean_11_14_k16=dg, depth_general=bool(dg is not None and dg >= 0.80 - EPS),
                       mean_11_14_by_kprime={k: mean([acc(v, L) for L in range(11, 15)]) for k, v in kp.items()})
    r1, r2 = runs[1], runs[2]
    if r1.get("no_window"):
        rd = "DEEP-LATENT not readable (%s); DEPTH-GENERAL %s (11-14 at K'=16: %s)" % (r1["no_window"], r1["depth_general"], r1["mean_11_14_k16"])
    elif r1.get("status") != "done" or "deep_latent" not in r1:
        rd = "pending (%s s1 %s)" % (cell, r1.get("status"))
    else:
        if r1["deep_latent"] and r2.get("status") != "done":
            rd = "s1 DEEP-LATENT at window %d..10 (s2 opens; the claim needs 2 / 2)" % lo
        elif r1["deep_latent"]:
            rd = "DEEP-LATENT (2 / 2)" if r2.get("deep_latent") else "s1 DEEP-LATENT, s2 not: not 2 / 2"
        else:
            rd = "not DEEP-LATENT (s1: DEEP at window %s, THINK-NECESSARY %s; pooled %.3f at K'=10 over %d..10)" % (r1["deep_at_window"], r1["think_necessary"], r1["pooled_window_k10"], lo)
        rd += "; DEPTH-GENERAL %s (11-14 at K'=16: %s)" % (r1["depth_general"], None if r1["mean_11_14_k16"] is None else round(r1["mean_11_14_k16"], 3))
        t = tc[1]
        if t.get("status") == "done" and t.get("mean_11_14") is not None:
            rd += "; BEYOND TEXT %s (TEXT-COT 11-14 %.3f)" % (bool(r1["depth_general"] and t["mean_11_14"] < 0.80), t["mean_11_14"])
    return runs, rd


def main():
    res = {}
    na = [S168.seed_boundary("s168_direct_na_s%d" % s, "na") for s in (1, 2)]
    ds = [v.get("D_star") for v in na if v.get("status") == "done"]
    dstar = min(d for d in ds if d is not None) if len(ds) == 2 and any(d is not None for d in ds) else None
    cur = {}
    for s in (1, 2):
        d, r, h = header("currdirect", s); cur[s] = dict(h, D_star=dstar_of(r.get("acc_by_depth") or {}) if r else None, acc_by_depth=(r or {}).get("acc_by_depth"))
    cdone = [v for v in cur.values() if v["status"] == "done"]
    dcur = min([v["D_star"] for v in cdone if v["D_star"] is not None], default=None) if cdone else None
    if cdone and all(v["D_star"] is None for v in cdone):
        dcur = 15                                                           # curriculum DIRECT >= 0.90 at every depth: no window
    res.update({"D_star": dstar, "D_star_curr": dcur, "currdirect": cur})
    tc = {}
    for s in (1,):
        d, r, h = header("textcot", s); tc[s] = dict(h, acc_by_depth=(r or {}).get("acc_by_depth"), mean_11_14=mean([acc((r or {}).get("acc_by_depth"), L) for L in range(11, 15)]) if r else None)
    res["textcot"] = tc
    if dstar is None or dcur is None:
        res["reading"] = "pending (D* %s, D*_curr %s)" % (dstar, dcur); print("READING:", res["reading"]); dump(res); return
    lo = max(dstar, dcur, 3); res["window"] = [lo, 10]
    for cell in ("stepaux", "stepauxrw"):
        runs, rd = arm_read(cell, lo, dcur, tc)
        res[cell] = runs; res["reading_" + cell] = rd; print("READING %s:" % cell, rd)
    a1, b1 = res["stepaux"][1], res["stepauxrw"][1]                          # note 168-12: the lever reading (REWEIGHT - STEP-AUX, s1), depths 3..10 at K' = 10
    if a1.get("status") == "done" and b1.get("status") == "done" and a1.get("pooled_3_10") is not None and b1.get("pooled_3_10") is not None             and a1.get("late_aux") is not None and b1.get("late_aux") is not None:
        dp, dl = b1["pooled_3_10"] - a1["pooled_3_10"], b1["late_aux"] - a1["late_aux"]
        if dp >= 0.05 - EPS and dl >= 0.10 - EPS:
            lev = "HELPS"
        elif dp <= -0.05 + EPS:
            cr, cc = b1.get("clip_rate"), a1.get("clip_rate")
            lev = "HURTS (clip-confounded)" if cr is not None and ((cc is not None and cr >= 2 * cc) or (cc is None and cr >= 0.5)) else "HURTS"
        elif abs(dp) < 0.03 and abs(dl) < 0.05:
            lev = "NO EFFECT"
        else:
            lev = "mixed"
        res["lever_168_12"] = {"d_pooled_3_10": dp, "d_late_aux_8_10": dl, "reading": lev}; print("LEVER (168-12):", lev, "dpooled %+.3f, dlate %+.3f" % (dp, dl))
    rd = res["reading_stepaux"]
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    dump(res)


def dump(res):
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s168s2.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
