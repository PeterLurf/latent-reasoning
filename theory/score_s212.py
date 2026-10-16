"""THEORY §212 scoring (VB-2G Stage 0 + Stage 1; registered text docs/THEORY_VB2G_DRAFT_2026-10-16.md at the commit named in §212, plus the
pre-stated Stage 0 notes 211-13 / 211-15 / 211-16 and §212's notes; Theory is the only scorer).

Part A, Stage 0 on the §211 cells (N2 s211_vb_pa_s1 = base, N1 s211_vb_carry_s1, N3 s211_vb_pamct_s1); descriptive labels and the go rules:
  RS (eval_rs.json): pooled n 11-16, r in {4,6}: D = RESTART - INTACT; control = n <= 8, all r (|D| <= 0.03 for a readable RS);
    HISTORY-KEYED D >= 0.20; HISTORY-FREE |D| <= 0.05; cells < 100 items UNREADABLE. HEAD-LIMITED: LONG-STEP(12..16) < 0.95 while LONG-STEP(<= 8) >= 0.99.
  HALT (eval_vbhalt.json, 211-13): HALT-EXACT pooled n 3..8 exact >= 0.90; NOT-PREMATURE acc_halt >= acc_n2 - 0.02 (n 3..8), - 0.05 (n 9..12);
    STALL-SHIFT shift >= 0.85 with qualifying >= 200; DEPTH-TIMING n 9 and n 10 each exact >= 0.60, halt_at_8 <= 0.10, frozen_nm8 <= 0.10;
    CLOCK-HOLD mean frozen_nm8 over n 9-12 >= 0.5; EDGE-COLLAPSE mean halt_at_8 over n 9-12 >= 0.5; STALL-COUNT frozen1 >= 0.5; NO-DONE never > 0.10 at n 3..8.
  PF (eval_pf.json): cells n_items >= 100; clock gap = weighted mean (S - S0[r]) over k + r > 8 minus the same over k + r <= 8 (in-support);
    CLOCK-KEYED gap <= -0.15 with in-support |mean| <= 0.03; POSITION-KEYED weighted OLS slope of S on s (within k) <= -0.01 per slot;
    CONTENT-KEYED |gap| <= 0.05 and |slope| * (s range) <= 0.05.
  HIST (eval_vb.json bank_ablate vs acc_n2): HISTORY-HURTS n 9-12 mean gain >= 0.10 with n <= 8 change >= -0.03; HISTORY-USED drop >= 0.03 (n <= 8)
    or >= 0.05 (n 9-16); HISTORY-IGNORED |change| <= 0.02 on both. EVAL-HISTORY-GAIN (eval_hist_arms.json): an arm raises n 9-12 by >= 0.10, n <= 8 >= -0.03.
  FOREIGN (eval_foreign.json): CLOCK-FOREIGN insert - replace <= -0.10 (mean over o and n 4..12).  REPAIR (eval_repair.json): >= 0.2.
  STEP trigger (eval_step_rows.json): 5-fold-by-item held-out dLL of adding hh to (t, alive, held, prov, age_max, bdepth) >= 5 nats / 1000 events AND the
    hh coefficient (per SD) among prov >= 1 rows >= +0.5.
  Go rules: G1_GO = not (every read model is HISTORY-FREE and HEAD-LIMITED); FT_GO = base N2 reads CLOCK-HOLD / EDGE-COLLAPSE / STALL-COUNT (and is DEEP);
    S_GO = the STEP trigger fires on a DEEP §211 cell.
  D3 Stage 0 (note 212-15; D3_v1.md): D3SEL (N1, eval_d3sel.json) ADD-VOIDMASK; LFK (eval_lfk.json, each cell) LENGTH-FREE-K and LFK_DISCRIMINATES
    (N1 meets, N2 and N3 do not); STABHIST (N2, eval_stabhist.json) OWN-HISTORY-STABILISES.
Part B, G1 s212_vb2g_s1: VB2G-C1 = (a) DEEP, PROBE-GAIN, SERIAL-ONSET, SERIAL-STEP (score_s211's definitions) and (b) DEPTH-GENERAL (mean acc_n2 n 9-12
  >= 0.80), SELF-SIMILAR (RS pooled n 9-16 all r |D| <= 0.05, control <= 0.03, every cell >= 100), STEP-FLAT (eval_stepflat.json) and (c) PF CONTENT-KEYED,
  NO-CLOCK (STALL >= 0.9) and (d) C5-VB. Reported: HISTORY-USE, KEEP-ACC (cap 0.995 at n 13-16). Censoring: dead (< 48k) or NO-ENTRY on true_answer_ce.
Part C, FT pair s212_ft_aug_s1 / s212_ft_ctrl_s1: AUG-BEATS = AUG DEPTH-TIMING met and CTRL not; CLOCK-HOLD per arm; NOT-PREMATURE n 9-12 per arm.
    python theory/score_s212.py
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
sys.path.insert(0, os.path.join(ROOT, "theory"))
import score_s211 as S211  # noqa: E402

EPS = 1e-9
STAGE0 = {"N2": "s211_vb_pa_s1", "N1": "s211_vb_carry_s1", "N3": "s211_vb_pamct_s1"}
G1, FT_AUG, FT_CTRL = "s212_vb2g_s1", "s212_ft_aug_s1", "s212_ft_ctrl_s1"


def jl(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def wmean(pairs):
    w = sum(n for n, _ in pairs)
    return sum(n * v for n, v in pairs) / w if w else None


def rs_read(rs, ns, rset, ctrl_ns=range(3, 9)):
    if rs is None:
        return None
    C = rs["cells"]; cell = lambda n, r: C.get("%d_%d" % (n, r))
    sel = [cell(n, r) for n in ns for r in rset if r < n and cell(n, r)]
    thin = [c for c in sel if c["n_items"] < 100]
    d = wmean([(c["n_items"], c["restart"] - c["intact"]) for c in sel if c["n_items"] >= 100])
    ctl = [cell(n, r) for n in ctrl_ns for r in (2, 4, 6, 8) if r < n and cell(n, r)]
    dctl = wmean([(c["n_items"], c["restart"] - c["intact"]) for c in ctl if c["n_items"] >= 100])
    ls = rs.get("long_step") or {}
    ls_lo = [ls[str(m)]["exact"] for m in range(1, 9) if str(m) in ls]; ls_hi = [ls[str(m)]["exact"] for m in range(12, 17) if str(m) in ls]
    head = bool(ls_lo and ls_hi and min(ls_lo) >= 0.99 - EPS and min(ls_hi) < 0.95)
    readable = d is not None and dctl is not None and abs(dctl) <= 0.03 + EPS                 # thin cells (< 100) are excluded, not fatal (211-15; note 212-2)
    return {"D": d, "D_control": dctl, "thin_cells": len(thin), "readable": readable, "HISTORY_KEYED": bool(readable and d >= 0.20 - EPS),
            "HISTORY_FREE": bool(readable and abs(d) <= 0.05 + EPS), "HEAD_LIMITED": head}


def halt_read(h):
    if h is None:
        return None
    B = h["by_n"]; g = lambda n, k: B[str(n)][k]
    exact_38 = wmean([(g(n, "items"), g(n, "exact")) for n in range(3, 9)])
    notprem = all(g(n, "acc_halt") >= g(n, "acc_n2") - 0.02 - EPS for n in range(3, 9)) and all(g(n, "acc_halt") >= g(n, "acc_n2") - 0.05 - EPS for n in range(9, 13))
    st = h["stall"]; ss = st["qualifying"] >= 200 and st["shift"] >= 0.85 - EPS
    dt = all(g(n, "exact") >= 0.60 - EPS and g(n, "halt_at_8") <= 0.10 + EPS and g(n, "frozen_nm8") <= 0.10 + EPS for n in (9, 10))
    o = {"HALT_EXACT": bool(exact_38 is not None and exact_38 >= 0.90 - EPS), "exact_3_8": exact_38, "NOT_PREMATURE": bool(notprem),
         "STALL_SHIFT": bool(ss), "stall_shift": st["shift"], "stall_qualifying": st["qualifying"], "DEPTH_TIMING": bool(dt)}
    o["C5_VB"] = o["HALT_EXACT"] and o["NOT_PREMATURE"] and o["STALL_SHIFT"] and o["DEPTH_TIMING"]
    o["CLOCK_HOLD"] = sum(g(n, "frozen_nm8") for n in range(9, 13)) / 4 >= 0.5 - EPS
    o["EDGE_COLLAPSE"] = sum(g(n, "halt_at_8") for n in range(9, 13)) / 4 >= 0.5 - EPS
    o["STALL_COUNT"] = st["frozen1"] >= 0.5 - EPS
    o["NO_DONE"] = any(g(n, "never") > 0.10 + EPS for n in range(3, 9))
    return o


def pf_read(pf):
    if pf is None:
        return None
    G, S0 = pf["grid"], pf["S0"]; cells = []
    for key, c in G.items():
        parts = key.split("_"); k = int(c.get("k", parts[0])); s = int(c.get("s", parts[1]))       # key "k_s" or "k_s_r" (note 212-2)
        if c["n_items"] >= 100 and str(c["r"]) in S0:
            cells.append((k, s, c["r"], c["n_items"], c["S"] - S0[str(c["r"])], c["S"]))
    ins = wmean([(n, d) for k, s, r, n, d, _ in cells if k + r <= 8]); out = wmean([(n, d) for k, s, r, n, d, _ in cells if k + r > 8])
    gap = None if ins is None or out is None else out - ins
    num = den = 0.0; smin, smax = 1e9, -1e9                                 # weighted within-k OLS slope of S on s
    for k in set(c[0] for c in cells):
        ck = [c for c in cells if c[0] == k]
        if len(ck) < 2:
            continue
        w = [c[3] for c in ck]; sm = sum(wi * c[1] for wi, c in zip(w, ck)) / sum(w); ym = sum(wi * c[5] for wi, c in zip(w, ck)) / sum(w)
        num += sum(wi * (c[1] - sm) * (c[5] - ym) for wi, c in zip(w, ck)); den += sum(wi * (c[1] - sm) ** 2 for wi, c in zip(w, ck))
        smin, smax = min(smin, min(c[1] for c in ck)), max(smax, max(c[1] for c in ck))
    slope = num / den if den > 0 else None
    clock = bool(gap is not None and gap <= -0.15 + EPS and ins is not None and abs(ins) <= 0.03 + EPS)
    pos = bool(slope is not None and slope <= -0.01 + EPS)
    content = bool(gap is not None and abs(gap) <= 0.05 + EPS and slope is not None and abs(slope) * (smax - smin) <= 0.05 + EPS)
    return {"gap": gap, "in_support": ins, "slope": slope, "CLOCK_KEYED": clock, "POSITION_KEYED": pos, "CONTENT_KEYED": content, "cells": len(cells)}


def hist_read(ev, arms):
    if ev is None:
        return None
    a, b = ev["acc_n2"], (ev.get("bank_ablate") or {}).get("acc_n2")
    o = {}
    if b:
        lo = sum(b[str(n)] - a[str(n)] for n in range(1, 9)) / 8; hi = sum(b[str(n)] - a[str(n)] for n in range(9, 13)) / 4
        hi16 = sum(b[str(n)] - a[str(n)] for n in range(9, 17)) / 8
        o.update(trunc_change_le8=lo, trunc_change_9_12=hi, HISTORY_HURTS=bool(hi >= 0.10 - EPS and lo >= -0.03 - EPS),
                 HISTORY_USED=bool(lo <= -0.03 + EPS or hi16 <= -0.05 + EPS), HISTORY_IGNORED=bool(abs(lo) <= 0.02 + EPS and abs(hi16) <= 0.02 + EPS))
    if arms:
        gains = {}
        for k, v in arms.items():
            if k not in ("v0off", "floor2", "recent1") or not isinstance(v, dict):     # files carry descriptive keys ("def", card, ...)
                continue
            c = v["acc_n2"]; lo = sum(c[str(n)] - a[str(n)] for n in range(1, 9)) / 8; hi = sum(c[str(n)] - a[str(n)] for n in range(9, 13)) / 4
            gains[k] = {"le8": lo, "9_12": hi, "gain": bool(hi >= 0.10 - EPS and lo >= -0.03 - EPS)}
        o["arms"] = gains; o["EVAL_HISTORY_GAIN"] = any(g["gain"] for g in gains.values())
    return o


def foreign_read(f):
    if f is None:
        return None
    d = []
    for o in f["insert"]:
        if o in f["replace"]:
            ai, ar = f["insert"][o]["acc_n2"], f["replace"][o]["acc_n2"]
            d += [ai[str(n)] - ar[str(n)] for n in range(4, 13) if str(n) in ai and str(n) in ar]
    m = sum(d) / len(d) if d else None
    return {"insert_minus_replace": m, "CLOCK_FOREIGN": bool(m is not None and m <= -0.10 + EPS)}


def logit_fit(X, y, lam=1e-3, it=60):
    X1 = np.c_[np.ones(len(X)), X]; w = np.zeros(X1.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X1 @ w))); W = p * (1 - p) + 1e-9
        H = X1.T @ (X1 * W[:, None]) + lam * np.eye(X1.shape[1]); w = w + np.linalg.solve(H, X1.T @ (y - p) - lam * w)
    return w


def ll(X, y, w):
    p = np.clip(1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w))), 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def step_trigger(rows):
    if not rows:
        return None
    base = ["t", "alive", "held", "prov", "age_max", "bdepth"]
    X0 = np.array([[r[k] for k in base] for r in rows], float); h = np.array([r["hh"] for r in rows], float)
    y = 1.0 - np.array([r["correct"] for r in rows], float)                  # model the error
    mu, sd = X0.mean(0), X0.std(0) + 1e-9; Z0 = (X0 - mu) / sd; hz = (h - h.mean()) / (h.std() + 1e-9); Z1 = np.c_[Z0, hz]
    items = np.array([r["item"] for r in rows]); uniq = np.unique(items); rng = np.random.default_rng(21212); fold = dict(zip(uniq, rng.integers(0, 5, len(uniq))))
    f = np.array([fold[i] for i in items]); dll = 0.0
    for k in range(5):
        tr, te = f != k, f == k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        dll += ll(Z1[te], y[te], logit_fit(Z1[tr], y[tr])) - ll(Z0[te], y[te], logit_fit(Z0[tr], y[tr]))
    per1k = 1000.0 * dll / len(rows)
    sel = np.array([r["prov"] >= 1 for r in rows])
    coef = None
    if sel.sum() >= 50 and len(np.unique(y[sel])) > 1:
        coef = float(logit_fit(Z1[sel], y[sel])[-1])
    return {"events": len(rows), "dLL_per_1k": per1k, "hh_coef_prov1": coef, "FIRES": bool(per1k >= 5 - EPS and coef is not None and coef >= 0.5 - EPS)}


def d3diag(d):
    """Note 212-11 (+ 212-12 pooling): item-weighted (n_steps-weighted) pooling over cells; target T in {31, 33}, m 8..10; exposed control m <= 8, T 25..29.
    Reports per-cell n for the target and the count of control cells where the arm differs from intact (the exposure proof)."""
    f = jl(d, "eval_d3diag.json")
    if f is None:
        return None
    I = f["intact"]["fresh_m_T"]
    def sel(G, tgt):
        out = []
        for k, c in G.items():
            m, T = (int(x) for x in k.split("_"))
            if (tgt and T in (31, 33) and 8 <= m <= 10) or ((not tgt) and m <= 8 and 25 <= T <= 29):
                out.append((k, c["n_steps"], c["exact"]))
        return out
    def pool(rows):
        w = sum(n for _, n, _ in rows); return sum(n * e for _, n, e in rows) / w if w else None
    res = {"target_cells_n": {k: n for k, n, _ in sel(I, True)}}
    it, ic = pool(sel(I, True)), pool(sel(I, False))
    for arm in ("gs24", "wpe_b1", "wpe_b2"):
        if arm not in f:
            continue
        G = f[arm]["fresh_m_T"]; at, ac = pool(sel(G, True)), pool(sel(G, False))
        diff = sum(1 for k, _, e in sel(G, False) if k in I and abs(I[k]["exact"] - e) > 0)
        rise, cdrop = (at - it), (ac - ic)
        unread = cdrop < -0.02 - EPS
        lab = "UNREADABLE (exposed control drops %.3f)" % cdrop if unread else ("BOTTLENECK / CAUSAL" if rise >= 0.30 - EPS else ("NOT" if rise < 0.10 else "INTERMEDIATE"))
        res[arm] = {"target": at, "target_intact": it, "rise": rise, "control": ac, "control_intact": ic, "control_change": cdrop,
                    "control_cells_differing": diff, "label": lab}
    return res


def d3sel(d):
    """Note 212-15 S0-1 (N1): D_R = pooled (restart - intact) over n 9-12, r in {4, 6}, cells >= 100 items, item-weighted; G_MV = the same pooling of
    (maskvoid - intact); readable iff >= 1 cell and D_R >= 0.20 (note 212-16; else UNREADABLE, no add-on); ADD-VOIDMASK iff readable, G_MV >= 0.5 D_R and acc_n2(maskvoid) >= acc_n2(intact) - 0.02 at every n 1..8. Descriptive: REFRESH share
    = pooled (refresh - intact) / D_R; GAP share = (D_R - pooled (gapped - intact)) / D_R, read only if gapped_supported and the gapped control (n 5-8, r in
    {2, 4}, pooled gapped - restart >= -0.02) holds; COMPACTION-TRIGGER (D3 v1 2.4) = GAP share >= 0.5."""
    f = jl(d, "eval_d3sel.json")
    if f is None:
        return None
    C = f["cells"]; keys = ["%d_%d" % (n, r) for n in range(9, 13) for r in (4, 6)]
    sel = [C[k] for k in keys if k in C and C[k]["n_items"] >= 100]
    pool = lambda arm: wmean([(c["n_items"], c[arm] - c["intact"]) for c in sel if c.get(arm) is not None])  # noqa: E731
    dr, gmv, grf = pool("restart"), pool("maskvoid"), pool("refresh")
    A = f["acc_n2"]; ctl = all(A["maskvoid"][str(n)] >= A["intact"][str(n)] - 0.02 - EPS for n in range(1, 9))
    readable = bool(len(sel) >= 1 and dr is not None and dr >= 0.20 - EPS)     # note 212-16 (Audit): else UNREADABLE, no add-on
    add = bool(readable and gmv is not None and gmv >= 0.5 * dr - EPS and ctl)
    gap = None; gctl = None
    if f.get("gapped_supported"):
        G = f.get("gapped_control") or {}
        gc = [G[k] for k in ("%d_%d" % (n, r) for n in range(5, 9) for r in (2, 4)) if k in G and G[k]["n_items"] >= 100]
        gctl = wmean([(c["n_items"], c["gapped"] - c["restart"]) for c in gc])
        dg = pool("gapped")
        if gctl is not None and gctl >= -0.02 - EPS and dg is not None and dr is not None and dr > EPS:
            gap = (dr - dg) / dr
    return {"D_R": dr, "G_MV": gmv, "maskvoid_control": ctl, "cells_read": len(sel), "readable": readable, "ADD_VOIDMASK": add,
            "REFRESH_share": (grf / dr) if (grf is not None and dr and dr > EPS) else None, "gapped_control": gctl, "GAP_share": gap,
            "COMPACTION_TRIGGER": bool(gap is not None and gap >= 0.5 - EPS)}


def lfk(d):
    """Note 212-15 S0-2 / D3 v1 5 (a'): LENGTH-FREE-K = all-kind strict per-slot accuracy at ge33 >= the same at le28 - 0.01; readable with >= 500
    slot-events at ge33 (else UNREADABLE = not met). Note 213-1: with a "fault" arm (S0-2b, N1), SENSITIVE iff the fault arm is readable, its ge33 accuracy is
    <= intact's - 0.02, and it reads not-met."""
    f = jl(d, "eval_lfk.json")
    if f is None:
        return None
    def read(S):
        acc = lambda k: (S[k]["strict_correct"] / S[k]["events"]) if (k in S and S[k]["events"]) else None  # noqa: E731
        lo, hi, n_hi = acc("all_le28"), acc("all_ge33"), (S.get("all_ge33") or {}).get("events", 0)
        readable = bool(n_hi >= 500 and lo is not None and hi is not None)
        return {"acc_le28": lo, "acc_29_32": acc("all_29_32"), "acc_ge33": hi, "events_ge33": n_hi, "readable": readable,
                "by_kind": {k: acc(k) for k in sorted(S)}, "MEETS": bool(readable and hi >= lo - 0.01 - EPS)}
    o = read(f["slot"])
    sens = lambda fo: bool(fo["readable"] and o["acc_ge33"] is not None and fo["acc_ge33"] <= o["acc_ge33"] - 0.02 + EPS and not fo["MEETS"])  # noqa: E731
    if "fault" in f:                                                        # note 213-1 S0-2b: the positive control (N1)
        fo = read(f["fault"]["slot"]); fo["paired"] = f["fault"].get("paired"); fo["realised"] = f["fault"].get("realised"); o["fault"] = fo
        o["SENSITIVE"] = sens(fo)
        rz = (f["fault"].get("realised") or {}).get("ge33") or {}
        low = bool(rz and fo["events_ge33"] and rz.get("faults_class_differs", 0) < 0.03 * fo["events_ge33"])
        o["fault_low_realised"] = low
        if not o["SENSITIVE"] and low and "fault2" in f:                    # note 213-2: one re-run at p 0.10 decides
            f2 = read(f["fault2"]["slot"]); f2["paired"] = f["fault2"].get("paired"); f2["realised"] = f["fault2"].get("realised")
            o["fault2"] = f2; o["SENSITIVE"] = sens(f2)
        elif not o["SENSITIVE"] and low:
            o["SENSITIVE"] = None                                           # pending the 213-2 re-run
    return o


def stabhist(d):
    """Note 212-15 S0-3 (N2): STAB = mean over n 3..6 of [(acc_n2 - acc_k24) self_ablate] - [(acc_n2 - acc_k24) intact]; OWN-HISTORY-STABILISES iff
    STAB >= 0.10 and the intact arm equals eval_vb's acc_n2 / acc_k['24'] exactly (the gate; else UNREADABLE). other_ablate's STAB is descriptive."""
    f = jl(d, "eval_stabhist.json"); ev = jl(d, "eval_vb.json")
    if f is None:
        return None
    gate = bool(ev and all(f["intact"]["acc_n2"][str(n)] == ev["acc_n2"][str(n)] and f["intact"]["acc_k24"][str(n)] == ev["acc_k"]["24"][str(n)]
                           for n in range(3, 9)))
    g = lambda arm, n: f[arm]["acc_n2"][str(n)] - f[arm]["acc_k24"][str(n)]  # noqa: E731
    st = lambda arm: (sum(g(arm, n) - g("intact", n) for n in range(3, 7)) / 4) if arm in f else None  # noqa: E731
    s_self = st("self_ablate")
    return {"gate_intact_equals_eval_vb": gate, "STAB_self": s_self, "STAB_other": st("other_ablate"),
            "OWN_HISTORY_STABILISES": bool(gate and s_self is not None and s_self >= 0.10 - EPS),
            "label": "UNREADABLE (intact gate)" if not gate else ("OWN-HISTORY-STABILISES" if s_self >= 0.10 - EPS else "not supported")}


def stage0():
    out = {}
    for tag, d in STAGE0.items():
        ev = jl(d, "eval_vb.json")
        rep = jl(d, "eval_repair.json"); rows = (jl(d, "eval_step_rows.json") or {}).get("rows")
        out[tag] = {"RS": rs_read(jl(d, "eval_rs.json"), range(11, 17), (4, 6)), "HALT": halt_read(jl(d, "eval_vbhalt.json")),
                    "PF": pf_read(jl(d, "eval_pf.json")), "HIST": hist_read(ev, jl(d, "eval_hist_arms.json")), "FOREIGN": foreign_read(jl(d, "eval_foreign.json")),
                    "REPAIR": None if rep is None else {"p": rep["p_right_after_wrong"], "n_events": rep.get("n_events"),
                                                        "GE_0_2": (rep["p_right_after_wrong"] is not None and rep["p_right_after_wrong"] >= 0.2 - EPS),
                                                        "informative": bool(rep.get("n_events") and rep["n_events"] >= 30)},
                    "STEP": step_trigger(rows), "DEEP": bool(ev and all(ev["acc_n2"][str(n)] >= 0.90 - EPS for n in range(3, 9))),
                    "D3DIAG": d3diag(d), "D3SEL": d3sel(d), "LFK": lfk(d), "STABHIST": stabhist(d)}
    rsd = [v["RS"] for v in out.values() if v["RS"]]
    g1 = None if not rsd else (not all(r["HISTORY_FREE"] and r["HEAD_LIMITED"] for r in rsd))
    b = out["N2"]; ft = None if b["HALT"] is None else bool(b["DEEP"] and (b["HALT"]["CLOCK_HOLD"] or b["HALT"]["EDGE_COLLAPSE"] or b["HALT"]["STALL_COUNT"]))
    stp = [v["STEP"]["FIRES"] for v in out.values() if v["STEP"] and v["DEEP"]]
    out["GO"] = {"G1": g1, "FT": ft, "S": (any(stp) if stp else None), "RS_models_read": len(rsd)}
    L = {k: out[k]["LFK"] for k in ("N1", "N2", "N3")}                      # 212-15 S0-2: the reading must discriminate before D3 uses it
    out["D3"] = {"ADD_VOIDMASK": (out["N1"]["D3SEL"] or {}).get("ADD_VOIDMASK"),
                 "LFK_DISCRIMINATES": (None if any(v is None for v in L.values()) else bool(L["N1"]["MEETS"] and not L["N2"]["MEETS"] and not L["N3"]["MEETS"])),
                 "LFK_SENSITIVE": (out["N1"]["LFK"] or {}).get("SENSITIVE"),
                 "OWN_HISTORY_STABILISES": (out["N2"]["STABHIST"] or {}).get("OWN_HISTORY_STABILISES")}
    return out


def g1_read():
    ev = jl(G1, "eval_vb.json")
    if ev is None:
        return {"label": "pending (not landed)"}
    a = S211.cell_read("G1", G1)                                             # (a) + censoring via score_s211's definitions (true_answer_ce: see note 212-1)
    recs = S211.log_recs(G1); e05 = S211.entry_on(recs, "true_answer_ce", 0.5); last = max(recs) if recs else 0
    dead, noent = last < S211.FULL, e05 is None
    depth = sum(ev["acc_n2"][str(n)] for n in range(9, 13)) / 4
    rs = rs_read(jl(G1, "eval_rs.json"), range(9, 17), (2, 4, 6, 8)); sf = jl(G1, "eval_stepflat.json")
    stepflat = bool(sf and sf.get("strata_dropped_frac", 1) <= 0.20 + EPS and sf["per_step_13_16_reweighted"] >= sf["per_step_le8"] - 0.02 - EPS and sf["per_step_13_16_reweighted"] >= 0.98 - EPS)
    pf = pf_read(jl(G1, "eval_pf.json")); h = halt_read(jl(G1, "eval_vbhalt.json"))
    part_a = bool(a.get("DEEP") and a.get("PROBE_GAIN") and (a.get("onset") or {}).get("SERIAL_ONSET") and a.get("SERIAL_STEP"))
    part_b = bool(depth >= 0.80 - EPS and rs and rs["readable"] and abs(rs["D"]) <= 0.05 + EPS and stepflat)
    part_c = bool(pf and pf["CONTENT_KEYED"] and a.get("NO_CLOCK"))
    part_d = bool(h and h["C5_VB"])
    ka = ev.get("keep_acc") or {}; kcap = [ka[str(n)] for n in range(13, 17) if str(n) in ka]
    o = {"A_C1": part_a, "B_C2": part_b, "C_clock": part_c, "D_C5": part_d, "DEPTH_9_12": depth, "RS": rs, "STEP_FLAT": stepflat, "PF": pf, "HALT": h,
         "HIST": hist_read(ev, None), "KEEP_ACC_13_16_min": min(kcap) if kcap else None, "KEEP_CAP": bool(kcap and min(kcap) < 0.995),
         "dead": dead, "no_entry_true_ce": noent, "s211_style": a.get("label")}
    o["VB2G_C1"] = bool(part_a and part_b and part_c and part_d and not dead and not noent)
    o["label"] = ("FAILURE (dead; rule b)" if dead else "NO-ENTRY (true_answer_ce; rule b)" if noent else
                  "VB2G-C1" if o["VB2G_C1"] else "parts: C1 %s / C2 %s / clock %s / C5 %s" % (part_a, part_b, part_c, part_d))
    return o


def ft_read():
    A, C = jl(FT_AUG, "eval_vbhalt.json"), jl(FT_CTRL, "eval_vbhalt.json")
    if A is None or C is None:
        return {"label": "pending"}
    ha, hc = halt_read(A), halt_read(C)
    return {"AUG": ha, "CTRL": hc, "AUG_BEATS": bool(ha["DEPTH_TIMING"] and not hc["DEPTH_TIMING"]),
            "label": "AUG-BEATS" if (ha["DEPTH_TIMING"] and not hc["DEPTH_TIMING"]) else "no AUG advantage on DEPTH-TIMING"}


def main():
    out = {"stage0": stage0(), "G1": g1_read(), "FT": ft_read()}
    print("READING: GO %s | D3 %s | G1 %s | FT %s" % (out["stage0"]["GO"], out["stage0"]["D3"], out["G1"]["label"], out["FT"]["label"]))
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s212.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
