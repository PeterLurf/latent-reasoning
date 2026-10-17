"""THEORY §213 scoring (D3 = SLVA on VERSION-BANK; registered text docs/research_bottleneck_2026-10-16/D3_v1.md at c9f3f980 (v1.1) + note 213-1's
field contract; Theory is the only scorer). Cells: D3-T s213_d3t_s1 (SLVA, table trained) and D3-F s213_d3f_s1 (SLVA + frozen pretrained table).
Per cell:
  censoring (rule b): dead (log < 48k) or NO-ENTRY (answer_ce, score_s211's entry rule) fails every reading.
  VB-C1: score_s211.cell_read (DEEP, PROBE-GAIN, SERIAL-ONSET, VALUE-SWAP SERIAL-STEP, NO-CLOCK).
  NO-KNEE (eval_steppos.json fresh_m_T): for m 9 and m 10 separately, pooled exact at T >= 33 >= pooled exact at T <= 29 - 0.05; each side >= 100 items
    (else UNREADABLE = not met).  COUNT (reported): min over m 9-12 of fresh_m exact >= 0.95.  STEP-GENERAL (stretch): fresh_m exact >= 0.95 at m 12-16.
  LENGTH-FREE-K (eval_lfk.json): score_s212.lfk (all-kind strict per-slot ge33 >= le28 - 0.01; >= 500 events at ge33).  LENGTH-FREE = NO-KNEE and LFK.
    Note 213-1: LFK counts only if LFK_DISCRIMINATES and LFK_SENSITIVE on the §211 cells (lfk_valid); else every cell reads pending.
  DEPTH-GENERAL: eval_vb acc_n2 mean over n 9-12 >= 0.80.  SELF-SIMILAR (reported): score_s212.rs_read on eval_rs.json, n 9-12, r in {2,4,6,8}: readable
    and |D| <= 0.05 (control within 0.03).
  HISTORY-USED (eval_d3hist.json): mean over n 6-8 of acc_n2 intact - bank_ablate_slva >= 0.10; UNREADABLE (not met) if the intact mean < 0.90 or the
    intact arm differs from eval_vb's acc_n2 at any n 6-8. Beside: SELF-ABLATE / OTHER-ABLATE drops; swap triple; RNN-IN-FUNCTION = newest_follow >= 0.95
    and old_follow <= 0.02 and not HISTORY-USED; OLD-SWAP follow is quoted beside HISTORY-USED.
  D3-C2 = LENGTH-FREE and DEPTH-GENERAL and VB-C1 ("C2 with the memory in form only"); D3-C2-ATTN = D3-C2 and HISTORY-USED ("C2 in a pass-global
    attention memory").
Pair (descriptive): TABLE = D3-F - D3-T on STEP-GENERAL mean m 12-16, DEPTH n 9-12 and LFK acc_ge33; entry steps; entry gap > 5k = CONFOUNDED.
ARGS-DIFF (reported): each cell's args.json vs s211_vb_pa_s1/args.json over shared keys outside the metadata whitelist; new keys listed.
    python theory/score_s213.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
sys.path.insert(0, os.path.join(ROOT, "theory"))
import score_s211 as S211  # noqa: E402
import score_s212 as S212  # noqa: E402

EPS = 1e-9
CELLS = {"D3-T": "s213_d3t_s1", "D3-F": "s213_d3f_s1"}
N2 = "s211_vb_pa_s1"
META = {"mode", "arch", "out", "device", "train_gpu", "train_driver", "torch_version", "cuda_version", "cudnn_version", "python_version", "image_tag",
        "eval_bs", "micro", "auto_micro", "mem_budget", "ckpt_every", "keep_ckpt_steps", "time_steps", "mem_check", "stop_at", "resume"}


def jl(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def no_knee(sp):
    G = sp["fresh_m_T"]; out = {}
    for m in (9, 10):
        lo = [(c["n_steps"], c["exact"]) for k, c in G.items() if int(k.split("_")[0]) == m and int(k.split("_")[1]) <= 29]
        hi = [(c["n_steps"], c["exact"]) for k, c in G.items() if int(k.split("_")[0]) == m and int(k.split("_")[1]) >= 33]
        nl, nh = sum(n for n, _ in lo), sum(n for n, _ in hi)
        el, eh = S212.wmean(lo), S212.wmean(hi)
        readable = nl >= 100 and nh >= 100
        out[str(m)] = {"n_le29": nl, "exact_le29": el, "n_ge33": nh, "exact_ge33": eh, "readable": readable,
                       "met": bool(readable and eh >= el - 0.05 - EPS)}
    out["NO_KNEE"] = all(out[str(m)]["met"] for m in (9, 10))
    return out


def hist(h, ev):
    mean = lambda arm: sum(h[arm]["acc_n2"][str(n)] for n in (6, 7, 8)) / 3  # noqa: E731
    gate = all(h["intact"]["acc_n2"][str(n)] == ev["acc_n2"][str(n)] for n in (6, 7, 8))
    I = mean("intact"); readable = bool(gate and I >= 0.90 - EPS)
    drop = {a: (I - mean(a)) if a in h else None for a in ("bank_ablate_slva", "self_ablate", "other_ablate")}
    # note 213-11 (Audit): an omitted arm reads None (unreadable), never 0 / not-met-by-default
    used = None if drop["bank_ablate_slva"] is None else bool(readable and drop["bank_ablate_slva"] >= 0.10 - EPS)
    sw = h.get("swap") or {}
    sw_ok = sw.get("n_pairs", 0) >= 100 and sw.get("newest_follow") is not None and sw.get("old_follow") is not None
    rnn = None if (not sw_ok or used is None) else bool(sw["newest_follow"] >= 0.95 - EPS and sw["old_follow"] <= 0.02 + EPS and not used)
    ds, do = drop["self_ablate"], drop["other_ablate"]                     # note 213-13: descriptive pattern of a HISTORY-USED cell
    pat = (None if used is not True else "OWN-SLOT (N2-like)" if (ds is not None and ds >= 0.10 - EPS) else
           "CORE/CODA" if (do is not None and do >= 0.10 - EPS) else
           "REDUNDANT (self and other each < 0.10)" if (ds is not None and do is not None) else "unresolved (arm omitted)")
    return {"intact_n6_8": I, "gate_intact_equals_eval_vb": gate, "readable": readable, "drop": drop, "HISTORY_USED": used, "HIST_PATTERN": pat, "swap": sw, "omitted": h.get("omitted"),
            "OLD_SWAP_follow": sw.get("old_follow"), "RNN_IN_FUNCTION": rnn}


def hold_read(d):
    """Note 213-14 S0-4 (N1): G_H = pooled (hold - intact) over the 212-18 cells (n 9-12, r in {4, 6}, >= 100 items); D_R from the same cells' restart;
    HOLD-FIXES iff G_H >= 0.5 D_R and hold acc_n2 >= intact - 0.02 at every n 1..8 (else not met; control failure = UNREADABLE)."""
    f = jl(d, "eval_hold.json")
    if f is None:
        return None
    C = f["cells"]; sel = [C[k] for k in ("%d_%d" % (n, r) for n in range(9, 13) for r in (4, 6)) if k in C and C[k]["n_items"] >= 100]
    pool = lambda arm: S212.wmean([(c["n_items"], c[arm] - c["intact"]) for c in sel if c.get(arm) is not None])  # noqa: E731
    dr, gh, ghv = pool("restart"), pool("hold"), pool("hold_v")
    A = f["acc_n2"]; ctl = all(A["hold"][str(n)] >= A["intact"][str(n)] - 0.02 - EPS for n in range(1, 9))
    fixes = bool(ctl and dr is not None and dr >= 0.20 - EPS and gh is not None and gh >= 0.5 * dr - EPS)
    return {"D_R": dr, "G_H": gh, "G_HV": ghv, "control": ctl, "cells_read": len(sel), "HOLD_FIXES": fixes,
            "label": "UNREADABLE (control)" if not ctl else ("HOLD-FIXES" if fixes else "not met")}


def step_by_t(d):
    """Note 213-18 (descriptive): exact-input in-trajectory step exactness by pass index t, pooled over n 9-16 (cells < 200 events excluded).
    PASS-COUNT-KEYED iff pooled t <= 8 >= 0.97 and pooled t >= 10 <= 0.90; GRADUAL-DRIFT iff pooled t 2-4 <= 0.95 or (t 2-4) - (t 5-8) >= 0.03 (and not
    PASS-COUNT-KEYED); else MIXED."""
    f = jl(d, "eval_stepbyt.json") or {}
    st = f.get("step_by_t") or (jl(d, "eval_lfk.json") or {}).get("step_by_t")
    if not st:
        return None
    cells = [(int(t_), c["events"], c["strict_exact"]) for n_, T_ in st.items() if 9 <= int(n_) <= 16 for t_, c in T_.items() if c["events"] >= 200]
    def pool(lo, hi):
        e = sum(ev for t_, ev, _ in cells if lo <= t_ <= hi); s = sum(x for t_, _, x in cells if lo <= t_ <= hi)
        return (s / e) if e else None
    a8, a10, a24, a58 = pool(2, 8), pool(10, 99), pool(2, 4), pool(5, 8)
    curve = {t_: pool(t_, t_) for t_ in sorted({c[0] for c in cells})}
    ch = {}                                                                 # Audit on 213-18: changed-slot accuracy by t, and the n mix per t
    for n_, T_ in st.items():
        if 9 <= int(n_) <= 16:
            for t_, c in T_.items():
                z = ch.setdefault(int(t_), [0, 0, {}]); z[0] += c.get("changed_events", 0); z[1] += c.get("changed_correct", 0); z[2][n_] = c["events"]
    changed_curve = {t_: ((z[1] / z[0]) if z[0] else None) for t_, z in sorted(ch.items())}
    n_mix = {t_: z[2] for t_, z in sorted(ch.items())}
    pc = bool(a8 is not None and a10 is not None and a8 >= 0.97 - EPS and a10 <= 0.90 + EPS)
    gd = bool(not pc and a24 is not None and (a24 <= 0.95 + EPS or (a58 is not None and a24 - a58 >= 0.03 - EPS)))
    return {"t_le8": a8, "t_ge10": a10, "t2_4": a24, "t5_8": a58, "curve": curve, "changed_curve": changed_curve, "n_mix_by_t": n_mix,
            "label": "PASS-COUNT-KEYED" if pc else ("GRADUAL-DRIFT" if gd else "MIXED")}


def hist24(d):
    """Note 213-20 (descriptive): STAB_arm = mean over n 3..6 of [(acc_n2 - acc_k24) arm] - [(acc_n2 - acc_k24) intact]; HISTORY-STABILISES iff STAB_bank
    >= 0.10 and the intact arm equals eval_vb (acc_n2, acc_k['24']) bitwise."""
    f, ev = jl(d, "eval_hist24.json"), jl(d, "eval_vb.json")
    if f is None or ev is None:
        return None
    gate = all(f["intact"]["acc_n2"][str(n)] == ev["acc_n2"][str(n)] and f["intact"]["acc_k24"][str(n)] == ev["acc_k"]["24"][str(n)] for n in range(3, 9))
    g = lambda arm, n: f[arm]["acc_n2"][str(n)] - f[arm]["acc_k24"][str(n)]  # noqa: E731
    st = {a: ((sum(g(a, n) - g("intact", n) for n in range(3, 7)) / 4) if a in f else None)
          for a in ("bank_ablate_slva", "self_ablate", "other_ablate", "dup_newest", "dup_foreign")}   # dup_foreign: 213-22, descriptive
    stab = bool(gate and st["bank_ablate_slva"] is not None and st["bank_ablate_slva"] >= 0.10 - EPS)
    sd = st["dup_newest"]                                                  # note 213-21: content control
    content = None if (not stab or sd is None) else ("CONTENT-STABILISES" if sd >= 0.10 - EPS else "COUNT-LIKE" if sd < 0.05 else "MIXED")
    lab = ("UNREADABLE (gate)" if not gate else "not met: stability without memory history (Audit, 213-20)" if not stab else
           "HISTORY-STABILISES, content control pending (213-21)" if content is None else
           "memory used causally for long-run stability (CONTENT-STABILISES)" if content == "CONTENT-STABILISES" else
           "stability depends on the bank's size, not its history (COUNT-LIKE)" if content == "COUNT-LIKE" else "HISTORY-STABILISES, MIXED content control")
    return {"gate": gate, "STAB": st, "HISTORY_STABILISES": stab if gate else None, "CONTENT": content, "omitted": f.get("omitted"), "label": lab}


def edge_collapse(d):
    """Note 213-24 (descriptive; D4 draft S0-1, docs/research_d4_2026-10-16/scripts/d4_edge_collapse.py's read()): on-schedule pass-events at t
    (alive(E_{t-1}) == n - t + 1 >= 3), n 10..14 pooled, from eval_vbhalt rows; h(t) = P(alive drop >= 2); t_fin = first t with alive 0.
    EDGE-COLLAPSE iff events(t 7..9) >= 300, h(7..9) >= 0.30, h(8) >= 2 h(5), median t_fin over n 10 / 11 / 12 within 1 pass.
    ORS_6_8 = sum h(t) events / sum (1 - e(t)) events over t 6..8, e = eval_stepbyt strict_exact pooled n 10..16 (t with >= 200 events)."""
    v = jl(d, "eval_vbhalt.json")
    if v is None or "rows" not in v:
        return None
    ev, tf = {}, {}
    for r in v["rows"]:
        n, al = r["n"], r["alive"]
        if 10 <= n <= 14:
            for t_ in range(2, len(al)):
                if al[t_ - 1] == n - t_ + 1 and al[t_ - 1] >= 3:
                    e_ = ev.setdefault(t_, [0, 0]); e_[0] += 1; e_[1] += int(al[t_ - 1] - al[t_] >= 2)
        if 9 <= n <= 16:
            tz = next((t_ for t_ in range(len(al)) if al[t_] == 0), None); tf.setdefault(n, []).append(99 if tz is None else tz)
    h = {t_: x[1] / x[0] for t_, x in ev.items() if x[0] >= 30}
    e79 = sum(ev.get(t_, [0, 0])[0] for t_ in (7, 8, 9)); c79 = sum(ev.get(t_, [0, 0])[1] for t_ in (7, 8, 9)); h79 = c79 / e79 if e79 else None
    med = {n: float(sorted(x)[len(x) // 2]) if len(x) % 2 else float(sum(sorted(x)[len(x) // 2 - 1:len(x) // 2 + 1]) / 2) for n, x in tf.items()}
    ok = e79 >= 300 and all(k in med for k in (10, 11, 12))
    ec = bool(ok and h79 >= 0.30 - EPS and h.get(8, 0) >= 2 * h.get(5, 1e9) and max(med[k] for k in (10, 11, 12)) - min(med[k] for k in (10, 11, 12)) <= 1)
    o = {"h_by_t": h, "h_7_9": h79, "events_7_9": e79, "median_t_fin": med, "readable": ok, "EDGE_COLLAPSE": ec if ok else None}
    sb = (jl(d, "eval_stepbyt.json") or {}).get("step_by_t")
    if sb:
        num = den = 0.0
        for t_ in (6, 7, 8):
            E = sum(sb.get(str(n), {}).get(str(t_), {}).get("events", 0) for n in range(10, 17)); X = sum(sb.get(str(n), {}).get(str(t_), {}).get("strict_exact", 0) for n in range(10, 17))
            if E >= 200 and t_ in h:
                num += h[t_] * E; den += (1 - X / E) * E
        o["ORS_6_8"] = num / den if den else None; o["ORD"] = None if o["ORS_6_8"] is None else bool(o["ORS_6_8"] >= 0.5 - EPS)
    return o


def acc_gain_after10(d):
    """Note 213-26 (descriptive, decode-free; REDTEAM_D4 blocker 2): per n 11..14, acc_n2 - acc at K' 10, and acc at K' 24 - acc at K' 10, from eval_vb acc_k;
    STOPS-AT-10 iff the mean of (acc_n2 - acc_k10) over n 11..14 is <= 0.02 (accuracy does not rise after pass 10)."""
    ev = jl(d, "eval_vb.json")
    if ev is None or "acc_k" not in ev or "10" not in ev["acc_k"]:
        return None
    K = ev["acc_k"]; g = {n: ev["acc_n2"][str(n)] - K["10"][str(n)] for n in range(11, 15)}; g24 = {n: K["24"][str(n)] - K["10"][str(n)] for n in range(11, 15)}
    m = sum(g.values()) / 4
    return {"gain_n2_minus_k10": g, "gain_k24_minus_k10": g24, "mean_gain": m, "STOPS_AT_10": bool(m <= 0.02 + EPS)}


def ageproj(d):
    """Note 213-27 (D4 Stage 0 premise test, descriptive): AGE-READABLE iff fit r2_heldout >= 0.5; G_arm = mean over n 11..14 of
    [acc_n2 - acc_k10] under the arm minus the same intact; control: arm acc_n2 >= intact - 0.02 at every n 1..8; AGE-UNLOCKS iff AGE-READABLE, age_clamp
    control holds, G_age_clamp >= +0.05 and G_rand_clamp <= +0.02; gate: |intact - eval_vb| <= 0.005 (acc_n2 n 1..14 and acc_k10)."""
    f, ev = jl(d, "eval_ageproj.json"), jl(d, "eval_vb.json")
    if f is None or ev is None:
        return None
    A = f["arms"]; I = A["intact"]
    gate = all(abs(I["acc_n2"][str(n)] - ev["acc_n2"][str(n)]) <= 0.005 + EPS for n in range(1, 15)) and         all(abs(I["acc_k10"][str(n)] - ev["acc_k"]["10"][str(n)]) <= 0.005 + EPS for n in range(1, 15))
    gi = lambda a: sum(A[a]["acc_n2"][str(n)] - A[a]["acc_k10"][str(n)] for n in range(11, 15)) / 4  # noqa: E731
    G = {a: gi(a) - gi("intact") for a in A if a != "intact"}
    ctl = {a: all(A[a]["acc_n2"][str(n)] >= I["acc_n2"][str(n)] - 0.02 - EPS for n in range(1, 9)) for a in A if a != "intact"}
    readable = f["fit"]["r2_heldout"] >= 0.5 - EPS
    unlocks = bool(gate and readable and ctl.get("age_clamp") and G.get("age_clamp", -1) >= 0.05 - EPS and G.get("rand_clamp", 1) <= 0.02 + EPS)
    lab = ("UNREADABLE (gate)" if not gate else "NO-LINEAR-AGE (r2 %.3f)" % f["fit"]["r2_heldout"] if not readable else
           "UNREADABLE (age_clamp control)" if not ctl.get("age_clamp") else "AGE-UNLOCKS" if unlocks else "not met")
    return {"gate": gate, "r2": f["fit"]["r2_heldout"], "r2_t_minus_n": f["fit"].get("r2_t_minus_n"), "r2_t_given_n": f["fit"].get("r2_t_given_n"),
            "G": G, "control": ctl, "AGE_UNLOCKS": unlocks,
            "depth_9_12": {a: sum(A[a]["acc_n2"][str(n)] for n in range(9, 13)) / 4 for a in A}, "label": lab}


def d4_stage0(d):
    """D4 v1 §7 Stage 0 readers (pre-stated before output). S0-3 eval_agerevert.json {arm: {D: {acc_k_pre {n}, acc_n2 {n}}}, intact {acc_n2}}:
    per arm (revert / mct): control = acc_n2 at D 2 >= 0.97 at every n 1..4; HIDDEN-WORK = ER pooled n 3..6 at D 4 > 0.30, ER = (mean acc_pre - 0.1) /
    (mean acc_n2 - 0.1); readable = control and not HIDDEN-WORK; drop78 = mean intact acc_n2 n 7..8 - mean arm acc_n2 n 7..8 at D 4; ARM-NULL iff readable
    and drop78 < 0.10. S0-2 eval_carrier.json {prestall {s: {acc_n2 {n}}}, foreign {acc_n2 {n}}}: BANK-BORNE iff drop at s 8 (pooled n 6..8) >= 0.15;
    BANK-NULL iff drop at s 8 <= 0.05 and FOREIGN o 6 drop (n 6..8) <= 0.05. AGE-CLOCK iff a readable arm has drop78 >= 0.20 and prestall s 4 drops n 7..8
    by <= 0.05. PROGRESS-ONLY (hard stop) iff both arms readable and ARM-NULL, BANK-NULL, and 213-27 AGE_UNLOCKS is False. Intact gates: |intact - eval_vb|
    <= 0.005 at every n."""
    ev = jl(d, "eval_vb.json"); ar, ca, ap = jl(d, "eval_agerevert.json"), jl(d, "eval_carrier.json"), ageproj(d)
    if ev is None or (ar is None and ca is None):
        return None
    E = ev["acc_n2"]; mean = lambda A, ns: sum(A[str(n)] for n in ns) / len(ns)  # noqa: E731
    o = {}
    if ar is not None:
        I = ar["intact"]["acc_n2"]; o["s03_gate"] = all(abs(I[str(n)] - E[str(n)]) <= 0.005 + EPS for n in range(1, 9)); arms = {}
        for arm in ("revert", "mct"):
            if arm not in ar:
                arms[arm] = None; continue
            A2, A4 = ar[arm]["2"], ar[arm]["4"]
            ctl = all(A2["acc_n2"][str(n)] >= 0.97 - EPS for n in range(1, 5))
            den = mean(A4["acc_n2"], range(3, 7)) - 0.1
            er = ((mean(A4["acc_k_pre"], range(3, 7)) - 0.1) / den) if den > EPS else None
            hidden = er is None or er > 0.30 + EPS
            drop78 = mean(I, (7, 8)) - mean(A4["acc_n2"], (7, 8)); readable = bool(ctl and not hidden and o["s03_gate"])
            arms[arm] = {"control": ctl, "ER": er, "HIDDEN_WORK": hidden, "readable": readable, "drop78": drop78,
                         "ARM_NULL": bool(readable and drop78 < 0.10 - EPS)}
        o["S0_3"] = arms
    if ca is not None:
        P = ca["prestall"]; F = ca["foreign"]["acc_n2"]; I2 = E
        d8 = mean(I2, (6, 7, 8)) - mean(P["8"]["acc_n2"], (6, 7, 8)); d4_78 = mean(I2, (7, 8)) - mean(P["4"]["acc_n2"], (7, 8))
        df = mean(I2, (6, 7, 8)) - mean(F, (6, 7, 8))
        o["S0_2"] = {"drop_s8": d8, "drop_s4_78": d4_78, "drop_foreign": df, "BANK_BORNE": bool(d8 >= 0.15 - EPS),
                     "BANK_NULL": bool(d8 <= 0.05 + EPS and df <= 0.05 + EPS)}
    a3, b2 = o.get("S0_3"), o.get("S0_2")
    if a3 and b2:
        o["AGE_CLOCK"] = bool(any(v and v["readable"] and v["drop78"] >= 0.20 - EPS for v in a3.values()) and b2["drop_s4_78"] <= 0.05 + EPS)
        both_null = all(v and v["readable"] and v["ARM_NULL"] for v in a3.values())
        o["PROGRESS_ONLY"] = bool(both_null and b2["BANK_NULL"] and ap is not None and ap["AGE_UNLOCKS"] is False)
    return o


def args_diff(d):
    a, b = jl(d, "args.json"), jl(N2, "args.json")
    if a is None or b is None:
        return None
    shared = {k: (b[k], a[k]) for k in a if k in b and k not in META and a[k] != b[k]}
    return {"shared_diffs": shared, "new_keys": sorted(k for k in a if k not in b), "mode": a.get("mode"), "ok": not shared}


def lfk_valid():
    """Note 213-1: (a') counts only if LFK_DISCRIMINATES (N1 meets; N2 and N3 do not, on the §211 cells) and LFK_SENSITIVE (N1's FAULT arm)."""
    L = {k: S212.lfk(v) for k, v in S212.STAGE0.items()}
    if any(v is None for v in L.values()) or L["N1"].get("SENSITIVE") is None:
        return None
    return bool(L["N1"]["MEETS"] and not L["N2"]["MEETS"] and not L["N3"]["MEETS"] and L["N1"]["SENSITIVE"])


def cell(tag, d):
    o = {"cell": d, "ARGS_DIFF": args_diff(d)}
    a = S211.cell_read(tag, d); o["s211"] = a
    if "DEEP" not in a:                                                     # pending / partial eval_vb.json: never scored
        o["label"] = a.get("label"); return o
    ev = jl(d, "eval_vb.json"); sp, lf, rs, h = jl(d, "eval_steppos.json"), S212.lfk(d), jl(d, "eval_rs.json"), jl(d, "eval_d3hist.json")
    miss = [n for n, v in (("eval_steppos", sp), ("eval_lfk", lf), ("eval_rs", rs), ("eval_d3hist", h)) if v is None]
    if miss:
        o["label"] = "pending (missing %s)" % miss; return o
    dead, noent = a["dead"], a["entry_answer_ce_05"] is None
    nk = no_knee(sp); fm = sp["fresh_m"]
    count = min(fm[str(m)]["exact"] for m in range(9, 13)) >= 0.95 - EPS
    sg = all(fm[str(m)]["exact"] >= 0.95 - EPS for m in range(12, 17)); sg_mean = sum(fm[str(m)]["exact"] for m in range(12, 17)) / 5
    valid = lfk_valid()
    if not valid:                                                           # note 213-1: (a') not validated -> LENGTH-FREE pending an amendment
        o["LFK_VALID"] = valid; o["label"] = "pending (LENGTH-FREE-K not validated: LFK_DISCRIMINATES and LFK_SENSITIVE needed; note 213-1)"; return o
    lfree = bool(nk["NO_KNEE"] and lf["MEETS"])
    dg = a["DEPTH_9_12"] >= 0.80 - EPS
    ss = S212.rs_read(rs, range(9, 13), (2, 4, 6, 8)); selfsim = bool(ss and ss["readable"] and abs(ss["D"]) <= 0.05 + EPS)
    hr = hist(h, ev)
    c2 = bool(lfree and dg and a["VB_C1"]); c2a = bool(c2 and hr["HISTORY_USED"] is True)
    if dead or noent:
        c2 = c2a = lfree = dg = False
    o.update(dead=dead, no_entry=noent, entry_answer_ce_05=a["entry_answer_ce_05"], VB_C1=a["VB_C1"], NO_KNEE=nk, LFK=lf, LENGTH_FREE=lfree, COUNT=count,
             STEP_GENERAL=sg, STEP_GENERAL_mean_12_16=sg_mean, DEPTH_9_12=a["DEPTH_9_12"], DEPTH_GENERAL=dg, RS=ss, SELF_SIMILAR=selfsim, HIST=hr,
             D3_C2=c2, D3_C2_ATTN=c2a,
             HALT=S212.halt_read(jl(d, "eval_vbhalt.json")) if jl(d, "eval_vbhalt.json") else None, STEP_BY_T=step_by_t(d),
             HIST24=hist24(d), EDGE=edge_collapse(d), ACC_GAIN_AFTER_10=acc_gain_after10(d))   # 213-17 add-on: descriptive (C5), not part of D3-C2
    o["label"] = ("FAILURE (dead; rule b)" if dead else "NO-ENTRY (rule b)" if noent else
                  "D3-C2-ATTN (C2 in a pass-global attention memory)" if c2a else
                  "D3-C2 (C2 with the memory in form only)" if c2 else
                  "parts: LENGTH-FREE %s / DEPTH-GENERAL %s (%.3f) / VB-C1 %s / HISTORY-USED %s" % (lfree, dg, a["DEPTH_9_12"], a["VB_C1"], hr["HISTORY_USED"]))
    o["label"] += " | OLD-SWAP follow %s%s%s" % (hr["OLD_SWAP_follow"], " | RNN-IN-FUNCTION" if hr["RNN_IN_FUNCTION"] else "",
                                                (" | history: " + hr["HIST_PATTERN"]) if hr["HIST_PATTERN"] else "")
    return o


def pair(T, F):
    if "D3_C2" not in T or "D3_C2" not in F:
        return {"label": "pending"}
    if T["dead"] or T["no_entry"] or F["dead"] or F["no_entry"]:
        return {"label": "censored (a cell is dead or NO-ENTRY; rule b)"}
    et, ef = T["entry_answer_ce_05"], F["entry_answer_ce_05"]
    conf = et is None or ef is None or abs(et - ef) > 5000
    return {"TABLE_step_general_12_16": F["STEP_GENERAL_mean_12_16"] - T["STEP_GENERAL_mean_12_16"], "TABLE_depth_9_12": F["DEPTH_9_12"] - T["DEPTH_9_12"],
            "TABLE_lfk_ge33": (F["LFK"]["acc_ge33"] or 0) - (T["LFK"]["acc_ge33"] or 0),
            "TABLE_lfk_29_32": (F["LFK"]["acc_29_32"] or 0) - (T["LFK"]["acc_29_32"] or 0),  # note 213-3: descriptive (N1's rare-row dip)
            "lfk_changed_ge33": {"T": T["LFK"]["by_kind"].get("changed_ge33"), "F": F["LFK"]["by_kind"].get("changed_ge33")},
            "entry_T": et, "entry_F": ef, "CONFOUNDED": conf,
            "label": "descriptive (one seed each)%s" % (" - CONFOUNDED (entry gap > 5k)" if conf else "")}


def main():
    out = {t: cell(t, d) for t, d in CELLS.items()}
    out["PAIR"] = pair(out["D3-T"], out["D3-F"])
    ref = {}                                                                # note 213-11: d3hist references on the §211 cells (harness check, descriptive)
    for k, d in S212.STAGE0.items():
        h, ev = jl(d, "eval_d3hist.json"), jl(d, "eval_vb.json")
        ref[k] = hist(h, ev) if (h and ev) else None
        if ref[k] and k in ("N2", "N3") and (ref[k]["HIST_PATTERN"] or "").startswith("REDUNDANT"):   # 213-13: PA arms lack a cross-slot-only arm
            ref[k]["HIST_PATTERN"] = "REDUNDANT or CROSS-SLOT GATHER (PA arms cannot separate; 213-13)"
    exp = {"N1": lambda r: r["HISTORY_USED"] is False and r["RNN_IN_FUNCTION"] is True,   # explicit True / False: None (omitted) fails
           "N2": lambda r: r["HISTORY_USED"] is True and r["RNN_IN_FUNCTION"] is False, "N3": lambda r: r["HISTORY_USED"] is True}
    ref["EXPECTATIONS_MET"] = None if any(ref[k] is None for k in exp) else all(exp[k](ref[k]) for k in exp)
    ref["HOLD_N1"] = hold_read(S212.STAGE0["N1"])
    ref["STEP_BY_T_N1"] = step_by_t(S212.STAGE0["N1"])
    ref["HIST24"] = {k: hist24(S212.STAGE0[k]) for k in ("N1", "N2")}
    ref["EDGE"] = {k: edge_collapse(S212.STAGE0[k]) for k in ("N1", "N2")}
    ref["ACC_GAIN_AFTER_10"] = {k: acc_gain_after10(S212.STAGE0[k]) for k in ("N1", "N2", "N3")}   # note 213-26   # note 213-24 references         # note 213-20 references                    # note 213-18 reference                         # note 213-14 S0-4
    ref["AGEPROJ_D3T"] = ageproj(CELLS["D3-T"])                          # note 213-27 (D4 Stage 0)
    ref["D4_STAGE0_D3T"] = d4_stage0(CELLS["D3-T"])                        # D4 v1 §7 S0-3 / S0-2 / PROGRESS-ONLY
    out["REFERENCE"] = ref
    print("READING: " + " || ".join("%s: %s" % (t, out[t]["label"]) for t in CELLS) + " || PAIR: " + out["PAIR"]["label"])
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s213.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
