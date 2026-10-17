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
             HALT=S212.halt_read(jl(d, "eval_vbhalt.json")) if jl(d, "eval_vbhalt.json") else None, STEP_BY_T=step_by_t(d))   # 213-17 add-on: descriptive (C5), not part of D3-C2
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
    ref["STEP_BY_T_N1"] = step_by_t(S212.STAGE0["N1"])                    # note 213-18 reference                         # note 213-14 S0-4
    out["REFERENCE"] = ref
    print("READING: " + " || ".join("%s: %s" % (t, out[t]["label"]) for t in CELLS) + " || PAIR: " + out["PAIR"]["label"])
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s213.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
