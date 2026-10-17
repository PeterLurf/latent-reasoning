"""THEORY §215 scoring (SELF-DEPTH-CHAIN round 1; registered text docs/research_selfdepth_2026-10-17/SELFDEPTH_v1.md at 5ded8703; Theory is the only scorer).
Cells boxes/runs_fsq/s215_ftself_s1 (p_self 0.5) and s215_ftctrl_s1 (p_self 0; SD's own control); shared s215_gate1/eval_gate.json, s215_pool/pool_meta.json.
Notation a(n) = eval_vb acc_n2[n] (K' n + 2); a_k(n) = eval_vb acc_k[k][n]. Per cell (v1 §4-§5):
  censoring (each fails every reading): dead (log last step < 12,000); FT-DIVERGED (a(n) < 0.99 or a_24(n) < 0.97 at any n 1-8); LABEL-NOISY (FT-SELF:
    eval_selfdepth label_audit pi_state < 0.99 or pi_label < 0.99); CONTAMINATED (eval_selfdepth contamination pool_test_hits > 0, or log self_reuse > 0);
    ARGS-DIFF mismatch vs s213_d3t_s1/args.json (only lr / steps / cur / c_sched / init_from / args_ref / self flags may differ; pause must be off).
  ABSORB: min a(n) n 9-12 >= 0.95 (in support; never C2 / C4). DEPTH-GENERAL mean n 9-12 >= 0.80 (reported, IN-SUPPORT).
  M = mean a(n) n 13-16; DBS (C2) M >= 0.80; MARGIN-GROWS M >= 0.63.
  STOP-LOC: G(k) = mean over n in [k + 1, min(k + 4, 16)] of a(n) - a_k(n), k 9..15 (no readability filter); STOP = min k with G(k) <= 0.02, else 16 ("16+");
    STOP-TO-SUPPORT iff STOP >= 13.
  K'-BEYOND: mean n 15, 16 of a(n) - a_14(n) >= 0.25. RISE-PAST-K: mean n 15, 16 of a(n) - a_16(n) (descriptive).
  STOP-MOVED: score_s213.acc_gain_after10 mean >= 0.10 (descriptive / attribution). THINK-GAIN-BEYOND, C2-TAIL, THINK-NECESSARY-DEEP descriptive.
  Mechanism (only under ABSORB, else UNLABELLED): FULL M >= 0.80; GROW 0.63 <= M < 0.80; SHIFT 0.40 <= M < 0.63 and STOP >= 13; MIXED 0.40 <= M < 0.63 and
    STOP <= 12; SHRINK M < 0.40.
  Kept: LENGTH-FREE (score_s213.no_knee and score_s212.lfk); VB-C1 (score_s211.cell_read parts with dead = 12,000 and no NO-ENTRY censor); K24-STABLE
    (a_24 >= 0.99 at n 1-8); KEEPS-SELF-STAB (score_s213.hist24 CONTENT-STABILISES and STAB_self >= 0.5; else MEM-TRADED).
  Claims: SD-SELF = ABSORB and LENGTH-FREE and VB-C1 and K24-STABLE; SD-C2 = SD-SELF and DBS; SD-C2-MEM = SD-C2 and KEEPS-SELF-STAB.
Pair (v1 §4.4): SELF-ATTRIBUTED iff SELF STOP-MOVED and CTRL not; FT-EFFECT iff CTRL STOP-MOVED; MEM attribution; CTRL missing -> UNREAD.
GATE-1 (eval_gate.json by_n {n: {n_kept, keep, pi, pi_E}}, pooled {pi, pi_E}): pi_pool >= 0.995 and pi_n >= 0.99 every n 9-12 and pi_E_pool >= 0.99 and
  keep_n >= 0.70 every n.
    python theory/score_s215.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
sys.path.insert(0, os.path.join(ROOT, "theory"))
import score_s211 as S211  # noqa: E402
import score_s212 as S212  # noqa: E402
import score_s213 as S213  # noqa: E402

EPS = 1e-9
CELLS = {"SELF": "s215_ftself_s1", "CTRL": "s215_ftctrl_s1"}
BASE, GATE, FULL = "s213_d3t_s1", "s215_gate1", 12000
ALLOWED = {"lr", "steps", "cur", "c_sched", "init_from", "args_ref", "self_pool", "p_self", "self_kmax", "self_kextra", "self_micro", "mem_check_self",
           "mem_check_steps", "pause", "pause_p", "pause_max", "keep_ckpt_steps", "stop_at"}


def jl(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def args_diff(d):
    a, b = jl(d, "args.json"), jl(BASE, "args.json")
    if a is None or b is None:
        return None
    shared = {k: (b[k], a[k]) for k in a if k in b and k not in S213.META and k not in ALLOWED and a[k] != b[k]}
    new = sorted(k for k in a if k not in b and k not in ALLOWED)
    pause_on = a.get("pause") not in (None, "none", "")
    return {"shared_diffs": shared, "unexpected_new_keys": new, "pause_on": pause_on, "ok": not shared and not new and not pause_on}


def stop_loc(ev):
    a, K = ev["acc_n2"], ev["acc_k"]; G = {}
    for k in range(9, 16):
        ns = [n for n in range(k + 1, min(k + 4, 16) + 1) if str(k) in K]
        if ns:
            G[k] = sum(a[str(n)] - K[str(k)][str(n)] for n in ns) / len(ns)
    stop = next((k for k in range(9, 16) if k in G and G[k] <= 0.02 + EPS), 16)
    return G, stop


def cell(tag, d):
    o = {"cell": d, "ARGS_DIFF": args_diff(d)}
    ev = jl(d, "eval_vb.json"); recs = S211.log_recs(d)
    if ev is None or not recs:
        o["label"] = "pending (not landed)" if not recs else "pending (eval_vb.json)"; return o
    need = [k for k in ("acc_n2", "acc_k", "probe", "test_meta", "vswap", "stall", "stepcap") if k not in ev]
    if need:
        o["label"] = "pending (eval_vb partial: %s)" % need; return o
    a, K = ev["acc_n2"], ev["acc_k"]; last = max(recs)
    dead = last < FULL
    reuse = max((r.get("self_reuse") or 0) for r in recs.values())
    diverged = any(a[str(n)] < 0.99 - EPS or K["24"][str(n)] < 0.97 - EPS for n in range(1, 9))
    sd = jl(d, "eval_selfdepth.json") or {}
    la = sd.get("label_audit") or {}
    noisy = (tag == "SELF") and bool(la) and (la.get("pi_state", 0) < 0.99 - EPS or la.get("pi_label", 0) < 0.99 - EPS)
    contam = bool((sd.get("contamination") or {}).get("pool_test_hits", 0) > 0 or reuse > 0)
    ad = o["ARGS_DIFF"]; args_bad = bool(ad is not None and not ad["ok"])
    c = S211.cell_read(tag, d)                                             # VB-C1 parts (its 48k dead / NO-ENTRY censors are ignored here)
    vb_c1 = bool(c.get("DEEP") and c.get("PROBE_GAIN") and (c.get("onset") or {}).get("SERIAL_ONSET") and c.get("SERIAL_STEP") and c.get("NO_CLOCK"))
    sp = jl(d, "eval_steppos.json"); lf = S212.lfk(d)
    lfree = bool(sp and S213.no_knee(sp)["NO_KNEE"] and lf and lf["MEETS"])
    k24 = all(K["24"][str(n)] >= 0.99 - EPS for n in range(1, 9))
    h = S213.hist24(d); keep_mem = bool(h and h.get("CONTENT") == "CONTENT-STABILISES" and (h["STAB"].get("self_ablate") or 0) >= 0.5 - EPS)
    absorb = min(a[str(n)] for n in range(9, 13)) >= 0.95 - EPS
    M = sum(a[str(n)] for n in range(13, 17)) / 4
    G, stop = stop_loc(ev)
    kb = sum(a[str(n)] - K["14"][str(n)] for n in (15, 16)) / 2
    rise = sum(a[str(n)] - K["16"][str(n)] for n in (15, 16)) / 2
    gain10 = (S213.acc_gain_after10(d) or {}).get("mean_gain")
    mech = ("UNLABELLED (not ABSORB)" if not absorb else "FULL" if M >= 0.80 - EPS else "GROW" if M >= 0.63 - EPS else
            ("SHIFT" if stop >= 13 else "MIXED") if M >= 0.40 - EPS else "SHRINK")
    censored = dead or diverged or noisy or contam or args_bad
    sd_self = bool(absorb and lfree and vb_c1 and k24) and not censored
    sd_c2 = bool(sd_self and M >= 0.80 - EPS)
    o.update(last_step=last, dead=dead, FT_DIVERGED=diverged, LABEL_NOISY=noisy, CONTAMINATED=contam, ARGS_BAD=args_bad, censored=censored,
             ABSORB=absorb, DEPTH_GENERAL=sum(a[str(n)] for n in range(9, 13)) / 4 >= 0.80 - EPS, M=M, DBS=M >= 0.80 - EPS, MARGIN_GROWS=M >= 0.63 - EPS,
             STOP_G=G, STOP=stop, STOP_TO_SUPPORT=stop >= 13, K_BEYOND=kb, K_BEYOND_MET=kb >= 0.25 - EPS, RISE_PAST_K=rise,
             STOP_MOVED=None if gain10 is None else bool(gain10 >= 0.10 - EPS), gain_after10=gain10,
             THINK_GAIN_BEYOND=sum(a[str(n)] - K["12"][str(n)] for n in range(13, 17)) / 4,
             C2_TAIL=all(K["24"][str(n)] >= a[str(n)] - 0.05 - EPS for n in range(13, 17)),
             THINK_NECESSARY_DEEP=all(max(K["0"][str(n)], K["1"][str(n)]) <= 0.15 + EPS for n in range(9, 13)),
             MECHANISM=mech, LENGTH_FREE=lfree, VB_C1=vb_c1, K24_STABLE=k24, KEEPS_SELF_STAB=keep_mem, MEM_TRADED=(not keep_mem) if h else None,
             SD_SELF=sd_self, SD_C2=sd_c2, SD_C2_MEM=bool(sd_c2 and keep_mem), MEMO=sd.get("memo"), label_audit=la)
    why = [w for w, f in (("dead", dead), ("FT-DIVERGED", diverged), ("LABEL-NOISY", noisy), ("CONTAMINATED", contam), ("ARGS-DIFF", args_bad)) if f]
    o["label"] = ("FAILURE (%s; rule b)" % ", ".join(why) if censored else
                  "SD-C2-MEM" if o["SD_C2_MEM"] else "SD-C2 (MEM-TRADED)" if sd_c2 else "SD-SELF (in support; not C2)" if sd_self else
                  "not SD-SELF") + " | %s | M %.3f | STOP %s | K'-BEYOND %.3f" % (mech, M, "16+" if stop == 16 else stop, kb)
    return o


def gate1():
    g = jl(GATE, "eval_gate.json")
    if g is None:
        return None
    B, P = g["by_n"], g["pooled"]
    ok = bool(P["pi"] >= 0.995 - EPS and P["pi_E"] >= 0.99 - EPS and all(str(n) in B and B[str(n)]["pi"] >= 0.99 - EPS and B[str(n)]["keep"] >= 0.70 - EPS
                                                                       for n in range(9, 13)))
    return {"GATE_1": ok, "pooled": P, "by_n": {n: {k: B[n].get(k) for k in ("n_kept", "keep", "pi", "pi_E")} for n in sorted(B, key=int)}}


def main():
    out = {t: cell(t, d) for t, d in CELLS.items()}
    S, C = out["SELF"], out["CTRL"]
    if "STOP_MOVED" in S and "STOP_MOVED" in C:
        out["PAIR"] = {"SELF_ATTRIBUTED": bool(S["STOP_MOVED"] and not C["STOP_MOVED"]), "FT_EFFECT": bool(C["STOP_MOVED"]),
                       "MEM": None if (S.get("censored") or C.get("censored")) else ("traded by the self items" if (S.get("MEM_TRADED") and not C.get("MEM_TRADED")) else
                               "traded by the FT (lr re-heat)" if (S.get("MEM_TRADED") and C.get("MEM_TRADED")) else "kept" if S.get("KEEPS_SELF_STAB") else None),
                       "EROSION_FT": bool(S.get("FT_DIVERGED") and C.get("FT_DIVERGED"))}
    else:
        out["PAIR"] = {"label": "UNREAD (CTRL missing or pending)"}
    out["GATE_1"] = gate1()
    print("READING: SELF %s || CTRL %s || PAIR %s || GATE-1 %s" % (S.get("label"), C.get("label"), out["PAIR"], (out["GATE_1"] or {}).get("GATE_1")))
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s215.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
