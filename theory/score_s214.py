"""THEORY §214 scoring (D4 option B: one 48k PAUSE-A cell from pretrained on D3-T's recipe; registered text docs/research_d4_2026-10-16/D4_v1.md at 10187c1b
(v1 + Addendum 1); Theory is the only scorer). Cell boxes/runs_fsq/s214_pausea_s1.
Kept / claim readings (all pi-free; score_s213.cell on the §213 contract files): LENGTH-FREE, VB-C1, DEPTH-GENERAL (mean acc_n2 n 9-12 >= 0.80), plus
  K24-STABLE: eval_vb acc_k['24'] >= 0.99 at every n 1-8;
  C2-TAIL (n 13-16): mean acc_n2 >= 0.30 and THINK-GAIN = mean(acc_n2 - acc_k10) >= 0.10 and acc_k24 >= acc_n2 - 0.05 at every n;
  KEEPS-SELF-STAB: score_s213.hist24 CONTENT-STABILISES and STAB_self >= 0.5 (MEM-TRADED iff not).
  D4-C2 = DEPTH-GENERAL and LENGTH-FREE and VB-C1 and K24-STABLE; D4-C2-TAIL = D4-C2 and C2-TAIL; D4-C2-MEM = D4-C2 and KEEPS-SELF-STAB.
Mechanism, the decode-free DEADLINE-BOUND ladder (D4_v1 §5). c = 0.10. acc(K') per n from eval_vbhalt rows (coda[t] == y). FIN-K(K_end): A = acc(K_end);
  readable iff A >= 0.18; the smallest K' >= 1 with acc(K') >= c + 0.9 (A - c) and acc(K'') >= c + 0.8 (A - c) for every K'' in [K', K_end].
  L0 KEPT iff STOP-GONE fails (ACC-GAIN-AFTER-10 = mean over n 11-14 of acc_n2 - acc_k10 < 0.10).
  L1 iff STOP-GONE and NEAR-9: FIN-K (K_end n + 2) OLS slope over n 9-14 >= 0.7 and spread >= 3, readable at all six n.
  L2 iff L1 and SHIFT-PLAIN: FIN-K slope over n 12-16 >= 0.7, readable at all five n.
  L3 iff L2 and SHIFT-START-D (eval_pause start_d, n 9-12 x D {0,2,4,6,8} with n + D in [12, 20]: FIN-K (K_end n + D + 2) regressed on n + D, slope >= 0.7,
     >= 12 cells readable) and PAUSE-LEARNED and not HIDDEN-WORK.
  PAUSE-LEARNED: eval_pause pause_inv, n 1-8, Q 6, acc at K' n + Q + 2: pooled >= 0.99 and every n >= 0.97. PAUSE-EARLY / HIDDEN-WORK: ER = (mean acc(K' n + Q - 1)
     - c) / (mean acc(K' n + Q + 2) - c) pooled over n 3-8; HIDDEN-WORK iff ER > 0.30.
  Labels: KEPT (L0); MOVED (a readable rung above the reached one fails); UNRESOLVED>Lk (the next rung unreadable); REMOVED<=16 (L3).
  MOVED scan (reported): t_d = the smallest t in 9..20 with FIN-K in [t - 1, t + 1] on >= 3 readable cells (plain n, D 0; START-D (n, D)) with n + D >= t + 2.
  START-D labels (descriptive, only when the plain cell is L0; D 4 vs D 0 on n 9-12): GATED-COUNT iff the FIN-K shift >= 3 with an acc change < +0.05 at every
    readable n 9-10; PAUSE-UNLOCKS iff mean acc change over n 9-12 >= +0.10; UNGATED iff the n 9 FIN-K shift <= +1 and acc falls >= 0.05. D 0 is taken from
    eval_pause start_d["n_0"] when present, else from the plain eval_vb acc_k of the same §170 items on the same card (the D 0 arm applies no pi).
Censoring (rule b): dead (log < 48k) or NO-ENTRY (answer_ce) fails every reading; never-finishing (FIN-K unreadable) is a failure; pi-arm gates failing make L3
  UNREADABLE. Basin entry vs D3-T 11k: gap > 5k CONFOUNDED (descriptive).
ARGS-DIFF vs s213_d3t_s1/args.json: only pause / pause_p / pause_max / kmax / mem_check_steps (+ metadata) may differ.
    python theory/score_s214.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
sys.path.insert(0, os.path.join(ROOT, "theory"))
import score_s213 as S213  # noqa: E402

EPS, C = 1e-9, 0.10
CELL, BASE = "s214_pausea_s1", "s213_d3t_s1"
ALLOWED = {"pause", "pause_p", "pause_max", "kmax", "mem_check_steps"}


def jl(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def ols(xs, ys):
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n; sxx = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx > 0 else None


def fin_k(curve, k_end):
    """curve {K': acc}; -> FIN-K or None (unreadable = never finishing)."""
    if k_end not in curve:
        return None
    A = curve[k_end]
    if A < 0.18 - EPS:
        return None
    hi, lo = C + 0.9 * (A - C), C + 0.8 * (A - C)
    for k in range(1, k_end + 1):
        if k in curve and curve[k] >= hi - EPS and all(curve.get(k2, -1) >= lo - EPS for k2 in range(k, k_end + 1)):
            return k
    return None


def curves_from_vbhalt(v):
    by = {}
    for r in v["rows"]:
        by.setdefault(r["n"], []).append(r)
    out = {}
    for n, rs in by.items():
        T = min(len(r["coda"]) for r in rs)
        out[n] = {k: sum(int(r["coda"][k] == r["y"]) for r in rs) / len(rs) for k in range(T)}
    return out


def curve_from_acc_k(ak):
    return {int(k): float(v) for k, v in ak.items()}


def ladder(d, ev):
    v = jl(d, "eval_vbhalt.json"); P = jl(d, "eval_pause.json")
    if v is None:
        return {"label": "pending (eval_vbhalt.json)"}
    cur = curves_from_vbhalt(v)
    fk = {n: fin_k(cur[n], n + 2) for n in range(9, 17) if n in cur}
    gain = sum(ev["acc_n2"][str(n)] - ev["acc_k"]["10"][str(n)] for n in range(11, 15)) / 4
    stop_gone = gain >= 0.10 - EPS
    o = {"ACC_GAIN_AFTER_10": gain, "STOP_GONE": stop_gone, "FIN_K": fk}

    def rung(ns):
        vals = [fk.get(n) for n in ns]
        if any(x is None for x in vals):
            return None, None
        return ols(list(ns), vals), max(vals) - min(vals)
    s1, sp1 = rung(range(9, 15)); s2, _ = rung(range(12, 17))
    o.update(slope_9_14=s1, spread_9_14=sp1, slope_12_16=s2)
    r1 = None if s1 is None else bool(s1 >= 0.7 - EPS and sp1 >= 3 - EPS)
    r2 = None if s2 is None else bool(s2 >= 0.7 - EPS)
    # pi arms
    pl = he = None; sd_slope = None; sd_cells = []
    if P is not None:
        Q = P.get("Q", 6); pin = P.get("pause_inv") or {}
        if pin:
            accs = {int(n): curve_from_acc_k(pin[n]["acc_k"]) for n in pin}
            fin = [accs[n].get(n + Q + 2) for n in range(1, 9) if n in accs]
            pl = bool(len(fin) == 8 and all(x is not None for x in fin) and sum(fin) / 8 >= 0.99 - EPS and min(fin) >= 0.97 - EPS)
            num = [accs[n].get(n + Q - 1) for n in range(3, 9)]; den = [accs[n].get(n + Q + 2) for n in range(3, 9)]
            if all(x is not None for x in num + den) and sum(den) / 6 - C > EPS:
                er = (sum(num) / 6 - C) / (sum(den) / 6 - C); o["ER_pause"] = er; he = bool(er > 0.30 + EPS)
        sd = P.get("start_d") or {}
        for key, blk in sd.items():
            n, D = (int(x) for x in key.split("_"))
            if 9 <= n <= 12 and 12 <= n + D <= 20 and not blk.get("extra"):     # descriptive extras (e.g. D 0 at n 9-11) never enter L3 or its floor
                f = fin_k(curve_from_acc_k(blk["acc_k"]), n + D + 2); sd_cells.append((n, D, f))
        rd = [(n + D, f) for n, D, f in sd_cells if f is not None]
        if len(rd) >= 12:
            sd_slope = ols([x for x, _ in rd], [y for _, y in rd])
    o.update(PAUSE_LEARNED=pl, HIDDEN_WORK=he, start_d_cells=sd_cells, start_d_slope=sd_slope)
    r3 = None if (sd_slope is None or pl is not True or he is not False) else bool(sd_slope >= 0.7 - EPS)
    # labels
    if not stop_gone:
        lab = "KEPT (L0)"
    elif r1 is None:
        lab = "UNRESOLVED>L0 (L1 unreadable)"
    elif not r1:
        lab = "MOVED (fails L1: t_d <= 12.25)"
    elif r2 is None:
        lab = "UNRESOLVED>L1 (L2 unreadable)"
    elif not r2:
        lab = "MOVED (fails L2: t_d in (12.25, 14.5])"
    elif r3 is None:
        lab = "UNRESOLVED>L2 (L3 unreadable or pi gates failed)"
    elif not r3:
        lab = "MOVED (fails L3: t_d in (14.5, 16.5])"
    else:
        lab = "REMOVED<=16 (L3)"
    o.update(L1=r1, L2=r2, L3=r3, label=lab)
    cells = [(n, 0, fk.get(n)) for n in range(9, 17)] + sd_cells
    td = None
    for t in range(9, 21):
        hit = [1 for n, D, f in cells if f is not None and n + D >= t + 2 and t - 1 <= f <= t + 1]
        if len(hit) >= 3:
            td = t; break
    o["MOVED_t_d"] = td
    # START-D labels (D4_v1 §5; descriptive, read only when the plain cell is L0): D 4 vs D 0 on n 9-12. D 0 comes from eval_pause start_d["n_0"] when
    # present (Audit asked for D 0 cells at n 9-11 as extras), else from the plain eval_vb acc_k curve of the same §170 items on the same card (the D 0 arm
    # applies no pi, so it is the plain rollout).
    if not stop_gone and P is not None:
        sd = P.get("start_d") or {}
        def cur_of(n, D):
            k = "%d_%d" % (n, D)
            if k in sd:
                return curve_from_acc_k(sd[k]["acc_k"])
            if D == 0:
                return {int(kk): float(v[str(n)]) for kk, v in ev["acc_k"].items() if str(n) in v}
            return None
        rows = {}
        for n in range(9, 13):
            c0, c4 = cur_of(n, 0), cur_of(n, 4)
            if c0 is None or c4 is None or (n + 2) not in c0 or (n + 6) not in c4:
                continue
            f0, f4 = fin_k(c0, n + 2), fin_k(c4, n + 6)
            rows[n] = {"acc_D0": c0[n + 2], "acc_D4": c4[n + 6], "fin_D0": f0, "fin_D4": f4,
                       "shift": (f4 - f0) if (f0 is not None and f4 is not None) else None}
        lab_sd = []
        r910 = [rows[n] for n in (9, 10) if n in rows and rows[n]["shift"] is not None]
        if r910 and all(r["shift"] >= 3 - EPS and r["acc_D4"] - r["acc_D0"] < 0.05 for r in r910):
            lab_sd.append("GATED-COUNT")
        if all(n in rows for n in range(9, 13)) and sum(rows[n]["acc_D4"] - rows[n]["acc_D0"] for n in range(9, 13)) / 4 >= 0.10 - EPS:
            lab_sd.append("PAUSE-UNLOCKS")
        if 9 in rows and rows[9]["shift"] is not None and rows[9]["shift"] <= 1 + EPS and rows[9]["acc_D0"] - rows[9]["acc_D4"] >= 0.05 - EPS:
            lab_sd.append("UNGATED")
        o["START_D_rows"] = rows; o["START_D_labels"] = lab_sd or ["none"]
    return o


def args_diff(d):
    a, b = jl(d, "args.json"), jl(BASE, "args.json")
    if a is None or b is None:
        return None
    shared = {k: (b[k], a[k]) for k in a if k in b and k not in S213.META and k not in ALLOWED and a[k] != b[k]}
    new = sorted(k for k in a if k not in b and k not in ALLOWED)
    return {"shared_diffs": shared, "unexpected_new_keys": new, "ok": not shared and not new}


def main():
    out = {"cell": CELL, "ARGS_DIFF": args_diff(CELL)}
    base = S213.cell("D4", CELL); out["s213"] = base
    if "VB_C1" not in base:
        out["label"] = base.get("label", "pending")
    else:
        ev = jl(CELL, "eval_vb.json"); A2, K24, K10 = ev["acc_n2"], ev["acc_k"]["24"], ev["acc_k"]["10"]
        k24 = all(K24[str(n)] >= 0.99 - EPS for n in range(1, 9))
        tail = (sum(A2[str(n)] for n in range(13, 17)) / 4 >= 0.30 - EPS and sum(A2[str(n)] - K10[str(n)] for n in range(13, 17)) / 4 >= 0.10 - EPS
                and all(K24[str(n)] >= A2[str(n)] - 0.05 - EPS for n in range(13, 17)))
        h = base.get("HIST24"); keep_mem = bool(h and h.get("CONTENT") == "CONTENT-STABILISES" and (h["STAB"].get("self_ablate") or 0) >= 0.5 - EPS)
        dead = base["dead"] or base["no_entry"]
        c2 = bool(base["DEPTH_GENERAL"] and base["LENGTH_FREE"] and base["VB_C1"] and k24) and not dead
        lad = ladder(CELL, ev)
        e = base["entry_answer_ce_05"]
        out.update(K24_STABLE=k24, C2_TAIL=tail, KEEPS_SELF_STAB=keep_mem, MEM_TRADED=(not keep_mem) if h else None, D4_C2=c2,
                   D4_C2_TAIL=bool(c2 and tail), D4_C2_MEM=bool(c2 and keep_mem), LADDER=lad, DEPTH_9_12=base["DEPTH_9_12"],
                   entry=e, CONFOUNDED_vs_D3T=(e is None or abs(e - 11000) > 5000))
        lab = ("FAILURE (dead / NO-ENTRY; rule b)" if dead else "D4-C2-MEM" if out["D4_C2_MEM"] else "D4-C2 (memory channel lost: MEM-TRADED)" if c2 else
               "not D4-C2 (DEPTH %.3f)" % base["DEPTH_9_12"])
        out["label"] = "%s | ladder %s | K24 %s | KEEPS-SELF-STAB %s" % (lab, lad.get("label"), k24, keep_mem)
    print("READING: " + str(out["label"]))
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s214.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
