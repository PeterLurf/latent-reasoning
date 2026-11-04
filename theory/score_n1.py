"""THEORY §271 N1 SHUFFLED-OBJECTS scorer (N1_v1 = docs/research_n1_2026-11-04/N1_v1.md; Theory is the only scorer; committed with fixtures before any S1 output).
Arms: LOOP-SI x 3 (of_s1 init, K 8, step-index cue); DIRECT-SI x 3 (one pass, also trained on 'After swap t, who has o*?' t <= m:
matched intermediate supervision; Note 271-1, the BINDING control); DIRECT-OF x 3 (of_s1 trunk, one pass; descriptive X1-ANS after Note 271-1);
descriptive LOOP-EN, TEXT-CoT, GENERIC.

Two modes
  --stage s1  (N1_v1 §5 S1) -> theory/results/score_n1_s1.json
      GO gate on the pilot (RUNS['pilot'], 4k gate eval), all four rows:
        L1p  eval_acc_so  by_cell/5/{1..4}/NUM n_correct / n_items >= 0.90 each m
        CWS  eval_ws_so   by_m/4/cws/{2,3,4} >= 0.90 each t (n: by_m/4/n/t)
        M3p  eval_swap_so by_cell/5_4_2: follow_cf >= 0.50 and >= mean_floor + 0.30; n_pairs >= 1000 else VOID cell
        M4p  eval_kcurve_so not_decoder: acc_vs_s2 >= 0.80 and acc_vs_s4 <= 0.05 (Note 271-1) (n 5 m 4, K' 2, S_2 != S_4 stratum)
      GO if all four; PARK if L1p < 0.50 at any m; else EXTEND (or NO-GO when --extended: the 8k re-gate already failed).
      DIRECT gate: DIRECT-SI kappa(n 5, m 8 MOVED) >= 0.70 in >= 2/3 seeds -> DEPTH-SPLIT, else DIRECT-ALIVE; DIRECT-OF reported beside.
  default (N1_v1 §6) -> theory/results/score_n1.json
      Per LOOP-SI seed, binding rows (Wilson 95% CI + NEAR on every cell; letters on raw values):
        L1   NUM n 5 m 1-4 acc >= 0.95 each m
        C2-1 NUM n 5 MOVED acc_moved m 6 >= 0.85, m 8 >= 0.80; read only if M3 passes (a seed failing M3 counts FAIL for C2-1)
        KC   kcurve acc_moved K'0, K'1 <= f(5,8) + 0.05; K'4 <= 0.302; K'8 - K'4 >= 0.40
        M1   ws by_m/8 cws t 5-8 >= 0.90        M2 swap 5_8_4      M3 swap 5_4_2 (same law as M3p)
        M4   not_decoder (as M4p)                M5 acc(k_touch >= 5) - acc(k_touch <= 2) >= -0.10 at n 5 m 8 NUM (all items)
        H1   long by_m/{1..4}: acc K'63 and K'64 >= acc K'8 - 0.02
      Seed LOOP-PASS iff all of L1 C2-1 KC M1-M5 H1; loop arm passes iff >= 2/3 seeds are LOOP-PASS.
      DIRECT-SI: D-L1 (NUM n 5 m 1-4 >= 0.90 each m) in >= 2/3 seeds, else X1 = VOID-CONTROL.
                 X1 kappa(5, 8) <= 0.25 in 3/3 seeds. X2 per seed: kappa(3,8) - kappa(5,8) >= 0.15 and kappa(3,8) >= 0.15 and
                 Wilson LB acc_moved(3,8) > f(3,8); TH6-SUPPORTED iff 3/3 (never gates N1-PASS).
      DIRECT-OF: the same rows, descriptive only (X1-ANS, TH6 DIRECT-OF); its missing files are listed, never pending.
      Labels: DEPTH-SPLIT (loop arm passes, DIRECT gate fired) > N1-PASS (loop arm + X1 with D-L1) > N1-LOOP-ONLY (X1 FAIL or
              VOID-CONTROL) ; else FAIL: <rows failing 2/3>.
              ORD-SCOPE: L2 (ORD n 5 m 1-4 >= 0.90 each m) in >= 2/3 seeds, independent of N1-PASS; precondition of B1.
              C2-KPRIME: C2-K (by_m/10: acc_moved K'12 >= 0.60 and >= K'8 + 0.20) in >= 2/3 seeds.
              B1 (conditional): binding only if ORD-SCOPE and C2-3 (ORD n 5 m 8 acc >= 0.60) in >= 2/3 seeds; then BBH five
                  (unparsed items counted wrong) >= 0.60 in >= 2/3 seeds.
              Routing for a seed whose C2-1 counts FAIL: LOOKUP-BOUND if lookup any L 5-12 < 0.80, else NORM-DRIFT-CANDIDATE if
                  median r_8 outside [0.67, 1.5] (kcurve s_norm at m 8 or m 12) and ws by_m/8 cws[4] - cws[8] > 0.10.
              Flags: DECODER (M3 fails while L1 passes), DECODER-SIDE (acc(5,8,NUM) > prod_{t<=8} cws_t + 0.05), P-M3 consistency.
      C2-1 arm: per-seed margins, seed mean +- SD; NEAR-SEED if a seed mean lies within SIGMA_SEED 0.090 of its bar.
Kappa = (acc - f) / (1 - f) from raw counts (n_correct_moved / n_moved); f is the per-cell frozen floor read as given from the shared floors file
(<runs>/n1_floors/floors_so.json); its sha256 is computed and echoed in the output, and checked against FROZEN when G0 has written it back.
Censoring (§9): without --final a missing / malformed binding file is pending; with --final the seed is FAIL: VOID and counts FAIL in k/3
(event-only k over seeds that reached the cell reported beside). A swap cell with n_pairs < 1000 (or any binding cell with n 0) is a VOID cell = FAIL.
Descriptive inputs that are missing are listed in desc_missing, never silently None.
    python -B theory/score_n1.py [--stage s1 [--pilot RUN] [--extended]] [--final] [--fixtures]
Every field path the scorer reads is listed in SCHEMA (one place); inject_so_eval.py must write exactly these.
"""
import argparse, hashlib, json, math, os, shutil, statistics, sys, tempfile

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402  (STABLE, as score_of / score_fp)
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_n1.json")
OUT_S1 = os.path.join(_HR, "theory", "results", "score_n1_s1.json")

# Run directories (edit here). The S2 resume of seed 1 lands in a new dir (resume-extension rule): put its name first in 'loop'.
RUNS = {"pilot": "loop_si_s1",
        "loop": ("loop_si_s1", "loop_si_s2", "loop_si_s3"),
        "direct_si": ("direct_si_s1", "direct_si_s2", "direct_si_s3"),   # Note 271-1: binding one-pass control (X1, D-L1, X2, DIRECT gate)
        "direct": ("direct_of_s1", "direct_of_s2", "direct_of_s3"),      # DIRECT-OF: descriptive (X1-ANS) after Note 271-1
        "en": "loop_en_s1", "textcot": "textcot_s1", "generic": "generic_s1"}

# Frozen floors file (N1_v1 §3.6 'floors', §6.1), shared by every arm; path relative to the runs dir. Its sha256 is echoed in the output.
FLOORS_REL = os.path.join("n1_floors", "floors_so.json")
# G0 write-back: {"floors_sha256": "<sha256 of the floors file>", "f": {"5/8/NUM": 0.326, ...}} (Note 271-2 values).
# Fail closed (Audit MED-2): while floors_sha256 is None the main path and --stage s1 return 'pending' (FLOORS-UNPINNED).
FROZEN = {"floors_sha256": None, "f": {}}
SIGMA_SEED = 0.090
NPAIRS = 1000

# ---------------------------------------------------------------------------------------------------------------------------
# SCHEMA: every JSON field path read by this scorer, '/'-separated, keys are strings ("5", "8", "NUM", K' "63").
# <n> in {3,4,5}; <m> in {1..12}; <mode> in {NUM, ORD, TRANSFER (n 5 m 1-8), OFFSET (n 5 m 4, o = 8)}; <t> pass index from 1.
# 'B' = binding read (missing -> pending / --final VOID), 'D' = descriptive read (missing -> listed in desc_missing).
SCHEMA = {
    "<runs>/n1_floors/floors_so.json (FLOORS_REL; the frozen floors set file, one for every arm)": {
        "set_sha": "D  sha of the TEST set the floors were computed on (compared with eval_acc_so set_sha; echoed)",
        "floors/<n>/<m>/<mode>/f": "B  frozen shortcut floor on the cell's MOVED items (M-7 list); kappa reads this per-cell f as given",
        "floors/<n>/<m>/<mode>/floor_hold": "D  P(S_m = S_0) on the cell",
        "(file)": "sha256 of the file bytes is computed and echoed; checked against FROZEN['floors_sha256'] when set",
    },
    "eval_acc_so.json": {
        "set_sha": "D  sha of the TEST set",
        "by_cell/<n>/<m>/<mode>/n_correct": "B  int, correct over all items at K' = max(8, m) (DIRECT: one pass; TEXT-CoT: greedy)",
        "by_cell/<n>/<m>/<mode>/n_items": "B  int (acc = n_correct / n_items, computed by the scorer)",
        "by_cell/<n>/<m>/<mode>/n_correct_moved": "B  int, correct on MOVED items (S_m != S_0 and k_touch >= 2), K' = max(8, m)",
        "by_cell/<n>/<m>/<mode>/n_moved": "B  int (acc_moved = n_correct_moved / n_moved; kappa from these raw counts)",
        "by_cell/<n>/<m>/<mode>/{acc,acc_moved}": "optional, not read (the scorer recomputes from counts)",
        "by_cell/<n>/<m>/<mode>/acc_moved_k8": "D  MOVED accuracy at K' 8 (m 9-12; equals acc_moved for m <= 8)",
        "by_cell/<n>/<m>/<mode>/sm1_confusion": "D  share of predictions equal to S_{m-1} (when != S_m)",
        "by_cell/5/8/NUM/by_k_touch/<k>/n": "B  int, all items (not only MOVED) with k_touch = k",
        "by_cell/5/8/NUM/by_k_touch/<k>/n_correct": "B  int",
        "by_cell/<n>/<m>/<mode>/items": "optional, not read: per item {pred, ans, k_touch, moved, s2, s4}",
        "TRANSFER / OFFSET cells": "D  n_correct, n_items only",
    },
    "eval_ws_so.json (loop arms; DIRECT writes {'arm': 'direct', 'n/a': true})": {
        "by_m/<m>/ws/<t>": "D  P(pick at pass t = S_t), m in {4, 8, 12}",
        "by_m/<m>/cws/<t>": "B  P(WS_t | WS_{t-1}); cws/1 = ws/1 (needed for the P-M3 product)",
        "by_m/<m>/n/<t>": "B  int, the CWS_t denominator (# items with WS_{t-1}; n/1 = n_items)",
    },
    "eval_swap_so.json": {
        "by_cell/<n>_<m>_<t>/{n_pairs,follow_cf,mean_floor}": "B  cells 5_4_2 (M3), 5_8_4 (M2); 5_6_3 descriptive",
        "by_cell/<n>_<m>_<t>/{stay_own,follow_dfin,follow_stale}": "D",
    },
    "eval_kcurve_so.json": {
        "acc_moved/<K'>": "B  n 5 m 8 NUM MOVED, K' in {0,1,2,4,6,7,8,9,12,16} (DIRECT: only '0')",
        "n_moved": "B  int",
        "by_m/<10|12>/acc_moved/<8|12>": "B  n 5 NUM MOVED at m 10 / 12, K' 8 and 12 (C2-K reads m 10)",
        "by_m/<10|12>/n_moved": "B  int",
        "not_decoder/{acc_vs_s2,acc_vs_s4,n}": "B  n 5 m 4 at K' 2 on the S_2 != S_4 stratum (M4 / M4p; bars >= 0.80 / <= 0.05, Note 271-1)",
        "s_norm/<8|12>/r_median/<t>": "D  median over items of ||S_t||_F / ||S_4||_F at n 5 m 8 / 12 (NORM-DRIFT routing)",
    },
    "eval_long_so.json": {
        "by_m/<1..4>/acc/<8|12|63|64>": "B  n 5 NUM, accuracy at K'",
        "by_m/<1..4>/n": "B  int",
        "by_m/<1..4>/sm1_confusion/<63|64>": "D  share answering S_{m-1} (parity)",
    },
    "eval_perm_label.json": {"by_m/<2..4>/{acc,n}": "D"},
    "eval_lookup.json": {"by_L/<1..12>/{acc,n}": "D  (routing: LOOKUP-BOUND)"},
    "eval_bbh.json": {"subsets/<three|five|seven>/per_item": "list of {parsed: bool, correct: bool}; acc = #(parsed and correct) / len; "
                      "five binds B1 when its precondition holds, else D"},
    "eval_beta_so.json": {"by_m/<5..12>/share_gt1": "D"},
}
LOOP_BIND = ("eval_acc_so", "eval_ws_so", "eval_swap_so", "eval_kcurve_so", "eval_long_so")
LOOP_DESC = ("eval_perm_label", "eval_lookup", "eval_beta_so", "eval_bbh")
PILOT_BIND = ("eval_acc_so", "eval_ws_so", "eval_swap_so", "eval_kcurve_so")
DIRECT_BIND = ("eval_acc_so",)
DIRECT_DESC = ("eval_kcurve_so", "eval_perm_label", "eval_lookup", "eval_bbh")
M4_S4 = 0.05   # Note 271-1 (Audit G-D): the S_4 half of M4 / M4p is <= 0.05 (was 0.10)
LOOP_ROWS = ("L1", "C2-1", "KC", "M1", "M2", "M3", "M4", "M5", "H1", "STABLE")   # STABLE: Audit G-D MED-1 (score_ao.stable)


class Malformed(Exception):
    pass


def jl(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def G(doc, path, src):
    cur = doc
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            raise Malformed("%s: %s" % (src, path))
        cur = cur[k]
    return cur


def num(doc, path, src):
    v = G(doc, path, src)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise Malformed("%s: %s (not a finite number)" % (src, path))
    return v


class Desc:
    """Descriptive reads: missing paths are recorded, never silently None."""

    def __init__(self):
        self.missing = []

    def get(self, doc, path, src):
        if doc is None:
            self.missing.append(src + " (file)"); return None
        try:
            return num(doc, path, src)
        except Malformed as e:
            self.missing.append(str(e)); return None

    def ratio(self, doc, base, kf, nf, src):
        k, n = self.get(doc, base + kf, src), self.get(doc, base + nf, src)
        return (k / n) if (k is not None and n) else None


def wilson(p, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def near(p, n, bar):
    """N1_v1 §6: NEAR = the bar lies within 2 SE of the reading."""
    return bool(n) and abs(p - bar) <= 2 * math.sqrt(max(p * (1 - p), 1e-12) / n)


def cell(v, bar, n, ge=True):
    n = int(n or 0); lo, hi = wilson(v, n)
    c = {"value": v, "bar": bar, "n": n, "ok": bool(n) and (v >= bar if ge else v <= bar), "margin": v - bar, "ci95": [lo, hi], "near": near(v, n, bar)}
    if not n:
        c["void"] = "n 0 (VOID cell)"
    return c


def diff_cell(p1, n1, p2, n2, bar, s1=1.0, s2=1.0, vp1=None, vp2=None):
    """Reading (p1/s1 - p2/s2) >= bar; SE treats the two samples as independent (an over-estimate for paired K' readings, disclosed).
    vp1 / vp2: the raw proportions whose binomial variance enters the SE (default p1 / p2). The kappa gap passes the raw accuracies
    (Audit LOW a: Var = a3(1-a3)/n3/(1-f3)^2 + a5(1-a5)/n5/(1-f5)^2, not p = a - f)."""
    n1, n2 = int(n1 or 0), int(n2 or 0); v = p1 / s1 - p2 / s2
    q1, q2 = (p1 if vp1 is None else vp1), (p2 if vp2 is None else vp2)
    se = math.sqrt(max(q1 * (1 - q1), 1e-12) / max(n1, 1) / s1 ** 2 + max(q2 * (1 - q2), 1e-12) / max(n2, 1) / s2 ** 2)
    c = {"value": v, "bar": bar, "n": [n1, n2], "ok": bool(n1 and n2) and v >= bar, "margin": v - bar, "ci95": [v - 1.96 * se, v + 1.96 * se],
         "near": bool(n1 and n2) and abs(v - bar) <= 2 * se}
    if not (n1 and n2):
        c["void"] = "n 0 (VOID cell)"
    return c


def row(cells, extra=True, **kw):
    return {"ok": all(c["ok"] for c in cells.values()) and bool(extra), "cells": cells, **kw}


def floor(FLR, n, m, mode, frozen):
    key = "%d/%d/%s" % (n, m, mode); f = num(FLR, "floors/" + key + "/f", "floors_so")
    if not 0.0 <= f < 1.0:
        raise Malformed("floors_so: floors/%s/f = %r (outside [0, 1))" % (key, f))
    if FLR.get("_sha_ok") is False:
        return f, False, "FLOOR-SHA-MISMATCH (floors file sha256 != frozen)"
    fz = (frozen or {}).get("f") or {}
    if key in fz and abs(fz[key] - f) > 1e-6:
        return f, False, "FLOOR-MISMATCH %s: json %.6f vs frozen %.6f" % (key, f, fz[key])
    if not fz:
        return f, True, None
    if key not in fz:
        return f, False, "FLOOR-MISMATCH %s: not in the frozen table" % key
    return f, True, None


def set_sha_note(A, FLR):
    a, b = A.get("set_sha"), FLR.get("set_sha")
    return [] if (a is None or b is None or a == b) else ["SET-SHA differs: eval_acc_so %s vs floors %s (reported; floors read as given)" % (a, b)]


def kappa(a, f):
    return (a - f) / (1 - f)


def counts(A, n, m, mode, moved=False):
    """Raw counts (k, n) -> (acc, n); Audit G-D (Note 271-1): kappa and every accuracy are computed from counts."""
    b = "by_cell/%d/%d/%s/" % (n, m, mode); kf, nf = ("n_correct_moved", "n_moved") if moved else ("n_correct", "n_items")
    k, N = num(A, b + kf, "eval_acc_so"), num(A, b + nf, "eval_acc_so")
    if k < 0 or N < 0 or k > N:
        raise Malformed("eval_acc_so: %s%s / %s (k > n or negative)" % (b, kf, nf))
    return (k / N if N else 0.0), int(N)


def acc_cell(A, n, m, mode, bar, moved=False, ge=True):
    a, N = counts(A, n, m, mode, moved)
    return dict(cell(a, bar, N, ge), k=round(a * N))


def swap_row(S, key):
    b = "by_cell/%s/" % key; npairs = int(num(S, b + "n_pairs", "eval_swap_so")); fo = num(S, b + "follow_cf", "eval_swap_so")
    mf = num(S, b + "mean_floor", "eval_swap_so")
    r = row({"follow_0.50": cell(fo, 0.50, npairs), "follow_mean+0.30": cell(fo, mf + 0.30, npairs)}, npairs >= NPAIRS, n_pairs=npairs, mean_floor=mf)
    if npairs < NPAIRS:
        r["void"] = "n_pairs %d < %d: VOID cell = FAIL" % (npairs, NPAIRS)
    return r


def not_decoder_row(K):
    n = num(K, "not_decoder/n", "eval_kcurve_so")
    return row({"vs_s2": cell(num(K, "not_decoder/acc_vs_s2", "eval_kcurve_so"), 0.80, n),
                "vs_s4": cell(num(K, "not_decoder/acc_vs_s4", "eval_kcurve_so"), M4_S4, n, ge=False)})


def cws_row(W, m, ts):
    return row({"t%d" % t: cell(num(W, "by_m/%d/cws/%d" % (m, t), "eval_ws_so"), 0.90, num(W, "by_m/%d/n/%d" % (m, t), "eval_ws_so")) for t in ts})


def bbh_acc(B, sub):
    items = G(B, "subsets/%s/per_item" % sub, "eval_bbh")
    if not isinstance(items, list) or not items:
        raise Malformed("eval_bbh: subsets/%s/per_item (empty or not a list)" % sub)
    ok = sum(1 for it in items if isinstance(it, dict) and it.get("parsed") is True and it.get("correct") is True)
    return ok / len(items), len(items), sum(1 for it in items if isinstance(it, dict) and it.get("parsed") is True)


def stable_row(runs, run):
    """score_ao.stable: None = no / blank log (pending, VOID with --final); False = FAIL: STABLE (n_nonfinite > 10 or an unrecovered loss jump)."""
    sok, smsg = AO.stable(runs, run)
    return sok, {"ok": sok is not None and sok is not False, "cells": {}, "detail": smsg}


def load(runs, run, names):
    return {x: jl(os.path.join(runs, run, x + ".json")) for x in names}


def censored(final, **kw):
    return {"label": "FAIL: VOID" if final else "pending", "status": "void" if final else "pending", **kw}


# ------------------------------------------------------------------ LOOP-SI seed
def seed_loop(runs, run, final, frozen, FLR):
    f = load(runs, run, LOOP_BIND + LOOP_DESC)
    miss = [x for x in LOOP_BIND if f[x] is None] + (["floors_so (shared)"] if FLR is None else [])
    if miss:
        return censored(final, missing=miss)
    sok, srow = stable_row(runs, run)
    if sok is None:   # a missing / blank log is pending (VOID with --final), never STABLE
        return censored(final, stable=srow["detail"])
    try:
        return _loop(f, frozen, FLR, srow)
    except Malformed as e:
        return censored(final, malformed=str(e))


def _loop(f, frozen, FLR, srow):
    A, W, S, K, L = (f[x] for x in LOOP_BIND); D = Desc(); rows = {"STABLE": srow}; notes = set_sha_note(A, FLR)
    rows["L1"] = row({"m%d" % m: acc_cell(A, 5, m, "NUM", 0.95) for m in range(1, 5)})
    rows["L2"] = row({"m%d" % m: acc_cell(A, 5, m, "ORD", 0.90) for m in range(1, 5)})
    rows["M3"] = swap_row(S, "5_4_2"); rows["M2"] = swap_row(S, "5_8_4")
    c6 = acc_cell(A, 5, 6, "NUM", 0.85, moved=True); c8 = acc_cell(A, 5, 8, "NUM", 0.80, moved=True)
    rows["C2-1"] = row({"m6": c6, "m8": c8}, rows["M3"]["ok"], read=rows["M3"]["ok"],
                       raw_ok=c6["ok"] and c8["ok"], **({} if rows["M3"]["ok"] else {"not_read": "M3 fails: C2-1 not read, counts FAIL"}))
    f58, okf, msg = floor(FLR, 5, 8, "NUM", frozen)
    if msg:
        notes.append(msg)
    kp = lambda k: num(K, "acc_moved/%s" % k, "eval_kcurve_so")  # noqa: E731
    nk = num(K, "n_moved", "eval_kcurve_so")
    rows["KC"] = row({"k0": cell(kp(0), f58 + 0.05, nk, ge=False), "k1": cell(kp(1), f58 + 0.05, nk, ge=False),
                      "k4": cell(kp(4), 0.202 + 0.10, nk, ge=False), "k8-k4": diff_cell(kp(8), nk, kp(4), nk, 0.40)}, okf, f=f58)
    rows["M1"] = cws_row(W, 8, range(5, 9))
    rows["M4"] = not_decoder_row(K)
    bk = G(A, "by_cell/5/8/NUM/by_k_touch", "eval_acc_so")
    if not isinstance(bk, dict) or not bk:
        raise Malformed("eval_acc_so: by_cell/5/8/NUM/by_k_touch (empty)")
    hn = hc = ln = lc = 0
    for k in bk:
        kn = int(num(A, "by_cell/5/8/NUM/by_k_touch/%s/n" % k, "eval_acc_so")); kc = int(num(A, "by_cell/5/8/NUM/by_k_touch/%s/n_correct" % k, "eval_acc_so"))
        if int(k) >= 5:
            hn += kn; hc += kc
        elif int(k) <= 2:
            ln += kn; lc += kc
    rows["M5"] = row({"hi-lo": diff_cell(hc / hn if hn else 0.0, hn, lc / ln if ln else 0.0, ln, -0.10)}, acc_hi=hc / hn if hn else None, acc_lo=lc / ln if ln else None)
    h1 = {}
    for m in range(1, 5):
        b = "by_m/%d/" % m; a8 = num(L, b + "acc/8", "eval_long_so"); n = num(L, b + "n", "eval_long_so")
        for k in ("63", "64"):
            h1["m%d_k%s" % (m, k)] = cell(num(L, b + "acc/" + k, "eval_long_so"), a8 - 0.02, n)
    rows["H1"] = row(h1)
    a12 = num(K, "by_m/10/acc_moved/12", "eval_kcurve_so"); a8_10 = num(K, "by_m/10/acc_moved/8", "eval_kcurve_so"); n10 = num(K, "by_m/10/n_moved", "eval_kcurve_so")
    rows["C2-K"] = row({"k12": cell(a12, 0.60, n10), "k12-k8": diff_cell(a12, n10, a8_10, n10, 0.20)})
    rows["C2-3"] = row({"m8": acc_cell(A, 5, 8, "ORD", 0.60)})   # descriptive row; B1 precondition
    b1 = None
    if f["eval_bbh"] is not None:
        try:
            a, n, npz = bbh_acc(f["eval_bbh"], "five"); b1 = dict(cell(a, 0.60, n), n_parsed=npz)
        except Malformed as e:
            D.missing.append(str(e))
    fail = [r for r in LOOP_ROWS if not rows[r]["ok"]]
    decoder = (not rows["M3"]["ok"]) and rows["L1"]["ok"]
    # descriptive
    acc58 = counts(A, 5, 8, "NUM")[0]
    cws18 = [D.get(W, "by_m/8/cws/%d" % t, "eval_ws_so") for t in range(1, 9)]
    pm3 = None
    if None not in cws18:
        prod = math.prod(cws18)
        pm3 = {"mean_cws_1_8": sum(cws18) / 8, "prod_cws_1_8": prod, "acc_5_8": acc58, "DECODER-SIDE": acc58 > prod + 0.05,
               "consistent_if_c2_1": (sum(cws18) / 8 >= 0.97) if rows["C2-1"]["ok"] else None}
    route = None
    if not rows["C2-1"]["ok"]:
        lk = [D.get(f["eval_lookup"], "by_L/%d/acc" % L_, "eval_lookup") for L_ in range(5, 13)]
        r8 = [D.get(K, "s_norm/%d/r_median/8" % m, "eval_kcurve_so") for m in (8, 12)]
        w4, w8 = D.get(W, "by_m/8/cws/4", "eval_ws_so"), D.get(W, "by_m/8/cws/8", "eval_ws_so")
        if None in lk:
            route = "unread (lookup missing)"
        elif min(lk) < 0.80:
            route = "LOOKUP-BOUND"
        elif None in (w4, w8) or all(x is None for x in r8):
            route = "unread (s_norm / cws missing)"
        elif any(x is not None and not (0.67 <= x <= 1.5) for x in r8) and w4 - w8 > 0.10:
            route = "NORM-DRIFT-CANDIDATE"
        else:
            route = "none (no pre-stated route)"
        route = {"route": route, "lookup_L5_12": lk, "r8_m8_m12": r8, "cws4_minus_cws8": (w4 - w8) if None not in (w4, w8) else None}
    desc = {"C2-2": {m: {"acc_moved": D.ratio(A, "by_cell/5/%d/NUM/" % m, "n_correct_moved", "n_moved", "eval_acc_so"),
                         "acc_moved_k8": D.get(A, "by_cell/5/%d/NUM/acc_moved_k8" % m, "eval_acc_so")} for m in range(9, 13)},
            "sm1_confusion_5_8": D.get(A, "by_cell/5/8/NUM/sm1_confusion", "eval_acc_so"),
            "long_sm1_k63": {m: D.get(L, "by_m/%d/sm1_confusion/63" % m, "eval_long_so") for m in range(1, 5)},
            "transfer": {m: D.ratio(A, "by_cell/5/%d/TRANSFER/" % m, "n_correct", "n_items", "eval_acc_so") for m in range(1, 9)},
            "offset": D.ratio(A, "by_cell/5/4/OFFSET/", "n_correct", "n_items", "eval_acc_so"),
            "swap_5_6_3": {k: D.get(S, "by_cell/5_6_3/" + k, "eval_swap_so") for k in ("n_pairs", "follow_cf", "mean_floor")},
            "swap_detail": {c: {k: D.get(S, "by_cell/%s/%s" % (c, k), "eval_swap_so") for k in ("stay_own", "follow_dfin", "follow_stale")} for c in ("5_4_2", "5_8_4")},
            "kcurve": {k: D.get(K, "acc_moved/%d" % k, "eval_kcurve_so") for k in (2, 6, 7, 9, 12, 16)},
            "perm_label": {m: D.get(f["eval_perm_label"], "by_m/%d/acc" % m, "eval_perm_label") for m in (2, 3, 4)},
            "lookup": {L_: D.get(f["eval_lookup"], "by_L/%d/acc" % L_, "eval_lookup") for L_ in range(1, 13)},
            "beta_share_gt1": {m: D.get(f["eval_beta_so"], "by_m/%d/share_gt1" % m, "eval_beta_so") for m in range(5, 13)},
            "bbh_three_seven": {}}
    if f["eval_bbh"] is not None:
        for sub in ("three", "seven"):
            try:
                desc["bbh_three_seven"][sub] = bbh_acc(f["eval_bbh"], sub)[0]
            except Malformed as e:
                D.missing.append(str(e))
    else:
        D.missing.append("eval_bbh (file)")
    lab = "LOOP-PASS" if not fail else "FAIL: " + ",".join(fail)
    if decoder:
        lab += " [DECODER: M3 fails with L1 passing]"
    seed_c2 = {m: ("NEAR-SEED" if abs(c["margin"]) < SIGMA_SEED else ("PASS" if c["ok"] else "FAIL")) for m, c in (("m6", c6), ("m8", c8))}
    return {"label": lab, "status": "ok", "rows": rows, "fail": fail, "decoder": decoder, "b1_five": b1,
            "c2_1_margins": {"m6": c6["margin"], "m8": c8["margin"]}, "c2_1_values": {"m6": c6["value"], "m8": c8["value"]}, "c2_1_seed_label": seed_c2,
            "near": sorted("%s.%s" % (r, k) for r, v in rows.items() for k, c in v["cells"].items() if c["near"]),
            "pm3": pm3, "routing": route, "notes": notes, "desc": desc, "desc_missing": D.missing}


# ------------------------------------------------------------------ DIRECT-OF seed
def seed_direct(runs, run, final, frozen, FLR):
    f = load(runs, run, DIRECT_BIND + DIRECT_DESC)
    miss = (["eval_acc_so"] if f["eval_acc_so"] is None else []) + (["floors_so (shared)"] if FLR is None else [])
    if miss:
        return censored(final, missing=miss)
    sok, srow = stable_row(runs, run)
    if sok is None:
        return censored(final, stable=srow["detail"])
    try:
        return _direct(f, frozen, FLR, srow)
    except Malformed as e:
        return censored(final, malformed=str(e))


def _direct(f, frozen, FLR, srow):
    A = f["eval_acc_so"]; D = Desc(); notes = set_sha_note(A, FLR); rows = {"STABLE": srow}
    rows["D-L1"] = row({"m%d" % m: acc_cell(A, 5, m, "NUM", 0.90) for m in range(1, 5)})
    f5, ok5, m5 = floor(FLR, 5, 8, "NUM", frozen); f3, ok3, m3 = floor(FLR, 3, 8, "NUM", frozen)
    notes += [x for x in (m5, m3) if x]
    a5, n5 = counts(A, 5, 8, "NUM", moved=True); a3, n3 = counts(A, 3, 8, "NUM", moved=True)
    k5, k3 = kappa(a5, f5), kappa(a3, f3)
    rows["X1"] = row({"kappa5<=0.25": cell(a5, f5 + 0.25 * (1 - f5), n5, ge=False)}, ok5, kappa=k5, f=f5, k_moved=round(a5 * n5), n_moved=n5)
    rows["GATE"] = row({"kappa5>=0.70": cell(a5, f5 + 0.70 * (1 - f5), n5)}, ok5, kappa=k5)
    lb3 = wilson(a3, n3)[0]
    rows["X2"] = row({"kappa_gap": diff_cell(a3 - f3, n3, a5 - f5, n5, 0.15, 1 - f3, 1 - f5, vp1=a3, vp2=a5), "kappa3>=0.15": cell(a3, f3 + 0.15 * (1 - f3), n3)},
                     ok3 and ok5 and lb3 > f3, kappa3=k3, kappa5=k5, wilson_lb3=lb3, f3=f3, f5=f5, lb3_gt_f3=lb3 > f3,
                     raw_gap=a3 - a5, gap_vs_1_over_n=(a3 - 1 / 3) - (a5 - 1 / 5))
    desc = {"kcurve_k0": D.get(f["eval_kcurve_so"], "acc_moved/0", "eval_kcurve_so"),
            "perm_label": {m: D.get(f["eval_perm_label"], "by_m/%d/acc" % m, "eval_perm_label") for m in (2, 3, 4)},
            "lookup": {L_: D.get(f["eval_lookup"], "by_L/%d/acc" % L_, "eval_lookup") for L_ in range(1, 13)},
            "depth_curve_5": {m: D.ratio(A, "by_cell/5/%d/NUM/" % m, "n_correct_moved", "n_moved", "eval_acc_so") for m in range(5, 13)}, "bbh": {}}
    if f["eval_bbh"] is None:
        D.missing.append("eval_bbh (file)")
    else:
        for sub in ("three", "five", "seven"):
            try:
                desc["bbh"][sub] = bbh_acc(f["eval_bbh"], sub)[0]
            except Malformed as e:
                D.missing.append(str(e))
    return {"label": "read" if srow["ok"] else "FAIL: STABLE (every DIRECT row counts FAIL)", "status": "ok", "rows": rows, "kappa5": k5, "kappa3": k3, "notes": notes, "desc": desc, "desc_missing": D.missing,
            "near": sorted("%s.%s" % (r, k) for r, v in rows.items() for k, c in v["cells"].items() if c["near"])}


# ------------------------------------------------------------------ descriptive arms
def desc_arm(runs, run, D):
    A = jl(os.path.join(runs, run, "eval_acc_so.json"))
    if A is None:
        D.missing.append(run + "/eval_acc_so (file)"); return {"m6_moved": None, "m8_moved": None}
    return {"m%d_moved" % m: D.ratio(A, "by_cell/5/%d/NUM/" % m, "n_correct_moved", "n_moved", run + "/eval_acc_so") for m in (6, 8)}


def count(seeds, r):
    k = sum(1 for s in seeds.values() if s.get("status") == "ok" and s["rows"][r]["ok"])
    reached = [n for n, s in seeds.items() if s.get("status") == "ok"]
    return {"k": k, "of": len(seeds), "k_event": k, "reached": len(reached)}


def load_floors(runs, frozen):
    """The frozen floors file: parsed doc (or None) + info echoed in the output (path, sha256 of the bytes, set_sha, frozen check)."""
    p = os.path.join(runs, FLOORS_REL); info = {"path": p}
    try:
        with open(p, "rb") as fh:
            raw = fh.read()
        doc = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError):
        info["missing"] = True; return None, info
    if not isinstance(doc, dict):
        info["missing"] = True; return None, info
    info["sha256"] = hashlib.sha256(raw).hexdigest(); info["set_sha"] = doc.get("set_sha")
    fz = (frozen or {}).get("floors_sha256")
    info["frozen_sha256"] = fz; info["sha_ok"] = (fz is None) or fz == info["sha256"]
    doc["_sha_ok"] = info["sha_ok"]
    return doc, info


def direct_summary(dp, pend):
    dc = lambda r: sum(1 for s in dp.values() if s["status"] == "ok" and s["rows"][r]["ok"] and s["rows"]["STABLE"]["ok"])  # noqa: E731
    dl1, x1k, gk, x2k = dc("D-L1"), dc("X1"), dc("GATE"), dc("X2")
    npend = sum(1 for s in dp.values() if s["status"] == "pending"); gate = gk >= 2
    if pend and not gate:
        x1 = "pending"
    elif gate:
        x1 = "NOT-READ (DIRECT gate fired: DEPTH-SPLIT)"
    elif dl1 < 2:
        x1 = "VOID-CONTROL"
    else:
        x1 = "PASS" if x1k == 3 else "FAIL"
    return {"D-L1_k3": dl1, "X1_k3": x1k, "X1": x1, "gate_k3": gk, "DIRECT_gate": "DEPTH-SPLIT" if gate else ("pending" if gk + npend >= 2 else "DIRECT-ALIVE"),
            "X2_k3": x2k, "TH6": "pending" if pend else ("TH6-SUPPORTED" if x2k == 3 else "TH6-NOT-SUPPORTED"),
            "reached": sum(1 for s in dp.values() if s["status"] == "ok"),
            "kappa5": {r: s.get("kappa5") for r, s in dp.items()}, "kappa3": {r: s.get("kappa3") for r, s in dp.items()},
            "x1_near": {r: s["rows"]["X1"]["cells"]["kappa5<=0.25"]["near"] for r, s in dp.items() if s["status"] == "ok"},
            "near_seed_x1": {r: abs(s["kappa5"] - 0.25) < SIGMA_SEED for r, s in dp.items() if s["status"] == "ok"}}


def unpinned(frozen, **kw):
    """Audit MED-2: floors fail closed. Until G0 pins FROZEN['floors_sha256'], nothing is scored."""
    if (frozen or {}).get("floors_sha256"):
        return None
    return dict({"section": 271, "label": "pending", "reason": "FLOORS-UNPINNED", "labels": {"main": "pending", "reason": "FLOORS-UNPINNED"},
                 "verdict": "pending (FLOORS-UNPINNED: FROZEN['floors_sha256'] is not set; G0 pins it)"}, **kw)


def score(runs=R, final=False, frozen=None):
    frozen = FROZEN if frozen is None else frozen
    up = unpinned(frozen, final=final)
    if up:
        return up
    FLR, finfo = load_floors(runs, frozen)
    o = {"section": 271, "final": final, "runs": {k: v for k, v in RUNS.items()}, "floors": finfo}
    lp = {r: seed_loop(runs, r, final, frozen, FLR) for r in RUNS["loop"]}
    sp = {r: seed_direct(runs, r, final, frozen, FLR) for r in RUNS["direct_si"]}
    dp = {r: seed_direct(runs, r, True, frozen, FLR) for r in RUNS["direct"]}   # DIRECT-OF descriptive: missing is reported, never pending
    o["loop"], o["direct_si"], o["direct_of"] = lp, sp, dp
    l_pend = any(s["status"] == "pending" for s in lp.values()); s_pend = any(s["status"] == "pending" for s in sp.values())
    rows = {r: count(lp, r) for r in LOOP_ROWS + ("L2", "C2-K", "C2-3")}
    for r, v in rows.items():   # k/3 counts never-reached seeds as FAIL; event-only k over reached seeds beside; NEAR seeds listed
        v["near_seeds"] = sorted(n for n, s in lp.items() if s["status"] == "ok" and any(c["near"] for c in s["rows"][r]["cells"].values()))
    o["rows_k3"] = rows
    seed_all = sum(1 for s in lp.values() if s["label"] == "LOOP-PASS"); o["loop_pass_k3"] = seed_all
    o["loop_pass_event"] = "%d/%d reached" % (seed_all, sum(1 for s in lp.values() if s["status"] == "ok"))
    loop_arm = seed_all >= 2
    # C2-1 arm summary: per-seed margins, mean +- SD over reached seeds, NEAR-SEED
    vals = {m: [s["c2_1_values"][m] for s in lp.values() if s["status"] == "ok"] for m in ("m6", "m8")}
    c2 = {"k": rows["C2-1"]["k"], "per_seed_margins": {r: s.get("c2_1_margins") for r, s in lp.items()},
          "per_seed_label": {r: s.get("c2_1_seed_label") for r, s in lp.items()}, "bars": {"m6": 0.85, "m8": 0.80}}
    for m, bar in (("m6", 0.85), ("m8", 0.80)):
        v = vals[m]
        c2[m] = {"mean": statistics.mean(v) if v else None, "sd": statistics.stdev(v) if len(v) > 1 else None, "n_seeds": len(v),
                 "near_seed": bool(v) and abs(statistics.mean(v) - bar) < SIGMA_SEED}
    # Audit LOW d: 'n/a' when the C2-1 arm fails; NEAR-SEED only qualifies an arm that passes
    c2["claim"] = "n/a" if rows["C2-1"]["k"] < 2 else ("NEAR-SEED" if (c2["m6"]["near_seed"] or c2["m8"]["near_seed"]) else "PASS")
    o["c2_1"] = c2
    # DIRECT-SI binds X1 / D-L1 / X2 / gate (Note 271-1); DIRECT-OF beside (X1-ANS)
    ds = direct_summary(sp, s_pend); dof = direct_summary(dp, False)
    o["direct_si_summary"] = ds
    o["direct_of_summary"] = dict(dof, note="descriptive (X1-ANS); missing seeds count as not reached",
                                  missing={r: s.get("missing") or s.get("malformed") for r, s in dp.items() if s["status"] != "ok"})
    gate, x1 = ds["gate_k3"] >= 2, ds["X1"]
    if l_pend or (s_pend and not gate):
        main = "pending"
    elif not loop_arm:
        bad = [r for r in LOOP_ROWS if rows[r]["k"] < 2]
        main = "FAIL: " + (",".join(bad) if bad else "SEED-CONJUNCTION (no 2/3 seeds pass every row)")
    elif gate:
        main = "DEPTH-SPLIT"
    elif x1 == "PASS":
        main = "N1-PASS"
    else:
        main = "N1-LOOP-ONLY (X1 %s)" % x1
    ord_scope = "pending" if l_pend else ("ORD-SCOPE" if rows["L2"]["k"] >= 2 else "ORD-SCOPE-FAIL")
    c2kp = "pending" if l_pend else ("C2-KPRIME" if rows["C2-K"]["k"] >= 2 else "C2-KPRIME-FAIL")
    th6 = ds["TH6"]
    if l_pend:
        b1 = "pending"
    elif not (ord_scope == "ORD-SCOPE" and rows["C2-3"]["k"] >= 2):
        b1 = "B1-NOT-BINDING (precondition L2 and C2-3 not both met)"
    else:
        bm = [s.get("b1_five") for s in lp.values() if s["status"] == "ok"]
        nmiss = sum(1 for s in lp.values() if s["status"] == "ok" and s.get("b1_five") is None)
        if nmiss and not final:
            b1 = "pending"
        else:   # with --final a missing BBH file counts FAIL for that seed
            b1 = "B1-PASS" if sum(1 for c in bm if c and c["ok"]) >= 2 else "B1-FAIL"
    routing = {r: s["routing"]["route"] for r, s in lp.items() if s["status"] == "ok" and s.get("routing")}
    flags = {r: [x for x, on in (("DECODER", s.get("decoder")), ("DECODER-SIDE", (s.get("pm3") or {}).get("DECODER-SIDE"))) if on]
             for r, s in lp.items() if s["status"] == "ok"}
    o["labels"] = {"main": main, "ORD-SCOPE": ord_scope, "C2-KPRIME": c2kp, "TH6": th6, "B1": b1, "X1": x1,
                   "X1-ANS (DIRECT-OF, descriptive)": dof["X1"], "TH6 DIRECT-OF (descriptive)": dof["TH6"], "routing": routing, "flags": flags}
    # descriptive arms
    D = Desc()
    en, gen, cot = desc_arm(runs, RUNS["en"], D), desc_arm(runs, RUNS["generic"], D), desc_arm(runs, RUNS["textcot"], D)
    cb = jl(os.path.join(runs, RUNS["textcot"], "eval_bbh.json")); cot_bbh = {}
    if cb is None:
        D.missing.append(RUNS["textcot"] + "/eval_bbh (file)")
    else:
        for sub in ("three", "five", "seven"):
            try:
                cot_bbh[sub] = bbh_acc(cb, sub)[0]
            except Malformed as e:
                D.missing.append(RUNS["textcot"] + "/" + str(e))
    lm = c2["m8"]["mean"]
    o["descriptive"] = {"P1_loop_en": en, "G1_generic": gen, "X4_textcot": dict(cot, bbh=cot_bbh),
                        "C3_loop_ge_cot_minus_0.05": (lm >= cot["m8_moved"] - 0.05) if (lm is not None and cot["m8_moved"] is not None) else None}
    allseeds = list(lp.items()) + list(sp.items()) + list(dp.items())
    dm = D.missing + ["%s: %s" % (r, x) for r, s in allseeds for x in s.get("desc_missing", [])]
    dm += ["%s: %s (DIRECT-OF, descriptive)" % (r, v) for r, v in o["direct_of_summary"]["missing"].items()]
    o["desc_missing"] = dm
    notes = ["%s: %s" % (r, x) for r, s in allseeds for x in s.get("notes", [])]
    if FLR is not None and not finfo["sha_ok"]:
        notes.append("floors file sha256 %s != frozen %s: every kappa / floor row FAILs" % (finfo["sha256"], finfo["frozen_sha256"]))
    if notes:
        o["floor_notes"] = notes
    o["verdict"] = "%s | %s | %s | %s | B1 %s | X1(DIRECT-SI) %s | X1-ANS(DIRECT-OF) %s | C2-1 k %d/3 (claim %s) | loop seeds %s | floors sha256 %s" % (
        main, ord_scope, c2kp, th6, b1, x1, dof["X1"], rows["C2-1"]["k"], c2["claim"], " / ".join(s["label"] for s in lp.values()),
        (finfo.get("sha256") or "MISSING")[:12])
    if routing:
        o["verdict"] += " | routing " + ", ".join("%s %s" % kv for kv in routing.items())
    return o


# ------------------------------------------------------------------ S1
def score_s1(runs=R, final=False, pilot=None, extended=False, frozen=None):
    frozen = FROZEN if frozen is None else frozen
    up = unpinned(frozen, stage="s1", final=final, gate="pending", direct_gate="pending")
    if up:
        return up
    FLR, finfo = load_floors(runs, frozen)
    p = pilot or RUNS["pilot"]; o = {"section": 271, "stage": "s1", "pilot": p, "extended": extended, "final": final, "floors": finfo}
    f = load(runs, p, PILOT_BIND); miss = [x for x in PILOT_BIND if f[x] is None]; gate = None
    if miss:
        gate = "FAIL: VOID" if final else "pending"; o["pilot_missing"] = miss
    else:
        try:
            A, W, S, K = (f[x] for x in PILOT_BIND)
            rows = {"L1p": row({"m%d" % m: acc_cell(A, 5, m, "NUM", 0.90) for m in range(1, 5)}), "CWS": cws_row(W, 4, (2, 3, 4)),
                    "M3p": swap_row(S, "5_4_2"), "M4p": not_decoder_row(K)}
            park = any(c["value"] < 0.50 for c in rows["L1p"]["cells"].values())
            o["rows"] = rows; o["near"] = sorted("%s.%s" % (r, k) for r, v in rows.items() for k, c in v["cells"].items() if c["near"])
            o["fail"] = [r for r, v in rows.items() if not v["ok"]]
            gate = "GO" if not o["fail"] else ("PARK" if park else ("NO-GO (EXTEND used)" if extended else "EXTEND"))
        except Malformed as e:
            gate = "FAIL: VOID" if final else "pending"; o["pilot_malformed"] = str(e)
    sp = {r: seed_direct(runs, r, final, frozen, FLR) for r in RUNS["direct_si"]}
    dp = {r: seed_direct(runs, r, True, frozen, FLR) for r in RUNS["direct"]}
    ds, dof = direct_summary(sp, False), direct_summary(dp, False)
    pick = lambda d: {r: ({"kappa5": s["kappa5"], "gate": s["rows"]["GATE"], "D-L1_desc": s["rows"]["D-L1"], "notes": s["notes"]} if s["status"] == "ok" else s)  # noqa: E731
                      for r, s in d.items()}
    o["direct_si"], o["direct_of"] = pick(sp), pick(dp)
    o["gate"], o["direct_gate"] = gate, ds["DIRECT_gate"]
    o["direct_of_gate_desc"] = "DEPTH-SPLIT-like" if dof["gate_k3"] >= 2 else "alive"
    o["verdict"] = "S1 %s | DIRECT gate (DIRECT-SI) %s (kappa>=0.70 in %d/3) | DIRECT-OF beside: kappa>=0.70 in %d/3 | floors sha256 %s%s" % (
        gate, ds["DIRECT_gate"], ds["gate_k3"], dof["gate_k3"], (finfo.get("sha256") or "MISSING")[:12],
        (" | pilot fails " + ",".join(o["fail"])) if o.get("fail") else "")
    return o


# ------------------------------------------------------------------ fixtures
DEL = object()
FL = {(5, 8): 0.326, (3, 8): 0.539, (4, 8): 0.393, (5, 6): 0.465, (5, 10): 0.265, (5, 12): 0.255}   # Note 271-2


def _acc_so(direct=False):
    bc = {}
    for n in (3, 4, 5):
        for m in range(1, 13):
            for mode in ("NUM", "ORD"):
                ni = 1500 if (m == 8 and mode == "NUM" and n in (3, 5)) else 500
                if direct:
                    a = 0.95 if m <= 4 else 0.45
                    am = {(5, 8): 0.35, (3, 8): 0.58}.get((n, m), 0.95 if m <= 4 else 0.40)
                else:
                    a = 0.98 if m <= 4 else (0.92 if m <= 8 else 0.70)
                    am = 0.98 if m <= 4 else {6: 0.95, 8: 0.90}.get(m, 0.90 if m <= 8 else 0.65)
                bc.setdefault(str(n), {}).setdefault(str(m), {})[mode] = {
                    "acc": a, "n_items": ni, "acc_moved": am, "n_moved": int(0.7 * ni), "f": FL.get((n, m), 0.30), "floor_hold": 0.2,
                    "acc_moved_k8": am if m <= 8 else 0.30, "sm1_confusion": 0.02}
    for m in range(1, 9):
        bc["5"][str(m)]["TRANSFER"] = {"acc": 0.9, "n_items": 500}
    bc["5"]["4"]["OFFSET"] = {"acc": 0.9, "n_items": 500}
    bc["5"]["8"]["NUM"]["by_k_touch"] = {"0": {"n": 100, "n_correct": 98}, "1": {"n": 200, "n_correct": 190}, "2": {"n": 172, "n_correct": 160},
                                         "3": {"n": 400, "n_correct": 370}, "4": {"n": 368, "n_correct": 340},
                                         "5": {"n": 160, "n_correct": 150}, "6": {"n": 100, "n_correct": 92}}
    return {"set_sha": "fixture", "by_cell": bc}


def _bbh(five=170):
    def sub(k):
        return {"per_item": [{"parsed": True, "correct": i < k} for i in range(250)]}
    return {"subsets": {"three": sub(200), "five": sub(five), "seven": sub(100)}}


def _loop_docs():
    return {"eval_acc_so": _acc_so(),
            "eval_ws_so": {"by_m": {str(m): {"ws": {str(t): 0.99 for t in range(1, m + 1)}, "cws": {str(t): 0.99 for t in range(1, m + 1)},
                                             "n": {str(t): 1000 for t in range(1, m + 1)}} for m in (4, 8, 12)}},
            "eval_swap_so": {"by_cell": {k: {"n_pairs": 1000, "follow_cf": 0.75, "stay_own": 0.1, "follow_dfin": 0.05, "follow_stale": 0.05, "mean_floor": 0.15}
                                         for k in ("5_4_2", "5_8_4", "5_6_3")}},
            "eval_kcurve_so": {"acc_moved": {"0": 0.30, "1": 0.31, "2": 0.40, "4": 0.22, "6": 0.36, "7": 0.57, "8": 0.90, "9": 0.90, "12": 0.90, "16": 0.89},
                               "n_moved": 1060, "by_m": {"10": {"acc_moved": {"8": 0.38, "12": 0.75}, "n_moved": 330},
                                                         "12": {"acc_moved": {"8": 0.24, "12": 0.70}, "n_moved": 330}},
                               "not_decoder": {"n": 300, "acc_vs_s2": 0.92, "acc_vs_s4": 0.03},
                               "s_norm": {str(m): {"r_median": {str(t): 1.0 for t in range(1, m + 1)}} for m in (8, 12)}},
            "eval_long_so": {"by_m": {str(m): {"acc": {"8": 0.98, "12": 0.98, "63": 0.98, "64": 0.98}, "n": 500, "sm1_confusion": {"63": 0.01, "64": 0.01}}
                                      for m in range(1, 5)}},
            "eval_perm_label": {"by_m": {str(m): {"acc": 0.7, "n": 500} for m in (2, 3, 4)}},
            "eval_lookup": {"by_L": {str(L_): {"acc": 0.97, "n": 200} for L_ in range(1, 13)}},
            "eval_beta_so": {"by_m": {str(m): {"share_gt1": 0.01} for m in range(5, 13)}},
            "eval_bbh": _bbh()}


def _direct_docs():
    return {"eval_acc_so": _acc_so(direct=True), "eval_ws_so": {"arm": "direct", "n/a": True}, "eval_kcurve_so": {"acc_moved": {"0": 0.35}, "n_moved": 1060},
            "eval_perm_label": {"by_m": {str(m): {"acc": 0.5, "n": 500} for m in (2, 3, 4)}},
            "eval_lookup": {"by_L": {str(L_): {"acc": 0.9, "n": 200} for L_ in range(1, 13)}}, "eval_bbh": _bbh(80)}


def _sel(sel):
    """Fixture selectors: 'loop' = LOOP-SI seeds, 'direct' = DIRECT-SI seeds (the binding control), 'dof' = DIRECT-OF seeds, 'floors' = floors file."""
    m = {"loop": RUNS["loop"], "direct": RUNS["direct_si"], "dof": RUNS["direct"], "floors": ("__floors__",)}
    if isinstance(sel, str):
        return list(m.get(sel, (sel,)))
    return list(sel)


def _counts_from_acc(doc):
    """Fixture writer: derive the raw counts the scorer reads (unless a patch set them) and drop the floor (it lives in the floors file)."""
    for nd in doc.get("by_cell", {}).values():
        for md in nd.values():
            for c in md.values():
                c.pop("f", None)
                if "acc" in c and "n_correct" not in c:
                    c["n_correct"] = round(c["acc"] * c["n_items"])
                if "acc_moved" in c and "n_correct_moved" not in c:
                    c["n_correct_moved"] = round(c["acc_moved"] * c["n_moved"])
    return doc


def _set(doc, path, v):
    ks = path.split("/"); cur = doc
    for k in ks[:-1]:
        cur = cur.setdefault(k, {})
    if v is DEL:
        cur.pop(ks[-1], None)
    else:
        cur[ks[-1]] = v


def _logs(d, run, jump=False):
    """score_ao.stable inputs: log.jsonl (80 rows to step 8000) + train_summary.json. jump: an unrecovered loss jump (5.0 from step 4100)."""
    with open(os.path.join(d, run, "log.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for i in range(1, 81):
            fh.write(json.dumps({"step": i * 100, "loss": 5.0 if (jump and i > 40) else 0.01}) + "\n")
    with open(os.path.join(d, run, "train_summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"n_nonfinite": 0}, fh)


def world(patches=(), drop=(), unstable=(), nolog=()):
    d = tempfile.mkdtemp(prefix="n1fx_")
    docs = {r: _loop_docs() for r in RUNS["loop"]}
    docs.update({r: _direct_docs() for r in RUNS["direct_si"]})
    docs.update({r: _direct_docs() for r in RUNS["direct"]})
    for r in (RUNS["en"], RUNS["generic"]):
        docs[r] = {"eval_acc_so": _acc_so()}
    docs[RUNS["textcot"]] = {"eval_acc_so": _acc_so(), "eval_bbh": _bbh(120)}
    fl = {}
    for n, nd in _acc_so()["by_cell"].items():
        for m, md in nd.items():
            for mode, c in md.items():
                if "f" in c:
                    fl.setdefault(n, {}).setdefault(m, {})[mode] = {"f": c["f"], "floor_hold": c["floor_hold"]}
    docs["__floors__"] = {"floors_so": {"set_sha": "fixture", "floors": fl}}
    for sel, fn, path, v in patches:
        for r in _sel(sel):
            _set(docs[r][fn], path, v)
    gone = {(r, fn) for sel, fn in drop for r in _sel(sel)}
    for r, fs in docs.items():
        dd = os.path.join(d, os.path.dirname(FLOORS_REL)) if r == "__floors__" else os.path.join(d, r)
        os.makedirs(dd, exist_ok=True)
        for fn, doc in fs.items():
            if (r, fn) not in gone:
                if fn == "eval_acc_so":
                    doc = _counts_from_acc(doc)
                with open(os.path.join(dd, fn + ".json"), "w", encoding="utf-8", newline="\n") as fh:
                    json.dump(doc, fh)
    bad = {r for sel in unstable for r in _sel(sel)}; gone_log = {r for sel in nolog for r in _sel(sel)}
    for r in list(RUNS["loop"]) + list(RUNS["direct_si"]) + list(RUNS["direct"]):
        if r not in gone_log:
            _logs(d, r, r in bad)
    return d


def pin(d, kw):
    """Fixtures always pass frozen explicitly (Audit MED-2). Default / floors_sha256 'MATCH' = the world's own floors-file sha256."""
    fz = dict(kw.get("frozen") or {"floors_sha256": "MATCH", "f": {}})
    if fz.get("floors_sha256") == "MATCH":
        try:
            with open(os.path.join(d, FLOORS_REL), "rb") as fh:
                fz["floors_sha256"] = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            fz["floors_sha256"] = "f" * 64   # file absent: pinned to a sha that cannot match
    return dict(kw, frozen=fz)


def fixtures():
    res = []
    S1, S2, S3 = RUNS["loop"]; D1, D2, D3 = RUNS["direct_si"]; F1 = RUNS["direct"][0]
    ACC = "eval_acc_so"; lab = lambda o: o["labels"]["main"]  # noqa: E731
    A = "by_cell/5/%d/NUM/"
    cases = [
        # (name, patches, drop, score kwargs, predicate)
        ("all pass -> N1-PASS, ORD-SCOPE, C2-KPRIME, TH6-NOT, B1-PASS", [], [], {},
         lambda o: lab(o) == "N1-PASS" and o["labels"]["ORD-SCOPE"] == "ORD-SCOPE" and o["labels"]["C2-KPRIME"] == "C2-KPRIME"
         and o["labels"]["TH6"] == "TH6-NOT-SUPPORTED" and o["labels"]["B1"] == "B1-PASS" and o["c2_1"]["claim"] == "PASS"),
        ("L1 fails (m 2 0.93)", [("loop", ACC, A % 2 + "acc", 0.93)], [], {}, lambda o: lab(o) == "FAIL: L1"),
        ("C2-1 fails (m 8 0.70)", [("loop", ACC, A % 8 + "acc_moved", 0.70)], [], {}, lambda o: lab(o) == "FAIL: C2-1"),
        ("C2-1 fails (m 6 0.80)", [("loop", ACC, A % 6 + "acc_moved", 0.70)], [], {}, lambda o: lab(o) == "FAIL: C2-1"),
        ("KC K'0 above floor + 0.05", [("loop", "eval_kcurve_so", "acc_moved/0", 0.60)], [], {}, lambda o: lab(o) == "FAIL: KC"),
        ("KC flat K'-curve (K'4 0.60)", [("loop", "eval_kcurve_so", "acc_moved/4", 0.60)], [], {}, lambda o: lab(o) == "FAIL: KC"),
        ("KC K'8 - K'4 < 0.40", [("loop", "eval_kcurve_so", "acc_moved/4", 0.30), ("loop", "eval_kcurve_so", "acc_moved/8", 0.65)], [], {},
         lambda o: lab(o) == "FAIL: KC"),
        ("M1 CWS t 6 0.85", [("loop", "eval_ws_so", "by_m/8/cws/6", 0.85)], [], {}, lambda o: lab(o) == "FAIL: M1"),
        ("M2 follow 0.30", [("loop", "eval_swap_so", "by_cell/5_8_4/follow_cf", 0.30)], [], {}, lambda o: lab(o) == "FAIL: M2"),
        ("M2 n_pairs 0 -> VOID cell -> FAIL", [("loop", "eval_swap_so", "by_cell/5_8_4/n_pairs", 0)], [], {}, lambda o: lab(o) == "FAIL: M2"),
        ("M2 n_pairs 999 -> VOID cell -> FAIL", [("loop", "eval_swap_so", "by_cell/5_8_4/n_pairs", 999)], [], {}, lambda o: lab(o) == "FAIL: M2"),
        ("M2 MEAN + 0.30 clause (follow 0.55, MEAN 0.30)", [("loop", "eval_swap_so", "by_cell/5_8_4/follow_cf", 0.55),
                                                           ("loop", "eval_swap_so", "by_cell/5_8_4/mean_floor", 0.30)], [], {}, lambda o: lab(o) == "FAIL: M2"),
        ("M3 fails (all seeds) -> C2-1 not read, DECODER", [("loop", "eval_swap_so", "by_cell/5_4_2/follow_cf", 0.30)], [], {},
         lambda o: lab(o) == "FAIL: C2-1,M3" and o["labels"]["flags"][S1] == ["DECODER"] and o["loop"][S1]["rows"]["C2-1"]["raw_ok"]),
        ("M3 fails in one seed -> that seed C2-1 FAIL, arm still N1-PASS", [(S2, "eval_swap_so", "by_cell/5_4_2/follow_cf", 0.30)], [], {},
         lambda o: lab(o) == "N1-PASS" and o["rows_k3"]["C2-1"]["k"] == 2 and "C2-1" in o["loop"][S2]["fail"]),
        ("M4 decoder (acc vs S_4 0.9)", [("loop", "eval_kcurve_so", "not_decoder/acc_vs_s4", 0.9)], [], {}, lambda o: lab(o) == "FAIL: M4"),
        ("M4 S_4 half alone at 0.07 (> 0.05, Note 271-1) -> FAIL: M4", [("loop", "eval_kcurve_so", "not_decoder/acc_vs_s4", 0.07)], [], {},
         lambda o: lab(o) == "FAIL: M4" and o["loop"][S1]["rows"]["M4"]["cells"]["vs_s2"]["ok"]),
        ("M4 S_2 half alone at 0.75 -> FAIL: M4", [("loop", "eval_kcurve_so", "not_decoder/acc_vs_s2", 0.75)], [], {}, lambda o: lab(o) == "FAIL: M4"),
        ("kappa from raw counts, not the acc field (n_correct_moved 872/1050 -> kappa 0.75) -> DEPTH-SPLIT",
         [((D1, D2), ACC, A % 8 + "n_correct_moved", 872)], [], {}, lambda o: lab(o) == "DEPTH-SPLIT"),
        ("counts k > n -> malformed -> pending", [(S1, ACC, A % 8 + "n_correct_moved", 2000)], [], {}, lambda o: lab(o) == "pending"),
        ("floors file missing -> pending", [], [("floors", "floors_so")], {}, lambda o: lab(o) == "pending" and not o["floors"].get("sha256")),
        ("floors file missing --final -> every seed VOID -> FAIL", [], [("floors", "floors_so")], {"final": True}, lambda o: lab(o).startswith("FAIL")),
        ("floors sha256 echoed; frozen sha mismatch -> KC (and X1) fail", [], [], {"frozen": {"floors_sha256": "0" * 64, "f": {}}},
         lambda o: lab(o) == "FAIL: KC" and len(o["floors"]["sha256"]) == 64 and o["floors"]["sha_ok"] is False),
        ("frozen sha matches -> N1-PASS", [], [], {"frozen": {"floors_sha256": "MATCH", "f": {}}}, lambda o: lab(o) == "N1-PASS" and o["floors"]["sha_ok"] is True),
        ("no pin (floors_sha256 None) -> pending FLOORS-UNPINNED", [], [], {"frozen": {"floors_sha256": None, "f": {}}},
         lambda o: lab(o) == "pending" and o["labels"]["reason"] == "FLOORS-UNPINNED"),
        ("no pin --final -> still pending FLOORS-UNPINNED", [], [], {"frozen": {"floors_sha256": None, "f": {}}, "final": True},
         lambda o: lab(o) == "pending" and o["reason"] == "FLOORS-UNPINNED"),
        ("DIRECT-OF extrapolates (descriptive X1-ANS FAIL) -> N1-PASS stands", [("dof", ACC, A % 8 + "acc_moved", 0.60)], [], {},
         lambda o: lab(o) == "N1-PASS" and o["labels"]["X1-ANS (DIRECT-OF, descriptive)"] == "FAIL"),
        ("DIRECT-OF missing -> reported in desc_missing, N1-PASS stands", [], [("dof", ACC)], {},
         lambda o: lab(o) == "N1-PASS" and any("DIRECT-OF" in x for x in o["desc_missing"])),
        ("DIRECT-OF kappa 0.75 alone does not fire the gate", [("dof", ACC, A % 8 + "acc_moved", 0.83)], [], {}, lambda o: lab(o) == "N1-PASS"),
        ("X2 read on DIRECT-SI only (DIRECT-OF 3/3 X2 -> TH6-NOT)", [("dof", ACC, "by_cell/3/8/NUM/acc_moved", 0.70)], [], {},
         lambda o: o["labels"]["TH6"] == "TH6-NOT-SUPPORTED" and o["labels"]["TH6 DIRECT-OF (descriptive)"] == "TH6-SUPPORTED"),
        ("per-seed C2-1 NEAR-SEED label (seed 1 m 8 0.85)", [(S1, ACC, A % 8 + "acc_moved", 0.85)], [], {},
         lambda o: o["c2_1"]["per_seed_label"][S1]["m8"] == "NEAR-SEED" and o["c2_1"]["per_seed_label"][S2]["m8"] == "PASS"
         and o["c2_1"]["per_seed_label"][S1]["m6"] == "PASS"),
        ("per-seed C2-1 PASS label when far above (m 6 0.97, m 8 0.95)", [("loop", ACC, A % 6 + "acc_moved", 0.97), ("loop", ACC, A % 8 + "acc_moved", 0.95)], [], {},
         lambda o: o["c2_1"]["per_seed_label"][S1] == {"m6": "PASS", "m8": "PASS"} and o["c2_1"]["claim"] == "PASS"),
        ("M5 strata (k>=5 at 0.60)", [("loop", ACC, A % 8 + "by_k_touch/5/n_correct", 96), ("loop", ACC, A % 8 + "by_k_touch/6/n_correct", 60)], [], {},
         lambda o: lab(o) == "FAIL: M5"),
        ("H1 parity oscillation (K'63 = S_{m-1}, acc 0.58)", [("loop", "eval_long_so", "by_m/3/acc/63", 0.58)], [], {}, lambda o: lab(o) == "FAIL: H1"),
        ("H1 K'64 drop", [("loop", "eval_long_so", "by_m/1/acc/64", 0.90)], [], {}, lambda o: lab(o) == "FAIL: H1"),
        ("X1 fails (DIRECT kappa 0.41, 3/3) -> N1-LOOP-ONLY", [("direct", ACC, A % 8 + "acc_moved", 0.60)], [], {},
         lambda o: lab(o) == "N1-LOOP-ONLY (X1 FAIL)"),
        ("X1 3/3 rule: one DIRECT seed extrapolates -> N1-LOOP-ONLY", [(D2, ACC, A % 8 + "acc_moved", 0.60)], [], {},
         lambda o: lab(o) == "N1-LOOP-ONLY (X1 FAIL)" and o["direct_si_summary"]["X1_k3"] == 2),
        ("D-L1 fails in 2/3 -> VOID-CONTROL, N1-LOOP-ONLY", [((D1, D2), ACC, A % 2 + "acc", 0.85)], [], {},
         lambda o: lab(o) == "N1-LOOP-ONLY (X1 VOID-CONTROL)" and o["labels"]["X1"] == "VOID-CONTROL"),
        ("D-L1 fails in 1/3 -> still read, N1-PASS", [(D1, ACC, A % 2 + "acc", 0.85)], [], {}, lambda o: lab(o) == "N1-PASS"),
        ("DIRECT gate kappa 0.75 in 2/3 -> DEPTH-SPLIT", [((D1, D2), ACC, A % 8 + "acc_moved", 0.83)], [], {}, lambda o: lab(o) == "DEPTH-SPLIT"),
        ("DIRECT gate kappa 0.75 in 1/3 -> not DEPTH-SPLIT (X1 FAIL)", [(D1, ACC, A % 8 + "acc_moved", 0.83)], [], {},
         lambda o: lab(o) == "N1-LOOP-ONLY (X1 FAIL)"),
        ("DEPTH-SPLIT with loop failing -> FAIL", [((D1, D2), ACC, A % 8 + "acc_moved", 0.83), ("loop", "eval_ws_so", "by_m/8/cws/6", 0.85)], [], {},
         lambda o: lab(o) == "FAIL: M1"),
        ("2/3 arm rule: one loop seed fails C2-1 -> N1-PASS", [(S3, ACC, A % 8 + "acc_moved", 0.70)], [], {},
         lambda o: lab(o) == "N1-PASS" and o["loop_pass_k3"] == 2),
        ("2/3 arm rule: two loop seeds fail C2-1 -> FAIL", [((S2, S3), ACC, A % 8 + "acc_moved", 0.70)], [], {}, lambda o: lab(o) == "FAIL: C2-1"),
        ("seed conjunction: each row 2/3 but no seed passes all", [(S1, "eval_ws_so", "by_m/8/cws/6", 0.85), (S2, "eval_long_so", "by_m/1/acc/64", 0.9),
                                                                  (S3, "eval_kcurve_so", "not_decoder/acc_vs_s4", 0.9)], [], {},
         lambda o: lab(o).startswith("FAIL: SEED-CONJUNCTION")),
        ("NEAR cell: L1 m 3 at 0.955 (n 500)", [("loop", ACC, A % 3 + "acc", 0.955)], [], {},
         lambda o: lab(o) == "N1-PASS" and "L1.m3" in o["loop"][S1]["near"]),
        ("NEAR-SEED: C2-1 m 8 seed mean 0.85 (within 0.090 of 0.80)", [("loop", ACC, A % 8 + "acc_moved", 0.85)], [], {},
         lambda o: lab(o) == "N1-PASS" and o["c2_1"]["claim"] == "NEAR-SEED" and o["c2_1"]["m8"]["near_seed"] and abs(o["c2_1"]["per_seed_margins"][S1]["m8"] - 0.05) < 1e-3),
        ("C2-1 seed mean +- SD", [(S1, ACC, A % 8 + "acc_moved", 0.95), (S3, ACC, A % 8 + "acc_moved", 0.85)], [], {},
         lambda o: abs(o["c2_1"]["m8"]["mean"] - 0.90) < 1e-3 and abs(o["c2_1"]["m8"]["sd"] - 0.05) < 1e-3),
        ("missing ws in seed 3 -> pending", [], [(S3, "eval_ws_so")], {}, lambda o: lab(o) == "pending" and o["loop"][S3]["label"] == "pending"),
        ("missing ws in seed 3 --final -> FAIL: VOID counts FAIL, event-only beside, arm 2/3", [], [(S3, "eval_ws_so")], {"final": True},
         lambda o: o["loop"][S3]["label"] == "FAIL: VOID" and lab(o) == "N1-PASS" and o["rows_k3"]["L1"]["k"] == 2 and o["rows_k3"]["L1"]["reached"] == 2),
        ("missing in seeds 2+3 --final -> FAIL", [], [((S2, S3), "eval_long_so")], {"final": True}, lambda o: lab(o).startswith("FAIL")),
        ("malformed acc_so (cell absent) -> pending; --final VOID", [(S1, ACC, "by_cell/5/6", DEL)], [], {},
         lambda o: lab(o) == "pending" and "by_cell/5/6" in o["loop"][S1]["malformed"]),
        ("malformed acc_so --final -> seed VOID", [(S1, ACC, "by_cell/5/6", DEL)], [], {"final": True},
         lambda o: o["loop"][S1]["label"] == "FAIL: VOID" and lab(o) == "N1-PASS"),
        ("missing DIRECT seed -> pending", [], [(D3, ACC)], {}, lambda o: lab(o) == "pending" and o["labels"]["TH6"] == "pending"),
        ("missing DIRECT seed --final -> X1 FAIL (3/3 not met)", [], [(D3, ACC)], {"final": True}, lambda o: lab(o) == "N1-LOOP-ONLY (X1 FAIL)"),
        ("L2 fails alone -> N1-PASS without ORD-SCOPE (+ B1 not binding)", [("loop", ACC, "by_cell/5/2/ORD/acc", 0.80)], [], {},
         lambda o: lab(o) == "N1-PASS" and o["labels"]["ORD-SCOPE"] == "ORD-SCOPE-FAIL" and o["labels"]["B1"].startswith("B1-NOT-BINDING")),
        ("ORD-SCOPE passes while N1 fails", [("loop", ACC, A % 8 + "acc_moved", 0.70)], [], {},
         lambda o: lab(o) == "FAIL: C2-1" and o["labels"]["ORD-SCOPE"] == "ORD-SCOPE"),
        ("C2-KPRIME fails (m 10 K'12 0.50), N1-PASS stands", [("loop", "eval_kcurve_so", "by_m/10/acc_moved/12", 0.50)], [], {},
         lambda o: lab(o) == "N1-PASS" and o["labels"]["C2-KPRIME"] == "C2-KPRIME-FAIL"),
        ("C2-K +0.20 clause (K'12 0.65 vs K'8 0.50)", [("loop", "eval_kcurve_so", "by_m/10/acc_moved/12", 0.65), ("loop", "eval_kcurve_so", "by_m/10/acc_moved/8", 0.50)],
         [], {}, lambda o: o["labels"]["C2-KPRIME"] == "C2-KPRIME-FAIL"),
        ("X2 3/3 -> TH6-SUPPORTED (N1-PASS unchanged)", [("direct", ACC, "by_cell/3/8/NUM/acc_moved", 0.70)], [], {},
         lambda o: o["labels"]["TH6"] == "TH6-SUPPORTED" and lab(o) == "N1-PASS"),
        ("X2 2/3 -> TH6-NOT-SUPPORTED", [((D1, D2), ACC, "by_cell/3/8/NUM/acc_moved", 0.70)], [], {}, lambda o: o["labels"]["TH6"] == "TH6-NOT-SUPPORTED"),
        ("X2 at chance (raw gap 0.25, kappa 0) -> TH6-NOT-SUPPORTED", [("direct", ACC, "by_cell/3/8/NUM/acc_moved", 0.54), ("direct", ACC, A % 8 + "acc_moved", 0.29)],
         [], {}, lambda o: o["labels"]["TH6"] == "TH6-NOT-SUPPORTED" and abs(o["direct_si"][D1]["rows"]["X2"]["raw_gap"] - 0.25) < 1e-3),
        ("X2 Wilson LB clause: kappa ok but LB <= f(3,8) (n_moved 30)", [("direct", ACC, "by_cell/3/8/NUM/acc_moved", 0.70), ("direct", ACC, "by_cell/3/8/NUM/n_moved", 30)],
         [], {}, lambda o: o["labels"]["TH6"] == "TH6-NOT-SUPPORTED" and not o["direct_si"][D1]["rows"]["X2"]["lb3_gt_f3"]),
        ("NORM-DRIFT-CANDIDATE when C2-1 fails with r_8 2.0 and CWS drop", [("loop", ACC, A % 8 + "acc_moved", 0.60), ("loop", "eval_kcurve_so", "s_norm/8/r_median/8", 2.0),
                                                                           ("loop", "eval_ws_so", "by_m/8/cws/8", 0.85)], [], {},
         lambda o: lab(o) == "FAIL: C2-1,M1" and set(o["labels"]["routing"].values()) == {"NORM-DRIFT-CANDIDATE"}),
        ("LOOKUP-BOUND takes precedence over NORM-DRIFT", [("loop", ACC, A % 8 + "acc_moved", 0.60), ("loop", "eval_kcurve_so", "s_norm/8/r_median/8", 2.0),
                                                          ("loop", "eval_ws_so", "by_m/8/cws/8", 0.85), ("loop", "eval_lookup", "by_L/9/acc", 0.5)], [], {},
         lambda o: set(o["labels"]["routing"].values()) == {"LOOKUP-BOUND"}),
        ("no route when r_8 in range", [("loop", ACC, A % 8 + "acc_moved", 0.60)], [], {},
         lambda o: set(o["labels"]["routing"].values()) == {"none (no pre-stated route)"}),
        ("routing absent when C2-1 passes", [], [], {}, lambda o: o["labels"]["routing"] == {}),
        ("P-M3 violation -> DECODER-SIDE flag", [("loop", "eval_ws_so", "by_m/8/cws/%d" % t, 0.97) for t in range(1, 9)] + [("loop", ACC, A % 8 + "acc", 0.95)], [], {},
         lambda o: o["labels"]["flags"][S1] == ["DECODER-SIDE"] and lab(o) == "N1-PASS"),
        ("BBH unparsed items counted wrong -> B1-FAIL", [("loop", "eval_bbh", "subsets/five/per_item",
                                                          [{"parsed": i >= 30, "correct": i < 170} for i in range(250)])], [], {},
         lambda o: o["labels"]["B1"] == "B1-FAIL" and abs(o["loop"][S1]["b1_five"]["value"] - 0.56) < 1e-9),
        ("BBH missing, precondition met: pending; --final B1-FAIL", [], [("loop", "eval_bbh")], {"final": True}, lambda o: o["labels"]["B1"] == "B1-FAIL"),
        ("frozen floor mismatch -> X1 / KC fail, noted", [], [], {"frozen": {"floors_sha256": "MATCH", "f": {"5/8/NUM": 0.25, "3/8/NUM": 0.539}}},
         lambda o: lab(o) == "FAIL: KC" and any("FLOOR-MISMATCH" in x for x in o.get("floor_notes", []))),
        ("frozen floors match -> N1-PASS", [], [], {"frozen": {"floors_sha256": "MATCH", "f": {"5/8/NUM": 0.326, "3/8/NUM": 0.539}}}, lambda o: lab(o) == "N1-PASS"),
        ("STABLE: unrecovered loss jump in every loop seed -> FAIL: STABLE", [], [], {"_unstable": ("loop",)}, lambda o: lab(o) == "FAIL: STABLE"),
        ("STABLE: one loop seed unstable -> 2/3, N1-PASS", [], [], {"_unstable": (S2,)},
         lambda o: lab(o) == "N1-PASS" and o["loop"][S2]["label"] == "FAIL: STABLE" and o["rows_k3"]["STABLE"]["k"] == 2),
        ("STABLE: one DIRECT-SI seed unstable -> X1 2/3 -> N1-LOOP-ONLY", [], [], {"_unstable": (D3,)},
         lambda o: lab(o) == "N1-LOOP-ONLY (X1 FAIL)" and o["direct_si_summary"]["X1_k3"] == 2),
        ("STABLE: blank log in a loop seed -> pending", [], [], {"_nolog": (S1,)}, lambda o: lab(o) == "pending" and o["loop"][S1]["label"] == "pending"),
        ("STABLE: blank log --final -> seed VOID", [], [], {"_nolog": (S1,), "final": True},
         lambda o: o["loop"][S1]["label"] == "FAIL: VOID" and lab(o) == "N1-PASS"),
        ("STABLE: blank DIRECT-SI log -> pending", [], [], {"_nolog": (D1,)}, lambda o: lab(o) == "pending"),
        ("KC K'1 alone above floor + 0.05", [("loop", "eval_kcurve_so", "acc_moved/1", 0.60)], [], {},
         lambda o: lab(o) == "FAIL: KC" and [k for k, c in o["loop"][S1]["rows"]["KC"]["cells"].items() if not c["ok"]] == ["k1"]),
        ("X2 kappa3 >= 0.15 clause alone (a3 0.60, a5 0.15) -> TH6-NOT-SUPPORTED",
         [("direct", ACC, "by_cell/3/8/NUM/acc_moved", 0.60), ("direct", ACC, A % 8 + "acc_moved", 0.15)], [], {},
         lambda o: o["labels"]["TH6"] == "TH6-NOT-SUPPORTED" and o["direct_si"][D1]["rows"]["X2"]["cells"]["kappa_gap"]["ok"]
         and not o["direct_si"][D1]["rows"]["X2"]["cells"]["kappa3>=0.15"]["ok"] and o["direct_si"][D1]["rows"]["X2"]["lb3_gt_f3"] and lab(o) == "N1-PASS"),
        ("X2 kappa-gap SE uses the raw accuracies (Audit LOW a)", [], [], {},
         lambda o: abs((o["direct_si"][D1]["rows"]["X2"]["cells"]["kappa_gap"]["ci95"][1] - o["direct_si"][D1]["rows"]["X2"]["cells"]["kappa_gap"]["value"]) / 1.96
                       - math.sqrt(0.58 * 0.42 / 1050 / (1 - 0.539) ** 2 + 0.35 * 0.65 / 1050 / (1 - 0.326) ** 2)) < 1e-3),
        ("C2-3 alone (ORD n 5 m 8 0.55) -> B1 not binding, N1-PASS, ORD-SCOPE", [("loop", ACC, "by_cell/5/8/ORD/acc", 0.55)], [], {},
         lambda o: lab(o) == "N1-PASS" and o["labels"]["ORD-SCOPE"] == "ORD-SCOPE" and o["labels"]["B1"].startswith("B1-NOT-BINDING")),
        ("floor >= 1 -> Malformed -> pending", [("floors", "floors_so", "floors/5/8/NUM/f", 1.0)], [], {},
         lambda o: lab(o) == "pending" and "outside [0, 1)" in o["loop"][S1]["malformed"]),
        ("descriptive inputs missing are listed (TEXT-CoT, lookup), N1-PASS unchanged", [], [(RUNS["textcot"], ACC), (S1, "eval_lookup")], {},
         lambda o: lab(o) == "N1-PASS" and any("textcot_s1/eval_acc_so (file)" in x for x in o["desc_missing"])
         and any("eval_lookup (file)" in x for x in o["desc_missing"]) and o["descriptive"]["C3_loop_ge_cot_minus_0.05"] is None),
    ]
    for name, pt, dr, kw, pred in cases:
        kw = dict(kw); d = world(pt, dr, kw.pop("_unstable", ()), kw.pop("_nolog", ()))
        try:
            o = score(d, **pin(d, kw)); ok = bool(pred(o)); msg = o["verdict"]
        except Exception as e:  # noqa: BLE001
            ok, msg = False, "EXC %r" % e
        finally:
            shutil.rmtree(d, ignore_errors=True)
        res.append((name, ok, msg))
    P = RUNS["pilot"]; DIRS = RUNS["direct_si"]; DOFS = RUNS["direct"]
    s1cases = [
        ("S1 all four rows -> GO, DIRECT-ALIVE", [], [], {}, lambda o: o["gate"] == "GO" and o["direct_gate"] == "DIRECT-ALIVE"),
        ("S1 CWS t 3 0.85 -> EXTEND", [(P, "eval_ws_so", "by_m/4/cws/3", 0.85)], [], {}, lambda o: o["gate"] == "EXTEND" and o["fail"] == ["CWS"]),
        ("S1 M3p follow 0.40 -> EXTEND", [(P, "eval_swap_so", "by_cell/5_4_2/follow_cf", 0.40)], [], {}, lambda o: o["gate"] == "EXTEND"),
        ("S1 M3p MEAN + 0.30 clause -> EXTEND", [(P, "eval_swap_so", "by_cell/5_4_2/follow_cf", 0.55), (P, "eval_swap_so", "by_cell/5_4_2/mean_floor", 0.30)], [], {},
         lambda o: o["gate"] == "EXTEND"),
        ("S1 M3p n_pairs 500 -> VOID cell -> EXTEND", [(P, "eval_swap_so", "by_cell/5_4_2/n_pairs", 500)], [], {}, lambda o: o["gate"] == "EXTEND"),
        ("S1 M4p acc vs S_2 0.70 -> EXTEND", [(P, "eval_kcurve_so", "not_decoder/acc_vs_s2", 0.70)], [], {}, lambda o: o["gate"] == "EXTEND"),
        ("S1 M4p acc vs S_4 0.15 -> EXTEND", [(P, "eval_kcurve_so", "not_decoder/acc_vs_s4", 0.15)], [], {}, lambda o: o["gate"] == "EXTEND"),
        ("S1 L1p m 3 0.85 -> EXTEND", [(P, ACC, A % 3 + "acc", 0.85)], [], {}, lambda o: o["gate"] == "EXTEND"),
        ("S1 L1p m 4 0.45 -> PARK", [(P, ACC, A % 4 + "acc", 0.45)], [], {}, lambda o: o["gate"] == "PARK"),
        ("S1 re-gate after EXTEND fails -> NO-GO", [(P, "eval_ws_so", "by_m/4/cws/3", 0.85)], [], {"extended": True}, lambda o: o["gate"].startswith("NO-GO")),
        ("S1 re-gate after EXTEND passes -> GO", [], [], {"extended": True}, lambda o: o["gate"] == "GO"),
        ("S1 pilot ws missing -> pending; DIRECT still read", [], [(P, "eval_ws_so")], {}, lambda o: o["gate"] == "pending" and o["direct_gate"] == "DIRECT-ALIVE"),
        ("S1 pilot ws missing --final -> FAIL: VOID", [], [(P, "eval_ws_so")], {"final": True}, lambda o: o["gate"] == "FAIL: VOID"),
        ("S1 DIRECT gate kappa 0.75 in 2/3 -> DEPTH-SPLIT", [((DIRS[0], DIRS[1]), ACC, A % 8 + "acc_moved", 0.83)], [], {},
         lambda o: o["direct_gate"] == "DEPTH-SPLIT" and o["gate"] == "GO"),
        ("S1 DIRECT-OF kappa 0.75 in 3/3 alone -> DIRECT-ALIVE (reported beside)", [("dof", ACC, A % 8 + "acc_moved", 0.83)], [], {},
         lambda o: o["direct_gate"] == "DIRECT-ALIVE" and o["direct_of_gate_desc"] == "DEPTH-SPLIT-like"),
        ("S1 M4p S_4 at 0.07 (bar 0.05) -> EXTEND", [(P, "eval_kcurve_so", "not_decoder/acc_vs_s4", 0.07)], [], {}, lambda o: o["gate"] == "EXTEND"),
        ("S1 floors sha echoed", [], [], {}, lambda o: len(o["floors"]["sha256"]) == 64),
        ("S1 no pin -> pending FLOORS-UNPINNED", [], [], {"frozen": {"floors_sha256": None, "f": {}}},
         lambda o: o["gate"] == "pending" and o["direct_gate"] == "pending" and o["reason"] == "FLOORS-UNPINNED"),
        ("S1 floors sha mismatch -> DIRECT-SI gate rows fail (no kill)", [((DIRS[0], DIRS[1]), ACC, A % 8 + "acc_moved", 0.83)], [],
         {"frozen": {"floors_sha256": "0" * 64, "f": {}}}, lambda o: o["direct_gate"] == "DIRECT-ALIVE" and o["floors"]["sha_ok"] is False),
        ("S1 DIRECT-SI unstable seeds do not fire the gate", [((DIRS[0], DIRS[1]), ACC, A % 8 + "acc_moved", 0.83)], [], {"_unstable": (DIRS[0],)},
         lambda o: o["direct_gate"] == "DIRECT-ALIVE"),
        ("S1 DIRECT gate kappa 0.75 in 1/3 -> DIRECT-ALIVE", [(DIRS[0], ACC, A % 8 + "acc_moved", 0.83)], [], {}, lambda o: o["direct_gate"] == "DIRECT-ALIVE"),
        ("S1 DIRECT gate 1 kill + 1 pending -> pending", [(DIRS[0], ACC, A % 8 + "acc_moved", 0.83)], [(DIRS[1], ACC)], {}, lambda o: o["direct_gate"] == "pending"),
        ("S1 DIRECT gate 2 kills + 1 pending -> DEPTH-SPLIT", [((DIRS[0], DIRS[1]), ACC, A % 8 + "acc_moved", 0.83)], [(DIRS[2], ACC)], {},
         lambda o: o["direct_gate"] == "DEPTH-SPLIT"),
        ("S1 DIRECT missing --final counts as no kill", [(DIRS[0], ACC, A % 8 + "acc_moved", 0.83)], [(DIRS[1], ACC)], {"final": True},
         lambda o: o["direct_gate"] == "DIRECT-ALIVE"),
        ("S1 NEAR flag on a gate cell (L1p m 2 0.905)", [(P, ACC, A % 2 + "acc", 0.905)], [], {}, lambda o: o["gate"] == "GO" and "L1p.m2" in o["near"]),
    ]
    for name, pt, dr, kw, pred in s1cases:
        kw = dict(kw); d = world(pt, dr, kw.pop("_unstable", ()), kw.pop("_nolog", ()))
        try:
            o = score_s1(d, **pin(d, kw)); ok = bool(pred(o)); msg = o["verdict"]
        except Exception as e:  # noqa: BLE001
            ok, msg = False, "EXC %r" % e
        finally:
            shutil.rmtree(d, ignore_errors=True)
        res.append((name, ok, msg))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], "::", r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true"); ap.add_argument("--final", action="store_true")
    ap.add_argument("--stage", choices=("s1", "full"), default="full"); ap.add_argument("--pilot", default=None); ap.add_argument("--extended", action="store_true")
    a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    if a.stage == "s1":
        o = score_s1(final=a.final, pilot=a.pilot, extended=a.extended); out = OUT_S1
    else:
        o = score(final=a.final); out = OUT
        if a.final and o["desc_missing"]:
            print("NOTE (--final): descriptive inputs missing:", "; ".join(o["desc_missing"]))
    print(o["verdict"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(o, fh, indent=1, default=str)
