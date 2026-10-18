"""THEORY §216 scoring (SD2 = SELF-DEPTH-CHAIN round 2 + the s2 lineage; registered text docs/research_selfdepth2_2026-10-18/SD2_v1.md at e9ade2ec, v1 +
Addendum 1; THEORY.md §216; Theory is the only scorer). Lineage table LIN (v1 §4.6); run dirs under R (boxes/runs_fsq; set_root() redirects every module).
Notation a(n) = acc at K' n + 2, a_k(n) = acc at K' k. Sources (v1 §4.0, D3): n <= 16-only readings from eval_vb.json (bs 32); every reading touching any
  n >= 17 from eval_deep.json only (deep_read asserts it opens nothing else). BS-CHECK compares the two at n 13-16 (descriptive; flagged > 0.005).
Pending (215-7; v1 §4.6, RT M5): a cell reads "pending (missing [...])" while any registered file / part / field is missing and reads nothing else first;
  pending never passes. --final <lineage>: still-missing registered files fail by rule b (log, args, draws_log, eval_vb, eval_selfdepth, pool_meta,
  eval_gate -> censor; eval_deep -> every n >= 17 reading fails, mechanism UNLABELLED; steppos / lfk -> LENGTH-FREE fails; hist24 / E4 -> KEEPS fails).
  Descriptive files never gate. A cell in NOT_ADMITTED reads UNREAD (never a pass), also under --final. Addendum 1, no mixed cards within a cell
  (Theory 2026-10-18): every registered eval file of the cell that carries a card field (eval_vb / deep / steppos / lfk / hist24 / selfdepth) and the
  lineage's bar-input eval_vb files (base, R1) must contain CARD_CLASS; a mismatch reads "pending (card mismatch: [...])" and is VOID at --final.
Censors (v1 §5.2, rule b; each fails every reading of the cell): dead = incomplete log (last logged step < 12,000 FT / < 48,000 s2 base; skipped
  non-finite steps, vbank's {"nonfinite": 1} records, are counted in nonfinite_steps and are not dead; Theory 2026-10-18);
  FT-DIVERGED (a(n) < 0.99 or a_24(n) < 0.97 at any n 1-8, eval_vb); LABEL-NOISY-2 (eval_selfdepth label_audit: pi >= 0.99 and pi_E >= 0.99 pooled and
  per window, each window's Wilson CI overlapping its gate window's CI, for pi and for pi_E); CONTAMINATED (pool_test_hits > 0 or pool_excl_hits > 0,
  self_reuse > 0, or the scorer's own re-check of the consumed pool records vs test_set(1000, 24) when the pool file still exists); ARGS_BAD (§4.6 allowed
  sets / required values; pause on); both labellers failing a window's gate (no round 2). eval_deep missing at --final, or items_sha256 / meta / acc-vs-items
  mismatch -> DEEP void: every n >= 17 reading fails (DBS-2, MARGIN-GROWS-2, LIT-C2-2), mechanism UNLABELLED.
R2 readings (r2_cell): ABSORB-2 min a(13..16) >= 0.95; RETAIN min a(9..12) >= 0.99 (else ERODED); DEPTH-GENERAL-2 mean a(13..16) >= 0.80 (desc.);
  THINK-NECESSARY-DEEP-2 max(a_0, a_1) <= 0.15 at n 13..16 (desc.); K'24-STABLE a_24 >= 0.99 at n 1..8; K'24-WIDE (desc.); LENGTH-FREE (score_s213.no_knee
  and score_s212.lfk MEETS); VB-C1 (score_s211.cell_read parts; its 48k dead / NO-ENTRY censors ignored). eval_deep: M2 = mean a(17..20); DBS-2 M2 >= 0.80;
  MARGIN-GROWS-2 M2 >= MG; M_far = mean a(21..24); LIT-C2-2 mean_{22..24}[a - a_20] >= 0.10; STOP-LOC-2 G(k) = mean over n in [k + 1, min(k + 4, 24)] of
  a(n) - a_k(n), k 13..23, STOP = min k with G <= 0.02 else "24+", STOP-TO-SUPPORT-2 iff STOP >= 17; n50 / n90 first downward crossing over n 13..24
  (linear; "> 24" never; "< 13" if a(13) below); WIDTH-2 = n50 - n90; KB2 mean_{19,20}[a - a_18] (">= 0.25" reported); RISE-PAST-K-2 mean_{19,20}
  [a - a_20]; C2-TAIL-2 a_28 >= a - 0.05 at n 17..20; NOT-SIZE-KEYED-2 T-split at T_train = pool_meta T_max_by_n[16], H_T flag iff the T > T_train deficit
  >= 0.10 at >= 2 of n 19..22 with >= 100 items per band (desc.).
Bars: r1_bars / r2_bars verbatim from v1_checks/v1_panels.py (ceil005), from the lineage's own base / R1 eval_vb (REF_EVAL_VB; card CARD_CLASS).
  Mechanism2 (only under ABSORB-2): FULL M2 >= max(0.80, MG) and M_far >= 0.60 (separability flag beside); GROW M2 >= MG; SHIFT SB <= M2 < MG and STOP >= 17;
  MIXED SB <= M2 < MG and STOP <= 16; SHRINK M2 < SB; UNLABELLED (not ABSORB-2) / (eval_deep void) / (bars degenerate: SB >= MG) / (censored).
Kept: KEEPS-SELF-STAB-2 = score_s213.hist24 gate and the E4 store consistent (hist24_items: every arm's per-n means and STAB to 1e-12) and CONTENT-STABILISES
  and STAB_self >= 0.5; else MEM-TRADED-2 + the POST HOC sub-label. MEM history base -> R1 -> R2; MEM-ERODING iff R2 < R1 - 0.05. MEMO-GAP-2 desc.
Gates: per window GATE-1's rule (pi_pool >= 0.995, pi_n >= 0.99 every n, pi_E_pool >= 0.99, keep_n >= 0.70 every n), recomputed item-weighted from by_n,
  and it must equal the file's GATE_2 / GATE_1p (and pooled_by_window); a failed window falls back to the base labeller's entry (BASE-LABELLED).
Claims (v1 §5.1): SD2-SELF = ABSORB-2 and RETAIN and LENGTH-FREE and VB-C1 and K'24-STABLE, uncensored; SD2-C2 = SD2-SELF and DBS-2; SD2-C2-MEM = SD2-C2
  and KEEPS-SELF-STAB-2; LIT-C2-2 on the same line. s2 lineage: base (COMPETENT-BASE = 48k completed and a(n) >= 0.99 and a_24(n) >= 0.97 at n 1..8)
  -> GATE-1-s2 -> R1-s2 (r1_cell = SELFDEPTH_v1 §4.2-§4.3 / §5.2 with lineage paths and the R1 bar formulas; round-2 condition) -> GATE-2-s2 -> R2-s2.
  x2 (v1 §4.5): PASS / FAIL (stage named) / NOT COMPETENT / UNREAD.
    python theory/score_s216.py [--final s1] [--final s2] [--not_admitted <cell>,...]
"""
import argparse
import builtins
import hashlib
import json
import math
import os
import sys

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = _HR if os.path.isdir(os.path.join(_HR, "boxes", "runs_fsq")) else "D:/Latent"    # a scratch copy reads (never writes) the repo
R = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(ROOT, "theory", "results", "score_s216.json")
for _p in (ROOT, os.path.join(ROOT, "theory")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import score_s211 as S211  # noqa: E402
import score_s212 as S212  # noqa: E402
import score_s213 as S213  # noqa: E402
import score_s215 as S215  # noqa: E402

EPS = 1e-9
LIN = {"s1": {"base": "s213_d3t_s1", "r1": "s215_ftself_s1", "r2": "s216_r2_s1", "pool_r1": "s215_pool", "pool_r2": "s216_pool_s1", "gate1": "s215_gate1",
              "gate2": "s216_gate2_s1"},
       "s2": {"base": "s216_d3t_s2", "r1": "s216_r1_s2", "r2": "s216_r2_s2", "pool_r1": "s216_pool_r1_s2", "pool_r2": "s216_pool_r2_s2", "gate1": "s216_gate1_s2",
              "gate2": "s216_gate2_s2"}}
SEED = {"s1": 1, "s2": 2}
REF_BASE, REF_FT = "s213_d3t_s1", "s215_ftself_s1"                       # ARGS-DIFF references (v1 §4.6)
POOL_FILE = {"pool_r1": "pool_r1.jsonl", "pool_r2": "pool_r2.jsonl"}      # s1 R2 named in v1 §2.3; the s2 file names follow it [ASSUMED]
FT_FULL, BASE_FULL = 12000, 48000
CARD_CLASS = "3070"        # Addendum 1: the 3070. If Theory records the 3070 as unrecoverable, set "4060" and REF_EVAL_VB to the 4060 re-evals' name (base / R1)
REF_EVAL_VB = "eval_vb.json"
NOT_ADMITTED, FINAL = set(), set()
ALLOW = {"base": {"seed"},
         "r1": {"seed", "init_from", "self_pool", "self_micro", "args_ref", "mem_check_self", "mem_check_steps", "keep_ckpt_steps", "stop_at", "draws_log_steps"},
         "r2": {"init_from", "self_pool", "self_kmax", "self_micro", "args_ref", "mem_check_self", "mem_check_steps", "keep_ckpt_steps", "stop_at"}}
# keys the current trainer writes that s213_d3t_s1/args.json predates, at vbank.py's argparse defaults (feature off): not "unexpected" for the s2 base
BASE_NEUTRAL_NEW = {"pause": "none", "pause_p": 0.5, "pause_max": 8, "mem_check_steps": 1, "self_pool": "", "p_self": 0.0, "self_kmax": 16, "self_kextra": 4,
                    "self_micro": 2, "mem_check_self": False, "draws_log_steps": 0, "init_from": ""}
VB_PARTS = ("acc_n2", "acc_k", "probe", "test_meta", "vswap", "stall", "stepcap")
CARD_FILES = ("eval_vb.json", "eval_deep.json", "eval_steppos.json", "eval_lfk.json", "eval_hist24.json", "eval_selfdepth.json")
CENSOR_FILES = {"log.jsonl", "args.json", "draws_log.json", "eval_vb.json", "eval_selfdepth.json", "pool_meta.json", "eval_gate.json"}
WIN = {"W2": range(13, 17), "W1": range(9, 13)}                           # W2 = GATE-2 (n 13-16), W1 = GATE-1' (n 9-12)
GATE_FLAG = {"W2": "GATE_2", "W1": "GATE_1p"}
WIN_ALIAS = {"W2": ("W2", "w2", "13-16", "13..16", "new", "GATE_2", "GATE-2", "gate2"), "W1": ("W1", "w1", "9-12", "9..12", "old", "GATE_1p", "GATE-1p", "gate1p")}
LA = {"pi": ("pi_label", "pi"), "pi_E": ("pi_state", "pi_E"), "pi_ci": ("pi_label_ci95", "pi_ci95"), "pi_E_ci": ("pi_state_ci95", "pi_E_ci95")}
E4_KEYS = {"items": "items", "ans_n2": "ans_n2", "ans_k24": "ans_k24", "y": "y", "idx": "idx"}
HIST_ARMS = ("intact", "bank_ablate_slva", "self_ablate", "other_ablate", "dup_newest", "dup_foreign")
LAYOUTS = {   # field layouts of files not yet built (v1 §2.6 B1 / B2 / E2 / E4) [ASSUMED]; only the alias tables above need editing if the built names differ
    "E1 eval_deep.json": "v1 §4.2 exactly; items[n].idx may index test_set(1000, 24), its n 13..24 slice, or the item's n (detected per n; used for alignment only)",
    "E2 eval_selfdepth.json": "round 1's fields + label_audit.by_window {W1, W2: {n_audit, pi_label, pi_state, pi_label_ci95, pi_state_ci95}} (aliases WIN_ALIAS, LA); "
                              "items_n2 / items_k24 {n 1..16: [0/1]}, T, idx {n: [...]}; memo {n 9..16}; contamination {pool_test_hits, pool_excl_hits}",
    "E4 eval_hist24.json": "items {arm: {n: {ans_n2: [...], ans_k24: [...]}}, y: {n: [...]}, idx: {n: [...]}} for every arm in the file, n 3..8 (E4_KEYS; y / idx "
                           "also accepted at the file's top level or inside items[arm][n])",
    "B2 eval_gate.json": "by_n {9..16: {window, labeller, items, n_kept, keep_kv, pi, pi_E, pi_ci95, pi_E_ci95, records}} for the round-1-model labeller; "
                         "pooled_by_window {W1, W2: {n_kept, pi, pi_E, pi_ci95, pi_E_ci95}}; GATE_2, GATE_1p (bool); fallback null or {W1|W2: {labeller, by_n, "
                         "pooled, GATE (bool)}}",
    "B1 pool_meta.json": "T_max_by_n {'16': int} (T_train)"}


def set_root(r):
    global R
    R = r
    for m in (S211, S212, S213, S215):
        m.R = r


def jl(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def mean(v):
    v = list(v)
    return sum(v) / len(v)


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def rel(d, f):
    return "boxes/runs_fsq/%s/%s" % (d, f)


def same_path(v, r):
    return isinstance(v, str) and v.replace("\\", "/").endswith(r)


def _win(dct, w):
    if isinstance(dct, dict):
        for k in WIN_ALIAS[w]:
            if k in dct:
                return dct[k]
    return None


def _la(x, k):
    for a in LA[k]:
        if isinstance(x, dict) and x.get(a) is not None:
            return x[a]
    return None


def ci_overlap(a, b):
    return bool(isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 2 and len(b) == 2 and a[0] <= b[1] + EPS and b[0] <= a[1] + EPS)


_TE = {}


def test_items(n_max):
    """boxes.nested170.test_set(1000, n_max) (CPU, deterministic; the import loads torch without touching CUDA)."""
    if n_max not in _TE:
        from boxes import nested170 as N
        _TE[n_max] = N.test_set(1000, n_max=n_max)
    return _TE[n_max]


def items_sha256():
    """v1 §4.2 (m9): sha256 of json.dumps([[syms, int(t)] ...], separators (",", ":"), ensure_ascii) over test_set(1000, 24) n 13..24 in test order."""
    its = [[it["syms"], int(it["t"])] for it in test_items(24) if 13 <= it["n"] <= 24]
    return hashlib.sha256(json.dumps(its, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- bars: verbatim from docs/research_selfdepth2_2026-10-18/v1_checks/v1_panels.py
def ceil005(x):
    return round(math.ceil(x / 0.005 - 1e-6) * 0.005, 3)


def ip(a, x):
    lo = math.floor(x); f = x - lo
    return a[lo] if f < 1e-12 else (1 - f) * a[lo] + f * a[lo + 1]


def r1_bars(a0):
    """Round-1 (B 8 -> 12) bars from the lineage's own base eval_vb: H_shift M = mean a0(9..12) (exact translation by +4)."""
    M0 = mean(a0[n] for n in range(9, 13)); MG = ceil005(M0 + 0.15)
    return {"M_shift": M0, "MARGIN_GROWS": MG, "SHRINK_below": ceil005(M0 - 0.08), "DBS": 0.80, "FULL_M": max(0.80, MG), "K_BEYOND": 0.25}


def r2_bars(a0, a1):
    """Round-2 (B 12 -> 16) bars from the lineage's own base and round-1 eval_vb (acc_n2):
    M2_shift = mean a1(13..16); M2_prop = mean_{d 1..4} a1(12 + 0.75 d) (linear in n); M2_shrink = mean a0(9..12);
    MG2 = ceil005(midpoint shift / prop); SH2 = ceil005(midpoint shrink / shift); FULL needs M2 >= max(0.80, MG2) and M_far >= 0.60 (absolute);
    separability flag: H_prop's M_far upper bound (a1(x > 16) := a1(16), monotone [ASSUMED]) >= 0.60 -> FULL 'not separable from H_prop'."""
    Ms = mean(a1[n] for n in range(13, 17)); Mp = mean(ip(a1, 12 + 0.75 * d) for d in range(1, 5)); Mk = mean(a0[n] for n in range(9, 13))
    MG = ceil005((Ms + Mp) / 2); SH = ceil005((Ms + Mk) / 2)
    mono = lambda x: a1[16] if x >= 16 else ip(a1, x)  # noqa: E731
    up = mean(mono(12 + 0.75 * d) for d in range(5, 9))
    return {"M2_shift": Ms, "M2_prop": Mp, "M2_shrink": Mk, "MARGIN_GROWS_2": MG, "SHRINK_below": SH, "FULL_M2": max(0.80, MG), "FULL_M_far": 0.60,
            "DBS_2": 0.80, "LIT_C2_2": 0.10, "Hprop_M_far_upper": up, "FULL_not_separable_flag": up >= 0.60 - EPS, "degenerate_SH_ge_MG": SH >= MG}
# ---------------------------------------------------------------- end verbatim


def curve(ev):
    return {int(n): v for n, v in ev["acc_n2"].items()}


def bar_inputs(lin):
    """(a0, a1, problems): the lineage's base and R1 acc_n2 (REF_EVAL_VB), each on CARD_CLASS (Addendum 1: every reading compares on one card)."""
    L = LIN[lin]; e0, e1 = jl(L["base"], REF_EVAL_VB), jl(L["r1"], REF_EVAL_VB); bad = []
    for d, e in ((L["base"], e0), (L["r1"], e1)):
        if e is None or "acc_n2" not in e:
            bad.append("%s/%s missing" % (d, REF_EVAL_VB))
        elif CARD_CLASS not in str(e.get("card", "")):
            bad.append("%s card %s (CARD_CLASS %s)" % (d, e.get("card"), CARD_CLASS))
    return (curve(e0) if e0 and "acc_n2" in e0 else None), (curve(e1) if e1 and "acc_n2" in e1 else None), bad


def card_mismatch(pairs):
    """Addendum 1 (no partial move within a cell): each listed file that exists and carries a card field (top level or meta) must contain CARD_CLASS."""
    bad = []
    for d, f in pairs:
        x = jl(d, f)
        for c in ([x.get("card"), (x.get("meta") or {}).get("card")] if isinstance(x, dict) else []):
            if c is not None and CARD_CLASS not in str(c):
                bad.append("%s/%s: %s" % (d, f, c))
    return bad


def _pend(miss):
    files = sorted(v for k, v in miss.items() if k != "card")
    return "pending (%s)" % "; ".join((["missing %s" % files] if files else []) + ([miss["card"]] if "card" in miss else []))


def _void_card(o, miss, extra):
    o.update(status="FAIL", censored=True, label="VOID (%s; Addendum 1; rule b)" % miss["card"], **extra)
    return o


def cross(a, L, lo=13, hi=24):
    """first downward crossing of L over n lo..hi, linear interpolation; '< lo' if a(lo) is already below; '> hi' if never."""
    if a[lo] < L - EPS:
        return "< %d" % lo
    for n in range(lo, hi):
        if a[n] >= L - EPS and L > a[n + 1]:
            return n + (a[n] - L) / (a[n] - a[n + 1])
    return "> %d" % hi


def stop_loc(a, Ak, ks, n_hi, cap):
    """score_s215.stop_loc generalised to a k-range and a source: G(k) = mean over n in [k + 1, min(k + 4, n_hi)] of a(n) - a_k(n); STOP = min k with G <= 0.02."""
    G = {k: mean(a[n] - Ak(k, n) for n in range(k + 1, min(k + 4, n_hi) + 1)) for k in ks}
    return G, next((k for k in ks if G[k] <= 0.02 + EPS), cap)


def _pos(n, idx, lo_n):
    """within-n positions for test-order indices idx (global index into test_set, into its n >= lo_n slice, or within n; detected)."""
    if not isinstance(idx, list) or not idx:
        return None
    for base in ((n - 1) * 1000, (n - lo_n) * 1000, 0):
        if all(isinstance(i, int) and base <= i < base + 1000 for i in idx):
            return [i - base for i in idx]
    return None


def ols(xs, ys):
    mx, my = mean(xs), mean(ys); den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else None


def deep_read(d, T_train=None):
    """v1 §4.2 / §4.3 on eval_deep.json ONLY (asserted: the open() calls made while reading are recorded). Void on a missing file, items_sha256 / meta
    mismatch (card contains CARD_CLASS, dtype fp32, tf32 false, eval_bs in {16, 8}, n 13..24, K' grid 0..28) or acc_k[K'][n] != mean(ans[K'] == y)."""
    sha = items_sha256(); TE = [it for it in test_items(24) if 13 <= it["n"] <= 24]     # built before the spy: the import opens files
    opened, _open = [], builtins.open

    def spy(f, *a, **k):
        opened.append(str(f)); return _open(f, *a, **k)
    builtins.open = spy
    try:
        o = _deep(os.path.join(R, d, "eval_deep.json"), sha, TE, T_train)
    finally:
        builtins.open = _open
    assert all(os.path.basename(p) == "eval_deep.json" for p in opened), ("deep_read opened", opened)
    o["files_opened"] = opened
    return o


def _deep(p, sha, TE, T_train):
    if not os.path.exists(p):
        return {"void": "eval_deep.json missing"}
    f = json.load(open(p)); m = f.get("meta") or {}
    chk = {"items_sha256": m.get("items_sha256") == sha, "card": CARD_CLASS in str(m.get("card", "")), "dtype": m.get("dtype") == "fp32",
           "tf32": m.get("tf32") is False, "eval_bs": m.get("eval_bs") in (16, 8), "n_range": (m.get("n_lo"), m.get("n_hi")) == (13, 24),
           "K_grid": list(m.get("K_grid") or []) == list(range(29))}
    A2, AK, IT = f.get("acc_n2") or {}, f.get("acc_k") or {}, f.get("items") or {}
    chk["fields"] = (sorted(A2, key=int) == [str(n) for n in range(13, 25)] and sorted(AK, key=int) == [str(k) for k in range(29)]
                     and all(sorted(AK[k], key=int) == [str(n) for n in range(13, 25)] for k in AK) and all(str(n) in IT for n in range(13, 25)))
    bad_acc = []
    if chk["fields"]:
        for n in range(13, 25):
            it = IT[str(n)]; y, ans = it.get("y") or [], it.get("ans") or []
            if not y or len(ans) != len(y) or any(len(r) != 29 for r in ans):
                bad_acc.append((n, "items shape")); continue
            for k in range(29):
                v = sum(int(r[k] == yy) for r, yy in zip(ans, y)) / len(y)
                if abs(v - AK[str(k)][str(n)]) > 1e-12:
                    bad_acc.append((n, k))
            if abs(A2[str(n)] - AK[str(n + 2)][str(n)]) > 1e-12:
                bad_acc.append((n, "acc_n2 != acc_k[n + 2]"))
    chk["acc_items"] = chk["fields"] and not bad_acc
    if not all(chk.values()):
        return {"void": "eval_deep %s" % sorted(k for k, v in chk.items() if not v), "checks": chk, "acc_mismatch": bad_acc[:10], "meta": m}
    a = {n: A2[str(n)] for n in range(13, 25)}; Ak = lambda k, n: AK[str(k)][str(n)]  # noqa: E731
    G, stop = stop_loc(a, Ak, range(13, 24), 24, "24+")
    n50, n90 = cross(a, 0.5), cross(a, 0.9)
    width = (n50 - n90) if (num(n50) and num(n90)) else None
    lit_n = {n: a[n] - Ak(20, n) for n in range(21, 25)}; ch = f.get("chance") or {}
    o = {"void": None, "checks": chk, "meta": m, "a": a, "M2": mean(a[n] for n in range(17, 21)), "M_far": mean(a[n] for n in range(21, 25)),
         "LIT_C2_2": mean(lit_n[n] for n in (22, 23, 24)), "LIT_by_n": lit_n, "G": G, "STOP": stop, "STOP_TO_SUPPORT_2": stop == "24+" or stop >= 17,
         "n50": n50, "n90": n90, "WIDTH_2": width, "KB2": mean(a[n] - Ak(18, n) for n in (19, 20)), "RISE_PAST_K_2": mean(a[n] - Ak(20, n) for n in (19, 20)),
         "C2_TAIL_2": all(Ak(28, n) >= a[n] - 0.05 - EPS for n in range(17, 21)), "K24_WIDE_deep": all(Ak(28, n) >= a[n] - 0.05 - EPS for n in range(13, 21))}
    o["KB2_ge_025"] = o["KB2"] >= 0.25 - EPS
    if all(str(n) in ch for n in (22, 23, 24)):                             # §5.3 guard (descriptive): LIT floored at chance, a(n) - chance(n)
        o["LIT_C2_2_floored"] = mean(a[n] - max(Ak(20, n), ch[str(n)]) for n in (22, 23, 24)); o["a_minus_chance"] = {n: a[n] - ch[str(n)] for n in (22, 23, 24)}
    o["y_matches_test"] = all(_pos(n, IT[str(n)].get("idx"), 13) is None or all(TE[(n - 13) * 1000 + q]["t"] == yy for q, yy in
                              zip(_pos(n, IT[str(n)]["idx"], 13), IT[str(n)]["y"])) for n in range(13, 25))
    ts = {}                                                                 # NOT-SIZE-KEYED-2 (descriptive)
    for n in range(17, 23):
        it = IT[str(n)]; T = it.get("T")
        if T_train is None or not T or len(T) != len(it["y"]):
            continue
        c = [int(r[n + 2] == yy) for r, yy in zip(it["ans"], it["y"])]; lo = [x for x, t in zip(c, T) if t <= T_train]; hi = [x for x, t in zip(c, T) if t > T_train]
        ts[n] = {"n_le": len(lo), "n_gt": len(hi), "a_le": mean(lo) if lo else None, "a_gt": mean(hi) if hi else None, "slope_T": ols(T, c)}
        ts[n]["deficit"] = (ts[n]["a_le"] - ts[n]["a_gt"]) if (lo and hi) else None
    q = [n for n in range(19, 23) if n in ts and ts[n]["n_le"] >= 100 and ts[n]["n_gt"] >= 100 and ts[n]["deficit"] >= 0.10 - EPS]
    o["TSPLIT"] = {"T_train": T_train, "by_n": ts, "qualifying_n": q} if ts else None
    o["H_T_flag"] = (len(q) >= 2) if ts else None
    o["M2_lowT"] = mean(ts[n]["a_le"] for n in range(17, 21)) if ts and all(n in ts and ts[n]["a_le"] is not None for n in range(17, 21)) else None
    return o


def bs_check(ev, d, sd):
    """BS-CHECK (v1 §4.0, descriptive): eval_deep vs eval_vb at n 13..16 on a(n) and every shared K'; item flips vs E2's items_n2 / items_k24 (K' n + 2, 24)."""
    f = jl(d, "eval_deep.json")
    if f is None or ev is None:
        return None
    A2, AK, IT = f.get("acc_n2") or {}, f.get("acc_k") or {}, f.get("items") or {}; diffs = []
    for n in range(13, 17):
        if str(n) in A2:
            diffs.append((abs(A2[str(n)] - ev["acc_n2"][str(n)]), n, "n+2"))
        for k in ev["acc_k"]:
            if k in AK and str(n) in AK[k]:
                diffs.append((abs(AK[k][str(n)] - ev["acc_k"][k][str(n)]), n, k))
    mx = max(diffs) if diffs else None; flips = {}
    for key, kk in (("items_n2", None), ("items_k24", 24)):
        E = (sd or {}).get(key) or {}; fl = 0; cmp_ = 0
        for n in range(13, 17):
            it = IT.get(str(n)); e = E.get(str(n))
            if not it or not e:
                continue
            pd = _pos(n, it.get("idx"), 13) or list(range(len(it["y"]))); pe = _pos(n, ((sd or {}).get("idx") or {}).get(str(n)), 1) or list(range(len(e)))
            em = dict(zip(pe, e)); K_ = n + 2 if kk is None else kk
            for q, r, yy in zip(pd, it["ans"], it["y"]):
                if q in em:
                    cmp_ += 1; fl += int(int(r[K_] == yy) != int(em[q]))
        flips[key] = {"compared": cmp_, "flips": fl} if cmp_ else None
    return {"max_abs_delta": mx[0] if mx else None, "at": {"n": mx[1], "K": mx[2]} if mx else None, "item_flips": flips,
            "FLAG": bool(mx and mx[0] > 0.005 + EPS)}


def args_diff(lin, role):
    """v1 §4.6 (RT B1): base vs s213_d3t_s1, R1 / R2 vs s215_ftself_s1; allowed sets + required values; new keys only if allowed (base: BASE_NEUTRAL_NEW)."""
    L = LIN[lin]; d = L[role]; a, b = jl(d, "args.json"), jl(REF_BASE if role == "base" else REF_FT, "args.json")
    if a is None or b is None:
        return None
    allowed = ALLOW[role] | ({"seed"} if (role == "r2" and lin == "s2") else set())
    must = ({"seed": 2} if role == "base" else
            {"seed": 2, "init_from": rel(L["base"], "model.pt"), "self_pool": rel(L["pool_r1"], POOL_FILE["pool_r1"])} if role == "r1" else
            {"seed": SEED[lin], "self_kmax": 20, "init_from": rel(L["r1"], "model.pt"), "self_pool": rel(L["pool_r2"], POOL_FILE["pool_r2"])})
    shared = {k: (b[k], a[k]) for k in a if k in b and k not in S213.META and k not in allowed and a[k] != b[k]}
    new = sorted(k for k in a if k not in b and k not in allowed and not (role == "base" and k in BASE_NEUTRAL_NEW and a[k] == BASE_NEUTRAL_NEW[k]))
    req = {k: (v, a.get(k)) for k, v in must.items() if not (same_path(a.get(k), v) if isinstance(v, str) else a.get(k) == v)}
    pause_on = a.get("pause") not in (None, "none", "")
    return {"ref": REF_BASE if role == "base" else REF_FT, "shared_diffs": shared, "unexpected_new_keys": new, "required_bad": req, "pause_on": pause_on,
            "ok": not shared and not new and not req and not pause_on}


def _recs(d):
    recs = S211.log_recs(d)
    if not recs:
        return None
    nan = sorted(s for s, r in recs.items() if any(isinstance(r.get(k), float) and not math.isfinite(r[k]) for k in ("loss", "answer_ce")))
    nf = []                                                                 # raw lines: log_recs keeps only the last record per step
    for ln in open(os.path.join(R, d, "log.jsonl")):
        if '"nonfinite"' in ln:
            r = json.loads(ln)
            if r.get("nonfinite"):
                nf.append(r.get("step"))
    cur = None
    for s in sorted(recs):
        cur = recs[s].get("pool_cursor", cur)
    return {"recs": recs, "last": max(recs), "nonfinite_steps": len(nf), "nonfinite_step_list": nf[:20], "nan_loss_records": nan[:10], "reuse": max((r.get("self_reuse") or 0) for r in recs.values()), "cursor": cur,
            "c_zero": all((r.get("c") or 0) == 0 for r in recs.values())}


def gate_rule(B, ns):
    """GATE-1's rule on by_n entries over ns, pooled pi / pi_E recomputed item-weighted from n_kept: (ok, pi_pool, pi_E_pool, problem)."""
    es = [(B or {}).get(str(n), (B or {}).get(n)) for n in ns]
    kv = lambda e: e.get("keep_kv", e.get("keep"))  # noqa: E731
    if any(not isinstance(e, dict) or not all(num(e.get(k)) for k in ("n_kept", "pi", "pi_E")) or not num(kv(e)) for e in es):
        return False, None, None, "by_n entries missing"
    nk = sum(e["n_kept"] for e in es)
    if nk <= 0:
        return False, None, None, "no kept items"
    pp, pe = sum(e["pi"] * e["n_kept"] for e in es) / nk, sum(e["pi_E"] * e["n_kept"] for e in es) / nk
    ok = pp >= 0.995 - EPS and pe >= 0.99 - EPS and all(e["pi"] >= 0.99 - EPS and kv(e) >= 0.70 - EPS for e in es)
    return bool(ok), pp, pe, None


def _gate_entry(B, P, flag, ns):
    ok, pp, pe, why = gate_rule(B, ns)
    pm = bool(isinstance(P, dict) and num(P.get("pi")) and num(P.get("pi_E")) and pp is not None and abs(P["pi"] - pp) <= 1e-9 and abs(P["pi_E"] - pe) <= 1e-9)
    cons = isinstance(flag, bool) and flag == ok and pm
    return {"recomputed": ok, "file_flag": flag, "pooled_matches": pm, "consistent": cons, "PASS": bool(ok and flag is True and cons), "pi_pool": pp,
            "pi_E_pool": pe, "problem": why, "ci": {"pi": (P or {}).get("pi_ci95"), "pi_E": (P or {}).get("pi_E_ci95")},
            "labeller": sorted({str(((B or {}).get(str(n)) or {}).get("labeller")) for n in ns})}


def gate1(lin):
    """GATE-1 (round-1 format, as score_s215.gate1) recomputed from by_n; it must equal the file's GATE_1 and pooled values."""
    g = jl(LIN[lin]["gate1"], "eval_gate.json")
    if g is None:
        return None
    e = _gate_entry(g.get("by_n"), g.get("pooled"), g.get("GATE_1"), range(9, 13)); e["GATE_1"] = e["PASS"]
    return e


def gate2(lin):
    """GATE-2 (W2, n 13-16) and GATE-1' (W1, n 9-12) per window (B2 layout, LAYOUTS); status SELF-LABELLED / BASE-LABELLED / BOTH-FAIL / UNDECIDED."""
    g = jl(LIN[lin]["gate2"], "eval_gate.json")
    if g is None:
        return None
    o = {}
    for w, ns in WIN.items():
        pr = _gate_entry(g.get("by_n"), _win(g.get("pooled_by_window"), w), g.get(GATE_FLAG[w]), ns); fb = _win(g.get("fallback"), w); fr = None
        if not pr["PASS"] and isinstance(fb, dict):
            fr = _gate_entry(fb.get("by_n"), fb.get("pooled"), fb.get("GATE", fb.get(GATE_FLAG[w])), ns); fr["labeller"] = fb.get("labeller")
        st = ("UNDECIDED" if not isinstance(pr["file_flag"], bool) else "SELF-LABELLED" if pr["PASS"] else
              "BASE-LABELLED" if (fr and fr["PASS"]) else "BOTH-FAIL" if (fr and isinstance(fr["file_flag"], bool)) else "UNDECIDED")
        if st == "UNDECIDED" and lin in FINAL:
            st = "BOTH-FAIL"                                                # --final: an undecided gate / fallback fails (rule b)
        o[w] = {"status": st, "primary": pr, "fallback": fr, "used": pr if pr["PASS"] else fr if (fr and fr["PASS"]) else None}
    return o


def label_audit2(la, g2):
    """LABEL-AUDIT-2 (v1 §3.3 / §4.4): pi >= 0.99 and pi_E >= 0.99 pooled and per window, each window's CI overlapping its (used) gate window's CI."""
    po = num(_la(la, "pi")) and num(_la(la, "pi_E")) and _la(la, "pi") >= 0.99 - EPS and _la(la, "pi_E") >= 0.99 - EPS; ws = {}
    for w in WIN:
        x = _win(la.get("by_window"), w) or {}; gu = ((g2 or {}).get(w) or {}).get("used") or {}; gci = gu.get("ci") or {}
        ws[w] = {"pi": _la(x, "pi"), "pi_E": _la(x, "pi_E"), "ci_overlap_pi": ci_overlap(_la(x, "pi_ci"), gci.get("pi")),
                 "ci_overlap_pi_E": ci_overlap(_la(x, "pi_E_ci"), gci.get("pi_E"))}
        ws[w]["ok"] = bool(num(ws[w]["pi"]) and num(ws[w]["pi_E"]) and ws[w]["pi"] >= 0.99 - EPS and ws[w]["pi_E"] >= 0.99 - EPS and ws[w]["ci_overlap_pi"]
                           and ws[w]["ci_overlap_pi_E"])
    pats = {}
    for r in la.get("kept_wrong") or []:
        k = "%s n%s" % (r.get("pattern"), r.get("n")); pats[k] = pats.get(k, 0) + 1
    return {"pi": _la(la, "pi"), "pi_E": _la(la, "pi_E"), "pooled_ok": bool(po), "windows": ws, "kept_wrong_patterns": pats,
            "ok": bool(po and all(v["ok"] for v in ws.values()))}


def label_audit1(la, g1):
    """R1-s2 LABEL-AUDIT (SELFDEPTH_v1 §4.3): pi_E >= 0.99 and pi >= 0.99, each CI overlapping GATE-1-s2's pooled CI."""
    gci = (g1 or {}).get("ci") or {}
    o = {"pi": _la(la, "pi"), "pi_E": _la(la, "pi_E"), "ci_overlap_pi": ci_overlap(_la(la, "pi_ci"), gci.get("pi")),
         "ci_overlap_pi_E": ci_overlap(_la(la, "pi_E_ci"), gci.get("pi_E"))}
    o["ok"] = bool(num(o["pi"]) and num(o["pi_E"]) and o["pi"] >= 0.99 - EPS and o["pi_E"] >= 0.99 - EPS and o["ci_overlap_pi"] and o["ci_overlap_pi_E"])
    return o


def contamination2(sd, info, pool_dir, pool_file):
    """CONTAMINATED (v1 §2.2 items 2 / 8, §4.4): E2's pool_test_hits / pool_excl_hits, self_reuse, and, while the pool file exists, the scorer's re-check
    of the consumed records (pool[:cursor]) against test_set(1000, n_max 24) by exact syms (only ever adds a failure)."""
    c = (sd or {}).get("contamination") or {}; p = os.path.join(R, pool_dir, pool_file); rc = None
    if os.path.exists(p) and info and info["cursor"]:
        test = {tuple(it["syms"]) for it in test_items(24)}; hits = rd = 0
        with open(p) as fh:
            for ln in fh:
                if rd >= info["cursor"]:
                    break
                rd += 1; hits += tuple(json.loads(ln)["syms"]) in test
        rc = {"consumed_read": rd, "test24_hits": hits}
    th, eh = c.get("pool_test_hits"), c.get("pool_excl_hits")
    bad = bool((num(th) and th > 0) or (num(eh) and eh > 0) or (info and info["reuse"] > 0) or (rc and rc["test24_hits"] > 0))
    return {"pool_test_hits": th, "pool_excl_hits": eh, "self_reuse": info["reuse"] if info else None, "recheck": rc, "CONTAMINATED": bad}


def _e4(it, f, arm, n, key):
    k = E4_KEYS[key]; v = ((it.get(arm) or {}).get(str(n)) or {}).get(k)
    if v is None:                                                          # y / idx: items[k][n] or the file's top-level k[n]
        for top in (it, f):
            t = top.get(k)
            if isinstance(t, dict) and t.get(str(n)) is not None:
                return t[str(n)]
    return v


def hist24_items(d, f=None):
    """E4 recompute (v1 §2.6 / G-S19): every arm's acc_n2 / acc_k24 at n 3..8 from the per-item store must equal the file's per-n means (1e-12); STAB by
    score_s213.hist24's formula from the item means. None if the file has no store."""
    f = f if f is not None else jl(d, "eval_hist24.json")
    it = (f or {}).get(E4_KEYS["items"])
    if not isinstance(it, dict):
        return None
    rec, bad = {}, []
    for arm in [a for a in HIST_ARMS if a in f]:
        rec[arm] = {"acc_n2": {}, "acc_k24": {}}
        for n in range(3, 9):
            y = _e4(it, f, arm, n, "y")
            for key, ak in (("acc_n2", "ans_n2"), ("acc_k24", "ans_k24")):
                ans = _e4(it, f, arm, n, ak)
                if not y or not ans or len(ans) != len(y):
                    bad.append((arm, key, n, "absent")); continue
                v = sum(int(p == q) for p, q in zip(ans, y)) / len(y); rec[arm][key][str(n)] = v
                if abs(v - f[arm][key][str(n)]) > 1e-12:
                    bad.append((arm, key, n, v, f[arm][key][str(n)]))
    st = None
    if not bad and "intact" in rec:
        g = lambda arm, n: rec[arm]["acc_n2"][str(n)] - rec[arm]["acc_k24"][str(n)]  # noqa: E731
        st = {a: (mean(g(a, n) - g("intact", n) for n in range(3, 7)) if a in rec else None) for a in HIST_ARMS if a != "intact"}
    return {"consistent": not bad and st is not None, "mismatch": bad[:20], "STAB": st}


def keeps2(d, need_items, missing=False):
    """KEEPS-SELF-STAB-2 (v1 §4.4) + the POST HOC MEM sub-label (D5; descriptive, exhaustive, never softens MEM-TRADED-2)."""
    h = None if missing else S213.hist24(d)
    hi = None if (missing or h is None) else hist24_items(d)
    items_ok = (not need_items and hi is None) or bool(hi and hi["consistent"] and all(
        (hi["STAB"].get(a) is None and h["STAB"].get(a) is None) or (hi["STAB"].get(a) is not None and h["STAB"].get(a) is not None
                                                                     and abs(hi["STAB"][a] - h["STAB"][a]) <= 1e-12) for a in h["STAB"]))
    gate = bool(h and h["gate"] and items_ok); ss = (h or {}).get("STAB", {}).get("self_ablate")
    keeps = bool(gate and h["CONTENT"] == "CONTENT-STABILISES" and ss is not None and ss >= 0.5 - EPS)
    sub = (None if keeps else "unreadable (gate)" if not gate else "STAB_self low with dup_newest destabilising" if (h["CONTENT"] == "CONTENT-STABILISES" and ss is not None)
           else "content channel not stabilising" if h["CONTENT"] != "CONTENT-STABILISES" else "unreadable (self arm omitted)")
    return {"hist24": h, "items": hi, "gate_with_items": gate, "STAB_self": ss, "KEEPS_SELF_STAB_2": keeps, "MEM_TRADED_2": not keeps,
            "MEM_SUBLABEL_POST_HOC": sub}


def stab_self(d):
    h = S213.hist24(d) if jl(d, "eval_hist24.json") is not None else None
    return (h or {}).get("STAB", {}).get("self_ablate")


def mem_history(lin, x):
    s0, s1 = stab_self(LIN[lin]["base"]), stab_self(LIN[lin]["r1"])
    return {"base": s0, "r1": s1, "r2": x, "MEM_ERODING": None if (x is None or s1 is None) else bool(x < s1 - 0.05 - EPS)}


def mechanism2(absorb2, dr, b, censored):
    if censored:
        return "UNLABELLED (censored)"
    if not absorb2:
        return "UNLABELLED (not ABSORB-2)"
    if dr.get("void"):
        return "UNLABELLED (DEEP void: %s)" % dr["void"]
    if b.get("void"):
        return "UNLABELLED (bars void: %s)" % b["void"]
    if b["degenerate_SH_ge_MG"]:
        return "UNLABELLED (bars degenerate)"
    M2 = dr["M2"]
    if M2 >= b["FULL_M2"] - EPS and dr["M_far"] >= b["FULL_M_far"] - EPS:
        return "FULL"
    if M2 >= b["MARGIN_GROWS_2"] - EPS:
        return "GROW"
    if M2 >= b["SHRINK_below"] - EPS:
        return "SHIFT" if dr["STOP_TO_SUPPORT_2"] else "MIXED"
    return "SHRINK"


def _sd_missing(sd, rnd):
    la = (sd or {}).get("label_audit") or {}; c = (sd or {}).get("contamination") or {}; m = []
    if not all(num(_la(la, k)) for k in ("pi", "pi_E")) or not all(_la(la, k) for k in ("pi_ci", "pi_E_ci")):
        m.append("label_audit")
    if rnd == 2 and not all(isinstance(_win(la.get("by_window"), w), dict) for w in WIN):
        m.append("label_audit.by_window")
    m += ["contamination." + k for k in (("pool_test_hits", "pool_excl_hits") if rnd == 2 else ("pool_test_hits",)) if not num(c.get(k))]
    return m


def _missing(d, files, ev_parts=VB_PARTS):
    miss = {}
    for f in files:
        if not os.path.exists(os.path.join(R, d, f)):
            miss[f] = "%s/%s" % (d, f)
    if "log.jsonl" not in miss and not S211.log_recs(d):
        miss["log.jsonl"] = "%s/log.jsonl (no step records)" % d
    if "eval_vb.json" not in miss and "eval_vb.json" in files:
        p = [k for k in ev_parts if k not in jl(d, "eval_vb.json")]
        if p:
            miss["eval_vb.json"] = "%s/eval_vb.json (parts %s)" % (d, p)
    return miss


REG_R2 = ("log.jsonl", "args.json", "draws_log.json", "eval_vb.json", "eval_deep.json", "eval_steppos.json", "eval_lfk.json", "eval_hist24.json", "eval_selfdepth.json")
REG_R1 = ("log.jsonl", "args.json", "draws_log.json", "eval_vb.json", "eval_steppos.json", "eval_lfk.json", "eval_hist24.json", "eval_selfdepth.json")


def r2_missing(lin):
    L = LIN[lin]; d = L["r2"]; miss = _missing(d, REG_R2)
    if "eval_hist24.json" not in miss and not isinstance((jl(d, "eval_hist24.json") or {}).get(E4_KEYS["items"]), dict):
        miss["eval_hist24.json"] = "%s/eval_hist24.json (E4 items)" % d
    if "eval_selfdepth.json" not in miss:
        m = _sd_missing(jl(d, "eval_selfdepth.json"), 2)
        if m:
            miss["eval_selfdepth.json"] = "%s/eval_selfdepth.json (fields %s)" % (d, m)
    if not os.path.exists(os.path.join(R, L["pool_r2"], "pool_meta.json")):
        miss["pool_meta.json"] = "%s/pool_meta.json" % L["pool_r2"]
    g2 = gate2(lin)
    if g2 is None:
        miss["eval_gate.json"] = "%s/eval_gate.json" % L["gate2"]
    elif any(g2[w]["status"] == "UNDECIDED" for w in WIN):
        miss["eval_gate.json"] = "%s/eval_gate.json (GATE_2 / GATE_1p / fallback undecided: %s)" % (L["gate2"], {w: g2[w]["status"] for w in WIN})
    a0, a1, bad = bar_inputs(lin)
    if a0 is None or a1 is None:
        miss["bars"] = "bar inputs %s" % bad
    cm = card_mismatch([(d, f) for f in CARD_FILES] + [(L["base"], REF_EVAL_VB), (L["r1"], REF_EVAL_VB)])
    if cm:
        miss["card"] = "card mismatch: %s" % cm
    return miss


def r2_cell(lin):
    """r2_cell(lineage) (v1 §4.3-§4.5, §5.1-§5.2)."""
    L = LIN[lin]; d = L["r2"]; o = {"cell": d, "lineage": lin}
    if d in NOT_ADMITTED:
        o.update(status="UNREAD", label="UNREAD (not admitted; never a pass)"); return o
    miss = r2_missing(lin)
    if miss and lin not in FINAL:
        o.update(status="PENDING", missing=sorted(miss.values()), label=_pend(miss)); return o
    o["missing_at_final"] = sorted(miss.values()); cm = sorted(miss[k] for k in miss if k in CENSOR_FILES)
    if "card" in miss:
        return _void_card(o, miss, dict(SD2_SELF=False, SD2_C2=False, SD2_C2_MEM=False, LIT_C2_2_MET=False, MECHANISM="UNLABELLED (censored)"))
    if cm:
        o.update(status="FAIL", censored=True, SD2_SELF=False, SD2_C2=False, SD2_C2_MEM=False, LIT_C2_2_MET=False, MECHANISM="UNLABELLED (censored)",
                 label="FAILURE (registered file missing at --final: %s; rule b)" % cm); return o
    g2 = gate2(lin); o["GATE_2"] = {w: {"status": g2[w]["status"], "pi_pool": (g2[w]["used"] or g2[w]["primary"])["pi_pool"]} for w in WIN}
    both = [w for w in WIN if g2[w]["status"] == "BOTH-FAIL"]
    if both:
        o.update(status="FAIL", censored=True, SD2_SELF=False, SD2_C2=False, SD2_C2_MEM=False, LIT_C2_2_MET=False, MECHANISM="UNLABELLED (censored)",
                 label="VOID (no round 2 in %s: both labellers failed %s; rule b)" % (lin, [GATE_FLAG[w] for w in both])); return o
    o["ARGS_DIFF"] = ad = args_diff(lin, "r2"); info = _recs(d); ev = jl(d, "eval_vb.json"); a, K = ev["acc_n2"], ev["acc_k"]
    A = lambda n: a[str(n)]; Ak = lambda k, n: K[str(k)][str(n)]  # noqa: E731
    dead = info["last"] < FT_FULL                                           # registered: an incomplete run (skipped non-finite steps are not dead)
    diverged = any(A(n) < 0.99 - EPS or Ak(24, n) < 0.97 - EPS for n in range(1, 9))
    sd = jl(d, "eval_selfdepth.json"); la = label_audit2(sd["label_audit"], g2); noisy = not la["ok"]
    ct = contamination2(sd, info, L["pool_r2"], POOL_FILE["pool_r2"]); args_bad = not ad["ok"]
    censored = dead or diverged or noisy or ct["CONTAMINATED"] or args_bad
    pm = jl(L["pool_r2"], "pool_meta.json") or {}; T_train = (pm.get("T_max_by_n") or {}).get("16")
    dr = deep_read(d, T_train) if "eval_deep.json" not in miss else {"void": "eval_deep.json missing at --final"}
    a0, a1, bad = bar_inputs(lin); b = {"void": bad} if (bad or a0 is None or a1 is None) else r2_bars(a0, a1)
    c = S211.cell_read("R2", d)                                            # VB-C1 parts (its 48k dead / NO-ENTRY censors are ignored, as §215)
    vb_c1 = bool(c.get("DEEP") and c.get("PROBE_GAIN") and (c.get("onset") or {}).get("SERIAL_ONSET") and c.get("SERIAL_STEP") and c.get("NO_CLOCK"))
    sp = jl(d, "eval_steppos.json") if "eval_steppos.json" not in miss else None; lf = S212.lfk(d) if "eval_lfk.json" not in miss else None
    lfree = bool(sp and S213.no_knee(sp)["NO_KNEE"] and lf and lf["MEETS"])
    hk = keeps2(d, True, missing=jl(d, "eval_hist24.json") is None)       # E4 store absent at --final -> items_ok False -> gate fails
    absorb2 = min(A(n) for n in range(13, 17)) >= 0.95 - EPS; retain = min(A(n) for n in range(9, 13)) >= 0.99 - EPS
    k24 = all(Ak(24, n) >= 0.99 - EPS for n in range(1, 9)); ok_deep = not dr.get("void")
    mech = mechanism2(absorb2, dr, b, censored)
    sd2_self = bool(absorb2 and retain and lfree and vb_c1 and k24) and not censored
    dbs2 = bool(ok_deep and dr["M2"] >= 0.80 - EPS) and not censored
    mg2 = bool(ok_deep and not b.get("void") and dr["M2"] >= b["MARGIN_GROWS_2"] - EPS) and not censored
    lit = bool(ok_deep and dr["LIT_C2_2"] >= 0.10 - EPS) and not censored
    sd2_c2 = sd2_self and dbs2; keeps = hk["KEEPS_SELF_STAB_2"] and not censored; sd2_c2_mem = sd2_c2 and keeps
    memo = sd.get("memo") or {}; gap = {n: (memo[n]["acc_pool"] - memo[n]["acc_fresh"]) for n in memo if num(memo[n].get("acc_pool")) and num(memo[n].get("acc_fresh"))}
    bs = bs_check(ev, d, sd) if "eval_deep.json" not in miss else None
    o.update(status="FAIL" if censored else "SCORED", last_step=info["last"], nonfinite_steps=info["nonfinite_steps"], c_zero=info["c_zero"], dead=dead,
             FT_DIVERGED=diverged, LABEL_NOISY_2=noisy, LABEL_AUDIT_2=la, CONTAMINATION=ct, CONTAMINATED=ct["CONTAMINATED"], ARGS_BAD=args_bad, censored=censored,
             ABSORB_2=absorb2 and not censored, RETAIN=retain and not censored, ERODED=not retain, DEPTH_GENERAL_2=mean(A(n) for n in range(13, 17)),
             THINK_NECESSARY_DEEP_2=all(max(Ak(0, n), Ak(1, n)) <= 0.15 + EPS for n in range(13, 17)),
             K24_WIDE_vb=all(Ak(24, n) >= A(n) - 0.01 - EPS for n in range(9, 17)), K24_STABLE=k24 and not censored, LENGTH_FREE=lfree and not censored,
             VB_C1=vb_c1 and not censored, DEEP=dr, BARS=b, DBS_2=dbs2, MARGIN_GROWS_2=mg2, LIT_C2_2_MET=lit, MECHANISM=mech,
             FULL_SEPARABILITY_FLAG=bool(mech == "FULL" and b.get("FULL_not_separable_flag")), KEEPS=hk, KEEPS_SELF_STAB_2=keeps,
             MEM_TRADED_2=None if censored else not keeps, MEM_HISTORY=mem_history(lin, hk["STAB_self"]), MEMO_GAP_2={"gap": gap, "ok": all(abs(v) <= 0.02 + EPS for v in gap.values()),
             "n": sorted(gap, key=int)}, BS_CHECK=bs, SD2_SELF=sd2_self, SD2_C2=sd2_c2, SD2_C2_MEM=sd2_c2_mem, T_train=T_train,
             LABELLER_KEPT_2=S213.chainlabel_fresh(d) if jl(d, "eval_chainlabel_fresh.json") else None, VB_C1_parts={k: c.get(k) for k in ("DEEP", "PROBE_GAIN",
             "SERIAL_STEP", "NO_CLOCK")})
    if lin == "s1":                                                        # G-S3 identity (descriptive here; the gate is G-S3)
        x, y = jl(d, "draws_log.json") or [], jl(REF_FT, "draws_log.json") or []
        o["G_S3_plain_hash_1_200"] = bool(len(x) >= 200 and len(y) >= 200 and all(p.get("plain_hash") == q.get("plain_hash") for p, q in zip(x[:200], y[:200])))
    why = [w for w, f in (("dead", dead), ("FT-DIVERGED", diverged), ("LABEL-NOISY-2", noisy), ("CONTAMINATED", ct["CONTAMINATED"]), ("ARGS-DIFF", args_bad)) if f]
    fmt = lambda v: ("%.3f" % v) if num(v) else str(v)  # noqa: E731
    head = ("FAILURE (%s; rule b)" % ", ".join(why) if censored else "SD2-C2-MEM" if sd2_c2_mem else "SD2-C2 (MEM-TRADED-2)" if sd2_c2 else
            "SD2-SELF (in support; not C2)" if sd2_self else "not SD2-SELF%s" % (" (ERODED)" if not retain else ""))
    o["label"] = head + " | %s%s | M2 %s | STOP %s | n50 %s | WIDTH-2 %s | LIT-C2-2 %s (%s) | MEM %s | BS-CHECK %s" % (
        mech, " [FULL not separable from H_prop]" if o["FULL_SEPARABILITY_FLAG"] else "", fmt(dr.get("M2")), dr.get("STOP"), fmt(dr.get("n50")),
        fmt(dr.get("WIDTH_2")), "met" if lit else "not met", fmt(dr.get("LIT_C2_2")),
        "n/a (censored)" if censored else "KEEPS-SELF-STAB-2" if keeps else "MEM-TRADED-2 (POST HOC: %s)" % hk["MEM_SUBLABEL_POST_HOC"],
        "FLAG %s" % fmt(bs["max_abs_delta"]) if (bs and bs["FLAG"]) else ("ok" if bs else "n/a"))
    o["WORDING"] = wording(o, g2, la)
    return o


def wording(o, g2, la):
    """v1 §5.1 pre-stated wording (only these sentences; bracketed values from eval_gate.json / eval_selfdepth.json)."""
    bl = [w for w in WIN if g2[w]["status"] == "BASE-LABELLED"]; pi = (g2["W2"]["used"] or {}).get("pi_pool")
    nm = {"W2": ("13–16", "GATE-2"), "W1": ("9–12", "GATE-1′")}
    fb = "".join(" Labels for n %s from the base model (BASE-LABELLED fallback after the round-1 model failed %s)." % nm[w] for w in bl)
    selft = "" if len(bl) == 2 else (" The labeller%s was itself self-trained;" % ((" for n %s" % nm[[w for w in WIN if w not in bl][0]][0]) if bl else ""))
    s = {"SD2_SELF": ("latent depth extended from 12 to 16 by distilling the round-1 model's own re-rooted single-step chain into the loop.%s%s labels are "
                      "gold-equivalent at π = %s (rule ≥ 0.995; LABEL-AUDIT-2 π = %s); no gold labels beyond n 8; in support; not C2, not C4."
                      % (fb, selft, "%.5f" % pi if num(pi) else pi, "%.4f" % la["pi"] if num(la["pi"]) else la["pi"])) if o["SD2_SELF"] else None}
    if s["SD2_SELF"] and len(bl) == 2:
        s["SD2_SELF"] = s["SD2_SELF"].replace(". labels are", ". Labels are")
    s["SD2_C2"] = ("depth extrapolation past the self-labelled range (n 17–20, untouched §170-extension items; C2 proxy; one lineage: an instance)"
                   if o["SD2_C2"] else None)
    s["LIT_C2_2"] = ("met: " if o["LIT_C2_2_MET"] else "not met: ") + "accuracy rises with latent passes past the trained budget at n 22–24 (literal C2)"
    s["MECHANISM"] = ("consistent with %s" % o["MECHANISM"]) if not o["MECHANISM"].startswith("UNLABELLED") else o["MECHANISM"]
    s["SD2_C2_MEM"] = "SD2-C2-MEM" if o["SD2_C2_MEM"] else ("MEM-TRADED-2 (MEM history %s)" % o["MEM_HISTORY"] if o["SD2_C2"] else None)
    return s


def r1_missing(lin):
    L = LIN[lin]; d = L["r1"]; miss = _missing(d, REG_R1)
    if "eval_selfdepth.json" not in miss:
        m = _sd_missing(jl(d, "eval_selfdepth.json"), 1)
        if m:
            miss["eval_selfdepth.json"] = "%s/eval_selfdepth.json (fields %s)" % (d, m)
    if not os.path.exists(os.path.join(R, L["pool_r1"], "pool_meta.json")):
        miss["pool_meta.json"] = "%s/pool_meta.json" % L["pool_r1"]
    g = jl(L["gate1"], "eval_gate.json")
    if g is None or not isinstance(g.get("GATE_1"), bool):
        miss["eval_gate.json"] = "%s/eval_gate.json%s" % (L["gate1"], "" if g is None else " (GATE_1 undecided)")
    e0 = jl(L["base"], REF_EVAL_VB)
    if e0 is None or "acc_n2" not in e0:
        miss["bars"] = "bar input %s/%s" % (L["base"], REF_EVAL_VB)
    cm = card_mismatch([(d, f) for f in CARD_FILES if f != "eval_deep.json"] + [(L["base"], REF_EVAL_VB)])   # R1's eval_deep is a baseline (V3)
    if cm:
        miss["card"] = "card mismatch: %s" % cm
    return miss


def r1_cell(lin):
    """R1-s2 (v1 §4.3 last paragraph): SELFDEPTH_v1 §4.2-§4.3 / §5.2 readings (score_s215.cell's, generalised) with the lineage's paths, the R1 bar
    formulas on the lineage's base eval_vb, LABEL-AUDIT vs GATE-1-<lineage>'s CI, contamination incl. test n_max 24; round-2 condition (v1 §9)."""
    L = LIN[lin]; d = L["r1"]; o = {"cell": d, "lineage": lin}
    if d in NOT_ADMITTED:
        o.update(status="UNREAD", label="UNREAD (not admitted; never a pass)"); return o
    miss = r1_missing(lin)
    if miss and lin not in FINAL:
        o.update(status="PENDING", missing=sorted(miss.values()), label=_pend(miss)); return o
    if "card" in miss:
        return _void_card(o, miss, dict(ROUND2_CONDITION=False))
    cm = sorted(miss[k] for k in miss if k in CENSOR_FILES)
    if cm:
        o.update(status="FAIL", censored=True, ROUND2_CONDITION=False, label="FAILURE (registered file missing at --final: %s; rule b)" % cm); return o
    o["ARGS_DIFF"] = ad = args_diff(lin, "r1"); g1 = gate1(lin); o["GATE_1"] = g1; info = _recs(d); ev = jl(d, "eval_vb.json"); a, K = ev["acc_n2"], ev["acc_k"]
    A = lambda n: a[str(n)]; Ak = lambda k, n: K[str(k)][str(n)]  # noqa: E731
    dead = info["last"] < FT_FULL
    diverged = any(A(n) < 0.99 - EPS or Ak(24, n) < 0.97 - EPS for n in range(1, 9))
    sd = jl(d, "eval_selfdepth.json"); la = label_audit1(sd["label_audit"], g1); noisy = not la["ok"]
    ct = contamination2(sd, info, L["pool_r1"], POOL_FILE["pool_r1"]); args_bad = not ad["ok"]; gate_bad = not g1["GATE_1"]
    censored = dead or diverged or noisy or ct["CONTAMINATED"] or args_bad or gate_bad
    a0, _, _ = bar_inputs(lin); b = r1_bars(a0) if (a0 and "bars" not in miss) else None    # bar input missing at --final -> bar readings fail
    c = S211.cell_read("R1", d)
    vb_c1 = bool(c.get("DEEP") and c.get("PROBE_GAIN") and (c.get("onset") or {}).get("SERIAL_ONSET") and c.get("SERIAL_STEP") and c.get("NO_CLOCK"))
    sp = jl(d, "eval_steppos.json") if "eval_steppos.json" not in miss else None; lf = S212.lfk(d) if "eval_lfk.json" not in miss else None
    lfree = bool(sp and S213.no_knee(sp)["NO_KNEE"] and lf and lf["MEETS"]); k24 = all(Ak(24, n) >= 0.99 - EPS for n in range(1, 9))
    hk = keeps2(d, False, missing="eval_hist24.json" in miss)
    absorb = min(A(n) for n in range(9, 13)) >= 0.95 - EPS; M = mean(A(n) for n in range(13, 17)); G, stop = S215.stop_loc(ev)
    kb = mean(A(n) - Ak(14, n) for n in (15, 16)); gain10 = (S213.acc_gain_after10(d) or {}).get("mean_gain")
    mech = ("UNLABELLED (censored)" if censored else "UNLABELLED (not ABSORB)" if not absorb else "UNLABELLED (bars void)" if b is None else
            "FULL" if M >= b["FULL_M"] - EPS else "GROW" if M >= b["MARGIN_GROWS"] - EPS else
            ("SHIFT" if stop >= 13 else "MIXED") if M >= b["SHRINK_below"] - EPS else "SHRINK")
    sd_self = bool(absorb and lfree and vb_c1 and k24) and not censored; sd_c2 = bool(sd_self and M >= 0.80 - EPS)
    cond = bool(absorb and not diverged and not noisy and not ct["CONTAMINATED"] and not dead and not args_bad and not gate_bad)
    memo = sd.get("memo") or {}; gap = {n: (memo[n]["acc_pool"] - memo[n]["acc_fresh"]) for n in memo if num(memo[n].get("acc_pool")) and num(memo[n].get("acc_fresh"))}
    o.update(status="FAIL" if censored else "SCORED", last_step=info["last"], dead=dead, nonfinite_steps=info["nonfinite_steps"], FT_DIVERGED=diverged, LABEL_NOISY=noisy, LABEL_AUDIT=la, CONTAMINATION=ct,
             CONTAMINATED=ct["CONTAMINATED"], ARGS_BAD=args_bad, GATE_1_FAILED=gate_bad, censored=censored, BARS=b, ABSORB=absorb and not censored, M=M,
             DEPTH_GENERAL=mean(A(n) for n in range(9, 13)) >= 0.80 - EPS, DBS=M >= 0.80 - EPS and not censored,
             MARGIN_GROWS=bool(b and M >= b["MARGIN_GROWS"] - EPS) and not censored, STOP_G=G, STOP=stop, STOP_TO_SUPPORT=stop >= 13, K_BEYOND=kb,
             K_BEYOND_MET=kb >= 0.25 - EPS and not censored, RISE_PAST_K=mean(A(n) - Ak(16, n) for n in (15, 16)), gain_after10=gain10,
             STOP_MOVED=None if gain10 is None else bool(gain10 >= 0.10 - EPS), THINK_GAIN_BEYOND=mean(A(n) - Ak(12, n) for n in range(13, 17)),
             C2_TAIL=all(Ak(24, n) >= A(n) - 0.05 - EPS for n in range(13, 17)), THINK_NECESSARY_DEEP=all(max(Ak(0, n), Ak(1, n)) <= 0.15 + EPS for n in range(9, 13)),
             MECHANISM=mech, GROW_REPLICATED=mech == "GROW", LENGTH_FREE=lfree, VB_C1=vb_c1, K24_STABLE=k24, KEEPS=hk, KEEPS_SELF_STAB=hk["KEEPS_SELF_STAB_2"],
             SD_SELF=sd_self, SD_C2=sd_c2, SD_C2_MEM=bool(sd_c2 and hk["KEEPS_SELF_STAB_2"]), MEMO_GAP={"gap": gap, "ok": all(abs(v) <= 0.02 + EPS for v in gap.values())},
             ROUND2_CONDITION=cond)
    why = [w for w, f in (("dead", dead), ("FT-DIVERGED", diverged), ("LABEL-NOISY", noisy), ("CONTAMINATED", ct["CONTAMINATED"]), ("ARGS-DIFF", args_bad),
                          ("GATE-1 failed", gate_bad)) if f]
    o["label"] = ("FAILURE (%s; rule b)" % ", ".join(why) if censored else "SD-C2-MEM" if o["SD_C2_MEM"] else "SD-C2 (MEM-TRADED)" if sd_c2 else
                  "SD-SELF (in support; not C2)" if sd_self else "not SD-SELF") + " | %s | M %.3f | STOP %s | K'-BEYOND %.3f | round-2 condition %s" % (
        mech, M, "16+" if stop == 16 else stop, kb, cond)
    return o


def base_cell(lin):
    """s2 base (v1 §2.4): ARGS-DIFF vs s213_d3t_s1 (seed 2); dead = incomplete 48k log -> x2 FAIL; COMPETENT-BASE on a completed run only:
    a(n) >= 0.99 and a_24(n) >= 0.97 at every n 1..8 (eval_vb acc part); entry step (score_s211.entry_on) vs s1's, > 5k = CONFOUNDED (descriptive)."""
    d = LIN[lin]["base"]; o = {"cell": d, "lineage": lin}
    if d in NOT_ADMITTED:
        o.update(status="UNREAD", label="UNREAD (not admitted; never a pass)"); return o
    miss = _missing(d, ("log.jsonl", "args.json", "eval_vb.json"), ev_parts=("acc_n2", "acc_k")); cm = card_mismatch([(d, "eval_vb.json")])
    if cm:
        miss["card"] = "card mismatch: %s" % cm
    if miss and lin not in FINAL:
        o.update(status="PENDING", missing=sorted(miss.values()), label=_pend(miss)); return o
    if "card" in miss:
        return _void_card(o, miss, {})
    if miss:
        o.update(status="FAIL", label="FAILURE (registered file missing at --final: %s; rule b)" % sorted(miss.values())); return o
    info = _recs(d); ad = args_diff(lin, "base"); ev = jl(d, "eval_vb.json"); a, K = ev["acc_n2"], ev["acc_k"]
    dead = info["last"] < BASE_FULL
    comp = all(a[str(n)] >= 0.99 - EPS and K["24"][str(n)] >= 0.97 - EPS for n in range(1, 9))
    e2, e1 = S211.entry_on(info["recs"], "answer_ce", 0.5), S211.entry_on(S211.log_recs(LIN["s1"]["base"]), "answer_ce", 0.5) or 11000
    o.update(ARGS_DIFF=ad, last_step=info["last"], nonfinite_steps=info["nonfinite_steps"], dead=dead, COMPETENT=None if dead else comp,
             entry_answer_ce_05=e2, entry_s1=e1, CONFOUNDED=e2 is None or abs(e2 - e1) > 5000, STAB_self=stab_self(d),
             VB_C1=S211.cell_read("BASE", d).get("VB_C1") if all(k in ev for k in VB_PARTS) else None)
    o["status"] = "FAIL" if (dead or not ad["ok"]) else "PASS" if comp else "NOT_COMPETENT"
    o["label"] = ("FAILURE (s2 base dead: last step %d < 48,000; rule b)" % info["last"] if dead else
                  "FAILURE (s2 base ARGS-DIFF; rule b)" if not ad["ok"] else "COMPETENT-BASE" if comp else "s2 base not competent (C6 data point)")
    return o


def gate1_stage(lin):
    g = jl(LIN[lin]["gate1"], "eval_gate.json")
    if g is None or not isinstance(g.get("GATE_1"), bool):
        return ("FAIL", "GATE-1-%s missing / undecided at --final" % lin) if lin in FINAL else ("PENDING", "GATE-1-%s eval_gate.json missing / undecided" % lin)
    e = gate1(lin)
    return ("PASS", "GATE-1-%s" % lin) if e["GATE_1"] else ("FAIL", "GATE-1-%s fails%s" % (lin, "" if e["consistent"] else " (file flag != recomputed)"))


def gate2_stage(lin):
    g = gate2(lin)
    if g is None:
        return ("FAIL", "GATE-2-%s missing at --final" % lin) if lin in FINAL else ("PENDING", "GATE-2-%s eval_gate.json missing" % lin)
    st = {w: g[w]["status"] for w in WIN}
    if any(v == "BOTH-FAIL" for v in st.values()):
        return "FAIL", "GATE-2-%s fails with both labellers %s" % (lin, st)
    if any(v == "UNDECIDED" for v in st.values()):
        return "PENDING", "GATE-2-%s undecided %s" % (lin, st)
    return "PASS", "GATE-2-%s %s" % (lin, st)


def cell_stage(c, kind):
    s = c["status"]
    if s in ("UNREAD", "PENDING", "FAIL"):
        return s, "%s: %s" % (c["cell"], c["label"])
    if kind == "r1":
        return ("PASS", "R1 round-2 condition met") if c["ROUND2_CONDITION"] else ("FAIL", "lineage stopped at round 1 (%s: %s)" % (c["cell"], c["label"]))
    return ("PASS", "SD2-C2") if c["SD2_C2"] else ("FAIL", "%s not SD2-C2 (%s)" % (c["cell"], c["label"]))


def lineage(lin):
    o = {"r2": r2_cell(lin), "gate2": gate2(lin), "bars_r2": None}
    a0, a1, bad = bar_inputs(lin); o["bars_r2"] = r2_bars(a0, a1) if (a0 and a1 and not bad) else {"void": bad}
    stages = []
    if lin == "s2":
        o["base"], o["gate1"], o["r1"] = base_cell(lin), gate1(lin), r1_cell(lin)
        o["bars_r1"] = r1_bars(a0) if (a0 and not [x for x in bad if LIN[lin]["base"] in x]) else None
        b = o["base"]; stages += [("base", ("NOT_COMPETENT", b["label"]) if b["status"] == "NOT_COMPETENT" else ("PASS", b["label"]) if b["status"] == "PASS"
                                            else (b["status"], "%s: %s" % (b["cell"], b["label"])))]
        stages += [("gate1", gate1_stage(lin)), ("r1", cell_stage(o["r1"], "r1"))]
    stages += [("gate2", gate2_stage(lin)), ("r2", cell_stage(o["r2"], "r2"))]
    row = {"status": "PASS", "stage": None, "why": "SD2-C2 in %s" % LIN[lin]["r2"]}
    for st, (s, why) in stages:
        if s != "PASS":
            row = {"status": s, "stage": st, "why": why}; break
    c = o["r2"]
    row.update(SD2_SELF=c.get("SD2_SELF"), SD2_C2=c.get("SD2_C2"), LIT_C2_2=c.get("LIT_C2_2_MET"), MECHANISM=c.get("MECHANISM"),
               WIDTH_2=(c.get("DEEP") or {}).get("WIDTH_2"), n50=(c.get("DEEP") or {}).get("n50"), MEM_HISTORY=c.get("MEM_HISTORY") or mem_history(lin, None))
    o["row"] = row; o["panels"] = panels(lin)
    return o


def x2(rows):
    """v1 §4.5: PASS (2/2) / FAIL (k of 2, stage named) / NOT COMPETENT (completed s2 base below the floor; the only exclusion) / UNREAD (not admitted, pending)."""
    s1, s2 = rows["s1"], rows["s2"]; k = sum(r["status"] == "PASS" for r in (s1, s2))
    if s2["status"] == "NOT_COMPETENT":
        return {"outcome": "NOT COMPETENT", "label": "s2 base not competent (C6 data point); SD2-C2 instance in %d lineage (s1 row: %s); x2 not evaluable"
                % (int(s1["status"] == "PASS"), s1["status"])}
    if any(r["status"] == "UNREAD" for r in (s1, s2)):
        return {"outcome": "UNREAD", "label": "UNREAD (a cell not admitted; never a pass)"}
    fails = {lin: r["stage"] + ": " + r["why"] for lin, r in rows.items() if r["status"] == "FAIL"}
    resolved = s2["status"] == "FAIL" or s2["stage"] != "base"            # NOT COMPETENT is no longer possible
    if fails and resolved:
        pend = [lin for lin, r in rows.items() if r["status"] == "PENDING"]
        return {"outcome": "FAIL", "label": "FAIL: SD2-C2 in %d of 2 lineages%s" % (k, " (%s pending)" % pend if pend else ""), "failing": fails}
    if any(r["status"] == "PENDING" for r in (s1, s2)):
        return {"outcome": "UNREAD", "label": "UNREAD (pending; never a pass)"}
    return {"outcome": "PASS", "label": "SD2-C2 in both competent lineages (2/2; base seeds 1 and 2); not a rate"}


FLOOR = 0.11


def panels(lin):
    """Descriptive reference panels (v1 §3.4 per-n forms, v1_panels.py's arithmetic) on the lineage's own base / R1 eval_vb; the assumed R1 extension past n 16
    is replaced by R1's eval_deep (n 17..24) when present, read only after the lineage's R2 cell has launched (D2 / V3). No bar uses them."""
    a0, a1, bad = bar_inputs(lin)
    if a0 is None or a1 is None or bad:
        return None
    A1, A0 = dict(a1), dict(a0); src = "assumed extension (v1 §3.4)"
    rd = None
    if launched(lin):
        rd = deep_read(LIN[lin]["r1"])
    if rd and not rd.get("void"):
        A1.update({n: rd["a"][n] for n in range(17, 25)}); src = "%s/eval_deep.json" % LIN[lin]["r1"]
        for n in range(25, 41):
            A1[n] = FLOOR
    else:
        try:
            ex = {n: a1[n] - FLOOR for n in (14, 15, 16)}; q1, q2 = ex[15] / ex[14], ex[16] / ex[15]; q3 = q2 * q2 / q1
            A1[17] = FLOOR + ex[16] * q3; A1[18] = FLOOR + (A1[17] - FLOOR) * q3 * (q2 / q1)
        except ZeroDivisionError:
            return {"note": "extension undefined (a1(14..15) at the floor)"}
        for n in range(19, 41):
            A1[n] = FLOOR
    for n in range(17, 41):
        A0[n] = FLOOR
    for n in range(-12, 1):
        A0[n] = A1[n] = 1.0

    def c_(a, L, lo, hi):
        for n in range(lo, hi):
            if n + 1 in a and a[n] >= L - EPS and L > a[n + 1]:
                return n + (a[n] - L) / (a[n] - a[n + 1])
        return None
    m1, m0 = c_(a1, 0.5, 12, 16), c_(a0, 0.5, 8, 16); delta = (m1 - 12 - (m0 - 8)) if (m1 is not None and m0 is not None) else None
    H = {"H_shift": lambda n: A1[n - 4], "H_prop": lambda n: ip(A1, 12 + 0.75 * (n - 16)), "SHRINK": lambda n: A0[n - 8]}
    if delta is not None:
        H["H_add_constant_growth"] = lambda n: ip(A1, n - 4 - delta)
    out = {"source_past_16": src, "delta_m50": delta}
    for h, f in H.items():
        a2 = {n: f(n) for n in range(13, 25)}; n50, n90 = cross(a2, 0.5), cross(a2, 0.9)
        out[h] = {"a": a2, "M2": mean(a2[n] for n in range(17, 21)), "M_far": mean(a2[n] for n in range(21, 25)), "n50": n50, "n90": n90,
                  "WIDTH_2": (n50 - n90) if (num(n50) and num(n90)) else None}
    return out


def launched(lin):
    d = LIN[lin]["r2"]
    return os.path.exists(os.path.join(R, d, "log.jsonl")) or os.path.exists(os.path.join(R, d, "args.json"))


def baselines():
    """G-S18 / V3 (descriptive, never gate): eval_deep on FT-SELF (required), D3-T s1 / FT-CTRL (optional) after s216_r2_s1 has launched; on s216_r1_s2
    (required) after s216_r2_s2 has launched. Not read before (D2)."""
    o = {}
    for lin, cells in (("s1", (("FT-SELF", "s215_ftself_s1", True), ("D3-T s1", "s213_d3t_s1", False), ("FT-CTRL", "s215_ftctrl_s1", False))),
                       ("s2", (("R1-s2", "s216_r1_s2", True),))):
        if not launched(lin):
            o[lin] = "not read (%s not launched; D2 / V3)" % LIN[lin]["r2"]; continue
        for tag, d, req in cells:
            if jl(d, "eval_deep.json") is None:
                o[tag] = "missing (%s; descriptive, never gates)" % ("required" if req else "optional"); continue
            dr = deep_read(d); o[tag] = {k: dr.get(k) for k in ("void", "M2", "M_far", "LIT_C2_2", "STOP", "n50", "n90", "WIDTH_2", "KB2")}
            o[tag]["BS_CHECK"] = bs_check(jl(d, "eval_vb.json"), d, jl(d, "eval_selfdepth.json"))
    return o


def score(final=(), not_admitted=()):
    FINAL.clear(); FINAL.update(final); NOT_ADMITTED.clear(); NOT_ADMITTED.update(not_admitted)
    out = {"registered": "docs/research_selfdepth2_2026-10-18/SD2_v1.md @ e9ade2ec", "R": R, "CARD_CLASS": CARD_CLASS, "REF_EVAL_VB": REF_EVAL_VB,
           "final": sorted(FINAL), "not_admitted": sorted(NOT_ADMITTED), "layouts_assumed": LAYOUTS}
    for lin in ("s1", "s2"):
        out[lin] = lineage(lin)
    rows = {lin: out[lin]["row"] for lin in ("s1", "s2")}
    out["X2"] = x2(rows)
    out["SD2_SELF_X2"] = None if any(r.get("SD2_SELF") is None for r in rows.values()) else bool(all(r["SD2_SELF"] for r in rows.values()))
    out["BASELINES"] = baselines()
    return out


def reading(out):
    s1, s2 = out["s1"], out["s2"]
    s2c = " -> ".join("%s %s" % (k, s2[k]["status"]) for k in ("base", "r1", "r2"))
    return "READING: s1 R2 %s || s2 %s (row %s%s) || x2 %s" % (s1["r2"]["label"], s2c, s2["row"]["status"],
                                                                  (" at " + s2["row"]["stage"]) if s2["row"]["stage"] else "", out["X2"]["label"])


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="append", default=[], choices=["s1", "s2"])
    ap.add_argument("--not_admitted", default=""); a = ap.parse_args(argv)
    out = score(a.final, [x for x in a.not_admitted.split(",") if x])
    print(reading(out).encode("ascii", "backslashreplace").decode("ascii"))
    json.dump(out, open(OUT, "w", newline=chr(10)), indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
