"""THEORY §219 scoring (TEXT-SELF: a matched text-CoT self-training baseline for the latent s1 self-depth lineage; registered text
docs/research_textself_2026-10-18/TS_v1.md at 146a25d0 = v1 + Addendum 1 (Theory's §9 decisions) + Addendum 2 (Audit's conditions); THEORY.md §219;
Theory is the only scorer). One lineage (s1, LIN, v1 §4.6): base s219_base_s1 (fp32 re-eval of s170_cot_s1/model.pt) -> T1 s219_t1_s1 -> T2 s219_t2_s1.
Run dirs under R (boxes/runs_fsq); set_root() redirects R / ROOT (fixtures); OUT is module-level.
Notation a_T(n) = eval_text acc_by_n[n] (free greedy generation over test_set(1000, 24), answer == y over all 1,000 items; CTX-OVERFLOW and
  non-termination are failures). n50 / n90: first downward crossing of 0.5 / 0.9 over n 1..24, linear interpolation; "> 24" never; "< 1" below at n 1.
Latent comparators (Addendum 2 item 4): CONSTANTS LATENT_N50 = D3-T s1 10.306 / FT-SELF 15.460 / R2-s1 20.539 (sources LATENT_SRC; the same n50 rule on
  eval_vb acc_n2 (n <= 16) + eval_deep acc_n2 (n >= 17)). The scorer never re-reads the latent files for a decision; the embedded curves LATENT_CURVES
  only reproduce the constants and the latent bar instance 0.63 / 0.40 / 0.785 / 0.60 (asserted at import). Latent files are read (read-only) only by
  descriptive rows (DEPTH-PAIRED, LABELLER-REACH, LATENT-FIXED-K', TRAIN-TOKENS latent side).
Pending (215-7; v1 §4.6): a cell reads "pending (missing [...])" while any registered file / part is missing (log, args, draws_log, eval_text, eval_textself's
  CPU parts (label_audit, contamination), the round's pool_meta and eval_gate (flag decided), s219_base_s1/eval_text.json; T2 also T1's eval_text = the
  round-2 bar input) and reads nothing else first; pending never passes. --final s1: a still-missing registered file is a failure (rule b); an undecided
  gate fails. Descriptive files / keys (MEMO-GAP-T, LABELLER-KEPT-T, CARD-GATE, LABELLER-FULL, KEPT-VS-DRAWN, EQUIV) never gate. NOT_ADMITTED cells read
  UNREAD (never a pass) and the comparison reads TEXT-CENSORED (not admitted) (v1 §8).
eval_text (text_read, v1 §2.7 / §4.6 / Addendum 2 item 1): items_sha256 recomputed (n 1..24); meta card in the allowed classes (CARD_CLASS; the RTX 2070 only
  after CARD-GATE G-T21 passes; one card per file), dtype fp32, tf32 false, cot_eval_bs 32, test_seed 17100, test_n_max 24, model_sha256 (base BASE_SHA; an
  FT cell's == sha256(model.pt) when model.pt is present); acc_by_n == mean(answer == y) per n (1e-12); every item's P == len(syms) + 2, C_struct ==
  len(cot_ids), overflow == (P + C_struct > 1025) and an overflow item's answer -1. Any mismatch -> VOID: every reading of that model fails (§5.2).
Censors (v1 §5.2, rule b; each fails every reading of the cell and every comparison row at and after its round): dead (log last step < 12,000);
  FT-DIVERGED-T (a_T(n) < 0.99 at any n 1..8); LABEL-NOISY-T (LABEL-AUDIT-T pi or pi_E < 0.99 pooled or per window, or a window's Wilson CI (pi or pi_E)
  not overlapping its gate's); CONTAMINATED (E2 pool_excl_hits / pool_test_hits / gold_key_hits > 0, self_reuse > 0, pool exhaustion (cursor > pool size),
  or the scorer's own re-check of the consumed pool records: exact key set {syms, n, chain_label, values}, no test_set(1000, 30) syms, no duplicate);
  ARGS-DIFF (args_diff below); SEAL-BREACH (T3: a trainer-written eval.json / eval_<k>k.json or a *.void_unsealed file in the cell dir, or a time-field
  order violation; while the cell is pending, a later-stage n >= 13 output of the model before its eval_text is reported as SEAL_WATCH); eval_text VOID;
  the round's gate / pool not on one allowed card; the round's gate failed (the cell should not exist).
ARGS-DIFF (T1, V2, Addendum 1 item 3; each FT cell vs s170_cot_s1/args.json): META_T = score_s213.META + {excl_survive_by_n, hostprobe_sps}; TEXT_NEUTRAL_NEW
  (29 keys at nested170's parser defaults, cot_eval_bs 64; a missing or different value -> ARGS_BAD); FT keys with required values (FT_REQ; stop_at 0 and
  mem_check_self false for the registered run only: smoke=True for G-T5 smokes); args_ref by the 216-7 content rule (score_s216.args_ref_check with META_T;
  an empty pointer equals the base's absent one); every other key equal to the base's; a non-META base key missing from the cell -> ARGS_BAD; any other new
  key -> ARGS_BAD.
Readings (v1 §4.2 / §4.3): T1 T-ABSORB min a(9..12) >= 0.95, T-DBS M = mean a(13..16) >= 0.80, T-SELF = T-ABSORB and uncensored, round-2 condition =
  T-ABSORB and uncensored; T2 T-ABSORB-2 min a(13..16) >= 0.95, T-RETAIN min a(9..12) >= 0.99, T-DBS-2 M2 = mean a(17..20) >= 0.80, M_far = mean a(21..24)
  (CTX-FEASIBLE beside), T-SELF-2 = T-ABSORB-2 and T-RETAIN and uncensored. Bars (r1_bars / r2_bars verbatim from score_s216 = SD2 v1_panels.py) on the
  text base fp32 curve a_0 and T1's a_1; mechanism T-FULL / T-GROW / T-SHIFT/MIXED / T-SHRINK, T2's T-FULL also M_far >= 0.60; UNLABELLED (not absorbed /
  censored / bars void / bars degenerate SB >= MG).
Comparison (v1 §4.4 + Addendum 1 item 1): Delta_r = n50_T(r) - LATENT_N50[r]; sentinels "> 24" -> 24 - n50_L + 0.01, "< 1" -> 1 - n50_L - 0.01; BASE-PARITY
  |Delta_0| <= 0.25 -> levels, else gain Delta_r - Delta_0 (both reported). PARITY |D1| <= 0.5 and |D2| <= 0.5; TEXT-AHEAD not PARITY and D2 > 0.5 and
  D1 >= -0.5; LATENT-AHEAD not PARITY and D2 < -0.5 and D1 <= 0.5; CROSSED otherwise; "marginal" if TEXT-/LATENT-AHEAD with |D2| <= 1.0. Round 1 not
  absorbed -> TEXT-NOT-ABSORBED; round 2 not absorbed -> TEXT-NOT-ABSORBED-2 (+ ROUND-1-ONLY); a round-2 window failed by both labellers ->
  TEXT-LABEL-STALL + ROUND-1-ONLY (PARITY-1 / TEXT-AHEAD-1 / LATENT-AHEAD-1 by Delta_1); GATE-T1 failed -> NOT RUN; a censored cell -> TEXT-CENSORED
  (r, reason): every comparison row fails. Every comparison line carries the §5.1 qualifiers, CTX-BOUND, TOKEN-COST and TRAIN-TOKENS (Addendum 2 item 2:
  never worded as compute-matched).
    python -B theory/score_s219.py [--final s1] [--not_admitted <cell>,...]
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = _HR if os.path.isdir(os.path.join(_HR, "boxes", "runs_fsq")) else "D:/Latent"    # a scratch copy reads (never writes) the repo
R = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(ROOT, "theory", "results", "score_s219.json")
for _p in (ROOT, os.path.join(ROOT, "theory")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import score_s213 as S213  # noqa: E402   (META only)

EPS = 1e-9
REGISTERED = "docs/research_textself_2026-10-18/TS_v1.md @ 146a25d0 (v1 + Addendum 1 + Addendum 2); THEORY.md §219"
LIN = {"s1": {"base_model": "s170_cot_s1", "base": "s219_base_s1", "t1": "s219_t1_s1", "t2": "s219_t2_s1", "pool_t1": "s219_pool_t1_s1",
              "pool_t2": "s219_pool_t2_s1", "gate_t1": "s219_gate_t1_s1", "gate_t2": "s219_gate_t2_s1",
              "latent": {"r0": "s213_d3t_s1", "r1": "s215_ftself_s1", "r2": "s216_r2_s1"}}}
POOL_FILE = {"pool_t1": "pool_t1.jsonl", "pool_t2": "pool_t2.jsonl"}       # v1 §4.6 lineage table
ROLE = {"t1": {"round": 1, "pool": "pool_t1", "gate": "gate_t1", "prev": "base_model", "micro": 2},
        "t2": {"round": 2, "pool": "pool_t2", "gate": "gate_t2", "prev": "t1", "micro": 4}}
BASE_SHA = "36f4659b99f08a65202b303f4af74ef20ac65f1e60d2a25f3e84aafbfb2e3392"   # V6 (s170_cot_s1/model.pt)
FT_FULL, SUPPORT = 12000, {0: 8, 1: 12, 2: 16}
CTX_ROWS = 1025                                                             # v1 §2.8: feasible iff P + C <= 1025
TEST_SEED, TEST_NMAX, EVAL_BS = 17100, 24, 32
CARD_CLASS = "3070"        # T5. §2.7 fallback: Theory sets "4060" (and LATENT_N50_BY_CARD["4060"] from the latent re-evals on that card) by a note
CARD_BACKUP = "2070"       # usable for whole files only after CARD-GATE (G-T21) passes on the base
CARDS = ("3070", "2070", "4060", "3090", "4090", "A100", "H100")
HASH_MODELS = True         # FT cells: meta model_sha256 == sha256(model.pt) when model.pt is present
BASE_NOTE = None           # G-T19: {"note": "219-x", "branch": "levels" | "gain"} once Theory records it; the scorer asserts its own branch matches
NOT_ADMITTED, FINAL = set(), set()

# ---- latent comparators (Addendum 2 item 4): constants, sources, rule. Never re-read for a decision.
LATENT_N50 = {"r0": 10.306, "r1": 15.460, "r2": 20.539}
LATENT_N50_BY_CARD = {"3070": LATENT_N50}
LATENT_SRC = {"r0": "boxes/runs_fsq/s213_d3t_s1/eval_vb.json acc_n2 (n 1..16) + eval_deep.json acc_n2 (n 17..24); D3-T s1 (THEORY §213 / §216)",
              "r1": "boxes/runs_fsq/s215_ftself_s1/eval_vb.json acc_n2 (n 1..16) + eval_deep.json acc_n2 (n 17..24); FT-SELF (THEORY §215)",
              "r2": "boxes/runs_fsq/s216_r2_s1/eval_vb.json acc_n2 (n 1..16) + eval_deep.json acc_n2 (n 17..24); R2-s1 (THEORY §216)"}
LATENT_RULE = ("n50 = the first downward crossing of 0.5 by a(n) over n 1..24 at K' n + 2, linear interpolation (v1 §4.2); eval_vb (bs 32) for n <= 16, "
               "eval_deep (bs 16) for n >= 17; the two files agree exactly at n 13..16; all RTX 3070 fp32 tf32 off")
_ONE = {n: 1.0 for n in range(1, 25)}
LATENT_CURVES = {   # acc_n2 copied from the LATENT_SRC files (fixture-checked against them read-only); n 13..16 identical in eval_vb and eval_deep
    "r0": {**_ONE, **{9: 0.923, 10: 0.603, 11: 0.266, 12: 0.127, 13: 0.141, 14: 0.127, 15: 0.135, 16: 0.115, 17: 0.119, 18: 0.103, 19: 0.099,
                        20: 0.093, 21: 0.089, 22: 0.092, 23: 0.099, 24: 0.103}},
    "r1": {**_ONE, **{13: 0.999, 14: 0.893, 15: 0.633, 16: 0.344, 17: 0.172, 18: 0.133, 19: 0.121, 20: 0.124, 21: 0.1, 22: 0.105, 23: 0.107,
                        24: 0.115}},
    "r2": {**_ONE, **{15: 0.997, 17: 0.998, 18: 0.979, 19: 0.862, 20: 0.659, 21: 0.364, 22: 0.206, 23: 0.142, 24: 0.121}}}
LATENT_BARS_EXPECTED = {"MG_R1": 0.63, "SB_R1": 0.40, "MG2_R2": 0.785, "SB2_R2": 0.60}     # v1 §4.3 (asserted)
LATENT_STAB = (0.782, 0.489, 0.437)                                         # v1 §5.1 latent cost line (MEM-TRADED-2, MEM-ERODING)
TEXT_BASE_REROOT_KEEP = {9: 1.000, 10: 0.9975, 11: 0.985, 12: 0.985, 13: 0.955, 14: 0.875, 15: 0.775, 16: 0.645, 20: 0.125}   # v1 §2.2 [MEASURED]
LATENT_D3T_CHAIN_KEEP = {9: 1.000, 10: 0.996, 11: 0.988, 12: 0.976, 13: 0.972, 14: 0.952, 15: 0.943, 16: 0.917, 20: 0.781}   # v1 §2.2 (SD2_v1 §2.1)

# ---- ARGS-DIFF (v1 §4.6, T1, V2; Addendum 1 item 3; Addendum 2 item 1)
META_T = set(S213.META) | {"excl_survive_by_n", "hostprobe_sps"}
TEXT_NEUTRAL_NEW = {"n_slots": 8, "anneal": "", "probe_loss_every": 0, "mem_layer": 6, "cache_lower": False, "anneal_gate": "", "csr": "", "iit": "",
                    "iit_tau": 1, "iit_nmax": 0, "grokfast": "", "gf_alpha": 0.98, "gf_lamb": 2.0, "cot_eval_bs": 64, "eval_only": False, "aux_step": 0.0,
                    "aux_fade": "36000,48000", "kmatch_impl": "masked", "add_norm": "", "aux_nopass": False, "aux_reweight": False, "aux_hold": False,
                    "kprimes": "", "path_indep": False, "excl_min_n": 1, "scratch": "", "warmup": 100, "k_window": "", "fast_add": False}
assert len(TEXT_NEUTRAL_NEW) == 29
FT_KEYS = {"steps", "lr", "init_from", "self_pool", "p_self", "self_micro", "no_final_eval", "stop_at", "draws_log_steps", "mem_check_self",
           "mem_check_steps", "args_ref", "out"}
SMOKE_FREE = {"stop_at", "mem_check_self"}                                  # G-T5 smokes (G-T4 --stop_at 2; G-T9 --mem_check_self): any value


def FT_REQ(lin, role):
    L, r = LIN[lin], ROLE[role]
    return {"steps": 12000, "lr": 3e-05, "init_from": "boxes/runs_fsq/%s/model.pt" % L[r["prev"]],
            "self_pool": "boxes/runs_fsq/%s/%s" % (L[r["pool"]], POOL_FILE[r["pool"]]), "p_self": 0.5, "self_micro": r["micro"], "no_final_eval": True,
            "stop_at": 0, "draws_log_steps": 12000, "mem_check_self": False}


WIN = {"W2": range(13, 17), "W1": range(9, 13)}                           # W2 = GATE-T2 (n 13-16), W1 = GATE-T1' (n 9-12) / GATE-T1
GATE_FLAG = {"W2": "GATE_T2", "W1": "GATE_T1p"}
WIN_ALIAS = {"W2": ("W2", "w2", "13-16", "13..16", "new", "GATE_T2", "GATE-T2", "gate_t2"),
             "W1": ("W1", "w1", "9-12", "9..12", "old", "GATE_T1p", "GATE-T1p", "GATE-T1\u2032", "gate_t1p", "GATE_T1", "GATE-T1")}
LA = {"pi": ("pi_label", "pi"), "pi_E": ("pi_state", "pi_E"), "pi_ci": ("pi_label_ci95", "pi_ci95"), "pi_E_ci": ("pi_state_ci95", "pi_E_ci95")}
POOL_KEYS = {"syms", "n", "chain_label", "values"}
ITEM_FIELDS = ("idx", "n", "y", "P", "C_struct", "overflow", "answer", "ntok", "eot", "form_valid", "strict", "first_bad_step", "first_bad_pos")
TRAINER_EVAL = re.compile(r"^eval(_\d+k)?\.json$")
LAYOUTS = {   # writer layouts of record (boxes/textself_ft.py, boxes/textself.py, boxes/text_eval.py, scratch/s219_gate.py, scratch/s219_build_pool.py at
    # 3b0bb06f; reconciled 2026-10-18 for Theory). Every field below is one the scorer reads; [desc.] = descriptive only.
    "E1 eval_text.json (text_eval --part text)": "meta {run, model, model_sha256, card, dtype 'fp32', tf32 false, cot_eval_bs 32 (asserted by the writer), "
                         "test_seed 17100, test_n_max 24, ns, n_items, items_sha256, budget_rule, commit, sec, t_start, t_end (ISO, UTC offset; seal order), "
                         "peak_gib}; items = a list of 24,000 dicts {idx (index into test_set(1000, 24)), n, y, P (= len(syms) + 2), C_struct, overflow, answer "
                         "(-1 = failure), ntok, eot, form_valid, strict, first_bad_step (1-based), first_bad_pos (1-based row = P + j + 1, comparable to "
                         "T_train = max(P + C))}; acc_by_n, acc_feasible_by_n, form_valid_by_n, strict_by_n, ntok_mean_by_n, first_bad_pos_q_by_n, chance_by_n "
                         "[desc.]. The scorer also accepts {n: list} and columnar {n: {field: list}} items. eval_text.partial.json (written while running) "
                         "is never read",
    "E2 eval_textself.json (text_eval --part textself)": "label_audit {n_audit, pi, pi_E, pi_ci95, pi_E_ci95, by_window {W1 (T1), W2 + W1 (T2): {same}}, "
                         "kept_wrong [{syms, n, step, op, a, b, value, gold, wrong_steps, label, gold_label, pattern}]}; contamination {pool_excl_hits, "
                         "pool_test_hits, gold_key_hits, self_reuse, CONTAMINATED} (CPU, registered); memo {n: {window, acc_pool, acc_gate, gap}} [desc.]; "
                         "labeller_kept {n: {keep, pi, pi_E, n_kept}} [desc.]; labeller_kept_meta {t_start} (the seal order of the n >= 13 labelling)",
    "B1 pool_meta.json (s219_build_pool)": "per_n {n: int}, ns, n_records, sha256 (of the pool file) [desc.], card, T_max_by_n {n: int} (T_train,r = its "
                         "max), windows, t_start (seal order for pool_t2 vs T1's eval_text)",
    "B2 eval_gate.json (s219_gate --stage score)": "by_n {n: {window, labeller, items, n_kept, keep, pi, pi_E, pi_ci95, pi_E_ci95}}; pooled_by_window {W: "
                         "{items, n_kept, pi, pi_E, pi_ci95, pi_E_ci95}}; GATE_T1 / GATE_T2 / GATE_T1p (bool); fallback null or {W: {labeller, by_n, pooled, "
                         "GATE}}; labeller_full flat {n: {window, keep, pi, pi_E, n_kept}} [desc.]; equiv {W: {by_n {n: {items, agree_keep, agree_values, "
                         "agree_fail_step, agree_all, share, PASS}}, disagreements, min_margin, PASS}} [desc.]; kept_vs_drawn [desc.]; card; t_start",
    "T1 log.jsonl (textself_ft)": "per record {step, loss, loss_last, plain_loss, self_loss, self_value_ce, self_copy_ce, self_frac, pool_cursor, n_self_steps, "
                         "self_reuse, lr, n_nonfinite, train_tokens {plain, self: {target_tokens, token_positions, padded_positions}}}; skipped steps "
                         "{step, nonfinite: 1}; a resume {step, resumed}",
    "T1 train_done.json (textself_ft, the cell's done marker)": "{steps, train_sec, n_nonfinite, n_self_steps, pool_cursor, self_reuse, train_tokens, "
                         "self_n_counts, model_sha256}: TRAIN-TOKENS' text side (else the last logged train_tokens) and a cross-check of the log [desc.]",
    "T1 draws_log.json": "list of {step, plain_hash, coin, n (plain) | self_n + U (self)}, steps 1..12,000",
    "T1 args.json": "vars(a) of boxes/textself_ft.py (nested170's 57 keys + the FT keys) + train_gpu, train_driver, env_info keys, hostprobe_sps, format",
    "other trainer files (never eval*.json)": "init_from.json, args_diff.json, mem_check.json, mem_check_self.json (G-T9), timing.json (G-T10), "
                         "ckpt_*.pt: none is read for a reading and none is a SEAL-BREACH; any eval.json / eval_<k>k.json / *.void_unsealed is"}


def set_root(r, root=None):
    global R, ROOT
    R = r
    if root is not None:
        ROOT = root


def jl(d, f):
    p = os.path.join(R, d, f)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except ValueError as e:
        return {"__unreadable__": str(e)[:200]}


def ok_json(x):
    return isinstance(x, (dict, list)) and not (isinstance(x, dict) and "__unreadable__" in x)


def mean(v):
    v = list(v)
    return sum(v) / len(v)


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _eq(x, y):
    if isinstance(x, bool) or isinstance(y, bool):
        return isinstance(x, bool) and isinstance(y, bool) and x == y
    if num(x) and num(y):
        return float(x) == float(y)
    return x == y


def same_path(v, r):
    if not isinstance(v, str):
        return False
    v = v.replace("\\", "/")
    return v == r or v.endswith("/" + r)


def fmt(v, p=3):
    return ("%.*f" % (p, v)) if num(v) else str(v)


def card_class(c):
    return next((k for k in CARDS if isinstance(c, str) and k in c), None)


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


def _ci(x):
    return isinstance(x, (list, tuple)) and len(x) == 2 and all(num(v) for v in x)


def ci_overlap(a, b):
    return bool(_ci(a) and _ci(b) and a[0] <= b[1] + EPS and b[0] <= a[1] + EPS)


def _time(x):
    if num(x):
        return float(x)
    if isinstance(x, str):
        try:
            t = datetime.datetime.fromisoformat(x.replace("Z", "+00:00"))
            return t.timestamp() if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------- test set (CPU, deterministic; nested170 imports torch without touching CUDA)
_TE = {}


def nested():
    from boxes import nested170 as N
    return N


def test_items(n_max):
    if n_max not in _TE:
        _TE[n_max] = nested().test_set(1000, n_max=n_max, seed=TEST_SEED)
    return _TE[n_max]


_TI = {}


def test_info():
    """test_set(1000, 24): per n syms, t, P = len(syms) + 2, C = len(cot_ids) (= C_struct, v1 §2.7), overflow = P + C > 1025; items_sha256 (v1 §2.7)."""
    if not _TI:
        N = nested(); TE = test_items(TEST_NMAX); by = {}
        for it in TE:
            b = by.setdefault(it["n"], {"syms": [], "t": [], "P": [], "C": [], "ovf": []}); P_, C_ = len(it["syms"]) + 2, len(N.cot_ids(it["root"]))
            b["syms"].append(it["syms"]); b["t"].append(int(it["t"])); b["P"].append(P_); b["C"].append(C_); b["ovf"].append(P_ + C_ > CTX_ROWS)
        its = [[it["syms"], int(it["t"])] for it in TE]
        _TI.update(by=by, sha=hashlib.sha256(json.dumps(its, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest())
        _TI["ctx"] = {"overflow_by_n": {n: sum(by[n]["ovf"]) for n in by}, "max_PC": max(p + c for n in by for p, c in zip(by[n]["P"], by[n]["C"])),
                      "exactly_1025_n24": sum(p + c == CTX_ROWS for p, c in zip(by[24]["P"], by[24]["C"]))}
        _TI["mean"] = {n: {"P": mean(by[n]["P"]), "C": mean(by[n]["C"])} for n in by}
    return _TI


# ---------------------------------------------------------------- bars: verbatim from theory/score_s216.py (= docs/research_selfdepth2_2026-10-18/v1_checks/v1_panels.py)
def ceil005(x):
    return round(math.ceil(x / 0.005 - 1e-6) * 0.005, 3)


def ip(a, x):
    lo = math.floor(x); f = x - lo
    return a[lo] if f < 1e-12 else (1 - f) * a[lo] + f * a[lo + 1]


def r1_bars(a0):
    """Round-1 (B 8 -> 12) bars from the lineage's own base: H_shift M = mean a0(9..12) (exact translation by +4)."""
    M0 = mean(a0[n] for n in range(9, 13)); MG = ceil005(M0 + 0.15)
    return {"M_shift": M0, "MARGIN_GROWS": MG, "SHRINK_below": ceil005(M0 - 0.08), "DBS": 0.80, "FULL_M": max(0.80, MG), "K_BEYOND": 0.25}


def r2_bars(a0, a1):
    """Round-2 (B 12 -> 16) bars from the lineage's own base and round-1 curves:
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


def cross(a, L, lo=1, hi=24):
    """v1 §4.2: first downward crossing of L over n lo..hi, linear interpolation; '< lo' if a(lo) is already below; '> hi' if never (score_s216.cross)."""
    if a[lo] < L - EPS:
        return "< %d" % lo
    for n in range(lo, hi):
        if a[n] >= L - EPS and L > a[n + 1]:
            return n + (a[n] - L) / (a[n] - a[n + 1])
    return "> %d" % hi


def latent_selfcheck():
    """Addendum 2 item 4 / v1 §4.3: the embedded curves reproduce the constants (3 decimals) and the latent bar instance."""
    n50 = {r: cross(LATENT_CURVES[r], 0.5) for r in LATENT_CURVES}
    b1, b2 = r1_bars(LATENT_CURVES["r0"]), r2_bars(LATENT_CURVES["r0"], LATENT_CURVES["r1"])
    bars = {"MG_R1": b1["MARGIN_GROWS"], "SB_R1": b1["SHRINK_below"], "MG2_R2": b2["MARGIN_GROWS_2"], "SB2_R2": b2["SHRINK_below"]}
    ok_n = all(num(n50[r]) and round(n50[r], 3) == LATENT_N50[r] for r in n50); ok_b = all(abs(bars[k] - LATENT_BARS_EXPECTED[k]) < 1e-9 for k in bars)
    return {"n50_from_curves": n50, "bars": bars, "n50_ok": ok_n, "bars_ok": ok_b}


_LSC = latent_selfcheck()
assert _LSC["n50_ok"] and _LSC["bars_ok"], ("latent constants / bars not reproduced", _LSC)


def delta(n50, L):
    """v1 §4.4 sentinels (m12)."""
    if num(n50):
        return n50 - L
    if n50 == "> 24":
        return 24 - L + 0.01
    if n50 == "< 1":
        return 1 - L - 0.01
    return None


def four_way(d1, d2):
    """v1 §4.4: exhaustive and exclusive; |Delta| exactly 0.5 is inside the band."""
    par = abs(d1) <= 0.5 + EPS and abs(d2) <= 0.5 + EPS
    ta = (not par) and d2 > 0.5 + EPS and d1 >= -0.5 - EPS
    la = (not par) and d2 < -0.5 - EPS and d1 <= 0.5 + EPS
    lab = "PARITY" if par else "TEXT-AHEAD" if ta else "LATENT-AHEAD" if la else "CROSSED"
    return lab, bool(lab in ("TEXT-AHEAD", "LATENT-AHEAD") and abs(d2) <= 1.0 + EPS)


def crossed_subcase(d1, d2):
    """Audit's G-T12 wording item (note 219-2): CROSSED's sub-case, label unchanged. Under four_way, CROSSED is reached only with
    (a) opposite-sign leads outside the band at both rounds ("order changed"), or (b) |D2| <= 0.5 with |D1| > 0.5 ("converged at round 2").
    "Diverged at round 2" (D1 in band, D2 out) never reaches CROSSED: TEXT-/LATENT-AHEAD absorb it (four_way's d1 >= -0.5 / <= 0.5 clauses)."""
    if abs(d2) <= 0.5 + EPS and abs(d1) > 0.5 + EPS:
        return "converged at round 2"
    if abs(d1) > 0.5 + EPS and abs(d2) > 0.5 + EPS and (d1 > 0) != (d2 > 0):
        return "order changed"
    return "unexpected (four_way should not give CROSSED here)"


def round1_only(d1):
    return "PARITY-1" if abs(d1) <= 0.5 + EPS else "TEXT-AHEAD-1" if d1 > 0 else "LATENT-AHEAD-1"


# ---------------------------------------------------------------- eval_text (E1)
_ITEMS, _SHA = {}, {}


def _pos(n, idx, lo_n=None):
    """within-n positions for indices idx (global into test_set(1000, 24), into its n >= lo_n slice (latent eval_deep), or within n; detected);
    must be a permutation of 0..999."""
    if not isinstance(idx, list) or len(idx) != 1000 or not all(isinstance(i, int) and not isinstance(i, bool) for i in idx):
        return None
    for base in ((n - 1) * 1000,) + (((n - lo_n) * 1000,) if lo_n else ()) + (0,):
        if all(base <= i < base + 1000 for i in idx):
            q = [i - base for i in idx]
            return q if len(set(q)) == 1000 else None
    return None


def _norm_items(IT):
    """-> {n: {field: list}} (file order) from a list of dicts, {n: list of dicts} or {n: {field: list}}; None if unreadable."""
    def cols(rs):
        return {f: [r.get(f) if isinstance(r, dict) else None for r in rs] for f in ITEM_FIELDS}
    if isinstance(IT, list):
        by = {}
        for r in IT:
            if not isinstance(r, dict) or not isinstance(r.get("n"), int):
                return None
            by.setdefault(r["n"], []).append(r)
        return {n: cols(rs) for n, rs in by.items()}
    if isinstance(IT, dict):
        out = {}
        for k, v in IT.items():
            try:
                n = int(k)
            except ValueError:
                return None
            if isinstance(v, list):
                out[n] = cols(v)
            elif isinstance(v, dict) and all(isinstance(v.get(f), list) for f in ("idx", "y", "answer")):
                L_ = len(v["idx"]); out[n] = {f: (list(v[f]) if isinstance(v.get(f), list) and len(v[f]) == L_ else [None] * L_) for f in ITEM_FIELDS}
            else:
                return None
        return out
    return None


def file_sha256(p):
    st = os.stat(p); key = (p, st.st_mtime_ns, st.st_size)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 22), b""):
                h.update(ch)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


_TR = {}


def text_read(d, fname="eval_text.json", expect_sha=None, cards_ok=None, forbid_sha=()):
    """v1 §2.7 / §4.6 text_read (cached per score() run). -> {void, checks, a, ...}; the normalised per-item store is kept in _ITEMS[(R, d, fname)]."""
    key = (R, d, fname, expect_sha, frozenset(cards_ok or {CARD_CLASS}), tuple(forbid_sha))
    if key not in _TR:
        _TR[key] = _text_read(d, fname, expect_sha, cards_ok, forbid_sha)
    return _TR[key]


def _text_read(d, fname, expect_sha, cards_ok, forbid_sha):
    f = jl(d, fname)
    if f is None:
        return {"void": "%s missing" % fname, "missing": True}
    if not ok_json(f) or not isinstance(f, dict):
        return {"void": "%s unreadable" % fname, "missing": True}
    m = f.get("meta") or {}; TI = test_info(); cards_ok = set(cards_ok or {CARD_CLASS}); card = m.get("card"); cc = card_class(card)
    chk = {"items_sha256": m.get("items_sha256") == TI["sha"], "card": isinstance(card, str) and cc in cards_ok, "dtype": m.get("dtype") == "fp32",
           "tf32": m.get("tf32") is False, "cot_eval_bs": _eq(m.get("cot_eval_bs"), EVAL_BS), "test_seed": _eq(m.get("test_seed"), TEST_SEED),
           "test_n_max": _eq(m.get("test_n_max"), TEST_NMAX)}
    if m.get("n_lo") is not None or m.get("n_hi") is not None:
        chk["n_range"] = (m.get("n_lo"), m.get("n_hi")) == (1, 24)
    cards = {c for c in ([card, f.get("card")] + list(m.get("cards") or [])) if c is not None}
    ms = m.get("model_sha256")
    if expect_sha is not None:
        chk["model_sha256"] = ms == expect_sha
    else:
        chk["model_sha256"] = isinstance(ms, str) and re.fullmatch(r"[0-9a-f]{64}", ms) is not None and ms not in forbid_sha   # not the base's / T1's model
        mp = os.path.join(R, d, "model.pt")
        if chk["model_sha256"] and HASH_MODELS and os.path.exists(mp):
            chk["model_sha256_file"] = file_sha256(mp) == ms
        td = jl(d, "train_done.json")
        if chk["model_sha256"] and isinstance(td, dict) and isinstance(td.get("model_sha256"), str):
            chk["model_sha256_train_done"] = td["model_sha256"] == ms               # the trainer's own record of the model it saved
    A = f.get("acc_by_n") or {}; IT = _norm_items(f.get("items"))
    chk["fields"] = bool(isinstance(A, dict) and sorted(A, key=lambda k: int(k)) == [str(n) for n in range(1, 25)] and all(num(A[k]) for k in A)
                         and IT is not None and sorted(IT) == list(range(1, 25)))
    bad, flags, per = [], [], {}
    if chk["fields"]:
        for n in range(1, 25):
            c = IT[n]; q = _pos(n, c["idx"]); ti = TI["by"][n]
            if q is None or any(v is not None and v != n for v in c["n"]):
                bad.append((n, "idx / n")); continue
            if [c["y"][j] for j in sorted(range(1000), key=lambda j: q[j])] != ti["t"]:
                bad.append((n, "y != test t")); continue
            P_ = [ti["P"][qq] for qq in q]; C_ = [ti["C"][qq] for qq in q]; ov = [ti["ovf"][qq] for qq in q]
            if c["P"] != P_ or c["C_struct"] != C_:
                bad.append((n, "P / C_struct != the item's")); continue
            if any(not (isinstance(v, bool) or v in (0, 1)) for v in c["overflow"]) or [bool(v) for v in c["overflow"]] != ov:
                bad.append((n, "overflow != the C_struct rule (P + C > 1025)")); continue
            if any(o and a_ != -1 for o, a_ in zip(ov, c["answer"])):
                bad.append((n, "an overflow item has an answer")); continue
            corr = [int(a_ == y_) for a_, y_ in zip(c["answer"], c["y"])]
            if abs(sum(corr) / 1000 - A[str(n)]) > 1e-12:
                bad.append((n, "acc_by_n != mean(answer == y)")); continue
            fv = [bool(v) for v in c["form_valid"]] if all(isinstance(v, bool) for v in c["form_valid"]) else None
            st = [bool(v) for v in c["strict"]] if all(isinstance(v, bool) for v in c["strict"]) else None
            if fv is not None and st is not None and any(s_ != (f_ and k_ == 1) for s_, f_, k_ in zip(st, fv, corr)):
                flags.append((n, "strict != form_valid and correct"))
            feas = [k_ for k_, o in zip(corr, ov) if not o]
            per[n] = {"q": q, "corr": corr, "ovf": ov, "P": P_, "C": C_, "ntok": c["ntok"], "fbp": c["first_bad_pos"], "fbs": c["first_bad_step"],
                      "fv": fv, "st": st, "a_feas": (sum(feas) / len(feas)) if feas else None, "n_feas": len(feas)}
            for key, val in (("acc_feasible_by_n", per[n]["a_feas"]), ("form_valid_by_n", mean(fv) if fv else None), ("strict_by_n", mean(st) if st else None)):
                fv_ = (f.get(key) or {}).get(str(n))
                if num(fv_) and num(val) and abs(fv_ - val) > 1e-9:
                    flags.append((n, "%s != items" % key))
    chk["items"] = chk["fields"] and not bad
    chk["one_card"] = len(cards) <= 1
    void = sorted(k for k, v in chk.items() if not v)
    if void:
        return {"void": "%s %s" % (fname, void), "checks": chk, "item_problems": bad[:10], "meta": m, "card_class": cc}
    _ITEMS[(R, d, fname)] = per
    a = {n: float(A[str(n)]) for n in range(1, 25)}; n50, n90 = cross(a, 0.5), cross(a, 0.9)
    return {"void": None, "checks": chk, "flags": flags, "meta": {k: m.get(k) for k in ("run", "model_sha256", "card", "dtype", "tf32", "cot_eval_bs", "commit",
                                                                                          "sec", "peak_gib", "budget_rule", "t_start", "t_end")},
            "card_class": cc, "a": a, "a_feasible": {n: per[n]["a_feas"] for n in per}, "form_valid": {n: mean(per[n]["fv"]) if per[n]["fv"] else None for n in per},
            "strict": {n: mean(per[n]["st"]) if per[n]["st"] else None for n in per}, "ntok_mean_by_n": f.get("ntok_mean_by_n"),
            "chance_by_n": f.get("chance_by_n"), "n50": n50, "n90": n90, "WIDTH": (n50 - n90) if (num(n50) and num(n90)) else None,
            "ctx_feasible_share": {n: per[n]["n_feas"] / 1000 for n in per}, "a24_feasible": per[24]["a_feas"]}


# ---------------------------------------------------------------- logs, args, gates, label audit, contamination, seal
def log_info(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    recs, nf, badl = {}, [], 0
    with open(p) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                badl += 1; continue
            if not isinstance(r, dict) or not isinstance(r.get("step"), int) or isinstance(r.get("step"), bool):
                continue
            if num(r.get("nonfinite")) and r["nonfinite"] > 0:
                nf.append(r["step"])
            recs[r["step"]] = r
    if not recs:
        return {"recs": {}, "last": None}
    cur = nss = None
    for s in sorted(recs):
        cur = recs[s].get("pool_cursor", cur); nss = recs[s].get("n_self_steps", nss)
    lastr = recs[max(recs)]
    return {"recs": recs, "last": max(recs), "nonfinite_records": len(nf), "nonfinite_steps": nf[:20], "bad_lines": badl,
            "reuse": max((r.get("self_reuse") or 0) for r in recs.values() if num(r.get("self_reuse") or 0)), "cursor": cur, "n_self_steps": nss,
            "nan_loss_records": sorted(s for s, r in recs.items() if isinstance(r.get("loss"), float) and not math.isfinite(r["loss"]))[:10],
            "self_value_ce_last": lastr.get("self_value_ce"), "self_copy_ce_last": lastr.get("self_copy_ce"),
            "train_tokens_last": next((recs[s_]["train_tokens"] for s_ in sorted(recs, reverse=True) if isinstance(recs[s_].get("train_tokens"), dict)), None),
            "train_tokens_step": next((s_ for s_ in sorted(recs, reverse=True) if isinstance(recs[s_].get("train_tokens"), dict)), None)}


def args_ref_check(a, b):
    """216-7 rule (score_s216.args_ref_check, META_T): args_ref may differ from the reference's iff the file it names (repo-relative) equals the reference
    args.json on every non-META_T key; equal pointers pass; an empty pointer equals the base's absent one; a missing or unreadable file fails."""
    va, vb = a.get("args_ref"), b.get("args_ref")
    if va == vb or (not va and not vb):
        return {"ok": True, "why": "same pointer"}
    p = va if (va and os.path.isabs(str(va))) else os.path.join(ROOT, str(va or ""))
    if not va or not os.path.isfile(p):
        return {"ok": False, "why": "args_ref file missing: %s" % va}
    try:
        with open(p) as fh:
            f = json.load(fh)
    except Exception as e:                                                  # noqa: BLE001
        return {"ok": False, "why": "args_ref unreadable: %s" % e}
    diff = sorted(k for k in set(f) | set(b) if k not in META_T and k != "args_ref" and f.get(k) != b.get(k))
    return {"ok": not diff, "why": "content equals the reference args" if not diff else "content differs: %s" % diff[:10], "file": va}


def args_diff(lin, role, a=None, smoke=False, smoke_pool=None):
    """v1 §4.6 ARGS-DIFF (T1, V2): the cell's args.json (or a given dict, G-T5) vs s170_cot_s1/args.json. smoke=True (G-T5 only) frees stop_at /
    mem_check_self and, with smoke_pool, takes the smoke pool path as self_pool's required value (note 219-3 item 3); the scored cell never uses either."""
    L = LIN[lin]; a = jl(L[role], "args.json") if a is None else a; b = jl(L["base_model"], "args.json")
    if not ok_json(a) or not ok_json(b):
        return None
    req_t = {k: v for k, v in FT_REQ(lin, role).items() if not (smoke and k in SMOKE_FREE)}
    if smoke and smoke_pool:
        req_t["self_pool"] = smoke_pool
    req = {k: (v, a.get(k)) for k, v in req_t.items() if not (same_path(a.get(k), v) if k in ("init_from", "self_pool") else (k in a and _eq(a.get(k), v)))}
    shared = {k: (b[k], a[k]) for k in a if k in b and k not in META_T and k not in FT_KEYS and not _eq(a[k], b[k])}
    dropped = sorted(k for k in b if k not in a and k not in META_T)                      # V2
    neutral = {k: (v, a.get(k, "<missing>")) for k, v in TEXT_NEUTRAL_NEW.items() if not (k in a and _eq(a[k], v))}
    new = sorted(k for k in a if k not in b and k not in META_T and k not in TEXT_NEUTRAL_NEW and k not in FT_KEYS)
    ar = args_ref_check(a, b)
    return {"ref": "%s/args.json" % L["base_model"], "smoke": smoke, "shared_diffs": shared, "dropped_base_keys": dropped, "neutral_bad": neutral,
            "unexpected_new_keys": new, "required_bad": req, "args_ref": ar,
            "ok": not shared and not dropped and not neutral and not new and not req and ar["ok"]}


def gate_rule(B, ns):
    """GATE-1's rule on by_n entries over ns, pooled pi / pi_E recomputed item-weighted from n_kept: (ok, pi_pool, pi_E_pool, problem) (score_s216)."""
    es = [(B or {}).get(str(n), (B or {}).get(n)) for n in ns]
    kv = lambda e: e.get("keep", e.get("keep_kv"))  # noqa: E731
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
    per_n = {n: {k: ((B or {}).get(str(n)) or {}).get(k) for k in ("keep", "pi", "pi_E", "n_kept")} for n in ns}
    return {"recomputed": ok, "file_flag": flag, "pooled_matches": pm, "consistent": cons, "PASS": bool(ok and flag is True and cons), "pi_pool": pp,
            "pi_E_pool": pe, "problem": why, "ci": {"pi": (P or {}).get("pi_ci95"), "pi_E": (P or {}).get("pi_E_ci95")}, "per_n": per_n,
            "labeller": sorted({str(((B or {}).get(str(n)) or {}).get("labeller")) for n in ns})}


def gate_descriptive(g):
    """LABELLER-FULL / KEPT-VS-DRAWN / EQUIV (descriptive; never gate). Writer layouts (scratch/s219_gate.py at 3b0bb06f): labeller_full flat {n: {window,
    keep, pi, pi_E, n_kept}}; equiv {W: {by_n {n: {items, agree_keep, agree_values, agree_fail_step, agree_all, share, PASS}}, disagreements, min_margin,
    PASS}}."""
    eq = g.get("equiv") if isinstance(g.get("equiv"), dict) else {}; es = {}
    for w, v in eq.items():
        if isinstance(v, dict):
            bn = v.get("by_n") if isinstance(v.get("by_n"), dict) else {}
            sh = [x.get("share") for x in bn.values() if isinstance(x, dict) and num(x.get("share"))]
            es[w] = {"PASS": v.get("PASS"), "min_share": min(sh) if sh else None,
                     "items": sum(((x or {}).get("items") or 0) for x in bn.values() if isinstance(x, dict)), "min_margin": v.get("min_margin"),
                     "n_disagreements": len(v.get("disagreements") or [])}
    lf = g.get("labeller_full") if isinstance(g.get("labeller_full"), dict) else {}
    return {"EQUIV": es, "LABELLER_FULL": {k: {f: (x or {}).get(f) for f in ("window", "keep", "pi", "pi_E", "n_kept")} for k, x in lf.items() if isinstance(x, dict)},
            "kept_vs_drawn": g.get("kept_vs_drawn")}


def g_t4_rule(model_diff, single_thread_bitwise=None):
    """G-T4 (note 219-3 item 2): PASS iff a single-threaded deterministic re-run is bitwise, or every tensor is bitwise except the tied wte with
    max |delta wte| <= 1e-6. model_diff {tensor name: max |delta|} (nonzero entries; zero entries allowed). The bitwise step-3 loss is checked by the gate."""
    if single_thread_bitwise is True:
        return {"PASS": True, "reading": "bitwise (single-threaded, deterministic)"}
    nz = {k: v for k, v in (model_diff or {}).items() if not (num(v) and v == 0)}
    bad = {k: v for k, v in nz.items() if "wte" not in k or not num(v) or v > 1e-6}
    mw = max((v for k, v in nz.items() if "wte" in k and num(v)), default=0.0)
    return {"PASS": not bad, "max_wte": mw, "non_wte_or_over_1e-6": bad,
            "reading": "bitwise" if not nz else ("bitwise except wte (max |Δ wte| %.3g ≤ 1e-6)" % mw) if not bad else "FAIL %s" % bad}


def gate_t1(lin):
    g = jl(LIN[lin]["gate_t1"], "eval_gate.json")
    if not ok_json(g):
        return None
    P = g.get("pooled") if isinstance(g.get("pooled"), dict) else _win(g.get("pooled_by_window"), "W1")
    e = _gate_entry(g.get("by_n"), P, g.get("GATE_T1"), range(9, 13))
    e["status"] = ("UNDECIDED" if not isinstance(g.get("GATE_T1"), bool) else "PASS" if e["PASS"] else "FAIL")   # undecided at --final: rule-b failure
    e["card"] = g.get("card"); e["card_class"] = card_class(g.get("card"))
    e["descriptive"] = gate_descriptive(g)
    return e


def gate_t2(lin):
    """GATE-T2 (W2, n 13-16) and GATE-T1' (W1, n 9-12) per window, labeller T1, BASE-LABELLED fallback (score_s216.gate2)."""
    g = jl(LIN[lin]["gate_t2"], "eval_gate.json")
    if not ok_json(g):
        return None
    o = {"card": g.get("card"), "card_class": card_class(g.get("card")), "descriptive": gate_descriptive(g)}
    for w, ns in WIN.items():
        pr = _gate_entry(g.get("by_n"), _win(g.get("pooled_by_window"), w), g.get(GATE_FLAG[w]), ns); fb = _win(g.get("fallback"), w); fr = None
        if not pr["PASS"] and isinstance(fb, dict):
            fr = _gate_entry(fb.get("by_n"), fb.get("pooled"), fb.get("GATE", fb.get(GATE_FLAG[w])), ns); fr["labeller"] = fb.get("labeller")
        st = ("UNDECIDED" if not isinstance(pr["file_flag"], bool) else "SELF-LABELLED" if pr["PASS"] else
              "BASE-LABELLED" if (fr and fr["PASS"]) else "BOTH-FAIL" if (fr and isinstance(fr["file_flag"], bool)) else "UNDECIDED")
        # an undecided window stays UNDECIDED: at --final it is a rule-b failure (TEXT-CENSORED r2), never TEXT-LABEL-STALL (whose ROUND-1-ONLY label could pass)
        o[w] = {"status": st, "primary": pr, "fallback": fr, "used": pr if pr["PASS"] else fr if (fr and fr["PASS"]) else None}
    return o


def label_audit_t(la, gwin, role):
    """LABEL-AUDIT-T (v1 §2.7 / §5.2): pi and pi_E >= 0.99 pooled and per window; each window's Wilson CI (pi, pi_E) overlapping its gate window's.
    gwin {W: gate entry used}. T1: one window (W1), the pooled values if by_window is absent."""
    po = num(_la(la, "pi")) and num(_la(la, "pi_E")) and _la(la, "pi") >= 0.99 - EPS and _la(la, "pi_E") >= 0.99 - EPS; ws = {}
    for w in (("W1",) if role == "t1" else ("W2", "W1")):
        x = _win(la.get("by_window"), w) or (la if role == "t1" else {}); gci = (gwin.get(w) or {}).get("ci") or {}
        ws[w] = {"pi": _la(x, "pi"), "pi_E": _la(x, "pi_E"), "ci_overlap_pi": ci_overlap(_la(x, "pi_ci"), gci.get("pi")),
                 "ci_overlap_pi_E": ci_overlap(_la(x, "pi_E_ci"), gci.get("pi_E"))}
        ws[w]["ok"] = bool(num(ws[w]["pi"]) and num(ws[w]["pi_E"]) and ws[w]["pi"] >= 0.99 - EPS and ws[w]["pi_E"] >= 0.99 - EPS and ws[w]["ci_overlap_pi"]
                           and ws[w]["ci_overlap_pi_E"])
    kw = {}
    for r in la.get("kept_wrong") or []:
        if isinstance(r, dict):
            k = "%s %s %s -> %s (gold %s)" % (r.get("op"), r.get("a"), r.get("b"), r.get("value", r.get("v")), r.get("gold")); kw[k] = kw.get(k, 0) + 1
    return {"pi": _la(la, "pi"), "pi_E": _la(la, "pi_E"), "pooled_ok": bool(po), "windows": ws, "kept_wrong_handles": kw,
            "ok": bool(po and all(v["ok"] for v in ws.values()))}


_TEST30 = {}


def contamination_t(lin, role, sd, info, pm):
    """CONTAMINATED (v1 §2.4 item 8): E2's hit counts, self_reuse, exhaustion (cursor > pool size), and, while the pool file exists, the scorer's own
    re-check of the consumed records pool[:cursor] (key set exactly POOL_KEYS; no test_set(1000, 30) syms; no duplicate). Only ever adds a failure."""
    c = (sd or {}).get("contamination") or {}; L = LIN[lin]; pk = ROLE[role]["pool"]; p = os.path.join(R, L[pk], POOL_FILE[pk]); rc = None
    pn = pm.get("per_n"); ns_ = pm.get("ns")
    size = (pm["n_records"] if num(pm.get("n_records")) else sum(pn.values()) if isinstance(pn, dict) and pn and all(num(v) for v in pn.values()) else
            pn * len(ns_) if num(pn) and isinstance(ns_, list) else None)
    cur = (info or {}).get("cursor")
    if os.path.exists(p) and num(cur):
        if not _TEST30:
            _TEST30["s"] = {tuple(it["syms"]) for it in test_items(30)}
        hits = rd = keybad = dup = 0; seen = set()
        with open(p) as fh:
            for ln in fh:
                if rd >= cur:
                    break
                rd += 1; r = json.loads(ln); s = tuple(r.get("syms") or ())
                keybad += set(r) != POOL_KEYS; hits += s in _TEST30["s"]; dup += s in seen; seen.add(s)
        rc = {"consumed_read": rd, "test30_hits": hits, "key_set_bad": keybad, "duplicates": dup, "file_sha256_matches_meta": file_sha256(p) == pm.get("sha256")}
    cnt = {k: c.get(k) for k in ("pool_excl_hits", "pool_test_hits", "gold_key_hits", "self_reuse")}
    exhausted = bool(num(cur) and num(size) and cur > size)
    bad = bool(any(num(v) and v > 0 for v in cnt.values()) or (info and info.get("reuse", 0) > 0) or exhausted
               or (rc and (rc["test30_hits"] or rc["key_set_bad"] or rc["duplicates"])))
    return dict(cnt, log_self_reuse=(info or {}).get("reuse"), cursor=cur, pool_size=size, exhausted=exhausted, recheck=rc, CONTAMINATED=bad)


def seal_check(lin, role):
    """SEAL-BREACH (T3, v1 §5.2 / V3 / G-T20)."""
    L = LIN[lin]; d = L[role]; p = os.path.join(R, d); files = sorted(os.listdir(p)) if os.path.isdir(p) else []
    tr = [f for f in files if TRAINER_EVAL.match(f) or f.endswith(".void_unsealed")]
    later = []                                                              # this model's outputs at n >= 13 other than eval_text
    if role == "t1":
        later += [(L["gate_t2"], "eval_gate.json", None), (L["pool_t2"], "pool_meta.json", None), (L["pool_t2"], POOL_FILE["pool_t2"], None)]
    sd = jl(d, "eval_textself.json")
    if isinstance(sd, dict) and sd.get("labeller_kept") is not None:
        later.append((d, "eval_textself.json", "labeller_kept"))
    ev = jl(d, "eval_text.json"); own = ok_json(ev)
    watch = [] if own else ["%s/%s%s" % (dd, ff, (":" + part) if part else "") for dd, ff, part in later if os.path.exists(os.path.join(R, dd, ff))]
    order = []
    if own:
        te = _time(((ev or {}).get("meta") or {}).get("t_end"))
        for dd, ff, part in later:
            x = jl(dd, ff) if ff.endswith(".json") else None
            ts = _time(((x.get(part + "_meta") or {}).get("t_start") or (x.get(part) or {}).get("t_start")) if part else x.get("t_start")) if isinstance(x, dict) else None
            if te is not None and ts is not None and ts < te:
                order.append("%s/%s%s started before %s/eval_text.json ended" % (dd, ff, (":" + part) if part else "", d))
    return {"trainer_eval_files": tr, "order_violations": order, "watch": watch, "BREACH": bool(tr or order),
            "order_check": "time fields" if own else "eval_text missing (presence watch only)"}


# ---------------------------------------------------------------- CARD-GATE (G-T21), base, FT cells
_CG = {}


def card_gate(lin):
    """G-T21 (descriptive; it gates the use of the 2070): base eval_text_2070.json vs eval_text.json: |a_2070(n) - a_3070(n)| <= 0.005 at every n 1..24 and
    |Delta n50| <= 0.02 (sentinels must agree), item flips listed."""
    if lin in _CG:
        return _CG[lin]
    d = LIN[lin]["base"]
    if jl(d, "eval_text_2070.json") is None:
        o = {"run": False, "PASS": False, "label": "not run (the 2070 takes no work)"}
    else:
        e3 = text_read(d, expect_sha=BASE_SHA, cards_ok={CARD_CLASS}); e2 = text_read(d, "eval_text_2070.json", expect_sha=BASE_SHA, cards_ok={CARD_BACKUP})
        if e3.get("void") or e2.get("void"):
            o = {"run": True, "PASS": False, "label": "unreadable (%s / %s)" % (e3.get("void"), e2.get("void"))}
        else:
            dn = {n: abs(e2["a"][n] - e3["a"][n]) for n in range(1, 25)}; i2, i3 = _ITEMS[(R, d, "eval_text_2070.json")], _ITEMS[(R, d, "eval_text.json")]
            flips = {}
            for n in range(1, 25):
                m2 = dict(zip(i2[n]["q"], i2[n]["corr"])); m3 = dict(zip(i3[n]["q"], i3[n]["corr"])); fl = [q for q in range(1000) if m2[q] != m3[q]]
                if fl:
                    flips[n] = fl[:50]
            dn50 = (abs(e2["n50"] - e3["n50"]) if (num(e2["n50"]) and num(e3["n50"])) else (0.0 if e2["n50"] == e3["n50"] else None))
            ok = all(v <= 0.005 + EPS for v in dn.values()) and dn50 is not None and dn50 <= 0.02 + EPS
            o = {"run": True, "PASS": bool(ok), "max_abs_delta_a": max(dn.values()), "at_n": max(dn, key=dn.get), "delta_n50": dn50, "item_flips": flips,
                 "label": "PASS (the 2070 may take whole files)" if ok else "FAIL (the 3070 only)"}
    _CG[lin] = o
    return o


def cards_ok(lin):
    return {CARD_CLASS} | ({CARD_BACKUP} if card_gate(lin)["PASS"] else set())


def base_cell(lin):
    """s219_base_s1 (v1 §2.7 / G-T19): the fp32 re-eval of s170_cot_s1; bars (§4.3), Delta_0, BASE-PARITY, BF16-VS-FP32, CARD-GATE."""
    L = LIN[lin]; d = L["base"]; o = {"cell": d}
    if d in NOT_ADMITTED:
        o.update(status="UNREAD", label="UNREAD (not admitted; never a pass)"); return o
    if not ok_json(jl(d, "eval_text.json")):
        if lin not in FINAL:
            o.update(status="PENDING", missing=["%s/eval_text.json" % d], label="pending (missing ['%s/eval_text.json'])" % d); return o
        o.update(status="FAIL", censored=True, label="FAILURE (registered file missing at --final: %s/eval_text.json; rule b)" % d); return o
    ev = text_read(d, expect_sha=BASE_SHA, cards_ok={CARD_CLASS}); o["EVAL"] = {k: v for k, v in ev.items() if k not in ("a_feasible",)}
    if ev["void"]:
        o.update(status="FAIL", censored=True, label="VOID (eval_text: %s; every reading of the base fails; §5.2)" % ev["void"]); return o
    a = ev["a"]; b = r1_bars(a); LN = latent_n50()
    d0 = delta(ev["n50"], LN["r0"]) if LN else None
    bf = jl(L["base_model"], "eval.json"); bfa = {int(k): v for k, v in ((bf or {}).get("acc_by_n") or {}).items()} if ok_json(bf) else {}
    bf16 = None
    if all(n in bfa for n in range(1, 17)):
        ext = dict(bfa); ext.update({n: 0.0 for n in range(17, 25)})
        bf16 = {"diff_by_n": {n: a[n] - bfa[n] for n in range(1, 17)}, "max_abs": max(abs(a[n] - bfa[n]) for n in range(1, 17)),
                "n50_bf16": cross(ext, 0.5, 1, 16), "n50_fp32": ev["n50"], "dtype_committed": (bf or {}).get("eval_dtype"),
                "note": ("not a pure dtype contrast: batch composition also differs (the committed eval is nested170.cot_answers, batches of equal prompt length "
                         "across every n in one pass; text_eval batches equal prompt lengths within each n), so Δ mixes dtype and batch effects")}
    branch = None if d0 is None else ("levels" if abs(d0) <= 0.25 + EPS else "gain")
    note_ok = None if BASE_NOTE is None else (BASE_NOTE.get("branch") == branch)
    o.update(status="SCORED", censored=False, a=a, n50=ev["n50"], n90=ev["n90"], WIDTH_T=ev["WIDTH"],
             n50_per_support=(ev["n50"] / SUPPORT[0]) if num(ev["n50"]) else None, BARS_R1=b, M_9_12=mean(a[n] for n in range(9, 13)),
             min_9_12=min(a[n] for n in range(9, 13)), M_13_16=mean(a[n] for n in range(13, 17)), DELTA_0=d0, BASE_PARITY=branch,
             BASE_NOTE=BASE_NOTE, BASE_NOTE_MATCHES=note_ok, BF16_VS_FP32=bf16, CARD_GATE=card_gate(lin), FORM_VALID=ev["form_valid"], STRICT=ev["strict"],
             CTX_FEASIBLE={"share_n24": ev["ctx_feasible_share"][24], "a24_feasible": ev["a24_feasible"]})
    o["label"] = "base fp32 | n50 %s | n90 %s | Δ0 %s -> %s | MG_T %s SB_T %s%s" % (
        fmt(ev["n50"]), fmt(ev["n90"]), fmt(d0), "BASE-PARITY levels" if branch == "levels" else "gain statistic (|Δ0| > 0.25)" if branch else "n/a",
        fmt(b["MARGIN_GROWS"]), fmt(b["SHRINK_below"]), "" if note_ok in (None, True) else " | MISMATCH with the G-T19 note %s" % BASE_NOTE)
    return o


def latent_n50():
    return LATENT_N50_BY_CARD.get(CARD_CLASS)


REG_FT = ("log.jsonl", "args.json", "draws_log.json", "eval_text.json", "eval_textself.json")


def _sd_missing(sd, role):
    la = (sd or {}).get("label_audit") or {}; c = (sd or {}).get("contamination") or {}; m = []
    if not all(num(_la(la, k)) for k in ("pi", "pi_E")) or not all(_ci(_la(la, k)) for k in ("pi_ci", "pi_E_ci")):
        m.append("label_audit")
    if role == "t2" and not all(isinstance(_win(la.get("by_window"), w), dict) for w in WIN):
        m.append("label_audit.by_window")
    if not num(c.get("pool_excl_hits")):
        m.append("contamination.pool_excl_hits")
    return m


def ft_missing(lin, role):
    """v1 §4.6 pending rule: every registered file / CPU part of an FT cell."""
    L = LIN[lin]; d = L[role]; r = ROLE[role]; miss = {}
    for f in REG_FT:
        x = jl(d, f) if f.endswith(".json") else (True if os.path.exists(os.path.join(R, d, f)) else None)
        if x is None or (f.endswith(".json") and not ok_json(x)):
            miss[f] = "%s/%s%s" % (d, f, "" if x is None else " (unreadable)")
    if "log.jsonl" not in miss:
        li = log_info(d) or {}
        if not li.get("recs"):
            miss["log.jsonl"] = "%s/log.jsonl (no step records)" % d
        elif not (num(li.get("cursor")) and num(li.get("n_self_steps"))):
            miss["log.jsonl"] = "%s/log.jsonl (pool_cursor / n_self_steps fields)" % d
    if "eval_textself.json" not in miss:
        m = _sd_missing(jl(d, "eval_textself.json"), role)
        if m:
            miss["eval_textself.json"] = "%s/eval_textself.json (CPU parts %s)" % (d, m)
    pm = jl(L[r["pool"]], "pool_meta.json")
    if not ok_json(pm):
        miss["pool_meta.json"] = "%s/pool_meta.json" % L[r["pool"]]
    elif not isinstance(pm.get("card"), str):
        miss["pool_meta.json"] = "%s/pool_meta.json (card field)" % L[r["pool"]]
    g = gate_t1(lin) if role == "t1" else gate_t2(lin)
    if g is None:
        miss["eval_gate.json"] = "%s/eval_gate.json" % L[r["gate"]]
    elif (g["status"] == "UNDECIDED") if role == "t1" else any(g[w]["status"] == "UNDECIDED" for w in WIN):
        miss["eval_gate.json"] = "%s/eval_gate.json (gate undecided)" % L[r["gate"]]
    if not ok_json(jl(L["base"], "eval_text.json")):
        miss["base_eval_text"] = "%s/eval_text.json" % L["base"]
    if role == "t2" and not ok_json(jl(L["t1"], "eval_text.json")):
        miss["t1_eval_text"] = "%s/eval_text.json (round-2 bar input)" % L["t1"]
    if role == "t2" and g is not None and not any(g[w]["status"] == "UNDECIDED" for w in WIN) and ok_json(pm):
        _, pmiss = provenance_t2(lin, g, pm)
        if pmiss:
            miss["provenance"] = "provenance fields %s" % pmiss
    return miss


def mech_text(rnd, M, b, absorb, censored, M_far=None):
    """v1 §4.2 text mechanism label (T-SHIFT/MIXED merged: STOP-LOC is undefined for text)."""
    if censored:
        return "UNLABELLED (censored)"
    if not absorb:
        return "UNLABELLED (not T-ABSORB%s)" % ("-2" if rnd == 2 else "")
    if b is None or b.get("void"):
        return "UNLABELLED (bars void: %s)" % (None if b is None else b["void"])
    MG, SB, FULL = ((b["MARGIN_GROWS"], b["SHRINK_below"], b["FULL_M"]) if rnd == 1 else (b["MARGIN_GROWS_2"], b["SHRINK_below"], b["FULL_M2"]))
    if SB >= MG - EPS:
        return "UNLABELLED (bars degenerate)"
    if M >= FULL - EPS and (rnd == 1 or M_far >= 0.60 - EPS):
        return "T-FULL"
    if M >= MG - EPS:
        return "T-GROW"
    if M >= SB - EPS:
        return "T-SHIFT/MIXED"
    return "T-SHRINK"


def ft_cell(lin, role):
    """T1 / T2 (v1 §4.2-§4.3, §5.1-§5.2)."""
    L = LIN[lin]; d = L[role]; r = ROLE[role]; rnd = r["round"]; o = {"cell": d, "round": rnd}
    if d in NOT_ADMITTED:
        o.update(status="UNREAD", label="UNREAD (not admitted; never a pass)"); return o
    miss = ft_missing(lin, role); seal = seal_check(lin, role)
    if miss and lin not in FINAL:
        w = (seal["watch"] + seal["trainer_eval_files"])
        o.update(status="PENDING", missing=sorted(miss.values()), SEAL_WATCH=w or None,
                 label="pending (missing %s)%s" % (sorted(miss.values()), (" | SEAL_WATCH %s: an n >= 13 output of this model exists before its eval_text "
                                                                           "(rename *.void_unsealed, §5.2)" % w) if w else "")); return o
    if miss:
        o.update(status="FAIL", censored=True, missing_at_final=sorted(miss.values()), SEAL=seal,
                 label="FAILURE (registered file missing at --final: %s%s; rule b)" % (sorted(miss.values()), "; SEAL-BREACH" if (seal["BREACH"] or seal["watch"]) else ""))
        return o
    ok_cards = cards_ok(lin); fb = (BASE_SHA,) + ((((jl(L["t1"], "eval_text.json") or {}).get("meta") or {}).get("model_sha256"),) if role == "t2" else ())
    ev = text_read(d, cards_ok=ok_cards, forbid_sha=tuple(x for x in fb if x)); o["EVAL"] = {k: v for k, v in ev.items() if k not in ("a_feasible",)}
    if ev["void"]:
        o.update(status="FAIL", censored=True, VOID=ev["void"], SEAL=seal, label="VOID (eval_text: %s; every reading of the model fails; §5.2)" % ev["void"])
        return o
    info = log_info(d); ad = args_diff(lin, role); sd = jl(d, "eval_textself.json"); pm = jl(L[r["pool"]], "pool_meta.json")
    a = ev["a"]; dead = info["last"] < FT_FULL; diverged = any(a[n] < 0.99 - EPS for n in range(1, 9))
    if role == "t1":
        g = gate_t1(lin); gwin = {"W1": g}; gate_bad = not g["PASS"]; o["GATE_T1"] = {k: g[k] for k in ("status", "pi_pool", "pi_E_pool", "consistent", "per_n")}
    else:
        g = gate_t2(lin); gwin = {w: g[w]["used"] for w in WIN}; gate_bad = any(g[w]["status"] == "BOTH-FAIL" for w in WIN)
        o["GATE_T2"] = {w: {"status": g[w]["status"], "pi_pool": (g[w]["used"] or g[w]["primary"])["pi_pool"]} for w in WIN}
    la = label_audit_t(sd["label_audit"], gwin, role); noisy = not la["ok"]
    ct = contamination_t(lin, role, sd, info, pm); args_bad = not (ad and ad["ok"])
    if role == "t2":
        pc, _ = provenance_t2(lin, g, pm); ct["provenance"] = pc
        if not all(pc.values()):
            ct["CONTAMINATED"] = True; ct["provenance_bad"] = sorted(k for k, v in pc.items() if not v)
    rc = {"gate": g.get("card_class"), "pool": card_class(pm.get("card"))}
    card_bad = not (rc["gate"] in ok_cards and rc["pool"] in ok_cards and rc["gate"] == rc["pool"])
    why = [w for w, f in (("dead", dead), ("FT-DIVERGED-T", diverged), ("LABEL-NOISY-T", noisy),
                          ("CONTAMINATED%s" % (" (provenance: %s)" % ct["provenance_bad"] if ct.get("provenance_bad") else ""), ct["CONTAMINATED"]), ("ARGS-DIFF", args_bad),
                          ("SEAL-BREACH", seal["BREACH"]), ("round not on one allowed card %s" % rc, card_bad),
                          ("GATE-T1 failed" if role == "t1" else "both labellers failed a window", gate_bad)) if f]
    censored = bool(why)
    e0 = text_read(L["base"], expect_sha=BASE_SHA, cards_ok={CARD_CLASS})
    T_train = max((pm.get("T_max_by_n") or {}).values(), default=None) if isinstance(pm.get("T_max_by_n"), dict) else None
    o.update(last_step=info["last"], nonfinite_records=info["nonfinite_records"], dead=dead, FT_DIVERGED_T=diverged, LABEL_AUDIT_T=la, LABEL_NOISY_T=noisy,
             CONTAMINATION=ct, CONTAMINATED=ct["CONTAMINATED"], ARGS_DIFF=ad, ARGS_BAD=args_bad, SEAL=seal, SEAL_BREACH=seal["BREACH"], ROUND_CARDS=rc,
             CARD_BAD=card_bad, GATE_BAD=gate_bad, censored=censored, censor_reasons=why, a=a, n50=ev["n50"], n90=ev["n90"], WIDTH_T=ev["WIDTH"],
             n50_per_support=(ev["n50"] / SUPPORT[rnd]) if num(ev["n50"]) else None, T_train=T_train,
             self_value_ce=info["self_value_ce_last"], self_copy_ce=info["self_copy_ce_last"], FORM_VALID=ev["form_valid"], STRICT=ev["strict"],
             MEMO_GAP_T=memo_gap(sd), LABELLER_KEPT_T=(sd or {}).get("labeller_kept"), G_T3=g_t3(lin, role), TRAIN_DONE=train_done_check(d, info),
             GATE_DESCRIPTIVE=g.get("descriptive"))
    if rnd == 1:
        absorb = min(a[n] for n in range(9, 13)) >= 0.95 - EPS; M = mean(a[n] for n in range(13, 17))
        b = r1_bars(e0["a"]) if not e0.get("void") else {"void": "base eval_text %s" % e0.get("void")}
        mech = mech_text(1, M, b, absorb, censored)
        o.update(T_ABSORB=absorb and not censored, T_ABSORB_raw=absorb, min_9_12=min(a[n] for n in range(9, 13)), M_T=M, T_DBS=bool(M >= 0.80 - EPS) and not censored,
                 T_SELF=bool(absorb) and not censored, BARS=b, MECHANISM=mech, ROUND2_CONDITION=bool(absorb) and not censored)
        head = ("FAILURE (%s; rule b)" % ", ".join(why) if censored else "T-SELF (in support)" if absorb else "not T-SELF (T-ABSORB failed)")
        o["label"] = head + " | %s | M_T %s (T-DBS %s) | n50 %s | n50/12 %s | round-2 condition %s" % (
            mech, fmt(M), "met" if o["T_DBS"] else "not met", fmt(ev["n50"]), fmt(o["n50_per_support"]), o["ROUND2_CONDITION"])
    else:
        e1 = text_read(L["t1"], cards_ok=ok_cards, forbid_sha=(BASE_SHA,))
        absorb2 = min(a[n] for n in range(13, 17)) >= 0.95 - EPS; retain = min(a[n] for n in range(9, 13)) >= 0.99 - EPS
        M2 = mean(a[n] for n in range(17, 21)); Mfar = mean(a[n] for n in range(21, 25))
        Mfar_f = mean(ev["a_feasible"][n] for n in range(21, 25)) if all(num(ev["a_feasible"][n]) for n in range(21, 25)) else None
        b = (r2_bars(e0["a"], e1["a"]) if not (e0.get("void") or e1.get("void")) else {"void": "base / T1 eval_text %s / %s" % (e0.get("void"), e1.get("void"))})
        mech = mech_text(2, M2, b, absorb2, censored, Mfar)
        o.update(T_ABSORB_2=absorb2 and not censored, T_ABSORB_2_raw=absorb2, min_13_16=min(a[n] for n in range(13, 17)), min_9_12=min(a[n] for n in range(9, 13)),
                 T_RETAIN=retain and not censored,
                 T_RETAIN_raw=retain, M2_T=M2, T_DBS_2=bool(M2 >= 0.80 - EPS) and not censored, M_far_T=Mfar, M_far_T_feasible=Mfar_f,
                 T_SELF_2=bool(absorb2 and retain) and not censored, BARS=b, MECHANISM=mech,
                 LABELLED_BY={w: g[w]["status"] for w in WIN})
        head = ("FAILURE (%s; rule b)" % ", ".join(why) if censored else "T-SELF-2 (in support)" if o["T_SELF_2"] else
                "not T-SELF-2 (%s)" % ", ".join(x for x, f in (("T-ABSORB-2 failed", not absorb2), ("T-RETAIN failed", not retain)) if f))
        o["label"] = head + " | %s | M2_T %s (T-DBS-2 %s) | M_far %s (feasible %s) | n50 %s | n50/16 %s" % (
            mech, fmt(M2), "met" if o["T_DBS_2"] else "not met", fmt(Mfar), fmt(Mfar_f), fmt(ev["n50"]), fmt(o["n50_per_support"]))
    o["status"] = "FAIL" if censored else "SCORED"
    o["WORDING"] = ft_wording(o, g, la, role)
    return o


def provenance_t2(lin, g2, pm):
    """Round-2 provenance (Audit's G-T13 item, relayed by Theory 2026-10-18): T2's init_from.json sha256, pool_meta.windows[W].labeller_sha256 of every
    SELF-LABELLED window and eval_gate.labeller_sha256 (the primary labeller) must equal T1's eval_text meta.model_sha256; a BASE-LABELLED window's
    pool_meta labeller_sha256 and its eval_gate fallback labeller_sha256 must equal BASE_SHA. Any mismatch -> CONTAMINATED (T2 void, rule b).
    -> (checks, missing fields)."""
    L = LIN[lin]; t1sha = ((jl(L["t1"], "eval_text.json") or {}).get("meta") or {}).get("model_sha256"); ini = jl(L["t2"], "init_from.json")
    g = jl(L["gate_t2"], "eval_gate.json") or {}; W = (pm or {}).get("windows") if isinstance((pm or {}).get("windows"), dict) else {}; chk, miss = {}, []
    if not (isinstance(ini, dict) and isinstance(ini.get("sha256"), str)):
        miss.append("%s/init_from.json (sha256)" % L["t2"])
    else:
        chk["init_from_sha256 == T1 model_sha256"] = ini["sha256"] == t1sha
    if not isinstance(g.get("labeller_sha256"), str):
        miss.append("%s/eval_gate.json (labeller_sha256)" % L["gate_t2"])
    else:
        chk["eval_gate labeller_sha256 == T1 model_sha256"] = g["labeller_sha256"] == t1sha
    for w in WIN:
        st = ((g2 or {}).get(w) or {}).get("status"); x = _win(W, w)
        if st not in ("SELF-LABELLED", "BASE-LABELLED"):
            continue
        want = t1sha if st == "SELF-LABELLED" else BASE_SHA
        if not (isinstance(x, dict) and isinstance(x.get("labeller_sha256"), str)):
            miss.append("%s/pool_meta.json (windows.%s.labeller_sha256)" % (L["pool_t2"], w))
        else:
            chk["pool %s labeller_sha256 == %s" % (w, "T1" if st == "SELF-LABELLED" else "BASE_SHA")] = x["labeller_sha256"] == want
        if st == "BASE-LABELLED":
            fb = _win(g.get("fallback"), w) or {}
            chk["eval_gate fallback %s labeller_sha256 == BASE_SHA" % w] = fb.get("labeller_sha256") == BASE_SHA
    return chk, miss


def train_done_check(d, info):
    """train_done.json (the trainer's done marker, note 219-3 item 1; descriptive): present, steps == 12,000 and equal to the log's last n_self_steps /
    pool_cursor / self_reuse. dead stays the registered rule (log last step < 12,000)."""
    td = jl(d, "train_done.json")
    if not isinstance(td, dict) or "__unreadable__" in td:
        return {"present": False, "flag": "train_done.json missing (TRAIN-TOKENS from the last logged train_tokens)"}
    eq = {k: td.get(k) == (info or {}).get(v) for k, v in (("n_self_steps", "n_self_steps"), ("pool_cursor", "cursor"))}
    eq["self_reuse"] = td.get("self_reuse") == (info or {}).get("reuse")
    return {"present": True, "steps": td.get("steps"), "steps_ok": td.get("steps") == FT_FULL, "matches_log": eq, "consistent": all(eq.values()) and td.get("steps") == FT_FULL,
            "n_nonfinite": td.get("n_nonfinite"), "model_sha256": td.get("model_sha256")}


def ft_wording(o, g, la, role):
    """v1 §5.1 pre-stated per-cell sentences (only these)."""
    s = {}
    if role == "t1":
        pi = g.get("pi_pool")
        s["T_SELF"] = (("text-CoT depth extended from 8 to 12 by fine-tuning a §170 TEXT-COT GPT-2 on its own re-rooted greedy single-step chain (form-filtered, "
                        "values never checked; labels gold-equivalent at π = %s; no gold labels beyond n 8; in support; not attributed: no text FT-CTRL).")
                       % fmt(pi, 5)) if o["T_SELF"] else None
        s["T_DBS"] = ("text-CoT accuracy past the self-labelled range (n 13–16, the same untouched test items as the latent arm): mean %s (%s 0.80; a "
                      "text-arm description, never C2)" % (fmt(o["M_T"]), "≥" if o["T_DBS"] else "<")) if not o["censored"] else None
    else:
        pi = (g["W2"]["used"] or {}).get("pi_pool"); bl = [w for w in WIN if g[w]["status"] == "BASE-LABELLED"]
        nm = {"W2": "13–16", "W1": "9–12"}
        lab = ("" if not bl else " Labels for n %s from the base model s170_cot_s1 (BASE-LABELLED fallback after the round-1 model failed the window's gate)."
               % " and ".join(nm[w] for w in bl))
        s["T_SELF_2"] = (("text-CoT depth extended from 12 to 16 by fine-tuning the round-1 text model on the round-1 text model's own re-rooted greedy "
                          "single-step chain (form-filtered, values never checked; labels gold-equivalent at π = %s; no gold labels beyond n 8; in support; "
                          "not attributed: no text FT-CTRL).%s") % (fmt(pi, 5), lab)) if o["T_SELF_2"] else None
        s["T_DBS_2"] = ("text-CoT accuracy past the self-labelled range (n 17–20, the same untouched test items as the latent arm): mean %s (%s 0.80; a "
                        "text-arm description, never C2)" % (fmt(o["M2_T"]), "≥" if o["T_DBS_2"] else "<")) if not o["censored"] else None
    return s


def memo_gap(sd):
    memo = (sd or {}).get("memo") or {}
    gap = {n: memo[n]["acc_pool"] - memo[n]["acc_gate"] for n in memo if isinstance(memo[n], dict) and num(memo[n].get("acc_pool")) and num(memo[n].get("acc_gate"))}
    return {"gap": gap, "flag": [n for n, v in gap.items() if abs(v) > 0.02 + EPS]} if gap else None


_COINS = {}


def coin_stream(seed=1, steps=FT_FULL, bs=64, kextra=4, p_self=0.5):
    """the latent's rng_s replay (vbank.draw_step l.755 / l.764): coin = rng_s.random() < p_self; on a self step bs scalar integers(0, kextra + 1) (U)."""
    key = (seed, steps, bs, kextra, p_self)
    if key not in _COINS:
        import numpy as np
        rng = np.random.default_rng(seed + 21800); c, U = [], []
        for _ in range(steps):
            k = bool(rng.random() < p_self); c.append(k)
            U.append([int(rng.integers(0, kextra + 1)) for _ in range(bs)] if k else None)
        _COINS[key] = (c, U)
    return _COINS[key]


def g_t3(lin, role):
    """G-T3 re-read (descriptive here; the gate is G-T3): draws_log coins == the replayed latent coin stream at all 12,000 steps (5,992 self steps),
    plain_hash logged at steps 1..200; T1 and T2 replay the same base plain stream (plain_hash identical)."""
    x = jl(LIN[lin][role], "draws_log.json")
    if not isinstance(x, list):
        return None
    coins, _ = coin_stream(); xs = sorted((r for r in x if isinstance(r, dict) and isinstance(r.get("step"), int)), key=lambda r: r["step"])
    got = [bool(r.get("coin")) for r in xs]
    o = {"records": len(xs), "coins_match": got == coins, "n_self_steps": sum(got), "plain_hash_1_200": all(r.get("plain_hash") for r in xs[:200]) and len(xs) >= 200}
    if role == "t2":
        y = jl(LIN[lin]["t1"], "draws_log.json")
        if isinstance(y, list):
            ys = sorted((r for r in y if isinstance(r, dict) and isinstance(r.get("step"), int)), key=lambda r: r["step"])
            o["plain_hash_equals_t1"] = len(xs) == len(ys) and all(p.get("plain_hash") == q.get("plain_hash") for p, q in zip(xs, ys))
    return o


# ---------------------------------------------------------------- descriptive comparison rows (v1 §4.4; Addendum 2 item 2)
def _latent_file(r, f):
    d = LIN["s1"]["latent"][r]; x = jl(d, f)
    return x if ok_json(x) else None


def token_cost(cell_d, rnd):
    """TOKEN-COST per n (round 1: n 13-16, T1 vs FT-SELF; round 2: n 17-20, T2 vs R2-s1): sequential steps and token-positions per correct answer.
    text: sum ntok / #correct and sum (P + ntok) / #correct over all 1,000 items (overflow items: not generated, 0 positions); latent: (n + 2) * 1,000 /
    #correct and (n + 2) * sum T_item / #correct, T_item = len(syms) + 2, #correct = round(1,000 * the constant curve)."""
    per = _ITEMS.get((R, cell_d, "eval_text.json")); TI = test_info(); lr = "r%d" % rnd; out = {}
    if per is None:
        return None
    for n in (range(13, 17) if rnd == 1 else range(17, 21)):
        p = per[n]; k = sum(p["corr"]); nt = [int(v) if num(v) else 0 for v in p["ntok"]]
        pos = sum(0 if o else (P_ + t) for P_, t, o in zip(p["P"], nt, p["ovf"])); kl = int(round(1000 * LATENT_CURVES[lr][n])); sT = sum(TI["by"][n]["P"])
        out[n] = {"text_steps_per_correct": (sum(nt) / k) if k else None, "text_positions_per_correct": (pos / k) if k else None,
                  "latent_steps_per_correct": ((n + 2) * 1000 / kl) if kl else None, "latent_positions_per_correct": ((n + 2) * sT / kl) if kl else None,
                  "text_correct": k, "latent_correct": kl}
    return out


def train_tokens(lin, role):
    """TRAIN-TOKENS (Addendum 2 item 2; definitions ASSUMED, the addendum gives none): target tokens and token-positions consumed in the round, both arms.
    text: self items exact from the pool file (pool[:cursor], C = the shape's chain length, P = len(syms) + 2) when it exists, else from draws_log self_n
      x the test-set per-n means; plain items (12,000 - n_self_steps) x 64, n ~ the base law (weights excl_survive_by_n, else uniform 1..8) x the
      test-set per-n means [expected]. Targets = C (the chain incl. EOT, prompt masked); positions = P + C.
    latent: the round's draws_log (n / self_n and U per item, exact) with K = min(n + U, kmax | self_kmax) (vbank.py l.735 / l.765) and T_item ~ the
      test-set per-n mean len(syms) + 2: positions = sum K T; targets = answer targets sum (K - n + 1) + state-slot targets sum K T [expected]."""
    L = LIN[lin]; d = L[role]; rnd = ROLE[role]["round"]; TI = test_info(); MT = TI["mean"]; out = {"definition": train_tokens.__doc__.split("\n", 1)[1].strip()}
    info = log_info(d) or {}; nss = info.get("n_self_steps"); cur = info.get("cursor"); a = jl(d, "args.json") or {}
    td = jl(d, "train_done.json"); tk = src = None
    if isinstance(td, dict) and isinstance(td.get("train_tokens"), dict):
        tk, src = td["train_tokens"], "train_done.json (exact)"
    elif isinstance(info.get("train_tokens_last"), dict):
        tk, src = info["train_tokens_last"], "log.jsonl step %s (exact)" % info.get("train_tokens_step")
    if tk is not None and all(isinstance(tk.get(k), dict) and all(num(tk[k].get(f)) for f in ("target_tokens", "token_positions")) for k in ("plain", "self")):
        out["text"] = {"self_items": (nss * 64) if num(nss) else None, "self_target_tokens": tk["self"]["target_tokens"], "self_positions": tk["self"]["token_positions"],
                       "self_padded_positions": tk["self"].get("padded_positions"), "plain_target_tokens": tk["plain"]["target_tokens"],
                       "plain_positions": tk["plain"]["token_positions"], "plain_padded_positions": tk["plain"].get("padded_positions"),
                       "target_tokens": tk["self"]["target_tokens"] + tk["plain"]["target_tokens"],
                       "positions": tk["self"]["token_positions"] + tk["plain"]["token_positions"], "self_method": src}
        return _latent_tokens(out, rnd, MT)
    if not num(nss):
        return dict(out, text=None, latent=None, note="n_self_steps unavailable")
    pk = ROLE[role]["pool"]; p = os.path.join(R, L[pk], POOL_FILE[pk]); st = sp = 0; how = None
    if os.path.exists(p) and num(cur):
        N = nested(); rd = 0
        with open(p) as fh:
            for ln in fh:
                if rd >= cur:
                    break
                rd += 1; s = json.loads(ln)["syms"]; C_ = chain_len(s, N); st += C_; sp += C_ + len(s) + 2
        how = "self exact (pool file, %d records)" % rd
    else:
        x = jl(d, "draws_log.json")
        if isinstance(x, list):
            for r in x:
                for n in (r.get("self_n") or []) if isinstance(r, dict) and r.get("coin") else []:
                    st += MT[n]["C"]; sp += MT[n]["C"] + MT[n]["P"]
            how = "self expected (draws_log self_n x test-set per-n means)"
    w = a.get("excl_survive_by_n") if isinstance(a.get("excl_survive_by_n"), dict) else {str(n): 1.0 for n in range(1, 9)}
    W = sum(float(v) for v in w.values()) or 1.0; npl = (FT_FULL - nss) * 64
    pt = npl * sum(float(w[k]) / W * MT[int(k)]["C"] for k in w); pp = npl * sum(float(w[k]) / W * (MT[int(k)]["C"] + MT[int(k)]["P"]) for k in w)
    out["text"] = {"self_items": nss * 64, "self_target_tokens": st, "self_positions": sp, "self_method": "%s; plain expected (fallback: no logged counts)" % how,
                   "plain_items": npl, "plain_target_tokens": pt, "plain_positions": pp, "target_tokens": st + pt, "positions": sp + pp}
    return _latent_tokens(out, rnd, MT)


def _latent_tokens(out, rnd, MT):
    """the latent arm (its logs carry no token counts): draws_log n / U exact, T_item the test-set per-n mean [ASSUMED definition]."""
    lr = "r%d" % rnd; dl, la = _latent_file(lr, "draws_log.json"), _latent_file(lr, "args.json")
    if not isinstance(dl, list) or not isinstance(la, dict):
        out["latent"] = None; out["latent_note"] = "latent draws_log / args not under R (descriptive only)"; return out
    km, skm = la.get("kmax", 12), la.get("self_kmax", 16); pos = ans = slot = 0.0; ni = 0
    for r in dl:
        if not isinstance(r, dict):
            continue
        ns, U, cap = ((r.get("self_n"), r.get("U"), skm) if r.get("coin") else (r.get("n"), r.get("U"), km))
        for n, u in zip(ns or [], U or []):
            K = min(n + u, cap); T = MT[n]["P"]; pos += K * T; ans += K - n + 1; slot += K * T; ni += 1
    out["latent"] = {"items": ni, "positions": pos, "answer_targets": ans, "state_slot_targets": slot, "target_tokens": ans + slot, "kmax": km, "self_kmax": skm,
                     "method": "draws_log n / U exact, T_item expected (test-set per-n mean)"}
    return out


def parse_syms(syms, N):
    """the §170 minimal-bracket rendering's inverse (precedence climbing, left-associative) -> nested170.Node; render(parse(s)) == s asserted."""
    pos = [0]

    def atom():
        s = syms[pos[0]]; pos[0] += 1
        if s == " (":
            nd = expr(1); assert syms[pos[0]] == " )"; pos[0] += 1; return nd
        return N.Node(v=int(s))

    def expr(minp):
        lhs = atom()
        while pos[0] < len(syms) and syms[pos[0]].strip() in N.PREC and N.PREC[syms[pos[0]].strip()] >= minp:
            op = syms[pos[0]].strip(); pos[0] += 1; lhs = N.Node(op=op, l=lhs, r=expr(N.PREC[op] + 1))
        return lhs
    nd = expr(1); assert pos[0] == len(syms) and N.render(nd) == list(syms), "parse_syms"
    return nd


def chain_len(syms, N):
    return len(N.cot_ids(parse_syms(syms, N)))


def mcnemar_p(b, c):
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def depth_paired(cell_d, rnd):
    """DEPTH-PAIRED (v1 §4.4): per item at n 13-16 (T1 vs FT-SELF eval_selfdepth items_n2) / n 17-20 (T2 vs R2-s1 eval_deep items at K' n + 2): 2 x 2 + McNemar."""
    per = _ITEMS.get((R, cell_d, "eval_text.json"))
    if per is None:
        return None
    lat = {}
    if rnd == 1:
        sd = _latent_file("r1", "eval_selfdepth.json")
        for n in range(13, 17):
            v = ((sd or {}).get("items_n2") or {}).get(str(n))
            if isinstance(v, list) and len(v) == 1000:
                lat[n] = [int(x) for x in v]
    else:
        de = _latent_file("r2", "eval_deep.json")
        for n in range(17, 21):
            it = ((de or {}).get("items") or {}).get(str(n)) or {}; y, ans = it.get("y"), it.get("ans")
            if isinstance(y, list) and isinstance(ans, list) and len(y) == 1000:
                q = _pos(n, it.get("idx"), 13) if it.get("idx") is not None else None
                q = q or list(range(1000)); c = [int(r_[n + 2] == yy) for r_, yy in zip(ans, y)]; lat[n] = [0] * 1000
                for qq, cc in zip(q, c):
                    lat[n][qq] = cc
    if not lat:
        return {"note": "latent per-item file not under R (descriptive only)"}
    out = {}
    for n, lc in lat.items():
        t = dict(zip(per[n]["q"], per[n]["corr"])); tab = {"both": 0, "text_only": 0, "latent_only": 0, "neither": 0}
        for q in range(1000):
            tab["both" if (t[q] and lc[q]) else "text_only" if t[q] else "latent_only" if lc[q] else "neither"] += 1
        out[n] = dict(tab, mcnemar_p=mcnemar_p(tab["text_only"], tab["latent_only"]))
    return out


def fail_pos(cell_d, rnd, T_train):
    """FAIL-POS (v1 §4.4): per n (r1 13-16, r2 17-20), among failing items, median first_bad_pos / first_bad_step and the share of first deviations at
    rows > T_train,r (items with a recorded first_bad_pos)."""
    per = _ITEMS.get((R, cell_d, "eval_text.json"))
    if per is None:
        return None
    out = {}; allp = []
    for n in (range(13, 17) if rnd == 1 else range(17, 21)):
        p = per[n]; fb = [v for v, k in zip(p["fbp"], p["corr"]) if not k and num(v)]; fs = [v for v, k in zip(p["fbs"], p["corr"]) if not k and num(v)]
        allp += fb
        out[n] = {"n_fail": sum(1 - k for k in p["corr"]), "n_with_pos": len(fb), "median_first_bad_pos": _median(fb), "median_first_bad_step": _median(fs),
                  "share_past_T_train": (sum(v > T_train for v in fb) / len(fb)) if (fb and num(T_train)) else None}
    out["pooled_share_past_T_train"] = (sum(v > T_train for v in allp) / len(allp)) if (allp and num(T_train)) else None
    out["T_train"] = T_train
    return out


def _median(v):
    v = sorted(v)
    return None if not v else (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2)


def labeller_reach(lin, t1, t2, g1):
    """LABELLER-REACH (v1 §4.4, RT m2): the output format's single-step reach (per-slot vs re-emitted copy), never depth."""
    o = {"wording": "the output format's single-step reach (per-slot decode vs re-emitted copy); never a depth reading",
         "base_vs_D3T": {"text_REROOT_keep_v1_2_2": TEXT_BASE_REROOT_KEEP, "latent_chain_keep_v1_2_2": LATENT_D3T_CHAIN_KEEP,
                         "text_GATE_T1_keep": {n: v["keep"] for n, v in (g1 or {}).get("per_n", {}).items()} if g1 else None}}
    for tag, c, lr in (("T1_vs_FTSELF", t1, "r1"), ("T2_vs_R2s1", t2, "r2")):
        lk = (c or {}).get("LABELLER_KEPT_T") if isinstance(c, dict) else None; cl = _latent_file(lr, "eval_chainlabel_fresh.json")
        o[tag] = {"text_keep": {n: (v or {}).get("keep") for n, v in lk.items()} if isinstance(lk, dict) else None,
                  "latent_keep": {n: (v or {}).get("keep") for n, v in (cl.get("by_n") or {}).items()} if isinstance(cl, dict) else None}
    return o


def latent_fixed_k():
    """LATENT-FIXED-K' (RT m13b, descriptive; latent files read-only): n50 at a fixed K' - acc_k[24] for r0 / r1 (eval_vb n <= 16, eval_deep n >= 17),
    acc_k[26] for r2 (eval_deep, n 13..24) - beside the K' n + 2 constants."""
    o = {}
    for r, k in (("r0", "24"), ("r1", "24"), ("r2", "26")):
        vb, de = _latent_file(r, "eval_vb.json"), _latent_file(r, "eval_deep.json")
        try:
            if r == "r2":
                a = {int(n): v for n, v in de["acc_k"][k].items()}; o[r] = {"K": int(k), "n50": cross(a, 0.5, 13, 24), "n50_Kn2": LATENT_N50[r]}
            else:
                a = {int(n): v for n, v in vb["acc_k"][k].items()}; a.update({int(n): v for n, v in de["acc_k"][k].items() if int(n) >= 17})
                o[r] = {"K": int(k), "n50": cross(a, 0.5, 1, 24), "n50_Kn2": LATENT_N50[r]}
        except (TypeError, KeyError, ValueError):
            o[r] = {"note": "latent eval file not under R (descriptive only)"}
    return o


# ---------------------------------------------------------------- the comparison (v1 §4.4 / §5.1; Addendum 1 item 1; Addendum 2 items 2 / 4)
QUAL = ("one lineage each (latent = the surviving s1 lineage; s2 censored at R1, 216-8); GPT-2 small with learned absolute positions (1,024 rows); "
        "Arithmetic(n, 11); two rounds (CTX-BOUND: GPT-2's 1,024 rows)")
LATENT_COST = "latent cost: MEM-TRADED-2, MEM-ERODING (STAB_self %.3f → %.3f → %.3f)" % LATENT_STAB


def _tc_str(tc, rnd):
    if not tc:
        return "TOKEN-COST n %s: n/a" % ("13–16" if rnd == 1 else "17–20")
    f = lambda v: ("%.1f" % v) if num(v) else "n/a"  # noqa: E731
    return "TOKEN-COST n %s (seq. steps | token-positions per correct answer, text vs latent): %s" % (
        "13–16" if rnd == 1 else "17–20", "; ".join("%d: %s vs %s | %s vs %s" % (n, f(v["text_steps_per_correct"]), f(v["latent_steps_per_correct"]),
                                                                             f(v["text_positions_per_correct"]), f(v["latent_positions_per_correct"]))
                                                    for n, v in sorted(tc.items())))


def _tt_str(tt, rnd):
    if not tt or not tt.get("text"):
        return "TRAIN-TOKENS r%d: n/a%s" % (rnd, (" (%s)" % tt.get("note")) if tt and tt.get("note") else "")
    t, l_ = tt["text"], tt.get("latent")
    g = lambda v: ("%.3g" % v) if num(v) else "n/a"  # noqa: E731
    return ("TRAIN-TOKENS r%d (target tokens / token-positions consumed; not compute-matched): text %s / %s (%s); latent %s / %s%s" % (
        rnd, g(t["target_tokens"]), g(t["positions"]), t["self_method"], g((l_ or {}).get("target_tokens")), g((l_ or {}).get("positions")),
        "" if l_ else " (%s)" % tt.get("latent_note")))


def comparison(lin, base, g1, t1, g2, t2):
    L = LIN[lin]; LN = latent_n50(); o = {"latent_n50": LN, "card": CARD_CLASS, "rows": {k: False for k in ("PARITY", "TEXT_AHEAD", "LATENT_AHEAD", "CROSSED")},
                                         "ROUND_1_ONLY": None, "marginal": False}

    def fin(status, outcome, words, rnd_cost=2, extra=None):
        o.update(status=status, outcome=outcome)
        if extra:
            o.update(extra)
        dsc = o.get("descriptive") or {}; tc = dsc.get("TOKEN_COST_r%d" % rnd_cost) or (dsc.get("TOKEN_COST_r1") if rnd_cost == 2 else None)
        tcr = rnd_cost if dsc.get("TOKEN_COST_r%d" % rnd_cost) else 1
        why_na = "pending" if status == "PENDING" else "not read"
        tts = "; ".join(_tt_str(dsc.get("TRAIN_TOKENS_r%d" % r), r) for r in (1, 2)) if dsc else "TRAIN-TOKENS: n/a (%s)" % why_na
        stat = o.get("statistic")
        o["line"] = "%s | %s | %s | %s | %s | %s | Δ0 %s, statistic %s | CTX-BOUND (GPT-2's 1,024 learned position rows; n 24: %s feasible)" % (
            outcome if not words else words, QUAL, LATENT_COST, _tc_str(tc, tcr) if dsc else "TOKEN-COST: n/a (%s)" % why_na, tts,
            "exposure matched in items and step indices, not in training tokens or FLOPs", fmt(o.get("DELTA", {}).get(0)), stat or "n/a",
            fmt(((base or {}).get("CTX_FEASIBLE") or {}).get("share_n24")))
        return o
    o["descriptive"] = {}
    if base["status"] == "PENDING":
        return fin("PENDING", "PENDING", "pending (%s)" % base["label"])
    if base["status"] in ("UNREAD", "FAIL"):
        return fin("CENSORED", "TEXT-CENSORED", "TEXT-CENSORED (r0, %s): every comparison row fails (rule b)" % base["label"])
    if LN is None:
        if lin in FINAL:
            return fin("CENSORED", "TEXT-CENSORED", "TEXT-CENSORED (latent comparators not recorded for card %s at --final; §2.7)" % CARD_CLASS)
        return fin("PENDING", "PENDING", "pending (latent comparator n50s for card %s not recorded: §2.7 fallback, Theory note)" % CARD_CLASS)
    D = {0: base["DELTA_0"]}; stat = base["BASE_PARITY"]; o.update(DELTA=D, statistic=stat, DELTA_0=D[0], n50_T={0: base["n50"]})
    o["descriptive"].update(ROUND_0={"delta_0_fp32": D[0], "delta_0_bf16": delta((base.get("BF16_VS_FP32") or {}).get("n50_bf16"), LN["r0"])},
                            RATIO={"text": {0: base["n50_per_support"]}, "latent": {int(r[1]): LN[r] / SUPPORT[int(r[1])] for r in LN}},
                            CTX_FEASIBLE={"base": base["CTX_FEASIBLE"]}, LATENT_FIXED_K=latent_fixed_k())

    def stat_d(r):
        return D[r] if stat == "levels" else D[r] - D[0]
    if g1 is None or g1["status"] == "UNDECIDED":
        if lin in FINAL:
            return fin("CENSORED", "TEXT-CENSORED", "TEXT-CENSORED (r1, GATE-T1 eval_gate missing / undecided at --final; rule b)")
        if t1["status"] != "UNREAD":
            return fin("PENDING", "PENDING", "pending (GATE-T1 eval_gate.json missing / undecided)")
    if g1 is not None and g1["status"] == "FAIL":
        fl = {n: v for n, v in g1["per_n"].items() if not (num(v["keep"]) and v["keep"] >= 0.70 - EPS and num(v["pi"]) and v["pi"] >= 0.99 - EPS)}
        return fin("NOT_RUN", "NOT RUN", "not run (GATE-T1: keep / π per failing n %s; pooled π %s, π^E %s)" % (fl or g1["per_n"], fmt(g1["pi_pool"], 5),
                                                                                                             fmt(g1["pi_E_pool"], 5)), 1)
    if t1["status"] == "UNREAD":
        return fin("CENSORED", "TEXT-CENSORED", "TEXT-CENSORED (r1, not admitted): every comparison row fails (rule b)", 1)
    if t1["status"] == "PENDING":
        return fin("PENDING", "PENDING", "pending (T1: %s)" % t1["label"], 1)
    if t1["status"] == "FAIL":
        return fin("CENSORED", "TEXT-CENSORED", "TEXT-CENSORED (r1, %s): every comparison row fails (rule b)" % t1["label"], 1)
    D[1] = delta(t1["n50"], LN["r1"]); o["n50_T"][1] = t1["n50"]
    o["descriptive"].update(TOKEN_COST_r1=token_cost(L["t1"], 1), TRAIN_TOKENS_r1=train_tokens(lin, "t1"), DEPTH_PAIRED_r1=depth_paired(L["t1"], 1),
                            FAIL_POS_r1=fail_pos(L["t1"], 1, t1.get("T_train")))
    o["descriptive"]["RATIO"]["text"][1] = t1["n50_per_support"]; o["descriptive"]["CTX_FEASIBLE"]["t1"] = (t1.get("EVAL") or {}).get("ctx_feasible_share", {}).get(24)
    o["DELTA_gain"] = {1: D[1] - D[0]}
    if not t1["T_ABSORB_raw"]:
        return fin("SCORED", "TEXT-NOT-ABSORBED", ("TEXT-NOT-ABSORBED: text-CoT self-training did not absorb the round-1 window under the matched recipe (min a_T(9..12) "
                                                   "= %s < 0.95); no comparison is read (Δ_1 = %s, descriptive). This cannot be separated from a fault in the new "
                                                   "trainer path.") % (fmt(t1["min_9_12"]), fmt(D[1])), 1)
    r1l = round1_only(stat_d(1))
    fp1 = (o["descriptive"]["FAIL_POS_r1"] or {}).get("pooled_share_past_T_train")
    r1w = "ROUND-1-ONLY: %s (Δ_1 = %s, %s)%s" % (r1l, fmt(stat_d(1)), stat, (" | FAIL-POS: %s of the text arm's first deviations at n 13–16 sit at rows past its "
                                                                             "trained %s (GPT-2's learned absolute positions)" % (fmt(fp1), t1.get("T_train")))
                                                if r1l == "LATENT-AHEAD-1" else "")
    if g2 is None or any(g2[w]["status"] == "UNDECIDED" for w in WIN):
        if lin in FINAL:
            return fin("CENSORED", "TEXT-CENSORED", "TEXT-CENSORED (r2, GATE-T2 eval_gate missing / undecided at --final; rule b)", 1)
        return fin("PENDING", "PENDING", "pending (GATE-T2 / GATE-T1′ eval_gate.json missing / undecided; round 1: Δ_1 = %s)" % fmt(D[1]), 1)
    both = [w for w in WIN if g2[w]["status"] == "BOTH-FAIL"]
    if both:
        kp = {w: {"T1": g2[w]["primary"]["per_n"], "base": (g2[w]["fallback"] or {}).get("per_n")} for w in both}
        o["ROUND_1_ONLY"] = r1l
        return fin("SCORED", "TEXT-LABEL-STALL", ("TEXT-LABEL-STALL: %s. The text format's single-step labeller did not reach the round-2 window under the "
                                                  "latent-calibrated gate (keep / π per failing n: %s): an output-format labeller limit (the latent copies per slot; "
                                                  "text must re-emit the expression), not a depth result. The latent arm passed the same gate (GATE-2, 216-1).")
                   % (r1w, kp), 1)
    if t2["status"] == "UNREAD":
        return fin("CENSORED", "TEXT-CENSORED", "TEXT-CENSORED (r2, not admitted): every comparison row fails (rule b)", 1)
    if t2["status"] == "PENDING":
        return fin("PENDING", "PENDING", "pending (T2: %s; round 1: Δ_1 = %s)" % (t2["label"], fmt(D[1])), 1)
    if t2["status"] == "FAIL":
        return fin("CENSORED", "TEXT-CENSORED", "TEXT-CENSORED (r2, %s): every comparison row fails (rule b)" % t2["label"], 1)
    D[2] = delta(t2["n50"], LN["r2"]); o["n50_T"][2] = t2["n50"]; o["DELTA_gain"][2] = D[2] - D[0]
    o["descriptive"].update(TOKEN_COST_r2=token_cost(L["t2"], 2), TRAIN_TOKENS_r2=train_tokens(lin, "t2"), DEPTH_PAIRED_r2=depth_paired(L["t2"], 2),
                            FAIL_POS_r2=fail_pos(L["t2"], 2, t2.get("T_train")), LABELLER_REACH=labeller_reach(lin, t1, t2, g1))
    o["descriptive"]["RATIO"]["text"][2] = t2["n50_per_support"]; o["descriptive"]["CTX_FEASIBLE"]["t2"] = (t2.get("EVAL") or {}).get("ctx_feasible_share", {}).get(24)
    o["BOTH_STATISTICS"] = {"levels": four_way(D[1], D[2]), "gain": four_way(D[1] - D[0], D[2] - D[0])}
    if not t2["T_ABSORB_2_raw"]:
        o["ROUND_1_ONLY"] = r1l
        return fin("SCORED", "TEXT-NOT-ABSORBED-2", ("TEXT-NOT-ABSORBED-2: text-CoT self-training did not absorb the round-2 window under the matched recipe (min "
                                                     "a_T(13..16) = %s < 0.95); no round-2 comparison is read (Δ_2 = %s, descriptive); %s. This cannot be separated "
                                                     "from a fault in the new trainer path.") % (fmt(t2["min_13_16"]), fmt(stat_d(2)), r1w))
    lab, marg = four_way(stat_d(1), stat_d(2)); o["rows"][lab.replace("-", "_")] = True; o["marginal"] = marg
    d1, d2 = fmt(stat_d(1)), fmt(stat_d(2)); fp2 = (o["descriptive"]["FAIL_POS_r2"] or {}).get("pooled_share_past_T_train"); mg = ", marginal (within plausible protocol bias)" if marg else ""
    words = {"PARITY": "PARITY: on Arithmetic(n, 11), round-matched text-CoT self-training and latent self-training moved the 50 %% depth by the same amount "
                       "(|Δn50| ≤ 0.5 at rounds 1 and 2: Δ = %s / %s)." % (d1, d2),
             "LATENT-AHEAD": "LATENT-AHEAD: latent self-training moved the 50 %% depth further than round-matched text-CoT self-training (Δn50 = %s / %s)%s | "
                             "FAIL-POS: %s of the text arm's first deviations at n 17–20 sit at rows past its trained %s (GPT-2's learned absolute positions)"
                             % (d1, d2, mg, fmt(fp2), t2.get("T_train")),
             "TEXT-AHEAD": "TEXT-AHEAD: round-matched text-CoT self-training moved the 50 %% depth further than latent self-training (Δn50 = %s / %s)%s"
                           % (d1, d2, mg),
             "CROSSED": "CROSSED (%s): %s (Δ = %s / %s)." % (crossed_subcase(stat_d(1), stat_d(2)), {"order changed": "the arms' order changed between rounds", "converged at round 2": "a round-1 lead closed into the parity band at round 2"}.get(crossed_subcase(stat_d(1), stat_d(2)), "see crossed_subcase"), d1, d2)}[lab]
    return fin("SCORED", lab + (" (marginal)" if marg else ""), words + (" | T-RETAIN failed (min a_T(9..12) = %s)" % fmt(t2["min_9_12"])
                                                                         if not t2["T_RETAIN_raw"] else ""))


# ---------------------------------------------------------------- driver
def lineage(lin):
    _CG.clear()
    base = base_cell(lin); g1 = gate_t1(lin); t1 = ft_cell(lin, "t1"); g2 = gate_t2(lin); t2 = ft_cell(lin, "t2")
    if t1.get("status") == "SCORED" and not t1.get("ROUND2_CONDITION") and t2.get("status") not in ("PENDING", "UNREAD"):
        t2["note"] = "T2 present although the round-2 condition failed: not read by the comparison"
    if t1.get("status") == "SCORED" and not t1.get("ROUND2_CONDITION"):
        t2 = dict(t2, status="NOT_RUN" if t2.get("status") == "PENDING" else t2.get("status"),
                  label=("not run (round-2 condition not met)" if t2.get("status") == "PENDING" else t2.get("label")))
    cmp_ = comparison(lin, base, g1, t1, g2, t2)
    return {"base": base, "card_gate": card_gate(lin), "gate_t1": g1, "t1": t1, "gate_t2": g2, "t2": t2, "comparison": cmp_}


def score(final=(), not_admitted=()):
    FINAL.clear(); FINAL.update(final); NOT_ADMITTED.clear(); NOT_ADMITTED.update(not_admitted); _ITEMS.clear(); _TR.clear()
    out = {"registered": REGISTERED, "R": R, "CARD_CLASS": CARD_CLASS, "final": sorted(FINAL), "not_admitted": sorted(NOT_ADMITTED),
           "LATENT": {"n50": LATENT_N50, "sources": LATENT_SRC, "rule": LATENT_RULE, "selfcheck": _LSC, "by_card": LATENT_N50_BY_CARD},
           "META_T": sorted(META_T), "TEXT_NEUTRAL_NEW": TEXT_NEUTRAL_NEW, "layouts_assumed": LAYOUTS}
    out["s1"] = lineage("s1")
    return out


def reading(out):
    s = out["s1"]
    return "READING: base %s || T1 %s || T2 %s || comparison %s" % (s["base"]["label"], s["t1"]["label"], s["t2"]["label"], s["comparison"]["line"])


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="append", default=[], choices=["s1"])
    ap.add_argument("--not_admitted", default=""); a = ap.parse_args(argv)
    out = score(a.final, [x for x in a.not_admitted.split(",") if x])
    print(reading(out).encode("ascii", "backslashreplace").decode("ascii"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
