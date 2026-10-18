"""THEORY §218 scoring (LANG-VB: VERSION-BANK / SLVA on language; registered text docs/research_lang_2026-10-18/LANG_v1.md at 83f2dea4 = v1 +
Addendum 1; THEORY.md §218; Theory is the only scorer). Cells under R (boxes/runs_fsq): DIRECT-SYM lvb_direct_s1, A1 lvb_sym_s1, A2 lvb_self_s1,
A2-CTRL lvb_ctrl_s1 (Stage A, ProsQA-SYM v1, eval card CARD['A']), B1 lvb_gsm_s1 (Stage B, GSM8K strict DECOMP, CARD['B']). NOT here: A0-1 (= THEORY
note 191-4, scored separately, v1 §2.8); A-TXT (optional, no registered reading in this scorer); G-L* gates other than the inputs named below.
Pending (215-7; v1 §4.3): a cell reads "pending (missing [...])" while any registered file / part / field is missing and reads nothing else first; pending
  never passes. --final <cell>: a still-missing registered file / decisive part fails by rule b (FAILURE); a card / dtype / eval_bs / items_sha mismatch reads
  "pending (mismatch ...)" before --final and VOID (rule b) at --final (SD2 Addendum-1 form, v1 §4 card fallback). Cells never launched (not admitted, POOL-SHORT,
  A2 / B not GO) read UNREAD = not met, never pending, never a pass.
Registered files (v1 §4.3): DIRECT log.jsonl / args.json / eval_direct.json (+ the G-L0 gate JSON, G-EVMEM); A1 log / args / eval_lvb.json (every part) / G-L0 /
  DIRECT's eval_direct.json (C1 band; A1 also waits for DIRECT's cell to be decided) / G-EVMEM; A2 = A1's list + pool_meta.json, eval_gate1.json, the labeller
  part (label_audit), A2-CTRL's eval_lvb.json; A2-CTRL log / args / eval_lvb.json (acc, hist24); B1 log / args / eval_lvb_gsm.json / interim GO JSON / B0-2's
  CODI-20 re-score (+ G-EVMEM-GSM; the reference boxes/runs_fsq/gsm_codi_20ep_s1/eval_items.json).
Censors (v1 §5.2, rule b; each fails every reading of the cell): dead (last logged step < FULL, a guard stop counts); NO-ENTRY (ProsQA plain_cand_ce never <= 0.35
  in two consecutive 1k bins; GSM8K valid_acc <= 0.15 at every check); CONTAMINATED (pool / test / GATE signature hit, a gold key in a pool record, self_reuse > 0);
  ARGS-DIFF beyond the pre-listed keys; LABEL-BUG (A2: any wrong label in GATE-1-L or LABEL-AUDIT-L -> VOID, re-gate); G-L20 failure other than POOL-SHORT.
UNREADABLE (= not met): < 100 exposed clean pairs in a decisive arm (incl. CF-L / CF-GSM); < 2,000 STALL items (ProsQA); PROBE-INVALID (PLANT or final state);
  FD-INVALID; FD(k, k) < 0.3 (GSM8K); a G-L0-failed band inside a decisive reading's range; an empty C1 band (< 2 depths); a per-item consistency assert failing
  in the reading's part. Per-D values with < 50 exposed clean pairs (CF: (D, j) cells with < 30) are excluded and listed, never passed.
Stage A readings (v1 §4.1; scorer recomputes every count and rate from the per-item / per-pair stores; file summaries must agree or the part is VOID):
  C1 band (DIRECT) = {D in 5..8 : acc_DIRECT(D') <= 0.65 for every D' in D..8}; D*_fresh = max D in 3..12 with acc > 0.65 (descriptive).
  DEEP-L acc_d2 >= 0.90 at every D 4-8. PROBE-GAIN-L: pooled over the C1 band (>= 2 depths) ACC(K' D+2) - PROBE(1) >= 0.35, PROBE-VALID(1) (PLANT lin and XOR
  >= 0.90) and PROBE(12) >= ACC - 0.05 on the band. SERIAL-ONSET-L: ON(D) = min grid t with PROBE(t') >= 0.90 for all grid t' >= t (never = failure); OLS slope
  (D 4-8) >= 0.6 and ON(8) - ON(4) >= 3; PROBE-VALID(0..2) and PROBE(12) >= ACC - 0.05 pooled D 4-8. SERIAL-STEP-L: (ii') follow >= 0.80 pooled D 4-8 and >= 0.60
  at every readable D, and (i') follow >= 0.80 pooled D 5-8 and >= 0.60 at every readable D (D 4 excluded by construction, listed); >= 100 exposed clean pairs per
  arm. NO-CLOCK-L STALL recovery >= 0.9 (>= 2,000 items). NO-PRECOMPUTE-L FD-VALID-L and AUROC <= 0.65 at every (t 1-2, k >= 3) cell, linear and MLP, pooled D 6-8.
  CF-L: decisive cells (D 4-8, j 3..D-1), share of exposed clean pairs with t* in {j-1, j} >= 0.75 and pair-weighted within-D Spearman(t*, j) over D 5-8 >= 0.60;
  t* = least-violations change point over t 0..D (ties -> smallest; v1_cf.py crossover verbatim). VB-C1-L = the seven, uncensored.
  Labels: ONESHOT-L, LOSES-ONE-PASS, COUNT-KEYED, RECOMPUTE-L, ANCHOR-TAGGED, HISTORY-READ, ALL-PERSONS signature, PRECOMPUTE-L, EMPTY-C1-BAND, CF-CARRY,
  CF-RECOMPUTE, CF-FAST, LAST-HOP-LATE (all descriptive). MEM: KEEPS-SELF-STAB-L (SELFDEPTH_v1 §4.3 form on D 3..6) else MEM-TRADED-L -> every later stage needs
  the user's explicit OK (USER_OK, set only from a recorded note).
  A2: ABSORB-L, LENGTH-L, K'24-STABLE, SD-SELF-L (= those + VB-C1-L re-read on A2), DEPTH-BEYOND-SELF-L, MARGIN-GROWS-L (bar from A1's file), BEATS-CTRL-L,
  SD-C2-L (A2-CTRL missing or censored -> not met). Beside: CAUSAL-DEEP-L (§217 rule), MEM, FALSE-IN, STOP-LAG, MEMO-GAP-L, kept-vs-drawn, LIT-C2-L UNINFORMATIVE.
Stage B (v1 §4.2 + Addendum 1): band m >= 3. PROBE-GAIN-GSM, SERIAL-STEP-GSM ((ii') and (i): base-item-clustered follow >= 0.6 and own <= 0.3, >= 100 exposed
  clean pairs and >= 50 distinct base items with >= 1 clean pair), NO-CLOCK-GSM, NOT-ONE-PASS-CHAIN (lin and MLP; FD(k, k) >= 0.3 else UNREADABLE), CF-GSM
  (clustered share in {j-1, j} >= 0.6 and Spearman >= 0.6 on (base, j) cluster means, >= 100 clean pairs, >= 50 base items); cluster-bootstrap 95 % CIs
  (BOOT_SEED fixed). BEATS-CODI20 = delta >= 0.03 and exact one-sided McNemar p <= 0.05 vs B0-2's CODI-20 re-score (== gsm_codi_20ep_s1/eval_items.json per item).
Stage flow (v1 §2.9): A2 GO = A1 uncensored and DEEP-L and NO-CLOCK-L and GATE-1-L (and the user's OK if A1 reads MEM-TRADED-L); A2 and A2-CTRL only together;
  B GO = A1 VB-C1-L (and the user's OK if any scored LANG cell reads MEM-TRADED-L); B1 interim GO at 24k (recomputed), else B1-24k.
    python -B theory/score_s218.py [--final a1] [--final a2] ... [--not_admitted a2,ctrl] [--user_ok A2,B1]
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
from fractions import Fraction

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = _HR if os.path.isdir(os.path.join(_HR, "boxes", "runs_fsq")) else "D:/Latent"    # a scratch copy reads (never writes) the repo
R = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(ROOT, "theory", "results", "score_s218.json")
for _p in (ROOT, os.path.join(ROOT, "theory")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EPS = 1e-9
REGISTERED = "docs/research_lang_2026-10-18/LANG_v1.md @ 83f2dea4 (v1 + Addendum 1)"
CELL = {"direct": "lvb_direct_s1", "a1": "lvb_sym_s1", "a2": "lvb_self_s1", "ctrl": "lvb_ctrl_s1", "b1": "lvb_gsm_s1"}
NAME = {"direct": "DIRECT-SYM", "a1": "A1", "a2": "A2", "ctrl": "A2-CTRL", "b1": "B1"}
STAGE = {"direct": "A", "a1": "A", "a2": "A", "ctrl": "A", "b1": "B"}
CARD = {"A": "4060", "B": "4060"}          # v1 §4 / N10. Whole-stage fallback (v1 §4): Theory sets CARD[stage] = "3070" in a note; never a partial move
DTYPE = "fp32"
LABEL_DTYPE = "bf16"                        # GATE-1-L and the pool run under bf16 autocast (v1 §2.4)
PATHS = {   # [ASSUMED] locations of the non-cell registered inputs (v1 does not name them); "@ROOT" = relative to ROOT, else to R. Readings never change
    "gate_l0": ("@ROOT", "scratch/gate_s218/g_l0.json"),                  # Experiments' G-L0 output (exists 2026-10-18 14:42; panel layout, read as built)
    "evmem_A": ("lvb_gates", "gate_evmem.json"), "evmem_B": ("lvb_gates", "gate_evmem_gsm.json"),
    "gate1": ("lvb_gate1_s1", "eval_gate1.json"), "pool": ("lvb_pool_s1", "pool_meta.json"), "b02": ("lvb_b02", "codi20_rescore.json"),
    "codi20": ("gsm_codi_20ep_s1", "eval_items.json"), "interim": ("lvb_gsm_s1", "interim_go.json")}
FULL = {"direct": 24000, "a1": 24000, "a2": 12000, "ctrl": 12000, "b1": 120000}
B1_INTERIM_STEP = 24000
CODI20_ACC, CODI4_ACC = 0.3791, 0.189
PER_D = 1000                                # SYM v1 test: 1,000 per D, D 3..16 (v1 §2.2 item 10)
N_GSM = 1319
D_TEST = tuple(range(3, 17))
K_GRID = tuple(range(0, 25))
PROBE_GRID = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12)
GSM_PROBE_GRID = (0, 1, 2, 3, 4, 6, 8, 12, 14)
L0_STREAMS = {"train": range(3, 9), "pool": range(9, 13), "gate": range(9, 13), "test": range(3, 17)}
L0_GATED = ("first", "random", "maxdeg", "la1")
POOL_KEYS = {"text_ids", "slot_idx", "stmts_parsed", "marks_sound", "stop_s", "label"}
POOL_KEPT, POOL_CAP, POOL_DEMAND = 106000, 530000, 383488
BOOT_SEED, BOOT_REPS = 218, 2000
GSM_MIN_BASE, CF_GSM_MIN_BASE = 50, 30      # Addendum 1 item 6; CF-GSM >= 30 distinct exposed base steps per Audit's R13 ruling (note 218-5)
NOT_ADMITTED, FINAL, USER_OK = set(), set(), set()
# ARGS-DIFF (v1 §4.3): reference cell / file and the keys allowed to differ; None = not pre-listed in v1 -> the cell reads pending until Theory sets it
ARGS_REF = {"direct": None, "a1": ("s213_d3t_s1", "args.json"), "a2": (CELL["a1"], "args.json"), "ctrl": (CELL["a2"], "args.json"), "b1": None}
ARGS_ALLOW = {"direct": None, "a1": None, "b1": None,
              "a2": {"init_from", "self_pool", "p_self", "self_kmax", "self_micro", "lr", "steps", "c_sched"},
              "ctrl": {"p_self", "self_pool"}}
ARGS_MUST = {"a2": {"init_from": "boxes/runs_fsq/%s/model.pt" % CELL["a1"], "p_self": 0.5, "lr": 3e-5, "steps": 12000, "self_kmax": 16, "seed": 1},
             "ctrl": {"p_self": 0.0, "self_pool": None}, "a1": {"seed": 1, "steps": 24000}, "direct": {"seed": 1, "steps": 24000},
             "b1": {"seed": 1, "steps": 120000}}   # recipe values of v1 §2.3 / §2.4 / §2.6 / §2.7 (RESOLVED R26)
META = {"mode", "arch", "out", "device", "train_gpu", "train_driver", "torch_version", "cuda_version", "cudnn_version", "python_version", "image_tag",
        "eval_bs", "micro", "auto_micro", "mem_budget", "ckpt_every", "keep_ckpt_steps", "time_steps", "mem_check", "stop_at", "resume"}   # = score_s213.META
PARTS_A = {"panel": ("panel",), "acc": ("acc_d2", "acc_k", "items"), "probe": ("probe",), "plant": ("plant", "probe_valid"),
           "fd": ("fd", "fd_valid", "fd_plant"), "vswap": ("vswap",), "stall": ("stall",), "cf": ("cf",), "hist24": ("hist24",),
           "causal_deep": ("causal_deep",), "labeller": ("gate1", "kept_vs_drawn", "label_audit")}
DECISIVE_A = ("panel", "acc", "probe", "plant", "fd", "vswap", "stall", "cf")
DESCRIPTIVE_A = ("hist24", "causal_deep")   # hist24 feeds MEM (routes the user's OK, not a censor); causal_deep is beside A2 positives
REQ_PARTS = {"a1": DECISIVE_A + DESCRIPTIVE_A, "a2": DECISIVE_A + DESCRIPTIVE_A + ("labeller",), "ctrl": ("acc", "hist24")}
NO_BS_PARTS = {"panel", "labeller", "codi_pair"}   # copies of gate outputs / a paired comparison: no eval_bs / items_sha of their own
PARTS_B = {"acc": ("acc_fixed14", "items", "by_m", "by_mbin", "acc_k", "acc_oracle", "acc_halt"), "codi_pair": ("codi_pair",),
           "probe": ("probe", "probe_valid_gsm", "plant_gsm"), "future": ("future",), "vswap": ("vswap",), "stall": ("stall",),
           "stepcap": ("stepcap",), "gsmswap": ("gsmswap",), "cf_gsm": ("cf_gsm",)}
DECISIVE_B = ("acc", "probe", "future", "vswap", "stall", "cf_gsm", "codi_pair")
DESCRIPTIVE_B = ("stepcap", "gsmswap")
HIST_ARMS = ("intact", "bank_ablate_slva", "self_ablate", "dup_newest")
VSWAP_RULE = {   # v1 §2.5 / §4.1 exposure table: admissible (D, t) and the exposure conditions recomputed per pair
    "ii_prime": {"D": range(4, 9), "t": lambda D: range(3, D), "need": ("pos_ok", "frontier_differs"), "donor": True},
    "i_prime": {"D": range(5, 9), "t": lambda D: range(3, D - 1), "need": ("pos_ok", "lt_differs", "opts_outside"), "donor": True},
    "ii": {"D": range(4, 9), "t": lambda D: range(3, D), "need": ("pos_ok", "frontier_differs"), "donor": True},
    "person": {"D": range(4, 9), "t": lambda D: range(3, D), "need": ("pos_ok",), "donor": True},
    "reset": {"D": range(4, 9), "t": lambda D: range(3, D), "need": (), "donor": False}}

LAYOUTS = {   # field layouts the scorer reads from files not yet built (v1 §4.1 / §4.2 names kept; [ASSUMED] where v1 is silent). Experiments: match these names.
    "every eval file": "top-level meta {card (str containing the card class, e.g. 'NVIDIA GeForce RTX 4060'), dtype 'fp32' ('bf16' for GATE-1-L / labeller), "
                       "eval_bs {part: int} (the G-EVMEM value per part), items_sha (str, or {part: str}) = the SYM test set's sha (G-L0 test_items_sha) / the "
                       "GSM8K test list's sha (B0-2 items_sha), card_by_part {part: card} optional}; the same fields at the file's top level (+ test_sha {D: sha}) are also read "
                       "(langvb_eval layout; RESOLVED R31)",
    "scratch/gate_s218/g_l0.json (G-L0; PATHS)": "the built panel layout (panel {D: {*_by_stream}}, full_rows) or streams {train, pool, gate, test: {D: {n_gate (>= 5000), invariant_failures (0), max_person_agnostic, chase {first, random, "
                                     "maxdeg, la1, la2, la3, la4}, soft, ubidir_max, udist, beyond_D, gate_pass}}} (train D 3-8, pool / gate D 9-12, test D 3-16); "
                                     "test_items_sha (str)",
    "lvb_gates/gate_evmem.json / gate_evmem_gsm.json (G-EVMEM)": "card, eval_bs {part: int} for every ProsQA part (+ 'direct') / every GSM8K part",
    "log.jsonl": "one JSON record per line: step (int); plain_cand_ce (ProsQA, per logged step); valid_acc (GSM8K Aug-valid checks); self_reuse; guard_stop "
                 "(true on a guard stop; counts as dead). Last occurrence of a step wins (resumes)",
    "eval_direct.json": "acc_by_D {D 3..16: float}, items {D: [0/1] x 1000}, c1_band [D...], d_star_fresh (int|null), meta (eval_bs part 'direct')",
    "eval_lvb.json acc": "acc_d2 {D 3..16}, acc_k {K' 0..24: {D}}, items {D: {K': [0/1] x 1000}} (acc_k[K'][D] == mean(items[D][K']); acc_d2[D] == acc_k[D+2][D]), "
                         "cand_ce_items, T_band (descriptive)",
    "eval_lvb.json panel": "panel {D: {..., gate_pass}} copied from G-L0 (checked against the gate JSON's verdict; flag only)",
    "eval_lvb.json probe": "probe {t in PROBE_GRID: {acc_by_D {D}, items {D: [0/1]} (D 3..8 at least), curve [[step, heldout_acc], ...] every 500, steps_run "
                           "(10000 | 20000)}}",
    "eval_lvb.json plant": "plant {t 0..2: {lin {acc, items [0/1]}, xor {acc, items}, eps, eps_rel (Addendum 1 item 5: eps / ||V_t||), pcs}}; probe_valid {t: bool}",
    "eval_lvb.json fd": "fd {lin|mlp: {t 1..2: {k 1..8-t: {auroc, n_pos, n_neg, pos [scores], neg [scores]}}}} pooled D 6-8 (pos / neg required at k >= 3); "
                        "fd_valid {s 3|4: auroc} or {lin|mlp: {s: auroc}}; fd_plant {auroc} or {lin|mlp: {auroc}}",
    "eval_lvb.json vswap": "vswap {ii_prime|i_prime|ii|person|reset: {n_candidates (= len(pairs)), n_exposed, n_clean, follow, own (reset: own = recovery), "
                           "by_D {D: {n_exposed, n_clean, follow, own}}, by_t, t2_descriptive, pairs [{rec, don, D, t, y_hyb, y_rec, y_don, ok_rec, ok_don, "
                           "pos_ok, frontier_differs (ii_prime, ii), lt_differs, opts_outside (i_prime)}]}}; pairs = every candidate (exposure is recomputed); "
                           "y_hyb may instead be 'follow' / 'own' / 'neither' (then y_rec / y_don unused); summary counts and rates are over the decisive "
                           "(D, t) window only, t = 2 (ii_prime) in t2_descriptive",
    "eval_lvb.json stall": "stall {recovery, by_D, items {D 4..8: {stalled [0/1], intact [0/1]}}} (intact defaults to items[D][str(D)] of the acc part)",
    "eval_lvb.json cf": "cf {D: {j: {n_candidates, n_exposed, n_clean, curve {t: P(answer = y')}, tstar [clean pairs], share_window, share_clean_kd, "
                        "share_clean_kd2}}, spearman_within_D {D}, share_window_pooled, last_hop {D: {tstar_hist, curve}}, pairs [{rec, don, D, j, "
                        "answers_by_t [1 if the answer at t equals y (recipient's o_R) else 0, t = 0..D], ok_rec, ok_don (K' = D), ok_rec_d2, ok_don_d2 "
                        "(K' = D + 2, Addendum 1 item 4), pos_ok, flip_ok}]}",
    "eval_lvb.json hist24": "hist24 {intact|bank_ablate_slva|self_ablate|dup_newest: {acc_d2 {D 3..8}, acc_k24 {D 3..8}}, items {arm: {D: {ok_d2 [0/1], ok_k24 [0/1]}}}}",
    "eval_lvb.json causal_deep": "causal_deep {ii_prime|i_prime: {by_D {D 9..16: {n_clean, follow, own}}}, stall {by_D {D: {n, recovery}}}} (+ per-pair stores)",
    "eval_lvb.json labeller (A2)": "gate1 (copy of eval_gate1.json's choice), kept_vs_drawn {D: {T, concepts, cross, decoys, frontier}}, label_audit {n (5,000), "
                                   "n_wrong, pi_label}, memo {D 9..12: {acc_pool, acc_fresh}} (MEMO-GAP-L, descriptive)",
    "lvb_gate1_s1/eval_gate1.json": "meta {card, dtype 'bf16'}, choice ('L-ROLL'|'L-RR'), chains {L-ROLL|L-RR: {D 9..12: {n, kept, yield, omit_rate, false_in, "
                                    "stop_lag, frontier_hist}}}, items {L-ROLL|L-RR: {D: [{keep (bool), label, gold, stop}]}}",
    "lvb_pool_s1/pool_meta.json": "kept_by_D {9..12}, draws_by_D {9..12}, total_kept, demand_replay (383488 at seed 1), rng_replay_equal (bool), key_set [..], "
                                  "sig_hits {test, gate, probe} (ints), gold_key_hits (int)",
    "eval_lvb_gsm.json acc": "acc_fixed14, items [{i, ok, pred_txt, m, m_bin}] x 1319, by_m, by_mbin, acc_k {K'}, acc_oracle, acc_halt",
    "eval_lvb_gsm.json probe": "probe {t in GSM_PROBE_GRID: {acc_by_m {m: acc}, items [0/1] x 1319 (test order), curve}}, probe_valid_gsm (bool), "
                               "plant_gsm {lin: {acc}, xor: {acc}}",
    "eval_lvb_gsm.json future": "future {lin|mlp: {t 0..12: {k >= t: {acc, n}}}} (incl. t = k)",
    "eval_lvb_gsm.json vswap": "vswap {ii_prime|i|ii|iv: {n_candidates, n_exposed, n_clean, follow, own, neither (pair-level), by_m, pairs [{rec, don, base (test "
                               "index), variant (0 = base), step, m_bin, y_rec, y_don, y_hyb (numbers; |a - b| < 1e-4), ok_rec, ok_don, pos_ok}]}}",
    "eval_lvb_gsm.json stall / stepcap": "stall {recovery, by_m, items [{i, stalled (0/1), intact (0/1)}]}; stepcap {by_t {t: {acc, n}}, by_op, by_digits, pooled}",
    "eval_lvb_gsm.json cf_gsm": "cf_gsm {j: {n_candidates, n_exposed, n_clean, curve, tstar}, share_window, spearman, last_step, late_fusion_median_tstar, pairs [{rec, "
                                "don, base, variant, j, m_bin, answers_by_t [1 iff answer == y, t = 0..14], ok_rec, ok_don, pos_ok, flip_ok}]}",
    "eval_lvb_gsm.json codi_pair": "codi_pair {delta, se_paired, mcnemar {b, c, p_one_sided}, n, by_m, by_parse} (b = B1 right & CODI wrong)",
    "lvb_b02/codi20_rescore.json (B0-2)": "meta {card, dtype}, items_sha, acc, items [{i, pred, gold, ok}] x 1319 (== gsm_codi_20ep_s1/eval_items.json ok per item)",
    "lvb_gsm_s1/interim_go.json (B0-6)": "meta {card, dtype}, step (24000), acc_fixed14_valid, probe1_valid, fd_mlp_1k_ge3, fd_mlp_kk_ge3, GO (bool)"}

RESOLVED = [   # ambiguities in the registered text and how this scorer reads them (each for Theory to confirm before any output)
    "R1 ARGS-DIFF: v1 names no pre-listed new keys for A1 vs D3-T, no reference for DIRECT-SYM ('its own reference for the plain path') and none for B1: "
    "ARGS_ALLOW / ARGS_REF are None there and the cell reads pending until Theory sets them (FAILURE at --final).",
    "R2 G-L0 band rule: a band D is usable iff in every stream covering D: n_gate >= 5000, 0 invariant failures, chase first / random / maxdeg / la1 < 0.6, "
    "max_person_agnostic < 0.6, ubidir_max == 0.5 and beyond_D == 0 (the last two failing = CODE-BUG, band unusable). A decisive reading whose D range holds a "
    "failed band is UNREADABLE; DIRECT's C1 band is UNREADABLE if any of D 5-8 failed; descriptive readings exclude and list the band.",
    "R3 PROBE(12) final-state validity is read on each reading's own pooling: the C1 band for PROBE-GAIN-L, D 4-8 for SERIAL-ONSET-L.",
    "R4 SERIAL-STEP-L pooled bars include the pairs of D values excluded from the per-D bar (< 50 clean pairs); CF-L's excluded (D, j) cells (< 30) leave both the "
    "share and the Spearman.",
    "R5 CF exposure = pos_ok and flip_ok; clean = ok_rec and ok_don at K' = D (fields, else derived from answers_by_t endpoints; endpoint mismatches are reported, "
    "not voiding). Within-D Spearman uses D with >= 2 distinct j after exclusion (D 4 has j = 3 only), weighted by clean pairs (v1_cf.py).",
    "R6 CF labels: CF-CARRY = median t* == 1; CF-RECOMPUTE = median (D - t*) == 0; CF-FAST = median (j - t*) >= 2, over the used decisive pairs. LAST-HOP-LATE = "
    "CF-L met and median (D - t*) == 0 over clean j = D pairs (pure serial also reads it: v1 PS-6 says j = D cannot separate serial from recompute).",
    "R7 RESET recovery = own rate (answer == y_rec) over clean exposed RESET pairs (no donor).",
    "R8 KEEPS-SELF-STAB-L uses SELFDEPTH_v1 §4.3 in full (intact == acc part bitwise, CONTENT-STABILISES: STAB_bank >= 0.10 and STAB_dup_newest >= 0.10, "
    "STAB_self >= 0.5; mean over D 3..6), on A1 / A2 / A2-CTRL (DIRECT-SYM has no bank); unreadable or missing at --final = MEM-TRADED-L.",
    "R9 A2-CTRL registered parts = acc + hist24 (its only readings: BEATS-CTRL input, ABSORB vs CTRL, MEM). At --final a2 a missing / censored / pending A2-CTRL "
    "fails BEATS-CTRL-L and SD-C2-L only (v1 §4.1 'A2-CTRL missing or censored -> SD-C2-L not met'), not every A2 reading.",
    "R10 LABEL-BUG counts a wrong label under either GATE-1-L chain (S-L' holds for any labeller) or in LABEL-AUDIT-L; A2 launched -> VOID (rule b); not launched -> "
    "A2 GO false (UNREAD).",
    "R11 No 'diverged' censor exists in v1 §5.2 for LANG cells (THEORY §218's header lists one); none is applied.",
    "R12 NO-ENTRY is applied to DIRECT-SYM too (it logs plain_cand_ce); a censored DIRECT-SYM makes the C1 band UNREADABLE (VB-C1-L not met).",
    "R13 CF-GSM (RULED, note 218-5, Audit): Addendum 1 item 6's cluster RULE applies to CF-GSM (base-step-clustered share and rho, cluster bootstrap); "
    "its minimum is >= 30 distinct exposed base steps with >= 1 clean pair (37 exist), fewer = UNREADABLE; (ii') / (i) keep >= 50 base items.",
    "R14 CF-GSM clustering: share = mean over base items of the per-item in-window share; Spearman on (base, j) cluster means of t* (pooled over j, not within "
    "m_bin); pair-level values reported beside.",
    "R15 The ProsQA '>= 2,000 STALL items' minimum is not applied to GSM8K (879 items at m >= 3 could never reach it).",
    "R16 ONE-PASS-CHAIN-L = some readable FD probe (lin or MLP) with FD(1, k>=3) >= 0.8 FD(k, k) pooled; LAST-STEP-LATE (GSM, descriptive) = median t* >= 12 over "
    "clean j = m_bin pairs (the LATE-FUSION-GSM threshold).",
    "R17 CAUSAL-DEEP-L applies §217's rule to both ii_prime and i_prime (follow >= 0.95, own <= 0.02 pooled over B, every D in B follow >= 0.90) plus STALL >= 0.95; "
    "B = {D 9..16 : acc_d2 >= 0.95}; D with < 100 clean pairs in either arm or < 500 STALL items excluded; empty B = NOT-APPLICABLE. Pooled STALL = n-weighted.",
    "R18 B GO uses the MEM status of every Stage A cell already scored (A1 always; A2 / A2-CTRL when not pending / UNREAD).",
    "R19 GSM vswap file follow / own are compared with the pair-level recompute (v1 §4.2 predates Addendum 1); decisions read the clustered means.",
    "R20 VB-C1-L wording: v1 §5.1's CF clause is replaced by Addendum 1 item 2's fixed wording ('hop-indexed consumption on >= 0.75 of clean exposed pairs "
    "(calibrated: requires ~0.9 item-level serial share)'); Stage B wording 'C4 via distillation from CoT, with CoT-parsed pointer annotations' (Addendum 1 item 1).",
    "R21 RNN-IN-FUNCTION / HISTORY-USED (v1 §5.1 constraint check) have no field in the §4.1 contract; read from an optional slva_diag part if present, else "
    "reported 'not in the field contract'.",
    "R22 PROBE extension rule (v1 §2.5): steps_run must be 10000 or 20000, and 10000 is invalid when the curve gains > 0.02 between 8k and 10k (PROBE(t) VOID).",
    "R23 GSM8K decisive pair readings (SERIAL-STEP-GSM, CF-GSM) use only recipients whose base item has m >= 3 (v1 §4.2 'band m >= 3'); the m = 2 exposed "
    "item's pairs are reported pair-level only.",
    "R24 G-L0 band failures also gate A2's own readings by D range (ABSORB 9-12, LENGTH / K'24 3-8, D 13-16, the MARGIN bar's A1 9-12).",
    "R25 A CF (D, j) cell with > 400 exposed pairs (v1 §2.5 'up to 400') is flagged, not voided (more pairs do not bias the reading).",
    "R26 ARGS_MUST holds the recipe values v1 states (A1 / DIRECT seed 1, 24,000 steps; A2 init_from A1, p_self 0.5, lr 3e-5, 12,000, self_kmax 16, seed 1; "
    "A2-CTRL p_self 0, self_pool none; B1 seed 1, 120,000); A2's 'c 0' has no stated c_sched string and is not asserted.",
    "R27 PATHS (G-L0, G-EVMEM, GATE-1-L, pool, B0-2, interim GO) are not named in v1; the names here are assumptions for Experiments to match or edit.",
    "R28 A1 is pending until DIRECT-SYM's cell is decided (not only until eval_direct.json exists), since a censored DIRECT-SYM voids the band.",
    "R29 PROBE / FD / swap / CF per-item stores are required (v1 §4 'every share or threshold has its per-item / per-pair outputs stored'); GSM future items are "
    "not listed in §4.2 and are not required.",
    "R30 Stage A items_sha reference = G-L0's test_items_sha if it records one (the built g_l0.json does not), else A1's file-level test sha; DIRECT-SYM / A2 / "
    "A2-CTRL must match it. Experiments' eval records one per-D test_sha per file (not per part, as v1 §4.3 asks): accepted as file-level, reported.",
    "R31 Both field layouts are read (aliases only, no reading changes): LAYOUTS' names and Experiments' langvb_eval names as of 15:10 (per-pair 'exposed', int "
    "y_hyb, cf.cells + selected pairs with clean / tstar / clean_k2, fd_slot_scores, fd_valid {s: {model}}, plant items {D: [...]}, causal_deep stall rows, "
    "top-level card / dtype / eval_bs / test_sha, eval_direct nested under 'direct', the G-L0 panel layout at scratch/gate_s218/g_l0.json). With a "
    "selected-only CF store, n_candidates / n_exposed per (D, j) come from the file (flagged), n_clean / shares / t* are recomputed."]


# ---------------------------------------------------------------- utilities
def jl(d, f):
    p = os.path.join(R, d, f)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def jp(key):
    d, f = PATHS[key]
    if d != "@ROOT":
        return jl(d, f)
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def pname(key):
    d, f = PATHS[key]
    return f if d == "@ROOT" else "%s/%s" % (d, f)


def has(d, f):
    return os.path.exists(os.path.join(R, d, f))


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else None


def median(v):
    v = list(v)
    return statistics.median(v) if v else None


def card_ok(c, stage):
    return c is not None and CARD[stage] in str(c)


def ge(x, b):
    return num(x) and num(b) and x >= b - EPS


def le(x, b):
    return num(x) and num(b) and x <= b + EPS


def rd(met, unreadable=None, **v):
    """a reading: met (bool, False whenever unreadable), status MET / NOT MET / UNREADABLE (why), plus its values."""
    if unreadable:
        return dict(met=False, readable=False, status="UNREADABLE (%s)" % unreadable, **v)
    return dict(met=bool(met), readable=True, status="MET" if met else "NOT MET", **v)


def ranks(v):   # verbatim from docs/research_lang_2026-10-18/v1_checks/v1_cf.py
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        k = i
        while k + 1 < len(v) and v[o[k + 1]] == v[o[i]]:
            k += 1
        for m in range(i, k + 1):
            r[o[m]] = (i + k) / 2.0
        i = k + 1
    return r


def spearman(a, b):   # verbatim from v1_cf.py (0.0 when either side is constant)
    ra, rb = ranks(a), ranks(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((x - ma) ** 2 for x in ra); vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(va * vb)


def crossover(ans, K):   # verbatim from v1_cf.py: t* minimising #{t < t*: ans = y} + #{t >= t*: ans = y'}; ties -> smallest t*
    ny_before = 0; nyp_after = sum(1 for a_ in ans if a_ == 0); best = (nyp_after, 0)
    for c in range(1, K + 2):
        if ans[c - 1] == 1:
            ny_before += 1
        else:
            nyp_after -= 1
        if ny_before + nyp_after < best[0]:
            best = (ny_before + nyp_after, c)
    return best[1]


def auroc(pos, neg):
    """Mann-Whitney AUROC = P(pos > neg) + P(tie) / 2, by average ranks."""
    if not pos or not neg:
        return None
    r = ranks(list(pos) + list(neg)); rp = sum(r[:len(pos)]) + len(pos)        # ranks() is 0-based
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ols(xs, ys):
    mx, my = mean(xs), mean(ys); den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else None


def mcnemar_one_sided(b, c):
    """exact one-sided McNemar p = P(X >= b), X ~ Bin(b + c, 1/2) (b = B1 right & CODI-20 wrong)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(Fraction(sum(math.comb(n, k) for k in range(b, n + 1)), 2 ** n))


def cluster_mean(groups):
    """base-item-clustered mean (Addendum 1 item 6): values averaged within each base item first, then over base items."""
    ms = [mean(v) for v in groups.values() if v]
    return mean(ms)


def cluster_boot(groups, reps=None, seed=None):
    """cluster-bootstrap 95 % CI of cluster_mean: base items resampled with replacement (fixed rng seed)."""
    ms = [mean(v) for v in groups.values() if v]
    if len(ms) < 2:
        return None
    rng = random.Random(BOOT_SEED if seed is None else seed); n = len(ms); bs = []
    for _ in range(reps or BOOT_REPS):
        bs.append(sum(ms[rng.randrange(n)] for _ in range(n)) / n)
    bs.sort()
    return [bs[int(0.025 * (len(bs) - 1))], bs[int(math.ceil(0.975 * (len(bs) - 1)))]]


def same(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return num(a) and num(b) and abs(a - b) <= tol


# ---------------------------------------------------------------- logs, gates, args
def log_info(d):
    """log.jsonl: last step (last occurrence per step wins), guard stop, plain_cand_ce by step, valid_acc checks, max self_reuse."""
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    steps, ce, va, guard, reuse = set(), {}, {}, False, 0
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("guard_stop") or r.get("GUARD_STOP"):
                guard = True
            if num(r.get("self_reuse")):
                reuse = max(reuse, r["self_reuse"])
            if "step" not in r:
                continue
            s = int(r["step"]); steps.add(s)
            if num(r.get("plain_cand_ce")):
                ce[s] = r["plain_cand_ce"]
            if num(r.get("valid_acc")):
                va[s] = r["valid_acc"]
    return {"last": max(steps) if steps else None, "n_steps": len(steps), "guard_stop": guard, "ce": ce, "valid_acc": va, "self_reuse": reuse}


def entry_prosqa(ce):
    """NO-ENTRY (v1 §5.2): entry iff plain_cand_ce (1k-bin means, bin = (step - 1) // 1000) <= 0.35 in two consecutive bins."""
    bins = {}
    for s, v in ce.items():
        bins.setdefault((s - 1) // 1000, []).append(v)
    bm = {b: mean(v) for b, v in bins.items()}
    first = next((b for b in sorted(bm) if b + 1 in bm and bm[b] <= 0.35 + EPS and bm[b + 1] <= 0.35 + EPS), None)
    return {"entered": first is not None, "entry_bin_end_step": None if first is None else 1000 * (first + 2), "n_bins": len(bm),
            "min_bin": min(bm.values()) if bm else None}


def entry_gsm(va):
    """NO-ENTRY (GSM8K): Aug-valid accuracy <= 0.15 at every check."""
    return {"entered": any(v > 0.15 + EPS for v in va.values()), "n_checks": len(va), "max_valid_acc": max(va.values()) if va else None}


def _l0_streams(g):
    """the per-stream rows of G-L0: the 'streams' layout (LAYOUTS), or Experiments' built layout (panel {D: {n_gate, invariant_failures, chase_by_stream,
    max_person_agnostic_by_stream, ubidir_by_stream {stream: {r: v}}, udist_by_stream, beyond_D_by_stream}}, full_rows {stream: {D: {n, *_invariant_fail}}})."""
    if isinstance(g.get("streams"), dict):
        return g["streams"]
    P, FR, S = g.get("panel") or {}, g.get("full_rows") or {}, {}
    for D, e in P.items():
        for st, ch in (e.get("chase_by_stream") or {}).items():
            fr = (FR.get(st) or {}).get(D) or {}
            inv = (fr.get("module_invariant_fail") or 0) + (fr.get("indep_invariant_fail") or 0) if fr else e.get("invariant_failures")
            ub = (e.get("ubidir_by_stream") or {}).get(st)
            ubv = (max(ub.values(), key=lambda v: abs(v - 0.5) if num(v) else 9) if ub else 0.5) if isinstance(ub, dict) else ub   # farthest from 1/2
            S.setdefault(st, {})[D] = {"n_gate": fr.get("n", e.get("n_gate")), "invariant_failures": inv, "chase": ch,
                                       "max_person_agnostic": (e.get("max_person_agnostic_by_stream") or {}).get(st), "ubidir_max": ubv,
                                       "udist": (e.get("udist_by_stream") or {}).get(st), "beyond_D": (e.get("beyond_D_by_stream") or {}).get(st)}
    return S


def gate_l0():
    """G-L0 (v1 §7; RESOLVED R2): per band D, usable iff every stream covering D passes; missing entries -> 'missing' (pending)."""
    g = jp("gate_l0")
    if g is None:
        return None
    S = _l0_streams(g); miss, bands, bug, incons = [], {}, [], []
    for D in D_TEST:
        bands[D] = []
    combos = {(st, D) for st, Ds in L0_STREAMS.items() for D in Ds} | {(st, int(D)) for st, v in S.items() for D in v if str(D).isdigit() and int(D) in D_TEST}
    for st, D in sorted(combos):                                            # required combos + every other (stream, band) the gate ran ("every stream")
        e = (S.get(st) or {}).get(str(D))
        if not isinstance(e, dict):
            miss.append("%s D%d" % (st, D)); continue
        why = []
        if not num(e.get("n_gate")) or e["n_gate"] < 5000:
            why.append("%s n_gate %s < 5000" % (st, e.get("n_gate")))
        if e.get("invariant_failures") != 0:
            why.append("%s invariant_failures %s" % (st, e.get("invariant_failures")))
        ch = e.get("chase") or {}
        for c in L0_GATED:
            if not num(ch.get(c)) or not ch[c] < 0.6 - EPS:
                why.append("%s chase %s %s >= 0.6" % (st, c, ch.get(c)))
        if not num(e.get("max_person_agnostic")) or not e["max_person_agnostic"] < 0.6 - EPS:
            why.append("%s person-agnostic %s >= 0.6" % (st, e.get("max_person_agnostic")))
        if not num(e.get("ubidir_max")) or abs(e["ubidir_max"] - 0.5) > 1e-12:
            why.append("%s uBIDIR %s != 0.5 (CODE-BUG)" % (st, e.get("ubidir_max"))); bug.append((st, D, "ubidir"))
        if e.get("beyond_D") != 0:
            why.append("%s beyond_D %s != 0 (CODE-BUG)" % (st, e.get("beyond_D"))); bug.append((st, D, "beyond_D"))
        if isinstance(e.get("gate_pass"), bool) and e["gate_pass"] != (not why):
            incons.append("%s D%d gate_pass %s vs recomputed %s" % (st, D, e["gate_pass"], not why))
        bands[D] += why
    ub = g.get("usable_bands")
    if isinstance(ub, list) and sorted(int(x) for x in ub) != sorted(D for D, w in bands.items() if not w):
        incons.append("file usable_bands %s != recomputed" % ub)
    return {"missing": miss, "usable": {D: not w for D, w in bands.items()}, "why": {D: w for D, w in bands.items() if w},
            "failed": sorted(D for D, w in bands.items() if w), "code_bug": bug, "flag_inconsistent": incons, "test_items_sha": g.get("test_items_sha"),
            "card": (g.get("meta") or {}).get("card", g.get("card")), "file_pass": g.get("pass"), "names_ok": g.get("names_ok")}


def evmem(stage):
    g = jp("evmem_" + stage)
    if g is None:
        return None
    eb = g.get("eval_bs")
    if not isinstance(eb, dict) and isinstance(g.get("evmem"), dict):       # langvb_eval part_evmem: {evmem: {part: {peak_gib_by_bs, eval_bs, D}, ...}}
        eb = {p_: v.get("eval_bs") for p_, v in g["evmem"].items() if isinstance(v, dict) and "eval_bs" in v}
    return {"eval_bs": eb or {}, "card": g.get("card", (g.get("meta") or {}).get("card"))}


def args_diff(k):
    """ARGS-DIFF (v1 §4.3): keys differing from the reference (META excluded) must be in ARGS_ALLOW[k]; required values ARGS_MUST[k]."""
    d = CELL[k]; a = jl(d, "args.json")
    if ARGS_REF.get(k) is None or ARGS_ALLOW.get(k) is None:
        return {"status": "UNLISTED", "ok": None, "why": "ARGS-DIFF %s not pre-listed in v1 (ARGS_REF / ARGS_ALLOW['%s'] unset; RESOLVED R1)" % (
            "reference" if ARGS_REF.get(k) is None else "allowed-key list", k)}
    b = jl(*ARGS_REF[k])
    if a is None or b is None:
        return {"status": "MISSING", "ok": None, "why": "args.json missing (%s)" % ("cell" if a is None else "/".join(ARGS_REF[k]))}
    allow = ARGS_ALLOW[k]
    shared = {x: (b[x], a[x]) for x in a if x in b and x not in META and x not in allow and a[x] != b[x]}
    new = sorted(x for x in a if x not in b and x not in META and x not in allow)
    gone = sorted(x for x in b if x not in a and x not in META and x not in allow)
    req = {}
    for x, v in (ARGS_MUST.get(k) or {}).items():
        got = a.get(x)
        if v is None:
            ok = got in (None, "", "none", "None")
        elif isinstance(v, str):
            ok = isinstance(got, str) and got.replace("\\", "/").endswith(v.split("boxes/runs_fsq/")[-1])
        elif isinstance(v, float):
            ok = num(got) and abs(got - v) <= 1e-12
        else:
            ok = got == v
        if not ok:
            req[x] = (v, got)
    ok = not shared and not new and not gone and not req
    return {"status": "OK" if ok else "BAD", "ok": ok, "ref": "/".join(ARGS_REF[k]), "shared_diffs": shared, "unexpected_new_keys": new,
            "missing_keys": gone, "required_bad": req}


def meta_check(f, stage, parts, evm, ref_sha, dtype=DTYPE, dtype_parts=None):
    """v1 §4.3 asserts: card / dtype per file (and per part when card_by_part / dtype_by_part present), eval_bs per part == G-EVMEM, items_sha per part ==
    the stage's reference. Returns (missing fields, mismatches). The fields may sit in a 'meta' block or at the file's top level (Experiments' langvb_eval
    layout: card, dtype, eval_bs, test_sha {D: sha}); a per-D sha map is one file-level sha (compared canonically)."""
    m = dict((f or {}).get("meta") or {}); miss, bad = [], []
    for key in ("card", "dtype", "eval_bs", "card_by_part", "dtype_by_part"):
        if key not in m and key in (f or {}):
            m[key] = f[key]
    if "items_sha" not in m:
        m["items_sha"] = (f or {}).get("items_sha", (f or {}).get("test_sha"))
    if is_dsha(m["items_sha"]):
        m["items_sha"] = canon_sha(m["items_sha"])
    if m.get("card") is None:
        miss.append("meta.card")
    elif not card_ok(m["card"], stage):
        bad.append("card %s (stage %s card %s)" % (m["card"], stage, CARD[stage]))
    for p, c in (m.get("card_by_part") or {}).items():
        if p in parts and not card_ok(c, stage):
            bad.append("part %s card %s" % (p, c))
    if m.get("dtype") is None:
        miss.append("meta.dtype")
    elif m["dtype"] != dtype:
        bad.append("dtype %s != %s" % (m["dtype"], dtype))
    for p, dt in (m.get("dtype_by_part") or {}).items():                   # e.g. the labeller part's bf16 copy inside an fp32 eval file
        want = (dtype_parts or {}).get(p, dtype)
        if p in parts and dt != want:
            bad.append("part %s dtype %s != %s" % (p, dt, want))
    eb, sh = m.get("eval_bs") or {}, m.get("items_sha")
    for p in parts:
        if p in NO_BS_PARTS:
            continue
        v = eb.get(p) if isinstance(eb, dict) else None
        if v is None:
            miss.append("meta.eval_bs.%s" % p)
        elif evm is not None and p in evm["eval_bs"] and v != evm["eval_bs"][p]:
            bad.append("part %s eval_bs %s != G-EVMEM %s" % (p, v, evm["eval_bs"][p]))
        elif evm is not None and p not in evm["eval_bs"]:
            miss.append("G-EVMEM eval_bs.%s" % p)
        s = sh.get(p) if isinstance(sh, dict) else sh
        s = canon_sha(s) if is_dsha(s) else s
        if not s:
            miss.append("meta.items_sha%s" % ("." + p if isinstance(sh, dict) else ""))
        elif ref_sha and s != canon_sha(ref_sha):
            bad.append("part %s items_sha %s != reference %s" % (p, str(s)[:24], str(canon_sha(ref_sha))[:24]))
    return miss, bad


def is_dsha(x):
    return isinstance(x, dict) and bool(x) and all(str(k).isdigit() for k in x)


def canon_sha(x):
    """a per-D sha map {D: sha} -> one canonical string; strings pass through."""
    return json.dumps({str(k): v for k, v in sorted(x.items(), key=lambda kv: int(kv[0]))}, separators=(",", ":")) if is_dsha(x) else x


def file_sha(f):
    """the file-level test-set sha of an eval file (meta.items_sha / items_sha / test_sha), canonical; None if absent or per-part only."""
    if not isinstance(f, dict):
        return None
    s = (f.get("meta") or {}).get("items_sha", f.get("items_sha", f.get("test_sha")))
    return canon_sha(s) if (isinstance(s, str) or is_dsha(s)) else None


def parts_missing(f, spec, parts):
    return [p for p in parts if not all(x in (f or {}) for x in spec[p])]


def pend_label(miss):
    files = sorted(v for k, v in miss.items() if k != "mismatch")
    return "pending (%s)" % "; ".join((["missing %s" % files] if files else []) + (["mismatch %s" % miss["mismatch"]] if "mismatch" in miss else []))


# ---------------------------------------------------------------- ProsQA part parsers (each returns missing fields -> pending, void -> UNREADABLE)
def p_acc(ev):
    a2, ak, it = ev.get("acc_d2") or {}, ev.get("acc_k") or {}, ev.get("items") or {}; miss, bad = [], []
    for D in D_TEST:
        sD = str(D)
        if sD not in a2 or not isinstance(it.get(sD), dict):
            miss.append("acc D%d" % D); continue
        for K in K_GRID:
            lst, v = it[sD].get(str(K)), (ak.get(str(K)) or {}).get(sD)
            if not isinstance(lst, list) or not num(v):
                miss.append("acc D%d K'%d" % (D, K)); continue
            if len(lst) != PER_D:
                bad.append("items D%d K'%d n %d != %d" % (D, K, len(lst), PER_D))
            elif abs(mean(lst) - v) > 1e-12:
                bad.append("acc_k[%d][%d] %s != mean(items) %s" % (K, D, v, mean(lst)))
        if not same(a2[sD], (ak.get(str(D + 2)) or {}).get(sD), 1e-12):
            bad.append("acc_d2[%d] != acc_k[%d][%d]" % (D, D + 2, D))
    a = {D: a2.get(str(D)) for D in D_TEST}
    return {"missing": miss[:20], "void": bad[:20] or None, "a": a, "ak": ak, "items": it}


def acc_items(acc, Ds, K_of=lambda D: D + 2):
    out = []
    for D in Ds:
        out += list(acc["items"][str(D)][str(K_of(D))])
    return out


def p_probe(ev):
    P = ev.get("probe") or {}; miss, bad, acc, items = [], {}, {}, {}
    for t in PROBE_GRID:
        e = P.get(str(t))
        if not isinstance(e, dict):
            miss.append("probe t%d" % t); continue
        ab, it, b = e.get("acc_by_D") or {}, e.get("items") or {}, []
        acc[t], items[t] = {}, {}
        for D in sorted(set(int(x) for x in ab) | set(range(3, 9))):
            lst, v = it.get(str(D)), ab.get(str(D))
            if not isinstance(lst, list) or not lst or not num(v):
                if D <= 8:
                    miss.append("probe t%d D%d" % (t, D))
                continue
            acc[t][D], items[t][D] = v, lst
            if abs(mean(lst) - v) > 1e-12:
                b.append("acc_by_D[%d] %s != mean(items) %s" % (D, v, mean(lst)))
        sr = e.get("steps_run"); cv = {int(s): a for s, a in (e.get("curve") or [])}
        if sr is None or not cv:
            miss.append("probe t%d steps_run / curve" % t)
        elif sr not in (10000, 20000):
            b.append("steps_run %s not in {10000, 20000}" % sr)
        elif sr == 10000 and (8000 not in cv or 10000 not in cv):
            b.append("curve lacks steps 8000 / 10000")
        elif sr == 10000 and cv[10000] - cv[8000] > 0.02 + EPS:
            b.append("extension rule: last-2k gain %.4f > 0.02 but steps_run 10000 (RESOLVED R22)" % (cv[10000] - cv[8000]))
        if b:
            bad[t] = b
    return {"missing": miss[:20], "void": bad or None, "acc": acc, "items": items}


def probe_pooled(pr, t, Ds):
    out = []
    for D in Ds:
        out += list(pr["items"][t][D])
    return mean(out)


def _pm(X, model):
    return X[model] if isinstance(X, dict) and model in X else X


def _fv(FV, model, s):
    """fd_valid AUROC at (t, k) = (s, 0): {s: auroc}, {model: {s: auroc}} or {s: {model: auroc | {auroc}}} (langvb_eval)."""
    for v in (((FV.get(model) or {}).get(s) if isinstance(FV.get(model), dict) else None), ((FV.get(s) or {}).get(model) if isinstance(FV.get(s), dict) else None),
              FV.get(s)):
        v = v.get("auroc") if isinstance(v, dict) else v
        if num(v):
            return v
    return None


def p_plant(ev):
    P, V = ev.get("plant") or {}, ev.get("probe_valid") or {}; miss, bad, valid, eps = [], [], {}, {}
    for t in (0, 1, 2):
        e = P.get(str(t))
        if not isinstance(e, dict) or not all(isinstance(e.get(x), dict) and num(e[x].get("acc")) for x in ("lin", "xor")):
            miss.append("plant t%d lin / xor acc" % t); continue
        for x in ("lin", "xor"):
            it = e[x].get("items")
            if isinstance(it, dict):                                         # langvb_eval: items {D: [0/1]} -> pooled in D order
                it = [v for D in sorted(it, key=int) for v in it[D]]
            if isinstance(it, list) and it and abs(mean(it) - e[x]["acc"]) > 1e-12:
                bad.append("plant t%d %s acc != mean(items)" % (t, x))
            elif not isinstance(it, list):
                miss.append("plant t%d %s items" % (t, x))
        valid[t] = ge(e["lin"]["acc"], 0.90) and ge(e["xor"]["acc"], 0.90)
        eps[t] = {"eps": e.get("eps"), "eps_rel": e.get("eps_rel"), "lin": e["lin"]["acc"], "xor": e["xor"]["acc"], "pcs": e.get("pcs"),
                  "flag": None if num(e.get("eps_rel")) else "eps_rel (eps / ||V_t||) not reported (Addendum 1 item 5; descriptive)"}
        if not isinstance(V.get(str(t)), bool):
            miss.append("probe_valid t%d" % t)
        elif V[str(t)] != valid[t]:
            bad.append("probe_valid[%d] %s != recomputed %s" % (t, V[str(t)], valid[t]))
    return {"missing": miss, "void": bad or None, "valid": valid, "plant": eps}


def p_fd(ev):
    F, FV, FP = ev.get("fd") or {}, ev.get("fd_valid") or {}, ev.get("fd_plant"); miss, bad, cells, valid = [], [], {}, {}
    for model in ("lin", "mlp"):
        for t in (1, 2):
            for k in range(1, 9 - t):
                e = ((F.get(model) or {}).get(str(t)) or {}).get(str(k))
                if not isinstance(e, dict) or not num(e.get("auroc")):
                    miss.append("fd %s t%d k%d" % (model, t, k)); continue
                pos, neg = e.get("pos"), e.get("neg")
                srows = (ev.get("fd_slot_scores") or {}).get("%s_t%d_k%d" % (model, t, k))
                if not isinstance(pos, list) and isinstance(srows, list):           # langvb_eval: fd_slot_scores {lin_t1_k3: [[item, slot, y, score], ...]}
                    pos = [r_[3] for r_ in srows if r_[2]]; neg = [r_[3] for r_ in srows if not r_[2]]
                if isinstance(pos, list) and isinstance(neg, list):
                    a = auroc(pos, neg)
                    if a is None or abs(a - e["auroc"]) > 1e-9 or e.get("n_pos") != len(pos) or e.get("n_neg") != len(neg):
                        bad.append("fd %s t%d k%d auroc %s / n vs per-slot scores %s" % (model, t, k, e["auroc"], a))
                elif k >= 3:
                    miss.append("fd %s t%d k%d per-slot scores (pos / neg)" % (model, t, k))
                cells[(model, t, k)] = {"auroc": e["auroc"], "n_pos": e.get("n_pos"), "n_neg": e.get("n_neg")}
        fp = _pm(FP, model)
        s3, s4 = _fv(FV, model, "3"), _fv(FV, model, "4")
        pa = fp.get("auroc") if isinstance(fp, dict) else fp
        if not (num(s3) and num(s4) and num(pa)):
            miss.append("fd_valid / fd_plant %s" % model); continue
        valid[model] = {"s3": s3, "s4": s4, "plant": pa, "ok": ge(s3, 0.90) and ge(s4, 0.90) and ge(pa, 0.90)}
    return {"missing": miss[:20], "void": bad[:20] or None, "cells": cells, "valid": valid}


def fo(p, tol=None):
    """(follow, own) of one pair: explicit follow / own, or y_hyb in {'follow', 'own', 'neither'}, or y_hyb compared with y_don / y_rec (tol for numbers)."""
    if p.get("follow") in (0, 1, True, False) and p.get("own") in (0, 1, True, False):
        return int(bool(p["follow"])), int(bool(p["own"]))
    y = p.get("y_hyb")
    if y in ("follow", "own", "neither"):
        return int(y == "follow"), int(y == "own")
    if "y_rec" not in p and (y is True or y is False or (isinstance(y, int) and y in (0, 1))):
        return int(y), 1 - int(y)                                            # ProsQA two-candidate readout: 1 = prefers the donor's answer (langvb_eval)
    if y is None or "y_rec" not in p:
        return None
    if tol is None:
        eq = lambda a, b: a is not None and b is not None and a == b  # noqa: E731
    else:
        eq = lambda a, b: num(a) and num(b) and abs(a - b) < tol  # noqa: E731
    return int(eq(y, p.get("y_don"))), int(eq(y, p.get("y_rec")))


def p_vswap_arm(arm, A):
    """one ProsQA swap arm: exposure, clean, follow / own recomputed from every candidate pair; the file's summaries must agree (else VOID)."""
    rule = VSWAP_RULE[arm]; pairs = (A or {}).get("pairs")
    if not isinstance(pairs, list):
        return {"missing": ["vswap.%s.pairs" % arm]}
    need = ("ok_rec",) + (("ok_don",) if rule["donor"] else ()) + ("D", "t"); miss, rows, t2 = set(), [], []
    for p in pairs:
        for f in need:
            if f not in p:
                miss.add("vswap.%s pair field %s" % (arm, f))
        comp = bool(rule["need"]) and all(f in p for f in rule["need"])     # exposure from its components (LAYOUTS) or the eval's per-pair 'exposed' flag
        if rule["need"] and not comp and "exposed" not in p:
            miss.add("vswap.%s pair exposure fields (%s, or exposed)" % (arm, "/".join(rule["need"])))
        x = fo(p)
        if x is None:
            miss.add("vswap.%s pair y_hyb / y_rec / y_don" % arm); continue
        D, t = p.get("D"), p.get("t")
        if not (num(D) and num(t)):
            continue
        cond = all(bool(p.get(f)) for f in rule["need"]) if comp else bool(p.get("exposed", True))
        clean_ = bool(p.get("ok_rec")) and (bool(p.get("ok_don")) if (rule["donor"] or "ok_don" in p) else True)   # v1 §2.5: recipient and its arm's donor
        exp = D in rule["D"] and t in rule["t"](D) and cond
        rows.append((D, t, exp, exp and clean_, x[0], x[1]))
        if arm == "ii_prime" and t == 2 and 4 <= D <= 8 and cond:
            t2.append((D, clean_, x[0], x[1]))
    if miss:
        return {"missing": sorted(miss)}
    ex = [r for r in rows if r[2]]; cl = [r for r in rows if r[3]]
    o = {"n_candidates": len(pairs), "n_exposed": len(ex), "n_clean": len(cl), "follow": mean(r[4] for r in cl), "own": mean(r[5] for r in cl), "by_D": {}}
    for D in sorted(set(r[0] for r in rows)):
        e_, c_ = [r for r in ex if r[0] == D], [r for r in cl if r[0] == D]
        o["by_D"][D] = {"n_exposed": len(e_), "n_clean": len(c_), "follow": mean(r[4] for r in c_), "own": mean(r[5] for r in c_)}
    o["by_t"] = {t: {"n_clean": len([r for r in cl if r[1] == t]), "follow": mean(r[4] for r in cl if r[1] == t)} for t in sorted(set(r[1] for r in cl))}
    t2c = [r for r in t2 if r[1]]
    o["t2_descriptive"] = {"n_clean": len(t2c), "follow": mean(r[2] for r in t2c), "own": mean(r[3] for r in t2c)} if arm == "ii_prime" else None
    o["n_pos_fail"] = sum(1 for p in pairs if "pos_ok" in p and not p.get("pos_ok"))    # G-L4 expects 0 (position identity; langvb_eval asserts it per batch)
    bad = []
    for f in ("n_candidates", "n_exposed", "n_clean"):
        if A.get(f) != o[f]:
            bad.append("%s %s != recomputed %s" % (f, A.get(f), o[f]))
    own_f = A.get("own", A.get("recovery"))
    for f, v in (("follow", A.get("follow")), ("own", own_f)):
        if not same(v, o[f]) and not (arm == "reset" and f == "follow"):
            bad.append("%s %s != recomputed %s" % (f, v, o[f]))
    for D, x in (A.get("by_D") or {}).items():
        y = o["by_D"].get(int(D))
        if y is None or x.get("n_exposed") != y["n_exposed"] or x.get("n_clean") != y["n_clean"] or not same(x.get("own"), y["own"]) or (
                arm != "reset" and not same(x.get("follow"), y["follow"])):
            bad.append("by_D[%s] != recomputed" % D)
    o["void"] = bad or None
    o["missing"] = [] if isinstance(A.get("by_D"), dict) else ["vswap.%s.by_D" % arm]
    return o


def p_vswap(ev):
    V = ev.get("vswap") or {}; out, miss, void = {}, [], {}
    for arm in VSWAP_RULE:
        if not isinstance(V.get(arm), dict):
            miss.append("vswap.%s" % arm); continue
        x = p_vswap_arm(arm, V[arm]); out[arm] = x; miss += x.get("missing") or []
        if x.get("void"):
            void[arm] = x["void"]
    return {"missing": miss[:20], "void": void or None, "arms": out}


def p_stall(ev, acc):
    S = ev.get("stall") or {}; it = S.get("items") or {}; st, inn, miss, byD = [], [], [], {}
    for D in range(4, 9):
        e = it.get(str(D))
        s = e.get("stalled") if isinstance(e, dict) else e
        i_ = e.get("intact") if isinstance(e, dict) and isinstance(e.get("intact"), list) else (
            (acc["items"].get(str(D)) or {}).get(str(D)) if isinstance(acc.get("items"), dict) else None)
        if not isinstance(s, list) or not isinstance(i_, list):
            miss.append("stall items D%d" % D); continue
        if len(s) != len(i_):
            return {"missing": [], "void": ["stall D%d: %d stalled vs %d intact items" % (D, len(s), len(i_))], "n": len(st)}
        st += s; inn += i_; byD[D] = {"n": len(s), "stalled": mean(s), "intact": mean(i_)}
    if miss or not num(S.get("recovery")):
        return {"missing": miss + ([] if num(S.get("recovery")) else ["stall.recovery"])}
    rec = (mean(st) / mean(inn)) if inn and mean(inn) else None
    bad = [] if same(rec, S["recovery"]) else ["recovery %s != recomputed %s" % (S["recovery"], rec)]
    return {"missing": [], "void": bad or None, "recovery": rec, "n": len(st), "by_D": byD}


def p_cf(ev):
    """CONSUMPTION-FRONT (v1 §2.5 / §3.6 PS-6): t* per pair recomputed from answers_by_t; per-(D, j) counts, shares and t* lists must agree (else VOID).
    Two stores are accepted: every candidate with pos_ok / flip_ok (LAYOUTS; exposure recomputed), or only the selected exposed recipients with per-pair
    clean / tstar / clean_k2 and the cells under cf.cells (langvb_eval; n_candidates / n_exposed then come from the file and are flagged)."""
    C = ev.get("cf") or {}; pairs = C.get("pairs"); CC = C.get("cells") if isinstance(C.get("cells"), dict) else C
    if not isinstance(pairs, list):
        return {"missing": ["cf.pairs"]}
    full = bool(pairs) and all("pos_ok" in p and "flip_ok" in p for p in pairs)
    miss, rows, endmis, tsbad = set(), [], 0, 0
    for p in pairs:
        for f in ("D", "j", "answers_by_t"):
            if f not in p:
                miss.add("cf pair field %s" % f)
        D, j, ans = p.get("D"), p.get("j"), p.get("answers_by_t")
        if not (num(D) and num(j)) or not isinstance(ans, list) or len(ans) != D + 1:
            miss.add("cf pair answers_by_t (length D + 1)"); continue
        ans = [int(bool(a)) for a in ans]; ends = ans[D] == 1 and ans[0] == 0       # t = 0 / D are x' / x's own runs (G-L4b)
        if "clean" in p:
            cl_ = bool(p["clean"])
        else:
            okr = bool(p["ok_rec"]) if "ok_rec" in p else ans[D] == 1; okd = bool(p["ok_don"]) if "ok_don" in p else ans[0] == 0; cl_ = okr and okd
        endmis += int(cl_ != ends)
        ts = crossover(ans, D)
        if "tstar" in p and p["tstar"] != ts:
            tsbad += 1
        exp = (bool(p.get("pos_ok")) and bool(p.get("flip_ok")) if full else True) and 4 <= D <= 8 and 1 <= j <= D
        d2 = (p.get("ok_rec_d2"), p.get("ok_don_d2"))
        c2 = bool(p["clean_k2"]) if "clean_k2" in p else ((bool(d2[0]) and bool(d2[1])) if None not in d2 else None)
        rows.append({"D": D, "j": j, "exp": exp, "clean": exp and cl_, "ts": ts, "ans": ans, "clean_d2": c2})
    if miss:
        return {"missing": sorted(miss)}
    cells, bad = {}, (["%d pairs' stored tstar != the least-violations change point" % tsbad] if tsbad else [])
    for r in rows:
        c = cells.setdefault((r["D"], r["j"]), {"cand": 0, "exp": [], "clean": []}); c["cand"] += 1
        if r["exp"]:
            c["exp"].append(r)
        if r["clean"]:
            c["clean"].append(r)
    out, flags = {}, ([] if full else ["exposure counts n_candidates / n_exposed read from the file: only selected exposed pairs are stored (v1 §4.3 asks "
                                       "recomputation from the pairs; n_clean, shares and t* are recomputed)"])
    for (D, j), c in sorted(cells.items()):
        cl, ex = c["clean"], c["exp"]; f = (CC.get(str(D)) or {}).get(str(j))
        o = {"n_candidates": c["cand"] if full else (f or {}).get("n_candidates"), "n_exposed": len(ex) if full else (f or {}).get("n_exposed"),
             "n_stored": len(ex), "n_clean": len(cl), "share_window": mean(int(j - 1 <= r["ts"] <= j) for r in cl),
             "tstar": sorted(r["ts"] for r in cl), "curve": {t: mean(1 - r["ans"][t] for r in cl) for t in range(D + 1)} if cl else {},
             "share_clean_kd": (len(cl) / len(ex)) if ex else None,
             "share_clean_kd2": mean(int(bool(r["clean_d2"])) for r in ex) if ex and all(r["clean_d2"] is not None for r in ex) else None}
        out[(D, j)] = o
        if not isinstance(f, dict):
            bad.append("cf[%d][%d] absent from the file" % (D, j)); continue
        for k in (("n_candidates", "n_exposed", "n_clean") if full else ("n_clean",)):
            if f.get(k) != o[k]:
                bad.append("cf[%d][%d] %s %s != recomputed %s" % (D, j, k, f.get(k), o[k]))
        if not same(f.get("share_window"), o["share_window"]):
            bad.append("cf[%d][%d] share_window %s != recomputed %s" % (D, j, f.get("share_window"), o["share_window"]))
        if sorted(f.get("tstar") or []) not in (o["tstar"], sorted(r["ts"] for r in ex)):
            bad.append("cf[%d][%d] tstar list != recomputed" % (D, j))
        if cl:
            fc = {int(t): v for t, v in (f.get("curve") or {}).items()}
            ce = {t: mean(1 - r["ans"][t] for r in ex) for t in range(D + 1)}
            if not (all(same(fc.get(t), v) for t, v in o["curve"].items()) and len(fc) == len(o["curve"])) and not (
                    all(same(fc.get(t), v) for t, v in ce.items()) and len(fc) == len(ce)):
                bad.append("cf[%d][%d] curve != recomputed (clean or exposed pairs)" % (D, j))
    flags += ["cf[%d][%d] %d exposed pairs used > 400 (v1 §2.5 'up to 400'; flagged, not voiding)" % (D, j, x["n_stored"]) for (D, j), x in out.items()
              if x["n_stored"] > 400]
    for D, v in CC.items():
        if str(D).isdigit() and isinstance(v, dict):
            for j in v:
                if str(j).isdigit() and (int(D), int(j)) not in out:
                    bad.append("cf[%s][%s] in the file but no pairs" % (D, j))
    return {"missing": [], "void": bad[:20] or None, "cells": out, "rows": rows, "endpoint_mismatch": endmis, "flags": flags, "full_candidate_store": full,
            "n_pos_fail": sum(1 for p in pairs if "pos_ok" in p and not p["pos_ok"]), "n_flip_fail": sum(1 for p in pairs if "flip_ok" in p and not p["flip_ok"]),
            "file_pooled": {"share_window_pooled": C.get("share_window_pooled"), "spearman_within_D": C.get("spearman_within_D")}}


def p_hist24(ev, acc):
    """hist24 (v1 §4.1, SD2 E4 scope): per-arm means at D 3..8 recomputed from the per-item store; intact == the acc part bitwise (the SELFDEPTH gate)."""
    H = ev.get("hist24") or {}; it = H.get("items") or {}; miss, bad, rec = [], [], {}
    for arm in HIST_ARMS:
        if not isinstance(H.get(arm), dict):
            miss.append("hist24.%s" % arm); continue
        rec[arm] = {"acc_d2": {}, "acc_k24": {}}
        for D in range(3, 9):
            x = (it.get(arm) or {}).get(str(D)) or {}
            for key, ik in (("acc_d2", "ok_d2"), ("acc_k24", "ok_k24")):
                lst, v = x.get(ik), (H[arm].get(key) or {}).get(str(D))
                if not isinstance(lst, list) or not lst or not num(v):
                    miss.append("hist24 %s D%d %s" % (arm, D, ik)); continue
                rec[arm][key][D] = mean(lst)
                if abs(mean(lst) - v) > 1e-12:
                    bad.append("hist24 %s %s[%d] != mean(items)" % (arm, key, D))
    if miss:
        return {"missing": miss[:20]}
    ak = acc.get("ak") or {}; a = acc.get("a") or {}
    gate = all(num(a.get(D)) and abs(rec["intact"]["acc_d2"][D] - a[D]) <= 1e-12 and abs(rec["intact"]["acc_k24"][D] - (ak.get("24") or {}).get(str(D), -9)) <= 1e-12
               for D in range(3, 9))
    g = lambda arm, D: rec[arm]["acc_d2"][D] - rec[arm]["acc_k24"][D]  # noqa: E731
    st = {arm: mean(g(arm, D) - g("intact", D) for D in range(3, 7)) for arm in HIST_ARMS if arm != "intact"}
    content = ("CONTENT-STABILISES" if (ge(st["bank_ablate_slva"], 0.10) and ge(st["dup_newest"], 0.10)) else
               "not stabilising (STAB_bank < 0.10)" if not ge(st["bank_ablate_slva"], 0.10) else "COUNT-LIKE / MIXED (STAB_dup_newest < 0.10)")
    keeps = bool(gate and not bad and content == "CONTENT-STABILISES" and ge(st["self_ablate"], 0.5))
    return {"missing": [], "void": bad or None, "gate": gate, "STAB": st, "CONTENT": content, "STAB_self": st["self_ablate"], "KEEPS_SELF_STAB_L": keeps}


def p_causal_deep(ev, a):
    """CAUSAL-DEEP-L (descriptive; §217 rule at D 9..16, RESOLVED R17)."""
    Cd = ev.get("causal_deep") or {}; miss = []
    for arm in ("ii_prime", "i_prime"):
        if not isinstance((Cd.get(arm) or {}).get("by_D"), dict):
            miss.append("causal_deep.%s.by_D" % arm)
    if not isinstance((Cd.get("stall") or {}).get("by_D"), dict):
        miss.append("causal_deep.stall.by_D")
    if miss:
        return {"missing": miss}
    B = [D for D in range(9, 17) if ge(a.get(D), 0.95)]; excl, ok_rows = [], {}
    for D in B:
        rows = {arm: (Cd[arm]["by_D"].get(str(D)) or {}) for arm in ("ii_prime", "i_prime")}; s0 = Cd["stall"]["by_D"].get(str(D)) or {}
        n_ = s0.get("n", s0.get("items")); rc = s0.get("recovery")                # langvb_eval stall_by rows: items (count), intact_kD, stalled_kD1
        if rc is None and num(s0.get("intact_kD")) and num(s0.get("stalled_kD1")) and s0["intact_kD"] > 0:
            rc = s0["stalled_kD1"] / s0["intact_kD"]
        s = {"n": n_ if num(n_) else 0, "recovery": rc}
        if any((r.get("n_clean") or 0) < 100 for r in rows.values()) or s["n"] < 500 or not num(rc):
            excl.append(D); continue
        ok_rows[D] = (rows, s)
    if not ok_rows:
        return {"missing": [], "label": "NOT-APPLICABLE (empty competent band)", "B": B, "excluded": excl, "CAUSAL_DEEP_L": None}
    pooled = {}
    for arm in ("ii_prime", "i_prime"):
        n = sum(ok_rows[D][0][arm]["n_clean"] for D in ok_rows)
        pooled[arm] = {"follow": sum(ok_rows[D][0][arm]["n_clean"] * ok_rows[D][0][arm]["follow"] for D in ok_rows) / n,
                       "own": sum(ok_rows[D][0][arm]["n_clean"] * ok_rows[D][0][arm]["own"] for D in ok_rows) / n,
                       "min_D_follow": min(ok_rows[D][0][arm]["follow"] for D in ok_rows)}
    ns = sum(ok_rows[D][1]["n"] for D in ok_rows); stall = sum(ok_rows[D][1]["n"] * ok_rows[D][1]["recovery"] for D in ok_rows) / ns
    met = all(ge(v["follow"], 0.95) and le(v["own"], 0.02) and ge(v["min_D_follow"], 0.90) for v in pooled.values()) and ge(stall, 0.95)
    return {"missing": [], "B": B, "B_used": sorted(ok_rows), "excluded": excl, "pooled": pooled, "stall": stall, "CAUSAL_DEEP_L": met,
            "label": "CAUSAL-DEEP-L" if met else "not CAUSAL-DEEP-L"}


def parse_A(ev, parts):
    """parse every listed part of an eval_lvb.json; returns {part: parsed} with acc parsed first (stall / hist24 read it)."""
    P = {}
    if "acc" in parts:
        P["acc"] = p_acc(ev)
    acc = P.get("acc") or {"items": {}, "a": {}, "ak": {}}
    for p in parts:
        if p == "acc":
            continue
        P[p] = (p_probe(ev) if p == "probe" else p_plant(ev) if p == "plant" else p_fd(ev) if p == "fd" else p_vswap(ev) if p == "vswap" else
                p_stall(ev, acc) if p == "stall" else p_cf(ev) if p == "cf" else p_hist24(ev, acc) if p == "hist24" else
                p_causal_deep(ev, acc["a"]) if p == "causal_deep" else {"missing": [] if isinstance(ev.get("panel"), dict) else ["panel"]} if p == "panel"
                else p_labeller(ev) if p == "labeller" else {"missing": []})
    return P


def p_labeller(ev):
    la = ev.get("label_audit") or {}; miss = []
    if not (num(la.get("n")) and num(la.get("n_wrong"))):
        miss.append("label_audit {n, n_wrong}")
    if not isinstance(ev.get("kept_vs_drawn"), dict):
        miss.append("kept_vs_drawn")
    if miss:
        return {"missing": miss}
    return {"missing": [], "void": None, "label_audit": la, "LABEL_BUG": la["n_wrong"] > 0 or (num(la.get("pi_label")) and la["pi_label"] < 1 - 1e-12),
            "n_audit_flag": None if la["n"] >= 5000 else "label_audit n %s < 5,000 (v1 §4.1)" % la["n"], "kept_vs_drawn": ev.get("kept_vs_drawn"),
            "memo": ev.get("memo"), "gate1_copy_choice": (ev.get("gate1") or {}).get("choice")}


# ---------------------------------------------------------------- ProsQA readings (v1 §4.1)
def failed_in(l0, Ds):
    return sorted(set(Ds) & set((l0 or {}).get("failed") or []))


def bad_part(x, name):
    if x is None:
        return "%s part absent" % name
    if x.get("missing"):
        return "%s fields missing at --final: %s" % (name, x["missing"][:3])
    if x.get("void"):
        return "%s VOID (consistency assert): %s" % (name, str(x["void"])[:240])
    return None


def arm_reading(x, Ds, l0, bar=0.80, per_bar=0.60):
    """SERIAL-STEP arm (v1 §4.1, A-1): pooled follow >= bar over Ds, >= per_bar at every D with >= 50 exposed clean pairs (others excluded and listed);
    >= 100 exposed clean pairs pooled else UNREADABLE; a G-L0-failed band in Ds -> UNREADABLE."""
    w = bad_part(x, "vswap arm")
    if w:
        return rd(False, w)
    by = x["by_D"]; n = sum((by.get(D) or {}).get("n_clean", 0) for D in Ds)
    fol = (sum(by[D]["n_clean"] * by[D]["follow"] for D in Ds if (by.get(D) or {}).get("n_clean")) / n) if n else None
    own = (sum(by[D]["n_clean"] * by[D]["own"] for D in Ds if (by.get(D) or {}).get("n_clean")) / n) if n else None
    rdD = [D for D in Ds if (by.get(D) or {}).get("n_clean", 0) >= 50]; exc = [D for D in Ds if D not in rdD]
    v = dict(n_clean=n, n_exposed=sum((by.get(D) or {}).get("n_exposed", 0) for D in Ds), follow=fol, own=own,
             per_D={D: {"n_clean": by[D]["n_clean"], "follow": by[D]["follow"], "own": by[D]["own"]} for D in Ds if D in by}, readable_D=rdD,
             excluded_listed={D: "%d < 50 exposed clean pairs" % ((by.get(D) or {}).get("n_clean", 0)) for D in exc})
    fb = failed_in(l0, Ds)
    if fb:
        return rd(False, "G-L0-failed band %s" % fb, **v)
    if n < 100:
        return rd(False, "%d < 100 exposed clean pairs" % n, **v)
    return rd(ge(fol, bar) and all(ge(by[D]["follow"], per_bar) for D in rdD), None, **v)


def cf_reading(cfp, l0):
    """CF-L (v1 §3.6 PS-6, §4.1; RESOLVED R4-R6) + the CF labels and descriptive rows."""
    w = bad_part(cfp, "cf")
    if w and (cfp is None or cfp.get("missing") or cfp.get("void")):
        return rd(False, w), {"CF_CARRY": None, "CF_RECOMPUTE": None, "CF_FAST": None, "LAST_HOP_LATE": None}, {}
    cells = cfp["cells"]; dec = [(D, j) for D in range(4, 9) for j in range(3, D)]
    excl = {"%d,%d" % c: "%d < 30 exposed clean pairs" % (cells.get(c) or {}).get("n_clean", 0) for c in dec if (cells.get(c) or {}).get("n_clean", 0) < 30}
    used = [c for c in dec if "%d,%d" % c not in excl]
    rows = [r for r in cfp["rows"] if r["clean"] and (r["D"], r["j"]) in used]; n = len(rows)
    share = mean(int(r["j"] - 1 <= r["ts"] <= r["j"]) for r in rows)
    parts = []
    for D in range(5, 9):
        pts = [r for r in rows if r["D"] == D]
        if len(set(r["j"] for r in pts)) >= 2:
            parts.append((len(pts), spearman([r["ts"] for r in pts], [r["j"] for r in pts]), D))
    rho = (sum(a * b for a, b, _ in parts) / sum(a for a, _, _ in parts)) if parts else None
    fb = failed_in(l0, range(4, 9))
    w = ("G-L0-failed band %s" % fb if fb else "%d < 100 exposed clean pairs" % n if n < 100 else
         "no D in 5..8 with >= 2 decisive j after exclusion" if rho is None else None)
    v = dict(n_clean=n, share_window=share, spearman_within_D=rho, spearman_by_D={D: {"n": a, "rho": b} for a, b, D in parts}, excluded_listed=excl,
             cells_used=["%d,%d" % c for c in used], file_pooled=cfp.get("file_pooled"), endpoint_mismatch=cfp.get("endpoint_mismatch"),
             flags=cfp.get("flags"), n_pos_fail=cfp.get("n_pos_fail"), n_flip_fail=cfp.get("n_flip_fail"))
    reading = rd(ge(share, 0.75) and ge(rho, 0.60), w, **v)
    lrows = [r for r in cfp["rows"] if r["clean"] and r["j"] == r["D"]]
    lab = {"median_tstar": median(r["ts"] for r in rows), "CF_CARRY": (median(r["ts"] for r in rows) == 1) if rows else None,
           "CF_RECOMPUTE": (median(r["D"] - r["ts"] for r in rows) == 0) if rows else None,
           "CF_FAST": ge(median(r["j"] - r["ts"] for r in rows), 2) if rows else None,
           "LAST_HOP_LATE": bool(reading["met"] and lrows and median(r["D"] - r["ts"] for r in lrows) == 0),
           "last_hop": {"n_clean": len(lrows), "median_D_minus_tstar": median(r["D"] - r["ts"] for r in lrows),
                        "share_tstar_D": mean(int(r["ts"] == r["D"]) for r in lrows)}}
    desc = {"cells": {"%d,%d" % c: {k_: v_ for k_, v_ in x.items() if k_ not in ("tstar", "curve")} for c, x in sorted(cells.items())},
            "j_le_2": {j: {"n_clean": len([r for r in cfp["rows"] if r["clean"] and r["j"] == j]),
                           "share_window": mean(int(r["j"] - 1 <= r["ts"] <= r["j"]) for r in cfp["rows"] if r["clean"] and r["j"] == j)} for j in (1, 2)}}
    return reading, lab, desc


def readings_A(P, l0, c1, ev):
    o = {}; acc, pr, pl, fd, vs, stl = (P.get(x) for x in ("acc", "probe", "plant", "fd", "vswap", "stall"))
    A = (acc or {}).get("a") or {}
    w = bad_part(acc, "acc") or ("G-L0-failed band %s" % failed_in(l0, range(4, 9)) if failed_in(l0, range(4, 9)) else None)
    o["DEEP_L"] = rd(all(ge(A.get(D), 0.90) for D in range(4, 9)), w, acc_d2={D: A.get(D) for D in range(4, 9)})
    wp = bad_part(acc, "acc") or bad_part(pr, "probe") or bad_part(pl, "plant")
    # PROBE-GAIN-L
    gain = accp = p1 = p12 = None; band = (c1 or {}).get("band") or []
    w = ("C1 band unreadable: %s" % (c1 or {}).get("why") if not (c1 or {}).get("readable") else
         "EMPTY-C1-BAND (band %s; v1 §2.6)" % band if len(band) < 2 else wp)
    if w is None:
        accp, p1, p12 = mean(acc_items(acc, band)), probe_pooled(pr, 1, band), probe_pooled(pr, 12, band); gain = accp - p1
        if not pl["valid"].get(1):
            w = "PROBE-INVALID(1) (PLANT lin %s / xor %s < 0.90)" % (pl["plant"].get(1, {}).get("lin"), pl["plant"].get(1, {}).get("xor"))
        elif p12 < accp - 0.05 - EPS:
            w = "PROBE-INVALID (final state on the band: PROBE(12) %.4f < ACC %.4f - 0.05)" % (p12, accp)
    o["PROBE_GAIN_L"] = rd(ge(gain, 0.35), w, gain=gain, band=band, acc_band=accp, probe1_band=p1, probe12_band=p12)
    # SERIAL-ONSET-L
    ON, slope, never = {}, None, []
    w = wp or ("G-L0-failed band %s" % failed_in(l0, range(4, 9)) if failed_in(l0, range(4, 9)) else None)
    if wp is None:
        for D in range(4, 9):
            vals = {t: pr["acc"].get(t, {}).get(D) for t in PROBE_GRID}
            ON[D] = next((t for i, t in enumerate(PROBE_GRID) if all(ge(vals[u], 0.90) for u in PROBE_GRID[i:])), None)
        never = [D for D in ON if ON[D] is None]
        if not never:
            slope = ols(list(ON), [ON[D] for D in ON])
        inv = [t for t in (0, 1, 2) if not pl["valid"].get(t)]
        p12_48, acc48 = probe_pooled(pr, 12, range(4, 9)), mean(acc_items(acc, range(4, 9)))
        if w is None and inv:
            w = "PROBE-INVALID(%s)" % inv
        elif w is None and p12_48 < acc48 - 0.05 - EPS:
            w = "PROBE-INVALID (final state D 4-8: PROBE(12) %.4f < ACC %.4f - 0.05)" % (p12_48, acc48)
    o["SERIAL_ONSET_L"] = rd(bool(ON) and not never and ge(slope, 0.6) and ON[8] - ON[4] >= 3, w, ON=ON, slope=slope,
                             never_reached=never, span_8_4=(ON[8] - ON[4]) if ON and not never else None)
    # SERIAL-STEP-L
    arms = (vs or {}).get("arms") or {}
    if vs is not None and vs.get("missing"):
        arms = {}
    ii = arm_reading(arms.get("ii_prime"), range(4, 9), l0); ip = arm_reading(arms.get("i_prime"), range(5, 9), l0)
    ip["excluded_by_construction"] = {4: "no admissible t <= D - 2 (v1 §4.1 exposure table)"}
    o["SERIAL_STEP_L"] = rd(ii["met"] and ip["met"], None if (ii["readable"] and ip["readable"]) else
                            "arm unreadable: ii' %s; i' %s" % (ii["status"], ip["status"]), ii_prime=ii, i_prime=ip)
    # NO-CLOCK-L
    w = bad_part(stl, "stall") or ("G-L0-failed band %s" % failed_in(l0, range(4, 9)) if failed_in(l0, range(4, 9)) else None)
    if w is None and stl["n"] < 2000:
        w = "%d < 2,000 STALL items" % stl["n"]
    o["NO_CLOCK_L"] = rd(ge((stl or {}).get("recovery"), 0.9), w, recovery=(stl or {}).get("recovery"), n=(stl or {}).get("n"))
    # NO-PRECOMPUTE-L
    w = bad_part(fd, "fd") or ("G-L0-failed band %s" % failed_in(l0, range(6, 9)) if failed_in(l0, range(6, 9)) else None)
    cells = (fd or {}).get("cells") or {}; dec = {k_: v_ for k_, v_ in cells.items() if k_[2] >= 3}
    if w is None:
        inv = [m for m in ("lin", "mlp") if not (fd["valid"].get(m) or {}).get("ok")]
        if inv:
            w = "FD-INVALID (%s: fd_valid s3 / s4 or fd_plant < 0.90)" % inv
    o["NO_PRECOMPUTE_L"] = rd(bool(dec) and all(le(v_["auroc"], 0.65) for v_ in dec.values()), w,
                              max_auroc_k_ge3=max((v_["auroc"] for v_ in dec.values()), default=None),
                              cells_k_ge3={"%s t%d k%d" % k_: v_["auroc"] for k_, v_ in sorted(dec.items())},
                              descriptive_k_le_2={"%s t%d k%d" % k_: v_["auroc"] for k_, v_ in sorted(cells.items()) if k_[2] < 3},
                              fd_valid=(fd or {}).get("valid"))
    # CF-L
    o["CF_L"], cflab, cfdesc = cf_reading(P.get("cf"), l0)
    comps = ("DEEP_L", "PROBE_GAIN_L", "SERIAL_ONSET_L", "SERIAL_STEP_L", "CF_L", "NO_CLOCK_L", "NO_PRECOMPUTE_L")
    # labels (descriptive)
    iiD, per, rs = (arm_reading(arms.get(a), range(4, 9), l0) for a in ("ii", "person", "reset"))
    lab = {"ONESHOT_L": ge(pr["acc"].get(2, {}).get(8), 0.8) if wp is None else None,
           "LOSES_ONE_PASS": (o["DEEP_L"]["readable"] and not o["DEEP_L"]["met"] and o["PROBE_GAIN_L"]["readable"] and gain < 0.15)
           if o["PROBE_GAIN_L"]["readable"] else None,
           "COUNT_KEYED": (o["NO_CLOCK_L"]["readable"] and not o["NO_CLOCK_L"]["met"]) if o["NO_CLOCK_L"]["readable"] else None,
           "RECOMPUTE_L": bool(ii["readable"] and rs["readable"] and ii["own"] > 0.3 + EPS and ge(rs["own"], 0.9)),
           "ANCHOR_TAGGED": bool(ii["readable"] and rs["readable"] and ii["own"] > 0.5 + EPS and le(rs["own"], 0.75)),
           "HISTORY_READ": bool(ii["readable"] and iiD["readable"] and ge(iiD["own"], 0.5) and ge(ii["follow"], 0.80)),
           "ALL_PERSONS_SIGNATURE": bool(ii["readable"] and per["readable"] and ge(per["own"], 0.80) and ge(ii["follow"], 0.80)),
           "PRECOMPUTE_L": bool(o["NO_PRECOMPUTE_L"]["readable"] and not o["NO_PRECOMPUTE_L"]["met"]),
           "EMPTY_C1_BAND": bool((c1 or {}).get("readable") and len(band) < 2),
           "RESET_recovery": rs.get("own"), "PERSON_own": per.get("own"), "ii_own": iiD.get("own"), "ii_prime_t2_descriptive": (arms.get("ii_prime") or {}).get("t2_descriptive")}
    lab.update({k_: cflab.get(k_) for k_ in ("CF_CARRY", "CF_RECOMPUTE", "CF_FAST", "LAST_HOP_LATE")})
    sd = (ev or {}).get("slva_diag")
    lab["RNN_IN_FUNCTION"] = sd.get("rnn_in_function") if isinstance(sd, dict) else "not in the field contract (RESOLVED R21)"
    lab["HISTORY_USED"] = sd.get("history_used") if isinstance(sd, dict) else "not in the field contract (RESOLVED R21)"
    desc = {"arms_descriptive": {"ii": iiD, "person": per, "reset": rs}, "cf": {"labels": cflab, **cfdesc},
            "DEPTH_L_mean_acc_d2_9_12": mean(A.get(D) for D in range(9, 13)) if all(num(A.get(D)) for D in range(9, 13)) else None,
            "PLANT": (pl or {}).get("plant"), "PROBE_by_t_D": {t: {D: v_ for D, v_ in x.items()} for t, x in ((pr or {}).get("acc") or {}).items()},
            "LIT_C2_L": "UNINFORMATIVE on SYM (v1 §3.7c; D <= 16 fits K' 16)"}
    return o, comps, lab, desc


VB_WORDING = ("supervised serial latent reachability in a pass-global attention memory, on natural-language premises (ProsQA-SYM v1: person-agnostic cues "
              "excluded by construction, single-parent chases gated < 0.6, uBIDIR ½ and UDIST 1 by construction), C1 band D %s beyond DIRECT-SYM's one-pass "
              "boundary, one seed; PREMISE-SWAP and slot-local splices follow ≥ 0.80; hop-indexed consumption on ≥ 0.75 of clean exposed pairs (CONSUMPTION-FRONT; "
              "calibrated: requires ≈ 0.9 item-level serial share); no future-distance decodable at k ≥ 3 by PLANT-certified probes")
VB_NEVER = ["graph search", "no shortcut", "C2", "C4 (REACH targets are simulator states)", "unsupervised", "the model is serial"]


def mem_read(P):
    """MEM (v1 §4.1; SELFDEPTH_v1 §4.3 form, RESOLVED R8): KEEPS-SELF-STAB-L else MEM-TRADED-L (unreadable counts as MEM-TRADED-L)."""
    h = P.get("hist24")
    w = bad_part(h, "hist24")
    if w:
        return {"KEEPS_SELF_STAB_L": False, "MEM_TRADED_L": True, "why": w + " (unreadable = MEM-TRADED-L; RESOLVED R8)"}
    return {"KEEPS_SELF_STAB_L": h["KEEPS_SELF_STAB_L"], "MEM_TRADED_L": not h["KEEPS_SELF_STAB_L"], "STAB": h["STAB"], "gate": h["gate"], "CONTENT": h["CONTENT"]}


# ---------------------------------------------------------------- cell plumbing
SOFT_FINAL = ("desc:", "ctrl")


def unread(o, why):
    o.update(status="UNREAD", label="UNREAD (%s; never a pass)" % why)
    return o


def _gate_pend(miss, ctx, stage):
    if stage == "A":
        l0 = ctx["l0"]
        if l0 is None:
            miss["gate_l0"] = pname("gate_l0")
        elif l0["missing"]:
            miss["gate_l0"] = "%s (entries %s)" % (pname("gate_l0"), l0["missing"][:5])
    evm, key = ctx["evm" + stage], "evmem_" + stage
    if evm is None:
        miss[key] = pname(key)
    elif not card_ok(evm.get("card"), stage):
        miss.setdefault("mismatch", []).append("G-EVMEM card %s (stage %s card %s)" % (evm.get("card"), stage, CARD[stage]))


def _args_pend(miss, ad):
    if ad["status"] == "UNLISTED":
        miss["args_unlisted"] = ad["why"]
    elif ad["status"] == "MISSING":
        miss["args_ref"] = ad["why"]


def _gate_final(o, miss, k):
    """215-7: pending while anything registered is missing; at --final: mismatch -> VOID, hard misses -> FAILURE (rule b); soft misses read on."""
    if not miss:
        return None
    if k not in FINAL:
        o.update(status="PENDING", missing=sorted(str(v) for v in miss.values()), label=pend_label(miss))
        return o
    if "mismatch" in miss:
        o.update(status="FAIL", censored=True, label="VOID (mismatch %s; v1 §4.3 / §5.2; rule b)" % miss["mismatch"])
        return o
    hard = sorted(str(v) for key, v in miss.items() if not key.startswith(SOFT_FINAL))
    if hard:
        o.update(status="FAIL", censored=True, missing_at_final=hard, label="FAILURE (registered file / part / field missing at --final: %s; rule b)" % hard)
        return o
    o["missing_at_final_soft"] = sorted(str(v) for v in miss.values())
    return None


def _a_missing(k, ctx):
    d = CELL[k]; miss = {}
    for f in ("log.jsonl", "args.json", "eval_lvb.json"):
        if not has(d, f):
            miss[f] = "%s/%s" % (d, f)
    info = log_info(d)
    if info is not None and not info["n_steps"]:
        miss["log.jsonl"] = "%s/log.jsonl (no step records)" % d
    elif info is not None and not info["ce"]:
        miss["log.jsonl"] = "%s/log.jsonl (no plain_cand_ce records)" % d
    ev = jl(d, "eval_lvb.json"); P = {}
    if ev is not None:
        parts = REQ_PARTS[k]; pm = parts_missing(ev, PARTS_A, parts)
        for p in pm:
            miss[("desc:" if p in DESCRIPTIVE_A else "part:") + p] = "%s/eval_lvb.json part %s" % (d, p)
        P = parse_A(ev, [p for p in parts if p not in pm])
        for p, x in P.items():
            if x.get("missing"):
                miss[("desc:" if p in DESCRIPTIVE_A else "field:") + p] = "%s/eval_lvb.json %s fields %s" % (d, p, x["missing"][:4])
        mm, bad = meta_check(ev, "A", parts, ctx["evmA"], ctx["shaA"], dtype_parts={"labeller": LABEL_DTYPE})
        if mm:
            miss["meta"] = "%s/eval_lvb.json meta %s" % (d, mm[:6])
        if bad:
            miss.setdefault("mismatch", []).extend(bad)
    _gate_pend(miss, ctx, "A")
    ad = args_diff(k); _args_pend(miss, ad)
    return miss, ev, P, info, ad


def censors_A(k, info, ev, ad, extra=()):
    dead = info["last"] is None or info["last"] < FULL[k] or info["guard_stop"]
    ent = entry_prosqa(info["ce"]); noent = not ent["entered"]
    c = (ev or {}).get("contamination") or {}
    contam = info["self_reuse"] > 0 or any(num(v) and v > 0 for v in c.values())
    args_bad = ad["ok"] is not True
    why = [w for w, f in (("dead (last step %s < %d%s)" % (info["last"], FULL[k], ", guard stop" if info["guard_stop"] else ""), dead),
                          ("NO-ENTRY (plain_cand_ce never <= 0.35 in two consecutive 1k bins)", noent), ("CONTAMINATED", contam),
                          ("ARGS-DIFF %s" % ({x: ad.get(x) for x in ("shared_diffs", "unexpected_new_keys", "missing_keys", "required_bad") if ad.get(x)}
                                            or ad.get("why")), args_bad)) + tuple(extra) if f]
    return {"dead": dead, "last_step": info["last"], "guard_stop": info["guard_stop"], "NO_ENTRY": noent, "entry": ent, "CONTAMINATED": contam,
            "ARGS_DIFF": ad, "why": why, "censored": bool(why)}


def _direct_body(ev):
    """eval_direct.json: fields at the top level (LAYOUTS) or under 'direct' (Experiments' langvb_eval part_direct); meta stays at the top level."""
    if not isinstance(ev, dict):
        return None, None
    return (ev["direct"] if isinstance(ev.get("direct"), dict) else ev), ev


def direct_cell(ctx):
    """DIRECT-SYM (v1 §2.6): the C1 band and D*_fresh, computed by the scorer from eval_direct.json."""
    k = "direct"; d = CELL[k]; o = {"cell": d, "name": NAME[k]}
    if k in NOT_ADMITTED:
        o["C1"] = {"readable": False, "why": "DIRECT-SYM not admitted"}
        return unread(o, "not admitted")
    miss = {}
    for f in ("log.jsonl", "args.json", "eval_direct.json"):
        if not has(d, f):
            miss[f] = "%s/%s" % (d, f)
    info = log_info(d)
    if info is not None and (not info["n_steps"] or not info["ce"]):
        miss["log.jsonl"] = "%s/log.jsonl (no step / plain_cand_ce records)" % d
    body, ev = _direct_body(jl(d, "eval_direct.json"))
    if ev is not None:
        fm = [x for x in ("acc_by_D", "items", "c1_band", "d_star_fresh") if x not in body]
        fm += ["acc D%d" % D for D in D_TEST if not (isinstance((body.get("items") or {}).get(str(D)), list) and num((body.get("acc_by_D") or {}).get(str(D))))]
        if fm:
            miss["field:direct"] = "%s/eval_direct.json fields %s" % (d, fm[:6])
        mm, bad = meta_check(ev, "A", ("direct",), ctx["evmA"], ctx["shaA"])
        if mm:
            miss["meta"] = "%s/eval_direct.json meta %s" % (d, mm)
        if bad:
            miss.setdefault("mismatch", []).extend(bad)
    _gate_pend(miss, ctx, "A"); ad = args_diff(k); _args_pend(miss, ad)
    r = _gate_final(o, miss, k)
    if r:
        r["C1"] = {"readable": False, "why": "DIRECT-SYM %s" % r["label"]}
        return r
    acc, it = body["acc_by_D"], body["items"]
    bad = ["acc_by_D[%d] %s != mean(items) (n %d)" % (D, acc[str(D)], len(it[str(D)])) for D in D_TEST
           if len(it[str(D)]) != PER_D or abs(mean(it[str(D)]) - acc[str(D)]) > 1e-12]
    cen = censors_A(k, info, ev, ad); a = {D: acc[str(D)] for D in D_TEST}
    band = [D for D in range(5, 9) if all(le(a[x], 0.65) for x in range(D, 9))]
    dstar = max((D for D in range(3, 13) if a[D] > 0.65 + EPS), default=None)
    fb = failed_in(ctx["l0"], range(5, 9))
    why = ("DIRECT-SYM censored: %s" % cen["why"] if cen["censored"] else "eval_direct VOID: %s" % bad[:3] if bad else
           "G-L0-failed band %s in D 5-8" % fb if fb else None)
    o["C1"] = {"readable": why is None, "why": why, "band": band if why is None else [], "n_depths": len(band), "EMPTY_C1_BAND": why is None and len(band) < 2,
               "d_star_fresh": dstar, "acc_by_D": a, "file_c1_band": body.get("c1_band"), "file_d_star_fresh": body.get("d_star_fresh"),
               "flag_file_differs": sorted(body.get("c1_band") or []) != band or body.get("d_star_fresh") != dstar,
               "flag_followup_d_star_ge_6": bool(dstar is not None and dstar >= 6)}
    o.update(status="FAIL" if (cen["censored"] or bad) else "SCORED", censored=cen["censored"], censors=cen, void=bad or None)
    o["label"] = ("FAILURE (%s; rule b)" % "; ".join(cen["why"]) if cen["censored"] else "VOID (eval_direct consistency: %s)" % bad[:2] if bad else
                  "C1 band %s%s | D*_fresh %s" % (band, " = EMPTY-C1-BAND (< 2 depths: VB-C1-L not met)" if len(band) < 2 else "", dstar))
    return o


def a1_cell(ctx, direct):
    k = "a1"; d = CELL[k]; o = {"cell": d, "name": NAME[k]}
    if k in NOT_ADMITTED:
        return unread(o, "not admitted")
    miss, ev, P, info, ad = _a_missing(k, ctx)
    if not has(CELL["direct"], "eval_direct.json"):
        miss["eval_direct.json"] = "%s/eval_direct.json (C1 band; v1 §2.6)" % CELL["direct"]
    elif direct["status"] == "PENDING":
        miss["direct_cell"] = "DIRECT-SYM cell pending (C1 band; v1 §2.6)"
    r = _gate_final(o, miss, k)
    if r:
        return r
    c1 = direct.get("C1") or {"readable": False, "why": "DIRECT-SYM %s" % direct["status"]}
    return _score_A(k, o, ctx, ev, P, info, ad, c1)


def _score_A(k, o, ctx, ev, P, info, ad, c1, extra=()):
    cen = censors_A(k, info, ev, ad, extra)
    rs, comps, lab, desc = readings_A(P, ctx["l0"], c1, ev)
    vb = all(rs[c]["met"] for c in comps) and not cen["censored"]
    mem = mem_read(P)
    o.update(status="FAIL" if cen["censored"] else "SCORED", censored=cen["censored"], censors=cen, READINGS=rs, VB_C1_L=vb,
             VB_C1_L_components={c: rs[c]["status"] for c in comps}, LABELS=lab, DESCRIPTIVE=desc, MEM=mem, C1=c1,
             G_L0={"failed": (ctx["l0"] or {}).get("failed"), "code_bug": (ctx["l0"] or {}).get("code_bug"),
                   "flag_inconsistent": (ctx["l0"] or {}).get("flag_inconsistent")},
             CAUSAL_DEEP_L=(P.get("causal_deep") or {}).get("label", bad_part(P.get("causal_deep"), "causal_deep")),
             _acc=(P.get("acc") or {}).get("a"), _acc_void=bad_part(P.get("acc"), "acc"), _ak=(P.get("acc") or {}).get("ak"))
    o["WORDING"] = {"VB_C1_L": (VB_WORDING % c1.get("band")) if vb else None, "never": VB_NEVER,
                    "CF_L": "hop-indexed consumption on ≥ 0.75 of clean exposed pairs (calibrated: requires ≈ 0.9 item-level serial share)"
                    if rs["CF_L"]["met"] and not cen["censored"] else None}
    mech = [x for x in ("PRECOMPUTE_L", "CF_CARRY", "CF_RECOMPUTE", "CF_FAST", "RECOMPUTE_L", "ANCHOR_TAGGED", "HISTORY_READ", "ALL_PERSONS_SIGNATURE",
                        "ONESHOT_L", "LOSES_ONE_PASS", "COUNT_KEYED", "EMPTY_C1_BAND", "LAST_HOP_LATE") if lab.get(x) is True]
    o["label"] = ("FAILURE (%s; rule b)" % "; ".join(cen["why"]) if cen["censored"] else
                  ("VB-C1-L" if vb else "not VB-C1-L (%s)" % ", ".join("%s %s" % (c, rs[c]["status"]) for c in comps if not rs[c]["met"]))) + \
        " | labels %s | MEM %s" % (mech or "none", "KEEPS-SELF-STAB-L" if mem["KEEPS_SELF_STAB_L"] else "MEM-TRADED-L (later stages need the user's explicit OK)")
    return o


# ---------------------------------------------------------------- A2 inputs, GO, A2-CTRL, A2
def gate1_read(g):
    """GATE-1-L (v1 §2.4 / G-L15): per-item KEEP / label / gold per chain; yield, wrong labels and the N9 choice recomputed."""
    if g is None:
        return None
    m = g.get("meta") or {}; miss, bad, out = [], [], {}
    if not card_ok(m.get("card"), "A"):
        bad.append("card %s (stage A card %s)" % (m.get("card"), CARD["A"]))
    if m.get("dtype") != LABEL_DTYPE:
        bad.append("dtype %s != %s (v1 §2.4)" % (m.get("dtype"), LABEL_DTYPE))
    for ch in ("L-ROLL", "L-RR"):
        out[ch] = {}
        for D in range(9, 13):
            lst = ((g.get("items") or {}).get(ch) or {}).get(str(D))
            if not isinstance(lst, list) or not lst:
                miss.append("items %s D%d" % (ch, D)); continue
            kept = [x for x in lst if x.get("keep")]; wrong = [x for x in kept if x.get("label") != x.get("gold")]
            out[ch][D] = {"n": len(lst), "kept": len(kept), "yield": len(kept) / len(lst), "n_wrong": len(wrong)}
            fc = ((g.get("chains") or {}).get(ch) or {}).get(str(D)) or {}
            if (fc.get("kept") is not None and fc["kept"] != len(kept)) or (num(fc.get("yield")) and not same(fc["yield"], len(kept) / len(lst))):
                bad.append("chains %s D%d kept / yield != items" % (ch, D))
            for x in ("omit_rate", "false_in", "stop_lag"):
                out[ch][D][x] = fc.get(x)
    if g.get("choice") not in ("L-ROLL", "L-RR"):
        miss.append("choice")
    if miss:
        return {"missing": miss}
    y = {ch: (out[ch][11]["kept"] + out[ch][12]["kept"]) / (out[ch][11]["n"] + out[ch][12]["n"]) for ch in out}
    choice = "L-RR" if y["L-RR"] - y["L-ROLL"] >= 0.05 - EPS else "L-ROLL"
    if g["choice"] != choice:
        bad.append("choice %s != recomputed %s (v1 §2.4 / N9: L-RR only if its D 11-12 yield beats L-ROLL's by >= 0.05)" % (g["choice"], choice))
    wrong = sum(out[ch][D]["n_wrong"] for ch in out for D in out[ch])
    yok = all(out[choice][D]["yield"] >= 0.2 - EPS for D in range(9, 13))
    pi = {ch: 1 - sum(out[ch][D]["n_wrong"] for D in out[ch]) / max(1, sum(out[ch][D]["kept"] for D in out[ch])) for ch in out}
    return {"missing": [], "problems": bad, "by_chain": out, "yield_11_12": y, "choice": choice, "LABEL_BUG": wrong > 0, "n_wrong": wrong, "pi_label": pi,
            "yield_ok": yok, "GATE_1_L": not bad and wrong == 0 and yok,
            "why": [w for w, f in (("problems %s" % bad, bad), ("LABEL-BUG: %d wrong labels" % wrong, wrong), ("kept yield < 0.2 at some D 9-12", not yok)) if f]}


def pool_read(pm):
    """G-L20 POOL (v1 §2.4 / §7): POOL-SHORT, CONTAMINATED (signature hits, gold keys), other G-L20 failures."""
    if pm is None:
        return None
    miss = [x for x in ("kept_by_D", "draws_by_D", "key_set", "sig_hits", "gold_key_hits", "rng_replay_equal") if x not in pm]
    if miss:
        return {"missing": miss}
    kept = {D: (pm["kept_by_D"] or {}).get(str(D)) for D in range(9, 13)}; draws = {D: (pm["draws_by_D"] or {}).get(str(D)) for D in range(9, 13)}
    short = [D for D in kept if not num(kept[D]) or kept[D] < POOL_KEPT]; at_cap = [D for D in short if num(draws[D]) and draws[D] >= POOL_CAP]
    sig = pm["sig_hits"]; sig_n = sum(v for v in sig.values() if num(v)) if isinstance(sig, dict) else (sig if num(sig) else 1)
    ks = set(pm["key_set"] or []); demand = pm.get("demand_replay", POOL_DEMAND); total = sum(v for v in kept.values() if num(v))
    contam = sig_n > 0 or (pm["gold_key_hits"] or 0) > 0 or bool(ks - POOL_KEYS)
    gl20 = [w for w, f in (("total kept %s < 1.1 x demand %s" % (total, demand), total < 1.1 * demand - EPS), ("key set %s != registered" % sorted(ks), ks != POOL_KEYS),
                           ("rng_s replay != v1_pool count", pm["rng_replay_equal"] is not True), ("demand_replay %s != 383,488 (seed 1)" % demand, demand != POOL_DEMAND)) if f]
    return {"missing": [], "kept_by_D": kept, "draws_by_D": draws, "POOL_SHORT": bool(short), "short": short, "short_at_cap": at_cap, "CONTAMINATED": contam,
            "sig_hits": sig, "gold_key_hits": pm["gold_key_hits"], "G_L20_fail": gl20}


def a2_go(a1, g1r, pr):
    """v1 §2.9 item 3: A2 GO = A1 uncensored and DEEP-L and NO-CLOCK-L and GATE-1-L (and the user's OK if A1 reads MEM-TRADED-L); then pool -> G-L20 ->
    A2 and A2-CTRL together (POOL-SHORT -> neither is launched: UNREAD)."""
    o = {"go": None, "why": [], "components": {}}
    if "a2" in NOT_ADMITTED or "ctrl" in NOT_ADMITTED:
        o.update(go=False, why=["A2 / A2-CTRL not admitted (launched only together)"]); return o
    if a1["status"] == "PENDING":
        o["why"] = ["A1 pending"]; return o
    if a1["status"] == "UNREAD":
        o.update(go=False, why=["A1 UNREAD"]); return o
    rs = a1.get("READINGS") or {}
    c = {"A1_uncensored": a1["status"] == "SCORED" and not a1.get("censored"), "DEEP_L": (rs.get("DEEP_L") or {}).get("met", False),
         "NO_CLOCK_L": (rs.get("NO_CLOCK_L") or {}).get("met", False),
         "GATE_1_L": None if (g1r is None or g1r.get("missing")) else g1r["GATE_1_L"]}
    mt = bool((a1.get("MEM") or {}).get("MEM_TRADED_L"))
    c["USER_OK_if_A1_MEM_TRADED"] = (not mt) or ("A2" in USER_OK)
    if pr is None or pr.get("missing"):
        c["POOL_NOT_SHORT"] = None
    elif pr["POOL_SHORT"]:
        c["POOL_NOT_SHORT"] = False if (sorted(pr["short_at_cap"]) == sorted(pr["short"]) or "a2" in FINAL) else None
    else:
        c["POOL_NOT_SHORT"] = True
    o["components"] = c; o["A1_MEM_TRADED_L"] = mt; o["POOL"] = pr
    if g1r is not None and not g1r.get("missing"):
        o["GATE_1_L_detail"] = {x: g1r.get(x) for x in ("choice", "yield_11_12", "LABEL_BUG", "n_wrong", "yield_ok", "why")}
    if any(v is False for v in c.values()):
        o.update(go=False, why=[x for x, v in c.items() if v is False] + (["A1 reads MEM-TRADED-L: the user's explicit OK is required (L6)"] if mt and "A2" not in USER_OK else [])
                 + (["GATE-1-L: %s" % g1r["why"]] if c["GATE_1_L"] is False else [])
                 + (["POOL-SHORT: kept < %d at D %s at the draw cap (v1 §2.4)" % (POOL_KEPT, pr["short"])] if c["POOL_NOT_SHORT"] is False else []))
    elif any(v is None for v in c.values()):
        o["why"] = [x for x, v in c.items() if v is None]
    else:
        o["go"] = True
    return o


def ctrl_cell(ctx, a2go):
    k = "ctrl"; d = CELL[k]; o = {"cell": d, "name": NAME[k]}
    if k in NOT_ADMITTED:
        return unread(o, "not admitted")
    if a2go["go"] is False:
        return unread(o, "A2 not GO (%s); A2-CTRL launches only with A2" % a2go["why"])
    if a2go["go"] is None and k in FINAL and not (has(d, "log.jsonl") or has(d, "args.json")):
        return unread(o, "A2 GO not established (%s); never launched" % a2go["why"])
    miss, ev, P, info, ad = _a_missing(k, ctx)
    if a2go["go"] is None:
        miss["a2_go"] = "A2 GO undecided (%s)" % a2go["why"]
    r = _gate_final(o, miss, k)
    if r:
        return r
    cen = censors_A(k, info, ev, ad); acc = P["acc"]; w = bad_part(acc, "acc"); A = acc.get("a") or {}
    o.update(status="FAIL" if cen["censored"] else "SCORED", censored=cen["censored"], censors=cen, acc_void=w,
             M_13_16=None if w else mean(A[D] for D in range(13, 17)), min_9_12=None if w else min(A[D] for D in range(9, 13)), MEM=mem_read(P))
    o["label"] = ("FAILURE (%s; rule b)" % "; ".join(cen["why"]) if cen["censored"] else "VOID acc (%s)" % w if w else
                  "A2-CTRL scored | mean acc_d2 D 13-16 %.4f | min D 9-12 %.4f | MEM %s" % (o["M_13_16"], o["min_9_12"],
                                                                                       "KEEPS-SELF-STAB-L" if o["MEM"]["KEEPS_SELF_STAB_L"] else "MEM-TRADED-L"))
    return o


SD_SELF_WORDING = ("latent reachability depth extended from 8 to 12 by training on the model's own SOUND-CLOSURE-verified chains (labels gold-equal by "
                   "construction, π = 1; no gold state beyond D 8; labeller %s; in support)")
SD_C2_WORDING = ("depth extrapolation past the self-labelled range (D 13–16, untouched SYM v1 items; margin grew vs the +4-translated A1 curve and beat the "
                 "matched plain fine-tune by ≥ 0.10; C2 proxy; one lineage: an instance)")


def a2_cell(ctx, a1, a2go, ctrl, g1r, direct, pr):
    k = "a2"; d = CELL[k]; o = {"cell": d, "name": NAME[k]}
    if k in NOT_ADMITTED:
        return unread(o, "not admitted")
    launched = has(d, "log.jsonl") or has(d, "args.json")
    if a2go["go"] is False:
        if g1r and not g1r.get("missing") and g1r.get("LABEL_BUG") and launched:
            o.update(status="FAIL", censored=True, label="VOID (LABEL-BUG at GATE-1-L: %d wrong labels; A2 voided and re-gated; rule b)" % g1r["n_wrong"])
            return o
        o["launched_despite_no_GO"] = launched; o["POOL"] = pr
        return unread(o, "A2 not GO: %s" % a2go["why"])
    miss, ev, P, info, ad = _a_missing(k, ctx)
    if a2go["go"] is None:
        miss["a2_go"] = "A2 GO undecided (%s)" % a2go["why"]
    if pr is None:
        miss["pool_meta.json"] = pname("pool")
    elif pr["missing"]:
        miss["pool_meta.json"] = "%s fields %s" % (pname("pool"), pr["missing"])
    elif pr["POOL_SHORT"]:
        miss["pool"] = "pool incomplete (kept < %d, draws < %d at D %s)" % (POOL_KEPT, POOL_CAP, [D for D in pr["short"] if D not in pr["short_at_cap"]])
    if g1r is None:
        miss["eval_gate1.json"] = pname("gate1")
    elif g1r.get("missing"):
        miss["eval_gate1.json"] = "%s fields %s" % (pname("gate1"), g1r["missing"][:4])
    if not has(CELL["ctrl"], "eval_lvb.json"):
        miss["ctrl"] = "%s/eval_lvb.json (A2-CTRL, for SD-C2-L)" % CELL["ctrl"]
    elif ctrl["status"] == "PENDING":
        miss["ctrl"] = "A2-CTRL cell pending (SD-C2-L)"
    if a1["status"] == "PENDING":
        miss["a1"] = "A1 pending (MARGIN-GROWS-L bar)"
    r = _gate_final(o, miss, k)
    if r:
        return r
    lb = P.get("labeller") or {}; bug = bool(lb.get("LABEL_BUG") or g1r.get("LABEL_BUG"))
    if bug:
        o.update(status="FAIL", censored=True, label="VOID (LABEL-BUG: label_audit n_wrong %s, GATE-1-L wrong %s; A2 voided and re-gated; rule b)" % (
            (lb.get("label_audit") or {}).get("n_wrong"), g1r.get("n_wrong")))
        return o
    extra = (("CONTAMINATED (pool: sig_hits %s, gold_key_hits %s)" % (pr["sig_hits"], pr["gold_key_hits"]), pr["CONTAMINATED"]),
             ("G-L20 failed: %s" % pr["G_L20_fail"], bool(pr["G_L20_fail"])))
    c1 = direct.get("C1") or {"readable": False, "why": "DIRECT-SYM %s" % direct["status"]}
    _score_A(k, o, ctx, ev, P, info, ad, c1, extra)
    cen = o["censors"]; acc = P["acc"]; w = bad_part(acc, "acc"); A = acc.get("a") or {}; AK = acc.get("ak") or {}

    def wb(Ds):                                                             # v1 §4: a band that fails G-L0 enters no reading
        fb = failed_in(ctx["l0"], Ds)
        return w or ("G-L0-failed band %s" % fb if fb else None)
    absorb = rd(not w and min(A[D] for D in range(9, 13)) >= 0.95 - EPS, wb(range(9, 13)), min_9_12=None if w else min(A[D] for D in range(9, 13)))
    length = rd(not w and all(A[D] >= 0.99 - EPS for D in range(3, 9)), wb(range(3, 9)))
    k24 = rd(not w and min(AK["24"][str(D)] for D in range(3, 9)) >= 0.97 - EPS, wb(range(3, 9)),
             min_k24_3_8=None if w else min(AK["24"][str(D)] for D in range(3, 9)))
    sd_self = bool(absorb["met"] and o["VB_C1_L"] and length["met"] and k24["met"]) and not cen["censored"]
    M = None if w else mean(A[D] for D in range(13, 17))
    dbs = rd(ge(M, 0.80), wb(range(13, 17)), M_13_16=M)
    a1A = a1.get("_acc") or {}; a1w = a1.get("_acc_void") or (None if all(num(a1A.get(D)) for D in range(9, 13)) else "A1 acc part unavailable")
    a1w = a1w or ("G-L0-failed band %s" % failed_in(ctx["l0"], range(9, 13)) if failed_in(ctx["l0"], range(9, 13)) else None)
    bar = None if a1w else mean(a1A[D] for D in range(9, 13)) + 0.10
    mg = rd(not w and ge(M, bar), wb(range(13, 17)) or (("MARGIN bar unreadable: %s" % a1w) if a1w else None), M_13_16=M, bar_A1_9_12_plus_010=bar)
    if ctrl["status"] == "SCORED" and not ctrl.get("censored") and not ctrl.get("acc_void"):
        bc = rd(not w and ge(M - ctrl["M_13_16"], 0.10), wb(range(13, 17)), M_A2=M, M_CTRL=ctrl["M_13_16"], delta=None if w else M - ctrl["M_13_16"])
    else:
        bc = rd(False, "A2-CTRL %s (%s): SD-C2-L not met (v1 §4.1; RESOLVED R9)" % (ctrl["status"], ctrl.get("label")))
    sd_c2 = bool(sd_self and dbs["met"] and mg["met"] and bc["met"])
    lab = lb.get("label_audit") or {}
    o.update(A2_READINGS={"ABSORB_L": absorb, "LENGTH_L": length, "K24_STABLE": k24, "DEPTH_BEYOND_SELF_L": dbs, "MARGIN_GROWS_L": mg, "BEATS_CTRL_L": bc},
             SD_SELF_L=sd_self, SD_C2_L=sd_c2, POOL=pr, GATE_1_L={x: g1r.get(x) for x in ("choice", "yield_11_12", "pi_label", "by_chain", "GATE_1_L")},
             BESIDE={"CAUSAL_DEEP_L": o.get("CAUSAL_DEEP_L"), "MEM": o["MEM"],
                     "FALSE_IN_STOP_LAG": {D: {x: g1r["by_chain"][g1r["choice"]][D].get(x) for x in ("false_in", "stop_lag", "omit_rate")} for D in range(9, 13)},
                     "MEMO_GAP_L": {D: (v.get("acc_pool") - v.get("acc_fresh")) for D, v in (lb.get("memo") or {}).items()
                                    if num(v.get("acc_pool")) and num(v.get("acc_fresh"))},
                     "kept_vs_drawn": lb.get("kept_vs_drawn"), "label_audit": lab, "label_audit_flag": lb.get("n_audit_flag"),
                     "ABSORB_vs_CTRL_9_12": {"A2_min": absorb.get("min_9_12"), "CTRL_min": ctrl.get("min_9_12")},
                     "LIT_C2_L": "UNINFORMATIVE on SYM (v1 §3.7c)", "flag_gate1_copy_choice_differs": lb.get("gate1_copy_choice") not in (None, g1r["choice"])})
    o["WORDING"]["SD_SELF_L"] = (SD_SELF_WORDING % g1r["choice"]) if sd_self else None
    o["WORDING"]["SD_C2_L"] = (SD_C2_WORDING + " | MEM %s | CAUSAL-DEEP-L %s | LIT-C2-L UNINFORMATIVE" % (
        "KEEPS-SELF-STAB-L" if o["MEM"]["KEEPS_SELF_STAB_L"] else "MEM-TRADED-L", o.get("CAUSAL_DEEP_L"))) if sd_c2 else None
    o["WORDING"]["never"] = VB_NEVER + ["literal C2"]
    o["label"] = ("FAILURE (%s; rule b)" % "; ".join(cen["why"]) if cen["censored"] else
                  "SD-C2-L" if sd_c2 else "SD-SELF-L (in support; not C2)" if sd_self else "not SD-SELF-L") + \
        " | VB-C1-L %s | ABSORB %s | D13-16 %s | MARGIN %s | BEATS-CTRL %s | MEM %s" % (
            o["VB_C1_L"], absorb["status"], dbs["status"], mg["status"], bc["status"], "KEEPS" if o["MEM"]["KEEPS_SELF_STAB_L"] else "MEM-TRADED-L")
    return o


# ---------------------------------------------------------------- Stage B (v1 §2.7, §4.2, Addendum 1)
def b_go(a1, a2, ctrl):
    """v1 §2.9 item 4: B GO = A1 reads VB-C1-L and (the user's explicit OK if any scored LANG cell reads MEM-TRADED-L; RESOLVED R18)."""
    o = {"go": None, "why": [], "components": {}}
    if "b1" in NOT_ADMITTED:
        o.update(go=False, why=["B1 not admitted (family remainder / admission)"]); return o
    if a1["status"] == "PENDING":
        o["why"] = ["A1 pending"]; return o
    if a1["status"] != "SCORED":
        o.update(go=False, why=["A1 %s (%s)" % (a1["status"], a1.get("label"))]); return o
    mem = [NAME[c] for c, x in (("a1", a1), ("a2", a2), ("ctrl", ctrl)) if x.get("status") in ("SCORED", "FAIL") and (x.get("MEM") or {}).get("MEM_TRADED_L")]
    ok = (not mem) or ("B1" in USER_OK)
    o.update(go=bool(a1["VB_C1_L"] and ok), components={"A1_VB_C1_L": a1["VB_C1_L"], "USER_OK_if_MEM_TRADED": ok}, MEM_TRADED_cells=mem)
    o["why"] = ([] if a1["VB_C1_L"] else ["A1 not VB-C1-L (D-13)"]) + ([] if ok else ["MEM-TRADED-L in %s: the user's explicit OK is required (L6)" % mem])
    return o


def interim_read(x):
    """B0-6 interim GO (v1 §2.7): GO iff acc_fixed14(valid) - PROBE(1)(valid) >= 0.05 and FD-MLP(1, k >= 3) < 0.8 FD-MLP(k, k) pooled (recomputed)."""
    if x is None:
        return None
    f = ("acc_fixed14_valid", "probe1_valid", "fd_mlp_1k_ge3", "fd_mlp_kk_ge3")
    miss = [y for y in f if not num(x.get(y))] + ([] if isinstance(x.get("GO"), bool) else ["GO"])
    if miss:
        return {"missing": miss}
    go = x["acc_fixed14_valid"] - x["probe1_valid"] >= 0.05 - EPS and x["fd_mlp_1k_ge3"] < 0.8 * x["fd_mlp_kk_ge3"] - EPS
    m = x.get("meta") or {}
    bad = [] if card_ok(m.get("card"), "B") else ["interim card %s" % m.get("card")]
    if m.get("dtype", DTYPE) != DTYPE:
        bad.append("interim dtype %s" % m.get("dtype"))
    return {"missing": [], "mismatch": bad, "GO": go, "file_GO": x["GO"], "flag_file_differs": go != x["GO"], **{y: x[y] for y in f}, "step": x.get("step")}


def b02_read(b, ref):
    """B0-2 (v1 §7): CODI-20 re-score on the Stage B card; per item == gsm_codi_20ep_s1/eval_items.json (0.3791)."""
    if b is None or ref is None:
        return None
    it = b.get("items"); m = b.get("meta") or {}
    if not isinstance(it, list) or len(it) != N_GSM or not b.get("items_sha"):
        return {"missing": ["B0-2 items (1319) / items_sha"]}
    bad = [] if card_ok(m.get("card"), "B") else ["B0-2 card %s" % m.get("card")]
    if m.get("dtype") is not None and m["dtype"] != DTYPE:
        bad.append("B0-2 dtype %s" % m["dtype"])
    ok = [int(bool(x.get("ok"))) for x in it]; rok = [int(bool(x.get("ok"))) for x in ref]
    diff = [i for i in range(N_GSM) if ok[i] != rok[i]]
    return {"missing": [], "mismatch": bad, "ok": ok, "acc": mean(ok), "items_sha": b["items_sha"], "B0_2_pass": not diff and len(ref) == N_GSM,
            "n_item_diffs": len(diff), "first_diffs": diff[:10]}


def p_gsm_acc(ev):
    it = ev.get("items")
    if not isinstance(it, list) or len(it) != N_GSM:
        return {"missing": ["acc items (%d)" % N_GSM]}
    miss = sorted(set("acc item field %s" % f for x in it for f in ("ok", "m") if f not in x))
    if miss:
        return {"missing": miss}
    bad = ["items[%d].i = %s" % (i, x.get("i")) for i, x in enumerate(it) if x.get("i", i) != i][:5]
    ok = [int(bool(x["ok"])) for x in it]; ms = [x["m"] for x in it]; acc = mean(ok)
    if not same(acc, ev.get("acc_fixed14"), 1e-12):
        bad.append("acc_fixed14 %s != mean(items.ok) %s" % (ev.get("acc_fixed14"), acc))
    for m_, v in (ev.get("by_m") or {}).items():
        val = v.get("acc") if isinstance(v, dict) else v; sel = [o_ for o_, mm in zip(ok, ms) if str(mm) == str(m_)]
        if sel and num(val) and abs(mean(sel) - val) > 1e-12:
            bad.append("by_m[%s] %s != mean(items) %s" % (m_, val, mean(sel)))
    band = [i for i, mm in enumerate(ms) if num(mm) and mm >= 3]
    return {"missing": [], "void": bad or None, "ok": ok, "m": ms, "m_bin": [x.get("m_bin") for x in it], "band": band, "acc": acc,
            "acc_band": mean(ok[i] for i in band), "n_band": len(band), "flag_n_band_ne_879": len(band) != 879}


def p_gsm_probe(ev, acc):
    P = ev.get("probe") or {}; miss, bad, band_acc = [], [], {}
    for t in GSM_PROBE_GRID:
        e = P.get(str(t)); it = (e or {}).get("items")
        if not isinstance(it, list) or len(it) != N_GSM:
            miss.append("probe t%d items (%d)" % (t, N_GSM)); continue
        it = [int(bool(x)) for x in it]; band_acc[t] = mean(it[i] for i in acc["band"])
        for m_, v in ((e.get("acc_by_m") or {}).items()):
            sel = [o_ for o_, mm in zip(it, acc["m"]) if str(mm) == str(m_)]
            if sel and num(v) and abs(mean(sel) - v) > 1e-12:
                bad.append("probe t%d acc_by_m[%s] != mean(items)" % (t, m_))
    pl = ev.get("plant_gsm") or {}
    lin = (pl.get("lin") or {}).get("acc") if isinstance(pl.get("lin"), dict) else pl.get("lin")
    xor = (pl.get("xor") or {}).get("acc") if isinstance(pl.get("xor"), dict) else pl.get("xor")
    if not (num(lin) and num(xor)):
        miss.append("plant_gsm lin / xor")
    if not isinstance(ev.get("probe_valid_gsm"), bool):
        miss.append("probe_valid_gsm")
    if miss:
        return {"missing": miss}
    if not num(acc.get("acc_band")):                                       # acc part absent: the cell is pending / FAILURE on that already
        return {"missing": [], "void": None, "band_acc": band_acc, "PROBE_VALID_GSM": None, "PLANT_GSM": ge(lin, 0.90) and ge(xor, 0.90), "plant": {"lin": lin, "xor": xor}}
    pv = band_acc[14] >= acc["acc_band"] - 0.03 - EPS
    if ev["probe_valid_gsm"] != pv:
        bad.append("probe_valid_gsm %s != recomputed %s" % (ev["probe_valid_gsm"], pv))
    return {"missing": [], "void": bad or None, "band_acc": band_acc, "PROBE_VALID_GSM": pv, "PLANT_GSM": ge(lin, 0.90) and ge(xor, 0.90), "plant": {"lin": lin, "xor": xor}}


def p_gsm_future(ev):
    F = ev.get("future") or {}; out, miss = {}, []
    for model in ("lin", "mlp"):
        G = F.get(model) or {}
        t1 = [((G.get("1") or {}).get(str(k)) or {}) for k in range(3, 13)]; kk = [((G.get(str(k)) or {}).get(str(k)) or {}) for k in range(3, 13)]
        if not (t1[0] and kk[0]):
            miss.append("future %s (1, 3) / (3, 3)" % model); continue

        def pool(es):
            es = [e for e in es if num(e.get("acc")) and num(e.get("n")) and e["n"] > 0]
            n = sum(e["n"] for e in es)
            return (sum(e["acc"] * e["n"] for e in es) / n) if n else None, n
        f1, n1 = pool(t1); fk, nk = pool(kk)
        out[model] = {"FD_1_k_ge3": f1, "n_1": n1, "FD_kk_ge3": fk, "n_kk": nk, "k_absent": [k for k, e in zip(range(3, 13), t1) if not e]}
    return {"missing": miss, "void": None, "by_model": out}


def _gsm_pairs(arm, A, acc, tol=1e-4):
    pairs = (A or {}).get("pairs")
    if not isinstance(pairs, list):
        return {"missing": ["vswap.%s.pairs" % arm]}
    miss = sorted(set("vswap.%s pair field %s" % (arm, f) for p in pairs for f in ("base", "step", "m_bin", "ok_rec", "ok_don", "pos_ok") if f not in p))
    if miss:
        return {"missing": miss}
    rows = []
    for p in pairs:
        x = fo(p, tol)
        if x is None:
            return {"missing": ["vswap.%s pair y_hyb / y_rec / y_don" % arm]}
        exp = p["pos_ok"] is True and num(p["step"]) and num(p["m_bin"]) and 2 <= p["step"] <= p["m_bin"] - 1
        b = p["base"]; inband = isinstance(b, int) and 0 <= b < N_GSM and num(acc["m"][b]) and acc["m"][b] >= 3
        rows.append({"base": b, "exp": exp, "clean": exp and bool(p["ok_rec"]) and bool(p["ok_don"]), "follow": x[0], "own": x[1], "band": inband})
    return {"missing": [], "rows": rows, "n_candidates": len(pairs)}


def gsm_arm(arm, A, acc):
    """SERIAL-STEP-GSM arm (Addendum 1 item 6): base-item-clustered follow / own over clean pairs of band (m >= 3) base items; >= 100 clean pairs and
    >= GSM_MIN_BASE distinct base items with >= 1 clean pair, else UNREADABLE; cluster-bootstrap 95 % CIs."""
    x = _gsm_pairs(arm, A, acc)
    if x.get("missing"):
        return x
    rows = x["rows"]; ex = [r for r in rows if r["exp"]]; cl = [r for r in rows if r["clean"]]
    pl = {"n_candidates": x["n_candidates"], "n_exposed": len(ex), "n_clean": len(cl), "follow": mean(r["follow"] for r in cl), "own": mean(r["own"] for r in cl),
          "neither": mean(1 - r["follow"] - r["own"] for r in cl)}
    bad = ["%s %s != recomputed %s" % (f, A.get(f), pl[f]) for f in ("n_candidates", "n_exposed", "n_clean") if A.get(f) != pl[f]]
    dec = [r for r in cl if r["band"]]; gf, gw = {}, {}
    for r in dec:
        gf.setdefault(r["base"], []).append(r["follow"]); gw.setdefault(r["base"], []).append(r["own"])
    cf_, cw = cluster_mean(gf), cluster_mean(gw)
    for f, cv in (("follow", cf_), ("own", cw)):
        if not same(A.get(f), pl[f]) and not same(A.get(f), cv):
            bad.append("%s %s != recomputed (pair-level %s / clustered %s)" % (f, A.get(f), pl[f], cv))
    return {"missing": [], "void": bad or None, "pair_level": pl, "n_clean_band": len(dec), "n_base": len(gf), "follow": cf_, "own": cw,
            "follow_ci95": cluster_boot(gf), "own_ci95": cluster_boot(gw), "pair_level_band": {"follow": mean(r["follow"] for r in dec), "own": mean(r["own"] for r in dec)}}


def gsm_arm_reading(x):
    w = bad_part(x, "vswap arm")
    if w:
        return rd(False, w)
    v = {k_: x[k_] for k_ in ("n_clean_band", "n_base", "follow", "own", "follow_ci95", "own_ci95", "pair_level", "pair_level_band")}
    if x["n_clean_band"] < 100:
        return rd(False, "%d < 100 exposed clean pairs (m >= 3)" % x["n_clean_band"], **v)
    if x["n_base"] < GSM_MIN_BASE:
        return rd(False, "%d < %d distinct base items with >= 1 clean pair (Addendum 1 item 6)" % (x["n_base"], GSM_MIN_BASE), **v)
    return rd(ge(x["follow"], 0.6) and le(x["own"], 0.3), None, **v)


def p_gsm_stall(ev):
    S = ev.get("stall") or {}; it = S.get("items")
    if not isinstance(it, list) or not it or not num(S.get("recovery")) or any(not isinstance(x, dict) or "stalled" not in x or "intact" not in x for x in it):
        return {"missing": ["stall {recovery, items [{i, stalled, intact}]}"]}
    st = [int(bool(x["stalled"])) for x in it]; inn = [int(bool(x["intact"])) for x in it]
    rec = mean(st) / mean(inn) if mean(inn) else None
    return {"missing": [], "void": None if same(rec, S["recovery"]) else ["recovery %s != recomputed %s" % (S["recovery"], rec)], "recovery": rec, "n": len(it)}


def p_gsm_stepcap(ev):
    S = ev.get("stepcap") or {}
    if num(S.get("pooled")):
        return {"missing": [], "pooled": S["pooled"]}
    bt = [v for v in (S.get("by_t") or {}).values() if isinstance(v, dict) and num(v.get("acc")) and num(v.get("n")) and v["n"] > 0]
    if not bt:
        return {"missing": ["stepcap pooled / by_t"]}
    return {"missing": [], "pooled": sum(v["acc"] * v["n"] for v in bt) / sum(v["n"] for v in bt)}


def p_gsm_cf(ev, acc):
    C = ev.get("cf_gsm") or {}; pairs = C.get("pairs")
    if not isinstance(pairs, list):
        return {"missing": ["cf_gsm.pairs"]}
    miss = sorted(set("cf_gsm pair field %s" % f for p in pairs for f in ("base", "j", "m_bin", "answers_by_t", "pos_ok", "flip_ok") if f not in p))
    if miss:
        return {"missing": miss}
    rows, cells, bad = [], {}, []
    for p in pairs:
        ans = p["answers_by_t"]
        if not isinstance(ans, list) or len(ans) != 15:
            return {"missing": ["cf_gsm answers_by_t (t 0..14)"]}
        ans = [int(bool(a)) for a in ans]
        okr = bool(p["ok_rec"]) if "ok_rec" in p else ans[14] == 1; okd = bool(p["ok_don"]) if "ok_don" in p else ans[0] == 0
        exp = p["pos_ok"] is True and p["flip_ok"] is True; b = p["base"]
        inband = isinstance(b, int) and 0 <= b < N_GSM and num(acc["m"][b]) and acc["m"][b] >= 3
        r = {"base": b, "j": p["j"], "m_bin": p["m_bin"], "exp": exp, "clean": exp and okr and okd, "ts": crossover(ans, 14), "band": inband}
        rows.append(r); c = cells.setdefault(p["j"], {"cand": 0, "exp": 0, "clean": []}); c["cand"] += 1; c["exp"] += int(exp)
        if r["clean"]:
            c["clean"].append(r["ts"])
    for j, c in cells.items():
        f = C.get(str(j))
        if not isinstance(f, dict) or f.get("n_candidates") != c["cand"] or f.get("n_exposed") != c["exp"] or f.get("n_clean") != len(c["clean"]) or (
                f.get("tstar") is not None and sorted(f["tstar"]) != sorted(c["clean"])):
            bad.append("cf_gsm[%s] counts / tstar != recomputed" % j)
    return {"missing": [], "void": bad or None, "rows": rows}


def cf_gsm_reading(x):
    """CF-GSM (v1 §4.2 + Addendum 1 item 6; RESOLVED R13 / R14 / R16)."""
    w = bad_part(x, "cf_gsm")
    if w:
        return rd(False, w), {"LATE_FUSION_GSM": None, "LAST_STEP_LATE": None}
    rows = x["rows"]; dec = [r for r in rows if r["clean"] and r["band"] and 3 <= r["j"] <= r["m_bin"] - 1]
    g = {}; pts = {}
    for r in dec:
        g.setdefault(r["base"], []).append(int(r["j"] - 1 <= r["ts"] <= r["j"])); pts.setdefault((r["base"], r["j"]), []).append(r["ts"])
    share = cluster_mean(g); keys = sorted(pts)
    rho = spearman([mean(pts[q]) for q in keys], [q[1] for q in keys]) if len(keys) >= 2 else None
    v = dict(n_clean=len(dec), n_base=len(g), share_window_clustered=share, share_ci95=cluster_boot(g), spearman_cluster_points=rho, n_points=len(keys),
             pair_level={"share_window": mean(int(r["j"] - 1 <= r["ts"] <= r["j"]) for r in dec),
                         "spearman": spearman([r["ts"] for r in dec], [r["j"] for r in dec]) if len(dec) >= 2 else None})
    w = ("%d < 100 exposed clean pairs" % len(dec) if len(dec) < 100 else
         "%d < %d distinct exposed base steps (Addendum 1 item 6 as ruled in note 218-5: CF-GSM minimum 30)"
         % (len(g), CF_GSM_MIN_BASE) if len(g) < CF_GSM_MIN_BASE else "Spearman undefined" if rho is None else None)
    allc = [r["ts"] for r in rows if r["clean"]]; last = [r["ts"] for r in rows if r["clean"] and r["j"] == r["m_bin"]]
    lab = {"LATE_FUSION_GSM": ge(median(allc), 12) if allc else None, "late_fusion_median_tstar": median(allc),
           "LAST_STEP_LATE": ge(median(last), 12) if last else None, "last_step_median_tstar": median(last), "n_last_step": len(last)}
    return rd(ge(share, 0.6) and ge(rho, 0.6), w, **v), lab


def codi_read(ev, acc, b02):
    """accuracy vs CODI-20 (v1 §4.2 / §6): paired on the 1,319 items vs B0-2's re-score; exact one-sided McNemar."""
    if b02 is None or b02.get("missing"):
        return rd(False, "B0-2 unavailable")
    if not b02["B0_2_pass"]:
        return rd(False, "B0-2 failed: %d items differ from gsm_codi_20ep_s1/eval_items.json" % b02["n_item_diffs"])
    a, c_ = acc["ok"], b02["ok"]; n = len(a)
    b = sum(1 for x, y in zip(a, c_) if x and not y); c = sum(1 for x, y in zip(a, c_) if y and not x)
    delta = (b - c) / n; p = mcnemar_one_sided(b, c); se = math.sqrt(max(0.0, (b + c) / n - delta ** 2) / n)
    f = ev.get("codi_pair") or {}; mc = f.get("mcnemar") or {}
    bad = [w for w, ok in (("delta", same(f.get("delta"), delta)), ("mcnemar.b", mc.get("b") == b), ("mcnemar.c", mc.get("c") == c),
                           ("mcnemar.p_one_sided", same(mc.get("p_one_sided"), p, 1e-9)), ("n", f.get("n") == n)) if not ok]
    lab = ("BEATS-CODI20" if delta >= 0.03 - EPS and p <= 0.05 + EPS else "ABOVE-NS" if delta >= 0.03 - EPS else "MATCHES" if abs(delta) < 0.03 - EPS
           else "BELOW" if delta <= -0.03 + EPS else "MATCHES")
    v = dict(delta=delta, b=b, c=c, n=n, p_one_sided=p, se_paired=se, acc_B1=mean(a), acc_CODI20=b02["acc"], file_mismatch=bad or None, LABEL=lab)
    if bad:
        return rd(False, "codi_pair VOID (file %s != recomputed)" % bad, **v)
    return rd(lab == "BEATS-CODI20", None, **v)


BEATS_WORDING = ("this B1 model's accuracy differs from this CODI-20 model's by Δ %+.4f on the same 1,319 items (paired; item-sampling SE %.4f, McNemar p %.3g); "
                 "one seed each, matched examples (7.66M), a system comparison (full fine-tune + dense per-step supervision vs LoRA CODI) — what it can claim: "
                 "these two trained models differ on these items; what it cannot: that the recipe beats CODI (P(BEATS | equal recipes) ≈ 0.12)")
GSM_WORDING = ("serial latent arithmetic in a pass-global attention memory on GSM8K (m ≥ 3), C4 via distillation from CoT, with CoT-parsed pointer annotations "
               "(Addendum 1 item 1: strict DECOMP, calculator-only sub-step values masked)")


def parse_B(ev, parts, acc_needed=True):
    P = {"acc": p_gsm_acc(ev)}
    acc = P["acc"] if not P["acc"].get("missing") else {"band": [], "m": [None] * N_GSM, "acc_band": None}
    for p in parts:
        if p == "acc" or p == "codi_pair" or p == "gsmswap":
            continue
        P[p] = (p_gsm_probe(ev, acc) if p == "probe" else p_gsm_future(ev) if p == "future" else p_gsm_stall(ev) if p == "stall" else
                p_gsm_stepcap(ev) if p == "stepcap" else p_gsm_cf(ev, acc) if p == "cf_gsm" else
                {"missing": [], "arms": {arm: gsm_arm(arm, (ev.get("vswap") or {}).get(arm), acc) for arm in ("ii_prime", "i", "ii", "iv")}} if p == "vswap" else {"missing": []})
        if p == "vswap":
            P[p]["missing"] = sorted(set(m_ for x in P[p]["arms"].values() for m_ in (x.get("missing") or [])))
    return P


def b1_cell(ctx, bgo):
    k = "b1"; d = CELL[k]; o = {"cell": d, "name": NAME[k]}
    if k in NOT_ADMITTED:
        return unread(o, "not admitted")
    launched = has(d, "log.jsonl") or has(d, "args.json")
    if bgo["go"] is False:
        o["launched_despite_no_GO"] = launched
        return unread(o, "B not GO: %s (D-13; Stage B not run in this design)" % bgo["why"])
    if bgo["go"] is None and k in FINAL and not launched:
        return unread(o, "B GO not established (%s)" % bgo["why"])
    miss = {}
    if bgo["go"] is None:
        miss["b_go"] = "B GO undecided (%s)" % bgo["why"]
    for f in ("log.jsonl", "args.json", "eval_lvb_gsm.json"):
        if not has(d, f):
            miss[f] = "%s/%s" % (d, f)
    info = log_info(d)
    if info is not None and (not info["n_steps"] or not info["valid_acc"]):
        miss["log.jsonl"] = "%s/log.jsonl (no step / valid_acc records)" % d
    ir = interim_read(jp("interim")); b2 = b02_read(jp("b02"), jp("codi20"))
    for key, x, name in (("interim", ir, pname("interim")), ("b02", b2, pname("b02") + " (+ " + pname("codi20") + ")")):
        if x is None:
            miss[key] = name
        elif x.get("missing"):
            miss[key] = "%s fields %s" % (name, x["missing"])
        elif x.get("mismatch"):
            miss.setdefault("mismatch", []).extend(x["mismatch"])
    ev = jl(d, "eval_lvb_gsm.json"); P = {}
    if ev is not None:
        parts = DECISIVE_B + DESCRIPTIVE_B; pm = parts_missing(ev, PARTS_B, parts)
        for p in pm:
            miss[("desc:" if p in DESCRIPTIVE_B else "part:") + p] = "%s/eval_lvb_gsm.json part %s" % (d, p)
        P = parse_B(ev, [p for p in parts if p not in pm])
        for p, x in P.items():
            if x.get("missing"):
                miss[("desc:" if p in DESCRIPTIVE_B else "field:") + p] = "%s/eval_lvb_gsm.json %s fields %s" % (d, p, x["missing"][:4])
        mm, bad = meta_check(ev, "B", parts, ctx["evmB"], (b2 or {}).get("items_sha") if b2 and not b2.get("missing") else None)
        if mm:
            miss["meta"] = "%s/eval_lvb_gsm.json meta %s" % (d, mm[:6])
        if bad:
            miss.setdefault("mismatch", []).extend(bad)
    _gate_pend(miss, ctx, "B"); ad = args_diff(k); _args_pend(miss, ad)
    r = _gate_final(o, miss, k)
    if r:
        return r
    go24 = ir["GO"]; full = FULL[k] if go24 else B1_INTERIM_STEP
    dead = info["last"] is None or info["last"] < full or info["guard_stop"]; ent = entry_gsm(info["valid_acc"]); args_bad = ad["ok"] is not True
    why = [w for w, f in (("dead (last step %s < %d%s)" % (info["last"], full, ", guard stop" if info["guard_stop"] else ""), dead),
                          ("NO-ENTRY (Aug-valid accuracy <= 0.15 at every check)", not ent["entered"]), ("CONTAMINATED", info["self_reuse"] > 0),
                          ("ARGS-DIFF %s" % {x: ad.get(x) for x in ("shared_diffs", "unexpected_new_keys", "missing_keys", "required_bad") if ad.get(x)}, args_bad)) if f]
    cen = bool(why); acc = P["acc"]; wa = bad_part(acc, "acc"); pr = P.get("probe")
    # PROBE-GAIN-GSM
    w = wa or bad_part(pr, "probe"); gain = None
    if w is None:
        gain = acc["acc_band"] - max(pr["band_acc"][0], pr["band_acc"][1])
        w = ("PROBE-INVALID-GSM (PROBE(14) %.4f < acc %.4f - 0.03)" % (pr["band_acc"][14], acc["acc_band"]) if not pr["PROBE_VALID_GSM"] else
             "PLANT-GSM failed (%s)" % pr["plant"] if not pr["PLANT_GSM"] else None)
    pg = rd(ge(gain, 0.10), w, gain=gain, acc_band=acc.get("acc_band"), probe_band=(pr or {}).get("band_acc"))
    arms = ((P.get("vswap") or {}).get("arms") or {})
    a_ii, a_i = gsm_arm_reading(arms.get("ii_prime")), gsm_arm_reading(arms.get("i"))
    ss = rd(a_ii["met"] and a_i["met"], None if (a_ii["readable"] and a_i["readable"]) else "arm unreadable: ii' %s; i %s" % (a_ii["status"], a_i["status"]),
            ii_prime=a_ii, i=a_i)
    st = P.get("stall"); nc = rd(ge((st or {}).get("recovery"), 0.9), bad_part(st, "stall"), recovery=(st or {}).get("recovery"), n=(st or {}).get("n"))
    fu = P.get("future"); w = bad_part(fu, "future"); bym = (fu or {}).get("by_model") or {}; unr = [m_ for m_ in ("lin", "mlp") if not ge((bym.get(m_) or {}).get("FD_kk_ge3"), 0.3)]
    if w is None and unr:
        w = "FD(k, k) < 0.3 pooled for %s (UNREADABLE, never a vacuous pass)" % unr
    shows = {m_: bool(ge(bym[m_]["FD_kk_ge3"], 0.3) and ge(bym[m_]["FD_1_k_ge3"], 0.8 * bym[m_]["FD_kk_ge3"])) for m_ in bym if num(bym[m_].get("FD_kk_ge3"))
             and num(bym[m_].get("FD_1_k_ge3"))}
    nop = rd(len(shows) == 2 and not any(shows.values()), w, by_model=bym, shows_one_pass_chain=shows)
    cfr, cflab = cf_gsm_reading(P.get("cf_gsm"))
    comps = {"PROBE_GAIN_GSM": pg, "SERIAL_STEP_GSM": ss, "CF_GSM": cfr, "NO_CLOCK_GSM": nc, "NOT_ONE_PASS_CHAIN": nop}
    vb = all(x["met"] for x in comps.values()) and not cen
    sc = P.get("stepcap")
    lab = {"PASS1_PRECOMPUTE_GSM": (gain < 0.05 - EPS) if num(gain) else None, "ONE_PASS_CHAIN_L": any(shows.values()) if shows else None,
           "RECOMPUTES_FROM_INPUT_L": bool(a_ii["readable"] and ge(a_ii["own"], 0.5)),
           "STEP_NOT_LEARNED": (sc["pooled"] < 0.5 - EPS) if sc and not sc.get("missing") else None, **cflab}
    if go24:
        acc_r = codi_read(ev, acc, b2) if not wa else rd(False, wa)
        if cen:
            acc_r = rd(False, "censored (rule b)", **{k_: v_ for k_, v_ in acc_r.items() if k_ not in ("met", "readable", "status")})
    else:
        acc_r = rd(False, "B1-24k: accuracy descriptive vs CODI4 %.3f (v1 §2.7)" % CODI4_ACC, acc_B1=acc.get("acc"), delta_vs_CODI4=None if wa else acc["acc"] - CODI4_ACC)
    o.update(status="FAIL" if cen else "SCORED", censored=cen, censors={"why": why, "dead": dead, "last_step": info["last"], "full_steps": full, "entry": ent,
             "ARGS_DIFF": ad}, INTERIM=ir, B1_24K=not go24, B0_2={x: b2.get(x) for x in ("B0_2_pass", "n_item_diffs", "acc")}, READINGS=comps, VB_C1_GSM=vb,
             LABELS=lab, ACCURACY=acc_r, n_band=acc.get("n_band"), flag_n_band_ne_879=acc.get("flag_n_band_ne_879"),
             descriptive={"ii": arms.get("ii"), "iv": arms.get("iv"), "gsmswap": "descriptive (§207 rule; not re-scored here)", "stepcap": sc})
    o["WORDING"] = {"VB_C1_GSM": GSM_WORDING if vb else None, "never": ["C4 qualified", "C4-legal distillation", "C2", "beyond text", "beats CODI (recipe)", "C3"],
                    "ACCURACY": (BEATS_WORDING % (acc_r["delta"], acc_r["se_paired"], acc_r["p_one_sided"])) if (go24 and acc_r.get("delta") is not None and not cen) else None,
                    "PASS1_PRECOMPUTE_GSM / ONE_PASS_CHAIN_L": "the §208 / §204 failure recurs in VERSION-BANK form" if (lab["PASS1_PRECOMPUTE_GSM"] or lab["ONE_PASS_CHAIN_L"]) else None}
    o["label"] = ("FAILURE (%s; rule b)" % "; ".join(why) if cen else ("VB-C1-GSM" if vb else "not VB-C1-GSM (%s)" % ", ".join(
        "%s %s" % (c, x["status"]) for c, x in comps.items() if not x["met"]))) + " | %s | accuracy %s" % (
        "B1-24k (interim no-GO)" if not go24 else "B1 120k", acc_r.get("LABEL", acc_r["status"]))
    return o


# ---------------------------------------------------------------- flow and output
def _strip(o):
    return {k_: v_ for k_, v_ in o.items() if not k_.startswith("_")}


def score(final=(), not_admitted=(), user_ok=()):
    FINAL.clear(); FINAL.update(final); NOT_ADMITTED.clear(); NOT_ADMITTED.update(not_admitted); USER_OK.clear(); USER_OK.update(user_ok)
    ctx = {"l0": gate_l0(), "evmA": evmem("A"), "evmB": evmem("B")}
    ctx["shaA"] = (canon_sha((ctx["l0"] or {}).get("test_items_sha")) or file_sha(jl(CELL["a1"], "eval_lvb.json"))   # Stage A test-set reference (RESOLVED R30)
                   or file_sha(_direct_body(jl(CELL["direct"], "eval_direct.json"))[1]))
    direct = direct_cell(ctx); a1 = a1_cell(ctx, direct); g1r = gate1_read(jp("gate1")); pr = pool_read(jp("pool")); a2go = a2_go(a1, g1r, pr)
    ctrl = ctrl_cell(ctx, a2go); a2 = a2_cell(ctx, a1, a2go, ctrl, g1r, direct, pr); bgo = b_go(a1, a2, ctrl); b1 = b1_cell(ctx, bgo)
    cells = {"direct": direct, "a1": a1, "a2": a2, "ctrl": ctrl, "b1": b1}
    mem = {NAME[c]: (x.get("MEM") or {}).get("MEM_TRADED_L") for c, x in cells.items() if x.get("MEM") is not None}
    flow = {"A2_GO": a2go, "B_GO": bgo, "B1_INTERIM_GO": b1.get("INTERIM"),
            "MEM_TRADED_L": {"cells": sorted(n for n, v in mem.items() if v),
                             "rule": "MEM-TRADED-L in any LANG cell -> every later stage (A2 after A1, B1, any further self round) needs the user's explicit OK (L6)",
                             "user_ok_recorded": sorted(USER_OK)}}
    primary = {"VB_C1_L (A1)": a1.get("VB_C1_L") if a1["status"] in ("SCORED", "FAIL") else a1["status"],
               "SD_C2_L (A2)": a2.get("SD_C2_L") if a2["status"] in ("SCORED", "FAIL") else a2["status"],
               "SD_SELF_L (A2)": a2.get("SD_SELF_L") if a2["status"] in ("SCORED", "FAIL") else a2["status"],
               "VB_C1_GSM (B1)": b1.get("VB_C1_GSM") if b1["status"] in ("SCORED", "FAIL") else b1["status"],
               "BEATS_CODI20 (B1)": (b1.get("ACCURACY") or {}).get("met") if b1["status"] in ("SCORED", "FAIL") else b1["status"]}
    return {"registered": REGISTERED, "R": R, "CARD": CARD, "final": sorted(FINAL), "not_admitted": sorted(NOT_ADMITTED), "user_ok": sorted(USER_OK),
            "PRIMARY": primary, "FLOW": flow, "G_L0": ctx["l0"], "cells": {c: _strip(x) for c, x in cells.items()},
            "layouts_assumed": LAYOUTS, "paths_assumed": {k_: "/".join(v_) for k_, v_ in PATHS.items()}, "resolved": RESOLVED,
            "not_scored_here": ["A0-1 (THEORY note 191-4; v1 §2.8)", "A-TXT (optional; no registered reading here)"]}


def reading(out):
    c = out["cells"]
    return "READING: " + " || ".join("%s %s" % (NAME[k_], c[k_]["label"]) for k_ in ("direct", "a1", "a2", "ctrl", "b1")) + \
        " || A2 GO %s || B GO %s" % (out["FLOW"]["A2_GO"]["go"], out["FLOW"]["B_GO"]["go"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="append", default=[], choices=sorted(CELL))
    ap.add_argument("--not_admitted", default="")
    ap.add_argument("--user_ok", default="", help="stages the user explicitly OK'd after a MEM-TRADED-L (A2, B1); only from a recorded note")
    a = ap.parse_args(argv)
    out = score(a.final, [x for x in a.not_admitted.split(",") if x], [x for x in a.user_ok.split(",") if x])
    print(reading(out).encode("ascii", "backslashreplace").decode("ascii"))
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    return out


if __name__ == "__main__":
    main()
