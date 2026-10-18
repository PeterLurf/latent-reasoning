# TEXT-SELF v1 (proposed §219): a matched text-CoT self-training baseline for the latent self-depth line — registration-ready text

Clock 2026-10-18 15:19 EDT (`date`). Opus design-revision subagent for Theory. $0.
**Status: registration-ready text, NOT registered.** Audit reviews it (G-T13 diff review and every gate in §7). The family cap is **$3.00**, so it falls under **the user's $30 standing rule (Audit approves)**. The text arm has no memory channel, so SELFDEPTH_v1 §9's MEM-TRADED user-OK rule does not apply. Repo HEAD at writing: 796a3343. §218 is LANG-VB; this is proposed as §219.
Inputs (all read): research6/TS_draft.md (= docs/research_textself_2026-10-18/TS_draft.md, written 12:04–13:17 at 598e75de) + scripts/; REDTEAM_TS.md + rt_ts/ (1 BLOCKER, 6 MAJOR, 14 MINOR); PROOFS_TS.md + proofs_ts/ (≈ 37 PASS, 17 CORRECTED, 0 FAIL); the frozen latent texts SELFDEPTH_v1.md (5ded8703) and SD2_v1.md (e9ade2ec), the structural model; THEORY §216 (registration, the s1 round-2 result) and notes 216-1…216-9; theory/score_s216.py (args_diff, BASE_NEUTRAL_NEW, args_ref_check = the 216-7 rule, the 215-7 pending rule and `--final`); theory/score_s213.META; boxes/nested170.py (argparse block l.635–658, main's auto-micro / eval-memory check l.925–950, the args.json writer l.956–966, the unconditional final eval l.1162–1180, the warm-up branch l.1039); boxes/runs_fsq/s170_cot_s1 (args.json, args_eval.json, eval.json meta, mem_check.json, model.pt sha256); s215_ftself_s1 / s216_r2_s1 args.json and auto_micro_self.json.
**Theory's decisions T1–T13** (relayed 2026-10-18) are applied and listed in §1.1.
**$0 checks** (research6/v1_checks/, JSON beside each script; CPU, ≤ 2 threads, nice 19, `python -B`, PYTHONDONTWRITEBYTECODE=1; nothing written under D:/Latent): §10.
Tags: [PROVED], [PROVED UNDER X], [DERIVED script] (exact arithmetic on measured inputs; script named), [MEASURED file], [VERIFIED code], [ASSUMED], [CONJECTURE].

---

## 0. One screen

| item | TEXT-SELF v1 |
|---|---|
| Question | The latent s1 lineage (§215 R1, §216 R2) moved its 50 % depth **10.31 → 15.46 → 20.54** (n50 / support 1.29 / 1.29 / 1.28). Run the same two self-training rounds on a **text chain-of-thought** GPT-2 of the same task, round for round: the same windows (n 9–12, then ⅔ n 13–16 + ⅓ n 9–12), form-only filter, gate rule, mixture (p_self 0.5, gold n ≤ 8 replay), 12k steps, lr 3e-5, and identical self exposure (**5,992 self steps = 383,488 items at the same step indices**). Does text-CoT self-training (a) absorb each window and (b) move its n50 by as much? A **baseline** for the latent line: never a C-criterion |
| Lineage (T2) | **One:** text s1 (from `s170_cot_s1`) against latent s1, the only uncensored latent lineage (R1-s2 FT-DIVERGED, 216-8). No text s2. PROOFS' s2 pool-exhaustion finding is therefore moot |
| Cells | `s219_base_s1` (eval-only: the fp32 re-eval of `s170_cot_s1/model.pt`, sha256 36f4659b…3392); `s219_t1_s1` (FT from the base); `s219_t2_s1` (FT from T1). Trainer: new module `boxes/textself_ft.py` (nested170.py unchanged), **`--no_final_eval` in every launch** (T3) |
| Labeller (T8) | The round's model's own greedy single step, re-rooted at every step (**REROOT**, the latent labeller's protocol), computed exactly by **TF-GREEDY** [PROVED incl. ties / EOT / caps; MEASURED: TF ≡ KV-cache REROOT generation on 1,100 / 1,100 fresh items + 400 / 400 draft items, FULL 600 / 600]. Base precondition: REROOT keep **0.992** pooled at n 9–12 (1 kept-wrong of 1,587), so round 1 is labellable. Whole-chain **FULL** is descriptive only (keep 0.55 / 0.06 / 0.00 at n 10 / 11 / 12; its kept chains are impure: π 0.96 at n 10–11). Round 2: labeller T1, BASE-LABELLED fallback per window, **no bridge** (T9) |
| Filter / gate | KEEP-T ∧ CV-T: form only, values and the final answer never checked (the latent's filter verbatim). Gate rule and π terms exactly the latent's (π = label, π^E = all values; RT m1) |
| Eval (T5, T10) | `eval_text.json`: free greedy generation over n 1–24 of test_set(1000, 24), **fp32, tf32 off, cot_eval_bs 32, local RTX 3070** (the latent comparators' card class); **the base re-eval first**. The RTX 2070 only behind a pre-stated CARD-GATE. a_T(n) = answer accuracy (context overflow and non-termination count as failures); STRICT (form-valid ∧ correct) descriptive |
| Comparison | Δ_r = n50_T(r) − n50_L(r), latent fixed at 10.31 / 15.46 / 20.54. **PARITY** iff \|Δ_1\| ≤ 0.5 ∧ \|Δ_2\| ≤ 0.5; **TEXT-AHEAD / LATENT-AHEAD** by round 2's sign with round 1 not contradicting; **CROSSED** otherwise. TEXT-AHEAD / LATENT-AHEAD with \|Δ_2\| ≤ 1.0 is worded **"marginal"** (T6). **BASE-PARITY:** if \|Δ_0\| > 0.25 on the fp32 base re-eval, every Δ is read as gain over each arm's own base (T6). ¬T-ABSORB reads **TEXT-NOT-ABSORBED**, never LATENT-AHEAD (T4). An unlabellable round-2 window reads **TEXT-LABEL-STALL** + the round-1 label: an output-format labeller limit, not a depth result (T9). **CTX-BOUND:** two rounds only (GPT-2's 1,024 position rows) |
| OC (T7) | Measured lineage spread **ŝ = 0.17** (3 df; 95 % CI 0.09–0.62) central, s 0.35 as sensitivity. **P(PARITY \| no difference) 0.964** (0.608 at s 0.35); P(PARITY \| 1-n difference) 0.0004 (0.035); P(LATENT-AHEAD \| additive panels) 1.000 (≥ 0.998 at s 0.35). Given both rounds readable (P 0.74): **LATENT-AHEAD 0.61, PARITY 0.26, TEXT-AHEAD 0.12, CROSSED 0.02**. Unconditional: LATENT-AHEAD 0.45, PARITY 0.19, TEXT-AHEAD 0.09, CROSSED 0.01; TEXT-NOT-ABSORBED 0.09 (P ≈ 0.94 under no FT effect, by construction); TEXT-LABEL-STALL 0.06; censored / not run 0.11 [DERIVED v1_oc; priors ASSUMED]. The LATENT-AHEAD headline is prior-driven; the discriminating power is the robust part |
| Cost (T12) | **Plan B′:** cloud RTX 3090 for T1 and T2 training only, **projected $0.70–2.15**; admission T1 ≤ $1.00 / T2 ≤ $1.45; guards $1.20 / $1.70; **family cap $3.00** (GUARD_STOP). Pools, gates and evals local (3070, $0, ≈ 9.7–15.3 h). Off-host checkpoint mirrors. Realised at registration: $0 |
| Approval | Under the user's $30 rule (cap $3.00): Audit approves, after the gates of §7 |

---

## 1. Change table (every review finding, with source)

The draft's design stands unless changed below: the matching table (draft §1.1), REROOT + TF-GREEDY, KEEP-T ∧ CV-T, the coin-stream replay, the per-cell readings and bar formulas, CTX-BOUND, the hypothesis panels.

### 1.1 Theory's decisions

| # | decision | applied in |
|---|---|---|
| T1 | **ARGS-DIFF** against `s170_cot_s1/args.json` with a **TEXT_NEUTRAL_NEW** map: each of the **30** nested170.py argparse keys absent from the base args.json, at its off default [DERIVED v1_argkeys: AST of nested170.py, 57 parser keys]; the runtime / info keys `excl_survive_by_n` and `hostprobe_sps` treated as META; checked on a **real smoke-run args.json** (gate G-T5); the **216-7 args_ref content rule** | §4.6, §7 G-T5 |
| T2 | **One lineage:** text s1 only, matched to latent s1. No text s2 (RT M1). PROOFS item 1 (the s2 coin stream takes 6,076 self steps = 388,864 items > a 384,000 pool) is moot | §0, §2.1, §6, §8 |
| T3 | **Seal the trainer's automatic final eval:** `--no_final_eval` in every s219 launch (RT M2). No T1 / T2 past-support output may exist before the registered eval_text | §2.5, §2.9, §5.2 (SEAL-BREACH), §7 G-T20 |
| T4 | ¬T-ABSORB (round 1) reads **TEXT-NOT-ABSORBED** (RT M3), never LATENT-AHEAD; Δ_1 descriptive | §4.4, §5.1, §6 |
| T5 | **Eval card:** every text eval local fp32 on the RTX 3070, the base's fp32 re-eval first; the 2070 only as a gated backup: per-n accuracy within 0.005 of the 3070 at every n and n50 within 0.02, on the base (RT M4) | §2.7, §7 G-T21 CARD-GATE, §6 |
| T6 | **Wording (RT M5):** one lineage per arm; GPT-2 learned absolute positions; the FAIL-POS line beside any LATENT-AHEAD; \|Δ_2\| ≤ 1 is "marginal". **Round-0 offset (RT M6):** BASE-PARITY \|Δ_0\| ≤ 0.25 on the fp32 re-eval, else Δ read as gain over each arm's own base | §4.4, §5.1, §6 |
| T7 | **Noise / OC:** ŝ = 0.17 (PROOFS: 3 df, CI 0.09–0.62) central, 0.35 sensitivity; every OC row recomputed exactly (PARITY / TEXT-AHEAD / LATENT-AHEAD / CROSSED / TEXT-NOT-ABSORBED / TEXT-LABEL-STALL; P(PARITY \| no difference), \| 1-n difference; both-rounds-readable P with the T2 stall ≈ 0.06 per RT) | §6, v1_oc |
| T8 | **Labeller:** re-rooted TF-GREEDY primary; **G-T16 compares per-step REROOT generation**, not the eval loop (PROOFS item 2); FULL descriptive only (kept chains impure) | §2.2, §7 G-T16 |
| T9 | **T2 stall:** the matched gate + base fallback (SD2's BASE-LABELLED), no bridge; a stall is worded "an output-format labeller limit (the latent copies per slot; text must re-emit the expression)", not a depth result | §2.2, §4.4, §5.1 |
| T10 | **Metric:** free-generation answer accuracy primary; STRICT descriptive. PARITY band ±0.5 kept | §4.2, §4.4 |
| T11 | **Exclusions:** PROOFS' fresh items (rng 23187, read by the base) join every pool / gate exclusion set, plus everything §216 excludes (test_set(1000, 24), SERIAL, 213-37, every latent pool and gate) and the sealed n 25–30 (216-5) | §2.4 |
| T12 | **Card / cost plan:** RT Plan B′ (train on cloud RTX 3090s; pools / gates / evals local), costed with PROOFS' corrections (MEMO-GAP-T generations, labeller padding +28 %); cap ≤ $6 with per-cell admission bounds and GUARD_STOP; off-host checkpoints; one GPU class per stage type | §8, v1_cost |
| T13 | **Every PROOFS CORRECTED item** (149 step-1 failures; 5 operations; n 25 fit 0.52–0.55; frontiers 45 / 59 and 313 / 531; the value-check wording; …) | §1.3 |

### 1.2 Red team (REDTEAM_TS.md)

| # | finding | v1 | verdict |
|---|---|---|---|
| B1 | ARGS-DIFF censors every cell by construction (30 new keys + 2 info keys) | T1: TEXT_NEUTRAL_NEW (§4.6, listed in full), META additions, the 216-7 args_ref rule, G-T5 on the real CPU-smoke and host-smoke args.json before launch. **Also (V2):** a reference key missing from the cell's args.json is ARGS_BAD (score_s216.args_diff does not check that direction) | accepted |
| M1 | latent s2 is dead; POOLED is 2-vs-1 | T2: s1 only; POOLED / UNPAIRED rows and their OC deleted; the claim line says "latent comparator = the surviving s1 lineage; s2 censored at R1 (216-8)". RT's optional $0 fp32 re-eval of `s170_cot_s2` as a text round-0 noise point: **not registered** (it would add a fourth spread point to ŝ only after the fact); listed in §9 | accepted |
| M2 | the trainer's built-in bf16 final eval reads n 13–16 before eval_text; it hangs CPU smokes | T3: `--no_final_eval` (required true by ARGS-DIFF); the trainer also skips nested170's eval-memory check, which generates on a test n-16 item (V3); `--stop_at` returns before any eval; SEAL-BREACH rule (§5.2) | accepted |
| M3 | ROUND-1-ONLY reads LATENT-AHEAD-1 under "text FT did nothing" | T4: TEXT-NOT-ABSORBED, with P(reads \| no FT effect) ≈ 0.94 stated (§6). ROUND-1-ONLY's three labels apply only to TEXT-LABEL-STALL. Round 2: see V5 | accepted |
| M4 | Plan A evaluates on another card class; the 3070 queue that motivated it is gone | T5 / T12: Plan B′; every eval_text on the 3070; CARD-GATE for the 2070 (§7 G-T21, OC §6) | accepted |
| M5 | one-lineage wording; GPT-2 position confound | T6: claim line pre-stated (§5.1) with "one lineage each (latent = the surviving s1 lineage); GPT-2 small, learned absolute positions (1,024 rows); Arithmetic(n, 11); two rounds"; FAIL-POS share beside any LATENT-AHEAD; "marginal" for \|Δ_2\| ≤ 1 | accepted |
| M6 | levels vs gain; round-0 offset | T6: levels registered; BASE-PARITY \|Δ_0\| ≤ 0.25 else the gain statistic Δ_r − Δ_0, decided now; both always reported; OC of both (§6) | accepted |
| m1 | gate π terms differ from the latent's | π_T := final value = gold (label), π^E_T := every value = gold; GATE and LABEL-NOISY-T on exactly the latent's terms (§2.2, §4.4) | accepted |
| m2 | P(TEXT-LABEL-STALL) over-stated; it is a labeller-format artefact | P(GATE-T2 passes \| T-ABSORB) 0.93 → stall ≈ 0.06 of outcomes (§6); wording T9; "no T1.5 bridge, no lowered bar" pre-stated (§2.2) | accepted |
| m3 | D1 applies to both labellers under REROOT | §2.3 reworded (with PROOFS P7.3): REROOT gives neither labeller extra information; the D1 asymmetry is in the trained target and at test time | accepted |
| m4 | "Literature's recommendation made explicit" is inaccurate | "the latent's filter applied verbatim; stricter than Literature's minimum (final answer / step count)" (§2.3) | accepted |
| m5 | G-T16 tests the wrong protocol; 90 / 90 is thin | T8: G-T16 = TF-GREEDY REROOT vs per-step REROOT KV-cache generation on all 250 gate items per n (incl. the rejected ones), plus FULL vs FULL generation for LABELLER-FULL; PROOFS' 1,100 + 400 + 600 CPU agreements cited | accepted |
| m6 | eval cost under-counted (MEMO-GAP-T; KV-cat context term) | T12: MEMO-GAP-T costed at 2,000 per set (matched to the latent's MEMO-GAP-2), eval_text 1.7–2.9 h with RT's context term as the upper end; G-T18 times the longest n-24 batch. Cut order step 2 (MEMO-GAP-T at 1,000 per set) pre-stated (§8) | accepted |
| m7 | pool spare 0.13 % | ≥ 1 % spare: T1 97,000 per n (388,000; spare 4,512 = 1.18 %), T2 65,000 / 32,500 per n (390,000; spare 6,512 = 1.70 %) [DERIVED v1_cost] | accepted |
| m8 | per-type micro changes the objective; G-T9 picks per host | self micro **fixed per round: T1 2, T2 4** = the latent cells' realised self micro (s215_ftself_s1 2, s216_r2_s1 4 [MEASURED auto_micro_self.json]); G-T9 only confirms the fit (projected 14.7 / 12.8 GiB); a host on which it does not fit is refused, the micro never changes | accepted |
| m9 | no text FT-CTRL; attribution | T-SELF wording adds "(not attributed: no text FT-CTRL)"; no text FT-CTRL is run (the comparison is recipe vs recipe) | accepted (wording) |
| m10 | shared module nested170.py | V1: trainer in a new module `boxes/textself_ft.py` importing nested170 read-only; labeller `boxes/textself.py`; eval `boxes/text_eval.py`; nested170.py sha256 pinned (G-T7) | accepted |
| m11 | OC-rule gaps | every named outcome carries P(pass \| no effect) / P(pass \| effect) (§6) | accepted |
| m12 | sentinels in Δ | n50_T "> 24" ⇒ Δ_r := 24 − n50_L(r) + 0.01 (TEXT-AHEAD side); "< 1" ⇒ Δ_r := 1 − n50_L(r) − 0.01 (LATENT-AHEAD side); fixture (§4.4, §4.6) | accepted |
| m13 | (a) adaptive controllers; (b) shape-informed budgets, latent fixed-K′; (c) re-grep seeds; (d) cut order; (e) T1 copy length 45 | (a) §2.5: no loss-reading controller on the textcot path [VERIFIED by RT: AnnealGate / KW / CSR are inject-only]; clip 1.0 reads mixed batches, as the latent's. (b) §4.4: both budgets are shape-informed; LATENT-FIXED-K′ descriptive row. (c) §2.4 item 9: re-grepped at 796a3343, 0 hits. (d) §8. (e) §1.3 | accepted |
| m14 | overflow fair; CTX-BOUND is a GPT-2 property | worded "GPT-2's 1,024 learned position rows", not a text-CoT property (§2.8, §5.1) | accepted |

### 1.3 Proofs (PROOFS_TS.md; CORRECTED items, T13; PASS items unchanged)

| # | finding | v1 |
|---|---|---|
| P4 | s2 coin stream 6,076 self steps = 388,864 > 384,000 | moot under T2 (no text s2); s1 replay exact (5,992 / 383,488; 12,000 / 12,000 coins and every U value) |
| P1 | TF-GREEDY proof PASS; the draft's 90 / 90 was FULL only | §2.2 fp32 caveat: "TF-GREEDY equals KV-cache greedy generation on 90 / 90 FULL items (draft) and on 1,100 / 1,100 REROOT + 600 / 600 FULL fresh items and 400 / 400 draft items (PROOFS_TS, rng 23187; 158 REROOT rejections; min top-2 margin ≥ 1e-3)"; G-T16 re-specified (T8) |
| P5 | ŝ = 0.167 (3 df), half the draft's central 0.35 | T7; §6 text: "P(PARITY \| true parity) is 0.96 at the measured in-house spread (ŝ 0.17, 3 df) and 0.61 at the conservative s 0.35; the text FT lineage noise is unmeasured" |
| P6 | MEMO-GAP-T and TF padding uncosted | T12 (§8); padded TF positions **471 (T1) / 729 (T2)** per item (+29 % / +27 %); pools ≈ 1.0–1.3 h (T1) and 1.6–2.0 h (T2) on the 3070 |
| P6 | memory slope 0.082 GiB per row (the 12.99 GiB point excludes AdamW's 0.93) | §3.6: T1 (rows 313) micro 2 **14.7 GiB**; T2 (rows 531) micro 2 23.7 > 21.0, micro 4 **12.8 GiB**; micro 2 / 4 unchanged |
| P6 | G-T18 local cap | 6.0 GiB (the local LATENT_MEM_CAP_GB) on an 8 GB card; projected eval peak ≈ 5.1 GiB |
| P6 | Plan B $0.7–2.0, 10–13 h of 3070 | superseded by Plan B′'s own costing (§8: $0.70–2.15; 9.7–15.3 h) |
| P6 | "TF-GREEDY is what makes the matched exposure affordable" overstated | §2.2: REROOT generation at bs 256–512 would take ≈ 2–4 h per pool [ASSUMED latency scaling], so TF-GREEDY is ≈ 1.5–3× faster; the G-T16 fallback (generation for the failing window) is affordable |
| P3 | n 25 fit 0.551 → range | "n 25 ≈ 0.52–0.55 fit" (pooled 0.535 ± 0.011); CTX-BOUND unaffected |
| P3 | frontier values | copies ≤ **45** (T1) / ≤ **59** (T2); rows ≈ **313 / 531** (pool-sized draws; pool_meta's T_max_by_n is the registered value); FULL failure window "**1.16–1.35 ×** the 147 trained rows" |
| P3 | E1 cap-bound rows | the eval loop breaks before feeding row 1,024 (a non-terminating row in a cap-bound batch is −1, not a zero-length forward); CPU stub test in G-T18 |
| P8 | "148 of the 165 REROOT failures at step 1" | **149** of 165 |
| P8 | "about 6 operations further" | **about 5** (D3-T 0.781 at n 20 vs text 0.775 at n 15) |
| P8 | "293k–(293k + 50 × batches)" | **290k–332k** sequential decode steps for eval_text n 1–24 |
| P7 | D1 wording; the π conjecture mis-motivated | §2.3: "a value check (apply(op, a, b) on the parsed handle) would be a gold step-value check; **it is equally available to the latent filter** (check_step holds the parsed handle), and both arms forgo it". The [CONJECTURE] π_text ≥ π_latent is dropped: both labels are local reads; any π difference comes from the decoders (autoregressive copy vs per-slot decode) |
| P2 | FULL kept chains impure | §2.2: "against 10 of 245 for FULL at n 10–11 (π 0.96; 5 of 25 at n 11)"; FULL also fails GATE's π_n ≥ 0.99 |
| P2 | Goodhart row "first steps on the longest inputs" | "its few failures are first steps (on longer-than-average inputs from n 12 on)" (§5.3) |

### 1.4 New in v1 (not in any review)

| # | item | why |
|---|---|---|
| V1 | Trainer in a new module `boxes/textself_ft.py`: its parser is a verbatim copy of nested170.main's 57 add_argument calls plus the FT keys; its textcot loss / schedule / clip / autocast / checkpoint code is a copy of nested170's textcot path. Gate **G-T7 PLAIN-EQUIV**: with `--p_self 0` and no `--init_from`, CPU-bitwise identical per-step losses (steps 1–3, log_every 1) to `nested170 --arm textcot --time_steps 3`, and args.json identical on every non-META key except the FT keys at their off values | RT m10's direction, made checkable; nested170.py stays byte-identical (sha256 pinned) while §217 / vbank import it |
| V2 | ARGS-DIFF also fails a **reference key missing** from the cell's args.json (non-META) | score_s216.args_diff checks only keys present in the cell; a dropped key (e.g. `lr`) would pass |
| V3 | SEAL mechanics: the trainer skips nested170's textcot eval-memory check (it generates on a test n-16 item, l.943–950) when `--no_final_eval`; G-T6 LOAD uses one synthetic forward, never `--eval_only`; G-T18 uses fresh n-24 shapes (rng 23195), never test items; GATE-T2 / GATE-T1′ start only after `s219_t1_s1/eval_text.json` and the round-2 condition note | every route by which a T1 / T2 output at n ≥ 13 could exist before eval_text is closed |
| V4 | The round-2 condition reads eval_textself's **CPU parts** (LABEL-AUDIT-T, CONTAMINATED), not its GPU descriptive parts (MEMO-GAP-T, LABELLER-KEPT-T) | the round-2 decision does not wait on descriptive generations |
| V5 | T4 applied to round 1 only, as written. An uncensored T2 failing T-ABSORB-2 is read four-way with the pre-stated qualifier "(text round-2 window not absorbed: min a(13..16) = x)" on the same line; its share (≈ 0.11 of outcomes) is in the OC. The round-2 analogue (TEXT-NOT-ABSORBED-2) is an open item with its own OC (§9 item 1) | the draft's OC silently assumed absorption in every both-rounds case |
| V6 | Base model pinned: `s170_cot_s1/model.pt` sha256 `36f4659b99f08a65202b303f4af74ef20ac65f1e60d2a25f3e84aafbfb2e3392` [MEASURED sha256sum] | eval_text meta and the T1 init assert it |
| V7 | Precondition exclusion as a superset: the **full** 200-per-n streams at n 9–20 of rng 23180 (draft) and rng 23187 (PROOFS), each regenerated by the scripts' own code (one stream, n 9..20 in order, redrawn on exclusion hits) | the scripts read subsets (100–200 per n); the stream superset needs no per-protocol bookkeeping |

---

## 2. Design (registration text)

### 2.1 What is matched, and what cannot be

| element | latent lineage s1 (§215 / §216) | TEXT-SELF v1 | matched? |
|---|---|---|---|
| Base | D3-T s1 (`s213_d3t_s1`): SLVA loop, gold n ≤ 8, 48k × bs 64, lr 1e-4, RTX 3090 | §170 TEXT-COT s1 (`s170_cot_s1`): GPT-2 small, Feng chain, gold n ≤ 8, 48k × bs 64, lr 1e-4, RTX 3090 [MEASURED args.json] | yes (task, generator, gold range, steps, bs, lr, card class) |
| Base depth | n50 10.31 (3070 fp32 eval_vb) | n50 10.44 on the committed **bf16** 3070 eval (Δ_0 +0.13); the fp32 re-eval (`s219_base_s1`) replaces it for every bar and for Δ_0 | read on the fp32 re-eval (BASE-PARITY, §4.4) |
| Labeller | the round's model, single step re-rooted at every step (SELFDEPTH_v1 §2.1) | the round's text model, single greedy step re-rooted at every step (REROOT), computed by TF-GREEDY (§2.2) | yes (own single step, fresh root, K = 1 / one emitted step) |
| Label information | x = [EOT] + render(cur) + [' ='], operands visible | the same prompt | yes: REROOT gives neither labeller extra information (§2.3) |
| Step object | nested170.cot_steps (vbank.gate_rerender) | nested170.cot_steps[1:] (cot_ids); the handle's brackets drop in the same step [VERIFIED by RT] | yes |
| Filter | KEEP ∧ CV (form only; values never checked) | KEEP-T ∧ CV-T (form only; values never checked) | yes (verbatim; it binds for text, ≈ never for the latent) |
| Gate | GATE-1 / GATE-2 / GATE-1′: 2,000 per n; π_pool ≥ 0.995 ∧ π_n ≥ 0.99 ∧ π^E_pool ≥ 0.99 ∧ keep_n ≥ 0.70 | the same rule, counts and π terms (π = label, π^E = all values) | yes (RT m1 fixed) |
| Round-1 / round-2 windows and mix | n 9–12, p_self 0.5; then ⅔ n 13–16 / ⅓ n 9–12 by pool composition | the same | yes |
| FT recipe | 12k steps, bs 64, lr 3e-5 (0.3 × base lr), warm-up 100 + cosine to 0, AdamW wd 0.01, clip 1.0, bf16 autocast, init_from, fresh AdamW | the same numbers (lr_at's built-in 100-step warm-up: `--warmup 100` takes the unchanged branch, nested170 l.1039 [VERIFIED]) | yes |
| Self micro | realised 2 (R1) / 4 (R2) [MEASURED auto_micro_self.json] | fixed 2 (T1) / 4 (T2) (RT m8) | yes |
| Self exposure | 5,992 self steps = 383,488 items per round (seed-1 coin stream) | the same 5,992 step indices (rng_s replay incl. 64 discarded integers(0, 5) per self step) [PROVED by stream identity; PROOFS P4] | yes (exact) |
| Plain stream | D3-T's law replayed from seed 1 each round (disclosed) | §170 TEXT-COT's law (uniform n 1–8, test ∪ SERIAL excluded) replayed from seed 1 each round (disclosed) | each arm's own base law |
| Target | R1 answer + R2 per-pass states E_self | the chain string compose_T(syms, values) | the same chain object, each arm's format |
| Test items | test_set(1000, 24) | the same items, paired per item | yes |
| Eval card / dtype | local RTX 3070, fp32, tf32 off (eval_vb bs 32; eval_deep bs 16) | local RTX 3070, fp32, tf32 off, cot_eval_bs 32 (T5) | yes |
| Test-time computation | K′ = n + 2 passes over T ≈ 3n + 4 rows (shape-derived budget) | free greedy generation of C(n) ≈ 1.5 n² tokens, budget C_struct + 50 (shape-derived) | **the object of comparison**; both budgets are shape-informed (RT m13b) |
| Censoring | the latent cell set includes K′24 / LENGTH-FREE / VB-C1 kept readings; the comparator is the **surviving** lineage (s2 censored at R1) | FT-DIVERGED-T only | asymmetric, stated (RT M1) |
| Control | FT-CTRL (§215 pair) | none (not attributed) | wording only (RT m9) |
| Positions | rows ≤ 61 trained, ≤ 83 read | rows ≤ 531 trained (T2), read to 767 at n 20, 1,085 at n 24 (GPT-2's learned absolute positions) | the GPT-2 position confound, stated on every LATENT-AHEAD (RT M5) |

**Not matchable, stated:** (i) the text model sees every intermediate expression at test time; the latent loop must hold them (D1, §2.3). (ii) Text positions grow ~n² (§3.3); at round 2 the text arm trains rows ≤ 531 and is read past them. (iii) The latent arm has a memory channel that self-training trades (MEM-TRADED / MEM-ERODING); the text arm has no counterpart, its cost is tokens and context.

### 2.2 Labeller and gates

- **REROOT chain** (the text analogue of SELFDEPTH_v1 §2.1; never the simulator): cur ← the input; repeat: prompt x = [EOT] + render(cur) + [' ='], the model's greedy continuation up to its first ' =' or EOT is the step s; s must pass CV-T (§2.3); cur ← parse(s); stop at a single number (label = its value) or at the step cap n + 4 (FAIL).
- **Why REROOT, not the whole chain (FULL, Lee 2502.01612's protocol)** [MEASURED, CPU fp32, base s170_cot_s1; the draft's items (rng 23180) and PROOFS' fresh items (rng 23187), 400 per n pooled at n 9–12]:

  | n | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 20 |
  |---|---|---|---|---|---|---|---|---|---|
  | REROOT keep (pooled) | 1.000 | 0.9975 | 0.985 | 0.985 | 0.955 | 0.875 | 0.775 | **0.645** [0.60, 0.69] | 0.125 |
  | REROOT kept-wrong | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 (fresh) | 0 |
  | FULL keep (pooled) | 0.995 | 0.55 | 0.0625 | 0.000 | 0 | 0 | 0 | 0 | — |
  | FULL kept-wrong / kept | 0 / 199 | 5 / 220 | 5 / 25 | — | | | | | |
  | latent D3-T chain keep (SD2_v1 §2.1) | 1.000 | 0.996 | 0.988 | 0.976 | 0.972 | 0.952 | 0.943 | 0.917 | 0.781 |

  (n 13–15: the draft's 200 per n only; n 16 and n 20 pooled draft + fresh, 400 and 200 items.) REROOT n 9–12: **0.992** [0.986, 0.995], **1 kept-wrong of 1,587** (π 0.9994). FULL: 10 kept-wrong of 245 kept at n 10–11 (**π 0.96**; 5 of 25 at n 11, π 0.80), so FULL fails GATE's keep_n ≥ 0.70 at n 11–12 **and** its π_n ≥ 0.99. FULL is reported at every gate as the descriptive **LABELLER-FULL** row (keep, π on the gate items).
- **TF-GREEDY is the labeller's computation.**
  - **Uniqueness [PROVED; PROOFS P7 PASS].** Given the current tree, a CV-T step is unique up to its value token: CV-T pins the handle (the leftmost-ready node; shape only), its replacement by one leaf, and the canonical rendering. A node is bracketed iff its operator's precedence is below its parent's or it is an equal-precedence right child; reducing the handle removes only its own brackets. Values are single GPT-2 tokens. So the CV-T step is c_k(v) = render(tree with the handle → leaf v), v ∈ {0..10}.
  - **Equivalence [PROVED in exact arithmetic; PROOFS P1 PASS incl. ties (both paths take the first maximal index of the same vector), EOT / ' =' inside a step (a deviation at a fixed slot rejects in both), length caps (the REROOT budget is vacuous: a valid step is shorter than its prompt; the step cap n + 4 is vacuous under CV-T), parse (render is canonical and the parser inverts it), batching (equal-length or masked)].** By induction over the emitted tokens, greedy decoding emits a CV-T step iff, teacher-forced on its own accepted prefix, the argmax equals c_k's token at every fixed slot and is a value token at the value slot. Two batched forwards per step (prompt + prefix → v; [v] + suffix + terminator) decide exactly what greedy decoding decides.
  - **fp32 caveat [ASSUMED negligible; measured]:** TF-GREEDY equals KV-cache greedy generation on 90 / 90 FULL items (draft) and on **1,100 / 1,100 REROOT + 600 / 600 FULL fresh items and 400 / 400 draft items** (PROOFS_TS, rng 23187; 158 REROOT rejections, same failing step; min top-2 logit margin ≥ 1e-3, ≥ 0.40 on REROOT at n ≤ 12). G-T16 re-measures it on the 3070 (§7).
  - **Gold-freeness.** The candidate uses the tree shape only (leftmost-ready handle, render positions, a leaf swap); values are always the model's argmax. N.apply / N.value / N.eval_tokens / N.cot_steps / N.cot_ids are patched to raise in the builder (G-T0). Gold enters only the separate gate and audit processes.
  - **Cost.** Token-positions per item **471 (T1) / 729 (T2)** padded at label batch 128 (364 / 572 unpadded) [DERIVED pc_cost]; pools ≈ **1.0–1.3 h (T1) and 1.6–2.0 h (T2)** on the 3070 [DERIVED v1_cost; throughput ASSUMED]. REROOT KV-cache generation at bs 256–512 would take ≈ 2–4 h per pool [ASSUMED latency scaling], so TF-GREEDY is ≈ 1.5–3× faster and the generation fallback is affordable.
- **Settings:** fp32, tf32 off (matmul and cudnn), label batch 128 (fixed), step cap n + 4; a round's gates, EQUIV and pool on **one card, the local RTX 3070** (as §215 / §216); the 2070 only for a whole round after CARD-GATE (§2.7).
- **Labeller identity** (pre-stated; never chosen on pool or test items):
  - round T1: `boxes/runs_fsq/s170_cot_s1/model.pt` (sha256 36f4659b…3392, asserted);
  - round T2: `s219_t1_s1/model.pt` on **both** windows, with the pre-stated **BASE-LABELLED** fallback per window (`s170_cot_s1`) after the round-1 model fails that window's gate; the fallback passes the same gate on the same gate items (separate eval_gate entry). **Both labellers failing a window → no T2** (no cell, no spend) and the comparison reads TEXT-LABEL-STALL (§4.4).
  - **No T1.5 bridge window, no lowered keep bar for text** (both unmatched; T9, RT m2).
- **Gates.** 2,000 held-out fresh items per n; scored against gold in a separate process; kept = KEEP-T ∧ CV-T. **GATE-T1** n 9–12 (rng 23260, labeller base); **GATE-T2** n 13–16 (rng 23360) and **GATE-T1′** n 9–12 (rng 23361), labeller T1. Rule per window (= GATE-1's): **π_pool ≥ 0.995 ∧ π_n ≥ 0.99 at every n ∧ π^E_pool ≥ 0.99 ∧ keep_n ≥ 0.70 at every n**, with π = share of kept chains whose label (the last value) equals gold and π^E = share whose every value equals gold (point-estimate count rules as SD2_v1: wrong_n ≤ ⌊0.01 kept_n⌋, wrong_pool ≤ ⌊0.005 kept_pool⌋).
  - Exact OC [DERIVED v1_oc, gate_oc]: GATE-T1 at the base's precondition keep (kept 2,000 / 1,995 / 1,970 / 1,970; c_n 20 / 19 / 19 / 19, c_pool 39): **P(pass \| w 0.003) 0.9985, \| w 0.005 0.499, \| w 0.007 0.012**, at the measured w ≈ 6e-4 ≈ 1.000. GATE-T2 at RT's additive T1 projection: 0.9986 / 0.505 / 0.0125; GATE-T1′: 0.999 / 0.542 / 0.015. The same shape as the latent's GATE-1 (0.9986 / 0.504 / 0.0125).
  - **GATE-T2's keep term is the binding one.** The base fallback cannot pass n 16 (pooled keep 0.645 [0.60, 0.69] < 0.70: P(keep_16 ≥ 0.70 at 2,000) ≈ 0 [DERIVED]), so W2 rests on T1's own single-step reach. RT's reach laws (the base's REROOT keep is a function of input length L_0: 0.90 at L_0 ≈ 42, 0.70 at 47.7, 0.50 at 51.4, trained maximum 31) project T1 (trained copies ≤ 45) at **0.997 / 0.995 / 0.989 / 0.978** (additive), 0.996 / 0.994 / 0.990 / 0.984 (ratio) and 0.987 / 0.974 / 0.949 / 0.903 (half transfer) at n 13–16; each passes keep_16 ≥ 0.70 with P ≈ 1; only near-zero transfer stalls [DERIVED rt_reach, v1_oc].
  - Descriptive at every gate: **LABELLER-FULL**, **KEPT-VS-DRAWN** (mean P + C, height, bracket count of kept vs drawn per n), **EQUIV** (G-T16).

### 2.3 Filter and target (the text analogues of KEEP ∧ COMPOSE-VALID), and the D1 statement

Per step k, with tree_{k−1} the parsed previous expression (tree_0 = the parsed input) and s_k the model's emitted tokens up to the terminator:
- **PARSE-T:** s_k parses over the §170 vocabulary. **FRAME-T:** \|s_k\| = \|c_k\|, every token outside the handle's span equals e_{k−1}'s. **SPAN-T:** exactly one contiguous span replaced, by exactly one token. **VALUE-SLOT:** that token is a value token (' 0' … ' 10'); **its value is never checked.** **ORDER-T:** the span is the leftmost-ready handle of tree_{k−1} (shape only). **CANON-T:** render(parse(s_k)) == s_k. **TERM-T:** the terminator is ' =' iff s_k has an operator, EOT iff s_k is a single number.
- **KEEP-T** = PARSE-T ∧ TERM-T at every step ∧ exactly n steps ∧ the last step a single number. **CV-T** = KEEP-T ∧ FRAME-T ∧ SPAN-T ∧ VALUE-SLOT ∧ ORDER-T ∧ CANON-T at every step. **Kept = KEEP-T ∧ CV-T ⟺ s_k = c_k(v_k) at every k** [PROVED]. The decomposed flags are logged per failure.
- **Record** (pool): keys exactly **{syms, n, chain_label, values}** (values = the model's v_1..v_n; chain_label = v_n). No gold field (root, t, y, gold, cot) ever.
- **Target:** compose_T(syms, values) = ids(c_1(v_1)) + [' ='] + … + ids(c_n(v_n)) + [EOT], from shape and the model's values only. Under CV-T it equals the model's own REROOT chain token for token; it equals N.cot_ids(gold root) iff every v_k is gold [PROVED]. Loss = the base's textcot CE on every target token, prompt masked.
- **The filter is the latent's filter applied verbatim**: stricter than Literature's minimum (final answer / step-count validity, LIT_FAITHFULNESS_SELFIMPROVE) (RT m4). It binds for text (frame / copy failures) and ≈ never for the latent (CV ≡ KEEP on D3-T).
- **D1, stated precisely** [PROVED, information argument; LIT_STEPFACTOR_D1, Bachmann & Nagarajan 2403.06963] (reworded per RT m3 / PROOFS P7.3):
  - **Under REROOT, D1 is not a labeller asymmetry.** Both labellers read the handle's operands from a fresh root (text: the prompt e_{k−1}; latent: x = [EOT] + render(cur) + [' ='] with one pass, K = 1). A value check (apply(op, a, b) on the parsed handle) would be a gold step-value check; **it is equally available to the latent filter** (check_step holds the parsed handle), and **both arms forgo it**. VALUE-SLOT accepts any value token; the builder's N.apply raises (G-T0).
  - **The D1 asymmetry is in the trained target and at test time:** text targets and generation keep every operand in context, while the latent's E_self targets and K′ = n + 2 rollout must hold values in slots across passes. That difference is the object of comparison; it is stated, not removed.
  - The draft's [CONJECTURE] "π_text ≥ π_latent because a local read vs a held value" is **dropped** (both labels are local reads). Any π difference comes from the decoders: autoregressive re-emission (text) vs per-slot decode (latent).
  - Value tokens are a small share of the text target: n of C(n) (n 12: 12 / 217.7 = 5.5 %; n 16: 16 / 385.5 = 4.15 %) [DERIVED pc_tokens]. The trainer logs `self_value_ce` and `self_copy_ce` separately.
- **π^E ≤ π for text:** a wrong intermediate value usually changes the label, but × 0 (and other absorbing cases) can mask it, so π^E ≤ π; the gate reads both, as the latent's.

### 2.4 Contamination rules (registration text; T11)

1. **Training items are fresh generated items only.** Pools: `nested170.generate(rng, n)`; T1 rng default_rng(23200), drawn until **97,000 kept per n at n 9–12** (388,000); T2 rng default_rng(23300), drawn until **65,000 kept per n at n 13–16 and 32,500 per n at n 9–12** (390,000). One fixed interleaved order each (permutation rng 23201 / 23301). Fresh-once; exhaustion or reuse = void.
2. **Exclusion by exact rendered syms** against:
   - **test_set(1000, n_max 30)** (30,000 items, seed 17100; covers test_set(1000, 24) and the **sealed n 25–30**, 216-5);
   - SERIAL (rng 17300, 2,200 items);
   - the 213-37 fresh items (rng 21337, n 9–24, as E3 consumes them);
   - **every latent gate**: s215_gate1, s216_gate2_s1 (GATE-2 + GATE-1′), s216_gate1_s2 (gate_items.json each);
   - **every latent pool**: s216_pool_r1_s2/pool_r1.jsonl (on disk); s215_pool (pool_r1, deleted; rebuilt and verified once, sha c2e226ae, 744d79fc) and s216_pool_s1 (pool_r2, deleted). A deleted pool is either regenerated from its pool_meta (seeds + builder commit) and sha-checked, or covered by the collision bound P(render = s) ≤ (1/11)·40⁻ⁿ per draw [PROVED, PROOFS_SELFDEPTH §8; ≤ 1e-4 expected collisions at n 9 against a 444k pool], stated in pool_meta;
   - **the TEXT-SELF precondition items** (read by the base): the full one-stream, 200-per-n draws at n 9–20 of **rng 23180** (draft; redrawn on hits against test_set(1000, 24) + SERIAL) and of **rng 23187** (PROOFS; redrawn on hits against the same set plus the rng-23180 items), each regenerated by the scripts' own code (docs/research_textself_2026-10-18/scripts/ts_label_check.py, proofs_ts/pc_reroot.py) and sha-recorded in pool_meta (V7);
   - the **G-T18 smoke items** (rng 23195, n 24 shapes, read by the base on the 3070);
   - every TEXT-SELF gate and pool already built (T2's pool excludes T1's pool and gates).
   Per n: draws, collisions and duplicates logged.
3. **No gold reaches training.** Builder gold patches (N.apply, N.value, N.eval_tokens, N.cot_steps, N.cot_ids raise) installed before generation; record keys exactly {syms, n, chain_label, values}, asserted by the loader; no root key in any self item; a self-batch smoke counts cot_ids / cot_steps calls on self items = 0 (G-T0). The trainer never opens a gate directory (grep: no `s219_gate` in boxes/textself_ft.py).
4. **Gold is used only for evaluation, the gates, LABEL-AUDIT-T and MEMO-GAP-T**, each in a separate process.
5. **Test items:** the text arm reads the same test_set(1000, 24) items as the latent arm. Text models were evaluated on n 1–16 before (s170 eval.json, bf16, read before this text; they inform predictions only). **n 17–24 have never been read on a text model** (the draft's / PROOFS' n 17–20 reads were fresh items).
6. **The sealed n 25–30 (216-5) are never evaluated or used**, except as exclusions (item 2).
7. **The labeller is fixed in advance** (§2.2). A round's gates and pool use one card (3070) and fixed settings (fp32, tf32 off, label batch 128).
8. **Post hoc:** pool ∩ (item 2) = ∅ re-verified on the pool file. Any hit, gold key, reuse (`self_reuse` > 0) or exhaustion → **CONTAMINATED** (censor).
9. **Seeds** (re-grepped at 796a3343 in docs/THEORY.md, docs/research_*/*.md, docs/research_*/**/*.py, boxes/*.py, theory/*.py, scratch/*.{py,sh}, excluding this design's own folder: **0 hits each**): precondition 23180 (draft), 23187 (PROOFS); shape-only, never shown to a model 23188–23194 (draft / PROOFS) and 23996–23999 (RT); **G-T18 smoke 23195** (model-read); T1 pool 23200 / 23201, GATE-T1 23260, LABEL-AUDIT-T1 23270, LABELLER-KEPT-T1 23280; T2 pool 23300 / 23301, GATE-T2 23360, GATE-T1′ 23361, LABEL-AUDIT-T2 23370, LABELLER-KEPT-T2 23380. Coin stream rng_s = default_rng(1 + 21800), the latent's. The draft's s2 seeds (23400–23580) are dropped.

### 2.5 Cells and training (s1 lineage; RTX 3090 only)

| element | spec |
|---|---|
| Cells | **`s219_t1_s1`**: FT from `boxes/runs_fsq/s170_cot_s1/model.pt`. **`s219_t2_s1`**: FT from `s219_t1_s1/model.pt`. **`s219_base_s1`**: eval-only directory holding the fp32 re-eval of `s170_cot_s1/model.pt` (model_sha256 asserted); `s170_cot_s1/` is never written |
| Trainer | `python -m boxes.textself_ft` (new; V1): parser = nested170's 57 keys verbatim + the FT keys; the textcot path copied (loss_items textcot, fsq120.lr_at, AdamW wd 0.01 via make_opt, clip_grad_norm 1.0, bf16 autocast, checkpointing); `--arm textcot` asserted |
| Init | `--init_from <model.pt>`: strict load (missing = unexpected = [] asserted, printed INIT_FROM), sha256 printed and asserted, fresh AdamW |
| Recipe | **s170_cot_s1/args.json with only** `--init_from`, `--steps 12000`, `--lr 3e-5`, `--self_pool`, `--p_self 0.5`, `--self_micro` (T1 2, T2 4), `--no_final_eval`, `--draws_log_steps 12000`, `--args_ref` and `--out` changed (§4.6). Unchanged: arm textcot, bs 64, n_max_train 8, seed 1, warm-up 100 then cosine to 0, AdamW wd 0.01, clip 1.0, bf16 autocast, ckpt_every 2000, auto_micro (plain micro 1), mem_budget 21.0 |
| Plain stream | rng = default_rng(1); train_items(rng, 64, 8, EXCL) as the base: **each round replays the base's first 12,000 plain batches** (disclosed, as SD2's D11; a(n) = 1.000 at n 1–8 [MEASURED eval.json]). G-T3 checks plain_hash at steps 1..200 against a separate-process replay |
| Coin stream | rng_s = default_rng(1 + 21800); per step `coin = rng_s.random() < 0.5`; on a self step 64 × `rng_s.integers(0, 5)` drawn and discarded (the latent's self-U draws). The plain batch is drawn first every step (and discarded on a self step). So **5,992 self steps = 383,488 self items at the latent's step indices** [PROVED by stream identity; PROOFS P4: coins 12,000 / 12,000 and every U value match s215_ftself_s1 and s216_r2_s1] |
| Self batch | the next 64 pool records at the cursor; target compose_T(syms, values); loss = the base's textcot CE (prompt masked); no root key exists (any gold path is a KeyError) |
| Pools (m7) | T1 **388,000** (97,000 per n; spare 4,512 = 1.18 %, 70.5 self steps); T2 **390,000** (65,000 / 32,500 per n; spare 6,512 = 1.70 %); one fixed interleaved order; expected consumed per n T1 95,872 (SD 28.9), T2 63,914.7 per new n (SD 29.8) and 31,957.3 per old n (SD 22.1) [DERIVED v1_cost]; the realised counts are fixed by the pool order and **precomputed before launch** (G-T3) |
| Micro | plain 1 (the base's); self **fixed** T1 2 / T2 4 (m8; = the latent's realised self micro); G-T9 confirms peak ≤ 21.0 GiB (projected 14.7 / 12.8) |
| Controllers (m13a) | none reads the loss on the textcot path (AnnealGate / KW / CSR are inject-only [VERIFIED by RT]); clip 1.0 reads mixed batches, as the latent's |
| Logging (log.jsonl, every 100) | step, loss, plain_loss, self_loss, **self_value_ce**, self_copy_ce, self_frac, pool_cursor, n_self_steps, self_reuse (must be 0), lr, grad-norm summary, nonfinite |
| Draw log | draws_log.json, steps 1..12,000: coin, self n list, plain_hash (sha1 of the plain batch's syms) |
| Checkpoints | full state every 2k (model, AdamW, np rng, torch rng, rng_s, cursor, counters, step, elapsed) **mirrored off-host** and sha-checked after the first copy (user rule 2026-10-15); resume tested before launch (G-T4); deleted at landing; model.pt kept (T1 is T2's labeller) |
| Seal (T3) | `--no_final_eval` (required true): no generation of any kind in the trainer; the textcot eval-memory check (a generation on a test n-16 item) is skipped; `--stop_at` returns before any eval; `--eval_at 0`, `--probe_loss_every 0` (required). The drain fetches model.pt, log.jsonl, args.json, draws_log.json, mem / time JSONs only |

### 2.6 Code edits (list only; none written; Audit's diff review G-T13)

| # | where | edit |
|---|---|---|
| T1 | boxes/textself_ft.py (new) | parser copy + FT keys (`--init_from`, `--self_pool`, `--p_self`, `--self_micro`, `--no_final_eval`, `--mem_check_self`, `--mem_check_steps`, `--stop_at`, `--draws_log_steps`, `--args_ref`); pool loader (sha256 vs pool_meta, key-set assert, n from syms); rng_s coin replay incl. the 64 discarded draws; compose_T targets (`SD-LABEL-MAP-T`: `it["values"] = rec["values"]`); per-type micro; log fields; draws_log; rng_s + cursor in the checkpoint and the resume branch; `--mem_check_self` (the 64 longest pool records of the deepest n, two optimiser steps, peak after step 2); no eval code path under `--no_final_eval` |
| T2 | boxes/textself.py (new) | candidate (shape only), TF-GREEDY REROOT and FULL labellers (batched, right-padded, per-row positions and key masks), CV-T decomposition, compose_T, record format, load_pool; independent REROOT / FULL KV-cache generation for G-T16. CPU references: scripts/ts_label_check.py (label_reroot / label_full), proofs_ts/pc_reroot.py (independent generation) |
| B1 | scratch/s219_build_pool.py (new; from s216_build_pool.py) | gates stage (GATE items, exclusion, gold patch) and pool stage (per-n targets, interleave, pool_meta with seeds, draws, collisions, labeller sha256, card, dtype, label_bs, T_max_by_n, L0_max_by_n, excl breakdown incl. the precondition-stream sha) |
| B2 | scratch/s219_gate.py (new) | gold scoring of GATE-T1 / GATE-T2 / GATE-T1′ (π, π^E, keep, Wilson CIs, per-item records), LABELLER-FULL, KEPT-VS-DRAWN, the BASE-LABELLED fallback, G-T16 EQUIV |
| E1 | boxes/text_eval.py `--part text` (new) | §2.7 contract (gold-free budget, CTX-OVERFLOW, break before feeding row 1,024, per-item store, items_sha256, model_sha256) |
| E2 | boxes/text_eval.py `--part textself` | CPU parts (LABEL-AUDIT-T, CONTAMINATED) and GPU parts (MEMO-GAP-T, LABELLER-KEPT-T), each its own key |
| S1 | theory/score_s219.py (new) | §4.6 |
| Q1 | scratch/s219_*.sh | stage chain (§2.9), drain waiters at launch, off-host mirrors, admission, GUARD_STOP |

**nested170.py is not edited** (sha256 pinned in G-T7); vbank.py imports it, and §217's runner / LANG-VB jobs may import it while TEXT-SELF builds.

### 2.7 Eval contract (per text model: `s219_base_s1`, `s219_t1_s1`, `s219_t2_s1`)

**eval_text.json** (`python -m boxes.text_eval --run <cell> --part text --cot_eval_bs 32 --device cuda:<3070>`):
- **Items:** test_set(1000, n_max 24), n 1–24 (24,000 items, test seed 17100), test order. **items_sha256** as SD2_v1 §4.2 (json.dumps of [[syms, int(t)]] over the 24,000 items, separators (",", ":"), ensure_ascii); the scorer recomputes it.
- **Generation:** greedy, KV cache, nested170.cot_answers' loop semantics (batched by equal prompt length, cot_eval_bs 32), **fp32, tf32 off**, on the **RTX 3070**.
  - **Budget, gold-free:** per batch min(max C_struct + 50, 1025 − P), C_struct = the item's shape-only chain length (equal to len(cot_ids) on all 24,000 test items [DERIVED, asserted by ts_tokens and pc_tokens]); the part never calls cot_ids. The loop **breaks before feeding row 1,024** (PROOFS P3).
  - **CTX-OVERFLOW:** an item with P + C_struct > 1025 is not generated and counts as a failure (§2.8).
  - **Non-termination:** no EOT within the budget, or a tail that is not one number → answer −1 (a failure).
- **Per-item store:** idx, n, y, P, C_struct, overflow, answer, ntok, eot, **form_valid**, **strict** (form_valid ∧ answer == y), first_bad_step, **first_bad_pos**.
- **Aggregates:** acc_by_n (**a_T(n)**: answer == y over all 1,000 items, overflow and non-termination = failure), acc_feasible_by_n, form_valid_by_n, strict_by_n, ntok_mean_by_n, first_bad_pos quantiles by n, chance_by_n (majority-class share).
- **Meta:** run, model_sha256, card, dtype fp32, tf32 false, cot_eval_bs 32, test_seed, test_n_max 24, items_sha256, budget rule, commit, sec, peak_gib.
- **Projected time** 1.7–2.9 h per model on the 3070 (290k–332k sequential decode steps; the upper end carries RT's KV-cat context term) [DERIVED pc_cost / rt_reach; ASSUMED form]. G-T18 times the longest batch.

**eval_textself.json** (per FT cell; gold only here, separate process):
- **CPU parts (registered; read for the round-2 condition):** **LABEL-AUDIT-T**: 5,000 random consumed records (T1 rng 23270, T2 23370): π and π^E, pooled and per window, Wilson 95 % CIs; kept-wrong records with (syms, step, op, a, b, the wrong value, the gold value). **CONTAMINATED**: pool ∩ §2.4 item 2 = ∅ (re-checked on the pool file), self_reuse = 0.
- **GPU parts (descriptive):** **MEMO-GAP-T**: \|a on the first 2,000 consumed pool items of n − a on the 2,000 gate items of n\| per window n (free generation, the eval settings), flagged > 0.02. **LABELLER-KEPT-T**: REROOT keep / π of the cell's model on 500 fresh items per n 9–24 (T1 rng 23280, T2 23380).

**Base first (G-T19):** `s219_base_s1/eval_text.json` exists, and a Theory note records its bars (§4.3), Δ_0 and the BASE-PARITY branch, **before any T1 output is read**. BF16-VS-FP32 (descriptive): the committed bf16 `s170_cot_s1/eval.json` against the fp32 re-eval at n 1–16.

**Card rule (T5).**
- Every registered text eval (all three eval_text files and every eval_textself GPU part), every gate, EQUIV and pool runs on **the local RTX 3070**, fp32, tf32 off.
- **The RTX 2070 is a backup only, behind CARD-GATE (G-T21):** `s219_base_s1` eval_text is also run on the 2070 (output `eval_text_2070.json`, descriptive); the 2070 may take later work only if **\|a_2070(n) − a_3070(n)\| ≤ 0.005 at every n 1–24 and \|Δn50\| ≤ 0.02**, with item flips listed. It then takes **whole files** (a model's eval_text on one card only; a round's gates + EQUIV + pool on one card only); the card is recorded per file and the scorer asserts one card per file.
- **3070 lost:** (1) lost then recovered → the incomplete parts re-run on the 3070; partial outputs kept as `*.void_gpulost`. (2) Not recoverable before the base eval exists → the text evals move together to one named local card, the RTX 4060 (fp32, cot_eval_bs 32), and **the latent comparators' n50s are re-evaluated on that card first** (D3-T and FT-SELF eval_vb acc part, R2 eval_deep), every Δ then reads on that card only (SD2 Addendum 1's logic). A partial move within a model's registered eval_text is not allowed.

### 2.8 Context overflow and CTX-BOUND (pre-stated)

- GPT-2 has 1,024 position rows (core.GPT2KV adds `wpe[p0:p0 + T]`); an item is context-feasible iff **P + C ≤ 1025** [PROOFS P3 PASS]. C is shape-only, so feasibility is a gold-free item property fixed before any model runs.
- Test items: **only n 24 overflows (29 / 1,000; max P + C 1,085; seven items exactly at 1,025); n ≤ 23 all fit** [DERIVED ts_tokens, pc_tokens]. Overflow items are failures in a_T(n) (the intrinsic token cost); CTX-FEASIBLE accuracy (971 items) beside every reading that touches n 24. Immaterial for every expected crossing (RT m14).
- **CTX-BOUND:** a text round 3 is context-infeasible on GPT-2: fresh shapes give feasible shares 0.96 (n 24), **≈ 0.52–0.55 (n 25)**, ≈ 0.08 (n 26), ≈ 0.003 (n 27), 0 (n ≥ 28). TEXT-SELF is registered for two rounds; the comparison ends at round 2. It is worded as **GPT-2's 1,024 learned position rows**, not a property of text CoT.

### 2.9 Order of operations (the seal, T3)

1. Registration commit (this text, seeds, TEXT_NEUTRAL_NEW, exclusion rule) → code (§2.6) → CPU gates (G-T0, T2, T4, T5, T7, T8, T12, T17) → Audit G-T13.
2. **3070:** G-T18 (fresh rng-23195 shapes, output to scratch/_tmp) → **base eval_text** → Theory note (G-T19: base bars, Δ_0, BASE-PARITY branch). GATE-T1 (+ LABELLER-FULL, EQUIV) → G-T15 → T1 pool → G-T1 / G-T17 (any order relative to the base eval; neither reads the other).
3. **Cloud 3090:** T1 deploy (HOSTPROBE, G-T9, G-T10, admission) → 12k steps, off-host mirrors → drain (no eval file; G-T20 SEAL).
4. **3070:** T1 eval_text → CPU: LABEL-AUDIT-T1, CONTAMINATED → Theory note: T1 readings + Δ_1 + the round-2 condition.
5. **3070:** only then GATE-T2 + GATE-T1′ (T1 labeller; base fallback per window) → G-T15 → T2 pool. T1's eval_textself GPU parts run in the gaps.
6. **Cloud 3090:** T2 → drain → G-T20.
7. **3070:** T2 eval_text → eval_textself → `score_s219 --final s1`.

No T1 or T2 output at any n ≥ 13 (generation, labelling, gate) exists before that model's eval_text.json (V3).

---

## 3. Mathematics

### 3.1 S0-T, the support identity for text [PROVED]

- **Steps.** Every kept chain of round r has exactly n ≤ B_r steps (B_1 12, B_2 16): KEEP-T. Every target is the composition of n CV-T steps.
- **Rows.** Every target token sits at a position ≤ T_train,r = max (P + C) over the consumed records: **≈ 313 at B 12 and ≈ 531 at B 16** (pool-sized draws; pool_meta's T_max_by_n is the registered value) [DERIVED pc_tokens]. The base trained rows ≤ 147 and copies (input length L_0) ≤ 31; T1 trains copies ≤ 45, T2 ≤ 59.
- **(a) Zero loss past support.** A model that fails on any item with n > B_r, or on any token at a position > T_train,r, pays no loss on the round-r mixture. **Stopping at the support is irreducible by loss in text too** (the text H_shift). Unlike the latent arm, whose rows grow ~3n, the text arm's step frontier and row frontier move together (rows ∝ n², §3.3), so its training cannot separate them.
- **(b)** T-ABSORB (n 9–12; n 13–16 in round 2) is in support. n 13–16 after T1 and n 17–24 after T2 are past it in steps **and** rows (at n 17–20 the read rows run to 767 against 531 trained).
- **(c) Pretrained rows.** GPT-2's rows > T_train,r carry only pretraining for this task (learned absolute positions). The latent arm's nuisance H_T (rows 61 trained vs 83 read) is the small version of what is intrinsic here.

### 3.2 S1-T, exposure under the replayed coin stream [PROVED by stream identity; DERIVED counts]

- rng_s = default_rng(seed + 21800); one `random()` per step; on a coin < 0.5, 64 `integers(0, 5)` draws (bound 5 matters: a bound of 1 consumes nothing in numpy). This is vbank.draw_step's consumption [VERIFIED vbank.py l.755, l.764]; PROOFS P4 replayed it and matched s215_ftself_s1's and s216_r2_s1's draws_log.json on **12,000 / 12,000 coins and every logged U** (5,992 / 5,992), so **both text rounds take their 5,992 self steps at the latent's step indices**.
- Self items per round **383,488**; per-n expectations as §2.5. Plain items: the base's first 12,000 batches, each round.
- A "stop past the trained rows" policy costs nothing on the loss (S0-T a); no deadline table is needed because no reading uses it.

### 3.3 Positions and context grow as n² [constant MEASURED; form DERIVED under uniform bracket drops]

- e_0 has L_0 = 2n + 1 + 2b tokens (b bracket pairs); step k removes one operator and one operand (−2) and some bracket pairs (−2 each). Then C = Σ_{k=1..n}(L_k + 1) = n(L_0 + 1) − n(n + 1) − 2Σ_k b_k ≈ (1 + β)n² + (1 − β)n with b = βn.
- **Measured** (test items): C̄ / n² = **1.489 / 1.506 / 1.507** at n 8 / 16 / 24; P̄(16) = 49.96 [DERIVED pc_tokens, PASS].
- **Test-time cost per item:**

| | text (free greedy generation) | latent (K′ = n + 2 passes) |
|---|---|---|
| sequential steps | C(n) ≈ 1.5 n²: 386 / 606 / 868 at n 16 / 20 / 24 | n + 2: 18 / 22 / 26 |
| token-positions | P + C ≈ 1.5 n² + 3n: 436 / 668 / 942 | (n + 2) × T: ≈ 899 / 1,365 / 1,921 |
| memory at test | KV cache over P + C rows | bank of (n + 2) × T states |

  The text format uses ≈ 0.48–0.49 of the token-positions but **21–33× more sequential steps** at n 16–24 [DERIVED, PROOFS PASS]. TOKEN-COST (§4.4) reports both per correct answer.

### 3.4 Why the text arm can extrapolate differently

- **(a) Per-token iid errors cannot explain the base curve** [DERIVED from MEASURED eval.json, PROOFS PASS]: a(8) = 1.000 on 1,000 items bounds ε ≤ 3 / (1,000 × 95.28) = 3.15e-5 per token, which predicts a(11) ≥ (1 − 3.15e-5)^{181.4} = 0.994; the measured a(11) is 0.234. The base's failure is **keyed** to something that changes between n 8 and n 11: the step count (> 8 untrained) or the position (> 147 untrained).
- **(b) Self-label errors do not accumulate with chain position under REROOT** (each step is a short fresh context); they depend on the copy length L_k. FULL breaks where REROOT keeps (§2.2).
- **(c) Per-step rewrite.** Each text step re-emits the remaining expression: L_k tokens, of which L_k − 1 are copies. A copy error breaks FRAME-T; the answer may still be right by chance (free-generation floor ≈ 0.04–0.10: base n 13–16 0.100 / 0.088 / 0.082 / 0.040 [MEASURED]).
- **(d) Value errors are rare in support** (a(n) = 1.000 at n 1–8, ≈ 36k value tokens) and exist past it (FULL 10 / 245 kept at n 10–11; REROOT 1 / 1,587 at n 9–12, 1 / 942 over PROOFS' fresh REROOT kept chains). Most failures are form failures: copies broken at a row frontier (FULL) or at a copy-length frontier (REROOT's step 1: **149 of 165** draft failures at n 9–16).
- **(e) Two frontiers, both moved by self-training.** The base fails at the edges of what it trained: rows (≤ 147; the FULL failure window sits at rows ≈ 170–200 = **1.16–1.35 ×** 147) and copy spans (≤ 31 tokens; REROOT keep 0.90 / 0.70 / 0.50 at L_0 ≈ 42 / 47.7 / 51.4, i.e. ≈ +11 tokens or ×1.35 past the trained maximum [DERIVED rt_reach]). T1's targets extend both (rows ≤ 313, copies ≤ 45), T2's again (rows ≤ 531, copies ≤ 59). Scaling the base's reach [ASSUMED], T1's row frontier (1.31 × 313 ≈ 410 rows ≈ n 15) binds before its copy frontier (≈ 56–61 tokens ≈ n 18–19) at test time; the copy frontier decides T1's **labelling** reach at n 13–16 (§2.2).
- **The labelling-reach gap is the output format** (RT m2, T9): the latent labeller decodes the next state per slot (every untouched slot copies itself at the same position: vbank.recode_targets / rerender), so its copy never fails (CV ≡ KEEP); the text labeller must re-emit the shortened expression token by token. At equal training (gold n ≤ 8) the latent single step reaches **about 5 operations further** at the 0.78 keep level (D3-T 0.781 at n 20 vs text 0.775 at n 15). LABELLER-REACH (§4.4) reports it; it is never read as depth.

### 3.5 Hypotheses and pre-stated panels [CONJECTURE for each H; DERIVED numbers ts_panels / pc_panels from the bf16 base curve; the fp32 re-eval replaces them before any T1 output is read]

Base s1 (bf16): a_0(9..12) = 0.995 / 0.707 / 0.234 / 0.123; floor c = mean a_0(13..16) = 0.0775; n50 10.438, n90 9.33.

| panel | rule (round r, support B_r, trained rows T_r) | r1: M (n 13–16) | r1 n50 | Δn50_1 vs 15.46 | r2: M2 (n 17–20) | r2 n50 | Δn50_2 vs 20.54 |
|---|---|---|---|---|---|---|---|
| **N-SHIFT** (step-keyed, additive) | a_r(n) = a_0(n − 4r) | 0.515 | 14.44 | **−1.02** | 0.515 | 18.44 | **−2.10** |
| **N-PROP** (step-keyed, proportional) | a_r(n) = a_0(8 + (n − B_r)·8 / B_r) | 0.749 | 15.66 | +0.20 | 0.888 | 20.88 | +0.34 |
| **POS-PROP** (row-keyed, proportional) | P(correct) = c + (1 − c) σ((1.31 − tot / T_r) / 0.076), fitted on n 9–12 | 0.742 | 15.56 | +0.10 | 0.873 | 20.51 | −0.03 |
| **POS-ADD** (row-keyed, additive) | P(correct) = c + (1 − c) σ((45.6 − (tot − T_r)) / 11.2), fitted on n 9–12 | 0.513 | 14.48 | **−0.98** | 0.531 | 18.54 | **−2.00** |
| BETTER (pre-stated offset) | text margin grows faster than the latent's | — | — | +1.5 | — | — | +3.0 |
| WORSE (pre-stated offset) | margin shrinks below N-SHIFT | — | — | −2.0 | — | — | −4.1 |

- PROOFS re-fitted POS-* on a 5× finer grid (x0 1.31 / w 0.076; x0 45.6 / w 11.2; SSE 0.00198 both, so **the base alone cannot split them**) and re-derived N-* exactly: PASS. With the pool-size row maxima (313 / 531) the POS-* deltas are +0.14 / +0.05 and −0.94 / −1.94: no reading changes.
- **What the comparison discriminates:** additive text extrapolation (steps or rows) predicts **LATENT-AHEAD** at both rounds (Δ ≈ −1, then ≈ −2); proportional extrapolation predicts **PARITY** (\|Δ\| ≤ 0.34); TEXT-AHEAD needs a text margin growing faster than the latent's ×1.29 law. H_pos vs H_n inside a family is descriptive (FAIL-POS).
- **Literature priors:** Lee 2502.01612 (reverse addition, margin ≈ 3–5 digits over ≈ 90 rounds: additive; NoPE, from scratch); Huang 2511.07378 Thm 5.1 (text CoT generalises a constant factor of the training length: proportional; one-layer NoPE assumptions); GPT-2's learned absolute positions (the classic length-generalisation failure: favours row-keyed failure). The latent's ×1.29 is an in-house empirical constant (216-9).
- **Priors** [ASSUMED, unchanged from the draft]: N-SHIFT 0.25, POS-ADD 0.20, N-PROP 0.15, POS-PROP 0.20, BETTER 0.10, WORSE 0.10 (additive 0.45, proportional 0.35).

### 3.6 S2-T label noise, memory and time

- **(a) Identity** [PROVED, as SELFDEPTH_v1 S2(a)]: π = P / (P + φ(1 − P)), P the labeller's strict-correct rate, φ the share of its wrong chains that pass CV-T. For text φ counts **value-only** errors (every copy or ordering error fails CV-T).
- **(b) The guard is direct counts** (GATE-T* at 2,000 per n, LABEL-AUDIT-T at 5,000), not a κ bound; the latent κ premises (A1′–A5) are unverified for text [ASSUMED], so no bound is quoted.
- **(c) Class mass** [PROVED, PROOFS_SELFDEPTH §3c]: a majority-wrong pattern class has mass ≤ 2(1 − π): ≤ 1.3e-3 at the base's measured REROOT π 0.9994. LABEL-AUDIT-T prints each kept-wrong handle (op, a, b → v), so a systematic arithmetic cell is visible (Lee App B.4).
- **(d) Compounding:** T2's labeller T1 was trained on base labels at 1 − π_1; LABEL-AUDIT-T2 compares its kept-wrong handles with T1's (the A5 check).
- **Training memory** [DERIVED pc_cost; form ASSUMED, validated by G-T9]: GiB ≈ 1.87 + 0.082·T / micro at bs 64 (the 12.99 GiB base point is a backward without AdamW's 0.93 GiB; slope (12.99 − 0.93) / 147). **T1 (rows 313) micro 2: 14.7 GiB; T2 (rows 531) micro 2 23.7 > 21.0, micro 4: 12.8 GiB.** Hence micro 2 / 4, equal to the latent's realised self micro.
- **Training time** [DERIVED ts_cost; form ASSUMED, G-T10 decides]: t(T) = 0.27 s·(T / 134)(1 + T / 4,608) / (1 + 134 / 4,608), calibrated on the base's steady 0.26–0.27 s/step (3090, plain batches of mean batch-max T 134). Self step **0.59 s (T1), 1.03 s (T2)**, plain 0.27 s: **1.43 h (T1), 2.16 h (T2)** steady; host variance up to ≈ 2× (memory note; the base's own first 12k steps ran at 0.48–0.74 s/step).
- **Eval memory** [DERIVED, PROOFS PASS]: fp32 KV 73.7 KB per token per item; at cot_eval_bs 32 × 1,024 rows 2.42 GB, ≈ 5.1 GiB with both caches live across a step plus weights: under the local 6.0 GiB cap.

### 3.7 The noise model for Δ, and BASE-PARITY [DERIVED v1_oc; model ASSUMED]

- **Measured spread** [DERIVED pc_oc from committed files]: three between-lineage n50 differences, each one draw of Δ under no difference when both arms' lineage SDs are equal: latent r0 +0.201, latent r1 −0.132 (R1-s2 censored; its n50 descriptive, 216-9), text r0 (bf16) +0.159. RMS **ŝ = 0.1666 (3 df; 95 % χ² CI 0.094–0.621)**. The data show no round-to-round growth (r1 0.13 < r0 0.20).
- **Model** (the draft's, kept as the conservative form): Δ_1 ~ N(m_1, s²), Δ_2 ~ N(m_2, 2s²), corr 0.5. Central s = ŝ; sensitivity s = 0.20 and **0.35**. The text-FT lineage noise itself is unmeasured (one text lineage).
- **Protocol bias** (RT M4 / M5): a constant offset b on both rounds shifts the readings (§6 bias rows). The card (T5), dtype (fp32 both), the seal (T3) and BASE-PARITY remove the known sources; a residual b ≈ ±0.15 costs P(PARITY \| no difference) 0.964 → 0.917.
- **BASE-PARITY** (T6): Δ_0 = n50_T(0, fp32) − 10.31. Under a shared proportional law an offset scales with support (Δ_0 · B_r / 8: ×1.5, ×2), so at \|Δ_0\| = 0.25 a proportional parity puts Δ_2 at the band edge 0.5. Hence: **\|Δ_0\| ≤ 0.25 → levels** (the registered statistic); **\|Δ_0\| > 0.25 → gain Δ^g_r = Δ_r − Δ_0** with the same four-way rule. Both are always reported. On the bf16 curve Δ_0 = +0.13 (levels branch expected); under levels, a proportional offset of +0.20 / +0.26 gives P(PARITY) 0.825 at ŝ.
- **Gain noise:** Var(Δ^g_1) = 2s²(1 − ρ_0), Var(Δ^g_2) = s²(3 − 2√2 ρ_0), with ρ_0 = corr(e_0, e_r) ∈ {0, 0.5} [DERIVED]. P(PARITY \| no difference) 0.900 (ρ_0 0) / 0.981 (ρ_0 0.5) at ŝ; 0.473 / 0.645 at s 0.35. The gain statistic adds round-0 noise unless lineage effects persist, so it is the fallback, not the default (RT M6).

---

## 4. Readings, definitions and the scorer's field contract

Notation: a_T(n) = eval_text `acc_by_n[n]` (free greedy generation; answer == y over all 1,000 items; overflow and non-termination are failures). Chance ≈ 0.04–0.11. Every per-cell reading is comparator-free. The latent numbers are the committed s1 readings and are never recomputed here.

### 4.1 Files

**Per FT cell** (boxes/runs_fsq/s219_t1_s1/, s219_t2_s1/):

| file | producer | fields read | readings |
|---|---|---|---|
| log.jsonl | trainer | step, loss, plain_loss, self_loss, self_value_ce, self_reuse, pool_cursor, n_self_steps, nonfinite | dead (last step < 12,000), reuse = 0 |
| args.json | trainer | vs s170_cot_s1/args.json (§4.6) | ARGS-DIFF |
| draws_log.json | trainer | coin, self_n, plain_hash | G-T3 (coins == the latent round's; plain_hash == the base stream's first 200) |
| **eval_text.json** | text_eval part `text` | meta, items, acc_by_n, acc_feasible_by_n, form_valid_by_n, strict_by_n, ntok_mean_by_n | §4.2–§4.5 |
| eval_textself.json | text_eval part `textself` | label_audit, contamination (CPU, registered); memo, labeller_kept (GPU, descriptive) | LABEL-AUDIT-T, CONTAMINATED, MEMO-GAP-T, LABELLER-KEPT-T |
| (absence) eval.json, eval_*k.json | — | must not exist | SEAL-BREACH (§5.2) |

**Per round:** `s219_pool_t<r>_s1/pool_meta.json` {seeds, per_n, draws_n, collide_n, dup_n, keep_n, sha256, builder_commit, labeller, labeller_sha256, card, dtype, label_bs, excl {set: hits}, precondition_stream_sha, T_max_by_n, L0_max_by_n}; `s219_gate_t<r>_s1/eval_gate.json` {by_n {window, labeller, items, n_kept, keep, pi, pi_E, pi_ci95, pi_E_ci95, records}, pooled_by_window, GATE_T1 / GATE_T2 / GATE_T1p, fallback, labeller_full {n: keep, pi}, kept_vs_drawn, equiv {items, agree_keep, agree_values, agree_fail_step, disagreements, min_margin}, card}.
**Base:** `s219_base_s1/eval_text.json` (required before any T1 reading); `eval_text_2070.json` (only if CARD-GATE is run).
**Latent side (read only, fixed):** s213_d3t_s1 / s215_ftself_s1 eval_vb.json; s216_r2_s1 eval_deep.json; s215_ftself_s1/eval_selfdepth.json (items_n2 at n 9–16) and s216_r2_s1/eval_deep.json items (DEPTH-PAIRED); s215_ftself_s1 / s213_d3t_s1 / s216_r2_s1 eval_chainlabel_fresh.json (LABELLER-REACH).

### 4.2 Per-cell readings (comparator-free)

| reading | exact definition | role | base (bf16; fp32 replaces) |
|---|---|---|---|
| **T-ABSORB** (T1) | min_{n 9..12} a_T(n) ≥ 0.95 | in support; enabler; text only, never a C-criterion | 0.123 (n 12) |
| **T-DBS** (T1) | M_T = mean_{n 13..16} a_T(n) ≥ 0.80 | text depth past the self-labelled range (descriptive of text) | 0.078 |
| **T-ABSORB-2** (T2) | min_{n 13..16} a_T(n) ≥ 0.95 | in support | — |
| **T-RETAIN** (T2) | min_{n 9..12} a_T(n) ≥ 0.99 | must not break | — |
| **T-DBS-2** (T2) | M2_T = mean_{n 17..20} a_T(n) ≥ 0.80 | past support | — |
| M_far_T (T2) | mean_{n 21..24} a_T(n) (n 24 overflow = failure; CTX-FEASIBLE beside) | enters T-FULL | — |
| **n50 / n90** | first downward crossing of 0.5 (0.9) by a_T(n) over n 1..24, linear interpolation; never crossing → "> 24"; below at n 1 → "< 1" | **the comparison input** | 10.44 / 9.33 |
| WIDTH_T | n50 − n90 (unreadable if either is a sentinel) | descriptive | 1.11 |
| n50 / support | n50 / B_r | descriptive (latent 1.29 / 1.29 / 1.28) | 1.30 |
| **T-SELF** (T1) | T-ABSORB ∧ ¬censored | text round-1 claim (in support) | — |
| **T-SELF-2** (T2) | T-ABSORB-2 ∧ T-RETAIN ∧ ¬censored | text round-2 claim | — |
| **Round-2 condition** | T-ABSORB ∧ ¬FT-DIVERGED-T ∧ ¬LABEL-NOISY-T ∧ ¬CONTAMINATED ∧ ¬dead ∧ ARGS-DIFF ok ∧ ¬SEAL-BREACH (T1); read from eval_text + eval_textself's CPU parts (V4) | T2 runs iff it holds | — |

**Text mechanism label** (the SELFDEPTH / SD2 bar formulas on the text lineage's own fp32 curves; only under T-ABSORB (T-ABSORB-2), else UNLABELLED): T1: **T-FULL** if M_T ≥ max(0.80, MG_T); **T-GROW** if MG_T ≤ M_T, not T-FULL; **T-SHIFT/MIXED** if SB_T ≤ M_T < MG_T; **T-SHRINK** if M_T < SB_T. T2: the same with M2_T and the round-2 bars, T-FULL also needing M_far_T ≥ 0.60. STOP-LOC is undefined for text (no pass index), so SHIFT and MIXED are merged and named **T-SHIFT/MIXED** (RT Q9).

### 4.3 Text bars by formula (registered; ceil005 = the smallest multiple of 0.005 ≥ x)

| bar | formula (a_0 = text base fp32, a_1 = T1) | s1 instance on the bf16 base [DERIVED pc_oc] |
|---|---|---|
| R1 MARGIN-GROWS MG_T | ceil005(mean_{n 9..12} a_0(n) + 0.15) | ceil005(0.515 + 0.15) = **0.665** |
| R1 SHRINK SB_T | ceil005(mean_{n 9..12} a_0(n) − 0.08) | **0.435** |
| R2 M2_shift,T | mean_{n 13..16} a_1(n) | from T1 |
| R2 M2_prop,T | mean_{d 1..4} a_1(12 + 0.75 d), linear in n | from T1 |
| R2 MG2_T | ceil005((M2_shift,T + M2_prop,T) / 2) | from T1 |
| R2 SB2_T | ceil005((mean_{n 9..12} a_0(n) + M2_shift,T) / 2) | from T1 |
| absolute | T-ABSORB 0.95, T-RETAIN 0.99, T-DBS / T-DBS-2 0.80, M_far 0.60 | — |

The latent R1 / R2 bars came from the same formulas: fed the latent curves they reproduce **0.63 / 0.40 / 0.785 / 0.60** (asserted; PROOFS PASS). Degenerate bars (SB ≥ MG) → UNLABELLED (bars degenerate).

### 4.4 Comparison readings (pre-stated; read only after the round's eval_text.json exists; the latent values are fixed committed numbers)

- **Inputs:** n50_L(0, 1, 2) = **10.31 / 15.46 / 20.54** (D3-T s1 eval_vb, FT-SELF eval_vb, R2-s1 eval_deep; all RTX 3070 fp32; THEORY §216 s1 result). n50_T(r) from eval_text. **Δ_r = n50_T(r) − n50_L(r)**; Δ_0 on `s219_base_s1`.
- **Statistic (T6, BASE-PARITY):** \|Δ_0\| ≤ 0.25 → the four-way rule reads Δ_1, Δ_2 (levels); \|Δ_0\| > 0.25 → it reads Δ^g_r = Δ_r − Δ_0 (gain over each arm's own base). Recorded in the G-T19 note before any T1 output; both statistics are always reported.
- **Sentinels (m12):** n50_T = "> 24" ⇒ Δ_r := 24 − n50_L(r) + 0.01 (the TEXT-AHEAD side); "< 1" ⇒ Δ_r := 1 − n50_L(r) − 0.01 (the LATENT-AHEAD side).
- **PARITY** iff \|Δ_1\| ≤ 0.5 ∧ \|Δ_2\| ≤ 0.5. **TEXT-AHEAD** iff ¬PARITY ∧ Δ_2 > 0.5 ∧ Δ_1 ≥ −0.5. **LATENT-AHEAD** iff ¬PARITY ∧ Δ_2 < −0.5 ∧ Δ_1 ≤ 0.5. **CROSSED** = every other case (reported with both Δ). Exhaustive and exclusive [PROVED by case split; fixture]. \|Δ\| exactly 0.5 is inside the band.
- **"Marginal" (T6):** TEXT-AHEAD / LATENT-AHEAD with \|Δ_2\| ≤ 1.0 is worded *"marginal (within plausible protocol bias)"*.
- **TEXT-NOT-ABSORBED (T4):** T1 uncensored ∧ ¬T-ABSORB. No PARITY-1 / TEXT-AHEAD-1 / LATENT-AHEAD-1 label; Δ_1 reported descriptively; T2 not run. It is a pre-stated outcome, not a censor, and **never** a LATENT-AHEAD reading: it cannot separate "text self-training does not absorb a +4 window in 12k steps" from a fault in new trainer code.
- **TEXT-LABEL-STALL (T9):** T1 met the round-2 condition, but GATE-T2 (or GATE-T1′) failed under T1 **and** under the base fallback. The comparison reads **ROUND-1-ONLY**: PARITY-1 iff \|Δ_1\| ≤ 0.5, else TEXT-AHEAD-1 / LATENT-AHEAD-1 by the sign of Δ_1, plus the row *"an output-format labeller limit: the text format's single-step labeller did not reach the round-2 window under the latent-calibrated gate (the latent copies per slot; text must re-emit the expression) — keep / π per failing n"*. The latent arm passed the same gate (GATE-2, 216-1: keep ≥ 0.954 at n 13–16). **Not a depth result**; never folded into LATENT-AHEAD.
- **T-ABSORB-2 failed (V5):** an uncensored T2 that fails T-ABSORB-2 is read four-way as registered, with *"(text round-2 window not absorbed: min a_T(13..16) = x)"* on the same line.
- **TEXT-CENSORED (r, reason):** a censored text cell (§5.2) fails every comparison row at and after its round (rule b): none of PARITY / TEXT-AHEAD / LATENT-AHEAD / ROUND-1-ONLY labels passes.
- **NOT RUN (GATE-T1):** GATE-T1 fails → no T1 cell, no spend; the comparison reads "not run (GATE-T1: keep / π per failing n)".
- **Descriptive comparison rows:**
  - **Round 0:** Δ_0 (fp32; bf16 +0.13).
  - **RATIO:** n50 / support per round, text vs latent.
  - **DEPTH-PAIRED:** per item at n 13–16 (T1 vs FT-SELF, items_n2 of s215_ftself_s1/eval_selfdepth.json) and n 17–20 (T2 vs R2-s1, eval_deep items at K′ n + 2): the 2 × 2 correct table and McNemar's exact p per n.
  - **TOKEN-COST** per n 13–16 (round 1) and 17–20 (round 2), both arms: sequential steps per correct answer (text Σ ntok / #correct; latent (n + 2)·1,000 / #correct) and token-positions per correct answer (text Σ(P + ntok) / #correct; latent (n + 2)·Σ T_item / #correct, T_item = len(syms) + 2).
  - **CTX-FEASIBLE** share per n and n-24 feasible accuracy; **CTX-BOUND** beside every comparison row.
  - **LABELLER-REACH** (Q13, RT m2): text REROOT keep vs the latent chain keep per n — base vs D3-T (§2.2 table), T1 (LABELLER-KEPT-T) vs FT-SELF (eval_chainlabel_fresh), T2 vs R2-s1 — worded as **the output format's single-step reach** (per-slot vs re-emitted copy), never depth.
  - **LATENT-FIXED-K′** (RT m13b): the latent n50 at a fixed K′ (acc_k[24] for r0 / r1 from eval_vb, acc_k[26] for r2 from eval_deep), beside the K′ n + 2 values: both arms' registered budgets are shape-informed.
  - **FAIL-POS:** per n, among failing items, the median first_bad_pos and first_bad_step, and **the share of first deviations at rows > T_train,r**. It is printed on the same line as any LATENT-AHEAD (T6).
- **Never:** any text reading worded as latent C1 / C2 / C4 or as any north-star criterion.

### 4.5 Descriptive readings (per text model)

| reading | definition | why |
|---|---|---|
| FORM-VALID f(n), STRICT s(n) | eval_text form_valid_by_n, strict_by_n | the strict analogue (chance ≈ 0) beside answer accuracy (T10) |
| FAIL-POS | §4.4 | splits row-keyed from step-keyed failure (§3.5) |
| LABELLER-FULL (each gate) | FULL keep / π on the gate items | what Lee's protocol would have kept |
| KEPT-VS-DRAWN (each gate) | mean P + C, height, brackets of kept vs drawn per n | selection (Literature failure mode 1) |
| LABELLER-KEPT-T | REROOT keep / π at n 9–24, fresh items | the next labeller's reach |
| MEMO-GAP-T | §2.7 | memorisation (flag > 0.02) |
| BF16-VS-FP32 (base) | committed bf16 acc_by_n vs the fp32 re-eval at n 1–16 | dtype sensitivity |
| CARD-GATE (if run) | 3070 vs 2070 base eval_text, per n and n50, item flips | card sensitivity |
| self_value_ce / self_copy_ce | log.jsonl | the arithmetic vs copy parts of the text loss (D1) |
| EQUIV (each gate) | TF-GREEDY vs REROOT generation | labeller exactness |

### 4.6 Scorer theory/score_s219.py (to write; fixture-checked before any output, G-T12)

- **Lineage table:** s1 {base_model s170_cot_s1, base_eval s219_base_s1, t1 s219_t1_s1, t2 s219_t2_s1, pool_t1 s219_pool_t1_s1 (pool_t1.jsonl), pool_t2 s219_pool_t2_s1 (pool_t2.jsonl), gate_t1 s219_gate_t1_s1, gate_t2 s219_gate_t2_s1, latent {r0 s213_d3t_s1, r1 s215_ftself_s1, r2 s216_r2_s1}}. One lineage only.
- **ARGS-DIFF** (T1; each FT cell vs `s170_cot_s1/args.json`):
  - **META_T** = score_s213.META ∪ {`excl_survive_by_n`, `hostprobe_sps`}. (META already holds out, device, train_gpu, train_driver, the env keys, eval_bs, micro, auto_micro, mem_budget, ckpt_every, keep_ckpt_steps, time_steps, mem_check, stop_at, resume.)
  - **TEXT_NEUTRAL_NEW** (each nested170.py argparse key absent from the base args.json, at its parser default; a cell value ≠ the listed value → ARGS_BAD) [DERIVED v1_argkeys, AST of nested170.py sha256 recorded]:
    `n_slots 8, anneal "", probe_loss_every 0, mem_layer 6, cache_lower false, anneal_gate "", csr "", iit "", iit_tau 1, iit_nmax 0, grokfast "", gf_alpha 0.98, gf_lamb 2.0, cot_eval_bs 64, eval_only false, aux_step 0.0, aux_fade "36000,48000", kmatch_impl "masked", add_norm "", aux_nopass false, aux_reweight false, aux_hold false, kprimes "", path_indep false, excl_min_n 1, scratch "", warmup 100, k_window "", fast_add false` (29 keys; the 30th, keep_ckpt_steps, is META). Each is inert on the textcot path at that value: a feature that is off at that value or used only by the memory arms (n_slots, mem_layer, cache_lower, csr, iit*, aux_*, kmatch_impl, add_norm, k_window, kprimes, path_indep, anneal*, probe_loss_every 0, grokfast "" with gf_* unused), deltamem only (fast_add), the pretrained trunk (scratch ""), the §170 exclusion protocol (excl_min_n 1), lr_at's unchanged warm-up branch (warmup 100, l.1039), no eval in the trainer (cot_eval_bs, eval_only) [VERIFIED code for warmup / excl_min_n / scratch; the rest by G-T7 PLAIN-EQUIV bitwise].
  - **Allowed FT keys with required values:** steps 12000; lr 3e-05; init_from = the lineage's previous model.pt (T1 boxes/runs_fsq/s170_cot_s1/model.pt, T2 boxes/runs_fsq/s219_t1_s1/model.pt); self_pool = the round's pool file; p_self 0.5; self_micro (T1 2, T2 4); **no_final_eval true**; stop_at 0 (the registered run); draws_log_steps 12000; mem_check_self false; mem_check_steps (any); **args_ref**: may differ iff the file it names equals `s170_cot_s1/args.json` on every non-META_T key (the 216-7 rule, score_s216.args_ref_check verbatim); out.
  - **Every other key**: shared keys equal to the base's (arm textcot, think_k 0, bs 64, seed 1, n_max_train 8, ckpt_passes false, mem_write / write_mix_start / write_mix_end null, log_every 100, per_n 1000, serial_pool 2200, eval_at 0, sched_scale 1.0, fp32 false, task "nested", format string); **a non-META base key missing from the cell → ARGS_BAD** (V2); a new key neither in TEXT_NEUTRAL_NEW nor allowed → ARGS_BAD.
  - **G-T5:** the rule is run on the **real** args.json of the CPU smoke (G-T4) and of each host smoke before launch; a mismatch fixes the trainer, not the rule. A rule defect found there may be amended only by a Theory note with Audit's ruling **before any s219 cell output exists** (the 216-6 / 216-7 pattern).
- **Pending rule** (215-7 / SD2 M5): a cell reads **"pending (missing [...])"** while any registered file is missing (log, args, draws_log, eval_text, eval_textself's CPU parts, the round's pool_meta and eval_gate, and `s219_base_s1/eval_text.json`), and reads nothing else first. Pending never passes. **`--final s1`** (run once Theory records the chain as done) turns a still-missing registered file into a failure (rule b). Descriptive files and keys (MEMO-GAP-T, LABELLER-KEPT-T, CARD-GATE, LABELLER-FULL) never gate.
- **Functions:** text_read (items_sha256 recomputed; meta asserts: card contains "3070" (or the CARD-GATE / fallback card recorded in the G-T19 note), fp32, tf32 false, cot_eval_bs 32, n 1–24, model_sha256 (base: 36f4659b…3392); acc_by_n == mean(answer == y) per n; overflow items exactly the C_struct rule's; one card per file); n_cross (n50 / n90 over n 1..24, sentinels); bars_text (§4.3; the latent instance must reproduce 0.63 / 0.40 / 0.785 / 0.60); mech_text; round2_condition; gate_t (recomputed item-weighted from by_n; must equal the file's flags; fallback); label_audit_t; contamination_t; seal_check; base_parity; compare (levels / gain, four-way, marginal, sentinels, TEXT-NOT-ABSORBED, TEXT-LABEL-STALL + ROUND-1-ONLY, the T-ABSORB-2 qualifier, TEXT-CENSORED, NOT RUN); depth_paired, token_cost, ctx_feasible, labeller_reach, latent_fixed_k, fail_pos.
- **Fixtures (G-T12)**, one per panel and rule: the four §3.5 panels (N-SHIFT → LATENT-AHEAD, N-PROP → PARITY, POS-PROP → PARITY, POS-ADD → LATENT-AHEAD); BETTER → TEXT-AHEAD; a crossed curve → CROSSED; \|Δ\| exactly 0.5 → inside; Δ_2 −0.8 → "LATENT-AHEAD (marginal)"; n50_T "> 24" → TEXT-AHEAD side; BASE-PARITY: Δ_0 0.30 → the gain statistic (a levels-PARITY curve with offset 0.3 reads PARITY on gain); ¬T-ABSORB → TEXT-NOT-ABSORBED (no round-1 four-way label); GATE-T2 fail under both labellers → TEXT-LABEL-STALL + PARITY-1; T2 ¬T-ABSORB-2 → four-way + qualifier; T1 dead → TEXT-CENSORED, every comparison row fails; LABEL-NOISY-T; CONTAMINATED; SEAL-BREACH (an eval.json in the T1 dir); ARGS-DIFF: lr 1e-4 fails, a TEXT_NEUTRAL_NEW key at a non-default fails, `no_final_eval` false fails, a dropped base key fails, args_ref to a file with one changed key fails, **the real G-T4 smoke args.json passes**; overflow items as failures; items_sha256 mismatch → void; missing eval_text → pending, then failure with --final; the latent bars reproduced; degenerate bars → UNLABELLED; T-FULL needing M_far ≥ 0.60.

---

## 5. Claims, wording, censoring, Goodhart

### 5.1 Claims and pre-stated wording

- **T-SELF** (T1) = T-ABSORB ∧ ¬censored. Only this wording: *"text-CoT depth extended from 8 to 12 by fine-tuning a §170 TEXT-COT GPT-2 on its own re-rooted greedy single-step chain (form-filtered, values never checked; labels gold-equivalent at π = [GATE-T1 pooled π]; no gold labels beyond n 8; in support; not attributed: no text FT-CTRL)."*
- **T-SELF-2** (T2): the same sentence, *"from 12 to 16 … the round-1 text model's own chain …"*, with GATE-T2's π; if a window is BASE-LABELLED, the sentence names the labeller per window (as SD2's m11).
- **T-DBS / T-DBS-2:** *"text-CoT accuracy past the self-labelled range (n 13–16 / 17–20, the same untouched test items as the latent arm): mean [M]"*, pass or fail stated against 0.80. A text-arm description, **never C2**.
- **Comparison wording** (exactly one applies; the statistic, levels or gain, is named):
  - **PARITY:** *"on Arithmetic(n, 11), round-matched text-CoT self-training and latent self-training moved the 50 % depth by the same amount (\|Δn50\| ≤ 0.5 at rounds 1 and 2: Δ = [Δ_1] / [Δ_2])."*
  - **LATENT-AHEAD:** *"latent self-training moved the 50 % depth further than round-matched text-CoT self-training (Δn50 = [Δ_1] / [Δ_2])"*, with *"marginal (within plausible protocol bias)"* if \|Δ_2\| ≤ 1.0, and on the same line the **FAIL-POS** row: *"[share] of the text arm's first deviations at n 17–20 sit at rows past its trained [T_train] (GPT-2's learned absolute positions)"*.
  - **TEXT-AHEAD:** the mirror of LATENT-AHEAD (with "marginal" if \|Δ_2\| ≤ 1.0).
  - **CROSSED:** *"the arms' order changed between rounds (Δ = [Δ_1] / [Δ_2])."*
  - **TEXT-NOT-ABSORBED:** *"text-CoT self-training did not absorb the round-1 window under the matched recipe (min a_T(9..12) = [x] < 0.95); no comparison is read (Δ_1 = [x], descriptive). This cannot be separated from a fault in the new trainer path."*
  - **TEXT-LABEL-STALL:** *"ROUND-1-ONLY: [PARITY-1 / TEXT-AHEAD-1 / LATENT-AHEAD-1] (Δ_1 = [x]). The text format's single-step labeller did not reach the round-2 window under the latent-calibrated gate (keep / π per failing n): an output-format labeller limit (the latent copies per slot; text must re-emit the expression), not a depth result."*
- **On the same line, always:** *"one lineage each (latent = the surviving s1 lineage; s2 censored at R1, 216-8); GPT-2 small with learned absolute positions (1,024 rows); Arithmetic(n, 11); two rounds (CTX-BOUND: GPT-2's 1,024 rows)"*; the latent cost (MEM-TRADED-2, MEM-ERODING: STAB_self 0.782 → 0.489 → 0.437); the text cost (TOKEN-COST at n 17–20: sequential steps and token-positions per correct answer, both arms); Δ_0 and the statistic used; the T-ABSORB-2 qualifier when it applies.
- **Never:** text readings worded as latent C1 / C2 / C4 or as any north-star criterion; PARITY worded as "latent reasoning is as good as text CoT" in general (one task, one model size, two rounds, one lineage each); LATENT-AHEAD worded as "latent beats CoT" without the qualifiers (the text arm was form-filtered exactly as the latent arm, it reads its own operands (D1), the latent arm traded memory, GPT-2's positions); TEXT-LABEL-STALL or TEXT-NOT-ABSORBED worded as LATENT-AHEAD.
- **For the north star:** a comparison row is context for C2 (the latent arm's depth extrapolation against the natural text baseline under identical self-training). It is not itself a C-criterion.

### 5.2 Censoring (rule b; each fails every reading of that cell and every comparison row at and after its round)

- **dead:** log last step < 12,000 (a guard stop included).
- **FT-DIVERGED-T:** a_T(n) < 0.99 at any n 1–8 (the base reads 1.000 at every n 1–8).
- **LABEL-NOISY-T:** LABEL-AUDIT-T π < 0.99 or π^E < 0.99, pooled or per window, or a window's Wilson CI (π or π^E) not overlapping its gate's.
- **CONTAMINATED:** §2.4 item 8.
- **ARGS-DIFF:** a mismatch per §4.6.
- **SEAL-BREACH (T3):** an eval.json or eval_<k>k.json (any trainer-written generation) in a cell directory, or a gate / labelling output of that model at n ≥ 13 produced before its eval_text.json (stage-script order, G-T20). Such a file is renamed `*.void_unsealed`, kept and disclosed (keep-aborted-logs rule), and the model's comparison rows fail.
- **eval_text:** missing at `--final`, or items_sha256 / meta / card mismatch → every reading of that model fails.
- **Values, not missing:** CTX-OVERFLOW items and non-terminating generations are failures in a_T(n); never-crossed n50 / n90 read as sentinels; chance-level answers count as given.
- **Pending never passes** (215-7).
- **Gates:** a failed GATE-T1 → NOT RUN (no cell, no spend). For T2, both labellers failing a window → no T2, the comparison reads TEXT-LABEL-STALL.
- **A trainer fault found after a reading** (e.g. a G-T7 / G-T8 re-check fails on the landed tree) voids the cell (not TEXT-NOT-ABSORBED); a fixed re-run needs a Theory note and Audit before any output of it, and is disclosed.
- **Not in the text arm** (stated so nobody "fixes" it): LENGTH-FREE, VB-C1, K′24-STABLE, KEEPS-SELF-STAB and THINK-NECESSARY are latent-loop readings with no text counterpart; the text arm's only kept reading is FT-DIVERGED-T.

### 5.3 Goodhart analysis

| target / route | finding |
|---|---|
| **Stop at the support** (fail past n > B_r or past row T_train,r) | Zero loss on the support (S0-T a), irreducible by loss as in the latent arm. It is what the panels measure (N-SHIFT / POS-ADD → LATENT-AHEAD), not something the design prevents |
| **The filter rewarding easy items** | REROOT keep 0.985–1.000 at n 9–12 (§2.2); its few failures are first steps (on longer-than-average inputs from n 12 on; at n 11 they are not longer), so round-1 selection is ≤ 1.5 % and not uniformly toward shorter inputs. KEPT-VS-DRAWN at every gate. T-ABSORB's denominator is **all** test items, so selection cannot inflate it. FULL (not used) keeps 0.06 / 0.00 at n 11 / 12 with π 0.80 at n 11 |
| **A value check sneaking into the filter** (D1) | VALUE-SLOT accepts any value token; the builder's N.apply raises (G-T0). An accidental value check would show as π = 1.000000 with zero kept-wrong at every n **and** keep < the strict-correct share; the gate script reports both |
| **Copying a systematic arithmetic error** (Lee App B.4) | LABEL-AUDIT-T prints each kept-wrong (op, a, b → v); class mass ≤ 2(1 − π) |
| **Memorising pool items** | fresh-once; MEMO-GAP-T (descriptive, as the latent's MEMO-GAP-2) |
| **The eval budget** | gold-free (C_struct + 50, context-capped); a model that never emits EOT fails; no oracle stopping, no per-item budget. Both arms' budgets are shape-informed (K′ n + 2 vs C_struct + 50); LATENT-FIXED-K′ beside |
| **The free-generation answer metric** | it credits a non-canonical chain that reaches the right number and lucky guesses (floor ≈ 0.04–0.10), as latent a(n) credits chance answers. STRICT s(n) (chance ≈ 0) beside; n50 moves < 0.1 n under a floor change of 0.05 at the measured slopes [DERIVED, draft] |
| **Overflow** | lowers a_T(24) by ≤ 0.029 by construction; enters only M_far_T and sentinel n50s; CTX-FEASIBLE beside |
| **Cross-arm bars** | no training objective reads n50 or Δ; the latent n50s are fixed committed numbers |
| **Card / dtype** | both arms read on the 3070 at fp32 (T5); the 2070 only behind CARD-GATE; a model's registered eval_text on one card |
| **The BASE-PARITY switch** | decided on the base re-eval before any T1 output (G-T19), so the statistic cannot be chosen on an FT outcome; both statistics always reported |
| **The seal** | the trainer has no generation path under `--no_final_eval`; G-T20; SEAL-BREACH censors |

---

## 6. OC (one lineage per arm; realised spend at registration $0) [arithmetic DERIVED v1_checks/v1_oc.py → v1_oc.json, exact quadrature / binomial sums; every prior ASSUMED]

**Structural factors** [ASSUMED]: T1 uncensored 0.95 (engineering + FT-DIVERGED-T); GATE-T1 ∧ LABEL-AUDIT-T1 0.99; T-ABSORB 0.90 (the base learned n ≤ 8 by step ≈ 600; REROOT labels at keep ≈ 0.99, π ≈ 0.9994); **GATE-T2 (either labeller) \| T-ABSORB 0.93** (RT m2: only near-zero copy-reach transfer stalls); T2 uncensored 0.95; LABEL-AUDIT-T2 0.99; T-ABSORB-2 0.85 (rows to 531, ⅔ dose); T-RETAIN 0.95. Panel priors §3.5. Lineage noise on M / M2: sd 0.06–0.12; on Δ: §3.7 (ŝ 0.1666 central, 0.20, 0.35).

**Outcome tree (unconditional):**

| outcome | P |
|---|---|
| both rounds readable (four-way) | **0.740** (of which T-ABSORB-2 failed, qualifier on the line: 0.111) |
| TEXT-LABEL-STALL (ROUND-1-ONLY) | **0.059** |
| TEXT-NOT-ABSORBED | **0.094** |
| TEXT-CENSORED / not run (T1 0.060, T2 0.047) | **0.106** |

**Per-cell readings:**

| reading | P(pass \| no effect) | P(pass \| effect) | headline |
|---|---|---|---|
| T-ABSORB | ≈ 0 (base min a(9..12) 0.123) | 0.90 [ASSUMED] | **0.85** (with censoring) |
| T-DBS (M_T ≥ 0.80) | ≈ 0 (base M 0.078); ≤ 0.009 under N-SHIFT / POS-ADD (sd 0.12) | 0.17–0.33 \| proportional panels (M 0.74–0.75); 0.80–0.95 \| BETTER | **0.15** (sd 0.09) |
| T-SELF-2 | ≈ 0 | — | **0.60** |
| T-DBS-2 (M2_T ≥ 0.80) | ≤ 0.012 under the additive panels (M2 0.52–0.53) | 0.73–0.93 \| proportional (M2 0.87–0.89); 0.89–0.99 \| BETTER | **0.23** (with T-SELF-2, sd 0.09) |
| GATE-T1 (π terms, kept ≈ 7,935) | w 0.007: 0.012; w 0.005: 0.499 | w 0.003: 0.9985; measured w ≈ 6e-4: ≈ 1.000 | ≈ 1 |
| GATE-T2 keep_16 ≥ 0.70 | zero transfer (T1 = base, keep 0.645): ≈ 0 (0.66: 7.6e-5) | additive 0.978 / ratio 0.984 / half 0.903: 1.000 | 0.93 [ASSUMED] |
| LABEL-AUDIT-T (5,000; point term, wrong ≤ 50) | π 0.985: 0.0013; π 0.99: 0.54 | π 0.992: 0.948; π 0.9994: 1.000 | ≈ 1 |

**Comparison, levels, by case** (four-way P; "marginal" = TEXT-AHEAD / LATENT-AHEAD with \|Δ_2\| ≤ 1.0):

| truth (Δ_1 / Δ_2) | s | PARITY | TEXT-AHEAD (marginal) | LATENT-AHEAD (marginal) | CROSSED |
|---|---|---|---|---|---|
| **no difference** (0 / 0) | **ŝ 0.167** / 0.20 / 0.35 | **0.964** / 0.915 / 0.608 | 0.017 (0.017) / 0.039 / 0.155 | 0.017 (0.017) / 0.039 / 0.155 | 0.002 / 0.008 / 0.082 |
| latent ahead 1 n per round (−1 / −1) | ŝ / 0.20 / 0.35 | **0.0004** / 0.002 / 0.035 | 0 | **0.983** (0.483) / 0.962 / 0.844 | 0.017 / 0.036 / 0.121 |
| text ahead 1 n per round (+1 / +1) | ŝ | 0.0004 | 0.983 (0.483) | 0 | 0.017 |
| latent ahead at round 2 only (−0.5 / −1) | ŝ | 0.016 | 0 | 0.983 (0.483) | 0.001 |
| band edge (+0.5 / +0.5) | ŝ | 0.333 | 0.500 (0.483) | 0 | 0.167 |
| proportional parity with offset (+0.20 / +0.26) | ŝ | 0.825 | 0.158 (0.157) | 0.001 | 0.016 |
| additive panels (N-SHIFT, POS-ADD), WORSE | ŝ / 0.35 | 0.000 / ≤ 0.0008 | 0 | **1.000** / ≥ 0.998 (marginal ≤ 0.021) | 0.000 / ≤ 0.0005 |
| N-PROP (+0.20 / +0.34) | ŝ / 0.35 | 0.745 / 0.504 | 0.244 (0.241) / 0.369 | 0.000 / 0.045 | 0.011 / 0.082 |
| POS-PROP (+0.10 / −0.03) | ŝ / 0.35 | 0.958 / 0.594 | 0.012 / 0.141 | 0.024 / 0.170 | 0.007 / 0.096 |
| BETTER (+1.5 / +3.0) | ŝ | 0 | 1.000 (0) | 0 | 0 |

**Protocol-bias rows** (constant b on both rounds, ŝ): no difference: b +0.15 → PARITY 0.917, TEXT-AHEAD 0.069; b +0.30 → 0.744 / 0.198 (mirror for b < 0). N-PROP: b +0.15 → PARITY 0.483 / TEXT-AHEAD 0.477; b +0.30 → 0.212 / 0.719. POS-PROP, b −0.30 → LATENT-AHEAD 0.239. −1 / −1, b +0.30 → LATENT-AHEAD 0.802. This is why T5 / T3 / BASE-PARITY and the "marginal" wording exist.

**Gain branch** (only if \|Δ_0\| > 0.25; §3.7): P(PARITY \| no difference) **0.900 / 0.981** (ρ_0 0 / 0.5) at ŝ, 0.473 / 0.645 at s 0.35; P(LATENT-AHEAD \| −1 / −1) 0.959 / 0.991 at ŝ, 0.795 / 0.872 at 0.35. Prior-weighted headline at ŝ: PARITY 0.29–0.33, LATENT-AHEAD 0.56–0.58, TEXT-AHEAD 0.11–0.13.

**ROUND-1-ONLY labels** (TEXT-LABEL-STALL branch, at ŝ): P(PARITY-1 \| no difference) **0.997**; P(LATENT-AHEAD-1 \| −1 at round 1) 0.999; P(LATENT-AHEAD-1 \| N-SHIFT) 0.999; prior-weighted PARITY-1 0.34 / TEXT-AHEAD-1 0.11 / LATENT-AHEAD-1 0.55.

**Named outcomes (m11):**

| outcome | P(reads \| no effect) | P(reads \| effect) | headline |
|---|---|---|---|
| **TEXT-NOT-ABSORBED** | **≈ 0.94** (= P(T1 uncensored): text FT does nothing → min a(9..12) ≈ 0.12, by construction) | 0.10 of uncensored T1 under a working trainer [ASSUMED] | 0.094 |
| **TEXT-LABEL-STALL** | ≈ 1 under zero copy-reach transfer (T1 ≡ base: keep_16 0.645 < 0.70) | ≈ 0 under additive / ratio / half transfer | 0.059 |
| PARITY | 0.964 \| no difference (by design; 0.608 at s 0.35) | 0.0004 \| 1-n difference (0.035 at 0.35); ≤ 0.0008 \| additive panels | 0.19 |
| LATENT-AHEAD | 0.017 \| no difference (0.155 at 0.35); 0.198 \| no difference with bias −0.30 | 0.983 \| −1 / −1; ≥ 0.998 \| additive panels | 0.45 |
| TEXT-AHEAD | 0.017 \| no difference (0.155 at 0.35); 0.198 with bias +0.30 | 0.983 \| +1 / +1; 1.000 \| BETTER | 0.09 |
| CROSSED | 0.002 \| no difference (0.082 at 0.35) | — | 0.01 |

**Headline** (prior-weighted over the panels; ¬T-ABSORB-2 cases inside the four-way modelled at Δ_2 = −3 [ASSUMED], so P(LATENT-AHEAD \| ¬T-ABSORB-2) 0.89):

| | s | PARITY | TEXT-AHEAD | LATENT-AHEAD | CROSSED |
|---|---|---|---|---|---|
| given both rounds absorbed (panels only) | ŝ / 0.35 | 0.303 / 0.195 | 0.139 / 0.184 | 0.555 / 0.590 | 0.003 / 0.032 |
| **given both rounds readable (0.740)** | **ŝ** / 0.35 | **0.258** / 0.165 | **0.118** / 0.156 | **0.606** / 0.629 | **0.019** / 0.050 |
| **unconditional** | **ŝ** / 0.35 | **0.191** / 0.122 | **0.087** / 0.116 | **0.448** / 0.465 | **0.014** / 0.037 |

plus, unconditional: TEXT-LABEL-STALL 0.059 (PARITY-1 0.020, TEXT-AHEAD-1 0.006, LATENT-AHEAD-1 0.033), TEXT-NOT-ABSORBED 0.094, TEXT-CENSORED / not run 0.106.

**CARD-GATE OC** (the 2070 backup; per-n net flips ≤ 5 of 1,000 at the 16 frontier n and \|Δn50\| ≤ 0.02): P(pass \| no systematic card effect) **0.9997 / 0.992 / 0.779 / 0.259** at a per-item correctness-flip rate q 0.001 / 0.002 / 0.005 / 0.01; P(the affected n passes \| a systematic +0.01 at one n) 1e-4 (q 0.001) / 0.022 (q 0.005). Flips need top-2 margins ≲ 1e-5 (PROOFS: no REROOT margin < 1e-3 at n ≤ 20), so q ≤ 0.002 is expected [ASSUMED]; a failure only keeps the work on the 3070.

**How to read it.**
- **PARITY is a strong reading at the measured spread:** P(PARITY \| true parity) is **0.96 at ŝ 0.17 (3 df)** and 0.61 at the conservative s 0.35; the text-FT lineage noise is unmeasured. A PARITY pass is strong evidence against a 1-n difference (P ≤ 0.0004 at ŝ, 0.035 at 0.35).
- **LATENT-AHEAD is the most likely outcome under the stated priors (0.61 given readable)** because the additive families (prior 0.45) predict ≈ −1 / −2 and are detected with P ≥ 0.998, and because a T2 that does not absorb its window reads LATENT-AHEAD (with the qualifier). The headline is prior-driven; the discrimination (additive vs proportional) is the robust part.
- **The informative split** is additive vs proportional text extrapolation. LATENT-AHEAD says the latent's proportional margin (×1.29 per round) is something text self-training on GPT-2 does not reproduce (with the GPT-2 position confound named via FAIL-POS); PARITY says the proportional law is not specific to the latent loop.
- **Rule check (OC block + censoring rule):** every registered reading above carries P(pass \| no effect) and P(pass \| effect) at its seed count (one lineage per arm). Never-reached events (overflow, non-termination, never-crossed n50, a never-run T2) count as failures, sentinels or named outcomes (§5.2, §4.4). Realised $ at registration: $0.

---

## 7. Gates (all green before any spend on the stage they guard; Audit reviews)

| gate | check |
|---|---|
| **G-T0 NO-GOLD** | Builder gold patches (N.apply, N.value, N.eval_tokens, N.cot_steps, N.cot_ids raise) before generation. Record key set {syms, n, chain_label, values} asserted at load; no root key in a self item. Grep: `chain_label` / `values` only in the writer and the loader's SD-LABEL-MAP-T line. A self-batch smoke counts cot_ids / cot_steps calls on self items = 0. No `s219_gate` string in boxes/textself_ft.py |
| G-T1 EXCLUSION | pool ∩ §2.4 item 2 = ∅ by exact syms; per-n draws / collisions / duplicates logged; the precondition streams regenerated and their per-n counts asserted (200 per n at n 9–20, each stream); n(syms) == generator n asserted |
| **G-T2 COMPOSE-T** | compose_T(syms, values) == the TF-GREEDY accepted token sequence, bitwise, on every gate-kept item; == N.cot_ids(gold root) on every kept item whose values equal gold and ≠ on every kept-wrong item (separate gold process). The decomposed CV-T flags agree with the candidate-equality test on 100k synthetic perturbed steps (drop, duplicate, swap, wrong span, wrong bracket, non-value token) |
| **G-T3 STREAMS** | T1 coins == s215_ftself_s1/draws_log.json coins and T2 coins == s216_r2_s1/draws_log.json coins at all 12,000 steps (5,992 self steps each); plain_hash at steps 1..200 == a separate-process replay of the base stream (rng seed 1, train_items); the realised per-n self counts == the values precomputed from the pool order |
| G-T4 INIT + RESUME | CPU fp32 smoke: `--init_from <base> --stop_at 2 --ckpt_every 1 --no_final_eval`: model at step 0 == the init model.pt bitwise; INIT_FROM prints missing [] / unexpected [] and the sha256; ckpt_last at steps 1 and 2; a resume runs step 3 with cursor / rng / rng_s / AdamW restored and its step-3 loss equals an uninterrupted run's; **no eval*.json written** |
| **G-T5 ARGS-DIFF** | score_s219.args_diff on the **real** args.json of G-T4 and of each host's G-T9 / G-T10 smoke reads ok; a copy with one changed key (lr), a TEXT_NEUTRAL_NEW key off its default, `no_final_eval` false, a dropped base key, and an args_ref to a one-key-changed file each read ARGS_BAD (T1, V2). Printed in each launch log |
| G-T6 LOAD | each new model.pt loads strictly into GPT2KV and runs **one forward on a synthetic input** (never `--eval_only`, never a test item) |
| **G-T7 PLAIN-EQUIV** (V1) | `boxes/nested170.py` sha256 equals the registration HEAD's (unchanged). `textself_ft --p_self 0` (no init_from, CPU, small bs, `--stop_at 3 --log_every 1 --no_final_eval`) vs `nested170 --arm textcot --time_steps 3 --log_every 1` with the same args: per-step losses bitwise equal at steps 1–3, args.json equal on every non-META key except the FT keys at their off values |
| G-T8 GRADFLOW | scratch/gradflow_check.py on a self batch: nonzero grad on every trunk parameter tensor; decorators by ast (no no_grad on the new paths) |
| **G-T9 MEM** | on each FT host: `--mem_check_self --mem_check_steps 2` on the 64 longest pool records of the deepest n (T1 n 12, T2 n 16) at bs 64 and the **fixed** self micro (T1 2, T2 4); peak after step 2 incl. AdamW ≤ 21.0 GiB (projected 14.7 / 12.8). Over → the host is refused (counts as a refusal); the micro never changes |
| G-T10 TIME | forced G-TIME (odd steps deep self, even plain) → the cell's training projection → §8 admission at the host's price |
| G-T11 OFF-HOST | full-state checkpoint mirror every 2k (sha-checked after the first copy); the resume is G-T4's; every local stage's outputs (pool + pool_meta, eval_gate, eval_text, eval_textself) committed or copied off the working card at the stage boundary |
| G-T12 FIXTURES | score_s219 fixtures (§4.6) pass, including the real G-T4 args.json, before any cell output is scored |
| **G-T13 DIFF REVIEW** (Audit) | T1 (parser copy, init_from, loader, coin replay incl. the 64 discarded draws, compose_T, per-type micro, logging, ckpt / resume, **no generation path under `--no_final_eval`**); T2 (TF-GREEDY: candidate from shape only, masks / positions / padding, REROOT / FULL loops, no simulator call; the independent generation used by G-T16); B1 / B2; E1 (gold-free budget, CTX-OVERFLOW, the row-1,024 break, per-item store, items_sha256, card / dtype meta); E2 |
| G-T14 LAUNCH | RTX 3090 only for training (card in every file); drain waiters armed at launch; no edit of boxes/textself*.py / text_eval.py while a deploy is in flight or a local job imports them |
| **G-T15 GATES** | GATE-T1 (T1), GATE-T2 + GATE-T1′ (T2) per §2.2 on the 3070 before the pool is used; a pool stage starts only if eval_gate.json reads PASS for its window (mechanical check in the stage script; fallback labeller per window) |
| **G-T16 EQUIV** (T8) | on the 3070, on all 250-per-n gate items (kept and rejected): **TF-GREEDY REROOT vs per-step REROOT KV-cache generation** (fresh prompt [EOT] + e_{k−1} + [' =']; greedy decode to the first ' =' / EOT; validity by CV-T): keep flag, every value and the failing step identical on ≥ 0.995 of items per n; every disagreement listed with its top-2 logit margin. FULL vs FULL generation for LABELLER-FULL (descriptive). Fail → that window's labeller is REROOT KV-cache generation at bs ≥ 256 (≈ 2–4 h per pool); Theory notes the cost before the pool runs |
| **G-T17 SMOKE-AT-SCALE** | locally before deploy: the builder's exclusion set at full size (30,000 test + SERIAL + 213-37 + the latent gates + the on-disk 440k s2 pool + any regenerated pool + the 4,800 precondition items + rng-23195 smoke items) and the trainer's pool load on a full-size 388k / 390k synthetic pool; timings logged (b310's O(N²) lesson) |
| **G-T18 EVAL-SMOKE** | on the 3070, the base model on **fresh** n-24 shapes (rng 23195; never test items): the 32 longest feasible items and a cap-bound batch; peak ≤ 6.0 GiB; the longest batch timed → the eval projection; a CPU stub-model test (never emits EOT) asserts the loop stops before row 1,024 and returns −1; output to scratch/_tmp, never into a cell directory |
| **G-T19 BASE FIRST** | `s219_base_s1/eval_text.json` (fp32, 3070, model_sha256 asserted) exists, and a Theory note records the §4.3 bars, Δ_0 and the BASE-PARITY branch, before any T1 output is read |
| **G-T20 SEAL** (T3) | AST / grep: no call to cot_answers / generate / eval code reachable in textself_ft.py when `--no_final_eval`; the G-T4 and host smokes leave no eval*.json; the drain's fetch list excludes eval files; the stage script starts GATE-T2 / GATE-T1′ only after `s219_t1_s1/eval_text.json` and the round-2 condition note exist |
| **G-T21 CARD-GATE** (T5; only if the 2070 is to be used) | base eval_text on the 2070 (`eval_text_2070.json`): \|a_2070(n) − a_3070(n)\| ≤ 0.005 at every n 1–24 and \|Δn50\| ≤ 0.02, item flips listed; pass → the 2070 may take whole files (§2.7); fail → the 3070 only |

---

## 8. Cost, caps, admission, guards, wall-clock [DERIVED v1_checks/v1_cost.py → v1_cost.json; price band $0.17–0.26/h and throughputs ASSUMED]

**Plan B′ (T12):** cloud RTX 3090 for T1 and T2 training only (two hosts in sequence, as §215 / §216); every pool, gate, EQUIV and eval on the local RTX 3070 at fp32.

| item | where | hours | projected $ | admission (G-TIME projection at the host's price) | in-flight guard (realised, incl. refused / BAD hosts) |
|---|---|---|---|---|---|
| G-T18 + base eval_text (`s219_base_s1`) | 3070 | 1.8–3.1 | 0 | — | — |
| GATE-T1 (8k) + LABELLER-FULL + EQUIV; T1 pool (388k, padded TF) | 3070 | 1.1–1.5 | 0 | — | — |
| **`s219_t1_s1`** (12k; self 0.59 s, plain 0.27 s per step; ×1–2 host variance; + 0.25–0.35 h deploy / HOSTPROBE / G-T9 / G-T10) | 3090 | 1.7–3.2 | **0.29–0.84** | ≤ **$1.00** | **$1.20** |
| T1 eval_text; eval_textself CPU parts (minutes) and GPU parts (MEMO-GAP-T 16k gens, LABELLER-KEPT-T) | 3070 | 2.1–3.6 | 0 | — | — |
| GATE-T2 + GATE-T1′ (16k) + EQUIV; T2 pool (390k, padded TF) | 3070 | 1.7–2.3 | 0 | — | — |
| **`s219_t2_s1`** (12k; self 1.03 s per step; + deploy) | 3090 | 2.4–4.7 | **0.41–1.22** | ≤ **$1.45** | **$1.70** |
| T2 eval_text; eval_textself (MEMO-GAP-T 32k gens, LABELLER-KEPT-T) | 3070 | 2.9–4.8 | 0 | — | — |
| off-host mirrors (≈ 6 × 1.5 GB per cell) + uploads | | | ≤ 0.10 [ASSUMED] | | |
| **Total** | | 3070 **9.7–15.3 h**; cloud 4.1–7.9 h | **$0.70–2.15** | admissions sum $2.45 | **family cap $3.00** (GUARD_STOP; guards sum $2.90 + $0.10 egress) |

- **≤ $30:** the family cap $3.00 is under the user's $30 auto-approval rule (Audit approves). No MEM-TRADED user OK is needed (no memory channel). Plan A for reference (one 3090 per lineage for everything, PROOFS-corrected): $1.9–5.6 per lineage; Plan B′ is cheaper and puts the text evals on the latent's eval card.
- **Admission and retries** (SELFDEPTH_v1 §8 with 215-2 / 215-4): a projection above its bound means no launch on that host; at most 2 more **measured** refusals per cell; BAD hosts (never start) do not use a try but count toward the guard. The admission bounds sit ≈ 19–20 % above the projected upper ends (RT m10's headroom logic). **Steps are never reduced** (12k is the registered budget); the micro is never changed (G-T9 refuses instead). **A guard stop is a dead run** (rule b). A cell no host admits is not run; its readings read **UNREAD** (never a pass) and the comparison reads TEXT-CENSORED (not admitted).
- **Cut order (pre-stated; RT m13d):** (1) no text s2 lineage (done, T2); (2) MEMO-GAP-T at 1,000 per set instead of 2,000 (SE of the gap ≈ 0.006 at a 0.98), only by a Theory note before any eval_textself output; (3) descriptive parts (LABELLER-FULL, LABELLER-KEPT-T, BF16-VS-FP32, DEPTH-PAIRED, TOKEN-COST, CARD-GATE) after the registered parts. **Never** cut: steps, gates, EQUIV, eval_text, LABEL-AUDIT-T.
- **Wall-clock:** the critical path is serial (base eval ∥ GATE-T1 + pool only if they interleave on the 3070; T2 labelling waits on T1's eval_text by the seal): **≈ 13–22 h** end to end after Audit's diff review [DERIVED v1_cost]. T1's eval_textself GPU parts overlap T2's cloud training.
- **The 3070 is the bottleneck.** At 15:09 EDT it was 44 % busy with §218 (LANG-VB G-L1b), the 4060 85 % (§217), the 2070 55 % (A0-1) [MEASURED nvidia-smi]. TEXT-SELF's 9.7–15.3 h of 3070 time is scheduled by Experiments (one eval card per experiment, same-card rule); the 2070 helps only after CARD-GATE.
- **Retention:** keep model.pt of T1 (the T2 labeller) and T2 (CTX-BOUND makes a text round 3 moot, so T2's model.pt may be deleted after scoring with Theory's OK); keep every eval / log / args / draws_log / pool_meta / eval_gate (with per-item records). Delete checkpoints and mirrors at landing; delete each pool after LABEL-AUDIT-T, CONTAMINATED and MEMO-GAP-T are scored (pool_meta + seeds + builder commit regenerate it).

---

## 9. Open items (for Theory / Audit)

1. **T4 at round 2 (V5).** v1 applies T4 to round 1 only, as written: an uncensored T2 that fails T-ABSORB-2 is read four-way with the qualifier; it mostly reads LATENT-AHEAD (P ≈ 0.89 in that branch, ≈ 0.11 of all outcomes). The alternative, **TEXT-NOT-ABSORBED-2** (round-1 label as ROUND-1-ONLY + Δ_2 descriptive), is computed in v1_oc (`structural`, "unconditional_s0.167": four-way 0.63, ROUND-1-ONLY 0.17, TEXT-NOT-ABSORBED 0.09; LATENT-AHEAD 0.35 unconditional). Round 2 reuses a trainer path that round 1 exercised, so RT M3's fault argument is weaker there; decide before registration.
2. **Trainer location (V1).** A new module with a parser copy and the PLAIN-EQUIV gate (nested170.py byte-identical) vs the draft's in-place nested170.py edit with G-T7 over all four §170 arms. v1 recommends the new module (vbank.py and §217 import nested170).
3. **cot_eval_bs in TEXT_NEUTRAL_NEW** is required at its parser default 64: the trainer runs no eval and skips the eval-memory check that halves it (nested170 l.943–950). The eval's own bs 32 lives in text_eval's meta. If Theory prefers it in META_T (as eval_bs is), one line changes.
4. **Deleted latent pools in the exclusion set:** regenerate from pool_meta and sha-check (s215 pool_r1 was rebuilt once, sha c2e226ae) or rely on the collision bound; both are valid, v1 allows either and pool_meta records which. Confirm.
5. **RT M1's optional $0 fp32 re-eval of `s170_cot_s2`** is not registered (it would be a fourth spread point for ŝ found after the OC was fixed). If run, descriptive only, after registration.
6. **The structural priors** (T-ABSORB 0.90, T-ABSORB-2 0.85, GATE-T2 0.93, uncensored 0.95) and the panel priors are the draft's / RT's, unmeasured for text; the LATENT-AHEAD headline depends on them.
7. **The precondition stream regeneration** depends on numpy's Generator stream (pool_meta records the numpy version; G-T1 asserts the per-n counts and that the regenerated draft items reproduce ts_label_check_s1.json's per-item keep at n 9–12, as RT's rt_reach did).
8. **`--final` trigger:** the Theory note that records the s1 chain as done should be named in the registration commit.
9. **Card fallback:** if the 3070 is unrecoverable before the base eval, every text eval moves to the 4060 with the latent comparators re-evaluated there first (§2.7). Experiments to confirm the 4060's fp32 eval memory (≤ 5.1 GiB projected) when §217 is done.

---

## 10. Verification log (this revision)

**$0 CPU checks** (research6/v1_checks/; CPU, ≤ 2 threads, nice 19, `python -B`, PYTHONDONTWRITEBYTECODE=1, OMP_NUM_THREADS=1; no RNG used; nothing written under D:/Latent):
- **v1_argkeys.py → v1_argkeys.json** [DERIVED; AST of boxes/nested170.py, no import]: 57 parser keys; **30 absent from s170_cot_s1/args.json** (the RT B1 list exactly); of those only keep_ckpt_steps is in score_s213.META; runtime attributes `excl_survive_by_n`, `task`; info extras `hostprobe_sps`, train_gpu / driver, env keys, `format`; base keys not in the parser are META or equal by construction (task, format); the textcot overrides (mem_write / write_mix_* null, think_k 0) match the base. nested170.py sha256 recorded.
- **v1_oc.py → v1_oc.json** (+ v1_oc.out) [DERIVED; exact Simpson quadrature (PROOFS pc_oc's rect, M 4,000), exact binomial / Poisson sums]: ŝ 0.16656 from pc_oc.json; the four-way OC at s ŝ / 0.20 / 0.35 for every case incl. the "marginal" split; reproduces PROOFS' values (P(PARITY \| 0 / 0) 0.964 / 0.915 / 0.608; headline 0.303 / 0.139 / 0.555 at ŝ, 0.195 / 0.184 / 0.590 / 0.032 at 0.35); bias rows; the gain branch (ρ_0 0 / 0.5); ROUND-1-ONLY labels; the v1 structural tree (sum_check 1.000); the ¬T-ABSORB-2 branch at Δ_2 −3 / −2; GATE-T1 / GATE-T2 / GATE-T1′ exact OC on the latent's count rules; GATE-T2 keep_16 under RT's reach projections; LABEL-AUDIT point term; CARD-GATE (Skellam per n, exact n50 enumeration at n 10–11); per-cell T-DBS / T-DBS-2 (the draft's ts_oc arithmetic on v1's factors).
- **v1_cost.py → v1_cost.json** [DERIVED; inputs ts_cost.json, pc_cost.json, rt_reach.json]: pool sizes and spare (388,000 / 390,000; 1.18 % / 1.70 %); hypergeometric per-n consumption; cloud hours / $ per cell, admission headroom (+19.8 % / +19.2 %); local 3070 stage hours (9.7–15.3 h); critical path (13.3–22.3 h).
- **Reproduced from the reviews without change** (cited, not re-run): PROOFS P1 (TF ≡ generation 1,500 / 1,500 + 400 / 400), P2 (fresh keep / purity), P3 (tokens, overflow, CTX-BOUND), P4 (stream replay), P5 (n50s, bars, panels), P6 (memory, padding, MEMO-GAP), P7 (filter); RT rt_reach (reach law, T1 projections, decode-step counts) and rt_oc (bias, POOLED, base-corrected, structural).
**Code / files read** [VERIFIED]: boxes/nested170.py l.148 (test_set), l.157 (train_items), l.635–659 (argparse), l.661–676 (textcot overrides, EXCL, excl_survive_by_n), l.794 (cot_answers), l.925–950 (auto-micro and the textcot eval-memory check: cot_answers on the longest test n-16 item, halving cot_eval_bs), l.956–966 (args.json = vars(a) + info), l.1039 (warm-up branch), l.1157–1180 (time_steps exit; model save; the unconditional final eval over test_set n 1–16); theory/score_s216.py l.1–110 (LIN, ALLOW, BASE_NEUTRAL_NEW, pending / --final), l.361–397 (args_diff, args_ref_check); theory/score_s213.py l.34–35 (META); s170_cot_s1 args.json, args_eval.json (cot_eval_bs 32, eval on the 3070), eval.json meta (bf16, RTX 3070), mem_check.json (12.99 GiB, micro 1); s215_ftself_s1 / s216_r2_s1 args.json and auto_micro_self.json (self micro 2 / 4); THEORY.md §216 (l.18625–18675), notes 216-1…216-9 (l.19011–19250); SD2_v1.md (all, incl. Addendum 1); SELFDEPTH_v1.md structure; the draft, REDTEAM_TS, PROOFS_TS and their scripts' headers and JSON.
**Directory listings only:** boxes/runs_fsq s215_* / s216_* (pool and gate files present: s216_pool_r1_s2/pool_r1.jsonl; s215_pool and s216_pool_s1 hold pool_meta only; gate_items.json in s215_gate1, s216_gate2_s1, s216_gate1_s2); scratch/a01_runs.sh header (A0-1 is §191 ProsQA, not nested170). `sha256sum s170_cot_s1/model.pt` (read only).
**Seeds grep** at 796a3343 (docs/THEORY.md, docs/research_*/*.md, docs/research_*/**/*.py, boxes/*.py, theory/*.py, scratch/*.py, scratch/*.sh; excluding docs/research_textself_2026-10-18/): 23180–23201, 23260–23262, 23270, 23280, 23290, 23300 / 23301, 23360 / 23361, 23370, 23380, 23390: 0 hits each. Within this design's folder, 23180 / 23187 are model-read (precondition), 23188–23194 shape-only or Monte-Carlo; 23195 is new (G-T18).
**Names grep:** `s219_`, `score_s219`, `textself_ft`, `text_eval.py`, `boxes/textself`: 0 hits; no §219 in THEORY.md.
**Still unbuilt (flagged):** every §2.6 edit; score_s219; the stage scripts. Memory, time, labelling and eval projections are model extrapolations; G-T9 / G-T10 / G-T16 / G-T18 decide.

*v1 written 2026-10-18 15:19–15:32 EDT (Opus design-revision subagent for Theory; $0; nothing committed, registered or launched).*

---

## Addendum 1 (2026-10-18 15:35 EDT; Theory's decisions on §9's open items, before review / registration)

1. **T4 applies to both rounds.** A T2 that fails T-ABSORB-2 reads **TEXT-NOT-ABSORBED-2**, never LATENT-AHEAD-2. §6's computed OC row for it is the registered one.
2. **The trainer is the new module boxes/textself_ft.py.** nested170.py stays untouched; no edit is made to a module in use.
3. **cot_eval_bs** goes in TEXT_NEUTRAL_NEW at 64, so a change is caught by ARGS-DIFF. It is not META.
4. **Deleted latent pools are not regenerated.** Exclusion against them guards fresh-once within an arm, not test contamination. Text pools exclude every test, gate, SERIAL, 213-37 and rng-23187 item and the sealed n 25–30, regenerated from seeds where needed. Text vs latent pool overlap is bounded by the collision bound (PROOFS_SELFDEPTH §8: P(render = s) ≤ (1/11)·40⁻ⁿ) and changes no reading.
5. **The structural priors stay [ASSUMED]** as §6 states. The sensitivity rows (s 0.17 / 0.35) are the registered reporting.
6. **3070 time.** Experiments schedules text pools and evals on the 3070 around §218. The card class is fixed (3070, fp32), and the 2070 is used only after CARD-GATE.

