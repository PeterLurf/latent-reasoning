# ORDERED-WORDS (OW): latent multi-step reasoning on a forward-order, mod-11 word task (draft, 2026-10-23 01:25 EDT)

**Status:** draft for Theory, then Literature (novelty, §12) and Audit (cap ≤ $30). Not registered. Nothing has been built or run, apart from the $0 CPU checks in this folder.
- **Follows:** Note 223-11 (§223 closed: NESTED-WORDS d ≥ 2 not learned one-pass) and Audit's 10-23 conditions.
- **Literature basis:** LIT_ONEPASS_DEPTH_THRESHOLD_2026-10-23.md (Lin 2503.07604: one-pass multi-step learning needs fixed premise order and small values).

## 0. What it can support, stated up front
- **C1 (the target), scoped:** "On a templated multi-step word task in natural-language frames (forward order, values mod 11), GPT-2 small's CODI-style latent steps do work past the measured one-pass boundary. Removing them (K′ 0 / 1) drops accuracy, and a mid-chain latent swap moves the answer to the donor-state counterfactual (serial use)."
  - One seed, templated task, forward order only. Never GSM8K. Order robustness is stage 2 (§11).
- **C4, qualified (CoT-distillation sense, as §175):** trained from final answers plus one text step sentence per latent ("<name> has <v> <u>."), i.e. the step text a CoT dataset supplies.
  - There are no non-text simulator states and no teacher model.
  - Strict C4 (answers only) is **not** claimed.
  - **Audit decides:** if generator-written step sentences count as hand-built states, the claim is C1 only.
- **C2, C3, C7: descriptive only.** C2 = depth extrapolation d 9–12 at K′ > the trained K. C3 = the TEXT comparator (§4). C7 = step-value decoding.
- **Not claimed:** C5, and C6 (one seed).

## 1. Task: OW v0 = NESTED-WORDS v1.2 with exactly two changes
- **Kept by reference** (`ow_words.py` imports boxes/nested_words.py and never edits it):
  - FRAMES (8 per op, frames 6–7 held out of training), NAMES, UNITS, GIVEN, QUESTION, the op set (including the forced cross-spine mul_s at d ≥ 3);
  - 0–2 distractor sentences;
  - `_statements` / `_finish` (rendering, R_t), `topo_shuffle`, and the regex solver.
- **Change 1, values mod 11:**
  - Every quantity is in 0..10, and every op is reduced mod 11, with no range rejection.
  - Constants and givens: add/sub 1..10, mul 2..10. S_0 ~ U(0..10). Distractor values 1..10.
  - mul_s keeps v1.2's rule (both factors ≥ 2, partner computed). A mul with S_{t−1} = 0 is redrawn, as in v1.2.
  - There is **no** "mod 11" sentence in the text: the convention is learned from data, as §170's expressions were.
- **Change 2, dependency (forward) order:**
  - S_0 first, then for t = 1..d: [G_t if step t reads a given] then S_t.
  - Each distractor goes at a uniform random slot; distractors read nothing, so any slot is a dependency order.
  - `order_mode="topo"` reproduces v1.2's random linear extension. It is used for the optional VAL-only pilot and for stage 2.
- **Added item keys:** R_step (the step sentences for t = 1..d; R_step[−1] == R[d]), cot, mod, order_mode. No v1.2 key is missing.
- **Example** (d 3, forward; Ted is a distractor): "Jim has 10 apples. Ted has 9 apples. Leo has 7 apples. Ivy has the apples Jim has beyond Leo's. Ian has as many apples as Ivy and Jim together. Ian's apples times Ivy's apples make Eli's apples. How many apples does Eli have?"
  - Values 10 → 3 → 2 (13 mod 11) → 6.
  - cot = "Ivy has 3 apples. Ian has 2 apples. Eli has 6 apples."
- **Seeds:** test 22900, pairs 22910, train 22930 (not used elsewhere in boxes/*.py). Test set: 1,000 per d, d 1..12, all 8 frames. Train: d 1..8 by CurGate, frames 0–5. Exclusion is by rendered match, as v1.2.

## 2. $0 generator diff check (F1-diff): `ow_gen_check.py` → `ow_gen_check.json`: **PASS**
- **Inputs:** nested_words sha256 cd659df8… (= the §223 frozen generator_sha256); ow_words sha256 5c0c9c08…
- **Sample:** 2,000 items per d, d 1..10, for each of OW-forward, OW-topo and v1.2.
- **Identity:**
  - all 10 shared objects are `is`-identical to v1.2's (`identity.all_shared` true);
  - re-implemented: `_operand`, `_draw_program`, `make`, `redraw` (the value law and the order call only);
  - `added_item_keys` = [R_step, cot, mod, order_mode]; `missing_item_keys` = [].
- **Values:**
  - `values.ow` min 0 / max 10 at every d;
  - `answer_max_share` 0.102–0.115 (chance 0.091);
  - `final_op_lookup_acc` (answer predicted from the last op alone) 0.090–0.112 ≈ chance.
  - v1.2 by contrast: `final_op_lookup_acc` 0.014–0.053 vs chance ≈ 0.01, so op identity leaked about 2–5× chance through the range rejection.
  - `restarts_per_item`: ow 0.225, nw 5.75.
- **Order (a finding):**
  - `items_with_dependency_violation` = **0 / 20,000 for v1.2 too**. v1.2 never stated a premise before its dependency, so its spine was already in forward order.
  - What OW actually changes is where the givens sit: `given_to_use_gap_mean_sentences` drops from 4.43 (v1.2; max 18) to 0.10 (OW; max 2, from distractor slots).
  - The program layout becomes fixed: `share_program_layout_canonical` goes from 0.22 to 1.00.
- **Solver:** the independent regex solver (mod 11) re-derives the answer from the text and from every R_t: `solver.mismatches` 0 (200 items per d per mode). Numbers 0..10 are single GPT-2 tokens.
- **Frames:** chi² vs uniform over the 8 frames, per op, is 2.7–12.6 (df 7). The same law as v1.2.
- **Shortcut panel** (OW-forward, all frames): at d ≥ 2 every single-feature rule (majority, first number, last number, sum mod 11, product mod 11) is ≤ 0.135 (gate ≤ 0.25).
  - At d 1, product mod 11 = 0.398. d = 1 is a single mul_v, so this is the task, not a shortcut.
  - `affine_drop_cross_spine` solves d 1–2 exactly (no cross product there). It is descriptive, as in v1.2.
- **Pairs** (t = ⌈d/2⌉; kept iff own / donor-final / cf distinct):
  - `kept_share` 0.59–0.74 at d 4..10. Token-length mismatches 0.
  - **d = 3 is structurally degenerate: kept_share 0.** With t = 2, the forced mul_s at t = 3 must take j = 1, so cf(a) = donor S_2 × donor S_1 = donor-final for every pair.
  - The same holds for v1.2 (checked: 300 / 300). §223's ps_eval d = 3 swaps would have kept 0 pairs; this is harmless, since those readings were never read.
  - **The serial reading pools d ≥ 4.**
  - The ideal-carrier cf floor is 0.089–0.140.
- **Exclusion:** train draws hitting a test signature = 0.0 at d 1 and 2 (20,000 draws each). Question tokens ≈ v1.2 (35 / 58 / 97 / 151 at d 1 / 3 / 6 / 10).
- **Rejected variant B** (bounded naturals 0..10, no wrap): `restarts_per_item` 21.2 at d 3; `answer_max_share` 0.375 / 0.24 / 0.175 at d 3 / 4 / 6. The answer law leaks, so mod 11 is used.

## 3. Arms (minimal set) and code changes
| cell | flags (beyond the common OW recipe) | role |
|---|---|---|
| **DIRECT** | --direct | one-pass baseline; fixes n\* and the band (G-W0) |
| **LAT** | --recode --lam 1 --recode_target step | CODI-style latent loop, question visible, a per-latent step-sentence decoder (M1 with a short target); **the C1 test cell** |
| **TEXT** | --direct --text_cot | text-CoT comparator (C3, descriptive); decoder = cot + "The answer is: N" |
| VAL-only (pilot only, optional) | --order_mode topo | descriptive attribution: values vs order |

- **Common recipe:** codi_ps (GPT-2 small + LoRA r128, bf16 autocast, clip 2.0, eff_bs 64) with the §223 lessons:
  - **lr 3e-4** (223-8 R1);
  - **cg_thr 0.45, cg_cap 6000**, need 2 × 1k bins (the 223-10 CurGate fix: the from-scratch first bin can leave the last-2 window);
  - d_stages 1..8, steps 48,000 (as 223-10), K = d + U{0..2} capped at 10.
- **No M2 bottleneck.** M2 blocked learning (223-7), and it makes K′ = 0 vacuous. Without M2 the answer can read the question, so C1 rests on removal plus the serial swap (§6).
- **Why the step-sentence target instead of 223's re-rooted R_t:**
  - [DERIVED] D4's serial model needs only S_t in z_t. The remaining program stays visible in the question, so R_t's program text is redundant for the answer path.
  - The step target equals TEXT's CoT unit, so LAT is "TEXT's CoT moved into latents" (a clean C3 / C4 pairing).
  - [MEASURED, 223-9 P2] M1's R_t value tokens stayed poorly decoded (`recode_val_ce` 2.19 at 7,600 vs `recode_ce` 0.42).
  - The step target is about 6–8 tokens vs 35–150, so the D head costs far less.
  - Not new: this is SIM-CoT 2509.20317's step decoder, cited as a recipe choice, not a claim.
- **codi_ps changes** (all flags default-off ⇒ bitwise §223 behaviour):
  1. `--task {nw, ow}` (ow → boxes/ordered_words.py = ow_words.py moved, seeds 22900/22910/22930) and `--order_mode {forward, topo}`.
     - args.json gains ordered_words_sha256 and order_mode.
  2. `--recode_target {reroot, step}`. step → tok_item's R = R_step tokens. The HOLD target for j > d stays R_d (= R_step[−1]).
  3. `--text_cot` (needs --direct): dec = [EOT] + enc(cot + "\n") + enc("The answer is: N") + [EOS].
     - The CE is over all dec tokens. num_ce / CurGate read the answer position only (CP.ans_pos already searches for ANS_PROMPT).
     - tok_item's `dec[1:5] == ANS_PROMPT` assert becomes "ANS_PROMPT present once".
- **ps_eval changes:**
  - the generator switch from run args (test set, pairs, and `cf_values` with OW.apply mod 11);
  - acc over d 1..12;
  - swaps / budget_front over d 4..8 (d 3 is degenerate);
  - **new part `k_abl`:** K′ ∈ {0, 1, ⌈d/2⌉, d, d+2} via `run(n_lat=…)`, plus MEAN-Z (every z_j replaced by its test-set mean at that j), d 1..12;
  - **new part `text`:** greedy decoding with a budget of len(cot tokens) + 20; the answer is parsed from "The answer is: N", anything else is wrong; generated tokens are stored per item;
  - decode reads the step targets (descriptive).
- **Build gates (before any pilot):**
  - Audit's G-D on the diff;
  - the flags-off equivalence gate: a ≥ 50-step bitwise run, 223 commit vs new, same card / seed (as 223-5(b));
  - the plain-path smoke;
  - per-term gradflow: the step-recode CE gives nonzero grad on the D head and on prj / LoRA through z_j; no detach;
  - `--args_only` single-change diffs (§3 table);
  - F2 memory / timing at the deepest shape (d 8, K 10) on the 4060 and the 3090.
- **Planned args diff vs scratch/s223_args_frozen_direct.json (b20663fb…), for DIRECT-OW:**
  - changed: lr 0.003 → 0.0003; cg_thr 0.3 → 0.45; cg_cap 2500 → 6000; steps 24000 → 48000; stream_seed 22330 → 22930;
  - new keys: task "ow", order_mode "forward", recode_target "reroot" (inert), text_cot false;
  - hashes: generator / module / test / pair shas, excl_n.
  - **LAT vs DIRECT-OW** = {direct, recode, lam, recode_target, cell}. **TEXT vs DIRECT-OW** = {text_cot, cell}. **VAL-only vs DIRECT-OW** = {order_mode}.

## 4. $0 pilots (pre-stated gates; local RTX 4060, registered kernel, checkpoints / resume)
- **P0, DIRECT learnability (binding).** DIRECT-OW, --stop_at 12000 (≈ 72 min at 223-9's 0.36 s/step).
  - **LP0:** after the first logged window, no 100-step mean gn_preclip > 10 or non-finite value, and no non-finite rows.
  - **G1 (the language is read):** the stage 0 → 1 advance is by threshold (stages.jsonl reason "threshold…").
  - **G2 (composition; binding):** pooled num_acc in the last completed 1k bin at d_max = 2 (d ~ U{1, 2}; before the 1 → 2 advance, or at stop) **≥ 0.80**, as 223-10.
  - **G3 (CurGate works):** the stage 1 → 2 advance is by threshold by step 12,000.
  - **Outcomes:**
    - All pass → P1.
    - G2 passes but G3 fails → THRESHOLD-MISSET. cg_thr for every cell := the 1k-bin num_ce of the first bin with pooled num_acc ≥ 0.80 at d_max 2, + 0.05, rounded up to 0.05. It is set from DIRECT only and recorded in a Note. No new pilot.
    - LP0, G1 or G2 fails → **OW stage 1 closes descriptively**: "forward order + mod 11 does not make d = 2 one-pass learnable for GPT-2 + LoRA r128 in 12k steps". No cloud, no further pilot.
  - Train-time accuracy at d_max ≥ 3 is a disclosed DIRECT-only preview. It does not fix n\*.
- **P0b, VAL-only (optional, descriptive, never gating):** the same as P0 with --order_mode topo, run if the card is free. It attributes G2 to values vs order, which feeds stage 2.
- **P1, latent and text learnability (only after P0 passes):** LAT and TEXT, --stop_at 12000.
  - **LP0** per cell.
  - **LP2:** LAT's stage 0 → 1 advance is by threshold.
  - **LP3:** LAT entry_step ≤ P0 DIRECT entry_step + 1,500.
  - **LP4:** LAT pooled num_acc at d_max 2 (last completed 1k bin) ≥ P0 DIRECT's − 0.10.
  - TEXT: LP0, plus a greedy eval at stop on 200 test items per reached d (descriptive; TEXT does not gate).
  - **Outcomes:**
    - LAT passes → an args re-freeze Note, then cloud DIRECT.
    - LAT fails → stage 1 closes descriptively, with no cloud spend (DIRECT serves only the latent cell, as 223-4).
  - A pilot that dies is resumed from its checkpoint, not failed.

## 5. G-W0: the depth band (from cloud DIRECT, before any LAT / TEXT launch)
- **223-6 guards:**
  - DIRECT test acc at d = 1 must be ≥ 0.80, and entry_step must not be None. Otherwise **NO-ENTRY**: every reading is UNREADABLE and no latent cell runs.
- **n\*** = the largest d in 1..8 with DIRECT acc > 0.50. **Band B** = {d ∈ n\*+1..8 : DIRECT acc ≤ 0.50}.
  - B needs **≥ 3 depths, with ≥ 2 of them ≥ 4** (d 3 has no informative pairs).
  - B is fixed in a dated Note from DIRECT alone (as 170-3).
- **B too small (n\* ≥ 6, i.e. one pass suffices in forward order)** → label **ONE-PASS-SUFFICES-OW**. No latent cells; stage 1 closes descriptively; realised spend is DIRECT only.
  - That outcome motivates stage 2 (order shuffling may create the band); stage 2 is not registered here.
- The shortcut panel already passes at the generator level (§2: every rule ≤ 0.135 at d ≥ 2).

## 6. Readings (LAT; pooled over B; test 1,000 items per d; K = d + 2 unless stated)
- **R1 DEEP-GAIN:** LAT acc − DIRECT acc ≥ 0.30, **and** LAT acc ≥ 0.60.
- **R2 REMOVE:** acc(K) − acc(K′ = 0) ≥ 0.30, **and** acc(K) − acc(K′ = 1) ≥ 0.30. MEAN-Z is descriptive.
- **R3 SERIAL (cf-follow):** PREFIX-SWAP(t = ⌈d/2⌉): the donor's z_1..z_t are fed as inputs, and the recipient's steps t+1..K run with the recipient's question visible.
  - Over kept pairs in B ∩ {d ≥ 4}, capped at 1,000 per d, with **N ≥ 1,000**:
  - follow(cf(a)) − floor ≥ **0.30**, where floor = the same swap with the next kept pair's donor, scored against this pair's cf(a).
  - SELF-SWAP must be bitwise identical to plain.
  - Descriptive: donor-final follow, cf_b, LEAK-SEG, MARKER-SWAP, BUDGET-FRONT.
- **Labels, in precedence:**
  - UNREADABLE (§8);
  - NO-GAIN (R1 fails);
  - **C1-LANG** (R1 ∧ R2 ∧ R3);
  - DEEP-NOT-SERIAL (R1 ∧ R2, R3 fails: the latent positions add compute but carry no portable state);
  - DEEP-UNREMOVED (R1, R2 fails).
  - Every label is worded "one seed, templated word task, forward order, mod 11".
- **Descriptive:**
  - C2 extrapolation: LAT vs DIRECT at d 9–12, K′ = d + 2 (beyond the trained K ≤ 10), and the K-sweep;
  - step-value decoding from z_t (C7-type, decodable ≠ used);
  - per-cell entry steps (basin-entry rule: a > 5k gap between LAT and DIRECT ⇒ attribution CONFOUNDED).
- **C3 (TEXT, descriptive, §225's axes):**
  - TEXT greedy acc by d;
  - (S) LAT K vs TEXT generated tokens per item;
  - (F) block-position compute: LAT 12·(P + K + n_dec) vs TEXT 12·(P + ntok), a proxy (one forward per position; KV cached in both).
  - C3-MATCH-S-OW (descriptive) holds if LAT ≥ TEXT − 0.02 at every d in B.

## 7. Derivation (assumptions, tagged claims, predictions)
- **A1 [ASSUMED]:** GPT-2 small + LoRA r128 can fit 11 × 11 op tables per frame family. §170 (full fine-tune) learned one-pass n ≤ 2 at ≥ 0.90 on mod-11 expressions; LoRA is less tested.
- **A2 [ASSUMED]:** Lin's one-pass result (from scratch, symbolic premises) transfers in direction to pretrained GPT-2 with paraphrase frames.
- **D1 [PROVEN]:** in Z_11 every regular step with a fixed operand is a bijection of the state (add/sub are translations; mul by b ∈ 2..10 is invertible because 11 is prime).
  - So for a program without mul_s, S_d is a bijection of S_0 given the operands. Any rule that ignores S_0 or any one step's operand is at chance given that input.
  - There is no early-exit shortcut, and the answer law carries no op identity. Measured: `final_op_lookup_acc` ≈ chance vs v1.2's 2–5× leak (§2).
- **D2 [MEASURED, §2]:** v1.2 already had 0 dependency violations. OW's order change is given placement (gap 4.43 → 0.10 sentences) plus one fixed program layout.
  - [DERIVED] If Lin's "unfixed order" means premises before their dependencies (Literature to confirm, §12 N4), then §223's d ≥ 2 failure is better explained by values / range leakage than by order. P0b tests this.
- **D3 [DERIVED]** (the swap logic; models defined on the kept pairs, where own, donor-final and cf(a) are distinct):
  - (a) **Serial carrier:** z_t holds S_t (and back-referenced S_j, j ≤ t, in earlier latents), and later latents apply the recipient's visible operands. PREFIX-SWAP(t) then gives **cf(a)**.
  - (b) **Relay or precompute:** z_{≤t} hold the donor's question or answer, giving **donor-final**.
  - (c) **Latents unused, or pause-like compute with no item state:** gives **own**.
  - Only (a) passes R3. This guard is what makes C1-LANG serial evidence rather than extra positions. **d = 3 is excluded** because cf(a) ≡ donor-final there (§2).
- **D4 [DERIVED]:** R2 alone is weak, because K′ = 0 is off-distribution for a cell trained with K ≥ d. R1 ∧ R2 ∧ R3 is required, and R3 carries the use claim (the §203 decodable-vs-used lesson).
- **D5 [ASSUMED]:** λ = 1 for the step-recode CE (the §175 / SIM-CoT value; not derived). [MEASURED, 223-7 / 223-9] the recode term does not starve the answer (M1 ≈ DIRECT).
- **D6 (Goodhart):** the recode target is a decodability target only. A model that decodes v_t but answers from the question fails R2 / R3 (it lands in DEEP-UNREMOVED or NO-GAIN), never C1-LANG.
- **Predictions [ASSUMED, shaded]:**

| event | P |
|---|---|
| P0 passes (DIRECT d ≤ 2 pooled ≥ 0.80) | 0.60 |
| P0b reaches G2 too, given P0 passes (order secondary at d ≤ 2) | 0.75 |
| band B valid (n\* ≤ 5), given P0 passes | 0.75 |
| P1 LAT passes, given P0 passes | 0.80 |
| R1, given B | 0.30 |
| R2, given R1 | 0.85 |
| R3, given R1 ∧ R2 | 0.45 |
| **C1-LANG overall** | **≈ 0.05** |
| DEEP-NOT-SERIAL overall | ≈ 0.05 |
| TEXT ≥ 0.90 on B, given it trains | 0.70 |

  - The likeliest outcomes are a P0 close (≈ 0.40) or NO-GAIN (≈ 0.25). Both are informative for the language-C1 route: the first for Lin-transfer at LoRA scale, the second for latent-vs-one-pass on language.

## 8. OC block and censoring
- **OC** (`ow_oc.py` → `ow_oc.json`; one seed; 20,000 reps; design effect 1.3; |B| = 3 × 1,000 items; N = 1,000 pairs):

| truth | P(R1) | P(R2) | P(R3) | P(C1-LANG) |
|---|---|---|---|---|
| no effect (LAT = DIRECT 0.30, K′ 0/1 = K, cf = floor 0.10) | 0.000 | 0.000 | 0.000 | **0.000** |
| effect, strong (0.80 / 0.30, K′0 0.25, cf 0.55) | 1.000 | 1.000 | 1.000 | **1.000** |
| effect, marginal (0.64 / 0.30, K′1 0.32, cf 0.43) | 0.999 | 0.922 | 0.924 | **0.852** |
| truth exactly at the bars | 0.382 | 0.336 | 0.509 | 0.123 |
| pause-like compute, no state (0.70, cf = floor) | 1.000 | 1.000 | 0.000 | **0.000** (DEEP-NOT-SERIAL 1.000) |
| relay / precompute (cf = floor; donor-final follows) | 1.000 | 1.000 | 0.000 | **0.000** |

- **Censoring (rule b):**
  - Dead, diverged, contaminated or ARGS-DIFF → every reading of that cell fails (UNREADABLE, never a pass).
  - DIRECT d = 1 < 0.80 or NO-ENTRY → G-W0 UNREADABLE → no latent cells. LAT NO-ENTRY → every LAT reading fails.
  - **Depths a cell never reached in its curriculum are scored as measured** (a wrong answer counts as wrong), and the final stage is reported; nothing is dropped.
  - Non-kept pairs are excluded, with the share reported. d = 3 is excluded structurally. < 1,000 kept pairs → R3 UNREADABLE → C1-LANG not passed (label DEEP-UNREAD).
  - TEXT output that is unparseable, does not terminate, or runs over budget → wrong.
  - A missing file reads pending (215-7). A pilot crash is resumed, not failed.

## 9. Realised-$ estimate (cloud RTX 3090, $0.22/h, one GPU class; pilots $0)
- **Rates:** DIRECT ≈ 0.15–0.25 s/step. LAT ≈ 0.6–1.5 s/step: the 4060's M1 ran 3.16 s/step with 35–150-token R_t targets, and the step target is ≈ 7 tokens. TEXT ≈ 0.25–0.5 s/step. F2 measures the real rates before admission.

| cell | train 48k | eval + setup | $ |
|---|---|---|---|
| DIRECT | 2.0–3.3 h | 0.5 h | 0.55–0.85 |
| LAT | 8–20 h | 2 h (k_abl, swaps, front, decode) | 2.20–4.85 |
| TEXT | 3.3–6.7 h | 1 h (greedy) | 0.95–1.70 |
| **total** | | | **≈ 3.7–7.4** |

- **Family cap $10** (≤ $30, so Audit can auto-approve). Shares: DIRECT $1.50, LAT $6.00, TEXT $2.50.
- Guards are set at projection × 1.3 with marks to the final step. Off-host full-state mirrors with a tested resume. Drain waiters are armed at launch.
- **Sequence:** P0 → P1 → cloud DIRECT → G-W0 Note → LAT + TEXT in parallel.
- Early closes spend $0 (P0 / P1) or ≈ $0.85 (ONE-PASS-SUFFICES-OW).

## 10. Not in the minimal set (and why)
- **PLAIN** (latent loop, answer CE only; the strict-C4 arm): +$2–5. CODI-type loops without step supervision have not learned latent use in our record. It can be added after a C1-LANG result.
- **M2 bottleneck:** blocked learning (223-7) and makes K′ = 0 vacuous.
- **Second seed:** only on C1-LANG, for C6.

## 11. Stage 2, ORDER-SHUFFLE (described, NOT registered)
- **Axis:** premise order as its own factor, at three levels on the same generator (`order_mode`):
  - (i) forward (stage 1);
  - (ii) topo = v1.2's random linear extension (givens float, gap ≈ 4.4 sentences);
  - (iii) free = any permutation of the premises, so statements may precede their dependencies (Lin's unfixed order, if N4 confirms).
- **Question:** does one-pass accuracy decay with order freedom (Lin) while the latent loop's does not? "Latent steps buy order robustness" would be a C1-on-language strengthening.
- **Shape:** DIRECT and LAT, fine-tuned from the stage-1 checkpoints or trained fresh on mixed order (to be decided at registration).
  - Readings: acc and cf-follow by order level within B.
  - Its own derivation, OC and $0 pilot.
- **Trigger:** registered only after stage 1 reads C1-LANG, DEEP-NOT-SERIAL or ONE-PASS-SUFFICES-OW. In the last case, order shuffling may create the band.

## 12. Novelty sentences for Literature (no "new" / "first" until checked)
- **N1:** a one-pass vs CODI-style latent comparison on a templated multi-step word task with paraphrase frames, names and distractors (forward order, mod 11), where removal (K′ 0 / 1) plus a mid-chain latent-prefix swap scored against the donor-state counterfactual establish serial latent use past a measured one-pass boundary.
  - Expected: narrow (CODI 2502.21074; Coconut; SIM-CoT 2509.20317; Lin 2503.07604; internal §221 / §223).
- **N2:** the counterfactual-follow reading (the recipient's remaining ops applied to the donor's intermediate state) as the guard separating serial state from relay / precompute.
  - Internal precedent: §221 CARRIER (221-2), §223 cf(a). External: SCIT 2608.27265; Liang & Pan (patching at step t).
- **N3 (stage 2):** premise-order shuffling as a separate axis for latent (not one-pass) reasoning: do latent steps remove Lin's order-shuffle decay?
- **N4 (fact check, not novelty):**
  - Lin's value regime (modular? bounded? range);
  - whether Lin's "unfixed order" includes premises before their dependencies (§7 D2 depends on it);
  - whether Arun IwMfUz9KNj's "layout diversity" covers frame paraphrases (OW keeps them).

## 13. Main risks
1. **LoRA vs full fine-tune:** §170's one-pass success used full fine-tuning at lr 1e-4. A P0 close would confound task and recipe. A full-FT DIRECT pilot would be a separate registration.
2. **Wrap-around semantics:** mod 11 makes "more / fewer / together" false in ordinary English on a sizeable share of add/sub statements (e.g. 3 + 10 → 2). GPT-2's priors may slow learning, and the task becomes "language with a convention".
3. **Order was already forward in v1.2** (0 violations). The order change is small (given placement), so OW's lever is mainly the value change. P0b attributes it, and stage 2 needs Lin's definition (N4).
4. **Lin's regime may make one pass too strong** (n\* ≥ 6 → ONE-PASS-SUFFICES-OW, ≈ 0.25 given P0 passes). The band may sit at d 6–8, where curriculum budgets are thinnest.
5. **LAT may track DIRECT** (223: M1 ≈ DIRECT) → NO-GAIN. Or it may use the latents as pause compute with the question visible → DEEP-NOT-SERIAL.
6. **CurGate threshold 0.45 was set on the 0–99 task;** the mod-11 CE scale differs (handled by the pre-stated THRESHOLD-MISSET rule).
7. **Engineering:** three new flags plus two eval parts need G-D, the equivalence gate and gradflow. Watch for lazy-import skew and no module edits during local pilots. Long local LAT pilots (≈ 3–5 h) run on an unstable box (checkpoints are mandatory).

## Files (this folder; $0, CPU, single thread)
- `ow_words.py`: the OW v0 generator prototype (moves to boxes/ordered_words.py at build).
- `ow_gen_check.py` → `ow_gen_check.json`: the F1-diff vs v1.2 (PASS).
- `ow_oc.py` → `ow_oc.json`: the OC block.
