# STEP-TOLD (ST) draft v0: a latent arm whose think pass t is told WHICH step to compute (2026-10-24)

**Status:** draft for Theory → red team → Literature → Audit. Not registered; no § number yet (cells named `st_*` here). Nothing committed by the drafter.
**Basis:** the user's 10-24 decision (after HOLD-C1, the next language design is a STEP-TOLD latent arm on GPT-2 small); §226 (one-pass DIRECT learns d = 1, not d = 2); §227 (text-CoT 1.000 at d 2–4); §228 / §229 (the §184 INJECT arm at chance for d ≥ 2; Notes 228-2, 229-1); REVIEW_LB #1 / #7 and §231 (Notes 231-2, 231-3: the think pass never computes the unrequested intermediate).
**$0 checks in this folder (all run; CPU, ≤ 2 threads, idle priority):** `st_cue_check.py` → `.json` (cue leak proofs, lengths, d = 2 pairs), `st_told_zs.py` → `.json` (zero-shot: do the §228 trunks read the question's name?), `st_oc.py` → `.json` (OC + cost). `st_ask_local.py` is the local GPU check P1-ASK (for Experiments; smoke-tested 3 steps on CPU, EXCL / TEST2 sha identical to §228).

## 0. Pre-stated before running st_told_zs.py (stub written 2026-10-24 11:48 EDT, kept verbatim; not commit-registered, so it is disclosed, not certified)
Items: TOPO, frames 0-5, dag on, stream seed 23241, 1,000 d = 1 + 1,000 d = 2 items. Models (closed, not blinded): s228_pdir2c_s1 (DIRECT), s228_plat2_s1 (LAT).
Variants on d = 2: ORIG (question about S_2), REPL (question re-targeted to S_1), APP (original text + the S_1 question appended), ASK0 (question re-targeted to S_0, lookup control).
Predictions [ASSUMED]:
- DIRECT: acc(v_1 | REPL) 0.35-0.62 (its d = 1 acc is 0.623); acc(v_1 | APP) 0.25-0.55; acc(v_2 | ORIG) ~0.09; acc(v_0 | ASK0) >= 0.6.
- LAT pass-1 aux value acc (teacher-forced): d = 1 ~0.77; d = 2 ORIG ~0.25; d = 2 REPL 0.45-0.77; d = 2 APP 0.35-0.70.
Rule (soft launch gate):
- TOLD-ZS-SUPPORT: DIRECT acc(v_1) >= 0.30 under REPL or APP, OR LAT pass-1 aux on d = 2 >= 0.45 under REPL or APP.
- TOLD-ZS-NULL: every REPL / APP reading <= its majority share + 0.10 -> no pilot spend; the draft returns to Theory / Audit.
- Otherwise WEAK: pilot allowed, prior P(LP2) lowered to 0.35.
- Cue format: APP is primary (it keeps the question and matches CUED-DIRECT); REPL replaces it only if REPL beats APP by >= 0.15 in BOTH models.

**ZS result (st_told_zs.json; run 11:51–12:22 EDT; $0; 1,843 s CPU). Majority shares: d2 v_0 0.110, v_1 0.109, v_2 0.107; d1 v_1 0.113.**

| reading | DIRECT s228_pdir2c_s1 | LAT s228_plat2_s1 (pass-1 aux, v_1) |
|---|---|---|
| D1 (d = 1, asked) | v_1 0.648 | 0.753 |
| ORIG (d = 2) | v_2 0.094, v_1 0.086 | 0.242 |
| REPL (asked S_1) | v_1 **0.087** | **0.239** |
| APP (S_1 cue appended) | v_1 **0.085** | **0.228** |
| ASK0 (asked the GIVEN S_0; positive control) | v_0 **0.084** (predicted >= 0.6) | – |
| CHAIN (cues S_1, S_2, ... at passes 1..8; descriptive) | – | answer v_2 0.098 (K' 2), 0.102 (K' 8) |

- **Literal reading under the rule: WEAK.** LAT's REPL / APP (0.239 / 0.228) are above majority + 0.10 = 0.209, so not NULL; nothing reaches SUPPORT. Format stays APP. As pre-stated: the pilot is allowed, and the prior P(LP2) is lowered to 0.35.
- **Disclosed, not a rule change:**
  - (i) The NULL line was mis-set for LAT. Its reference should have been ORIG (0.242), not the majority: the cue moves LAT's aux by −0.003 / −0.014 and DIRECT's v_1 by +0.001 / −0.001. **The cue has no zero-shot effect in either model.**
  - (ii) **The ASK0 positive control fails** (0.084 vs >= 0.6). On d = 2 items DIRECT outputs one value 35 % of the time (pred_share_max 0.35–0.37) whatever the question asks, even when it asks a GIVEN value.
  - So the check shows the trained trunks do not read the question's name (F4). It cannot test whether name-reading is *learnable* (A1); by 231-2's logic, a failed positive control leaves that question unread.
- **Consequence:** the binding $0 gate for A1 is the training check P1-ASK (sec. 6.1), not ZS. Audit to rule whether WEAK + a failed positive control is acceptable as the ZS outcome, or whether ZS should read UNREADABLE for A1 (the drafter's recommendation). Either way, no cloud spend comes before P1-ASK.

## 1. Claim and scope
- **C1 only, narrowed to C1-TOLD:** "given the step order by name (the plan), the latent loop executes it serially: pass t computes v_t from the state written by pass t − 1". One seed, GPT-2 small, TOPO word programs (mod 11, frames 0–5).
- **What the cue gives away, stated plainly:** the plan (which quantity, in which order), never a value. Finding the plan is part of what one-pass DIRECT fails at, so a cue-matched one-pass comparator is binding (§5).
- **C4 is not addressed:** the cue is a test-time scaffold built from the generator. A deployed model would have to produce its own cue (open question 3).
- A null speaks about this arm (INJECT + a per-pass name cue) at GPT-2 small, not about latent reasoning on language.

## 2. The arm: one change to §228's LAT, and the design choices
**Single change:** the think input at pass t is the item text plus a cue naming S_t's quantity, instead of the identical text every pass:
- `think_t = [EOT] + enc(" " + text + " " + QUESTION(u, S_min(t, d)))`, i.e. "… How many cups does Pat have? How many cups does Leo have?" at pass 1 of a d = 2 item. 7 extra tokens at every d (st_cue_check C4).
- Passes t > d repeat the S_d cue (hold). Decoder, aux (read-only on S_t, t ≤ d, value target R_step[t]), K = 8, lr, bs, write mix, aux weights and the item stream are §228's.

**Choice 1, the cue form: the question template with S_t's name, appended (APP).** Justified against the two alternatives:
- *Learned per-pass embedding (UT-style timestep, 1807.03819): rejected.* It tells pass t its index, not its target. Under TOPO order the pass would still have to locate S_t structurally (the computed node whose inputs are all given, for t = 1). That structural inference is exactly what §228's passes never did (REVIEW_LB check T: v_1 0.11–0.16 on d = 2). A pass index was already in the aux prompt (" Step t :") and did not help.
- *A bare "compute S_t" template: rejected.* It adds a token pattern the base has never seen. The question template is the one format a trained trunk maps to "compute this entity" (F1).
- *Re-targeting the question (REPL): rejected as primary.* It deletes the final target from passes t < d and has no information-matched one-pass form. APP keeps the question, and CUED-DIRECT gets the same tokens. Rule in §0: REPL only if it beats APP by ≥ 0.15 in both ZS models.
- The cue goes **at the end**: causal attention keeps the text rows identical across passes, and the last cue token is where a d = 1 question's answer forms (check T: U_LAST 0.60 > U_EAGER 0.53 on d = 1).

**Choice 2, the lower-block cache: exact per distinct cue.** gpt_lower is valid only for identical inputs (its docstring). With one d per step, a batch has d distinct think inputs (passes t ≥ d share the S_d cue). So: compute the lower blocks once per distinct cue and reuse them at the passes that share it [DERIVED: identical input → identical lower rows]. Lower-block work is d per step instead of 1; at d ≤ 2 that is 1.5, at U{1..8} 4.5. Gate: fp64 equality vs uncached (as §228's cache_lower gate).

**Deviations from §228 LAT, listed:** (s1) the per-pass cue; (s2) the per-distinct-cue lower cache (§228: one shared cache; §184: uncached); (s3) passes t > d see the S_d cue; (s4) +7 think tokens; (s5) new logging-only key `aux_vacc_by_d_pass` (REVIEW_LB #1); (s6) new eval parts (MEAN-AT-t, CUE-ONLY, D-SWAP2). Everything else is §228's, including §228's own listed deviations (a)–(g) from §184.

## 3. Derivation (a): why a step cue lets pass t compute v_t
Notation: chain S_0 → S_1 → … → S_d; S_t's one defining sentence reads S_{t−1} plus an operand (a given value, a constant, or S_j for add_s / mul_s); v_t ∈ Z_11.

**Facts.**
- **F1 [MEASURED]** one pass computes the value of the entity its question names when that entity is one op from given values: DIRECT d = 1 0.998 (P0d, 226-2), 0.623 at 8k (s228_pdir2c_s1, `eval_acc by_d["1"].acc`); LAT d = 1 1.000.
- **F2 [MEASURED]** the unasked intermediate is never computed: v_1 on d = 2 items is at chance in the trained write source u (U_EAGER 0.16, U_LAST 0.11) and in S_1 (0.16) (REVIEW_LB T). Frozen Qwen gives 0.112 and GPT-2 0.10–0.11 (231-2). Pass-1 aux value acc is 0.77 on d = 1 and 0.25 on d = 2 (T).
- **F3 [MEASURED]** a value present in the input is held: v_0 0.988 (Qwen RMS12), 1.000 (GPT-2 LN6) (231-3). Where u holds v_1 (d = 1), S_1 keeps it (S1_ALL 0.55 vs U_LAST 0.60).
- **F4 [MEASURED, st_told_zs]** the trained §228 trunks do not read the question's name. Re-targeting the question to S_1 leaves v_1 at chance: DIRECT 0.087 (REPL) / 0.085 (APP) vs 0.086 uncued. LAT's pass-1 aux reads 0.239 / 0.228 vs 0.242 uncued. Asking for the GIVEN S_0 (ASK0) also gives chance (0.084). On d = 2 items DIRECT emits one value 35 % of the time whatever is asked (st_told_zs.json). So 'asked = computed' (F1) holds only for the entity the training always asked.
- **F5 [PROVEN, generator; st_cue_check C6 14,000 / 14,000]** for t ≥ 2, S_t's sentence reads S_{t−1}, so v_t (t ≥ 2) is never one op from given values. The question always names S_d, the only computed entity no sentence reads, so in training the name is redundant with structure. This explains F4: reading the name was never needed.

**Assumptions.**
- **A1 [ASSUMED; tested at $0 by P1-ASK]** with training pressure, one pass learns *target-conditioned one-hop computation*: for a named X whose inputs are readable at that pass (text values, or the block-6 memory read), it computes v_X at the cue positions. F1 shows this for X = the sink. F4 shows it is not yet present for other names, so it must be learned. P1-ASK tests exactly this (pass 1's job, one pass, no memory).
- **A2 [ASSUMED; MEASURED on token expressions in §184]** a value present in the top write source at pass t is stored in S_t and is readable by the block-6 read at pass t + 1 as an input to that pass's computation. Storage holds on language (F3). Read-into-compute on language is untested: §228's d = 1 answers bypassed the memory (228-2).
- **A3 [ASSUMED]** one op fits one pass. At pass t ≥ 2: blocks 0–5 at the cue positions locate S_t's sentence and its operand (the cue is in the input, so the lower blocks see it), block 6 reads v_{t−1}, and blocks 6–11 apply the mod-11 op. At pass 1 all 12 blocks do a d = 1 job, which F1 shows fits.
- **A4 [ASSUMED]** the additive / delta heads keep earlier writes readable for back-references (add_s, mul_s reading S_j, j < t − 1).

**Derived.**
- **D1 [DERIVED from A1–A4, induction on t]** pass 1, told S_1: S_1's inputs are given (F5), so by A1 v_1 forms at the cue positions, and by A2 S_1 holds it. Pass t, told S_t: the inputs are v_{t−1} (S_{t−1}, read at block 6) and S_t's operand (text, or S_j by A4), so v_t forms (A1 with the read as the readable input, A3). Passes d + 1..K re-write v_d. The decoder (question = S_d) only *reads* v_d from S_K, a 0-hop step like F3.
- **D2 [DERIVED] why §228 failed under the same A1–A4:** every pass was asked S_d. For d ≥ 2, S_d's inputs are not readable at pass 1 (F5), so nothing targeted forms (F2 is the measured instance). S_1 is empty, and pass 2 faces the same problem: a fixed point. The only way out was to infer S_1 structurally, which no pass is asked to do, and the trunks do not even read names (F4). The aux names S_t only at read time, after the write. This matches 228-2 (pass-1 aux 0.51 pooled = 0.77 on d = 1 / 0.25 on d = 2).
- **D3 [ASSUMED, interpretation] why §184 worked uncued:** in nested token expressions the next reducible subexpression is syntactically local (innermost brackets), so "what to compute next" is a local pattern, not an unasked structural inference.
- **D4 [PROVEN, generator + st_cue_check] the cue is not a lookup:** v_t (t ≥ 1) is written nowhere in the text (it contains only v_0, operands and distractor values), and the cue contains no value (§4). The cue does give the plan, so CUED-DIRECT is required (§5).

**Prediction (a), pass-1 aux value acc on d = 2 items** (`aux_vacc_by_d_pass["2"]["1"]`, last 1k bin, 8k steps) [ASSUMED]:
- under ST, given P1-ASK PASS: **point 0.75, 80 % interval 0.30–0.95; P(≥ 0.80 = LP2) ≈ 0.35**. The §0 WEAK rule applied literally (the draft's pre-ZS value was 0.55, point 0.85);
- under the null (A1 false): ≈ 0.25 (the check-T value).
- Why above §228's d = 1 level (0.77): in §228 the d = 1 answer bypassed the memory, so aux and answer gradients pulled apart. Under ST the d = 2 answer needs v_1 in S_1, so they agree.
- Also predicted: pass-2 aux on d = 2 ≥ 0.60 if serial (P ≈ 0.5); CUED-DIRECT d = 2 < 0.40 at 8k (P ≈ 0.8).

## 4. Leak control (b): the cue names the step, never its value
**$0 proofs (st_cue_check.json, all PASS):**
- **C1 no number:** 0 / 4,000 items (d 1–8, every t ≤ 8) have a digit in any cue string, and 0 have an answer token among the cue tokens.
- **C2 skeleton-only:** on 2,000 same-skeleton redraw pairs (values differ in 1,992), the cue token lists are bitwise equal at every t in 2,000 / 2,000. The cue is therefore a function of the skeleton: I(cue; values | skeleton) = 0 [PROVEN].
  - A corollary used in §7: a STATE-SWAP between same-skeleton pairs keeps the cue by construction.
- **C3 best cue lookup:** (name, t), (unit, t), (name, unit, t) → majority v_t on 9,000 held-out rows gives 0.097 / 0.107 / 0.095 vs majority 0.095 (bound 0.125).
- **Build assert (G-CUE in the module):** the cue function takes only (names, unit, d, t). Every think input is asserted to contain no answer token past the text.

**CUE-ONLY control (pre-stated, binding guard at every pilot):** on TEST2 at the pilot's depths, every pass's think input is `[EOT] + enc(cue_t)` alone (no item text); the decoder is unchanged.
- Read: pass-1 and pass-2 aux value acc (teacher-forced) and the answer acc.
- **Each must be ≤ majority + 0.05 (≈ 0.16).** Any above → **LEAK**: the pilot is VOID whatever the other gates say, and the build is audited.
- OC: false alarm 0.000 at the majority; power 1.000 at 0.25 (st_oc).

## 5. Comparators (c): CUED-DIRECT is binding; plain DIRECT is descriptive
- **Is naming the target a lookup that one pass with the same cue could solve?** Not a lookup (D4). But the cue hands over the plan, and if finding the plan is what breaks one pass at d = 2, a cued one pass could solve d = 2. So **CUED-DIRECT** (`st_pcd2_s1`), the plain trunk on `dec = [EOT] + enc(" " + text + Σ_{t=1..d} " " + QUESTION(u, S_t)) + enc(" The answer is:")`, gets every cue LAT-ST sees, in order, in one pass.
  - It is the binding comparator for LP3 / LP4 and Stage 1's R1, and the source of G-W0's band. R1 then measures the loop, not the plan.
  - CD d = 1 < 0.80 at the pilot → flag CD-UNDERTRAINED (s228_pdir2c_s1 reached only d = 1 0.623 in 8k). LP3 is still read; Stage 1's G-W0 guard (CD d = 1 ≥ 0.80) decides later.
- **Plain DIRECT (descriptive):** reuse **s228_pdir2c_s1 iff the pilot host is an RTX 3090** (same class; one-GPU-class rule).
  - The new code lives in a new module (§9), so boxes/inject_ow.py stays byte-identical: s228_pdir2c_s1's module sha (249c7376…) is unchanged, and its items, d-sequence and ihash are the pilot's [DERIVED: same EXCL / stream; the st_ask_local smoke reproduced excl_n 160,598 and test2_sha 2a51c14b…].
  - Otherwise run a fresh `st_pdir2_s1` on the pilot host (+$0.05).
- **Exposure:** all cells share seed 1, the d law and the stream, so ihash is identical per 1k bin (VOID otherwise, as score_lat).

## 6. Sequence, gates and readings (d)
### 6.1 $0, before any spend
1. **Done (this draft):** G-CUE C1–C3 PASS; C6 PASS; ZS (§0).
2. **P1-ASK (local RTX 4060, ≈ 30 min, $0; Experiments):** `st_ask_local.py --d_max 2 --steps 8000` is §228's P-DIR2 with every input given the S_1 cue and the target v_1 (any d).
   - **PASS iff TEST2 acc(d = 2) ≥ 0.70 (target v_1), read only if acc(d = 1) ≥ 0.80.**
   - d = 1 < 0.80 → UNDERTRAINED → one fresh 16k run, same rule.
   - **FAIL → no cloud spend;** the told-pass premise fails at its first step (A1), and that goes to the user.
   - TEST2 d 3–8 descriptive (one hop with 1–6 downstream sentences).
   - OC: 1.000 if learned (0.95 / 0.92), 0.000 if not (0.15), 0.52 at the bar.
3. **Build packet (Experiments; Audit G-D):**
   - G1: inject_st `--cue none` bitwise = inject_ow, both arms, 50 steps on the 4060.
   - G2: the per-distinct-cue cache in fp64 vs uncached.
   - G3: per-term gradflow.
   - G4: a plain-path smoke.
   - G5: mem_check at the d = 2 / 4 / 8 shapes.
   - G6: args diff vs s228_plat2_s1 = {cue, cache mode, log key} only.
   - G7: CUE-ONLY on a random-init model reads chance.
   - G8: score_st fixtures and a field-path check against inject_st_eval.

### 6.2 Pilot P-2 (one cloud RTX 3090; d ~ U{1, 2}, 8k; cells st_plst2_s1, st_pcd2_s1, plain DIRECT per §5)
| gate | rule | role |
|---|---|---|
| LP0 (per arm, self-calibrated, REVIEW_LB #12) | each of LAT-ST and CD: no non-finite row; no 100-step mean gn_preclip after window 10 > 2 × max(windows 3–10). Replays clean on all four §228 / §229 pilot logs (0 windows); the fixed 10 / 38.1 lines are dropped because DIRECT's own windows reach 38.4 (s228_pdir2c_s1, window 2) | binding |
| LP1 | LAT-ST d = 1 train acc, last 1k bin ≥ 0.90 | binding |
| LP2 (on d = 2 items only) | `aux_vacc_by_d_pass["2"]["1"]` last 1k bin ≥ 0.80; d = 1 and pass 2 descriptive | binding |
| **LP3** | LAT-ST TEST2 d = 2 acc (K′ 8) ≥ 0.70 **and** ≥ CD d = 2 + 0.30 | **binding** |
| **LP5 USE** | MEAN-AT-1 on TEST2 d = 2: acc(intact) − acc(S_1 := the TEST2-d-2 mean of S_1; passes 2..8 own, with their own cues) ≥ 0.30 | **binding** |
| LEAK | CUE-ONLY (§4) | binding guard (VOID) |
| descriptive | K′1 on d = 2; pass-2 aux; CD d = 1; plain DIRECT d = 2; D-SWAP2: STATE-SWAP(1) on ≤ 1,000 kept d = 2 pairs drawn as st_cue_check C5 (stream 23244, redraw rng 23245; kept share 0.63, cf(a) = cf_b in all kept, 0 duplicate sigs in 200k d = 2 draws); outside the training EXCL, so the build asserts 0 sig overlap with a replay of the 8k training stream; follow / own / donor-final shares | — |

- **Outcomes:**
  - PASS = LP0–LP3 ∧ LP5 ∧ ¬LEAK → P-4.
  - LP3 fail → **ST-NOT-LEARNABLE-D2**, stop (no fallback inside ST).
  - LP3 pass with LP5 fail → **TOLD-RECOMPUTE**, stop: the gain is not serial use of the state.
  - Censoring: a dead or non-finite cell → UNREADABLE, and one re-run on the same class; a crash is resumed; a missing file reads pending.

### 6.3 Pilot P-4 (same host class; d ~ U{1..4}, 16k; LAT-ST + CD)
- **LP4 (binding):** LAT d = 3 ≥ 0.60 and d = 4 ≥ 0.50, each ≥ CD + 0.25.
- **LP5-4:** MEAN-AT-⌈d/2⌉ drop ≥ 0.30, pooled over d 3–4.
- LEAK on d 3–4.
- d ≥ 3 is where computed back-references (mul_s) first occur (§228 fix 3).

### 6.4 Stage 1 (pre-named; launched only on P-4 PASS, inside the family cap)
- **Design:** §228's Stage 1 (U{1..8}, 48k, one RTX 3090 class, identical d-sequence, TEST2 / PAIRS2 / floors as build_sets), with CD in DIRECT's place.
- **G-W0 band from CD:** CD ≤ 0.50 and not rising; guards CD d = 1 ≥ 0.80 and entry_d1; ≥ 3 depths, ≥ 2 of them ≥ 4. Plain DIRECT is descriptive (it shows the one-pass band with no plan).
- **R1:** acc_LAT − acc_CD ≥ 0.30 and acc_LAT ≥ 0.60, pooled over B.
- **R2:** §228's pooled MEAN-STATE (S := the TEST2 mean of S_8 at every pass); acc_LAT − acc_MEAN ≥ 0.30 and acc_MEAN ≤ acc_CD + 0.10.
- **R2′ MEAN-AT-t\* (new, binding):** on B depths ≥ 4, S_{t\*} := the per-d mean after pass t\* = ⌈d/2⌉; passes t\*+1..8 run on the item, with their own cues; acc_LAT − acc_MEANAT ≥ 0.30 pooled.
- **R3:** §228's STATE-SWAP at t\* on same-skeleton pairs (the cue is identical by C2), U = {cf(a), cf_b}, floor = max(redraw, permutation), N ≥ 2,000, follow_U ≥ 0.50 and ≥ floor + 0.30, SELF bitwise.
- **Labels:**
  - **C1-TOLD** = R1 ∧ R2 ∧ R2′ ∧ R3;
  - **TOLD-RECOMPUTE** = R1 ∧ (¬R2′ or swap own-share ≥ 0.50);
  - DEEP-NOT-SERIAL (R1 ∧ R2 ∧ R2′ ∧ ¬R3, with sub-labels RELAY / OWN as §228);
  - **CUE-ONE-PASS** (CD leaves < 3 band depths: the plan alone makes one pass suffice → UNREAD, reported as a finding);
  - NO-GAIN (¬R1).
- **Descriptive cue readings:** CUE-SKIP (at pass t\*+1 the cue names S_{t\*+2}; a told-serial model's answer moves to the predictable skip counterfactual) and CUE-FROZEN (the S_1 cue at every pass).
- A second seed only on C1-TOLD, with separate approval.

## 7. Does R3 still mean serial use when each pass is told what to compute? (the separation)
- **The worry:** pass t, told S_t, could compute v_t from the question and ignore the state.
- **Bound [DERIVED, F5]:** for t ≥ 2 that needs t ops in one pass. CD, the same trunk with the same tokens in one pass, measures exactly that ability. Where CD is at chance (the band), per-pass recompute of v_t (t ≥ 2) has no one-pass route.
- **Pre-stated readings that separate the routes:**
  - STATE-SWAP with the cue kept: automatic, since same-skeleton donors have identical cues (C2).
  - MEAN-AT-t (state removed at one pass, cue kept).
  - Pooled MEAN-STATE.

| truth | LP3 / R1 | LP5 / R2′ (MEAN-AT-t) | R3 swap (cue kept) | label |
|---|---|---|---|---|
| told-serial (pass t reads S_{t−1}) | pass | drop ≈ acc − 0.1 → pass | follows U | **C1-TOLD** |
| per-pass recompute from question + cue | pass only if CD also can → R1 fails | no drop → fail | own | TOLD-RECOMPUTE / NO-GAIN |
| plan alone suffices one pass | fail (CD high) | – | – | CUE-ONE-PASS / NO-GAIN |
| relay (pass 1 computes all, later passes copy) | needs pass 1 to do d ops (F2 says it doesn't) | drop at t\* only if the copy is read | donor-final | DEEP-NOT-SERIAL (RELAY) |
| pause / scratch (state used, not as values) | pass | drop | own or scattered | DEEP-NOT-SERIAL |
| pass-1-only (stores v_1, nothing after) | fail (LP3) | – | – | ST-NOT-LEARNABLE-D2 |

- The decoder reading v_{d−1} and doing the last op itself still counts as serial use (the state carries the intermediate), and R3 follows U in that case.

## 8. OC and cost (st_oc.json; one seed; 20,000 reps)
**P(pass | truth):**

| gate / truth | no effect | strong serial | marginal | at bars | recompute | cue solves one pass | pass-1 only |
|---|---|---|---|---|---|---|---|
| P-2 PASS (LP2 ∧ LP3 ∧ LP5 ∧ ¬LEAK) | **0.000** | **1.000** | 0.877 | 0.104 | **0.000** | **0.000** | **0.000** |

- P1-ASK: 0.000 (no told) / 1.000 (learned) / 0.52 (at bar).
- LP4: 0.000 (no effect) / 1.000 (strong) / 0.157 (at bars).
- R2′ (N 3,000): 1.000 (serial, drop 0.55) / 0.000 (recompute, drop 0.03) / 0.51 (at bar).
- R1 / R2 / R3: §228's OC (lat_checks E) carries over unchanged (same N, rules and floors). The comparator changes; the sampling does not.
- LEAK: false alarm 0.000, power at 0.25 = 1.000.

**Cost** (measured inputs: s228_plat2_s1 5,740 s and s229_plat2c_s1 7,398 s train at d ≤ 2 on a 3090; s228_pdir2c_s1 617 s; LAT eval_acc 411 s; realised $0.58 / $0.54 → $0.20–0.27 per host-hour):
- Work factors [ASSUMED token-linear]: LAT-ST vs §228 LAT ×1.18 (d ≤ 2), ×1.64 (d ≤ 4), ×2.76 (d ≤ 8); CD ×1.24 (d ≤ 2), ×2.58 (d ≤ 8).

| stage | host-h | $ (low–high) | guard ×1.3 |
|---|---|---|---|
| P1-ASK (local 4060) | ≈ 0.5 | 0 | 0 |
| P-2 (LAT-ST + CD; + fresh DIRECT only if not a 3090) | 2.6–3.1 | 0.52–0.84 (+0.05) | **1.16** |
| P-4 | 6.5–8.0 | 1.31–2.17 | 2.82 |
| Stage 1 (LAT-ST + CD + plain DIRECT, 48k; eval 3 h assumed) | 34.5–42.1 | 6.90–11.37 | 14.78 |
| **family cap (guard)** | | | **$18.76 → cap $19** (Audit-approvable under the $30 rule; one design, not split) |

- Stage-1 LAT-ST is ≈ 30–36 h of wall clock on one 3090. Off-host full-state checkpoints and a tested resume are required (the 10-15 rule); guard marks run to the final step.
- **Priors [ASSUMED]:** P1-ASK 0.5 (lowered from 0.6 after F4: no name-reading prior in the trained trunks); P-2 | ASK 0.35 (§0 WEAK rule); P-4 | P-2 0.6; C1-TOLD | P-4 0.4 → **P(C1-TOLD) ≈ 0.04**. **Expected spend ≈ $2.0** (at the high rate).

## 9. Code (no existing file edited)
- `boxes/inject_st.py` (new): imports inject_ow read-only; `InjectST(InjectOW)` overrides think_inputs per pass (a list of d distinct inputs, padded and masked), the per-distinct-cue lower cache, `--cue {none, step, all}` (all = CD's decoder input), and the `aux_vacc_by_d_pass` key.
- `boxes/inject_st_eval.py` (new): acc (K′ 0..14), mean_s (+ MEAN-AT-t), cue_only, swaps (+ D-SWAP2), decode. Every per-item output is stored.
- `theory/score_st.py` (new): LP0–LP5, LEAK, P-4, Stage 1, with fixtures (told-serial, recompute, cue-one-pass, leak, relay) before any output.
- inject_ow.py, inject_ow_eval.py, deltamem.py and nested170.py stay byte-identical. That keeps s228_pdir2c_s1 valid and avoids lazy-import skew for live jobs.

## 10. Novelty sentences for Literature (no "new" / "first" before the check)
- **N1:** "Each think pass of a memory-state latent loop on a pretrained LM (INJECT / DeltaMem, GPT-2 small) is told by name which intermediate quantity to compute, never its value, and a cue-matched one-pass comparator (CUED-DIRECT, the same tokens in one pass) credits the depth gain to the loop rather than to the plan."
  - Nearest known (ours): planning tokens 2310.05707 (discrete step-type tokens in text CoT); Universal Transformers 1807.03819 (per-step timestep embeddings: an index, not a target); Shapiro 2608.11233 (loop retrofit).
  - To locate: least-to-most / successive / decomposed prompting and self-ask (subquestions in text); any latent or looped model with per-step target cues.
- **N2:** "Diagnosis: a looped latent arm on word programs fails because no pass is ever asked for the intermediate. v_1 is at chance at the write source on d = 2 while the given value decodes at 0.988, and trained trunks ignore the question's name." Related: Biran 2406.12775 ("Hopping Too Late"); Yang 2402.16837 (latent multi-hop).
- **N3:** "MEAN-AT-t (state removed at one pass, cue kept) plus same-skeleton state swaps (cue identical by construction) as the readings that separate serial use of a told state from per-pass recompute."

## 11. Open questions for the red team
1. Is P1-ASK the right $0 gate for A1? It is one pass with no memory or aux, so it could fail where LAT-ST (aux + K passes) would learn, or pass while pass t ≥ 2's read-into-compute (A2, untested on language) still fails.
2. Is CUED-DIRECT information-matched? It sees all cues at once and in order; LAT sees one per pass, and its decoder sees no cue. Would a CD with only the S_1 cue be the stricter control at d = 2?
3. Does C1-TOLD satisfy the north-star C1 ("step-specific blocking shows serial use") when the plan is given? What is the shortest path to a self-cued arm (the aux decoder emitting the next name as a discrete token), and does that reintroduce text-CoT (C3 / C4)?
4. With the cue repeated for t > d, the pass sequence reveals d. Can a model exploit "the cue stopped changing" (hold) in a way that games R2′ or R3?
5. Are LP2's last-bin clustering (495 d = 2 steps × 64 items, N_eff 495 assumed) and LP5's pairing modelled conservatively enough?
6. Is the per-distinct-cue lower cache exact under right-padding with variable text length? (The cue sits at different absolute positions per item; the cache key is the whole input.)
7. APP puts two questions in the input. Could a REPL-style input be learned faster? The §0 rule lets ZS decide only on a ≥ 0.15 margin in both models.
8. F4 shows the trunks never read names. Does that make A1 a learning problem the 8k pilot is too short for (§227: text needed ≈ 6k post-introduction steps for d = 2)? Should P-2 run 12k?

## 12. Main risks
1. **A1 fails** (P1-ASK): one pass cannot learn name-conditioned one-hop computation with downstream sentences present. F4 shows no prior for it.
2. **A2 fails on language:** pass t ≥ 2 cannot use v_{t−1} from the block-6 read (untested; §228's d = 1 bypassed the memory). This would show as LP2 pass with pass-2 aux and LP3 fail.
3. **The plan alone suffices one pass** (CD solves d = 2): LP3 fails as written. That would be informative (the plan was the one-pass bottleneck), but it ends ST at d = 2.
4. **Recompute:** caught by LP5 / R2′ and R3's own-share. It costs P-2 or Stage-1 spend before it is caught.
5. **Scope:** C1-TOLD is narrower than C1 (plan given) and leaves C4 open (a test-time generator scaffold). A pass must be worded with the plan disclosed.
6. **Build:** per-pass inputs, the cache, three eval parts and a new scorer. All are in new modules with an equivalence gate (G1). Stage 1 needs ≈ 35 h of single-host wall clock.

## Files (this folder)
- `ST_draft.md` (this file).
- `st_cue_check.py` / `.json`: G-CUE C1–C3, C4 lengths, C5 d = 2 pairs, C6 chain.
- `st_told_zs.py` / `.json`: zero-shot TOLD check on the closed §228 models.
- `st_oc.py` / `.json`: OC + cost.
- `st_ask_local.py`: the P1-ASK local wrapper (GPU; not run).
