# 02 RED TEAM: N1 SHUFFLED-OBJECTS ("loop is necessary"), an independent review (2026-11-04)

Reviewer: an independent red-team sub-agent working read-only. It ran no training and read no stored outputs; the only file it read under `boxes/runs_fsq/` was `bun_s1/args.json`, for the recipe. Its CPU checks are pure-Python simulations of the task's own generator law, kept in the scratchpad (`so_base.py`, `swap.py`, `oc.py`). Target: `02_n1_s5_loop_necessary_task.md` (abbreviated "the design" below; § numbers refer to it unless THEORY is named).

**Verdict: GO-WITH-FIXES.** Four HIGH fixes (F1-F4) must be in place before registration. As written, the binding expressivity row X2 passes at chance, the C1 swap battery keeps zero pairs, the C2 cells mix depth with labels the model never saw, and the "one pass fails" comparison does not start the two arms from the same weights. All four fixes cost $0 or close to it. The task is worth running: it is the cheapest public depth split this project can reach.

## Checks that passed (the design's claims hold)

- k_touch ~ Binomial(m, 2/n) (**PROVED**): the swap pair is drawn independently of the state, and P(S_{t-1} ∈ {a_t, b_t}) = 2/n. My simulation matches. Hold floor P(S_m = S_0) = 0.333 / 0.25 / 0.20 at n 3 / 4 / 5 for m 8.
- DeltaMem has β = 2σ(·) ∈ (0, 2) at `deltamem.py` 98 and 188, so the eigenvalue α(1 − β) ∈ (−α, α) along k (header line 5). The design's statement is correct.
- The §262 name-token assert applies only when `cue == "step"` (`inject_aw.py` 91-92). `cue_inputs` asserts that all passes share one padding (`inject_st.py` 77).
- In GPT-2, " 1" to " 12" are one token each (352 … 1105). The first BPE tokens of the 7 BBH names are distinct (" Gertrude" → 402 / 861 / 81 / 2507; first token " G").
- The Barrington citation is right: a width-5 permutation BP decides its output by where one point goes, so tracking a single holder in S5 words is NC¹-hard. Status: **DERIVED-UNDER-ASSUMPTIONS** (TC⁰ ≠ NC¹, log-precision transformer ⊆ TC⁰, worst case as m → ∞). See F8 for what that does and does not license.

## Findings

### F1 HIGH: X2 (the "solvable-group asymmetry") passes at chance level

Evidence (simulation, 400k draws per cell, `so_base.py`). On the MOVED stratum (S_m ≠ S_0, k_touch ≥ 2), any strategy that just avoids the initial holder scores 1/(n−1): **0.50 at n 3 and 0.25 at n 5**. That is a 0.25 gap with no tracking at all, and the X2 bar is 0.15. A bag-of-swaps MAP (order-free, so no state tracking) scores 0.539 / 0.325 at m 8. Uniform guessing over n gives 0.333 / 0.20, still a 0.13 gap. `oc.py` puts P(X2 passes in ≥ 2/3 seeds | DIRECT exactly at chance) at **0.99997** (0.995 for the bag-MAP floor). X2 therefore cannot carry the expressivity reading. The §6 sentence that the n 3 vs n 5 contrast "is the expressivity reading" is false as scored. The `floor_major` = 1/n in §6 is also the wrong floor for MOVED cells.

Fix:
- Score X1 and X2 on floor-normalised accuracy, κ = (acc − f_n,m)/(1 − f_n,m).
- f_n,m = the maximum, over a pre-stated list of shortcut baselines, of each baseline's accuracy on that cell's own test items. Compute it at build time ($0) and print it in the registration. The list: non-initial uniform, bag-of-swaps MAP, last-2-swaps MAP, partner of the last swap that involves S_0, most frequent name, and truncated tracking S_4 and S_{m−1}.
- X2 becomes κ(n 3) − κ(n 5) ≥ 0.15 in ≥ 2/3 seeds. Also require κ(n 3) ≥ 0.15 with its Wilson lower bound > 0: DIRECT must actually beat the n 3 floor, or the asymmetry is between two chance levels.
- Use 1/(n−1) or f_n,m wherever MOVED cells name a chance floor.

### F2 HIGH: the `swap_so` pair law is empty, and its donor cannot separate "carry and update" from "donor look-ahead"

Evidence. §7 makes the donor the recipient's text with another asked object o′ and the **same swaps**. Then cf(t), the recipient's swaps t+1..m applied to the donor's S_t, is the donor's own continuation, so **cf(t) = donor-final identically**. The keep rule "own ≠ cf(t) ≠ donor-final" therefore keeps 0 pairs. Simulation: kept 0.0000 at (n 5, m 8, t 4), (5, 4, 2), (3, 4, 2) and (5, 6, 3). The code already records exactly this degeneracy for OW at d 3, t 2 (`inject_aw_eval.py` 365-367: "cf_a(2) = donor final: 0 kept"). M2 and M3 would come back n_pairs 0, and M3 is the precondition for reading C2-1. Without the keep rule the problem is worse: a model whose memory at pass t already holds the donor's final answer (a look-ahead or buffer reading, cf. THEORY §162 BUFFER-LIKE) predicts the same token as genuine serial carrying.

Fix:
- Donor = the same people, objects, h_0 and o*, with an **independently redrawn swap sequence of the same length and the same frame indices**. Names are single tokens, so the token length is equal. Because the SI cue depends only on (t, o*), the donor's and recipient's cues are identical, and the existing equal-cue assert can be kept.
- Keep a pair iff cf ∉ {own, donor-final, donor S_t}. Kept shares are 0.485 (n 5, m 8, t 4), 0.382 (5, 4, 2) and 0.297 (3, 4, 2), so 1,000 pairs need at most about 3,400 tries.
- Report follow_cf, stay_own, follow_dfin, follow_stale (donor S_t) and the MEAN floor.
- G0 asserts n_pairs = 1,000 in every swap cell.

### F3 HIGH: C2-1 and X1 mix depth with swap labels the model has never seen; the design's fix is optional

Evidence. Training has m ≤ 4, so the NUM text carries only "Swap 1-4" and the SI cue only " 1".." 4". At test, passes 5-12 must match the cue's " 8" to the text's "Swap 8:". bun saw " 5".." 10" only as arithmetic values (`NUM_IDS` = " 0".." 10", `nested170.py` 20) and never saw " 11" / " 12" at all. A loop failure at m 6 / 8 is therefore ambiguous: lookup or loop? A DIRECT failure (X1) is ambiguous in the same way. The design leans on X2 to separate the two (§6 "X1 is read only with X2"), but X2 is void (F1). The offset-window remedy sits in §10 confound 1 only as a conditional S2 add-on. ORD and the BBH-five cells need counting past 4 as well: B1 is m 5 > trained 4.

Fix:
- Make **offset labels** standard for every arm's training stream: swap labels are a consecutive window [o+1, o+m] with o ~ U{0..12−m}. Then every label 1-12 is seen at training depth ≤ 4, and the pass index is no longer tied to the label number. The cue names the label of the t-th swap.
- Add a $0, eval-only LOOKUP probe at m 8: a one-pass question "Who trades in swap L?" on the frozen model, L = 1..12. It shows whether label lookup itself extrapolates.
- Make B1 binding only together with C2-3, since five-objects needs count-to-5.

### F4 HIGH: DIRECT, PAUSE and TEXT-CoT do not start from the same weights as the loop arms

Evidence. The LOOP arms start from `bun_s1/model.pt`, the end of a lineage of 68k+ steps of multi-hop OW training (THEORY §268 row; `bun_s1/args.json` init_model dch_s1). DIRECT, PAUSE-12 and TEXT-CoT start from the s233 trunk. "Matched compute = same items, steps, batch, base" (§3) is not true of init. If X1 passes, it may be the init gap and not the missing loop that holds DIRECT back.

Fix:
- Add **DIRECT-BUN**: bun_s1's GPT-2 weights with the DeltaMem read off (S ≡ 0 or the branch removed). Same data and 8k steps, 3 seeds, about $0.25 each. This is the binding X1 / X2 arm. DIRECT-s233 becomes descriptive.
- Report the loop's own K′ 0 and K′ 1 at m 8. They are free, and they are the NORTH_STAR C1 "K′ = 0 / 1 far below K" rows.

### F5 MED: the aux value index points at " the", not at the holder

Evidence. `tok_ow` sets `vidx = q.index(HAS_TOK) + 1` (`inject_ow.py` 70; HAS_TOK = " has", line 42). In the design's aux sentence "After swap t, <S_t> has the <o*>.", the token after " has" is " the". The "partners" and "positions" families ("is dancing with", "is playing") contain no " has" at all, so `.index` raises. vidx feeds the training-log aux accuracy (`inject_aw.py` 400-402) and the eval aux part (`inject_aw_eval.py` 77). Under the gate-on-mechanism rule, a pilot that reads aux accuracy would see ≈ 1.0 for " the" and pass falsely.

Fix: `tok_so` sets vidx to the position of S_t's name token, asserted unique in the sentence for every family. G0 asserts that `dec(aux[vidx]) == " " + S_t` on 20k items.

### F6 MED: the S1 pilot gate checks accuracy only

Evidence. §3 S1 gates on NUM n 5 accuracy at m ≤ 4 and m 5. Confound 3 (§10) says each SI pass input is a standalone question that one pass can answer at t ≤ 4. A per-pass-DIRECT model passes this gate while its memory goes unused (cf. §253, where accuracy passed and pass 2 never formed).

Fix: the GO gate also needs M3 in range (fixed law F2, follow_cf ≥ 0.50 and ≥ MEAN + 0.30 at m 4, t 2), M4 (K′ 2) on the S_2 ≠ S_4 stratum (bar ≤ 0.05; the all-items floor is about 0.40, which weakens the test), and WS at passes 2-4 ≥ 0.90.

### F7 MED: the OC block lists priors only; it has no P(pass | no effect) or P(pass | effect), and two binding rows are noisy

Evidence (`oc.py`; MOVED counts of about 353 at n 5, m 8 and 301 at m 6; σ_seed 0.09; m 6 and m 8 noise drawn independently, which is pessimistic). P(C2-1 ≥ 2/3 seeds) by true m 8 mean: 0.001 at 0.70, 0.013 at 0.75, 0.12 at 0.80, 0.43 at 0.85, 0.79 at 0.90, 0.96 at 0.95. The false-pass rate is low, but power is poor unless the effect is ≥ 0.90. M5's k_touch ≥ 6 stratum has P = 0.050 × 500 ≈ 25 items. With equal true accuracy (0.85) it fails falsely in 9.8% of seeds.

Fix:
- Put this table in the OC block, along with the X2 null row from F1.
- Pool M5 over NUM + ORD, or use k_touch ≥ 5 (about 87 items), or raise the cell to 1,500 items at m 8.
- Censoring: a seed that never reaches a cell (OOM after the one VOID-RERUN, or a missing eval file) counts as FAIL in the 2/3 count, not as excluded. State this explicitly.

### F8 MED: the expressivity claims are tagged too strongly

Evidence. For m ≤ 12, a 12-layer transformer can compose 12 permutations by an associative scan in ⌈log₂ 12⌉ = 4 levels, for S5 as well as S3. Barrington and Barrington-Thérien separate solvable from non-solvable groups only at constant depth as m → ∞ (2210.10749 gives O(log m) shortcuts for all semiautomata). A DIRECT failure at m 8 after training on m ≤ 4 is therefore a learning and length-generalisation fact, not a circuit-class fact. Hardness is also worst-case, and the test words are random transpositions. §1 states the finite-m caveat but keeps "the expressivity reading" for X2.

Fix: tag the §1 complexity argument DERIVED-UNDER-ASSUMPTIONS (asymptotic, worst-case) and the X2 prediction CONJECTURE (learning dynamics). Falsifier: X2 (F1 form) fails while X1 passes, which gives the N1-LOOP-ONLY label. Phrase the claim as "the loop extrapolates where one pass trained identically does not", not "one pass cannot".

### F9 MED: cost is understated about 2.5×

Evidence. `bun_s1/args.json` "hostprobe_sps": 0.6206 is the host benchmark's `sec_per_step` (`scratch/gen_vast_job.sh` 59-61, "b225 reference 0.657"), not bun's training step time. Bun-sized runs take about 1.6 s/step. K 12 vs K 8 and 60-110 vs about 70 tokens put N1 at about 2.4-3 s/step, so 8k steps ≈ 5.3-6.7 h ≈ $1.1-1.4 at $0.21/h, plus eval (36 cells × 500 × K′ 12, K′ 64 hold, 750 × ≤ 7 BBH options) of about $0.3.

Fix: per loop run ≈ $1.5, so S1 + S2 as listed ≈ $12-14 and a guard of $18. That is still under $30. Dropping FROM-TRUNK (F15) saves about $1.5.

### F10 MED: the C1 scope limit is inherited unchanged; the SI cue is a step counter on chain-ordered text

Evidence. The SI cue gives the pass index t, and the swaps are rendered in chain order, so swap t is the t-th swap sentence. This is the project's existing scope ("plan given + chain-ordered text"; THEORY §252 and §253 found the loops position-bound). SO has no hidden plan, so the cue leaks nothing about the answer. But N1 does not test content selection either, and a pass can be the one-hop lookup "t-th sentence vs memory pointer". EN removes even the lookup: the pass becomes a pure state update on given transitions.

Fix:
- Add eval-only **PERM-LABEL** cells: NUM text with swap sentences printed in a permuted order but labelled by their true index, so the cue's label must be found by content.
- Add an optional descriptive **GENERIC** arm: the same cue "Who has the o* now?" on every pass (the §260 analogue), where the model must count steps itself.
- State in the label that N1-PASS is scoped to chain-ordered text.

### F11 MED: no K′-curve for C2 and no K′ 0 / 1 rows for C1

Evidence. NORTH_STAR C2 asks for "accuracy rises with latent steps past the trained budget". The design fixes K′ = 12 for every m. Project rule (11-03): a K′-curve for C2. NORTH_STAR C1 asks for "K′ = 0 and K′ = 1 far below K"; M4 covers only m 4.

Fix: at n 5, m 8, MOVED, report acc at K′ ∈ {0, 1, 4, 6, 7, 8, 9, 12, 16}. Binding: acc(K′ 4) ≤ f + 0.10 and acc(K′ 8) − acc(K′ 4) ≥ 0.40. This also absorbs the F4 K′ 0 / 1 rows.

### F12 MED: BBH comparison

Evidence:
- (a) §2 says NAMES exclude BBH's names. But `NAMES` (`nested_words.py` 33) contains **"Eve" and "Bob"**, and both are BBH names. The claimed name transfer is only partial.
- (b) BBH inputs end with "At the end of the game, Bob has the\nOptions:\n(A) …". §7 never says whether the question and options are stripped before the SI cue is appended. Passes may see a question that does not match the cue.
- (c) BBH-five has m = 5, beyond the trained depth, and its ORD count goes past 4 (F3).
- (d) A model fine-tuned on matched synthetic data is not comparable to the few-shot BBH numbers in 2210.09261.

Fix:
- Remove Eve / Bob from training NAMES, or report BBH by name overlap.
- Pre-state the BBH rendering: strip the "At the end …" clause and the options, then append the SI cue.
- Label B1 "task-specific fine-tune transfer", never SOTA.
- The option-wise logit comparison of the same X token across runs is sound, because the token prior cancels. Keep it.

### F13 LOW: holding an involution; H1 cannot see parity

Evidence. The hold passes repeat the swap-m cue. The natural per-pass rule "apply swap c to the pointer" is an involution, so a model that fails to hold oscillates S_{m−1} ↔ S_m. K′ 64 vs K′ 12 adds 52 passes, an even number, so H1 cannot see an oscillating model. Answering S_{m−1} already scores 0.56 on MOVED at n 5, m 8.

Fix: H1 compares K′ 63 and K′ 64 with K′ 12. Report the S_{m−1} confusion rate per cell.

### F14 LOW: code-touch list is incomplete or misquoted

- The `NUM_IDS` sites in `inject_aw.py` loss_reg are 363 / 378 / 389 / 417 (and 424 on the ask path). The design's "385 / 416" is wrong.
- `inject_aw_eval.py` 768 hard-codes the probability width as 11, `np.zeros((N, K + 1, 11))`; it must be len(VAL_IDS).
- `tok_aw` → `tok_st` runs the G-CUE loop for any cue ≠ "none" (`inject_st.py` 55-57), and that loop reads `it["unit"]` / `names.spine`. `tok_so` must not route through `tok_st`. Say so, and have G5 cover it.
- The per-item restricted argmax over people_ids is new code at every readout site; list it in G1 / G5.

### F15 LOW: arm hygiene

- **FROM-TRUNK:** 8k steps is uninformative given the 68k-step formation path (the design admits this). Drop it.
- **BETA-POS:** replacing 2σ by σ halves every β at init, so bun_s1's function is perturbed from step 0. Use β = min(2σ(g), 1 − ε) so the function is preserved wherever β < 1, and run the arm only if E1 shows a non-trivial share of β > 1.
- **PAUSE-12 X3:** a binding row on 1 seed. Make it descriptive or give it 3 seeds.
- **Lineage scope:** all LOOP seeds share the bun_s1 init. Scope C2 as "bun_s1 lineage, 3 data-order seeds" (one-lineage rule until X7).

## Minimal staged plan I would approve

- **S0 ($0, CPU):**
  - Apply F1-F5 and F14.
  - G0 additions:
    - shortcut-floor table f_n,m per cell (F1);
    - swap-pair counts of 1,000 with the fixed law (F2);
    - an offset-label histogram showing every label 1-12 seen (F3);
    - vidx decode assert (F5);
    - decontamination with Eve / Bob handled.
  - Fixtures cover X2-at-chance (must FAIL), n_pairs 0 (must VOID) and parity oscillation (H1 must FAIL).
- **S1 (≈ $1.6, guard $3):**
  - Runs:
    - LOOP-SI seed 1 to 4k with offset labels: about 2.9 h ≈ $0.6, plus eval at about $0.2.
    - DIRECT-BUN × 3: about $0.75.
  - GO requires all four:
    - L1-pilot;
    - M3 in range (fixed law);
    - M4 on the S_2 ≠ S_4 stratum;
    - pass 2-4 WS ≥ 0.90.
  - DIRECT gate: κ-normalised (F1). If DIRECT-BUN κ(n 5, m 8) ≥ 0.70 in 2/3, the "necessary" framing is dead.
- **S2 (≈ $8-9, guard $12):** LOOP-SI × 3 (8k), TEXT-CoT × 1, LOOP-EN × 1, PAUSE-12 × 1 (descriptive), K′-curve and PERM-LABEL cells eval-only. GENERIC and BETA-POS run only if S2 has budget left and E1 or the PERM-LABEL cells call for them.
- **Total ≈ $10-11 (guard $15), under the $30 line.**
