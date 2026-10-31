# UNSCAFFOLDED v1: can training on permuted statement order make the loop look its steps up (by name, then without names)? (C1 scope) (2026-10-30)

**Status:** v1 after REVIEW_NS (GO-with-fixes, all adopted) and the $0 stage (Note 251-16). Supersedes NS_draft.md. The scorer theory/score_ns.py is written before any cloud output.

## 1. Why
- **Note 251-16:** the C1 loops (rcb_plaw4_s1 / s2) compute by sentence position.
  - With the op sentences permuted against the dependencies (TEST-PERM), d 2–4 fall to 0.15–0.20, and even pass 1 falls to 0.10, although its cue names S_1.
  - A probe reads the pass index from S_t at 0.98.
  - So "plan given" understated the scaffold: the text is also in chain order, and every C1 claim depends on that.
- **Question (north star C1; GSM8K text is not in solution order):** does training with permuted order make the loop locate each step's sentence?
  - **NP:** by the entity the cue names.
  - **GP:** with a generic cue, where it must also decide which entity comes next.

## 2. Arms (warm start; REG law; d ~ U{1..4}; K 8; 16k final, no extension; RTX 3090; micro pinned; per-pass aux unchanged: a C1 test, not C4)
| Arm | Runs | Init | Change vs rcb_plaw4 args |
|---|---|---|---|
| NP (named, permuted training) | np_s1 / np_s2 | rcb_plaw4_s1 / s2 | `--perm_p 0.5` |
| GP (generic cue, permuted training) | gp_s1 / gp_s2 | rcb_plaw4_s1 / s2 | `--perm_p 0.5 --cue generic` |

- **`--perm_p 0.5`:** each training item is rendered with its op sentences in a dependency-violating order (≥ 1 inversion for d ≥ 2, seeded per item) with probability 0.5, else in chain order.
  - Givens and distractors keep their positions.
  - OW.solve(rendered) == answer is asserted.
  - The 0.5 mix keeps chain-order competence (TEST2), so the args diff is one flag for NP.
- **Generic cue:** " What is the next value? The answer is:" (6 + 4 tokens) at every pass. The decoder is unchanged (full text + the final question).
- **Disclosed:** the aux target "<S_t> has <v_t> <u>." supervises step t's entity in training, so GP is "no plan input at test", not "no plan supervision".
- **Comparators:**
  - rcb_plaw4_s1 / s2 on TEST-PERM: 0.15–0.20 (Note 251-16; step-0 reference for both arms);
  - GENERIC-0 (step-0 reference for GP);
  - CD-AN.

## 3. G-D and pilot (before cloud)
- **G-D:**
  - default-off bitwise;
  - the PLAN-FREE assert inside `passes()` (GP: identical cue tails at every pass, except deliberate interventions);
  - every eval builder threaded (cue_only / cue_wrong / c7 / read_mask / transfer use ST.cue_str today; REVIEW_NS #3);
  - CUE_LEN for the generic cue;
  - the perm renderer solve assert on 10k items;
  - gradflow;
  - args diff.
- **$0 local pilot** (one card, 2k steps, seed 1, both arms; a P-CTRL without flags is not needed, the step-0 values are known). Read the train-batch ws_readout d 2 pass 1 on permuted items at 1k / 2k.
  - **Rule per arm:** GO to cloud if ≥ 0.50 at 2k (step 0: 0.10); PARK the arm if ≤ 0.20; otherwise GO for NP only.

## 4. Readings (per seed; TEST-PERM = TEST2 programs re-rendered as in 251-15, driver 622fb717)
**Binding, on TEST-PERM:**
- ACC d 2–4 ≥ 0.90 at K′ 8.
- NOT-DECODER: d 4 at K′ 2 ≤ 0.20.
- LP5-D4: mean state at t 2 on TEST-PERM d 4 items, drop ≥ 0.30.

**Binding, on TEST2:**
- RETAIN: d 1–4 ≥ 0.98.
- SWAP4 (SWAP-MID d 4 t 2, chain-order pairs) ≥ 0.50 and ≥ floor + 0.30.

**Guards:**
- STABLE.
- NP: CUE-ONLY (named).
- GP: CUE-ONLY is vacuous, so it is replaced by NOT-DECODER, and cue_wrong is dropped.

**Labels (first match):**
- VOID / UNSTABLE;
- NOT-FORMED (TEST2 d 2 < 0.90 at the end);
- LOOKUP (NP) / PLAN-FREE-LOOKUP (GP): all binding pass;
- ORDER-ONLY: RETAIN passes, TEST-PERM ACC fails;
- PARTIAL: TEST-PERM d 2 ≥ 0.90, d 3–4 fail.
- Per arm overall: PASS (both seeds) / SPLIT / FAIL.

**Descriptive:**
- the add_s-partner-ambiguity stratum (0.076 / 0.177 at d 3 / 4);
- WS per pass on TEST-PERM;
- the S_t probe for spine identity after training;
- K′ = d and 8;
- the learning curve on permuted batches;
- STILL-RISING if TEST-PERM train acc rises over the last 2k (descriptive only; the verdict stands at 16k).

## 5. OC (per seed)
| Arm | P(pass \| the loop can learn sentence lookup) | P(pass \| it stays position-bound) |
|---|---|---|
| NP | 0.5 | 0.03 |
| GP | 0.3 | 0.02 |

- **Rationale:** a named cue gives a lookup key that attention can match against the subject name of the defining sentence; GP must also infer the next entity from S.
- **Censoring:**
  - an unfinished run or a crashed eval is VOID-RERUN once (fresh host; --keep_final_ckpt for re-eval);
  - a pilot-PARKed arm is not run.

## 6. Cost
- Up to 4 × 16k (§251 cells ~$0.7–1.15) ≈ $3.5–4.6; guard $5. Credit $25.7.
- **Build:** --perm_p (training renderer), --cue generic (tok paths + eval builders), the TEST-PERM eval part (from driver 622fb717 into inject_aw_eval), the PLAN-FREE assert.
