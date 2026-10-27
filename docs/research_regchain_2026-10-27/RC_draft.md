# REG-CHAIN draft: a C1 chaining test where no pass position predicts its op (2026-10-27)

**Status:** draft for red team + Literature. Chosen by Theory under the user's standing rules (relayed by Audit 10-26: auto-select, no approval questions); C1 first.

## 1. Why
The OW generator (boxes/ow_words.py `_draw_program`, dag=True) forces one cross-state product mul_s at ts ~ U{3..d}. Its consequences so far:
- d 3's step 3 is always (mul_s, src 1). At d 4 step 3 is mul_s about 46% of the time.
- §234 P-4 s1: pass 3 learned "S_2·S_1" (non-mul_s pass 3 = 0.103; Note 234-7).
- §234 P-4 s2: pass 2 merges the fixed final step (prints v_3 0.783 at d 3; Note 234-9).
- §235: decorrelating d 3 only moved the prior (mul_s at d 4 pass 3 0.998 → 0.193; Note 235-1).
- All three are the same confound. The test sets carry it too: TEST2 d 3 is all mul_s, so no test on them can show op-reading at the last step.
- A valid C1 chaining test on this generator needs programs where the op at every pass is independent of the pass index, in training and test alike.

## 2. Design (single change vs P-4: the program law, train + test)
- **REG law:** OW.Stream kwarg `mul_s_p` (default 1.0 = current). At 0.0 no program has the forced mul_s. Every step's op is drawn iid from OPS_REG (add_c, sub_c, mul_c, add_v, sub_v, mul_v, add_s), so the op is independent of t and d.
  - add_s keeps state references to an older state (j ≤ t − 2), so memory addressing is still exercised.
  - At 0.0 the ts draw is skipped. At 1.0 the stream is bitwise current (flags-off gate).
- **Training:** LAT-AW and CD-AN on REG, d ~ U{1..4}, 16k, γ = 1, askname init, one RTX 3090 per run, off-host ckpts every 1k. Args diff vs aw_plaw4_s1 = {mul_s_p 0.0} + seed + name.
- **Seeds:** LAT-AW s1 and s2 (seed variance is real: s1 vs s2 took different routes at d 3); CD-AN s1.
- **Eval:** P-4's parts (acc, lp5, mean_s, cue_only, cue_wrong, swap_k, swap_1, probe, ws, lp5h, swap_kd) on REG sets with new seeds.
  - TEST2-REG d 1..12 (×1,000), pairs and floors by the same laws, all built with mul_s_p 0.0.
  - Exclusion: training REG items exclude the REG test sigs.
  - The ws items carry sig; the scorer regenerates ops.

## 3. Derivation (tagged)
- [A1] Pass t's ':' output is argmax p(v | text, cue_t, read(S_{t−1})).
- [A2] Under REG, I(op_t; t, d) = 0 by construction [exact]. So no function of pass position alone beats 1/7 on op_t, and value accuracy at pass t ≤ P(op read correctly) + chance terms.
- [C1, derived] A pass-position shortcut cannot pass CHAIN at any pass on REG. That removes the confound behind 234-7, 234-9 and 235-1.
  - Merging steps (pass t printing v_{t+1}) needs op_{t+1} from the text, which is legitimate reading. CHAIN still reads v_t at pass t, so a merged route fails CHAIN visibly.
- [C2, derived] H-POS (the failures were position priors): REG training gives pass 3 / 4 the same op coverage as pass 2. Prediction: per-op WS at passes 3–4 is near pass 2's (≥ 0.9 on every op at d 3; ≥ 0.5 at d 4 pass 4).
- [C3] H-CAP (passes ≥ 3 can't read op + state): per-op WS at pass 3 stays ≤ 0.5 even on REG.
- [Measured prior] P-4 s1/s2 passes 1–2 are ≥ 0.98 on every regular op, so the readout works where coverage was even.

## 4. Readings (binding: PASS needs both LAT seeds)
- LP0 each arm.
- **ACC:** LAT TEST2-REG d 3 ≥ 0.60 and d 4 ≥ 0.50, each ≥ CD-AN + 0.25.
- **CHAIN:** WS-READOUT ≥ 0.50 at every pass 2..d, d 3 and d 4.
- **OP-ALL:** per-op WS at pass 3, pooled over d 3–4, ≥ 0.50 for every op with n ≥ 50 (no op left behind).
- **LP5-4:** the mid-state mean drop pooled over d 3–4 ≥ 0.30.
- **SWAP-MID d 3** (t 1): follow_U ≥ 0.50 and ≥ floor + 0.30.
- **CUE-ONLY guard** on d 3–4, else VOID.
- **Labels:**
  - per seed: CHAIN-ALL / MERGED (pass t prints v_{t+1} ≥ 0.5 at some pass) / CAP (OP-ALL fails);
  - overall: PASS (both seeds) / SPLIT (one seed) / FAIL.
- **Descriptive:**
  - d 5–6 accuracy and WS (C2 extrapolation beyond the trained depth);
  - SWAP-MID d 4;
  - the substitution table per pass;
  - CUE-WRONG; the aux staircase.
- **Status mapping:**
  - PASS → "C1, scaffolded (plan given), d ≤ 4, 2 seeds, op-unconfounded";
  - SPLIT → "1 of 2 seeds";
  - FAIL with CAP → the op-target aux design (pre-named next; not run here).

## 5. OC
| World | Per seed | Both seeds |
|---|---|---|
| H-POS | ~0.7 | ~0.5 |
| H-CAP | ~0.01 | — |

- Prior H-POS ~0.7, so P(PASS) ~0.35 and P(at least SPLIT) ~0.6.

## 6. Censoring
As P-4 (crash → resume + dedupe; missing parts → pending, FAIL at --final).

## 7. Cost
- P-4 realised $0.97 for LAT + CD on one host, so three runs ≈ $1.5. Admit ≤ $2.0, guard $6.3.
- Running: realised so far (~$4.9 incl. seed 2) + $6.3 ≈ $11.2 ≤ $30.
- Build: the kwarg + flags-off gate; REG eval sets + exclusion + set shas; P-4 eval parts on REG (no code change beyond set construction); a d-share and op-share check on 50k training draws (each op ≈ 1/7 at every t).
