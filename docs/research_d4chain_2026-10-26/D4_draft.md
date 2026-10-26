# D4-DECOR draft: decorrelate pass index from op in training (2026-10-26)

**Status:** draft for red team + Literature.
- User decision (10-26, via Audit): "Fix depth-4 chain", ~$2–5 pilot, same recipe.
- Prior: §234 P-4 FAIL (CHAIN), Note 234-7. $0 diagnostics: Audit c0943199 (scratch/audit_d4_chain/diag.json) + Theory (below).

## 1. Failure, from $0 data (aw_plaw4_s1, TEST2 d = 4)
- The chain breaks at **pass 3**, not pass 4 (Audit's locus probes):
  - v_2 is stored in S_2 (0.998) and read at pass 3's ':' (0.998);
  - v_3 is computed at pass 3 only about half the time (':' argmax 0.523; probe from u 0.420);
  - the write carries what was computed (S_3 holds v_3 at 0.453).
- **By op at step 3** (Audit, regenerated items, sigs match): pass 3 is right on (mul_s, src 1) at 0.999 (n 930 / 2000) and at chance on every other op. Passes 1–2 are 0.97–1.00 on every op.
- **Counterfactual check (Theory):** when pass 3 is wrong, its argmax equals S_2·S_1 mod 11 on 0.505 of items (n 507; chance ≈ 0.09). On those items pass 3 applies the forced cross-state product whatever the step's op.
- **Generator fact** (boxes/ow_words.py `_draw_program`, dag=True):
  - for d ≥ 3 exactly one step ts ~ U{3..d} is the forced mul_s, with partner j ≤ ts − 2;
  - so at d = 3, step 3 is always (mul_s, j = 1); at d = 4 it is mul_s about half the time (ts = 3) and step 4 the other half;
  - d ≤ 2 has no mul_s, so §234 P-2's d = 2 reading is unaffected (no confound at pass 2).
- **Reading (H-POS):** training on d ~ U{1..4} ties pass 3 to "multiply S_2 by S_1". All d = 3 items say so, and so do about half of the d = 4 items. The trunk learned the position prior rather than reading step 3's op from the text.
  - Pass 4 sees mul_s or a regular op only on d = 4 items (25% of the mix) and stays at chance all run (log staircase: pass 3 plateaus at ~0.5 from 9k; pass 4 stays ~0.09).
- **Rival (H-CAP):** pass ≥ 3 lacks capacity to read an op and a state from text plus memory, and mul_s is the one case it fits. The two predict different outcomes under the change below.

## 2. Change (single, data only)
- New OW.Stream kwarg `mul_s_p` (default 1.0 = current behaviour, bitwise; flags-off gate). With probability 1 − mul_s_p a d ≥ 3 program skips the forced mul_s (ts := None, every step a regular op from OPS_REG, which includes add_s).
- **Training stream only: mul_s_p 0.5.**
  - Then P(mul_s | pass 3) is 0.5 at d 3 and ≈ 0.25 at d 4 (was 1.0 / ≈ 0.47);
  - P(mul_s | pass 4, d 4) ≈ 0.25.
  - Pass index no longer predicts op.
- **Every eval set is unchanged:** TEST2, VAL, pair sets, swap and lp5h sets are built with mul_s_p 1.0, so they are P-4's items exactly (gate: sigs equal aw_plaw4_s1's eval items).
- **Arms:** LAT-AW and CD-AN, both on the decorrelated mix. Args diff vs aw_plaw4_s1 / aw_pcd4_s1 = {mul_s_p 1.0 → 0.5} (+ run name).
- Everything else as P-4: askname init, γ = 1, d ~ U{1..4}, 16k, one RTX 3090, off-host checkpoints every 1k.

## 3. Derivation (tagged)
- [A1] The trunk at pass t outputs argmax_v p(v | text, cue_t, read(S_{t−1})).
- [A2] Under training mix M it fits E_M[CE], which rewards any feature f(t) that predicts op_t.
- [A3] Under M_P-4, I(op_3; t = 3) is large (d 3: op_3 ≡ mul_s). So a solution that ignores op_3's text and keeps "t = 3 → S_2·S_1" reaches loss ≈ 0 on d 3 and ≈ 0.47 accuracy on d 4 pass 3. That is the observed plateau (0.48–0.52, 9k–15k) [measured].
- [C1, derived] Under M_0.5, the shortcut's d 3 pass-3 accuracy falls to ≈ 0.5 and its d 4 accuracy to ≈ 0.25. The gradient toward reading op_3 is no longer cancelled by the d 3 majority, so H-POS predicts pass 3 learns to read the op.
- [C2, derived] H-CAP predicts pass-3 accuracy ≈ the mul_s share under any mix, so on P-4's d 4 TEST2 ≈ 0.47 again (and d 3 drops, since d 3 TEST2 is all mul_s while training sees half).
- [C3, conjecture] Pass 4 benefits too: half its d 4 training steps are now regular ops at a non-final position after a correct pass 3.
- **Predictions on P-4's TEST2 (binding gates = P-4's, unchanged):**
  - H-POS: CHAIN at d 4 (WS ≥ 0.50 at passes 2–4), pass-3 by-op split ≥ 0.80 on non-mul_s ops; d 3 stays ≥ 0.95.
  - H-CAP: pass 3 at d 4 ≤ 0.55 and non-mul_s ops at chance.

## 4. Readings
- **Binding:** score_aw4's P-4 gates on the new runs (LP0 each arm; LP4; CHAIN; LP5-4; CUE-ONLY guard), with run names d4d_plaw4_s1 / d4d_pcd4_s1.
- **Descriptive:**
  - WS by (pass, op_t, src_j) on d 3–4 (eval_ws items must carry ops / src_j; build change);
  - the S_2·S_1 counterfactual share at pass 3 (shortcut-use rate, P-4 = 0.505);
  - the log staircase (aux value accuracy by pass);
  - MEAN-AT-1, SWAP-MID, CUE-WRONG as P-4.
- **Labels on FAIL:**
  - D4-CAP: non-mul_s pass-3 ≤ 0.55;
  - D4-PASS4: pass 3 ≥ 0.80 but pass 4 < 0.50, so the next step is a depth / mix question for pass 4, not more of this change;
  - D4-REGRESS: d 3 < 0.90.

## 5. OC (1 seed)
- P(PASS | H-CAP) ≈ 0.02 (CHAIN needs pass 3 ≥ 0.50 on items that are ~53% non-mul_s).
- P(PASS | H-POS, pass 4 also learns) ≈ 0.75; P(PASS | H-POS, pass 4 lags) ≈ 0.2.
- Prior on H-POS ≈ 0.7 from the counterfactual share, so P(PASS) ≈ 0.35.

## 6. Censoring
As P-4: a crash means resume (dedupe); missing parts mean pending, FAIL at --final.

## 7. Cost
- P-4 realised $0.97 for both arms. Admit ≤ $1.30, guard $4.17 (P-4's).
- §234 + this: realised $3.01 + guard $4.17 = $7.18 ≤ $30.
- Build: the generator kwarg + flags-off bitwise gate, eval-set sig identity, ops / src_j in ws items. G-D at Audit.
