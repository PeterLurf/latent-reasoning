# ANSWER-WRITE v1: write the latent state from the answer position (2026-10-25)

**Status:** v1 = AW_draft.md (0982fcd9), changed as listed here; where they differ, this file wins. Red team: REVIEW_AW.md (GO with fixes 1–7; 8–10 adopted). Literature (addendum in LIT_STEPTOLD_2026-10-24.md, 9ce7aeb5):
- N1 NARROW: "a query-conditioned variant of Coconut-style last-hidden-state feedback, written into a recurrent memory (plan given)". Priors: Coconut, CODI 2502.21074, RMT 2207.06881 (verified).
- N2 / N3 NARROW.

## 1. Treatment (fix 3)
- At pass t the input is `text + " How many <u> does <S_t> have?" + " The answer is:"`. Only the ':' state writes the INJECT state, with **γ = 1 from step 1** (`--write_mix_start 0 --write_mix_end 0`, part of the treatment). ST-AN uses the same schedule.
- No think-pass answer CE in P-2.
- Init: s233_askname_s1 (sha c00f7ec0…). Deviations w1–w6 as the draft.

## 2. $0 gates, in order, rules committed before any init read (fixes 1, 8)
- **G-AW0 (build / presence):** a v_1 probe at mem.ln_w(h_w) at ':' on the init ≥ 0.80, with layer-0, v_0 and read == 0 controls.
  - The bar is 0.80 (changed from 0.90 on the base-GPT-2 reference before any askname read; .void kept; Audit to confirm from git that no aw_gate_askname.json predates this commit).
  - One pre-named UNREADABLE fix: v0ctl_u0|v0 ≥ 0.80.
- **G-AW1 FROZEN-USE (the main risk, A1; fix 1):** frozen askname trunk + DeltaMem with a single pass-1 write (γ = 1) of the d = 2 item's ':' state. Fit only W_o (the read-out into the residual) on train items. Read v_2 from the decoder pass.
  - **Pass: TEST d = 2 v_2 acc ≥ 0.50**, with controls (a) S := 0 (≤ chance + 0.05) and (b) a readout fit through LP2's path (a reference).
  - **A FAIL goes to the user before P-2** (the frozen trunk cannot use a read-in value; a new design is needed).

## 3. P-2 (one RTX 3090, d ≤ 2, 12k) readings
- As §233 P-2: LP0 self-calibrated; LP1; LP2 pass-1 aux on d = 2 ≥ 0.80; LP3 binding (≥ 0.70 and ≥ CD-AN + 0.30); LP5 (mean pass-1 state, cue kept, drop ≥ 0.30); MEAN-STATE; CUE-ONLY guard; CUE-WRONG label.
- **STATE-USE binding:** SWAP-K on 1,000 same-skeleton d = 2 pairs: follow_U ≥ 0.50 and ≥ floor + 0.30.
- **Failure diagnosis (fix 2):**
  - an end-of-run `u1|v1` probe on the trained model separates AW-SOURCE-DRIFT (the ':' code eroded) from AW-NOT-STORED (written, not in S);
  - WS-READOUT on a held-out set every 1k steps;
  - label AW-WEAK-READ when LP2 passes but LP3 / LP5 fail.
  - No reading-forcing mechanism in P-2. Warm-starting W_o or ELIDE batches are pre-named as new designs.
- **D2 / D3 restated (fix 5):** the read is 12 per-head scalars × v_h (delta-head scalars are negative at 22 % of positions). D3 holds for S_1 only. `read1|v1` replaces `s1|v1`. The K′ 1 / K′ 2 decoder readings and the pass-1 / pass-2 key cosine are logged.

## 4. Leak fixture (fix 4)
- FX-LEAK drives the real loss and eval entry points with S_K hooked.
- The planted positive-control leak is a value stashed on the module, not an argument.
- It asserts `dec_logits` identity, plus a batch-composition check.

## 5. Sequence and cost (fixes 6, 7, 9)
- G-AW0 → G-AW1 → P-2 → (on pass) ST-AN (args pinned now: §233 LAT-ST on the askname init, γ = 1, P-2 rule) → P-4 (chaining required: WS-READOUT ≥ 0.50 at pass ≥ 2 on d ≥ 3) → Stage 1 (re-costed from P-2's measured s/step).
- **Running cost rule:** realised spend + remaining guards ≤ $30 at every launch; otherwise the user decides. One UNREADABLE re-run of P-2 counts against it.
- P-2 admit ≤ $1.85, guard $2.31. Family guard $29.81 (cap $30). Expected spend ≈ $4.2.

## 6. OC
- PASS 0.000 under no effect, stored-but-unread, recompute and init-solves-one-pass; 1.000 under strong serial use and "pass 1 + decoder" (labelled DEC-STEP2; P-4 must show chaining); 0.88 marginal.
- P(P-2) ≈ 0.19; P(C1-TOLD) ≈ 0.04.
