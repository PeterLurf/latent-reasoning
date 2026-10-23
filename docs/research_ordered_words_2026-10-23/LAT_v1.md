# LAT-OW v1: a C1 test of the §184 latent recipe on English-templated word programs (2026-10-23)

**Status:** v1 = LAT_v1_draft.md (ad2a3be4), changed as listed here; where they differ, this file wins. Red team: REVIEW_LAT.md (GO with fixes; fixes 1–10 adopted). Literature: the addendum to LIT_ONEPASS_DEPTH_THRESHOLD_2026-10-23.md (N1 / N2 narrow; N3 (a) / (b) "not found"). Basis: §226 (one-pass DIRECT learns d = 1, not d = 2) and §227 (serial text-CoT 1.000 at d 2–4).

## 1. Claim and scope (fix 2)
- **C1 only**, on "English-templated modular arithmetic (TOPO layout), GPT-2 small, one seed (a second only on a pass)".
- A null speaks about **this recipe (the §184 INJECT port)**, not about language.
- **Deviations from §184, listed:**
  - (a) the step target is a single value, where §184's is the whole partly reduced expression;
  - (b) name lookup among distractors;
  - (c) 4–5× longer think inputs;
  - (d) one d per step;
  - (e) cache_lower (§184 ran uncached; gated in §8 of the draft);
  - (f) uniform d ~ U{1..8} from step 0;
  - (g) English frames.
- **Pre-named fallback:** if the pilot fails LP3 / LP4, the single allowed follow-up is a cumulative step target (the partly reduced program as §184's expression analogue), registered separately. This is not a retune.
- **Disclosed:** LAT gets per-step targets that DIRECT does not get (as in §184).

## 2. Readings (replacing draft §4 where named)
- **R1 DEEP-GAIN:** unchanged (acc_LAT − acc_DIR ≥ 0.30 and acc_LAT ≥ 0.60, pooled over band B).
- **R2 REMOVE (fixes 1, 4):**
  - (i) K′1 is read only on band depths ≥ 3 and compared with DIRECT on the same depths.
  - (ii) **The one-pass reference is a binding MEAN-STATE read:** S is replaced by the TEST2-mean of S₈ at every pass. It requires acc(8) − acc(mean-state) ≥ 0.30 and acc(mean-state) ≤ acc_DIR + 0.10 on B.
  - K′0 (bias zeroed) and the K′ onset by depth are descriptive.
- **R3 SERIAL (fixes 5, 6):** the draft's STATE-SWAP after pass ⌈d/2⌉, scored on cf(a) ∪ cf_b, at d ≥ 4, N ≥ 2,000, floor = max(3-donor same-skeleton redraw, permutation), SELF-SWAP bitwise.
  - **Plus follow_U ≥ 0.50** (the union-follow share itself, not only its margin over the floor).
  - R3 is reported per depth.
  - **LEAK-SEG-H is descriptive** (it guards a channel STATE-SWAP does not have). The existing BUDGET-FRONT arm gives a STATE-TIMED vs DEFERRED sub-label.
- **C1-LANG** = R1 ∧ R2 ∧ R3. R1 ∧ R2 without R3 = DEEP-NOT-SERIAL.

## 3. G-W0 band (fix 7)
- As in the draft: exposure-matched (DIRECT and LAT draw the identical d-sequence, ≈ 6,000 steps per depth), with the 223-6 guards.
- A depth enters B only if DIRECT ≤ 0.50 **and DIRECT is not still rising:** OLS slope of the last 4 d-specific 1k bins ≤ +0.02 per bin (else DIRECT-STILL-RISING removes that depth).
- B needs ≥ 3 depths, ≥ 2 of them ≥ 4.
- **Reworded:** "P0d's d = 2 fail is supported by its flat d = 2 trend", replacing "exposure-adequate".

## 4. Futility and late learning (fix 8)
- The 24k futility stop (LAT d = 2 train acc < 0.50) also requires no rise over the last 4 bins (slope ≤ +0.02 per bin). If it is still rising, there is one re-check at 32k with the same rule.
- At 48k a LAT-STILL-RISING flag is reported for any band depth with a rising last 4 bins (§184 s1 entered its deep depths only at 42–44k).

## 5. Pilot (fix 3)
- The draft's P-DIR2 / P-LAT2 run locally, $0 (d ~ U{1, 2}, 8k; LP0–LP3 as in the draft).
- **Plus P-DIR4 / P-LAT4** (d ~ U{1..4}, the cloud batch law, 16k steps), **on one cloud RTX 3090 (≈ $1.0–1.3)**. d ≥ 3 is where memory back-references (computed states, mul_s) first occur: 0 % of d = 2 items, 100 % of d ≥ 3.
- **LP4 (binding):** LAT TEST2 acc at d = 3 ≥ 0.60 and at d = 4 ≥ 0.50, each ≥ DIRECT + 0.25 on the same depths.
- Both pilots must pass before Stage 1. Failing either → the pre-named fallback (§1) or stop.

## 6. Batches and cost (fixes 9, 10)
- One d per step is kept, disclosed: the aux weight per step swings 0.37–2.31. Clip rate is logged by depth.
- Mixed-d batches replace it if F2's measured cost per step is within the share.
- **Cost:**
  - LAT share $9.00 (guard $11.00); DIRECT share $3.00 (guard $3.60).
  - TEXT moves to Stage 2, under §227's CurGate recipe, as the C3 comparator after a C1 pass.
  - Stage 1 ≤ $12 incl. the cloud pilot; a second seed only on C1-LANG; family cap $20.
  - F2 re-measures s/step before admission (the draft's 1.01 s/step vs the review's 1.15–1.41).

## 7. OC (draft §5 kept; fix-induced changes)
- With fixes 1 / 4 / 6, a hop-per-pass serial model passes R2 with P ≈ 0.9 (vs 0.05 in the draft).
- A 38 %-follow carrier no longer passes R3 (follow_U).
- The mechanism nulls (pause, relay, §207 recompute, floor collapse, leak) stay at P(C1-LANG) ≈ 0.
- Overall prior P(C1-LANG) ≈ **0.08** [ASSUMED].

## 8. Unchanged from the draft
Cells, TEST2, code gates (port-kernel identity vs nested170, fp64 cache_lower check, flags-off bitwise, G-D, the d-sequence binding check), the scorer score_lat.py (fixtures first), and the novelty wording (N1 / N2 narrow; N3 "not found").
