# Multiplication on the task-agnostic carrier: design draft (2026-10-07; design only, no registration or spend until the gates read)

The user asked "when can we run multiplication". Their standing rule: multiplication returns only through the task-agnostic harness
(fsq120, §128's recipe), not §98's task-specific one. §128 excluded it because the state does not fit an 8-d FSQ code.

## 1. Why the width results gate it
* §98's mapping (kept for comparability):
  - op j = the multiplier digit b_j;
  - the operand a is visible in every segment;
  - **the only carried state is the running sum S_j = a × (b_1 … b_j)** (digits least-significant first, as in the bench files).
* Capacity needed: 4×4 has S_j < 10⁸ (26.6 bits); 5×5 has S_j < 10¹⁰ (33.2 bits).

| Carrier | Codes | Bits | 4×4 | 5×5 |
|---|---|---|---|---|
| FSQ-8 (§128) | 8⁸ | 24 | too small | too small |
| FSQ-8 at 16 levels | 16⁸ | 32 | fits | too small |
| FSQ-16 (§133) | 8¹⁶ | 48 | fits | fits |
| FSQ-64 / NOISE-64 / CONT-64 (§133) | — | ≥ 192 / continuous | fits | fits |
| GROW 8 → 64 (§136) | grows | ≥ 192 at the end | fits | fits |

* So multiplication needs a **§133 wide arm that passes stage 1**, which is G1 below. FSQ-16 is the smallest sufficient width, and it passed its proof
  slot (0.996, ρ 0.17).

## 2. Item and op encoding (one family adapter; FINAL only, so C4-compatible)
* **Segment j:** [EOT, carried code, a's n digit tokens, b_j, a position token j].
  - The operand sits in context at every segment, like §132's in-context tables (rung 1 PASS).
  - The position token makes the shift explicit.
* **Start:** a fixed start code (as for FF / CYC / MOD / RDFA), so S_0 = 0.
* **Answer:** the product's digits (the bench's `####` field, least-significant first), decoded from the final code by the answer decoder, as for
  every family.
  - **No step labels:** the bench's P_j / S_j text is never used, so the run is final-answer-only.
* **Data:**
  - train on the bench train splits (`bench/data/mult_train/{4x4,5x5}_train.txt`) plus generated shorter-multiplier problems for the
    curriculum (same rule);
  - score on the **1,000 published test items** (`bench/data/mult_{4x4,5x5}_test.txt`), the same items as §98's comparators.
* **Fallback encoding** (only if the per-digit step is too hard to enter): ops = (a_i, b_j) digit pairs in column order, giving n² simpler steps and
  the same carried state.

## 3. Curriculum, recipe and arms
* **Length curriculum over the multiplier length**: §120's L_max idea.
  - Chains of L = 1 … n multiplier digits, with L_max 1 / 2 / 3 / 4 from steps 0 / 2k / 5k / 9k (4×4).
  - A shorter chain's answer is a × (b_1 … b_L) = S_L, so the decoder also learns to read intermediate sums. That makes "prefix accuracy" exact
    here.
* **4×4 first; 5×5 only after a 4×4 PASS.** 5×5 starts from scratch on the same recipe, with L_max to 5.
* **Arms (draft):**
  - **W:** the best stage-1 wide carrier, with §128's restart-on-plateau slots.
  - **W + COPY:** §134's copy rows, with no restarts. Copy targets are other problems' products, which are numbers of the same kind as the
    carried S_j, so the anchor fits this task unusually well.
  - Optional, when its gate reads: **GROW** (if §136 is safe).
* **Comparators** (same 1,000 items):
  - TC-M (text CoT) 1.0;
  - ICoT-SI 0.998 (4×4) and 0.996 (5×5; 0.95 published);
  - GEN-M 0.163 / 0.005 (§98's latent arm, the k = 4 hand-off wall).
* **Rules sketch** (fixed at registration):
  - **M-PASS:** ≥ 3/4 seeds at ≥ 0.99 on the test items, SERIAL, and ρ ≤ 0.28 (with the continuous clause where relevant).
  - **SERIAL check:** inject a donor's code at step 2. The answer must follow the donor's S_2 continued with the recipient's a and remaining
    digits, which is computable exactly.
  - **M-SOTA:** median ≥ ICoT-SI on the same items.
  - **Descriptive C2:** 4×4-trained models on 4×8 (generated).
* **The §135-revised idea** (a state-only teacher target, a warm-start fade, on COPY's base) would be framed **on this fsq120 family**, not §98's
  harness. The teachers would be §98's kept TC-M models. It waits for the user's go, and it would be a separate registration after W / W + COPY
  read.

## 4. Cost and runtime (estimates)
* A segment is about 8 tokens and chains are ≤ 4–5 segments, against S5's ≤ 32 segments of about 3 tokens. So a 24k-step run should take
  about 30–60 min on a 3090 (S5 runs: about 1.4 h), about $0.1–0.3 per run.
* **4×4:** 2 arms × 4 seeds = 8 runs, about $2–3. **5×5:** the same, about $3. **About $6 in all**, plus a 4× budget variant (96k steps, as §98 gave
  GEN-M) only if a 24k arm is close.

## 5. Gates and ETA
* **G1:** a §133 wide arm reads stage-1 PASS (≥ 3/4 slots). Slots 2–4 are running; **expected within about 12 h.**
* **G2:** the §134 guard (COPY does no harm on FF / CYC / MOD / RDFA). It is deploying; **expected within about 6 h.** Without it, the W + COPY arm
  waits.
* **G3 (optional):** §136 stage 1 shows growth is safe, which adds the GROW arm; **about 24 h.** It does not block.
* **Then:** Literature's check on this design (about 1–2 h), registration plus the scorer (Theory, about 1 h), and the family adapter (the
  experiments session: multi-token items, digit answers, donor ground truth; CPU smoke and CUDA gradflow; about 3–6 h). The 4×4 proof run follows
  (about 1 h).
* **ETA:**
  - first 4×4 proof result **2026-10-07 evening**, if G1 / G2 read by midday;
  - 4×4 four-seed result **2026-10-08 morning**;
  - 5×5 **2026-10-08 evening**.
