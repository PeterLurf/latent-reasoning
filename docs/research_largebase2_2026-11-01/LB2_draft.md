# LARGER-BASE DISTRACT draft (§258): the §256 recipe on a ~0.8B base, staged, with same-base reference arms (2026-11-01)

**Status:** draft for red team. Requested by the user via Audit (credit > $100; 24–48 GB GPUs; ≤ $30 per design).

## 1. Why, and base choice (the math)
- §256 gives C2-partial on GPT-2 small (124M): untrained passes 5–7 compute, d 6 0.98 / 0.96. The question is whether the recipe holds and improves with scale (C1 / C2 scope, and the user's GSM8K path).
- **Earlier large-base attempts:**
  - §228 / §229 closed the INJECT language line on GPT-2 small, before the AW / askname recipe.
  - §231 Qwen2.5-0.5B: the EAGER probe was at chance, and Note 252-4 few-shot was at chance even on CHAIN.
  - Both predate the answer-position write plus the askname init. The working recipe needs an init trained to answer single-step questions at ':', so §231's "eager" premise does not apply. The init is trained, not assumed.
- **Base candidates (code path, tokenizer, cost):**
  - **GPT-2 large (774M; 36 × 1280).** Same architecture, tokenizer and code path as the whole pipeline: the trunk loader changes n_layer / d only; numbers 0-10 are single tokens; the askname data is unchanged. DeltaMem goes at the middle block (18 of 36), mirroring 6 of 12.
  - **Qwen2.5-0.5B.** New tokenizer: digits split, so "10" is 2 tokens and the answer readout changes. A new block / attention port (RoPE, GQA) and a new DeltaMem insertion. A large build, and §231's base showed no eager arithmetic.
  - **Pick GPT-2 large;** GPT-2 medium (355M) is the pre-named fallback if the cost projection fails (§3 S0).
- **Cost model:** per-token FLOPs scale with parameters, about 6.2× small. Activations scale roughly with layers × width, about 5× small.
  - P-2 small peaked at 14.4 GiB at micro 1 (bs 64), so large needs about 70 GiB activations at micro 1 → micro 4–8 on a 48 GB card (A6000 / L40S), plus ~12.4 GB of fp32 AdamW state.
  - Step time is measured in S0. If large runs ~5× rcb's 0.87 s/step, 16k ≈ 19 h ≈ $9 at ~$0.45/h.

## 2. Stages (one seed; pilot; RTX A6000 / L40S 48 GB; host sha asserts)
- **S0 INSTALL ($1-2):**
  - load GPT-2 large into the trunk and insert DeltaMem at block 18 (W_o = 0, fade-in): step-0 max |Δlogits| = 0;
  - mem_check at the P-2 and REG d 4 + 2-insert shapes; G-TIME (s/step);
  - **projection rule:** if the projected S1-S3 + R total exceeds $26, switch to GPT-2 medium (same stages) before spending more.
- **S1 ASKNAME-L:** the s233 recipe (direct one-pass askname classes + perm_p 0.5, per Note 252-5) on the base, 8k.
  - Gate: P1-ASK+NAME (ST_v1) ≥ 0.98 per class; TEST-PERM d 2 pass-1 ≥ 0.95.
- **S2 P2-L:** aw_plaw2's args (d ≤ 2, aux), chain order (perm 0, Note 254-1), 12k.
  - Gate: TEST2 d 2 ≥ 0.90 AND pass-2 aux ≥ 0.90 AND WS d 2 pass 2 ≥ 0.90 (mechanism gate; memory: gate on the mechanism).
- **S3 DIS-L:** REG d ≤ 4 + --distract_p 1 from S2, 16k (the rcb + §256 steps merged).
  - Binding: RETAIN (TEST2 d 1-4 ≥ 0.98, NOT-DECODER, SWAP4, LP5-D4); C2 (d 6 ≥ 0.90, d 7 ≥ 0.80, WS d 6 passes 1-6 ≥ 0.90).
  - Descriptive: §256 held-out insert families, TEST-PERM, d 8-10.
- **R TEXT-L (same-base reference, Audit's rule):** text CoT on the same base, trained on the same REG d ≤ 4 stream (+ distract inserts), with exposure matched to S3, evaluated d 1-8 under compute-matched budgets.
  - It answers whether text CoT on the same base extrapolates as far (the fair C2 comparison) and gives the C3 compute reference.
  - Also CD-AN (direct one-pass) on the same base.

## 3. Readings and labels (S3)
- **LARGE-C2** (RETAIN + C2) / LARGE-C1-ONLY / FAIL.
- **TEXT-L comparison (descriptive):** d 6-8 acc per arm; the FLOPs per item at each arm's operating point.

## 4. OC (sketch)
- P(S1 gate) ≈ 0.9; P(S2 mechanism gate) ≈ 0.7; P(LARGE-C2 | S2 passes) ≈ 0.6.

## 5. Cost
- S0 $1-2; S1 ~$3; S2 ~$5; S3 ~$9; R ~$5 (+ CD-AN ~$1). Total ≈ $24-25; guard $30.
- **Build:**
  - n_layer / d-generic trunk and DeltaMem placement;
  - the text-CoT harness for the REG stream (REVIEW_C3L #5 lists the pieces);
  - mem / G-TIME;
  - host-built sets.
