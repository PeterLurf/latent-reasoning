# Synthesis spec v0 (Theory, 2026-10-15): VERSION-BANK + READY-RECODE

Sources (read these in full; they are the detailed designs), all in this folder:
- A_bounded_core.md (architecture: VERSION-BANK);
- B_training_paradigm.md (STAGGER + SLOT-IIT, Props H/D/B/M);
- C_forced_recoding.md (LEDGER + RE-POSE, teacher forcing, TOM, EDIT test).

Prior art: D:/Latent/docs/LIT_PASSUSE_PRIORART_2026-10-15.md.

User constraints:
- (21:58) invent a training paradigm and a recurrent memory architecture that force and reward the use of passes and the re-coding of information; attention allowed;
- (22:07) the memory must be pass-global attention memory, not RNN memory.

## Architecture = A's VERSION-BANK (VB-12), unchanged unless noted
- **Split:** prelude = GPT-2 blocks 0–5, run once on x = [EOT, problem, " ="]. Shared core = blocks 6–9, looped K times. Coda = blocks 10–11 + ln_f + LM head.
- **Pass t:**
  - each core layer ℓ attends over the current version's slots plus ALL bank entries V_0..V_{t−1}, all positions (keys/values from block ℓ's own ln_1/c_attn applied to the stored vectors);
  - the attention-logit bias on a bank entry from version s is β_{ℓ,h} + γ_{ℓ,h}·(t−1−s), with β init −5 and γ ≤ 0, init 0;
  - write V_t = output after block 9, appended (never overwritten).
- **Pass input, two variants:**
  - CARRY: U_t⁰ = N(V_{t−1}), a per-slot RMS reset to the pass-0 scale, gain g init 1. Pass 1 is exactly pretrained GPT-2 when β = −∞.
  - PURE (the user's strict form): U_t⁰ = learned q + scaled wpe[j]; everything else is read from the bank.
- **No pass index anywhere** (positions reused; relative age only).
- **Coda:** blocks 10–11 over V_t (+ decoder tokens prelude-encoded among themselves only), reading the bank (bias init −5). No problem tokens.
  - Nested answer at the " =" slot (CE over 11 values).
  - K′ = 0 is the coda on V_0.

## Training paradigm READY-RECODE
- **(R1) Readiness-window answer loss (A):** CE at the coda after pass t, for t ∈ [τ_i, K_i] only. τ = the item's step count (n on nested; m CoT equations on GSM8K); K_i = min(τ + U{0..4}, K_max = 12). No answer loss before τ.
- **(R2) Per-pass recode target (A):**
  - nested: a LINEAR per-slot head R (LN + Linear 768→20 classes) on every slot of V_t must give E_t, the slot-aligned expression after t Feng reductions (consumed slots VOID; the reduced node's last slot = its value; HOLD E_n for t > n);
  - GSM8K: the text recode "<<e_t>>" via the coda with a question-free NOPASS prompt.
- **(R3) Teacher-forced versions (from C; a separate cell for attribution):** with probability p_TF (1 until 4k, linear to 0 at 24k), the bank versions s ≤ t−1 that pass t reads are replaced by sg teacher versions V^T_s = core(prelude(render_VOID(e_s))), where render_VOID keeps length T by writing a filler token at consumed positions. So each pass's task becomes "one reduction from a correct version".
- **(R4) Curriculum:** n_max 3 → 5 → 8 gated (A); GSM8K K_max 2 → 5 → 10.
- **NOT adopted, with reasons:**
  - marginal-pass rewards (B's Prop M; published sabotage evidence, Cho 2505.23060);
  - destructive / lifetime memory (user constraint);
  - per-pass rotations (C §0.5);
  - training-only past-slot dropout (B §1.5: it trains re-reading).
  - STAGGER is reserved as the GSM8K add-on if PASS1-PRECOMPUTE recurs. It needs per-pass input snapshots: in VB, the bank's input entry becomes V_0^(p) = prelude(x^(p)) on window rows.

## Decisive readings (comparator-free), nested
- **ITER-GAIN_N** = ACC(K′ 12) − max(ACC(K′ 0), ACC(K′ 1)), pooled n 3..8; bar ≥ 0.50.
- **ONSET / SERIAL-PROFILE (A):** coverage ≥ 0.8 at every n; slope of the median onset on n ≥ 0.6; med(8) − med(3) ≥ 3.
- **SKIP-DELAY (A):** skipping pass ⌈n/2⌉ drops K′ = n to ≤ 0.3 × intact and recovers at K′ = n + 1 to ≥ 0.9 × intact.
- **SERIAL-SLOT (B; untrained):** splice the donor's single version V_t (donor = the recipient with one handle's operands changed) into the recipient's bank, keeping the recipient's V_0..V_{t−1}. The answer must follow the hybrid; chance-floored.
- **BANK-ABLATE (A; G10, carry-only):** mask versions 1..t−2 at every pass.
- **Labels:** VB-C1, TWO-STAGE, ONESHOT-PRECOMPUTE (falsifies the premise), NO-ENTRY, LOSES-ONE-PASS, CARRY-ONLY, COUNT-KEYED, TF-ONLY (R3 cell).

## Stage plan
- **Stage 0 ($0, local):** gates G1–G7 (A §7.1); the render_VOID/E_t self-check; B's G-ALIGN (WINDOW-ACC on existing nested models; descriptive).
- **Stage 1 (cloud 3090, seed 1, 48k):**
  - N1 CARRY + slot recode;
  - N2 PURE + slot recode;
  - N3 CARRY + slot recode + TF versions (R3).
  - About $1.3–2 each; cap $8.
- **Stage 2 (GSM8K, only after a VB-C1 cell):** VB-12 (+ VB-8) with the text recode; STAGGER add-on if PASS1-PRECOMPUTE.
