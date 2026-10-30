# KV-LAT draft: a FLOPs-competitive latent loop, with the latent state as appended, KV-cached positions (C3 + C1) (2026-10-29)

**Status:** draft for red team. Theory's choice under the user's standing rules. It is the design implication of Note 246-0 (C3: latency yes, FLOPs no), staged as a small pilot within the ~$10 headroom.

## 1. Why
- The §234–§240 loop re-runs blocks 6–11 over the whole prompt every pass, reading DeltaMem at every position. That costs 1.5–3× text-CoT's FLOPs at d 2–4 (Note 246-0).
- Append-and-cache designs (Coconut 2412.06769, CODI 2502.21074 §3.5, pause tokens 2310.02226; Literature) cost about one token's forward pass per latent step.
- No FLOP-matched latent-vs-CoT measurement exists (Literature), so a FLOP-matched C3 reading would be new as a measurement. The architecture itself is Coconut-family (no novelty claim for it).

## 2. Architecture (one change vs §234 P-2: the memory mechanism)
- The prompt [EOT] + text is run once and its KV is cached.
- **Pass t:** feed the cue tokens c_t ("How many <u> does <S_t> have? The answer is:") plus nothing else, attending to the cached prompt and to the latent positions L_1..L_{t−1}.
  - The last-layer state at c_t's ':' is mapped by W_lat (D×D, identity init) to an input embedding, appended as latent position L_t (KV cached).
  - The cue tokens of pass t are NOT kept in the cache (only L_t is), so the context grows by one position per pass.
- **Decoder:** SFX tokens attending to the prompt plus L_1..L_K → the answer, with argmax over the 11 number tokens as before.
- **Init and training:** askname init (s233_askname_s1); the P-2 recipe (d ≤ 2, 12k, bs 64, lr 1e-4); per-pass aux CE on v_t at pass t's ':' (as P-2; this is not the C4 question here); W_lat trained.
  - DeltaMem is removed for this arm.
- **Analytic cost per item:** prompt prefill (12·blk(P)) + Σ_t 12·(|c_t| cached decode steps) + decoder (|SFX| cached steps).
  - At d 2, K′ 1, that is ≈ prefill + ~14 cached token-steps, against text-CoT's prefill + 16 tokens, roughly FLOP-parity.
  - Serial layers ≈ 12·(Σ|c_t| + |SFX|), more than the DeltaMem loop's 12·(K′ + 1), since cue tokens are decoded serially. Disclosed: KV-LAT trades latency for FLOPs. A variant that feeds c_t in parallel as one prefill chunk keeps the serial depth at 12 per pass; it is the default here, since the cue is known in advance.

## 3. Readings (binding, per seed; P-2 gates adapted)
- **LP3:** d 2 acc ≥ 0.70 and ≥ CD-AN + 0.30.
- **LP5:** L_1 replaced by its TEST2 mean drops d 2 acc ≥ 0.30.
- **STATE-USE:** L_1 swapped with a donor's: follow_U ≥ 0.50 and ≥ floor + 0.30.
- **CUE-ONLY guard.** **STABLE** (the §244 rule).
- **C3-F (binding):** executed FLOPs at the cheapest config reaching LP3's accuracy ≤ 1.1× text-CoT's analytic FLOPs (Note 246-0 formula; TEXT assumed at ceiling, disclosed).
- **Descriptive:** serial layers; the K′ curve; d 3–4 (out of range).
- **Labels:** PASS (all) / C1-ONLY (C1 gates, C3-F fails) / NOT-FORMED / VOID / UNSTABLE. Overall: both seeds.

## 4. OC (sketch)
- P(C1 gates | KV-LAT): Coconut-style loops with per-step supervision train readily, and the askname ':' write gives pass 1 its value, so ≈ 0.6 per seed.
- P(C3-F | formed) ≈ 0.9 (arithmetic).
- Both seeds ≈ 0.35.

## 5. Cost
- Build: a new arm in inject_aw (an appended-latent path with a KV cache), gradflow, a flags-off gate on the DeltaMem path, and an eval for the swap / mean on L_1. This is the main effort.
- Two 12k seeds ≈ $1.0 (cheaper per step than the DeltaMem loop); guard $2.5.
- Running: realised $20.1 + $2.5 = $22.6 ≤ $30.
