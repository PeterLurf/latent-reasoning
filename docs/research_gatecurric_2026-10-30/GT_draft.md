# GATE-CURRIC draft: can a competence-gated depth curriculum replace the per-pass anchor? (C4-partial) (2026-10-30)

**Status:** draft for red team. Theory's choice (Note 251-7; Literature's untested alternative to the anchor, ATC 2609.19717).

## 1. Why
- §251: AEX (aux on d ≤ 2) extends to d 4; AEX0 (no aux after the seed) fails in both seeds.
- Note 251-7 gives the mechanism.
  - Every model solves d 2 with 1 pass plus the decoder.
  - Without aux, the pass-2 write erodes: WS d 2 pass 2 falls from 1.00 to 0.22 in AEX0 s2.
  - d 3 needs S_2 to hold v_2, so d 3 never forms.
  - In AEX0, half the batches (d ≤ 2) reward the shortcut, and d 3 / d 4 get only 25% each.
- ATC (Literature 10-30) learns chains from answers alone with:
  - a competence gate (held-out acc > 0.9 for 5 evals);
  - rehearsal λ = 0.1;
  - backtracking.
- **Question:** does putting ~90% of batches on the next depth, which can only be solved with the pass-2 write, restore the write and extend the chain without per-pass targets?

## 2. Arms (single change vs AEX0)
- **GATE:** gate_s1 / gate_s2 run AEX0's recipe exactly: init aw_plaw2_s1, REG law, `--aux_fade 0,0`, 16k, K 8, micro 2, the same seeds, RTX 3090.
- The only change is a new flag `--d_curric gate` replacing the uniform d law:
  - **Stage s** starts at 3, since the seed covers d ≤ 2. Each step draws d = s with p 0.9, else d ~ U{1..s−1} (rehearsal 0.1). The rng is seeded as in the uniform law.
  - **Gate eval** every 250 steps, no_grad, model.eval(): answer acc at K′ 8 on a fixed dev set of 200 items per depth 1–4. It uses the REG law from a new seed, and its sigs are added to the training exclusion and disjoint from TEST2.
  - **Advance** s → s + 1 (max 4) when dev acc at s > 0.9 on 5 consecutive gate evals.
  - **Backtrack:** if dev acc at any d < s drops below 0.9 on 2 consecutive evals, s := the smallest such d (minimum 2). Then advance again as above.
  - Stage, dev accs and the d histogram are logged at every gate eval.
- **Comparators:** AEX0 (aex0_s1 / s2: the same recipe with the uniform law; FAIL) and AEX (the anchored reference). No new CTRL runs.
- Args diff vs aex0_s{1,2} = {d_curric gate}.

## 3. Readings (score_ae gates, d 4-centred; per seed; binding)
- ACC4 ≥ 0.50 and ≥ CD + 0.25.
- NOT-DECODER: d 4 at K′ 2 ≤ 0.20.
- SWAP4: SWAP-MID d 4 t 2 ≥ 0.50 and ≥ floor + 0.30.
- LP5-D4 ≥ 0.30.
- CUE-ONLY guard; STABLE.
- Labels as in §251 (EXTEND / EXTEND-D3-ONLY / NOT-EXTEND / VOID / UNSTABLE). Per arm: EXTEND (both) / SPLIT / FAIL.
- **Mechanism (descriptive, pre-stated prediction):** EXTEND seeds have WS d 2 pass 2 ≥ 0.9 and d 3 at K′ 1 ≤ 0.2. NOT-EXTEND seeds have WS d 2 pass 2 < 0.5.
- Stage trace: the step at which s reaches 4, if it does, and any backtracks.
- **Censoring:** if s never reaches 4 by 16k, d 4 was seen only as rehearsal. The run is still scored as registered, and its d 4 gates will likely fail; the label stands.

## 4. OC (per seed)
- P(EXTEND | depth pressure suffices) ≈ 0.4; P(EXTEND | per-pass targets needed) ≈ 0.03.
- P(EXTEND-D3-ONLY or better | pressure suffices) ≈ 0.6.
- Both seeds EXTEND: ≈ 0.16 vs ≈ 0.001.

## 5. Cost
- 2 × 16k on RTX 3090 (AEX0 realised $0.80–0.81 each; gate evals add ~5%) ≈ $1.7; guard $2.5. Credit $25.7 → ≥ $23.
- **Build:** `--d_curric gate` (default off = bitwise current), the dev set and exclusion, and gate logging.
- **G-D:**
  - default-off bitwise vs the current module;
  - a CPU / GPU smoke forcing an advance and a backtrack with a mocked dev acc;
  - dev-set disjointness from TEST2, pairs and frames;
  - the step-0 gate eval on aw_plaw2_s1 reproduces REG d 1–4 ≈ 1.0 / 1.0 / 0.03 / 0.10.
