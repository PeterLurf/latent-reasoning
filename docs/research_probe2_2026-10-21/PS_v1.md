# PROBE-S v1 (§222 on registration): certified probe readings for §218's A1, with content-equivariant plants (2026-10-21)

**Status:** v1 + REVIEW_PS edits E1–E6 (targeted Opus review: no blockers; no bar changed), cap $0, eval-only. Successor to §218's probe readings, which are structurally UNREADABLE (218-18 / 218-24 / 218-25). Draft: P2_draft.md (6bb096e8). Review: REVIEW_P2.md (Opus; the Frame Lemma).

## 0. Honesty lines
- **§218's VB-C1-L stays NOT MET** whatever this design reads. The cause was a positive-control design flaw (218-25), found before any A1 output.
- **Designed after §218's control failure and after a feasibility check on the untrained model only.** The check used the registered streams (22500 / 22551–22554 / 22593–22597 / 22601–22604):
  - S1 (registered read path), cexor at t 0: ×1.5 → 0.748, **×10 → 1.000**. ×10 was seen.
  - S2 (antisymmetric head), cexor at t 1 ×1.5: 0.496.
  - The remaining feasibility arms and the K2-cexor oracle may land before or after this freeze. Every rule below is mechanical, so their values cannot change it.
- **No A1 output of any kind has been read.** A1 has not landed.
- **No new spend trigger.** A2 and B1 stay under §218's rules and the user's decisions (218-18 item 1).

## 1. Candidates (the choice rule is pre-stated in §3)
- **S1 (minimal change):** §218's registered probe, unchanged in read path, recipe, budget and extension rule (Probe(m, 1e-4), pr.scores frozen-decoder sA − sB, 10,000 steps, bs 64, the 20k extension rule). **Only the PLANT definitions change:**
  - **PLANT-lin-S** = antisymmetric lin: o1 row += ε(2z−1)u, o2 row −= ε(2z−1)u.
  - **PLANT-xor-S** = content-equivariant xor (REVIEW_P2 §2.4): a fair coin picks the target row T. T += ε(s1u + s2u2) with s1s2 = +1, and the other row += ε(s3u + s4u2) with s3s4 = −1. The label is T.
  - Both are content-separable, so the order-blind readout can read them (Frame Lemma, 218-25). For PLANT-xor-S, linear accuracy on either row is ½, a genuine XOR.
  - u, u2, ε: as §218 (each model's own V_t PCs 40–60, ε = √λ₅₀, rng 22550), and × m (§2).
- **S2 (fallback):** PROBE2 with an antisymmetric head g = f(hA, hB) − f(hB, hA), f = MLP([hA, hB]) → 256 ReLU → 1. hA / hB are the coda copy's last-layer states at each option's name positions. Coda copy lr 1e-4, head lr 1e-3, wd 0.01, same budget and extension rule.
  - Option order is randomised per item for every PROBE2(t), in training and in readout.
  - **S2 additionally requires** the review's guards before any A1 reading:
    - SERIAL-ORACLE: a planted gold-REACH frontier code on untrained / reference V_t, read with slope ≥ 0.8, span ≥ 4 and gain ≥ 0.40 in every candidate band;
    - SELFCOMP null: PROBE2(1) on the untrained model's V_1 ≤ 0.55.
    - If either fails, S2's ONSET / GAIN readings are declared non-informative.

## 2. Plant strength m (from the oracle only; Audit)
- **m = the smallest m ∈ {1.5, 2, 3}** such that, on the untrained model at **both** t 0 and t 1:
  - K2-anti-lin (logreg on [r1, r2]) ≥ 0.985;
  - **and** K2-cexor (the rule oracle: per-row sign product, pick the +1 row) ≥ 0.985.
- If no m qualifies, m = 3, flagged "ceiling below 0.985". S1 / S2 may still certify at that m, and the flag sits on every reading.
- m is never chosen from where a probe passes.

## 3. Certification and the choice rule
- **CERT(S)** holds iff, on **both** the untrained LangSLVA **and** the re-run G-L1c reference model (same-model gate SAME, 0.984375; items disjoint from the SYM test set and GATE):
  - LATE(PLANT-lin-S, m) ≥ 0.90 and LATE(PLANT-xor-S, m) ≥ 0.90, at **each t ∈ {0, 1, 2}**;
  - under S's recipe at the registered budget (10k steps, plus the extension rule if the last 2k gain > 0.02);
  - LATE = the last three held-out evaluations;
  - and the controls hold:
    - cexor-NOPLANT (coin labels, no plant) LATE ≤ 0.55. This is an exact null for any trained probe: labels are independent of every input, so P(LATE > 0.55) ≤ 0.003 per cell at n 1,000.
    - shuffled-label cexor LATE ≤ 0.55. This is ½ only in expectation: a label-blind readout of planted rows can correlate with T through the plant axes (u + u2 vs u − u2). With per-batch permutation it is a plumbing smoke, not leak evidence.
    - PROBE(0) (no plant; S2 with randomised order) within 0.50 ± 0.015 on the SYM test set.
- **Fresh streams for certification** (roles pinned; held-out / test generators restarted at every read; no stream serves two roles). None was used in feasibility.
  - pool / held-out: PS.Stream 22700. This is also §216's pool_seed, under a different generator and data; recorded, harmless.
  - batches: 22701.
  - plant_dirs coefficient generator: 22750 (a parameter replacing the hard-coded 22550, for certification only).
  - PLANT-lin-S z: train 22751, held-out / test 22752.
  - PLANT-xor-S coin T: train 22753, held-out / test 22754.
  - PLANT-xor-S signs: s1, then s3, as two sequential draws of one generator, with s2 = s1 and s4 = −s3. Train 22755, held-out / test 22756.
  - cexor-NOPLANT coin: train 22757; held-out reads 22754's coins.
  - shuffled-label control: a fresh permutation of each training batch's targets, generator 22758 (never a fixed permutation of the pool).
- **A1's plant streams:**
  - u, u2, ε: rng 22550 (§218's plant_dirs) on A1's own pool rows.
  - PLANT-lin-S z: as §218 (22551 train / 22553 held-out and test).
  - PLANT-xor-S: coin T = the first array and s1 = the second array (0 → −1, 1 → +1) of §218's xor draw (22552 train / 22554 held-out and test).
  - s3: a new generator, 22558 train / 22559 held-out and test.
- **Card / dtype:** the RTX 4060, fp32 eval (A1's probe-eval card and dtype), for every certification run.
- **Choice rule (pre-stated): S1 if CERT(S1); else S2 if CERT(S2) (with S2's guards); else NO CERTIFIABLE PROBE.**
  - In the last case A1's probe readings stay §218's (UNREADABLE), and that is reported.

## 4. Readings on A1 (with the selected S; A1's registered files as §218's)
- **PROBE-VALID-S(t)** on A1 itself:
  - PLANT-lin-S and PLANT-xor-S, with A1's own plant directions at m, both ≥ 0.90 at t ∈ {0, 1, 2};
  - **and**, at each t ∈ {0, 1, 2} paired with that t's plants, K2-anti-lin ≥ 0.985 and K2-cexor ≥ 0.97 on A1's own planted option rows at m.
    - K2-cexor is the rule oracle, centred on A1's pooled unplanted option-row mean at that t from its K2 train split.
    - K2 streams: coin / signs 22611 / 22612 train and 22613 / 22614 test; lin z 22571 / 22572. As in feasibility, no answer information is read.
    - Below either bar at any t, that t reads "plant at ceiling on A1" = PROBE-INVALID(t). The K2 runs may run before A1's probe parts.
- **DIAG-ALIGN** (descriptive, never gates; computed and read only with A1's probe parts). On A1's unplanted held-out option rows at t 0–2, centred as K2:
  - d′ of (r_gold − r_other)·u and ·u2;
  - the gold accuracy of sign((r_o1 − r_o2)·u) and of the cexor sign-agreement rule;
  - cos²(u, ĝ_t) and cos²(u2, ĝ_t) for ĝ_t = mean(r_gold − r_other), against 1/768;
  - var(option rows along u, u2) / λ50.
  - At t = 0 the gold readings are ½ / 0 in expectation (LANG_v1 §3.5).
- **PROBE-GAIN-S:** §218's PROBE-GAIN-L formula with PROBE-VALID-S(1) and the final-state check: pooled over §218's C1 band (DIRECT-SYM's), ACC(K′ D+2) − PROBE(1) ≥ 0.35.
- **SERIAL-ONSET-S:** §218's ON(D) rule (slope ≥ 0.6, ON(8) − ON(4) ≥ 3) with PROBE-VALID-S(t), t 0–2.
- **VB-C1-L-S** = DEEP-L ∧ PROBE-GAIN-S ∧ SERIAL-ONSET-S ∧ SERIAL-STEP-L ∧ CONSUMPTION-FRONT ∧ STALL / NO-CLOCK-L ∧ NO-PRECOMPUTE-L. The non-probe components are §218's registered readings, as score_s218 reads them.
- **Under S1, PROBE(t) itself is §218's registered probe output.** Only its validity certificate changes, so PROBE-GAIN-S and SERIAL-ONSET-S have exactly §218's meaning.
- **Wording:** *"under a certified probe (S; lin and content-equivariant xor plants certified on two reference models and on A1; plant strength fixed by an oracle rule), … one seed; a successor reading to §218, whose VB-C1-L is not met by a control-design flaw; never C2 / C4"*.

## 5. Derivation summary [tags]
- **Frame Lemma [PROVED at t = 0]** (218-25): order-dependent plants are unreadable by an order-blind readout.
- **Content-equivariant plants are readable by it [DERIVED; construction in REVIEW_PS §1(b)].**
  - Four ReLU units in block 10's MLP write φ(r_j) = |a_j + b_j| − |a_j − b_j| into each slot row (a = r·u, b = r·u2; φ > 0 iff the signs agree).
  - A block-11 head at the shared "###" decoder row attends with logit ∝ φ_j and copies name(j). So Δ ∝ (κ_own − κ_cross)(φ(r_A) − φ(r_B)).
  - **Assumption:** the per-name score is name-bound, κ_own > κ_cross. This is evidenced by anti-lin LATE 1.000 on both models (218-24), which would be ½ without such a binding.
  - LayerNorm preserves signs relative to its β-offset exactly (positive homogeneity), and up to second order in ε_rel otherwise.
  - **[MEASURED in feasibility]** cexor at t 0 ×10 → 1.000.
- **Plant validity on A1 needs no orthogonality [PROVED].**
  - The coin T and the signs (s1, s3) come from registered generators. They are independent of each other, of the item (text, display order, names, gold) and of A1. u, u2 and ε depend only on A1's unplanted pool rows.
  - Hence every function of A1's unplanted inputs reads T at exactly ½, however u and u2 align with A1's own answer information. A PROBE-VALID-S pass requires reading the plant.
  - A1's own signal along u and u2 is nuisance for the plant task. It can make the plant harder (the K2 guards at each t catch swamping, conservatively), and it can never produce a pass by itself.
  - "Harder, never easier" is not claimed: it holds only for a nuisance independent of the rest of the base along u, and validity does not need it.
- **D3 directions [DERIVED].**
  - Under S1 the probe is unchanged, so PROBE(t), PROBE-GAIN and ON(D) are §218's statistics exactly; no capacity change, no direction issue.
  - Only PROBE-VALID changes: it can now pass for a working probe. Before, it could not pass by construction.
  - Under S2 the review's direction analysis (REVIEW_P2 §4) applies, which is why the guards in §1 are required.
- **PROBE(0) = ½ for any probe [PROVED, REVIEW_P2 §5]**, conditional on the (A, B) order being display order or an independent coin.

## 6. OC
- **P(VB-C1-L-S | no serial use) ≤ 0.004.** It is set by the non-probe components (CF-L, splices), as §218's v1_oc. Certification adds no false-pass route: it is checked on known plants and on separate models.
- **Under S1** (corrected per REVIEW_PS §3):
  - v1's 0.063 / 0.398 already price a PLANT certificate at ≈ 0.95 (LANG_v1 §6). So PROBE-VALID-S on A1 enters as × P(VALID-S_A1 | CERT) / 0.95 ≈ 0.9 / 0.95 [ASSUMED].
  - P(CERT(S1)) ≈ 0.6 [ASSUMED from feasibility] is a new gate, independent of A1's mechanism. A failed certification counts as a fail (§7).
  - **P(VB-C1-L-S | serial use) ≈ 0.398 × 0.947 × 0.6 ≈ 0.23** (0.38 given CERT).
  - **P(VB-C1-L-S | no serial use) ≈ 0.0025 (≤ 0.004).**
  - **P(VB-C1-L-S) ≈ 0.036** (0.018–0.048 for P(CERT) 0.3–0.8).
  - The S2 branch (S1 not certified, S2 certified; P(CERT S2 | ¬CERT S1) ≈ 0.3 [ASSUMED]) adds ≈ 0.003, for a total ≈ 0.039. **A low-odds successor, worth running because it costs $0.**
- **Under S2,** the review's numbers apply (0.043 / 0.027 / 0.016 for probe-component pass rates of 0.7 / 0.6 / 0.5).
- A null is likely, as in §218. The design's value is that A1's probe readings become readable at all.
- **Realised $: $0.**

## 7. Censoring
- No certifiable probe → successor readings UNREAD.
- PROBE-VALID-S failure, or "plant at ceiling on A1" → PROBE-INVALID → UNREADABLE (not met).
- Every §218 censor on A1 applies.
- A missing file reads pending (215-7).

## 8. Order and cost
1. Registration and freeze.
2. theory/score_s222.py + fixtures.
3. Certification on the 4060 (S1 first; S2 only if S1 fails): K2 runs for m, then K3-style runs on the untrained and reference models at t 0–2, plus the controls.
4. A1's probe parts under the selected S, on the 4060.
5. score_s222.

$0, local. A1's probe parts wait for certification; its non-probe parts run at landing and are not read before §218's scoring point.
