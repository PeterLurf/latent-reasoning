# PROBE2: a probe read path that can certify itself, for §218's A1 (DRAFT, 2026-10-21, not registered)

**Status:** draft, $0, eval-only. Successor to §218's probe readings, which are structurally UNREADABLE under 218-22 → (iii) (Notes 218-18 / 218-24). It is drafted after §218's control failure and before any A1 probe output exists, and must be registered and frozen before any A1 probe part runs. §218's VB-C1-L stays NOT MET whatever PROBE2 reads.

## 0. What failed and why (facts; Notes 218-17, 218-22, 218-24, K3-DIAG)
- §218's PROBE(t) and PLANT use a trainable copy of the coda (blocks 10–11, β/γ) that reads V_t plus the bank. They score each option by the **frozen** decoder path's likelihood of "### name.". The choice is sA − sB.
- **Symmetric plant** (registered): chance at ×1 / ×10 / ×100 (K3). It adds the same code to both option rows, so sA and sB move together. K1: |Δ(sA − sB)| = 0.018 at ×1, 0.22 at ×10, against a per-item name-likelihood prior |sA − sB| ≈ 5.4.
- **Antisymmetric lin plant:** learned at 1.000 (untrained and reference). Wiring is sound: OVERFIT-32, param identity, controls A / B.
- **Antisymmetric xor plant:** chance on the untrained model (0.522), on the trained reference (0.515), and at 3× lr (0.532). Yet the sign-product oracle on the option rows reads 0.89 / 0.98 / 0.998 at ×1 / 1.5 / 2 (K2-anti-xor).

## 1. Derivation [tags]
**D1 (why the registered read path misses xor) [DERIVED, with the empirical check above].**
- The registered decision statistic is Δ = Σ log p_dec(name_A | h_A) − Σ log p_dec(name_B | h_B), with the decoder and LM head frozen. Only h is trainable, through blocks 10–11.
- With the antisymmetric xor plant, the label is y = sign(a) · sign(b), where a = (r1 − r2)·u and b = (r1 − r2)·u2.
- To realise y the trainable blocks must produce, at the option positions, a representation whose frozen-decoder likelihood difference flips sign with the product sign(a)·sign(b). Two obstacles:
  - (i) it must overcome a per-item name prior about 25× the planted shift at ×10 (K1);
  - (ii) the frozen readout is a fixed, near-monotone map of h along the directions that move name likelihood, so the product has to be computed inside two attention / MLP blocks and then written along exactly those directions.
- Nothing in the architecture forbids this [the representability is ASSUMED-possible]. But K3 shows it is not found within the registered budget, while the linear case (one direction, no product) is found at once.
- So the limit is the read path's effective capacity under the registered recipe, not the information [MEASURED].

**D2 (why PROBE2 can read xor) [DERIVED].**
- PROBE2 keeps the same trainable coda copy, so it has access to every slot of V_t plus the bank. That is the registered probe's reach, so information anywhere in the state stays reachable.
- It replaces the frozen-decoder readout with a learned head: g(hA, hB) = MLP([hA, hB, hA − hB, hA ⊙ hB]) → 2 logits.
  - The input contains the elementwise product. With the antisymmetric xor plant, hA − hB carries ±2εa-type and ±2εb-type components through the residual stream. Hence (hA − hB)⊙(hA − hB)-type and hA ⊙ hB terms carry a·b-type components.
  - A one-hidden-layer ReLU MLP represents sign(a)·sign(b) on its margin region exactly (XOR needs 2 hidden units).
- No name likelihood is used, so the name prior is absent by construction.
- [PREDICTED] PROBE2 learns anti-lin and anti-xor at ×1.5 to ≥ 0.90 within 2,000 steps on the untrained model. This is the feasibility test.

**D3 (direction of PROBE2 relative to the registered probe, per reading) [DERIVED].**
- PROBE2 has weakly more decision capacity, plus a learned head.
- **PROBE2-VALID (positive control):** easier. That is the point, and it is certified on known plants.
- **PROBE2(12) final-state check** (PROBE(12) ≥ ACC − 0.05): easier.
- **PROBE2-GAIN-L = ACC(K′ D+2) − PROBE2(1) ≥ 0.35:** PROBE2(1) ≥ the registered PROBE(1) in expectation (more capacity), so the gain is **harder** to pass. Conservative against precompute: a precomputed answer that the registered probe could not read is now caught.
- **SERIAL-ONSET2 / ON(D)** (the first t with PROBE2 ≥ 0.90 at all later t): onsets move **earlier or stay the same**.
  - Under true serial use (the answer forms at about t ≈ D), onsets track computation and the slope stays about 1.
  - Under a hidden precompute that PROBE2 can read, onsets move early at every D and the slope falls.
  - So the slope bar (≥ 0.6) gets **harder** to pass under precompute and is unchanged under serial use.
- **PROBE2(0) = ½:** the §3.5 exchangeability theorem holds for **any** probe (SYM v1 makes V_0 carry no option-asymmetric information about reachability). It stays as the null check.

## 2. Certification (before any A1 probe output; all known-answer; untrained model + the G-L1c reference model only)
- **Plants:** antisymmetric lin and xor, with u / ε from plant_dirs (registered).
- **m:** the 218-22 m-rule on each certification model (K2-anti-lin ≥ 0.985: 1.5 on the untrained model; the reference's own value from its K2-anti).
- **PROBE2-CERT holds iff, on BOTH models:**
  - LATE(PROBE2 anti-lin, ×m) ≥ 0.90 and LATE(PROBE2 anti-xor, ×m) ≥ 0.90, at t 0–2, under the PROBE2 recipe at the registered budget (10,000 steps, bs 64, the registered streams, LATE = the last three held-out evaluations);
  - and PROBE2(0) without a plant is 0.50 ± 0.03.
- If PROBE2-CERT fails, no successor readings exist, A1's probe readings stay §218's (UNREADABLE), and it is reported.
- **On A1 itself:** PROBE2-VALID(t) is read per model with the antisymmetric plant at the certified m (lin and xor ≥ 0.90), exactly as §218's PROBE-VALID with the new read path and plant.

## 3. Readings on A1 (DIRECT-SYM where defined)
- **PROBE2-GAIN-L:** pooled over §218's C1 band (DIRECT-SYM's, as registered), ACC(K′ D+2) − PROBE2(1) ≥ 0.35. Requires PROBE2-VALID(1) and the final-state check.
- **SERIAL-ONSET2-L:** §218's ON(D) rule with PROBE2 (slope ≥ 0.6, ON(8) − ON(4) ≥ 3). Requires PROBE2-VALID(t), t 0–2.
- **VB-C1-L2** = DEEP-L ∧ PROBE2-GAIN-L ∧ SERIAL-ONSET2-L ∧ SERIAL-STEP-L ∧ CONSUMPTION-FRONT ∧ STALL ∧ NO-PRECOMPUTE-L. The non-probe components are §218's registered readings, taken as read by score_s218.
- **Wording:** "under a certified option-choice probe (PROBE2; lin + xor plants certified on two reference models), …". §218's VB-C1-L stays NOT MET (a control-design flaw); VB-C1-L2 is a successor reading, one seed, never C2 / C4.
- **No spend triggers.** A2 and B1 stay under §218's rules and the user's decisions; VB-C1-L2 does not open Stage B.

## 4. OC (to be computed in v1)
- PROBE2-GAIN-L at ≥ 1,000 pooled items per band: under "answer readable after pass 1" PROBE2(1) ≈ ACC, so P(pass) ≈ 0. Under true serial use, P(pass) is as §218's PROBE-GAIN-L.
- VB-C1-L2's P(pass | no serial use) ≈ §218's 0.004 or lower (a stronger probe).
- To be recomputed with §218's v1_oc.py and PROBE2's certified power.

## 5. Censoring
- PROBE2-CERT failure → no successor readings.
- PROBE2-VALID(t) failure on A1 → UNREADABLE (not met).
- Every §218 censor on A1 applies.
- A missing file reads pending.

## 6. Cost and order
- **Cost:** $0, local.
- **Order:** feasibility test (2k steps, untrained, ×1.5 / ×10, Audit-approved) → v1 with proofs / red team → registration and freeze → certification (both models, 10k steps) → A1 probe parts under PROBE2.
- A1's probe parts wait. Its non-probe parts run at landing.

## 7. Open
1. Should the head also see a pooled readout over all slots, in case precompute sits off the option rows? The coda copy already attends over all slots before hA / hB, so this is likely redundant.
2. lr 1e-3 for the head vs 1e-4 for the coda copy: should both be fixed now?
3. Should DIRECT-SYM get PROBE2 readings? Its §218 role is the C1 band only, so its probe readings would be descriptive.
