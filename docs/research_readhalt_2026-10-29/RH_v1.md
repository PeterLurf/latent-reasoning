# READ-HALT v1: C7 (round-trip and edit) and C5 (readout halting) on the §240 language loop ($0, 2026-10-29)

**Status:** Theory's choice under the user's standing rules, on Audit's suggestion. Eval-only, $0 (local 4060). Rules are pre-stated before any output.
- Models: lh_fh_s1 / lh_fh_s2 (§240). TEST2-DF.
- No red-team subagent (eval-only, usage rule); Audit G-D covers the build.

## 1. C7: a faithfully readable thought (NORTH_STAR C7: round trip + edit; decodable is not enough)
- **The translation of thought t** is the value printed at pass t's ':' (argmax over the 11 number tokens), rendered as the step sentence "<S_t> has <x> <unit>." (the R_step format).
- **Re-encode (the model's own write path):** at pass t, replace the model's write with the write computed from the same cue c_t over an augmented text, original text + " <S_t> has <x> <unit>." Passes t + 1..K′ and the decoder then run on the original inputs. Only the pass-t write changes.
- **Cells (1,000 items each):** d 3, t ∈ {1, 2}; d 4, t ∈ {2, 3}. K′ = d, the binding read.
- **ROUND-TRIP (binding):** x = the decoded value x̂_t. Per-item final-answer agreement with intact ≥ 0.95, pooled over cells and per cell ≥ 0.90.
- **EDIT (binding):** x′ = a uniformly random value ≠ x̂_t (seeded). The predicted answer a′ is the generator's answer with S_t := x′ (recomputing later steps). follow(a′) ≥ 0.50 and ≥ floor + 0.30, where floor = P(answer = a′) under intact (≈ 1/11), pooled; per cell follow ≥ 0.40.
- **Control (descriptive):** the augmented text with x = x̂_t used for every pass (not just t), so the re-encode is not just an extra-text effect; also the oracle x = gold v_t.
- **Labels:**
  - C7-PASS (both);
  - ROUND-TRIP-ONLY (edit fails: the text is accepted but not causally specific);
  - REJECTED (round trip fails: the model rejects text-encoded states, cf. Audit's §234 d_oracle finding);
  - per model; overall needs both.

## 2. C5: halting from the model's own readout (NORTH_STAR C5: spends more steps on harder inputs)
- **Rule (fixed now, no tuning):** decode after each K′ = 0..8. Halt at the first K′ where the decoder's max softmax probability over the 11 number tokens is ≥ τ = 0.90. If never reached, K′ = 8.
- **Binding:**
  - HALT-ACC: acc at halt ≥ acc(K′ 8) − 0.02 at each d 1–4;
  - HARDER-MORE: mean K′_halt strictly increases from d 1 to d 4;
  - SAVING: mean K′_halt(d) ≤ d + 1 at d 1–4.
- **Labels, first match:** COSTLY (HALT-ACC fails) / NO-SIGNAL (HARDER-MORE fails) / NO-SAVING (SAVING fails) / C5-READOUT (all three). Per model; overall both.
- **Scope:** the threshold is applied to an untrained readout. That is readout halting, not learned halting, and the claim says "C5-partial (readout halting; τ fixed a priori)". Externalisation ("emits a checkpoint when it helps") is not tested.
- **Descriptive:** d 5–6 (does K′_halt grow, and does it stop early on items it gets wrong?); calibration (max-prob vs correctness).

## 3. Build (Experiments; Audit G-D)
- **Part `c7`:** eval_c7.json {cell: {n, intact_acc, rt_agree, edit_follow, floor, ctrl_aug_all, oracle_rt}}, with per-item records {sig, x_hat, x_edit, a_edit, ans_intact, ans_rt, ans_edit}. Gate: x = x̂_t re-encoding with an EMPTY augmentation reproduces intact bitwise (path check).
- **Part `halt`:** eval_halt.json {d: {acc_k8, acc_halt, mean_khalt, khalt_hist}}, with per-item {sig, maxprob[K′ 0..8], pred[K′]}.
- Both on the 4060, fp32, one eval_bs per model. TEST2-DF d 1–6.

## 4. OC
- **C7:** P(ROUND-TRIP | the write path accepts text-encoded values) ≈ 0.7. Audit's §234 oracle finding argues for rejection, so the prior is 0.4, giving P(C7-PASS, both) ≈ 0.2.
- **C5:** P(C5-READOUT) ≈ 0.6 (the decoder is at 1.000 from K′ = d − 1; confidence should track it).
- Under no effect (random confidence): P(HARDER-MORE) ≈ 0.04.

## 5. Censoring
Missing parts mean pending. Each cell's n ≥ 900, else that cell is VOID.

## 6. Cost
$0.
