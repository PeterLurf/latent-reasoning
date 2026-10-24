# REVIEW_HC: red team of HC_draft.md v0 (43cce7d8) (2026-10-24)

Reviewer: Theory red-team subagent. Read: HC_draft.md, hc_ckpt_check(.out.json), hc_c1_noise_cpu(.out.json); THEORY §211 / §213 / §224 / §230 rows, Notes 230-1 / 230-2, the §213 HISTORY-USED / OLD-SWAP notes (grep only); boxes/vbank.py rollout; boxes/vbank_eval.py (part_vswap, make_pairs, main); boxes/vbank_diag.py (part_d3hist, part_khorizon, main); theory/score_s211.py header; NORTH_STAR C1. Unblinded files only (no s130 TEST replays, no s162 outputs, no Note 216-5 sealed set). Literature's addendum (LIT_FIXEDHOLD_NOVELTY_2026-10-21.md, "HOLD-C1" section) is folded into issue 12.

**Check script:** `review_scripts/rv_hc_margins.py` → `rv_hc_margins.out.json`. CPU, 2 threads, idle priority. It reads only committed eval_vb.json files and imports the draft's own hc_c1_noise_cpu functions. It recomputes the OC products, the minimum detectable harms, the paired item-bootstrap noise of slope_h, and two unquantised onset statistics.

## Verdict: GO to v1 with fixes

The question is right, and so is the model family. Stage A is cheap. No issue is fatal. But issues 1–7 must be fixed before registration:
- Stage A cannot run as planned: the 4060 breaks the user's local-short rule, and it cannot hold K′ 128 at bs 32.
- One decisive statistic (slope_h) cannot detect onset harm.
- Two derivation steps misread or overstate the evidence (own_iv direction; "harmless mixing").
- The s2 "replicate" is not a replicate.
- The predicted main failure mode, COEXIST-SHIFTED, is not fully pre-registered.

## Issues, ranked by severity

### 1. [MUST] Stage A on the local 4060 breaks the standing rule and is infeasible at the registered shape. Move it to one cloud 3090.
- **The rule.** ~14–16 h of evals is a long run, not a short test. The box is crash-prone (0x133 crashes). The user's rule is: local cards for short, memory-checked tests only; long work on the cloud.
- **Memory.** khorizon at K′ 128, bs 32 peaks at **0.538 / 0.541 of a 24 GB 3090** (s230_fhftfix_s1 / s224_fhft_s2 `k_tried`), i.e. ≈ 13 GB.
  - The 4060 has 8 GB. The c1hold K′ 128 rollout cannot run there at bs 32.
  - The khorizon fallback (halved bs) then breaks the K2 identity gate, which needs bs 32 to equal part_vswap bitwise.
- **Card.** The 3090 is also the card class of every HOLD / DEPTH / NOLOSS file. The base re-reads s224_base_s1_re / s2_re already exist on the 3090 (acc / probe / stall; card effect vs 3070 ≤ 0.0012). So a 3090 plan puts **every reading on one class**, and G-BASE is mostly done already.
- **Fix: one 3090 host, staged dirs hc_<run>_3090/.** The draft's never-the-committed-dir rule is kept. Timings are scaled from committed part_sec: 3070 → 3090 measured ×0.55 on acc / probe; khorizon 3090 ≈ 2,500 s.

| step (one 3090) | models | per model | total |
|---|---|---|---|
| upload 8 × 503 MB model.pt + args + calibration; env; smoke | – | – | ≈ 0.6 h |
| acc, probe, vswap, stall (full battery; acc / stall re-runs also give a free same-class reproducibility check vs the committed 3090 files) | HC s1, FT-CTRL s1 (s215_ftctrl_s1's model.pt), HC s2 (issue 4), FT-CTRL s2 | ≈ 300 + 1,700 + 90 + 20 s ≈ 0.59 h | 2.4 h |
| vswap + acc / stall (probe exists on the 3090 in s224_base_s*_re) | BASE s1, BASE s2 | ≈ 0.12 h | 0.25 h |
| part_d3hist (existing, G-D'd; issue 3) | HC s1, FT-CTRL s1, HC s2, FT-CTRL s2 | ≈ 165 s | 0.2 h |
| c1hold (SWAP-HOLD + arms; issue 6) | same 4 | ≈ 0.15 h (≈ 4.4k pairs at K′ 128 ≈ 0.18 × khorizon) | 0.6 h |
| RD-FT s1 / s2 probe + vswap (descriptive; optional) | 2 | ≈ 0.5 h | 1.0 h |
| **Stage A total** | | | **≈ 4.1 h without RD-FT, ≈ 5.1 h with it** |

- **Cost.** At $0.22/h: **≈ $0.90 central without RD-FT, ≈ $1.15 with it.** Host-speed variance (~2×, memory) gives ≤ $2.3. **Cap $2.5.**
- **Order:** s1 set first (≈ 1.7 h), then drain, then score COEXIST(s1). The s2 set runs on the same host regardless (≈ $0.4; it is needed to read a SPLIT).
- **If Stage B is kept (issue 4):** run it on the same host after the s1 score. That avoids a second setup, and one host is also one class.
- **Optional:** two 3090 hosts in parallel (s1 / s2) cost the same and take half the wall clock (memory: parallel cheap hosts).
- Ship eval files only. The staged dirs have no log.jsonl, so the drain cannot blank a training log.

### 2. [MUST] slope_h cannot serve as a C1-KEPT statistic. Replace it with an unquantised onset statistic.
- slope_h is 1.2857 in all 8 models because it is an OLS of **medians on T_SET**. It moves only in jumps of about one T_SET level per height.
- The paired item bootstrap of the **difference** for an FT pair has SD 0.039–0.082 [MEASURED, rv_hc_margins c_d_pairs], even though the point difference is 0 in all 4 pairs. Item-level onset agreement within an FT pair is only 0.74–0.79, because the probe is retrained each time.
- So δ = 0.364. Its 90 %-fail harm is **0.583**, i.e. 45 % of the slope. P(pass | −0.40 harm) is **0.38 for one lineage**: the draft's 0.146 assumes both lineages are harmed.
- **Fix:** keep slope_h ≥ 0.8 inside the VB-C1 absolute (no rule change), but drop it from C1-KEPT. Add two statistics, both MEASURED on the committed 3070 FT pairs:

| new decisive S | point (s1 / s2 base) | max FT \|d\| | max paired SD | suggested δ | harm |
|---|---|---|---|---|---|
| slope of per-height MEAN onset (n 3–8, heights ≥ 30 items) | 1.134 / 1.094 | 0.017 | 0.031 | **0.10** | ↓ |
| mean onset (n 3–8, items with onset) | 4.568 / 4.782 | 0.049 | 0.012 | **0.10** (T units) | ↓ (precompute) |
| PROBE(3, 4, 6) mean | 0.602 / 0.570 | 0.007 | ≈ 0.005 (binomial, not bootstrapped) | **0.03** | two-sided |

- **PROBE(3–6) should be decisive.** F7's ambiguity concerns splice arms, not the probe. F8 itself says PROBE reads V_t plus the bank, so redistribution stays visible to it.
- With these statistics, an onset harm of ≥ 0.15 in either lineage is caught with P ≥ 0.9.

### 3. [MUST] Key risk (Q1): VALUE-SWAP stays a fair test, and the existing arms already separate "rejects an inconsistent splice" from "lost serial use". F8 has own_iv backwards.
- **Fairness.** SERIAL-STEP (ii) tests whether the newest state carries the computation. That is what VB-C1 registered, and the base models pass it as RNN-in-function: d3hist newest_follow 1.000, bank / old_follow 0.000 [MEASURED, s213 / s215 eval_d3hist].
  - History use alone does not break (ii). §211 N2 was HISTORY-USED with follow_ii 1.000.
  - In the history-using PA cells, OLD-SWAP follow was ≤ 0.016 (THEORY, the §213 line "support read, not content read").
  - (ii) can fail by splice rejection only if HC reads **content** from older versions.
  - F5's bank_ablate 0.641 (s2) does not show that. Masking versions 1..t−2 can hurt through a support read (attention renormalisation), as in the PA cells.
  - So (ii) is fair and stays decisive for COEXIST.
- **own_iv is misread.** Arm (iv) = (ii) + V_0 masked after t (vbank_eval).
  - Under (iv), "own" can come only from the recipient's V_1..V_{t−1}. So **own_iv ↑ = history restore (F7's mechanism)**, not recompute from V_0.
  - Recompute-from-V_0 is **own_ii ↑ with own_iv ↓**, the registered RECOMPUTE-V0 label in score_s211.
  - Fix F8 and §4.3. own_iv becomes the first discriminator.
- **Fix: register a pre-stated decomposition**, used only to label a (ii) failure:
  - (a) own_iv / own_ii (history vs V_0);
  - (b) **part_d3hist OLD-SWAP** (donor V_1..V_t at readers ≥ t + 2, recipient V_t): this directly measures history content reads, it exists, and it is G-D'd;
  - (c) arm (v);
  - (d) **arm (v) with V_0 masked after t** (the draft's Q4).
  - Run part_d3hist (also gives bank_ablate_slva / self / other) on HC and FT-CTRL, both lineages.
- **Is arm (v) a sound discriminator? Only partly.** With the recipient's V_0 kept, a (v) failure means the model recomputes from V_0. Arm (iv) already shows that.
  - A (v) pass shows **chain-level** latent use only. The donor's resampled leaves are also present in donor V_1..V_{t−1}, so (v) cannot show that step t's state was used.
  - Step-specific use must then come from STALL (stored V_t := V_{t−1}) and the PROBE onset.
  - Fix SWAP-REPAIR as: follow_v ≥ 0.90 ∧ own_v ≤ 0.05 ∧ follow_v(V_0 off) ≥ 0.90 ∧ own_iv ≥ 0.5·own_ii (history, not V_0) ∧ STALL and PROBE / onset C1-KEPT met.
  - Label it **COEXIST-SHIFTED (chain-level serial use; newest state not sufficient)**. It is still not COEXIST.
- **Derivation.** "A C1 statistic can change only through shared weights" (F2 / F9) is true of every eval, so it constrains nothing. The operative fact: pass T_far's SLVA queries / keys are trained by the anchor on a bank whose entries are computing-pass versions, and the read policy is shared across regimes.
  - A change in the computing-pass read pattern is therefore the expected **first-order** effect.
  - F5's size (0.83 → 0.60) refutes F3 / F4's "small cross-term" reading. Retag F4 as [REFUTED by F5].
  - F6 ("mixing is harmless where versions agree") covers natural rollouts only, and in s2 older versions are needed. Retag it [ASSUMED], to be tested by (b).
  - Natural-rollout acc cannot bound how the changed read policy treats off-manifold spliced histories.

### 4. [MUST] s232 is not a replicate of HC-near s2, and is nearly redundant with it.
- The draws logs show **identical batch streams within a seed**: s224_fhft_s2, s224_ftctrl_s2 and s230 / s215 share the same plain_hash at steps 1–2 [MEASURED, draws_log.json].
- s232 would differ from s224_fhft_s2 only in λ 0 for the first 500 steps and 1.6 vs 1.618. Its trajectory is therefore near-identical. §4.6's "HC-near shown beside s232 (FT-replicate spread)" is wrong (memory: dedupe identical trajectories).
- s232's HOLD is also effectively seen: HC-near s2 H(128) = 0.990.
- **Fix (simpler, saves ≈ $1 and a gate):** take **s224_fhft_s2 as HC s2**. Scope: "§224 / §230 anchored far FT, λ ≈ 1.6 (s2: 500-step λ ramp)". Its args diff vs FT-CTRL s2 is {far, fh}.
- Spend the money instead on a blind lineage (issue 5).
- If the exact recipe is insisted on, keep s232, but call it "exact-recipe re-run, same draws", never a replicate, and drop the FT-replicate wording.
- Pairing note: HC and FT-CTRL share draws, so HC − FT-CTRL is a **paired** contrast. The OC's "two independent FTs (√2 sd)" is conservative. Fine, but say so.

### 5. [MUST] Seeds (Q3): two lineages give no variance estimate for a seed-dependent effect. Pre-register s3 now, with its trigger.
- The F5 response is strongly lineage-dependent: bank_ablate n 8 is 0.985 (s1) vs 0.641 (s2).
- For HOLD, no lineage is blind: s1 is §230's confirmed seed, and s2 is the motivating seed (issue 4).
- So the only new evidence in COEXIST is the C1 battery after the FT. For C1 it is new in both lineages; the claim does not "rest on s2".
- Two lineages is the project minimum for the scoped wording ("in 2 lineages"). But the C1 × C6 status upgrade should need one fully blind lineage. An s3 decided by "Audit's ruling at that time" is outcome-dependent.
- **Fix: pre-register now.**
  - s3 = D3-T s3 base (48k), FT-CTRL s3 and HC s3 (λ 1.6 fixed). RD-FT s3 is not needed.
  - Evals: C1 battery ×3, khorizon ×2, d3hist + c1hold on HC s3. One 3090 class. ≈ $3.3, cap $4.
  - **Trigger:** COEXIST(s1) ∧ COEXIST(s2) (blind confirmation before any status upgrade), or SPLIT (s3 decides "2 / 3 lineages" wording).
  - **Censoring:** an s3 base that fails VB-C1 itself = NO-BASE. Report it and do not count it as a COEXIST fail. One retry with s4 is allowed only if pre-stated.

### 6. [MUST] c1hold part (K2): the identity gate cannot run as written; memory and G-D packet.
- **Identity.** part_vswap stores only pooled counts, not per-pair answers, so "bitwise share 1.0 vs part_vswap's per-pair answers" has no reference file.
- **Fix:** inside c1hold, reproduce vswap_class's (n, t) grouping and chunks at bs 32. Compute (ii) once from a K = n + 2 rollout and once from the coda row n + 2 of the K′ 128 rollout. Require:
  - per-pair share 1.0;
  - pooled n_pairs / follow_ii / own_ii / follow_iv / own_iv equal to the same-card eval_vb vswap **exactly**.
- vbank.rollout's bank is list-grown with no K-dependent shapes, so passes ≤ n + 2 should be bitwise across K. The gate verifies this.
- **Memory.** Run the check on the c1hold shape: spliced K′ 128 rollout + the held recipient / donor K = n + 2 rollouts, deepest (n 8, longest) chunk, on the 3090. Expect ≈ 0.55–0.6 of the card.
  - Do not reuse khorizon's fits(), which measures the intact arm only.
  - If bs 32 does not fit: SWAP-HOLD may use a halved bs. follow_ii(n + 2) and agree must then be computed against a same-bs K = n + 2 reference, never against eval_vb.
- **Arms.** Implement (v) with the existing kwargs (edit at t → donor V_t; ver_sub = donor V_1..V_{t−1}; default REPLACE indexing) and (v, V_0 off) via v0_off_after = t. No new rollout code.
- **G-D packet:**
  - @torch.no_grad on the part (insert-above-decorated-def lesson: check with ast);
  - unique top-level key "c1hold" (main() key-collision lesson), with a post-run assert that the arm keys are dicts;
  - add "c1hold" to main()'s choices and dispatch (it needs `run` for the identity read);
  - plain-path smoke (no new flags) of vbank_eval on a base;
  - the main() choices / dispatch diff read in full;
  - a known-answer check: arm (v) on the BASE must give follow ≈ 1.0 (as newest_follow 1.000);
  - commit before deploy, and no module edits while the deploy is in flight.

### 7. [MUST] Pre-register what COEXIST-SHIFTED(s1) launches, and what C1-KEPT reads when a reference is unreadable.
- **Stage B / s3 gating.** The draft prior puts the most likely s1 failure on follow_ii / own_ii (P(COEXIST(s1)) ≈ 0.62). State whether COEXIST-SHIFTED(s1) launches the s2 / s3 cells.
- **Recommended:** yes for the s2 readings (they are evals only under issue 4), so the shifted reading can reach 2 lineages. Its wording is pre-stated: "C1 at chain level, single-state sufficiency lost after the hold FT".
- **Unreadable references.** If a reference statistic is UNREADABLE (vswap < 100 clean pairs, PROBE-INVALID) on BASE or FT-CTRL, that comparison is UNREADABLE and C1-KEPT is not met (rule b). Add it to §6.
- **Clean pairs.** Clean pairs are model-specific, so follow_ii(HC) and follow_ii(ref) use different pair subsets. Report the overlap and follow_ii on the intersection, descriptively.

### 8. [SHOULD] Saturation (Q2): the margins are honest as practical floors, not as "measured noise". Fix the wording, and consider tighter floors.
- **Recomputed [MEASURED, rv_hc_margins]:**
  - joint P(C1-KEPT | no harm, 2 lineages) = **0.954** (reproduced, independence);
  - P(COEXIST | no harm) = **0.662** as written, **0.656** with the VB-C1 absolute counted for both seeds.
- The statistics share items and HC is common to both references, so dependence is positive. The independence product is therefore a lower bound on P(keep): conservative.
- The harm at which C1-KEPT fails with 50 % / 90 % probability:
  - follow_ii / own_ii: 0.020 / 0.020;
  - STALL: 0.020 / 0.021;
  - DEEP_min: 0.020 / 0.022;
  - PROBE-GAIN: 0.030 / 0.041;
  - PROBE(2): 0.030 / 0.045.
- So C1-KEPT detects **small absolute harms (2–4 points)**, not "only large" ones. P(pass | harm) < 0.001 at the draft's harm sizes is true but uninformative: those harms are 5–7× δ.
- δ is 14–100× the null sd for the saturated statistics (follow_ii δ/sd 59; STALL 22; DEEP 14).
- **Fixes:**
  - reword §4.2 / S1 from "within measured noise" to **"within pre-registered margins (0.01–0.03 absolute; measured FT noise ≤ 0.007)"**;
  - optionally set the saturated floors to **0.01**: false-fail P stays ≈ 0 (sd ≤ 0.0014) and sensitivity doubles;
  - restate the OC row as the minimum detectable harm, plus P(COEXIST | harm in **one** lineage). With slope_h that is ≈ 0.38 × the other factors ≈ 0.25, not ≤ 0.15. With issue 2's statistics it is ≈ 0.
- The real saturation risk is that argmax readouts at 1.000 hide a margin loss. If c1hold is being built anyway, log the donor-answer log-prob margin for arm (ii), descriptively.

### 9. [SHOULD] The margin rule's "self-item FTs bound a matched FT from above" is unproven.
- A self-item FT has a different objective, not a larger version of the same perturbation. Only one matched null pair exists (s213 → s215_ftctrl).
- The floors bind every δ anyway, so the margins are unaffected. Retag the sentence [ASSUMED].
- After Stage A, report FT-CTRL s2 − BASE s2 as the second matched null.

### 10. [SHOULD] Single-change diffs and arg nulls.
- FT-CTRL s1 (s215_ftctrl_s1) predates the far arguments. Its args show null where HC has values (far_n, far_t, far_w, fh_dose, fh_lam_cap, fh_target).
- State that null = default-off, and that G-FH-EQ v2 / the §224 flags-off bitwise gate covers the plain path. Cite the gate file.
- Confirm from the queue logs that s215_ftctrl_s1 was trained on a 3090. args.json does not record the card; train_sec 4,878 matches the 3090 FT-CTRL s2 (5,132), which is consistent but not proof.
- An HC s2 under issue 4 has diff {far, fh} vs FT-CTRL s2 and {fh} vs RD-FT s2. That is one change each.

### 11. [SHOULD] S3 / descriptive claims need the registered rule.
- By the registered HISTORY-USED rule (BANK-ABLATE pooled n 6–8 drop ≥ 0.10), HC s1 is **not** history-using: n 6–8 = 0.995 / 0.993 / 0.985, a drop ≈ 0.01.
- HC-near s2 is: 0.999 / 0.978 / 0.641, a drop ≈ 0.13.
- So "a stability objective induced history use" holds in **1 / 2** lineages, and only as support vs content once OLD-SWAP (issue 3b) is read. Reword S3 as "shifted the version reads in 2 / 2 (carry_likeness 0.82 → 0.60 / 0.69); HISTORY-USED in 1 / 2".

### 12. [SHOULD] Novelty (Literature addendum, 2026-10-24).
- **S1: OK, narrow wording.** Viakhirev 2608.18222 shows that a fixed-point objective can "settle but EM = 0": settling and computing are separable. That is the motivation for testing C1 after the hold FT; cite it in §3. No paper measures causal intermediate-state use before vs after a stability objective in a looped model. Replace "within measured noise" (issue 8).
- **S2: NARROW.** 2609.01924 (Jacobian lens on Huginn) already runs a swap-at-recurrence-k persistence test: 30 / 30 to k = 12, with ~2-recurrence memory (recompute from input). Also SCIT 2608.27265 and Liang & Pan 2602.00449: patching at step t, short horizon.
  - Cite 2609.01924, not only 2502.05171.
  - New only as a **state** (not input) splice with a **128-pass** hold.
- **S3: OK as descriptive** (not found). Reword per issue 11.
- No "new" / "first" in any of S1–S3.

### 13. [NIT] Smaller points.
- **RD-FT (descriptive):** it can be dropped from Stage A (−1.0 h, −$0.22). The attribution split HC − RD-FT is only needed if FAR-CHANGED-C1 fires, so run it then. Pre-state that.
- **Contingency arithmetic:** the §6 "P(Stage B launches) ≈ 0.62" disappears under issue 4.
- **K3 G-BASE on the 3090:** acc / probe / stall vs s224_base_s*_re must be bitwise (same class). vswap vs the 3070 counts must be within 0.01.
- **K6:** the stray-CUDA-context note is moot on the cloud. Keep the 'PART x done' waiters and drain waiters armed at launch.
- **§4.3 table:** add the issue 2 statistics. Move slope_h and coverage / gap_8_3 to "VB-C1 absolute only".
- **SWAP-HOLD:** keep it secondary. The anchor never saw spliced histories, so a fail mixes C1 × C6 with off-distribution holding. Report FT-CTRL / BASE (chance at K′ 128) only as context.

## Simpler alternatives considered
- **Evals-only design (recommended core):** HC s1 = s230, HC s2 = s224_fhft_s2. One 3090, ≈ $1, no training. The decisive question, whether C1 survives the hold FT, is fully answered by it. A blind s3 ($3.3) carries the confirmation.
- **Skip c1hold entirely:** run part_d3hist plus the existing (iv) arm for the decomposition, and drop SWAP-HOLD. This saves a build + G-D, but loses the only joint C1 × C6 reading. Keep c1hold, small, with arms (ii) K′ 128, (v) and (v, V_0 off).
- **INJECT port:** rightly rejected (weaker battery, base deleted).
