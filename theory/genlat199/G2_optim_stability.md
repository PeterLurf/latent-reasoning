# G2: general optimisation and training-stability practice, mapped onto the DeltaMem seed lottery (2026-10-13)

Research subagent for the Theory session (the user's request: "general improvements to latent training ... that would also practically
improve our methodologies"). Read-only on D:\Latent. CPU only (3 threads, idle priority). No GPU, no cloud, no repo writes.

**How to read this note.**
- **Source tags.**
  - **[abstract]**: from the arXiv abstract, fetched 2026-10-13. No method detail is stated as fact from an abstract alone.
  - **[full text]**: read in the paper's own text (McCandlish from the PDF text).
  - **[full text\*]**: read in the arXiv / ar5iv HTML through a fetch-and-summarise tool, with quotes as returned. Re-check these numbers against the PDF before they enter THEORY.
  - **[ours]**: computed here from repo files.
- **Theory classes**, following Part II of `docs/THEORY_SEEDSTAB_2026-10-13.md` (= `seedstab/MATH_selection.md`):
  - **TILT**: changes μ_Z.
  - **√**: changes σ_Z (noise, batch).
  - **WINDOW**: changes T_w (a √-lever in Part II's window regime).
  - **REMOVE-BASIN**: removes the contractive basin.
  - **ESCAPE**: barrier crossing (Part II §3.4).
  - **NEUTRAL**: provably or plausibly no effect on P(basin).
  - **METHOD**: measurement only.
- **Mechanism status:** DERIVED-UNDER-ASSUMPTIONS (DUA) or CONJECTURE. Recommendations with no statable mechanism are ranked lower.
- **Not repeated here** (see the existing notes):
  - Grokfast (LIT_S196);
  - Power / Omnigrok / Varma / Slingshot / Softmax Collapse / L2-SP (LIT_SEED_RELIABILITY, LIT_S196);
  - restart cost and early discarding in general (Part II §6);
  - curricula and budget annealing (LIT_OPERATOR_FIRST);
  - Summers & Dinneen and Frankle as background (LIT_SEED_STABILITY);
  - Gradient Flossing (LIT_SEED_STABILITY_CANDIDATES);
  - pass-index and scaffold issues (LIT_SCHEDULE_FADE);
  - random K and truncation in Huginn (LIT_RAND_K).

---

## 0. Bottom line

**The three highest-value actions, all under about $3:**
1. **NULL-REROLL (METHOD; neither tilt nor √).** Before any single-seed "rescue" of s3 is read as a lever effect (§196 GROKFAST-S3 is running now), measure what an *inert* perturbation does to s3.
   - Part II §3.2's flip formula implies an inert change, which re-draws the rounding noise, makes s3 enter and meet §196's ACCELERATES bar with probability **0.12–0.60** (ρ_Z 0.9 → 0) (§2.1).
   - Cost: 2 inert replays to 25k on a 3090, about $2.
2. **NO-DECAY group for biases, gains and gates (TILT, small, known sign; free), plus a free early diagnostic.**
   - **Our harness weight-decays every parameter** (single AdamW group, `fsq120.make_opt`, wd 0.01), including the delta heads' retention logits (gate.bias[:6], init 5).
   - [ours] In all 15 final models, decay accounts for most of these logits' motion: the pure-decay path ends at 4.8817 by 48k.
   - **All 5 failed models sit on the pure-decay path** (mean excess −0.003 … +0.013). **All 10 converged models sit above it** (+0.016 … +0.037). That is perfect separation (exact one-sided p = 1/3003, post hoc among 14 fields inspected, so about 0.005 after Bonferroni).
   - Within RMS: s1 +0.019, s2 +0.022, s3 −0.002.
   - **Lever:** decay is a deterministic push on α toward forgetting, i.e. toward the contractive basin. It acts directly against the only first-order escape route of the open-loop regime (Part I Lemma 5.2(ii): α → 1). It is small in the decision window and free to remove.
   - **Diagnostic:** the excess over the decay path, Δ_α(t), is a zero-cost candidate early order parameter.
3. **RE-WARM s3 (ESCAPE vs TILT discriminator, about $0.8).** Restart s3's final model at constant LR (1e-4; 3e-4) for 10k steps.
   - **Part II** (C is an attractor; entry only by Kramers escape, with a data bound k ≤ 1.25e-5 per step) predicts P(entry in 10k at 1e-4) ≤ 0.12.
   - **Part I §5.4's drive threshold** (ε(∞) = ε₀ exp(2∫ηD)) predicts entry, if the late-pass drive D persists. That is plausible: s3's t8 aux accuracy is 0.316.
   - The outcome decides whether WSD / longer high-LR phases (a TILT in probit under Part I) are worth adopting.

**One-line verdicts on the slice:**
- **Weight averaging cannot move the lottery.** It is NEUTRAL by construction when not fed back, but it can help final quality and C6.
- **No modern optimiser has seed-variance evidence.** Muon has a grokking-speed result and a plausible mechanism (§2.9).
- **Schedule shape before 6k is irrelevant:** cosine LR is ≥ 0.92 × peak through 6k.
- **Truncated BPTT would be a negative tilt.** Keep full BPTT.
- **Determinism / precision work cannot raise P.** Treat the card class as part of the seed.

---

## 1. New harness facts from this pass (descriptive; final models only; post hoc)

Scripts: `genlat/trunk_drift.py`, `genlat/scalars.py` (output `trunk_drift.json`). Models: the 15 in FACTS.md (10 converged, 5 failed).

### F1: weight decay applies to every parameter, and dominates the delta heads' retention logits [ours]

**The optimiser.** `fsq120.make_opt` builds `AdamW(m.parameters(), lr, weight_decay=0.01)`: one group, no bias / norm exclusion. nested170 calls it with lr_new = 1.

**Decay alone.** With PyTorch's decoupled AdamW, a parameter with no gradient follows b_t = b_0·Π(1 − η_t λ).
- With nested170's schedule (warm-up 100, cosine over 48k), the factor is 0.99414 / 0.99226 / 0.98282 / 0.97633 at 6k / 8k / 20k / 48k.
- So the α-logit goes 5 → 4.971 / 4.961 / 4.914 / 4.882.

**Effect on the realised α.**
- d log α / d logit = 1 − α = 0.0067 at logit 5. So the decay shifts every token's log α by −0.0002 (6k) to −0.0008 (48k), assuming the logits stay near 5.
- Per pass of L writing tokens (Part I Thm 4: ‖P‖ ≤ Π_τ α_τ), Δλ_pass is −0.006 (6k) / −0.018 (20k) / −0.025 (48k) at L = 30.
- It scales with L: at L = 300 (ProsQA-length writes, *if* that harness shares make_opt and per-token α; not checked) it would be 10× larger.
- For scale: the λ_eff boundary between basins is P ≥ −0.009 vs C ≤ −0.023 (Part II §1).

**Excess of the mean α-bias over the pure-decay value 4.8817 (48k):**

| converged | Δ_α | failed | Δ_α |
|---|---|---|---|
| s190_rmsnopass_s1 | +0.037 | s188_rwwin_s1 | +0.013 |
| s190_rmsnopass_s2 | +0.035 | s182_hold_s1 | +0.004 |
| s184_reweight_s2 | +0.026 | s185_den_s1 | +0.004 |
| s187_nopass_s1 / s2 | +0.022 / +0.022 | **s185_rms_s3** | **−0.002** |
| s185_rms_s2 / s1 | +0.022 / +0.019 | s182_nofade_s1 | −0.003 |
| s184_reweight_s1 | +0.021 | | |
| s182_window_s2 / s1 | +0.019 / +0.016 | | |

**Reading.**
- Decay moved every model's α-bias down by 0.118. The gradient's net push was +0.016 to +0.037 in converged models and about 0 in failed ones. In converged models the gradient pushes retention up against the decay; in failed ones it does not.
- **This is a consequence readout, not a cause.** It fits the C basin having no pressure to retain. The β-logits (init 0) are unaffected by decay.
- The gate weight norm also separates: converged 2.96–3.53, failed 2.69–3.06 (AUC 0.94). That difference is gradient-driven; decay moves the norm by only about 2.4%.
- **Limits:**
  - The realised α also depends on gate.weight·u, which was not measured here. `DeltaMem.alpha_rec` records the realised mean α per head and can be enabled at eval.
  - Recipes are confounded across groups. The within-RMS seeds agree.

### F2: trunk drift from pretrained GPT-2 is the same in both basins [ours]
- Relative drift ‖θ − θ_GPT2‖/‖θ_GPT2‖ per block:
  - converged: block 0 0.072–0.075, block 6 0.124–0.127, mean 0.103–0.107;
  - failed: 0.072–0.074, 0.122–0.125, 0.103–0.107.
  - AUC of the block mean is 0.54.
- The memory qkv drift from its block-6 copy (AUC 0.44) and ‖W_o‖ (AUC 0.40) do not separate either.
- **Reading:** under AdamW's near-sign-magnitude updates (Balles & Hennig 1705.07774 [abstract]: Adam's direction is the sign; its magnitude is set by the relative variance), drift is set by the LR schedule, not by which circuit is learned.
- **Consequence:** there is no final-weight premise for "P is more parameter-efficient than C". So an L2-SP or weight-decay *efficiency tilt* (Varma 2309.02390; see LIT_SEED_RELIABILITY) has no support at this level (§3).

### F3: the clip binds on every step in the decision window, and per-step weights vary by about ±28% [ours, §194 s2 = s190_rmsnopass_s2 log.jsonl]

| steps | median gn_preclip | p10–p90 | SD(log gn), detrended | clip_frac_100 |
|---|---|---|---|---|
| 2–4k | 5.57 | 3.99–9.34 | 0.250 | 1.00 |
| 4–6k | 5.10 | 3.15–8.00 | 0.307 | 1.00 |
| 6–8k | 4.35 | 3.33–5.26 | 0.186 | 1.00 |
| 24–48k | 2.82 | 1.14–7.84 | 0.839 | 0.94 |

- With clipping always binding, AdamW's input is g_t/‖g_t‖. The effective per-step weight 1/‖g_t‖ therefore has a CV of about 0.25–0.31 in the decision window.
- Note CLIP-1 (THEORY) found a detrended corr(gn, aux_ce) of +0.67 over 1–12k. So high-aux-loss batches get down-weighted in exactly the window where selection happens (§2.5).

---

## 2. Ranked recommendations (top 12)

Each entry gives: evidence; math basis (i) source level, (ii) mechanism for us with its Part II class, (iii) falsifiable prediction; whether we tried it; cost; cheap test.

### 2.1 NULL-REROLL: calibrate single-seed rescue tests with inert perturbations (METHOD) — rank 1

**Evidence.**
- Summers & Dinneen 2103.04514 [abstract]:
  - all sources of nondeterminism have similar effects on model diversity;
  - "even one-bit changes in initial parameters result in models converging to vastly different values".
- Zhuang et al. 2106.11872 [abstract]:
  - tooling nondeterminism leaves top-line accuracy unchanged but makes some sub-distributions "far more sensitive";
  - determinism overhead is up to 746% / 241% / 196% across GPU types.
- Ours: GPU-class and kernel-order changes flip outcomes (FACTS); same seed + same card replays exactly.

**Math basis.**
- **(i) Source and assumption.** Part II §3.2 gives P_flip = 2[Φ(z) − Φ₂(z, z; ρ_Z)] (PROVED for the Gaussian decision variable, MC-checked). ρ_Z is the share of decision variance accumulated before rounding chaos decorrelates two runs. The sources are empirics only.
- **(ii) Mechanism (DUA).** An inert change (a different micro-batch split, or a 1-ulp perturbation at step 0) leaves μ_Z and σ_Z unchanged and re-draws the noise realisation after decorrelation. For a seed that failed:

  P(enter | inert) = [Φ(z) − Φ₂(z, z; ρ_Z)] / (1 − Φ(z)) = **0.60 / 0.40 / 0.17** at ρ_Z = 0 / 0.5 / 0.9 (z = 0.25).

  - §196's ACCELERATES also needs entry ≤ 25k and final < 0.1. That has q ≈ 0.7 across converged models (7/10 have final < 0.1), and 2/2 within RMS.
  - **So an inert perturbation reads ACCELERATES with probability ≈ 0.12–0.60.**
  - The exact same-seed control proves the filter *caused* the trajectory change. It does not separate "the filter tilts toward P" from "any perturbation re-rolls".
- **(iii) Prediction.**
  - If chaos decorrelates early (k* ≪ T_d, Part II P10), at least 1 of 2 inert s3 replays enters by 25k with probability 1 − (1 − 0.6)² = 0.84.
  - If ρ_Z ≥ 0.9, 0/2 enter with probability ≥ 0.68.
  - 0/2 supports single-seed designs. Any entry means single-seed rescue readings are uninterpretable and lever claims need fresh-seed panels.

**Tried?** No. §196's registration relies on the exact control (THEORY §196 "Exact control (Audit)").

**Cost.** About $1 per 25k replay on a $0.18/h 3090 (0.74 s/step). Zero code: `--micro 2` instead of the auto value changes only the summation order.

**Power (for any lever claim) [ours, exact Fisher one-sided, α = 0.05]:**

| comparison | n per arm | power |
|---|---|---|
| p 0.6 → 0.95 | 10 / 15 / 20 | 0.43 / 0.67 / 0.81 |
| p 0.6 → 0.9 | 20 | 0.62 |

- **One-arm against a known base rate p₀ = 0.6:** 6/6 is needed for p < 0.05 (P(6/6 | 0.6) = 0.047), and a true p = 0.95 passes only with probability 0.74.
- **The cheapest decisive lever design is therefore "6 fresh seeds, all enter"** (about $6 at 25k). Alternatively a validated 8k proxy (Part II P2), or the small-scale proxy (§2.7).

### 2.2 NO-DECAY group + Δ_α log (TILT, small, known sign; plus a free early diagnostic) — rank 2

**Evidence.**
- F1 [ours].
- He et al. 1812.01187, "Bag of Tricks…" [full text\*]: "no bias decay" applies weight decay only to conv / FC weights, excluding biases and BN γ / β. It was part of their large-batch heuristics; no standalone number is attributed to it.
- Wang & Aitchison 2405.13698 [full text\*]: AdamW weights are an EMA of updates with τ_iter = 1/(ηλ), and the optimal τ in epochs is roughly constant. PyTorch's decay is η_t·λ.
  - Ours: ηλ = 1e-6, so **τ = 10⁶ steps ≫ 48k**. Decay is effectively inert on matrices, but not on a large-valued scalar like the α-logit (5.0), whose decay force η λ b is 5× that of a unit-scale weight.
- TRM for contrast [full text\*]: wd 1.0 (Sudoku / Maze) at lr 1e-4, so τ = 10⁴.

**Math basis.**
- **(i) Source.**
  - Bag of Tricks is a heuristic. Wang & Aitchison is a derivation (an EMA identity, exact for decoupled decay) plus empirics.
  - Part I Thm 4 (PROVED): the open-loop delta map has ‖P‖ ≤ Π_τ α_τ.
  - Part I Lemma 5.2(ii) (PROVED under smoothness): in the open-loop-like regime the only first-order route to more late-pass separation is changing P, i.e. α → 1, β → 0 / 2.
- **(ii) Mechanism (DUA).**
  - Decay adds a constant, seed-independent force −η_t λ b on the α-logit. That is a deterministic drift toward lower α, i.e. more per-pass contraction. Its size is Δλ_pass ≈ −0.006 by 6k and −0.018 by 20k at L = 30.
  - In Part II terms it is a negative constant inside b(t) (a TILT toward C), acting on exactly the coordinate Lemma 5.2(ii) names as the escape route.
  - **Magnitude is small** in the decision window: about 2% of the needed climb from the init's λ ≈ −0.3 to about 0. It grows to the scale of the basin boundary by entry time (about 19k).
  - Assumption: the α-logit is a component of the selection coordinate m. That is plausible by Thm 4, not shown.
  - Decay on the log-gains s_add / s_del pulls g toward 1, not 0. That semantics is incoherent either way.
- **(iii) Predictions.**
  - (a) **Lever:** |ΔP| ≤ 0.1 at panel scale. That is undetectable at n ≤ 10 (§2.1 power), so adopt it on hygiene grounds as a gated flag for new registrations, not as a tested lever.
  - (b) **Diagnostic (the main falsifiable one).** Δ_α(t) = mean α-logit(t) − 5·Π_{s≤t}(1 − η_s λ). Logged every 100 steps, it separates eventual P from C seeds by 8k with AUC ≥ 0.8 (Part II P2's bar).
    - With no-decay, use Δ_α(t) = mean α-logit − 5.
    - **Falsified if** the 8k AUC is < 0.65, i.e. the push appears only after entry (a consequence, not a precursor).

**Tried?** No. No bias / norm exclusion exists in fsq120, nested170 or inject169.

**Cost.**
- About 10 lines in make_opt: a no-decay group for every parameter with ndim ≤ 1, plus mem.gate.bias, mem.s_add / s_del / add_gain and all LN gains and biases.
- Also gate_* bitwise flags-off equality.
- Logging Δ_α and the realised mean α (`alpha_rec`) costs nothing.

**Cheap test.**
- Prospective: log Δ_α on every new nested run, at zero cost.
- Retrospective: the 4060 pilot196_s3_stage2 checkpoints (kept until Theory reads 196-2) and any surviving 8k checkpoints give Δ_α(t) by reading only gate.bias from each .pt (CPU, seconds).

### 2.3 RE-WARM s3's final model; WSD / constant-LR as the follow-up (ESCAPE vs TILT discriminator) — rank 3

**Evidence.**
- Gupta et al. 2308.04014 [abstract]: re-warming first raises loss, then helps downstream (Pythia-410M).
- Ibrahim et al. 2403.08763 [abstract]: re-warm + re-decay + replay matches re-training.
- SGDR 1608.03983 [abstract]: warm restarts.
- WSD: MiniCPM 2404.06395 [abstract] introduces it.
- Hägele et al. 2405.18392 [abstract]: constant LR + cooldown scales like cosine, and SWA along the trajectory helps.
- Wen et al. 2410.05192 [abstract]:
  - a river-valley derivation (the stable phase progresses along the river under large oscillations; the decay reveals it);
  - WSD-S.
- Huginn 2502.05171 [full text\*]: the main run used a constant schedule after a 4096-step warm-up. After two failed runs it dropped the peak LR to 4×10⁻⁵.

**Math basis.**
- **(i) Source.** Restarts and re-warming are empirics; Wen is a derivation under an assumed landscape.
- **(ii) Mechanism.** Two of our own models disagree, and this test separates them.
  - **Part I §5.4 (PROVED r = 2 kernel; CONJECTURE mapping):** ε(∞) = ε₀ exp(2∫η D). If ln ε₀ ~ N(μ, s²), then P = Φ((μ − ln ε_needed + 2∫ηD)/s): **linear in ∫ηD inside Φ, a TILT.**
    - Cosine gives ∫η over 6–48k = 1.81 (lr × steps); WSD with a final 20% linear decay gives 3.72 (×2.06).
    - A 10k re-warm at 1e-4 adds 1.0, against the original 20–48k tail's 0.66.
  - **Part II (C is an attractor):** entry from deep in C is ESCAPE only.
    - The rate is about e^{−2ΔU/σ²} with σ² ∝ η/B.
    - Data bound: 0 escapes in 6 failed runs × about 40k steps gives k ≤ 1.25e-5 per step at the LRs they had.
- **(iii) Predictions.**
  - Arm R1 (constant 1e-4, fresh AdamW, 100-step warm-up, γ = fade = 1, 10k steps from s185_rms_s3/model.pt):
    - Part II predicts P(entry(0.5) within 10k) ≤ 1 − e^{−0.125} = 0.12;
    - Part I predicts entry if D is still large (s3's t8 aux accuracy 0.316, t6 0.587).
  - Arm R3 (3e-4) is a practical rescue attempt: both models allow escape there, Part II only through the exponent.
  - **Discriminator:** R1 enters → Part I's drive account holds, and WSD (a TILT under it) is worth a fresh-seed panel. R1 never enters and R3 does → escape (barrier) behaviour, so WSD is not a lever.
  - A late R1 entry (> 8k) with a shrinking aux loss beforehand is Part I's signature. Entry right after the LR jump is Kramers'.

**Also (not new):**
- the LR during the open loop sets the C head start in τ = η·steps units (Part II C4);
- `--warmup 2000` (the flag exists, §178) halves η·T_h, a weaker form of FAIR-START (i).

**Tried?**
- Not on nested170.
- fsq120 re-warmed at removals (§126 / §136 forks) and ran restart-on-plateau. LIT_S129 flagged the LR-restart confound (a CLOCK-ONLY control).

**Cost.**
- About 2 h per arm on a 3090 (10k × 0.74 s), so about $0.4 per arm.
- Needs a `--fork_from model.pt` path in nested170 (fsq120 has `--fork_from`), plus the γ = fade = 1 schedule offsets.

### 2.4 Free optimisation logging package: per-group gradient / update norms and the branch gradient-noise scale (METHOD; measures σ_Z's ingredients) — rank 4

**Evidence.**
- McCandlish et al. 1812.06162 [full text]:
  - B_simple = tr(Σ)/|G|² (Eq. 2.9) is the batch at which the estimated and true gradients become close: E|G_est − G|²/|G|² = B_simple/B (Eq. 2.10);
  - App. A.1 gives unbiased estimators from two batch sizes: |G|² ≈ (B_big|G_big|² − B_small|G_small|²)/(B_big − B_small) and S ≈ (|G_small|² − |G_big|²)/(1/B_small − 1/B_big), each EMA-smoothed before taking the ratio;
  - the noise scale "increases as the loss decreases";
  - Adam-preconditioned gradients gave "mixed results".
- Mosbach et al. 2006.04884 [full text\*]: failed BERT fine-tunes have vanishing gradients in the bottom layers. Their fix (small LR with bias correction, 20 epochs rather than 3) cut seed SD on CoLA from 25.6 to 1.8 (MRPC 3.9 → 0.8, RTE 4.5 → 2.7; 25 seeds).

**Math basis.**
- **(i) Source.** McCandlish is a derivation under a local quadratic model and is exact as an estimator. Mosbach is empirics.
- **(ii) Mechanism (DUA).**
  - Part II's window regime has z = b√(T_w B)/σ_g in step units, which is η-independent at a fixed step count.
  - B_simple over the selection-relevant parameters (mem.qkv, gate, W_o, LN_w, gains) estimates σ_g²/|G|² for the branch drift. That is the per-sample inverse SNR, i.e. z's denominator up to the projection on m.
  - Assumption: the drift along m is a fixed fraction of the branch gradient norm.
- **(iii) Predictions.**
  - (a) If B_simple(branch) in 2–6k ≲ 64, batch size is a weak lever: Part II P7's gain needs B ≫ B_simple to be cheap.
  - (b) Part II §5's stochastic-collapse ratio κ_SC = η_eff s²/(2βB): failing seeds show a higher branch B_simple in the window, or no difference, in which case noise is not the selector.
  - (c) Mosbach-type check: per-group gradient norms of the failed seeds are *not* vanishing. Clip rate 1.0 says the total is large, but the branch share is unknown.

**Tried?** Total gn_preclip and clip_frac_100 are logged (note 194-1); §196 logs the AdamW update norm, slow share, max logit and weight norm. Per-group norms and B_simple are not logged.

**Cost.**
- About 30 lines.
- A **bitwise-neutral** design for B_simple: on logging steps only, after the normal full-batch backward, save the .grad buffers, zero them, backward the first half-batch into them (same items, no RNG draw), record the per-group norms, then restore the saved grads before clipping.
- That doubles the cost on 1% of steps, plus 0.5 GB on a 3090, and leaves the training trajectory identical: gate it like gate_gnlog.

### 2.5 Relative instead of absolute clipping (ZClip / AdaGC style) (TILT, small) — rank 5

**Evidence.**
- ZClip 2504.02507 [abstract]: the clip threshold adapts via z-score anomaly detection on gradient-norm statistics, clipping spikes "while not interfering with convergence otherwise".
- AdaGC 2502.11034 [abstract]: per-tensor clipping relative to an EMA of historical clipped norms; spike score 0 on Llama-2 7B, Mixtral 8×1B and ERNIE 10B-A1.4B, with +1.32 / +1.27 / +2.48% downstream over global clipping.
- Zhang et al. 1905.11881 [abstract]: a theorem that, under a relaxed (gradient-norm-dependent) smoothness, clipped / normalised GD converges arbitrarily faster than fixed-step GD. That is why clipping helps at all.

**Math basis.**
- **(i) Source.** Zhang is a theorem (assumes (L0, L1)-smoothness). ZClip / AdaGC are heuristics plus empirics on spikes, which are not our failure mode.
- **(ii) Mechanism (DUA).**
  - Adam cancels any constant rescaling (Part II §5). Only step-to-step variation of the clip factor matters.
  - With clipping always binding, step t enters Adam's m with weight 1/‖g_t‖, and SD(log‖g‖) = 0.25–0.31 in 2–6k (F3).
  - With corr(gn, aux_ce) ≈ +0.67 (CLIP-1, 1–12k), high-aux-loss batches are down-weighted by about 0.67 × 0.28 ≈ 19% per SD of aux loss.
  - If late-pass (P-discriminating) signal concentrates in those batches, this is a negative contribution to b_P (a TILT toward C) of order ≤ 20%.
  - Relative clipping (threshold = EMA mean + 3 SD) makes steady-state weights uniform.
- **(iii) Prediction.**
  - z rises by ≤ 20%, so P rises 0.6 → ≤ 0.66 (Part II C1 linear in b): undetectable at n ≤ 20.
  - Mechanism-level check instead: under relative clipping the per-step weight's correlation with the batch's deep-item share (log the batch mean n) goes from negative to about 0.
  - If CLIP-1's sign does not replicate in the 2–6k window of §196's gn log, drop this lever.

**Tried?** No. CLIP-1 is descriptive; §188's "clipped away" note was corrected in §192.

**Cost.** About 15 lines: a flag; the threshold from an EMA of gn_preclip. Free to run.

**Cheap test (CPU, minutes).** Recompute CLIP-1 restricted to 2–6k on §194 s2 and on §196's log when it lands, adding gn against the batch deep-item share if it is logged.

### 2.6 Weight averaging (EMA / LAWA / SWA): evaluation-side only, plus a linear-connectivity diagnostic (NEUTRAL for P; METHOD) — rank 6

**Evidence.**
- TRM 2510.04871 [full text\*]: EMA 0.999 "to reduce this problem and improves stability" (HRM overfits then diverges on small data). Sudoku-Extreme 79.9% without vs 87.4% with.
- Morales-Brotons et al. 2411.18704 [abstract]: EMA solutions differ from last-iterate ones and are better in generalisation, robustness to label noise, prediction consistency, calibration and transfer; EMA needs less LR decay.
- Kaddour 2209.14981 [abstract]: latest-k checkpoint averaging saves "dozens of epochs".
- Sanyal et al. 2306.03241 [abstract]: gains are larger at high LR and with spaced checkpoints; nanoGPT-2 125M–770M.
- Ajroldi et al. 2502.06761 [abstract]: on AlgoPerf, averaging "significantly accelerates training", with mild generalisation gains.
- SWA 1803.05407 [abstract]: flatter solutions.
- Soups 2203.05482 [abstract]: fine-tuned models "often appear to lie in a single low error basin".
- **Against soups:** Juneja et al. 2205.12411 [abstract]: fine-tuned text classifiers form *distinct clusters*, linearly connected within and separated by loss barriers across, and the clusters correspond to different generalisation strategies.
- Neyshabur et al. 2008.11687 [abstract]: from pretrained weights, the model "stays in the same basin".
- McCoy et al. 1911.02969 [abstract]: 100 BERT seeds, MNLI dev 83.6–84.8%, HANS subject–object swap 0.00–66.2%.

**Math basis.**
- **(i) Source.** These are empirics. Polyak–Juditsky averaging theory (convex; not on arXiv) is not used here.
- **(ii) Mechanism.**
  - **PROVED, trivially:** a shadow average that is not fed back leaves the training map, and hence μ_Z, σ_Z and P(basin), unchanged. It can only cut within-basin iterate noise (∝ η/B) at evaluation.
  - Fed-back averaging (Lookahead 1907.08610 [abstract]: "lowers the variance of its inner optimizer") is a √-lever, weak per Part II.
  - Value for us is evaluation-side: borderline runs (window_s1 0.239, rmsnopass_s2 0.228 final loss) and K′16 / C6 readings may be partly iterate noise.
  - LMC is a direct check of Part II's two-basin premise.
- **(iii) Predictions.**
  - (a) EMA(0.999) eval ≥ raw on every converged run, with the largest gain on borderline runs. Entry statistics are identical by construction.
  - (b) LMC [CONJECTURE from the two-basin model]: the loss barrier on the linear path between s185_rms_s1 and s2 (P–P) is smaller than between s1 / s2 and s3 (P–C). Interpolants near s3 are C-like on path_indep delta_cos (≥ 0.69).
    - **Falsified if** the P–C barrier is ≤ the P–P barrier. Then "two basins" is a progress difference within one basin, and the reduced model's bistability needs revisiting.

**Tried?** No weight EMA anywhere: THEORY's EMA hits are standardisation statistics. No LMC.

**Cost.**
- EMA shadow: about 20 lines, 0.5 GB fp32 on a 3090, evaluated beside the raw model.
- LMC: CPU only, forward passes on 256 training-distribution items × 5 α × 3 pairs at K = 8; about 10–20 min at 3 threads. It needs a scratch eval script around nested170's model builder (no repo writes).

### 2.7 A small-scale proxy for the lottery, so that P(basin) can be estimated over 20–100 seeds (METHOD; empirical precedent only) — rank 7

**Evidence.** Wortsman et al. 2309.14322 [abstract + full text\*]:
- large-scale instabilities (attention-logit growth, output-logit divergence) "also appear in small models when training at high learning rates", and their mitigations are "equally effective" there;
- they define LR sensitivity as the expected excess loss over three orders of magnitude of LR;
- longer warm-up and LR-independent weight decay reduce it.
- No seed variance is reported.

**Math basis.**
- **(i) Source.** Empirics only.
- **(ii) Mechanism (CONJECTURE).**
  - Part II's P = Φ(μ_Z/σ_Z) depends on (a, b_P, β_C, r_C, T_h, σ). A proxy keeps the lottery if it keeps the *sign structure*: the open-loop head start (Lemma 1), the contractive delta init α^L (Thm 4), and the pass-token shortcut (Lemma 4).
  - It need not keep GPT-2 scale. The risk is that C's "hare" (the pretrained trunk's one-shot reductions) is absent in a scratch trunk.
- **(iii) Gate before use.** Over ≥ 20 seeds:
  - 0.2 < P < 0.8;
  - the path_indep separation (delta_cos);
  - the hare signature (Part II P1: the lowest-early-loss seeds fail more).
  - **Falsified as a proxy if** any of the three fails.

**Tried?** `--scratch L,D,H` exists (§178), but it was never used for seed statistics.

**Cost.** Candidates are GPT-2 with n_max 4–5 and K 4–6 on 8k steps, or a scratch 4-layer trunk. Local cards (short, memory-checked) at about 20–40 min per seed would make 20 seeds about one card-night.

**Payoff.** §2.1's power table: a 0.6 → 0.95 claim needs about 20 seeds per arm.

### 2.8 Separate LR for the memory branch vs the trunk (LoRA+ / discriminative fine-tuning) (TILT under Part I §5.3–5.4; sign uncertain under Part II) — rank 8

**Evidence.**
- LoRA+ 2402.12354 [abstract]: equal LRs for the zero-init factor B and A "does not allow efficient feature learning" at large width (scaling argument); the right ratio gives +1–2% and about 2× speed.
- ULMFiT 1801.06146 [full text\*]: discriminative fine-tuning with η_{l−1} = η_l/2.6. Ablation IMDb / TREC-6 / AG error: 6.87 / 6.86 / 5.81 → 5.57 / 6.21 / 5.62 with "discr".
- 2608.11233 [full text\*]: a 10× LR group for the new prelude projection; "any parameter that needs its own learning rate must be a separate tensor in its own optimizer group".

**Math basis.**
- **(i) Source.** LoRA+ is a derivation (width scaling, assumes the large-width limit); ULMFiT is empirics.
- **(ii) Mechanism.**
  - **Part I §5.3 (DERIVED, Rawal analogy).** The closed-loop gain ε is a product of branch factors (W_o, LN_w gain, write alignment, γ). For r = 2, ε(∞) = ε₀ exp(2∫η_branch D). A branch-LR multiplier c raises the exponent linearly: a TILT in probit, as in §2.3.
  - **Part II C4 (CONJECTURE).** C's speed r_C has a trunk component (one-shot reductions in the shared-trunk step decoder) and a branch component (the open-loop input encoding). A trunk-LR cut lowers r_C and β_C (toward P); a branch-LR raise also speeds C's encoding. The sign is uncertain.
- **(iii) Prediction.** Branch ×3 vs ×1 at a fixed trunk LR:
  - Part I predicts earlier entry (entry − 6k shrinks about c-fold) and higher P;
  - Part II predicts no change or a fall if C's encoding speeds up more.
  - Read on the proxy (§2.7), or on 6 fresh seeds.

**Tried?** §127 LR10 (fsq120, new LN / W_down / W_up at 10× LR): ESCAPE (it passes the 2k plateau flag) without learning; not adopted. That was a different harness and failure mode. Never on nested170.

**Cost.** make_opt already supports a `mult` group (lr_new). It needs a branch parameter list for nested170.

### 2.9 Muon on the memory branch's matrices only (TILT, CONJECTURE) — rank 9

**Evidence.**
- Tveit et al. 2504.16041 [abstract]: on 7 numerical (mostly modular-arithmetic) tasks, Muon cut the mean grokking epoch from 153.09 to 102.89 vs AdamW (t = 5.02, p = 6.3e-8).
- Liu et al. 2502.16982 (Moonlight) [abstract + full text\*]:
  - Muon needs weight decay and a per-parameter update scale (0.2·O_t·√max(A, B));
  - AdamW handles RMSNorm, LM head and embeddings;
  - **an AdamW-pretrained model fine-tuned with Muon shows no significant advantage** (MMLU 50.2 vs 55.7 for Muon–Muon); Qwen2.5-7B SFT is on par.

**Math basis.**
- **(i) Source.** Moonlight's scaling is empirics plus an update-RMS-matching heuristic. Tveit is empirics.
- **(ii) Mechanism (CONJECTURE).**
  - Part I Lemma 5.2(i) (PROVED): in a contractive regime any update moves the common (pass-independent) mode far more than the differential (progress) mode.
  - If the progress direction lives in low-singular-value directions of the branch matrices' gradient, Muon's orthogonalised update gives those directions the same step as the dominant input-summary direction. That raises b_P relative to C's rich-get-richer gain a: a TILT.
  - Assumption: progress is a distinct singular direction of the update. Not checked.
  - Applying it to the pretrained trunk is not recommended (Moonlight's mismatch).
- **(iii) Prediction.**
  - Mechanism: the per-step ratio ‖δΔ_r‖/‖δS_r‖ (Lemma 5.2's differential-to-common ratio) measured at a 6k checkpoint is higher under branch-Muon than AdamW.
  - Outcome: on the proxy, P rises and entry is earlier.

**Tried?** No optimiser other than AdamW appears in THEORY.

**Cost.** About 60 lines (Newton–Schulz), plus a hybrid param split and a flags-off gate.

### 2.10 Keep full backprop through all passes; never truncate (TILT if violated) — rank 10 ("keep doing")

**Evidence.**
- TRM [full text\*]: Sudoku-Extreme with "1-step gradient" (HRM's approximation) 56.5% vs full backprop through the last recursion 87.4%.
- Tallec & Ollivier 1705.08209 [abstract]: truncation "favors short-term dependencies"; the estimate is biased; ARTBP unbiases it.
- Huginn [full text\*] truncates to the last k = 8 of r̄ = 32 iterations (memory-driven).
- Phantom gradients (2111.05177) and JFB (2103.12803) [abstract] are cheaper approximate gradients for implicit models.

**Math basis.**
- **(i) Source.** Tallec is a derivation (bias of truncated BPTT); TRM is empirics.
- **(ii) Mechanism (DUA).**
  - Pass s's write parameters receive the late-pass reward only through ∂S_t/∂S_s for t > s. Truncating to the last k passes deletes those terms for t − s > k. That lowers b_P.
  - C's input-encoding gradient is pass-local and survives. So truncation is a TILT toward C.
  - Part I Lemma 5.2(iii): the non-attenuated escape signal itself needs the cross-pass Jacobian chain.
  - Gradient checkpointing is exact (NEUTRAL; a re-roll only through recompute rounding).
- **(iii) Prediction (not worth running).** A k = 2 truncated arm has lower P and a more negative 8k λ_eff. Cheap approximations (phantom / JFB) are expected to hurt for the same reason.

**Tried?** nested170 uses full BPTT; LIT_S155 recommended it ("truncation lowers φ").

### 2.11 Batch size as a √-lever, sized by §2.4's noise scale (√) — rank 11

**Evidence.**
- Malladi et al. 2205.10287 [abstract]: proved SDE approximations for RMSprop / Adam and a "square root scaling rule" when changing batch size.
- Smith et al. 1711.00489 [abstract]: increasing batch size reproduces LR decay, Adam included.
- Zhang et al. 2410.21676 [abstract]: critical batch size scales mainly with data size.

**Math basis.**
- **(i) Source.** Malladi is a theorem under SDE-approximation assumptions; the others are empirics.
- **(ii) Mechanism (DUA).**
  - In the noise-dominated Adam regime, v ≈ σ²/B. So the update ≈ η(√B·G/σ + ξ): drift ∝ √B per step, and noise O(1).
  - Over k steps z ∝ √(kB)·G/σ. That is √ in samples, independent of η, which confirms Part II's √-lever at fixed LR.
  - B 64 → 256 doubles z. The cost is 4× per step.
- **(iii) Prediction.** As Part II P7 (P 0.6–0.67 → 0.69–0.81), and Part I §5.5 (batch 32 fails more than 128). Only worth running if §2.4 shows B_simple(branch) ≫ 64.

**Tried?** Only as a budget change (§168: bs 32 vs 64), never as a lever.

### 2.12 Precision and determinism: record, don't spend (NEUTRAL) — rank 12

**Evidence.**
- Micikevicius et al. 1710.03740 [abstract]: keep an fp32 master copy of the weights.
- Zamirai et al. 2010.06192 [abstract]: in pure-bf16 training, nearest rounding cancels small weight updates; stochastic rounding or Kahan summation recovers fp32 accuracy (to −0.1 … +0.2%).
- D'Angelo et al. 2310.04415 [full text\*]: GPT-2-124M at lr 1e-3 without weight decay "exhibit[s] late-training divergence for all random seeds when using bfloat16", while fp32 and λ = 0.1 / 0.3 stay stable.
- Zhuang 2106.11872: determinism overhead (above).

**Math basis.**
- **(i) Source.** Empirics.
- **(ii) Mechanism.**
  - **PROVED within Part II:** rounding enters only as a re-draw of the noise realisation after chaotic decorrelation (§3.2). Determinism cannot change μ_Z or σ_Z, so cannot raise P. Stochastic rounding adds a noise of about 2⁻⁸ per op, negligible against SGD noise.
  - [ours] Our Adam step is about lr = 1e-4 per coordinate on weights of order 0.02–0.1, i.e. 1e-3–5e-3 relative. That is at or below bf16's 2⁻⁸ = 3.9e-3. **Pure-bf16 weights would silently drop many updates**, so the fp32 master weights we have (autocast) are essential; keep them.
  - We are 10× below D'Angelo's divergent LR.
- **(iii) Prediction.** The same card and driver replay bitwise (observed). Cross-card flips follow §2.1's rate, and fp32 vs bf16 changes P by less than seed noise.

**Advice.** Record gpu and driver (done). Treat card class as part of the seed; one GPU class per experiment (memory note).

---

## 3. Below the line: no statable mechanism, or not recommended

| item | evidence (level) | Part II class / why not |
|---|---|---|
| **Strong weight decay / L2-SP as an efficiency tilt** | Varma 2309.02390, Power, Omnigrok (prior notes); TRM wd 1.0 [full text\*] | The premise is unsupported here (F2: trunk drift is identical in P and C). Strong decay on the branch shrinks the multiplicative loop gain ε (Part I §5.3), so it is predicted **harmful** (a TILT toward C). Not recommended without a premise. |
| **β2 = 0.95 / Adam turn-on transient** | Wortsman 2304.13013 [abstract]: spikes 1–8 iterations after AdamW's second moment under-estimates g²; StableAdamW (AdamW–Adafactor hybrid). Huginn and TRM use β2 0.95 [full text\*] | LN_w has exactly zero gradient for 2000 steps (Lemma 1), then switches on. [ours, simulated] For a gradient growing linearly from step 2000, Adam's m̂/√v̂ is 4.7 / 3.0 / 1.9 at τ = 100 / 300 / 1000 steps after onset at β2 = 0.999, vs 1.0–1.1 at 0.95 (steady state ≈ 1). Adam scales drift and noise equally (Part II §2), so the class is NEUTRAL (timing only). Low; log LN_w's update RMS in 2–3k. |
| **QK-norm / z-loss / sandwich or peri-LN** | Wortsman 2309.14322 [full text\*]: qk-layernorm and z-loss (1e-4) fix attention-logit growth and output-logit divergence at high LR; AdamW ε → 1e-15 when grad RMS nears 1e-8. Huginn [full text\*]: bad run 1 hit token-correlation collapse; bad run 2 "learned early to ignore the incoming state" (validation ppl the same at 1 or 32 recurrences); fixed by sandwich norm + a learned adapter + embedding scale + peak LR 4e-5, with no ablation. Peri-LN 2502.02732 [abstract]; Dehghani 2302.05442 / Henry 2010.04245 [abstract] | No mechanism for basin selection: our failure is a plateau (0 non-finite), delta q / k are already L2-normalised, and the additive read is RMS-normalised. Huginn's bad run 2 is the closest external precedent of our C basin (state ignored), but it is unablated. Conditional: only if §196's max-logit log shows growth. ε: [ours] the mean per-coordinate grad RMS after the clip is about 9e-5 ≫ 1e-8, so ε is not binding. |
| **AGC** | Brock 2102.06171 [full text\*]: unit-wise ‖G‖/max(‖W‖, 10⁻³) > λ (λ = 0.01 typical); "always better to not clip the final linear layer" | Unsuitable: zero-init W_o has ‖W‖ = 0, so it is clipped to λ·10⁻³ per unit; under Adam the per-unit factor mostly cancels. Do not adopt. |
| **Dropout / stochastic depth / label smoothing / Mixout** | Müller 1906.02629 [abstract] (calibration, tighter clusters); Mixout 1909.11299 [abstract] (stochastic mixing toward a target model; "stability of finetuning ... greatly increase" on GLUE) | Dropout and depth noise raise σ (a negative √). Label smoothing has no clear tilt (CE gradients on fitted early passes already vanish). Mixout ≈ L2 toward pretrained weights, which F2 does not support. Random K is covered in LIT_RAND_K. |
| **Stratified batches (exact n-mix per batch)** | Dodge 2002.06305 [full text\*]: init (2048-parameter head) and data order contribute comparably; stop the worst after 20–30% of training (41 started, 11 kept, budget 20). Shumailov 2104.09667 [abstract]: order alone can block learning or reset progress | A √-lever: it removes the multinomial n-count variance. [ours] The deep-item (n ≥ 6) share per batch of 64 has CV 16%. Probably a small part of σ_g; free, low. Deliberate orderings are tilts, already covered (curriculum, §184 reweight). Our init is deterministic, so all seed variance is order + rounding (consistent with Dodge's split). |
| **Lion / SOAP / Adam-mini / AdEMAMix / schedule-free** | Lion 2302.06675 [abstract]: sign updates, smaller LR, gains grow with batch. SOAP 2409.11321 [abstract]: > 40% fewer iterations in the large-batch regime. Adam-mini 2406.16793 [abstract]: −50% optimiser memory. AdEMAMix 2409.03137 [abstract]: a second slow EMA; 1.3B on 101B tokens ≈ AdamW on 197B; slower forgetting. Schedule-free 2405.15682 [abstract]: no stopping time; won AlgoPerf self-tuning | No seed-variance evidence and no statable mechanism. AdEMAMix's slow EMA is a symmetric slow-component amplifier like Grokfast (Part II §7: not a lever; wait for §196). Adam-mini is only a memory convenience for the 8 GB local cards. |

---

## 4. Verified IDs (all resolved on arXiv 2026-10-13, exact titles; level of reading)

| ID | Title | Read |
|---|---|---|
| 1803.05407 | Averaging Weights Leads to Wider Optima and Better Generalization | abstract |
| 2209.14981 | Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging | abstract |
| 2306.03241 | Early Weight Averaging meets High Learning Rates for LLM Pre-training | abstract |
| 2203.05482 | Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time | abstract |
| 2411.18704 | Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits | abstract |
| 2502.06761 | When, Where and Why to Average Weights? | abstract |
| 2510.04871 | Less is More: Recursive Reasoning with Tiny Networks | full text\* |
| 2506.21734 | Hierarchical Reasoning Model | abstract |
| 2008.11687 | What is being transferred in transfer learning? | abstract |
| 2205.12411 | Linear Connectivity Reveals Generalization Strategies | abstract |
| 1911.02969 | BERTs of a feather do not generalize together: Large variability in generalization across models with similar test set performance | abstract |
| 2502.16982 | Muon is Scalable for LLM Training | abstract + full text\* |
| 2504.16041 | Muon Optimizer Accelerates Grokking | abstract |
| 2409.11321 | SOAP: Improving and Stabilizing Shampoo using Adam | abstract |
| 2302.06675 | Symbolic Discovery of Optimization Algorithms | abstract |
| 2406.16793 | Adam-mini: Use Fewer Learning Rates To Gain More | abstract |
| 2409.03137 | The AdEMAMix Optimizer: Better, Faster, Older | abstract |
| 2405.15682 | The Road Less Scheduled | abstract |
| 2304.09871 | A Theory on Adam Instability in Large-Scale Machine Learning | abstract |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities | abstract + full text\* |
| 2304.13013 | Stable and low-precision training for large-scale vision-language models | abstract |
| 2404.06395 | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies | abstract |
| 2405.18392 | Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations | abstract |
| 2410.05192 | Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective | abstract |
| 2403.08763 | Simple and Scalable Strategies to Continually Pre-train Large Language Models | abstract |
| 2308.04014 | Continual Pre-Training of Large Language Models: How to (re)warm your model? | abstract |
| 1608.03983 | SGDR: Stochastic Gradient Descent with Warm Restarts | abstract |
| 2406.09405 | Why Warmup the Learning Rate? Underlying Mechanisms and Improvements | abstract |
| 2405.13698 | How to set AdamW's weight decay as you scale model and dataset size | full text\* |
| 2310.04415 | Why Do We Need Weight Decay in Modern Deep Learning? | full text\* |
| 2305.17212 | Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks | abstract |
| 1812.01187 | Bag of Tricks for Image Classification with Convolutional Neural Networks | full text\* |
| 2102.06171 | High-Performance Large-Scale Image Recognition Without Normalization | full text\* |
| 2504.02507 | ZClip: Adaptive Spike Mitigation for LLM Pre-Training | abstract |
| 2502.11034 | AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping | abstract |
| 1905.11881 | Why gradient clipping accelerates training: A theoretical justification for adaptivity | abstract |
| 2502.05171 | Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach | full text\* |
| 2302.05442 | Scaling Vision Transformers to 22 Billion Parameters | abstract |
| 2010.04245 | Query-Key Normalization for Transformers | abstract |
| 2502.02732 | Peri-LN: Revisiting Normalization Layer in the Transformer Architecture | abstract |
| 2608.11233 | Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets | full text\* (App. A.2 dead start: "with the mixing gate at zero, the bridge contributed nothing ... so the gate stayed at zero"; fixed by an identity-biased live gate. Ours avoids this: g_add is calibrated non-zero, and the gate bias is non-zero with weight 0) |
| 1901.09321 | Fixup Initialization: Residual Learning Without Normalization | abstract |
| 1705.08209 | Unbiasing Truncated Backpropagation Through Time | abstract |
| 2111.05177 | On Training Implicit Models | abstract |
| 2103.12803 | JFB: Jacobian-Free Backpropagation for Implicit Networks | abstract |
| 1710.03740 | Mixed Precision Training | abstract |
| 2010.06192 | Revisiting BFloat16 Training | abstract |
| 2106.11872 | Randomness In Neural Network Training: Characterizing The Impact of Tooling | abstract |
| 2103.04514 | Nondeterminism and Instability in Neural Network Optimization | abstract |
| 1812.06162 | An Empirical Model of Large-Batch Training | full text (PDF) |
| 2205.10287 | On the SDEs and Scaling Rules for Adaptive Gradient Algorithms | abstract |
| 1711.00489 | Don't Decay the Learning Rate, Increase the Batch Size | abstract |
| 2410.21676 | How Does Critical Batch Size Scale in Pre-training? | abstract |
| 2002.06305 | Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping | full text\* |
| 2006.04884 | On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines | full text\* |
| 2006.05987 | Revisiting Few-sample BERT Fine-tuning | abstract |
| 2104.09667 | Manipulating SGD with Data Ordering Attacks | abstract |
| 1909.11299 | Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models | abstract |
| 1906.02629 | When Does Label Smoothing Help? | abstract |
| 2312.16903 | Spike No More: Stabilizing the Pre-training of Large Language Models | abstract |
| 2402.12354 | LoRA+: Efficient Low Rank Adaptation of Large Models | abstract |
| 1801.06146 | Universal Language Model Fine-tuning for Text Classification | full text\* |
| 1907.04595 | Towards Explaining the Regularization Effect of Initial Large Learning Rate in Training Neural Networks | abstract |
| 1907.08610 | Lookahead Optimizer: k steps forward, 1 step back | abstract |
| 1705.07774 | Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients | abstract |
| 2011.09468 | Gradient Starvation: A Learning Proclivity in Neural Networks | abstract |

**Leads not pursued.**
- Li, Wei & Ma 1907.04595 [abstract]: a small initial LR first memorises "easy-to-fit" patterns, and a large-LR model ignores an immediately memorisable patch until annealing. It is a provable two-layer analogue of our "hare" (Part II P1).
- Its implication (a higher LR in the open loop delays the C shortcut) conflicts with Part II C4's T_h·η reading. That is unresolved, and a candidate question for Theory.

---

## 5. Files
- `genlat/G2_optim_stability.md`: this note.
- `genlat/trunk_drift.py`, `genlat/scalars.py`, `genlat/trunk_drift.json`: F1 / F2. These read 15 model.pt files and the cached openai-community/gpt2; they write only the JSON.
- `genlat/fetch_arx.py`, `genlat/pdftxt.py`, `genlat/arx/meta1.txt`: ID verification dumps. `genlat/pdf/` holds the McCandlish PDF text.
- No GPU or cloud use, no repo writes, no .pt files created.
