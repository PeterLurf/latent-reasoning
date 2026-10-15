# JOINT-COT-L1 on GSM8K-Aug: design + derivation DRAFT (2026-10-14)

> **Theory review (2026-10-15 00:00 EDT):** the layer-split premise was checked in code (deltamem.py: read_fixed at block-6 entry, so layers 1-6 sit before the read). The CODI details follow Literature bc8be322 plus this draft's own v3 read. Disclosed process slips: one read-only git log; the probe ran ~10 min at Normal priority before Idle. **Not registered; needs the user's OK** (pair $4.04 central / $6.00 high; J1 alone $2.02 / $3.00).

Opus subagent for Theory. **Not registered.** Theory reviews; registration needs the user's OK (new spend). D:\Latent read only; CPU probes at ≤ 3 threads, Idle priority, in theory/jointcot/ (§13). Lead: Audit (JOINT-COT-L1 on the stepaux recipe). CODI facts: Audit's v3 read plus Literature's addendum (docs/LIT_WARMCOT_CHECK_2026-10-14.md), **re-checked here against the arXiv v3 HTML** (§1.3).

**Tags.** PROVED (full argument; "PROVED (code)" = the harness as committed, lines cited) · DERIVED|A… (follows from named, unverified assumptions) · CONJECTURE · DATA (read from files; **POST-HOC** = numbers seen while drafting, which includes every §200 / §202 result and the probes) · calc.

**Notation.** γ_T = the teacher-CE weight (CODI's γ); β = the coupling weight (CODI's β); γ_w = the memory write mix (code `mem.gamma`, 0 to 2k then ramped to 1 at 6k). Layer l = the output of block l−1 (l = 1..12, before ln_f). The memory read enters the residual **before block 6**, i.e. between l = 6 and l = 7 ("post-read layers" = l 7..12).

---

## 0. Summary

1. **Recommendation:** run JOINT-COT-L1 on the **W base** (init = C's trunk, W's flags, 12k steps, full fine-tuning, lr 1e-4), as a **2-cell pair**: **J0** = W + the joint teacher CE (γ_T = 2); **J1** = J0 + the L1 coupling. Comparators W, C+, A and C have all landed ($0). J1 − J0 is CODI's "w/o L1" ablation on our architecture; J0 − W is trunk protection by the joint CE.
2. **Where to couple:** at the student decoder's "###" token, against the teacher's "###" token, on the **post-read layers 7–12 only**. Layers 1–6 sit before the read, so they get **zero memory gradient** (PROVED, code) and only train the decoder to imitate the teacher without the memory, which is the decoder-copy Goodhart. The teacher target is CODI's **final-step-dropped** sequence (T2), for m ≥ 2 only, with C's full-CoT CE (T1) kept for retention. CODI keeps the last step only in an ablation, which loses 12 points (31.7 vs 43.7, verified).
3. **Mechanism:** (M1) a joint CE puts a fixed point on the CoT loss, and the excess CoT loss scales as 1/γ_T² (DERIVED|quadratic). W had γ_T = 0, so nothing held its forgetting back. (M3) The teacher CE's gradient never reaches the memory or S (PROVED, code), so shared weights alone change only what the trunk computes from q: CODI's 24.5. (M2) The L1 is the only term that gives the read a dense, item-specific target **downstream of the memory**. The T2 target asks M_K for the earlier results, the content D1 shows the string aux never demands.
4. **Gradient crowding (DATA + DERIVED):** A's pre-clip gradient is κ ≈ 5.9× C's (5.2–6.3 by phase). Under Adam the teacher gets ≈ γ_T/√(γ_T² + κ²) of a text run's step: 0.32 at γ_T = 2. From raw GPT-2 the teacher would reach only cot_valid ≈ 0.21 by 24k (calc). This is why the **warm** base is chosen and the raw 24k design is not. It also makes Audit's prediction (i), "≥ 0.30 throughout", coherent: it cannot hold from raw init, where C itself is at 0.16 at 2k.
5. **Coupling vs LoRA:** the derivation predicts that the **coupling matters more**. Protection without coupling leaves the memory's targets unchanged (M3), and CODI's own LoRA + joint-CE arm without L1 is 24.5, near its no-CoT 19.1. Full fine-tuning is the base, so no cell has LoRA, and J1 − J0 isolates the coupling. LoRA is a conditional third cell (§6.5).
6. **Predictions (seed 1, central values):** COT-RETAIN ≥ 0.30 in J0 and J1; COUPLE = J1 − J0 ≈ +0.03 (P(read ≥ T) ≈ 0.55); J1 ≈ 0.185 (W 0.146); deep K′10 − K′1 ≥ T_D (W 0.005); K′0 does not rise by more than T (the Goodhart guard). No SOTA: about 6 epochs in total (§7).
7. **Cost:** **$2.02 central / $3.00 high per cell** (7.7 h at A's 1.82 s/step plus a teacher 0.45 s/step; $1.42 on a W-speed host). The pair is **$4.04 / $6.00, flagged above $4**. The raw 24k pair is $7.8 / $11.7, and its high is above $10, so it is the user's call (§8).
8. **Main risk:** the coupling's memory part is an end-of-chain target (at S_K), so one-pass dominance can survive. KD-ROUTE and THINK-NECESSARY-DEEP read this, and the per-pass coupling (option c, KaVa-like) is the named follow-up.

---

## 1. Facts

### 1.1 Our runs (DATA, POST-HOC; boxes/runs_fsq/*/eval.json, log.jsonl)

| cell | test | K′0 / K′1 / K′10 | deep (m ≥ 3, n 879) K′0 / K′1 / K′10 | recall share | train s/step (3090) | clip rate |
|---|---|---|---|---|---|---|
| A `gsm_lat_aug_s1` (raw, 24k) | 0.147 | 0.083 / 0.125 / 0.147 | 0.064 / 0.097 / 0.110 (ratio 0.88) | 0.48 | **1.820** (43,678 s) | 1.0 |
| W `gsm_lat_warm_s1` (C init, 12k) | 0.146 | 0.111 / 0.129 / 0.146 | 0.081 / 0.091 / 0.096 (ratio **0.95**; gain 0.015 < T_D) | 0.34 | 1.103 (13,230 s) | 1.0 |
| C `gsm_txt_aug_s1` (24k text) | 0.321 | — | — | 0.42 | **0.255** (6,116 s, micro 1) | 0.064 |
| C+ `gsm_txt_aug_cont_s1` (C + 12k) | 0.336 | — | — | 0.42 | 0.423 (another host) | 0.001 |

* The deep figures come from eval_m_kprime.json (3070, bs 64, note 202-1's script). W's deep row is new here: its loop gain on m ≥ 3 (0.015) is **below T_D**, so W's deep stratum is essentially one-pass or direct.
* W's cot_valid (the shared trunk's own TEXT-COT, Aug valid): 0.072 / 0.092 / 0.116 / 0.128 / 0.106 / 0.114 at 2–12k. C's valid curve: 0.16 / 0.21 / 0.26 / 0.29 / 0.31 / 0.32 / 0.33 / 0.35 at 2–16k, flat after.
* **Pre-clip gradient norms (median by phase, log.jsonl):** A 9.0 / 6.5 / 4.0 / 3.5 / 3.7 against C 1.75 / 1.04 / 0.71 / 0.60 / 0.59 (steps <1k / 1–2k / 2–6k / 6–12k / 12–24k), so **κ = 5.2 / 6.2 / 5.7 / 5.9 / 6.3**. The write-path parameters carry only 0.25–0.43 of A's norm (write_grad_norm), so the bulk sits in the trunk, W_o and the read.
* **Paired A vs W on the 1,319:** 187 discordant items, so the paired SE of a difference is **0.0104** (calc_jointcot.json).
* **m mix:** test m 0 / 1 / 2 / ≥ 3 = 18 / 65 / 357 / 879. Aug train (filtered, 380,567) has m ≤ 1 for 16.4% (probe_kd.json train_m_share).

### 1.2 Harness (PROVED, code)
* **H1, the decoder read site.** The answer decoder is `ProsMem.dec_loss`, which runs `fwd_read` on [EOT] + pre + a1 + EOS (prosqa168s2.py l.138–144; deltas5.py l.91–98). `read_fixed` adds `fade · W_o[read_add ‖ read_del]` to the residual **entering block 6** at every position (deltas5 l.95–96; deltamem l.158–176). So h_S^l for l ≤ 6 is a function of the decoder tokens only: ∂h_S^l/∂S ≡ 0.
* **H2, the "###" positions.** a1 = ENC.encode("### " + ans) starts with ANS = 21017 (asserted in probe_kd.py). The student's "###" is at index 1 + |pre| and predicts the first answer token. The teacher's "###" in pre + cot + a1 (C's sequence, gsm8k_stepaux l.243–249) is at |pre| + |cot|. Both are the token right before the answer, the analogue of CODI's ":".
* **H3, the text arm is the plain trunk.** `g(g.embed(x))` (core.GPT2KV.forward) has no memory. Its loss reaches only g's parameters, never `m.mem.*` or S.
* **H4, the eval path.** `answers()` = `think_states` (K passes on the identical [EOT, q]) → `greedy_after` on [EOT] + pre + [ANS] reading S_K (gsm8k_stepaux l.296–311). No text CoT is read or generated, and no training loss enters it.
* **H5, the global clip.** clip_grad_norm_(·, 1.0) runs after all micro-chunk backwards (l.343), and A clips on every step. AdamW (wd 0.01) follows, and the lr is warm-up 100 + cosine (fsq120.lr_at).
* **H6, decorators.** `greedy_after`, `cot_batched`, `aux_greedy` and `margins` are @torch.no_grad, and `think_states` uses `with torch.no_grad()`. None of them is on a training-loss path. `dec_loss`, `aux_logits`, `fwd_read` and `GPT2KV.forward` are undecorated.

### 1.3 CODI 2502.21074, checked here against the v3 HTML (arxiv.org/html/2502.21074v3, fetched 2026-10-14; title verified on the abstract page)
* **Verified:**
  - L = α L_student + β L_KD + γ L_teacher (Eq. 1). Appendix F: "β determines the weight of the distillation loss", and γ is the teacher-task weight. Defaults are α = β = γ = 1; GPT-2 uses β = 1, and LLaMA-1b β = 20.
  - L_KD = (1/M) Σ_l |sg[h_T^l] − h_S^l| (Eq. 5), at the token right before the answer (":" of "The answer is:"), with sg on the teacher.
  - "We normalize each layer's hidden activations by dividing them by the standard deviation of the teacher's corresponding hidden activations within the current batch." Figure A1's code divides the loss by `teacher_hidden[:, pos-1].std()`, a **scalar** std, and uses **smooth_l1** there.
  - "We exclude the final CoT step … the model might directly copy the result from the last CoT step to the token responsible for generating the exact answer token, bypassing the reasoning process."
  - Table 2 (GPT-2): no-CoT-SFT 19.1; CODI 43.7; separate static teacher 27.1; static teacher + a multitask student 42.2; **w/o L1 24.5**; **w/ CoT last step 31.7**; w/o projection 42.5.
  - Training: LoRA r 128 / α 32, bf16, lr 3e-3, 40 epochs, AdamW, cosine with 3% warm-up, effective batch 128.
  - Table A5 (teacher weight γ): 20 epochs 36.3 / 38.4 / 41.6 / 40.8 at γ 0.5 / 1 / 2 / 3; 40 epochs 38.2 / 43.7 / 41.9 / – ("a higher γ accelerates convergence but leads to lower final performance").
  - Table A6 (20 epochs): the baseline prompt gives 39.0 ± 0.9 (5 runs).
* **Inconsistency in CODI:** the main-text implementation paragraph says "for LLaMA … set γ to 20, as … its distillation loss has a much smaller magnitude". That is β by Eq. 1 and Appendix F. **Appendix F is taken.**
* **Not verified:**
  - whether M counts the embedding layer, and whether the std is per layer (the text) or one scalar over all layers (Figure A1 is ambiguous);
  - LoRA's target modules;
  - the epoch count of the Table 2 ablations;
  - whether the student's answer-position states are post-ln_f at the last layer.
  None of these changes the design below, which states its own choice for each.

---

## 2. Mechanism / derivation

**Setup.** θ = the shared GPT-2 trunk, φ = DeltaMem's parameters, S_K = the memory after K passes.
* Student loss: L_S(θ, φ) = answer CE + λ·aux CE (A's).
* Teacher loss: L_T(θ) = T1 CE (C's full-CoT loss) + T2 answer CE (§3).
* Coupling: L_KD(θ, φ).
* J0 minimises L_S + γ_T·L_T, and J1 minimises L_S + γ_T·L_T + β·L_KD.

**M1. Why the joint CE prevents §202's forgetting (the gradient-sharing argument).**
* **(a) PROVED (first order).** For any update Δθ = −P∇(L_S + γ_T L_T) with P ≻ 0 (SGD, or Adam with v frozen over a step):
  ΔL_T = −⟨∇L_T, ∇L_S⟩_P − γ_T‖∇L_T‖²_P.
  At γ_T = 0 (W), L_T moves only through the cross term, and conflict (a negative inner product) raises it with no opposing term. That is consistent with W's 0.35 → 0.07 within 2k, while the lr was re-warmed and the write path was open-loop (DATA).
* **(b) DERIVED|A-QUAD.** Assume L_T is quadratic around θ_C with Hessian H, and the student's mean gradient in the conflict subspace is ḡ_S, constant near θ_C. A fixed point of gradient flow on L_S + γ_T L_T does not depend on the preconditioner. So γ_T H δ* = −ḡ_S, and **the excess CoT loss is ½ δ*ᵀHδ* = ḡ_Sᵀ H⁻¹ ḡ_S / (2γ_T²)**: bounded, and 4× smaller at γ_T = 2 than at γ_T = 1.
  - In the warm design the teacher only has to **hold** C's competence, so retention is set by this fixed point and not by the teacher's rate.
* **(c) DERIVED|A-ADAM (diagonal second moments; cross-moment ≈ 0; per-parameter κ ≈ the global norm ratio).** In shared parameters Adam divides by √(E g_S² + γ_T² E g_T²). So the teacher's step is a share s_T = γ_T/√(γ_T² + κ²) of a solo text run's step, and the student keeps s_S = κ/√(γ_T² + κ²).
  - With κ = 5.9: s_T = 0.17 / 0.32 / 0.45 / 0.65 at γ_T = 1 / 2 / 3 / 5, and s_S = 0.99 / 0.95 / 0.89 / 0.76.
  - The global clip rescales g_S and g_T together, so Adam is invariant to it to first order (A clips every step).
* **(d) calc, raw-init consequence.** Matching integrated cosine lr ("gradient-flow time"), a raw-init joint teacher reaches C's step 2.0k / 4.0k / 5.7k / 8.6k by 24k at γ_T = 1 / 2 / 3 / 5. That predicts **cot_valid ≈ 0.16 / 0.21 / 0.25 / 0.30 at 24k** (κ 4 → 0.18 / 0.25 / 0.29 / 0.31). So a raw-init JOINT keeps a weak teacher for most of its run, and Audit's (i) is unreachable there. This is the derivation-side reason for the warm base.
  - It also matches CODI Table A5's direction: a higher γ is faster early and costs the student later (s_S falls).

**M2. What the L1 transmits.**
* **(a) PROVED (H1).** The l ≤ 6 terms have ∂/∂S ≡ 0 and ∂/∂φ ≡ 0. The l ≥ 7 terms reach S_K through ∂r/∂S · fade · W_oᵀ · J_lᵀ · sign(e_l)/σ_l, where r is the read and J_l the Jacobian of layer l with respect to the block-6 input. That gradient is first order in W_o, so it is zero while W_o = 0 (WARMCOT Lemma O): in the warm cell it opens with the 500-step fade-in.
* **(b) DERIVED, the information argument.** The answer CE constrains h_S at "###" only through the LM head at the top layer, a single token. On the ≈ 85% of items the student gets wrong, its gradient there is mostly "push down the wrong tokens" (precision; STEPFACTOR D3). The L1 gives a bounded, dense target, sign(e)/σ_l, at six layers whether or not the answer is right.
* **(c) What the target contains (DATA, probe_kd, C's trunk):**
  - Under T1 (full CoT), the teacher's "###" state already holds the answer by copy: logit-lens top-1 of the first answer token is **0.75 / 0.94 / 0.96 / 0.96 at l 9–12**, against 0.00 / 0.10 at l 7–8. The student's direct state gives 0.024.
  - Under T2 (last line dropped), it does not (C was never trained on T2; top-1 0.024, equal to the student's). It holds the earlier lines' contribution instead: 56–82% residual L1 (§3).
  - So a T1 coupling at layers 9–12 is mostly **an answer-embedding regression, redundant with the answer CE**: CODI's "shortcut" in our terms. A T2 coupling asks for what lines 1..m−1 add at "###", i.e. the **earlier results** and their roles. That is the memory content STEPFACTOR D1 proves the string aux never demands (value tokens are memory-free under teacher forcing).
* **(d) DERIVED|A-OP (one pass evaluates at most one operation level; STEPFACTOR D0).** v_{m−1} has level ≥ 2 on deep items, so M_1 cannot hold it, and the T2 term's memory part needs ≥ 2 passes on m ≥ 3 items. This is the coupling's predicted lever on DEEP-MULTIPASS (P-J1e).
  - It is an end-of-chain target, at S_K only. Credit assignment through the K passes is as for the answer CE. The per-pass aux supplies plans per pass, not results.

**M3. Why CODI's no-L1 joint arm does not lift, here (PROVED, code H3 + DERIVED).**
* ∂L_T/∂φ ≡ 0 and ∂L_T/∂S ≡ 0, so the teacher changes the student only through θ, and θ acts on different inputs:
  - the think pass sees [EOT, q];
  - the decoder sees [EOT, q, "\n", "###"] plus the read;
  - the teacher sees q, "\n" and the CoT.
* So the joint CE can change **what the trunk computes from q** (the WARMCOT tilt, kept rather than forgotten) but never **what the memory is asked to carry**. That is set by L_S alone, exactly as in W.
* So J0 = W + retention, and the lift it can buy is bounded by the tilt mechanism. W tested that at init: "faster, same ceiling".
* CODI's own number: joint + LoRA without L1 = 24.5 against no-CoT 19.1, i.e. +5 over no-CoT at 40 epochs, against +19 for the coupling. Their reading: "treating them as independent tasks".

---

## 3. Where to couple

**Options:**
* (a) the decoder's "###" at all 12 layers (CODI literally);
* **(a′) the decoder's "###" at layers 7–12;**
* (b) at the read (one layer-6 target, with the decoder trunk detached so that only the memory can reduce it);
* (c) per pass: pass t's top state at the last think token against the teacher's state at the end of CoT line t (KaVa-like).

**Which path gradient descent prefers (DERIVED|WARMCOT Lemma S + gradient starvation).**
* Split the target into an item-independent offset, a q-computable part and a residual.
* The **direct path** (trunk on [EOT, q, \n, ###]) reduces the first two at first order in θ, with no product structure.
* The **memory path** reduces anything, but its gradient is a product, W_o × (item-specific read content) × (think-pass content), so it is slow wherever the read does not already carry the content.
* So gradient descent removes the offset and the q-computable part through the direct path first, and the coupling's **memory pressure is the residual**.
* **Probe (DATA, POST-HOC; probe_kd.json; S0 = the student exactly at init, W_o = 0; T2; C's trunk; normalised L1 on Aug valid, m ≥ 2, n 468; ridge fit on 600 train items).** The shares are defined as:
  - offset = (L1_init − L1_const)/L1_init, with a per-dim constant fit on train;
  - linear-direct = (L1_const − L1_lin)/L1_init, from a ridge map of the student's own S0 state at the same layer;
  - residual = L1_lin/L1_init.

| l | σ_T | L1 at init | offset | linear-direct | **residual** | R²_lin | T1 answer top-1 (logit lens) |
|---|---|---|---|---|---|---|---|
| 7 | 3.04 | 0.369 | 0.30 | 0.14 | **0.56** | 0.41 | 0.000 |
| 8 | 3.68 | 0.441 | 0.19 | 0.19 | **0.62** | 0.42 | 0.103 |
| 9 | 5.00 | 0.505 | 0.21 | 0.10 | **0.69** | 0.24 | 0.746 |
| 10 | 6.66 | 0.552 | 0.15 | 0.06 | **0.79** | 0.12 | 0.944 |
| 11 | 8.49 | 0.539 | 0.14 | 0.04 | **0.82** | 0.09 | 0.957 |
| 12 | 12.37 | 0.460 | 0.14 | 0.04 | **0.82** | 0.11 | 0.962 |
| 1–6 (reference) | 2.2–3.0 | 0.18–0.33 | 0.21–0.30 | 0.15–0.28 | 0.45–0.55 | 0.43–0.70 | — |

  - T2's own answer top-1 at l 12 is **0.024**, equal to S0's 0.024 (the linear imitation gives 0.026).
  - Raw GPT-2, for the raw design: T2's L1 at init is only 0.11–0.25 at l 7–12, and T1's answer top-1 is **0.000 at every layer**. Raw's "###" states barely depend on the CoT.
* **Reading:**
  - On C's trunk, **56–82% of the post-read L1 at init survives both a constant and a linear map of the student's own direct state**, and the share rises with depth.
  - The item-generic offset is 14–30%, and the linearly direct-reachable part only 4–19%. So on this proxy most of the coupling's item-specific signal at l ≥ 9 is memory-or-nothing.
  - Two caveats:
    1. a trained trunk is more expressive than a linear map, so the true direct share is larger;
    2. part of the residual is the teacher's CoT-length-dependent position, which nothing should reproduce.
  - Under T1, layers 9–12 carry the copied answer (top-1 0.75–0.96). This is the CODI shortcut: those layers become an answer regression.
  - Raw-init targets are weak and generic, another reason for the warm base.

**Interaction with D1 and one-pass dominance.**
* A's decoder finishes s ≥ 2 levels from q + M_1 (STEPFACTOR D4), and W's deep stratum is one-pass (§1.1).
* A T2 coupling sets the decoder's job at s = 1 (finish the last step from results in memory), and on deep items it asks M_K for results that M_1 cannot hold (M2(d)).
* So (a′) pushes against one-pass dominance on m ≥ 3, and **only there**. On m ≤ 2 items, M_1 can hold v_1, and D4's pooled s = 1 bound (0.51–0.78) still fails the pooled THINK-NECESSARY. **The pooled rule is predicted to fail; the deep multi-pass gain (DEEP-MULTIPASS) is the coupling's test.**

**Why not (a), (b) or (c) as the primary.**
* **(a):** the l ≤ 6 half carries zero memory gradient (PROVED, H1). It trains blocks 0–5, which are also the think pass's cached lower blocks, to imitate CoT-conditioned states from q alone. That is pure direct-path pressure, the decoder-copy Goodhart.
* **(b):** a single-layer, layer-6 target. The probe shows little answer content below layer 9 (logit lens T1 0.00 / 0.10, T2 0.004 / 0.006 at l 7–8). It is a new construction with no published ablation. Named as **J1-MEM** (`--kd_detach_trunk`, which closes the direct path by construction), the variant to run if J1 fails the K0-GUARD.
* **(c):** the input is identical every pass, so **the pass-varying part of a per-pass target can be met only through the memory** (PROVED: with S fixed, every pass computes the same state, so the L1 floor over passes is the per-dim median of the targets).
  - Its Goodhart risk is a free pass counter (STEPFACTOR D5: the additive read ∝ t). An item-independent component c(t) of the teacher's line-end states can be met by the counter alone.
  - Probe (T1 line-end states t 1–3, m ≥ 3, n 313, l 7–12): the counter explains only **R²_counter = 0.09–0.18** of the pass-varying variance on C's trunk, so most of it is item-specific. On raw GPT-2 it explains 0.79–0.82, i.e. raw's states are mostly generic.
  - (c) targets D1 most directly (results per pass), but it needs a pass ↔ line alignment and idle passes for t > m, and its closest prior art (KaVa 2510.02312) aligns KV caches, not hidden states.
  - **The follow-up (JOINT-L1-STEP)** if J1 lifts the pooled LOOP gain but fails DEEP-MULTIPASS.
* **Teacher format.**
  - **T2 for m ≥ 2 only**, with an answer-only CE: "###" is input-forced and not predicted, so it never conflicts with T1's "<<" after line m−1.
  - **T1 stays as C's full CE**: retention, and cot_valid stays comparable with C, W and C+.
  - For m ≤ 1 the T2 sequence is pre + a1, the **student's direct input without EOT**. Probe LEAK: L1(T2, S0) = 0.09–0.12, against 0.37–0.55 for the m ≥ 2 T2 targets at l 7–12 (n 26), near-identical states. A T2 CE there would train the K′0 path directly (PROVED, structural), so m ≤ 1 items get neither the T2 CE nor the coupling (16.4% of Aug train, 6.3% of test).
  - CODI applies the drop to every item. On our removal test that would be a built-in leak.

---

## 4. Goodhart

| # | route | why it is (not) closed | detector (logged / eval.json) |
|---|---|---|---|
| G1 | **Decoder copy (Audit iii):** the decoder reproduces the teacher state without the passes | The teacher is never run at test (H4), so the decoder can only reach z from (q, M_K). The open risk is the direct path absorbing the q-computable part (K′0 ↑) or M_1 sufficing (K′1 ≈ K′10). Closed by design: l ≤ 6 excluded (H1); m ≤ 1 excluded (LEAK) | **K0-GUARD:** K′0(J1) ≤ K′0(W) + T. **KD-ROUTE:** L_KD at S_0, S_1 and S_K on a fixed 128-item Aug-valid batch every 2k and at the end; memory share = (L_KD(S0) − L_KD(SK))/L_KD(S0), multi-pass share = (L_KD(S1) − L_KD(SK))/L_KD(S0). **DEEP-MULTIPASS**; SHORTCUT-RATIO, pooled and deep; LOOP-FORMED |
| G2 | **Match the state without the answer:** the L1 is spread over 6 × 768 dims, and the generic offset is a free reduction | The answer CE still binds the answer. The early fall of the L1 is the offset (a bias), not transfer | KD-ANS: mean L_KD on wrong vs right Aug-valid items at the end (≈ 0 gap means the coupling does not track the answer). L_KD reported against its item-independent floor (L1_const on the same batch) |
| G3 | **Scale gaming** | PROVED: σ_T is sg'd, so the student cannot move the denominator. A common rescaling c of teacher and student states at a layer (which shared weights could produce) leaves Σ\|z − h\|/σ_T invariant (degree-0 homogeneous), whereas the raw L1 is degree 1, so a uniform shrink of layers 7–12 would reduce it. Per-layer σ makes the layers equal-weighted (σ_T rises 3.0 → 12.4, i.e. 4.1× from l 7 to 12). The top-5 dims carry 1.4–3.4% of the L1, so a scalar σ is not outlier-driven. A per-dim std would blow up on near-constant dims (rejected) | logged σ_T per layer, and the ratio σ_S/σ_T |
| G4 | **Teacher pulled toward the student** | sg blocks the direct gradient (targets detached at collection, §9). Shared weights still co-move z and h at "###" (the same token, the same blocks 6–11), and a "context-blind ###" would cut the L1 for both. γ_T·L_T2 resists it: the teacher's "###" must compute y from its lines (DERIVED) | **TEACHER-CONTEXT:** the teacher's between-item variance at "###" (mean over dims of var over items / σ_T², l 7–12) every log step; a fall > 30% from its 1k value flags homogenisation. **T2-ACC** (the teacher's own in-forward answer) at each 2k |
| G5 | **Recall:** the coupling met by memorising q → z on train items (≈ 6 exposures per item) | Not closed | recall share (fit_fix − fit_rs) against W 0.34 / A 0.48. **KD-GEN** = L_KD(valid batch) / L_KD(fixed train batch) at each 2k; > 1.5 means memorised |
| G6 | **Pass counter** (option c only) | Not used in J1 | R²_counter (probe); would be re-measured in-run for c |

---

## 5. Gradient flow (from the code; one assertion per term)

| term | forward | reaches | must NOT reach | assertion (gradflow gate, fp32 CPU) |
|---|---|---|---|---|
| L_ans (A's) | `dec_loss` → `fwd_read` | g (all blocks; wte tied in g.logits; wpe), mem ln / qkv (q rows via the read, k/v via S) / gate / wo / s_add / s_del / ln_w (γ_w > 0), and the think passes via S_K | — | nonzero on g and on mem.wo (fade 1, W_o perturbed, see note) |
| L_aux (A's) | `aux_logits` → `fwd_read` on S_t | as L_ans, through S_t | — | as A (existing gradflow) |
| γ_T·L_T1 (C's CE) | `g(g.embed(x))` on pre + cot + a1 + EOS | g only | mem.*, S | grad(mem.*) is None or 0; grad(g) ≠ 0 |
| γ_T·L_T2 (answer CE, m ≥ 2) | the same plain forward on pre + lines[:m−1] + a1 + EOS, labels a1[1:] + EOS | g only | mem.*, S | as L_T1; plus the label mask starts at p_T + 1 |
| β·L_KD (l 7–12) | student: the dec_loss forward with states captured at p_S; teacher: the T2 forward with states captured at p_T and **detached at capture** | student side: g blocks 0..l−1 per term, mem read params, S_K → write path → think passes | teacher graph (the targets have requires_grad False); σ_T detached | (i) grad(mem.wo) ≠ 0 and grad(mem.qkv k/v rows) ≠ 0 with fade = 1, γ_w = 1 and W_o ~ N(0, 1e-3); (ii) **KD restricted to l 1–6 gives grad(mem.*) ≡ 0** (an index check: an off-by-one at the read site fails it); (iii) z.requires_grad is False |

* **Note (Lemma O):** at step 0 W_o = 0 and fade = 0, so every memory path, the coupling's included, has zero gradient. The gate must run at fade = 1 with W_o perturbed, or a check run at W_o = 0 passes vacuously.
* **Order and identity:** per micro-chunk:
  1. the teacher forward(s) → γ_T·L_T/micro backward, keeping detached z;
  2. the student forward → (L_S + β L_KD)/micro backward;
  3. after all chunks, the clip and step (H5).
  Gradients accumulate, so this equals one backward of the sum (JOINT-EXACT gate, fp64). The teacher's targets use θ before the step, which is CODI's semantics.
* **sg placement:** z = h.detach() in the teacher forward that also feeds L_T1 / L_T2. Its CE still uses the live graph, so there is **no `torch.no_grad` around the teacher forward**. σ_T = z[l].std() over the micro-chunk (scalar per layer). With micro 4 that is 16 items, against CODI's batch of 128. The design uses the micro-chunk std and logs it; a running mean is a variant.
* **Decorators:** a new helper inserted near the decorated `greedy_after` / `cot_batched` must not inherit `@torch.no_grad`, so there is an ast check (the memory lesson about inserting above a decorated def).

---

## 6. Design

### 6.1 Cells (RTX 3090 for every cell; seed 1; GPU and driver recorded; final checkpoint for every reading, the Aug-valid-selected one beside it; **the test is never used for selection**)

| cell | dir | init | flags (on top of W's exactly: stepaux, K 10, --cache_lower, --fast_add, --chunk_delta 64, --aux_fade 12000,12000, γ_w 2k → 6k, bs 64, lr 1e-4, 12k, --cot_valid) | role |
|---|---|---|---|---|
| **J0** | `gsm_lat_jcot_s1` | C's model.pt (`--init_from`) | `--joint_cot 2` (T1 CE + T2 answer CE, γ_T 2) | protection by the joint CE (CODI's "w/o L1") |
| **J1** | `gsm_lat_jcotl1_s1` | C's model.pt | J0 + `--kd_beta 1 --kd_layers 7-12 --kd_teacher drop --kd_norm teacher_std --kd_loss l1` | + the coupling |
| W, C+, A, C | landed | — | — | comparators ($0) |

* **The same batches:** all cells draw seed 1's index sequence (WARMCOT F7). J0 / J1 / W see the identical first 12k batches.
* **Weights:** α = 1 (A's student: answer + λ aux), β = 1 (CODI's GPT-2 value), γ_T = 2. γ_T = 2 is CODI's best at the shortest budget reported (20 epochs, Table A5), and in the warm design it gives 4× less excess CoT loss than γ_T = 1 (M1b) at a student share s_S = 0.95 (M1c).
* **The lr stays 1e-4, full fine-tuning.** It is A's / W's / C's lr in this parametrisation. CODI's 3e-3 belongs to LoRA (scale α/r = 0.25). Its per-step entrywise ΔW is ≈ 1.8e-4 with random-sign accumulation over r and up to 1.7e-3 when coherent (calc, PEFT-default A init; CODI's init is unverified), so transplanting 3e-3 to full fine-tuning would be a 30× step.

### 6.2 Readings (a new theory/score_s20x_jcot.py, fixture-checked before any output; T = 0.028, T_D = 0.034 on m ≥ 3)
* **COUPLE** = J1 − J0 (paired): HELPS ≥ +T / HURTS ≤ −T / NEUTRAL. **The method reading.**
* **PROTECT** = J0 − W, read the same way.
* **TOTAL** = J1 − W. Beside it J1 − A (Audit's ii; the exposure is 24k text + 12k joint against 24k latent, stated) and **R1-J** = J1 − C+ (equal exposure).
* **COT-RETAIN** (J0, J1; Audit's i): min over the 2k records of cot_valid ≥ 0.30.
* **LOOP-FORMED:** K′10 − K′0 ≥ T. **THINK-NECESSARY** (pooled, as registered) and **-DEEP** (m ≥ 3, note 202-1).
* **DEEP-MULTIPASS:** K′10 − K′1 on m ≥ 3 ≥ T_D.
  - W reads 0.005 and A 0.013. The within-model paired SE is 0.009 (W, 64 discordant of 879).
  - This replaces a ratio reading: W's deep K′1/K′10 = 0.95 has a bootstrap SE of **0.094**, too noisy for a one-seed threshold.
* **SHORTCUT-RATIO:** K′1/K′10, pooled and deep. Descriptive, with its bootstrap SE.
* **K0-GUARD:** K′0(J1) ≤ K′0(W) + T = 0.139.
* **KD-ROUTE:** memory share ≥ 0.10 at the final checkpoint (G1).
* **Descriptive:**
  - recall share; KD-GEN; KD-ANS; TEACHER-CONTEXT; T2-ACC; σ_T by layer;
  - valid and cot_valid curves; aux exact by pass and by grounding; path_indep; IDLE; STABLE-12 / 16; FILLER;
  - GSM-Symbolic (eval-only, local); timed s/item and Meter FLOPs;
  - basin entry (descriptive, with entry steps for every J vs W contrast).
* **The deep fields** (acc_by_m_kprime) come from the training host's eval, and the W reference is re-run **on the same card and bs as J0 / J1's eval** (the bf16 batch-size rule).

### 6.3 OC block (rule a; one seed per cell; calc_jointcot.json)
* The paired SE of a difference is 0.0104 (A vs W, 187 discordant); 0.0135 is the high case. Seed SD σ_s is per run, and the two runs are independent.

| σ_s | P(read ≥ T \| Δ 0) | Δ +0.02 | Δ +0.03 | Δ +0.04 | Δ +0.06 |
|---|---|---|---|---|---|
| 0 | 0.003 (0.019 at SE 0.0135) | 0.22 | 0.58 | 0.88 | 1.00 |
| 1 point | 0.055 | 0.32 | 0.55 | 0.75 | 0.97 |
| 2 points | 0.18 | 0.40 | 0.53 | 0.65 | 0.86 |

* **Deep (T_D):** P(≥ T_D \| 0) = 0.003 / 0.036 / 0.13, and P(≥ T_D \| +0.05) = 0.89 / 0.80 / 0.69 at σ_s 0 / 1 / 2 points.
* **Reading:** at the central COUPLE (+0.03) the read is a coin flip, and the false-pass rate is up to 18% if the seed SD is 2 points (unknown on GSM8K; A vs B vs W differ by ≤ 1.3 points across recipes). **A single-seed COUPLE-HELPS is worded "on seed 1".** A rate claim needs ≥ 3 seeds per cell (stage 2, a new decision).
* **Realised $ of comparable cells:** A ≈ $3.0 (12.1 h at $0.248/h), W ≈ $1.10 (+ $0.20 for failed hosts), C+ ≈ $0.37, all of §200 $7.8.

### 6.4 Censoring (rule b)
* **COT-RETAIN:** a missing 2k record (crash, failed eval) counts as a value below 0.30, i.e. a failure. The event-only minimum is printed beside it.
* **KD-ROUTE onset:** the first log step with memory share ≥ 0.10. Never reached means a failure, printed as "> 12k", with the event-only mean onset beside it (seeds > 1).
* **VALID-CEILING:** the first 2k record with Aug-valid ≥ 0.154 (W's best). Never reached is a failure for the "faster" descriptor.
* **A cell that dies before 12k:** every reading of it is a failure, not missing. Logs are kept (*.void), and a re-run happens only under a new name and with approval.

### 6.5 Coupling vs trunk protection (Audit's ask)
* **Which matters more at about 4 epochs: the coupling (J1 − J0)** (DERIVED, M3 + CODI data):
  1. protection only keeps the trunk's q-features, and the memory's targets are unchanged;
  2. CODI's protected, joint, no-L1 arm sits 5 points above its no-CoT;
  3. our forgetting mechanism (M1a) needs γ_T = 0, and the joint CE removes it in every cell.
* **So full fine-tuning is the right base:**
  - it is A's / W's / C's parametrisation and lr;
  - the joint CE does the protecting (M1b), and COT-RETAIN checks it;
  - neither cell has LoRA, so a CODI-shaped J1 − J0 is attributable to the coupling.
* **Conditional third cell J1-LoRA** (C's trunk frozen + LoRA r 128 / α 32 at 3e-3 on the trunk, DeltaMem at 1e-4, otherwise J1; ≈ $1.9 / $2.8): run **only if COT-RETAIN fails in J0 or J1**. That is where protection, not coupling, would be the binding factor. It would be a separate decision.
* **If only one cell is affordable:** J1 alone (TOTAL vs W). Its gain is then "joint CE + coupling", not separable, and it is worded so.
* **The raw 24k pair** (CODI's convention, vs A; $7.8 / $11.7) is not recommended: M1d predicts a weak teacher (cot_valid ≈ 0.21 at γ_T 2) for most of the run.

---

## 7. Predictions / falsifiers (seed 1; RTX 3090; final checkpoint; central values)

| id | prediction | from | falsified if |
|---|---|---|---|
| P-J0a | COT-RETAIN(J0): min cot_valid ≥ 0.30 (central 0.33; W 0.07–0.13) | M1a/b | < 0.30 at any 2k record, censoring included: then M1b's fixed point at γ_T 2 is too loose, and J1-LoRA becomes the next test |
| P-J1a | COT-RETAIN(J1) ≥ 0.30 (central 0.32) | M1 | < 0.30 |
| P-J0b | **PROTECT NEUTRAL** (J0 − W central +0.01) | M3 + CODI 24.5 | ≥ +T: retention alone lifts, and M3's bound is too pessimistic (the tilt mechanism works once kept) |
| **P-J1b** | **COUPLE-HELPS: J1 − J0 ≥ +T** (central +0.03; P ≈ 0.55) | M2 | NEUTRAL or HURTS. With KD-ROUTE ≥ 0.10 that falsifies M2's "transmitted content is answer-relevant" (G2); with KD-ROUTE < 0.10 the coupling never entered the memory (G1) |
| P-J1c | TOTAL: J1 − W ≥ +T, J1 test ∈ [0.15, 0.25] (central **0.185**); J1 − A ≥ +T (Audit ii) | M1 + M2 | J1 − W < +T |
| P-J1d | R1-J: J1 − C+ reads **below** (central −0.15) | WARMCOT D4 (ρ < 1) | matches or above |
| P-J1e | LOOP-FORMED (central K′10 − K′0 = 0.06). THINK-NECESSARY pooled **fails** (central K′1/K′10 0.75). **DEEP-MULTIPASS: deep K′10 − K′1 ≥ T_D** (central 0.04; W 0.005, A 0.013). OC: P(read \| true 0.005) ≈ 0.001, P(read \| true 0.04) ≈ 0.75 at SE 0.009 (seed variance not included) | M2(d), D4 | deep K′10 − K′1 < T_D: no multi-pass lift on deep items. Pooled THINK-NECESSARY passing is *not* a falsifier |
| **P-J1f** | **K0-GUARD holds** (central K′0 0.10 ≤ 0.139) | §3 (l ≤ 6 and m ≤ 1 excluded) | K′0 > 0.139: the gain is read as direct-path strengthening (Goodhart G1), not loop transfer, and J1-MEM is the next variant |
| P-J1g | KD-ROUTE: memory share ≥ 0.10 (central 0.2); multi-pass share ≥ 0.02 (central 0.05) | §3 split | memory share < 0.10 |
| P-J1h | recall share ≥ 0.25 (central 0.35), descriptive | WARMCOT D7 | ≤ 0.10 |
| P-J1i | KD-GEN ≤ 1.3 (central 1.15) | G5 | > 1.5 (a memorised coupling) |
| P-J1j | T2-ACC rises from C's 0.024 to ≥ 0.15 by 12k (central 0.25), descriptive | M1 | stays < 0.05 (the T2 target never becomes informative) |

* **Without anchoring on 43.7:** CODI's lowest budget point is 20 epochs (38.4 at γ 1, 41.6 at γ 2). Ours is 4 text + 2 latent epochs, a 6 + 6-head memory read at one layer (not CODI's six latent tokens with full attention), and full fine-tuning. The derivation gives the direction of COUPLE, not its size: **size CONJECTURE**.
* **No derivation predicts SOTA or a C3 match** (R1-J below).

---

## 8. Cost (calc_jointcot.json; $0.21–0.28/h, central $0.25; + $0.10 per host; eval ≈ 0.1 h (A 192 s, W 136 s); high = 1.35 × hours at $0.28)

* **Per-step:** A's realised 1.820 s/step (latent, micro 4), plus the teacher:
  - T1 = C's realised 0.255 s/step (micro 1; mean CoT 28 tokens);
  - T2 = 0.776 × T1's tokens per item (the m ≥ 2 share included, 4,000 Aug-train items);
  - so **+0.452 s/step (+25%)**.
  - The capture and L1 cost under 1%. The KD diagnostics are two no-grad decoder forwards on one chunk every 100 steps.
  - At W's realised host speed (1.103 s/step) the teacher adds 41%.

| cell | hours | $ central / high | on a W-speed host |
|---|---|---|---|
| J0 or J1 (warm, 12k) | 7.7 | **2.02 / 3.00** | 1.42 |
| **pair J0 + J1** | 15.3 | **4.04 / 6.00: FLAG (> $4; under Audit's $10)** | 2.84 |
| J1 only | 7.7 | 2.02 / 3.00 | 1.42 |
| J1-LoRA (conditional) | ≈ 7 | ≈ 1.9 / 2.8 | — |
| raw 24k J (one cell) | 15.2 | 3.91 / 5.86 | — |
| raw 24k pair | 30.5 | 7.82 / 11.73: **the high is over $10, the user's call** | — |

* Gates are about $0.1 (one 3090 `--time_steps 200`); CPU gates are $0.
* **`--time_steps 200` on a 3090 replaces the projection, and it is restated to Audit and the user before launch.** Realised $ replaces the projection at landing.

---

## 9. Engineering contract (Experiments implements; Audit reviews the diff; every new parameter optional; no module edit while a run lazily imports these files)

**Flags, in gsm8k_stepaux.py only.** ProsMem, deltas5.fwd_read and core are shared with §168s2 / §191 and stay untouched.
* `--joint_cot G` (default 0 = off): γ_T. T1 = the textcot arm's exact sequence and labels (l.243–249). T2 = pre + lines[:m−1] + a1 + EOS for m ≥ 2, with labels only on a1[1:] + EOS. Logged: t1_ce, t2_ce.
* `--kd_beta B` (default 0), `--kd_layers 7-12`, `--kd_teacher drop|full`, `--kd_norm teacher_std|none`, `--kd_loss l1|smoothl1`, `--kd_m_min 2`.
* `--kd_detach_trunk` (default off; J1-MEM only): the student-side capture runs a decoder forward with the trunk weights detached (the read live).

**Capture.** Two new functions in gsm8k_stepaux.py:
* `fwd_plain_layers(g, x, pos)`, mirroring GPT2KV.forward with DM._block, returns (final hidden, {l: h_l[rows, pos]}).
* `dec_loss_layers(m, S, its, dev)`, mirroring ProsMem.dec_loss, returns (CE, states at p_S).
* **Gates:** their outputs equal the originals bitwise on the plain path. probe_kd's same construction gave max |Δ| = 0.0 for both trunks against GPT2KV.forward (fp32 CPU).

**Positions.** Assert ids[p_S] == ids[p_T] == ANS in every batch, and T2 == T1 minus exactly the last line.

**Logging (per log step).**
* L_KD per layer; σ_T per layer; σ_S/σ_T; TEACHER-CONTEXT;
* KD-ROUTE (S_0 / S_1 / S_K, no grad);
* per-term trunk grad norms every 500 steps (separate backwards on one chunk), with κ̂ = ‖g_S‖/‖g_T‖ for M1c.

**At each valid eval:** cot_valid_acc (exists), t2_valid_acc, kd_valid / kd_train (KD-GEN).

**eval.json (additive):** kd_route{SK, S1, S0, share_mem, share_multipass}, kd_ans, t2_acc, cot_valid_final, acc_by_m_kprime (exists).

**Memory.** Teacher first, its graph freed after its backward, then the student. The peak is ≈ max(A's 16.9 GiB, the teacher ≈ 3.9 + 3.0 GiB per 16-item chunk), so micro 4 holds. One combined backward would be ≈ 24 GiB and is not allowed.

**Gates (before launch):**
1. **Plain-path bitwise:** all new flags off; a CPU fp32 construct + 1 step from a fixed seed gives a bitwise-identical loss, grads and params against the pre-edit file.
2. **JOINT-EXACT (fp64, CPU, 4 items):** the two-backward update equals one backward of the sum (relative max |Δ| ≤ 1e-12).
3. **Per-term gradflow:** §5's table, every assertion (fade 1, γ_w 1, W_o perturbed), plus the ast decorator check.
4. **Positions and formats:** the asserts above, plus a 200-item round trip (T2 re-rendered == T1 minus its last line).
5. **mem_check** at the composite longest item with all terms on; ≤ 21 GiB at micro 4.
6. **INIT-IDENTITY** (§202's gate: step 0 = C's trunk; K′ constant).
7. **Eval-path invariance (C1):** an eval-only run of a 50-step smoke model with and without the joint flags gives bitwise-equal eval fields.
8. **`--time_steps 200`** for J1 on a 3090.
9. **Scorer:** a field-path grep plus a synthetic-run fixture before any output (the scorer field-path rule).

**C1 (PROVED, code, given the contract):**
* The think input is identical on every pass (the --cache_lower assert is unchanged).
* The memory is the only cross-pass variable.
* The decoder is read-only (fwd_read never writes).
* The teacher, T2 and the coupling exist only inside the training loss. `answers()` / `think_states` are untouched, and gate 7 checks it.
* At test there is no CoT text, and K′0 is S = 0 (the read adds only fade·b_o).

---

## 10. Prior art (IDs verified on the arXiv abstract pages, 2026-10-14)

| work | what it does | relation |
|---|---|---|
| **CODI** 2502.21074 (v3 full text read, §1.3) | a joint teacher CE + latent student + an all-layer L1 at the pre-answer token, sg on the teacher, teacher-std normalisation, the final step dropped, LoRA | **the method transplanted**. Our changes: post-read layers only, T2 for m ≥ 2 with T1 kept, full fine-tuning, a recurrent identical-input memory with a single-layer read |
| Coconut 2412.06769 | a CoT stage 0, then a latent curriculum | the W base is its stage-0 convention |
| ICoT-KD, Deng et al. 2311.01460 | teacher hidden states distilled "vertically" into a no-CoT student | the hidden-state-distillation ancestor. Note: it distils into the **direct path**, the route our l ≤ 6 exclusion closes |
| SIM-CoT 2509.20317 | step-level supervision of each latent by an auxiliary decoder | the closest to our per-pass aux; complementary to the coupling |
| PCCoT 2506.18582 | CODI-family Jacobi-parallel latents | the SOTA bar (49.48) |
| **KaVa** 2510.02312 | self-distillation of a teacher's compressed KV cache into latent tokens, aligning stepwise KV trajectories | prior art for the per-step coupling (option c) |
| FitNets 1412.6550 | hint-based intermediate-layer distillation | the generic reason a dense hidden-state target speeds optimisation (CONJECTURE transfer) |
| LoRA 2106.09685 | low-rank adapters | CODI's parametrisation; J1-LoRA |
| Orgad et al. 2410.02707 (title verified; content not read) | as CODI cites it: the token just before the answer "encodes essential reasoning information" | CODI's justification for the distillation token |
| Dilgren & Wiegreffe 2604.04902 | post-hoc removal tests on Coconut / CODI | ours is pre-registered with the run |

* **Novelty: PARTLY KNOWN.** The coupling is CODI's, the init is Coconut's, and the stepwise variant is KaVa-adjacent. New are:
  1. the read-site layer restriction, derived (H1);
  2. the T1-retain / T2-target split, which closes CODI's m ≤ 1 direct-answer leak for a removal test;
  3. the coupling's memory-route share (L_KD at S_0 / S_1 / S_K) as a pre-registered Goodhart detector beside K′0 / K′1;
  4. the crowding account (κ) of why a raw-init joint teacher stays weak at about 4 epochs.

---

## 11. Claim status

| claim | status |
|---|---|
| H1–H6 (read site, positions, plain teacher, eval path, clip, decorators) | PROVED (code) |
| l ≤ 6 coupling terms carry zero memory gradient | PROVED (code, H1) |
| teacher CE never reaches φ or S (M3) | PROVED (code, H3) |
| M1a first-order L_T change; forgetting unopposed at γ_T 0 | PROVED |
| excess CoT loss ∝ 1/γ_T² | DERIVED\|A-QUAD |
| Adam teacher share s_T = γ_T/√(γ_T²+κ²); κ ≈ 5.9 | DERIVED\|A-ADAM; κ DATA |
| raw-init teacher cot_valid ≈ 0.21 at γ_T 2 by 24k | calc under A-ADAM + gradient-flow time |
| T1 target ≈ answer copy at l 9–12; T2 holds earlier-line content | DATA (probe, C's trunk; POST-HOC) |
| T2 coupling needs ≥ 2 passes on deep items | DERIVED\|A-OP |
| direct path absorbs the offset / q-computable part first | DERIVED\|Lemma S + gradient starvation; linear-proxy DATA |
| normalised L1 is invariant to common rescaling | PROVED |
| per-pass target's pass-varying part is memory-only | PROVED |
| m ≤ 1 T2 = the student's direct input (leak) | PROVED (structural) + DATA |
| COUPLE > PROTECT at about 4 epochs | DERIVED (M3) + CODI DATA (24.5 / 43.7) |
| COUPLE size (+0.03), J1 0.185 | CONJECTURE |
| C1 preserved | PROVED (code, given the §9 contract and gate 7) |

---

## 12. Risks
1. **The end-of-chain target.** The coupling acts at S_K, and credit through K passes is the answer CE's problem too. The deep stratum may stay one-pass (W's is), in which case DEEP-MULTIPASS fails and (c) is the follow-up.
2. **Crowding and conflict size are unmeasured in the joint run.** κ comes from separate runs, and the per-parameter κ is unknown. COT-RETAIN and κ̂ (logged) read it, and J1-LoRA is the fallback.
3. **The T2 cold start.** C never answered after m − 1 lines (T2 top-1 0.024), so until T2-ACC rises the coupling pulls toward an untrained state. The W_o fade (500) and γ_w (2–6k) delay the memory route anyway. No β warm-up, as in CODI.
4. **The narrow channel.** One read at block 6 (6 + 6 heads, 64 × 64) must carry the earlier results in a form blocks 6–11 turn into the teacher's states. CODI's latents are attended at every layer.
5. **σ over 16 items** (micro 4) is noisier than CODI's 128. A running mean is a variant if σ_T's coefficient of variation exceeds 0.2 in the smoke.
6. **Exposure.** J vs A mixes init and exposure; C+ is the equal-exposure text control.
7. **One seed, one GPU class** (3090). Same-seed only, entry steps beside every contrast.
8. **POST-HOC.** §200 / §202, W's deep row and the probes were all seen while drafting. CODI's unverified details are in §1.3.

## 13. Files (theory/jointcot/)
* `probe_kd.py` → `probe_kd.json`, `probe_kd.log`:
  - T1 / T2 / S0 states at "###" (raw GPT-2 and C);
  - L1 decomposition; answer content; COUNTER; LEAK.
  - CPU fp32, 3 threads, Idle, 23 min (1,403 s). Its forward is bitwise equal to GPT2KV.forward (gate 0.0).
* `calc_jointcot.py` → `calc_jointcot.json`: κ, shares, the raw-teacher mapping, OC, cost, LoRA rate.
* Rerun: `python -I <script>` from this folder.
