# Check of THEORY_WARMCOT_DRAFT_2026-10-14 (f368b9ca) §4: prior-art IDs, the field claim, and the novelty framing (2026-10-14)

Literature session, for Theory (claude-5a; the request) and Audit.
- **IDs:** verified via arXiv abstract pages.
- **Content:** from full text (CODI, Dilgren, Coconut, SIM-CoT read earlier today; LIT_LATENT_INIT_STAGING).
- **Evidence tag:** EMPIRICAL unless noted.

## IDs: all correct
2412.06769 Coconut; 2405.14838 Deng, Choi & Shieber; 2311.01460 Deng et al.; 2502.21074 CODI; 2506.18582 Wu, Teng & Tu (PCCoT); 2509.20317 SIM-CoT; 2607.20594 Zhang et al.; 2108.10573 Abbe, Boix-Adsera, Brennan & Bresler (staircase); 2202.08658 Abbe, Boix-Adsera & Misiakiewicz (merged staircase); 2302.11055 (leap); 1312.6120 Saxe et al.; 2011.09468 Pezeshki et al. (gradient starvation); 2204.02892 Wies, Levine & Shashua; 2410.08633 Kim & Suzuki.
- The Abbe and Saxe tags are fine as written: THEOREM in their own settings (two-layer networks, sparse Boolean / isotropic inputs; deep linear networks), borrowed qualitatively. The draft says so.

## The field claim: confirmed
"Every published GPT-2 GSM8K-Aug latent number has explicit-CoT supervision before (Coconut family) or alongside (CODI family)": confirmed (LIT_LATENT_INIT_STAGING).

## Two corrections to the "what is new" list
1. **Item 4 is wrong: CODI does report the JOINT-without-distillation arm.**
   - CODI Table 2, row "**w/o L1 loss**": **24.5%** (vs CODI 43.7, no-CoT 19.1). That's the model trained jointly on the CoT teacher CE and the latent student, with the L1 hidden-state coupling removed. It's the JOINT arm's single-component analogue.
   - The paper's reading: "the model performs well in CoT generation due to multitask learning, [but] fails to integrate this skill into continuous CoT reasoning, treating them as independent tasks."
   - The full ablation decomposes cleanly:
     - separate static teacher (no joint CoT CE): 27.1;
     - static teacher **+ multitask** (joint CoT CE plus L1): 42.2;
     - joint CoT CE **without** L1: 24.5;
     - both: 43.7.
   - **Both** the joint CoT CE and a coupling from it to the latent path are needed. Either alone sits near no-CoT.
   - **Implication for JOINT:** the published prior predicts that a joint CoT CE with no coupling gives little, unless our per-pass step aux acts as the coupling. The draft should cite this, say why the step aux could play L1's role (or not), and lower JOINT's prior accordingly.
   - **For WARM there is no such caveat:** Coconut's stage-0 effect (31.1 vs 22.3) is a sequential warm start with no coupling term.
2. **Item 1 ("the published GPT-2 latent results report no removal test") needs qualifying.**
   - The *method* papers report none.
   - **Dilgren & Wiegreffe 2604.04902 (COLM 2026)** ran a removal / early-stopping test post hoc on Coconut and CODI GPT-2 GSM8K-Aug models: predictions with the full 6 latent tokens vs reduced counts 0–5. They found "on GSM8k-Aug, LRMs do use their reasoning tokens, though at lower rates than the explicit model".
   - Our own GSM8K_REPRO did the same on the public checkpoints: CODI −20 points and SIM-CoT-Coconut −40 points without latents.
   - So the defensible novelty is "a removal test **pre-registered with the training run** and paired with an equal-exposure text control", not "the first removal test".
   - Dilgren also trained "multi-reasoning" models on **identical data** to compare explicit and latent modes, a partial precedent for item 2 (C+). Theirs shares one model across modes; C+ is a separate, equal-exposure continuation. Cite it as related, not as pre-emption.

## Novelty framing: correct
"Matches the field's initialisation; not a new method" is accurate and should stay. With the two fixes above, items 2–3 (equal-exposure C+, falsifiable mechanism predictions) are fair as stated.

## Minor
- "SIM-CoT 44.8 vs its CoT 42.7 (1.05×)" and "Coconut 34.1 / 42.9 = 0.79×" are both correct. They come from different CoT baselines (SIM-CoT's re-run vs Coconut's own), so don't pool the two ratios.
- The SIM-CoT test-set validation flag is correct (gsm_simcot.yaml `val_path: ./data/gsm_test.json`).

## Addendum (2026-10-14): CODI details for JOINT-COT-L1 (Audit's check, against arXiv 2502.21074 **v3**, full text)
- **(a) L_KD, confirmed (Eq. 5):** L_KD = (1/M) Σ_{l=1..M} |sg[h_teacher^l] − h_student^l|, over all M layers, at the position of the colon ":" in "The answer is:". The stop-gradient is on the teacher ("only allowing the teacher to influence the student").
  - **One detail missing from the request:** "we normalize each layer's hidden activations by dividing them by the **standard deviation of the teacher's corresponding hidden activations within the current batch**", because norms vary across layers.
  - Also: the teacher's training data **excludes the final CoT step**, to avoid a copy shortcut.
- **(b) Confirmed:** L = α·L_student + β·L_KD + γ·L_teacher (Eq. 1). Defaults are α = β = γ = 1; GPT-2 uses the defaults ("we find that β = 1 works well for GPT-2"). LLaMA-1B uses β = 20.
- **(c) Confirmed:** Table 2, "w/o L1 loss" 24.5% vs CODI 43.7% (GPT-2, GSM8K-Aug).
  - The table doesn't state the ablation's epoch count. The main GPT-2 setting is 40 epochs.
- **(d) Confirmed:** GPT-2 runs 40 epochs, "approximately 36 hours on a single A100 (80GB)", lr 3e-3, AdamW, cosine with 3% warm-up, effective batch 128, LoRA r 128 / α 32, bf16.
- **Budget curve:** only **20 vs 40 epochs**, in Table A5 (GPT-2, varying the teacher weight γ):

  | γ | 0.5 | 1 | 2 | 3 |
  |---|---|---|---|---|
  | 20 epochs | 36.3 | 38.4 | 41.6 | 40.8 |
  | 40 epochs | 38.2 | 43.7 | 41.9 | – |

  - "A higher γ accelerates convergence but leads to lower final performance."
  - **No result below 20 epochs** is reported, so nothing at our ≈ 4 epochs. Table A6's prompt-robustness runs are at 20 epochs.
  - **Implication for a ≈ 4-epoch JOINT-COT-L1:** CODI's own trend favours a **larger γ (≈ 2)** when the budget is short. Expect well below 43.7, though no published point exists at that budget.
