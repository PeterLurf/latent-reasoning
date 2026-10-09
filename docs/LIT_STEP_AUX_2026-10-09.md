# Prior check for STEP-AUX: a SIM-CoT-style per-pass auxiliary step decoder on INJECT / READ-SLOTS memory (removed at test) (2026-10-09)

Literature session, for Audit (the lead) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Details:** from the full HTML of 2509.20317 and 2606.20075, and SIM-CoT's code (read for LIT_S149_SOTA_PROBE_FACTS).

## 1. SIM-CoT 2509.20317
- **Authors:** Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, et al. (InternLM).
- **GPT-2 on GSM8K-Aug, absolute numbers** (Table 1, as in LIT_BENCH_VERIFY / GSM8K_REPRO):
  - **SIM-CoT + Coconut 44.8** vs *their* Coconut re-run 36.6, so **+8.2**;
  - SIM-CoT + CODI 42.6 vs their CODI re-run 42.0;
  - explicit SFT-CoT 42.7, so **+2.1 for SIM-CoT + Coconut over explicit CoT**.
  - Our bench reproduced SIM-CoT + Coconut at **44.8** (checkpoint_28) and SIM-CoT + CODI at 39.4 (not reproduced, −3.2).
  - Reported OOD: SVAMP, GSM-Hard, MultiArith.
- **Loss:** L = λ_step L_step + λ_lm L_ans-lm, with L_step = −Σ_k Σ_t log p_φ(y_{k,t} | z_k, y_{k,<t}): CE on step k's text tokens, given latent k.
  - **No numeric λ is stated in the text.** The code's default `explain_loss_factor` is **1.0** (SIM-CoT CODI/src/model.py:116). Treat it as
    λ_step = λ_lm = 1 unless their scripts override it.
- **What the aux decoder reads: the latent only (plus a step-index prompt), not the question.**
  - Coconut variant: the decoder's input = the latent pair's embeddings preceded by the text "Step k of the solution". The last pair gets
    "Step 3 and all the remaining steps" (coconut.py:483–485).
  - CODI variant: latent j concatenated with step j's tokens.
  - The decoder is a separate model (φ), discarded at inference.

## 2. Per-iteration aux decoding on looped / recurrent-depth models; extrapolation with vs without step supervision
- **Huginn 2502.05171 / Ouro 2510.25741:** **no per-iteration step supervision** (LM pretraining; Ouro adds entropy-regularised exit).
  - Lu et al. 2507.02199 decode Huginn's per-iteration states post hoc (logit lens) and find little interpretable latent CoT. That's
    analysis, not training.
- **Recurrent-depth retrofit 2608.11233: the closest "per-loop step supervision" precedent.**
  - Intermediate-step supervision trains **one task step per loop**; the mechanism **persists after outcome-only annealing**.
  - **It extrapolates to about 1.5× the supervised depth.** A same-size **scratchpad (text) model matched it within the trained horizon
    but collapsed beyond it** (84% vs 72% overall).
- **LOTUS 2606.31779:** gold step-text CE per latent (no looping).
- **Without step supervision:**
  - Fan 2409.15647 (no intermediate supervision, adaptive loops; length generalisation);
  - 2604.07822 (multi-hop depth extrapolation from scratch);
  - **ProsQA-Ext 2609.35643:** latent models "optimised end to end *without* intermediate reasoning traces" generalise to 7–12 hops better
    than CoT and pause models, which lean on local graph features.
- **No paper ablates the *same* looped model with vs without a per-iteration aux decoder on extrapolation.** That comparison would be new.
- **The flag for STEP-AUX and C2:**
  - Binding pass t to CoT step t teaches a *training-length* schedule. Passes beyond the longest training CoT have no target.
  - SIM-CoT pools the tail ("all the remaining steps") for that reason.
  - Expect the aux to help in-distribution and entry (cf. our §111 READ). Its effect on K beyond training is open, and could be negative (the
    retrofit's 1.5× ceiling).
  - **Pre-register a K′ > K test with vs without aux, and fade the aux (as §142 / 2608.11233's annealing) before reading C2.**

## 3. 2606.20075, "What Makes Effective Supervision in Latent CoT" (Chen, Leong, Guo, Wang, …; 2026-06-18)
- **Diagnosis:** outcome-only supervision causes a **"dual collapse"**: gradient attenuation along the latent chain, and representational
  drift (semantic drift of latents).
- **Process supervision has two complementary dimensions:**
  - **Trajectory supervision:** dense stepwise signals.
  - **Space supervision:** preserving the latent manifold's semantic structure.
- **The key result for STEP-AUX:**
  - **Rigid geometric compression** (forcing latents into alignment with token embeddings or teacher states) acts as a "destructive prior"
    that **collapses the reasoning space**.
  - **Generative reconstruction** (an *auxiliary decoder* recovering step text S_t from latent L_t: L_GR = −log D_ψ(S_t | L_t), with ψ trained
    jointly; SIM-CoT is the cited example) is a flexible "semantic tether" that **better preserves information capacity**.
- **A diagnostic:** the Unified Latent Probe (ULP). A shared conditional decoder recovers step text from frozen latents, as an estimate of the
  mutual information between latents and steps. "Information-performance binding": accuracy tracks the recoverable step information.
  - GPT-2 is among the models used.
- **For us:** STEP-AUX is the generative-reconstruction form, which this paper favours over hidden-state alignment (CODI / KaVa-style). The
  ULP is a cheap descriptive to log (step recoverability per pass).

## Verdict
- New as scoped: a per-pass generative step decoder on a pretrained LM's *memory* state, under INJECT / READ-SLOTS.
- The precedents favour generative reconstruction (SIM-CoT; 2606.20075).
- The open risk is C2: per-pass step binding may cap extrapolation.
- **Pre-register:** a K′ > K test with vs without aux, an aux fade before C2 is read, a decoder that sees only the memory (as SIM-CoT), and ULP-style
  per-pass recoverability as a descriptive.
