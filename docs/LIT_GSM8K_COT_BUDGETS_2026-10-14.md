# Training budgets behind the published GPT-2 GSM8K-Aug CoT-SFT bars, against our §200 TEXT-COT (0.321 at 4.0 epochs) (2026-10-14)

Literature session, for Audit and Theory (both asked).
- **Sources:** full text (arXiv PDFs via pdftotext) and the Coconut repo's `args/gsm_cot.yaml` (GitHub, main).
- **Evidence tag:** every row is EMPIRICAL.
- **Exposure:** epochs × 385,620 training items. Ours: 24k steps × bs 64 = 1.54M examples = **4.0 epochs**.

| CoT-SFT bar (GPT-2 small) | epochs (× ours) | batch | lr / schedule | full FT / LoRA | checkpoint rule | source |
|---|---|---|---|---|---|---|
| **Coconut's CoT, 42.9 ± 0.2** | **25 (6.3×)** | 32 × 4 GPUs = 128 | **1e-4**, constant (no scheduler in the config), wd 0.01, fp32 | full FT | `save_only_improve: True` on gsm_valid, i.e. **best validation** epoch | 2412.06769 Table 1; repo gsm_cot.yaml |
| **Dilgren & Wiegreffe CoT = our bench checkpoint, 41.8 (bench)** | **25 (6.3×)**; the bench uses `checkpoint_25` | 128 | 1e-4, wd 0.01, no bf16 | full FT (Coconut code) | checkpoint_25 (the last epoch) | 2604.04902 App. C, Table 6 |
| **CODI-reported CoT-SFT, 44.1** | **40 (10×)**: the A-section says the setup holds "for all experiments (CoT-SFT, No-CoT-SFT, and CODI)"; the 40 epochs is stated for GPT-2 | 128 | **3e-3**, cosine, 3% warm-up | **LoRA r 128, α 32**, bf16 | not stated | 2502.21074 App. A |
| **SIM-CoT re-run, 42.7** (reused by LOTUS) | **not stated** for CoT-SFT | 128 | Adam (0.9, 0.999), wd 0.1; CoT-SFT lr not stated | not stated | **early stopping, patience 3 epochs** | 2509.20317 App. E |
| **iCoT (Deng 2023) explicit CoT, 40.7% (0.41)**; ICoT-SI's 0.41 matches it, so it's probably reused | **15 (3.75×)** | **32** | **5e-5** (AdamW) | full FT | not stated | 2311.01460 App. E, Table 4; 2405.14838 Table 3 |
| PCCoT's "CoT 44.1" | copied from CODI ("The CoT baseline is taken from CODI") | | | | | 2506.18582 §4.2 |
| LOTUS's "explicit CoT 42.7" | SIM-CoT's numbers, reused (LIT_GSM8K_AUG_BARS) | | | | | 2606.31779 |
| *(latent, for reference)* PCCoT 49.48 | 40 | 128 | 3e-3, cosine, warm-up 0.03, wd 0.01, clip 1.0 | LoRA r 128, α 32 | not stated; seeds 0 / 1 / 2 | 2506.18582 Table 7 |
| *(latent)* LOTUS 44.1 | 30 | 128 | | | best GSM8K **validation** ("final-epoch checkpoints perform similarly") | 2606.31779 |

**CoT format:** all of the above use Deng et al.'s **GSM8K-Aug** (equation-only steps, e.g. `<<600*.4=240>>`). The -NL variant is separate and gives lower numbers: CODI-reported CoT on NL is 34.8.

## Reading
1. **Audit's prediction holds wherever a budget is stated.** Every published CoT bar trained on **≥ 3.75× our exposure**:
   - iCoT: 15 epochs → 40.7, the smallest budget;
   - Coconut / Dilgren: 25 epochs → 42.9 / 41.8, same lr as ours (1e-4), batch 128;
   - CODI: 40 epochs, LoRA at 3e-3 → 44.1.
   - The closest like-for-like comparison is **Coconut / Dilgren**: same code family, same lr, full FT. It differs from §200 C in **6.3× the epochs**, batch 128 vs 64, and best-validation (Coconut) or last-epoch (Dilgren) selection.
2. **Budget is likely but not proven.**
   - None of these papers reports a CoT-SFT learning curve, so we can't read off its accuracy at 4 epochs.
   - Our validation is flat at ~0.35 from 16k (≈ 2.7 epochs) under a **constant** lr. A plateau at constant lr doesn't rule out gains from more epochs or a decay (CODI uses cosine).
   - The one published budget curve is for CODI, not CoT: at the default γ = 1, 20 epochs gave 38.4% and 40 epochs 43.7% (CODI Table A5).
3. **Cheap ways to settle it in-house** (suggestions only):
   - evaluate the Dilgren CoT run's earlier checkpoints, if any are public (only `checkpoint_25` is in our bench);
   - continue §200 C to 25 epochs at the Coconut config (1e-4, bs 128) and read validation at 4 / 8 / 16 / 25;
   - check that §200 C's format and answer parsing match Deng's GSM8K-Aug (`####` answer) and the bench harness, so the 41.8 comparison is matched.
4. **Batch / lr are confounded with epochs.** CODI's 3e-3 is a LoRA lr and doesn't transfer to full FT. Coconut / Dilgren at full-FT 1e-4 is the right template.
