# Remedies for the entry basin as posterior / bottleneck collapse: a map for an 8-d carried code under final-answer supervision (2026-10-05)

Literature session, at the user's request ("do more research") via Audit; Theory copied. All arXiv IDs verified via the
arXiv API on 2026-10-05 (or earlier, per LIT_INDEX).

**The symptom** is the classic collapse pair: the decoder (GPT-2) ignores the code, so the encoder gets no useful
gradient, so the code stays uninformative ("lagging inference", He 1901.05534). Loss sits at ln 120 for about 2k steps.

Columns: **TA** = task-agnostic. **Early** = acts within the first 2k steps. **Seq** = shown in recurrent or sequential
latent models.

| Remedy | Key reference(s) | TA | Early | Seq | Adaptation to our carrier |
|---|---|---|---|---|---|
| **Free bits / minimum information** | Kingma et al. 1606.04934 (free bits); **δ-VAE 1901.03416** (a committed rate floor by construction) | ✓ | ✓ (from step 0) | ✓ (autoregressive decoders) | a deterministic FSQ code has no KL. The analogue is a **batch code-entropy floor**: penalise code-usage perplexity below a target, so the code can't be near-constant |
| **Aggressive encoder training** | **He et al. 1901.05534** (inner-loop encoder updates until the mutual information plateaus, then a decoder step) | ✓ | ✓ (only in the early phase) | ✓ (LSTM text VAEs) | several writer (W_down, FSQ input) steps per reader step in the first 2k |
| **Word / input dropout** | Bowman et al. 1511.06349 | ✓ | ✓ | ✓ | drop the decoder's *alternative* information routes. Here: mask or drop item-token context at the answer decoder, so only the code can explain the answer |
| **Auxiliary predictive objectives on z** | **CPC 1807.03748** (contrastive prediction of future latents); **Z-Forcing 1711.05411** (an auxiliary cost making stochastic-RNN latents predict a backward state); bag-of-words loss 1703.10960; Next-Latent Prediction 2511.05963 | ✓ | ✓ | ✓ (all sequential) | **a gradient into the code that doesn't depend on the decoder reading it**: e.g. z_j must predict (contrastively) z_{j+1} or the next segment's hidden state |
| **KL / commitment annealing** | Bowman; cyclical annealing 1903.10145; VQ commitment β (1711.00937) | ✓ | ✓ | ✓ | Our ANNEAL (wide → narrow) failed. Cyclical schedules are the untested variant |
| **Skip-to-decoder / MI maximisation** | **Dieng et al. 1807.04863** (skip connections from latent to every decoder layer prevent collapse); InfoVAE 1706.02262 (an MI term) | ✓ | ✓ | ✓ (text) | **feed the code directly to the answer head and to later GPT-2 layers** (a residual skip), a short gradient path from code to loss. Cheap |
| **VQ codebook-collapse fixes** | Łańcucki 2005.08520 (re-initialisation, k-means init); Jukebox 2005.00341 (random restarts of dead codes); FSQ 2309.15505 (no codebook); **rotation trick 2410.06424** (better gradients through VQ than straight-through) | ✓ | ✓ | partly | FSQ already avoids dead codes. The rotation trick is relevant if moving to VQ, since ST gradients are weak early |
| **Optimiser / LR tricks** | Grokfast 2405.20233 (amplify slow gradients); separate higher LR for new modules (LoRA+ 2402.12354; prompt tuning at lr 0.3, 2104.08691); shrink-and-perturb 1910.08475 (escaping warm-start plateaus); early triage of bad seeds (Dodge 2002.06305) | ✓ | ✓ | mixed | **a 10–100× LR on the new writer and reader modules** vs GPT-2's body; restart-on-plateau as disclosed best-of-n |

## The 2–3 most promising for a pretrained-LM recurrent carrier
1. **Skip-to-decoder** (Dieng 1807.04863). Add z_b (projected) directly into the answer head's input and into the
   upper GPT-2 residual stream. The decoder then has a short, strong gradient path from code to loss from step 0. That
   attacks the "decoder ignores code" half directly, and it's cheap and task-agnostic.
2. **A decoder-independent auxiliary predictive objective on the code** (CPC 1807.03748 / Z-Forcing 1711.05411 /
   Next-Latent 2511.05963).
   - A contrastive loss making z_j predict the next segment's code or hidden state (other batch items as negatives).
   - It supplies encoder gradient even while the decoder ignores the code: the "encoder gets no useful gradient" half.
   - Label-free and task-agnostic.
3. **Aggressive writer updates plus a separate high LR for new modules** (He 1901.05534; LoRA+ / prompt-tuning LRs).
   Run k writer steps per reader step in the first 2k, with the new modules' LR at 10–100× GPT-2's. It targets the
   lagging-inference dynamics directly.
   - Optionally add a **batch code-entropy floor** (the free-bits analogue) to forbid the constant code.

**Scoring note.** Each acts in the first 2k steps, so read the **plateau rate at 2k** (the 123-3 flag) on the known
basin seed (S5 seed 2) as the primary outcome, against the §126 controls.
