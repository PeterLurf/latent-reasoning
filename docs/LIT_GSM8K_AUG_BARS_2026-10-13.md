# RiM 2605.30343 and the GPT-2 GSM8K-Aug bars (LOTUS vs CODI vs PCCoT), for the GSM8K-Aug pilot draft (2026-10-13)

Literature session, for Audit (the request) and Theory.
- **Sources:** full text from the arXiv PDFs (pdftotext) and the arXiv HTML for LOTUS Tables 1 / 13 (2026-10-13). IDs verified.
- **Evidence tags:** all sources here are EMPIRICAL (benchmark tables).

## Bars for GPT-2 (124M), trained on GSM8K-Aug, GSM8K test (1,319), greedy

| method | number | whose run | seeds | source |
|---|---|---|---|---|
| **PCCoT** | **49.48 ± 0.31** | PCCoT's own | 3 | PCCoT 2506.18582 Table 1. LOTUS Table 13 lists it as 49.5 (same number, cited). |
| Coconut + SIM-CoT | 44.8 | SIM-CoT's re-run | n/s | SIM-CoT. LOTUS Table 13 reuses it. |
| CoT-SFT, as reported by CODI | 44.1 | CODI's run (in a figure) | n/s | PCCoT: "The CoT baseline is taken from CODI". **This is not CODI's own method score.** |
| LOTUS | 44.1 ± 0.7 | LOTUS's own | 3 | LOTUS Table 1. |
| **CODI, CODI's own** | **43.7** | CODI's own | n/s | CODI 2502.21074, Table 2 (ablations) / Table A3. PCCoT also cites 43.7. |
| **Explicit CoT, as tabled in LOTUS** | **42.7** | **SIM-CoT's re-run**, reused | n/s | LOTUS Table 1 / 13. Identical to SIM-CoT's SFT-CoT row in every column (42.7 / 9.0 / 85.0 / 41.6 / 45.2; LIT_STEP_AUX_GSM8K), so it isn't a LOTUS run. |
| CODI + SIM-CoT / CODI (re-run) | 42.6 / 42.0 | SIM-CoT's re-runs | n/s | as above |
| Coconut | 34.1 ± 1.5 (own) / 36.6 (SIM-CoT re-run) | | | |
| No-CoT | 19.1 | CODI (LOTUS reuses it) | | |
| **RiM** (LoRA; see below) | 33.6 ± 0.2 (held-out checkpoint selection) / 35.5 (test-selected) | RiM's own | 16 split repeats | RiM Tables 1 / 6 |
| **Our bench's own CoT** | **41.8** | ours | | LIT_STEP_AUX_GSM8K. The matched reference. |

("n/s" = no seed count stated where I looked; not checked further.)

**Corrections to the request's premise:**
- **"CODI's own 44.1" is wrong.** CODI's own GPT-2 score is **43.7**. 44.1 is the **CoT-SFT** baseline that CODI reported and PCCoT copied.
- **"Explicit CoT 42.7" in LOTUS is SIM-CoT's re-run, reused, not a LOTUS run.** The two published explicit-CoT references for GPT-2 are 44.1 (CODI's) and 42.7 (SIM-CoT's), a 1.4-point spread from re-running alone.
- **For the pilot:**
  - The latent SOTA bar is **PCCoT 49.48 ± 0.31** (3 runs; the only seeded GPT-2 bar above 45).
  - The CoT bar should be **our own matched CoT run** (bench 41.8), quoted next to 42.7 / 44.1, not one of the published numbers alone.

## RiM: Aichberger & Hochreiter 2605.30343, "Unlocking the Working Memory of Large Language Models for Latent Reasoning" (Preprint)
- **Architecture:**
  - **Memory blocks** are *fixed* sequences of special `<m>` tokens appended after the question. Nothing is generated, so the whole latent phase is **one forward pass**: TTFT 7.6 ms for GPT-2, equal to no-CoT; Coconut is about 7× slower, CoT about 27×.
  - A custom block-causal attention mask: blocks attend to the question and earlier blocks, and per-block readout branches attend to the question and blocks so far.
  - **Two-stage curriculum:**
    - Stage 1: one block per reasoning step; the readout after block k predicts step k.
    - Stage 2: step supervision is dropped; a fixed K blocks; the readout after *each* block predicts the final answer (an "anytime" objective).
  - For GPT-2 they first run **Coconut's Stage 0** (CoT warm start).
  - Training: **rank-128 LoRA**, bf16, batch 128, about 8 epochs.
- **GPT-2 numbers (GSM8K, greedy; mean ± s.e. over 16 split repeats; Table 1 / Table 2 / Table 6):**

  | method | GSM8K | GSM-Hard | note |
  |---|---|---|---|
  | SFT w/o CoT | 15.4 | 3.5 | |
  | Coconut (with Stage 0) | 31.1 | 7.1 | without Stage 0: 22.3 |
  | **RiM, final block** | **33.6** | 7.8 | the deployable number |
  | RiM, any block | 39.5 | 9.4 | oracle: best block per item |
  | SFT w/ CoT | 39.8 | 8.4 | their LoRA run |
  | RiM, test-selected checkpoint ("official convention") | **35.5** | | vs Coconut official 34.1, DART 24.7 |

  - pass@8: RiM final block 49.1, any block 78.1.
- **Removal / causal tests: none found.** The paper has:
  - a memory-budget sweep (blocks K × tokens per block M; after Stage 2 accuracy is about flat over budgets, "~43%" in Fig. 6a, which I couldn't read off the PDF text for GPT-2);
  - representation analyses (block representations become block-specific and input-dependent);
  - linear probes that predict correctness from the block representations (Table 5).
  - There is **no** K = 0 / ablate-the-blocks / corrupt-or-patch-the-block test, and no comparison with the same model and *no* blocks under the Stage-2 objective. Their "SFT w/o CoT" is a separately trained model.
  - So, in our terms, there is no C1 evidence.
- **Checkpoint-selection note (useful for our pilot's hygiene):**
  - RiM selects checkpoints by cross-validation on GSM8K test splits (16 repeats).
  - It says the "official numbers" convention is to **select the best checkpoint on the full evaluation set** (stated about the Coconut / DART numbers). Test-selected numbers are optimistic.
  - **Our pilot should fix its checkpoint rule (last, or chosen on a train-derived split) in the registration.** I haven't checked PCCoT's / CODI's selection rule.
- **Relevance:**
  - It's the closest *fixed-slot, single-pass* latent design (cf. LLoCoT, PCCoT's parallel Jacobi tokens; our READ-SLOTS).
  - Its Stage 1 → Stage 2 (step supervision, then answer-only on every block) parallels our STEP-AUX → answer schedule.
  - It's **not a GPT-2 SOTA**: 33.6 / 35.5, below CODI and PCCoT, though it's a LoRA, 8-epoch budget.
