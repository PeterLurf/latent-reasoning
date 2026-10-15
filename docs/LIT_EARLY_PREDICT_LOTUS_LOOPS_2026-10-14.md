# ID checks on Audit's early-prediction lead, and LOTUS's per-loop ablations (2026-10-14)

Literature session, for Audit (the request) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Sources:** the two grokking papers from their abstracts; LOTUS from full text (§4.3, Tables 4–8).

## 1. Early prediction of a run's outcome
- **2306.13253:** Notsawo, Zhou, Pezeshki, Rish & Dumas, "Predicting Grokking Long Before it Happens: A look into the loss landscape of models which grok". **Verified.**
  - The venue "ICLR 2024" is **not on the abstract page** (comments: 26 pages, 30 figures). Treat it as unconfirmed.
  - **Method:** a spectral signature of the early learning curve; the low-frequency Fourier amplitude detects early oscillations that precede grokking.
  - **Scope (EMPIRICAL):** framed as predicting **whether grokking will occur for given hyper-parameters**, to cut hyper-parameter search. Seed-to-seed discrimination at fixed hyper-parameters isn't claimed in the abstract.
  - **Using it for seed selection is an extrapolation;** check their seed experiments before relying on it.
- **2604.20923:** Golwala, "ILDR: Geometric Early Detection of Grokking". **Verified** (single author).
  - **Method:** the inter- / intra-class distance ratio on penultimate-layer representations, on **held-out** data, framed via Fisher's LDA criterion.
  - **Tasks:** modular arithmetic and **S5 composition**.
  - **Results:** it leads the transition by 9–73% of the budget, with longer leads on more complex algebra. Over **8 seeds** the lead is 950 ± 250 steps (CV 26%).
  - It also reports that optimiser interventions fired at the ILDR threshold give "bidirectional control" over the transition.
  - **Tag:** EMPIRICAL, plus a heuristic grounding.
  - **Scope:** it **times** an eventual transition. The abstract doesn't show that it separates seeds that **never** enter from late enterers. That's the property a seed-selection rule needs, and it should be tested on our failed seeds post hoc ($0: existing logs or checkpoints).
- **Related in-house / prior:**
  - Frankle 1912.05671: the outcome is decided early (LIT_SEED_STABILITY);
  - Gopalani & Hu 2506.13688: plateau signatures (LIT_PLATEAU_SIGNATURE);
  - our "hare" post-hoc signal (§199 A1.4).

## 2. LOTUS 2606.31779 (Fan, Svete, Lee): per-loop use. **No one-pass dominance at 3B.**
- **Architecture:**
  - a looped padded transformer: the full model runs over [question; K = 6 latent blocks × c = 25 tokens] for R iterations;
  - the question's KV cache is computed once;
  - the latent positions' states carry across iterations, and all blocks are refined in parallel;
  - CE on each latent position's gold CoT token.
- **Inference-time R** (one model trained at R = 6, Llama-3.2-3B, GSM8K test; Table 6):

  | R | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|
  | acc | 22.7 | 40.0 | 55.0 | 63.5 | 68.7 | **70.0** | 69.3 |

  - The gain is spread over loops 2–6. R = 1 (22.7) is even below that backbone's no-CoT (38.3), because truncation is off-distribution for a model trained at R = 6.
- **Training-time R** (separate models; Table 5): R = 2: 14.6, 3: 23.2, 4: 52.6, 5: 68.1, 6: 70.0.
- **Supervision ablation** (Table 4):
  - no latent supervision (answer CE only): 63.3;
  - CODI-style only: 64.4;
  - per-iteration supervision through the main head: 68.2 vs post-loop 70.0;
  - aux decoder, per-iteration: 69.9 vs post-loop 68.4.
- **Width:** c = 1 gives 49.7 vs c = 25 70.0; inference-time c below the trained value hurts by up to 19 points.
- **Reading for us:**
  - LOTUS shows **graded per-loop use** (each extra loop adds accuracy up to the trained R), not one-pass dominance.
  - But its cross-loop channel is the **full residual stream at 150 latent positions** (a wide explicit state), not a narrow memory. Its supervision is **per-position gold-token CE** on the latents themselves, which is closer to text CoT than to our memory-only design.
  - Every ablation is at **3B**. GPT-2 appears only in Table 1 (44.1 ± 0.7, 3 seeds). The seed count for the ablations isn't stated where I read.
  - A removal reading at our scale is still unknown.
