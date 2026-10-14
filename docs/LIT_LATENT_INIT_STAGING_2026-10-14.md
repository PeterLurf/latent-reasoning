# How the published GPT-2 GSM8K-Aug latent numbers are initialised and staged: do they start from a CoT-trained model? (2026-10-14)

Literature session, for Audit (the request) and Theory.
- **Sources:** full text (arXiv PDFs), the Coconut repo README and `args/` configs, and the SIM-CoT repo's `Coconut/args/gsm_simcot.yaml` (GitHub, main; 2026-10-14).
- **Evidence tag:** all EMPIRICAL (recipe facts).

## Answer: not all of them. There are two families.

| method (GPT-2 GSM8K-Aug) | init | CoT signal during latent training | source |
|---|---|---|---|
| **Coconut 34.1** | **A CoT-trained model.** "In the initial stage, the model is trained on regular CoT instances"; **6 epochs** of CoT, then 3 stages × 3 epochs, plus a final stage, up to 50 epochs. The repo says to train `gsm_cot.yaml` first, then "**select a checkpoint as the initialization of Coconut (the validation accuracy is expected to be around 40%)**" via `load_model_path` (the README's "to initialize coconut from a CoT-tuned model"). | Remaining language steps stay as targets in the early stages (curriculum replacement). | 2412.06769 §3 / §4; repo README and gsm_coconut.yaml |
| **SIM-CoT + Coconut 44.8** | **A trained Coconut model**, which itself started from CoT. The config loads `./ckpts/gsm-coconut/checkpoint_24` (resume 3, c = 2, max_latent_stage 5, 35 epochs, bs 8 per GPU, lr 1e-4). The paper only says "Following Hao et al." | Step-level aux decoder (SIM-CoT) plus the Coconut curriculum. | 2509.20317 App. E; SIM-CoT repo gsm_simcot.yaml |
| **CODI 43.7** | **Raw pretrained GPT-2 (LoRA).** There's no CoT pre-stage: one model plays teacher (explicit CoT, teacher-forced) and student (continuous thoughts), "**jointly**" trained, "share the same LLM". Why one model: "the model must first learn to perform explicit CoT reasoning before it can effectively compress…". A separate *pre-trained* teacher is an ablation (Table 2: 27.1%, far below 43.7). | **The CoT cross-entropy (teacher task) runs throughout**, plus L1 distillation of hidden states at the answer-prompt token across all layers. | 2502.21074 §3, §4.4 |
| **PCCoT 49.48** | **As CODI: raw GPT-2 + LoRA**, joint teacher / student ("we adopt CODI for training"). | The teacher-task CoT cross-entropy runs throughout. | 2506.18582 §3–4 |
| SIM-CoT + CODI 42.6 | As CODI (strictly follows Shen et al.'s configuration). | CODI's plus SIM-CoT's step decoder | 2509.20317 App. E |
| *(context)* RiM 33.6 | For GPT-2, "we adopt the same initial Stage 0 used for Coconut w/ Stage 0" (CoT warm start). Coconut *without* Stage 0 is 22.3 vs 31.1 with it (RiM Table 4). | | 2605.30343 |
| *(context)* Dilgren Coconut 33.1 (bench) | "stage 0 training is replaced with **checkpoint 6 of the CoT model**" | | 2604.04902 Table 4 |

## Reading for §200 (latent cells from raw GPT-2)
1. **Every published GPT-2 latent number has explicit CoT supervision, either beforehand or alongside.**
   - The Coconut family (Coconut, SIM-CoT + Coconut, RiM, Dilgren) starts from a **CoT-tuned checkpoint**: about 6 CoT epochs, with validation around 40% per the Coconut README.
   - The CODI family (CODI, PCCoT, SIM-CoT + CODI) starts from **raw GPT-2** but trains the **CoT teacher task jointly for all 40 epochs**, with hidden-state distillation from it.
   - **No published GPT-2 GSM8K-Aug latent result trains from raw GPT-2 without either.**
2. **Size of the effect, where measured:**
   - Coconut with vs without Stage 0: **31.1 vs 22.3** (RiM's re-run, Table 4: a 9-point drop without the CoT warm start).
   - CODI with a separate static teacher, i.e. no joint CoT learning: **27.1 vs 43.7** (CODI Table 2).
   - So a raw-GPT-2 latent cell **without** a CoT stage or a joint CoT loss should be expected to sit well below the published bars, independent of the architecture.
3. **For a matched §200 comparison, two options:**
   - (a) initialise the latent cells from the §200 TEXT-COT checkpoint (the Coconut convention; the README targets about 40% validation, while ours is ~0.35);
   - (b) add a joint CoT teacher loss on the same model (the CODI convention).
   - Our STEP-AUX is a per-pass step decoder, closer to SIM-CoT's than to a full CoT teacher task. Either way, state which family the cell belongs to when quoting the bars.
4. **A hygiene flag:** the SIM-CoT + Coconut config sets `val_path: ./data/gsm_test.json`, so epoch-wise "validation" runs on the **test** set. That's consistent with RiM's remark that official numbers are test-selected (LIT_GSM8K_AUG_BARS). Coconut's own config uses `gsm_valid.json`.
