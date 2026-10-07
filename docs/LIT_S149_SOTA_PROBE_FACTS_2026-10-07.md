# Facts for §149 SOTA-PROBE: released GPT-2 GSM8K-Aug latent reasoners, their latent formats and step alignment (2026-10-07)

Literature session, for Theory; Audit copied.
- **Sources:** the HF API (file listings, 2026-10-07); the official configs and code (facebookresearch/coconut `args/gsm_coconut.yaml`,
  `dataset.py`; InternLM/SIM-CoT `Coconut/args/gsm_simcot.yaml`, `Coconut/coconut.py`, `CODI/src/model.py`); our GSM8K_REPRO.md
  and `theory/codi_probe.py`, which already load all of them.

## 1. Checkpoints (all public; accuracy = published / our bench harness, GSM8K-Aug test, n = 1,319)

| Model | HF id (file) | Thoughts | Published | Ours (GSM8K_REPRO) |
|---|---|---|---|---|
| **Coconut** (Dilgren 2604.04902, matched set, MIT) | `connordilgren/gpt2-gsm8k-coconut` (`checkpoint_33`, a raw state dict) | 6 | 33.1 (D&W T1; Coconut paper 34.1 ± 1.5) | 33.1 |
| **CODI** (authors) | `zen-E/CODI-gpt2` (`pytorch_model.bin`, LoRA, merged at load) | 6 | 43.7 (D&W re-run 42.2; SIM-CoT re-run 42.0) | 42.5 |
| **SIM-CoT + Coconut** (authors) | `internlm/SIM_COT-GPT2-Coconut` (`checkpoint_28`; the repo's eval names `checkpoint_30`, which isn't public) | 10 | 44.8 | 44.8 |
| **SIM-CoT + CODI** (authors) | `internlm/SIM_COT-GPT2-CODI` (safetensors) | 6 | 42.6 | 39.4 (not reproduced, −3.2) |
| References | `connordilgren/gpt2-gsm8k-cot`, `connordilgren/gpt2-gsm8k-no-cot` (`checkpoint_25`) | — | 41.6 / 16.8 | 41.8 / 16.5 |

- **Dilgren has no plain GSM8K CODI**, only `gpt2-gsm8k-multimode-codi` (multimode = a different training). Use `zen-E/CODI-gpt2`.

## 2. Latent formats
- **Coconut (official config; Dilgren matches):**
  - c_thought 2, max_latent_stage 3, `pad_latent_to_max: True`, 3 epochs per stage.
  - Past the last stage, `dataset.py` sets n_skip_steps = 10000: **all** language steps are removed and replaced by 3 × 2 = 6
    thoughts. That is the final-checkpoint format.
  - Tokens: question + "\n" + `<|start-latent|>` + 6 × `<|latent|>` + `<|end-latent|>`, then "### N".
  - **A thought is the previous position's last hidden state fed back raw** as the next input embedding, with no projection.
- **SIM-CoT + Coconut:** the same Coconut format with **c_thought 2, max_latent_stage 5** → 10 thoughts (`pad_latent_to_max`).
  Continued from a Coconut checkpoint (checkpoint_24) for 35 epochs. Raw hidden fed back. The auxiliary decoder is dropped at
  inference.
- **CODI:**
  - question + `<bot>`, then 6 latents, then `<eot>`, then "The answer is: N".
  - **Projected:** latent = prj(last hidden), with prj = Linear 768→768, GELU, Linear 768→768, LayerNorm (`codi_probe.py`:59–61).
  - The first latent is prj(h at `<bot>`).
- **SIM-CoT + CODI:** the CODI format with the same prj.

## 3. Thought ↔ step alignment
- **Coconut:** pair (2t−1, 2t) was trained to replace step t, **for t ≤ 3 only, and only during the curriculum.** The final stage
  removes all steps with the same 6 thoughts, so steps ≥ 4 have no slot. The alignment is a training-history prior, not a guarantee
  at the final checkpoint.
- **SIM-CoT + Coconut (documented in code):** the auxiliary decoder reads thought pair k and is trained to decode
  "Step k of the solution" for k = 1..4. **Pair 5 decodes "Step 3 and all the remaining steps"** (the prompt string in
  `coconut.py`:485; the label groups past max_latent_stage − 1 are concatenated).
  - So pair k ↔ step k (k ≤ 4), and pair 5 ↔ steps ≥ 5.
  - **The strongest documented alignment of the four.**
- **SIM-CoT + CODI (code):** the steps are split into num_latent + 1 groups (`get_steps(…, num_latent+1)`). The latent at forward
  index j (j = 0 is prj(h_bot)) decodes group j. So latent j ↔ step j+1.
  - I didn't read `get_steps` in full: check how the last group absorbs extra steps, and how latents beyond a problem's step count
    are padded (labels −100).
- **CODI (original):** **no step alignment.** Distillation is on the hidden state at the answer-preceding token. SERIAL there needs a
  probe-defined alignment (descriptive), with ablation + ρ as the rule, as you proposed.

## 4. GSM8K-Aug test data
- **Local:** `bench/data/gsm8k_aug/test.txt` (from Deng et al.'s Internalize_CoT_Step_by_Step data; the same file Coconut's
  `prepare_gsm8k.py` uses).
- **Format, one item per line:** `question||<<16-3-4=9>> <<9*2=18>> #### 18`. Equation steps are space-separated `<<expr=val>>`, and
  the answer follows `####`. Same 1,319 items and order as the openai test (LIT_BENCH_VERIFY).
- `bench/data/gsm8k_aug_icotsi/` is ICoT-SI's variant. Don't mix them.

## 5. Flags for the donor design
1. **GSM8K steps form a DAG, not a Markov chain.** Later equations reference earlier values *and* question numbers. Every thought
   can see the question (L1: recomputation is possible).
   - Donor ground truth needs an explicit substitution rule: insert the donor's step-t value into the recipient's chain and
     re-evaluate the downstream equations.
   - Restrict to recipient / donor pairs with the **same equation template** (operators and dependency graph, different numbers)
     for a clean prediction. Otherwise the "donor-consistent answer" is undefined.
2. **Distribution:** most items have 2–4 steps; the latents add little at ≥ 5 (GSM8K_REPRO per-step table, ≤ 0.07 at 6). Score
   donor tests by step count.
3. **ρ for raw-hidden models (Coconut, SIM-CoT + Coconut):** perturb in the fed-back embedding space, at a scale relative to the
   thought norm (raw hiddens have large, outlier-dominated norms in GPT-2). For CODI, perturb after prj (LayerNorm-bounded).
   - 2607.09698's prediction to test: **Coconut expanding, CODI contracting.**
4. **The checkpoint caveat:** SIM-CoT + Coconut is checkpoint_28 (public), not the repo's checkpoint_30. Our harness reproduces its
   44.8 regardless.
