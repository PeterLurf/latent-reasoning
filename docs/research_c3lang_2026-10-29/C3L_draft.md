# C3-LANG draft: the latent loop vs text-CoT at matched compute, on the language word programs (2026-10-29)

**Status:** draft for red team + Literature. Theory's choice under the user's standing rules (north star: C3 after C4 = C2; C4's AUX-FREE line paused at §245; C2 not met at §243). Audit agreed, with points (1)–(3) adopted. Budget ≤ $5.

## 1. Question
- C3 (docs/NORTH_STAR.md): "at matched compute, equal or better accuracy than written CoT on tasks that need many steps". The user's operative C4 standard asks the same thing: does latent use latent space more effectively than text?
- No language-loop C3 reading exists. §225 was synthetic.

## 2. Arms (no new latent training)
- **LAT:** lh_fh_s1 / lh_fh_s2 (§240: the DF law, d ≤ 4 trained, hold fixed point; C1 gates passed at d ≤ 4).
- **TEXT:** txt_s1 / txt_s2, GPT-2 text-CoT trained from the same askname init (s233_askname_s1) on the same DF law, d ~ U{1..4}.
  - Target: question → step sentences R_step ("<name> has <v> <unit>.") → " The answer is <v_d>." + EOS. Teacher-forced CE on target tokens.
  - Harness: codi_ps.py --text_cot --direct (§227), re-pointed to the DF stream and the askname init. 16k steps, P0d full-FT recipe.
  - Training budget is not the matched quantity; inference compute is.
  - **UNDERTRAINED gate:** TEXT free-decode d ≤ 4 must be ≥ 0.95, else one +8k extension, then final. Disclosed: the LAT lineage had more total training steps.
- **CD-AN** (one pass, no loop): §238's df_pcd4_s1, reference.

## 3. Compute accounting (Audit (1); §225's method)
- **(S) sequential forward passes:**
  - LAT: K′ think passes + 1 decoder pass;
  - TEXT: generated tokens (each one decoding step with a KV cache), counted from the first generated token to EOS.
- **(F) FLOPs actually executed:**
  - LAT counts per pass the trunk blocks run on the tokens they run on (lower blocks are cached per distinct input, inject_aw cache_lower), plus the DeltaMem read / write and the decoder pass. Logged by an instrumented eval: executed_flops per item.
  - TEXT counts prompt prefill + Σ per-token decode with the KV cache (attention over the growing context).
  - An exact attention-inclusive count for both.
- **Budget grid B:** per d, accuracy as a function of (S) and of (F).
  - LAT: K′ ∈ {1..8, 16}.
  - TEXT: max_new_tokens truncation at a grid of budgets. Truncated = wrong (no parse), the same censoring as §227.

## 4. Readings (TEST2-DF, d 1..8, all items)
- **Binding, per seed pair (LAT_s vs TEXT_s):**
  - **C3-F:** at every d ∈ {2, 3, 4}, at every grid FLOPs budget where either arm reaches 0.5, acc_LAT ≥ acc_TEXT − 0.02.
  - **C3-S:** the same on (S).
  - **TEXT-AHEAD:** acc_TEXT ≥ acc_LAT + 0.05 at some budget for ≥ 2 of d 2–4.
- **Labels:**
  - C3-MATCHED-F / C3-MATCHED-S (each axis);
  - TEXT-AHEAD-F / S;
  - MIXED;
  - UNDERTRAINED (TEXT's gate fails after the extension → no C3 reading).
- **Descriptive:**
  - d 5–8 (out of range for both; extrapolation compare);
  - B*(d, 0.9) per axis;
  - TEXT's per-step value accuracy in the decoded chain;
  - error types.

## 5. OC (sketch)
- At d ≤ 4, LAT is at 1.000 with K′ = d − 1 passes, about d forward passes total (each over the full prompt, but lower blocks are cached).
- TEXT needs ≈ 10 tokens per step plus ~8 answer tokens, about 10 d + 8 sequential decode steps (each cheap with a KV cache).
- So on (S) LAT very likely wins (P ≈ 0.9). On (F) the outcome depends on the per-pass prompt recompute vs cached decode (P(C3-MATCHED-F) ≈ 0.5).
- Overall P(both axes MATCHED) ≈ 0.45.

## 6. Cost
- Two TEXT 16k runs ≈ $0.7 each (+ an optional $0.35 extension); LAT evals are local $0. Guard $3.
- Running: realised $20.1 + $3 = $23.1 ≤ $30, leaving room for one follow-up (Audit (3)).
